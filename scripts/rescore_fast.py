"""Retrain on signals that survive new accounts.

The 85 out-of-fold score used whole-window statistics of accounts that do not
exist in the test split, and the live score was 59. This script keeps the
behavioural features, adds an explicit burst-and-chain description of each
payment, and refits. The chain and fan scores use only the payments around
the one being scored, so they are defined the same way on unseen accounts.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from dd.loading import load_labels, load_split

SEED = 20261024
FAN_WINDOW = 15 * 60
CHAIN_WINDOW = 30 * 60
MAX_HOPS = 6


def _groups(keys, order):
    groups = defaultdict(list)
    for i in order:
        groups[keys[i]].append(i)
    return groups


def structural_features(payments: pd.DataFrame) -> pd.DataFrame:
    n = len(payments)
    ts = payments["timestamp"].to_numpy(dtype="float64")
    usd = payments["usd"].to_numpy(dtype="float64")
    src = payments["src"].to_numpy()
    dst = payments["dst"].to_numpy()
    order = np.argsort(ts, kind="mergesort")
    outgoing = _groups(src, order)
    incoming = _groups(dst, order)

    fan_out = np.ones(n)
    fan_out_span = np.zeros(n)
    fan_out_cv = np.zeros(n)
    fan_in = np.ones(n)
    fan_in_span = np.zeros(n)
    fan_in_cv = np.zeros(n)

    def fill_fan(groups, count, span, cv, partner):
        for idxs in groups.values():
            tt = ts[idxs]
            vv = usd[idxs]
            pp = partner[idxs]
            for j, i in enumerate(idxs):
                lo = int(np.searchsorted(tt, tt[j] - FAN_WINDOW, side="left"))
                hi = int(np.searchsorted(tt, tt[j] + FAN_WINDOW, side="right"))
                count[i] = len(set(pp[lo:hi].tolist()))
                span[i] = tt[hi - 1] - tt[lo]
                block = vv[lo:hi]
                cv[i] = float(np.nanstd(block) / (np.nanmean(block) + 1e-9)) if hi - lo > 1 else 0.0

    fill_fan(outgoing, fan_out, fan_out_span, fan_out_cv, dst)
    fill_fan(incoming, fan_in, fan_in_span, fan_in_cv, src)

    # Greedy link: the receiver's next outgoing payment that carries a similar
    # amount within half an hour. A real chain peels a little at every hop.
    nxt = np.full(n, -1, dtype=np.int32)
    link_dt = np.full(n, np.nan)
    link_ratio = np.full(n, np.nan)
    for i in range(n):
        idxs = outgoing.get(dst[i])
        if not idxs:
            continue
        tt = ts[idxs]
        lo = int(np.searchsorted(tt, ts[i] + 1e-6, side="left"))
        hi = int(np.searchsorted(tt, ts[i] + CHAIN_WINDOW, side="right"))
        if lo >= hi:
            continue
        ratios = usd[idxs[lo:hi]] / (usd[i] + 1e-9)
        ok = (ratios >= 0.55) & (ratios <= 1.15)
        if not ok.any():
            continue
        # prefer the soonest hop that still looks like the same money
        pick = lo + int(np.flatnonzero(ok)[0])
        nxt[i] = idxs[pick]
        link_dt[i] = ts[idxs[pick]] - ts[i]
        link_ratio[i] = ratios[int(np.flatnonzero(ok)[0])]

    chain_len = np.ones(n)
    chain_decay = np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if seen[i] or nxt[i] < 0:
            continue
        path = []
        j = i
        guard = 0
        while j >= 0 and not seen[j] and guard < MAX_HOPS:
            seen[j] = True
            path.append(j)
            j = int(nxt[j])
            guard += 1
        # only count a path once, then share its length with every hop
        length = float(len(path))
        if length < 2:
            continue
        for j in path:
            chain_len[j] = length
        amounts = usd[path]
        chain_decay[path] = (amounts[0] - amounts[-1]) / (amounts[0] + 1e-9)

    # A payment that is itself a hop, even if we started the walk elsewhere.
    hop = nxt >= 0
    out = pd.DataFrame(
        {
            "TRANSACTION_HASH": payments["TRANSACTION_HASH"].to_numpy(),
            "fan_out_15m": fan_out,
            "fan_out_span_15m": fan_out_span,
            "fan_out_cv_15m": fan_out_cv,
            "fan_in_15m": fan_in,
            "fan_in_span_15m": fan_in_span,
            "fan_in_cv_15m": fan_in_cv,
            "chain_hops": chain_len,
            "chain_decay": chain_decay,
            "chain_next_dt": link_dt,
            "chain_next_ratio": link_ratio,
            "chain_is_hop": hop.astype("float64"),
        }
    )
    return out


KEEP_PREFIXES = ("amt_", "time_", "conv_", "flow_", "vel_", "str_", "rel_")
KEEP_EXACT = {
    "pair_is_first", "pair_dt_prev", "pair_amt_vs_prev", "pair_amt_vs_mean", "pair_cv",
    "pair_reciprocal", "pair_share_of_src", "pair_share_of_dst", "pair_seq",
    "src_flow_through", "dst_flow_through", "src_flow_imbalance", "dst_flow_imbalance",
    "src_out_cv", "dst_in_cv", "src_in_cv", "dst_out_cv", "src_degree_ratio", "dst_degree_ratio",
    "g_reciprocal_edge", "g_jaccard", "g_common_neighbours", "g_two_hop_paths", "g_same_community",
    "g_src_clustering", "g_dst_clustering",
}
DROP = {"time_offset_days", "conv_delivered_asset_code"}


def select_columns(frame: pd.DataFrame) -> list:
    cols = []
    for c in frame.columns:
        if c == "TRANSACTION_HASH" or c in DROP:
            continue
        if c.startswith(KEEP_PREFIXES) or c in KEEP_EXACT or c.startswith(("fan_", "chain_")):
            cols.append(c)
    return cols


def fit_and_score(train_x, y, test_x):
    import lightgbm as lgb

    params = dict(
        objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=48,
        min_data_in_leaf=30, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
        lambda_l2=1.5, verbose=-1, num_threads=4, seed=SEED,
    )
    oof = np.zeros(len(train_x))
    preds = np.zeros(len(test_x))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(folds.split(train_x, y)):
        run = dict(params, seed=SEED + fold, bagging_seed=SEED + fold + 1, feature_fraction_seed=SEED + fold + 2)
        model = lgb.train(
            run,
            lgb.Dataset(train_x.iloc[tr], y[tr]),
            num_boost_round=1200,
            valid_sets=[lgb.Dataset(train_x.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof[va] = model.predict(train_x.iloc[va], num_iteration=model.best_iteration)
        preds += model.predict(test_x, num_iteration=model.best_iteration) / 5
        print(f"  fold {fold} best {model.best_iteration} AP {average_precision_score(y[va], oof[va]):.4f}", flush=True)
    return oof, preds


def main():
    train_dir, test_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print("structural features", flush=True)
    train_pay = load_split(train_dir)["payments"]
    test_pay = load_split(test_dir)["payments"]
    train_struct = structural_features(train_pay)
    test_struct = structural_features(test_pay)

    train = pd.read_parquet("cache/train_features.parquet").merge(train_struct, on="TRANSACTION_HASH")
    test = pd.read_parquet("cache/test_features.parquet").merge(test_struct, on="TRANSACTION_HASH")
    labels = load_labels(train_dir).set_index("TRANSACTION_HASH")
    y = labels["is_suspicious"].reindex(train["TRANSACTION_HASH"]).to_numpy().astype(int)

    # The structural score on its own, before any fitting, so we can see the signal.
    raw = (
        np.log1p(train_struct["fan_out_15m"]) + np.log1p(train_struct["fan_in_15m"])
        + train_struct["chain_hops"] + train_struct["chain_is_hop"]
    )
    print(f"unfitted structure AP {average_precision_score(y, raw.to_numpy()):.4f}", flush=True)

    cols = select_columns(train)
    print(f"fitting {len(cols)} features", flush=True)
    oof, preds = fit_and_score(train[cols], y, test[cols])
    ap = average_precision_score(y, oof)
    k = max(1, int(round(len(y) * 0.01)))
    order = np.argsort(-oof)
    precision = y[order[:k]].mean()
    recall = y[order[:k]].sum() / y.sum()
    print(f"OOF AP {ap:.4f}  score {100 * ap:.2f}  top1 precision {precision:.3f} recall {recall:.3f}", flush=True)

    spread = pd.Series(preds).rank(method="average", pct=True).to_numpy()
    submission = pd.DataFrame({"TRANSACTION_HASH": test["TRANSACTION_HASH"].to_numpy(), "score": np.round(spread, 6)})
    assert submission["TRANSACTION_HASH"].is_unique
    submission.to_csv(out_path, index=False)
    print(f"wrote {len(submission)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
