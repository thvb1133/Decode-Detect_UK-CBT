"""Score every test transaction from time-local account bursts.

A suspicious payment sits in a tight cluster: the same accounts trade again
within a short gap, as a fan, a chain, a cycle, or a relay. Linking payments
that share an account and follow each other inside a time gap makes those
clusters. The same features are built inside each split, so a test payment is
only compared with other test payments. Labels are read after the features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (60, 120, 180, 300, 600, 1800, 7200, 28800)
SEED = 42
N_FOLDS = 5


def _rates(path: Path) -> pd.DataFrame:
    rates = pd.read_csv(path)
    asset = rates["ASSET"].astype(str)
    issuer = asset.str.split(":", n=1).str[0].where(asset.str.contains(":"), "")
    symbol = asset.str.split(":", n=1).str[-1].str.upper()
    hour = pd.to_datetime(rates["TIMESTAMP"], unit="s", utc=True).dt.floor("h")
    out = pd.DataFrame(
        {
            "hour": hour,
            "symbol": symbol,
            "issuer": issuer,
            "usd_price": pd.to_numeric(rates["PRICE_MICRO_USD"], errors="coerce") / 1_000_000,
        }
    )
    return out.dropna(subset=["hour", "usd_price"]).drop_duplicates(["hour", "symbol", "issuer"])


def load_payments(folder: Path) -> pd.DataFrame:
    """One row per transaction, plus every account that touches it."""
    transactions = pd.read_parquet(
        folder / "transactions.parquet",
        columns=["TRANSACTION_HASH", "TIMESTAMP", "FEE", "ACCOUNT"],
    )
    transfers = pd.read_parquet(
        folder / "transfers.parquet",
        columns=[
            "TRANSACTION_HASH", "TRANSFER_INDEX", "FROM_ADDRESS", "TO_ADDRESS",
            "TOKEN_ADDRESS", "SYMBOL", "VALUE", "TRANSFER_TYPE",
        ],
    )
    transfers["VALUE"] = pd.to_numeric(transfers["VALUE"], errors="coerce").fillna(0.0)
    transfers = transfers.merge(
        transactions[["TRANSACTION_HASH", "TIMESTAMP"]], on="TRANSACTION_HASH", how="left"
    )
    transfers["TIMESTAMP"] = pd.to_numeric(transfers["TIMESTAMP"], errors="coerce").astype("int64")
    transfers["hour"] = pd.to_datetime(transfers["TIMESTAMP"], unit="s", utc=True).dt.floor("h")
    symbol = transfers["SYMBOL"].astype(str).str.upper()
    issuer = transfers["TOKEN_ADDRESS"].astype(str)
    issuer = issuer.where(~symbol.eq("XRP"), "")
    transfers["symbol"] = symbol
    transfers["issuer"] = issuer
    priced = transfers.merge(_rates(folder / "market_rates.csv"), on=["hour", "symbol", "issuer"], how="left")
    if priced["usd_price"].isna().any():
        fallback = _rates(folder / "market_rates.csv").groupby(["hour", "symbol"], as_index=False)["usd_price"].median()
        priced = priced.merge(fallback, on=["hour", "symbol"], how="left", suffixes=("", "_fallback"))
        priced["usd_price"] = priced["usd_price"].fillna(priced["usd_price_fallback"])
        priced = priced.drop(columns=["usd_price_fallback"])
    priced["usd_price"] = priced["usd_price"].fillna(1.0)
    priced["usd"] = priced["VALUE"].clip(lower=0) * priced["usd_price"]

    priced = priced.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    first = priced.drop_duplicates("TRANSACTION_HASH", keep="first")
    last = priced.drop_duplicates("TRANSACTION_HASH", keep="last")
    stats = priced.groupby("TRANSACTION_HASH").agg(
        n_legs=("usd", "size"),
        n_assets=("symbol", "nunique"),
        usd=("usd", "max"),
        path_frac=("TRANSFER_TYPE", lambda s: float((s == "path_payment").mean())),
    )
    endpoints = pd.concat(
        [
            priced[["TRANSACTION_HASH", "FROM_ADDRESS"]].rename(columns={"FROM_ADDRESS": "account"}),
            priced[["TRANSACTION_HASH", "TO_ADDRESS"]].rename(columns={"TO_ADDRESS": "account"}),
        ],
        ignore_index=True,
    )
    accounts = endpoints.groupby("TRANSACTION_HASH")["account"].agg(lambda values: tuple(pd.unique(values)))
    out = pd.DataFrame({"TRANSACTION_HASH": stats.index})
    out = out.merge(stats, left_on="TRANSACTION_HASH", right_index=True)
    out["sender"] = first.set_index("TRANSACTION_HASH").loc[out["TRANSACTION_HASH"], "FROM_ADDRESS"].to_numpy()
    out["receiver"] = last.set_index("TRANSACTION_HASH").loc[out["TRANSACTION_HASH"], "TO_ADDRESS"].to_numpy()
    indexed_first = first.set_index("TRANSACTION_HASH")
    indexed_last = last.set_index("TRANSACTION_HASH")
    out["src_symbol"] = indexed_first.loc[out["TRANSACTION_HASH"], "symbol"].to_numpy()
    out["dst_symbol"] = indexed_last.loc[out["TRANSACTION_HASH"], "symbol"].to_numpy()
    out["src_usd"] = indexed_first.loc[out["TRANSACTION_HASH"], "usd"].to_numpy(dtype=float)
    out["dst_usd"] = indexed_last.loc[out["TRANSACTION_HASH"], "usd"].to_numpy(dtype=float)
    out["src_units"] = indexed_first.loc[out["TRANSACTION_HASH"], "VALUE"].to_numpy(dtype=float)
    out["dst_units"] = indexed_last.loc[out["TRANSACTION_HASH"], "VALUE"].to_numpy(dtype=float)
    out["time"] = pd.to_numeric(
        transactions.set_index("TRANSACTION_HASH").loc[out["TRANSACTION_HASH"], "TIMESTAMP"], errors="coerce"
    ).to_numpy().astype(np.int64)
    out["fee"] = pd.to_numeric(
        transactions.set_index("TRANSACTION_HASH").loc[out["TRANSACTION_HASH"], "FEE"], errors="coerce"
    ).fillna(0.0).to_numpy()
    out["accounts"] = accounts.loc[out["TRANSACTION_HASH"]].to_numpy()
    out["conv"] = (out["src_symbol"] != out["dst_symbol"]).astype(float)
    out["hour"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.hour.astype(int)
    out["dow"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.day_of_week.astype(int)
    return out.sort_values("time").reset_index(drop=True)


def _union_find(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int8)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for left, right in pairs:
        a, b = find(left), find(right)
        if a == b:
            continue
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1
    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


def _burst_frame(pay: pd.DataFrame, window: int) -> pd.DataFrame:
    """Features of the connected burst that contains each payment."""
    n = len(pay)
    times = pay["time"].to_numpy()
    senders = pay["sender"].to_numpy()
    receivers = pay["receiver"].to_numpy()
    usd = pay["usd"].to_numpy(dtype=float)
    touched: dict[str, list[int]] = {}
    for index, accounts in enumerate(pay["accounts"]):
        for account in accounts:
            touched.setdefault(account, []).append(index)
    pairs: list[tuple[int, int]] = []
    for members in touched.values():
        members.sort(key=lambda i: times[i])
        for left, right in zip(members, members[1:]):
            if left != right and times[right] - times[left] <= window:
                pairs.append((left, right))
    labels = _union_find(n, pairs)
    # Compact ids.
    _, labels = np.unique(labels, return_inverse=True)
    frame = pay[["sender", "receiver", "usd", "n_legs", "conv", "n_assets"]].copy()
    frame["comp"] = labels
    frame["time"] = times
    grouped = frame.groupby("comp")
    count = grouped.size()
    span = grouped["time"].max() - grouped["time"].min()
    usd_mean = grouped["usd"].mean()
    usd_std = grouped["usd"].std().fillna(0.0)
    n_sender = grouped["sender"].nunique()
    n_receiver = grouped["receiver"].nunique()
    max_out = frame.groupby(["comp", "sender"]).size().groupby("comp").max()
    max_in = frame.groupby(["comp", "receiver"]).size().groupby("comp").max()
    account_count = (
        pd.DataFrame({"comp": np.repeat(labels, [len(a) for a in pay["accounts"]]),
                      "account": [a for accounts in pay["accounts"] for a in accounts]})
        .drop_duplicates()
        .groupby("comp").size()
    )
    activity = pd.Series(0, dtype=float)
    for account, members in touched.items():
        activity[account] = len(set(members))
    endpoint_activity = np.array([activity[s] + activity[r] for s, r in zip(senders, receivers)], dtype=float)
    act_sum = pd.Series(endpoint_activity).groupby(labels).sum()
    conv_frac = grouped["conv"].mean()
    legs_mean = grouped["n_legs"].mean()
    assets = grouped["n_assets"].max()
    prefix = f"w{window}_"
    out = pd.DataFrame(index=pay.index)
    comp = pd.Series(labels, index=pay.index)
    out[prefix + "n"] = comp.map(count).to_numpy()
    out[prefix + "span"] = comp.map(span).fillna(0).to_numpy()
    out[prefix + "gap"] = out[prefix + "span"] / np.maximum(out[prefix + "n"] - 1, 1)
    out[prefix + "accounts"] = comp.map(account_count).to_numpy()
    out[prefix + "senders"] = comp.map(n_sender).to_numpy()
    out[prefix + "receivers"] = comp.map(n_receiver).to_numpy()
    out[prefix + "density"] = out[prefix + "n"] / out[prefix + "accounts"]
    out[prefix + "usd_mean"] = np.log1p(comp.map(usd_mean).to_numpy())
    out[prefix + "usd_cv"] = (comp.map(usd_std).to_numpy() / (comp.map(usd_mean).to_numpy() + 1e-9))
    out[prefix + "max_out"] = comp.map(max_out).to_numpy()
    out[prefix + "max_in"] = comp.map(max_in).to_numpy()
    out[prefix + "star"] = np.maximum(out[prefix + "max_in"], out[prefix + "max_out"]) / out[prefix + "n"]
    out[prefix + "focus"] = (2 * out[prefix + "n"]) / comp.map(act_sum).to_numpy()
    out[prefix + "conv"] = comp.map(conv_frac).to_numpy()
    out[prefix + "legs"] = comp.map(legs_mean).to_numpy()
    out[prefix + "assets"] = comp.map(assets).to_numpy()
    out[prefix + "pos"] = (times - comp.map(frame.groupby("comp")["time"].min()).to_numpy()) / np.maximum(out[prefix + "span"], 1)
    return out


def _window_ratios(pay: pd.DataFrame) -> pd.DataFrame:
    """How crowded this account's recent half-window is, compared with its usual rate."""
    out = pd.DataFrame(index=pay.index)
    span = max(int(pay["time"].max() - pay["time"].min()), 1)
    for role, column in (("s", "sender"), ("r", "receiver")):
        codes, _ = pd.factorize(pay[column])
        codes = codes.astype(np.int64)
        times = pay["time"].to_numpy()
        order = np.lexsort((times, codes))
        key_codes = codes[order]
        key_times = times[order]
        # prefix counts per account via a global search on code*BIG+time
        big = np.int64(10**12)
        packed = key_codes.astype(np.int64) * big + key_times.astype(np.int64)
        totals = pd.Series(codes).value_counts()
        for window in (300, 900, 3600, 10800, 28800):
            queries = codes.astype(np.int64) * big + times.astype(np.int64)
            left = np.searchsorted(packed, queries - window, side="left")
            right = np.searchsorted(packed, queries + window, side="right")
            # searchsorted is only valid within the same account. Clamp to that account's slice.
            # Recompute bounds per row using the packed key, then drop crossings by checking code.
            count = (right - left).astype(float)
            # Remove hits that belong to a different account: the packed order is grouped by code,
            # so a window that crosses the code boundary is exactly the part outside this code's block.
            code_start = np.searchsorted(key_codes, codes, side="left")
            code_end = np.searchsorted(key_codes, codes, side="right")
            left = np.maximum(left, code_start)
            right = np.minimum(right, code_end)
            count = (right - left).astype(float)
            expected = totals.loc[codes].to_numpy(dtype=float) * (2 * window) / span
            out[f"{role}_rel_{window}"] = count / (expected + 1.0)
            out[f"{role}_n_{window}"] = count
    return out


def _account_window(codes: np.ndarray, times: np.ndarray, values: np.ndarray, query_codes: np.ndarray, query_times: np.ndarray, offset_lo: int, offset_hi: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((times, codes))
    sorted_codes = codes[order]
    sorted_times = times[order]
    cum = np.concatenate([[0.0], np.cumsum(values[order])])
    big = np.int64(10**12)
    packed = sorted_codes.astype(np.int64) * big + sorted_times.astype(np.int64)
    queries = query_codes.astype(np.int64) * big + query_times.astype(np.int64)
    left = np.searchsorted(packed, queries + offset_lo, side="left")
    right = np.searchsorted(packed, queries + offset_hi, side="right")
    code_start = np.searchsorted(sorted_codes, query_codes, side="left")
    code_end = np.searchsorted(sorted_codes, query_codes, side="right")
    left = np.maximum(left, code_start)
    right = np.minimum(right, code_end)
    return cum[right] - cum[left], (right - left).astype(float)


def _flow_features(pay: pd.DataFrame) -> pd.DataFrame:
    """Pass-through, repeated amounts, and whether money arrived just before it left."""
    n = len(pay)
    codes, _ = pd.factorize(pd.concat([pay["sender"], pay["receiver"]], ignore_index=True))
    sender = codes[:n].astype(np.int64)
    receiver = codes[n:].astype(np.int64)
    times = pay["time"].to_numpy()
    usd = pay["dst_usd"].to_numpy(dtype=float)
    out = pd.DataFrame(index=pay.index)
    out["log_dst"] = np.log1p(np.clip(usd, 0, None))
    out["log_src"] = np.log1p(np.clip(pay["src_usd"].to_numpy(dtype=float), 0, None))
    out["slippage"] = pay["src_usd"].to_numpy(dtype=float) / (usd + 1e-9) - 1
    units = pay["dst_units"].to_numpy(dtype=float)
    out["round_100"] = ((np.round(usd) % 100 == 0) & (np.abs(usd - np.round(usd)) < 0.5)).astype(float)
    out["round_unit"] = (np.abs(units - np.round(units)) < 1e-6).astype(float)

    sent = pd.DataFrame({"account": sender, "other": receiver, "usd": usd}).groupby("account")
    received = pd.DataFrame({"account": receiver, "other": sender, "usd": usd}).groupby("account")
    out_n = sent.size()
    in_n = received.size()
    out_sum = sent["usd"].sum()
    in_sum = received["usd"].sum()
    out_uniq = sent["other"].nunique()
    in_uniq = received["other"].nunique()
    out_mean = sent["usd"].mean()
    in_mean = received["usd"].mean()
    out["s_out_n"] = out_n.reindex(sender).to_numpy()
    out["r_in_n"] = in_n.reindex(receiver).to_numpy()
    out["s_in_n"] = in_n.reindex(sender).fillna(0).to_numpy()
    out["r_out_n"] = out_n.reindex(receiver).fillna(0).to_numpy()
    out["s_cp"] = (out_uniq.reindex(sender).to_numpy() / np.maximum(out["s_out_n"], 1))
    out["r_cp"] = (in_uniq.reindex(receiver).to_numpy() / np.maximum(out["r_in_n"], 1))
    out["s_pass"] = (out_sum.reindex(sender).fillna(0).to_numpy() / (in_sum.reindex(sender).fillna(0).to_numpy() + 1))
    out["r_pass"] = (out_sum.reindex(receiver).fillna(0).to_numpy() / (in_sum.reindex(receiver).fillna(0).to_numpy() + 1))
    out["usd_vs_sender"] = out["log_dst"] - np.log1p(out_mean.reindex(sender).to_numpy())
    out["usd_vs_receiver"] = out["log_dst"] - np.log1p(in_mean.reindex(receiver).to_numpy())
    out["share_sender"] = usd / (out_sum.reindex(sender).to_numpy() + 1e-9)
    out["share_receiver"] = usd / (in_sum.reindex(receiver).to_numpy() + 1e-9)

    pair = pd.Series(sender * (int(codes.max()) + 1) + receiver)
    out["pair_n"] = pair.map(pair.value_counts()).to_numpy(dtype=float)
    out["pair_first"] = pair.groupby(pair).cumcount().eq(0).astype(float).to_numpy()
    paid = set(zip(sender.tolist(), receiver.tolist()))
    out["cycle2"] = [float((right, left) in paid) for left, right in zip(sender.tolist(), receiver.tolist())]

    amount_key = pd.Series(list(zip(pay["sender"], pay["dst_symbol"], np.round(units, 2))))
    out["same_amount_sender"] = amount_key.map(amount_key.value_counts()).to_numpy(dtype=float)
    recv_key = pd.Series(list(zip(pay["receiver"], pay["dst_symbol"], np.round(units, 2))))
    out["same_amount_receiver"] = recv_key.map(recv_key.value_counts()).to_numpy(dtype=float)

    for name, window in (("10m", 600), ("1h", 3600), ("6h", 21600), ("24h", 86400)):
        funded, funded_n = _account_window(receiver, times, usd, sender, times, -window, -1)
        onward, onward_n = _account_window(sender, times, usd, receiver, times, 1, window)
        out[f"funded_n_{name}"] = funded_n
        out[f"funded_ratio_{name}"] = np.clip(funded / (usd + 1e-9), 0, 100)
        out[f"forward_n_{name}"] = onward_n
        out[f"forward_ratio_{name}"] = np.clip(onward / (usd + 1e-9), 0, 100)
    week = 8 * 86400
    out["sender_relay"] = np.clip(
        _account_window(sender, times, usd, sender, times, 0, week)[0] / (_account_window(receiver, times, usd, sender, times, -week, 0)[0] + 1e-9),
        0, 5,
    )
    out["receiver_relay"] = np.clip(
        _account_window(sender, times, usd, receiver, times, 0, week)[0] / (_account_window(receiver, times, usd, receiver, times, -week, 0)[0] + 1e-9),
        0, 5,
    )
    return out


def _graph_features(pay: pd.DataFrame) -> pd.DataFrame:
    """Two-hop routes, 3-cycles, and a short PageRank on the payment graph."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n = len(pay)
    codes, _ = pd.factorize(pd.concat([pay["sender"], pay["receiver"]], ignore_index=True))
    sender = codes[:n]
    receiver = codes[n:]
    size = int(codes.max()) + 1
    usd = pay["dst_usd"].to_numpy(dtype=float)
    adjacency = sparse.csr_matrix((np.ones(n), (sender, receiver)), shape=(size, size))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    transpose = adjacency.T.tocsr()
    ones = np.ones(size)
    _, component = connected_components(adjacency, directed=False)
    component_size = np.bincount(component)[component]
    weighted = sparse.csr_matrix((usd, (sender, receiver)), shape=(size, size))
    weighted.sum_duplicates()
    strength = np.asarray(weighted.sum(1)).ravel()
    inverse = np.divide(1.0, strength, out=np.zeros(size), where=strength > 0)
    transition = sparse.diags(inverse) @ weighted
    rank = np.full(size, 1 / size)
    dangling = strength == 0
    for _ in range(20):
        rank = 0.15 / size + 0.85 * (transition.T @ rank + rank[dangling].sum() / size)
    three_cycle = np.asarray(adjacency[receiver].multiply(transpose[sender]).sum(1)).ravel()
    out = pd.DataFrame(index=pay.index)
    out["s_paths2"] = (adjacency @ (adjacency @ ones))[sender]
    out["r_paths2"] = (transpose @ (transpose @ ones))[receiver]
    out["s_component"] = component_size[sender]
    out["r_component"] = component_size[receiver]
    out["s_pagerank"] = rank[sender]
    out["r_pagerank"] = rank[receiver]
    out["cycle3"] = three_cycle
    return out


def feature_matrix(pay: pd.DataFrame) -> pd.DataFrame:
    parts = [
        pd.DataFrame(
            {
                "log_usd": np.log1p(pay["usd"].clip(lower=0)),
                "log_fee": np.log1p(pay["fee"].clip(lower=0)),
                "n_legs": pay["n_legs"],
                "n_assets": pay["n_assets"],
                "conv": pay["conv"],
                "path_frac": pay["path_frac"],
                "hour": pay["hour"],
                "dow": pay["dow"],
            },
            index=pay.index,
        ),
        _flow_features(pay),
        _graph_features(pay),
    ]
    for window in WINDOWS:
        parts.append(_burst_frame(pay, window))
    parts.append(_window_ratios(pay))
    matrix = pd.concat(parts, axis=1)
    matrix.index = pay["TRANSACTION_HASH"].to_numpy()
    return matrix.replace([np.inf, -np.inf], np.nan)


def _rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").div(len(values)).to_numpy()


def _models(seed: int, scale: float):
    import lightgbm as lgb

    boost = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=10.0,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    trees = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    return boost, trees


def score_folder(root: Path) -> Path:
    print("building features", flush=True)
    matrices = {}
    payments = {}
    for split in ("train", "validation", "test"):
        pay = load_payments(root / "data" / "raw" / split)
        payments[split] = pay
        matrices[split] = feature_matrix(pay)
        print(f"  {split}: {matrices[split].shape}", flush=True)
    labels = pd.read_csv(root / "data" / "raw" / "train" / "labels.csv")
    train_x = matrices["train"]
    labels = labels.set_index("TRANSACTION_HASH").loc[train_x.index]
    y = labels["is_suspicious"].astype(int).to_numpy()
    groups = np.where(
        labels["scenario_id"].notna(),
        "sc:" + labels["scenario_id"].astype(str),
        "tx:" + labels.index.astype(str),
    )
    median = train_x.median()
    columns = list(train_x.columns)

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(columns=columns).fillna(median)

    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train_x, y, groups))
    positives = max(int(y.sum()), 1)
    scale = (len(y) - positives) / positives
    oof = {"boost": np.zeros(len(y)), "trees": np.zeros(len(y))}
    test_scores = {"boost": np.zeros(len(matrices["test"])), "trees": np.zeros(len(matrices["test"]))}
    prepared_train = prepare(train_x)
    prepared_test = prepare(matrices["test"])
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        boost, trees = _models(SEED + fold, scale)
        for name, model in (("boost", boost), ("trees", trees)):
            model.fit(prepared_train.iloc[fit_idx], y[fit_idx])
            oof[name][valid_idx] = model.predict_proba(prepared_train.iloc[valid_idx])[:, 1]
            test_scores[name] += model.predict_proba(prepared_test)[:, 1] / N_FOLDS
        print(f"fold {fold} done", flush=True)
    blend = 0.3 * _rank01(oof["boost"]) + 0.7 * _rank01(oof["trees"])
    print(f"grouped CV  boost {100 * average_precision_score(y, oof['boost']):.1f}")
    print(f"grouped CV  trees {100 * average_precision_score(y, oof['trees']):.1f}")
    print(f"grouped CV  blend {100 * average_precision_score(y, blend):.1f}")
    for difficulty in ("easy", "medium", "hard"):
        mask = labels["difficulty"].isin([difficulty, "normal"]).to_numpy()
        if y[mask].sum() == 0:
            continue
        print(f"  AP {difficulty:6} {100 * average_precision_score(y[mask], blend[mask]):.1f}")
    test_blend = 0.3 * _rank01(test_scores["boost"]) + 0.7 * _rank01(test_scores["trees"])
    # 'first' makes every score unique so equal scores are not tied.
    unique_scores = pd.Series(test_blend).rank(method="first").div(len(test_blend)).to_numpy()
    submission = pd.DataFrame(
        {"TRANSACTION_HASH": matrices["test"].index.astype(str), "score": unique_scores}
    )
    destination = root / "submission" / "predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    print(f"wrote {destination} ({len(submission):,} rows)")
    return destination


def main() -> int:
    score_folder(ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
