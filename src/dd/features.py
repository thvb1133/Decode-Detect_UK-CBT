"""Transaction-level feature construction.

Every feature is a function of public ledger data for a single split. Nothing
reads labels, and nothing crosses a split boundary. The output always has
exactly one row per ``TRANSACTION_HASH``.

The feature groups mirror how an analyst actually reads a laundering case:

``amount``     what was moved, and how contrived the figure looks
``conversion`` what the payment did with assets on the way through
``account``    who the two parties are across the whole window
``pair``       what this particular relationship normally looks like
``velocity``   what else happened around this payment in time
``flow``       whether value arrived shortly before, or left shortly after
``structure``  repeated near-identical amounts, round-number splitting
``graph``      where the two parties sit in the account graph
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"10m": 600, "1h": 3600, "6h": 21600, "24h": 86400}
FORWARD_WINDOWS = {"1h": 3600, "6h": 21600, "24h": 86400}
EPS = 1e-9


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _safe_ratio(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return np.where(np.abs(b) > EPS, a / np.where(np.abs(b) > EPS, b, 1.0), np.nan)


def _trailing_zeros(units):
    """Count trailing decimal zeros of an integer amount, capped at 9.

    Deliberately round figures are a classic structuring tell, and this is the
    scale-free way to measure roundness across assets whose units differ by
    orders of magnitude.
    """
    out = np.zeros(len(units), dtype="float64")
    vals = np.asarray(units, dtype="float64")
    remaining = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
    for k in range(9):
        divisible = np.isfinite(remaining) & (np.mod(remaining, 10 ** (k + 1)) == 0)
        out += divisible.astype("float64")
        if not divisible.any():
            break
    return out


class _AccountEvents:
    """Per-account event arrays sorted by time, for window queries.

    Built once per split and queried by binary search, which keeps the
    neighbourhood features linear in the number of events they touch rather
    than quadratic in the size of the split.
    """

    def __init__(self, accounts, times, values, partners):
        accounts = np.asarray(accounts, dtype=object)
        times = np.asarray(times, dtype="float64")
        # sort by account, then by time within each account
        order = np.lexsort((times, accounts.astype(str)))
        self.accounts = accounts[order]
        self.times = times[order]
        self.values = np.asarray(values, dtype="float64")[order]
        self.partners = np.asarray(partners, dtype=object)[order]
        uniq, starts = np.unique(self.accounts.astype(str), return_index=True)
        ends = np.append(starts[1:], len(self.accounts))
        self.slices = {a: (int(s), int(e)) for a, s, e in zip(uniq, starts, ends)}

    def window(self, account, lo, hi):
        """Return ``(times, values, partners)`` for events with ``lo <= t < hi``."""
        span = self.slices.get(account)
        if span is None:
            return None
        s, e = span
        t = self.times[s:e]
        i = int(np.searchsorted(t, lo, side="left"))
        j = int(np.searchsorted(t, hi, side="left"))
        if i >= j:
            return None
        return t[i:j], self.values[s + i : s + j], self.partners[s + i : s + j]


def _build_events(payments):
    out = _AccountEvents(payments["src"], payments["timestamp"], payments["usd"], payments["dst"])
    inn = _AccountEvents(payments["dst"], payments["timestamp"], payments["usd"], payments["src"])
    return out, inn


# --------------------------------------------------------------------------
# feature groups
# --------------------------------------------------------------------------
def amount_features(payments: pd.DataFrame) -> pd.DataFrame:
    usd = payments["usd"].to_numpy()
    units = payments["delivered_units"].to_numpy()
    ts = payments["timestamp"].to_numpy()
    dt = pd.to_datetime(ts, unit="s", utc=True)

    f = pd.DataFrame(index=payments.index)
    f["amt_usd"] = usd
    f["amt_log_usd"] = np.log1p(np.clip(usd, 0, None))
    f["amt_log_units"] = np.log1p(np.clip(units, 0, None))
    f["amt_fee_usd"] = payments["fee_usd"].to_numpy()
    f["amt_fee_share"] = _safe_ratio(payments["fee_usd"], usd)
    f["amt_trailing_zeros"] = _trailing_zeros(units)
    # roundness of the USD valuation itself, which survives asset conversion
    for step in (10, 100, 1000):
        f[f"amt_round_{step}"] = (np.abs(np.remainder(usd, step)) < 1e-6).astype("float64")
    f["amt_cents"] = np.remainder(np.round(usd * 100), 100)
    f["amt_frac_part"] = usd - np.floor(usd)
    f["amt_is_integer"] = (f["amt_frac_part"] < 1e-6).astype("float64")
    # distance below the next round threshold: structuring sits just under one
    for step in (100, 1000, 10000):
        f[f"amt_below_{step}"] = _safe_ratio(np.remainder(usd, step), step)

    f["time_hour"] = dt.hour.to_numpy().astype("float64")
    f["time_dow"] = dt.day_of_week.to_numpy().astype("float64")
    f["time_minute"] = dt.minute.to_numpy().astype("float64")
    f["time_second"] = dt.second.to_numpy().astype("float64")
    f["time_is_night"] = ((f["time_hour"] < 6) | (f["time_hour"] >= 22)).astype("float64")
    f["time_is_weekend"] = (f["time_dow"] >= 5).astype("float64")
    f["time_offset_days"] = (ts - ts.min()) / 86400.0
    return f


def conversion_features(payments: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """What the payment did with assets: direct, converted, or bridged."""
    f = pd.DataFrame(index=payments.index)
    hashes = payments["TRANSACTION_HASH"].to_numpy()

    grp = transfers.groupby("TRANSACTION_HASH")
    legs = grp.size().rename("n_legs")
    assets = grp["SYMBOL"].nunique().rename("n_symbols")
    tokens = grp["TOKEN_ADDRESS"].nunique().rename("n_tokens")
    hops = grp["TO_ADDRESS"].nunique().rename("n_hops")
    is_path = grp["TRANSFER_TYPE"].apply(lambda s: float((s == "path_payment").any())).rename("is_path")
    agg = pd.concat([legs, assets, tokens, hops, is_path], axis=1).reindex(hashes)

    f["conv_n_legs"] = agg["n_legs"].to_numpy()
    f["conv_n_symbols"] = agg["n_symbols"].to_numpy()
    f["conv_n_tokens"] = agg["n_tokens"].to_numpy()
    f["conv_n_hops"] = agg["n_hops"].to_numpy()
    f["conv_is_path_payment"] = agg["is_path"].to_numpy()
    f["conv_has_paths_field"] = payments["has_paths"].to_numpy().astype("float64")
    f["conv_is_bridge"] = (agg["n_legs"].to_numpy() >= 3).astype("float64")

    sendmax = payments["sendmax_usd"].to_numpy()
    usd = payments["usd"].to_numpy()
    f["conv_has_sendmax"] = np.isfinite(sendmax).astype("float64")
    # SendMax is the spending ceiling, so the gap bounds the conversion cost
    f["conv_sendmax_spread"] = _safe_ratio(sendmax - usd, usd)
    f["conv_sendmax_log_ratio"] = np.log1p(np.clip(sendmax, 0, None)) - np.log1p(np.clip(usd, 0, None))

    delivered = payments["delivered_asset"].to_numpy()
    source = payments["sendmax_asset"].to_numpy()
    f["conv_asset_changed"] = np.array(
        [float(s is not None and s != d) for s, d in zip(source, delivered)], dtype="float64"
    )
    f["conv_delivers_xrp"] = np.array([float(a == "XRP") for a in delivered], dtype="float64")
    codes, _ = pd.factorize(pd.Series(delivered))
    f["conv_delivered_asset_code"] = codes.astype("float64")
    return f


def account_features(payments: pd.DataFrame) -> pd.DataFrame:
    """Whole-window profile of the sender and the receiver.

    This is an offline investigation over a fixed window, so using the full
    window for account profiles is allowed and is what an analyst would do.
    """
    out = payments.groupby("src").agg(
        out_count=("usd", "size"),
        out_sum=("usd", "sum"),
        out_mean=("usd", "mean"),
        out_std=("usd", "std"),
        out_max=("usd", "max"),
        out_min=("usd", "min"),
        out_median=("usd", "median"),
        out_degree=("dst", "nunique"),
        out_first=("timestamp", "min"),
        out_last=("timestamp", "max"),
    )
    inn = payments.groupby("dst").agg(
        in_count=("usd", "size"),
        in_sum=("usd", "sum"),
        in_mean=("usd", "mean"),
        in_std=("usd", "std"),
        in_max=("usd", "max"),
        in_min=("usd", "min"),
        in_median=("usd", "median"),
        in_degree=("src", "nunique"),
        in_first=("timestamp", "min"),
        in_last=("timestamp", "max"),
    )
    profile = out.join(inn, how="outer")
    for col in ("out_count", "in_count", "out_sum", "in_sum", "out_degree", "in_degree"):
        profile[col] = profile[col].fillna(0.0)

    profile["life_span"] = (
        profile[["out_last", "in_last"]].max(axis=1) - profile[["out_first", "in_first"]].min(axis=1)
    )
    profile["total_count"] = profile["out_count"] + profile["in_count"]
    profile["total_usd"] = profile["out_sum"].fillna(0) + profile["in_sum"].fillna(0)
    # a pure conduit receives and forwards nearly the same value
    profile["flow_through"] = _safe_ratio(
        np.minimum(profile["in_sum"].fillna(0), profile["out_sum"].fillna(0)),
        np.maximum(profile["in_sum"].fillna(0), profile["out_sum"].fillna(0)),
    )
    profile["flow_imbalance"] = _safe_ratio(
        profile["in_sum"].fillna(0) - profile["out_sum"].fillna(0),
        profile["in_sum"].fillna(0) + profile["out_sum"].fillna(0),
    )
    profile["rate_per_hour"] = _safe_ratio(profile["total_count"], np.maximum(profile["life_span"], 1.0) / 3600.0)
    profile["out_cv"] = _safe_ratio(profile["out_std"], profile["out_mean"])
    profile["in_cv"] = _safe_ratio(profile["in_std"], profile["in_mean"])
    profile["degree_ratio"] = _safe_ratio(profile["out_degree"], profile["in_degree"] + 1.0)

    keep = [
        "out_count", "in_count", "out_sum", "in_sum", "out_mean", "in_mean",
        "out_std", "in_std", "out_max", "in_max", "out_median", "in_median",
        "out_degree", "in_degree", "life_span", "total_count", "total_usd",
        "flow_through", "flow_imbalance", "rate_per_hour", "out_cv", "in_cv", "degree_ratio",
    ]
    profile = profile[keep]

    src = profile.reindex(payments["src"].to_numpy())
    dst = profile.reindex(payments["dst"].to_numpy())
    f = pd.DataFrame(index=payments.index)
    for col in keep:
        f[f"src_{col}"] = src[col].to_numpy()
        f[f"dst_{col}"] = dst[col].to_numpy()

    usd = payments["usd"].to_numpy()
    f["rel_amt_vs_src_mean"] = _safe_ratio(usd, src["out_mean"].to_numpy())
    f["rel_amt_vs_src_max"] = _safe_ratio(usd, src["out_max"].to_numpy())
    f["rel_amt_vs_src_median"] = _safe_ratio(usd, src["out_median"].to_numpy())
    f["rel_amt_vs_dst_mean"] = _safe_ratio(usd, dst["in_mean"].to_numpy())
    f["rel_amt_vs_dst_max"] = _safe_ratio(usd, dst["in_max"].to_numpy())
    f["rel_amt_z_src"] = _safe_ratio(usd - src["out_mean"].to_numpy(), src["out_std"].to_numpy())
    f["rel_amt_z_dst"] = _safe_ratio(usd - dst["in_mean"].to_numpy(), dst["in_std"].to_numpy())
    f["rel_amt_share_src_out"] = _safe_ratio(usd, src["out_sum"].to_numpy())
    f["rel_amt_share_dst_in"] = _safe_ratio(usd, dst["in_sum"].to_numpy())
    f["rel_degree_gap"] = src["out_degree"].to_numpy() - dst["in_degree"].to_numpy()
    f["rel_activity_gap"] = np.log1p(src["total_count"].to_numpy()) - np.log1p(dst["total_count"].to_numpy())
    return f


def pair_features(payments: pd.DataFrame) -> pd.DataFrame:
    """What this specific sender-receiver relationship normally looks like.

    A large payment into an established commercial relationship is ordinary;
    the same payment on a brand-new edge is not.
    """
    df = payments[["src", "dst", "timestamp", "usd"]].copy()
    df["pair"] = df["src"].astype(str) + ">" + df["dst"].astype(str)
    grp = df.groupby("pair")["usd"]
    stats = pd.DataFrame(
        {
            "pair_count": grp.size(),
            "pair_sum": grp.sum(),
            "pair_mean": grp.mean(),
            "pair_std": grp.std(),
            "pair_max": grp.max(),
        }
    )
    stats["pair_cv"] = _safe_ratio(stats["pair_std"], stats["pair_mean"])
    span = df.groupby("pair")["timestamp"].agg(["min", "max"])
    stats["pair_span"] = span["max"] - span["min"]

    joined = stats.reindex(df["pair"].to_numpy())
    f = pd.DataFrame(index=payments.index)
    for col in stats.columns:
        f[col] = joined[col].to_numpy()

    ordered = df.sort_values("timestamp", kind="mergesort")
    seq = ordered.groupby("pair").cumcount()
    prev_t = ordered.groupby("pair")["timestamp"].shift(1)
    prev_v = ordered.groupby("pair")["usd"].shift(1)
    f["pair_seq"] = seq.reindex(payments.index).to_numpy().astype("float64")
    f["pair_is_first"] = (f["pair_seq"] == 0).astype("float64")
    f["pair_dt_prev"] = (ordered["timestamp"] - prev_t).reindex(payments.index).to_numpy()
    f["pair_amt_vs_prev"] = _safe_ratio(payments["usd"].to_numpy(), prev_v.reindex(payments.index).to_numpy())
    f["pair_amt_vs_mean"] = _safe_ratio(payments["usd"].to_numpy(), f["pair_mean"].to_numpy())
    f["pair_reciprocal"] = (
        pd.Series(df["dst"].astype(str) + ">" + df["src"].astype(str))
        .isin(set(df["pair"]))
        .to_numpy()
        .astype("float64")
    )
    f["pair_share_of_src"] = _safe_ratio(f["pair_sum"].to_numpy(), payments.groupby("src")["usd"].transform("sum").to_numpy())
    f["pair_share_of_dst"] = _safe_ratio(f["pair_sum"].to_numpy(), payments.groupby("dst")["usd"].transform("sum").to_numpy())
    return f


def velocity_features(payments: pd.DataFrame, out_events, in_events) -> pd.DataFrame:
    """Local burst activity around each payment, for both parties."""
    n = len(payments)
    ts = payments["timestamp"].to_numpy(dtype="float64")
    src = payments["src"].to_numpy()
    dst = payments["dst"].to_numpy()

    cols = {}
    for tag, width in WINDOWS.items():
        cols[f"vel_src_out_{tag}"] = np.zeros(n)
        cols[f"vel_src_out_fanout_{tag}"] = np.zeros(n)
        cols[f"vel_src_out_usd_{tag}"] = np.zeros(n)
        cols[f"vel_dst_in_{tag}"] = np.zeros(n)
        cols[f"vel_dst_in_fanin_{tag}"] = np.zeros(n)
        cols[f"vel_dst_in_usd_{tag}"] = np.zeros(n)

    for i in range(n):
        t = ts[i]
        for tag, width in WINDOWS.items():
            got = out_events.window(src[i], t - width, t + width)
            if got is not None:
                _, vals, partners = got
                cols[f"vel_src_out_{tag}"][i] = len(vals)
                cols[f"vel_src_out_usd_{tag}"][i] = np.nansum(vals)
                cols[f"vel_src_out_fanout_{tag}"][i] = len(set(partners.tolist()))
            got = in_events.window(dst[i], t - width, t + width)
            if got is not None:
                _, vals, partners = got
                cols[f"vel_dst_in_{tag}"][i] = len(vals)
                cols[f"vel_dst_in_usd_{tag}"][i] = np.nansum(vals)
                cols[f"vel_dst_in_fanin_{tag}"][i] = len(set(partners.tolist()))

    f = pd.DataFrame(cols, index=payments.index)
    usd = payments["usd"].to_numpy()
    for tag in WINDOWS:
        # fan-out and fan-in bursts are the defining shape of the smurfing motifs
        f[f"vel_src_fanout_rate_{tag}"] = _safe_ratio(f[f"vel_src_out_fanout_{tag}"], f[f"vel_src_out_{tag}"])
        f[f"vel_dst_fanin_rate_{tag}"] = _safe_ratio(f[f"vel_dst_in_fanin_{tag}"], f[f"vel_dst_in_{tag}"])
        f[f"vel_amt_share_src_{tag}"] = _safe_ratio(usd, f[f"vel_src_out_usd_{tag}"])
        f[f"vel_amt_share_dst_{tag}"] = _safe_ratio(usd, f[f"vel_dst_in_usd_{tag}"])

    ordered = payments.sort_values("timestamp", kind="mergesort")
    f["vel_src_dt_prev"] = (
        (ordered["timestamp"] - ordered.groupby("src")["timestamp"].shift(1)).reindex(payments.index).to_numpy()
    )
    f["vel_src_dt_next"] = (
        (ordered.groupby("src")["timestamp"].shift(-1) - ordered["timestamp"]).reindex(payments.index).to_numpy()
    )
    f["vel_dst_dt_prev"] = (
        (ordered["timestamp"] - ordered.groupby("dst")["timestamp"].shift(1)).reindex(payments.index).to_numpy()
    )
    f["vel_dst_dt_next"] = (
        (ordered.groupby("dst")["timestamp"].shift(-1) - ordered["timestamp"]).reindex(payments.index).to_numpy()
    )
    return f


def flow_features(payments: pd.DataFrame, out_events, in_events) -> pd.DataFrame:
    """Layering evidence: value arriving just before, and leaving just after.

    The strongest single signal for chain, stack and multi-stage motifs is an
    amount landing on an account and a near-identical amount leaving it a short
    time later. ``match`` is measured in log space so it is symmetric in the
    direction of the discrepancy and insensitive to scale.
    """
    n = len(payments)
    ts = payments["timestamp"].to_numpy(dtype="float64")
    usd = payments["usd"].to_numpy(dtype="float64")
    src = payments["src"].to_numpy()
    dst = payments["dst"].to_numpy()
    log_amt = np.log1p(np.clip(usd, 0, None))

    cols = {}
    for tag in FORWARD_WINDOWS:
        for name in (
            "fwd_count", "fwd_usd", "fwd_ratio", "fwd_best_match", "fwd_best_dt",
            "fwd_fanout", "fwd_latency", "fwd_within_5pct", "fwd_within_1pct",
            "bwd_count", "bwd_usd", "bwd_ratio", "bwd_best_match", "bwd_best_dt",
            "bwd_fanin", "bwd_latency", "bwd_within_5pct", "bwd_within_1pct",
        ):
            cols[f"flow_{name}_{tag}"] = np.full(n, np.nan)

    for i in range(n):
        t, a, la = ts[i], usd[i], log_amt[i]
        for tag, width in FORWARD_WINDOWS.items():
            # value leaving the receiver after this payment lands
            got = out_events.window(dst[i], t + 1e-6, t + width)
            if got is None:
                cols[f"flow_fwd_count_{tag}"][i] = 0.0
                cols[f"flow_fwd_usd_{tag}"][i] = 0.0
                cols[f"flow_fwd_ratio_{tag}"][i] = 0.0
                cols[f"flow_fwd_within_5pct_{tag}"][i] = 0.0
                cols[f"flow_fwd_within_1pct_{tag}"][i] = 0.0
            else:
                tt, vals, partners = got
                diff = np.abs(np.log1p(np.clip(vals, 0, None)) - la)
                k = int(np.nanargmin(diff)) if np.isfinite(diff).any() else 0
                cols[f"flow_fwd_count_{tag}"][i] = len(vals)
                cols[f"flow_fwd_usd_{tag}"][i] = np.nansum(vals)
                cols[f"flow_fwd_ratio_{tag}"][i] = np.nansum(vals) / (a + EPS)
                cols[f"flow_fwd_best_match_{tag}"][i] = diff[k]
                cols[f"flow_fwd_best_dt_{tag}"][i] = tt[k] - t
                cols[f"flow_fwd_fanout_{tag}"][i] = len(set(partners.tolist()))
                cols[f"flow_fwd_latency_{tag}"][i] = tt[0] - t
                rel = np.abs(_safe_ratio(vals - a, a))
                cols[f"flow_fwd_within_5pct_{tag}"][i] = float(np.nansum(rel <= 0.05))
                cols[f"flow_fwd_within_1pct_{tag}"][i] = float(np.nansum(rel <= 0.01))

            # value arriving at the sender shortly before it moves on
            got = in_events.window(src[i], t - width, t - 1e-6)
            if got is None:
                cols[f"flow_bwd_count_{tag}"][i] = 0.0
                cols[f"flow_bwd_usd_{tag}"][i] = 0.0
                cols[f"flow_bwd_ratio_{tag}"][i] = 0.0
                cols[f"flow_bwd_within_5pct_{tag}"][i] = 0.0
                cols[f"flow_bwd_within_1pct_{tag}"][i] = 0.0
            else:
                tt, vals, partners = got
                diff = np.abs(np.log1p(np.clip(vals, 0, None)) - la)
                k = int(np.nanargmin(diff)) if np.isfinite(diff).any() else 0
                cols[f"flow_bwd_count_{tag}"][i] = len(vals)
                cols[f"flow_bwd_usd_{tag}"][i] = np.nansum(vals)
                cols[f"flow_bwd_ratio_{tag}"][i] = a / (np.nansum(vals) + EPS)
                cols[f"flow_bwd_best_match_{tag}"][i] = diff[k]
                cols[f"flow_bwd_best_dt_{tag}"][i] = t - tt[k]
                cols[f"flow_bwd_fanin_{tag}"][i] = len(set(partners.tolist()))
                cols[f"flow_bwd_latency_{tag}"][i] = t - tt[-1]
                rel = np.abs(_safe_ratio(vals - a, a))
                cols[f"flow_bwd_within_5pct_{tag}"][i] = float(np.nansum(rel <= 0.05))
                cols[f"flow_bwd_within_1pct_{tag}"][i] = float(np.nansum(rel <= 0.01))

    f = pd.DataFrame(cols, index=payments.index)
    for tag in FORWARD_WINDOWS:
        # a payment that is both the tail of one hop and the head of the next
        f[f"flow_chain_score_{tag}"] = np.exp(-f[f"flow_bwd_best_match_{tag}"]) * np.exp(
            -f[f"flow_fwd_best_match_{tag}"]
        )
        f[f"flow_pass_through_{tag}"] = np.minimum(f[f"flow_fwd_ratio_{tag}"], 1.0 / (f[f"flow_bwd_ratio_{tag}"] + EPS))
        f[f"flow_peel_{tag}"] = _safe_ratio(f[f"flow_fwd_usd_{tag}"], f[f"flow_fwd_count_{tag}"])
    return f


def structuring_features(payments: pd.DataFrame, out_events, in_events) -> pd.DataFrame:
    """Repeated near-identical amounts, the signature of smurfing and stacking."""
    n = len(payments)
    ts = payments["timestamp"].to_numpy(dtype="float64")
    usd = payments["usd"].to_numpy(dtype="float64")
    src = payments["src"].to_numpy()
    dst = payments["dst"].to_numpy()
    log_amt = np.log1p(np.clip(usd, 0, None))

    twins_out = np.zeros(n)
    twins_in = np.zeros(n)
    spread_out = np.full(n, np.nan)
    spread_in = np.full(n, np.nan)
    rank_out = np.full(n, np.nan)
    rank_in = np.full(n, np.nan)
    width = WINDOWS["24h"]

    for i in range(n):
        t, la = ts[i], log_amt[i]
        got = out_events.window(src[i], t - width, t + width)
        if got is not None:
            _, vals, _ = got
            lv = np.log1p(np.clip(vals, 0, None))
            twins_out[i] = float(np.nansum(np.abs(lv - la) < 0.05)) - 1.0
            spread_out[i] = float(np.nanstd(lv))
            rank_out[i] = float(np.nanmean(vals <= usd[i]))
        got = in_events.window(dst[i], t - width, t + width)
        if got is not None:
            _, vals, _ = got
            lv = np.log1p(np.clip(vals, 0, None))
            twins_in[i] = float(np.nansum(np.abs(lv - la) < 0.05)) - 1.0
            spread_in[i] = float(np.nanstd(lv))
            rank_in[i] = float(np.nanmean(vals <= usd[i]))

    f = pd.DataFrame(index=payments.index)
    f["str_twins_src_24h"] = twins_out
    f["str_twins_dst_24h"] = twins_in
    f["str_amt_spread_src_24h"] = spread_out
    f["str_amt_spread_dst_24h"] = spread_in
    f["str_amt_rank_src_24h"] = rank_out
    f["str_amt_rank_dst_24h"] = rank_in
    f["str_uniformity_src"] = np.exp(-f["str_amt_spread_src_24h"].to_numpy())
    f["str_uniformity_dst"] = np.exp(-f["str_amt_spread_dst_24h"].to_numpy())
    return f


def graph_features(payments: pd.DataFrame, cycle_length=4, max_cycles=200_000) -> pd.DataFrame:
    """Position of both parties in the account graph for the window.

    The account graph is small (hundreds of nodes), so exact centralities and
    short-cycle enumeration are affordable and preferable to approximations.
    """
    import networkx as nx

    edges = (
        payments.groupby(["src", "dst"])
        .agg(weight=("usd", "sum"), count=("usd", "size"))
        .reset_index()
    )
    g = nx.DiGraph()
    for s, d, w, c in edges[["src", "dst", "weight", "count"]].itertuples(index=False):
        g.add_edge(s, d, weight=float(w) if np.isfinite(w) else 0.0, count=float(c))
    undirected = g.to_undirected()

    pagerank = nx.pagerank(g, weight="weight")
    pagerank_u = nx.pagerank(g.reverse(copy=True), weight="weight")
    betweenness = nx.betweenness_centrality(g, weight=None)
    clustering = nx.clustering(undirected)
    core = nx.core_number(undirected)
    try:
        hubs, authorities = nx.hits(g, max_iter=200)
    except Exception:
        hubs = authorities = {node: np.nan for node in g}

    scc = {node: i for i, comp in enumerate(nx.strongly_connected_components(g)) for node in comp}
    scc_size = {}
    for comp in nx.strongly_connected_components(g):
        for node in comp:
            scc_size[node] = len(comp)
    communities = nx.community.greedy_modularity_communities(undirected)
    community = {node: i for i, comp in enumerate(communities) for node in comp}

    # Short directed cycles: the cycle motif, and the return path of a round
    # trip. Enumeration is capped because a dense hub-and-spoke graph has
    # combinatorially many short cycles and the counts saturate long before
    # the enumeration finishes.
    cycle_nodes: dict = {}
    cycle_edges: set = set()
    for seen, cycle in enumerate(nx.simple_cycles(g, length_bound=cycle_length)):
        if seen >= max_cycles:
            break
        for node in cycle:
            cycle_nodes[node] = cycle_nodes.get(node, 0) + 1
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            cycle_edges.add((a, b))

    nodes = list(g.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype="float32")
    for s, d in g.edges():
        adj[index[s], index[d]] = 1.0
    two_hop = adj @ adj
    neighbours = {node: set(undirected.neighbors(node)) for node in nodes}

    src = payments["src"].to_numpy()
    dst = payments["dst"].to_numpy()
    f = pd.DataFrame(index=payments.index)

    def lookup(table, keys, default=np.nan):
        return np.array([table.get(k, default) for k in keys], dtype="float64")

    for tag, keys in (("src", src), ("dst", dst)):
        f[f"g_{tag}_pagerank"] = lookup(pagerank, keys)
        f[f"g_{tag}_pagerank_rev"] = lookup(pagerank_u, keys)
        f[f"g_{tag}_betweenness"] = lookup(betweenness, keys)
        f[f"g_{tag}_clustering"] = lookup(clustering, keys)
        f[f"g_{tag}_core"] = lookup(core, keys)
        f[f"g_{tag}_hub"] = lookup(hubs, keys)
        f[f"g_{tag}_authority"] = lookup(authorities, keys)
        f[f"g_{tag}_scc_size"] = lookup(scc_size, keys, 1.0)
        f[f"g_{tag}_in_cycle"] = lookup(cycle_nodes, keys, 0.0)

    f["g_same_scc"] = (lookup(scc, src, -1.0) == lookup(scc, dst, -2.0)).astype("float64")
    f["g_same_community"] = (lookup(community, src, -1.0) == lookup(community, dst, -2.0)).astype("float64")
    f["g_edge_in_cycle"] = np.array([float((s, d) in cycle_edges) for s, d in zip(src, dst)])
    f["g_reciprocal_edge"] = np.array([float(g.has_edge(d, s)) for s, d in zip(src, dst)])
    f["g_common_neighbours"] = np.array(
        [float(len(neighbours.get(s, set()) & neighbours.get(d, set()))) for s, d in zip(src, dst)]
    )
    union = np.array([float(len(neighbours.get(s, set()) | neighbours.get(d, set()))) for s, d in zip(src, dst)])
    f["g_jaccard"] = _safe_ratio(f["g_common_neighbours"].to_numpy(), union)
    f["g_two_hop_paths"] = np.array(
        [two_hop[index[s], index[d]] if s in index and d in index else np.nan for s, d in zip(src, dst)]
    )
    f["g_pagerank_gap"] = f["g_src_pagerank"] - f["g_dst_pagerank"]
    f["g_core_gap"] = f["g_src_core"] - f["g_dst_core"]
    f["g_hub_authority"] = f["g_src_hub"] * f["g_dst_authority"]

    edge_weight = edges.set_index(["src", "dst"])
    pairs = pd.MultiIndex.from_arrays([src, dst])
    f["g_edge_weight"] = edge_weight["weight"].reindex(pairs).to_numpy()
    f["g_edge_count"] = edge_weight["count"].reindex(pairs).to_numpy()
    return f


def build_features(split: dict) -> pd.DataFrame:
    """Assemble every feature group into one row per ``TRANSACTION_HASH``."""
    payments = split["payments"]
    out_events, in_events = _build_events(payments)

    parts = [
        amount_features(payments),
        conversion_features(payments, split["transfers"]),
        account_features(payments),
        pair_features(payments),
        velocity_features(payments, out_events, in_events),
        flow_features(payments, out_events, in_events),
        structuring_features(payments, out_events, in_events),
        graph_features(payments),
    ]
    features = pd.concat(parts, axis=1)
    features.insert(0, "TRANSACTION_HASH", payments["TRANSACTION_HASH"].to_numpy())
    features = features.replace([np.inf, -np.inf], np.nan)
    assert features["TRANSACTION_HASH"].is_unique, "feature table must have one row per transaction"
    return features
