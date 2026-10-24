"""Flow features for the cases a single degree count misses.

The 42.5 leaderboard run found easy fan-ins and almost none of the medium or
hard flows. Those cases are a chain: money arrives, pauses, and leaves in a
similar amount. This module measures that chain from the transfer list only.
It never reads labels.

Wallet statistics use the whole supplied window, because the test file is part
of the ledger the model is allowed to see. They do not use anything from a
label file.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86400.0


def _epoch(times: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(times, utc=True, errors="coerce")
    return parsed.astype("int64").to_numpy() / 1e9


def _max_distinct(times: np.ndarray, keys: np.ndarray, window: float) -> int:
    if len(times) == 0 or not np.isfinite(times).any():
        return 0
    order = np.argsort(times)
    times = times[order]
    keys = keys[order]
    counts: dict = {}
    left = 0
    best = 0
    for right, moment in enumerate(times):
        if not np.isfinite(moment):
            continue
        key = keys[right]
        counts[key] = counts.get(key, 0) + 1
        while times[left] < moment - window:
            previous = keys[left]
            counts[previous] -= 1
            if counts[previous] == 0:
                del counts[previous]
            left += 1
        best = max(best, len(counts))
    return best


def _hold_stats(in_times: np.ndarray, in_amounts: np.ndarray, out_times: np.ndarray, out_amounts: np.ndarray) -> tuple[float, float]:
    """Median wait from an incoming payment to the next similar outgoing one."""
    if len(in_times) == 0 or len(out_times) == 0:
        return -1.0, 0.0
    order = np.argsort(out_times)
    out_times = out_times[order]
    out_amounts = out_amounts[order]
    delays = []
    matched = 0
    for moment, amount in zip(in_times, in_amounts):
        if not np.isfinite(moment):
            continue
        position = int(np.searchsorted(out_times, moment, side="left"))
        if position >= len(out_times):
            continue
        delay = float(out_times[position] - moment)
        delays.append(delay)
        later = out_amounts[position]
        if amount > 0 and delay <= 600 and abs(later - amount) / amount <= 0.15:
            matched += 1
    if not delays:
        return -1.0, 0.0
    return float(np.median(delays)), matched / len(in_times)


def wallet_flow_table(edges: pd.DataFrame) -> pd.DataFrame:
    times = _epoch(edges["time"]) if "time" in edges.columns else np.full(len(edges), np.nan)
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for sender, receiver, amount, moment in zip(
        edges["sender"].astype(str), edges["receiver"].astype(str), edges["amount"].to_numpy(dtype=float), times
    ):
        if sender == receiver:
            continue
        outgoing[sender].append((moment, receiver, amount))
        incoming[receiver].append((moment, sender, amount))

    wallets = set(incoming) | set(outgoing)
    only_out = {}
    rows = []
    for wallet in wallets:
        ins = incoming.get(wallet, [])
        outs = outgoing.get(wallet, [])
        in_keys = np.array([item[1] for item in ins], dtype=object)
        out_keys = np.array([item[1] for item in outs], dtype=object)
        in_times = np.array([item[0] for item in ins], dtype=float)
        out_times = np.array([item[0] for item in outs], dtype=float)
        in_amounts = np.array([item[2] for item in ins], dtype=float)
        out_amounts = np.array([item[2] for item in outs], dtype=float)
        in_sum = float(in_amounts.sum()) if len(in_amounts) else 0.0
        out_sum = float(out_amounts.sum()) if len(out_amounts) else 0.0
        in_wallets = len(set(in_keys.tolist())) if len(in_keys) else 0
        out_wallets = len(set(out_keys.tolist())) if len(out_keys) else 0
        hold, fast_share = _hold_stats(in_times, in_amounts, out_times, out_amounts)
        max_out = float(out_amounts.max()) if len(out_amounts) else 0.0
        retention = 1.0 - out_sum / in_sum if in_sum > 0 else 0.0
        exit_ratio = max_out / in_sum if in_sum > 0 else 0.0
        in_burst_1h = _max_distinct(in_times, in_keys, HOUR)
        out_burst_1h = _max_distinct(out_times, out_keys, HOUR)
        if out_wallets == 1:
            only_out[wallet] = out_keys[0]
        rows.append(
            {
                "wallet": wallet,
                "in_wallets": in_wallets,
                "out_wallets": out_wallets,
                "in_sum": in_sum,
                "out_sum": out_sum,
                "retention": retention,
                "exit_ratio": exit_ratio,
                "in_burst_1h": in_burst_1h,
                "out_burst_1h": out_burst_1h,
                "in_burst_24h": _max_distinct(in_times, in_keys, DAY),
                "out_burst_24h": _max_distinct(out_times, out_keys, DAY),
                "median_hold_s": hold,
                "fast_forward_share": fast_share,
                "is_collector": int(in_burst_1h >= 5 and out_wallets <= 2 and exit_ratio >= 0.6),
                "is_distributor": int(out_burst_1h >= 5 and in_wallets <= 2),
                "is_conduit": int(
                    in_wallets == 1 and out_wallets == 1 and in_sum > 0 and 0.85 <= out_sum / in_sum <= 1.15
                ),
            }
        )

    table = pd.DataFrame(rows).set_index("wallet")
    depths = {}

    def depth(wallet: str, seen: set[str]) -> int:
        if wallet in depths:
            return depths[wallet]
        if wallet in seen or not int(table.at[wallet, "is_conduit"]) or wallet not in only_out:
            depths[wallet] = 0
            return 0
        seen.add(wallet)
        nxt = only_out[wallet]
        value = 1 + (depth(nxt, seen) if nxt in table.index else 0)
        seen.remove(wallet)
        depths[wallet] = min(value, 8)
        return depths[wallet]

    table["layer_depth"] = [depth(wallet, set()) for wallet in table.index]
    return table


def build_flow_features(edges: pd.DataFrame) -> pd.DataFrame:
    """One row per TRANSACTION_HASH, taken from its largest-value leg."""
    required = {"TRANSACTION_HASH", "sender", "receiver", "amount"}
    missing = required - set(edges.columns)
    if missing:
        raise KeyError(f"edges are missing {sorted(missing)}")
    if edges.empty:
        return pd.DataFrame(columns=["TRANSACTION_HASH"])
    wallets = wallet_flow_table(edges)
    primary = edges.sort_values("amount", ascending=False).drop_duplicates("TRANSACTION_HASH")
    sender = wallets.add_prefix("sender_").reindex(primary["sender"].astype(str).to_numpy())
    receiver = wallets.add_prefix("receiver_").reindex(primary["receiver"].astype(str).to_numpy())
    out = pd.concat(
        [
            primary[["TRANSACTION_HASH"]].reset_index(drop=True),
            sender.reset_index(drop=True),
            receiver.reset_index(drop=True),
        ],
        axis=1,
    )
    out["flow_layer_depth"] = out[["sender_layer_depth", "receiver_layer_depth"]].max(axis=1)
    out["flow_burst"] = out[["sender_out_burst_1h", "receiver_in_burst_1h"]].max(axis=1)
    numeric = out.columns.drop("TRANSACTION_HASH")
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out
