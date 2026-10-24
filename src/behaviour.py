"""Person 4 — behaviour features from the wallet's own history.

Every count uses only transfers at or before the transaction being scored.
Labels are never read. The same function is applied to train and test, so the
features exist for both splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86400.0


def _epoch(times: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(times, utc=True, errors="coerce")
    return parsed.astype("int64").to_numpy() / 1e9


def build_behaviour_features(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per TRANSACTION_HASH. ``clean`` must contain sender, receiver, time, paid_usd, is_conversion."""
    rows = clean.loc[:, ["TRANSACTION_HASH", "sender", "receiver", "time", "paid_usd", "is_conversion"]].copy()
    rows["time_s"] = _epoch(rows["time"])
    rows = rows.sort_values(["time_s", "TRANSACTION_HASH"]).reset_index(drop=True)

    events = pd.concat(
        [
            rows.assign(wallet=rows["sender"], direction=-1, counterparty=rows["receiver"]),
            rows.assign(wallet=rows["receiver"], direction=1, counterparty=rows["sender"]),
        ],
        ignore_index=True,
    )
    events = events.sort_values(["wallet", "time_s", "TRANSACTION_HASH"]).reset_index(drop=True)

    feature_rows = []
    for wallet, group in events.groupby("wallet", sort=False):
        times = group["time_s"].to_numpy()
        usd = group["paid_usd"].to_numpy(dtype=float)
        direction = group["direction"].to_numpy()
        conversion = group["is_conversion"].to_numpy(dtype=float)
        counterparties = group["counterparty"].to_numpy()
        hashes = group["TRANSACTION_HASH"].to_numpy()
        roles = np.where(direction < 0, "sender", "receiver")
        inflow = np.cumsum(np.where(direction > 0, usd, 0.0))
        outflow = np.cumsum(np.where(direction < 0, usd, 0.0))
        conversions = np.cumsum(conversion)
        value_sum = np.cumsum(usd)
        value_sq = np.cumsum(usd * usd)
        finite = np.isfinite(times)
        for index in range(len(group)):
            if not finite[index]:
                feature_rows.append((hashes[index], roles[index], _zeros()))
                continue
            now = times[index]
            start_hour = int(np.searchsorted(times, now - HOUR, side="left"))
            start_day = int(np.searchsorted(times, now - DAY, side="left"))
            past_n = index - start_day
            hour_n = index - start_hour
            day_in = _window_sum(inflow, start_day, index)
            day_out = _window_sum(outflow, start_day, index)
            if index:
                mean = value_sum[index - 1] / index
                variance = value_sq[index - 1] / index - mean * mean
                deviation = (usd[index] - mean) / (np.sqrt(max(variance, 0.0)) + 1e-6)
                gap = now - times[index - 1]
            else:
                deviation = 0.0
                gap = np.nan
            recent_start = max(start_day, index - 40)
            recent = usd[recent_start:index]
            same_amount = 0.0
            if len(recent) >= 3 and np.mean(recent) > 0:
                same_amount = float(np.std(recent) / (np.mean(recent) + 1e-6))
            rapid = 0
            if index and direction[index] < 0 and direction[index - 1] > 0 and gap <= 120:
                previous = usd[index - 1]
                if previous > 0 and abs(usd[index] - previous) / previous < 0.05:
                    rapid = 1
            feature_rows.append(
                (
                    hashes[index],
                    roles[index],
                    {
                        "tx_count_1h": hour_n,
                        "tx_count_24h": past_n,
                        "total_inflow_24h": max(day_in, 0.0),
                        "total_outflow_24h": max(day_out, 0.0),
                        "unique_counterparties": (
                            len(set(counterparties[recent_start:index].tolist())) if past_n else 0
                        ),
                        "conversion_count": float(conversions[index - 1]) if index else 0.0,
                        "avg_time_between_tx": float(np.mean(np.diff(times[max(0, index - 10): index + 1]))) if index else 0.0,
                        "amount_deviation": float(deviation) if np.isfinite(deviation) else 0.0,
                        "seconds_since_prev": float(gap) if np.isfinite(gap) else -1.0,
                        "amount_cv_24h": same_amount,
                        "rapid_passthrough": rapid,
                        "is_new_wallet": int(index == 0),
                    },
                )
            )

    long = pd.DataFrame(
        [{"TRANSACTION_HASH": tx, "role": role, **values} for tx, role, values in feature_rows]
    )
    # A wallet can appear twice in one transaction (self-payment). Keep the sender view.
    long = long.drop_duplicates(["TRANSACTION_HASH", "role"], keep="first")
    sender = long[long["role"] == "sender"].drop(columns=["role"]).set_index("TRANSACTION_HASH")
    receiver = long[long["role"] == "receiver"].drop(columns=["role"]).set_index("TRANSACTION_HASH")
    sender.columns = [f"sender_{column}" for column in sender.columns]
    receiver.columns = [f"receiver_{column}" for column in receiver.columns]
    combined = sender.join(receiver, how="outer")
    # The names from the team brief, taken from whichever side is more active.
    for column in [
        "tx_count_1h", "tx_count_24h", "total_inflow_24h", "total_outflow_24h",
        "unique_counterparties", "conversion_count", "avg_time_between_tx", "amount_deviation",
    ]:
        combined[column] = combined[[f"sender_{column}", f"receiver_{column}"]].max(axis=1)
    combined = combined.reset_index()
    numeric = combined.columns.drop("TRANSACTION_HASH")
    combined[numeric] = combined[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return combined


def _window_sum(cumsum: np.ndarray, start: int, index: int) -> float:
    """Sum of a cumsum over the half-open interval [start, index)."""
    if index <= start:
        return 0.0
    total = float(cumsum[index - 1])
    if start > 0:
        total -= float(cumsum[start - 1])
    return max(total, 0.0)


def _zeros() -> dict:
    return {
        "tx_count_1h": 0,
        "tx_count_24h": 0,
        "total_inflow_24h": 0.0,
        "total_outflow_24h": 0.0,
        "unique_counterparties": 0,
        "conversion_count": 0.0,
        "avg_time_between_tx": 0.0,
        "amount_deviation": 0.0,
        "seconds_since_prev": -1.0,
        "amount_cv_24h": 0.0,
        "rapid_passthrough": 0,
        "is_new_wallet": 1,
    }
