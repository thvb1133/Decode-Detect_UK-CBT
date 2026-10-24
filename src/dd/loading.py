"""Loading and label-free valuation of the synthetic ledger splits.

A *split* is a directory holding ``transactions.parquet``, ``transfers.parquet``,
``blocks.parquet`` and ``market_rates.csv``. The training split additionally
holds ``labels.csv``, which is only ever read through :func:`load_labels`.

Everything here is derived from public fields. Labels never enter feature
construction, and each split is processed in isolation: wallets and hashes are
disjoint between splits, so cross-split aggregation would be meaningless as
well as unsafe.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
MICRO = 1_000_000
HOUR = 3600


def _decode_amount(wire):
    """Return ``(asset, units)`` for an XRPL amount in JSON wire form.

    XRP arrives as a string of integer drops; an issued currency arrives as an
    object. Units are integers scaled by 1e6 in both cases, which keeps token
    amounts exact at the six decimal places the dataset uses.
    """
    if wire is None:
        return None, np.nan
    if isinstance(wire, str):
        return "XRP", int(wire)
    return f"{wire['issuer']}:{wire['currency']}", int(Decimal(wire["value"]) * MICRO)


def load_rates(split_dir: Path) -> dict:
    rates = pd.read_csv(Path(split_dir) / "market_rates.csv")
    return dict(
        zip(zip(rates["TIMESTAMP"].astype("int64"), rates["ASSET"]), rates["PRICE_MICRO_USD"].astype("int64"))
    )


def _value_usd(rates, timestamps, assets, units):
    """Value integer asset units in reference USD at the hour of each timestamp."""
    out = np.full(len(units), np.nan)
    hours = (np.asarray(timestamps, dtype="int64") // HOUR) * HOUR
    for i, (hour, asset, unit) in enumerate(zip(hours, assets, units)):
        if asset is None or unit != unit:
            continue
        price = rates.get((int(hour), asset))
        if price is None:
            continue
        out[i] = unit * price / (MICRO * MICRO)
    return out


def load_split(split_dir) -> dict:
    """Load one split and project it into an effective payment table.

    The returned ``payments`` frame carries exactly one row per transaction:
    the customer-visible sender, destination and delivered value in reference
    USD, plus the conversion metadata needed downstream.
    """
    split_dir = Path(split_dir)
    tx = pd.read_parquet(split_dir / "transactions.parquet")
    transfers = pd.read_parquet(split_dir / "transfers.parquet")
    rates = load_rates(split_dir)

    tx["TIMESTAMP"] = tx["TIMESTAMP"].astype("int64")
    transfers["TIMESTAMP"] = transfers["TIMESTAMP"].astype("int64")

    extra = [json.loads(s) for s in tx["EXTRA_TRANSACTION_FIELDS"]]
    delivered = [json.loads(s) for s in tx["DELIVERED_AMOUNT"]]

    dest = [e.get("Destination") for e in extra]
    send_max = [e.get("SendMax") for e in extra]
    has_paths = np.array([("Paths" in e) for e in extra], dtype=bool)

    dlv_asset, dlv_units = zip(*(_decode_amount(d) for d in delivered))
    snd_asset, snd_units = zip(*(_decode_amount(s) for s in send_max))

    ts = tx["TIMESTAMP"].to_numpy()
    payments = pd.DataFrame(
        {
            "TRANSACTION_HASH": tx["TRANSACTION_HASH"].to_numpy(),
            "timestamp": ts,
            "src": tx["ACCOUNT"].to_numpy(),
            "dst": np.asarray(dest, dtype=object),
            "fee_xrp": tx["FEE"].astype("float64").to_numpy(),
            "ledger_index": tx["LEDGER_INDEX"].astype("int64").to_numpy(),
            "tx_index": tx["TRANSACTION_INDEX"].astype("int64").to_numpy(),
            "delivered_asset": np.asarray(dlv_asset, dtype=object),
            "delivered_units": np.asarray(dlv_units, dtype="float64"),
            "sendmax_asset": np.asarray(snd_asset, dtype=object),
            "sendmax_units": np.asarray(snd_units, dtype="float64"),
            "has_paths": has_paths,
        }
    )
    payments["usd"] = _value_usd(rates, ts, payments["delivered_asset"], payments["delivered_units"])
    payments["sendmax_usd"] = _value_usd(rates, ts, payments["sendmax_asset"], payments["sendmax_units"])
    payments["xrp_price"] = [rates.get((int(t) // HOUR * HOUR, "XRP"), np.nan) / MICRO for t in ts]
    payments["fee_usd"] = payments["fee_xrp"] * payments["xrp_price"]

    payments = payments.sort_values(["timestamp", "ledger_index", "tx_index"], kind="mergesort").reset_index(drop=True)
    return {"transactions": tx, "transfers": transfers, "payments": payments, "rates": rates}


def load_labels(split_dir) -> pd.DataFrame:
    return pd.read_csv(Path(split_dir) / "labels.csv")
