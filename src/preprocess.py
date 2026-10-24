"""Person 1 — one clean row per transaction, with USD amounts.

Reads the organiser layout ``data/raw/train`` and ``data/raw/test`` (a
``validation`` folder is included in the graph only). XRP ``SOURCE_VALUE`` is
converted from drops. Token amounts are priced with ``market_rates.csv`` at the
hour the ledger closed. Labels are attached to the training split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph_features import resolve_column

DROPS_PER_XRP = 1_000_000
RIPPLE_EPOCH = 946_684_800  # 2000-01-01 UTC, the XRPL time origin

HASH_CANDIDATES = ["TRANSACTION_HASH", "TX_HASH", "HASH", "TRANSACTIONHASH"]
SENDER_CANDIDATES = [
    "ACCOUNT", "SENDER", "SOURCE_ADDRESS", "FROM_ADDRESS", "ACCOUNT_ADDRESS", "FROM",
]
RECEIVER_CANDIDATES = [
    "DESTINATION", "RECEIVER", "DESTINATION_ADDRESS", "TO_ADDRESS", "TO",
]
SYMBOL_CANDIDATES = [
    "SYMBOL", "CURRENCY", "SOURCE_CURRENCY", "AMOUNT_CURRENCY", "TOKEN_SYMBOL",
]
ISSUER_CANDIDATES = [
    "TOKEN_ADDRESS", "ISSUER", "SOURCE_ISSUER", "ISSUER_ADDRESS", "AMOUNT_ISSUER",
]
VALUE_CANDIDATES = ["SOURCE_VALUE", "AMOUNT", "VALUE", "AMOUNT_VALUE"]
DEST_VALUE_CANDIDATES = ["DESTINATION_VALUE", "DELIVERED_VALUE", "DEST_VALUE"]
LEDGER_CANDIDATES = ["LEDGER_INDEX", "LEDGERINDEX", "BLOCK_INDEX", "BLOCK_NUMBER"]
TIME_CANDIDATES = ["CLOSE_TIME", "TIMESTAMP", "DATETIME", "TIME", "CLOSED_AT", "DATE"]
FEE_CANDIDATES = ["FEE", "FEE_DROPS", "TRANSACTION_FEE"]
LEG_CANDIDATES = ["TRANSFER_INDEX", "LEG_INDEX", "TRANSFER_ID", "LEG", "INDEX"]
LABEL_CANDIDATES = ["LABEL", "IS_SUSPICIOUS", "SUSPICIOUS", "TARGET", "Y"]
RATE_CANDIDATES = [
    "PRICE_MICRO_USD", "USD_PRICE", "PRICE_USD", "RATE", "PRICE", "USD_RATE", "CLOSE",
]


def find_raw_root(root: Path) -> Path:
    for candidate in (root / "data" / "raw", root / "raw", root / "data"):
        if (candidate / "train").is_dir():
            return candidate
    raise SystemExit(
        "Could not find data/raw/train. Unzip the hackathon training and test "
        "downloads so the parquet files are under data/raw/train and data/raw/test."
    )


def _first_file(folder: Path, names: set[str]) -> Path | None:
    if not folder.exists():
        return None
    wanted = {name.lower() for name in names}
    for path in folder.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def _read(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_split(raw_root: Path, split: str) -> dict[str, pd.DataFrame]:
    folder = raw_root / split
    return {
        "transactions": _read(_first_file(folder, {"transactions.parquet"})),
        "transfers": _read(_first_file(folder, {"transfers.parquet"})),
        "blocks": _read(_first_file(folder, {"blocks.parquet"})),
        "rates": _read(_first_file(folder, {"market_rates.csv", "market_rates.parquet"})),
        "labels": _read(_first_file(folder, {"labels.csv", "labels.parquet"})),
    }


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None
    return resolve_column(df, candidates, required=False)


def _parse_json(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _amount_from_json(value) -> tuple[float, str, str]:
    """Return (number, symbol, issuer) from a delivered-amount field."""
    parsed = _parse_json(value)
    if isinstance(parsed, (int, float)):
        return float(parsed), "", ""
    if isinstance(parsed, str):
        try:
            return float(parsed), "", ""
        except ValueError:
            return np.nan, "", ""
    if isinstance(parsed, dict):
        if "value" in parsed:
            try:
                number = float(parsed["value"])
            except (TypeError, ValueError):
                number = np.nan
            return number, str(parsed.get("currency") or parsed.get("symbol") or ""), str(
                parsed.get("issuer") or parsed.get("token_address") or ""
            )
        if "XRP" in parsed or "xrp" in parsed:
            raw = parsed.get("XRP", parsed.get("xrp"))
            try:
                return float(raw), "XRP", ""
            except (TypeError, ValueError):
                return np.nan, "XRP", ""
    try:
        return float(value), "", ""
    except (TypeError, ValueError):
        return np.nan, "", ""


def _to_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        median = float(numeric.dropna().median())
        if median > 1e12:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median > 1.2e9:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
        if median > 1e8:
            return pd.to_datetime(numeric + RIPPLE_EPOCH, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _rates_table(rates: pd.DataFrame) -> pd.DataFrame:
    if rates.empty:
        return pd.DataFrame(columns=["hour", "symbol", "issuer", "usd_price"])
    time_col = _col(rates, TIME_CANDIDATES + ["HOUR", "RATE_HOUR"])
    price_col = _col(rates, RATE_CANDIDATES)
    symbol_col = _col(rates, SYMBOL_CANDIDATES)
    issuer_col = _col(rates, ISSUER_CANDIDATES)
    asset_col = _col(rates, ["ASSET"])
    if time_col is None or price_col is None:
        return pd.DataFrame(columns=["hour", "symbol", "issuer", "usd_price"])
    if asset_col and symbol_col is None:
        asset = rates[asset_col].astype(str)
        issuer = asset.str.split(":", n=1).str[0].where(asset.str.contains(":"), "")
        symbol = asset.str.split(":", n=1).str[-1]
    else:
        issuer = rates[issuer_col].astype(str) if issuer_col else ""
        symbol = rates[symbol_col].astype(str) if symbol_col else ""
    price = pd.to_numeric(rates[price_col], errors="coerce")
    if price_col.upper() == "PRICE_MICRO_USD":
        price = price / 1_000_000
    out = pd.DataFrame(
        {
            "hour": _to_datetime(rates[time_col]).dt.floor("h"),
            "symbol": pd.Series(symbol, index=rates.index).astype(str).str.upper(),
            "issuer": pd.Series(issuer, index=rates.index).astype(str).replace({"nan": "", "None": ""}),
            "usd_price": price,
        }
    )
    return out.dropna(subset=["hour", "usd_price"]).drop_duplicates(["hour", "symbol", "issuer"])


def _xrp_mask(symbol: pd.Series, value: pd.Series) -> pd.Series:
    is_xrp = symbol.fillna("").str.upper().isin({"XRP", "XRP.IOU", ""})
    # Native payments are stored in drops. A median above 1,000 means the
    # column has not already been divided by 1,000,000.
    median = pd.to_numeric(value, errors="coerce").where(is_xrp).median()
    return is_xrp & bool(pd.notna(median) and median > 1000)


def build_edges(transfers: pd.DataFrame, when: pd.Series, rates: pd.DataFrame) -> pd.DataFrame:
    """One row per transfer leg, with a USD amount."""
    if transfers.empty:
        return pd.DataFrame(columns=["TRANSACTION_HASH", "sender", "receiver", "amount", "time", "symbol", "leg"])
    hash_col = resolve_column(transfers, HASH_CANDIDATES, label="transaction hash")
    sender_col = resolve_column(transfers, SENDER_CANDIDATES, label="sender")
    receiver_col = resolve_column(transfers, RECEIVER_CANDIDATES, label="receiver")
    value_col = _col(transfers, VALUE_CANDIDATES)
    symbol_col = _col(transfers, SYMBOL_CANDIDATES)
    issuer_col = _col(transfers, ISSUER_CANDIDATES)
    leg_col = _col(transfers, LEG_CANDIDATES)
    dest_value_col = _col(transfers, DEST_VALUE_CANDIDATES)

    edges = pd.DataFrame(
        {
            "TRANSACTION_HASH": transfers[hash_col].astype(str),
            "sender": transfers[sender_col].astype(str),
            "receiver": transfers[receiver_col].astype(str),
            "symbol": transfers[symbol_col].astype(str).str.upper() if symbol_col else "XRP",
            "issuer": transfers[issuer_col].astype(str) if issuer_col else "",
            "leg": pd.to_numeric(transfers[leg_col], errors="coerce").fillna(0) if leg_col else np.arange(len(transfers)),
        }
    )
    raw_value = (
        pd.to_numeric(transfers[value_col], errors="coerce").reset_index(drop=True)
        if value_col
        else pd.Series(np.nan, index=edges.index)
    )
    native = raw_value.astype(float)
    mask = _xrp_mask(edges["symbol"], raw_value)
    native.loc[mask] = native.loc[mask] / DROPS_PER_XRP
    edges["native_amount"] = native.fillna(0.0)
    if dest_value_col:
        dest = pd.to_numeric(transfers[dest_value_col], errors="coerce").reset_index(drop=True).astype(float)
        dest_mask = _xrp_mask(edges["symbol"], dest)
        dest.loc[dest_mask] = dest.loc[dest_mask] / DROPS_PER_XRP
        edges["dest_native"] = dest.fillna(edges["native_amount"])
    else:
        edges["dest_native"] = edges["native_amount"]

    edges["time"] = edges["TRANSACTION_HASH"].map(when)
    edges["hour"] = pd.to_datetime(edges["time"], utc=True, errors="coerce").dt.floor("h")
    priced = rates.rename(columns={"usd_price": "price"})
    edges = edges.merge(priced, on=["hour", "symbol", "issuer"], how="left")
    if edges["price"].isna().any() and not priced.empty:
        by_symbol = priced.groupby(["hour", "symbol"], as_index=False)["price"].median()
        edges = edges.merge(by_symbol, on=["hour", "symbol"], how="left", suffixes=("", "_symbol"))
        edges["price"] = edges["price"].fillna(edges["price_symbol"])
        edges = edges.drop(columns=["price_symbol"])
    edges["price"] = edges["price"].fillna(1.0)
    edges["amount"] = edges["native_amount"] * edges["price"]
    edges["dest_amount"] = edges["dest_native"] * edges["price"]
    return edges


def _fee_xrp(transactions: pd.DataFrame) -> pd.Series:
    fee_col = _col(transactions, FEE_CANDIDATES)
    if fee_col is None:
        return pd.Series(0.0, index=transactions.index)
    fee = pd.to_numeric(transactions[fee_col], errors="coerce").fillna(0.0)
    # The ledger burns 10 to 15 drops. Values that small are already drops.
    if fee.median() >= 1:
        return fee / DROPS_PER_XRP
    return fee


def aggregate(transactions: pd.DataFrame, edges: pd.DataFrame, blocks: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    hash_col = resolve_column(transactions, HASH_CANDIDATES, label="transaction hash")
    out = pd.DataFrame({"TRANSACTION_HASH": transactions[hash_col].astype(str)})
    out["fee_xrp"] = _fee_xrp(transactions).to_numpy()

    type_col = _col(transactions, ["TRANSACTION_TYPE", "TYPE", "TX_TYPE"])
    out["tx_type"] = transactions[type_col].astype(str) if type_col else "Payment"

    ledger_col = _col(transactions, LEDGER_CANDIDATES)
    time_col = _col(transactions, TIME_CANDIDATES)
    if time_col:
        out["time"] = _to_datetime(transactions[time_col]).to_numpy()
    elif ledger_col and not blocks.empty:
        block_ledger = _col(blocks, LEDGER_CANDIDATES)
        block_time = _col(blocks, TIME_CANDIDATES)
        if block_ledger and block_time:
            lookup = pd.Series(
                _to_datetime(blocks[block_time]).to_numpy(),
                index=blocks[block_ledger].to_numpy(),
            )
            out["time"] = transactions[ledger_col].map(lookup).to_numpy()
        else:
            out["time"] = pd.NaT
    else:
        out["time"] = pd.NaT
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")

    delivered_col = _col(transactions, ["DELIVERED_AMOUNT"])
    if delivered_col:
        parsed = transactions[delivered_col].map(_amount_from_json)
        out["delivered_native"] = [item[0] for item in parsed]
    else:
        out["delivered_native"] = np.nan

    if edges.empty:
        out["paid_usd"] = 0.0
        out["delivered_usd"] = 0.0
        out["n_legs"] = 0
        out["is_conversion"] = 0
        out["n_assets"] = 0
        out["sender"] = ""
        out["receiver"] = ""
    else:
        ordered = edges.sort_values(["TRANSACTION_HASH", "leg"])
        first = ordered.drop_duplicates("TRANSACTION_HASH", keep="first").set_index("TRANSACTION_HASH")
        last = ordered.drop_duplicates("TRANSACTION_HASH", keep="last").set_index("TRANSACTION_HASH")
        stats = ordered.groupby("TRANSACTION_HASH").agg(
            n_legs=("amount", "size"),
            n_assets=("symbol", "nunique"),
            paid_usd=("amount", "first"),
            max_usd=("amount", "max"),
        )
        out = out.merge(stats, on="TRANSACTION_HASH", how="left")
        out["sender"] = out["TRANSACTION_HASH"].map(first["sender"]).fillna("")
        out["receiver"] = out["TRANSACTION_HASH"].map(last["receiver"]).fillna("")
        out["delivered_usd"] = out["TRANSACTION_HASH"].map(last["dest_amount"])
        out["is_conversion"] = (
            out["TRANSACTION_HASH"].map(first["symbol"]).fillna("")
            != out["TRANSACTION_HASH"].map(last["symbol"]).fillna("")
        ).astype(int)
        out["n_legs"] = out["n_legs"].fillna(0)
        out["n_assets"] = out["n_assets"].fillna(0)
        out["paid_usd"] = out["paid_usd"].fillna(0.0)
        out["max_usd"] = out["max_usd"].fillna(0.0)

    out["delivered_usd"] = out["delivered_usd"].fillna(out.get("delivered_native", 0)).fillna(out["paid_usd"])
    out["log_paid_usd"] = np.log1p(out["paid_usd"].clip(lower=0))
    out["log_delivered_usd"] = np.log1p(out["delivered_usd"].clip(lower=0))
    out["log_fee_xrp"] = np.log1p(out["fee_xrp"].clip(lower=0))
    out["value_ratio"] = out["delivered_usd"] / (out["paid_usd"] + 1e-9)
    out["hour_of_day"] = out["time"].dt.hour.fillna(0).astype(int)
    out["day_of_week"] = out["time"].dt.day_of_week.fillna(0).astype(int)

    if not labels.empty:
        label_hash = resolve_column(labels, HASH_CANDIDATES, label="transaction hash")
        label_col = resolve_column(labels, LABEL_CANDIDATES, label="label")
        label_map = pd.Series(pd.to_numeric(labels[label_col], errors="coerce").to_numpy(), index=labels[label_hash].astype(str))
        out["label"] = out["TRANSACTION_HASH"].map(label_map)
    return out


def build_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train rows, test rows, and the USD edge list for every split."""
    raw_root = find_raw_root(root)
    shared_rates = _rates_table(_read(_first_file(raw_root, {"market_rates.csv", "market_rates.parquet"})))
    frames = []
    edges = []
    train = test = None
    for split in ("train", "validation", "test"):
        tables = load_split(raw_root, split)
        if tables["transactions"].empty:
            continue
        rates = _rates_table(tables["rates"])
        if rates.empty:
            rates = shared_rates
        # Time is attached after a first pass so transfer pricing can use it.
        preliminary = aggregate(tables["transactions"], pd.DataFrame(), tables["blocks"], pd.DataFrame())
        when = pd.Series(preliminary["time"].to_numpy(), index=preliminary["TRANSACTION_HASH"])
        split_edges = build_edges(tables["transfers"], when, rates)
        split_edges["split"] = split
        clean = aggregate(tables["transactions"], split_edges, tables["blocks"], tables["labels"])
        clean["split"] = split
        if split == "train":
            train = clean
        elif split == "test":
            test = clean
        frames.append(clean)
        edges.append(split_edges)
    if train is None or test is None:
        raise SystemExit(f"Need both train and test transactions under {raw_root}")
    all_edges = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame()
    return train, test, all_edges
