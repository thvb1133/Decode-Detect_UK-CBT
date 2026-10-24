"""End-to-end smoke test on a tiny ledger shaped like the organiser files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_all import run  # noqa: E402


def _write_split(folder: Path, start: int, count: int, label: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    hashes = [f"TX{i:08d}" for i in range(start, start + count)]
    ledgers = list(range(start, start + count))
    # The last quarter are fan-in payments into one collector, and those are the
    # labelled suspicious rows. They sit late enough to fall inside the time split.
    senders = []
    receivers = []
    values = []
    labels = []
    for offset, tx in enumerate(hashes):
        if offset > count * 0.7:
            senders.append(f"rMule{offset}")
            receivers.append("rCollector")
            values.append(50_000_000)  # 50 XRP, in drops
            labels.append(1)
        else:
            senders.append(f"rA{offset % 17}")
            receivers.append(f"rB{(offset * 3) % 19}")
            values.append(1_000_000 + offset)  # just over 1 XRP
            labels.append(0)
    transactions = pd.DataFrame(
        {
            "TRANSACTION_HASH": hashes,
            "LEDGER_INDEX": ledgers,
            "ACCOUNT": senders,
            "FEE": 12,
            "TRANSACTION_TYPE": "Payment",
            "DELIVERED_AMOUNT": ["50"] * count,
            "EXTRA_TRANSACTION_FIELDS": ["{}"] * count,
        }
    )
    transfers = pd.DataFrame(
        {
            "TRANSACTION_HASH": hashes,
            "TRANSFER_INDEX": 0,
            "ACCOUNT": senders,
            "DESTINATION": receivers,
            "SOURCE_VALUE": values,
            "SYMBOL": "XRP",
            "TOKEN_ADDRESS": "",
        }
    )
    blocks = pd.DataFrame(
        {
            "LEDGER_INDEX": ledgers,
            "CLOSE_TIME": [1_700_000_000 + i * 30 for i in ledgers],
        }
    )
    rates = pd.DataFrame(
        {
            "CLOSE_TIME": [pd.Timestamp(1_700_000_000, unit="s", tz="UTC")],
            "SYMBOL": ["XRP"],
            "TOKEN_ADDRESS": [""],
            "USD_PRICE": [0.5],
        }
    )
    transactions.to_parquet(folder / "transactions.parquet", index=False)
    transfers.to_parquet(folder / "transfers.parquet", index=False)
    blocks.to_parquet(folder / "blocks.parquet", index=False)
    rates.to_csv(folder / "market_rates.csv", index=False)
    if label:
        pd.DataFrame({"TRANSACTION_HASH": hashes, "LABEL": labels}).to_csv(folder / "labels.csv", index=False)


def test_flat_folder_splits_train_and_test_without_blocks(tmp_path):
    from preprocess import build_from_folder

    docs = tmp_path / "docs"
    docs.mkdir()
    train_tx = pd.DataFrame(
        {"TRANSACTION_HASH": ["a", "b"], "LEDGER_INDEX": [1, 2], "ACCOUNT": ["s", "s"], "FEE": [12, 12], "TRANSACTION_TYPE": ["Payment", "Payment"]}
    )
    test_tx = pd.DataFrame(
        {"TRANSACTION_HASH": ["c"], "LEDGER_INDEX": [3], "ACCOUNT": ["s"], "FEE": [12], "TRANSACTION_TYPE": ["Payment"]}
    )
    transfers = pd.DataFrame(
        {
            "TRANSACTION_HASH": ["a", "b", "c"],
            "ACCOUNT": ["s", "s", "s"],
            "DESTINATION": ["r", "r", "r"],
            "SOURCE_VALUE": [1_000_000, 2_000_000, 3_000_000],
            "SYMBOL": ["XRP", "XRP", "XRP"],
        }
    )
    labels = pd.DataFrame({"TRANSACTION_HASH": ["a", "b"], "LABEL": [0, 1]})
    rates = pd.DataFrame({"TIMESTAMP": [1789862400, 1789866000], "ASSET": ["XRP", "XRP"], "PRICE_MICRO_USD": [500000, 500000]})
    train_tx.to_parquet(docs / "transactions.parquet", index=False)
    test_tx.to_parquet(docs / "transactions_test.parquet", index=False)
    transfers.to_parquet(docs / "transfers.parquet", index=False)
    labels.to_csv(docs / "labels (1).csv", index=False)
    rates.to_csv(docs / "market_rates.csv", index=False)
    train, test, edges = build_from_folder(docs)
    assert set(train["TRANSACTION_HASH"]) == {"a", "b"}
    assert set(test["TRANSACTION_HASH"]) == {"c"}
    assert train["label"].notna().all()
    assert "label" not in test.columns or test["label"].isna().all()
    assert len(edges) == 3


def test_market_rates_micro_usd_are_scaled():
    from preprocess import _rates_table

    rates = pd.DataFrame(
        {
            "TIMESTAMP": [1789862400],
            "ASSET": ["XRP"],
            "PRICE_MICRO_USD": [648957],
        }
    )
    priced = _rates_table(rates)
    assert priced.loc[0, "symbol"] == "XRP"
    assert abs(priced.loc[0, "usd_price"] - 0.648957) < 1e-9


def test_pipeline_writes_a_valid_submission(tmp_path):
    _write_split(tmp_path / "data" / "raw" / "train", 0, 80, label=True)
    _write_split(tmp_path / "data" / "raw" / "test", 1000, 30, label=False)
    destination = run(tmp_path, fast=True)
    submission = pd.read_csv(destination)
    assert list(submission.columns) == ["TRANSACTION_HASH", "score"]
    assert submission["TRANSACTION_HASH"].is_unique
    assert len(submission) == 30
    assert submission["score"].between(0, 1).all()
    graph = pd.read_csv(tmp_path / "data" / "processed" / "graph_features.csv")
    behaviour = pd.read_csv(tmp_path / "data" / "processed" / "behaviour_features.csv")
    assert "label" not in graph.columns
    assert "label" not in behaviour.columns
    assert {"sender_in_degree", "cycle_indicator", "cluster_size", "is_same_community"} <= set(graph.columns)
    assert "tx_count_24h" in behaviour.columns
    # 50,000,000 drops is 50 XRP. The priced amount must be in XRP-sized units,
    # not raw drops.
    train = pd.read_csv(tmp_path / "data" / "processed" / "train_clean.csv")
    assert 20 < train["paid_usd"].max() < 1000
