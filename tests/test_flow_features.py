"""The flow features must fire on a collector and on a layered chain."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_features import build_flow_features  # noqa: E402


def _edges(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["TRANSACTION_HASH", "sender", "receiver", "amount", "minute"])
    frame["time"] = pd.Timestamp("2026-10-20", tz="UTC") + pd.to_timedelta(frame["minute"], unit="m")
    return frame.drop(columns=["minute"])


def test_collector_burst_is_visible_on_the_incoming_payments():
    rows = [(f"in{i}", f"mule{i}", "hub", 10.0, i) for i in range(6)]
    rows.append(("out", "hub", "exit", 58.0, 8))
    features = build_flow_features(_edges(rows)).set_index("TRANSACTION_HASH")
    assert features.loc["in0", "receiver_in_burst_1h"] >= 6
    assert features.loc["in0", "receiver_is_collector"] == 1
    assert features.loc["out", "sender_exit_ratio"] > 0.5


def test_layered_chain_has_depth():
    rows = [
        ("hop1", "source", "mid1", 10.0, 0),
        ("hop2", "mid1", "mid2", 9.8, 1),
        ("hop3", "mid2", "dest", 9.7, 2),
    ]
    features = build_flow_features(_edges(rows)).set_index("TRANSACTION_HASH")
    assert features.loc["hop1", "receiver_layer_depth"] >= 1
    assert features.loc["hop2", "sender_is_conduit"] == 1
    assert features["flow_layer_depth"].max() >= 1
