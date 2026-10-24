"""Tests for the graph feature pipeline.

A hand-built graph covers the pattern definitions exactly, and the synthetic
dataset checks the end-to-end contract (one row per TRANSACTION_HASH, patterns
recovered where they were implanted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph_features import (  # noqa: E402
    FAN_THRESHOLD,
    build_graph,
    build_graph_features,
    transaction_features,
)
from src.make_sample_data import generate  # noqa: E402


def make_edges(triples: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"TRANSACTION_HASH": h, "sender": s, "receiver": r, "amount": 100.0}
         for h, s, r in triples]
    )


def features_for(triples: list[tuple[str, str, str]]) -> pd.DataFrame:
    edges = make_edges(triples)
    return transaction_features(edges, build_graph(edges)).set_index("TRANSACTION_HASH")


def test_fan_in_is_flagged_on_the_collector_side():
    triples = [(f"t{i}", f"mule{i}", "collector") for i in range(FAN_THRESHOLD + 2)]
    out = features_for(triples)
    assert out["fan_in_pattern"].eq(1).all()
    assert out["fan_out_pattern"].eq(0).all()
    assert out["receiver_in_degree"].eq(FAN_THRESHOLD + 2).all()
    assert out["sender_out_degree"].eq(1).all()


def test_fan_out_is_flagged_on_the_source_side():
    triples = [(f"t{i}", "source", f"leaf{i}") for i in range(FAN_THRESHOLD + 2)]
    out = features_for(triples)
    assert out["fan_out_pattern"].eq(1).all()
    assert out["fan_in_pattern"].eq(0).all()
    assert out["sender_out_degree"].eq(FAN_THRESHOLD + 2).all()


def test_scatter_gather_detects_reconvergence():
    triples = []
    for i in range(FAN_THRESHOLD + 1):
        triples.append((f"a{i}", "source", f"mid{i}"))
        triples.append((f"b{i}", f"mid{i}", "sink"))
    out = features_for(triples)
    outbound = out.loc[[f"a{i}" for i in range(FAN_THRESHOLD + 1)]]
    assert outbound["scatter_gather_pattern"].eq(1).all()


def test_gather_scatter_detects_funnel_then_spray():
    triples = []
    for i in range(FAN_THRESHOLD + 1):
        triples.append((f"a{i}", f"src{i}", "hub"))
        triples.append((f"b{i}", "hub", f"dst{i}"))
    out = features_for(triples)
    inbound = out.loc[[f"b{i}" for i in range(FAN_THRESHOLD + 1)]]
    assert inbound["gather_scatter_pattern"].eq(1).all()


def test_cycle_indicator_marks_only_edges_on_a_cycle():
    triples = [("c1", "a", "b"), ("c2", "b", "c"), ("c3", "c", "a"), ("x1", "c", "d")]
    out = features_for(triples)
    assert out.loc[["c1", "c2", "c3"], "cycle_indicator"].eq(1).all()
    assert out.loc["x1", "cycle_indicator"] == 0


def test_reciprocal_edge_and_community_flags():
    triples = [("t1", "a", "b"), ("t2", "b", "a"), ("t3", "x", "y")]
    out = features_for(triples)
    assert out.loc[["t1", "t2"], "is_reciprocal_edge"].eq(1).all()
    assert out.loc["t3", "is_reciprocal_edge"] == 0
    # Two disconnected pairs cannot share a community.
    assert out["is_same_community"].eq(1).all()
    assert out.loc["t1", "cluster_size"] == 2


def test_self_transfers_do_not_create_edges():
    triples = [("t1", "a", "a"), ("t2", "a", "b")]
    out = features_for(triples)
    assert out.loc["t1"].drop(labels=[]).eq(0).all()
    assert out.loc["t2", "sender_out_degree"] == 1


@pytest.fixture(scope="module")
def sample_paths(tmp_path_factory):
    transactions, transfers = generate(n_background=500, n_wallets=1000)
    directory = tmp_path_factory.mktemp("sample")
    transactions.to_parquet(directory / "transactions.parquet", index=False)
    transfers.to_parquet(directory / "transfers.parquet", index=False)
    return directory, transfers


def test_one_row_per_transaction_hash(sample_paths):
    directory, transfers = sample_paths
    out = build_graph_features(
        directory / "transfers.parquet", directory / "transactions.parquet"
    )
    assert out["TRANSACTION_HASH"].is_unique
    assert set(out["TRANSACTION_HASH"]) == set(transfers["TRANSACTION_HASH"])
    assert not out.isna().any().any()


def test_implanted_patterns_are_recovered(sample_paths):
    directory, transfers = sample_paths
    out = build_graph_features(
        directory / "transfers.parquet", directory / "transactions.parquet"
    )
    truth = transfers.drop_duplicates("TRANSACTION_HASH")[["TRANSACTION_HASH", "pattern"]]
    merged = out.merge(truth, on="TRANSACTION_HASH")

    recall = {
        "fan_in": merged.loc[merged["pattern"] == "fan_in", "fan_in_pattern"].mean(),
        "fan_out": merged.loc[merged["pattern"] == "fan_out", "fan_out_pattern"].mean(),
        "cycle": merged.loc[merged["pattern"] == "cycle", "cycle_indicator"].mean(),
    }
    assert recall["fan_in"] == 1.0
    assert recall["fan_out"] == 1.0
    assert recall["cycle"] == 1.0

    scatter = merged[merged["pattern"] == "scatter_gather"]
    assert scatter["scatter_gather_pattern"].mean() > 0.4
    gather = merged[merged["pattern"] == "gather_scatter"]
    assert gather["gather_scatter_pattern"].mean() > 0.4


def test_no_label_column_leaks_into_features(sample_paths):
    directory, _ = sample_paths
    out = build_graph_features(
        directory / "transfers.parquet", directory / "transactions.parquet"
    )
    forbidden = {"label", "is_suspicious", "target", "y", "pattern"}
    assert forbidden.isdisjoint({c.lower() for c in out.columns})
