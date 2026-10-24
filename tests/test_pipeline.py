"""Correctness tests for valuation, feature construction and submission shape.

These run against a small synthetic ledger built in the fixtures below, so they
need no organiser data and finish in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dd import features as featmod
from dd import model as modelmod
from dd.loading import XRP_TOKEN, load_split

HOUR = 3600
BASE = 1788220800  # an exact hour boundary, so every event shares one rate row


def _amount_xrp(drops):
    return json.dumps(str(int(drops)))


def _amount_iou(issuer, currency, value):
    return json.dumps({"currency": currency, "issuer": issuer, "value": f"{value:.6f}"})


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    """A six-payment ledger containing one deliberate pass-through chain."""
    root = tmp_path_factory.mktemp("split")
    issuer = "rISSUER00000000000000000000000000"

    rows = [
        # (hash, offset seconds, sender, receiver, delivered json, sendmax json)
        ("H1", 0, "rA", "rB", _amount_xrp(10_000_000), None),
        ("H2", 60, "rB", "rC", _amount_xrp(9_900_000), None),  # forwards ~99% one minute later
        ("H3", 120, "rC", "rA", _amount_xrp(9_800_000), None),  # closes a three-account cycle
        ("H4", 1800, "rD", "rE", _amount_iou(issuer, "USD", 25.0), _amount_xrp(40_000_000)),
        ("H5", 2000, "rD", "rE", _amount_iou(issuer, "USD", 25.0), None),
        ("H6", 2200, "rE", "rF", _amount_xrp(1_000_000), None),
    ]

    transactions = pd.DataFrame(
        {
            "LEDGER_INDEX": [1000 + i for i in range(len(rows))],
            "TIMESTAMP": [BASE + r[1] for r in rows],
            "DATETIME": pd.to_datetime([BASE + r[1] for r in rows], unit="s", utc=True),
            "LEDGER_HASH": [f"L{i}" for i in range(len(rows))],
            "TRANSACTION_HASH": [r[0] for r in rows],
            "TRANSACTION_INDEX": list(range(len(rows))),
            "ACCOUNT": [r[2] for r in rows],
            "TRANSACTION_TYPE": ["Payment"] * len(rows),
            "FEE": [0.000015] * len(rows),
            "DELIVERED_AMOUNT": [r[4] for r in rows],
            "SUCCESS": [True] * len(rows),
            "EXTRA_TRANSACTION_FIELDS": [
                json.dumps(
                    {"Amount": json.loads(r[4]), "Destination": r[3], **({"SendMax": json.loads(r[5])} if r[5] else {})}
                )
                for r in rows
            ],
        }
    )

    transfers = pd.DataFrame(
        {
            "LEDGER_INDEX": [1000 + i for i in range(len(rows))],
            "TIMESTAMP": [BASE + r[1] for r in rows],
            "TRANSACTION_HASH": [r[0] for r in rows],
            "TRANSFER_TYPE": ["direct_payment"] * 3 + ["path_payment", "direct_payment", "direct_payment"],
            "TRANSFER_INDEX": [0] * len(rows),
            "FROM_ADDRESS": [r[2] for r in rows],
            "TO_ADDRESS": [r[3] for r in rows],
            "TOKEN_ADDRESS": [XRP_TOKEN] * 3 + [issuer, issuer, XRP_TOKEN],
            "SYMBOL": ["XRP"] * 3 + ["USD", "USD", "XRP"],
            "SOURCE_VALUE": [10.0] * len(rows),
            "VALUE": [10.0] * len(rows),
        }
    )

    rates = pd.DataFrame(
        {
            "TIMESTAMP": [BASE, BASE, BASE + HOUR, BASE + HOUR],
            "ASSET": ["XRP", f"{issuer}:USD"] * 2,
            "PRICE_MICRO_USD": [500_000, 1_000_000, 500_000, 1_000_000],
        }
    )

    transactions.to_parquet(root / "transactions.parquet", index=False)
    transfers.to_parquet(root / "transfers.parquet", index=False)
    rates.to_csv(root / "market_rates.csv", index=False)
    return root


def test_xrp_is_valued_at_the_hourly_reference_price(ledger):
    payments = load_split(ledger)["payments"].set_index("TRANSACTION_HASH")
    # 10,000,000 drops = 10 XRP, priced at 0.50 USD
    assert payments.loc["H1", "usd"] == pytest.approx(5.0)


def test_issued_currency_is_valued_from_its_own_issuer_price(ledger):
    payments = load_split(ledger)["payments"].set_index("TRANSACTION_HASH")
    # 25 USD tokens at 1.00 USD each, a different asset from XRP
    assert payments.loc["H4", "usd"] == pytest.approx(25.0)


def test_sendmax_is_a_ceiling_not_the_delivered_amount(ledger):
    payments = load_split(ledger)["payments"].set_index("TRANSACTION_HASH")
    row = payments.loc["H4"]
    # SendMax of 40 XRP = 20 USD is recorded separately and never replaces the delivery
    assert row["sendmax_usd"] == pytest.approx(20.0)
    assert row["usd"] == pytest.approx(25.0)


def test_one_row_per_transaction(ledger):
    table = featmod.build_features(load_split(ledger))
    assert len(table) == 6
    assert table["TRANSACTION_HASH"].is_unique


def test_pass_through_is_detected_on_the_chain(ledger):
    table = featmod.build_features(load_split(ledger)).set_index("TRANSACTION_HASH")
    # rB receives on H1 and forwards 99% of it one minute later on H2
    assert table.loc["H1", "flow_fwd_count_1h"] == 1
    assert table.loc["H1", "flow_fwd_best_match_1h"] < 0.05
    assert table.loc["H1", "flow_fwd_within_5pct_1h"] == 1
    # rF never forwards anything, so the last hop has no onward flow
    assert table.loc["H6", "flow_fwd_count_1h"] == 0


def test_backward_flow_sees_the_incoming_leg(ledger):
    table = featmod.build_features(load_split(ledger)).set_index("TRANSACTION_HASH")
    # H2 is sent by rB, who received a near-identical amount 60 seconds earlier
    assert table.loc["H2", "flow_bwd_count_1h"] == 1
    assert table.loc["H2", "flow_bwd_best_match_1h"] < 0.05
    assert table.loc["H2", "flow_bwd_latency_1h"] == pytest.approx(60.0)


def test_cycle_membership_is_detected(ledger):
    table = featmod.build_features(load_split(ledger)).set_index("TRANSACTION_HASH")
    # rA -> rB -> rC -> rA is a closed three-account cycle
    for h in ("H1", "H2", "H3"):
        assert table.loc[h, "g_edge_in_cycle"] == 1.0
    # rE -> rF is not part of any cycle
    assert table.loc["H6", "g_edge_in_cycle"] == 0.0


def test_repeated_identical_amounts_are_counted(ledger):
    table = featmod.build_features(load_split(ledger)).set_index("TRANSACTION_HASH")
    # rD sends 25 USD twice within the window
    assert table.loc["H4", "str_twins_src_24h"] == 1
    assert table.loc["H5", "str_twins_src_24h"] == 1


def test_pair_novelty(ledger):
    table = featmod.build_features(load_split(ledger)).set_index("TRANSACTION_HASH")
    assert table.loc["H4", "pair_is_first"] == 1.0
    assert table.loc["H5", "pair_is_first"] == 0.0
    assert table.loc["H4", "pair_count"] == 2


def test_features_never_see_labels(ledger):
    """A label column present in the split directory must not change features."""
    before = featmod.build_features(load_split(ledger))
    pd.DataFrame({"TRANSACTION_HASH": ["H1"], "is_suspicious": [1]}).to_csv(Path(ledger) / "labels.csv", index=False)
    after = featmod.build_features(load_split(ledger))
    pd.testing.assert_frame_equal(before, after)


def test_top_k_metrics_break_ties_by_hash_ascending():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    hashes = np.array(["D", "A", "C", "B"])
    # all tied, so the budget of 2 goes to hashes A and B, both positive
    precision, recall, k = modelmod.top_k_metrics(y, scores, hashes, fraction=0.5)
    assert k == 2
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_rank_blend_is_monotone_in_each_model():
    class Stub:
        best_iteration = 1

        def __init__(self, values):
            self.values = np.asarray(values, dtype="float64")

        def predict(self, X, num_iteration=None):
            return self.values

    features = pd.DataFrame({"TRANSACTION_HASH": list("abcd"), "x": [0.0, 1.0, 2.0, 3.0]})
    blended = modelmod.predict([Stub([0.1, 0.2, 0.3, 0.4]), Stub([9.0, 10.0, 11.0, 12.0])], features)
    # the two models disagree wildly in scale but agree on order, so the blend keeps it
    assert list(np.argsort(blended)) == [0, 1, 2, 3]


def test_no_infinities_survive_feature_building(ledger):
    table = featmod.build_features(load_split(ledger))
    numeric = table.drop(columns=["TRANSACTION_HASH"]).to_numpy(dtype="float64")
    assert not np.isinf(numeric).any()
