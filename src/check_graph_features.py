"""Validate graph_features.csv before handing it to the ML engineer.

Checks the contract the rest of the team relies on: exactly one row per
TRANSACTION_HASH, full coverage of the transactions file, no missing values,
no label-like columns, and no constant columns that would waste model capacity.

    python src/check_graph_features.py \
        --features outputs/graph_features.csv \
        --transactions data/raw/transactions.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from graph_features import HASH_COLUMNS, read_table, resolve_column

LABEL_LIKE = {"label", "labels", "is_suspicious", "suspicious", "target", "y", "pattern"}


def check(features_path: Path, transactions_path: Path | None) -> tuple[list[str], list[str]]:
    features = pd.read_csv(features_path)
    problems: list[str] = []
    warnings: list[str] = []

    if "TRANSACTION_HASH" not in features.columns:
        return ["TRANSACTION_HASH column is missing"], warnings

    duplicates = int(features["TRANSACTION_HASH"].duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicated TRANSACTION_HASH rows")

    missing_values = features.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        problems.append(f"columns with missing values: {dict(missing_values)}")

    leaked = LABEL_LIKE & {c.lower() for c in features.columns}
    if leaked:
        problems.append(f"label-like columns present: {sorted(leaked)}")

    non_numeric = [
        c for c in features.columns.drop("TRANSACTION_HASH")
        if not pd.api.types.is_numeric_dtype(features[c])
    ]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric}")

    constant = [
        c for c in features.columns.drop("TRANSACTION_HASH")
        if features[c].nunique(dropna=False) <= 1
    ]
    if constant:
        warnings.append(f"constant columns (carry no signal, consider dropping): {constant}")

    if transactions_path is not None:
        transactions = read_table(transactions_path)
        hash_col = resolve_column(transactions, HASH_COLUMNS, label="transaction hash")
        expected = set(transactions[hash_col].astype(str))
        actual = set(features["TRANSACTION_HASH"].astype(str))
        if expected - actual:
            problems.append(f"{len(expected - actual)} transactions have no feature row")
        if actual - expected:
            problems.append(f"{len(actual - expected)} feature rows have no transaction")

    print(f"rows: {len(features):,}  feature columns: {features.shape[1] - 1}")
    return problems, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/graph_features.csv"))
    parser.add_argument("--transactions", type=Path, default=None)
    args = parser.parse_args()

    problems, warnings = check(args.features, args.transactions)
    for warning in warnings:
        print(f"  warning: {warning}")
    if problems:
        print("FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("OK - safe to hand over to the ML engineer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
