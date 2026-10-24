"""Smoke test: do the graph features separate implanted AML structures?

Runs only against the synthetic sample, where the generator records which
pattern each transaction belongs to. The pattern column is a stand-in for the
real labels and is used nowhere in feature construction - this script exists to
prove the features carry signal before the real labels arrive.

    python src/sample_signal_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/graph_features.csv"))
    parser.add_argument("--transfers", type=Path, default=Path("data/sample/transfers.parquet"))
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    truth = (
        pd.read_parquet(args.transfers)
        .drop_duplicates("TRANSACTION_HASH")[["TRANSACTION_HASH", "pattern"]]
    )
    data = features.merge(truth, on="TRANSACTION_HASH", how="inner")
    y = (data["pattern"] != "background").astype(int)
    X = data.drop(columns=["TRANSACTION_HASH", "pattern"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_valid)[:, 1]

    ap = average_precision_score(y_valid, scores)
    baseline = y_valid.mean()
    print(f"positives in validation: {int(y_valid.sum())} / {len(y_valid)} ({baseline:.3f})")
    print(f"Average Precision: {ap:.4f}   (random baseline {baseline:.4f})")
    print(f"Hackathon-style score: {100 * ap:.1f}")

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(12)
    )
    print("\ntop features:")
    for name, value in importances.items():
        print(f"  {name:<34} {value:.4f}")

    print("\nrecall by implanted pattern (score above the 90th percentile):")
    threshold = pd.Series(scores).quantile(0.90)
    valid_patterns = data.loc[y_valid.index, "pattern"]
    flagged = pd.Series(scores, index=y_valid.index) >= threshold
    for pattern, group in flagged.groupby(valid_patterns):
        print(f"  {pattern:<16} {group.mean():.2f}  (n={len(group)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
