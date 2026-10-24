"""Build every team file and the leaderboard submission.

Run this from the unzipped group project, after the hackathon downloads have
been unpacked into ``data/raw/train`` and ``data/raw/test``:

    pip install -r requirements.txt
    python src/run_all.py

The file to upload at hackathon.ukcbt.org is ``submission/predictions.csv``.
It has two columns, TRANSACTION_HASH and score, one row per test transaction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from behaviour import build_behaviour_features  # noqa: E402
from graph_features import features_from_frame  # noqa: E402
from model import run_experiments  # noqa: E402
from preprocess import build_tables  # noqa: E402


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}  ({len(frame):,} rows, {frame.shape[1]} columns)")


def run(root: Path, fast: bool = False) -> Path:
    train, test, edges = build_tables(root)
    print(f"train rows {len(train):,}   test rows {len(test):,}   transfer legs {len(edges):,}")
    if "label" not in train.columns or train["label"].isna().all():
        raise SystemExit("Training labels were not found. labels.csv must sit in data/raw/train.")

    graph = features_from_frame(edges[["TRANSACTION_HASH", "sender", "receiver", "amount"]])
    behaviour = build_behaviour_features(pd.concat([train, test], ignore_index=True))
    forbidden = {"label", "is_suspicious", "target", "pattern"}
    leaked = forbidden & {column.lower() for column in list(graph.columns) + list(behaviour.columns)}
    if leaked:
        raise SystemExit(f"Label columns leaked into features: {sorted(leaked)}")

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(graph, on="TRANSACTION_HASH", how="left").merge(
            behaviour, on="TRANSACTION_HASH", how="left"
        )

    train_features = attach(train)
    test_features = attach(test.drop(columns=["label"], errors="ignore"))
    processed = root / "data" / "processed"
    _write(train.drop(columns=["sender", "receiver"], errors="ignore"), processed / "train_clean.csv")
    _write(test.drop(columns=["label", "sender", "receiver"], errors="ignore"), processed / "test_clean.csv")
    _write(graph, processed / "graph_features.csv")
    _write(behaviour, processed / "behaviour_features.csv")
    _write(train_features, processed / "train_features.csv")
    _write(test_features, processed / "test_features.csv")

    scores, submission = run_experiments(train_features, test_features, fast=fast)
    _write(scores, root / "submission" / "experiment_scores.csv")
    destination = root / "submission" / "predictions.csv"
    _write(submission, destination)
    missing = set(test["TRANSACTION_HASH"].astype(str)) - set(submission["TRANSACTION_HASH"])
    if missing:
        raise SystemExit(f"{len(missing)} test transactions have no score")
    print("upload this file at hackathon.ukcbt.org, step 2:")
    print(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fast", action="store_true", help="tiny models, for a smoke test")
    args = parser.parse_args()
    run(args.root, fast=args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
