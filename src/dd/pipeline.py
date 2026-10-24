"""End-to-end pipeline: raw splits in, predictions CSV out.

Run as ``python -m dd.pipeline --train <dir> --test <dir> --out predictions.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dd import features as featmod
from dd import model as modelmod
from dd.loading import load_labels, load_split


def build_split_features(split_dir, cache_dir=None, name=None):
    """Build (or reuse a cached copy of) the feature table for one split."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{name or Path(split_dir).name}_features.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
    split = load_split(split_dir)
    table = featmod.build_features(split)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
    return table


def run(train_dir, test_dir, out_path, cache_dir=None, n_splits=5, n_repeats=3, report_path=None):
    started = time.time()
    print(f"[1/5] building training features from {train_dir}", flush=True)
    train_features = build_split_features(train_dir, cache_dir, "train")
    print(f"      {train_features.shape[0]} rows x {train_features.shape[1] - 1} features", flush=True)

    labels = load_labels(train_dir).set_index("TRANSACTION_HASH")
    y = labels["is_suspicious"].reindex(train_features["TRANSACTION_HASH"]).to_numpy()
    assert not np.isnan(y.astype("float64")).any(), "every training transaction needs a label"
    y = y.astype(int)

    print(f"[2/5] cross-validating ({n_splits} folds x {n_repeats} repeats)", flush=True)
    cv = modelmod.cross_validate(train_features, y, n_splits=n_splits, n_repeats=n_repeats)
    precision, recall, k = modelmod.top_k_metrics(y, cv["oof"], train_features["TRANSACTION_HASH"])
    print(f"      out-of-fold AP = {cv['ap']:.4f}  (score {100 * cv['ap']:.2f})", flush=True)
    print(f"      top 1% (k={k}): precision {precision:.4f}, recall {recall:.4f}", flush=True)

    print("[3/5] building test features", flush=True)
    test_features = build_split_features(test_dir, cache_dir, "test")
    print(f"      {test_features.shape[0]} rows x {test_features.shape[1] - 1} features", flush=True)

    missing = set(modelmod.feature_columns(train_features)) - set(test_features.columns)
    assert not missing, f"test split is missing features: {sorted(missing)[:5]}"
    test_features = test_features[["TRANSACTION_HASH"] + modelmod.feature_columns(train_features)]

    print("[4/5] scoring test transactions", flush=True)
    scores = modelmod.predict(cv["models"], test_features)
    # AP depends only on the ranking, so map to [0,1] by rank for a well spread file
    spread = pd.Series(scores).rank(method="average", pct=True).to_numpy()
    submission = pd.DataFrame(
        {"TRANSACTION_HASH": test_features["TRANSACTION_HASH"].to_numpy(), "score": np.round(spread, 6)}
    )
    assert submission["TRANSACTION_HASH"].is_unique, "duplicate test hash in submission"
    assert submission["score"].between(0, 1).all(), "scores must lie in [0, 1]"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"      wrote {len(submission)} rows to {out_path}", flush=True)

    print("[5/5] feature importance", flush=True)
    top = modelmod.importance(cv["models"], train_features, top=30)
    print(top.to_string(index=False), flush=True)

    report = {
        "oof_average_precision": float(cv["ap"]),
        "leaderboard_equivalent_score": float(100 * cv["ap"]),
        "average_precision_by_repeat": [float(v) for v in cv["ap_by_repeat"]],
        "top_1pct_precision": float(precision),
        "top_1pct_recall": float(recall),
        "top_1pct_k": int(k),
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_test": int(len(submission)),
        "n_features": int(len(modelmod.feature_columns(train_features))),
        "mean_best_iteration": cv["best_iteration"],
        "seed": modelmod.SEED,
        "runtime_seconds": round(time.time() - started, 1),
        "top_features": top.to_dict("records"),
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"done in {report['runtime_seconds']}s", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a ledger split for suspicious payments.")
    parser.add_argument("--train", required=True, help="directory holding the labelled training split")
    parser.add_argument("--test", required=True, help="directory holding the split to score")
    parser.add_argument("--out", default="submission/predictions.csv")
    parser.add_argument("--cache", default=None, help="optional directory for cached feature tables")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default=None, help="optional path for a JSON metrics report")
    args = parser.parse_args(argv)
    run(args.train, args.test, args.out, args.cache, args.folds, args.repeats, args.report)


if __name__ == "__main__":
    main()
