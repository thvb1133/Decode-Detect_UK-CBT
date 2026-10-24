"""Fit several differently-shaped LightGBM configurations and rank-blend them.

A single boosting configuration overfits a particular slice of this feature
space: a deep, low-regularisation model is better on the easy high-volume
motifs, and a shallower, heavily-subsampled one is better on the hard cases
hidden inside the relay traffic. Blending their rankings takes both, and
average precision only cares about the ordering.

The blend weights are chosen on out-of-fold predictions, so they are fitted
without ever seeing a transaction the contributing models were trained on.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from dd import model as modelmod
from dd.loading import load_labels
from dd.pipeline import build_split_features

CONFIGS = {
    # deep and permissive: leans on the dense, well-populated feature groups
    "deep": dict(modelmod.LGB_PARAMS, num_leaves=127, min_data_in_leaf=15, learning_rate=0.03, feature_fraction=0.7),
    # shallow and heavily subsampled: forced to spread across weaker signals
    "shallow": dict(
        modelmod.LGB_PARAMS, num_leaves=31, min_data_in_leaf=40, learning_rate=0.02,
        feature_fraction=0.45, bagging_fraction=0.7, lambda_l2=5.0,
    ),
    # balanced reference configuration
    "balanced": dict(modelmod.LGB_PARAMS),
}


def _rank(values):
    return pd.Series(values).rank(pct=True).to_numpy()


def search_weights(oofs: dict, y: np.ndarray, step=0.1):
    """Grid-search non-negative blend weights on out-of-fold rankings."""
    names = sorted(oofs)
    ranked = {name: _rank(oofs[name]) for name in names}
    grid = [w for w in product(np.arange(0, 1 + step / 2, step), repeat=len(names)) if sum(w) > 0]
    best, best_ap = None, -1.0
    for weights in grid:
        total = sum(weights)
        blended = sum(w * ranked[n] for w, n in zip(weights, names)) / total
        ap = average_precision_score(y, blended)
        if ap > best_ap:
            best_ap, best = ap, {n: round(w / total, 4) for w, n in zip(weights, names)}
    return best, best_ap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="submission/predictions.csv")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--report", default="artifacts/metrics.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)

    train_features = build_split_features(args.train, args.cache, "train")
    labels = load_labels(args.train).set_index("TRANSACTION_HASH")
    y = labels["is_suspicious"].reindex(train_features["TRANSACTION_HASH"]).to_numpy().astype(int)

    oofs, models, single_ap = {}, {}, {}
    for name, params in CONFIGS.items():
        print(f"fitting {name}", flush=True)
        cv = modelmod.cross_validate(train_features, y, args.folds, args.repeats, params)
        oofs[name] = cv["oof"]
        models[name] = cv["models"]
        single_ap[name] = cv["ap"]
        print(f"  {name} out-of-fold AP = {cv['ap']:.4f}", flush=True)

    weights, blend_ap = search_weights(oofs, y)
    print(f"blend weights {weights} -> out-of-fold AP = {blend_ap:.4f}", flush=True)

    best_single = max(single_ap, key=single_ap.get)
    if blend_ap <= single_ap[best_single]:
        print(f"blend does not beat {best_single}; keeping it alone", flush=True)
        weights = {n: (1.0 if n == best_single else 0.0) for n in CONFIGS}
        blend_ap = single_ap[best_single]

    blended_oof = sum(w * _rank(oofs[n]) for n, w in weights.items() if w > 0)
    precision, recall, k = modelmod.top_k_metrics(y, blended_oof, train_features["TRANSACTION_HASH"])
    print(f"top 1% (k={k}): precision {precision:.4f}, recall {recall:.4f}", flush=True)

    test_features = build_split_features(args.test, args.cache, "test")
    test_features = test_features[["TRANSACTION_HASH"] + modelmod.feature_columns(train_features)]
    scores = sum(w * _rank(modelmod.predict(models[n], test_features)) for n, w in weights.items() if w > 0)
    spread = pd.Series(scores).rank(method="average", pct=True).to_numpy()

    submission = pd.DataFrame(
        {"TRANSACTION_HASH": test_features["TRANSACTION_HASH"].to_numpy(), "score": np.round(spread, 6)}
    )
    assert submission["TRANSACTION_HASH"].is_unique
    assert submission["score"].between(0, 1).all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"wrote {len(submission)} rows to {args.out}", flush=True)

    report = {
        "oof_average_precision": float(blend_ap),
        "leaderboard_equivalent_score": float(100 * blend_ap),
        "single_model_average_precision": {k2: float(v) for k2, v in single_ap.items()},
        "blend_weights": weights,
        "top_1pct_precision": float(precision),
        "top_1pct_recall": float(recall),
        "top_1pct_k": int(k),
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_test": int(len(submission)),
        "n_features": int(len(modelmod.feature_columns(train_features))),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": modelmod.SEED,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
