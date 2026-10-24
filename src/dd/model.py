"""Model training, honest evaluation and test scoring.

Scoring on the leaderboard is ``100 x average precision``, so everything here
optimises ranking quality rather than calibrated probability. The ensemble is
a gradient-boosted tree model over the engineered features, blended across
repeated stratified folds; averaging fold models is what turns a single noisy
ranking into a stable one on a 3% positive rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

SEED = 20261024

LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 25,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "verbose": -1,
    "num_threads": 0,
    "seed": SEED,
}

NUM_ROUNDS = 3000
EARLY_STOPPING = 200


def feature_columns(features: pd.DataFrame) -> list:
    return [c for c in features.columns if c != "TRANSACTION_HASH"]


def top_k_metrics(y_true, scores, hashes, fraction=0.01):
    """Precision and recall in the top ``fraction`` of the ranking.

    Ties are broken by transaction hash ascending, matching how the organisers
    resolve a tied review budget.
    """
    k = max(1, int(round(len(scores) * fraction)))
    order = np.lexsort((np.asarray(hashes, dtype=str), -np.asarray(scores, dtype="float64")))
    selected = np.zeros(len(scores), dtype=int)
    selected[order[:k]] = 1
    return (
        precision_score(y_true, selected, zero_division=0),
        recall_score(y_true, selected, zero_division=0),
        k,
    )


def cross_validate(features: pd.DataFrame, y: np.ndarray, n_splits=5, n_repeats=3, params=None):
    """Repeated stratified CV returning out-of-fold scores and fold models.

    Out-of-fold predictions are averaged over repeats, which both reduces the
    variance of the AP estimate and gives the blending weights something stable
    to be fitted against.
    """
    import lightgbm as lgb

    params = dict(params or LGB_PARAMS)
    cols = feature_columns(features)
    X = features[cols]
    oof = np.zeros((n_repeats, len(y)))
    models = []
    best_iters = []

    for repeat in range(n_repeats):
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + repeat)
        for fold, (tr, va) in enumerate(folds.split(X, y)):
            run = dict(params)
            run["seed"] = SEED + 1000 * repeat + fold
            run["bagging_seed"] = run["seed"] + 1
            run["feature_fraction_seed"] = run["seed"] + 2
            train_set = lgb.Dataset(X.iloc[tr], label=y[tr])
            valid_set = lgb.Dataset(X.iloc[va], label=y[va], reference=train_set)
            booster = lgb.train(
                run,
                train_set,
                num_boost_round=NUM_ROUNDS,
                valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            oof[repeat, va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
            models.append(booster)
            best_iters.append(booster.best_iteration)

    mean_oof = oof.mean(axis=0)
    per_repeat_ap = [average_precision_score(y, oof[r]) for r in range(n_repeats)]
    return {
        "oof": mean_oof,
        "oof_by_repeat": oof,
        "models": models,
        "ap": average_precision_score(y, mean_oof),
        "ap_by_repeat": per_repeat_ap,
        "best_iteration": int(np.mean(best_iters)),
    }


def predict(models, features: pd.DataFrame) -> np.ndarray:
    """Average the rank of every fold model rather than its raw probability.

    Fold models stop at different iterations and are not calibrated against
    each other, so averaging ranks is the blend that AP actually rewards.
    """
    cols = feature_columns(features)
    X = features[cols]
    ranks = np.zeros(len(X))
    for booster in models:
        raw = booster.predict(X, num_iteration=booster.best_iteration)
        ranks += pd.Series(raw).rank(pct=True).to_numpy()
    return ranks / len(models)


def importance(models, features: pd.DataFrame, top=40) -> pd.DataFrame:
    cols = feature_columns(features)
    gains = np.zeros(len(cols))
    for booster in models:
        gains += booster.feature_importance(importance_type="gain")
    table = pd.DataFrame({"feature": cols, "gain": gains / len(models)})
    return table.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
