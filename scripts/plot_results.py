"""Render the evaluation evidence: PR curve, score separation and importances."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from dd import model as modelmod
from dd.loading import load_labels
from dd.pipeline import build_split_features

INK = "#111827"
ACCENT = "#2563eb"
WARN = "#dc2626"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--out", default="artifacts/results.png")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    features = build_split_features(args.train, args.cache, "train")
    labels = load_labels(args.train).set_index("TRANSACTION_HASH")
    y = labels["is_suspicious"].reindex(features["TRANSACTION_HASH"]).to_numpy().astype(int)
    difficulty = labels["difficulty"].reindex(features["TRANSACTION_HASH"]).to_numpy()

    cv = modelmod.cross_validate(features, y, args.folds, args.repeats)
    scores = cv["oof"]
    ap = average_precision_score(y, scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Decode & Detect — out-of-fold evaluation on {len(y):,} labelled transactions",
        fontsize=15, fontweight="bold", color=INK,
    )

    precision, recall, _ = precision_recall_curve(y, scores)
    ax = axes[0, 0]
    ax.plot(recall, precision, color=ACCENT, lw=2.2)
    ax.axhline(y.mean(), color=WARN, ls="--", lw=1.2, label=f"random baseline ({y.mean():.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision–recall  (AP = {ap:.4f}, score {100 * ap:.2f})", color=INK)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 60)
    ax.hist(scores[y == 0], bins=bins, color="#9ca3af", alpha=0.85, label="legitimate", log=True)
    ax.hist(scores[y == 1], bins=bins, color=WARN, alpha=0.85, label="suspicious", log=True)
    ax.set_xlabel("out-of-fold suspicion score")
    ax.set_ylabel("transactions (log scale)")
    ax.set_title("Score separation", color=INK)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    order = np.argsort(-scores)
    ranked = y[order]
    budgets = np.arange(1, len(ranked) + 1)
    captured = np.cumsum(ranked) / max(1, y.sum())
    ax.plot(100 * budgets / len(ranked), 100 * captured, color=ACCENT, lw=2.2)
    ax.plot([0, 100], [0, 100], color="#9ca3af", ls="--", lw=1.2, label="random review")
    for mark in (1, 5, 10):
        idx = int(len(ranked) * mark / 100) - 1
        ax.scatter([mark], [100 * captured[idx]], color=WARN, zorder=5)
        ax.annotate(
            f"{100 * captured[idx]:.0f}% at top {mark}%",
            (mark, 100 * captured[idx]), textcoords="offset points", xytext=(8, -4), fontsize=9, color=INK,
        )
    ax.set_xlabel("share of transactions reviewed (%)")
    ax.set_ylabel("suspicious transactions caught (%)")
    ax.set_title("Analyst review curve", color=INK)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    top = modelmod.importance(cv["models"], features, top=15).iloc[::-1]
    ax.barh(top["feature"], top["gain"], color=ACCENT)
    ax.set_xlabel("mean gain")
    ax.set_title("Most informative features", color=INK)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    rows = []
    for level in ("easy", "medium", "hard"):
        mask = (difficulty == level) & (y == 1)
        if mask.any():
            rows.append(
                {
                    "difficulty": level,
                    "positives": int(mask.sum()),
                    "median_percentile": round(float(pd.Series(scores).rank(pct=True).to_numpy()[mask].mean()), 4),
                    "caught_in_top_5pct": int((scores[mask] >= np.quantile(scores, 0.95)).sum()),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
