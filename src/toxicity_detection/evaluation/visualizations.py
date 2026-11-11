"""Visualization generators for baseline evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: list[list[int]],
    out_path: str | Path,
    title: str = "Baseline Test Set Confusion Matrix",
) -> None:
    """Generate and save an annotated confusion matrix plot.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        out_path: File path to save the output PNG.
        title: Plot title.
    """
    cm_arr = np.array(cm)
    total = cm_arr.sum()

    labels = [["Clean (0)", "Toxic (1)"], ["Clean (0)", "Toxic (1)"]]
    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(2):
        for j in range(2):
            count = cm_arr[i, j]
            pct = count / total * 100
            annot[i, j] = f"{count:,}\n({pct:.1f}%)"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels[0],
        yticklabels=labels[1],
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(title, fontsize=13, pad=12, weight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_model_comparison(
    candidates: list[dict[str, Any]],
    out_path: str | Path,
    title: str = "Validation Macro-F1 Across TF-IDF Configurations",
) -> None:
    """Generate and save a comparison bar chart of candidate models on the validation set.

    Args:
        candidates: List of candidate result dictionaries.
        out_path: File path to save the output PNG.
        title: Plot title.
    """
    names = [c["name"] for c in candidates]
    macro_f1s = [c["metrics"]["macro_f1"] for c in candidates]
    accs = [c["metrics"]["accuracy"] for c in candidates]

    x = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, macro_f1s, width, label="Macro-F1", color="#2b5c8f")
    plt.bar(x + width / 2, accs, width, label="Accuracy", color="#6baed6")

    plt.ylabel("Score", fontsize=11)
    plt.title(title, fontsize=13, pad=12, weight="bold")
    plt.xticks(x, names, rotation=35, ha="right", fontsize=9)
    plt.ylim(0.85, 0.96)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight max macro-f1
    best_idx = int(np.argmax(macro_f1s))
    plt.scatter(
        [x[best_idx] - width / 2],
        [macro_f1s[best_idx] + 0.003],
        color="crimson",
        marker="*",
        s=120,
        zorder=5,
        label="Best Candidate",
    )

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
