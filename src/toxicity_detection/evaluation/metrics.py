"""Classification and sliced evaluation metrics for toxicity detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | list[int] | pd.Series,
    y_pred: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for binary toxicity classification.

    Args:
        y_true: True binary labels (0 = clean, 1 = toxic).
        y_pred: Predicted binary labels (0 = clean, 1 = toxic).
        y_prob: Predicted probabilities for the positive class (toxic = 1).

    Returns:
        Dictionary of computed metrics including Macro-F1, per-class metrics,
        PR-AUC, ROC-AUC, and confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

    # Per-class precision, recall, and F1 (pos_label=0 for clean, pos_label=1 for toxic)
    prec_clean = float(
        precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
    )
    rec_clean = float(
        recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
    )
    f1_clean = float(
        f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
    )

    prec_toxic = float(
        precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    )
    rec_toxic = float(
        recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    )
    f1_toxic = float(
        f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision_clean": round(prec_clean, 4),
        "recall_clean": round(rec_clean, 4),
        "f1_clean": round(f1_clean, 4),
        "precision_toxic": round(prec_toxic, 4),
        "recall_toxic": round(rec_toxic, 4),
        "f1_toxic": round(f1_toxic, 4),
        "confusion_matrix": cm,
        "support_clean": int((y_true_arr == 0).sum()),
        "support_toxic": int((y_true_arr == 1).sum()),
        "support_total": len(y_true_arr),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        # Handle cases where only one class is present in true labels
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) > 1:
            metrics["pr_auc"] = round(
                float(average_precision_score(y_true_arr, y_prob_arr)), 4
            )
            metrics["roc_auc"] = round(
                float(roc_auc_score(y_true_arr, y_prob_arr)), 4
            )
        else:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None

    return metrics


def compute_source_sliced_metrics(
    df: pd.DataFrame,
    y_true_col: str = "label_binary",
    y_pred_col: str = "label_pred",
    y_prob_col: str | None = "pred_prob_toxic",
    source_group_col: str = "source_group",
) -> dict[str, Any]:
    """Compute overall and source-sliced performance metrics.

    Args:
        df: DataFrame containing predictions, ground truth, and source metadata.
        y_true_col: Column name for ground truth binary labels.
        y_pred_col: Column name for predicted binary labels.
        y_prob_col: Column name for predicted positive-class probabilities.
        source_group_col: Column name designating domain / source slice.

    Returns:
        Dictionary mapping slice names ('overall', 'native_twitter', 'jigsaw',
        'other_sources') to their respective classification metrics.
    """
    results: dict[str, Any] = {}

    # 1. Overall metrics
    y_prob = df[y_prob_col] if y_prob_col and y_prob_col in df.columns else None
    results["overall"] = compute_classification_metrics(
        y_true=df[y_true_col], y_pred=df[y_pred_col], y_prob=y_prob
    )

    # 2. Slice metrics
    source_groups = ["native_twitter", "jigsaw", "other_sources"]
    for sg in source_groups:
        slice_df = df[df[source_group_col] == sg]
        if len(slice_df) == 0:
            continue
        slice_prob = (
            slice_df[y_prob_col]
            if y_prob_col and y_prob_col in slice_df.columns
            else None
        )
        results[sg] = compute_classification_metrics(
            y_true=slice_df[y_true_col],
            y_pred=slice_df[y_pred_col],
            y_prob=slice_prob,
        )

    return results
