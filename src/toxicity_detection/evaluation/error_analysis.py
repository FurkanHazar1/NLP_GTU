"""Error analysis and deterministic extraction of misclassified validation examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_validation_errors(
    val_df: pd.DataFrame,
    y_pred: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series,
    n_samples: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract a balanced, deterministic sample of misclassified validation instances.

    Args:
        val_df: Validation DataFrame containing record_id, text_raw, label_binary,
            source, and source_group.
        y_pred: Predicted binary labels on the validation set.
        y_prob: Predicted probabilities for the toxic class (1).
        n_samples: Total number of misclassified records to sample (default: 30).
        random_state: Random state for deterministic sampling (default: 42).

    Returns:
        DataFrame containing formatted error records with human review template fields.
    """
    df = val_df.copy().reset_index(drop=True)
    df["label_pred"] = np.asarray(y_pred, dtype=int)
    df["pred_prob_toxic"] = np.asarray(y_prob, dtype=float).round(4)
    df["label_true"] = df["label_binary"].astype(int)

    # Identify false positives (clean misclassified as toxic)
    fp_mask = (df["label_true"] == 0) & (df["label_pred"] == 1)
    fp_df = df[fp_mask].copy()
    fp_df["error_type"] = "false_positive"

    # Identify false negatives (toxic misclassified as clean)
    fn_mask = (df["label_true"] == 1) & (df["label_pred"] == 0)
    fn_df = df[fn_mask].copy()
    fn_df["error_type"] = "false_negative"

    n_fp_target = n_samples // 2
    n_fn_target = n_samples - n_fp_target

    # Deterministic sorting before sampling guarantees invariance across environments
    fp_sampled = (
        fp_df.sort_values(by="record_id")
        .sample(n=min(len(fp_df), n_fp_target), random_state=random_state)
    )
    fn_sampled = (
        fn_df.sort_values(by="record_id")
        .sample(n=min(len(fn_df), n_fn_target), random_state=random_state)
    )

    sampled = pd.concat([fp_sampled, fn_sampled], ignore_index=True)
    # If one category has fewer than its target, backfill from the other
    if len(sampled) < n_samples:
        needed = n_samples - len(sampled)
        if len(fp_sampled) < n_fp_target and len(fn_df) > len(fn_sampled):
            remaining_fn = fn_df[~fn_df["record_id"].isin(fn_sampled["record_id"])]
            extra_fn = remaining_fn.sort_values(by="record_id").sample(
                n=min(len(remaining_fn), needed), random_state=random_state
            )
            sampled = pd.concat([sampled, extra_fn], ignore_index=True)
        elif len(fn_sampled) < n_fn_target and len(fp_df) > len(fp_sampled):
            remaining_fp = fp_df[~fp_df["record_id"].isin(fp_sampled["record_id"])]
            extra_fp = remaining_fp.sort_values(by="record_id").sample(
                n=min(len(remaining_fp), needed), random_state=random_state
            )
            sampled = pd.concat([sampled, extra_fp], ignore_index=True)

    # Format human review columns
    sampled["review_category"] = "pending_review"
    sampled["review_notes"] = ""

    output_columns = [
        "record_id",
        "text_raw",
        "label_true",
        "label_pred",
        "pred_prob_toxic",
        "source",
        "source_group",
        "error_type",
        "review_category",
        "review_notes",
    ]

    return sampled[output_columns].sort_values(by=["error_type", "record_id"]).reset_index(drop=True)


def export_validation_errors(errors_df: pd.DataFrame, out_path: str | Path) -> None:
    """Save error analysis DataFrame to CSV.

    Args:
        errors_df: DataFrame of error analysis records.
        out_path: File path to save CSV.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors_df.to_csv(path, index=False, encoding="utf-8")
