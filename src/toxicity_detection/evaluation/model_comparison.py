"""Model comparison routines, aligned prediction generation, and visualization generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from toxicity_detection.models.berturk import (
    DEFAULT_MAX_LENGTH,
    DynamicPaddingCollator,
    ToxicityDataset,
)
from toxicity_detection.training.trainer import evaluate_model


def generate_aligned_test_predictions(
    test_df: pd.DataFrame,
    tfidf_pipeline: Pipeline,
    berturk_model: PreTrainedModel,
    berturk_tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Generate aligned predictions and toxic-class probabilities for TF-IDF and BERTurk on the test set.

    Args:
        test_df: Test set DataFrame containing record_id, text_raw, and label_binary.
        tfidf_pipeline: Fitted TF-IDF baseline Pipeline.
        berturk_model: Fine-tuned BERTurk model.
        berturk_tokenizer: Associated BERTurk tokenizer.
        device: CUDA or CPU device.
        max_length: Maximum sequence length.
        batch_size: Evaluation batch size.

    Returns:
        DataFrame containing aligned predictions and probabilities indexed by record_id.
    """
    df_out = test_df[
        ["record_id", "text_raw", "label_binary", "source", "source_group"]
    ].copy()
    df_out["label_true"] = df_out["label_binary"].astype(int)

    # 1. TF-IDF Predictions
    tfidf_preds = tfidf_pipeline.predict(df_out["text_raw"])
    tfidf_probs = tfidf_pipeline.predict_proba(df_out["text_raw"])[:, 1]
    df_out["tfidf_pred"] = tfidf_preds.astype(int)
    df_out["tfidf_prob_toxic"] = np.round(tfidf_probs, 4)

    # 2. BERTurk Predictions
    test_dataset = ToxicityDataset(
        texts=df_out["text_raw"].tolist(), labels=df_out["label_true"].tolist()
    )
    collator = DynamicPaddingCollator(
        tokenizer=berturk_tokenizer, max_length=max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    _, berturk_preds, berturk_probs = evaluate_model(
        berturk_model, test_loader, device
    )
    df_out["berturk_pred"] = berturk_preds.astype(int)
    df_out["berturk_prob_toxic"] = np.round(berturk_probs, 4)

    return df_out.sort_values(by="record_id").reset_index(drop=True)


def generate_berturk_validation_errors(
    val_df: pd.DataFrame,
    tfidf_pipeline: Pipeline,
    berturk_preds: np.ndarray,
    berturk_probs: np.ndarray,
    n_samples: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract categorized validation errors comparing BERTurk and TF-IDF behavior.

    Categorizes errors into:
    - shared_error: Both models are wrong
    - berturk_corrected: TF-IDF is wrong, BERTurk is correct
    - berturk_introduced: TF-IDF is correct, BERTurk is wrong

    Args:
        val_df: Validation DataFrame.
        tfidf_pipeline: Fitted TF-IDF baseline Pipeline.
        berturk_preds: Predictions from fine-tuned BERTurk on validation set.
        berturk_probs: Predicted probabilities from BERTurk on validation set.
        n_samples: Total number of samples to select (default: 30).
        seed: Random seed.

    Returns:
        DataFrame containing categorized errors and human review template fields.
    """
    df = val_df[
        ["record_id", "text_raw", "label_binary", "source", "source_group"]
    ].copy()
    df["label_true"] = df["label_binary"].astype(int)

    tfidf_preds = tfidf_pipeline.predict(df["text_raw"])
    tfidf_probs = tfidf_pipeline.predict_proba(df["text_raw"])[:, 1]

    df["tfidf_pred"] = tfidf_preds.astype(int)
    df["tfidf_prob"] = np.round(tfidf_probs, 4)
    df["berturk_pred"] = np.asarray(berturk_preds, dtype=int)
    df["berturk_prob"] = np.round(berturk_probs, 4)

    # Classify error dynamics
    berturk_wrong = df["berturk_pred"] != df["label_true"]
    tfidf_wrong = df["tfidf_pred"] != df["label_true"]

    df["error_type"] = np.where(
        berturk_wrong,
        np.where(df["berturk_pred"] == 1, "false_positive", "false_negative"),
        "correct",
    )

    df["model_interaction"] = "other"
    df.loc[berturk_wrong & tfidf_wrong, "model_interaction"] = "shared_error"
    df.loc[berturk_wrong & (~tfidf_wrong), "model_interaction"] = (
        "berturk_introduced_error"
    )
    df.loc[(~berturk_wrong) & tfidf_wrong, "model_interaction"] = (
        "berturk_corrected_tfidf"
    )

    # Sample representative instances
    errors_pool = df[berturk_wrong | tfidf_wrong].copy()

    # Stratified deterministic sample across interaction categories
    sampled_dfs = []
    rng_seed = seed
    for cat in [
        "shared_error",
        "berturk_introduced_error",
        "berturk_corrected_tfidf",
    ]:
        sub = errors_pool[errors_pool["model_interaction"] == cat]
        target_n = n_samples // 3
        if len(sub) > 0:
            sample_sub = sub.sort_values(by="record_id").sample(
                n=min(len(sub), target_n), random_state=rng_seed
            )
            sampled_dfs.append(sample_sub)
            rng_seed += 1

    combined = pd.concat(sampled_dfs, ignore_index=True)
    if len(combined) < n_samples:
        remaining = errors_pool[
            ~errors_pool["record_id"].isin(combined["record_id"])
        ]
        needed = n_samples - len(combined)
        if len(remaining) > 0:
            fill_sample = remaining.sort_values(by="record_id").sample(
                n=min(len(remaining), needed), random_state=rng_seed
            )
            combined = pd.concat([combined, fill_sample], ignore_index=True)

    combined["review_category"] = "pending_review"
    combined["review_notes"] = ""

    cols = [
        "record_id",
        "text_raw",
        "label_true",
        "berturk_pred",
        "berturk_prob",
        "tfidf_pred",
        "tfidf_prob",
        "model_interaction",
        "error_type",
        "source",
        "source_group",
        "review_category",
        "review_notes",
    ]
    return combined[cols].sort_values(by=["model_interaction", "record_id"]).reset_index(drop=True)


def plot_training_curves(
    history: list[dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot training and validation loss along with validation Macro-F1 across epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    _fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.set_xlabel("Epoch", fontsize=11, weight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", color="#c0392b", fontsize=11, weight="bold")
    l1 = ax1.plot(
        epochs,
        train_loss,
        "o-",
        color="#e74c3c",
        label="Train Loss",
        linewidth=2,
    )
    l2 = ax1.plot(
        epochs,
        val_loss,
        "s--",
        color="#c0392b",
        label="Val Loss",
        linewidth=2,
    )
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Macro-F1", color="#2980b9", fontsize=11, weight="bold")
    l3 = ax2.plot(
        epochs,
        val_f1,
        "^g-",
        color="#2980b9",
        label="Val Macro-F1",
        linewidth=2,
    )
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    # Combine legends
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title("BERTurk Fine-Tuning Convergence History", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_model_comparison_slices(
    tfidf_slices: dict[str, Any],
    berturk_slices: dict[str, Any],
    out_path: str | Path,
) -> None:
    """Plot comparative Macro-F1 scores across source slices for TF-IDF vs. BERTurk."""
    slices = ["overall", "native_twitter", "jigsaw"]
    display_names = ["Overall Test", "Native Twitter", "Translated Jigsaw"]

    tfidf_scores = [tfidf_slices[s]["macro_f1"] for s in slices]
    berturk_scores = [berturk_slices[s]["macro_f1"] for s in slices]

    x = np.arange(len(slices))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        tfidf_scores,
        width,
        label="TF-IDF + LR Baseline",
        color="#7f8c8d",
    )
    plt.bar(
        x + width / 2,
        berturk_scores,
        width,
        label="BERTurk Transformer",
        color="#2980b9",
    )

    plt.ylabel("Macro-F1 Score", fontsize=11, weight="bold")
    plt.title("Test Set Macro-F1 by Source Domain: TF-IDF vs. BERTurk", fontsize=13, weight="bold", pad=12)
    plt.xticks(x, display_names, fontsize=10)
    plt.ylim(0.85, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    for i in range(len(slices)):
        diff = berturk_scores[i] - tfidf_scores[i]
        sign = "+" if diff >= 0 else ""
        plt.text(
            x[i] + width / 2,
            berturk_scores[i] + 0.005,
            f"{sign}{diff:.4f}",
            ha="center",
            fontsize=9,
            weight="bold",
            color="#16a085" if diff >= 0 else "#c0392b",
        )

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
