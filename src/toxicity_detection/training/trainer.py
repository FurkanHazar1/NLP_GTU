"""Supervised training and evaluation routines for BERTurk."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from toxicity_detection.evaluation import compute_classification_metrics
from toxicity_detection.models.berturk import (
    DEFAULT_MAX_LENGTH,
    DynamicPaddingCollator,
    ToxicityDataset,
    load_berturk_model,
)
from toxicity_detection.training.seed import set_seed

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> float:
    """Execute one full training epoch with mixed-precision and gradient accumulation.

    Args:
        model: PyTorch classification model.
        loader: Training DataLoader.
        optimizer: Torch optimizer (AdamW).
        scheduler: Learning rate scheduler.
        device: CUDA or CPU device.
        grad_clip: Maximum gradient norm for clipping.
        grad_accum_steps: Number of forward batches before optimizer step.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # Use bfloat16 on modern CUDA GPUs, otherwise standard fp32
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
        else:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        total_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset and return average loss, predictions, and toxic probabilities.

    Args:
        model: PyTorch model.
        loader: Evaluation DataLoader.
        device: Target torch device.

    Returns:
        Tuple of (avg_loss, predictions_array, toxic_probabilities_array).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_probs: list[float] = []

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.float().cpu().numpy().tolist())

    avg_loss = (
        (total_loss / len(loader))
        if (len(loader) > 0 and total_loss > 0)
        else 0.0
    )
    return avg_loss, np.array(all_preds, dtype=int), np.array(all_probs, dtype=float)


def train_berturk_with_early_stopping(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    patience: int = 2,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[int, dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Fine-tune BERTurk with validation checkpointing and early stopping.

    Args:
        train_df: Training DataFrame with text_raw and label_binary.
        val_df: Validation DataFrame with text_raw and label_binary.
        tokenizer: Pretrained BERTurk tokenizer.
        max_length: Maximum sequence token length.
        batch_size: Effective batch size.
        learning_rate: Peak learning rate for AdamW.
        max_epochs: Maximum epochs.
        patience: Early stopping patience in epochs.
        warmup_ratio: Proportion of warmup steps in linear schedule.
        weight_decay: L2 penalty for AdamW.
        grad_clip: Maximum gradient norm for clipping.
        seed: Random seed.
        device: Target device (defaults to CUDA if available).

    Returns:
        Tuple of (best_epoch, best_val_metrics, training_history, best_val_preds, best_val_probs).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    # 1. Dataset & DataLoaders with Dynamic Padding
    train_dataset = ToxicityDataset(
        texts=train_df["text_raw"].tolist(),
        labels=train_df["label_binary"].tolist(),
    )
    val_dataset = ToxicityDataset(
        texts=val_df["text_raw"].tolist(),
        labels=val_df["label_binary"].tolist(),
    )

    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model Initialization
    model = load_berturk_model()
    model.to(device)

    # 3. Optimizer & Linear Warmup Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_training_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # 4. Training Loop with Validation Checkpointing
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 1
    best_val_metrics: dict[str, Any] = {}
    best_preds = np.zeros(len(val_df), dtype=int)
    best_probs = np.zeros(len(val_df), dtype=float)
    no_improvement_count = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss, val_preds, val_probs = evaluate_model(
            model=model, loader=val_loader, device=device
        )
        duration = time.time() - t0

        val_metrics = compute_classification_metrics(
            y_true=val_df["label_binary"],
            y_pred=val_preds,
            y_prob=val_probs,
        )
        val_f1 = val_metrics["macro_f1"]

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": val_f1,
            "val_accuracy": val_metrics["accuracy"],
            "val_pr_auc": val_metrics["pr_auc"],
            "duration_seconds": round(duration, 2),
            "metrics": val_metrics,
        }
        history.append(record)
        logger.info(
            "  Stage 1 Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Macro-F1: %.4f (Acc: %.4f, AP: %.4f) [%.1fs]",
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            val_f1,
            val_metrics["accuracy"],
            val_metrics["pr_auc"],
            duration,
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_preds = val_preds
            best_probs = val_probs
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                # Early stopping triggered
                break

    return best_epoch, best_val_metrics, history, best_preds, best_probs
