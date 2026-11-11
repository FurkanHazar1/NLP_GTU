"""Paired bootstrap hypothesis testing and confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary Macro-F1 rapidly using bincount.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Macro-averaged F1 score.
    """
    code = 2 * y_true + y_pred
    bc = np.bincount(code, minlength=4)
    tn, fp, fn, tp = bc[0], bc[1], bc[2], bc[3]

    denom_0 = 2 * tn + fp + fn
    f1_0 = (2 * tn / denom_0) if denom_0 > 0 else 0.0

    denom_1 = 2 * tp + fp + fn
    f1_1 = (2 * tp / denom_1) if denom_1 > 0 else 0.0

    return (f1_0 + f1_1) / 2.0


def paired_bootstrap_macro_f1(
    y_true: np.ndarray | list[int],
    y_pred_a: np.ndarray | list[int],
    y_pred_b: np.ndarray | list[int],
    n_bootstraps: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    delta_min: float = 0.020,
) -> dict[str, Any]:
    """Execute paired bootstrap resampling to compare two models on the same test set.

    Calculates:
        Delta = F1(Model A) - F1(Model B)
    across n_bootstraps paired resamples with replacement.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_a: Predicted binary labels from Model A (e.g. BERTurk).
        y_pred_b: Predicted binary labels from Model B (e.g. TF-IDF).
        n_bootstraps: Number of bootstrap iterations (default: 10,000).
        alpha: Significance level for two-sided percentile interval (default: 0.05 for 95% CI).
        seed: Random seed for deterministic resampling (default: 42).
        delta_min: Meaningful difference threshold (default: 0.020).

    Returns:
        Dictionary containing observed scores, delta, percentile confidence interval,
        empirical p-value, and practical significance conclusions.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_a = np.asarray(y_pred_a, dtype=int)
    y_b = np.asarray(y_pred_b, dtype=int)

    n_samples = len(y_t)
    if not (len(y_a) == len(y_b) == n_samples):
        raise ValueError("Arrays y_true, y_pred_a, and y_pred_b must have identical lengths")

    obs_f1_a = _fast_macro_f1(y_t, y_a)
    obs_f1_b = _fast_macro_f1(y_t, y_b)
    obs_delta = obs_f1_a - obs_f1_b

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_yt = y_t[idx]
        f1_a = _fast_macro_f1(sample_yt, y_a[idx])
        f1_b = _fast_macro_f1(sample_yt, y_b[idx])
        deltas[i] = f1_a - f1_b

    # Percentile confidence interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(deltas, lower_pct))
    ci_upper = float(np.percentile(deltas, upper_pct))

    # Empirical two-sided p-value against null hypothesis (Delta <= 0)
    # Proportion of bootstrap replicates where delta <= 0
    p_val_le_zero = float((deltas <= 0.0).mean())
    p_val_ge_zero = float((deltas >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(p_val_le_zero, p_val_ge_zero))

    # Statistical and practical decision criteria
    is_statistically_significant = bool(ci_lower > 0.0 or ci_upper < 0.0)
    exceeds_practical_threshold = bool(obs_delta >= delta_min)

    return {
        "model_a_macro_f1": round(obs_f1_a, 4),
        "model_b_macro_f1": round(obs_f1_b, 4),
        "observed_delta": round(obs_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": f"{int((1 - alpha) * 100)}%",
        "n_bootstraps": n_bootstraps,
        "bootstrap_mean_delta": round(float(np.mean(deltas)), 4),
        "bootstrap_std_delta": round(float(np.std(deltas)), 4),
        "p_value": round(p_value, 6),
        "delta_min": delta_min,
        "is_statistically_significant": is_statistically_significant,
        "exceeds_practical_threshold": exceeds_practical_threshold,
        "both_criteria_met": bool(
            is_statistically_significant and exceeds_practical_threshold
        ),
    }
