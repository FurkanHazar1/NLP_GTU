"""Evaluation and error analysis modules."""

from __future__ import annotations

from toxicity_detection.evaluation.bootstrap import paired_bootstrap_macro_f1
from toxicity_detection.evaluation.error_analysis import (
    export_validation_errors,
    extract_validation_errors,
)
from toxicity_detection.evaluation.metrics import (
    compute_classification_metrics,
    compute_source_sliced_metrics,
)
from toxicity_detection.evaluation.model_comparison import (
    generate_aligned_test_predictions,
    generate_berturk_validation_errors,
    plot_model_comparison_slices,
    plot_training_curves,
)
from toxicity_detection.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
)

__all__ = [
    "compute_classification_metrics",
    "compute_source_sliced_metrics",
    "export_validation_errors",
    "extract_validation_errors",
    "generate_aligned_test_predictions",
    "generate_berturk_validation_errors",
    "paired_bootstrap_macro_f1",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_model_comparison_slices",
    "plot_training_curves",
]
