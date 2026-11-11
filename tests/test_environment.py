"""Minimal environment and stack verification test suite.

Verifies:
- Scientific computing imports and basic operations (NumPy, Pandas, SciPy).
- Small scikit-learn Logistic Regression training example.
- Basic PyTorch tensor operations on CPU and GPU (if available).
- Transformers imports and configuration initialization without model downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from transformers import AutoConfig, BertConfig


def test_scientific_computing():
    """Verify NumPy, Pandas, and SciPy basic functionality."""
    # NumPy
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dot_product = np.dot(a, b)
    assert dot_product == 32.0

    # Pandas
    df = pd.DataFrame({"text_len": [10, 20, 30], "label": [0, 1, 0]})
    assert df["text_len"].mean() == 20.0
    assert len(df) == 3

    # SciPy
    res = stats.describe(a)
    assert res.nobs == 3


def test_sklearn_logistic_regression():
    """Verify small scikit-learn Logistic Regression training example."""
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42,
    )
    clf = LogisticRegression(solver="lbfgs", random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (50,)
    accuracy = (preds == y).mean()
    assert accuracy > 0.6  # Baseline sanity check


def test_pytorch_tensor_operations():
    """Verify PyTorch tensor creation, arithmetic, and GPU availability."""
    # CPU operations
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    z = torch.matmul(x, y)
    expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
    assert torch.allclose(z, expected)

    # CUDA device operations (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        x_cuda = x.to(device)
        y_cuda = y.to(device)
        z_cuda = torch.matmul(x_cuda, y_cuda)
        assert z_cuda.is_cuda
        assert torch.allclose(z_cuda.cpu(), expected)


def test_transformers_configuration():
    """Verify Transformers imports and configuration without model downloads."""
    # Custom BertConfig (no external download required)
    bert_config = BertConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    assert bert_config.vocab_size == 32000
    assert bert_config.hidden_size == 768
    assert bert_config.model_type == "bert"

    # Verify AutoConfig class availability
    assert hasattr(AutoConfig, "from_pretrained")


def test_explainability_and_visualization_imports():
    """Verify SHAP, Matplotlib, and Seaborn imports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap

    assert hasattr(shap, "Explainer")
    fig, ax = plt.subplots()
    sns.lineplot(x=[1, 2, 3], y=[4, 5, 6], ax=ax)
    plt.close(fig)
