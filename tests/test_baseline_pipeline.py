"""Tests for TF-IDF baseline models, evaluation metrics, and error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from toxicity_detection.config import (
    PROCESSED_TEST_PATH,
    TEST_CHECKSUM_PATH,
    TEST_MEMBERSHIP_PATH,
)
from toxicity_detection.data import compute_file_sha256, compute_membership_hash
from toxicity_detection.evaluation import (
    compute_classification_metrics,
    compute_source_sliced_metrics,
    extract_validation_errors,
)
from toxicity_detection.models import (
    build_char_vectorizer,
    build_combined_vectorizer,
    build_tfidf_pipeline,
    build_word_vectorizer,
    load_model_artifact,
    save_model_artifact,
    turkish_lowercase_preprocessor,
)


def test_turkish_lowercase_preprocessor():
    """Verify Turkish-aware lowercasing preserving dotted and dotless I."""
    raw = "İSTANBUL ve ISPARTA IŞIKLARI"
    expected = "istanbul ve ısparta ışıkları"
    assert turkish_lowercase_preprocessor(raw) == expected


def test_feature_vectorizers_sparsity():
    """Verify that word, char, and combined extractors produce sparse matrices without densifying."""
    corpus = [
        "Bu bir test metnidir ve örnek bir cümledir.",
        "İkinci test metni burada örnek olarak yer alır.",
        "Üçüncü test metnidir ve Türkçe karakterler içerir: ç, ğ, ö, ş, ü.",
    ]

    word_vec = build_word_vectorizer()
    char_vec = build_char_vectorizer()
    combined_vec = build_combined_vectorizer()

    for vec in [word_vec, char_vec, combined_vec]:
        matrix = vec.fit_transform(corpus)
        assert issparse(matrix), f"Expected sparse matrix, got {type(matrix)}"
        assert matrix.shape[0] == 3
        assert matrix.shape[1] > 0


def test_pipeline_isolation_no_leakage():
    """Verify that vectorizer vocabulary is not modified or leaked during inference."""
    train_texts = ["zararlı ve zehirli içerik", "temiz ve zararsız mesaj"]
    train_labels = [1, 0]

    val_texts = ["tamamen yeni kelimeler içeren bilinmeyen cümle"]

    pipeline = build_tfidf_pipeline(
        representation="word", c_val=1.0, preprocessor_type="turkish_lower"
    )
    pipeline.fit(train_texts, train_labels)

    feature_step = pipeline.named_steps["features"]
    vocab_before = dict(feature_step.vocabulary_)

    # Inference on validation data containing out-of-vocabulary words
    pipeline.predict(val_texts)
    pipeline.predict_proba(val_texts)

    vocab_after = dict(feature_step.vocabulary_)
    assert vocab_before == vocab_after, (
        "Vocabulary was altered during inference on validation text!"
    )


def test_classification_metrics_computation():
    """Verify metric calculations including Macro-F1, PR-AUC, and Confusion Matrix."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]  # 1 TN, 1 FP, 1 FN, 1 TP
    y_prob = [0.1, 0.8, 0.4, 0.9]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["precision_clean"] == 0.5
    assert metrics["recall_clean"] == 0.5
    assert metrics["precision_toxic"] == 0.5
    assert metrics["recall_toxic"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["pr_auc"] is not None
    assert metrics["roc_auc"] is not None


def test_source_sliced_metrics():
    """Verify computation of sliced metrics across source groups."""
    df = pd.DataFrame(
        {
            "label_binary": [0, 1, 0, 1, 0],
            "label_pred": [0, 1, 0, 0, 0],
            "pred_prob_toxic": [0.1, 0.9, 0.2, 0.4, 0.05],
            "source_group": [
                "native_twitter",
                "native_twitter",
                "jigsaw",
                "jigsaw",
                "other_sources",
            ],
        }
    )

    sliced = compute_source_sliced_metrics(df)
    assert "overall" in sliced
    assert "native_twitter" in sliced
    assert "jigsaw" in sliced
    assert "other_sources" in sliced
    assert sliced["overall"]["support_total"] == 5
    assert sliced["native_twitter"]["support_total"] == 2
    assert sliced["jigsaw"]["support_total"] == 2
    assert sliced["other_sources"]["support_total"] == 1


def test_error_analysis_extraction():
    """Verify extraction of exactly 30 misclassified validation examples."""
    np.random.seed(42)
    # Generate 100 validation examples with deliberate errors
    n_rows = 100
    y_true = np.random.choice([0, 1], size=n_rows)
    # Invert 40 predictions to create 20 FP and 20 FN approximately
    y_pred = y_true.copy()
    y_pred[:40] = 1 - y_pred[:40]
    y_prob = np.where(y_pred == 1, 0.85, 0.15)

    dummy_val = pd.DataFrame(
        {
            "record_id": [f"id_{i:04d}" for i in range(n_rows)],
            "text_raw": [f"Örnek metin {i}" for i in range(n_rows)],
            "label_binary": y_true,
            "source": ["told"] * n_rows,
            "source_group": ["native_twitter"] * n_rows,
        }
    )

    errors_df = extract_validation_errors(
        dummy_val, y_pred, y_prob, n_samples=30, random_state=42
    )

    assert len(errors_df) == 30
    assert set(errors_df["error_type"].unique()).issubset(
        {"false_positive", "false_negative"}
    )
    assert "review_category" in errors_df.columns
    assert (errors_df["review_category"] == "pending_review").all()
    assert "review_notes" in errors_df.columns

    # Verify determinism
    errors_df2 = extract_validation_errors(
        dummy_val, y_pred, y_prob, n_samples=30, random_state=42
    )
    pd.testing.assert_frame_equal(errors_df, errors_df2)


def test_model_serialization_roundtrip(tmp_path):
    """Verify joblib serialization and deserialization of fitted pipeline."""
    train_texts = [
        "harika bir gün ve çok güzel bir hava",
        "kötü bir gün ve iğrenç bir hava",
    ]
    train_labels = [0, 1]

    pipe = build_tfidf_pipeline("word", c_val=1.0)
    pipe.fit(train_texts, train_labels)

    config = {"model_name": "test_word_model", "c_val": 1.0}
    save_model_artifact(pipe, config, tmp_path)

    loaded_pipe, loaded_config = load_model_artifact(tmp_path)
    assert loaded_config["model_name"] == "test_word_model"

    preds_original = pipe.predict(train_texts)
    preds_loaded = loaded_pipe.predict(train_texts)
    np.testing.assert_array_equal(preds_original, preds_loaded)


def test_test_set_lockfile_integrity():
    """Verify that the test set has remained completely unchanged."""
    recorded_file_sha = TEST_CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    actual_file_sha = compute_file_sha256(PROCESSED_TEST_PATH)
    assert recorded_file_sha == actual_file_sha

    test_df = pd.read_parquet(PROCESSED_TEST_PATH)
    recorded_mem_sha = TEST_MEMBERSHIP_PATH.read_text(encoding="utf-8").strip()
    actual_mem_sha = compute_membership_hash(test_df)
    assert recorded_mem_sha == actual_mem_sha
