"""Tests for the deterministic data preparation pipeline, canonicalization, and partitions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from toxicity_detection.config import (
    MANIFEST_PATH,
    PROCESSED_ALL_PATH,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    QUARANTINE_CONFLICTS_PATH,
    TEST_CHECKSUM_PATH,
    TEST_MEMBERSHIP_PATH,
)
from toxicity_detection.data import (
    canonicalize_text,
    compute_file_sha256,
    compute_membership_hash,
    compute_record_id,
    filter_scam_records,
    map_and_validate_binary_labels,
)


def test_canonicalize_text_rules():
    """Verify that canonicalize_text properly implements Policy 4/5."""
    # 1. Whitespace collapsing and stripping
    raw = "  Bu   bir \t test \n  cümlesidir.   "
    assert canonicalize_text(raw) == "bu bir test cümlesidir."

    # 2. Turkish-aware lowercasing
    # Dotted capital I (İ) -> dotted small i (i)
    assert canonicalize_text("İSTANBUL") == "istanbul"
    # Dotless capital I (I) -> dotless small i (ı)
    assert canonicalize_text("ISPARTA") == "ısparta"
    # Mixed Turkish characters
    assert (
        canonicalize_text("ÇİĞDEM VE IŞIL ŞÖLENİ")
        == "çiğdem ve ışıl şöleni"
    )

    # 3. Unicode NFC form
    decomposed = "c\u0327"  # 'c' + combining cedilla (NFD)
    composed = "\u00e7"  # 'ç' (NFC)
    assert canonicalize_text(decomposed) == canonicalize_text(composed)

    # 4. Input validation
    with pytest.raises(TypeError):
        canonicalize_text(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canonicalize_text(12345)  # type: ignore[arg-type]


def test_compute_record_id():
    """Verify cryptographic SHA-256 record ID computation."""
    text = "bu bir deneme metnidir"
    record_id = compute_record_id(text)
    assert len(record_id) == 64
    assert all(c in "0123456789abcdef" for c in record_id)

    # Invariant to identical string
    assert compute_record_id(text) == record_id

    # Non-string raises TypeError
    with pytest.raises(TypeError):
        compute_record_id(None)  # type: ignore[arg-type]


def test_filter_scam_records_validation():
    """Verify scam record isolation and exception handling."""
    dummy_data = {
        "Dolandırıcılık": [1.0, 1.0, 0.0],
        "Küfür": [0.0, 1.0, 0.0],
        "Tehdit": [0.0, 0.0, 0.0],
        "Hakaret": [0.0, 0.0, 0.0],
        "Cinsel İçerik": [0.0, 0.0, 0.0],
        "Irkçılık": [0.0, 0.0, 0.0],
        "Tespit": [0.0, 0.0, 1.0],
    }
    df = pd.DataFrame(dummy_data)

    # Only row 0 is pure scam (Dolandırıcılık=1, all other flags=0)
    # Row 1 has Küfür=1, so it is toxic, not pure scam
    retained, scam = filter_scam_records(df, expected_count=1)
    assert len(scam) == 1
    assert len(retained) == 2

    # Expecting 2 when only 1 exists should raise ValueError
    with pytest.raises(ValueError, match="Expected exactly 2"):
        filter_scam_records(df, expected_count=2)


def test_map_and_validate_binary_labels():
    """Verify binary label mapping and error raising for invalid labels."""
    df_hf_valid = pd.DataFrame({"is_toxic": [0, 1, 0]})
    df_kg_valid = pd.DataFrame({"Tespit": [1.0, 0.0, 1.0]})

    df_hf_out, df_kg_out = map_and_validate_binary_labels(
        df_hf_valid, df_kg_valid
    )
    assert list(df_hf_out["label_binary"]) == [0, 1, 0]
    assert list(df_kg_out["label_binary"]) == [0, 1, 0]

    # Invalid Overfit-GM label
    df_hf_invalid = pd.DataFrame({"is_toxic": [0, 2]})
    with pytest.raises(ValueError, match="Unexpected values in Overfit-GM"):
        map_and_validate_binary_labels(df_hf_invalid, df_kg_valid)

    # Invalid Kaggle label
    df_kg_invalid = pd.DataFrame({"Tespit": [1.0, -1.0]})
    with pytest.raises(ValueError, match="Unexpected values in Kaggle"):
        map_and_validate_binary_labels(df_hf_valid, df_kg_invalid)


def test_quarantine_conflicts_file_content():
    """Verify that quarantine_conflicts.parquet exists and satisfies all specifications."""
    assert QUARANTINE_CONFLICTS_PATH.exists()
    df = pd.read_parquet(QUARANTINE_CONFLICTS_PATH)

    # Exactly 4 physical rows (2 for internal Kaggle tie, 2 for cross-dataset conflict)
    assert len(df) == 4

    expected_cols = [
        "dataset_origin",
        "original_row_index",
        "record_id",
        "text_raw",
        "text_canonical",
        "label_binary",
        "raw_label_info",
        "quarantine_reason",
    ]
    assert list(df.columns) == expected_cols

    # Verify reasons
    reasons = set(df["quarantine_reason"])
    assert reasons == {"kaggle_internal_tie", "cross_dataset_disagreement"}

    # Verify 2 unique canonical keys
    assert df["text_canonical"].nunique() == 2


def test_processed_partitions_integrity():
    """Verify row counts, zero overlap, and class distributions across train, val, and test."""
    assert PROCESSED_ALL_PATH.exists()
    assert PROCESSED_TRAIN_PATH.exists()
    assert PROCESSED_VAL_PATH.exists()
    assert PROCESSED_TEST_PATH.exists()

    df_all = pd.read_parquet(PROCESSED_ALL_PATH)
    df_train = pd.read_parquet(PROCESSED_TRAIN_PATH)
    df_val = pd.read_parquet(PROCESSED_VAL_PATH)
    df_test = pd.read_parquet(PROCESSED_TEST_PATH)

    # Invariant 1: Total counts
    assert len(df_all) == 84866
    assert len(df_train) == 63650
    assert len(df_val) == 12731
    assert len(df_test) == 8485
    assert len(df_train) + len(df_val) + len(df_test) == len(df_all)

    # Invariant 2: Zero leakage between partitions
    train_ids = set(df_train["record_id"])
    val_ids = set(df_val["record_id"])
    test_ids = set(df_test["record_id"])
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Invariant 3: Zero canonical text leakage
    train_texts = set(df_train["text_canonical"])
    val_texts = set(df_val["text_canonical"])
    test_texts = set(df_test["text_canonical"])
    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
    assert len(val_texts.intersection(test_texts)) == 0

    # Invariant 4: Overall class balance
    clean_total = (df_all["label_binary"] == 0).sum()
    toxic_total = (df_all["label_binary"] == 1).sum()
    assert clean_total == 41376
    assert toxic_total == 43490

    # Invariant 5: Partition class counts
    assert (df_train["label_binary"] == 0).sum() == 31032
    assert (df_train["label_binary"] == 1).sum() == 32618
    assert (df_val["label_binary"] == 0).sum() == 6207
    assert (df_val["label_binary"] == 1).sum() == 6524
    assert (df_test["label_binary"] == 0).sum() == 4137
    assert (df_test["label_binary"] == 1).sum() == 4348

    # Invariant 6: Slice counts
    assert (df_all["source_group"] == "native_twitter").sum() == 49163
    assert (df_all["source_group"] == "jigsaw").sum() == 35603
    assert (df_all["source_group"] == "other_sources").sum() == 100

    assert (df_train["source_group"] == "native_twitter").sum() == 36872
    assert (df_val["source_group"] == "native_twitter").sum() == 7375
    assert (df_test["source_group"] == "native_twitter").sum() == 4916

    assert (df_train["source_group"] == "jigsaw").sum() == 26702
    assert (df_val["source_group"] == "jigsaw").sum() == 5341
    assert (df_test["source_group"] == "jigsaw").sum() == 3560

    assert (df_train["source_group"] == "other_sources").sum() == 76
    assert (df_val["source_group"] == "other_sources").sum() == 15
    assert (df_test["source_group"] == "other_sources").sum() == 9


def test_checksums_and_manifest_validity():
    """Verify cryptographic test lockfiles and manifest JSON structure."""
    assert TEST_CHECKSUM_PATH.exists()
    assert TEST_MEMBERSHIP_PATH.exists()
    assert MANIFEST_PATH.exists()

    # Verify physical file checksum matches stored checksum
    physical_hash_on_disk = TEST_CHECKSUM_PATH.read_text(
        encoding="utf-8"
    ).strip()
    actual_physical_hash = compute_file_sha256(PROCESSED_TEST_PATH)
    assert physical_hash_on_disk == actual_physical_hash

    # Verify membership hash matches stored checksum
    df_test = pd.read_parquet(PROCESSED_TEST_PATH)
    membership_hash_on_disk = TEST_MEMBERSHIP_PATH.read_text(
        encoding="utf-8"
    ).strip()
    actual_membership_hash = compute_membership_hash(df_test)
    assert membership_hash_on_disk == actual_membership_hash

    # Verify manifest JSON
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["pipeline_version"] == "1.0.0"
    assert manifest["random_seed"] == 42
    assert manifest["filtering_and_deduplication"]["retained_total_records"] == 84866
    assert manifest["partitions"]["train"]["count"] == 63650
    assert manifest["partitions"]["val"]["count"] == 12731
    assert manifest["partitions"]["test"]["count"] == 8485
    assert manifest["checksums"]["test_parquet_sha256"] == actual_physical_hash
    assert (
        manifest["checksums"]["test_membership_sha256"]
        == actual_membership_hash
    )
