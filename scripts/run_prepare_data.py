#!/usr/bin/env python3
"""CLI runner for the deterministic data preparation and partitioning pipeline."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from toxicity_detection.config import (
    KAGGLE_TWEETS_EXPECTED_SHA256,
    KAGGLE_TWEETS_RAW_PATH,
    MANIFEST_PATH,
    OVERFIT_GM_EXPECTED_SHA256,
    OVERFIT_GM_RAW_PATH,
    PROCESSED_ALL_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    QUARANTINE_CONFLICTS_PATH,
    QUARANTINED_RAW_TEXTS,
    RANDOM_SEED,
    TEST_CHECKSUM_PATH,
    TEST_MEMBERSHIP_PATH,
)
from toxicity_detection.data import (
    canonicalize_text,
    compute_file_sha256,
    compute_membership_hash,
    compute_record_id,
    deduplicate_and_merge,
    filter_scam_records,
    identify_quarantine_rows,
    map_and_validate_binary_labels,
    read_xlsx_pure_python,
    save_quarantine_records,
    stratified_partition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_data")


def main() -> int:
    """Run the complete end-to-end data preparation pipeline."""
    start_time = datetime.now(UTC)
    logger.info("=== Starting Data Preparation Pipeline (Phase 3.2) ===")

    # Step 1: Verify raw file checksums
    logger.info("Verifying raw dataset cryptographic checksums...")
    hf_hash = compute_file_sha256(OVERFIT_GM_RAW_PATH)
    if hf_hash != OVERFIT_GM_EXPECTED_SHA256:
        raise ValueError(
            f"Overfit-GM hash mismatch: expected {OVERFIT_GM_EXPECTED_SHA256}, got {hf_hash}"
        )
    logger.info("Overfit-GM SHA-256 verified: %s", hf_hash)

    kg_hash = compute_file_sha256(KAGGLE_TWEETS_RAW_PATH)
    if kg_hash != KAGGLE_TWEETS_EXPECTED_SHA256:
        raise ValueError(
            f"Kaggle tweets hash mismatch: expected {KAGGLE_TWEETS_EXPECTED_SHA256}, got {kg_hash}"
        )
    logger.info("Kaggle Tweets SHA-256 verified: %s", kg_hash)

    # Step 2: Read raw datasets
    logger.info("Reading raw Overfit-GM CSV...")
    df_hf = pd.read_csv(OVERFIT_GM_RAW_PATH)
    logger.info("Loaded %d Overfit-GM records", len(df_hf))

    logger.info("Reading raw Kaggle XLSX via pure Python parser...")
    df_kg = read_xlsx_pure_python(KAGGLE_TWEETS_RAW_PATH)
    logger.info("Loaded %d Kaggle records", len(df_kg))

    # Step 3: Canonicalize text and compute record IDs
    logger.info("Applying Policy 4/5 canonicalization to Overfit-GM...")
    df_hf["text_canonical"] = df_hf["text"].apply(canonicalize_text)
    df_hf["record_id"] = df_hf["text_canonical"].apply(compute_record_id)

    logger.info("Applying Policy 4/5 canonicalization to Kaggle Tweets...")
    df_kg["text_canonical"] = df_kg["Paylaşım"].apply(canonicalize_text)
    df_kg["record_id"] = df_kg["text_canonical"].apply(compute_record_id)

    # Step 4: Map binary labels
    logger.info("Mapping and validating binary labels...")
    df_hf, df_kg = map_and_validate_binary_labels(df_hf, df_kg)

    # Step 5: Filter pure scam records from Kaggle
    logger.info("Filtering pure scam records from Kaggle...")
    df_kg_nonscam, df_kg_scams = filter_scam_records(df_kg, expected_count=501)
    logger.info(
        "Isolated %d pure scam records (remaining non-scam Kaggle: %d)",
        len(df_kg_scams),
        len(df_kg_nonscam),
    )

    # Step 6: Identify and export quarantined records
    logger.info("Identifying unresolved binary label conflicts for quarantine...")
    quarantine_canonical_keys = {
        canonicalize_text(t) for t in QUARANTINED_RAW_TEXTS
    }
    quarantine_df = identify_quarantine_rows(
        df_hf=df_hf,
        df_kg=df_kg_nonscam,
        quarantine_canonical_keys=quarantine_canonical_keys,
    )
    logger.info(
        "Identified %d physical rows across %d conflict keys",
        len(quarantine_df),
        len(quarantine_canonical_keys),
    )
    save_quarantine_records(quarantine_df, QUARANTINE_CONFLICTS_PATH)
    logger.info("Saved quarantine parquet: %s", QUARANTINE_CONFLICTS_PATH)

    # Step 7: Deduplicate and merge corpus
    logger.info("Executing deduplication and cross-dataset harmonization...")
    combined_df, deduplication_stats = deduplicate_and_merge(
        df_hf=df_hf,
        df_kg_nonscam=df_kg_nonscam,
        quarantine_canonical_keys=quarantine_canonical_keys,
    )
    logger.info(
        "Retained supervised corpus: %d unique records (clean: %d, toxic: %d)",
        deduplication_stats["total_retained"],
        deduplication_stats["clean_count"],
        deduplication_stats["toxic_count"],
    )

    # Step 8: Partition into Train, Val, Test
    logger.info(
        "Partitioning corpus into Train (75%%), Val (15%%), Test (10%%)..."
    )
    train_df, val_df, test_df = stratified_partition(
        combined_df, seed=RANDOM_SEED
    )
    logger.info("Train partition size: %d", len(train_df))
    logger.info("Val partition size: %d", len(val_df))
    logger.info("Test partition size: %d", len(test_df))

    # Unified all.parquet
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Step 9: Save processed parquet partitions
    logger.info("Writing processed Parquet partitions...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(PROCESSED_ALL_PATH, index=False)
    train_df.to_parquet(PROCESSED_TRAIN_PATH, index=False)
    val_df.to_parquet(PROCESSED_VAL_PATH, index=False)
    test_df.to_parquet(PROCESSED_TEST_PATH, index=False)
    logger.info("Saved all partitions to %s", PROCESSED_DIR)

    # Step 10: Compute and write checksums
    logger.info("Computing test set physical checksum and semantic membership hash...")
    test_physical_checksum = compute_file_sha256(PROCESSED_TEST_PATH)
    test_membership_checksum = compute_membership_hash(test_df)

    TEST_CHECKSUM_PATH.write_text(test_physical_checksum.strip() + "\n", encoding="utf-8")
    TEST_MEMBERSHIP_PATH.write_text(test_membership_checksum.strip() + "\n", encoding="utf-8")
    logger.info("Saved %s: %s", TEST_CHECKSUM_PATH.name, test_physical_checksum)
    logger.info("Saved %s: %s", TEST_MEMBERSHIP_PATH.name, test_membership_checksum)

    # Step 11: Generate manifest.json
    logger.info("Generating dataset manifest...")
    end_time = datetime.now(UTC)

    # Compute slice counts for reporting
    slices_data = {}
    for sg in ["native_twitter", "jigsaw", "other_sources"]:
        tr_cnt = int((train_df["source_group"] == sg).sum())
        va_cnt = int((val_df["source_group"] == sg).sum())
        te_cnt = int((test_df["source_group"] == sg).sum())
        slices_data[sg] = {
            "train": tr_cnt,
            "val": va_cnt,
            "test": te_cnt,
            "total": tr_cnt + va_cnt + te_cnt,
        }

    manifest = {
        "pipeline_version": "1.0.0",
        "generated_at": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "random_seed": RANDOM_SEED,
        "raw_sources": {
            "overfit_gm": {
                "path": str(OVERFIT_GM_RAW_PATH.relative_to(Path.cwd())),
                "sha256": hf_hash,
                "record_count": len(df_hf),
            },
            "kaggle_tweets": {
                "path": str(KAGGLE_TWEETS_RAW_PATH.relative_to(Path.cwd())),
                "sha256": kg_hash,
                "record_count": len(df_kg),
            },
        },
        "filtering_and_deduplication": {
            "raw_total_records": len(df_hf) + len(df_kg),
            "scam_records_removed": len(df_kg_scams),
            "overfit_gm_internal_duplicates_removed": deduplication_stats["hf_internal_duplicates_dropped"],
            "kaggle_nonscam_internal_duplicates_removed": deduplication_stats["kg_internal_duplicates_dropped"],
            "cross_dataset_overlap_dropped_from_kaggle": deduplication_stats["cross_dataset_overlap_keys"],
            "quarantined_unique_keys": deduplication_stats["quarantined_keys_count"],
            "quarantined_physical_rows": len(quarantine_df),
            "retained_total_records": deduplication_stats["total_retained"],
        },
        "class_distribution": {
            "clean_count": deduplication_stats["clean_count"],
            "toxic_count": deduplication_stats["toxic_count"],
            "clean_percentage": round(deduplication_stats["clean_count"] / deduplication_stats["total_retained"] * 100, 2),
            "toxic_percentage": round(deduplication_stats["toxic_count"] / deduplication_stats["total_retained"] * 100, 2),
        },
        "partitions": {
            "train": {
                "count": len(train_df),
                "clean_count": int((train_df["label_binary"] == 0).sum()),
                "toxic_count": int((train_df["label_binary"] == 1).sum()),
                "file": str(PROCESSED_TRAIN_PATH.relative_to(Path.cwd())),
            },
            "val": {
                "count": len(val_df),
                "clean_count": int((val_df["label_binary"] == 0).sum()),
                "toxic_count": int((val_df["label_binary"] == 1).sum()),
                "file": str(PROCESSED_VAL_PATH.relative_to(Path.cwd())),
            },
            "test": {
                "count": len(test_df),
                "clean_count": int((test_df["label_binary"] == 0).sum()),
                "toxic_count": int((test_df["label_binary"] == 1).sum()),
                "file": str(PROCESSED_TEST_PATH.relative_to(Path.cwd())),
            },
        },
        "slices": slices_data,
        "checksums": {
            "test_parquet_sha256": test_physical_checksum,
            "test_membership_sha256": test_membership_checksum,
        },
        "schema": {
            "columns": list(all_df.columns),
            "dtypes": {c: str(all_df[c].dtype) for c in all_df.columns},
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved manifest: %s", MANIFEST_PATH)

    logger.info("=== Data Preparation Pipeline Successfully Completed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
