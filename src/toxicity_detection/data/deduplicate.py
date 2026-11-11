"""Corpus deduplication, cross-dataset collision resolution, and metadata harmonization."""

from __future__ import annotations

from typing import Any

import pandas as pd

from toxicity_detection.config import SOURCE_GROUP_MAPPING


def deduplicate_and_merge(
    df_hf: pd.DataFrame,
    df_kg_nonscam: pd.DataFrame,
    quarantine_canonical_keys: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute corpus deduplication, cross-dataset overlap resolution, and metadata harmonization.

    Pipeline steps:
    1. Deduplicate Overfit-GM internally on text_canonical (keep first occurrence).
    2. Deduplicate Kaggle non-scam internally on text_canonical (keep first occurrence).
    3. Identify cross-dataset shared keys (Overfit-GM retained, Kaggle duplicate dropped).
    4. Remove quarantined binary conflict keys from both pools.
    5. Standardize schemas and concatenate into a single unified DataFrame.

    Args:
        df_hf: Overfit-GM DataFrame with text_canonical, record_id, and label_binary.
        df_kg_nonscam: Kaggle DataFrame (scams removed) with text_canonical, record_id,
            and label_binary.
        quarantine_canonical_keys: Set of canonical keys to quarantine.

    Returns:
        Tuple of (combined_df, accounting_metrics_dict).

    Raises:
        ValueError: If intermediate or final record counts deviate from specifications.
    """
    raw_hf_count = len(df_hf)
    raw_kg_nonscam_count = len(df_kg_nonscam)

    # 1. Deduplicate Overfit-GM internally
    hf_dedup = df_hf.drop_duplicates(subset=["record_id"], keep="first").copy()
    hf_duplicates_dropped = raw_hf_count - len(hf_dedup)
    if hf_duplicates_dropped != 111:
        raise ValueError(
            f"Expected 111 Overfit-GM internal duplicates, found {hf_duplicates_dropped}"
        )
    if len(hf_dedup) != 77689:
        raise ValueError(
            f"Expected 77,689 unique Overfit-GM keys, found {len(hf_dedup)}"
        )

    # 2. Deduplicate Kaggle non-scam internally
    kg_dedup = df_kg_nonscam.drop_duplicates(
        subset=["record_id"], keep="first"
    ).copy()
    kg_duplicates_dropped = raw_kg_nonscam_count - len(kg_dedup)
    if kg_duplicates_dropped != 71:
        raise ValueError(
            f"Expected 71 Kaggle internal duplicates, found {kg_duplicates_dropped}"
        )
    if len(kg_dedup) != 9427:
        raise ValueError(
            f"Expected 9,427 unique Kaggle non-scam keys, found {len(kg_dedup)}"
        )

    # 3. Cross-dataset overlap resolution
    hf_record_ids = set(hf_dedup["record_id"])
    kg_record_ids = set(kg_dedup["record_id"])
    cross_overlap_ids = hf_record_ids.intersection(kg_record_ids)
    if len(cross_overlap_ids) != 2248:
        raise ValueError(
            f"Expected 2,248 cross-dataset collisions, found {len(cross_overlap_ids)}"
        )

    # Kaggle-only keys (excluding overlap with HF)
    kg_only = kg_dedup[~kg_dedup["record_id"].isin(hf_record_ids)].copy()
    if len(kg_only) != 7179:
        raise ValueError(
            f"Expected 7,179 Kaggle-only unique keys, found {len(kg_only)}"
        )

    # 4. Quarantine conflict removal
    hf_retained = hf_dedup[
        ~hf_dedup["text_canonical"].isin(quarantine_canonical_keys)
    ].copy()
    kg_retained = kg_only[
        ~kg_only["text_canonical"].isin(quarantine_canonical_keys)
    ].copy()

    if len(hf_retained) != 77688:
        raise ValueError(
            f"Expected 77,688 retained Overfit-GM records, found {len(hf_retained)}"
        )
    if len(kg_retained) != 7178:
        raise ValueError(
            f"Expected 7,178 retained Kaggle records, found {len(kg_retained)}"
        )

    # 5. Harmonize schema
    # Overfit-GM records
    hf_final = pd.DataFrame(
        {
            "record_id": hf_retained["record_id"].astype(str),
            "text_raw": hf_retained["text"].astype(str),
            "text_canonical": hf_retained["text_canonical"].astype(str),
            "label_binary": hf_retained["label_binary"].astype("int64"),
            "source": hf_retained["source"].astype(str),
            "source_group": hf_retained["source"].map(SOURCE_GROUP_MAPPING),
            "target_original": hf_retained["target"].astype(str),
        }
    )

    # Kaggle records
    kg_final = pd.DataFrame(
        {
            "record_id": kg_retained["record_id"].astype(str),
            "text_raw": kg_retained["Paylaşım"].astype(str),
            "text_canonical": kg_retained["text_canonical"].astype(str),
            "label_binary": kg_retained["label_binary"].astype("int64"),
            "source": "kaggle_tweets",
            "source_group": SOURCE_GROUP_MAPPING["kaggle_tweets"],
            "target_original": None,
        }
    )

    combined_df = (
        pd.concat([hf_final, kg_final], ignore_index=True)
        .sort_values(by="record_id")
        .reset_index(drop=True)
    )

    total_retained = len(combined_df)
    if total_retained != 84866:
        raise ValueError(
            f"Expected exactly 84,866 retained instances, found {total_retained}"
        )

    clean_count = int((combined_df["label_binary"] == 0).sum())
    toxic_count = int((combined_df["label_binary"] == 1).sum())

    if clean_count != 41376 or toxic_count != 43490:
        raise ValueError(
            f"Class balance mismatch: clean={clean_count} (exp 41,376), "
            f"toxic={toxic_count} (exp 43,490)"
        )

    stats = {
        "raw_hf_count": raw_hf_count,
        "raw_kg_nonscam_count": raw_kg_nonscam_count,
        "hf_internal_duplicates_dropped": hf_duplicates_dropped,
        "kg_internal_duplicates_dropped": kg_duplicates_dropped,
        "hf_unique_keys": len(hf_dedup),
        "kg_unique_keys": len(kg_dedup),
        "cross_dataset_overlap_keys": len(cross_overlap_ids),
        "kg_only_keys": len(kg_only),
        "quarantined_keys_count": len(quarantine_canonical_keys),
        "hf_retained_count": len(hf_retained),
        "kg_retained_count": len(kg_retained),
        "total_retained": total_retained,
        "clean_count": clean_count,
        "toxic_count": toxic_count,
    }

    return combined_df, stats
