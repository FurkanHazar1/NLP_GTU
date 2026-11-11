"""Quarantine protocol for unresolved binary label conflicts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def identify_quarantine_rows(
    df_hf: pd.DataFrame,
    df_kg: pd.DataFrame,
    quarantine_canonical_keys: set[str],
) -> pd.DataFrame:
    """Extract all physical records matching quarantined canonical keys across datasets.

    Args:
        df_hf: Overfit-GM DataFrame with text_canonical, record_id, and label_binary.
        df_kg: Kaggle DataFrame with text_canonical, record_id, and label_binary.
        quarantine_canonical_keys: Set of canonical text keys to quarantine.

    Returns:
        DataFrame containing detailed metadata and raw fields for all quarantined rows.
    """
    rows = []

    # Check Overfit-GM
    hf_matches = df_hf[df_hf["text_canonical"].isin(quarantine_canonical_keys)]
    for idx, row in hf_matches.iterrows():
        raw_info = {
            "source": row.get("source"),
            "target": row.get("target"),
            "is_toxic": int(row.get("is_toxic")),
        }
        rows.append(
            {
                "dataset_origin": "overfit_gm",
                "original_row_index": int(idx),
                "record_id": str(row["record_id"]),
                "text_raw": str(row["text"]),
                "text_canonical": str(row["text_canonical"]),
                "label_binary": int(row["label_binary"]),
                "raw_label_info": json.dumps(raw_info, ensure_ascii=False),
                "quarantine_reason": "cross_dataset_disagreement",
            }
        )

    # Check Kaggle
    kg_matches = df_kg[df_kg["text_canonical"].isin(quarantine_canonical_keys)]
    for idx, row in kg_matches.iterrows():
        raw_info = {
            "Tespit": float(row.get("Tespit")),
            "Küfür": float(row.get("Küfür", 0.0)),
            "Tehdit": float(row.get("Tehdit", 0.0)),
            "Hakaret": float(row.get("Hakaret", 0.0)),
            "Dolandırıcılık": float(row.get("Dolandırıcılık", 0.0)),
            "Cinsel İçerik": float(row.get("Cinsel İçerik", 0.0)),
            "Irkçılık": float(row.get("Irkçılık", 0.0)),
        }
        # Distinguish reason: key 1 has 2 occurrences in Kaggle
        canonical_key = str(row["text_canonical"])
        count_in_kg = int((df_kg["text_canonical"] == canonical_key).sum())
        reason = (
            "kaggle_internal_tie"
            if count_in_kg > 1
            else "cross_dataset_disagreement"
        )

        rows.append(
            {
                "dataset_origin": "kaggle_tweets",
                "original_row_index": int(idx),
                "record_id": str(row["record_id"]),
                "text_raw": str(row["Paylaşım"]),
                "text_canonical": canonical_key,
                "label_binary": int(row["label_binary"]),
                "raw_label_info": json.dumps(raw_info, ensure_ascii=False),
                "quarantine_reason": reason,
            }
        )

    df_quarantine = pd.DataFrame(rows)
    return df_quarantine.sort_values(
        by=["quarantine_reason", "dataset_origin", "original_row_index"]
    ).reset_index(drop=True)


def save_quarantine_records(df_quarantine: pd.DataFrame, path: str | Path) -> None:
    """Save quarantined conflict rows to parquet format.

    Args:
        df_quarantine: DataFrame of quarantined rows.
        path: Destination path for the parquet file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_quarantine.to_parquet(out_path, index=False)
