"""Deterministic stratified partitioning, slice verification, and cryptographic membership hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from toxicity_detection.config import RANDOM_SEED, STRATUM_SPLIT_COUNTS


def stratified_partition(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition the retained corpus into Train (75%), Val (15%), and Test (10%).

    Stratification is performed on composite stratum: f"{source_group}_{label_binary}".
    To ensure absolute determinism across platforms and environments, records within
    each stratum are sorted by record_id prior to seeded pseudo-random sampling.

    Args:
        df: Harmonized corpus DataFrame containing record_id, source_group, and label_binary.
        seed: Random seed for deterministic shuffling (default: 42).

    Returns:
        Tuple of (train_df, val_df, test_df) with 'split' column attached.

    Raises:
        ValueError: If partition sizes or stratum distributions deviate from specification.
    """
    df_work = df.copy()
    df_work["stratum"] = (
        df_work["source_group"].astype(str)
        + "_"
        + df_work["label_binary"].astype(str)
    )

    train_parts = []
    val_parts = []
    test_parts = []

    for stratum_name, counts in STRATUM_SPLIT_COUNTS.items():
        stratum_sub = df_work[df_work["stratum"] == stratum_name]
        expected_total = counts["train"] + counts["val"] + counts["test"]

        if len(stratum_sub) != expected_total:
            raise ValueError(
                f"Stratum {stratum_name} count mismatch: got {len(stratum_sub)}, "
                f"expected {expected_total}"
            )

        # Deterministic sorting before shuffle guarantees invariance to input row order
        shuffled = stratum_sub.sort_values(by="record_id").sample(
            frac=1.0, random_state=seed
        )

        n_train = counts["train"]
        n_val = counts["val"]

        train_slice = shuffled.iloc[:n_train].copy()
        val_slice = shuffled.iloc[n_train : n_train + n_val].copy()
        test_slice = shuffled.iloc[n_train + n_val :].copy()

        train_slice["split"] = "train"
        val_slice["split"] = "val"
        test_slice["split"] = "test"

        train_parts.append(train_slice)
        val_parts.append(val_slice)
        test_parts.append(test_slice)

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        .drop(columns=["stratum"])
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        .drop(columns=["stratum"])
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        .drop(columns=["stratum"])
        .reset_index(drop=True)
    )

    # Invariant validations
    if len(train_df) != 63650:
        raise ValueError(
            f"Expected 63,650 train records, got {len(train_df)}"
        )
    if len(val_df) != 12731:
        raise ValueError(
            f"Expected 12,731 validation records, got {len(val_df)}"
        )
    if len(test_df) != 8485:
        raise ValueError(f"Expected 8,485 test records, got {len(test_df)}")

    # Zero overlap check
    train_ids = set(train_df["record_id"])
    val_ids = set(val_df["record_id"])
    test_ids = set(test_df["record_id"])

    if train_ids.intersection(val_ids):
        raise ValueError("Data leakage detected: train and val share record_ids!")
    if train_ids.intersection(test_ids):
        raise ValueError(
            "Data leakage detected: train and test share record_ids!"
        )
    if val_ids.intersection(test_ids):
        raise ValueError("Data leakage detected: val and test share record_ids!")

    return train_df, val_df, test_df


def compute_membership_hash(test_df: pd.DataFrame) -> str:
    """Compute a deterministic semantic hash of test set membership.

    Constructed by sorting test record IDs ascending and hashing lines of
    f"{record_id}\t{label_binary}\n". This is invariant to physical storage
    format or parquet compression changes.

    Args:
        test_df: Test set DataFrame containing record_id and label_binary.

    Returns:
        64-character hexadecimal SHA-256 string.
    """
    sorted_records = test_df[["record_id", "label_binary"]].sort_values(
        by="record_id"
    )
    lines = [
        f"{r['record_id']}\t{int(r['label_binary'])}\n"
        for _, r in sorted_records.iterrows()
    ]
    payload = "".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 checksum of an arbitrary binary or text file.

    Args:
        file_path: Path to target file.

    Returns:
        64-character hexadecimal SHA-256 string.

    Raises:
        FileNotFoundError: If target file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
