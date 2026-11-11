"""Data processing, canonicalization, deduplication, and partitioning module."""

from __future__ import annotations

from toxicity_detection.data.canonicalize import canonicalize_text, compute_record_id
from toxicity_detection.data.deduplicate import deduplicate_and_merge
from toxicity_detection.data.filter import (
    filter_scam_records,
    map_and_validate_binary_labels,
    read_xlsx_pure_python,
)
from toxicity_detection.data.partition import (
    compute_file_sha256,
    compute_membership_hash,
    stratified_partition,
)
from toxicity_detection.data.quarantine import (
    identify_quarantine_rows,
    save_quarantine_records,
)

__all__ = [
    "canonicalize_text",
    "compute_file_sha256",
    "compute_membership_hash",
    "compute_record_id",
    "deduplicate_and_merge",
    "filter_scam_records",
    "identify_quarantine_rows",
    "map_and_validate_binary_labels",
    "read_xlsx_pure_python",
    "save_quarantine_records",
    "stratified_partition",
]
