"""Deterministic canonicalization and cryptographic record ID generation."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def canonicalize_text(text: str) -> str:
    """Apply Policy 4/5 canonicalization to a given text string.

    Steps:
    1. Validate input string.
    2. Normalize Unicode to NFC form.
    3. Collapse multiple whitespace characters (spaces, tabs, newlines) into a single space.
    4. Strip leading and trailing whitespace.
    5. Apply Turkish-aware lowercasing (İ -> i, I -> ı, then .lower()).

    Args:
        text: Input string to canonicalize.

    Returns:
        Canonicalized text string.

    Raises:
        TypeError: If input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}: {text!r}")

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Collapse internal whitespace and strip outer whitespace
    collapsed = re.sub(r"\s+", " ", normalized).strip()

    # 3. Turkish-aware lowercasing
    return collapsed.replace("İ", "i").replace("I", "ı").lower()


def compute_record_id(canonical_text: str) -> str:
    """Compute the deterministic SHA-256 identifier for a canonicalized string.

    Args:
        canonical_text: The canonical UTF-8 string.

    Returns:
        64-character hexadecimal SHA-256 digest.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(canonical_text, str):
        raise TypeError(f"Expected str, got {type(canonical_text).__name__}")
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
