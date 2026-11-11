"""Data filtering, scam isolation, and binary target mapping."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd


def read_xlsx_pure_python(file_path: str | Path) -> pd.DataFrame:
    """Parse Excel (.xlsx) file into a pandas DataFrame without external engine dependencies.

    Extracts XML files from the zipped OpenXML container and maps shared strings.

    Args:
        file_path: Path to the .xlsx spreadsheet file.

    Returns:
        DataFrame containing parsed rows and columns with original headers.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If sheet data is missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {path}")

    with zipfile.ZipFile(path) as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.findall("main:si", ns):
                texts = [node.text or "" for node in si.iter() if node.tag.endswith("t")]
                shared_strings.append("".join(texts))

        sheet_tree = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_data = sheet_tree.find("main:sheetData", ns)
        if sheet_data is None:
            raise ValueError("No main:sheetData element found in sheet1.xml")

        rows = []
        for row in sheet_data.findall("main:row", ns):
            row_data = {}
            for c in row.findall("main:c", ns):
                cell_ref = c.attrib.get("r", "")
                col_letter = "".join([ch for ch in cell_ref if ch.isalpha()])
                cell_type = c.attrib.get("t", "")
                v = c.find("main:v", ns)
                val = v.text if v is not None else None
                if cell_type == "s" and val is not None:
                    val = shared_strings[int(val)]
                row_data[col_letter] = val
            rows.append(row_data)

        df_raw = pd.DataFrame(rows)
        header = df_raw.iloc[0].values
        df = df_raw.iloc[1:].copy()
        df.columns = header
        return df


def filter_scam_records(
    df_kg: pd.DataFrame, expected_count: int = 501
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isolate pure commercial/phishing scam records from the Kaggle dataset.

    Pure scam records are defined as having Dolandırıcılık == 1.0 with all other
    toxicity category flags (Küfür, Tehdit, Hakaret, Cinsel İçerik, Irkçılık) equal to 0.

    Args:
        df_kg: Kaggle DataFrame with category flags and Tespit column.
        expected_count: Expected number of pure scam records (default: 501).

    Returns:
        Tuple of (retained_df, scam_df).

    Raises:
        ValueError: If expected scam record count does not match.
    """
    toxic_category_cols = ["Küfür", "Tehdit", "Hakaret", "Cinsel İçerik", "Irkçılık"]
    all_needed_cols = ["Dolandırıcılık", "Tespit"] + toxic_category_cols

    for col in all_needed_cols:
        if col not in df_kg.columns:
            raise KeyError(f"Required Kaggle column missing: {col}")

    # Convert numeric columns for filtering
    dolandiricilik = df_kg["Dolandırıcılık"].astype(float)
    other_toxic = df_kg[toxic_category_cols].astype(float).sum(axis=1)

    scam_mask = (dolandiricilik == 1.0) & (other_toxic == 0.0)
    actual_scams = int(scam_mask.sum())

    if actual_scams != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} pure scam records, found {actual_scams}"
        )

    scam_df = df_kg[scam_mask].copy()
    retained_df = df_kg[~scam_mask].copy()

    return retained_df, scam_df


def map_and_validate_binary_labels(
    df_hf: pd.DataFrame, df_kg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map source labels to standardized binary toxicity target (0 = clean, 1 = toxic).

    Overfit-GM:
        is_toxic: 0 -> 0 (clean), 1 -> 1 (toxic).

    Kaggle Tweets:
        Tespit: 1.0 -> 0 (clean), 0.0 -> 1 (toxic).

    Args:
        df_hf: Overfit-GM DataFrame with is_toxic column.
        df_kg: Kaggle DataFrame with Tespit column.

    Returns:
        Tuple of updated (df_hf, df_kg) with label_binary column added.

    Raises:
        ValueError: If unexpected label values are encountered.
    """
    # Overfit-GM validation and mapping
    if "is_toxic" not in df_hf.columns:
        raise KeyError("Column 'is_toxic' not found in Overfit-GM DataFrame")

    hf_labels = set(df_hf["is_toxic"].unique())
    if not hf_labels.issubset({0, 1}):
        raise ValueError(f"Unexpected values in Overfit-GM is_toxic: {hf_labels}")

    df_hf_out = df_hf.copy()
    df_hf_out["label_binary"] = df_hf_out["is_toxic"].astype(int)

    # Kaggle validation and mapping
    if "Tespit" not in df_kg.columns:
        raise KeyError("Column 'Tespit' not found in Kaggle DataFrame")

    kg_numeric_labels = df_kg["Tespit"].astype(float)
    kg_labels = set(kg_numeric_labels.unique())
    if not kg_labels.issubset({0.0, 1.0}):
        raise ValueError(f"Unexpected values in Kaggle Tespit: {kg_labels}")

    df_kg_out = df_kg.copy()
    # In Kaggle: Tespit = 1.0 means clean (0), Tespit = 0.0 means toxic (1)
    df_kg_out["label_binary"] = (kg_numeric_labels == 0.0).astype(int)

    return df_hf_out, df_kg_out
