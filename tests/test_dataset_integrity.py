"""Tests for raw dataset acquisition, cryptographic integrity, corpus accounting, and label invariants."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


def get_file_sha256(path: str) -> str:
    """Calculate SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def read_xlsx_pure_python(file_path: str) -> pd.DataFrame:
    """Parse Excel file into pandas DataFrame without openpyxl dependency."""
    with zipfile.ZipFile(file_path) as z:
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
        assert sheet_data is not None

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


def canonical_key_p4(t: str) -> str:
    """Apply Policy 4 canonicalization: NFC + collapsed whitespace + Turkish lowercasing."""
    t_nfc = unicodedata.normalize("NFC", str(t))
    t_ws = re.sub(r"\s+", " ", t_nfc).strip()
    return t_ws.replace("İ", "i").replace("I", "ı").lower()


def test_overfit_gm_integrity():
    """Verify Overfit-GM raw CSV file existence, checksum, record count, and schema."""
    csv_path = "data/raw/overfit_gm/turkish_toxic_language.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    expected_sha256 = "436aa9250d21d0617cb0aafdf77cad9c1aa85642d2b3fd1cd9d1aedab2d3fbc6"
    assert get_file_sha256(csv_path) == expected_sha256

    df = pd.read_csv(csv_path)
    assert len(df) == 77800
    assert list(df.columns) == ["text", "target", "source", "is_toxic"]
    assert set(df["target"].unique()) == {"OTHER", "PROFANITY", "INSULT", "RACIST", "SEXIST"}

    # Verify Turkish Unicode characters
    turkish_chars = set("çğışöüÇĞİŞÖÜ")
    all_text = " ".join(df["text"].iloc[:100].tolist())
    assert bool(set(all_text) & turkish_chars)


def test_kaggle_tweets_integrity():
    """Verify Kaggle raw XLSX file existence, checksum, record count, and schema."""
    xlsx_path = "data/raw/kaggle_tweets/Veri_Seti_10000.xlsx"
    assert os.path.exists(xlsx_path), f"File {xlsx_path} does not exist"

    expected_sha256 = "b713c8efc748134c417d250bb7c27bd4ab652abb1955af2495108ccfdb41c22a"
    assert get_file_sha256(xlsx_path) == expected_sha256

    df = read_xlsx_pure_python(xlsx_path)
    assert len(df) == 9999
    expected_cols = [
        "Paylaşım",
        "Tespit",
        "Küfür",
        "Tehdit",
        "Hakaret",
        "Dolandırıcılık",
        "Cinsel İçerik",
        "Irkçılık",
    ]
    assert list(df.columns) == expected_cols

    # Verify Turkish Unicode characters
    turkish_chars = set("çğışöüÇĞİŞÖÜ")
    all_text = " ".join(df["Paylaşım"].iloc[:100].tolist())
    assert bool(set(all_text) & turkish_chars)


def test_corpus_accounting_policies():
    """Verify independent corpus accounting under both Policy 2 (stripped) and Policy 4 (canonical)."""
    csv_path = "data/raw/overfit_gm/turkish_toxic_language.csv"
    xlsx_path = "data/raw/kaggle_tweets/Veri_Seti_10000.xlsx"

    df_hf = pd.read_csv(csv_path)
    df_kg = read_xlsx_pure_python(xlsx_path)

    # 1. Total raw records
    assert len(df_hf) == 77800
    assert len(df_kg) == 9999
    assert len(df_hf) + len(df_kg) == 87799

    # 2. Policy 2: Whitespace-stripped counts
    hf_p2 = set(df_hf["text"].str.strip())
    kg_p2 = set(df_kg["Paylaşım"].str.strip())
    assert len(hf_p2.intersection(kg_p2)) == 2088
    assert len(hf_p2.union(kg_p2)) == 85648

    # 3. Policy 4: Canonical (collapsed whitespace + Turkish lowercase + NFC)
    hf_p4 = df_hf["text"].apply(canonical_key_p4)
    kg_p4 = df_kg["Paylaşım"].apply(canonical_key_p4)
    assert hf_p4.nunique() == 77689
    assert kg_p4.nunique() == 9928

    s_hf_p4 = set(hf_p4)
    s_kg_p4 = set(kg_p4)
    assert len(s_hf_p4.intersection(s_kg_p4)) == 2248
    assert len(s_hf_p4.union(s_kg_p4)) == 85369


def test_scam_records_isolation():
    """Verify that pure scam records are exactly 501 and have zero overlap with Overfit-GM."""
    xlsx_path = "data/raw/kaggle_tweets/Veri_Seti_10000.xlsx"
    csv_path = "data/raw/overfit_gm/turkish_toxic_language.csv"

    df_kg = read_xlsx_pure_python(xlsx_path)
    df_hf = pd.read_csv(csv_path)

    for c in ["Küfür", "Tehdit", "Hakaret", "Dolandırıcılık", "Cinsel İçerik", "Irkçılık", "Tespit"]:
        df_kg[c] = df_kg[c].astype(float)

    scam_mask = df_kg["Dolandırıcılık"] == 1.0
    assert scam_mask.sum() == 501

    # Verify no other toxic flags are positive in scam records
    toxic_cols = ["Küfür", "Tehdit", "Hakaret", "Cinsel İçerik", "Irkçılık"]
    assert (df_kg.loc[scam_mask, toxic_cols].sum(axis=1) == 0).all()

    # Verify Tespit is 0.0 for all scam records
    assert (df_kg.loc[scam_mask, "Tespit"] == 0.0).all()

    # Verify zero overlap with Overfit-GM under canonical key
    kg_scam_keys = set(df_kg.loc[scam_mask, "Paylaşım"].apply(canonical_key_p4))
    hf_keys = set(df_hf["text"].apply(canonical_key_p4))
    assert len(kg_scam_keys.intersection(hf_keys)) == 0


def test_retained_corpus_reconciliation():
    """Verify final retained corpus size (84,866) and class balance under Policy 4."""
    xlsx_path = "data/raw/kaggle_tweets/Veri_Seti_10000.xlsx"
    csv_path = "data/raw/overfit_gm/turkish_toxic_language.csv"

    df_kg = read_xlsx_pure_python(xlsx_path)
    df_hf = pd.read_csv(csv_path)

    df_hf["canon_key"] = df_hf["text"].apply(canonical_key_p4)
    df_kg["canon_key"] = df_kg["Paylaşım"].apply(canonical_key_p4)

    for c in ["Dolandırıcılık", "Tespit"]:
        df_kg[c] = df_kg[c].astype(float)

    # Exclude scams
    df_kg_clean = df_kg[df_kg["Dolandırıcılık"] != 1.0].copy()

    # Deduplicate Overfit-GM (keep first)
    df_hf_dedup = df_hf.drop_duplicates(subset=["canon_key"], keep="first").copy()
    assert len(df_hf_dedup) == 77689

    # Kaggle non-overlapping
    s_hf = set(df_hf_dedup["canon_key"])
    df_kg_only = df_kg_clean[~df_kg_clean["canon_key"].isin(s_hf)].drop_duplicates(
        subset=["canon_key"], keep="first"
    )
    assert len(df_kg_only) == 7179

    # Conflict keys
    kg_conflict_key = canonical_key_p4(
        "bize karşı savunma oynayıp 1 puan aldığı için sevinen başakşehir gs ye 5 atıp şov yapıyor  siz neyin kafasını yaş"
    )
    cross_conflict_key = canonical_key_p4(
        "milletin çocuğu nasıl bu kadar uslu sessiz akıllı olabiliyor benimki tam bir fırlama eslemin peşinde koşmaktan bir"
    )
    quarantine_keys = {kg_conflict_key, cross_conflict_key}

    df_hf_final = df_hf_dedup[~df_hf_dedup["canon_key"].isin(quarantine_keys)]
    df_kg_final = df_kg_only[~df_kg_only["canon_key"].isin(quarantine_keys)]

    total_retained = len(df_hf_final) + len(df_kg_final)
    assert total_retained == 84866

    # Verify class balance
    toxic_hf = (df_hf_final["is_toxic"] == 1).sum()
    toxic_kg = (df_kg_final["Tespit"] == 0.0).sum()
    assert toxic_hf + toxic_kg == 43490
    assert total_retained - (toxic_hf + toxic_kg) == 41376


def test_kaggle_label_consistency_invariants():
    """Verify Kaggle Tespit column semantics and category flag relationships."""
    xlsx_path = "data/raw/kaggle_tweets/Veri_Seti_10000.xlsx"
    df_kg = read_xlsx_pure_python(xlsx_path)

    flag_cols = ["Küfür", "Tehdit", "Hakaret", "Dolandırıcılık", "Cinsel İçerik", "Irkçılık"]
    for c in flag_cols + ["Tespit"]:
        df_kg[c] = df_kg[c].astype(float)

    total_flags = df_kg[flag_cols].sum(axis=1)

    # Invariant 1: Tespit = 1.0 (Clean) NEVER has any category flags positive
    clean_mask = df_kg["Tespit"] == 1.0
    assert clean_mask.sum() == 4731
    assert (total_flags[clean_mask] == 0).all()

    # Invariant 2: Whenever at least one category flag is positive, Tespit is ALWAYS 0.0
    toxic_flag_mask = total_flags > 0
    assert toxic_flag_mask.sum() == 4930
    assert (df_kg["Tespit"][toxic_flag_mask] == 0.0).all()

    # Invariant 3: The unflagged toxic discrepancy subset has exactly 338 records
    discrepancy_mask = (df_kg["Tespit"] == 0.0) & (total_flags == 0)
    assert discrepancy_mask.sum() == 338


def test_cross_dataset_label_agreement_invariant():
    """Verify label agreement on overlapping records between Overfit-GM and Kaggle."""
    df_hf = pd.read_csv("data/raw/overfit_gm/turkish_toxic_language.csv")
    df_kg = read_xlsx_pure_python("data/raw/kaggle_tweets/Veri_Seti_10000.xlsx")

    df_hf["text_clean"] = df_hf["text"].str.strip()
    df_kg["text_clean"] = df_kg["Paylaşım"].str.strip()
    df_kg["Tespit"] = df_kg["Tespit"].astype(float)

    merged = pd.merge(df_hf, df_kg, on="text_clean")
    assert len(merged) == 2107
    assert merged["text_clean"].nunique() == 2088

    # Agreement between Overfit-GM is_toxic and Kaggle (Tespit == 0.0)
    kg_toxic = (merged["Tespit"] == 0.0).astype(int)
    matches = merged["is_toxic"] == kg_toxic
    assert matches.sum() == 2106
    assert (~matches).sum() == 1  # Exactly 1 disagreement out of 2,107


def test_metadata_inventory_json():
    """Verify machine-readable dataset_inventory.json schema and validity."""
    json_path = "data/metadata/dataset_inventory.json"
    assert os.path.exists(json_path)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "datasets" in data
    assert "overfit_gm" in data["datasets"]
    assert "kaggle_tweets" in data["datasets"]
    assert data["datasets"]["overfit_gm"]["record_count"] == 77800
    assert data["datasets"]["kaggle_tweets"]["record_count"] == 9999
