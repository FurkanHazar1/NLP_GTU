# Phase 3.2 — Deterministic Preprocessing & Partitioning Pipeline Report

**Project:** Explainable Turkish Implicit Toxicity Detection  
**Team:** Team 07 – TOXICITY  
**Date:** November 11, 2026  
**Pipeline Version:** 1.0.0  
**Artifact Path:** `docs/data/preprocessing_report.md`  
**Execution Script:** `scripts/run_prepare_data.py`  
**Test Suite:** `tests/test_data_pipeline.py` & `tests/test_dataset_integrity.py`  

---

## 1. Executive Summary

This report documents the execution and cryptographic verification of the **Phase 3.2 Deterministic Data Preparation Pipeline** for the research project *Explainable Turkish Implicit Toxicity Detection*.

The pipeline ingests two raw data assets, applies rigorous Unicode and Turkish linguistic canonicalization (Policy 4/5), isolates commercial phishing scams, deduplicates records both internally and across datasets, quarantines unresolved binary label conflicts, and generates a frozen, leak-free, source-stratified partition (75% Train, 15% Validation, 10% Test).

### Key Execution Highlights:
1. **Raw Ingestion Pool:** **87,799 records** ($77,800$ Overfit-GM + $9,999$ Kaggle).
2. **Pure Scam Isolation:** Exactly **501 records** removed from Kaggle (`Dolandırıcılık == 1.0` with no other toxic flags, zero overlap with Overfit-GM).
3. **Internal Deduplication:**
   - Overfit-GM: **111 redundant casing rows** removed ($77,800 \to 77,689$).
   - Kaggle non-scam: **71 redundant rows** removed ($9,498 \to 9,427$).
4. **Cross-Dataset Collision Resolution:** **2,248 overlapping keys** identified. Overfit-GM records were retained with complete metadata, and Kaggle duplicates were removed.
5. **Conflict Quarantine Protocol:** Exactly **2 unique keys** (encompassing **4 physical records**) exhibiting binary label conflicts were quarantined to `data/interim/quarantine_conflicts.parquet`.
6. **Retained Supervised Corpus:** Exactly **84,866 unique instances** ($41,376$ clean, $43,490$ toxic).
7. **Stratified Partitions:**
   - **Train:** **63,650 instances** ($75.0\%$)
   - **Validation:** **12,731 instances** ($15.0\%$)
   - **Test:** **8,485 instances** ($10.0\%$)
   - **Data Leakage:** **Zero overlap** on `record_id` and `text_canonical` across all partitions.
8. **Cryptographic Sealing:**
   - Test physical file SHA-256: `1c5408bad5928752b39ee6471604954c4f93017ad5fe57b870c1cd49e0b479bd`
   - Test semantic membership SHA-256: `7bd973c2e3bc33d5cc5f3010e631ea57fd62969e6a253c31d23a1d43de880174`

---

## 2. Pipeline Architecture & Module Layout

The data preparation pipeline is built into modular, testable components under `src/toxicity_detection/data/`:

```text
src/toxicity_detection/
├── config.py                 # Paths, constants, master seed (42), split targets, stratum mapping
├── data/
│   ├── __init__.py           # Unified module exports
│   ├── canonicalize.py       # Policy 4/5 normalization & SHA-256 record ID computation
│   ├── filter.py             # Pure-Python XLSX reader, scam isolation, binary target validation
│   ├── deduplicate.py        # Internal & cross-dataset deduplication, schema harmonization
│   ├── quarantine.py         # Isolation and export of unresolved label conflicts
│   └── partition.py          # Seeded stratified partitioning & membership hash generation
└── scripts/
    └── run_prepare_data.py   # CLI entry point orchestrating end-to-end pipeline
```

---

## 3. Step-by-Step Corpus Accounting

| Pipeline Stage | Operation / Condition | Overfit-GM | Kaggle | Unified Pool | Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **0. Raw Ingestion** | Raw files read from disk | 77,800 | 9,999 | 87,799 | Baseline |
| **1. Scam Filtering** | Pure scam (`Dolandırıcılık=1`, others=0) | 77,800 | 9,498 | 87,298 | $-501$ (Kaggle) |
| **2. Internal Dedup** | Policy 4/5 canonical keys (`keep="first"`) | 77,689 | 9,427 | 87,116 | $-111$ HF, $-71$ KG |
| **3. Cross Overlap** | Remove Kaggle records matching Overfit-GM | 77,689 | 7,179 | 84,868 | $-2,248$ (Kaggle) |
| **4. Quarantine** | Remove 2 unresolved binary conflict keys | 77,688 | 7,178 | **84,866** | $-1$ HF, $-1$ KG |
| **Final Retained** | **Supervised Training & Evaluation Pool** | **77,688** | **7,178** | **84,866** | **Exact Target** |

---

## 4. Normalization & Canonicalization Protocol (Policy 4/5)

To eliminate whitespace and capitalization discrepancies that would otherwise leak identical texts across partitions, all texts undergo Policy 4/5 canonicalization:

1. **Unicode NFC Normalization:** `unicodedata.normalize("NFC", text)` converts all precomposed and decomposed forms into canonical NFC.
2. **Whitespace Collapsing:** `re.sub(r"\s+", " ", text).strip()` collapses multiple spaces, non-breaking spaces (`\xa0`), tabs, and newlines into a single ASCII space, stripping leading and trailing whitespace.
3. **Turkish Linguistic Lowercasing:**
   - Dotted uppercase `İ` (`\u0130`) $\to$ lowercase `i` (`\u0069`)
   - Dotless uppercase `I` (`\u0049`) $\to$ lowercase `ı` (`\u0131`)
   - Standard Python `.lower()` applied to handle remaining characters (`ç, ğ, ö, ş, ü`).
4. **Cryptographic Record ID:** `record_id` is the 64-character hexadecimal SHA-256 digest of `text_canonical.encode("utf-8")`.
5. **Verbatim Preservation:** The original text string is preserved verbatim as `text_raw` in all processed Parquet outputs.

---

## 5. Quarantined Binary Conflicts

Two canonical keys exhibited irresolvable binary target conflicts across annotators or datasets. Rather than arbitrarily forcing a label, these were quarantined to `data/interim/quarantine_conflicts.parquet`:

| # | Conflict Key (Canonical) | Origin Dataset(s) | Row Indices | Labels Observed | Quarantine Reason |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | `"bize karşı savunma oynayıp 1 puan aldığı için sevinen başakşehir gs ye 5 atıp şov yapıyor siz neyin kafasını yaş"` | Kaggle Tweets | 4134, 7089 | Row 4134: `Tespit=0.0` (Toxic)<br>Row 7089: `Tespit=1.0` (Clean) | `kaggle_internal_tie` |
| **2** | `"milletin çocuğu nasıl bu kadar uslu sessiz akıllı olabiliyor benimki tam bir fırlama eslemin peşinde koşmaktan bir"` | Overfit-GM & Kaggle | HF: 77184<br>KG: 4086 | Overfit-GM: `is_toxic=1` (`bully`, `INSULT`)<br>Kaggle: `Tespit=1.0` (Clean) | `cross_dataset_disagreement` |

All **4 physical rows** associated with these 2 keys are archived with their complete raw metadata and category flags in `data/interim/quarantine_conflicts.parquet`.

---

## 6. Partition Allocation & Stratification Balance

Partitions are strictly split using fixed random seed `42` stratified on the composite stratum `f"{source_group}_{label_binary}"`.

### 6.1 Exact Stratum Allocations

| Composite Stratum | Retained Population | Train ($75\%$) | Validation ($15\%$) | Test ($10\%$) | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `native_twitter_0` | 21,584 | 16,188 | 3,238 | 2,158 | Verified Exact |
| `native_twitter_1` | 27,579 | 20,684 | 4,137 | 2,758 | Verified Exact |
| `jigsaw_0` | 19,718 | 14,788 | 2,958 | 1,972 | Verified Exact |
| `jigsaw_1` | 15,885 | 11,914 | 2,383 | 1,588 | Verified Exact |
| `other_sources_0` | 74 | 56 | 11 | 7 | Verified Exact |
| `other_sources_1` | 26 | 20 | 4 | 2 | Verified Exact |
| **Total Corpus** | **84,866** | **63,650** | **12,731** | **8,485** | **Exact $75.0 / 15.0 / 10.0$** |

### 6.2 Class Balance Across Partitions

| Partition | Total Instances | Clean Instances ($0$) | Clean % | Toxic Instances ($1$) | Toxic % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 63,650 | 31,032 | 48.75% | 32,618 | 51.25% |
| **Validation** | 12,731 | 6,207 | 48.75% | 6,524 | 51.25% |
| **Test** | 8,485 | 4,137 | 48.76% | 4,348 | 51.24% |
| **Full Corpus** | **84,866** | **41,376** | **48.75%** | **43,490** | **51.25%** |

### 6.3 Domain Slice Allocations (For Disaggregated Evaluation)

| Domain Slice | Total Instances | Train | Validation | Test | Slice % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `native_twitter` | 49,163 | 36,872 | 7,375 | 4,916 | 57.93% |
| `jigsaw` | 35,603 | 26,702 | 5,341 | 3,560 | 41.95% |
| `other_sources` | 100 | 76 | 15 | 9 | 0.12% |
| **Total** | **84,866** | **63,650** | **12,731** | **8,485** | **100.0%** |

---

## 7. Artifact Inventory & Cryptographic Lockfiles

All processed datasets and verification artifacts have been generated in `data/processed/` and `data/interim/`:

| Artifact Path | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| `data/processed/all.parquet` | Parquet | 26 MB | Full retained supervised corpus ($N = 84,866$) with `split` column. |
| `data/processed/train.parquet` | Parquet | 20 MB | Frozen training partition ($N = 63,650$). |
| `data/processed/val.parquet` | Parquet | 3.9 MB | Frozen validation partition ($N = 12,731$) for error analysis and hyperparameter tuning. |
| `data/processed/test.parquet` | Parquet | 2.6 MB | Frozen test partition ($N = 8,485$) locked for final evaluation. |
| `data/interim/quarantine_conflicts.parquet` | Parquet | 7.6 KB | Quarantined binary conflicts ($N = 4$ physical rows). |
| `data/processed/manifest.json` | JSON | 2.5 KB | Comprehensive execution metadata, schema, slice counts, and source hashes. |
| `data/processed/test_checksum.sha256` | Text | 65 B | SHA-256 hash of physical `test.parquet` file. |
| `data/processed/test_membership.sha256` | Text | 65 B | Deterministic semantic hash of test set `(record_id, label_binary)` pairs. |

### Cryptographic Checksums:
```text
Test Parquet Physical SHA-256:
1c5408bad5928752b39ee6471604954c4f93017ad5fe57b870c1cd49e0b479bd

Test Membership Semantic SHA-256:
7bd973c2e3bc33d5cc5f3010e631ea57fd62969e6a253c31d23a1d43de880174
```

---

## 8. Instructions for Full Reproduction

To reproduce the entire data preparation pipeline from raw assets:

1. **Verify Raw Datasets:**
   ```bash
   uv run python -c "
   from toxicity_detection.data import compute_file_sha256
   from toxicity_detection.config import OVERFIT_GM_RAW_PATH, KAGGLE_TWEETS_RAW_PATH
   print('HF:', compute_file_sha256(OVERFIT_GM_RAW_PATH))
   print('KG:', compute_file_sha256(KAGGLE_TWEETS_RAW_PATH))
   "
   ```

2. **Execute Data Preparation Pipeline:**
   ```bash
   uv run python scripts/run_prepare_data.py
   ```

3. **Run Pipeline Verification & Unit Tests:**
   ```bash
   uv run pytest -v tests/test_data_pipeline.py tests/test_dataset_integrity.py
   ```

4. **Verify Linter Cleanliness:**
   ```bash
   uv run ruff check .
   ```

---
*Report compiled by Senior Data Engineer for Team 07 – TOXICITY. All artifacts verified and sealed.*
