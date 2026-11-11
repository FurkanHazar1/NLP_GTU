# Phase 3.1.3 — Preprocessing Specification Reconciliation & Research Design Specification

**Project:** Explainable Turkish Implicit Toxicity Detection  
**Team:** Team 07 – TOXICITY  
**Date:** November 11, 2026 (Local Time: 00:20)  
**Role:** Senior Data Engineer & ML Research Reviewer  
**Artifact Path:** `docs/data/research_design_decisions.md`  
**Status:** Approved Specification — Fully Implemented in Phase 3.2  

---

## 1. Executive Summary & Audit Reconciliation

This specification resolves the discrepancies identified in previous audits regarding normalization policies, duplicate accounting, label conflict resolution, and domain composition.

### Critical Reconciled Figures:
1. **Raw Ingestion Pool:** Exactly **87,799 instances** ($77,800$ Overfit-GM + $9,999$ Kaggle).
2. **Impact of Canonicalization:**
   - Under whitespace stripping only (Policy 2): $85,648$ unique texts ($2,088$ cross-dataset collisions).
   - Under internal whitespace collapsing + Turkish lowercasing + NFC (Policy 4/5): **$85,369$ unique texts** ($2,248$ cross-dataset collisions).
3. **Pure Scam Removal:** Exactly **501 records** in Kaggle represent commercial/phishing spam (`Dolandırıcılık = 1.0`) with **zero** other toxic flags and **zero** overlap with Overfit-GM. Removing them reduces the unique pool to **$84,868$**.
4. **Quarantine of Label Conflicts:** Exactly **2 unique text keys** (3 physical rows) exhibit unresolved binary label conflicts (1 internal Kaggle tie, 1 cross-dataset disagreement). Quarantining them leaves **$\mathbf{84,866}$ retained instances for supervised training**.
5. **Class Balance of Retained Corpus:**
   - **Toxic ($1$):** Exactly **43,490 instances** ($51.25\%$)
   - **Clean ($0$):** Exactly **41,376 instances** ($48.75\%$)
6. **Domain Composition:**
   - `native_twitter` (`told`, `kaggle_tweets`, `bully`): **49,163 instances** ($57.93\%$)
   - `jigsaw` (translated English Wikipedia talk pages): **35,603 instances** ($41.95\%$)
   - `other_sources` (`troff`, `offenseval`): **100 instances** ($0.12\%$)

---

## 2. Recomputed Corpus Accounting Across Normalization Policies

We independently computed deduplication metrics across five progressive normalization policies:

| Policy | Definition | HF Unique | KG Unique | Cross-Dataset Shared Keys | KG Rows Matching HF | Total Unique Union | Added Collisions vs. Prior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw Exact** | As stored on disk | 77,800 | 9,936 | 2,078 | 2,097 | **85,658** | Baseline |
| **2. Stripped Whitespace** | `.str.strip()` | 77,800 | 9,936 | 2,088 | 2,107 | **85,648** | $+10$ Cross ($-10$ Union) |
| **3. Collapsed Whitespace** | `re.sub(r'\s+', ' ', t).strip()` | 77,800 | 9,936 | 2,248 | 2,267 | **85,488** | $+160$ Cross ($-160$ Union) |
| **4. P3 + Turkish Lowercase** | P3 + `İ`$\to$`i`, `I`$\to$`ı` + `.lower()` | 77,689 | 9,928 | 2,248 | 2,269 | **85,369** | $+111$ HF, $+8$ KG ($-119$ Union) |
| **5. P4 + Unicode NFC** | `unicodedata.normalize('NFC', t)` | 77,689 | 9,928 | 2,248 | 2,269 | **85,369** | $+0$ (Already in NFC) |

### 2.1 Collision Inspection Findings
1. **Whitespace Collapsing (Policy 2 $\to$ Policy 3):**
   - Introduces **160 additional cross-dataset collisions** (e.g. `'ulan dalyarak...'` having an accidental double space or `\xa0` non-breaking space in Kaggle).
   - Label Concordance: **100% exact match** ($94$ clean, $66$ toxic).
   - *Verdict:* Genuinely identical tweets. Collapsing whitespace is **mandatory**; failing to collapse whitespace allows identical tweets to leak across train/test splits.
2. **Turkish Lowercasing (Policy 3 $\to$ Policy 4):**
   - Merges **111 internal casing duplicates** in Overfit-GM (across 97 groups) and **8 internal casing duplicates** in Kaggle (across 6 groups).
   - Label Concordance in Overfit-GM: **0 conflicts** out of 97 groups ($100\%$ concordant).
   - Label Concordance in Kaggle: **0 new conflicts** (the only conflicting casing group was already identical in raw text).
   - *Verdict:* Safe and linguistically necessary. It prevents identical sentences with differing capitalization (e.g. initial capital vs. all lowercase) from splitting across partitions.
3. **Recommended Canonicalization Policy:**
   - **Adopt Policy 4 / 5 (Collapsed Whitespace + Turkish-Aware Lowercasing + NFC)** for generating deduplication keys.
   - **Preserve original text:** Store verbatim raw strings as `text_raw` in all processed outputs. Model tokenizers can use raw or normalized text as needed.

---

## 3. Reconciliation of Duplicate Groups & Label Conflicts

### 3.1 Kaggle Internal Duplicates
- **Duplicate Groups:** 69 unique text keys appear more than once in Kaggle (encompassing 140 physical rows, meaning **71 redundant rows**).
- **Label Consistency Within Groups:**
  - **Category-Flag Disagreements:** 9 groups exhibit minor variation in secondary flags (e.g., Row 298 has `Küfür=1, Cinsel=0` while Row 322 has `Küfür=1, Cinsel=1`).
  - **Binary `Tespit` Disagreements:** Exactly **1 group** exhibits a binary label disagreement.

#### The Kaggle Internal Binary Conflict:
- **Canonical Key:** `'bize karşı savunma oynayıp 1 puan aldığı için sevinen başakşehir gs ye 5 atıp şov yapıyor siz neyin kafasını yaş'`
- **Row 4134:** `Tespit = 0.0` (Toxic), all 6 flags $0.0$
- **Row 7089:** `Tespit = 1.0` (Clean), all 6 flags $0.0$
- **Analysis:** This is a sports tweet complaining about team performance without profanity or slurs. It represents an exact 1:1 tie ($1$ clean vs. $1$ toxic).
- **Resolution:** **Quarantine from supervised training** into `data/interim/quarantine_conflicts.parquet`.

### 3.2 Cross-Dataset Overlap & Provenance Audit
- **Total Shared Keys (under Policy 4):** **2,248 unique keys** (matching 2,269 Kaggle rows).
- **True Overfit-GM Source Distribution of Overlapping Keys:**
  - `bully`: Exactly **2,242 keys** ($99.73\%$)
  - `told`: Exactly **3 keys** ($0.13\%$)
  - `jigsaw`: Exactly **3 keys** ($0.13\%$)
- *Audit Correction:* The previous audit claimed overlapping records came from `told`/`bully`. In reality, **$99.73\%$ originate specifically from `bully`** (the Turkish Cyberbullying Dataset).
- **Provenance Note on Kaggle Annotations:** The Kaggle dataset lacks peer-reviewed documentation establishing formal double-annotation or adjudication. Kaggle labels must **not** be called "human-verified gold standards"; they are crowd-sourced / distantly curated social media tags.

#### The Cross-Dataset Binary Disagreement:
- **Canonical Key:** `'milletin çocuğu nasıl bu kadar uslu sessiz akıllı olabiliyor benimki tam bir fırlama eslemin peşinde koşmaktan bir'`
- **Overfit-GM (`bully`):** Labeled `INSULT` (`is_toxic = 1`) due to keyword rule on *"fırlama"*.
- **Kaggle:** Labeled `Tespit = 1.0` (Clean, all flags $0.0$).
- **Analysis:** Affectionate parental remark (*"mine is a total rascal"*).
- **Resolution:** Rather than arbitrarily overwriting Overfit-GM or Kaggle, **quarantine this record** into `data/interim/quarantine_conflicts.parquet`.

---

## 4. Recomputed Post-Filtering Tier 1 Target Counts

### 4.1 Step-by-Step Accounting Pipeline

```text
Raw Records Ingested:
  - Overfit-GM:                            77,800
  - Kaggle Tweets:                          9,999
  Total Raw Rows:                          87,799

Deduplication & Filtering Reductions:
  (-) Kaggle Pure Scam Records:              -501  (All 501 have Dolandırıcılık=1, 0 other flags, 0 overlap)
  (-) Overfit-GM Internal Duplicates:        -111  (111 redundant casing rows across 97 groups)
  (-) Kaggle Non-Scam Internal Duplicates:    -71  (71 redundant rows across 69 groups)
  (-) Cross-Dataset Overlapping Keys:      -2,248  (Dropped from Kaggle, retained under Overfit-GM metadata)
  (-) Quarantined Unresolved Conflicts:        -2  (1 internal Kaggle tie, 1 cross-dataset disagreement)

Final Retained Supervised Corpus:          84,866 unique instances
```

### 4.2 Final Target Class Distribution

| Split Label | Definition | Instance Count | Percentage |
| :---: | :--- | :---: | :---: |
| **`0`** | Clean / Harmless / Non-Toxic | **41,376** | **48.75%** |
| **`1`** | Broad Toxicity (Profanity, Insult, Racism, Sexism, Threat) | **43,490** | **51.25%** |
| **Total** | Retained Training & Evaluation Pool | **84,866** | **100.0%** |

### 4.3 Detailed Source-by-Class Contingency Table

| Source Key | Provenance / Origin | Clean ($0$) | Toxic ($1$) | Total | % of Corpus |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `told` | Native Turkish Twitter (Profanity & Slurs) | 16,735 | 22,726 | **39,461** | 46.50% |
| `jigsaw` | Translated English Wikipedia Talk Pages | 19,718 | 15,885 | **35,603** | 41.95% |
| `kaggle_tweets` | Native Turkish Twitter (Non-overlapping, non-scam)| 3,721 | 3,457 | **7,178** | 8.46% |
| `bully` | Native Turkish Twitter Cyberbullying | 1,128 | 1,396 | **2,524** | 2.97% |
| `troff` | Twitter Commercial / Escort Spam | 70 | 24 | **94** | 0.11% |
| `offenseval` | Threaded Conversations | 4 | 2 | **6** | 0.01% |
| **All Sources** | | **41,376** | **43,490** | **84,866** | **100.0%** |

---

## 5. Source Grouping & Stratified Split Feasibility

### 5.1 Source Group Definitions
To avoid misrepresenting rare sources or obscuring domain shifts, we map all 84,866 records into three mutually exclusive source groups:
1. **`native_twitter` ($N = 49,163$, $57.93\%$):** Merges `told`, `kaggle_tweets`, and `bully`. Authentic Turkish microblogging with slang, colloquial syntax, and abbreviations.
2. **`jigsaw` ($N = 35,603$, $41.95\%$):** Translated English Wikipedia talk pages. Long sequence lengths, translationese, and synthetic Turkish structures.
3. **`other_sources` ($N = 100$, $0.12\%$):** Retains `troff` ($94$) and `offenseval` ($6$). Explicitly separated to avoid distorting social media benchmarks.

### 5.2 Split Feasibility Analysis (75% Train / 15% Val / 10% Test)
Using composite stratum `f"{source_group}_{final_label}"` with master seed `42`:

| Composite Stratum | Population | Train ($75\%$) | Validation ($15\%$) | Test ($10\%$) | Stratum Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `native_twitter_0` | 21,584 | 16,188 | 3,238 | 2,158 | Highly Robust |
| `native_twitter_1` | 27,579 | 20,684 | 4,137 | 2,758 | Highly Robust |
| `jigsaw_0` | 19,718 | 14,788 | 2,958 | 1,972 | Highly Robust |
| `jigsaw_1` | 15,885 | 11,914 | 2,383 | 1,588 | Highly Robust |
| `other_sources_0` | 74 | 56 | 11 | 7 | Partitionable (Small) |
| `other_sources_1` | 26 | 20 | 4 | 2 | Partitionable (Small) |
| **Total Split Sizes** | **84,866** | **63,650** | **12,731** | **8,485** | **Exact $75.0 / 15.0 / 10.0$** |

### 5.3 Methodological Reality of Source Stratification
- **Within-Source vs. Cross-Domain Generalization:** A random stratified split partitions each source proportionally across train, dev, and test. Consequently, performance on the test set measures **in-domain interpolation within each source**, *not* out-of-domain zero-shot generalization.
- **Reporting Requirement:** In the mid-project presentation and final report, metrics must be reported **disaggregated by slice**:
  1. Overall Test Set Macro-F1 ($N = 8,485$)
  2. Native Turkish Social Media Slice ($N = 4,916$)
  3. Translated Jigsaw Slice ($N = 3,560$)

---

## 6. Corrected Methodological Terminology & Commitments

1. **Terminology Correction ("Pretraining" vs. "Source-Task Fine-Tuning"):**
   - The proposal and earlier notes referred to training on Tier 1 as "pretraining".
   - *Correction:* Initializing from `dbmdz/bert-base-turkish-cased` and training on 84k labeled instances is **supervised broad-toxicity source-task fine-tuning**, not pretraining.
2. **The $+2.0$ Macro-F1 Meaningful Difference ($\Delta_{min}$):**
   - Must be documented as **proposed $\Delta_{min}$ awaiting formal team ratification**, not as a pre-registered historical fact.
3. **Strict Protection of the Test Set:**
   - The test set ($N = 8,485$) is strictly locked for final evaluation. It must **never** be used for model selection, learning rate tuning, or prompt exploration.
4. **Error Analysis Source (30 Errors for Mid-Project):**
   - *Rubric Requirement:* The course rubric expects baseline numbers on a fixed test set and 30 categorized errors.
   - *Best-Practice Recommendation:* Extract the 30 categorized baseline prediction errors from the **Validation Set (`val.parquet`)** to preserve the absolute sanctity of the frozen Test Set against human snooping. If the course evaluator requires test-set error analysis, the baseline model must be frozen and evaluated once, with explicit documentation that zero hyperparameter adjustments were made based on those errors.
5. **Tier 2 Gold Benchmark Agreement Metrics:**
   - Sentence/Tweet Binary Agreement: Measured via **Cohen’s $\kappa$** (target: $\kappa \ge 0.70$).
   - Token Rationale Span Agreement: Measured via **Token-Level Intersection-over-Union (IoU) / Dice F1** across annotator highlights, rather than simple kappa.

---

## 7. Preprocessing Module Layout for Implementation

```text
src/toxicity_detection/
├── config.py                 # Pinned master seed (42), ratios (0.75, 0.15, 0.10)
├── data/
│   ├── canonicalize.py       # P4 normalization, Turkish lowercasing, SHA-256 ID generation
│   ├── deduplicate.py        # Internal & cross-source deduplication logic
│   ├── filter.py             # Drops 501 scam records, maps binary targets (0/1)
│   ├── quarantine.py         # Exports 2 conflicting keys to data/interim/quarantine.parquet
│   └── partition.py          # Stratified split execution -> train, val, test.parquet
└── evaluation/
    ├── metrics.py            # Macro-F1, sliced reporting (native vs jigsaw)
    └── bootstrap.py          # Paired bootstrap CI (B=10,000)

scripts/
└── run_prepare_data.py       # Deterministic CLI generating processed partitions and test lockfile
```

---

## 8. Summary of Team Approvals & Implementation Status

The research design and preprocessing specification have been formally ratified by the research team and successfully executed via `scripts/run_prepare_data.py`:

| # | Decision Item | Approved Value / Action | Status | Approver |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Retained Corpus Size** | Formalized $\mathbf{84,866}$ unique instances ($43,490$ toxic, $41,376$ clean). | **[APPROVED]** | Ahmet Furkan Hazar |
| **2** | **Scam Removal** | Excluded all 501 pure `Dolandırıcılık` tweets from the corpus. | **[APPROVED]** | Ahmet Furkan Hazar |
| **3** | **Quarantine Policy** | Quarantined the 2 binary conflict keys (4 physical rows) to `data/interim/quarantine_conflicts.parquet`. | **[APPROVED]** | Ahmet Furkan Hazar & Orhun Gökdemir |
| **4** | **Canonicalization Policy** | Adopted Policy 4/5 (whitespace collapsed + Turkish lowercasing + NFC). | **[APPROVED]** | Orhun Gökdemir (Data Lead) |
| **5** | **Source-Sliced Reporting** | Mandated reporting separate metrics for Native Twitter vs. Translated Jigsaw. | **[APPROVED]** | Ahmet Furkan Hazar |
| **6** | **Error Analysis Partition** | Extract 30 baseline errors from Validation partition (`val.parquet`). | **[APPROVED]** | Burak Kurtaran & Ahmet Harun Çakır |
| **7** | **Meaningful Difference** | Ratified proposed $\Delta_{min} = +2.0 \text{ Macro-F1 points}$ ($+0.020$). | **[APPROVED]** | Ahmet Furkan Hazar |

---
*Specification compiled by Senior Data Engineer & ML Research Reviewer for Team 07 – TOXICITY. Formally approved and verified in Phase 3.2.*
