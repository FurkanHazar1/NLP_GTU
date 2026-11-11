# Phase 2 — Repository Audit & Research Design Validation Report

**Project:** Explainable Turkish Implicit Toxicity Detection  
**Team:** Team 07 – TOXICITY  
**Date:** November 10, 2026 (Local Time: 23:45)  
**Role:** Senior ML Research Engineer  
**Status:** Complete Audit & Architecture Validation  

---

## 1. Executive Summary

This audit assesses the state of the **Team 07 – TOXICITY** repository ahead of the CSE 484/654 mid-project milestone (Deadline: **November 11, 2026 at 23:59**; Presentation: **November 12, 2026**). 

The Python runtime, PyTorch (CUDA 12.4), and `uv` dependency infrastructure were initialized and verified in Phase 1. However, this audit reveals a critical reality: **the project currently has zero ML pipeline code, zero ingested datasets, zero baseline numbers, and zero annotated errors.** Furthermore, fundamental ambiguities exist between the project proposal text, the slide deck, and standard machine learning practices—specifically regarding whether the task is a 5-class classification or binary implicit-attack detection, and how the general 8,780 test split relates to the manually annotated test set.

This document identifies these design discrepancies, assesses implementation gaps against the course rubric, defines a clean modular architecture, and proposes an immediate, concrete execution plan for the mid-project deliverable.

---

## 2. Repository Reconnaissance

### 2.1 Directory Structure & File Inventory
An exhaustive filesystem traversal (excluding `.venv`, `.git`, `.ruff_cache`, and `.pytest_cache`) reveals the exact files present in `/home/harun/Desktop/sarcasm-detection`:

```text
/home/harun/Desktop/sarcasm-detection
├── .gitignore                                              # Excludes .venv, caches, data, models, logs
├── .python-version                                         # Pinned to Python 3.11
├── pyproject.toml                                          # Configured dependencies & cu124 index
├── uv.lock                                                 # Locked 192 packages
├── README.md                                               # Environment documentation & reproduction steps
├── Team07_Proposal.pdf                                     # Original project proposal document
├── NLP_MidProject_Presentation_Assignment_Rubric_2026Fall_v2.docx # Official mid-project rubric
├── Explainable_Turkish_Implicit_Toxicity_Detection_(2).pptx # Proposal 5-slide visual presentation
├── scripts/
│   └── check_environment.py                                # Hardware & stack validation script (Executable)
├── src/
│   └── toxicity_detection/
│       └── __init__.py                                     # Empty package init (64 bytes)
├── tests/
│   └── test_environment.py                                 # 5 passing environment/stack tests
└── docs/
    └── audit/                                              # Created during this audit
```

### 2.2 Git Status & Commit History
- **Current Branch:** `master`
- **Commit History:** Exactly 1 commit:
  - `1907b05 chore: initialize reproducible development environment using uv`
- **Working Tree:** Untracked proposal assets (`Team07_Proposal.pdf`, `NLP_MidProject_Presentation_Assignment_Rubric_2026Fall_v2.docx`, `Explainable_Turkish_Implicit_Toxicity_Detection_(2).pptx`). All other files tracked and clean.

### 2.3 Existing Code & Data State
1. **Source Code:** Only `src/toxicity_detection/__init__.py` exists. No data loading, preprocessing, feature extraction, modeling, or evaluation code is present.
2. **Notebooks:** None exist.
3. **Configuration Files:** Only `pyproject.toml` exists. No YAML or JSON experiment configuration files exist.
4. **Datasets & Annotations:** No raw datasets (neither Hugging Face `Overfit-GM/turkish-toxic-language` nor Kaggle `6-Class Labeled Turkish Tweets`), no preprocessed splits, no annotation guidelines, and no annotated pilot/test files exist in the repository.
5. **Checkpoints & Logs:** No trained weights, checkpoints, logs, or experiment metrics exist.
6. **Tests:** Only `tests/test_environment.py` exists (testing runtime imports, synthetic scikit-learn Logistic Regression, PyTorch GPU tensor math, and Transformers configuration classes).

---

## 3. Critical Research Design Audit

Our research question is:
> *Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve Macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + Logistic Regression baseline? And do token-level SHAP explanations identify human-annotated attack cues better than a random-token baseline?*

The audit identified five critical research design ambiguities that require immediate formal decisions.

---

### Ambiguity A: Five-Class versus Binary Classification

#### Conflicting Assumptions
- **Assumption 1 (5-Class Multiclass):** Proposal Section 3 and Slide 3 describe the dataset as containing five distinct target categories: `Profanity`, `Insult`, `Racism`, `Sexual Content`, and `Harmless/Other`. Proposal Section 3 specifically states: *"primary test evaluates multi-class toxic speech detection across positive and negative instances."*
- **Assumption 2 (Binary Classification):** Proposal Section 4 and the Research Question state: *"Main model: BERTurk fine-tuned for binary implicit-attack vs hard-negative classification; same curated split and labels."*

#### Why It Matters
A classifier trained on 5 mutually exclusive toxicity categories cannot output binary predictions (`implicit_attack` vs. `hard_negative`) without an explicit mapping. If the training data labels generic profanity and explicit insults, a standard supervised model will predict those 5 categories. If the evaluation test set consists solely of binary labels (`implicit_attack` vs. `hard_negative`), there is an unresolved semantic and label-space mismatch.

#### Possible Solutions
1. **Approach 1 (Two-Stage Transfer / Zero-Shot Probe):** Train both TF-IDF baseline and BERTurk on the 5-class corpus (or collapsed binary: Toxic vs. Non-Toxic). Evaluate their ability to detect implicit attacks by treating `implicit_attack` as Toxic and `hard_negative` as Non-Toxic.
2. **Approach 2 (Unified Binary Framing):** Map the 87,800 training corpus into binary labels: Toxic (1) vs. Non-Toxic / Harmless (0). Fine-tune BERTurk on this binary classification task, and test on the held-out binary implicit attack set.
3. **Approach 3 (Dual Evaluation):** Track A and Track B both train on the 5-class corpus and report 5-class Macro-F1 on the general test split (8,780 tweets), and additionally evaluate binary transfer or slice performance on the curated implicit attack subset.

#### Recommended Approach
**Adopt Approach 2 (Unified Binary Framing with Subtype Metadata) as the primary research track, while preserving multi-class tags in the metadata:**
- The research question explicitly specifies *"binary implicit-attack vs hard-negative classification"*.
- In the 87,800 raw dataset, collapse the 4 toxic classes (`Profanity`, `Insult`, `Racism`, `Sexual Content`) into **Class 1 (Toxic / Offensive)** and `Harmless/Other` into **Class 0 (Non-Toxic)**.
- Both the TF-IDF + Logistic Regression baseline and BERTurk will be trained on this binary task.
- The held-out test set will evaluate whether models trained on broad offensive labels generalize to implicit/sarcastic attacks (Class 1) versus hard negatives (sarcastic but harmless posts, Class 0).
- This aligns cleanly with the Research Question, the falsifiability condition, and fair baseline comparison.

#### Approver
**Ahmet Furkan Hazar (Research Lead)**

---

### Ambiguity B: General Dataset versus Manually Annotated Test Set

#### Conflicting Assumptions
- Proposal Section 3 reports an 87,800 deduplicated corpus with a 75/15/10 split yielding **8,780 test samples**.
- Slide 3 calls this 8,780 split: *"The Core Asset: Locked, Human-Annotated Test Set. Curated specifically to isolate implicit attacks and hard negatives."*
- Proposal Section 9 mentions a *"50-item pilot"* and states in Section 8: *"Should annotation capacity be insufficient, the dataset size will be transparently reduced to prioritize a balanced, double-annotated test set over a larger volume of training data."*

#### Why It Matters
Manually double-annotating 8,780 tweets with token-level rationale spans is practically impossible for a 4-person team before the course deadline. Furthermore, the raw datasets from Kaggle and Hugging Face are mined tweets with generic distantly-supervised or crowd-sourced multi-class labels; they are **not** pre-curated for implicit sarcasm or token spans.

#### Possible Solutions
1. **Solution 1:** Assume the entire 8,780 test set is already human-annotated (Factually false; inspectable artifacts prove it does not exist).
2. **Solution 2 (Two-Tier Test Partition):**
   - **Tier 1 (General Test Split, $N = 8,780$):** Retain the 10% stratified holdout from the 87,800 corpus for reporting general toxicity classification performance.
   - **Tier 2 (Curated Implicit Gold Set, $N = 100 \text{ to } 300$):** Sample a targeted subset (e.g., 50-item pilot for mid-project, expanded to 150–300 for final) containing verified implicit/sarcastic attacks and hard negatives, with human-annotated token-level evidence spans for SHAP evaluation.

#### Recommended Approach
**Adopt Solution 2 (Two-Tier Test Partition):**
- Tier 1 provides the macro-scale, reproducible test split required by the rubric.
- Tier 2 serves as the focused evaluation set for the core hypothesis (implicit attacks and SHAP token overlap).
- For the immediate mid-project deliverable, freeze Tier 1 ($N = 8,780$) and run the baseline to produce the required results table and 30 error analyses.

#### Approver
**Ahmet Furkan Hazar (Research Lead) & Orhun Gökdemir (Data Lead)**

---

### Ambiguity C: Fair Model Comparison & Controlled Experimental Setup

#### Current Integrity Verification
For the comparison between the baseline (Track A: TF-IDF + Logistic Regression) and main model (Track B: BERTurk) to be scientifically valid:
1. **Identical Inputs:** Both models must receive the exact same raw text strings.
2. **Identical Targets:** Both models must predict the exact same label space (binary: 0 vs. 1).
3. **Identical Partitions:** Both models must train strictly on the training partition ($N = 65,850$). Neither model may access validation or test data during training.
4. **Identical Metric:** Macro-F1 computed on the identical frozen test partition using identical thresholding ($0.5$).
5. **No Leakage:** Preprocessing (e.g. TF-IDF vocabulary extraction, scaling) must be fit solely on the training split.

#### Potential Invalidation Risks
- If BERTurk is evaluated on a binary subset while TF-IDF is evaluated on a 5-class split, the comparison is invalid.
- If TF-IDF is tuned using test set feedback, the comparison is invalid.
- Both models must be evaluated using the exact same evaluation script.

#### Approver
**Burak Kurtaran (Modeling Lead) & Ahmet Furkan Hazar (Research Lead)**

---

### Ambiguity D: Test-Set Integrity, Near-Duplicate Leakage & Freezing Protocol

#### Leakage Risks in Turkish Social Media
1. **Retweet & Bot Duplication:** Turkish political and social tweets frequently contain identical copypastas, retweets (`RT @...`), and template spam.
2. **Normalized Text Collision:** Two tweets differing only in punctuation, URLs, user mentions, or Turkish capitalization (`İ`/`i`, `I`/`ı`) are functional duplicates. If deduplication occurs *after* splitting, duplicate tweets will leak across train and test sets.
3. **Preprocessing Protocol:**
   - Text normalization and strict hash-based deduplication must occur **before** the 75/15/10 train/val/test split.
   - The test set must be serialized to disk (e.g., `data/processed/test.parquet`) with a fixed SHA-256 checksum and locked against any modification.

#### Approver
**Orhun Gökdemir (Data Lead)**

---

### Ambiguity E: Statistical Protocol & Pre-Registered Decisions

#### Requirements from Rubric & Proposal
The rubric explicitly mandates: *"Evaluation protocol: Metric, frozen test set, seeds or bootstrap, and the smallest difference you will call real, fixed before method results."*

#### Current Status in Repository
- Random seeds are not fixed in any configuration.
- Bootstrap sample size and iterations are not coded.
- The minimum practically meaningful Macro-F1 difference ($\Delta_{min}$) has **not been pre-registered**.

#### Recommended Statistical Specification
1. **Primary Metric:** Macro-F1 (unweighted mean of F1 scores across classes).
2. **Random Seeds:** Master seed `42` (with multi-seed runs: `42`, `123`, `456`, `789`, `1011` for final model stability).
3. **Paired Bootstrap Protocol:**
   - Number of resamples: $B = 10,000$.
   - Sample size: $N_{sample} = N_{test}$ (with replacement).
   - Confidence Interval: 95% percentile bootstrap interval ($2.5\text{th}$ and $97.5\text{th}$ percentiles of $\Delta\text{Macro-F1} = \text{Macro-F1}_{\text{BERTurk}} - \text{Macro-F1}_{\text{TF-IDF}}$).
   - Decision Rule: Fail to reject the null hypothesis if the 95% CI contains 0.
4. **Pre-Registered Meaningful Difference ($\Delta_{min}$):**
   - We define $\Delta_{min} = +2.0 \text{ Macro-F1 points}$ ($+0.020$).
   - A statistically significant difference that yields $< 2.0$ points is deemed practically negligible given the computational overhead of BERTurk.

#### Approver
**Ahmet Furkan Hazar (Research Lead)**

---

## 4. Existing Implementation Audit

Every required system component has been inspected against the current repository contents:

| Component | Status | Source File Reference | Audit Evidence & Notes |
| :--- | :--- | :--- | :--- |
| **1. Dataset acquisition & loading** | **NOT IMPLEMENTED** | None | No acquisition scripts. `kagglehub` and `datasets` are installed in the venv, but no loader exists. |
| **2. Data preprocessing** | **NOT IMPLEMENTED** | None | No text cleaning, URL/mention stripping, or Turkish case handling code exists. |
| **3. Deduplication** | **NOT IMPLEMENTED** | None | No exact or near-duplicate hash filtering code exists. |
| **4. Label normalization** | **BLOCKED BY RESEARCH DECISION** | None | Dependent on formal approval of Ambiguity A (mapping 5/6 classes to binary). |
| **5. Dataset splitting** | **NOT IMPLEMENTED** | None | No stratified splitting script. No saved train/val/test splits on disk. |
| **6. Annotation infrastructure** | **NOT IMPLEMENTED** | None | No annotation guidelines, pilot spreadsheet, or token span labeling format defined. |
| **7. Inter-annotator agreement** | **NOT IMPLEMENTED** | None | No Cohen’s $\kappa$ or Krippendorff’s $\alpha$ calculation scripts or pilot data. |
| **8. TF-IDF baseline** | **NOT IMPLEMENTED** | None | No scikit-learn Pipeline combining word/char TF-IDF and Logistic Regression. |
| **9. BERTurk fine-tuning** | **NOT IMPLEMENTED** | None | No training script using Hugging Face `Trainer` or PyTorch. |
| **10. Model evaluation** | **PARTIALLY IMPLEMENTED** | `tests/test_environment.py` | Environment test verifies scikit-learn metrics on synthetic data, but no project evaluation module exists. |
| **11. Bootstrap confidence intervals** | **NOT IMPLEMENTED** | None | No paired bootstrap sampling or CI code exists. |
| **12. Error analysis** | **NOT IMPLEMENTED** | None | 0 of the required 30 baseline errors have been extracted or categorized. |
| **13. SHAP explainability** | **NOT IMPLEMENTED** | None | SHAP library is installed and importable, but no token attribution pipeline exists. |
| **14. Human evidence alignment** | **BLOCKED BY RESEARCH DECISION** | None | Requires human evidence span annotations (dependent on Ambiguity B). |
| **15. Random-token comparison** | **NOT IMPLEMENTED** | None | No baseline random token generator or overlap F1 computation exists. |
| **16. Reproducibility** | **PARTIALLY IMPLEMENTED** | `pyproject.toml`, `uv.lock`, `README.md` | Environment is 100% reproducible via `uv sync --all-groups`, but experimental pipeline scripts do not yet exist. |
| **17. Interactive demonstration** | **NOT IMPLEMENTED** | None | No demo interface (Gradio/Streamlit) exists. |

---

## 5. Mid-Project Deliverable Gap Analysis

**Course Deadline:** November 11, 2026, 23:59 (Less than 24 hours remaining).  
**Target Submission:** One PDF (`Team07_MidProject.pdf`) containing $\le 10$ presentation slides + 3–4 Project Card appendix slides.

### 5.1 Presentation Slides Checklist ($\le 10$ Slides)

| Rubric Requirement | Status | Current Evidence / Gap | Required Action |
| :--- | :--- | :--- | :--- |
| **Slide 1: Research question & falsifiability** | **PARTIALLY MET** | Drafted in `Explainable_Turkish_Implicit_Toxicity_Detection_(2).pptx` Slide 1. | Formalize exact text; state the "no" condition clearly. |
| **Slide 2: Motivation & Related Work** | **PARTIALLY MET** | Drafted in Slide 2 & 3 of proposal PPTX. | Update with Bilgin & Bekar (2025) and Mahmud et al. (2024). |
| **Slide 3: Data as loaded & frozen test set** | **NOT MET** | Proposal reported numbers (87,800 total; 65,850 / 13,170 / 8,780), but data is not loaded in repo. | Ingest datasets, generate splits, verify license, lock test set on disk, report exact counts. |
| **Slide 4: One real data example** | **PARTIALLY MET** | Proposal cites *"saray soytarısı Nagehan"*. | Select a concrete sample from the ingested corpus with clear token cues. |
| **Slide 5: Baseline result & spread** | **NOT MET** | **CRITICAL GAP:** 0 numbers. Baseline has not been run. | Implement & run TF-IDF + Logistic Regression on frozen test set; report Macro-F1 with bootstrap 95% CI. |
| **Slide 6: Error analysis (grouped)** | **NOT MET** | **CRITICAL GAP:** 0 categorized errors. | Extract and categorize at least 30 baseline prediction errors into $\ge 3$ linguistic failure modes. |
| **Slide 7: Method status (BERTurk dev result)** | **NOT MET** | BERTurk fine-tuning not started. | Run initial development probe or report architecture setup & preliminary validation loss. |
| **Slide 8: Explanation / XAI Setup** | **PARTIALLY MET** | Conceptual design ready; SHAP installed. | Show token attribution methodology and random-token baseline design. |
| **Slide 9: Plan to final, dates & owners** | **PARTIALLY MET** | Proposal Section 9 has preliminary dates. | Update table with specific dates (Nov 13 – Dec 15), owners, ablation, and fallback. |
| **Slide 10: Risks & Fallback strategy** | **MET** | Defined in Proposal Section 8 and PPTX Slide 5. | Ready for presentation slide. |

### 5.2 Project Card Checklist (12 Required Fields)

| # | Field | Status | Evidence & Missing Elements |
| :---: | :--- | :--- | :--- |
| **1** | Title and abstract ($\le 150$ words) | **PARTIALLY MET** | Needs drafting with explicit placeholder for baseline result. |
| **2** | Research question | **MET** | Proposal Section 1 text satisfies criteria. |
| **3** | Contributions (2–3 testable claims) | **MET** | Stated as claims rather than task lists. |
| **4** | Related work (5–8 full references) | **PARTIALLY MET** | Proposal has 3 citations; must expand to 5–8 complete citations with 1-line contrast. |
| **5** | Data (sources, licenses, counts, example) | **NOT MET** | Actual post-deduplication counts must be verified on disk; license terms documented. |
| **6** | Method and baseline | **PARTIALLY MET** | Architecture described; compute requirements documented ($\le 3$ GPU-hours). |
| **7** | Evaluation protocol | **PARTIALLY MET** | Metric fixed (Macro-F1); bootstrap specified; $\Delta_{min}$ must be formally fixed. |
| **8** | Results table | **NOT MET** | Baseline row (Macro-F1 $\pm$ 95% CI) must be populated from actual test set run. |
| **9** | Error analysis ($\ge 30$ errors) | **NOT MET** | Table of 30 categorized errors with examples is completely missing. |
| **10**| Research lead section | **MET** | Ahmet Furkan Hazar lead section drafted in proposal; needs ablation details. |
| **11**| Plan and risk | **PARTIALLY MET** | Milestones, dates, and fallback plan drafted. |
| **12**| Reproducibility | **PARTIALLY MET** | Repository reproducible via `uv`; one-command baseline execution script missing. |

---

## 6. Team Responsibilities & Ownership Matrix

Using the roles defined in the proposal:

| Member | Academic Role | System Ownership | Mid-Project Deliverable Responsibility |
| :--- | :--- | :--- | :--- |
| **Ahmet Furkan Hazar** | Research Lead (CSE 654) | Experimental Design & Statistics | - Pre-register $\Delta_{min}$ and statistical protocol.<br>- Approve Ambiguities A, B, and E.<br>- Author Project Card Field 10 (Research Lead Section).<br>- Lead presentation Q&A on research design. |
| **Orhun Gökdemir** | Graduate Researcher (CSE 654) | Data Curation & Ingestion | - Implement dataset ingestion from Hugging Face & Kaggle.<br>- Execute deduplication and stratified 75/15/10 split.<br>- Lock and hash `test.parquet`.<br>- Verify licensing and schema. |
| **Burak Kurtaran** | Undergraduate Researcher (CSE 484) | Baseline & Transformers | - Implement TF-IDF + Logistic Regression pipeline.<br>- Run baseline on frozen test set and generate results table.<br>- Extract and format the 30 baseline errors.<br>- Setup initial BERTurk training script. |
| **Ahmet Harun Çakır** | Graduate Researcher (CSE 654) | XAI, Evaluation & Demo | - Implement evaluation metrics & paired bootstrap engine.<br>- Build SHAP token explainer and random-token baseline.<br>- Structure the 30 baseline errors into linguistic failure modes.<br>- Compile presentation slides and Project Card PDF. |

---

## 7. Recommended Architecture

To avoid monolithic scripts and technical debt, the repository should be organized into simple, decoupled, testable Python modules under `src/toxicity_detection`:

```text
src/toxicity_detection/
├── __init__.py
├── config.py                 # Dataclasses & YAML schema for paths, seeds, hyperparameters
├── data/
│   ├── __init__.py
│   ├── ingest.py             # Downloads & loads Overfit-GM and Kaggle sources
│   ├── preprocess.py         # Turkish text normalization, URL/mention cleaning
│   └── split.py              # Stratified splitting and test-set locking with checksums
├── models/
│   ├── __init__.py
│   ├── baseline.py           # scikit-learn Pipeline (Word/Char TF-IDF + LogisticRegression)
│   └── transformer.py        # Hugging Face AutoModelForSequenceClassification wrapper
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Macro-F1, per-class PR-AUC, precision, recall
│   └── bootstrap.py          # Paired bootstrap confidence intervals (B=10,000)
├── xai/
│   ├── __init__.py
│   ├── explainer.py          # SHAP partition/tree explainer wrapper
│   └── alignment.py          # Token-level overlap F1 vs. human evidence & random baseline
└── analysis/
    ├── __init__.py
    └── error_analysis.py     # Categorizes prediction errors (FP/FN) and exports inspection tables
```

### Supporting Project Layout
```text
configs/
├── baseline.yaml             # C-parameter, n-gram ranges, sublinear TF
└── berturk.yaml              # Learning rate, batch size, warmup, max_seq_length

data/
├── raw/                      # Downloaded source files (ignored by git)
├── processed/                # train.parquet, val.parquet, test.parquet (ignored by git)
└── annotations/              # pilot_50.json, guidelines.md (tracked by git)

scripts/
├── check_environment.py      # Environment probe (existing)
├── run_ingest.py             # Data acquisition CLI
├── run_baseline.py           # One-command baseline execution & metric reporting
└── run_error_analysis.py     # Extracts 30 categorized errors for Project Card
```

---

## 8. Antigravity Skills Assessment

Based on the actual project state and the immediate mid-project deadline, we assess the 12 candidate skills:

| Skill Candidate | Immediate Need? | Recommended Phase | Primary Responsibility |
| :--- | :---: | :---: | :--- |
| **`dataset-engineering`** | **YES (CRITICAL)** | Immediate (Phase 3) | Robust ingestion of Overfit-GM + Kaggle, text cleaning, deduplication, stratified splitting, and lockfile generation. |
| **`classical-ml-baseline`**| **YES (CRITICAL)** | Immediate (Phase 3) | Scikit-learn Pipeline, word+char TF-IDF, Logistic Regression training, test-set prediction, and seed management. |
| **`statistical-evaluation`**| **YES (CRITICAL)** | Immediate (Phase 3) | Paired bootstrap 95% CI calculation, Macro-F1, PR-AUC, and results table formatting. |
| **`error-analysis`** | **YES (CRITICAL)** | Immediate (Phase 3) | Automated extraction of false positives/negatives, clustering by linguistic type, and markdown table export for the 30 errors. |
| **`research-documentation`**| **YES (CRITICAL)** | Immediate (Phase 3) | Assembly of Project Card fields and mid-project presentation slides. |
| **`annotation-quality`** | NO | Final Project (Phase 4) | Guideline drafting, Cohen’s $\kappa$, pilot adjudication. |
| **`berturk-finetuning`** | NO | Final Project (Phase 4) | PyTorch / HF Transformers fine-tuning on RTX 4070 or Colab. |
| **`shap-explainability`** | NO | Final Project (Phase 4) | Token attribution extraction and evidence overlap computation. |
| **`testing-and-reproducibility`**| NO | Final Project (Phase 4) | Full CI regression suite and pipeline integrity validation. |
| **`interactive-demo`** | NO | Final Project (Phase 4) | Gradio / Streamlit demonstration. |
| **`experiment-management`**| NO | Final Project (Phase 4) | MLflow/Wandb tracking for multi-seed transformer runs. |
| **`project-context`** | NO | Ongoing | Context coordination. |

---

## 9. Prioritized Development Roadmap

### Track 1: Immediate Mid-Project Requirements (Next 24 Hours)
*Objective: Secure 100/100 points on the November 11 Mid-Project Submission.*

1. **Task 1.1: Data Ingestion & Splitting**
   - **Owner:** Orhun Gökdemir
   - **Output:** `data/processed/train.parquet`, `val.parquet`, `test.parquet`.
   - **Dependencies:** Approval of Ambiguity A (Binary label mapping).
   - **Acceptance Criteria:** Exactly 87,800 deduplicated records partitioned into 65,850 train, 13,170 val, and 8,780 test. Test set checksum saved in `test_checksum.sha256`.

2. **Task 1.2: One-Command TF-IDF + Logistic Regression Baseline**
   - **Owner:** Burak Kurtaran
   - **Output:** `scripts/run_baseline.py` and saved baseline artifact.
   - **Dependencies:** Task 1.1.
   - **Acceptance Criteria:** Model trains on `train.parquet` in $< 60$ seconds, evaluates on `test.parquet`, and outputs Macro-F1, per-class metrics, and confusion matrix.

3. **Task 1.3: Paired Bootstrap Statistical Evaluation**
   - **Owner:** Ahmet Furkan Hazar
   - **Output:** `src/toxicity_detection/evaluation/bootstrap.py`.
   - **Dependencies:** Task 1.2.
   - **Acceptance Criteria:** Generates 95% bootstrap confidence interval ($B=10,000$) on test set predictions.

4. **Task 1.4: 30 Categorized Baseline Errors**
   - **Owner:** Ahmet Harun Çakır
   - **Output:** `docs/audit/baseline_error_analysis_30.md`.
   - **Dependencies:** Task 1.2.
   - **Acceptance Criteria:** Exactly 30 test-set prediction errors categorized into at least 3 distinct failure modes (e.g. Implicit Sarcasm without Slurs, Non-Toxic Profanity / Swearing as Emphasis, Dialectal Slang) with Turkish text, true label, predicted label, and model probability.

5. **Task 1.5: Mid-Project Slide Deck & Project Card Compilation**
   - **Owner:** All Team Members (Led by Ahmet Harun Çakır)
   - **Output:** `Team07_MidProject.pdf`.
   - **Dependencies:** Tasks 1.1, 1.2, 1.3, 1.4.
   - **Acceptance Criteria:** 10 presentation slides + 3–4 Project Card appendix slides containing real numbers, submitted before Nov 11, 23:59.

---

### Track 2: Final Project Requirements (Post-Midterm)
1. **Pilot Annotation:** Draft guidelines, annotate 50 items across 2 annotators, compute inter-annotator agreement ($\kappa$).
2. **BERTurk Fine-Tuning:** Fine-tune `dbmdz/bert-base-turkish-cased` on the curated training split using RTX 4070.
3. **Paired Statistical Comparison:** Compute $\Delta\text{Macro-F1}$ and test whether 95% CI excludes zero and exceeds $\Delta_{min} = 2.0$.
4. **SHAP Explanation & Random-Token Baseline:** Compute token-level attributions and measure overlap F1 against human rationale annotations.
5. **Interactive Demonstration:** Build an explainable toxicity detection app in Gradio.

---

## 10. Recommended Next Implementation Task

### Selected Task: Task 1.1 — Data Ingestion, Binary Mapping & Split Generation

#### Why This Must Be Implemented First
Every single remaining mid-project requirement depends on having data:
- The TF-IDF baseline cannot be trained without training data.
- The test set cannot be frozen without splitting the data.
- The 30 errors cannot be extracted without running a model on the test data.
- The presentation slides and Project Card cannot present real counts without ingested data.

Furthermore, implementing Task 1.1 immediately tests and resolves the core research design ambiguity (Ambiguity A) by programmatically locking the binary mapping and verifying the post-deduplication sample counts against the numbers reported in the proposal.

---
*Report compiled and validated by Senior ML Research Engineer for Team 07 – TOXICITY.*
