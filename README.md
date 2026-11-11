# Team 07 – TOXICITY: Explainable Turkish Implicit Toxicity Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Course:** CSE 484 / CSE 654 Natural Language Processing — Fall 2026  
**Institution:** Gebze Technical University, Department of Computer Engineering  
**Team Members:**
- **Ahmet Furkan Hazar** (26420101019) – CSE 654 (*Research Lead*)
- **Ahmet Harun Çakır** (26420101002) – CSE 654
- **Orhun Gökdemir** (254201008011) – CSE 654
- **Burak Kurtaran** (210104004240) – CSE 484

---

## 1. Research Overview & The Two-Tier Architecture

### 1.1 Primary Research Question (Unchanged from Proposal)
> *"Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + logistic-regression baseline, and do token-level SHAP explanations identify human-annotated attack cues?"*

**Pre-Registered Negative Result ("No") Conditions:**
1. **Performance Gate:** Fine-tuned BERTurk fails to achieve a statistically significant Macro-F1 improvement over the linear baseline on the held-out implicit attack benchmark ($\Delta \le 0$, $p \ge 0.05$, or 95% bootstrap CI includes zero).
2. **Explainability Gate:** Token-level SHAP explanation highlights fail to overlap annotators' marked rationale cues better than an empirical random-token baseline.

### 1.2 Two-Tier Strategy & Scientific Boundary Notice
During our dataset audit, we observed that public Turkish toxicity datasets capture broad, explicit toxicity and contain significant machine-translation artifacts (~45.8% translated Jigsaw Wikipedia discussions), with zero annotations for implicit attacks or token rationales.

To preserve scientific rigor, Team 07 established a **source-aware two-tier architecture**:
* **Tier 1 (Completed for Mid-Project):** Broad binary Turkish toxicity classification on a verified, deduplicated, source-stratified corpus of $N = 84,866$ records. Establishes the foundational model pipeline and classical/deep baseline benchmarks.
* **Tier 2 (Underway for Final Milestone):** A dedicated, double-annotated evaluation benchmark of 300 standalone Turkish posts capturing implicit attacks, sarcastic mockery, conversational hard negatives (proverbs, negated insults), and token-level human rationale spans.

> [!IMPORTANT]
> **Scientific Scope Boundary:** Tier 1 measures broad toxicity only. High accuracy on Tier 1 does **not** prove that BERTurk can detect implicit attacks or sarcasm. That claim requires evaluation on the held-out Tier 2 benchmark.

---

## 2. Experimental Results Summary

All models were evaluated on the **cryptographically locked test set** ($N = 8,485$ samples, physical SHA-256 `1c5408ba...`, semantic SHA-256 `7bd973c2...`):

| Evaluation Metric | Baseline: TF-IDF + LogReg ($C=10.0$) | BERTurk (`dbmdz/bert-base-turkish-cased`) | Difference ($\Delta$) | Practical Threshold ($\Delta_{\min} = 0.020$) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** (Primary) | 0.9393 | **0.9654** | **+0.0261** | **PASSED (+0.0061 margin)** |
| **Accuracy** | 0.9393 | **0.9655** | **+0.0262** | PASSED |
| **Toxic Recall** | 0.9271 | **0.9809** | **+0.0538** | Substantial Gain |
| **Toxic F1** | 0.9400 | **0.9668** | +0.0268 | — |
| **Average Precision (AP)**| **0.9862** | 0.9854 | -0.0008 | Comparable |
| **ROC-AUC** | 0.9851 | **0.9906** | **+0.0055** | — |

### Key Experimental Insights:
1. **Statistical Significance ($B = 10,000$ Paired Bootstrap, Seed 42):**
   * Observed Difference: $\Delta = \mathbf{+0.0261}$
   * 95% Bootstrap CI: **$[+0.0207, +0.0317]$**
   * Empirical Bootstrap $p$-value: $p = 1 / 10,001 \approx \mathbf{0.00010}$ ($p < 0.001$).
   * Practical Significance: The lower bound ($+0.0207$) strictly exceeds $\Delta_{\min} = 0.020$.
2. **False Negative Collapse:** False Negatives dropped from **317 to 83**—a **73.8% reduction in missed toxic comments**, raising Toxic Recall from 92.71% to **98.09%**.
3. **Consistent Multi-Domain Gains:**
   * **Native Twitter ($N = 4,916$):** Baseline $0.9373 \to$ BERTurk **$0.9620$** ($\Delta = +0.0247$).
   * **Jigsaw Wikipedia ($N = 3,560$):** Baseline $0.9397 \to$ BERTurk **$0.9688$** ($\Delta = +0.0291$).
4. **Error Analysis (30 Curated Cases):** Lexical models fail on implicit curses (*"kanı kurusun"*), metaphors (*"lağım medyası"*), and trigger false alarms on negated trigger words (*"aptal değiliz"*) and benign proverbs (*"ağır sözü kaldırmaz"*).

---

## 3. Mid-Project Deliverables

All formal mid-project milestone materials are committed to the repository:

- **Presentation PDF:** [`Team07_MidProject.pdf`](Team07_MidProject.pdf) (14 pages: 10 talk slides + 4 graded Project Card appendix slides).
- **Presentation Source:** [`Team07_MidProject.pptx`](Team07_MidProject.pptx) (Editable widescreen 16:9 presentation deck).
- **Source Markdown & Project Card:** [`docs/mid_project_report.md`](docs/mid_project_report.md) (Complete slide transcripts, all 12 Project Card fields, citations, and error taxonomy).
- **High-Resolution Figures:** [`reports/mid_project/`](reports/mid_project/) (Source splits, comparative bar chart, confusion matrices, error taxonomy).
- **Research Phase Reports:**
  - Data Preprocessing Report: [`docs/data/preprocessing_report.md`](docs/data/preprocessing_report.md)
  - Baseline Evaluation Report: [`reports/baseline/baseline_report.md`](reports/baseline/baseline_report.md)
  - BERTurk Evaluation Report: [`reports/berturk/berturk_report.md`](reports/berturk/berturk_report.md)

---

## 4. System & Runtime Requirements

- **Operating System:** Linux x86_64 (tested on Ubuntu 24.04 / Zorin OS 18.1)
- **Python Version:** `>=3.11,<3.13` (CPython 3.11.18 managed by `uv`)
- **Package & Dependency Manager:** [uv](https://docs.astral.sh/uv/) (`>=0.5.0`, tested with `0.13.7`)
- **Hardware Acceleration:** NVIDIA GPU supporting CUDA 12.x/13.x (tested on NVIDIA GeForce RTX 4070 Laptop GPU, 8GB VRAM, Driver 580.178.04)

---

## 5. Quickstart & Installation

All dependencies are locked in `uv.lock`.

### 5.1 Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 5.2 Sync Dependencies
```bash
uv sync --all-groups
```

### 5.3 Verify Environment & GPU Detection
```bash
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 6. End-to-End Reproduction Commands

Every result reported in our mid-project submission can be reproduced with a single command:

```bash
# 1. Run deterministic data preparation and cryptographic hash validation
uv run python scripts/run_prepare_data.py

# 2. Train and evaluate the classical TF-IDF + Logistic Regression baseline
uv run python scripts/run_tfidf_baseline.py

# 3. Evaluate the fine-tuned BERTurk model, compute source slices, and run 10,000 paired bootstraps
uv run python scripts/run_berturk.py --eval-only

# 4. Generate the official presentation deck (Team07_MidProject.pptx)
uv run --with python-pptx python scripts/generate_presentation.py

# 5. Run the comprehensive automated test suite (34 unit tests)
uv run pytest -v

# 6. Check linting and code quality
uv run ruff check .
```

---

## 7. Project Structure

```text
.
├── Team07_MidProject.pdf          # Final submission presentation PDF (14 slides)
├── Team07_MidProject.pptx         # Editable presentation source (16:9 widescreen)
├── Team07_Proposal.pdf            # Approved initial project proposal
├── pyproject.toml                 # Project metadata and dependency definitions
├── uv.lock                        # Pinned cryptographic dependency lockfile
├── README.md                      # Project documentation and reproduction guide
├── data/
│   ├── metadata/
│   │   └── dataset_inventory.json # Raw corpus accounting metadata
│   └── processed/
│       ├── manifest.json          # Preprocessing parameters and split hashes
│       ├── test_checksum.sha256   # Cryptographic lockfile for test.parquet
│       └── test_membership.sha256 # Cryptographic lockfile for test record IDs
├── docs/
│   ├── audit/                     # Initial repository and research design audits
│   ├── data/                      # Preprocessing, canonicalization, and design reports
│   └── mid_project_report.md      # Complete text of mid-project talk & Project Card
├── models/
│   ├── baselines/                 # Baseline model hyperparameters and configs
│   └── berturk/                   # BERTurk hyperparameters, tokenizer, and config
├── reports/
│   ├── baseline/                  # Baseline evaluation metrics, errors, and curves
│   ├── berturk/                   # BERTurk evaluation metrics, bootstrap, and curves
│   └── mid_project/               # High-resolution charts generated for presentation
├── scripts/
│   ├── generate_presentation.py   # Script generating Team07_MidProject.pptx
│   ├── run_prepare_data.py        # Deterministic data preparation CLI
│   ├── run_tfidf_baseline.py      # TF-IDF baseline training and evaluation CLI
│   └── run_berturk.py             # BERTurk evaluation and paired bootstrap CLI
├── src/
│   └── toxicity_detection/        # Core research package
│       ├── config.py              # Centralized paths and seed configurations
│       ├── data/                  # Canonicalization, deduplication, filtering, partitioning
│       ├── evaluation/            # Metrics, bootstrap, error analysis, plotting
│       ├── models/                # Baseline and BERTurk architectures and collators
│       └── training/              # Training loop, seed management, early stopping
└── tests/                         # Automated test suite (34 passing tests)
```

---

## 8. Transparency & AI Assistance Disclosure

In compliance with academic integrity guidelines:
* **ChatGPT (OpenAI):** Used for initial literature discovery, proposal critical review, and prompt design.
* **Antigravity (Google DeepMind):** Used as an interactive pair-programming assistant for implementing data cleaning pipelines, PyTorch dynamic padding collators, automated test suites, and formatting document artifacts.
* **Integrity Commitment:** All experimental data, metrics, error categorizations, and analytical claims were computed directly on host hardware and verified against underlying machine-readable artifacts. No AI tool was permitted to fabricate experimental numbers or simulate human annotations.
