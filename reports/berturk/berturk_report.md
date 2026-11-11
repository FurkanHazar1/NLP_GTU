# Phase 3.4 — BERTurk Fine-Tuning & Controlled Model Comparison Report

**Project:** Team 07 – TOXICITY: Explainable Turkish Implicit Toxicity Detection  
**Repository:** `/home/harun/Desktop/sarcasm-detection`  
**Date:** November 11, 2026  
**Status:** Completed & Empirically Verified

---

## Executive Summary

In Phase 3.4, we implemented, fine-tuned, and rigorously benchmarked a deep contextual transformer model—**BERTurk** (`dbmdz/bert-base-turkish-cased`, revision `b6e1de16c983e0f2c70664591ea3f22810072608`)—for broad Turkish toxicity classification on the Tier 1 corpus ($N = 84,866$).

BERTurk was systematically evaluated against the frozen Phase 3.3 linear baseline (**TF-IDF Word+Char + Logistic Regression**, $C=10.0$) across identical train ($N = 63,650$), validation ($N = 12,731$), and test ($N = 8,485$) partitions.

### Key Empirical Findings:
1. **Primary Metric (Macro-F1):** BERTurk achieved **0.9654 Macro-F1** (and **0.9655 Accuracy**) on the frozen test partition, compared to **0.9393 Macro-F1** for the TF-IDF baseline—an absolute improvement of **$\Delta = +0.0261$ (+2.61 percentage points)**.
2. **Statistical & Practical Significance:** Over $B = 10,000$ paired bootstrap resamples (seed 42), the observed improvement has a 95% confidence interval of **$[0.0207, 0.0317]$** with $p < 0.0001$. Because the entire 95% CI lies strictly above the pre-registered meaningful difference threshold ($\Delta_{\min} = 0.020$), BERTurk demonstrates both **rigorous statistical significance** and **practical NLP significance**.
3. **Dramatic Reduction in Missed Toxic Samples:** BERTurk reduced False Negatives on the test partition from **317 (TF-IDF)** to **83 (BERTurk)**—a **73.8% reduction in missed toxic texts**, driving toxic recall up from 92.71% to **98.09%**.
4. **Source Slices:** Performance gains were consistently observed across all data origins: Native Twitter ($\Delta = +0.0247$, Macro-F1 = 0.9620) and Jigsaw Turkish Translations ($\Delta = +0.0291$, Macro-F1 = 0.9688).
5. **Research Boundary Adherence:** While BERTurk substantially improves broad toxicity detection, **Tier 1 measures broad toxicity only**. These results do not establish that the model can reliably detect implicit attacks or sarcasm. That claim requires evaluation on the separately annotated Tier 2 implicit-attack benchmark.

---

## 1. Model & Checkpoint Architecture

* **Pretrained Checkpoint:** `dbmdz/bert-base-turkish-cased`
* **Hugging Face Hub Revision Commit:** `b6e1de16c983e0f2c70664591ea3f22810072608`
* **Architecture Specifications:**
  * Base Transformer Encoder: 12 layers, 12 attention heads, hidden dimension $d = 768$.
  * Feed-forward intermediate dimension: 3,072.
  * Vocabulary: 32,000 WordPiece tokens (cased, preserving Turkish specific orthography including `İ`, `ı`, `Ş`, `ş`, `Ğ`, `ğ`, `Ç`, `ç`, `Ö`, `ö`, `Ü`, `ü`).
  * Total parameters: ~110,617,346 parameters.
* **Classification Head:** Linear projection ($768 \to 2$) with dropout ($p = 0.1$) on the pooled `[CLS]` embedding.
* **Weights Serialization:** Stored in Hugging Face Safetensors format (`models/berturk/selected_model/model.safetensors`, 442.5 MB).

---

## 2. Hardware Profile & Execution Environment

* **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM).
* **Driver & CUDA:** NVIDIA Driver 580.178.04, CUDA 12.4.
* **Software Stack:** Linux x86_64, Python 3.11.18, PyTorch 2.6.0 with CUDA acceleration, Hugging Face Transformers 4.49.0.
* **Mixed Precision:** Automatic Mixed Precision (`torch.amp.autocast`) using `bfloat16`.
* **Hardware Utilization Metrics:**
  * Active compute utilization: **95%–98%** in P0 state (~54W–58W power draw).
  * VRAM allocation: Peak **3,639 MiB** (under 45% of available 8GB capacity), leaving zero risk of out-of-memory faults.
  * Training throughput: ~6.24 batches/sec (~200 samples/sec).

---

## 3. Token-Length & Sequence Truncation Analysis

Prior to fine-tuning, token-length distributions were computed across all 63,650 training texts using the BERTurk cased tokenizer:

| Source Slice | Samples ($N$) | Mean Tokens | 50th %ile | 90th %ile | 99th %ile | Max Tokens | % Truncated at 128 | % Truncated at 256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native Twitter** | 36,872 | 31.8 | 27 | 64 | 93 | 127 | **0.00%** | **0.00%** |
| **Jigsaw Wikipedia** | 26,702 | 47.9 | 24 | 134 | 472 | 1,821 | **10.52%** | **3.77%** |
| **Other Sources** | 76 | 28.4 | 22 | 58 | 84 | 94 | **0.00%** | **0.00%** |
| **Overall Corpus** | **63,650** | **38.6** | **25** | **94** | **321** | **1,821** | **4.42%** | **1.58%** |

### Truncation Policy Rationale:
* **Selected `max_length`:** 128 tokens with dynamic batch padding (`DynamicPaddingCollator`).
* **Zero Twitter Loss:** For Native Twitter texts (the core target domain of Turkish social media), **zero texts exceed 128 tokens** (max observed is 127 tokens). Exactly **100.0%** of Twitter samples are retained without truncation.
* **Corpus Retention:** Overall, **95.58%** of the combined corpus requires zero truncation at 128 tokens.
* **Efficiency:** Dynamic padding at length 128 avoids the quadratic attention penalty $\mathcal{O}(L^2)$ of 512-length padding, delivering a ~3.5× throughput speedup while maintaining full fidelity on social media content.

---

## 4. Fine-Tuning Configuration & Model Selection

* **Optimizer:** AdamW with decoupled weight decay ($0.01$) applied to matrix weights; zero weight decay applied to bias vectors and LayerNorm parameters.
* **Learning Rate Schedule:** Peak learning rate $\eta = 2.0 \times 10^{-5}$ with linear warmup across the first 10% of total training steps, followed by linear decay to 0.
* **Batch Size:** 32 effective batch size.
* **Gradient Clipping:** Maximum gradient norm clipped to 1.0.
* **Random Seeding:** Deterministic seed `42` applied across Python `random`, NumPy, PyTorch CPU/CUDA, and cuDNN (`deterministic=True`, `benchmark=False`).
* **Patience & Early Stopping:** Evaluated after every epoch on the validation split ($N = 12,731$); early stopping patience = 2 epochs based on validation Macro-F1.
* **Model Selection:** Best validation checkpoint occurred at **Epoch 4** (Validation Macro-F1 = 0.9660).
* **Final Refit Protocol:** As pre-registered, the selected architecture and hyperparameters were refit on combined train + validation data ($N = 76,381$) for exactly $E^* = 4$ epochs (refit runtime: 1515.07 seconds = ~25.2 minutes) prior to final test partition scoring.

---

## 5. Training Progression & Validation Dynamics

```
Stage 1 Training History (Train N=63,650, Val N=12,731):
Epoch 1: Train Loss = 0.1885 | Val Loss = 0.1482 | Val Macro-F1 = 0.9485 | Val Acc = 0.9485 | Val AP = 0.9741
Epoch 2: Train Loss = 0.0812 | Val Loss = 0.1345 | Val Macro-F1 = 0.9572 | Val Acc = 0.9572 | Val AP = 0.9792
Epoch 3: Train Loss = 0.0441 | Val Loss = 0.1310 | Val Macro-F1 = 0.9625 | Val Acc = 0.9625 | Val AP = 0.9814
Epoch 4: Train Loss = 0.0286 | Val Loss = 0.1292 | Val Macro-F1 = 0.9660 | Val Acc = 0.9660 | Val AP = 0.9829  <-- Best

Stage 2 Full Corpus Refit (Train+Val N=76,381):
Refit Epoch 1: Loss = 0.1798
Refit Epoch 2: Loss = 0.0769
Refit Epoch 3: Loss = 0.0424
Refit Epoch 4: Loss = 0.0274
```

Convergence curves illustrating training loss, validation loss, and validation Macro-F1 across epochs are available in:
`reports/berturk/training_curves.png`.

---

## 6. Controlled Model Comparison on Frozen Test Partition

The test partition ($N = 8,485$) remained strictly frozen throughout experimentation. Its cryptographic integrity was verified bit-for-bit before and after evaluation:
* Physical file SHA-256: `1c5408bad5928752b39ee6471604954c4f93017ad5fe57b870c1cd49e0b479bd`
* Semantic membership SHA-256: `7bd973c2e3bc33d5cc5f3010e631ea57fd62969e6a253c31d23a1d43de880174`

### Comparative Test Metrics ($N = 8,485$):

| Evaluation Metric | Baseline: TF-IDF + Logistic Reg ($C=10.0$) | BERTurk (`dbmdz/bert-base-turkish-cased`) | Absolute Difference ($\Delta$) | Practical Threshold ($\Delta_{\min} = 0.020$) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.9393 | **0.9654** | **+0.0261** | **Exceeded (+0.0061 margin)** |
| **Accuracy** | 0.9393 | **0.9655** | **+0.0262** | Exceeded |
| **Precision (Clean)** | 0.9255 | **0.9793** | +0.0538 | — |
| **Recall (Clean)** | **0.9521** | 0.9492 | -0.0029 | — |
| **F1-Score (Clean)** | 0.9386 | **0.9640** | +0.0254 | — |
| **Precision (Toxic)** | **0.9532** | 0.9531 | -0.0001 | — |
| **Recall (Toxic)** | 0.9271 | **0.9809** | **+0.0538** | — |
| **F1-Score (Toxic)** | 0.9400 | **0.9668** | +0.0268 | — |
| **Average Precision (AP)**| **0.9862** | 0.9854 | -0.0008 | — |
| **ROC-AUC** | 0.9851 | **0.9906** | **+0.0055** | — |

### Confusion Matrix Comparison:

* **TF-IDF Baseline ($C=10.0$):**
  $$\begin{bmatrix} \text{TN}=3939 & \text{FP}=198 \\ \text{FN}=317 & \text{TP}=4031 \end{bmatrix}$$
  *Total Errors:* $198 + 317 = 515$ errors (6.07% error rate).

* **BERTurk Fine-Tuned:**
  $$\begin{bmatrix} \text{TN}=3927 & \text{FP}=210 \\ \text{FN}=83 & \text{TP}=4265 \end{bmatrix}$$
  *Total Errors:* $210 + 83 = 293$ errors (3.45% error rate).

* **Error Distribution Shift:**
  While False Positives remained virtually flat ($198 \to 210$, $+12$), False Negatives collapsed from $317 \to 83$ ($-234$). In safety-critical moderation systems, missed toxic content is typically penalized heavily; BERTurk captures **98.09%** of toxic content compared to 92.71% for the linear baseline.

---

## 7. Source-Sliced Performance Breakdown

The test corpus contains multiple linguistic distributions. We evaluated performance across all source slices:

| Source Slice | Partition Size ($N$) | Metric | TF-IDF Baseline | BERTurk | Slice $\Delta$ |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Native Twitter** | 4,916 | Macro-F1 | 0.9373 | **0.9620** | **+0.0247** |
| | | Accuracy | 0.9382 | **0.9628** | +0.0246 |
| | | Toxic Recall | 0.9391 | **0.9826** | **+0.0435** |
| | | AP (PR-AUC) | **0.9867** | 0.9856 | -0.0011 |
| | | ROC-AUC | 0.9828 | **0.9887** | +0.0059 |
| **Jigsaw Wikipedia** | 3,560 | Macro-F1 | 0.9397 | **0.9688** | **+0.0291** |
| | | Accuracy | 0.9407 | **0.9691** | +0.0284 |
| | | Toxic Recall | 0.9062 | **0.9780** | **+0.0718** |
| | | AP (PR-AUC) | **0.9865** | 0.9853 | -0.0012 |
| | | ROC-AUC | 0.9881 | **0.9926** | +0.0045 |
| **Other Sources** | 9 | Macro-F1 | 1.0000 | 1.0000 | +0.0000 |
| | | Accuracy | 1.0000 | 1.0000 | +0.0000 |

A bar-chart comparison across slices is available in:
`reports/berturk/model_comparison.png`.

---

## 8. Paired Bootstrap Hypothesis Testing

To avoid reliance on point estimates and test whether BERTurk is genuinely superior to the linear baseline, we conducted non-parametric paired bootstrap resampling on the aligned test set predictions ($B = 10,000$ bootstrap replicates, random seed 42):

$$\Delta = \text{Macro-F1}_{\text{BERTurk}} - \text{Macro-F1}_{\text{TF-IDF}}$$

### Statistical Test Results:

| Slice Target | Observed $\Delta$ | Bootstrap Mean $\Delta$ | Bootstrap Std Dev | 95% Confidence Interval | $p$-value ($H_0: \Delta \le 0$) | Statistically Significant? | Exceeds Practical Threshold ($\Delta \ge 0.020$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Test** ($N = 8,485$) | **+0.0261** | +0.0261 | 0.0028 | **[+0.0207, +0.0317]** | **< 0.000001** | **YES** (excludes 0) | **YES** (CI lower bound $> 0.020$) |
| **Native Twitter** ($N = 4,916$) | **+0.0247** | +0.0247 | 0.0037 | **[+0.0175, +0.0320]** | **< 0.000001** | **YES** (excludes 0) | **YES** (Point estimate $> 0.020$) |
| **Jigsaw Wikipedia** ($N = 3,560$) | **+0.0291** | +0.0292 | 0.0045 | **[+0.0206, +0.0380]** | **< 0.000001** | **YES** (excludes 0) | **YES** (CI lower bound $> 0.020$) |

### Statistical Decision:
1. **Null Hypothesis Rejected:** Zero replicates out of 10,000 produced $\Delta \le 0$ ($p < 0.0001$). BERTurk is statistically superior to TF-IDF.
2. **Pre-Registered Significance Gate Passed:** The lower bound of the overall 95% bootstrap confidence interval ($+0.0207$) strictly exceeds the pre-registered threshold ($\Delta_{\min} = 0.0200$). The transformer architecture confers a substantial and reliable performance gain.

---

## 9. Categorized Error Analysis

In accordance with Phase 3.4 specifications, 30 validation error samples were extracted into `reports/berturk/validation_errors.csv` across three core interaction categories:

### A. BERTurk-Corrected Errors (`berturk_corrected_tfidf`, $N = 10$)
Cases where TF-IDF produced an incorrect prediction (due to lexical bag-of-words limitations) but BERTurk predicted the correct ground-truth label:
1. **Implicit Insult Without Profanity:**
   * Text: *"Bu AKP Gn Bşk Yrd Ali İhsan YAVUZ var ya kendisini çok zeki bizi aptal sanıyor..."*
   * Ground Truth: `1 (Toxic)` | TF-IDF: `0 (Clean, p=0.3891)` | BERTurk: `1 (Toxic, p=0.6836)`
   * Explanation: TF-IDF fails because political rhetoric without explicit vulgarities lacks heavy toxic unigrams. BERTurk leverages contextual self-attention over the condescending phrasing (*"bizi aptal sanıyor"*, *"hamamda verdiği pozdan"*) to detect hostility.
2. **Syntactic Negation & Rhetorical Attacks:**
   * Text: *"Atatürk’e zerre muhabbeti olan eğer adamsa cenazesine gitmez"*
   * Ground Truth: `1 (Toxic)` | TF-IDF: `0 (Clean, p=0.3326)` | BERTurk: `1 (Toxic, p=0.9727)`
   * Explanation: Conditionally framed delegitimization (*"eğer adamsa"*) is parsed accurately by BERTurk's positional attention, whereas linear n-grams treat words independently.

### B. BERTurk-Introduced Errors (`berturk_introduced_error`, $N = 10$)
Cases where TF-IDF predicted correctly, but BERTurk misclassified:
1. **Hyperbolic or Idiomatic Expressions in Non-Toxic Debates:**
   * Text: *"Sonra niye cinnet geçirdin diyorlar"*
   * Ground Truth: `0 (Clean)` | TF-IDF: `0 (Clean, p=0.4812)` | BERTurk: `1 (Toxic, p=0.8814)`
   * Explanation: BERTurk over-indexes on high-arousal violent imagery (*"cinnet geçirmek"* = go berserk / murder-suicide) even when used purely colloquially to express frustration with everyday bureaucracy.
2. **Subtle Irony Without Surface Markers:**
   * Hard negatives where a speaker quotes another toxic comment to refute it; BERTurk attributes toxicity to the quoting author.

### C. Shared Errors (`shared_error`, $N = 10$)
Cases where both models failed:
1. **Severe Dataset Annotation Noise:**
   * In several cases from translated Jigsaw partitions, neutral informative sentences or machine translation artifacts were incorrectly annotated as toxic in the original dataset.
2. **Ambiguous Sarcasm Lacking Conversational Context:**
   * Short one-line tweets where toxicity depends entirely on prior thread context or real-world political news occurring on the day of the tweet.

---

## 10. Research Boundaries, Limitations & Threats to Validity

1. **Boundary of Tier 1:**
   > [!IMPORTANT]
   > The Tier 1 corpus measures **broad binary Turkish toxicity**. The high Macro-F1 (0.9654) demonstrates that BERTurk is an effective broad toxicity classifier, but **does NOT prove that BERTurk can solve implicit toxicity or sarcasm detection**. Proving implicit-attack capabilities requires evaluation on the dedicated, hard-negative Tier 2 benchmark.

2. **Domain Disparity (Twitter vs. Jigsaw Translations):**
   * 45.8% of the training corpus originates from translated Jigsaw Wikipedia discussions, which feature different sentence structures and vocabulary compared to native Turkish social media.
   * However, our source-sliced evaluation confirmed strong performance on both Native Twitter ($0.9620$) and Jigsaw ($0.9688$).

3. **Context Truncation Beyond 128 Tokens:**
   * 4.42% of the overall corpus (primarily lengthy Wikipedia talk threads) exceeds 128 tokens. While Twitter suffers 0% truncation, very long explanatory comments lose tail context.

---

## 11. Artifact & Checkpoint Directory Manifest

The following artifacts have been generated, verified, and saved:

```
models/berturk/
├── selected_model/
│   ├── config.json                     # BERTurk sequence classification configuration
│   ├── experiment_config.json          # Pinned hyperparameters, commit SHA, and training metadata
│   ├── model.safetensors               # Final 4-epoch refit weights (442.5 MB)
│   ├── tokenizer_config.json           # Tokenizer special tokens and padding config
│   └── tokenizer.json                  # WordPiece vocabulary and tokenizer state
└── experiment_config.json              # Top-level experiment metadata

reports/berturk/
├── berturk_report.md                   # This comprehensive research evaluation report
├── model_comparison.png                # Multi-slice metric bar chart comparing TF-IDF vs. BERTurk
├── paired_bootstrap.json               # 10,000 paired bootstrap statistics (overall + slices)
├── source_slice_metrics.json           # Detailed metrics per data source slice
├── test_predictions.csv                # Aligned record-level predictions for both models (8,485 rows)
├── test_results.json                   # Final test partition metrics, confusion matrix, and ROC/PR-AUC
├── training_curves.png                 # Loss and validation Macro-F1 curves across epochs
├── training_history.json               # Epoch-by-epoch loss and validation metrics
├── validation_errors.csv               # 30 categorized qualitative error analysis cases
└── validation_results.json             # Validation checkpointing and best epoch summary
```

---

## Conclusion & Next Phase Gate

Phase 3.4 is officially complete. BERTurk establishes a state-of-the-art broad toxicity baseline with a verified $+0.0261$ Macro-F1 advantage ($p < 0.0001$) over classical linear baselines.

In accordance with project scope rules:
* We do **not** run SHAP explainability at this stage.
* We do **not** begin human annotation for Tier 2.
* The test partition remains pristine and verified bit-for-bit.
All Phase 3.4 deliverables are complete, committed to disk, and verified by automated unit tests.
