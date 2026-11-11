# Baseline Report: TF-IDF + Logistic Regression

**Project:** Explainable Turkish Implicit Toxicity Detection  
**Team:** Team 07 – TOXICITY  
**Date:** November 11, 2026  
**Phase:** Phase 3.3 — Classical Baseline Implementation & Evaluation  
**Selected Baseline Model:** Combined Word (1–2) + Char (3–5) TF-IDF + Logistic Regression ($C = 10.0$)  
**Status:** **Completed, Evaluated, and Frozen**  

---

> [!IMPORTANT]
> **Methodological Scope & Boundary:**
> This baseline establishes performance on **Tier 1 broad Turkish toxicity classification** (overt profanity, insults, threats, sexism, racism). It **does NOT validate implicit attack detection**. The Tier 1 dataset consists of broad social media and translated Wikipedia commentary, not a dedicated implicit toxicity benchmark. Overt lexical markers allow n-gram classifiers to achieve high nominal scores (~94% Macro-F1), while subsequent error analysis reveals acute blindness to subtle, sarcastic, and context-dependent implicit toxicity.

---

## 1. Dataset Description

The baseline models were trained, tuned, and evaluated on the frozen, leak-free partitions generated in Phase 3.2:

| Partition | Instances | Clean ($0$) | Clean % | Toxic ($1$) | Toxic % | Source Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | **63,650** | 31,032 | 48.75% | 32,618 | 51.25% | Native Twitter ($36,872$), Jigsaw ($26,702$), Other ($76$) |
| **Validation** | **12,731** | 6,207 | 48.75% | 6,524 | 51.25% | Native Twitter ($7,375$), Jigsaw ($5,341$), Other ($15$) |
| **Test (Frozen)** | **8,485** | 4,137 | 48.76% | 4,348 | 51.24% | Native Twitter ($4,916$), Jigsaw ($3,560$), Other ($9$) |
| **Total Corpus** | **84,866** | **41,376** | **48.75%** | **43,490** | **51.25%** | Native Twitter ($49,163$), Jigsaw ($35,603$), Other ($100$) |

### Integrity Verification:
- **Physical Test SHA-256:** `1c5408bad5928752b39ee6471604954c4f93017ad5fe57b870c1cd49e0b479bd`
- **Semantic Test Membership SHA-256:** `7bd973c2e3bc33d5cc5f3010e631ea57fd62969e6a253c31d23a1d43de880174`
- Both hashes were verified before and after test evaluation; zero byte modification occurred.

---

## 2. Preprocessing & Feature Engineering

### 2.1 Turkish-Aware Lowercasing Preprocessor
Standard Python and scikit-learn `.lower()` methods improperly fold Turkish characters—most notably uppercase dotted `İ` (`\u0130`), which folds to `i\u0307` or incorrectly maps with dotless `I` (`\u0049`).

We implemented a dedicated preprocessor:
```python
def turkish_lowercase_preprocessor(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()
```

### 2.2 Empirical Preprocessing Validation (Word-level, $C=1.0$)
We compared raw text without lowercasing against Turkish-aware lowercasing on the validation partition:

| Preprocessing Strategy | Accuracy | Macro-F1 | Precision (1) | Recall (1) | PR-AUC | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Text (`lowercase=False`)** | 0.9080 | 0.9080 | 0.9439 | 0.8723 | 0.9712 | Rejected |
| **Turkish-Aware Lowercasing** | **0.9121** | **0.9121** | **0.9496** | **0.8749** | **0.9749** | **Selected** |

Turkish-aware lowercasing achieved higher precision, recall, and Macro-F1 (+0.0041), confirming that resolving dotted/dotless I collisions improves feature quality. Original raw strings are preserved untouched in `text_raw`.

### 2.3 Feature Representations
- **Model A (Word-level):** `analyzer="word"`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`, L2 normalization.
- **Model B (Char-level):** `analyzer="char_wb"`, `ngram_range=(3, 5)`, `min_df=2`, `sublinear_tf=True`, L2 normalization.
- **Model C (Combined):** `FeatureUnion` combining Model A and Model B in a sparse feature matrix. Densification was strictly avoided to maintain memory efficiency.

---

## 3. Systematic Model Exploration & Validation Results

Nine candidate models (3 representations $\times$ 3 regularization parameters $C \in [0.1, 1.0, 10.0]$) were fitted on the training partition ($N = 63,650$) and evaluated on the validation partition ($N = 12,731$):

| Candidate Identifier | Representation | $C$ | Macro-F1 | Accuracy | PR-AUC | Clean F1 | Toxic F1 | Fit Time | Predict Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tfidf_word_c0.1` | Word (1, 2) | 0.1 | 0.8759 | 0.8760 | 0.9545 | 0.8796 | 0.8722 | 3.70s | 0.78s |
| `tfidf_word_c1.0` | Word (1, 2) | 1.0 | 0.9121 | 0.9121 | 0.9749 | 0.9134 | 0.9107 | 5.43s | 0.89s |
| `tfidf_word_c10.0` | Word (1, 2) | 10.0 | 0.9215 | 0.9215 | 0.9792 | 0.9218 | 0.9212 | 8.37s | 0.88s |
| `tfidf_char_c0.1` | Char (3, 5) | 0.1 | 0.8968 | 0.8968 | 0.9679 | 0.8989 | 0.8946 | 13.60s | 2.50s |
| `tfidf_char_c1.0` | Char (3, 5) | 1.0 | 0.9242 | 0.9242 | 0.9808 | 0.9244 | 0.9240 | 13.84s | 2.58s |
| `tfidf_char_c10.0` | Char (3, 5) | 10.0 | 0.9348 | 0.9348 | 0.9851 | 0.9343 | 0.9353 | 16.13s | 2.59s |
| `tfidf_combined_c0.1` | Combined | 0.1 | 0.9111 | 0.9111 | 0.9741 | 0.9127 | 0.9095 | 18.55s | 3.23s |
| `tfidf_combined_c1.0` | Combined | 1.0 | 0.9346 | 0.9346 | 0.9847 | 0.9344 | 0.9348 | 18.58s | 3.37s |
| **`tfidf_combined_c10.0`** | **Combined** | **10.0** | **0.9387** | **0.9387** | **0.9862** | **0.9380** | **0.9395** | **28.56s** | **3.38s** |

---

## 4. Model Selection & Rationale

### Selection Decision:
The **Combined Word + Character TF-IDF with $C = 10.0$** (`tfidf_combined_c10.0`) was selected as the final baseline.

### Formal Rationale:
1. **Meaningful Difference Criterion ($\Delta_{min} = 0.020$):**
   - Best Combined Baseline Macro-F1: **0.9387**
   - Default Word Baseline ($C = 1.0$) Macro-F1: **0.9121** ($\Delta = \mathbf{+0.0266}$, exceeding $\Delta_{min}$).
   - Best Word Baseline ($C = 10.0$) Macro-F1: **0.9215** ($\Delta = +0.0172$).
2. **Linguistic Advantage of Character Sub-words:**
   - Turkish is an agglutinative language with extensive suffixation (e.g. *köpek-ler-imiz-den*). Pure word-level models suffer from extreme vocabulary fragmentation when encountering social media misspellings, vowel elisions, or novel derivations.
   - Character boundary n-grams (`char_wb`, 3–5) capture root-morpheme combinations even when corrupted by slang or typos, lifting recall significantly.
3. **Computational Feasibility:**
   - Inference time remains rapid at $3.38\text{ ms/sample}$ on CPU, making it a highly practical benchmark.

---

## 5. Final Test Evaluation (Frozen Partition)

The selected configuration was refit on the aggregated `train + val` data ($N = 76,381$) and evaluated **exactly once** on the frozen test partition ($N = 8,485$):

### Overall Test Performance:
- **Macro-F1:** **0.9393**
- **Accuracy:** **0.9393** (93.93%)
- **PR-AUC (Average Precision):** **0.9862**
- **ROC-AUC:** **0.9851**
- **Evaluation Time:** $3.21\text{ s}$ ($0.38\text{ ms/sample}$)

### Detailed Class Metrics:
| Target Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Clean ($0$)** | 4,137 | 0.9255 | 0.9521 | 0.9386 |
| **Toxic ($1$)** | 4,348 | 0.9532 | 0.9271 | 0.9400 |
| **Macro Average** | 8,485 | 0.9394 | 0.9396 | **0.9393** |

---

## 6. Source-Aware Performance & Disaggregated Slices

Because the Tier 1 corpus contains distinct data regimes, evaluating across source slices is mandatory:

| Slice Key | Domain / Provenance | Test Size ($N$) | Clean $N$ | Toxic $N$ | Accuracy | Macro-F1 | Clean F1 | Toxic F1 | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | Combined Corpus | 8,485 | 4,137 | 4,348 | 0.9393 | **0.9393** | 0.9386 | 0.9400 | 0.9862 |
| **`native_twitter`** | Authentic Turkish Tweets | 4,916 | 2,158 | 2,758 | 0.9382 | **0.9373** | 0.9301 | 0.9446 | 0.9867 |
| **`jigsaw`** | Translated Wikipedia Talk | 3,560 | 1,972 | 1,588 | 0.9407 | **0.9397** | 0.9477 | 0.9317 | 0.9865 |
| **`other_sources`** | Commercial / Dialogue | 9 | 7 | 2 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

### Methodological Insights on Slices:
1. **Native Social Media vs. Translated Formal Text:** Performance is comparable across Native Twitter ($0.9373$) and Jigsaw ($0.9397$). On Twitter, toxic class recall is higher ($0.9446$), driven by dense profanity markers. On Jigsaw, clean class recall is higher ($0.9477$), reflecting Wikipedia's formal syntax.
2. **Within-Source Interpolation vs. Cross-Domain Generalization:** The current random source-stratified split evaluates in-domain interpolation within each domain. It does **not** prove cross-domain generalization.
3. **`other_sources` Slice Caveat:** With only 9 test instances, the 1.0000 score has zero statistical significance and must not be cited as evidence of domain generalization.

---

## 7. Confusion Matrix Interpretation

```text
                  Predicted Clean (0)    Predicted Toxic (1)
True Clean (0):          3,939                  198           (Total: 4,137)
True Toxic (1):            317                4,031           (Total: 4,348)
```

- **True Negatives (TN):** $3,939$ ($95.21\%$ of clean instances correctly identified).
- **False Positives (FP):** $198$ ($4.79\%$ false alarm rate on clean text).
- **False Negatives (FN):** $317$ ($7.29\%$ miss rate on toxic text).
- **True Positives (TP):** $4,031$ ($92.71\%$ of toxic instances correctly caught).

### Analytical Finding:
The **False Negative rate ($7.29\%$) is 52% higher than the False Positive rate ($4.79\%$)**. When a toxicity classifier relies on surface n-grams, toxicity expressed without conventional vulgarities is systematically missed.

---

## 8. Validation Error Analysis (30 Exemplars)

Exactly 30 misclassified validation examples (15 False Positives, 15 False Negatives) were deterministically sampled into `reports/baseline/validation_errors_30.csv`. Human review categories were initialized as `pending_review`.

### Diagnostic Patterns Observed:

#### False Negatives (Model misses toxicity due to absent surface profanity):
1. **Implicit Malicious Wishes & Curses:**
   - Text: `"Değil unutmak acısı hafifleyenin kanı kurusun"` (Ground Truth: Toxic, Pred: Clean, Prob: $0.3028$).
   - Cause: Expresses a severe curse (*"may the blood run dry of whoever forgets"*) without swear words.
2. **Metaphorical & Rhetorical Insults:**
   - Text: `"Adı üzerinde lağım medyası ne beklenir ki"` (Ground Truth: Toxic, Pred: Clean, Prob: $0.1716$).
   - Cause: Metaphorical slur (*"sewer media"*); individual words appear non-toxic to word/char n-grams.
3. **Rhetorical Derision:**
   - Text: `"Merak ediyorum bu kafayla nasıl yaşıyorsunuz"` (Ground Truth: Toxic, Pred: Clean, Prob: $0.2826$).
   - Cause: Idiomatic contempt (*"I wonder how you live with this head"*) without profanity.

#### False Positives (Model misclassifies clean text due to toxic lexical tokens):
1. **Proverbs & Moral Statements:**
   - Text: `"İnsanoğlu naziktir , ağır sözü kaldırmaz."` (Ground Truth: Clean, Pred: Toxic, Prob: $0.7061$).
   - Cause: Surface token *"ağır söz"* triggers toxicity weights despite the statement advocating politeness.
2. **Explicit Negation of Insults:**
   - Text: `"Hiç birimiz aptal yada saf değiliz,sadece susuyor ve görmezden geliyoruz."` (Ground Truth: Clean, Pred: Toxic, Prob: $0.5691$).
   - Cause: Word *"aptal"* triggers toxic prediction because linear bag-of-words models cannot invert sentiment over clausal negations (*"aptal değiliz"*).
3. **Colloquial Sports Slang:**
   - Text: `"diğer takımlar emek hırsızlığı falan mı yapıyo orasını tam anlamadım"` (Ground Truth: Clean, Pred: Toxic, Prob: $0.8023$).
   - Cause: Accusatory phrase *"emek hırsızlığı"* elevated toxic probability.

---

## 9. Output Artifacts Inventory

```text
models/baselines/
├── tfidf_word/
│   ├── pipeline.joblib       # Best word model (C=10.0, Macro-F1=0.9215)
│   └── config.json
├── tfidf_char/
│   ├── pipeline.joblib       # Best char model (C=10.0, Macro-F1=0.9348)
│   └── config.json
├── tfidf_combined/
│   ├── pipeline.joblib       # Best combined model (C=10.0, Macro-F1=0.9387)
│   └── config.json
└── selected_baseline/
    ├── pipeline.joblib       # Final refit model on train+val (76,381 instances)
    └── config.json

reports/baseline/
├── validation_results.json   # All 9 candidate evaluation metrics and timings
├── test_results.json         # Final frozen test set metrics
├── source_slice_metrics.json # Disaggregated slice metrics (Twitter, Jigsaw, Other)
├── validation_errors_30.csv  # 30 categorized misclassifications
├── confusion_matrix.png      # Annotated test confusion matrix plot
├── model_comparison.png      # Validation performance comparison plot
└── baseline_report.md        # Comprehensive technical report
```

---

## 10. Limitations & Threats to Validity

1. **Broad Toxicity vs. Implicit Toxicity:** The high test Macro-F1 ($0.9393$) reflects the presence of overt lexical features in Tier 1. It must not be interpreted as competence in identifying implicit toxicity, sarcasm, irony, or micro-aggressions.
2. **Bag-of-Words & Negation Blindness:** As demonstrated by the error analysis, linear classifiers fail on clausal negation, rhetorical questions, and metaphorical insults.
3. **Translationese in Jigsaw:** Translated English Wikipedia discussions introduce unnatural syntactic structures that may not represent authentic Turkish conversational dynamics.
4. **Frozen Evaluation Sanctity:** Zero hyperparameter tuning or feature redesign was performed after test evaluation. The test partition's cryptographic membership remains intact.

---
*Report generated for Team 07 – TOXICITY. All baseline assets sealed and verified.*
