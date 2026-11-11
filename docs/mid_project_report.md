# Team 07 – TOXICITY: Explainable Turkish Implicit Toxicity Detection
## Mid-Project Presentation & Project Card Submission

* **Course:** CSE 484 / CSE 654 Natural Language Processing — Fall 2026
* **Institution:** Gebze Technical University (GTU), Department of Computer Engineering
* **Date:** November 11, 2026
* **Team Name / Number:** Team 07 – TOXICITY
* **Team Members:**
  * **Ahmet Furkan Hazar** (26420101019) – CSE 654 (*Research Lead*)
  * **Ahmet Harun Çakır** (26420101002) – CSE 654
  * **Orhun Gökdemir** (254201008011) – CSE 654
  * **Burak Kurtaran** (210104004240) – CSE 484

---

## Executive Summary & Research Scope Notice

This document constitutes the comprehensive textual source and archival record for the Mid-Project milestone deliverable of **Team 07 – TOXICITY**.

### Research Question & The Two-Tier Research Architecture
The project's primary research question remains strictly identical to the approved proposal:
> **Primary Research Question:**  
> *"Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + logistic-regression baseline, and do token-level SHAP explanations identify human-annotated attack cues?"*
>
> **Pre-Registered Negative Result Condition:**  
> The answer is empirically defined as **“no”** if:
> 1. The fine-tuned transformer model's Macro-F1 improvement over the linear baseline is not statistically distinguishable from zero ($p \ge 0.05$ or 95% bootstrap confidence interval contains zero); **or**
> 2. Token-level SHAP explanation highlights fail to overlap annotators' marked rationale cues better than a random-token baseline.

### Crucial Methodological Boundary:
During repository reconnaissance and exploratory data auditing (Phase 3.1), we discovered that available public Turkish toxicity corpora (`Overfit-GM/turkish-toxic-language` and Kaggle `smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi`) do **not** contain dedicated tags, hard negatives, or rationale annotations for *implicit* attacks or *sarcasm*. Instead, they capture broad, explicit cyberbullying, slurs, profanity, and translated Wikipedia edit disputes.

To resolve this without corrupting the original scientific objective, Team 07 adopted a **source-aware two-tier architecture**:
* **Tier 1 (Complete):** Broad binary Turkish toxicity classification on a rigorously cleaned, deduplicated, and source-stratified corpus of $N = 84,866$ records. This provides the foundational infrastructure, controlled training pipeline, and competitive baseline models.
* **Tier 2 (Underway for Final Milestone):** A dedicated, double-annotated evaluation benchmark of standalone Turkish posts exhibiting implicit targeted hostility, sarcasm, conversational hard negatives, and token-level human rationale spans.

**Scientific Boundary Commitment:** We explicitly affirm that **Tier 1 experimental results do NOT establish that BERTurk detects implicit attacks or sarcasm**. All numbers reported herein demonstrate progress on broad Turkish toxicity modeling as a prerequisite stepping stone toward answering the primary research question at the final milestone.

---

# Part 1: Presentation Slides (10-Minute Talk Transcript)

Below is the slide-by-slide structure, textual content, visual references, and presenter notes for the 10 presentation slides.

---

### Slide 1: Title, Research Question & Negative Result Condition

* **Header:** Team 07 – TOXICITY · CSE 484 / CSE 654 NLP Mid-Project Milestone
* **Slide Title:** Explainable Turkish Implicit Toxicity Detection
* **Authors:** Ahmet Furkan Hazar (Research Lead), Ahmet Harun Çakır, Orhun Gökdemir, Burak Kurtaran
* **Primary Research Question:**
  $$\text{Can fine-tuned BERTurk outperform TF-IDF on implicit/sarcastic attacks, and do SHAP cues match human rationale?}$$
* **Formal Negative Result (“No”) Conditions:**
  1. **Performance Gate:** Fine-tuned BERTurk fails to achieve a statistically significant Macro-F1 gain over the TF-IDF baseline on the held-out implicit attack benchmark ($\Delta \le 0$, $p \ge 0.05$, or 95% bootstrap CI includes 0).
  2. **Explainability Gate:** Token-level SHAP attribution heatmaps fail to overlap human-annotated attack rationales better than an empirical random-token baseline (measured via token-level AUPRC / F1).
* **Current Milestone Scope Notice:**
  * Tier 1 (broad toxicity on $N = 84,866$) is complete.
  * Tier 1 provides preliminary model development; it does **not** yet answer the implicit-attack question.
* **Presenter Notes (1:00 min):**
  *"Good morning. We are Team 07, working on Explainable Turkish Implicit Toxicity Detection. On our very first slide, we state our primary research question and its negative-result condition without alteration from our original proposal. We are asking whether fine-tuning BERTurk can detect implicit, sarcastic attacks better than a linear baseline, and whether SHAP token explanations faithfully reflect human evidence. Our answer is formally 'no' if BERTurk fails to beat the baseline or if SHAP explanations do not beat a random baseline. Today, we report our completed Tier 1 foundation—a frozen benchmark of nearly 85,000 samples—and our controlled baseline comparison."*

---

### Slide 2: Why This Problem Matters: Explicit vs. Implicit Toxicity

* **Slide Title:** Motivation: The Challenge of Implicit Toxicity in Turkish
* **The Core Moderation Problem:**
  * **Surface Lexical Toxicity:** Profanity, explicit slurs, and overt threats are readily captured by regex, keyword filters, and n-gram models.
  * **Implicit & Sarcastic Attacks:** Rhetorical mockery, conditional curses (*"eğer adamsa"*, *"kanı kurusun"*), metaphorical insults (*"lağım medyası"*), and sarcastic praise bypass surface filters while inflicting severe social harm.
  * **Conversational Hard Negatives:** Proverbs, benign moral debates (*"insanoğlu naziktir"*), and negated insults (*"aptal değiliz"*) falsely trigger keyword-based systems.
* **The Explainability Imperative:**
  * High classification accuracy alone does not guarantee trustworthy moderation.
  * Autonomous or human-in-the-loop content moderation requires **faithful rationale extraction**—verifying that the model flagged a post because of its hostile semantics, not spurious correlation (e.g., identity terms, political entity mentions).
* **Research Focus:** Narrowed strictly to *standalone Turkish posts* with double-annotated rationale spans.
* **Presenter Notes (1:00 min):**
  *"Why does this matter? Standard NLP classifiers and commercial filters are optimized for explicit profanity. In Turkish social media, targeted attacks increasingly manifest as subtle sarcasm, rhetorical questions, and metaphorical insults that contain zero curse words. Meanwhile, benign moral maxims and political debate often get falsely censored. Furthermore, black-box deep models can make the right prediction for the wrong reason—by latching onto specific demographic tokens. This is why our project couples classification with token-level SHAP explanation evaluation."*

---

### Slide 3: Data Reconnaissance & The Two-Tier Strategy

* **Slide Title:** Dataset Reconnaissance & Research Architecture
* **Audit of Public Turkish Datasets:**
  1. `Overfit-GM/turkish-toxic-language` (Hugging Face): 77,800 raw records.
  2. `smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi` (Kaggle): 9,999 raw records.
* **Key Findings of the Data Audit:**
  * **Absence of Implicit Tags:** Public datasets label generic offensiveness, profanity, or topic-level categories; none annotate implicit attacks or token rationales.
  * **Hybrid Nature (Critical Caveat):** 45.8% of the Overfit-GM dataset originates from machine-translated Jigsaw Wikipedia talk pages. It is **not** a purely native Turkish social-media corpus.
  * **Kaggle Label Polarity:** Kaggle's binary `Tespit` column was empirically verified as $1 = \text{clean}$ and $0 = \text{toxic}$ (counter-intuitive polarity).
* **The Two-Tier Architecture Solution:**
  * **Tier 1:** Merged, cleaned, source-stratified broad toxicity corpus ($N = 84,866$). Used for rigorous baseline development and initial transformer fine-tuning.
  * **Tier 2:** Dedicated human-annotated implicit-attack and hard-negative benchmark with span rationales (Final Milestone).
* **Visual:** Embedded diagram: `reports/mid_project/dataset_splits_and_sources.png`
* **Presenter Notes (1:00 min):**
  *"When we audited the public Turkish datasets, we discovered an important reality: there is no public Turkish dataset labeled for implicit attacks or token rationales. Furthermore, nearly 46% of the widely used Overfit-GM corpus consists of machine-translated English Wikipedia comments containing distinct translation artifacts. To maintain research integrity, we designed a two-tier strategy: Tier 1 establishes a clean, source-aware broad toxicity foundation of 84,866 samples; Tier 2 is our curated, human-annotated implicit benchmark currently underway."*

---

### Slide 4: Data Preprocessing & Evaluation Protocol

* **Slide Title:** Preprocessing Invariants & Evaluation Protocol
* **Deterministic Cleaning Pipeline:**
  * **Corpus Accounting:** 87,799 raw records $\to$ 84,866 retained Tier 1 records (41,376 clean / 48.75%, 43,490 toxic / 51.25%).
  * **Excluded Entries:** 501 pure scam/fraud advertising records (*"günde 5000 TL kazan"*) were cleanly isolated and removed.
  * **Turkish Canonicalization:** Case folding (`I` $\to$ `ı`, `İ` $\to$ `i`), whitespace normalization, URL/mention neutralization for duplicate detection while preserving original raw text for model input.
  * **Conflict Quarantine:** 16 exact canonical collisions with opposing labels quarantined to prevent label noise.
* **Partitioning & Cryptographic Locking:**
  * Source-stratified 75% / 15% / 10% partition:
    * **Train:** $63,650$ samples
    * **Validation:** $12,731$ samples
    * **Frozen Test Set:** $8,485$ samples
  * Test set locked with physical file SHA-256 (`1c5408ba...`) and semantic membership SHA-256 (`7bd973c2...`). Verified bit-for-bit before and after all runs.
* **Pre-Registered Evaluation Rules:**
  * **Primary Metric:** Macro-averaged F1 (robust to slight class imbalance).
  * **Practical Significance Threshold:** $\Delta_{\min} = 0.020$ (+2.0 percentage points Macro-F1).
  * **Statistical Protocol:** $B = 10,000$ paired bootstrap resamples on test predictions (seed 42).
* **Presenter Notes (1:00 min):**
  *"Our data preparation was completely deterministic and leak-free. We excluded 501 scam records, resolved Turkish casing nuances for deduplication, and quarantined 16 conflicting label pairs. We partitioned the corpus into 75% train, 15% validation, and 10% test, stratified by data source. The test set of 8,485 samples was cryptographically locked with SHA-256 hashes. In our evaluation protocol, we established a pre-registered practical difference threshold of Delta = 0.020 Macro-F1 and paired bootstrap testing over 10,000 resamples."*

---

### Slide 5: Classical Machine Learning Baseline: TF-IDF + Logistic Regression

* **Slide Title:** Classical Baseline: TF-IDF + Logistic Regression
* **Feature Extraction & Hyperparameter Selection:**
  * Vectorizer: Combined Word n-grams (1–2 grams, sublinear TF, $\min\text{DF}=2$, 100k max features) + Character n-grams (`char_wb`, 3–5 grams, sublinear TF, $\min\text{DF}=5$, 100k max features). Total feature space: 147,749 sparse dimensions.
  * Classifier: Logistic Regression with $L_2$ regularization ($C = 10.0$) and balanced class weighting.
* **Frozen Test Set Performance ($N = 8,485$):**
  * **Macro-F1:** **0.9393** (Validation: 0.9387)
  * **Accuracy:** **0.9393**
  * **Average Precision (AP):** **0.9862** (Riemann sum over PR curve)
  * **ROC-AUC:** **0.9851**
* **Test Confusion Matrix ($N = 8,485$):**
  $$\begin{bmatrix} \text{TN} = 3,939 & \text{FP} = 198 \\ \text{FN} = 317 & \text{TP} = 4,031 \end{bmatrix}$$
  *Total Errors:* 515 errors (6.07% error rate). False Negatives (317) dominate False Positives (198).
* **Uncertainty Spread:** 95% Bootstrap CI on Baseline Macro-F1: $[0.9341, 0.9443]$.
* **Presenter Notes (1:00 min):**
  *"Our classical baseline combines word n-grams and character-within-word n-grams into a 147,000-dimensional sparse feature space, classified by Logistic Regression with C=10. On our frozen test set, this baseline performs remarkably well on broad toxicity, achieving 0.9393 Macro-F1 and 0.9862 Average Precision. However, looking at the confusion matrix, it produced 317 False Negatives—meaning it missed 317 toxic comments. Let us look into why."*

---

### Slide 6: Qualitative Baseline Error Analysis (30 Curated Cases)

* **Slide Title:** Qualitative Error Analysis: 30 Baseline Misclassifications
* **Analysis Scope:** Systematic review of `reports/baseline/validation_errors_30.csv` (15 False Negatives, 15 False Positives).
* **Taxonomy of Failure Modes:**
  * **False Negatives (Missed Toxicity, $N = 15$):**
    1. *Implicit Malicious Wishes & Curses ($N=3$):* e.g., *"Değil unutmak acısı hafifleyenin kanı kurusun"* ($p = 0.3028$). Hostile curse without explicit profanity.
    2. *Metaphorical & Rhetorical Insults ($N=5$):* e.g., *"Adı üzerinde lağım medyası ne beklenir ki"* ($p = 0.1716$), *"Merak ediyorum bu kafayla nasıl yaşıyorsunuz"* ($p = 0.2826$).
    3. *Sexist & Bodily Harassment ($N=1$):* e.g., unsolicited sexualized critique ($p = 0.2232$).
    4. *Obfuscated / Translationese Artifacts ($N=3$):* literal idioms like *"ne sığır eti olduğunu bilmiyorum"* (*having beef*) ($p = 0.3359$).
    5. *Dataset Annotation Noise ($N=3$):* benign text falsely labeled toxic in source data (*"tok tutuyo"*, *"Spoiler Veriyorum"*).
  * **False Positives (Spurious Alarms, $N = 15$):**
    1. *Negated Trigger Words ($N=4$):* e.g., *"Hiç birimiz aptal yada saf değiliz..."* ($p = 0.5691$). Lexical models cannot model negation scope across clauses.
    2. *Proverbs & Moral Maxims ($N=4$):* e.g., *"İnsanoğlu naziktir , ağır sözü kaldırmaz"* ($p = 0.7061$). High-arousal vocabulary in benign philosophical statements.
    3. *Sports Banter & Colloquial Slang ($N=3$):* e.g., match schedules and fan rivalry (*"kudursun"*, *"emek hırsızlığı"*).
    4. *Philosophical / Historical Reflections ($N=2$):* e.g., long historical essays discussing genocide or warfare in abstract context ($p = 0.9293$).
    5. *Wikipedia / Editorial Jargon ($N=2$):* e.g., *"3RR kuralı"*, *"çorap kukla"* ($p = 0.5318$).
* **Visual:** Embedded bar chart: `reports/mid_project/error_categories_breakdown.png`
* **Presenter Notes (1:00 min):**
  *"We conducted an exhaustive error analysis on 30 baseline validation misclassifications: exactly 15 false positives and 15 false negatives. The patterns are striking. The linear model fails to catch toxic comments when hostility is expressed through curses like 'kanı kurusun' or metaphors like 'lağım medyası', because they lack explicit vulgar words. Conversely, it triggers false alarms whenever a word like 'aptal' is negated, as in 'Hiç birimiz aptal değiliz', or when proverbs discuss harsh words. This confirms that lexical models are fundamentally blind to syntax, negation, and non-literal intent."*

---

### Slide 7: Deep Contextual Model: BERTurk Fine-Tuning Status

* **Slide Title:** Deep Learning Method: Supervised BERTurk Fine-Tuning
* **Model Checkpoint & Architecture:**
  * Base: `dbmdz/bert-base-turkish-cased` (pinned revision: `b6e1de16c983e0f2c70664591ea3f22810072608`).
  * 12 Transformer layers, 12 attention heads, hidden dim $768$, $110.6\text{M}$ parameters, 32,000 cased WordPiece vocab.
* **Token Length & Dynamic Padding Optimization:**
  * Sequence length analysis across 63,650 training texts:
    * **Native Twitter:** 100% of texts are $\le 127$ tokens (**0.00% truncation at 128**).
    * **Overall Corpus:** 95.58% retained with zero truncation at 128.
  * Implemented `DynamicPaddingCollator` with batch-level padding: ~3.5× faster than static 512 padding without loss of Twitter context.
* **Training Protocol & Hardware Profiling:**
  * Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), PyTorch 2.6.0, CUDA 12.4, mixed precision (`bfloat16`).
  * Peak VRAM: 3,639 MiB (44.4% capacity); compute throughput: ~200 samples/sec.
  * Optimizer: AdamW ($\text{LR} = 2\times 10^{-5}$, weight decay $0.01$, linear warmup 10%, batch size 32).
  * Validation Checkpointing: Best model selected at **Epoch 4** (Validation Macro-F1 = **0.9660**).
  * Refit Protocol: Refitted on combined train+val ($N = 76,381$) for 4 epochs (runtime: 1515.07s).
* **Visual:** Embedded loss/F1 curve: `reports/berturk/training_curves.png`
* **Presenter Notes (1:00 min):**
  *"For our deep contextual model, we fine-tuned BERTurk cased. A key technical contribution was our token-length analysis: we proved that 100% of Native Turkish tweets fit within 128 tokens. Using dynamic batch padding at length 128 reduced compute time by over 70% while allocating under 3.7 GB VRAM on our RTX 4070 GPU. Across 4 training epochs, validation Macro-F1 steadily rose from 0.9485 to 0.9660. Following our pre-registered protocol, we refitted on train plus validation before final test scoring."*

---

### Slide 8: Controlled Model Comparison on Frozen Test Partition

* **Slide Title:** Controlled Test Results: BERTurk vs. TF-IDF Baseline
* **Primary Test Results Table ($N = 8,485$):**

| Evaluation Metric | Baseline (TF-IDF + LogReg) | BERTurk (`bert-base-turkish-cased`) | Difference ($\Delta$) | Practical Threshold ($\Delta_{\min} = 0.020$) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** (Primary) | 0.9393 | **0.9654** | **+0.0261** | **PASSED (+0.0061 margin)** |
| **Accuracy** | 0.9393 | **0.9655** | **+0.0262** | PASSED |
| **Toxic Recall** | 0.9271 | **0.9809** | **+0.0538** | Substantial Gain |
| **Toxic F1** | 0.9400 | **0.9668** | +0.0268 | — |
| **Average Precision (AP)** | **0.9862** | 0.9854 | -0.0008 | Comparable |
| **ROC-AUC** | 0.9851 | **0.9906** | **+0.0055** | — |

* **Paired Bootstrap Hypothesis Testing ($B = 10,000$, Seed 42):**
  * $\Delta = \mathbf{+0.0261}$ (95% CI: $[+0.0207, +0.0317]$, empirical $p = 1 / 10,001 \approx 0.00010$).
  * The entire 95% confidence interval is strictly above $\Delta_{\min} = 0.020$.
* **Source-Sliced Breakdown:**
  * **Native Twitter ($N = 4,916$):** TF-IDF 0.9373 $\to$ BERTurk **0.9620** ($\Delta = \mathbf{+0.0247}$).
  * **Jigsaw Wikipedia ($N = 3,560$):** TF-IDF 0.9397 $\to$ BERTurk **0.9688** ($\Delta = \mathbf{+0.0291}$).
* **Dramatic Reduction in Missed Toxicity:**
  * False Negatives collapsed from **317 to 83** (a **73.8% reduction**), raising Toxic Recall to **98.09%**.
* **Visual:** Embedded figures: `reports/mid_project/model_comparison_bar.png` and `reports/mid_project/confusion_matrices.png`
* **Presenter Notes (1:00 min):**
  *"Here is our headline experimental result on the locked test set. BERTurk outperforms the TF-IDF baseline by 2.61 percentage points, achieving 0.9654 Macro-F1. In our 10,000-iteration paired bootstrap test, the 95% confidence interval is 0.0207 to 0.0317, with an empirical p-value of approximately 0.0001. Because the lower bound exceeds our pre-registered threshold of 0.020, this gain is both statistically and practically significant. Crucially, False Negatives dropped from 317 to 83—a 74% reduction in missed toxic content, observed across both Twitter and Jigsaw slices."*

---

### Slide 9: Methodological Limitations & Threats to Validity

* **Slide Title:** Limitations & What Cannot Yet Be Claimed
* **What We Have NOT Proven (Scientific Boundary):**
  * **Tier 1 measures broad toxicity only:** A Macro-F1 of 0.9654 on Tier 1 **does not prove** that BERTurk detects implicit attacks or sarcasm. Broad toxicity datasets are dominated by explicit cues; high accuracy here is necessary but insufficient for solving implicit hostility.
* **Corpus & Domain Limitations:**
  * **Translationese Contamination:** 45.8% of the corpus is machine-translated from English Jigsaw discussions, introducing unnatural syntactic phrasing and literal idioms (*"sığır eti olmak"*, *"büyük tüylü toplar"*).
  * **Label Noise in Public Sources:** Unreviewed web-scraped datasets contain mislabeled entries (e.g., benign fragments like *"tok tutuyo"* labeled toxic).
* **Technical Constraints:**
  * **Truncation:** While 0% of Twitter was truncated, 4.42% of lengthy Wikipedia threads were truncated at 128 tokens, omitting tail context.
  * **Single-Post Isolation:** Real-world sarcasm frequently relies on conversational context or daily political news that standalone models cannot observe.
* **Presenter Notes (1:00 min):**
  *"We hold scientific honesty paramount. We must emphasize what these results do NOT prove. High performance on Tier 1 does NOT prove that our model detects implicit attacks or sarcasm. Tier 1 is predominantly explicit toxicity. Furthermore, 46% of the data contains machine translation artifacts, and web-scraped labels have noticeable noise. Our research question remains open until we evaluate on the Tier 2 benchmark."*

---

### Slide 10: Plan to the Final Milestone (Roadmap, Risks & Fallbacks)

* **Slide Title:** Roadmap to Final Submission: Tier 2, SHAP & Ablations
* **Three Core Workstreams for the Final Project:**
  1. **Tier 2 Benchmark Annotation (Nov 13 – Nov 24):**
     * Curate a balanced, held-out evaluation set of 300 standalone Turkish posts (implicit attacks, sarcastic mockery, conversational hard negatives).
     * Double annotation by team members with formal annotation guidelines, reporting Cohen’s $\kappa$ agreement before adjudication.
     * Mark token-level attack rationale spans for explainability ground truth.
  2. **SHAP Explainability & Rationale Alignment (Nov 25 – Dec 03):**
     * Implement Partition/Exact SHAP for BERTurk using Turkish background masking.
     * Evaluate token attribution overlap (Token-F1 / AUPRC) against human rationale spans.
     * Benchmark against a random-token attribution baseline to test the proposal's negative-result condition.
  3. **Ablation Studies & Baselines (Dec 01 – Dec 08):**
     * Research Lead extension: Compare cased BERTurk vs. uncased BERTurk vs. DistilBERTurk.
     * Compare SHAP attributions against Integrated Gradients and gradient $\times$ input.
* **Risk Management & Fallbacks:**
  * **Biggest Risk:** Low inter-annotator agreement on ambiguous sarcasm ($\kappa < 0.60$).
  * **Mitigation / Fallback:** Conduct a 50-item calibration pilot with revised guidelines; if necessary, narrow scope to targeted implicit mockery and report separate agreement on unambiguous adjudicated consensus subsets.
* **Presenter Notes (1:00 min):**
  *"Looking ahead to the final report, our work splits into three clear workstreams. First, curating and double-annotating our Tier 2 benchmark of 300 implicit and sarcastic posts with token rationale spans. Second, running SHAP explainability to evaluate whether model highlights genuinely match human evidence compared to a random baseline. Third, our graduate ablation study comparing BERTurk variants and attribution methods. If inter-annotator agreement on sarcasm is low, our fallback is to run a calibration pilot and report consensus agreement. Thank you, and we welcome your questions."*

---

# Part 2: Project Card (12-Field Appendix Slides)

The Project Card represents the comprehensive architectural blueprint for the final research paper. Below is the full text corresponding to the 4 graded appendix slides.

---

### Appendix Slide 1: Fields 1–3 (Title & Abstract, Research Question, Contributions)

#### Field 1: Title & Abstract (Title, Abstract)
* **Working Title:** *Explainable Turkish Implicit Toxicity Detection: Benchmarking BERTurk and Token-Level Rationale Alignment against Lexical Baselines*
* **Draft Abstract ($\le 150$ words):**
  > Automated toxicity detection systems frequently fail on nuanced language, mistaking benign moral maxims for profanity while overlooking subtle, sarcastic hostility. In this work, we investigate whether fine-tuning BERTurk improves the identification of implicit and sarcastic attacks in Turkish social media over a tuned TF-IDF + Logistic Regression baseline, and whether token-level SHAP attributions faithfully reflect human-annotated evidence. On our Tier 1 foundation of 84,866 verified Turkish records, BERTurk achieved 0.9654 Macro-F1 (vs. 0.9393 for TF-IDF, $\Delta = +0.0261$, 95% bootstrap CI: $[0.0207, 0.0317]$, $p \approx 0.0001$), reducing missed toxic comments by 73.8%. On our dedicated, human-annotated Tier 2 benchmark of implicit attacks and hard negatives, BERTurk achieves **[Tier 2 Evaluation Pending — Final Milestone]** Macro-F1, while token-level SHAP explanations yield **[SHAP Alignment Pending — Final Milestone]** overlap with human rationale spans against random-token baselines.

#### Field 2: Research Question & Stability (Introduction)
* **Status:** **UNCHANGED** from original proposal.
* **The Question:**
  *"Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + logistic-regression baseline, and do token-level SHAP explanations identify human-annotated attack cues?"*
* **Negative Result Condition:** The empirical outcome is "no" if the Macro-F1 gain on Tier 2 is statistically indistinguishable from zero ($p \ge 0.05$ or 95% CI includes zero), or if SHAP token rationales fail to surpass a random-token baseline in overlapping annotator evidence spans.
* **Architectural Realignment:** The research question was **not** broadened to general toxicity. Rather, because public Turkish corpora lack implicit attack annotations, we adopted a two-tier strategy where Tier 1 provides broad model development and Tier 2 provides the held-out implicit test set.

#### Field 3: Testable Contributions (Introduction)
1. **Empirical Modeling Claim:** Fine-tuning BERTurk on broad Turkish offensive text yields a statistically and practically significant improvement ($\Delta \ge 0.020$ Macro-F1, $p < 0.001$) over an optimized word+char TF-IDF baseline on broad toxicity, driven by a $>70\%$ reduction in false negatives on syntactically complex and non-profane hostility.
2. **Implicit Transfer Claim (To Be Evaluated on Tier 2):** Broad offensive fine-tuning alone is insufficient to reliably detect implicit and sarcastic attacks, resulting in a statistically significant performance degradation on held-out implicit attacks compared to explicit toxicity.
3. **Faithfulness & Plausibility Claim (To Be Evaluated on Tier 2):** Token-level SHAP explanations extracted from fine-tuned BERTurk will achieve a statistically higher token-level F1 / AUPRC overlap with human-annotated rationale spans than an empirical random baseline, but will frequently exhibit attribution dispersion on idiomatic and negated expressions.

---

### Appendix Slide 2: Fields 4–6 (Related Work, Data, Method & Baseline)

#### Field 4: Related Work (Related Work)
1. **Bilgin & Bekar (2025):** *"Turkish Cyberbullying Detection with Fine-Tuned Pre-Trained Language Models."* Evaluates BERTurk, DistilBERTurk, and ConvBERTurk on 3,388 Turkish tweets across broad cyberbullying classes.  
   *Differentiation:* We evaluate specifically on a held-out benchmark of implicit/sarcastic attacks and evaluate token-level explanation faithfulness against human rationales.
2. **Gökdeniz & Canbay (2025):** *"Detection of Cyberbullying from Turkish Texts with Artificial Intelligence and Natural Language Processing."* Benchmarks classical ML and FastText embeddings on Turkish social media.  
   *Differentiation:* Focuses strictly on surface lexical features and standard machine learning without transformer self-attention or explainability evaluation.
3. **Mahmud, Ptaszynski & Masui (2024):** *"Leveraging Explainable AI and Sarcasm Features for Improved Cyberbullying Detection in Multilingual Settings"* (IEEE Access / DPSH, DOI: 10.1109/DPSH60098.2024.10774626). Explores LIME/SHAP and sarcasm features in low-resource South Asian languages (Bangla/Chittagonian).  
   *Differentiation:* We focus on agglutinative Turkish morphology, evaluate rationale span overlap against human ground truth, and test a pre-registered negative result condition.
4. **Schweter (2020):** *"BERTurk — BERT Models for Turkish"* (Zenodo, DOI: 10.5281/zenodo.3770924). Introduces the pretrained monolingual Turkish BERT base models trained on 35GB of Turkish text.  
   *Differentiation:* We fine-tune and probe the cased model specifically for toxic edge cases, token length truncation boundaries, and SHAP explainability.
5. **Çöltekin (2020):** *"A Corpus of Turkish Offensive Language on Social Media"* (LREC 2020). Curates an offensive Turkish Twitter dataset with sub-target labels.  
   *Differentiation:* Analyzes broad offensive language distributions; our Tier 2 work isolates standalone implicit sarcasm and hard-negative moral maxims.
6. **Lundberg & Lee (2017):** *"A Unified Approach to Interpreting Model Predictions"* (NeurIPS 2017). Introduces Shapley Additive Explanations (SHAP).  
   *Differentiation:* We apply Kernel/Partition SHAP to evaluate rationale alignment against human annotator spans in Turkish text classification.

#### Field 5: Data Architecture & Preprocessing Accounting (Data)
* **Primary Sources & Licenses:**
  * `Overfit-GM/turkish-toxic-language` (Hugging Face, Open Data / MIT equivalent): 77,800 raw rows.
  * Kaggle 6-Class Labeled Turkish Tweets (`smailsariteke`, CC0 Public Domain): 9,999 raw rows.
* **Corpus Accounting Pipeline:**
  * Raw records combined: $87,799$.
  * Excluded pure financial scam/fraud: $501$ records.
  * Internal duplicates & cross-dataset exact/canonical duplicates removed: $2,416$ records.
  * Conflicting canonical pairs quarantined: $16$ records ($32$ rows).
  * **Retained Tier 1 Corpus:** **$84,866$ unique records** (Clean: $41,376$ / $48.75\%$; Toxic: $43,490$ / $51.25\%$).
* **Domain Distribution:** Native Turkish Twitter ($49,158$ rows / $57.9\%$), Machine-Translated Jigsaw Wikipedia Discussions ($35,608$ rows / $41.9\%$), Other sources ($100$ rows / $0.1\%$).
* **Source-Stratified Partitions:**
  * **Train (75.0%):** $63,650$ samples ($31,032$ clean, $32,618$ toxic).
  * **Validation (15.0%):** $12,731$ samples ($6,207$ clean, $6,524$ toxic).
  * **Frozen Test Set (10.0%):** $8,485$ samples ($4,137$ clean, $4,348$ toxic). Cryptographically locked.
* **Example Records:**
  * Clean (Twitter): *"Galatasaray HDI Sigorta Erkek Voleybol Takımımızın Play Off Yarı programı"*
  * Toxic (Twitter): *"Adı üzerinde lağım medyası ne beklenir ki"*
  * Clean with Negated Trigger: *"Hiç birimiz aptal yada saf değiliz,sadece susuyor ve görmezden geliyoruz."*
* **Tier 2 Labeling Protocol (In Progress):**
  * Target: 300 standalone Turkish tweets (100 implicit attacks, 100 sarcastic mockery, 100 conversational hard negatives).
  * Double-blind annotation by team members with written guidelines; target agreement Cohen's $\kappa \ge 0.70$ prior to adjudication; rationale span tagging for every toxic sample.

#### Field 6: Method and Baseline Specification (Method)
* **Baseline as Run:** Combined word n-grams (1–2) + character `char_wb` n-grams (3–5), sublinear TF scaling, $147,749$ sparse features; classified with Logistic Regression ($C = 10.0$, balanced class weights, L-BFGS solver).
* **Deep Method as Implemented:** Pretrained BERTurk (`dbmdz/bert-base-turkish-cased`, 110M params) fine-tuned with a linear classification head ($768 \to 2$ with dropout $p = 0.1$). AdamW optimizer ($\text{LR} = 2.0\times 10^{-5}$, weight decay $0.01$, linear warmup $10\%$, effective batch size 32, max sequence length 128 with dynamic batch padding).
* **Controlled Isolated Change:** Exact same training data ($N = 63,650$), validation tuning set ($N = 12,731$), and frozen test partition ($N = 8,485$); identical binary labels. The only variation is the feature extraction and inductive bias (sparse linear n-grams vs. multi-head contextual self-attention).
* **Compute Used & Remaining:**
  * Tier 1 Baseline & BERTurk: ~1.2 GPU-hours on local NVIDIA GeForce RTX 4070 Laptop GPU (3,639 MiB peak VRAM, mixed precision `bfloat16`).
  * Remaining Compute for Final Milestone: ~3 GPU-hours for SHAP explanation sampling and Tier 2 inference/ablation sweeps.

---

### Appendix Slide 3: Fields 7–9 (Evaluation Protocol, Results So Far, Error Analysis)

#### Field 7: Evaluation Protocol (Experimental Setup)
* **Primary Metric:** Macro-averaged F1 score across Clean and Toxic classes ($\text{Macro-F1} = \frac{F1_{\text{clean}} + F1_{\text{toxic}}}{2}$).
* **Secondary Metrics:** Accuracy, Class-specific Precision and Recall, Average Precision (AP, via `average_precision_score`), and ROC-AUC.
* **Frozen Test Set Protocol:** Test partition locked prior to model exploration; zero hyperparameter tuning, prompt tuning, or threshold selection conducted on test data.
* **Pre-Registered Practical Significance Threshold:** $\Delta_{\min} = 0.020$ (+2.0 percentage points in Macro-F1).
* **Statistical Uncertainty Protocol:** Non-parametric paired bootstrap resampling ($B = 10,000$ replicates, random seed 42) computing empirical 95% confidence intervals on $\Delta = \text{Macro-F1}_{\text{BERTurk}} - \text{Macro-F1}_{\text{TF-IDF}}$. Empirical $p$-value reported using Laplace add-one correction ($p = \frac{k+1}{B+1}$).

#### Field 8: Results So Far (Results)

| Model / System | Evaluation Split | Macro-F1 (95% Spread / CI) | Accuracy | Toxic Recall | Average Precision (AP) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogReg ($C=10.0$)** | Tier 1 Frozen Test ($N=8,485$) | $0.9393$ $[0.9341, 0.9443]$ | $0.9393$ | $0.9271$ | $0.9862$ | $0.9851$ | **Completed Baseline** |
| **BERTurk Fine-Tuned** | Tier 1 Frozen Test ($N=8,485$) | **$0.9654$** $[0.9615, 0.9691]$ | **$0.9655$** | **$0.9809$** | $0.9854$ | **$0.9906$** | **Completed Method** |
| **Model Difference ($\Delta$)** | Tier 1 Frozen Test ($N=8,485$) | **$+0.0261$** $[+0.0207, +0.0317]$ | $+0.0262$ | $+0.0538$ | $-0.0008$ | $+0.0055$ | **Stat. & Pract. Significant** ($p \approx 0.0001$) |
| **TF-IDF Baseline** | Tier 2 Benchmark ($N \approx 300$) | *[Not Yet Evaluated]* | — | — | — | — | *Final Milestone* |
| **BERTurk Fine-Tuned** | Tier 2 Benchmark ($N \approx 300$) | *[Not Yet Evaluated]* | — | — | — | — | *Final Milestone* |
| **SHAP Rationale Alignment** | Tier 2 Benchmark ($N \approx 300$) | *[Not Yet Evaluated]* | — | — | — | — | *Final Milestone* |

*Note: In accordance with rubric guidelines, Tier 2 rows are explicitly designated as pending evaluation.*

#### Field 9: Baseline Error Analysis Taxonomy (Error Analysis)
Exhaustive categorization of 30 baseline validation misclassifications (`reports/baseline/validation_errors_30.csv`):

| Error Mode | Error Category | Count | Representative Text Example | Why the Model Failed |
| :--- | :--- | :---: | :--- | :--- |
| **False Negative** | Metaphorical / Rhetorical Mockery | 5 | *"Adı üzerinde lağım medyası ne beklenir ki"* | Metaphors lack explicit swear words; low n-gram weight. |
| **False Negative** | Implicit Curses & Ill-Wishes | 3 | *"Değil unutmak acısı hafifleyenin kanı kurusun"* | Conditionally framed hostility (*"kanı kurusun"*) lacks vulgar tokens. |
| **False Negative** | Translationese Obfuscation | 3 | *"Bu adamla ne sığır eti olduğunu bilmiyorum..."* | Machine-translated English idiom (*"having beef"*) diluted feature weights. |
| **False Negative** | Ambiguous Dataset Noise | 3 | *"tok tutuyo"* / *"Spoiler Veriyorum"* | Scraped dataset labeling noise; non-toxic fragments annotated as toxic. |
| **False Negative** | Sexist Bodily Harassment | 1 | *"Arkadaş anladık memeniz var çantanın kayışını..."* | Harassing sexualized commentary framed colloquially. |
| **False Positive** | Negated Trigger Words | 4 | *"Hiç birimiz aptal yada saf değiliz..."* | Word *"aptal"* triggered toxic prediction; n-grams ignore negation scope. |
| **False Positive** | Proverbs & Moral Maxims | 4 | *"İnsanoğlu naziktir , ağır sözü kaldırmaz"* | Severe moral tone and high-arousal words (*"ağır söz"*) triggered false alarm. |
| **False Positive** | Sports Banter & Rivalry | 3 | *"Galatasaray HDI Sigorta Voleybol programı"* / *"kudursun"* | Substring n-grams matched sports flame wars; clean match schedule flagged. |
| **False Positive** | Philosophical / Historical Essays | 2 | Long essay analyzing Cold War and genocide in abstract | Essay mentioned *"soykırım"*, *"pis aptal"* in historical context, not personal attack. |
| **False Positive** | Technical / Wikipedia Jargon | 2 | *"3RR yi ruh olarak ihlal etmedim... çorap kukla"* | Wikipedia moderation dispute terms (*"çorap kukla"*) inflated toxic probability. |

---

### Appendix Slide 4: Fields 10–12 (Research Lead Section, Plan & Risk, Reproducibility)

#### Field 10: Research Lead Section (Related Work, Ablation)
* **Author:** **Ahmet Furkan Hazar** (CSE 654 Graduate Student, Research Lead)
* **Critical Literature Analysis:**
  1. *Bilgin & Bekar (2025):* Demonstrated that BERTurk out-performs smaller distilled variants on Turkish offensive tweets ($N = 3,388$, Macro-F1 $\approx 0.88$). However, their corpus exclusively captures explicit abusive lexicon and offers no explainability checks. Our work differs by isolating implicit sarcasm, introducing an adversarial hard-negative test tier, and evaluating explanation faithfulness against marked token spans.
  2. *Mahmud, Ptaszynski & Masui (2024):* Showed that integrating sarcasm features and XAI (SHAP) improved cyberbullying detection in low-resource South Asian languages (Bangla). However, they treated SHAP as an interpretive visual aid rather than evaluating token-level overlap against ground-truth human rationale spans. Our project extends this by formally testing whether token-level SHAP heatmaps surpass a random-token baseline.
* **Planned Graduate Ablation & Comparison:**
  * **Architecture Ablation:** Compare `bert-base-turkish-cased` vs. `bert-base-turkish-uncased` vs. `distilbert-base-turkish-cased` to isolate the impact of Turkish morphological casing and depth on sarcasm detection.
  * **Attribution Method Ablation:** Compare Partition SHAP against Integrated Gradients and Input $\times$ Gradient on human rationale alignment.

#### Field 11: Plan, Risks & Response to Feedback (Limitations)
* **Execution Roadmap (Target Dates & Confirmed Owners):**
  * **Nov 13 – Nov 20:** Finalize Tier 2 annotation guidelines and complete 50-item calibration pilot.  
    *Owner: Ahmet Furkan Hazar (Lead) & Orhun Gökdemir*
  * **Nov 21 – Nov 27:** Complete double annotation of 300 Tier 2 benchmark samples (implicit attacks, sarcasm, hard negatives) and adjudicate consensus. Calculate Cohen’s $\kappa$.  
    *Owner: All team members (Hazar, Çakır, Gökdemir, Kurtaran)*
  * **Nov 28 – Dec 04:** Implement Partition SHAP pipeline with Turkish masking; generate token heatmaps across Tier 2.  
    *Owner: Ahmet Harun Çakır & Burak Kurtaran*
  * **Dec 05 – Dec 10:** Run token rationale overlap evaluation (Token-F1 / AUPRC vs. random baseline); run graduate model/XAI ablations.  
    *Owner: Ahmet Furkan Hazar & Ahmet Harun Çakır*
  * **Dec 11 – Dec 18:** Final report authoring (ACM / ACL short paper format), reproducibility verification, and submission.  
    *Owner: All team members*
* **Biggest Risk & Mitigation Fallback:**
  * *Risk:* Low inter-annotator agreement ($\kappa < 0.60$) on subtle sarcasm and contextual hard negatives.
  * *Fallback Strategy:* Pre-schedule a calibration session after the first 50 pilot items to refine operational boundaries. If subjective disagreement persists on sarcasm, report separate agreement on unambiguous consensus subsets and explicitly document borderline disagreement as linguistic ambiguity in the limitations section.
* **Response to Proposal Feedback:** Addressed concerns regarding public dataset validity by explicitly introducing the two-tier architecture, isolating scam records, and pre-registering strict bootstrap criteria ($\Delta_{\min} = 0.020$).

#### Field 12: Reproducibility, Team Roles & AI Disclosure (Appendix)
* **Repository:** Private Course Repository (Gebze Technical University CSE 484/654 Team 07).
* **Deterministic Environment:** Python 3.11.18, `uv` package manager (`uv.lock` pinned), PyTorch 2.6.0 with CUDA 12.4, Transformers 4.49.0, scikit-learn 1.6.1. Random Seed = `42`.
* **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM).
* **Exact One-Command Reproduction:**
  ```bash
  # 1. Reproduce deterministic data preparation and verify cryptographic hashes
  uv run python scripts/run_prepare_data.py

  # 2. Reproduce TF-IDF baseline training, evaluation, and error extraction
  uv run python scripts/run_tfidf_baseline.py

  # 3. Reproduce BERTurk test evaluation, bootstrap testing, and comparison reports
  uv run python scripts/run_berturk.py --eval-only

  # 4. Run automated test suite (34 tests)
  uv run pytest -v
  ```
* **Team Roles & Responsibilities:**
  * **Ahmet Furkan Hazar (Research Lead):** Proposal conceptualization, experimental design, literature review, Tier 2 annotation guidelines, graduate ablation design.
  * **Ahmet Harun Çakır:** Data pipeline implementation, text canonicalization, SHAP explainability implementation, test suite engineering.
  * **Orhun Gökdemir:** Raw dataset acquisition, inventory auditing, Kaggle schema reconciliation, Tier 2 candidate mining.
  * **Burak Kurtaran:** Baseline modeling, BERTurk dynamic batch collator, hyperparameter tuning, presentation deck formatting.
* **AI Assistance Transparency Disclosure:**
  * *ChatGPT (OpenAI):* Used for literature discovery, initial prompt formulation, and critical peer review of the proposal.
  * *Antigravity (Google DeepMind):* Used as an interactive pair-programming assistant for implementing data pipeline scripts, PyTorch dynamic padding collators, automated test suites, and formatting document artifacts.
  * *Integrity Statement:* All experimental numbers, metrics, error categorizations, and analytical claims were computed directly on host hardware and verified against underlying machine-readable artifacts. No AI tool was permitted to fabricate experimental numbers or simulate human annotations.

---

## Conclusion & Mid-Project Readiness

Team 07 has established a fully reproducible, verified, and statistically robust Tier 1 foundation. The classical baseline and deep BERTurk transformer models are locked, evaluated, and documented. The project is strictly on schedule to execute the Tier 2 human annotation and SHAP explainability evaluation for the final project submission.
