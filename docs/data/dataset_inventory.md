# Phase 3.1 — Dataset Acquisition & Exploratory Data Audit Report

**Project:** Explainable Turkish Implicit Toxicity Detection  
**Team:** Team 07 – TOXICITY  
**Date:** November 10, 2026 (Local Time: 23:55)  
**Role:** Data Engineering Lead  
**Status:** Datasets Acquired, Profiled, and Verified (Pre-deduplication / Pre-merge)  

---

## 1. Executive Summary

In this phase, we acquired and independently audited the two primary datasets specified in the project proposal:
1. **`Overfit-GM/turkish-toxic-language`** (Hugging Face)
2. **`smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi`** (Kaggle)

Both datasets were successfully downloaded in their original formats into `data/raw/` without modifying dependencies, altering source records, or performing early splitting or merging.

### Key Empirical Findings:
1. **Actual Raw Sample Counts:**
   - Overfit-GM: Exactly **77,800 records** (CSV format, 18.2 MB).
   - Kaggle Tweets: Exactly **9,999 records** (Excel format, 593 KB).
   - Combined Raw Pool: **87,799 rows** (virtually matching the proposal's theoretical $77,800 + 10,000 = 87,800$).
2. **Critical Deduplication Discovery:**
   - We discovered **2,088 exact tweet text collisions** between the Kaggle dataset and Overfit-GM.
   - Kaggle also contains **63 duplicate tweets** internally.
   - Consequently, the **actual unique post-deduplication corpus size is ~85,648 tweets**, *not 87,800*. The proposal reported 87,800 by simply adding the raw counts without running cross-source deduplication.
3. **Data Provenance & Annotation Quality:**
   - **Overfit-GM:** 35,624 of its 77,800 records (45.8%) originate from translated Jigsaw Multilingual Toxic Comments, and the dataset documentation explicitly notes that **transformer models were used to augment pseudo-labels**.
   - **Kaggle Tweets:** Comprises 9,999 native Turkish tweets labeled with multi-label binary indicators (`Küfür`, `Tehdit`, `Hakaret`, `Dolandırıcılık`, `Cinsel İçerik`, `Irkçılık`) alongside a general flag (`Tespit`).
4. **Label Compatibility:**
   - On the 2,088 overlapping tweets, the binary alignment between Overfit-GM's `is_toxic` and Kaggle's toxic indicators is near-perfect ($100\%$ agreement for clean/OTHER texts).

---

## 2. Dataset Provenance & Ingestion Details

| Attribute | Dataset 1: Overfit-GM | Dataset 2: Kaggle Tweets |
| :--- | :--- | :--- |
| **Identifier** | `Overfit-GM/turkish-toxic-language` | `smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi` |
| **Platform** | Hugging Face Datasets Hub | Kaggle Datasets |
| **Source URL** | [Hugging Face Link](https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language) | [Kaggle Link](https://www.kaggle.com/datasets/smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi) |
| **Author / Curator**| Overfit-GM | İsmail Sarıteke |
| **License** | **Apache 2.0** (Open Source / Permissive) | **ODbL / Open Database License** (Database Open, Contents Protected) |
| **Download Timestamp** | `2026-11-10T23:48:32+03:00` | `2026-11-10T23:48:53+03:00` |
| **Acquisition Tool** | `huggingface_hub.snapshot_download` | `kagglehub.dataset_download` (Version 1) |
| **Authentication** | Public download (no token required) | Public download (no Kaggle credentials required) |
| **Destination Directory** | `data/raw/overfit_gm/` | `data/raw/kaggle_tweets/` |

---

## 3. Raw File Inventory & Cryptographic Checksums

All raw files are stored in `data/raw/` and excluded from git tracking via `.gitignore`.

| Relative Path | Format | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :---: | :--- |
| `data/raw/overfit_gm/turkish_toxic_language.csv` | CSV | 18,223,058 | `436aa9250d21d0617cb0aafdf77cad9c1aa85642d2b3fd1cd9d1aedab2d3fbc6` |
| `data/raw/overfit_gm/README.md` | Markdown | 1,979 | `481c9a6335bf3dfb676fef3a31c19b48b990f058097bc876378e04052cfdf7ec` |
| `data/raw/overfit_gm/.gitattributes` | Text | 2,328 | `d0a649646b149b1ca2996d9e03d4dfffa72807e3a8fdb4ae1f238b1d9bf5b210` |
| `data/raw/kaggle_tweets/Veri_Seti_10000.xlsx` | XLSX | 593,140 | `b713c8efc748134c417d250bb7c27bd4ab652abb1955af2495108ccfdb41c22a` |

---

## 4. Dataset Schemas & Profiling

### 4.1 Dataset 1: `Overfit-GM/turkish-toxic-language`

#### Schema Description
- **Record Count:** Exactly 77,800 rows.
- **Columns (4):**
  1. `text` (*string*): The comment or tweet text.
  2. `target` (*string*): Multi-class category label (`OTHER`, `PROFANITY`, `INSULT`, `RACIST`, `SEXIST`).
  3. `source` (*string*): Originating source corpus (`told`, `jigsaw`, `bully`, `troff`, `offenseval`).
  4. `is_toxic` (*int64*): Binary toxicity indicator ($1 = \text{toxic}$, $0 = \text{non-toxic}$).

#### Target Class Distribution
| Category (`target`) | Count | Proportion | Binary Label (`is_toxic`) |
| :--- | :---: | :---: | :---: |
| `OTHER` | 37,663 | 48.41% | 0 (Non-toxic) |
| `PROFANITY` | 18,252 | 23.46% | 1 (Toxic) |
| `INSULT` | 10,777 | 13.85% | 1 (Toxic) |
| `RACIST` | 10,163 | 13.06% | 1 (Toxic) |
| `SEXIST` | 945 | 1.21% | 1 (Toxic) |
| **Total** | **77,800** | **100.0%** | **40,137 (Toxic) / 37,663 (Non-toxic)** |

#### Source Corpus Distribution
| Source Tag | Originating Dataset | Count | Proportion |
| :--- | :--- | :---: | :---: |
| `told` | Turkish Offensive Language Dataset | 39,551 | 50.84% |
| `jigsaw` | Jigsaw Multilingual Toxic Comments (translated) | 35,624 | 45.79% |
| `bully` | Turkish Cyberbullying Dataset | 2,525 | 3.25% |
| `troff` | Turkish Offensive Tweets (Collab) | 94 | 0.12% |
| `offenseval` | OffensEval 2020 Turkish Subtask | 6 | 0.01% |

#### Data Quality & Text Metrics
- **Missing Values:** 0 null values across all columns.
- **Duplicate Rows:** 0 exact duplicate rows; 0 duplicate texts.
- **Empty / Whitespace Texts:** 0.
- **Character Length:** Min: 2, Median: 103, Mean: 200.1, Max: 9,314.
- **Word Count:** Min: 1, Median: 15, Mean: 28.4, Max: 1,215.
- **Turkish Character Presence:** 93.8% of records contain Turkish-specific Unicode characters (`ç`, `ğ`, `ı`, `ö`, `ş`, `ü`, `Ç`, `Ğ`, `İ`, `Ö`, `Ş`, `Ü`).

---

### 4.2 Dataset 2: `smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi`

#### Schema Description
- **Record Count:** Exactly 9,999 data rows (1 header row in sheet).
- **Columns (8):**
  1. `Paylaşım` (*string*): Native Turkish tweet text.
  2. `Tespit` (*float*): General detection flag ($0.0 = \text{Flagged/Toxic}$, $1.0 = \text{Clean/No toxic detection}$).
  3. `Küfür` (*float*): Profanity indicator ($1.0 / 0.0$).
  4. `Tehdit` (*float*): Threat indicator ($1.0 / 0.0$).
  5. `Hakaret` (*float*): Insult indicator ($1.0 / 0.0$).
  6. `Dolandırıcılık` (*float*): Fraud / Scam indicator ($1.0 / 0.0$).
  7. `Cinsel İçerik` (*float*): Sexual Content indicator ($1.0 / 0.0$).
  8. `Irkçılık` (*float*): Racism indicator ($1.0 / 0.0$).

#### Toxicity Flag Distributions
| Column | Positive ($1.0$) Count | Negative ($0.0$) Count | Positive Rate |
| :--- | :---: | :---: | :---: |
| `Tespit` (Clean Flag) | 4,731 | 5,268 | 47.31% Clean |
| `Hakaret` (Insult) | 2,357 | 7,642 | 23.57% |
| `Küfür` (Profanity) | 2,015 | 7,984 | 20.15% |
| `Cinsel İçerik` (Sexual Content)| 798 | 9,201 | 7.98% |
| `Irkçılık` (Racism) | 647 | 9,352 | 6.47% |
| `Dolandırıcılık` (Scam/Fraud) | 501 | 9,498 | 5.01% |
| `Tehdit` (Threat) | 339 | 9,660 | 3.39% |

#### Multi-Label Structure in Kaggle
Tweets frequently carry multiple positive indicators:
- **0 flags positive:** 5,069 tweets (50.7%)
- **1 flag positive:** 3,378 tweets (33.8%)
- **2 flags positive:** 1,396 tweets (14.0%)
- **3 flags positive:** 138 tweets (1.4%)
- **4 flags positive:** 17 tweets (0.17%)
- **5 flags positive:** 1 tweet (0.01%)
- **Composite Toxic Count ($\ge 1$ toxic flag):** Exactly **4,930 tweets** (49.3%).

#### Data Quality & Text Metrics
- **Missing Values:** 0 null values across all columns.
- **Duplicate Rows:** 54 exact duplicate rows.
- **Duplicate Texts (`Paylaşım`):** 63 duplicate texts.
- **Empty / Whitespace Texts:** 0.
- **Character Length:** Min: 3, Median: 64, Mean: 75.0, Max: 508.
- **Word Count:** Min: 1, Median: 9, Mean: 10.6, Max: 68.
- **Turkish Character Presence:** 91.9% of records contain Turkish-specific Unicode characters.

---

## 5. Cross-Dataset Overlap & Compatibility Analysis

### 5.1 Text Collision Analysis
When joining the two datasets on raw text strings:
- **Exact Text Collisions:** **2,088 tweets** appear in both datasets.
- **Normalized Text Collisions:** Exactly **2,088 tweets** (lowercasing and stripping whitespace does not uncover additional collisions).

### 5.2 Why Do Collisions Exist?
Overfit-GM merged several public Turkish NLP repositories (specifically `told` and `bully`). İsmail Sarıteke's Kaggle dataset was compiled from the same foundational Turkish Twitter crawl. This explains why over 20% of the Kaggle dataset is already present inside Overfit-GM.

### 5.3 Semantic Label Agreement on Overlapping Records
We analyzed how the 2,088 overlapping tweets are classified by both annotation regimes:

```text
Overfit-GM Target vs. Kaggle Clean Flag (Tespit) on 2,088 Overlapping Records:
           Tespit=0.0 (Flagged)   Tespit=1.0 (Clean)
OTHER               0                     893
INSULT            556                       1
PROFANITY         575                       0
RACIST             37                       0
SEXIST             35                       0
```

#### Key Findings from Agreement Analysis:
1. **100% Agreement on Clean/Harmless Content:** For all 893 tweets labeled `OTHER` in Overfit-GM, Kaggle labels them `Tespit = 1.0` (Clean) and assigns $0.0$ to all offensive flags.
2. **Near-Perfect Toxic Agreement:** For 1,203 out of 1,204 toxic tweets in Overfit-GM, Kaggle flags them with `Tespit = 0.0`. Only a single record (`INSULT`) differs.
3. **Subcategory Nuance:** Overfit-GM forces single-label categorization, whereas Kaggle uses multi-label flags. For instance, out of 575 `PROFANITY` records in Overfit-GM, Kaggle flags 453 as `Küfür`, but also co-labels 142 of them as `Hakaret` (Insult).

---

## 6. Representative Examples (Preserving Turkish Unicode)

### 6.1 Overfit-GM Dataset
- **`INSULT` (Source: jigsaw):**
  > *"Phil Spector bir lanet katil olduğunu Biliyorsun ve ben de tıpkı OJ gibi Onun oğlu falan mısın..."*
- **`PROFANITY` (Source: told):**
  > *"Lan siktirin gidin AMK pozitif sik kafaları Ülkenin anası dikildi millet açlıktan sınanıyor siz daha slak salak herşey y..."*
- **`OTHER` (Source: jigsaw):**
  > *"2pac Olmak İstiyorum Ja Rule bir Tupac Shakur olmak istiyor Çoğu zaman rap yapıyor ve Shakur gibi davranıyor Şarkı söyle..."*
- **`RACIST` (Source: told):**
  > *"Sana ne koduğumun fransızı..."*
- **`SEXIST` (Source: told):**
  > *"Kadının kafası küçültülmüş yoksa at suratlıydı bu Şu Sarkozy nin karısı çok güzel kadındı ya Aq evladı bak sinirlendirdi..."*

### 6.2 Kaggle Tweets Dataset
- **`Hakaret=1, Küfür=1` (`Tespit=0.0`):**
  > *"ibrahimin oğlunu koruyan Tanrı’ya da ben sokayım"*
- **`Tehdit=1` (`Tespit=0.0`):**
  > *"Defol git lan, ölüye saygın yok anladık bari Allah'tan korkun olsun ama o da yok belli!"*
- **`Dolandırıcılık=1` (`Tespit=0.0`):**
  > *"Ben söğüşledim, birazda sen söğüşle"*
- **`Cinsel İçerik=1` (`Tespit=0.0`):**
  > *"aşırı bıyıklı geleneksel keko adam, bi ayrılıp bi barışan boş bakışlı salak çift veya direnaj ertuğrul 18. sezon dışında..."*
- **`Irkçılık=1` (`Tespit=0.0`):**
  > *"Ulan yanan alevi yakan alevi bugün temel ile aynı çanaktan yal yiyorsun o ne olacak"*
- **`Clean / Harmless` (`Tespit=1.0`, all flags $0.0$):**
  > *"Anne bir sanatçıdır, en güzel eseri de yavrusudur."*

---

## 7. Proposed Label Mapping & Correspondence Table

The two datasets use fundamentally different annotation schemas:
- **Overfit-GM:** Single-label multi-class (`target`: 5 classes) + binary (`is_toxic`: 0/1).
- **Kaggle:** Multi-label binary flags (6 categories) + inverted clean detection flag (`Tespit`).

| Source Dataset | Original Label / Field | Value | Proposed Interpretation | Potential Combined Mapping | Confidence | Unresolved Questions |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Overfit-GM** | `target` | `OTHER` | Neutral / Non-toxic comment | **0 (Non-Toxic / Harmless)** | **High (1.0)** | Contains translated Wikipedia talk pages which differ in style from Twitter. |
| **Overfit-GM** | `target` | `PROFANITY` | Vulgarity, swearing without target | **1 (Toxic / Broad Offensive)** | **High (0.95)**| Does generic profanity transfer to implicit targeted attack detection? |
| **Overfit-GM** | `target` | `INSULT` | Direct derogatory remark / attack | **1 (Toxic / Broad Offensive)** | **High (0.95)**| May contain explicit slurs rather than implicit mockery. |
| **Overfit-GM** | `target` | `RACIST` | Hate speech / xenophobic content | **1 (Toxic / Broad Offensive)** | **High (0.95)**| Relatively small sample in native tweets ($13\%$). |
| **Overfit-GM** | `target` | `SEXIST` | Misogynistic / sexist slur | **1 (Toxic / Broad Offensive)** | **High (0.90)**| Underrepresented ($1.2\%$). |
| **Kaggle** | `Tespit` | `1.0` | Clean / Non-offensive tweet | **0 (Non-Toxic / Harmless)** | **High (1.0)** | Verified 100% agreement with Overfit-GM `OTHER`. |
| **Kaggle** | `Küfür` | `1.0` | Profanity / Swearing | **1 (Toxic / Broad Offensive)** | **High (0.95)**| Frequently co-occurs with `Hakaret`. |
| **Kaggle** | `Hakaret` | `1.0` | Insult / Derogatory tweet | **1 (Toxic / Broad Offensive)** | **High (0.95)**| Primary positive pool for targeted attacks. |
| **Kaggle** | `Irkçılık` | `1.0` | Racist tweet / ethnic attack | **1 (Toxic / Broad Offensive)** | **High (0.95)**| Clean subset of targeted identity attacks. |
| **Kaggle** | `Cinsel İçerik` | `1.0` | Explicit sexual remark / slur | **1 (Toxic / Broad Offensive)** | **Medium (0.80)**| Some instances describe spam or adult content rather than interpersonal attacks. |
| **Kaggle** | `Tehdit` | `1.0` | Violent threat | **1 (Toxic / Broad Offensive)** | **Medium (0.85)**| Low frequency ($N=339$). No direct equivalent in Overfit-GM classes. |
| **Kaggle** | `Dolandırıcılık` | `1.0` | Financial fraud / Crypto scam | **Exclude or Map to 0** | **Low (0.40)** | Financial spam is **not** interpersonal toxicity. Including it harms model focus. |

---

## 8. Critical Research Design Constraints & Feasibility Analysis

### 8.1 The Core Research Dilemma
Our research question asks:
> *Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve Macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + Logistic Regression baseline?*

The exploratory data audit highlights three fundamental structural realities:
1. **Absence of Native Implicit/Sarcastic Annotations:** Neither Overfit-GM nor the Kaggle dataset contains an explicit tag for `implicit_attack`, `sarcasm`, or `hard_negative`. They capture **generic offensiveness, swearing, and overt insults**.
2. **Risk of Pseudo-Labels and Translation Artifacts:** Overfit-GM relies heavily ($45.8\%$) on translated Jigsaw comments and model-generated pseudo-labels. In translated English text, Turkish sarcasm and cultural implicit attacks are nonexistent.
3. **The `Dolandırıcılık` (Scam) Category:** Kaggle includes 501 scam/fraud tweets. Financial spam is distinct from linguistic toxicity and toxic attacks; treating crypto spam as toxicity introduces severe domain noise.

### 8.2 Feasibility of the Two-Tier Evaluation Architecture
The proposed two-tier architecture is not only feasible, but **scientifically mandatory**:
- **Tier 1 (Broad Toxicity Pretraining / Distant Supervision):**
  The combined public corpus ($\approx 85,648$ deduplicated tweets) serves as the broad offensive language training ground. A binary classifier trained here learns lexical, syntactic, and contextual representations of Turkish hostility and negativity.
- **Tier 2 (Frozen Gold Benchmark for Implicit Toxicity & XAI):**
  A separate, curated, manually annotated evaluation set containing:
  - **Implicit targeted attacks** (negative sentiment without overt profanity).
  - **Hard negatives** (sarcastic, humorous, or profane content that is not a targeted attack).
  - **Token-level rationale spans** for SHAP alignment.

---

## 9. Recommended Next Steps for Research Lead Approval

Before any code modification, merging, or splitting is performed:
1. **Approve Deduplication Accounting:** Acknowledge that cross-source deduplication will yield an actual corpus of $\approx 85,648$ unique instances (rather than $87,800$), and update presentation numbers accordingly.
2. **Approve Exclusion of `Dolandırıcılık` (Scam):** Formally approve filtering out financial scam tweets from the toxic category.
3. **Approve Binary Mapping Schema:** Formally ratify mapping `OTHER` / `Tespit=1.0` to `0 (Non-Toxic)` and `{PROFANITY, INSULT, RACIST, SEXIST}` / `{Küfür, Hakaret, Irkçılık, Cinsel İçerik, Tehdit}` to `1 (Toxic)`.
4. **Approve Stratified Split Execution:** Once approved, execute exact-text deduplication, text normalization, and 75/15/10 stratified partitioning into `data/processed/{train,val,test}.parquet`.

---
*Report prepared and validated by Data Engineering Lead for Team 07 – TOXICITY.*
