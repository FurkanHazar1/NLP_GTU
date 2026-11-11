#!/usr/bin/env python3
"""Generate the complete, publication-grade Team 07 Mid-Project presentation deck."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]  # Blank slide

    # Colors
    NAVY = RGBColor(27, 54, 93)      # #1B365D Primary Brand
    TECH_BLUE = RGBColor(41, 128, 185) # #2980B9 Accent Blue
    DARK_TEXT = RGBColor(44, 62, 80)  # #2C3E50 Body Text
    MUTED_GRAY = RGBColor(127, 140, 141) # #7F8C8D Subtitle / Caption
    LIGHT_BG = RGBColor(248, 249, 250) # #F8F9FA Background
    WHITE = RGBColor(255, 255, 255)
    ALERT_RED = RGBColor(192, 57, 43)  # #C0392B Error / Negative
    SUCCESS_GREEN = RGBColor(39, 174, 96) # #27AE60 Success
    CARD_BG = RGBColor(255, 255, 255)
    CARD_BORDER = RGBColor(218, 225, 231)
    ACCENT_BG = RGBColor(235, 245, 251) # Light blue card fill
    WARN_BG = RGBColor(253, 237, 236)   # Light red card fill

    def add_header(slide, title_text, category_text="TEAM 07 — TOXICITY · CSE 484 / 654 NLP MID-PROJECT"):
        """Add standardized top header with category banner and slide title."""
        # Top banner background
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = NAVY
        top_bar.line.fill.background()

        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.733), Inches(0.28))
        tf = cat_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = category_text.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TECH_BLUE

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(11.733), Inches(0.55))
        tf2 = title_box.text_frame
        tf2.word_wrap = True
        tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = WHITE

        # Bottom subtle footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(11.733), Inches(0.3))
        tff = footer_box.text_frame
        tff.margin_left = tff.margin_top = tff.margin_right = tff.margin_bottom = 0
        pf = tff.paragraphs[0]
        pf.text = "Gebze Technical University · Computer Engineering Dept. | Team 07: Hazar, Çakır, Gökdemir, Kurtaran"
        pf.font.size = Pt(9)
        pf.font.color.rgb = MUTED_GRAY

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        """Create a rounded card container for grouping content."""
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title & Research Question & Negative Result Condition
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_header(s1, "Explainable Turkish Implicit Toxicity Detection", "TEAM 07 — TOXICITY · CSE 484 / 654 NLP MID-PROJECT TALK (10 MIN)")

    # Left Column: Team info & Research Question Card
    add_card(s1, Inches(0.8), Inches(1.3), Inches(11.733), Inches(1.4), ACCENT_BG, TECH_BLUE)
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "PRIMARY RESEARCH QUESTION (UNCHANGED FROM PROPOSAL):"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    
    p2 = tf.add_paragraph()
    p2.text = "“Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + logistic-regression baseline, and do token-level SHAP explanations identify human-annotated attack cues?”"
    p2.font.size = Pt(13)
    p2.font.bold = True
    p2.font.color.rgb = NAVY

    # Middle Left Card: Negative Result Conditions
    add_card(s1, Inches(0.8), Inches(2.9), Inches(5.7), Inches(2.7), WARN_BG, ALERT_RED)
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(3.05), Inches(5.3), Inches(2.4))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FORMAL NEGATIVE RESULT (“NO”) CONDITIONS:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED

    bullets_no = [
        ("Condition 1 (Classification Gate):", "Fine-tuned BERTurk fails to achieve a statistically significant Macro-F1 improvement over the baseline (p ≥ 0.05 or 95% bootstrap CI includes zero)."),
        ("Condition 2 (Explainability Gate):", "Token-level SHAP explanation highlights fail to overlap annotators' evidence better than a random-token baseline (measured via token AUPRC / F1)."),
        ("Scientific Integrity Rule:", "If either failure condition occurs on the held-out benchmark, we report a formal negative result without pivoting.")
    ]
    for b_title, b_desc in bullets_no:
        p = tf.add_paragraph()
        p.text = f"• {b_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_TEXT
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.size = Pt(9.5)

    # Middle Right Card: Scope Notice & Team Roles
    add_card(s1, Inches(6.8), Inches(2.9), Inches(5.733), Inches(2.7), CARD_BG, CARD_BORDER)
    tbox = s1.shapes.add_textbox(Inches(7.0), Inches(3.05), Inches(5.333), Inches(2.4))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "MILESTONE SCOPE NOTICE & TEAM ROLES:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY

    notes = [
        ("Current Milestone (Tier 1 Complete):", "Broad binary toxicity foundation (84,866 samples). Demonstrates preliminary model development and baseline establishment; does NOT answer the implicit research question."),
        ("Final Milestone (Tier 2 Pending):", "Double-annotated implicit/sarcastic benchmark with rationale spans + SHAP token alignment evaluation."),
        ("Team 07 Members:", "Ahmet Furkan Hazar (Research Lead, CSE 654), Ahmet Harun Çakır (CSE 654), Orhun Gökdemir (CSE 654), Burak Kurtaran (CSE 484).")
    ]
    for n_title, n_desc in notes:
        p = tf.add_paragraph()
        p.text = f"• {n_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_TEXT
        run = p.add_run()
        run.text = n_desc
        run.font.bold = False
        run.font.size = Pt(9.5)

    # Bottom Banner: Core Message
    add_card(s1, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.1), CARD_BG, TECH_BLUE)
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.333), Inches(0.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TODAY'S 10-MINUTE TALK OVERVIEW:"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "1. Data Reconnaissance & Two-Tier Strategy  →  2. Preprocessing & Cryptographic Locking  →  3. TF-IDF Baseline (0.9393 Macro-F1)  →  4. 30 Qualitative Error Analysis  →  5. Fine-Tuned BERTurk (0.9654 Macro-F1, +0.0261 Δ, p≈0.0001)  →  6. Roadmap to Final Milestone (Tier 2 Annotation & SHAP)."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 2: Why This Problem Matters: Explicit vs. Implicit Toxicity
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation: The Challenge of Implicit Toxicity in Turkish")

    # Card 1: The Moderation Blindspot
    add_card(s2, Inches(0.8), Inches(1.3), Inches(3.7), Inches(5.5))
    tbox = s2.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(3.3), Inches(5.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. THE MODERATION BLINDSPOT"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    blindspots = [
        ("Surface Explicit Toxicity:", "Profanity, slurs, and direct violent threats are easily flagged by keyword filters, regex, and n-gram classifiers."),
        ("Implicit Targeted Hostility:", "Attacks increasingly avoid profane words, using sarcasm, rhetorical mockery, conditional curses ('kanı kurusun'), and metaphorical insults ('lağım medyası')."),
        ("Evasion Strategy:", "Offenders exploit lexical filters deliberately. Keyword-based systems leave users unprotected against sophisticated cyberbullying.")
    ]
    for b_title, b_desc in blindspots:
        p = tf.add_paragraph()
        p.text = f"\n{b_title}\n"
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = DARK_TEXT

    # Card 2: Conversational Hard Negatives
    add_card(s2, Inches(4.8), Inches(1.3), Inches(3.7), Inches(5.5))
    tbox = s2.shapes.add_textbox(Inches(5.0), Inches(1.45), Inches(3.3), Inches(5.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. HARD NEGATIVE OVER-CENSORSHIP"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    hard_negs = [
        ("Negated Insults:", "Sentences like 'Hiç birimiz aptal değiliz' (None of us are stupid) contain toxic triggers ('aptal') but express benign self-defense."),
        ("Proverbs & Moral Maxims:", "Proverbs like 'İnsanoğlu naziktir, ağır sözü kaldırmaz' discuss harsh speech educationally but get flagged by unigram presence ('ağır söz')."),
        ("Sports Banter & Metaphors:", "Colloquial phrases like 'kudursun' (let them rage) or sports scheduling trigger false alarms in keyword-driven systems.")
    ]
    for h_title, h_desc in hard_negs:
        p = tf.add_paragraph()
        p.text = f"\n{h_title}\n"
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = ALERT_RED
        run = p.add_run()
        run.text = h_desc
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = DARK_TEXT

    # Card 3: The Explainability Imperative
    add_card(s2, Inches(8.8), Inches(1.3), Inches(3.733), Inches(5.5), ACCENT_BG, TECH_BLUE)
    tbox = s2.shapes.add_textbox(Inches(9.0), Inches(1.45), Inches(3.333), Inches(5.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. THE EXPLAINABILITY IMPERATIVE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE

    xai_points = [
        ("High Accuracy ≠ Trustworthy:", "Black-box neural networks can achieve >95% accuracy by exploiting spurious demographic tokens or political proper nouns rather than true hostility."),
        ("Faithful Human Rationales:", "Moderation requires knowing WHY a post was flagged. Explanations must highlight human-agreeable rationale spans, not background noise."),
        ("Our Core Hypothesis:", "Token-level SHAP attributions must correlate with human-marked attack cues significantly better than a random baseline to be deemed usable in production moderation.")
    ]
    for x_title, x_desc in xai_points:
        p = tf.add_paragraph()
        p.text = f"\n{x_title}\n"
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = NAVY
        run = p.add_run()
        run.text = x_desc
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 3: Data Reconnaissance & The Two-Tier Strategy
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Data Reconnaissance & The Two-Tier Research Design")

    # Left Column: Public Data Findings & Two Tier Architecture
    add_card(s3, Inches(0.8), Inches(1.3), Inches(5.7), Inches(5.5))
    tbox = s3.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(5.3), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "PUBLIC DATASET AUDIT & ARCHITECTURAL PIVOT"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    audit_bullets = [
        ("Source 1: Overfit-GM (Hugging Face):", "77,800 raw records. 45.8% originates from translated Jigsaw Wikipedia talk pages. Contains distinct machine-translation artifacts ('sığır eti olmak')."),
        ("Source 2: Kaggle 6-Class Labeled Tweets:", "9,999 raw records. Tespit column was empirically inferred: 1 = clean, 0 = toxic (contrary to conventional binary schemes)."),
        ("The Critical Problem:", "Neither dataset annotates implicit attacks, sarcasm, hard negatives, or token rationales. Direct evaluation on them cannot answer our proposal's research question."),
        ("The Two-Tier Architecture:", "\n• Tier 1 (Complete): Rigorous broad binary toxicity corpus (84,866 samples). Serves as training foundation and preliminary baseline development.\n• Tier 2 (Underway): Dedicated human-annotated implicit-attack benchmark (300 posts: implicit, sarcasm, hard negatives + token rationales).")
    ]
    for b_title, b_desc in audit_bullets:
        p = tf.add_paragraph()
        p.text = f"{b_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = DARK_TEXT

    # Right Column: Visual Breakdown
    if os.path.exists("reports/mid_project/dataset_splits_and_sources.png"):
        s3.shapes.add_picture("reports/mid_project/dataset_splits_and_sources.png", Inches(6.8), Inches(1.4), width=Inches(5.733))

    # Bottom summary box on right
    add_card(s3, Inches(6.8), Inches(5.4), Inches(5.733), Inches(1.4), ACCENT_BG, TECH_BLUE)
    tbox = s3.shapes.add_textbox(Inches(7.0), Inches(5.5), Inches(5.333), Inches(1.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TIER 1 CORPUS COMPOSITION SUMMARY (N = 84,866):"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "• Native Turkish Twitter: 49,158 records (57.9%)\n• Jigsaw Wikipedia Translations: 35,608 records (41.9%)\n• Other Sources: 100 records (0.1%)\n• Total Balance: 41,376 Clean (48.75%) | 43,490 Toxic (51.25%)"
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 4: Preprocessing & Evaluation Protocol
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Deterministic Preprocessing Pipeline & Evaluation Protocol")

    # Card 1: Cleaning & Invariants
    add_card(s4, Inches(0.8), Inches(1.3), Inches(5.7), Inches(5.5))
    tbox = s4.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(5.3), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "DATA CLEANING & DEDUPLICATION INVARIANTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    preproc_steps = [
        ("Scam Exclusion (-501 records):", "Isolated and filtered 501 financial fraud spam tweets ('günde 5000 TL kazan', gambling bots) that distorted toxicity vocabulary."),
        ("Turkish-Aware Canonicalization:", "Preserved Turkish dotted/dotless I (İ/i, I/ı), normalized whitespace, and stripped URLs/mentions solely for duplicate hashing. Raw text was preserved intact for model input."),
        ("Deduplication & Collision Removal:", "Removed 2,416 exact and canonical duplicate rows across sources. Quarantined 16 conflicting label pairs (32 rows) to prevent label corruption."),
        ("Source-Stratified Partitioning:", "Exact 75% / 15% / 10% stratified allocation ensuring identical Twitter vs. Jigsaw domain proportions across all splits:\n• Train: 63,650 rows | Val: 12,731 rows | Test: 8,485 rows."),
        ("Cryptographic Test Set Locking:", "Locked test partition before running models:\n• Physical SHA-256: 1c5408ba...\n• Semantic SHA-256: 7bd973c2...\nVerified bit-for-bit before and after all evaluations.")
    ]
    for s_title, s_desc in preproc_steps:
        p = tf.add_paragraph()
        p.text = f"• {s_title} "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = s_desc
        run.font.bold = False
        run.font.size = Pt(9)
        run.font.color.rgb = DARK_TEXT

    # Card 2: Evaluation Rules & Decision Gates
    add_card(s4, Inches(6.8), Inches(1.3), Inches(5.733), Inches(5.5), CARD_BG, CARD_BORDER)
    tbox = s4.shapes.add_textbox(Inches(7.0), Inches(1.45), Inches(5.333), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "PRE-REGISTERED EVALUATION DECISION GATES"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    eval_rules = [
        ("Primary Metric: Macro-Averaged F1", "Harmonic mean of F1_clean and F1_toxic. Robust to class imbalance and equally penalizes false alarms and missed attacks."),
        ("Secondary Metric: Average Precision (AP)", "Calculated via precision-recall Riemann sum (average_precision_score). Distinct from trapezoidal PR-AUC."),
        ("Pre-Registered Meaningful Difference (Δmin = 0.020):", "To guard against claiming trivial gains as breakthroughs, we require Δ = F1_BERTurk - F1_TFIDF ≥ +0.020 (2.0 percentage points) to establish practical NLP superiority."),
        ("Paired Bootstrap Resampling (B = 10,000):", "Non-parametric paired bootstrap with seed 42. Reports empirical 95% confidence intervals and Laplace-corrected p-value (p = (k+1)/(B+1))."),
        ("Source-Sliced Performance Mandate:", "Every model must be scored separately on Native Twitter and Jigsaw slices to detect domain-shift vulnerabilities.")
    ]
    for r_title, r_desc in eval_rules:
        p = tf.add_paragraph()
        p.text = f"• {r_title}\n  "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = NAVY
        run = p.add_run()
        run.text = r_desc
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 5: Classical Baseline: TF-IDF + Logistic Regression
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Classical Baseline: TF-IDF + Logistic Regression (C = 10.0)")

    # Left Column: Configuration & Metrics Table
    add_card(s5, Inches(0.8), Inches(1.3), Inches(6.0), Inches(5.5))
    tbox = s5.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(5.6), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "BASELINE SPECIFICATION & FROZEN TEST METRICS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    specs = [
        ("Feature Space (147,749 Dimensions):", "Combined Word n-grams (1–2, min_df=2, sublinear TF) + Character n-grams (char_wb, 3–5, min_df=5, sublinear TF)."),
        ("Classifier:", "Logistic Regression with balanced class weights, L2 penalty (C = 10.0), L-BFGS solver, max_iter=1000."),
        ("Validation Macro-F1:", "0.9387 (C=10.0 selected over C=0.1 [0.9121] and C=1.0 [0.9284])."),
        ("Frozen Test Results (N = 8,485):", "\n• Macro-F1: 0.9393 (95% CI: [0.9341, 0.9443])\n• Accuracy: 0.9393\n• Toxic Recall: 0.9271 | Toxic Precision: 0.9532\n• Clean Recall: 0.9521 | Clean Precision: 0.9255\n• Average Precision (AP): 0.9862 | ROC-AUC: 0.9851"),
        ("Source Slices:", "Native Twitter: 0.9373 Macro-F1 | Jigsaw: 0.9397 Macro-F1")
    ]
    for s_title, s_desc in specs:
        p = tf.add_paragraph()
        p.text = f"• {s_title} "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = s_desc
        run.font.bold = False
        run.font.size = Pt(9)
        run.font.color.rgb = DARK_TEXT

    # Right Column: Confusion Matrix & Key Takeaway
    add_card(s5, Inches(7.1), Inches(1.3), Inches(5.433), Inches(3.2), CARD_BG, CARD_BORDER)
    tbox = s5.shapes.add_textbox(Inches(7.3), Inches(1.45), Inches(5.0), Inches(2.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TEST CONFUSION MATRIX (N = 8,485)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "                      Predicted Clean     Predicted Toxic\nTrue Clean (4,137)       3,939 (TN)           198 (FP)\nTrue Toxic (4,348)         317 (FN)          4,031 (TP)"
    p2.font.size = Pt(10)
    p2.font.name = "Courier New"
    p2.font.bold = True
    p2.font.color.rgb = DARK_TEXT

    p3 = tf.add_paragraph()
    p3.text = "\n• Total Errors: 515 misclassifications (6.07% error rate)\n• Core Weakness: 317 False Negatives (missed toxic samples) severely dominate 198 False Positives."
    p3.font.size = Pt(9.5)
    p3.font.color.rgb = ALERT_RED

    add_card(s5, Inches(7.1), Inches(4.7), Inches(5.433), Inches(2.1), ACCENT_BG, TECH_BLUE)
    tbox = s5.shapes.add_textbox(Inches(7.3), Inches(4.8), Inches(5.0), Inches(1.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "KEY TAKEAWAY ON THE CLASSICAL BASELINE:"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Linear TF-IDF is surprisingly competitive on broad toxicity (0.9393 Macro-F1) when vulgar keywords and overt slurs are present. However, it exhibits a major blindspot on 317 toxic cases that lack explicit vocabulary, while triggering false alarms on negated profanity."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 6: Qualitative Baseline Error Analysis (30 Curated Cases)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Baseline Error Analysis: 30 Curated Misclassifications")

    # Left: Taxonomy & Chart
    if os.path.exists("reports/mid_project/error_categories_breakdown.png"):
        s6.shapes.add_picture("reports/mid_project/error_categories_breakdown.png", Inches(0.8), Inches(1.3), width=Inches(5.6))

    # Right: Representative Error Examples Table Card
    add_card(s6, Inches(6.6), Inches(1.3), Inches(5.933), Inches(5.5))
    tbox = s6.shapes.add_textbox(Inches(6.8), Inches(1.45), Inches(5.533), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "REPRESENTATIVE CASES (reports/baseline/validation_errors_30.csv)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY

    error_cases = [
        ("FN 1: Implicit Malicious Curse (True=1, Pred=0, p=0.3028)",
         "“Değil unutmak acısı hafifleyenin kanı kurusun”",
         "Hostile curse ('may their blood dry up') contains zero profanity. TF-IDF misses it completely."),
        ("FN 2: Metaphorical Insult (True=1, Pred=0, p=0.1716)",
         "“Adı üzerinde lağım medyası ne beklenir ki”",
         "Metaphor ('sewer media') conveys hostility without explicit unigram insults."),
        ("FP 1: Negated Trigger Word (True=0, Pred=1, p=0.5691)",
         "“Hiç birimiz aptal yada saf değiliz, sadece susuyor ve...”",
         "Toxic token 'aptal' triggers false alarm; model cannot parse negation scope 'değiliz'."),
        ("FP 2: Proverbs / Moral Advice (True=0, Pred=1, p=0.7061)",
         "“İnsanoğlu naziktir , ağır sözü kaldırmaz.”",
         "Educational maxim flagged due to 'ağır söz' unigram presence."),
        ("FP 3: Benign Sports Schedule (True=0, Pred=1, p=0.6077)",
         "“Galatasaray HDI Sigorta Erkek Voleybol Takımımızın...”",
         "Clean volleyball match schedule triggered by char n-gram substring matches.")
    ]
    for c_title, c_text, c_reason in error_cases:
        p = tf.add_paragraph()
        p.text = f"• {c_title}\n  "
        p.font.bold = True
        p.font.size = Pt(9)
        p.font.color.rgb = ALERT_RED if "FN" in c_title else TECH_BLUE
        
        run = p.add_run()
        run.text = f"{c_text}\n  "
        run.font.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = NAVY

        run2 = p.add_run()
        run2.text = f"→ {c_reason}"
        run2.font.size = Pt(8.5)
        run2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 7: Deep Contextual Model: BERTurk Fine-Tuning Status
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Deep Learning Method: Supervised BERTurk Fine-Tuning")

    # Left: Architecture & Sequence Length Optimization
    add_card(s7, Inches(0.8), Inches(1.3), Inches(5.7), Inches(5.5))
    tbox = s7.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(5.3), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "MODEL CONFIGURATION & SEQUENCE OPTIMIZATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    bert_specs = [
        ("Base Model Checkpoint:", "dbmdz/bert-base-turkish-cased (commit SHA: b6e1de16c983e0f2c70664591ea3f22810072608). 12 layers, 768 hidden dim, 110.6M parameters, 32k cased vocab."),
        ("Token Length Analysis (63,650 Train Texts):", "\n• Native Twitter: Max length is 127 tokens! Exactly 0.00% truncation at max_length=128 (100% Twitter context retained).\n• Overall Corpus: 95.58% requires zero truncation at 128."),
        ("Dynamic Padding Collator:", "Dynamic batch padding avoided the O(L²) quadratic cost of static 512 padding, achieving ~200 samples/sec throughput."),
        ("Hardware Profile (NVIDIA RTX 4070 Laptop GPU):", "Peak VRAM: 3,639 MiB (44.4% of 8GB capacity), mixed precision (bfloat16). Fast, reproducible, and zero out-of-memory risk."),
        ("Training Dynamics (Train N=63,650, Val N=12,731):", "AdamW (LR = 2e-5, weight decay 0.01, linear warmup 10%, batch size 32). Validation Macro-F1 rose steadily from 0.9485 (Ep 1) → 0.9572 (Ep 2) → 0.9625 (Ep 3) → 0.9660 (Ep 4).")
    ]
    for b_title, b_desc in bert_specs:
        p = tf.add_paragraph()
        p.text = f"• {b_title} "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.size = Pt(9)
        run.font.color.rgb = DARK_TEXT

    # Right: Training Curve Plot
    if os.path.exists("reports/berturk/training_curves.png"):
        s7.shapes.add_picture("reports/berturk/training_curves.png", Inches(6.8), Inches(1.3), width=Inches(5.733))

    # Bottom summary box on right
    add_card(s7, Inches(6.8), Inches(5.6), Inches(5.733), Inches(1.2), ACCENT_BG, TECH_BLUE)
    tbox = s7.shapes.add_textbox(Inches(7.0), Inches(5.7), Inches(5.333), Inches(1.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "MODEL SELECTION & FULL REFIT PROTOCOL:"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "Best validation checkpoint occurred at Epoch 4 (Macro-F1 = 0.9660). Following our pre-registered protocol, the model was refitted on combined train+val (76,381 rows) for exactly 4 epochs (refit runtime: 1515.07s) prior to test scoring."
    p2.font.size = Pt(9)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 8: Controlled Model Comparison on Frozen Test Partition
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Controlled Test Results: BERTurk vs. TF-IDF Baseline")

    # Top: Comparative Visuals
    if os.path.exists("reports/mid_project/model_comparison_bar.png"):
        s8.shapes.add_picture("reports/mid_project/model_comparison_bar.png", Inches(0.8), Inches(1.3), width=Inches(5.7))
    if os.path.exists("reports/mid_project/confusion_matrices.png"):
        s8.shapes.add_picture("reports/mid_project/confusion_matrices.png", Inches(6.8), Inches(1.3), width=Inches(5.733))

    # Bottom Left Card: Headline Metrics & Statistical Significance
    add_card(s8, Inches(0.8), Inches(4.6), Inches(5.7), Inches(2.2), ACCENT_BG, TECH_BLUE)
    tbox = s8.shapes.add_textbox(Inches(1.0), Inches(4.7), Inches(5.3), Inches(2.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "HEADLINE METRICS ON FROZEN TEST SET (N = 8,485):"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "• TF-IDF Baseline Macro-F1: 0.9393  |  Accuracy: 0.9393\n• Fine-Tuned BERTurk Macro-F1: 0.9654  |  Accuracy: 0.9655\n• Observed Improvement: Δ = +0.0261 (+2.61 percentage points)\n• 10,000 Paired Bootstrap 95% CI: [+0.0207, +0.0317] (exceeds Δmin = 0.020)\n• Empirical Bootstrap p-value: p = 1 / 10,001 ≈ 0.00010 (p < 0.001)"
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    # Bottom Right Card: Source Slices & FN Reduction
    add_card(s8, Inches(6.8), Inches(4.6), Inches(5.733), Inches(2.2), CARD_BG, CARD_BORDER)
    tbox = s8.shapes.add_textbox(Inches(7.0), Inches(4.7), Inches(5.333), Inches(2.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "DOMAIN SLICES & FALSE NEGATIVE COLLAPSE:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "• False Negatives: Collapsed from 317 to 83 (a 73.8% reduction!)\n• Toxic Recall: Surged from 92.71% to 98.09% (+5.38% gain)\n• Native Twitter Slice (N = 4,916): F1 rose 0.9373 → 0.9620 (Δ = +0.0247)\n• Jigsaw Wikipedia Slice (N = 3,560): F1 rose 0.9397 → 0.9688 (Δ = +0.0291)\n• Conclusion: BERTurk is rigorously superior on broad toxicity."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 9: Methodological Limitations & What Cannot Yet Be Claimed
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Limitations & Scientific Boundary: What Cannot Yet Be Claimed")

    # Card 1: The Core Scientific Boundary
    add_card(s9, Inches(0.8), Inches(1.3), Inches(11.733), Inches(1.8), WARN_BG, ALERT_RED)
    tbox = s9.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(11.333), Inches(1.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CRITICAL SCIENTIFIC BOUNDARY: TIER 1 ≠ IMPLICIT ATTACK DETECTION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED

    p2 = tf.add_paragraph()
    p2.text = "While BERTurk achieves 0.9654 Macro-F1 on the Tier 1 test set, this result does NOT establish that our model can detect implicit attacks or sarcasm. Tier 1 measures broad toxicity, which is heavily populated by explicit hostility. A model can achieve high accuracy on Tier 1 while remaining completely ineffective at detecting subtle, context-dependent sarcasm. Proving implicit toxicity detection strictly requires evaluating on our held-out, double-annotated Tier 2 benchmark."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = DARK_TEXT

    # Three Columns: Specific Limitations
    add_card(s9, Inches(0.8), Inches(3.3), Inches(3.7), Inches(3.5))
    tbox = s9.shapes.add_textbox(Inches(1.0), Inches(3.45), Inches(3.3), Inches(3.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. TRANSLATIONESE BIAS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "\n• 41.9% of the Tier 1 corpus is machine-translated from English Jigsaw discussions.\n• Contains unnatural phrasing, translated syntax, and non-native idioms ('sığır eti olmak', 'büyük tüylü toplar').\n• Does not reflect pure organic Turkish social-media discourse."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    add_card(s9, Inches(4.8), Inches(3.3), Inches(3.7), Inches(3.5))
    tbox = s9.shapes.add_textbox(Inches(5.0), Inches(3.45), Inches(3.3), Inches(3.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. PUBLIC ANNOTATION NOISE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "\n• Scraped web corpora contain noisy labels.\n• Our error analysis revealed non-toxic fragments ('tok tutuyo', 'Spoiler Veriyorum') mislabeled as toxic in source data.\n• Human review of Kaggle tags revealed 338 flagged rows with zero subcategory indicators."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    add_card(s9, Inches(8.8), Inches(3.3), Inches(3.733), Inches(3.5))
    tbox = s9.shapes.add_textbox(Inches(9.0), Inches(3.45), Inches(3.333), Inches(3.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. CONTEXT ISOLATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = "\n• Posts are classified in standalone isolation.\n• Sarcasm frequently depends on conversational thread context, author intent, or external real-world news.\n• Narrowing our task to standalone posts avoids halluncinating context but excludes thread-level mockery."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SLIDE 10: Plan to the Final Milestone
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Roadmap to Final Submission: Tier 2 Annotation, SHAP & Ablations")

    # Left: Workstream Roadmap Cards
    add_card(s10, Inches(0.8), Inches(1.3), Inches(6.8), Inches(5.5))
    tbox = s10.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(6.4), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "REMAINING MILESTONES, OWNERS & TARGET DATES"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY

    milestones = [
        ("Tier 2 Guideline & Calibration Pilot (Nov 13 – Nov 20):",
         "Draft formal annotation protocol for implicit attacks, sarcasm, and hard negatives. Label 50-item pilot to calibrate agreement.",
         "Owner: Ahmet Furkan Hazar (Lead) & Orhun Gökdemir"),
        ("Tier 2 Double Annotation & Consensus (Nov 21 – Nov 27):",
         "Double-blind annotation of 300 standalone Turkish posts with token rationale spans. Target Cohen's κ ≥ 0.70 before adjudication.",
         "Owner: All team members (Hazar, Çakır, Gökdemir, Kurtaran)"),
        ("SHAP Attribution Pipeline Implementation (Nov 28 – Dec 04):",
         "Implement Partition/Kernel SHAP for BERTurk using Turkish text masking. Compute token-level heatmaps across Tier 2 samples.",
         "Owner: Ahmet Harun Çakır & Burak Kurtaran"),
        ("Rationale Alignment & Graduate Ablations (Dec 05 – Dec 10):",
         "Evaluate SHAP token-level overlap (AUPRC/F1) against human spans vs. random baseline. Run cased vs. uncased vs. DistilBERTurk ablation.",
         "Owner: Ahmet Furkan Hazar & Ahmet Harun Çakır"),
        ("Final Short Paper Report Authoring (Dec 11 – Dec 18):",
         "Draft final report in short research paper format. Run full reproduction audit.",
         "Owner: All team members")
    ]
    for m_title, m_desc, m_owner in milestones:
        p = tf.add_paragraph()
        p.text = f"• {m_title}\n  "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TECH_BLUE
        run = p.add_run()
        run.text = f"{m_desc}\n  "
        run.font.size = Pt(9)
        run.font.color.rgb = DARK_TEXT
        run2 = p.add_run()
        run2.text = f"[{m_owner}]"
        run2.font.italic = True
        run2.font.size = Pt(8.5)
        run2.font.color.rgb = NAVY

    # Right: Risk Analysis & Fallback Strategy
    add_card(s10, Inches(7.8), Inches(1.3), Inches(4.733), Inches(5.5), WARN_BG, ALERT_RED)
    tbox = s10.shapes.add_textbox(Inches(8.0), Inches(1.45), Inches(4.333), Inches(5.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CRITICAL RISK & FALLBACK MITIGATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED

    risks = [
        ("The Biggest Project Risk:", "Low inter-annotator agreement (Cohen's κ < 0.60) on subjective sarcasm and subtle moral maxims, threatening benchmark validity."),
        ("Proactive Mitigation Strategy:", "Execute an iterative 50-item calibration pilot with written boundary cases (distinguishing sarcastic mockery from benign irony)."),
        ("Fallback Contingency Plan:", "If high agreement cannot be achieved across broad sarcasm, narrow the operational scope to 'targeted mockery containing derogatory entity references' and report separate metrics on the unambiguous adjudicated consensus subset."),
        ("Negative Result Commitment:", "If BERTurk fails to outperform TF-IDF on Tier 2 or SHAP fails to beat a random baseline, we will report the negative result with thorough empirical analysis rather than altering test sets.")
    ]
    for r_title, r_desc in risks:
        p = tf.add_paragraph()
        p.text = f"\n{r_title}\n"
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = NAVY
        run = p.add_run()
        run.text = r_desc
        run.font.bold = False
        run.font.size = Pt(9)
        run.font.color.rgb = DARK_TEXT

    # =========================================================================
    # APPENDIX SLIDE 11: Project Card Fields 1–3
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Project Card (Part 2 — Graded Appendix): Fields 1, 2, 3", "PROJECT CARD APPENDIX · SLIDE 1 OF 4 (GRADED, NOT PRESENTED)")

    # Field 1: Title and Abstract
    add_card(s11, Inches(0.8), Inches(1.3), Inches(11.733), Inches(2.1))
    tbox = s11.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 1: TITLE & DRAFT ABSTRACT (BECOMES: TITLE, ABSTRACT)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Working Title: Explainable Turkish Implicit Toxicity Detection: Benchmarking BERTurk and Token Rationale Alignment Against Lexical Baselines"
    p2.font.size = Pt(10)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph()
    p3.text = "Draft Abstract (≤150 words): Automated toxicity detection systems frequently fail on nuanced language, mistaking benign moral maxims for profanity while overlooking subtle, sarcastic hostility. In this work, we investigate whether fine-tuning BERTurk improves the identification of implicit and sarcastic attacks in Turkish social media over a tuned TF-IDF + Logistic Regression baseline, and whether token-level SHAP attributions faithfully reflect human-annotated evidence. On our Tier 1 foundation of 84,866 verified Turkish records, BERTurk achieved 0.9654 Macro-F1 (vs. 0.9393 for TF-IDF, Δ = +0.0261, 95% bootstrap CI: [0.0207, 0.0317], p ≈ 0.0001), reducing missed toxic comments by 73.8%. On our dedicated, human-annotated Tier 2 benchmark of implicit attacks and hard negatives, BERTurk achieves [Tier 2 Evaluation Pending — Final Milestone] Macro-F1, while token-level SHAP explanations yield [SHAP Alignment Pending — Final Milestone] overlap with human rationale spans against random-token baselines."
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = DARK_TEXT

    # Field 2: Research Question
    add_card(s11, Inches(0.8), Inches(3.55), Inches(5.7), Inches(3.25))
    tbox = s11.shapes.add_textbox(Inches(1.0), Inches(3.65), Inches(5.3), Inches(3.05))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 2: RESEARCH QUESTION (BECOMES: INTRODUCTION)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Status: UNCHANGED from proposal."
    p2.font.size = Pt(9.5)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph()
    p3.text = "“Can a Turkish BERTurk classifier fine-tuned on offensive-language labels improve macro-F1 on a manually annotated, held-out set of implicitly targeted or sarcastic attacks over a TF-IDF + logistic-regression baseline, and do token-level SHAP explanations identify human-annotated attack cues?”\n\nNegative Result Condition: Defined empirically as 'no' if the fine-tuned model's Macro-F1 improvement over the baseline is not statistically distinguishable from zero (p ≥ 0.05 or 95% CI includes zero), or if SHAP highlights do not overlap annotator evidence better than a random-token baseline.\n\nTwo-Tier Strategy Justification: Public datasets lack implicit attack annotations. We adopted Tier 1 for broad baseline modeling and Tier 2 for dedicated implicit testing without changing our research goal."
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = DARK_TEXT

    # Field 3: Contributions
    add_card(s11, Inches(6.8), Inches(3.55), Inches(5.733), Inches(3.25))
    tbox = s11.shapes.add_textbox(Inches(7.0), Inches(3.65), Inches(5.333), Inches(3.05))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 3: TESTABLE CONTRIBUTIONS (BECOMES: INTRODUCTION)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Three concrete, testable empirical claims:"
    p2.font.size = Pt(9.5)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph()
    p3.text = "1. Empirical Baseline Claim (Supported): Supervised fine-tuning of BERTurk on broad Turkish offensive text produces a statistically and practically significant Macro-F1 gain (Δ ≥ 0.020, p < 0.001) over an optimized word+char TF-IDF baseline on broad toxicity, driven by a >70% reduction in false negatives on syntactically complex and non-profane hostility.\n\n2. Implicit Transfer Claim (To Be Evaluated on Tier 2): Supervised fine-tuning on broad offensive labels alone is insufficient to generalize to implicit attacks and sarcasm, leading to a significant drop in Macro-F1 on held-out implicit attacks compared to broad toxicity.\n\n3. Faithfulness Alignment Claim (To Be Evaluated on Tier 2): Token-level SHAP attributions from BERTurk achieve a statistically higher token-level F1 / AUPRC overlap with human-annotated attack rationales than an empirical random baseline, but suffer attribution dispersion on negated expressions."
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = DARK_TEXT

    # =========================================================================
    # APPENDIX SLIDE 12: Project Card Fields 4–6
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Project Card (Part 2 — Graded Appendix): Fields 4, 5, 6", "PROJECT CARD APPENDIX · SLIDE 2 OF 4 (GRADED, NOT PRESENTED)")

    # Field 4: Related Work
    add_card(s12, Inches(0.8), Inches(1.3), Inches(11.733), Inches(2.2))
    tbox = s12.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(2.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 4: RELATED WORK — VERIFIED CITATIONS & DIFFERENTIATION (BECOMES: RELATED WORK)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE

    refs = [
        ("Bilgin & Bekar (2025):", "Turkish Cyberbullying Detection with Fine-Tuned Pre-Trained Language Models. Benchmarks BERTurk variants on 3,388 tweets for broad cyberbullying. Differs: We focus specifically on held-out implicit attacks and evaluate SHAP token faithfulness against human rationale spans."),
        ("Gökdeniz & Canbay (2025):", "Detection of Cyberbullying from Turkish Texts with Artificial Intelligence and NLP. Evaluates classical ML and FastText on Turkish tweets. Differs: Examines surface lexical features only without transformer attention or explanation alignment."),
        ("Mahmud, Ptaszynski & Masui (2024):", "Leveraging Explainable AI and Sarcasm Features for Improved Cyberbullying Detection (IEEE Access / DPSH, DOI: 10.1109/DPSH60098.2024.10774626). Explores SHAP and sarcasm in South Asian languages. Differs: We address agglutinative Turkish morphology and test token-level overlap against human rationales."),
        ("Schweter (2020):", "BERTurk - BERT models for Turkish (Zenodo, DOI: 10.5281/zenodo.3770924). Releases monolingual Turkish BERT. Differs: We fine-tune and probe the cased model specifically for implicit toxicity and sequence truncation dynamics."),
        ("Lundberg & Lee (2017):", "A Unified Approach to Interpreting Model Predictions (NeurIPS 2017). Defines SHAP framework. Differs: We adapt Partition SHAP to Turkish text classification to evaluate explanation overlap against human-marked evidence spans.")
    ]
    for r_auth, r_desc in refs:
        p = tf.add_paragraph()
        p.text = f"• {r_auth} "
        p.font.bold = True
        p.font.size = Pt(8.5)
        p.font.color.rgb = NAVY
        run = p.add_run()
        run.text = r_desc
        run.font.bold = False
        run.font.size = Pt(8)
        run.font.color.rgb = DARK_TEXT

    # Field 5: Data
    add_card(s12, Inches(0.8), Inches(3.65), Inches(5.7), Inches(3.15))
    tbox = s12.shapes.add_textbox(Inches(1.0), Inches(3.75), Inches(5.3), Inches(2.95))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 5: DATA ARCHITECTURE (BECOMES: DATA)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• Sources & Licenses: Overfit-GM (HF, Open Data, 77,800 rows); Kaggle 6-Class Labeled Turkish Tweets (CC0, 9,999 rows).\n• Cleaning Accounting: 87,799 raw combined → 501 scam excluded → 2,416 duplicates removed → 16 conflicting pairs quarantined → 84,866 retained Tier 1 records (41,376 clean / 48.75%, 43,490 toxic / 51.25%).\n• Domain Composition: Native Twitter (49,158 / 57.9%), Jigsaw Wikipedia Translations (35,608 / 41.9%), Other (100 / 0.1%).\n• Partitions: Train (63,650), Val (12,731), Frozen Test (8,485).\n• Tier 2 Annotation (Underway): 300 standalone Turkish posts double-annotated for implicit toxicity, sarcasm, and token rationale spans."
    p2.font.size = Pt(8.5)
    p2.font.color.rgb = DARK_TEXT

    # Field 6: Method and Baseline
    add_card(s12, Inches(6.8), Inches(3.65), Inches(5.733), Inches(3.15))
    tbox = s12.shapes.add_textbox(Inches(7.0), Inches(3.75), Inches(5.333), Inches(2.95))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 6: METHOD & BASELINE (BECOMES: METHOD)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• Baseline as Run: Word n-grams (1–2) + Char n-grams (3–5, char_wb) with Logistic Regression (C=10.0, balanced weights, 147,749 features).\n• Method as Implemented: dbmdz/bert-base-turkish-cased (110M params) with linear classification head. AdamW (LR=2e-5, weight decay=0.01, linear warmup 10%, batch size 32, max_length 128 with dynamic padding).\n• Controlled Change: Exactly identical training data (63,650), validation set (12,731), test set (8,485), and binary labels. Only feature representation and inductive bias changed.\n• Compute Used: ~1.2 GPU-hours on local NVIDIA GeForce RTX 4070 Laptop GPU (3,639 MiB peak VRAM, bfloat16). Remaining needed: ~3 GPU-hours for SHAP sampling on Tier 2."
    p2.font.size = Pt(8.5)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # APPENDIX SLIDE 13: Project Card Fields 7–9
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Project Card (Part 2 — Graded Appendix): Fields 7, 8, 9", "PROJECT CARD APPENDIX · SLIDE 3 OF 4 (GRADED, NOT PRESENTED)")

    # Field 7: Evaluation Protocol
    add_card(s13, Inches(0.8), Inches(1.3), Inches(11.733), Inches(1.7))
    tbox = s13.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 7: EVALUATION PROTOCOL (BECOMES: EXPERIMENTAL SETUP)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• Primary Metric: Macro-averaged F1 score across Clean and Toxic classes. Secondary: Accuracy, Toxic Recall, Average Precision (AP), ROC-AUC.\n• Frozen Test Set: Locked before experiments; verified bit-for-bit with physical SHA-256 (1c5408ba...) and semantic membership SHA-256 (7bd973c2...). Zero test tuning.\n• Statistical Protocol: Non-parametric paired bootstrap (B = 10,000 resamples, random seed 42) computing empirical 95% confidence intervals and Laplace-corrected p-value.\n• Pre-Registered Practical Threshold: Δmin = 0.020 (+2.0 percentage points Macro-F1). The lower bound of the 95% CI must exceed Δmin to claim practical significance."
    p2.font.size = Pt(8.5)
    p2.font.color.rgb = DARK_TEXT

    # Field 8: Results So Far Table
    add_card(s13, Inches(0.8), Inches(3.15), Inches(11.733), Inches(2.0))
    tbox = s13.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(11.333), Inches(1.8))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 8: RESULTS TABLE (BECOMES: RESULTS)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE

    # Draw table
    table_shape = s13.shapes.add_table(5, 7, Inches(1.0), Inches(3.6), Inches(11.333), Inches(1.4))
    table = table_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(1.1)
    table.columns[4].width = Inches(1.2)
    table.columns[5].width = Inches(1.4)
    table.columns[6].width = Inches(1.4)

    headers = ["Model / System", "Evaluation Set", "Macro-F1 (95% CI)", "Accuracy", "Toxic Recall", "Avg Precision", "Status"]
    rows_data = [
        ["TF-IDF + LogReg (C=10.0)", "Tier 1 Frozen Test", "0.9393 [0.934, 0.944]", "0.9393", "0.9271", "0.9862", "Completed Baseline"],
        ["BERTurk Fine-Tuned", "Tier 1 Frozen Test", "0.9654 [0.961, 0.969]", "0.9655", "0.9809", "0.9854", "Completed Method"],
        ["Model Difference (Δ)", "Tier 1 Frozen Test", "+0.0261 [+0.021, +0.032]", "+0.0262", "+0.0538", "-0.0008", "p ≈ 0.0001 (Δ ≥ 0.020)"],
        ["BERTurk vs. Baseline", "Tier 2 Benchmark", "[NOT YET EVALUATED]", "—", "—", "—", "Final Milestone"]
    ]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(8.5)
            p.font.bold = True
            p.font.color.rgb = WHITE

    for r_idx, r in enumerate(rows_data):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(8)
                p.font.color.rgb = DARK_TEXT
                if c_idx == 0 or c_idx == 2:
                    p.font.bold = True

    # Field 9: Error Analysis Summary
    add_card(s13, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.5))
    tbox = s13.shapes.add_textbox(Inches(1.0), Inches(5.35), Inches(11.333), Inches(1.4))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 9: BASELINE ERROR ANALYSIS TAXONOMY (30 CASES FROM reports/baseline/validation_errors_30.csv)"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• False Negatives (15 cases): Metaphorical Mockery (5 cases, e.g. 'lağım medyası'), Implicit Curses (3 cases, 'kanı kurusun'), Translationese Idioms (3 cases, 'sığır eti olmak'), Annotation Noise (3 cases, 'tok tutuyo'), Sexist Commentary (1 case). Lexical models fail when hostility lacks profanity.\n• False Positives (15 cases): Negated Trigger Words (4 cases, e.g. 'aptal değiliz'), Proverbs / Moral Maxims (4 cases, 'ağır sözü kaldırmaz'), Sports Rivalry / Scheduling (3 cases, 'kudursun'), Philosophical Essays (2 cases), Technical / Wiki Jargon (2 cases). Lexical models trigger false alarms on high-arousal words regardless of negation or benign intent."
    p2.font.size = Pt(8)
    p2.font.color.rgb = DARK_TEXT

    # =========================================================================
    # APPENDIX SLIDE 14: Project Card Fields 10–12
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Project Card (Part 2 — Graded Appendix): Fields 10, 11, 12", "PROJECT CARD APPENDIX · SLIDE 4 OF 4 (GRADED, NOT PRESENTED)")

    # Field 10: Research Lead Section
    add_card(s14, Inches(0.8), Inches(1.3), Inches(11.733), Inches(2.0))
    tbox = s14.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.8))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 10: RESEARCH LEAD SECTION — AHMET FURKAN HAZAR (CSE 654) (BECOMES: RELATED WORK, ABLATION)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Author: Ahmet Furkan Hazar (Research Lead)\n1. Critical Analysis: Bilgin & Bekar (2025) benchmarked BERTurk variants on 3,388 offensive Turkish tweets, demonstrating transformer superiority over classical models. However, their dataset is confined to overt profanity and lacks explainability verification. Mahmud et al. (2024) demonstrated the utility of SHAP for cyberbullying in South Asian languages, but treated attributions as visual aids rather than evaluating token overlap against human rationale spans. Our project addresses both gaps: isolating implicit sarcasm and testing whether SHAP attributions surpass random baselines on human-marked attack cues.\n2. Planned Graduate Ablations: (a) Architecture Ablation: bert-base-turkish-cased vs. bert-base-turkish-uncased vs. distilbert-base-turkish-cased to quantify the contribution of Turkish orthographic casing and depth to sarcasm detection; (b) Attribution Method Ablation: Partition SHAP vs. Integrated Gradients vs. Input×Gradient on human rationale overlap."
    p2.font.size = Pt(8)
    p2.font.color.rgb = DARK_TEXT

    # Field 11: Plan and Risk
    add_card(s14, Inches(0.8), Inches(3.45), Inches(5.7), Inches(3.35))
    tbox = s14.shapes.add_textbox(Inches(1.0), Inches(3.55), Inches(5.3), Inches(3.15))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 11: PLAN, RISKS & FALLBACK (BECOMES: LIMITATIONS)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• Execution Timeline & Owners:\n  - Nov 13–20: Tier 2 Guideline & 50-item Pilot (A. F. Hazar, O. Gökdemir)\n  - Nov 21–27: Double Annotation of 300 Tier 2 samples & Cohen's κ calculation (All Members)\n  - Nov 28–Dec 04: SHAP implementation & token attribution extraction (A. H. Çakır, B. Kurtaran)\n  - Dec 05–10: Rationale overlap evaluation & graduate ablations (A. F. Hazar, A. H. Çakır)\n  - Dec 11–18: Final short paper report & reproduction audit (All Members)\n• Biggest Risk: Low inter-annotator agreement (κ < 0.60) on subtle sarcasm.\n• Mitigation & Fallback: Calibration session after 50 pilot items. If disagreement persists, narrow scope to targeted implicit mockery and report consensus subsets separately.\n• Response to Proposal Feedback: Fully addressed by introducing two-tier architecture, isolating scam data, and pre-registering Δmin = 0.020."
    p2.font.size = Pt(8)
    p2.font.color.rgb = DARK_TEXT

    # Field 12: Reproducibility & AI Disclosure
    add_card(s14, Inches(6.8), Inches(3.45), Inches(5.733), Inches(3.35))
    tbox = s14.shapes.add_textbox(Inches(7.0), Inches(3.55), Inches(5.333), Inches(3.15))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "FIELD 12: REPRODUCIBILITY & DISCLOSURE (BECOMES: APPENDIX)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p2 = tf.add_paragraph()
    p2.text = "• Environment: Python 3.11.18, uv (uv.lock pinned), PyTorch 2.6.0+cu124, Transformers 4.49.0, scikit-learn 1.6.1. Seed: 42. RTX 4070 Laptop GPU (8GB VRAM).\n• One-Command Exact Reproduction:\n  uv run python scripts/run_prepare_data.py   # Data preparation & lockfile check\n  uv run python scripts/run_tfidf_baseline.py # TF-IDF baseline training & eval\n  uv run python scripts/run_berturk.py --eval-only # BERTurk eval & bootstrap\n  uv run pytest -v                           # Automated test suite (34 tests)\n• Member Roles: A. F. Hazar (Lead, research design, annotation guidelines); A. H. Çakır (pipeline, canonicalization, SHAP, test suite); O. Gökdemir (data acquisition, inventory audit, candidate mining); B. Kurtaran (baseline modeling, collator, hyperparameter tuning).\n• AI Assistance Transparency: ChatGPT (OpenAI) used for initial literature discovery and prompt drafting. Antigravity (Google DeepMind) used as pair-programming assistant for implementing scripts, collators, test suites, and slides. Zero AI data fabrication."
    p2.font.size = Pt(8)
    p2.font.color.rgb = DARK_TEXT

    prs.save("Team07_MidProject.pptx")
    print("Successfully saved Team07_MidProject.pptx (14 slides: 10 talk + 4 appendix)")

if __name__ == "__main__":
    create_presentation()
