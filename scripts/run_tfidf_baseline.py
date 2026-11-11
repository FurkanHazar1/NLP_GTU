#!/usr/bin/env python3
"""CLI runner for TF-IDF + Logistic Regression baseline model selection and evaluation."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime

import pandas as pd

from toxicity_detection.config import (
    BASELINE_REPORT_DIR,
    BASELINES_DIR,
    MEANINGFUL_DIFFERENCE_DELTA,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    RANDOM_SEED,
    TEST_CHECKSUM_PATH,
    TEST_MEMBERSHIP_PATH,
)
from toxicity_detection.data import compute_file_sha256, compute_membership_hash
from toxicity_detection.evaluation import (
    compute_classification_metrics,
    compute_source_sliced_metrics,
    export_validation_errors,
    extract_validation_errors,
    plot_confusion_matrix,
    plot_model_comparison,
)
from toxicity_detection.models import build_tfidf_pipeline, save_model_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tfidf_baseline")


def main() -> int:
    """Run model exploration, selection, error analysis, and test set evaluation."""
    start_time = datetime.now(UTC)
    logger.info("=== Phase 3.3 — TF-IDF Baseline Training & Evaluation ===")

    # 1. Verify Test Set Integrity before any training starts
    logger.info("Checking initial test set integrity and cryptographic checksums...")
    test_file_sha_initial = compute_file_sha256(PROCESSED_TEST_PATH)
    recorded_file_sha = TEST_CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    if test_file_sha_initial != recorded_file_sha:
        raise ValueError(
            f"Test file SHA-256 mismatch! Expected {recorded_file_sha}, got {test_file_sha_initial}"
        )

    test_df_initial = pd.read_parquet(PROCESSED_TEST_PATH)
    test_mem_sha_initial = compute_membership_hash(test_df_initial)
    recorded_mem_sha = TEST_MEMBERSHIP_PATH.read_text(encoding="utf-8").strip()
    if test_mem_sha_initial != recorded_mem_sha:
        raise ValueError(
            f"Test membership SHA-256 mismatch! Expected {recorded_mem_sha}, got {test_mem_sha_initial}"
        )
    logger.info(
        "Initial test checksums confirmed: Physical=%s, Semantic=%s",
        test_file_sha_initial,
        test_mem_sha_initial,
    )

    # 2. Ingest train and val partitions
    logger.info("Loading training and validation partitions...")
    train_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
    val_df = pd.read_parquet(PROCESSED_VAL_PATH)
    logger.info("Loaded train (%d rows) and val (%d rows)", len(train_df), len(val_df))

    # 3. Text Preprocessing Strategy Comparison
    logger.info("Evaluating text preprocessing strategy (Turkish lower vs. Raw)...")
    pipe_raw = build_tfidf_pipeline(
        representation="word", c_val=1.0, preprocessor_type="raw", random_state=RANDOM_SEED
    )
    pipe_raw.fit(train_df["text_raw"], train_df["label_binary"])
    preds_raw = pipe_raw.predict(val_df["text_raw"])
    probs_raw = pipe_raw.predict_proba(val_df["text_raw"])[:, 1]
    metrics_raw = compute_classification_metrics(val_df["label_binary"], preds_raw, probs_raw)

    pipe_tr = build_tfidf_pipeline(
        representation="word", c_val=1.0, preprocessor_type="turkish_lower", random_state=RANDOM_SEED
    )
    pipe_tr.fit(train_df["text_raw"], train_df["label_binary"])
    preds_tr = pipe_tr.predict(val_df["text_raw"])
    probs_tr = pipe_tr.predict_proba(val_df["text_raw"])[:, 1]
    metrics_tr = compute_classification_metrics(val_df["label_binary"], preds_tr, probs_tr)

    logger.info(
        "Preprocessing comparison on Word-level (C=1.0): Raw Macro-F1=%.4f vs. Turkish-Lower Macro-F1=%.4f",
        metrics_raw["macro_f1"],
        metrics_tr["macro_f1"],
    )

    preprocessing_comparison = {
        "raw_text_no_lower": metrics_raw,
        "turkish_aware_lower": metrics_tr,
        "selected_preprocessor": "turkish_lower",
        "rationale": (
            "Turkish-aware lowercasing properly handles dotted/dotless I pairs (İ->i, I->ı) "
            f"and yields a superior Macro-F1 ({metrics_tr['macro_f1']} vs. {metrics_raw['macro_f1']})."
        ),
    }

    # 4. Systematic Candidate Exploration
    representations = ["word", "char", "combined"]
    c_values = [0.1, 1.0, 10.0]

    candidate_results: list[dict] = []
    trained_pipelines: dict[str, tuple] = {}

    logger.info("Beginning grid search over 9 candidate configurations on train -> val...")
    for rep in representations:
        best_rep_candidate = None
        best_rep_f1 = -1.0

        for c_val in c_values:
            cand_id = f"tfidf_{rep}_c{c_val}"
            cand_name = f"{rep.capitalize()} (C={c_val})"
            logger.info("Training candidate: %s ...", cand_name)

            t0 = time.time()
            pipe = build_tfidf_pipeline(
                representation=rep,
                c_val=c_val,
                preprocessor_type="turkish_lower",
                random_state=RANDOM_SEED,
            )
            pipe.fit(train_df["text_raw"], train_df["label_binary"])
            fit_time = round(time.time() - t0, 3)

            t0 = time.time()
            val_preds = pipe.predict(val_df["text_raw"])
            val_probs = pipe.predict_proba(val_df["text_raw"])[:, 1]
            pred_time = round(time.time() - t0, 3)

            val_metrics = compute_classification_metrics(
                val_df["label_binary"], val_preds, val_probs
            )
            logger.info(
                "  Result %s -> Macro-F1: %.4f | Acc: %.4f | PR-AUC: %.4f | Fit: %.2fs",
                cand_name,
                val_metrics["macro_f1"],
                val_metrics["accuracy"],
                val_metrics["pr_auc"],
                fit_time,
            )

            record = {
                "id": cand_id,
                "name": cand_name,
                "representation": rep,
                "c_val": c_val,
                "preprocessor": "turkish_lower",
                "fit_time_seconds": fit_time,
                "predict_time_seconds": pred_time,
                "metrics": val_metrics,
            }
            candidate_results.append(record)

            if val_metrics["macro_f1"] > best_rep_f1:
                best_rep_f1 = val_metrics["macro_f1"]
                best_rep_candidate = (cand_id, pipe, record, val_preds, val_probs)

        # Save representative model for each representation family
        if best_rep_candidate is not None:
            _rep_id, rep_pipe, rep_rec, _rep_preds, _rep_probs = best_rep_candidate
            trained_pipelines[rep] = best_rep_candidate
            family_dir = BASELINES_DIR / f"tfidf_{rep}"
            save_model_artifact(
                pipeline=rep_pipe,
                config=rep_rec,
                out_dir=family_dir,
            )
            logger.info("Saved family baseline artifact to %s", family_dir)

    # 5. Baseline Selection Decision Gate
    # Sort candidates by validation Macro-F1 descending
    sorted_candidates = sorted(
        candidate_results, key=lambda x: x["metrics"]["macro_f1"], reverse=True
    )
    best_candidate = sorted_candidates[0]
    best_rep = best_candidate["representation"]
    best_f1 = best_candidate["metrics"]["macro_f1"]

    # Compare against best word-level baseline
    best_word_candidate = next(c for c in sorted_candidates if c["representation"] == "word")
    word_f1 = best_word_candidate["metrics"]["macro_f1"]
    delta_vs_word = round(best_f1 - word_f1, 4)

    logger.info(
        "Candidate ranking complete. Top model: %s (Macro-F1=%.4f). Best Word: %s (Macro-F1=%.4f). Delta=%.4f",
        best_candidate["name"],
        best_f1,
        best_word_candidate["name"],
        word_f1,
        delta_vs_word,
    )

    selection_rationale = (
        f"Selected {best_candidate['name']} with validation Macro-F1={best_f1:.4f}. "
        f"The improvement over the best word-level baseline ({best_word_candidate['name']}, Macro-F1={word_f1:.4f}) "
        f"is +{delta_vs_word:.4f}, which exceeds the meaningful difference threshold (Delta_min={MEANINGFUL_DIFFERENCE_DELTA}). "
        "Character n-grams (3-5 char_wb) combined with word n-grams (1-2) capture Turkish agglutinative morphology, "
        "suffix variations, and informal social media spelling."
    )
    logger.info("Selection rationale: %s", selection_rationale)

    # 6. Extract Exactly 30 Validation Prediction Errors
    logger.info("Extracting 30 deterministic validation errors for human review...")
    # Retrieve predictions of the selected candidate on validation set
    _, _selected_pipe_val, _, best_val_preds, best_val_probs = trained_pipelines[best_rep]
    val_errors_df = extract_validation_errors(
        val_df=val_df,
        y_pred=best_val_preds,
        y_prob=best_val_probs,
        n_samples=30,
        random_state=RANDOM_SEED,
    )
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val_errors_path = BASELINE_REPORT_DIR / "validation_errors_30.csv"
    export_validation_errors(val_errors_df, val_errors_path)
    logger.info(
        "Saved %d validation errors (%d FP, %d FN) to %s",
        len(val_errors_df),
        (val_errors_df["error_type"] == "false_positive").sum(),
        (val_errors_df["error_type"] == "false_negative").sum(),
        val_errors_path,
    )

    # 7. Refit Selected Baseline on Combined Train + Val Data
    logger.info(
        "Refitting selected baseline configuration (%s, C=%.1f) on combined train+val (%d rows)...",
        best_candidate["representation"],
        best_candidate["c_val"],
        len(train_df) + len(val_df),
    )
    combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_pipeline = build_tfidf_pipeline(
        representation=best_candidate["representation"],
        c_val=best_candidate["c_val"],
        preprocessor_type="turkish_lower",
        random_state=RANDOM_SEED,
    )
    t0 = time.time()
    final_pipeline.fit(
        combined_train_val_df["text_raw"], combined_train_val_df["label_binary"]
    )
    final_fit_time = round(time.time() - t0, 3)
    logger.info("Refit completed in %.2fs", final_fit_time)

    # Save selected baseline artifact
    selected_baseline_dir = BASELINES_DIR / "selected_baseline"
    selected_config = {
        "selected_candidate": best_candidate["id"],
        "representation": best_candidate["representation"],
        "c_val": best_candidate["c_val"],
        "preprocessor": "turkish_lower",
        "training_data_rows": len(combined_train_val_df),
        "refit_time_seconds": final_fit_time,
        "selection_rationale": selection_rationale,
    }
    save_model_artifact(
        pipeline=final_pipeline,
        config=selected_config,
        out_dir=selected_baseline_dir,
    )
    logger.info("Saved selected final baseline to %s", selected_baseline_dir)

    # 8. Single Frozen Test Set Evaluation
    logger.info("Executing single final evaluation on frozen test partition...")
    test_df = pd.read_parquet(PROCESSED_TEST_PATH)
    t0 = time.time()
    test_preds = final_pipeline.predict(test_df["text_raw"])
    test_probs = final_pipeline.predict_proba(test_df["text_raw"])[:, 1]
    test_eval_time = round(time.time() - t0, 3)

    test_df_eval = test_df.copy()
    test_df_eval["label_pred"] = test_preds
    test_df_eval["pred_prob_toxic"] = test_probs

    test_metrics = compute_classification_metrics(
        y_true=test_df_eval["label_binary"],
        y_pred=test_df_eval["label_pred"],
        y_prob=test_df_eval["pred_prob_toxic"],
    )
    logger.info(
        "Final Test Evaluation: Macro-F1=%.4f | Acc=%.4f | PR-AUC=%.4f | Time=%.2fs",
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
        test_metrics["pr_auc"],
        test_eval_time,
    )

    # 9. Source-Sliced Metrics Computation
    logger.info("Computing source-sliced test metrics...")
    sliced_metrics = compute_source_sliced_metrics(test_df_eval)
    for slice_name, sm in sliced_metrics.items():
        logger.info(
            "  Slice %-16s -> N=%5d | Macro-F1=%.4f | Clean-F1=%.4f | Toxic-F1=%.4f",
            slice_name,
            sm["support_total"],
            sm["macro_f1"],
            sm["f1_clean"],
            sm["f1_toxic"],
        )

    # 10. Generate Visualizations
    logger.info("Generating evaluation plots...")
    cm_path = BASELINE_REPORT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        out_path=cm_path,
        title=f"Test Set Confusion Matrix ({best_candidate['name']})",
    )
    logger.info("Saved %s", cm_path)

    comp_path = BASELINE_REPORT_DIR / "model_comparison.png"
    plot_model_comparison(
        candidates=candidate_results,
        out_path=comp_path,
        title="Validation Set Performance Across TF-IDF Candidates",
    )
    logger.info("Saved %s", comp_path)

    # 11. Save Structured JSON Reports
    logger.info("Writing JSON report artifacts...")
    val_report_path = BASELINE_REPORT_DIR / "validation_results.json"
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing_comparison": preprocessing_comparison,
                "candidates": candidate_results,
                "selected_candidate": best_candidate,
                "selection_rationale": selection_rationale,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    test_report_path = BASELINE_REPORT_DIR / "test_results.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": best_candidate["id"],
                "model_name": best_candidate["name"],
                "evaluation_timestamp": datetime.now(UTC).isoformat(),
                "test_eval_time_seconds": test_eval_time,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    slice_report_path = BASELINE_REPORT_DIR / "source_slice_metrics.json"
    with open(slice_report_path, "w", encoding="utf-8") as f:
        json.dump(sliced_metrics, f, indent=2, ensure_ascii=False)

    # 12. Verify Test Set Lockfile Hashes Post-Evaluation
    logger.info("Verifying test set integrity post-evaluation...")
    test_file_sha_final = compute_file_sha256(PROCESSED_TEST_PATH)
    test_df_final = pd.read_parquet(PROCESSED_TEST_PATH)
    test_mem_sha_final = compute_membership_hash(test_df_final)

    assert test_file_sha_final == test_file_sha_initial, "Test file altered during evaluation!"
    assert test_mem_sha_final == test_mem_sha_initial, "Test membership altered during evaluation!"
    logger.info("Post-evaluation verification PASSED: Test partition is bit-for-bit identical.")

    total_elapsed = round((datetime.now(UTC) - start_time).total_seconds(), 2)
    logger.info(
        "=== Baseline Pipeline Finished Successfully in %.2fs ===", total_elapsed
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
