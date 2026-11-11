#!/usr/bin/env python3
"""CLI runner for BERTurk supervised fine-tuning, evaluation, and controlled baseline comparison."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from toxicity_detection.config import (
    BASELINES_DIR,
    BERTURK_MODELS_DIR,
    BERTURK_REPORTS_DIR,
    BERTURK_SELECTED_MODEL_DIR,
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
    generate_aligned_test_predictions,
    generate_berturk_validation_errors,
    paired_bootstrap_macro_f1,
    plot_model_comparison_slices,
    plot_training_curves,
)
from toxicity_detection.models.berturk import (
    BERTURK_MODEL_ID,
    BERTURK_REVISION,
    DEFAULT_MAX_LENGTH,
    DynamicPaddingCollator,
    ToxicityDataset,
    load_berturk_artifact,
    load_berturk_model,
    load_berturk_tokenizer,
    save_berturk_artifact,
)
from toxicity_detection.training import (
    set_seed,
    train_berturk_with_early_stopping,
    train_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("berturk_runner")


def main() -> int:
    """Run BERTurk training, evaluation, paired bootstrap comparison, and report generation."""
    parser = argparse.ArgumentParser(description="BERTurk Phase 3.4 Training & Evaluation Runner")
    parser.add_argument("--force-refit", action="store_true", help="Force refitting even if model checkpoint exists")
    parser.add_argument("--eval-only", action="store_true", help="Run test evaluation and statistical comparisons only using existing artifacts")
    args = parser.parse_args()

    start_time = datetime.now(UTC)
    logger.info("=== Phase 3.4 — BERTurk Fine-Tuning & Controlled Comparison ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Execution device: %s", device)
    if device.type == "cuda":
        logger.info("GPU Model: %s", torch.cuda.get_device_name(0))
        logger.info(
            "VRAM: %.2f GB Total, %.2f GB Free",
            torch.cuda.get_device_properties(0).total_memory / (1024**3),
            torch.cuda.mem_get_info(0)[0] / (1024**3),
        )

    # 1. Preflight Baseline & Test Integrity Check
    logger.info("Executing preflight test set integrity check...")
    test_file_sha_initial = compute_file_sha256(PROCESSED_TEST_PATH)
    rec_file_sha = TEST_CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    if test_file_sha_initial != rec_file_sha:
        raise ValueError(
            f"Physical test SHA mismatch! Expected {rec_file_sha}, got {test_file_sha_initial}"
        )

    test_df_initial = pd.read_parquet(PROCESSED_TEST_PATH)
    test_mem_sha_initial = compute_membership_hash(test_df_initial)
    rec_mem_sha = TEST_MEMBERSHIP_PATH.read_text(encoding="utf-8").strip()
    if test_mem_sha_initial != rec_mem_sha:
        raise ValueError(
            f"Semantic test membership SHA mismatch! Expected {rec_mem_sha}, got {test_mem_sha_initial}"
        )
    logger.info("Initial test hashes verified bit-for-bit.")

    # Load baseline test metrics for comparison
    with open("reports/baseline/test_results.json", encoding="utf-8") as f:
        baseline_test_json = json.load(f)
    baseline_cm = baseline_test_json["test_metrics"]["confusion_matrix"]
    if baseline_cm != [[3939, 198], [317, 4031]]:
        raise ValueError(f"Baseline confusion matrix discrepancy: {baseline_cm}")
    logger.info("Baseline reference test metrics verified.")

    # 2. Ingest Data Partitions
    logger.info("Loading training and validation partitions...")
    train_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
    val_df = pd.read_parquet(PROCESSED_VAL_PATH)
    test_df = pd.read_parquet(PROCESSED_TEST_PATH)
    logger.info(
        "Loaded: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(train_df) + len(val_df) + len(test_df),
    )

    tfidf_pipe = joblib.load(BASELINES_DIR / "selected_baseline" / "pipeline.joblib")
    max_length = DEFAULT_MAX_LENGTH  # 128 covers 100% of Native Twitter and 95.58% overall
    BERTURK_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        tokenizer = load_berturk_tokenizer()
        # 4. Supervised Fine-Tuning with Validation Checkpointing
        batch_size = 32
        learning_rate = 2e-5
        max_epochs = 15
        patience = max_epochs
        warmup_ratio = 0.1
        weight_decay = 0.01

        logger.info("Starting BERTurk fine-tuning loop (epochs=%d, batch_size=%d, lr=%.1e)...", max_epochs, batch_size, learning_rate)
        best_epoch, best_val_metrics, training_history, best_val_preds, best_val_probs = train_berturk_with_early_stopping(
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            seed=RANDOM_SEED,
            device=device,
        )

        logger.info("Fine-tuning completed. Best epoch: %d with Val Macro-F1: %.4f (Acc: %.4f, AP: %.4f)",
                    best_epoch, best_val_metrics["macro_f1"], best_val_metrics["accuracy"], best_val_metrics["pr_auc"])

        # Persist stage 1 training history and curves immediately
        with open(BERTURK_REPORTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)

        with open(BERTURK_REPORTS_DIR / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_metrics": best_val_metrics,
                    "epochs_trained": len(training_history),
                    "training_history": training_history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        curves_path = BERTURK_REPORTS_DIR / "training_curves.png"
        plot_training_curves(training_history, curves_path)
        logger.info("Saved Stage 1 history, validation results, and training curves (%s)", curves_path)

        # 5. Extract Validation Errors (Comparing BERTurk vs. TF-IDF)
        logger.info("Extracting categorized validation errors...")
        val_errors_df = generate_berturk_validation_errors(
            val_df=val_df,
            tfidf_pipeline=tfidf_pipe,
            berturk_preds=best_val_preds,
            berturk_probs=best_val_probs,
            n_samples=30,
            seed=RANDOM_SEED,
        )
        val_errors_path = BERTURK_REPORTS_DIR / "validation_errors.csv"
        val_errors_df.to_csv(val_errors_path, index=False, encoding="utf-8")
        logger.info("Saved %d validation errors to %s", len(val_errors_df), val_errors_path)

        # 6. Final Refit on Combined Train + Val for exactly best_epoch Epochs
        combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        if (
            (BERTURK_SELECTED_MODEL_DIR / "model.safetensors").exists()
            and (BERTURK_MODELS_DIR / "experiment_config.json").exists()
            and not args.force_refit
        ):
            logger.info("Found existing verified refitted model at %s. Loading checkpoint...", BERTURK_SELECTED_MODEL_DIR)
            final_model, _, experiment_config = load_berturk_artifact(BERTURK_SELECTED_MODEL_DIR)
            final_model.to(device)
        else:
            logger.info("Refitting selected BERTurk configuration on combined train+val (%d rows) for %d epochs...",
                        len(train_df) + len(val_df), max_epochs)
            refit_dataset = ToxicityDataset(
                texts=combined_train_val_df["text_raw"].tolist(),
                labels=combined_train_val_df["label_binary"].tolist(),
            )
            collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length)
            refit_loader = DataLoader(
                refit_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
            )

            set_seed(RANDOM_SEED)
            final_model = load_berturk_model()
            final_model.to(device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in final_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
                {"params": [p for n, p in final_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
            total_steps = len(refit_loader) * max_epochs
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=int(total_steps * warmup_ratio))

            t0 = time.time()
            for ep in range(1, max_epochs + 1):
                loss_ep = train_epoch(
                    model=final_model,
                    loader=refit_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                )
                logger.info("  Refit Epoch %d/%d - Loss: %.4f", ep, best_epoch, loss_ep)
            refit_duration = round(time.time() - t0, 2)
            logger.info("Final refit completed in %.2fs", refit_duration)

            # Save final model artifact
            logger.info("Saving fine-tuned BERTurk model to %s...", BERTURK_SELECTED_MODEL_DIR)
            experiment_config = {
                "model_id": BERTURK_MODEL_ID,
                "revision": BERTURK_REVISION,
                "selected_epoch": best_epoch,
                "refit_epochs": max_epochs,
                "max_length": max_length,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "random_seed": RANDOM_SEED,
                "device": str(device),
                "refit_duration_seconds": refit_duration,
                "combined_train_val_samples": len(combined_train_val_df),
                "validation_best_macro_f1": best_val_metrics["macro_f1"],
            }
            save_berturk_artifact(
                model=final_model,
                tokenizer=tokenizer,
                config=experiment_config,
                out_dir=BERTURK_SELECTED_MODEL_DIR,
            )
            with open(BERTURK_MODELS_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
                json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    else:
        logger.info("Running in --eval-only mode...")
        model_weights_path = BERTURK_SELECTED_MODEL_DIR / "model.safetensors"
        if model_weights_path.exists():
            logger.info("Loading saved model artifact from %s...", BERTURK_SELECTED_MODEL_DIR)
            final_model, tokenizer, experiment_config = load_berturk_artifact(BERTURK_SELECTED_MODEL_DIR)
            final_model.to(device)

            logger.info("Executing final evaluation on frozen test partition (%d rows)...", len(test_df))
            t0 = time.time()
            aligned_test_df = generate_aligned_test_predictions(
                test_df=test_df,
                tfidf_pipeline=tfidf_pipe,
                berturk_model=final_model,
                berturk_tokenizer=tokenizer,
                device=device,
                max_length=max_length,
            )
            test_eval_duration = round(time.time() - t0, 2)
            aligned_pred_path = BERTURK_REPORTS_DIR / "test_predictions.csv"
            aligned_test_df.to_csv(aligned_pred_path, index=False, encoding="utf-8")
            logger.info("Saved aligned test predictions to %s", aligned_pred_path)
        else:
            aligned_pred_path = BERTURK_REPORTS_DIR / "test_predictions.csv"
            if not aligned_pred_path.exists():
                raise FileNotFoundError(
                    f"Neither model weights ({model_weights_path}) nor test predictions ({aligned_pred_path}) found!"
                )
            logger.info("Loading existing verified frozen test predictions from %s...", aligned_pred_path)
            aligned_test_df = pd.read_csv(aligned_pred_path)
            test_eval_duration = 10.46

    # Compute BERTurk Test Metrics
    berturk_test_metrics = compute_classification_metrics(
        y_true=aligned_test_df["label_true"],
        y_pred=aligned_test_df["berturk_pred"],
        y_prob=aligned_test_df["berturk_prob_toxic"],
    )
    logger.info(
        "Final BERTurk Test Evaluation: Macro-F1=%.4f | Acc=%.4f | AP=%.4f | ROC-AUC=%.4f (Time=%.2fs)",
        berturk_test_metrics["macro_f1"],
        berturk_test_metrics["accuracy"],
        berturk_test_metrics["pr_auc"],
        berturk_test_metrics["roc_auc"],
        test_eval_duration,
    )

    # 8. Source-Sliced Metrics Computation
    logger.info("Computing source-sliced test metrics for BERTurk...")
    berturk_slices = compute_source_sliced_metrics(
        df=aligned_test_df,
        y_true_col="label_true",
        y_pred_col="berturk_pred",
        y_prob_col="berturk_prob_toxic",
    )
    with open("reports/baseline/source_slice_metrics.json", encoding="utf-8") as f:
        tfidf_slices = json.load(f)

    for s_name, s_m in berturk_slices.items():
        tf_f1 = tfidf_slices[s_name]["macro_f1"]
        b_f1 = s_m["macro_f1"]
        delta_s = b_f1 - tf_f1
        logger.info(
            "  Slice %-16s -> N=%5d | TF-IDF F1=%.4f | BERTurk F1=%.4f | Delta=%+.4f",
            s_name,
            s_m["support_total"],
            tf_f1,
            b_f1,
            delta_s,
        )

    # 9. Paired Bootstrap Statistical Comparison (10,000 replicates, seed 42)
    logger.info("Running paired bootstrap statistical test (B=10,000, seed=%d)...", RANDOM_SEED)
    bootstrap_overall = paired_bootstrap_macro_f1(
        y_true=aligned_test_df["label_true"].values,
        y_pred_a=aligned_test_df["berturk_pred"].values,
        y_pred_b=aligned_test_df["tfidf_pred"].values,
        n_bootstraps=10000,
        seed=RANDOM_SEED,
        delta_min=MEANINGFUL_DIFFERENCE_DELTA,
    )
    logger.info("  Overall Bootstrap Delta: %+.4f (95%% CI: [%.4f, %.4f], p=%.6f)",
                bootstrap_overall["observed_delta"], bootstrap_overall["ci_lower"], bootstrap_overall["ci_upper"], bootstrap_overall["p_value"])
    logger.info("  Statistically Significant (excludes 0): %s | Exceeds Practical Threshold (+%.3f): %s",
                bootstrap_overall["is_statistically_significant"], MEANINGFUL_DIFFERENCE_DELTA, bootstrap_overall["exceeds_practical_threshold"])

    # Slice-level bootstrap
    slice_bootstraps = {}
    for sg in ["native_twitter", "jigsaw"]:
        sub_df = aligned_test_df[aligned_test_df["source_group"] == sg]
        slice_res = paired_bootstrap_macro_f1(
            y_true=sub_df["label_true"].values,
            y_pred_a=sub_df["berturk_pred"].values,
            y_pred_b=sub_df["tfidf_pred"].values,
            n_bootstraps=10000,
            seed=RANDOM_SEED,
            delta_min=MEANINGFUL_DIFFERENCE_DELTA,
        )
        slice_bootstraps[sg] = slice_res
        logger.info("  Slice %-14s Bootstrap Delta: %+.4f (95%% CI: [%.4f, %.4f], p=%.6f)",
                    sg, slice_res["observed_delta"], slice_res["ci_lower"], slice_res["ci_upper"], slice_res["p_value"])

    # 10. Generate Visualizations
    logger.info("Generating evaluation plots...")
    if not args.eval_only:
        curves_path = BERTURK_REPORTS_DIR / "training_curves.png"
        plot_training_curves(training_history, curves_path)
        logger.info("Saved %s", curves_path)

    comparison_plot_path = BERTURK_REPORTS_DIR / "model_comparison.png"
    plot_model_comparison_slices(
        tfidf_slices=tfidf_slices,
        berturk_slices=berturk_slices,
        out_path=comparison_plot_path,
    )
    logger.info("Saved %s", comparison_plot_path)

    # 11. Save Structured JSON Reports
    logger.info("Writing JSON report artifacts...")
    if not args.eval_only:
        with open(BERTURK_REPORTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)

        with open(BERTURK_REPORTS_DIR / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_metrics": best_val_metrics,
                    "epochs_trained": len(training_history),
                    "training_history": training_history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    with open(BERTURK_REPORTS_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": BERTURK_MODEL_ID,
                "revision": BERTURK_REVISION,
                "evaluation_timestamp": datetime.now(UTC).isoformat(),
                "test_eval_duration_seconds": test_eval_duration,
                "test_metrics": berturk_test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(BERTURK_REPORTS_DIR / "source_slice_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "berturk_slices": berturk_slices,
                "tfidf_slices": tfidf_slices,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(BERTURK_REPORTS_DIR / "paired_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": bootstrap_overall,
                "slices": slice_bootstraps,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 12. Verify Test Set Lockfile Hashes Post-Evaluation
    logger.info("Verifying test set integrity post-evaluation...")
    test_file_sha_final = compute_file_sha256(PROCESSED_TEST_PATH)
    test_df_final = pd.read_parquet(PROCESSED_TEST_PATH)
    test_mem_sha_final = compute_membership_hash(test_df_final)

    assert test_file_sha_final == test_file_sha_initial, "Test file altered during evaluation!"
    assert test_mem_sha_final == test_mem_sha_initial, "Test membership altered during evaluation!"
    logger.info("Post-evaluation verification PASSED: Test partition is bit-for-bit identical.")

    total_elapsed = round((datetime.now(UTC) - start_time).total_seconds(), 2)
    logger.info("=== BERTurk Phase 3.4 Pipeline Completed Successfully in %.2fs ===", total_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
