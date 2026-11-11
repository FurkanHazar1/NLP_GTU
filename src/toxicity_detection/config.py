"""Configuration constants and paths for toxicity detection research."""

from __future__ import annotations

from pathlib import Path

# Base repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# Raw dataset files
OVERFIT_GM_RAW_PATH = RAW_DIR / "overfit_gm" / "turkish_toxic_language.csv"
KAGGLE_TWEETS_RAW_PATH = RAW_DIR / "kaggle_tweets" / "Veri_Seti_10000.xlsx"

# Expected SHA-256 checksums
OVERFIT_GM_EXPECTED_SHA256 = (
    "436aa9250d21d0617cb0aafdf77cad9c1aa85642d2b3fd1cd9d1aedab2d3fbc6"
)
KAGGLE_TWEETS_EXPECTED_SHA256 = (
    "b713c8efc748134c417d250bb7c27bd4ab652abb1955af2495108ccfdb41c22a"
)

# Processed output paths
PROCESSED_ALL_PATH = PROCESSED_DIR / "all.parquet"
PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train.parquet"
PROCESSED_VAL_PATH = PROCESSED_DIR / "val.parquet"
PROCESSED_TEST_PATH = PROCESSED_DIR / "test.parquet"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
TEST_CHECKSUM_PATH = PROCESSED_DIR / "test_checksum.sha256"
TEST_MEMBERSHIP_PATH = PROCESSED_DIR / "test_membership.sha256"

# Interim output paths
QUARANTINE_CONFLICTS_PATH = INTERIM_DIR / "quarantine_conflicts.parquet"

# Baseline models and reports paths
MODELS_DIR = REPO_ROOT / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
REPORTS_DIR = REPO_ROOT / "reports"
BASELINE_REPORT_DIR = REPORTS_DIR / "baseline"

# BERTurk models and reports paths
BERTURK_MODELS_DIR = MODELS_DIR / "berturk"
BERTURK_SELECTED_MODEL_DIR = BERTURK_MODELS_DIR / "selected_model"
BERTURK_REPORTS_DIR = REPORTS_DIR / "berturk"

# Statistical evaluation thresholds
MEANINGFUL_DIFFERENCE_DELTA = 0.020

# Random seed and split configuration
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}

# Source group mappings
SOURCE_GROUP_MAPPING = {
    "told": "native_twitter",
    "kaggle_tweets": "native_twitter",
    "bully": "native_twitter",
    "jigsaw": "jigsaw",
    "troff": "other_sources",
    "offenseval": "other_sources",
}

# Exact per-stratum target split allocations
STRATUM_SPLIT_COUNTS = {
    "native_twitter_0": {"train": 16188, "val": 3238, "test": 2158},
    "native_twitter_1": {"train": 20684, "val": 4137, "test": 2758},
    "jigsaw_0": {"train": 14788, "val": 2958, "test": 1972},
    "jigsaw_1": {"train": 11914, "val": 2383, "test": 1588},
    "other_sources_0": {"train": 56, "val": 11, "test": 7},
    "other_sources_1": {"train": 20, "val": 4, "test": 2},
}

# Quarantined unresolved binary conflict raw texts (prior to canonicalization)
QUARANTINED_RAW_TEXTS = [
    "bize karşı savunma oynayıp 1 puan aldığı için sevinen başakşehir gs ye 5 atıp şov yapıyor  siz neyin kafasını yaş",
    "milletin çocuğu nasıl bu kadar uslu sessiz akıllı olabiliyor benimki tam bir fırlama eslemin peşinde koşmaktan bir",
]
