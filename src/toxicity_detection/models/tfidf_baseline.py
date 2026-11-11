"""Classical TF-IDF and Logistic Regression baseline pipelines."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def turkish_lowercase_preprocessor(text: str) -> str:
    """Apply Turkish-aware lowercasing to text prior to TF-IDF tokenization.

    Converts dotted uppercase 'İ' to 'i' and dotless uppercase 'I' to 'ı'
    before applying standard lowercase.

    Args:
        text: Input string.

    Returns:
        Turkish-lower-cased string.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.replace("İ", "i").replace("I", "ı").lower()


def build_word_vectorizer(
    preprocessor: Callable[[str], str] | None = turkish_lowercase_preprocessor,
) -> TfidfVectorizer:
    """Build word-level TF-IDF vectorizer (Model A).

    Configuration:
    - analyzer: 'word'
    - ngram_range: (1, 2)
    - min_df: 2
    - sublinear_tf: True
    - norm: 'l2'

    Args:
        preprocessor: Text preprocessor callable.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        preprocessor=preprocessor,
    )


def build_char_vectorizer(
    preprocessor: Callable[[str], str] | None = turkish_lowercase_preprocessor,
) -> TfidfVectorizer:
    """Build character-level TF-IDF vectorizer (Model B).

    Configuration:
    - analyzer: 'char_wb'
    - ngram_range: (3, 5)
    - min_df: 2
    - sublinear_tf: True
    - norm: 'l2'

    Args:
        preprocessor: Text preprocessor callable.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        preprocessor=preprocessor,
    )


def build_combined_vectorizer(
    preprocessor: Callable[[str], str] | None = turkish_lowercase_preprocessor,
) -> FeatureUnion:
    """Build combined word and character TF-IDF feature union (Model C).

    Maintains sparse feature representation without densifying.

    Args:
        preprocessor: Text preprocessor callable.

    Returns:
        Configured FeatureUnion instance.
    """
    return FeatureUnion(
        [
            ("word", build_word_vectorizer(preprocessor=preprocessor)),
            ("char", build_char_vectorizer(preprocessor=preprocessor)),
        ]
    )


def build_tfidf_pipeline(
    representation: str = "combined",
    c_val: float = 1.0,
    preprocessor_type: str = "turkish_lower",
    random_state: int = 42,
) -> Pipeline:
    """Construct an end-to-end TF-IDF + Logistic Regression pipeline.

    Args:
        representation: Feature representation: 'word', 'char', or 'combined'.
        c_val: Inverse regularization parameter C for Logistic Regression.
        preprocessor_type: 'turkish_lower' or 'raw'.
        random_state: Random state for deterministic solver.

    Returns:
        Scikit-learn Pipeline.

    Raises:
        ValueError: If representation or preprocessor_type is invalid.
    """
    if preprocessor_type == "turkish_lower":
        preproc: Callable[[str], str] | None = turkish_lowercase_preprocessor
    elif preprocessor_type == "raw":
        preproc = None
    else:
        raise ValueError(f"Unknown preprocessor_type: {preprocessor_type}")

    if representation == "word":
        feature_extractor = build_word_vectorizer(preprocessor=preproc)
    elif representation == "char":
        feature_extractor = build_char_vectorizer(preprocessor=preproc)
    elif representation == "combined":
        feature_extractor = build_combined_vectorizer(preprocessor=preproc)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    clf = LogisticRegression(
        C=c_val,
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        random_state=random_state,
    )

    return Pipeline([("features", feature_extractor), ("classifier", clf)])


def save_model_artifact(
    pipeline: Pipeline,
    config: dict[str, Any],
    out_dir: str | Path,
) -> None:
    """Serialize fitted model pipeline and metadata configuration.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        config: Metadata and hyperparameters dictionary.
        out_dir: Destination directory.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, path / "pipeline.joblib")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_model_artifact(in_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Deserialize model pipeline and metadata configuration.

    Args:
        in_dir: Directory containing pipeline.joblib and config.json.

    Returns:
        Tuple of (fitted_pipeline, config_dict).

    Raises:
        FileNotFoundError: If pipeline.joblib or config.json is missing.
    """
    path = Path(in_dir)
    pipeline_path = path / "pipeline.joblib"
    config_path = path / "config.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing pipeline artifact: {pipeline_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config artifact: {config_path}")

    pipeline: Pipeline = joblib.load(pipeline_path)
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    return pipeline, config
