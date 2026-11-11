"""BERTurk sequence classification model loading, dataset encapsulation, and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

BERTURK_MODEL_ID = "dbmdz/bert-base-turkish-cased"
BERTURK_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
DEFAULT_MAX_LENGTH = 128


class ToxicityDataset(Dataset):
    """PyTorch Dataset for Turkish toxicity text classification."""

    def __init__(self, texts: list[str], labels: list[int] | None = None) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int | None]:
        label = self.labels[idx] if self.labels is not None else None
        return self.texts[idx], label


class DynamicPaddingCollator:
    """Collate function that dynamically pads tokenized sequences to the batch maximum."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, int | None]]) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        has_labels = batch[0][1] is not None

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if has_labels:
            labels = [item[1] for item in batch]
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)

        return encoded


def load_berturk_tokenizer(
    model_id: str = BERTURK_MODEL_ID,
    revision: str = BERTURK_REVISION,
) -> PreTrainedTokenizerBase:
    """Load pretrained BERTurk cased tokenizer.

    Args:
        model_id: Hugging Face model identifier.
        revision: Pinned git commit hash or revision tag.

    Returns:
        Loaded tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        do_lower_case=False,
    )


def load_berturk_model(
    model_id: str = BERTURK_MODEL_ID,
    revision: str = BERTURK_REVISION,
    num_labels: int = 2,
) -> PreTrainedModel:
    """Load pretrained BERTurk model for sequence classification.

    Args:
        model_id: Hugging Face model identifier.
        revision: Pinned git commit hash or revision tag.
        num_labels: Number of classification labels (default: 2 for binary).

    Returns:
        Loaded PyTorch model instance.
    """
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        num_labels=num_labels,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        revision=revision,
    )


def save_berturk_artifact(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: dict[str, Any],
    out_dir: str | Path,
) -> None:
    """Save fine-tuned BERTurk model weights, tokenizer files, and metadata configuration.

    Args:
        model: Trained PyTorch transformer model.
        tokenizer: Associated tokenizer.
        config: Metadata and training parameters dictionary.
        out_dir: Directory where artifacts will be saved.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    with open(path / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_berturk_artifact(
    in_dir: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Load fine-tuned BERTurk model, tokenizer, and metadata from disk.

    Args:
        in_dir: Directory containing saved model files and experiment_config.json.
        device: Target torch device for model parameters.

    Returns:
        Tuple of (model, tokenizer, config_dict).
    """
    path = Path(in_dir)
    config_file = path / "experiment_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing experiment configuration: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()

    return model, tokenizer, config
