"""Tests for BERTurk dataset formatting, dynamic padding, paired bootstrap, and comparison routines."""

from __future__ import annotations

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from toxicity_detection.evaluation.bootstrap import (
    _fast_macro_f1,
    paired_bootstrap_macro_f1,
)
from toxicity_detection.models.berturk import (
    DynamicPaddingCollator,
    ToxicityDataset,
    load_berturk_artifact,
    save_berturk_artifact,
)
from toxicity_detection.training.seed import set_seed


def test_reproducible_seeding():
    """Verify that set_seed ensures reproducible random generation."""
    set_seed(42)
    val_a = np.random.rand(5)
    tensor_a = torch.randn(5)

    set_seed(42)
    val_b = np.random.rand(5)
    tensor_b = torch.randn(5)

    np.testing.assert_array_equal(val_a, val_b)
    torch.testing.assert_close(tensor_a, tensor_b)


def test_fast_macro_f1_against_sklearn():
    """Verify that _fast_macro_f1 exactly matches sklearn f1_score."""
    from sklearn.metrics import f1_score

    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

    fast_score = _fast_macro_f1(y_true, y_pred)
    expected_score = f1_score(y_true, y_pred, average="macro")

    assert np.isclose(fast_score, expected_score)


def test_paired_bootstrap_significance():
    """Verify that paired_bootstrap_macro_f1 correctly computes confidence intervals and significance."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 20)
    # Model A is perfect
    y_pred_a = y_true.copy()
    # Model B is mostly wrong
    y_pred_b = 1 - y_true.copy()

    res = paired_bootstrap_macro_f1(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        n_bootstraps=500,
        seed=42,
        delta_min=0.020,
    )

    assert res["observed_delta"] > 0.5
    assert res["ci_lower"] > 0.0
    assert res["is_statistically_significant"] is True
    assert res["exceeds_practical_threshold"] is True
    assert res["both_criteria_met"] is True


def test_paired_bootstrap_null_case():
    """Verify that identical predictions yield observed delta 0 and non-significance."""
    y_true = np.array([0, 1, 0, 1] * 25)
    y_pred = y_true.copy()

    res = paired_bootstrap_macro_f1(
        y_true=y_true,
        y_pred_a=y_pred,
        y_pred_b=y_pred,
        n_bootstraps=200,
        seed=42,
        delta_min=0.020,
    )

    assert res["observed_delta"] == 0.0
    assert res["is_statistically_significant"] is False
    assert res["exceeds_practical_threshold"] is False


def test_dynamic_padding_collator_and_dataset(tmp_path):
    """Verify dynamic padding collator outputs correct tensor shapes without full downloads."""
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbu\nbir\ntest\nmetnidir\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=False)

    texts = ["bu bir test", "bu bir test metnidir"]
    labels = [0, 1]

    dataset = ToxicityDataset(texts=texts, labels=labels)
    assert len(dataset) == 2
    assert dataset[0] == ("bu bir test", 0)

    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=128)
    batch = collator([dataset[0], dataset[1]])

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 2
    # Padded dynamically to the max length in batch
    assert (
        batch["input_ids"].shape[1] == 6
    )  # [CLS] bu bir test metnidir [SEP] = 6 tokens
    assert batch["labels"].tolist() == [0, 1]


def test_mock_bert_model_serialization(tmp_path):
    """Verify model save and load roundtrip with a lightweight synthetic BertConfig."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)

    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntest\n", encoding="utf-8"
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab_file))

    out_dir = tmp_path / "saved_model"
    exp_config = {"architecture": "bert_tiny", "max_length": 64}
    save_berturk_artifact(model, tokenizer, exp_config, out_dir)

    loaded_model, loaded_tokenizer, loaded_config = load_berturk_artifact(
        out_dir, device="cpu"
    )

    assert loaded_config["architecture"] == "bert_tiny"
    assert len(loaded_tokenizer) == len(tokenizer)

    model.eval()
    dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        out_orig = model(dummy_input).logits
        out_loaded = loaded_model(dummy_input).logits
    torch.testing.assert_close(out_orig, out_loaded)
