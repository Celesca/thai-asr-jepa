"""Unit tests for Thai character CTC tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer


def test_tokenizer_encode_decode():
    tokenizer = ThaiCharTokenizer()
    text = "สวัสดีชาวโลก 123"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_special_tokens():
    tokenizer = ThaiCharTokenizer()
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.space_token_id == 2


def test_ctc_greedy_collapse():
    tokenizer = ThaiCharTokenizer()
    # Simulate greedy tokens with repeats and blank tokens (0)
    # Target: "กา" -> [k_id, a_id]
    k_id = tokenizer.vocab["ก"]
    a_id = tokenizer.vocab["า"]

    raw_tokens = [0, k_id, k_id, 0, 0, a_id, a_id, 0]
    decoded = tokenizer.decode_ctc(raw_tokens)
    assert decoded == "กา"

    # Tensor input
    tensor_tokens = torch.tensor(raw_tokens)
    assert tokenizer.decode_ctc(tensor_tokens) == "กา"


def test_save_load_vocab():
    tokenizer = ThaiCharTokenizer()
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_path = Path(tmp_dir) / "test_vocab.json"
        tokenizer.save(vocab_path)
        assert vocab_path.is_file()

        loaded_tok = ThaiCharTokenizer.load(vocab_path)
        assert loaded_tok.vocab == tokenizer.vocab
        assert loaded_tok.vocab_size == tokenizer.vocab_size
