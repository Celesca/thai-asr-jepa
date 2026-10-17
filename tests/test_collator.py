"""Unit tests for ASR and SSL batch collators."""

from __future__ import annotations

import pytest
import torch

from thai_audio_jepa_asr.data.collator import ASRDataCollator, SSLMaskCollator


def test_asr_collator_padding():
    collator = ASRDataCollator(pad_token_id=0)
    batch = [
        {
            "id": "utt1",
            "spectrogram": torch.randn(1, 256, 128),
            "tokens": torch.tensor([10, 15, 20]),
            "text": "ข้อความที่หนึ่ง",
        },
        {
            "id": "utt2",
            "spectrogram": torch.randn(1, 256, 128),
            "tokens": torch.tensor([5, 6, 7, 8, 9]),
            "text": "ข้อความที่สองยาวกว่า",
        },
    ]

    out = collator(batch)
    assert out["spectrograms"].shape == (2, 1, 256, 128)
    assert out["targets"].shape == (2, 5)  # Max length is 5
    assert torch.equal(out["target_lengths"], torch.tensor([3, 5]))
    # Verify padding
    assert out["targets"][0, 3].item() == 0
    assert out["targets"][0, 4].item() == 0


def test_ssl_mask_collator():
    collator = SSLMaskCollator(grid_size=(16, 8), nenc=1, npred=2)
    batch = [
        {"id": "utt1", "spectrogram": torch.randn(1, 256, 128)},
        {"id": "utt2", "spectrogram": torch.randn(1, 256, 128)},
    ]

    out = collator(batch)
    assert out["spectrograms"].shape == (2, 1, 256, 128)
    assert len(out["context_masks"]) == 1
    assert len(out["prediction_masks"]) == 2
    # Check that mask indices are within [0, 127]
    assert out["context_masks"][0].max() < 128
    assert out["prediction_masks"][0].max() < 128
