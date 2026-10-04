"""Unit tests for forward pass through Audio-JEPA + CTC head."""

from __future__ import annotations

import pytest
import torch

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel


def test_encoder_forward_shape():
    encoder = AudioJEPAEncoder(
        input_size=(256, 128),
        patch_size=(16, 16),
        embed_dim=192,  # Lightweight ViT-Tiny for fast unit tests
        depth=2,
        num_heads=3,
    )
    dummy_input = torch.randn(2, 1, 256, 128)
    out = encoder(dummy_input)
    # 256//16 = 16, 128//16 = 8 -> 128 patches
    assert out.shape == (2, 128, 192)


def test_ctc_model_forward_and_loss():
    encoder = AudioJEPAEncoder(
        input_size=(256, 128),
        patch_size=(16, 16),
        embed_dim=192,
        depth=2,
        num_heads=3,
    )
    vocab_size = 85
    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=vocab_size,
        temporal_adapter_mode="upsample_conv",
        upsample_factor=4,  # 16 * 4 = 64 frames
        pad_token_id=0,
    )

    batch_size = 2
    spec = torch.randn(batch_size, 1, 256, 128)
    # Targets with length 20 (<= 64)
    targets = torch.randint(1, vocab_size, (batch_size, 20))
    target_lengths = torch.tensor([20, 20], dtype=torch.long)

    out = model(spec, targets=targets, target_lengths=target_lengths)

    assert "logits" in out
    assert "loss" in out
    # Shape [B, 64, vocab_size]
    assert out["logits"].shape == (batch_size, 64, vocab_size)
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])
    assert out["loss"].item() > 0.0


def test_encoder_freezing_modes():
    encoder = AudioJEPAEncoder(embed_dim=192, depth=4, num_heads=3)
    # Frozen mode
    frozen_model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=50,
        freeze_encoder=True,
    )
    for p in frozen_model.encoder.parameters():
        assert not p.requires_grad
    for p in frozen_model.ctc_head.parameters():
        assert p.requires_grad

    # Partial unfreeze mode (last 1 block)
    encoder_partial = AudioJEPAEncoder(embed_dim=192, depth=4, num_heads=3)
    partial_model = AudioJEPACTCAudioModel(
        encoder=encoder_partial,
        vocab_size=50,
        freeze_encoder=False,
        unfreeze_last_n_blocks=1,
    )
    for p in partial_model.encoder.blocks[0].parameters():
        assert not p.requires_grad
    for p in partial_model.encoder.blocks[-1].parameters():
        assert p.requires_grad
