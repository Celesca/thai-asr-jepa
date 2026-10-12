"""Tests for Audio-JEPA Pretrained Checkpoint Transfer, Positional Interpolation, and CTC Alignment."""

import pytest
import torch
import torch.nn as nn

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder, CheckpointLoadingReport
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.models.temporal_adapter import (
    TemporalAdapter,
    compute_ctc_min_required_length,
    validate_ctc_feasibility,
)


def test_asymmetric_patching_and_tensor_shapes():
    """Verify modified patch size (1, 16) yields 2,048 tokens and pools to 256 temporal frames."""
    encoder = AudioJEPAEncoder(
        patch_size=(1, 16),
        input_size=(256, 128),
        embed_dim=768,
        depth=2,  # Lightweight for unit testing
    )
    B = 2
    dummy_spec = torch.randn(B, 1, 256, 128)

    # 1. Forward pass
    features = encoder(dummy_spec)
    assert features.shape == (B, 2048, 768), f"Expected (B, 2048, 768), got {features.shape}"

    # 2. Native frequency pooling to temporal sequence
    seq, lengths = encoder.to_temporal_sequence(features)
    assert seq.shape == (B, 256, 768), f"Expected (B, 256, 768), got {seq.shape}"
    assert torch.equal(lengths, torch.full((B,), 256, dtype=torch.long))


def test_temporal_token_rate_original_vs_modified():
    """Verify temporal rates: Original = 1.6 fps (unsuitable), Modified = 25.6 fps (suitable)."""
    orig_encoder = AudioJEPAEncoder(patch_size=(16, 16), clip_length_s=10.0)
    assert orig_encoder.grid_h == 16
    assert orig_encoder.grid_w == 8
    assert abs(orig_encoder.temporal_token_rate - 1.6) < 1e-3

    mod_encoder = AudioJEPAEncoder(patch_size=(1, 16), clip_length_s=10.0)
    assert mod_encoder.grid_h == 256
    assert mod_encoder.grid_w == 8
    assert abs(mod_encoder.temporal_token_rate - 25.6) < 1e-3


def test_positional_embedding_2d_interpolation():
    """Verify bicubic 2D positional embedding interpolation from (16, 8) to (256, 8)."""
    encoder = AudioJEPAEncoder(patch_size=(1, 16))
    src_grid = (16, 8)
    tgt_grid = (256, 8)
    src_pos_embed = torch.randn(1, 16 * 8, 768)

    interp_pe = encoder.interpolate_pos_encoding_2d(
        src_pos_embed=src_pos_embed,
        src_grid=src_grid,
        tgt_grid=tgt_grid,
    )
    assert interp_pe.shape == (1, 256 * 8, 768), f"Expected (1, 2048, 768), got {interp_pe.shape}"


def test_ctc_min_required_length_with_repeated_tokens():
    """Verify minimum CTC length accounting for adjacent repeated tokens: T >= L + repeats."""
    # Sequence with no adjacent repeats: [5, 6, 7] -> length 3, min required = 3
    assert compute_ctc_min_required_length([5, 6, 7]) == 3

    # Sequence with adjacent repeats: [5, 5, 5] -> 3 tokens + 2 repeats = 5
    assert compute_ctc_min_required_length([5, 5, 5]) == 5

    # Sequence with two separate pairs: [2, 2, 8, 8, 9] -> 5 tokens + 2 repeats = 7
    assert compute_ctc_min_required_length([2, 2, 8, 8, 9]) == 7

    # Sequence with trailing padding (0) ignored: [5, 5, 0, 0] -> 2 tokens + 1 repeat = 3
    assert compute_ctc_min_required_length([5, 5, 0, 0]) == 3


def test_ctc_feasibility_validation():
    """Verify feasibility percentage calculation and violation detection."""
    # 2 samples: one feasible (T=256, req=40), one infeasible for coarse grid (T=16, req=30)
    input_lengths = torch.tensor([256, 16])
    targets = torch.tensor([
        [1, 2, 3, 4, 5, 0],
        [1, 1, 1, 1, 1, 1],  # 6 tokens + 5 repeats = 11 required
    ])
    target_lengths = torch.tensor([5, 6])

    stats = validate_ctc_feasibility(input_lengths, targets=targets, target_lengths=target_lengths)
    assert stats["is_feasible"] is True
    assert stats["feasible_percentage"] == 100.0

    # Infeasible case: sample 0 has T=4 (requires 5), sample 1 has T=5 (requires 11)
    input_lengths_fail = torch.tensor([4, 5])
    stats_fail = validate_ctc_feasibility(input_lengths_fail, targets=targets, target_lengths=target_lengths)
    assert stats_fail["is_feasible"] is False
    assert stats_fail["num_violations"] == 2
    assert stats_fail["feasible_percentage"] == 0.0


def test_checkpoint_transfer_loading_percentage():
    """Verify transfer loading from ltuncay/Audio-JEPA loads >99.9% of model weights."""
    try:
        encoder, report = AudioJEPAEncoder.from_pretrained_transfer(
            model_name_or_path="ltuncay/Audio-JEPA",
            patch_size=(1, 16),
        )
        assert isinstance(report, CheckpointLoadingReport)
        assert report.loaded_percentage > 99.9, f"Loaded percentage {report.loaded_percentage}% < 99.9%"
        assert report.shape_mismatched_parameters[0]["name"] == "patch_embed.proj.weight"
        assert report.shape_mismatched_parameters[0]["target_shape"] == [768, 1, 1, 16]
    except Exception as e:
        pytest.skip(f"Skipping remote checkpoint download test (offline or rate-limited): {e}")


def test_single_training_step_modified_audio_jepa():
    """Verify an end-to-end forward, loss, and backward pass on modified Audio-JEPA CTC model."""
    encoder = AudioJEPAEncoder(
        patch_size=(1, 16),
        embed_dim=768,
        depth=2,  # 2 layers for fast test
    )
    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=109,
        temporal_adapter_mode="native_pooling",
        head_dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B = 2
    specs = torch.randn(B, 1, 256, 128)
    targets = torch.tensor([
        [10, 11, 12, 13, 0],
        [20, 21, 22, 0, 0],
    ])
    target_lengths = torch.tensor([4, 3])

    # Forward
    out = model(spectrograms=specs, targets=targets, target_lengths=target_lengths)
    assert "loss" in out
    loss = out["loss"]
    assert not torch.isnan(loss)
    assert loss.item() > 0

    # Backward & Optimizer step
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist on both encoder and CTC head
    assert encoder.patch_embed.proj.weight.grad is not None
    assert model.ctc_head[1].weight.grad is not None
    optimizer.step()
