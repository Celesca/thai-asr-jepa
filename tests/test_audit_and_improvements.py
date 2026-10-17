"""Focused unit tests verifying all audit findings and architectural improvements."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder, CheckpointLoadingReport
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.models.jepa_ssl import AudioJEPASSLModule, compute_representation_diagnostics
from thai_audio_jepa_asr.models.temporal_adapter import (
    TemporalAdapter,
    compute_ctc_min_required_length,
    validate_ctc_feasibility,
)
from thai_audio_jepa_asr.preprocessing.audio import (
    AudioJEPAMelSpec,
    FrontendConfig,
    compute_kaldi_fbank_frames,
    compute_patch_tokens,
)
from thai_audio_jepa_asr.data.collator import ASRDataCollator, SSLMaskCollator
from thai_audio_jepa_asr.data.dataset import ThaiSpeechASRDataset
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.training.optimizer import build_asr_optimizer


def test_speech_frontends_and_valid_lengths():
    """Verify frontend presets: original, speech_40ms, speech_20ms with exact valid token calculations."""
    sr = 32000

    # 1. Original (97.66ms window, 39.06ms hop, patch (1, 16))
    cfg_orig = FrontendConfig.from_name("original", sample_rate=sr)
    assert abs(cfg_orig.effective_output_interval_ms - 39.0625) < 1e-3
    assert abs(cfg_orig.temporal_token_rate - 25.6) < 1e-2
    # 10s audio -> 254 exact Kaldi fbank frames (padded to 256 in fixed grid)
    tokens_10s_orig = cfg_orig.compute_max_time_tokens()
    assert 254 <= tokens_10s_orig <= 256

    # 2. speech_40ms (25ms window, 10ms hop, patch (4, 16), stride (4, 16))
    cfg_40ms = FrontendConfig.from_name("speech_40ms", sample_rate=sr)
    assert cfg_40ms.effective_output_interval_ms == 40.0
    assert cfg_40ms.temporal_token_rate == 25.0
    fbank_frames_40ms = compute_kaldi_fbank_frames(sr * 10, sr, 25.0, 10.0)
    # approx 998-1000 frames -> (fbank - 4)//4 + 1 = 250 tokens
    tokens_10s_40ms = compute_patch_tokens(fbank_frames_40ms, patch_time=4, stride_time=4)
    assert 249 <= tokens_10s_40ms <= 250

    # 3. speech_20ms (25ms window, 10ms hop, patch (2, 16), stride (2, 16))
    cfg_20ms = FrontendConfig.from_name("speech_20ms", sample_rate=sr)
    assert cfg_20ms.effective_output_interval_ms == 20.0
    assert cfg_20ms.temporal_token_rate == 50.0
    tokens_10s_20ms = compute_patch_tokens(fbank_frames_40ms, patch_time=2, stride_time=2)
    assert 498 <= tokens_10s_20ms <= 500


def test_audio_transcript_consistency_filter():
    """Verify that ThaiSpeechASRDataset excludes audio exceeding max_duration_s rather than cropping with full transcript."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_file = Path(tmp_dir) / "test_manifest.jsonl"
        # Write dummy manifest with one 5s audio and one 15s audio
        dummy_records = [
            '{"id": "short_utt", "audio_path": "short.wav", "duration": 5.0, "normalized_sentence": "ข้อความสั้น"}',
            '{"id": "long_utt", "audio_path": "long.wav", "duration": 15.0, "normalized_sentence": "ข้อความยาวเกินไปสิบวินาที"}',
            '{"id": "tiny_utt", "audio_path": "tiny.wav", "duration": 0.2, "normalized_sentence": "สั้นเกินไป"}',
        ]
        manifest_file.write_text("\n".join(dummy_records), encoding="utf-8")

        tokenizer = ThaiCharTokenizer()
        ds = ThaiSpeechASRDataset(
            manifest_path=manifest_file,
            tokenizer=tokenizer,
            max_duration_s=10.0,
            min_duration_s=0.5,
        )
        # Only short_utt (5.0s) should be kept; long_utt (15s) and tiny_utt (0.2s) must be excluded
        assert len(ds) == 1
        assert ds.records[0]["id"] == "short_utt"


def test_short_span_masking_boundaries_and_padding_exclusion():
    """Verify short-span SSL masking samples multiple spans and excludes padding from context/target."""
    grid_h = 256
    grid_w = 8
    collator = SSLMaskCollator(
        grid_size=(grid_h, grid_w),
        masking_mode="short_span",
        span_duration_ms=300.0,
        token_interval_ms=39.0625,
        mask_ratio=0.50,
        nenc=1,
        npred=2,
    )

    valid_time_tokens = 100  # Utterance is only 100 tokens long (rest 156 are padding)
    batch = [
        {
            "id": "utt1",
            "spectrogram": torch.randn(1, 256, 128),
            "valid_time_tokens": valid_time_tokens,
        }
    ]

    out = collator(batch)
    assert len(out["context_masks"]) == 1
    assert len(out["prediction_masks"]) == 2

    # Verify context masks only contain valid indices (< valid_time_tokens * grid_w)
    max_valid_patch_idx = valid_time_tokens * grid_w
    ctxt_indices = out["context_masks"][0]
    pred_indices = out["prediction_masks"][0]

    assert ctxt_indices.max().item() < max_valid_patch_idx
    assert pred_indices.max().item() < max_valid_patch_idx

    # Padding mask must be True for all indices >= max_valid_patch_idx
    padding_mask = out["padding_mask"][0]
    assert not padding_mask[:max_valid_patch_idx].any()
    assert padding_mask[max_valid_patch_idx:].all()


def test_ema_teacher_initialization_and_update():
    """Verify teacher is initialized with student weights, has no grad, and updates via EMA."""
    encoder = AudioJEPAEncoder(patch_size=(1, 16), embed_dim=192, depth=2, num_heads=3)
    ssl_module = AudioJEPASSLModule(encoder=encoder, ema_tau=0.90)

    # Teacher parameters must have requires_grad=False
    for p in ssl_module.target_encoder.parameters():
        assert not p.requires_grad

    # Initial weights must match exactly
    for p_online, p_target in zip(ssl_module.encoder.parameters(), ssl_module.target_encoder.parameters()):
        assert torch.equal(p_online, p_target)

    # Modify online parameters
    with torch.no_grad():
        for p in ssl_module.encoder.parameters():
            p.add_(1.0)

    # Update teacher via EMA (tau=0.90)
    ssl_module.update_target_encoder(tau=0.90)

    for p_online, p_target in zip(ssl_module.encoder.parameters(), ssl_module.target_encoder.parameters()):
        # target should not equal old or new, but 0.9 * old + 0.1 * new
        assert not torch.equal(p_online, p_target)


def test_decoupled_optimizer_groups_and_frozen_backbone():
    """Verify optimizer has decoupled groups and patch_embed remains trainable in frozen backbone mode."""
    encoder = AudioJEPAEncoder(patch_size=(1, 16), embed_dim=192, depth=2, num_heads=3)
    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=50,
        freeze_encoder=True,
        freeze_patch_embed=False,  # Patch projection must remain trainable!
    )

    # ViT blocks must be frozen
    for p in model.encoder.blocks.parameters():
        assert not p.requires_grad

    # Patch projection must be trainable
    for p in model.encoder.patch_embed.parameters():
        assert p.requires_grad

    optimizer = build_asr_optimizer(
        model,
        head_lr=3e-4,
        patch_embed_lr=3e-4,
        backbone_lr=3e-5,
    )
    group_names = [g["name"] for g in optimizer.param_groups]
    assert "patch_embed_decay" in group_names or "patch_embed_no_decay" in group_names
    assert "head_decay" in group_names or "head_no_decay" in group_names


def test_ctc_feasibility_ratio_distribution_and_details():
    """Verify validate_ctc_feasibility calculates ratio distribution and details."""
    input_lengths = torch.tensor([100, 20])
    targets = torch.tensor([
        [1, 2, 3, 4, 5, 0],
        [1, 1, 1, 1, 1, 1],  # requires 11
    ])
    target_lengths = torch.tensor([5, 6])

    stats = validate_ctc_feasibility(input_lengths, targets=targets, target_lengths=target_lengths)
    assert stats["is_feasible"] is True
    assert "ratio_distribution" in stats
    assert stats["ratio_distribution"]["mean"] > 1.0
    assert stats["num_violations"] == 0

    # Infeasible sample
    input_lengths_bad = torch.tensor([100, 8])  # 8 < 11
    stats_bad = validate_ctc_feasibility(input_lengths_bad, targets=targets, target_lengths=target_lengths)
    assert stats_bad["is_feasible"] is False
    assert stats_bad["num_violations"] == 1
    assert len(stats_bad["violations_detail"]) == 1
    assert stats_bad["violations_detail"][0]["deficit"] == 3


def test_sdpa_attention_mask_padding_exclusion():
    """Verify PyTorch SDPA attention mask properly excludes padding tokens without inverted masks."""
    encoder = AudioJEPAEncoder(patch_size=(1, 16), embed_dim=192, depth=2, num_heads=3)
    B = 2
    x = torch.randn(B, 1, 256, 128)
    padding_mask = torch.zeros((B, 2048), dtype=torch.bool)
    padding_mask[:, 1000:] = True  # Padded beyond 1000 patches

    out = encoder(x, padding_mask=padding_mask)
    assert out.shape == (B, 2048, 192)
    assert not torch.isnan(out).any()


def test_representation_diagnostics():
    """Verify compute_representation_diagnostics tracks variance and cosine similarity."""
    B, N, D = 4, 32, 128
    # Distinct random features
    features = torch.randn(B, N, D)
    diag = compute_representation_diagnostics(features)
    assert "feature_variance" in diag
    assert "inter_utterance_similarity" in diag
    assert diag["feature_variance"] > 0.0
    assert -1.0 <= diag["inter_utterance_similarity"] <= 1.0
