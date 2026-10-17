"""Unit tests for CTC sequence length validation."""

from __future__ import annotations

import pytest
import torch

from thai_audio_jepa_asr.models.temporal_adapter import validate_ctc_feasibility


def test_ctc_feasibility_valid():
    input_lengths = torch.tensor([64, 64, 64])
    target_lengths = torch.tensor([25, 40, 60])
    stats = validate_ctc_feasibility(input_lengths, target_lengths, raise_on_failure=True)
    assert stats["is_feasible"] is True
    assert stats["num_violations"] == 0
    assert stats["violation_rate"] == 0.0


def test_ctc_feasibility_invalid_detection():
    # Audio-JEPA unmodified has T=16, which fails for long transcripts
    input_lengths = torch.tensor([16, 16, 16])
    target_lengths = torch.tensor([10, 25, 50])  # 2 violations (25 > 16, 50 > 16)
    stats = validate_ctc_feasibility(input_lengths, target_lengths, raise_on_failure=False)
    assert stats["is_feasible"] is False
    assert stats["num_violations"] == 2
    assert stats["num_valid"] == 1


def test_ctc_feasibility_raises_error():
    input_lengths = torch.tensor([16])
    target_lengths = torch.tensor([30])
    with pytest.raises(ValueError, match="CTC Feasibility Warning"):
        validate_ctc_feasibility(input_lengths, target_lengths, raise_on_failure=True)
