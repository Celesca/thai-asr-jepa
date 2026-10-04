"""Unit tests for deterministic split assignment and reproducibility."""

from __future__ import annotations

import pytest
from thai_audio_jepa_asr.data.porjai import deterministic_split_bucket, assign_split


def test_split_bucket_determinism():
    uid1 = "thai-central_000123"
    uid2 = "thai-central_999888"

    bucket1_first = deterministic_split_bucket(uid1)
    bucket1_second = deterministic_split_bucket(uid1)
    assert bucket1_first == bucket1_second
    assert 0 <= bucket1_first < 100

    bucket2_first = deterministic_split_bucket(uid2)
    bucket2_second = deterministic_split_bucket(uid2)
    assert bucket2_first == bucket2_second


def test_split_assignment_ratios():
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(1000):
        uid = f"sample_{i:05d}"
        bucket = deterministic_split_bucket(uid)
        split = assign_split(bucket, train_ratio=0.8, val_ratio=0.1)
        counts[split] += 1

    # Over 1000 samples, train should be roughly 800, val roughly 100, test roughly 100
    assert 740 <= counts["train"] <= 860
    assert 70 <= counts["val"] <= 140
    assert 70 <= counts["test"] <= 140
