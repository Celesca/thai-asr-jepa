"""Unit tests for Thai text normalization."""

from __future__ import annotations

import pytest
from thai_audio_jepa_asr.preprocessing.text_normalizer import ThaiTextNormalizer


def test_thai_digits_conversion():
    normalizer = ThaiTextNormalizer(digits_policy="thai_to_arabic")
    text = "หมายเลข ๑๒๓๔๕"
    norm = normalizer.normalize(text)
    assert norm == "หมายเลข 12345"


def test_arabic_to_thai_digits():
    normalizer = ThaiTextNormalizer(digits_policy="arabic_to_thai")
    text = "หมายเลข 123"
    norm = normalizer.normalize(text)
    assert norm == "หมายเลข ๑๒๓"


def test_remove_punctuation():
    normalizer = ThaiTextNormalizer(remove_punctuation=True)
    text = "สวัสดีครับ, สบายดีไหม? (ใช่ครับ!)"
    norm = normalizer.normalize(text)
    assert norm == "สวัสดีครับ สบายดีไหม ใช่ครับ"


def test_collapse_whitespace():
    normalizer = ThaiTextNormalizer(collapse_whitespace=True)
    text = "   ทดสอบ    ข้อความ     หลายช่องว่าง   "
    norm = normalizer.normalize(text)
    assert norm == "ทดสอบ ข้อความ หลายช่องว่าง"


def test_remove_zero_width():
    normalizer = ThaiTextNormalizer(remove_zero_width=True)
    text = "คำ\u200bว่า\u200cมี\ufeffช่องว่างแฝง"
    norm = normalizer.normalize(text)
    assert "\u200b" not in norm
    assert "\ufeff" not in norm
    assert norm == "คำว่ามีช่องว่างแฝง"


def test_sara_am_decomposition():
    normalizer = ThaiTextNormalizer()
    # Decomposed NIKHAHIT + SARA AA
    text = "ก\u0E4D\u0E32"
    norm = normalizer.normalize(text)
    # Recomposed to SARA AM
    assert norm == "กำ"
