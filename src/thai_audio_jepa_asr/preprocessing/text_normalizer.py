"""Configurable Thai text normalizer for ASR."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Thai digits to Arabic digits mapping
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI_TO_ARABIC_MAP = str.maketrans(THAI_DIGITS, ARABIC_DIGITS)
ARABIC_TO_THAI_MAP = str.maketrans(ARABIC_DIGITS, THAI_DIGITS)

# Common Thai punctuation and English punctuation
PUNCTUATION_PATTERN = re.compile(r"[\.,\?!:;\"\(\)\[\]\{\}«»“”‘’—–_\-–…\/\\\|\*&\^%\$#@~`]")


class ThaiTextNormalizer:
    """Configurable Thai text normalizer.

    Supports Unicode normalization, digit mapping, punctuation handling,
    and whitespace collapsing without silently dropping meaningful content.
    """

    def __init__(
        self,
        unicode_form: str = "NFKC",
        remove_punctuation: bool = True,
        digits_policy: str = "thai_to_arabic",
        collapse_whitespace: bool = True,
        lowercase: bool = True,
        remove_zero_width: bool = True,
    ) -> None:
        """Initialize the normalizer with configuration options.

        Args:
            unicode_form: 'NFKC' or 'NFC' (default: 'NFKC').
            remove_punctuation: Whether to remove punctuation marks.
            digits_policy: One of 'thai_to_arabic', 'keep', 'remove', 'arabic_to_thai'.
            collapse_whitespace: Whether to collapse multiple whitespaces into a single space.
            lowercase: Whether to convert Latin characters to lowercase.
            remove_zero_width: Whether to remove zero-width spaces / non-joiners.
        """
        self.unicode_form = unicode_form
        self.remove_punctuation = remove_punctuation
        self.digits_policy = digits_policy
        self.collapse_whitespace = collapse_whitespace
        self.lowercase = lowercase
        self.remove_zero_width = remove_zero_width

    def to_dict(self) -> Dict[str, Any]:
        """Export normalizer settings as a serializable dict."""
        return {
            "unicode_form": self.unicode_form,
            "remove_punctuation": self.remove_punctuation,
            "digits_policy": self.digits_policy,
            "collapse_whitespace": self.collapse_whitespace,
            "lowercase": self.lowercase,
            "remove_zero_width": self.remove_zero_width,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ThaiTextNormalizer:
        """Instantiate normalizer from configuration dict."""
        return cls(**d)

    def normalize(self, text: str) -> str:
        """Normalize Thai transcript text according to configuration.

        Args:
            text: Raw input string.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        # 1. Unicode normalization
        if self.unicode_form:
            text = unicodedata.normalize(self.unicode_form, text)

        # 2. Lowercase latin characters
        if self.lowercase:
            text = text.lower()

        # 3. Remove zero-width spaces (U+200B, U+FEFF, etc.)
        if self.remove_zero_width:
            text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # 4. Standardize Thai vowel order (e.g. sara e + sara a = sara ae)
        # Fix repetitive vowels and tone marks using standard thai character patterns
        # Duplicate tone marks or misplaced tone marks:
        text = re.sub(r"([\u0E48-\u0E4B])[\u0E48-\u0E4B]+", r"\1", text)
        # Sara am decomposition check: NIKHAHIT (\u0E4D) + SARA AA (\u0E32) -> SARA AM (\u0E33)
        text = text.replace("\u0E4D\u0E32", "\u0E33")

        # 5. Digits policy
        if self.digits_policy == "thai_to_arabic":
            text = text.translate(THAI_TO_ARABIC_MAP)
        elif self.digits_policy == "arabic_to_thai":
            text = text.translate(ARABIC_TO_THAI_MAP)
        elif self.digits_policy == "remove":
            text = re.sub(r"[0-9\u0E50-\u0E59]", "", text)
        elif self.digits_policy == "keep":
            pass
        else:
            raise ValueError(f"Unknown digits_policy: {self.digits_policy}")

        # 6. Punctuation handling
        if self.remove_punctuation:
            text = PUNCTUATION_PATTERN.sub(" ", text)

        # 7. Whitespace normalization
        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text
