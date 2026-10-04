"""Character-level Thai CTC Tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch


# Canonical Thai characters set
THAI_CONSONANTS = [chr(c) for c in range(0x0E01, 0x0E2F)]  # ก - ฮ
THAI_VOWELS = [
    "\u0E30", "\u0E31", "\u0E32", "\u0E33", "\u0E34", "\u0E35", "\u0E36", "\u0E37",
    "\u0E38", "\u0E39", "\u0E40", "\u0E41", "\u0E42", "\u0E43", "\u0E44", "\u0E45", "\u0E47",
]
THAI_TONES = ["\u0E48", "\u0E49", "\u0E4A", "\u0E4B"]
THAI_SPECIALS = ["\u0E4C", "\u0E46", "\u0E2F"]  # Thanthakhat, Maiyamok, Paiyannoi
ARABIC_NUMERALS = [str(i) for i in range(10)]
LATIN_LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)]


class ThaiCharTokenizer:
    """Character-level tokenizer for CTC Thai speech recognition.

    Special tokens:
        <pad> / <blank> -> 0 (CTC blank index)
        <unk> -> 1
        | (space) -> 2
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SPACE_TOKEN = "|"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize tokenizer with given vocabulary or canonical Thai alphabet."""
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self.build_canonical_vocab()

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab[self.PAD_TOKEN]
        self.unk_token_id = self.vocab[self.UNK_TOKEN]
        self.space_token_id = self.vocab[self.SPACE_TOKEN]

    @classmethod
    def build_canonical_vocab(cls) -> Dict[str, int]:
        """Construct the canonical vocabulary of Thai characters, digits, and common latin letters."""
        tokens = [cls.PAD_TOKEN, cls.UNK_TOKEN, cls.SPACE_TOKEN]

        for c in THAI_CONSONANTS:
            if c not in tokens:
                tokens.append(c)
        for v in THAI_VOWELS:
            if v not in tokens:
                tokens.append(v)
        for t in THAI_TONES:
            if t not in tokens:
                tokens.append(t)
        for s in THAI_SPECIALS:
            if s not in tokens:
                tokens.append(s)
        for d in ARABIC_NUMERALS:
            if d not in tokens:
                tokens.append(d)
        for l in LATIN_LETTERS:
            if l not in tokens:
                tokens.append(l)

        return {tok: idx for idx, tok in enumerate(tokens)}

    @classmethod
    def build_from_texts(cls, texts: List[str], min_freq: int = 1) -> ThaiCharTokenizer:
        """Build tokenizer vocabulary directly from a corpus of texts."""
        vocab = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.SPACE_TOKEN: 2,
        }
        counts: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                if ch == " ":
                    continue
                counts[ch] = counts.get(ch, 0) + 1

        for ch, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if count >= min_freq and ch not in vocab:
                vocab[ch] = len(vocab)

        return cls(vocab=vocab)

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Encode a normalized text string into a list of token IDs.

        Replaces spaces with the SPACE_TOKEN ('|'). Unknown characters map to UNK_TOKEN.
        """
        token_ids: List[int] = []
        for ch in text:
            if ch == " ":
                token_ids.append(self.space_token_id)
            elif ch in self.vocab:
                token_ids.append(self.vocab[ch])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids: List[int], remove_special: bool = True) -> str:
        """Decode a list of token IDs back into text."""
        chars: List[str] = []
        for tid in token_ids:
            if remove_special and tid in (self.pad_token_id, self.unk_token_id):
                continue
            if tid == self.space_token_id:
                chars.append(" ")
            else:
                chars.append(self.inv_vocab.get(tid, self.UNK_TOKEN))
        return "".join(chars)

    def decode_ctc(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode a sequence of CTC greedy predictions with blank collapsing."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        collapsed: List[int] = []
        prev = -1
        for tid in token_ids:
            if tid != prev:
                if tid != self.pad_token_id:
                    collapsed.append(tid)
                prev = tid

        return self.decode(collapsed, remove_special=True)

    def save(self, filepath: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> ThaiCharTokenizer:
        """Load vocabulary from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
