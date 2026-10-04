"""Speech recognition evaluation metrics: CER and Thai WER."""

from __future__ import annotations

from typing import Dict, List, Tuple
import jiwer
import pythainlp


def compute_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute Character Error Rate (CER) and edit operation breakdown.

    Args:
        references: List of ground-truth strings.
        hypotheses: List of predicted strings.

    Returns:
        Dictionary with 'cer', 'insertions', 'deletions', 'substitutions', 'hits'.
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have equal length"

    # Convert characters to space-separated strings for jiwer
    ref_chars = [" ".join(list(r)) for r in references]
    hyp_chars = [" ".join(list(h)) for h in hypotheses]

    out = jiwer.process_words(ref_chars, hyp_chars)
    total_ref_chars = sum(len(r) for r in references)

    return {
        "cer": round(out.wer, 4),
        "char_insertions": out.insertions,
        "char_deletions": out.deletions,
        "char_substitutions": out.substitutions,
        "char_hits": out.hits,
        "total_ref_chars": total_ref_chars,
    }


def compute_thai_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute Word Error Rate (WER) using PyThaiNLP dictionary-based word segmentation.

    Because written Thai has no word spaces, standard whitespace-based WER is misleading.
    We tokenize each Thai sentence into words using PyThaiNLP (newmm engine) before computing WER.

    Args:
        references: List of ground-truth Thai strings.
        hypotheses: List of predicted Thai strings.

    Returns:
        Dictionary with 'wer', 'word_insertions', 'word_deletions', 'word_substitutions', 'word_hits'.
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have equal length"

    tokenized_refs: List[str] = []
    tokenized_hyps: List[str] = []

    for ref, hyp in zip(references, hypotheses):
        # Segment into words separated by spaces
        ref_words = pythainlp.tokenize.word_tokenize(ref, engine="newmm", keep_whitespace=False)
        hyp_words = pythainlp.tokenize.word_tokenize(hyp, engine="newmm", keep_whitespace=False)
        tokenized_refs.append(" ".join(ref_words))
        tokenized_hyps.append(" ".join(hyp_words))

    out = jiwer.process_words(tokenized_refs, tokenized_hyps)
    total_ref_words = sum(len(r.split()) for r in tokenized_refs)

    return {
        "wer": round(out.wer, 4),
        "word_insertions": out.insertions,
        "word_deletions": out.deletions,
        "word_substitutions": out.substitutions,
        "word_hits": out.hits,
        "total_ref_words": total_ref_words,
    }


def compute_asr_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute both CER and Thai WER simultaneously."""
    cer_stats = compute_cer(references, hypotheses)
    wer_stats = compute_thai_wer(references, hypotheses)
    return {**cer_stats, **wer_stats}
