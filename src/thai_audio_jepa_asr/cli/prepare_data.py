"""CLI command: Prepare Porjai Thai dataset manifests and cached audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thai_audio_jepa_asr.data.porjai import prepare_porjai_dataset
from thai_audio_jepa_asr.preprocessing.text_normalizer import ThaiTextNormalizer
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.utils.config import resolve_config
from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("prepare_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Porjai Thai dataset manifests.")
    parser.add_argument("--config", default="configs/data/porjai.yaml", help="Path to data configuration YAML.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override maximum samples to process.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    config = resolve_config(args.config)
    data_cfg = config.get("data", config)

    output_dir = args.output_dir or data_cfg.get("output_dir", "data_cache/porjai")
    max_samples = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples", None)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    streaming = data_cfg.get("streaming", True)
    sample_rate = data_cfg.get("sample_rate", 32000)
    cache_audio = data_cfg.get("cache_audio", True)

    norm_cfg = config.get("normalization", {})
    normalizer = ThaiTextNormalizer.from_dict(norm_cfg) if norm_cfg else ThaiTextNormalizer()

    logger.info("Preparing Porjai dataset: output_dir=%s, max_samples=%s", output_dir, max_samples)

    summary = prepare_porjai_dataset(
        output_dir=output_dir,
        normalizer=normalizer,
        max_samples=max_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        streaming=streaming,
        sample_rate=sample_rate,
        cache_audio=cache_audio,
    )

    # Build canonical character tokenizer vocabulary and save alongside manifests
    vocab_file = Path(output_dir) / "vocab.json"
    tokenizer = ThaiCharTokenizer()
    tokenizer.save(vocab_file)
    logger.info("Saved Thai character vocabulary (%d tokens) to: %s", tokenizer.vocab_size, vocab_file)

    logger.info("Data preparation completed successfully.")
    return 0


if __name__ == "__main__":
    code = main()
    import os
    os._exit(code)
