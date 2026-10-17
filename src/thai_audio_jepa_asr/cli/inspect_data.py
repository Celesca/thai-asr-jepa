"""CLI command: Inspect Porjai Thai dataset schema and sample records."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

import datasets

from thai_audio_jepa_asr.data.porjai import DATASET_NAME
from thai_audio_jepa_asr.preprocessing.audio import load_audio
from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("inspect_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Porjai Thai Central dataset schema and samples.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset identifier.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample records to display.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Inspecting Hugging Face Dataset: {args.dataset}")
    print("=" * 70)

    try:
        ds = datasets.load_dataset(args.dataset, split="train", streaming=True)
        ds = ds.cast_column("audio", datasets.Audio(decode=False))

        samples = []
        for idx, item in enumerate(ds):
            if idx >= args.num_samples:
                break

            audio_info = item.get("audio", {})
            audio_bytes = audio_info.get("bytes")
            audio_len = len(audio_bytes) if audio_bytes else 0

            # Inspect audio duration using load_audio
            dur_str = "unknown"
            if audio_bytes:
                try:
                    wf, sr = load_audio(audio_bytes)
                    dur_str = f"{wf.shape[1] / sr:.2f}s @ {sr}Hz"
                except Exception as e:
                    dur_str = f"decode error ({e})"

            sample_info = {
                "sample_idx": idx,
                "utterance_id": item.get("utterance"),
                "sentence": item.get("sentence"),
                "audio_bytes_len": audio_len,
                "decoded_duration": dur_str,
            }
            samples.append(sample_info)

        print(f"Successfully streamed {len(samples)} sample records:")
        print(json.dumps(samples, ensure_ascii=False, indent=2))
        print("=" * 70)
        print("Schema inspection completed successfully.")
        sys.stdout.flush()
        return 0

    except Exception as e:
        logger.error("Dataset inspection failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    code = main()
    import os
    os._exit(code)
