"""Dataset preparation and deterministic splitting for Porjai Thai dataset."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import soundfile as sf
import torch

from thai_audio_jepa_asr.preprocessing.audio import load_audio, DEFAULT_SAMPLE_RATE
from thai_audio_jepa_asr.preprocessing.text_normalizer import ThaiTextNormalizer

logger = logging.getLogger(__name__)

DATASET_NAME = "CMKL/Porjai-Thai-voice-dataset-central"


def deterministic_split_bucket(utterance_id: str) -> int:
    """Map utterance_id to a deterministic integer bucket [0, 99]."""
    h = hashlib.md5(utterance_id.encode("utf-8")).hexdigest()
    return int(h, 16) % 100


def assign_split(bucket: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """Assign split name based on deterministic bucket value.

    Default: 80% train, 10% val, 10% test.
    """
    train_thresh = int(train_ratio * 100)
    val_thresh = train_thresh + int(val_ratio * 100)
    if bucket < train_thresh:
        return "train"
    elif bucket < val_thresh:
        return "val"
    else:
        return "test"


def prepare_porjai_dataset(
    output_dir: str | Path,
    normalizer: Optional[ThaiTextNormalizer] = None,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    streaming: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    cache_audio: bool = True,
) -> Dict[str, Any]:
    """Download, preprocess, split, and save Porjai Thai dataset manifests.

    Args:
        output_dir: Directory where manifests and cached audio are stored.
        normalizer: Text normalizer instance.
        max_samples: Optional limit on total samples to process (useful for smoke tests).
        train_ratio: Fraction of samples for training.
        val_ratio: Fraction of samples for validation.
        streaming: If True, streams the dataset to avoid downloading 7.5GB at once.
        sample_rate: Target audio sampling rate.
        cache_audio: If True, saves processed mono WAV files locally.

    Returns:
        Summary dictionary containing counts, durations, and paths.
    """
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifests"
    audio_dir = output_dir / "audio"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if cache_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    if normalizer is None:
        normalizer = ThaiTextNormalizer()

    raw_dir = Path("data_cache/raw_porjai")
    local_parquets = list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("**/*.parquet")) if raw_dir.is_dir() else []
    if local_parquets:
        logger.info("Found %d local parquet file(s) in %s. Loading from local disk...", len(local_parquets), raw_dir)
        ds = datasets.load_dataset("parquet", data_files=[str(p) for p in sorted(local_parquets)], split="train")
    else:
        logger.info("Loading dataset %s (streaming=%s)...", DATASET_NAME, streaming)
        ds = datasets.load_dataset(DATASET_NAME, split="train", streaming=streaming)
    # Load dataset with decode=False on audio to avoid system torchcodec/ffmpeg errors
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    invalid_files: List[Dict[str, Any]] = []

    total_processed = 0
    total_duration_sec = 0.0

    for idx, item in enumerate(ds):
        if max_samples is not None and max_samples > 0 and total_processed >= max_samples:
            break

        utterance_id = item.get("utterance", f"sample_{idx:06d}")
        raw_sentence = item.get("sentence", "")
        audio_item = item.get("audio", {})

        if not raw_sentence or not audio_item:
            invalid_files.append({"id": utterance_id, "reason": "empty transcript or audio"})
            continue

        try:
            audio_source = audio_item.get("bytes") or audio_item.get("path")
            waveform, sr = load_audio(audio_source, target_sr=sample_rate)
            duration_s = waveform.shape[1] / float(sr)

            if duration_s <= 0.1:
                invalid_files.append({"id": utterance_id, "reason": "audio too short (<0.1s)"})
                continue

            normalized_sentence = normalizer.normalize(raw_sentence)
            if not normalized_sentence:
                invalid_files.append({"id": utterance_id, "reason": "normalized text is empty"})
                continue

            # Save WAV file to cache
            if cache_audio:
                wav_path = audio_dir / f"{utterance_id}.wav"
                sf.write(str(wav_path), waveform.squeeze(0).numpy(), samplerate=sr)
                audio_ref = str(wav_path)
            else:
                audio_ref = audio_item.get("path", utterance_id)

            bucket = deterministic_split_bucket(utterance_id)
            split_name = assign_split(bucket, train_ratio=train_ratio, val_ratio=val_ratio)

            record = {
                "id": utterance_id,
                "audio_path": audio_ref,
                "duration": round(duration_s, 3),
                "sentence": raw_sentence,
                "normalized_sentence": normalized_sentence,
                "char_count": len(normalized_sentence),
                "bucket": bucket,
            }
            splits[split_name].append(record)
            total_duration_sec += duration_s
            total_processed += 1

            if total_processed % 200 == 0:
                logger.info("Processed %d samples (%.2f hours)...", total_processed, total_duration_sec / 3600.0)

        except Exception as e:
            logger.warning("Error processing %s: %s", utterance_id, e)
            invalid_files.append({"id": utterance_id, "reason": str(e)})

    # Write manifests
    summary: Dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "sample_rate": sample_rate,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "total_samples": total_processed,
        "total_duration_hours": round(total_duration_sec / 3600.0, 4),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_durations_sec": {k: round(sum(r["duration"] for r in v), 2) for k, v in splits.items()},
        "invalid_count": len(invalid_files),
        "normalization_config": normalizer.to_dict(),
        "manifest_paths": {},
    }

    for split_name, records in splits.items():
        manifest_file = manifest_dir / f"{split_name}.jsonl"
        with open(manifest_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["manifest_paths"][split_name] = str(manifest_file)

    invalid_file = manifest_dir / "invalid_files.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_files, f, ensure_ascii=False, indent=2)
    summary["invalid_file_path"] = str(invalid_file)

    summary_file = manifest_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Dataset preparation complete: %s", summary["split_counts"])
    return summary
