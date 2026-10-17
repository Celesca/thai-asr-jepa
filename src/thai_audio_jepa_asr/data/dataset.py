"""PyTorch Datasets for Thai ASR and SSL adaptation with audio-transcript consistency."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec, load_audio, DEFAULT_SAMPLE_RATE
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL manifest records."""
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ThaiSpeechASRDataset(Dataset):
    """Dataset for Supervised CTC Thai Speech Recognition.

    Strictly preserves audio-transcript consistency:
    - Never crops an utterance while retaining its full transcript.
    - Excludes utterances longer than max_duration_s or shorter than min_duration_s.
    - Logs excluded sample counts and excluded hours.
    - Supports nested subsets for low-resource evaluations (budget_ratio or budget_hours).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: ThaiCharTokenizer,
        mel_spec_transform: Optional[AudioJEPAMelSpec] = None,
        budget_ratio: Optional[float] = None,
        budget_hours: Optional[float] = None,
        subsample_seed: int = 42,
        max_duration_s: float = 10.0,
        min_duration_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.mel_transform = mel_spec_transform or AudioJEPAMelSpec()
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s

        all_records = load_manifest(self.manifest_path)
        total_count = len(all_records)
        total_raw_hours = sum(r.get("duration", 0.0) for r in all_records) / 3600.0

        # Filter complete utterances strictly within duration bounds
        valid_records = []
        excluded_long = 0
        excluded_short = 0
        excluded_hours = 0.0

        for r in all_records:
            dur = float(r.get("duration", 0.0))
            if dur > self.max_duration_s:
                excluded_long += 1
                excluded_hours += dur
            elif dur < self.min_duration_s:
                excluded_short += 1
                excluded_hours += dur
            else:
                valid_records.append(r)

        excluded_hours = excluded_hours / 3600.0
        kept_hours = sum(r.get("duration", 0.0) for r in valid_records) / 3600.0

        logger.info(
            "ASR Dataset [%s]: Total %d utterances (%.2f h). Kept %d (%.2f h). "
            "Excluded %d long (>%.1fs), %d short (<%.1fs), total excluded: %.2f h.",
            self.manifest_path.name,
            total_count,
            total_raw_hours,
            len(valid_records),
            kept_hours,
            excluded_long,
            self.max_duration_s,
            excluded_short,
            self.min_duration_s,
            excluded_hours,
        )

        # Apply nested budget subset if requested
        if budget_hours is not None and budget_hours > 0:
            # Deterministic subset matching target hours
            rng = np.random.RandomState(subsample_seed)
            shuffled_indices = rng.permutation(len(valid_records))
            cum_hours = 0.0
            selected_indices = []
            for idx in shuffled_indices:
                r = valid_records[idx]
                cum_hours += float(r.get("duration", 0.0)) / 3600.0
                selected_indices.append(idx)
                if cum_hours >= budget_hours:
                    break
            self.records = [valid_records[i] for i in sorted(selected_indices)]
            logger.info(
                "Applied budget_hours=%.2f: Selected %d utterances (retained %.2f hours).",
                budget_hours,
                len(self.records),
                cum_hours,
            )
        elif budget_ratio is not None and 0.0 < budget_ratio < 1.0:
            rng = np.random.RandomState(subsample_seed)
            indices = rng.permutation(len(valid_records))
            num_samples = max(1, int(len(valid_records) * budget_ratio))
            selected_indices = indices[:num_samples]
            self.records = [valid_records[i] for i in sorted(selected_indices)]
            selected_hours = sum(r.get("duration", 0.0) for r in self.records) / 3600.0
            logger.info(
                "Applied budget_ratio=%.2f: Selected %d utterances (retained %.2f hours).",
                budget_ratio,
                len(self.records),
                selected_hours,
            )
        else:
            self.records = valid_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        audio_path = record["audio_path"]
        text = record["normalized_sentence"]
        utterance_id = record["id"]

        waveform, _ = load_audio(audio_path, target_sr=self.mel_transform.sr)
        # Never crop complete utterance in supervised CTC
        spectrogram, valid_frames, valid_tokens = self.mel_transform(
            waveform,
            pad_to_target=True,
            allow_crop=False,
        )
        token_ids = self.tokenizer.encode(text)

        return {
            "id": utterance_id,
            "spectrogram": spectrogram,  # [1, time_bins, 128]
            "valid_fbank_frames": valid_frames,
            "valid_time_tokens": valid_tokens,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "duration": float(record.get("duration", 0.0)),
        }


class ThaiAudioSSLDataset(Dataset):
    """Dataset for Thai Continued Self-Supervised Adaptation of Audio-JEPA.

    Uses unlabeled speech audio only.
    Random cropping is permitted because transcripts are not used.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        mel_spec_transform: Optional[AudioJEPAMelSpec] = None,
        max_duration_s: float = 10.0,
        min_duration_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.mel_transform = mel_spec_transform or AudioJEPAMelSpec()
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s

        records = load_manifest(self.manifest_path)
        self.records = [
            r for r in records
            if float(r.get("duration", 0.0)) >= min_duration_s
        ]
        logger.info(
            "SSL Dataset [%s]: Loaded %d unlabeled speech samples (min_dur=%.1fs, max_dur=%.1fs).",
            self.manifest_path.name,
            len(self.records),
            min_duration_s,
            max_duration_s,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        audio_path = record["audio_path"]
        utterance_id = record["id"]

        waveform, _ = load_audio(audio_path, target_sr=self.mel_transform.sr)
        max_samples = int(self.max_duration_s * self.mel_transform.sr)

        # Random audio crop for SSL when audio exceeds max duration
        if waveform.shape[-1] > max_samples:
            max_start = waveform.shape[-1] - max_samples
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start : start + max_samples]

        spectrogram, valid_frames, valid_tokens = self.mel_transform(
            waveform,
            pad_to_target=True,
            allow_crop=True,
        )

        return {
            "id": utterance_id,
            "spectrogram": spectrogram,
            "valid_fbank_frames": valid_frames,
            "valid_time_tokens": valid_tokens,
            "duration": float(record.get("duration", 0.0)),
        }
