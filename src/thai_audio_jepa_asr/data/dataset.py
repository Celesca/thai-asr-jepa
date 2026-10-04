"""PyTorch Datasets for Thai ASR and SSL adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec, load_audio, DEFAULT_SAMPLE_RATE
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer


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
    """Dataset for Supervised CTC Thai Speech Recognition."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: ThaiCharTokenizer,
        mel_spec_transform: Optional[AudioJEPAMelSpec] = None,
        budget_ratio: float = 1.0,
        subsample_seed: int = 42,
        max_duration_s: float = 10.0,
        min_duration_s: float = 0.5,
    ) -> None:
        """Initialize dataset.

        Args:
            manifest_path: Path to .jsonl manifest file.
            tokenizer: Character tokenizer instance.
            mel_spec_transform: Kaldi fbank transform module.
            budget_ratio: Fraction of data to use (1.0 = 100%, 0.5 = 50%, etc.).
            subsample_seed: Deterministic seed for budget subsampling.
            max_duration_s: Filter out audio longer than this.
            min_duration_s: Filter out audio shorter than this.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.mel_transform = mel_spec_transform or AudioJEPAMelSpec()
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s

        records = load_manifest(self.manifest_path)
        # Filter duration
        valid_records = [
            r for r in records
            if self.min_duration_s <= r.get("duration", 0) <= self.max_duration_s
        ]

        # Apply low-resource budget if < 1.0
        if 0.0 < budget_ratio < 1.0:
            rng = np.random.RandomState(subsample_seed)
            indices = rng.permutation(len(valid_records))
            num_samples = max(1, int(len(valid_records) * budget_ratio))
            selected_indices = indices[:num_samples]
            self.records = [valid_records[i] for i in sorted(selected_indices)]
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
        spectrogram = self.mel_transform(waveform)  # Shape [1, 256, 128]
        token_ids = self.tokenizer.encode(text)

        return {
            "id": utterance_id,
            "spectrogram": spectrogram,  # [1, 256, 128]
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "duration": record.get("duration", 0.0),
        }


class ThaiAudioSSLDataset(Dataset):
    """Dataset for Thai-specific Continued Self-Supervised Adaptation of Audio-JEPA."""

    def __init__(
        self,
        manifest_path: str | Path,
        mel_spec_transform: Optional[AudioJEPAMelSpec] = None,
        max_duration_s: float = 10.0,
        min_duration_s: float = 0.5,
    ) -> None:
        """Initialize SSL dataset using unlabeled audio only."""
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.mel_transform = mel_spec_transform or AudioJEPAMelSpec()
        records = load_manifest(self.manifest_path)
        self.records = [
            r for r in records
            if min_duration_s <= r.get("duration", 0) <= max_duration_s
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        audio_path = record["audio_path"]
        waveform, _ = load_audio(audio_path, target_sr=self.mel_transform.sr)
        spectrogram = self.mel_transform(waveform)  # Shape [1, 256, 128]
        return {
            "id": record["id"],
            "spectrogram": spectrogram,
        }
