"""Audio loading, resampling, and mel-spectrogram extraction matching Audio-JEPA."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Audio-JEPA defaults from upstream configuration
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_CLIP_LENGTH_S = 10.0
DEFAULT_N_MELS = 128
DEFAULT_TARGET_TIME_BINS = 256


def load_audio(
    source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[torch.Tensor, int]:
    """Load audio from file path or bytes, convert to mono, and resample.

    Args:
        source: File path or raw audio bytes.
        target_sr: Sampling rate to resample to (default: 32000 Hz).

    Returns:
        Tuple of (waveform: Tensor of shape [1, num_samples], sample_rate: int).
    """
    if isinstance(source, (str, Path)):
        try:
            import soundfile as sf
            data, sr = sf.read(str(source), dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.transpose(0, 1)
        except Exception:
            import av
            with av.open(str(source)) as container:
                stream = container.streams.audio[0]
                sr = stream.rate
                frames = [f.to_ndarray() for f in container.decode(stream)]
            if not frames:
                return torch.zeros((1, target_sr), dtype=torch.float32), target_sr
            arr = np.concatenate(frames, axis=-1).astype(np.float32)
            waveform = torch.from_numpy(arr)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
    elif isinstance(source, (bytes, io.BytesIO)):
        bio = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
        # Try PyAV for robust decoding of mp3/wav/ogg bytes
        try:
            import av
            with av.open(bio) as container:
                stream = container.streams.audio[0]
                sr = stream.rate
                frames = [frame.to_ndarray() for frame in container.decode(stream)]
            if not frames:
                waveform = torch.zeros((1, target_sr), dtype=torch.float32)
                return waveform, target_sr
            arr = np.concatenate(frames, axis=-1)
            # Ensure float32 in [-1, 1]
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            waveform = torch.from_numpy(arr)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        except Exception:
            bio.seek(0)
            waveform, sr = torchaudio.load(bio)
    else:
        raise TypeError(f"Unsupported audio source type: {type(source)}")

    # Convert to mono if multichannel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


class AudioJEPAMelSpec(torch.nn.Module):
    """Kaldi-fbank mel-spectrogram extractor exactly matching Audio-JEPA."""

    def __init__(
        self,
        sr: int = DEFAULT_SAMPLE_RATE,
        n_mels: int = DEFAULT_N_MELS,
        clip_length_s: float = DEFAULT_CLIP_LENGTH_S,
        target_time_bins: int = DEFAULT_TARGET_TIME_BINS,
        f_min: int = 20,
        f_max: Optional[int] = None,
        use_log: bool = True,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.clip_length_s = clip_length_s
        self.target_time_bins = target_time_bins
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        self.use_log = use_log

        # Kaldi hop length and frame length in milliseconds
        self.hop_length_ms = (clip_length_s * 1000.0) / target_time_bins
        self.frame_length_ms = 2.5 * self.hop_length_ms

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Kaldi fbank spectrogram of shape [1, target_time_bins, n_mels].

        Args:
            waveform: Tensor of shape [1, num_samples] or [num_samples].

        Returns:
            Tensor of shape [1, target_time_bins, n_mels].
        """
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform_1d = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform_1d = waveform
        else:
            waveform_1d = waveform.mean(dim=0)

        # Zero-mean normalization as expected by Kaldi
        centered_waveform = waveform_1d - waveform_1d.mean()

        spec = kaldi.fbank(
            centered_waveform.unsqueeze(0),
            sample_frequency=self.sr,
            frame_length=self.frame_length_ms,
            frame_shift=self.hop_length_ms,
            num_mel_bins=self.n_mels,
            low_freq=self.f_min,
            high_freq=self.f_max,
            use_log_fbank=self.use_log,
            window_type="hanning",
        )

        # spec shape is (time, n_mels)
        t_len = spec.shape[0]
        if t_len < self.target_time_bins:
            pad = self.target_time_bins - t_len
            spec = torch.cat([spec, torch.zeros(pad, self.n_mels, dtype=spec.dtype, device=spec.device)], dim=0)
        elif t_len > self.target_time_bins:
            spec = spec[: self.target_time_bins]

        return spec.unsqueeze(0)
