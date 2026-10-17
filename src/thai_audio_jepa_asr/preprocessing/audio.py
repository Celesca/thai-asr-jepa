"""Audio loading, resampling, and configurable speech-oriented frontend extraction."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Audio-JEPA defaults
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_CLIP_LENGTH_S = 10.0
DEFAULT_N_MELS = 128
DEFAULT_TARGET_TIME_BINS = 256


@dataclass
class FrontendConfig:
    """Configuration for speech frontend extraction and patching."""

    name: str = "original"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_mels: int = DEFAULT_N_MELS
    frame_length_ms: float = 97.65625  # 2.5 * 39.0625ms for original
    frame_shift_ms: float = 39.0625   # 10000ms / 256 frames
    patch_time: int = 1
    patch_freq: int = 16
    stride_time: int = 1
    stride_freq: int = 16
    f_min: int = 20
    f_max: Optional[int] = None
    use_log: bool = True
    max_duration_s: float = DEFAULT_CLIP_LENGTH_S

    @classmethod
    def from_name(
        cls,
        name: str = "original",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_mels: int = DEFAULT_N_MELS,
        max_duration_s: float = DEFAULT_CLIP_LENGTH_S,
    ) -> FrontendConfig:
        """Create standard frontend configurations.

        Supported presets:
            - 'original': 97.66ms window, 39.06ms hop, patch (1, 16), stride (1, 16). 25.6 fps.
            - 'speech_40ms': 25.0ms window, 10.0ms hop, patch (4, 16), stride (4, 16). 25.0 fps.
            - 'speech_20ms': 25.0ms window, 10.0ms hop, patch (2, 16), stride (2, 16). 50.0 fps.
        """
        f_max = sample_rate // 2
        if name == "original":
            return cls(
                name="original",
                sample_rate=sample_rate,
                n_mels=n_mels,
                frame_length_ms=97.65625,
                frame_shift_ms=39.0625,
                patch_time=1,
                patch_freq=16,
                stride_time=1,
                stride_freq=16,
                f_min=20,
                f_max=f_max,
                max_duration_s=max_duration_s,
            )
        elif name == "speech_40ms":
            return cls(
                name="speech_40ms",
                sample_rate=sample_rate,
                n_mels=n_mels,
                frame_length_ms=25.0,
                frame_shift_ms=10.0,
                patch_time=4,
                patch_freq=16,
                stride_time=4,
                stride_freq=16,
                f_min=20,
                f_max=f_max,
                max_duration_s=max_duration_s,
            )
        elif name == "speech_20ms":
            return cls(
                name="speech_20ms",
                sample_rate=sample_rate,
                n_mels=n_mels,
                frame_length_ms=25.0,
                frame_shift_ms=10.0,
                patch_time=2,
                patch_freq=16,
                stride_time=2,
                stride_freq=16,
                f_min=20,
                f_max=f_max,
                max_duration_s=max_duration_s,
            )
        else:
            raise ValueError(f"Unknown frontend preset name: {name}. Use 'original', 'speech_40ms', or 'speech_20ms'.")

    @property
    def effective_output_interval_ms(self) -> float:
        """Effective time span per token in milliseconds."""
        return self.frame_shift_ms * self.stride_time

    @property
    def temporal_token_rate(self) -> float:
        """Tokens per second along the temporal axis."""
        return 1000.0 / self.effective_output_interval_ms

    def compute_max_fbank_frames(self) -> int:
        """Compute expected fbank frames for max_duration_s."""
        total_samples = int(self.max_duration_s * self.sample_rate)
        return compute_kaldi_fbank_frames(
            total_samples,
            self.sample_rate,
            self.frame_length_ms,
            self.frame_shift_ms,
        )

    def compute_max_time_tokens(self) -> int:
        """Compute expected ViT time tokens for max_duration_s."""
        fbank_frames = self.compute_max_fbank_frames()
        return compute_patch_tokens(fbank_frames, self.patch_time, self.stride_time)


def compute_kaldi_fbank_frames(
    num_samples: int,
    sample_rate: int,
    frame_length_ms: float,
    frame_shift_ms: float,
) -> int:
    """Calculate exact number of frames produced by Kaldi fbank.

    Formula matching Kaldi/torchaudio:
        window_size = int(sample_rate * frame_length_ms / 1000)
        window_shift = int(sample_rate * frame_shift_ms / 1000)
        frames = max(0, int((num_samples - window_size) / window_shift) + 1)
    """
    window_size = int(sample_rate * frame_length_ms / 1000.0)
    window_shift = int(sample_rate * frame_shift_ms / 1000.0)
    if num_samples < window_size:
        return 0
    return max(0, int((num_samples - window_size) / window_shift) + 1)


def compute_patch_tokens(
    num_fbank_frames: int,
    patch_time: int,
    stride_time: int,
) -> int:
    """Calculate ViT temporal token count after 2D Conv Patch Embedding."""
    if num_fbank_frames < patch_time:
        return 0
    return max(0, int((num_fbank_frames - patch_time) / stride_time) + 1)


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

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


class AudioJEPAMelSpec(torch.nn.Module):
    """Kaldi-fbank mel-spectrogram extractor with configurable speech frontend."""

    def __init__(
        self,
        frontend_type: str = "original",
        sr: int = DEFAULT_SAMPLE_RATE,
        n_mels: int = DEFAULT_N_MELS,
        clip_length_s: float = DEFAULT_CLIP_LENGTH_S,
        target_time_bins: Optional[int] = None,
        f_min: int = 20,
        f_max: Optional[int] = None,
        use_log: bool = True,
        config: Optional[FrontendConfig] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self.cfg = config
        else:
            try:
                self.cfg = FrontendConfig.from_name(
                    name=frontend_type,
                    sample_rate=sr,
                    n_mels=n_mels,
                    max_duration_s=clip_length_s,
                )
            except ValueError:
                # Custom fallback
                hop = (clip_length_s * 1000.0) / (target_time_bins or DEFAULT_TARGET_TIME_BINS)
                self.cfg = FrontendConfig(
                    name=frontend_type,
                    sample_rate=sr,
                    n_mels=n_mels,
                    frame_length_ms=2.5 * hop,
                    frame_shift_ms=hop,
                    patch_time=1,
                    patch_freq=16,
                    stride_time=1,
                    stride_freq=16,
                    f_min=f_min,
                    f_max=f_max if f_max is not None else sr // 2,
                    use_log=use_log,
                    max_duration_s=clip_length_s,
                )

        self.sr = self.cfg.sample_rate
        self.n_mels = self.cfg.n_mels
        self.clip_length_s = self.cfg.max_duration_s
        self.f_min = self.cfg.f_min
        self.f_max = self.cfg.f_max if self.cfg.f_max is not None else self.sr // 2
        self.use_log = self.cfg.use_log
        self.hop_length_ms = self.cfg.frame_shift_ms
        self.frame_length_ms = self.cfg.frame_length_ms

        if target_time_bins is not None:
            self.target_time_bins = target_time_bins
        else:
            if self.cfg.name == "original":
                self.target_time_bins = 256
            elif self.cfg.name in ("speech_40ms", "speech_20ms"):
                self.target_time_bins = 1000
            else:
                self.target_time_bins = self.cfg.compute_max_fbank_frames()

    def compute_valid_tokens(self, valid_fbank_frames: int) -> int:
        """Compute valid time tokens after patch embedding convolution."""
        return compute_patch_tokens(
            valid_fbank_frames,
            self.cfg.patch_time,
            self.cfg.stride_time,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        pad_to_target: bool = True,
        allow_crop: bool = False,
    ) -> Tuple[torch.Tensor, int, int]:
        """Compute Kaldi fbank spectrogram preserving natural time axis.

        Args:
            waveform: Tensor of shape [1, num_samples] or [num_samples].
            pad_to_target: If True, pads time dimension up to self.target_time_bins.
            allow_crop: If True (used only for SSL random crops), crops exceeding frames.

        Returns:
            Tuple of:
                - spec: Tensor of shape [1, time_bins, n_mels]
                - valid_fbank_frames: Exact number of non-padded fbank time frames
                - valid_time_tokens: Exact number of valid ViT temporal tokens
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

        actual_t = spec.shape[0]
        valid_fbank_frames = actual_t

        if allow_crop and actual_t > self.target_time_bins:
            spec = spec[: self.target_time_bins]
            valid_fbank_frames = self.target_time_bins

        valid_time_tokens = self.compute_valid_tokens(valid_fbank_frames)

        if pad_to_target:
            target_t = max(self.target_time_bins, valid_fbank_frames)
            if spec.shape[0] < target_t:
                pad = target_t - spec.shape[0]
                spec = torch.cat([spec, torch.zeros(pad, self.n_mels, dtype=spec.dtype, device=spec.device)], dim=0)

        return spec.unsqueeze(0), valid_fbank_frames, valid_time_tokens
