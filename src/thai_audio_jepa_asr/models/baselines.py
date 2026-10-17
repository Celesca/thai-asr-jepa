"""Unified speech encoder interface and baseline models (Wav2Vec2 / HuBERT).

Allows Audio-JEPA, Wav2Vec2, HuBERT, and future SSL encoders to be plugged into the
same CTC speech recognition downstream pipeline with identical evaluation metrics.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseSpeechEncoder(nn.Module, abc.ABC):
    """Abstract base class for all speech representation encoders."""

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output representations."""
        pass

    @abc.abstractmethod
    def forward(self, audio_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech inputs.

        Args:
            audio_inputs: Audio waveform [B, T_samples] or Mel-spectrogram [B, 1, H, W].

        Returns:
            Tuple of:
                - features: Tensor of shape [B, T_out, embed_dim]
                - lengths: Tensor of shape [B]
        """
        pass


class AudioJEPAAdapter(BaseSpeechEncoder):
    """Wraps AudioJEPAEncoder and TemporalAdapter under the unified BaseSpeechEncoder interface."""

    def __init__(
        self,
        encoder: nn.Module,
        temporal_adapter: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal_adapter = temporal_adapter

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def forward(self, audio_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(audio_inputs, context_masks=None)
        temporal_feats, lengths = self.temporal_adapter(encoded)
        return temporal_feats, lengths


class GenericWav2Vec2Encoder(BaseSpeechEncoder):
    """Wav2Vec2 / HuBERT baseline encoder implementation.

    Uses Hugging Face transformers if installed, or provides an internal
    1D-convolutional + Transformer architecture as a lightweight baseline.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        embed_dim: int = 768,
        sample_rate: int = 16000,
        freeze_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.model_name = model_name

        try:
            from transformers import Wav2Vec2Model
            logger.info("Initializing Wav2Vec2 model from Hugging Face: %s", model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.is_hf = True
            if freeze_feature_extractor:
                self.model.freeze_feature_encoder()
        except Exception as e:
            logger.info("Transformers Wav2Vec2 not available (%s); using internal baseline convolutional encoder.", e)
            self.is_hf = False
            # Lightweight 1D conv feature extractor + transformer layer fallback
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=10, stride=5),
                nn.GELU(),
                nn.Conv1d(128, 256, kernel_size=8, stride=4),
                nn.GELU(),
                nn.Conv1d(256, embed_dim, kernel_size=4, stride=2),
                nn.GELU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=2048, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, audio_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_hf:
            # audio_inputs expected to be waveform [B, T_samples]
            if audio_inputs.ndim == 3 and audio_inputs.shape[1] == 1:
                audio_inputs = audio_inputs.squeeze(1)
            outputs = self.model(audio_inputs)
            feats = outputs.last_hidden_state
            lengths = torch.full((feats.shape[0],), feats.shape[1], dtype=torch.long, device=feats.device)
            return feats, lengths
        else:
            if audio_inputs.ndim == 2:
                audio_inputs = audio_inputs.unsqueeze(1)
            elif audio_inputs.ndim == 4:
                # If spectrogram [B, 1, 256, 128], flatten along freq for 1D conv
                audio_inputs = audio_inputs.mean(dim=-1)
            feats = self.feature_extractor(audio_inputs).transpose(1, 2)
            feats = self.transformer(feats)
            lengths = torch.full((feats.shape[0],), feats.shape[1], dtype=torch.long, device=feats.device)
            return feats, lengths
