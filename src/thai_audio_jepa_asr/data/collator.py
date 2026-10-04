"""Batch collators for Supervised ASR and Masked Latent SSL."""

from __future__ import annotations

import math
from typing import Any, Dict, List
import torch
import torch.nn.utils.rnn as rnn_utils


class ASRDataCollator:
    """Collates audio spectrograms and target transcripts for CTC training."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch items into tensors for CTC loss.

        Args:
            batch: List of dicts with 'spectrogram', 'tokens', 'text', 'id'.

        Returns:
            Dictionary containing:
                - spectrograms: Tensor of shape [B, 1, time_bins, mel_bins]
                - targets: Padded Tensor of shape [B, max_target_len]
                - target_lengths: Tensor of shape [B]
                - texts: List of reference strings
                - ids: List of utterance ids
        """
        spectrograms = torch.stack([item["spectrogram"] for item in batch], dim=0)
        target_list = [item["tokens"] for item in batch]
        target_lengths = torch.tensor([len(t) for t in target_list], dtype=torch.long)
        padded_targets = rnn_utils.pad_sequence(
            target_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        return {
            "spectrograms": spectrograms,
            "targets": padded_targets,
            "target_lengths": target_lengths,
            "texts": [item["text"] for item in batch],
            "ids": [item["id"] for item in batch],
        }


class SSLMaskCollator:
    """Collates spectrograms and generates Audio-JEPA multi-block context and prediction masks.

    Supports both contiguous temporal masking on fine temporal patch grids (e.g. 256 x 8)
    and standard 2D block masking.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (256, 8),  # 256 time x 8 freq patches = 2048 total for modified Audio-JEPA
        enc_mask_scale: tuple[float, float] = (0.2, 0.7),
        pred_mask_scale: tuple[float, float] = (0.1, 0.25),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        masking_mode: str = "temporal",  # "temporal" (contiguous time spans) or "block"
        nenc: int = 1,
        npred: int = 2,
        min_keep: int = 16,
    ) -> None:
        self.height, self.width = grid_size
        self.num_patches = self.height * self.width
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.masking_mode = masking_mode
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep

    def _sample_block(self, scale_range: tuple[float, float]) -> torch.Tensor:
        """Sample a 2D block mask of shape [height, width] with 1s inside the block."""
        min_s, max_s = scale_range
        scale = min_s + torch.rand(1).item() * (max_s - min_s)
        ar = self.aspect_ratio[0] + torch.rand(1).item() * (self.aspect_ratio[1] - self.aspect_ratio[0])

        max_keep = int(self.height * self.width * scale)
        h = max(1, min(self.height - 1, int(round(math.sqrt(max_keep * ar)))))
        w = max(1, min(self.width - 1, int(round(math.sqrt(max_keep / ar)))))

        top = torch.randint(0, max(1, self.height - h + 1), (1,)).item()
        left = torch.randint(0, max(1, self.width - w + 1), (1,)).item()

        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        mask[top : top + h, left : left + w] = True
        return mask

    def _sample_temporal_span(self, scale_range: tuple[float, float]) -> torch.Tensor:
        """Sample contiguous temporal mask spanning time steps across all frequency bins."""
        min_s, max_s = scale_range
        scale = min_s + torch.rand(1).item() * (max_s - min_s)
        time_span = max(1, min(self.height - 1, int(round(self.height * scale))))
        top = torch.randint(0, max(1, self.height - time_span + 1), (1,)).item()

        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        mask[top : top + time_span, :] = True
        return mask

    def _sample_mask(self, scale_range: tuple[float, float]) -> torch.Tensor:
        """Sample mask according to configured masking mode."""
        if self.masking_mode == "temporal":
            return self._sample_temporal_span(scale_range)
        return self._sample_block(scale_range)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with sampled context and prediction masks."""
        spectrograms = torch.stack([item["spectrogram"] for item in batch], dim=0)
        B = spectrograms.shape[0]

        # Sample masks per batch
        context_masks_list: List[torch.Tensor] = []
        prediction_masks_list: List[torch.Tensor] = []

        for _ in range(self.nenc):
            block = self._sample_mask(self.enc_mask_scale)
            keep_indices = torch.nonzero(~block.flatten()).squeeze(-1)
            if len(keep_indices) < self.min_keep:
                keep_indices = torch.arange(self.num_patches)[: self.min_keep]
            # Expand to batch dimension: shape [B, N_keep]
            context_masks_list.append(keep_indices.unsqueeze(0).repeat(B, 1))

        for _ in range(self.npred):
            block = self._sample_mask(self.pred_mask_scale)
            pred_indices = torch.nonzero(block.flatten()).squeeze(-1)
            if len(pred_indices) < 2:
                pred_indices = torch.arange(4)
            prediction_masks_list.append(pred_indices.unsqueeze(0).repeat(B, 1))

        return {
            "spectrograms": spectrograms,
            "context_masks": context_masks_list,
            "prediction_masks": prediction_masks_list,
            "ids": [item["id"] for item in batch],
        }
