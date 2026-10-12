"""Batch collators for Supervised ASR and Masked Latent SSL with explicit padding handling."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.utils.rnn as rnn_utils


class ASRDataCollator:
    """Collates audio spectrograms, valid lengths, and target transcripts for CTC training."""

    def __init__(
        self,
        pad_token_id: int = 0,
        grid_w: int = 8,
        patch_time: int = 1,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.grid_w = grid_w
        self.patch_time = patch_time

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch items into tensors for CTC loss.

        Args:
            batch: List of dicts with 'spectrogram', 'tokens', 'text', 'id', 'valid_time_tokens'.

        Returns:
            Dictionary containing:
                - spectrograms: Tensor of shape [B, 1, time_bins, mel_bins]
                - targets: Padded Tensor of shape [B, max_target_len]
                - target_lengths: Tensor of shape [B]
                - valid_time_tokens: Exact valid sequence length for each utterance [B]
                - padding_mask: Boolean tensor [B, total_patches] where True indicates padding
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

        valid_time_tokens = torch.tensor(
            [item.get("valid_time_tokens", item.get("valid_fbank_frames", spectrograms.shape[2])) for item in batch],
            dtype=torch.long,
        )

        B = len(batch)
        time_dim = spectrograms.shape[2]
        grid_h = max(1, time_dim // self.patch_time)
        total_patches = grid_h * self.grid_w

        # Padding mask: True means padding (do not attend / do not train)
        padding_mask = torch.zeros((B, total_patches), dtype=torch.bool)
        for b in range(B):
            v_tok = int(valid_time_tokens[b].item())
            if v_tok < grid_h:
                padding_mask[b, v_tok * self.grid_w :] = True

        return {
            "spectrograms": spectrograms,
            "targets": padded_targets,
            "target_lengths": target_lengths,
            "valid_time_tokens": valid_time_tokens,
            "padding_mask": padding_mask,
            "texts": [item["text"] for item in batch],
            "ids": [item["id"] for item in batch],
        }


class SSLMaskCollator:
    """Collates spectrograms and generates speech-appropriate context and prediction masks.

    Supports:
        - 'short_span': Samples multiple 200-400 ms spans across valid speech positions only.
        - 'long_span': Ablation preserving original long temporal span masking.
        - 'block': Standard 2D block masking.
    Excludes padding tokens from context, prediction, and loss.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (256, 8),
        masking_mode: str = "short_span",  # "short_span", "long_span", or "block"
        span_duration_ms: float = 300.0,    # Target span duration for short_span
        token_interval_ms: float = 39.0625, # Time resolution per token
        mask_ratio: float = 0.50,          # Total masking ratio
        enc_mask_scale: tuple[float, float] = (0.2, 0.7),
        pred_mask_scale: tuple[float, float] = (0.1, 0.25),
        nenc: int = 1,
        npred: int = 2,
        min_keep: int = 8,
    ) -> None:
        self.height, self.width = grid_size
        self.num_patches = self.height * self.width
        self.masking_mode = masking_mode
        self.span_duration_ms = span_duration_ms
        self.token_interval_ms = token_interval_ms
        self.mask_ratio = mask_ratio
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep

    def _sample_short_span_mask(
        self,
        valid_time_tokens: int,
    ) -> Tuple[torch.Tensor, float]:
        """Sample multiple contiguous temporal spans over valid positions only.

        Returns:
            Tuple of (mask: 2D boolean tensor [height, width], actual_mask_ratio: float).
        """
        valid_t = max(2, min(self.height, valid_time_tokens))
        span_tokens = max(1, min(valid_t // 2, int(round(self.span_duration_ms / self.token_interval_ms))))
        target_masked_tokens = max(1, int(round(valid_t * self.mask_ratio)))

        masked_time_steps = set()
        attempts = 0
        max_attempts = 50

        while len(masked_time_steps) < target_masked_tokens and attempts < max_attempts:
            attempts += 1
            max_start = max(0, valid_t - span_tokens)
            start_t = torch.randint(0, max_start + 1, (1,)).item()
            for t in range(start_t, min(valid_t, start_t + span_tokens)):
                masked_time_steps.add(t)

        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        for t in masked_time_steps:
            mask[t, :] = True

        actual_ratio = len(masked_time_steps) / float(valid_t)
        return mask, actual_ratio

    def _sample_long_temporal_span(self, scale_range: tuple[float, float], valid_time_tokens: int) -> torch.Tensor:
        """Original long temporal span ablation."""
        valid_t = max(2, min(self.height, valid_time_tokens))
        min_s, max_s = scale_range
        scale = min_s + torch.rand(1).item() * (max_s - min_s)
        time_span = max(1, min(valid_t - 1, int(round(valid_t * scale))))
        top = torch.randint(0, max(1, valid_t - time_span + 1), (1,)).item()

        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        mask[top : top + time_span, :] = True
        return mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with masks respecting valid audio lengths."""
        spectrograms = torch.stack([item["spectrogram"] for item in batch], dim=0)
        B = spectrograms.shape[0]

        valid_tokens_list = [item.get("valid_time_tokens", self.height) for item in batch]
        min_valid_t = min(valid_tokens_list) if valid_tokens_list else self.height

        context_masks_list: List[torch.Tensor] = []
        prediction_masks_list: List[torch.Tensor] = []
        measured_ratios: List[float] = []

        for _ in range(self.nenc):
            if self.masking_mode == "short_span":
                block, ratio = self._sample_short_span_mask(min_valid_t)
                measured_ratios.append(ratio)
            elif self.masking_mode == "long_span":
                block = self._sample_long_temporal_span(self.enc_mask_scale, min_valid_t)
            else:
                # 2D block
                min_s, max_s = self.enc_mask_scale
                scale = min_s + torch.rand(1).item() * (max_s - min_s)
                h = max(1, min(min_valid_t - 1, int(round(min_valid_t * scale))))
                w = max(1, min(self.width, int(round(self.width * scale))))
                top = torch.randint(0, max(1, min_valid_t - h + 1), (1,)).item()
                left = torch.randint(0, max(1, self.width - w + 1), (1,)).item()
                block = torch.zeros((self.height, self.width), dtype=torch.bool)
                block[top : top + h, left : left + w] = True

            # Exclude padding from context: only consider valid time positions
            valid_mask_2d = torch.zeros((self.height, self.width), dtype=torch.bool)
            valid_mask_2d[:min_valid_t, :] = True

            # Unmasked valid positions become context
            unmasked_valid = (~block) & valid_mask_2d
            keep_indices = torch.nonzero(unmasked_valid.flatten()).squeeze(-1)
            if len(keep_indices) < self.min_keep:
                keep_indices = torch.nonzero(valid_mask_2d.flatten()).squeeze(-1)[: self.min_keep]

            context_masks_list.append(keep_indices.unsqueeze(0).repeat(B, 1))

        for _ in range(self.npred):
            if self.masking_mode == "short_span":
                # Sample prediction mask from masked regions or dedicated prediction spans
                pred_block, _ = self._sample_short_span_mask(min_valid_t)
            elif self.masking_mode == "long_span":
                pred_block = self._sample_long_temporal_span(self.pred_mask_scale, min_valid_t)
            else:
                min_s, max_s = self.pred_mask_scale
                scale = min_s + torch.rand(1).item() * (max_s - min_s)
                h = max(1, min(min_valid_t - 1, int(round(min_valid_t * scale))))
                w = max(1, min(self.width, int(round(self.width * scale))))
                top = torch.randint(0, max(1, min_valid_t - h + 1), (1,)).item()
                left = torch.randint(0, max(1, self.width - w + 1), (1,)).item()
                pred_block = torch.zeros((self.height, self.width), dtype=torch.bool)
                pred_block[top : top + h, left : left + w] = True

            valid_mask_2d = torch.zeros((self.height, self.width), dtype=torch.bool)
            valid_mask_2d[:min_valid_t, :] = True
            masked_valid = pred_block & valid_mask_2d

            pred_indices = torch.nonzero(masked_valid.flatten()).squeeze(-1)
            if len(pred_indices) < 2:
                pred_indices = torch.nonzero(valid_mask_2d.flatten()).squeeze(-1)[:4]

            prediction_masks_list.append(pred_indices.unsqueeze(0).repeat(B, 1))

        # Build padding mask [B, num_patches] for attention
        padding_mask = torch.zeros((B, self.num_patches), dtype=torch.bool)
        for b in range(B):
            v_tok = valid_tokens_list[b]
            if v_tok < self.height:
                padding_mask[b, v_tok * self.width :] = True

        return {
            "spectrograms": spectrograms,
            "context_masks": context_masks_list,
            "prediction_masks": prediction_masks_list,
            "padding_mask": padding_mask,
            "measured_mask_ratios": measured_ratios,
            "ids": [item["id"] for item in batch],
        }
