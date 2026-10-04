"""CTC Speech Recognition Model with Audio-JEPA Encoder."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.temporal_adapter import TemporalAdapter, validate_ctc_feasibility

logger = logging.getLogger(__name__)


class AudioJEPACTCAudioModel(nn.Module):
    """End-to-End Thai Speech Recognition Model using Audio-JEPA and CTC."""

    def __init__(
        self,
        encoder: AudioJEPAEncoder,
        vocab_size: int,
        temporal_adapter_mode: str = "native_pooling",
        upsample_factor: int = 1,
        freeze_encoder: bool = False,
        unfreeze_last_n_blocks: Optional[int] = None,
        head_dropout: float = 0.1,
        pad_token_id: int = 0,
        loading_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AudioJEPA CTC Model.

        Args:
            encoder: Pretrained or initialized AudioJEPAEncoder.
            vocab_size: Number of tokens in tokenizer vocabulary.
            temporal_adapter_mode: 'native_pooling' (default), 'unmodified', or 'upsample_conv'.
            upsample_factor: Upsampling factor for 'upsample_conv' (default: 1).
            freeze_encoder: If True, completely freezes encoder parameters.
            unfreeze_last_n_blocks: If specified, freezes all blocks except the last N.
            head_dropout: Dropout probability before the CTC projection layer.
            pad_token_id: CTC blank token index (default: 0).
            loading_report: Optional CheckpointLoadingReport dictionary.
        """
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.loading_report = loading_report

        self.temporal_adapter = TemporalAdapter(
            embed_dim=encoder.embed_dim,
            mode=temporal_adapter_mode,
            grid_h=encoder.grid_h,
            grid_w=encoder.grid_w,
            upsample_factor=upsample_factor,
            dropout=head_dropout,
        )

        # CTC Head: dropout -> linear projection to vocab size
        self.ctc_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(encoder.embed_dim, vocab_size),
        )

        self.ctc_loss = nn.CTCLoss(
            blank=pad_token_id,
            reduction="mean",
            zero_infinity=True,
        )

        # Apply freezing policies
        self._apply_freezing_policy(freeze_encoder, unfreeze_last_n_blocks)

    def _apply_freezing_policy(
        self,
        freeze_encoder: bool,
        unfreeze_last_n_blocks: Optional[int],
    ) -> None:
        """Freeze or unfreeze encoder layers based on configuration."""
        if freeze_encoder:
            logger.info("Freezing all encoder parameters.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif unfreeze_last_n_blocks is not None and unfreeze_last_n_blocks > 0:
            logger.info("Freezing patch embed and lower blocks; unfreezing last %d blocks.", unfreeze_last_n_blocks)
            for param in self.encoder.parameters():
                param.requires_grad = False
            total_blocks = len(self.encoder.blocks)
            start_unfreeze = max(0, total_blocks - unfreeze_last_n_blocks)
            for i in range(start_unfreeze, total_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            for param in self.encoder.norm.parameters():
                param.requires_grad = True
        else:
            logger.info("Full fine-tuning mode: all encoder parameters are trainable.")
            for param in self.encoder.parameters():
                param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.ctc_head.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_trainable": encoder_trainable,
            "head_trainable": head_trainable,
        }

    def forward(
        self,
        spectrograms: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for training or inference.

        Args:
            spectrograms: Mel-spectrogram tensor of shape [B, 1, H, W].
            targets: Optional padded targets of shape [B, max_target_len].
            target_lengths: Optional target length tensor of shape [B].
            padding_mask: Optional attention padding mask [B, num_patches].

        Returns:
            Dictionary containing 'logits', 'log_probs', 'input_lengths', and optionally 'loss'.
        """
        # Encoder pass without masking
        encoded = self.encoder(spectrograms, context_masks=None, padding_mask=padding_mask)

        # Temporal adapter pass
        temporal_feats, input_lengths = self.temporal_adapter(encoded)  # [B, T_out, embed_dim]

        # CTC projection
        logits = self.ctc_head(temporal_feats)  # [B, T_out, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        result: Dict[str, Any] = {
            "logits": logits,
            "log_probs": log_probs,
            "input_lengths": input_lengths,
        }

        # Calculate loss if targets are provided
        if targets is not None and target_lengths is not None:
            # Validate CTC feasibility with adjacent repeated token handling
            _ = validate_ctc_feasibility(
                input_lengths=input_lengths,
                targets=targets,
                target_lengths=target_lengths,
                raise_on_failure=False,
            )

            # CTCLoss expects log_probs in [T, B, C]
            log_probs_t = log_probs.transpose(0, 1)
            loss = self.ctc_loss(log_probs_t, targets, input_lengths, target_lengths)
            result["loss"] = loss

        return result
