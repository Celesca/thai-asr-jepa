"""Thai Continued Self-Supervised Adaptation of Audio-JEPA with Representation Diagnostics."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thai_audio_jepa_asr.models.audio_jepa import (
    AudioJEPAEncoder,
    TransformerBlock,
    get_2d_sincos_pos_embed,
)

logger = logging.getLogger(__name__)


def norm_mse_loss(pred: torch.Tensor, target: torch.Tensor, norm_pix_loss: bool = True) -> torch.Tensor:
    """Normalized MSE Loss matching upstream Audio-JEPA."""
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    # L2 normalize feature vectors
    target_norm = F.normalize(target, dim=-1, p=2)
    pred_norm = F.normalize(pred, dim=-1, p=2)
    # Cosine distance formulation of MSE: 2 - 2 * cos_sim
    loss = 2.0 - 2.0 * (target_norm * pred_norm).sum(dim=-1)
    return loss.mean()


def compute_representation_diagnostics(features: torch.Tensor) -> Dict[str, float]:
    """Compute representation health statistics over latent features.

    Args:
        features: Tensor of shape [B, N, D] from teacher or student.

    Returns:
        Dictionary with:
            - feature_variance: Mean variance across embedding dimension.
            - inter_utterance_similarity: Mean cosine similarity between different utterances.
            - norm_mean: Mean L2 norm of representations.
    """
    B, N, D = features.shape
    feat_var = float(features.var(dim=-1).mean().item())
    norm_mean = float(features.norm(dim=-1, p=2).mean().item())

    inter_sim = 0.0
    if B > 1:
        # Utterance-level mean embedding
        u_emb = F.normalize(features.mean(dim=1), dim=-1, p=2)  # [B, D]
        sim_mat = torch.mm(u_emb, u_emb.t())  # [B, B]
        mask = ~torch.eye(B, dtype=torch.bool, device=sim_mat.device)
        inter_sim = float(sim_mat[mask].mean().item())

    return {
        "feature_variance": round(feat_var, 4),
        "inter_utterance_similarity": round(inter_sim, 4),
        "norm_mean": round(norm_mean, 4),
    }


class VisionTransformerPredictor(nn.Module):
    """ViT Predictor predicting masked patch targets from context embeddings."""

    def __init__(
        self,
        num_patches_h: int = 16,
        num_patches_w: int = 8,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches_h * num_patches_w
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, num_patches_h, num_patches_w)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

    def forward(
        self,
        context_features: torch.Tensor,
        context_masks: List[torch.Tensor],
        prediction_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predict target patch representations given context embeddings and mask coordinates."""
        B = context_features.shape[0] // len(context_masks)

        # Project context features to predictor dimension
        x = self.predictor_embed(context_features)

        # Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        all_pos = []
        for m in context_masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_pos.append(torch.gather(x_pos_embed, dim=1, index=mask_keep))
        x = x + torch.cat(all_pos, dim=0)

        N_ctxt = x.shape[1]

        # Gather target positional embeddings
        target_pos_embs = []
        for m in prediction_masks:
            pos_keep = m.unsqueeze(-1).repeat(1, 1, self.predictor_embed.out_features)
            target_pos_embs.append(torch.gather(x_pos_embed, dim=1, index=pos_keep))
        target_pos = torch.cat(target_pos_embs, dim=0)

        # Mask tokens with target positional embeddings
        pred_tokens = self.mask_token.repeat(target_pos.size(0), target_pos.size(1), 1) + target_pos

        if target_pos.size(0) != x.size(0):
            repeat_factor = target_pos.size(0) // x.size(0)
            x = x.repeat(repeat_factor, 1, 1)

        x_full = torch.cat([x, pred_tokens], dim=1)

        for blk in self.blocks:
            x_full = blk(x_full)

        x_full = self.predictor_norm(x_full)
        preds = x_full[:, N_ctxt:]
        preds = self.predictor_proj(preds)
        return preds


class AudioJEPASSLModule(nn.Module):
    """Full Audio-JEPA Self-Supervised Adaptation architecture."""

    def __init__(
        self,
        encoder: AudioJEPAEncoder,
        predictor_depth: int = 6,
        predictor_embed_dim: int = 384,
        ema_tau: float = 0.996,
        ema_tau_end: float = 1.0,
        use_ema_schedule: bool = False,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # Online / context student encoder

        # Target teacher encoder initialized with identical adapted weights
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.ema_tau = ema_tau
        self.ema_tau_end = ema_tau_end
        self.use_ema_schedule = use_ema_schedule
        self.norm_pix_loss = norm_pix_loss

        self.predictor = VisionTransformerPredictor(
            num_patches_h=encoder.patch_embed.num_patches_h,
            num_patches_w=encoder.patch_embed.num_patches_w,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
        )

    def get_current_ema_tau(self, step: int, total_steps: int) -> float:
        """Calculate current EMA tau using cosine schedule if enabled."""
        if not self.use_ema_schedule or total_steps <= 1:
            return self.ema_tau
        progress = min(1.0, max(0.0, step / float(total_steps)))
        # Cosine ramp from ema_tau to ema_tau_end
        return self.ema_tau_end - (self.ema_tau_end - self.ema_tau) * 0.5 * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def update_target_encoder(self, tau: Optional[float] = None) -> None:
        """Update target teacher encoder strictly via Exponential Moving Average (EMA)."""
        current_tau = self.ema_tau if tau is None else tau
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(current_tau).add_(p_online.data, alpha=1.0 - current_tau)

    def forward(
        self,
        spectrograms: torch.Tensor,
        context_masks: List[torch.Tensor],
        prediction_masks: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform one masked latent prediction training step.

        Args:
            spectrograms: Tensor of shape [B, 1, H, W].
            context_masks: List of index tensors indicating context patches.
            prediction_masks: List of index tensors indicating target patches.
            padding_mask: Optional boolean padding mask [B, num_patches].

        Returns:
            Dictionary containing 'loss' and 'representation_stats'.
        """
        # 1. Forward pass online encoder with context masks
        h_context = self.encoder(spectrograms, context_masks=context_masks, padding_mask=padding_mask)

        # 2. Forward pass target encoder without masks to get full representation
        with torch.no_grad():
            h_target_full = self.target_encoder(spectrograms, context_masks=None, padding_mask=padding_mask)

        # Representation health diagnostics
        rep_stats = compute_representation_diagnostics(h_target_full)

        # 3. Predict each target mask separately
        total_loss = 0.0
        for m in prediction_masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, h_target_full.size(-1))
            h_target_m = torch.gather(h_target_full, dim=1, index=mask_keep)
            z_pred_m = self.predictor(h_context, context_masks, [m])
            total_loss = total_loss + norm_mse_loss(z_pred_m, h_target_m, norm_pix_loss=self.norm_pix_loss)

        loss = total_loss / max(1, len(prediction_masks))

        return {
            "loss": loss,
            "representation_stats": rep_stats,
        }
