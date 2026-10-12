"""Optimizer configuration with decoupled component-wise learning rates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_asr_optimizer(
    model: nn.Module,
    encoder_lr: Optional[float] = None,
    backbone_lr: Optional[float] = None,
    patch_embed_lr: Optional[float] = None,
    head_lr: float = 3e-4,
    backbone_lr_ratio: float = 0.1,
    patch_embed_lr_ratio: float = 1.0,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with three decoupled component learning rate groups:

    1. Pretrained backbone (Transformer blocks + norm) -> backbone_lr
    2. New patch projection (patch_embed Conv2d) -> patch_embed_lr
    3. Temporal adapter & CTC prediction head -> head_lr

    Args:
        model: AudioJEPACTCAudioModel instance.
        encoder_lr: Legacy alias for backbone_lr.
        backbone_lr: Learning rate for ViT backbone blocks.
        patch_embed_lr: Learning rate for patch projection layer.
        head_lr: Learning rate for temporal adapter and CTC head.
        backbone_lr_ratio: Ratio of backbone_lr to head_lr if backbone_lr not specified.
        patch_embed_lr_ratio: Ratio of patch_embed_lr to head_lr if patch_embed_lr not specified.
        weight_decay: Weight decay factor for non-bias / non-norm weights.
        betas: Adam beta coefficients.
        eps: Adam epsilon stability factor.

    Returns:
        Configured AdamW optimizer.
    """
    effective_backbone_lr = backbone_lr or encoder_lr or (head_lr * backbone_lr_ratio)
    effective_patch_embed_lr = patch_embed_lr or (head_lr * patch_embed_lr_ratio)
    effective_head_lr = head_lr

    backbone_decay, backbone_no_decay = [], []
    patch_decay, patch_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in ["bias", "LayerNorm", "norm", "pos_embed", "mask_token"])

        if "encoder.patch_embed" in name:
            if is_no_decay:
                patch_no_decay.append(param)
            else:
                patch_decay.append(param)
        elif "encoder" in name:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            # Temporal adapter or CTC head
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    param_groups = [
        {"params": backbone_decay, "lr": effective_backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"},
        {"params": backbone_no_decay, "lr": effective_backbone_lr, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": patch_decay, "lr": effective_patch_embed_lr, "weight_decay": weight_decay, "name": "patch_embed_decay"},
        {"params": patch_no_decay, "lr": effective_patch_embed_lr, "weight_decay": 0.0, "name": "patch_embed_no_decay"},
        {"params": head_decay, "lr": effective_head_lr, "weight_decay": weight_decay, "name": "head_decay"},
        {"params": head_no_decay, "lr": effective_head_lr, "weight_decay": 0.0, "name": "head_no_decay"},
    ]

    valid_groups = [g for g in param_groups if len(g["params"]) > 0]

    logger.info(
        "ASR Optimizer: %d parameter groups configured (Backbone LR: %.2e, PatchEmbed LR: %.2e, Head LR: %.2e).",
        len(valid_groups),
        effective_backbone_lr,
        effective_patch_embed_lr,
        effective_head_lr,
    )
    for g in valid_groups:
        logger.info("  Group '%s': %d tensors, LR: %.2e, Weight Decay: %.2e", g["name"], len(g["params"]), g["lr"], g["weight_decay"])

    return torch.optim.AdamW(valid_groups, betas=betas, eps=eps)
