"""Optimizer configuration with component-wise learning rates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


def build_asr_optimizer(
    model: nn.Module,
    encoder_lr: float = 1e-4,
    head_lr: float = 5e-4,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with decoupled learning rates for encoder vs CTC head.

    Args:
        model: AudioJEPACTCAudioModel instance.
        encoder_lr: Learning rate for ViT encoder parameters.
        head_lr: Learning rate for temporal adapter and CTC head parameters.
        weight_decay: Weight decay factor for non-bias / non-norm weights.
        betas: Adam beta coefficients.
        eps: Adam epsilon stability factor.

    Returns:
        Configured AdamW optimizer.
    """
    encoder_decay, encoder_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in ["bias", "LayerNorm", "norm", "pos_embed", "mask_token"])

        if "encoder" in name:
            if is_no_decay:
                encoder_no_decay.append(param)
            else:
                encoder_decay.append(param)
        else:
            # Temporal adapter or CTC head
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    param_groups = [
        {"params": encoder_decay, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]

    # Filter out empty parameter groups
    valid_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(valid_groups, betas=betas, eps=eps)
