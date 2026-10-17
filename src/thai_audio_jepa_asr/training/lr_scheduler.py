"""Learning rate schedulers with warmup."""

from __future__ import annotations

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as a fraction of initial LR (default: 0.01).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
