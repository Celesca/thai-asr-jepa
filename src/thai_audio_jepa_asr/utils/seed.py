"""Deterministic seed management for reproducible experiments."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Set seeds across Python random, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures CuDNN for deterministic algorithms.

    Returns:
        The seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # In PyTorch 2.x, set deterministic algorithms if desired
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    return seed
