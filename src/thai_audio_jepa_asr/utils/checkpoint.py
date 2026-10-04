"""Checkpoint saving, loading, and inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "checkpoint_latest.pt",
    is_best: bool = False,
) -> Path:
    """Save training state safely to disk.

    Args:
        state_dict: Dictionary containing model, optimizer, step, metrics, etc.
        checkpoint_dir: Directory where checkpoints are stored.
        filename: Filename for this checkpoint.
        is_best: If True, also copies to 'best_checkpoint.pt'.

    Returns:
        Path to the saved checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_path = ckpt_dir / filename
    temp_path = ckpt_dir / f"{filename}.tmp"

    torch.save(state_dict, temp_path)
    os.replace(temp_path, target_path)

    if is_best:
        best_path = ckpt_dir / "best_checkpoint.pt"
        torch.save(state_dict, best_path)

    return target_path


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint dictionary safely.

    Args:
        checkpoint_path: Path to checkpoint file.
        map_location: Target device for loading tensors.

    Returns:
        Checkpoint dictionary.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location, weights_only=False)
