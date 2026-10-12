"""Temporal Resolution and CTC Alignment Validation.

Implements exact CTC sequence feasibility validation accounting for adjacent repeated tokens:
    T_output >= target_length + adjacent_repeated_token_count
Supports both modified fine temporal patching (native encoder tokens) and
original coarse patching (Configuration A diagnostic baseline).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_ctc_min_required_length(tokens: Union[List[int], torch.Tensor]) -> int:
    """Compute minimum CTC output length required to align a token sequence.

    Formula:
        min_required_length = target_length + adjacent_repeated_token_count
    In CTC, consecutive identical non-blank tokens require at least one blank token between them.
    """
    if isinstance(tokens, torch.Tensor):
        tokens_list = tokens.tolist()
    else:
        tokens_list = list(tokens)

    # Filter out padding tokens (0) if present at end
    valid_tokens = [t for t in tokens_list if t != 0]
    target_length = len(valid_tokens)
    if target_length <= 1:
        return target_length

    adjacent_repeats = 0
    for i in range(target_length - 1):
        if valid_tokens[i] == valid_tokens[i + 1]:
            adjacent_repeats += 1

    return target_length + adjacent_repeats


def validate_ctc_feasibility(
    input_lengths: torch.Tensor,
    target_lengths: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    raise_on_failure: bool = False,
) -> Dict[str, Any]:
    """Verify that output sequence length T is sufficient for target transcript length.

    Computes:
        T_output >= target_length + adjacent_repeated_token_count

    Args:
        input_lengths: Tensor of output time frame lengths [B].
        target_lengths: Tensor of target token sequence lengths [B].
        targets: Optional target token matrix [B, max_len] to inspect repeated adjacent tokens.
        raise_on_failure: If True, raises ValueError when any sample violates feasibility.

    Returns:
        Dictionary summarizing feasibility statistics and percentage.
    """
    # Handle case where user passed 2D targets as second argument positionally
    if target_lengths is not None and target_lengths.ndim > 1:
        targets = target_lengths
        target_lengths = None

    total_samples = len(input_lengths)
    if total_samples == 0:
        return {
            "is_feasible": True,
            "feasible_percentage": 100.0,
            "total_samples": 0,
            "num_valid": 0,
            "num_violations": 0,
            "violation_rate": 0.0,
        }

    # Calculate exact minimum length required per sample
    min_required_lengths = []
    if targets is not None:
        for i in range(total_samples):
            t_len = int(target_lengths[i].item()) if target_lengths is not None else targets.shape[1]
            sample_tokens = targets[i][:t_len]
            min_len = compute_ctc_min_required_length(sample_tokens)
            min_required_lengths.append(min_len)
        required_tensor = torch.tensor(min_required_lengths, dtype=torch.long, device=input_lengths.device)
    elif target_lengths is not None:
        required_tensor = target_lengths
    else:
        raise ValueError("Either targets or target_lengths must be provided.")

    violations = (input_lengths < required_tensor)
    num_violations = int(violations.sum().item())
    num_feasible = total_samples - num_violations
    feasible_percentage = (num_feasible / total_samples) * 100.0
    violation_rate = float(num_violations / total_samples)

    # Compute ratio distribution: T_valid / (U + R)
    ratios = (input_lengths.float() / required_tensor.float().clamp(min=1.0)).cpu()
    violations_detail: List[Dict[str, Any]] = []

    if num_violations > 0:
        for i in range(total_samples):
            if bool(violations[i].item()):
                t_val = int(input_lengths[i].item())
                req_val = int(required_tensor[i].item())
                violations_detail.append({
                    "sample_idx": i,
                    "t_valid": t_val,
                    "required_min_length": req_val,
                    "deficit": req_val - t_val,
                    "reason": f"Output length {t_val} < required length {req_val} (U+R)",
                })

    stats = {
        "is_feasible": (num_violations == 0),
        "total_samples": total_samples,
        "num_valid": num_feasible,
        "num_feasible": num_feasible,
        "num_violations": num_violations,
        "violation_rate": round(violation_rate, 4),
        "feasible_percentage": round(feasible_percentage, 2),
        "min_input_len": int(input_lengths.min().item()),
        "max_input_len": int(input_lengths.max().item()),
        "max_required_len": int(required_tensor.max().item()),
        "mean_input_len": float(input_lengths.float().mean().item()),
        "mean_required_len": float(required_tensor.float().mean().item()),
        "ratio_distribution": {
            "min": round(float(ratios.min().item()), 3),
            "mean": round(float(ratios.mean().item()), 3),
            "median": round(float(ratios.median().item()), 3),
            "max": round(float(ratios.max().item()), 3),
        },
        "violations_detail": violations_detail,
    }

    if num_violations > 0:
        msg = (
            f"CTC Feasibility Warning: {num_feasible}/{total_samples} samples ({feasible_percentage:.1f}%) "
            f"feasible. {num_violations} violations! (T_out min: {stats['min_input_len']}, "
            f"Required max: {stats['max_required_len']}, T/(U+R) min: {stats['ratio_distribution']['min']})"
        )
        if raise_on_failure:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    else:
        logger.info(
            "CTC Feasibility: 100%% (%d/%d samples feasible, T/(U+R) mean: %.2f, min: %.2f).",
            total_samples,
            total_samples,
            stats["ratio_distribution"]["mean"],
            stats["ratio_distribution"]["min"],
        )

    return stats


class TemporalAdapter(nn.Module):
    """Adapter connecting 2D ViT patch representations to CTC sequence modeling.

    Supported modes:
        - 'native_pooling': (Recommended for Modified Audio-JEPA) Takes fine temporal patches
          (e.g. 256 time steps) and pools across frequency bins without post-encoder temporal interpolation.
        - 'unmodified': (Configuration A baseline) Takes original coarse patches (16 time bins, 1.6 fps)
          and pools across frequency bins, used for honest baseline diagnosis.
        - 'upsample_conv': Optional 1D convolution interpolation layer if used as auxiliary comparison.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mode: str = "native_pooling",
        grid_h: int = 256,
        grid_w: int = 8,
        upsample_factor: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.upsample_factor = upsample_factor

        if mode == "upsample_conv" and upsample_factor > 1:
            self.upsampler = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
        else:
            self.upsampler = None

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert encoder patch representations into temporal sequence.

        Args:
            x: Tensor of shape [B, num_patches, embed_dim].
            valid_lengths: Optional true sequence lengths [B] of valid temporal tokens.

        Returns:
            Tuple of (temporal_features: [B, T_out, embed_dim], lengths: [B]).
        """
        B, N, D = x.shape
        assert N == self.grid_h * self.grid_w, f"Token count mismatch: expected {self.grid_h * self.grid_w}, got {N}"

        # Reshape to [B, grid_h, grid_w, D]
        x_2d = x.view(B, self.grid_h, self.grid_w, D)
        # Pool across frequency dimension: [B, grid_h, D]
        x_temporal = x_2d.mean(dim=2)

        if valid_lengths is not None:
            lengths = valid_lengths.to(x.device)
        else:
            lengths = torch.full((B,), self.grid_h, dtype=torch.long, device=x.device)

        if self.mode in ("native_pooling", "unmodified") or self.upsampler is None:
            return x_temporal, lengths

        elif self.mode == "upsample_conv":
            x_conv = x_temporal.transpose(1, 2)
            target_t = self.grid_h * self.upsample_factor
            x_interp = torch.nn.functional.interpolate(x_conv, size=target_t, mode="linear", align_corners=False)
            x_out = self.upsampler(x_interp).transpose(1, 2)
            out_lengths = lengths * self.upsample_factor
            return x_out, out_lengths

        raise ValueError(f"Unknown TemporalAdapter mode: {self.mode}")
