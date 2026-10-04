"""Audio-JEPA Vision Transformer Encoder with configurable temporal patching.

Supports both original coarse patching (16, 16) and fine temporal patching (e.g. 1, 16 or 2, 16)
for downstream CTC speech recognition. Includes 2D positional embedding interpolation,
attention/padding masking, and robust checkpoint transfer reporting.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


@dataclass
class CheckpointLoadingReport:
    """Detailed report of pretrained checkpoint transfer."""
    checkpoint_source: str
    total_model_parameters: int
    loaded_parameters_count: int
    loaded_percentage: float
    loaded_parameter_names: List[str]
    missing_parameter_names: List[str]
    unexpected_parameter_names: List[str]
    shape_mismatched_parameters: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding."""
    grid_h_arr = np.arange(grid_h, dtype=np.float32)
    grid_w_arr = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_arr, grid_h_arr)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h, grid_w])

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding from grid coordinates."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class Attention(nn.Module):
    """Multi-Head Self-Attention compatible with Audio-JEPA checkpoint weights."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # PyTorch native scaled dot-product attention
        # Note: attn_mask can be boolean or float
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    """Standard ViT MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block matching Audio-JEPA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            qkv_proj_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Configurable 2D Patch Embedding for Mel-Spectrogram inputs.

    Supports asymmetric patch sizes, e.g. (patch_time, patch_freq) = (1, 16) or (16, 16).
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 128),
        patch_size: Tuple[int, int] = (1, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches_h = input_size[0] // patch_size[0]
        self.num_patches_w = input_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class AudioJEPAEncoder(nn.Module):
    """Audio-JEPA Vision Transformer Encoder with configurable temporal patching.

    Keeps upstream depth, width, heads, LayerNorm, and GELU.
    Allows asymmetric patch size (e.g. patch_size=(1, 16)) to preserve fine temporal resolution
    before Transformer encoding, enabling CTC alignment on Thai speech without post-encoder upsampling.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 128),
        patch_size: Tuple[int, int] = (1, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        clip_length_s: float = 10.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.clip_length_s = clip_length_s

        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_h = self.patch_embed.num_patches_h
        self.grid_w = self.patch_embed.num_patches_w

        # Positional embedding initialized with 2D sin-cos
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))

        # Upstream transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    @property
    def temporal_rate(self) -> float:
        """Encoder time tokens per second."""
        return float(self.grid_h) / float(self.clip_length_s)

    @property
    def temporal_token_rate(self) -> float:
        """Alias for temporal_rate: encoder time tokens per second."""
        return self.temporal_rate

    def interpolate_pos_encoding_2d(
        self,
        src_pos_embed: torch.Tensor,
        src_grid: Tuple[int, int],
        tgt_grid: Tuple[int, int],
    ) -> torch.Tensor:
        """Interpolate 2D sinusoidal positional embeddings between different patch grids."""
        src_h, src_w = src_grid
        tgt_h, tgt_w = tgt_grid
        if (src_h, src_w) == (tgt_h, tgt_w):
            return src_pos_embed

        dim = src_pos_embed.shape[-1]
        # Reshape to [1, dim, src_h, src_w]
        pe_2d = src_pos_embed.reshape(1, src_h, src_w, dim).permute(0, 3, 1, 2)
        pe_interp = F.interpolate(pe_2d, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False)
        # Reshape back to [1, tgt_h * tgt_w, dim]
        return pe_interp.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, dim)

    def to_temporal_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert 2D ViT patch representations into 1D temporal frame sequence.

        Reshapes [B, grid_h * grid_w, embed_dim] into [B, grid_h, grid_w, embed_dim]
        and pools across frequency bins (mean pooling).

        Returns:
            Tuple of (temporal_features: [B, grid_h, embed_dim], lengths: [B]).
        """
        B, N, D = x.shape
        assert N == self.grid_h * self.grid_w, f"Token mismatch: expected {self.grid_h * self.grid_w}, got {N}"
        x_2d = x.view(B, self.grid_h, self.grid_w, D)
        # Pool across frequency bins: [B, grid_h, embed_dim]
        x_temporal = x_2d.mean(dim=2)
        lengths = torch.full((B,), self.grid_h, dtype=torch.long, device=x.device)
        return x_temporal, lengths

    def forward(
        self,
        x: torch.Tensor,
        context_masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Audio-JEPA encoder.

        Args:
            x: Mel-spectrogram tensor of shape [B, 1, H, W] or [B, H, W].
            context_masks: Mask indices for SSL masked latent prediction training.
            padding_mask: Optional boolean attention padding mask [B, num_patches].

        Returns:
            Encoded patch representations [B, N_tokens, embed_dim].
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)
        x = x + self.pos_embed

        if context_masks is not None:
            if isinstance(context_masks, list):
                all_x = []
                for m in context_masks:
                    mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
                    all_x.append(torch.gather(x, dim=1, index=mask_keep))
                x = torch.cat(all_x, dim=0)
            else:
                mask_keep = context_masks.unsqueeze(-1).repeat(1, 1, x.size(-1))
                x = torch.gather(x, dim=1, index=mask_keep)

        # Convert boolean padding mask to additive attention mask if provided
        attn_mask = None
        if padding_mask is not None:
            # Shape for SDPA: [B, 1, 1, N]
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

    @classmethod
    def from_pretrained_transfer(
        cls,
        model_name_or_path: str = "ltuncay/Audio-JEPA",
        filename: str = "JEPA.ckpt",
        patch_size: Tuple[int, int] = (1, 16),
        input_size: Tuple[int, int] = (256, 128),
        clip_length_s: float = 10.0,
        device: str | torch.device = "cpu",
        interpolate_pos_embed: bool = True,
    ) -> Tuple[AudioJEPAEncoder, CheckpointLoadingReport]:
        """Transfer pretrained Audio-JEPA weights into an encoder with target patch size.

        Reuses transformer blocks, LayerNorm, attention, and MLP weights when shapes match.
        Interpolates positional embeddings and reinitializes shape-mismatched patch projection.
        Generates and returns an explicit CheckpointLoadingReport.
        """
        path = Path(model_name_or_path)
        if path.is_file():
            ckpt_path = str(path)
        else:
            logger.info("Downloading checkpoint %s from %s...", filename, model_name_or_path)
            ckpt_path = hf_hub_download(model_name_or_path, filename)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = state.get("state_dict", state)

        # Extract encoder keys
        source_sd: Dict[str, torch.Tensor] = {}
        for k, v in raw_sd.items():
            if k.startswith("encoder.") and not k.startswith("encoder_"):
                source_sd[k[len("encoder.") :]] = v

        # Instantiate target model
        model = cls(
            input_size=input_size,
            patch_size=patch_size,
            clip_length_s=clip_length_s,
        )
        target_sd = model.state_dict()

        loaded_names: List[str] = []
        shape_mismatched: List[Dict[str, Any]] = []
        unexpected_names = [k for k in source_sd if k not in target_sd]
        compatible_sd: Dict[str, torch.Tensor] = {}

        for name, target_tensor in target_sd.items():
            if name in source_sd:
                source_tensor = source_sd[name]
                if source_tensor.shape == target_tensor.shape:
                    compatible_sd[name] = source_tensor
                    loaded_names.append(name)
                elif name == "pos_embed" and interpolate_pos_embed:
                    # Positional embedding 2D interpolation from (16, 8) to target grid
                    src_grid = (16, 8)  # Original Audio-JEPA grid
                    tgt_grid = (model.grid_h, model.grid_w)
                    interpolated_pe = model.interpolate_pos_encoding_2d(source_tensor, src_grid, tgt_grid)
                    compatible_sd[name] = interpolated_pe
                    loaded_names.append(name)
                    logger.info("Interpolated pos_embed from %s to %s", src_grid, tgt_grid)
                else:
                    shape_mismatched.append({
                        "name": name,
                        "source_shape": list(source_tensor.shape),
                        "target_shape": list(target_tensor.shape),
                    })
                    logger.info(
                        "Reinitializing parameter '%s' due to shape mismatch: source %s != target %s",
                        name,
                        list(source_tensor.shape),
                        list(target_tensor.shape),
                    )

        missing_names = [k for k in target_sd if k not in compatible_sd]

        # Load compatible weights
        model.load_state_dict(compatible_sd, strict=False)

        total_params = sum(p.numel() for p in model.parameters())
        loaded_param_count = sum(compatible_sd[k].numel() for k in compatible_sd if k in target_sd)
        loaded_pct = (loaded_param_count / max(1, total_params)) * 100.0

        report = CheckpointLoadingReport(
            checkpoint_source=str(model_name_or_path),
            total_model_parameters=total_params,
            loaded_parameters_count=loaded_param_count,
            loaded_percentage=round(loaded_pct, 4),
            loaded_parameter_names=loaded_names,
            missing_parameter_names=missing_names,
            unexpected_parameter_names=unexpected_names,
            shape_mismatched_parameters=shape_mismatched,
        )

        logger.info(
            "Audio-JEPA Checkpoint Transfer: %d/%d parameters (%.2f%%) initialized from pretrained weights.",
            loaded_param_count,
            total_params,
            loaded_pct,
        )

        return model, report

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "ltuncay/Audio-JEPA",
        filename: str = "JEPA.ckpt",
        device: str | torch.device = "cpu",
    ) -> AudioJEPAEncoder:
        """Backward-compatible loader using transfer logic with default coarse patching."""
        model, _ = cls.from_pretrained_transfer(
            model_name_or_path=model_name_or_path,
            filename=filename,
            patch_size=(16, 16),
            device=device,
        )
        return model
