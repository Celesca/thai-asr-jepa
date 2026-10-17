"""Configuration file loading and resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Dictionary of parsed YAML contents.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_config(config_path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load config and recursively resolve any `defaults` or `base` configs.

    Args:
        config_path: Path to the main YAML config file.
        overrides: Optional runtime overrides dictionary.

    Returns:
        Fully resolved configuration dictionary.
    """
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent
    config = load_yaml(config_path)

    # If the config specifies a 'base' or 'defaults'
    if "base" in config:
        base_paths = config.pop("base")
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged: Dict[str, Any] = {}
        for bp in base_paths:
            resolved_bp = (base_dir / bp).resolve() if not Path(bp).is_absolute() else Path(bp)
            base_cfg = resolve_config(resolved_bp)
            merged = deep_merge(merged, base_cfg)
        config = deep_merge(merged, config)

    if overrides:
        config = deep_merge(config, overrides)

    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
