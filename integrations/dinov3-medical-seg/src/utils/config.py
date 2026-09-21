"""YAML configuration loading with deep merge support."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str, overrides: dict = None) -> Dict[str, Any]:
    """Load a YAML config file with optional override dict.

    Supports _base_ inheritance: if config has a _base_ key pointing to
    another YAML file, that base config is loaded first and overridden
    by the current config.

    Args:
        config_path: path to YAML config file
        overrides: dict of dotted keys to override (e.g. {'training.lr': 1e-4})

    Returns:
        merged config dict
    """
    config = _load_yaml(config_path)

    # Handle _base_ inheritance
    if "_base_" in config:
        base_path = config.pop("_base_")
        raw_paths = base_path if isinstance(base_path, list) else [base_path]
        config_dir = os.path.dirname(os.path.abspath(config_path))
        base_paths = [
            path if os.path.isabs(path) else os.path.normpath(os.path.join(config_dir, path))
            for path in raw_paths
        ]

        # Load and merge bases (later bases override earlier)
        base_config = {}
        for bp in base_paths:
            base_config = deep_merge(base_config, load_config(bp))
        config = deep_merge(base_config, config)

    # Resolve relative model_path against the project root directory so that
    # training and inference work regardless of the current working directory.
    model_path = config.get("model", {}).get("model_path")
    if model_path and not os.path.isabs(model_path):
        # config.py lives at <project_root>/src/utils/config.py → go up 2 levels
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config["model"]["model_path"] = os.path.normpath(os.path.join(project_root, model_path))

    # Apply CLI overrides
    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    return config


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, key: str, value: Any):
    """Set a nested dict value by dotted key: 'training.lr' → d['training']['lr']."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Auto-convert types
    existing = d.get(keys[-1])
    if existing is not None and not isinstance(value, type(existing)):
        value = type(existing)(value)
    d[keys[-1]] = value
