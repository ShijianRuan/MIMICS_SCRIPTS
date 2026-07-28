"""Configuration loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = {
    "model": {
        "base_model_dir": "",
        "fold": "0",
        "checkpoint_name": "checkpoint_final.pth",
        "strategy": "clopa_in",
    },
    "data": {
        "manifest": "",
        "label_values": [1],
        "validation_fraction": 0.2,
        "patch_size": [128, 128, 128],
        "foreground_patch_probability": 0.5,
        "num_workers": 0,
        "prepared_cache_dir": "",
        "keep_prepared_cache": True,
        "augmentation": {
            "enabled": True,
            "flip_probability": 0.5,
            "intensity_scale_range": [0.9, 1.1],
            "intensity_shift_range": [-0.1, 0.1],
            "noise_std_range": [0.0, 0.05],
        },
    },
    "prompts": {
        "mode": "clicks",
        "interaction_steps": 5,
        "point_radius": 4,
        "center_bias": 8.0,
        "interaction_decay": 0.9,
    },
    "training": {
        "output_dir": "",
        "epochs": 10,
        "steps_per_epoch": 50,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "mixed_precision": True,
        "seed": 20260724,
        "validation_batches": 8,
        "device": "auto",
        "resume": True,
        "status_path": "",
        "cancel_path": "",
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Training configuration must be a JSON/YAML object.")
    return value


def _resolve_path(value: str, base: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    prompts = config["prompts"]
    training = config["training"]

    if model["strategy"] not in {"clopa_in", "clopa_conv", "full"}:
        raise ValueError("model.strategy must be clopa_in, clopa_conv, or full.")
    if prompts["mode"] != "clicks":
        raise ValueError(
            "Only prompts.mode=clicks is enabled in the validated first release. "
            "Mixed prompt training requires a separate equivalence study."
        )

    patch_size = tuple(int(value) for value in data["patch_size"])
    if len(patch_size) != 3 or any(value < 64 or value % 32 for value in patch_size):
        raise ValueError(
            "data.patch_size must contain three multiples of 32, each at least 64."
        )
    if not 0.0 <= float(data["foreground_patch_probability"]) <= 1.0:
        raise ValueError("data.foreground_patch_probability must be in [0, 1].")
    if not 0.0 <= float(data["validation_fraction"]) < 1.0:
        raise ValueError("data.validation_fraction must be in [0, 1).")
    if not data.get("label_values"):
        raise ValueError(
            "data.label_values must contain at least one foreground value."
        )
    if "strict_geometry" in data and not bool(data["strict_geometry"]):
        raise ValueError(
            "data.strict_geometry=false is unsafe and no longer supported. "
            "Image and label must describe the same physical voxel grid."
        )
    augmentation = data["augmentation"]
    if not 0.0 <= float(augmentation["flip_probability"]) <= 1.0:
        raise ValueError("data.augmentation.flip_probability must be in [0, 1].")
    for key in ("intensity_scale_range", "intensity_shift_range", "noise_std_range"):
        values = augmentation[key]
        if len(values) != 2 or float(values[0]) > float(values[1]):
            raise ValueError(
                "data.augmentation.{} must be an ordered pair.".format(key)
            )
    if float(augmentation["intensity_scale_range"][0]) <= 0:
        raise ValueError("data.augmentation.intensity_scale_range must be positive.")
    if float(augmentation["noise_std_range"][0]) < 0:
        raise ValueError("data.augmentation.noise_std_range cannot be negative.")

    positive_ints = (
        ("prompts.interaction_steps", prompts["interaction_steps"]),
        ("prompts.point_radius", prompts["point_radius"]),
        ("training.epochs", training["epochs"]),
        ("training.steps_per_epoch", training["steps_per_epoch"]),
        ("training.batch_size", training["batch_size"]),
        ("training.gradient_accumulation", training["gradient_accumulation"]),
        ("training.validation_batches", training["validation_batches"]),
    )
    for name, value in positive_ints:
        if int(value) < 1:
            raise ValueError("{} must be at least 1.".format(name))
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training["weight_decay"]) < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if int(data["num_workers"]) < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if float(prompts["center_bias"]) <= 0:
        raise ValueError("prompts.center_bias must be positive.")
    if not 0.0 < float(prompts["interaction_decay"]) <= 1.0:
        raise ValueError("prompts.interaction_decay must be in (0, 1].")

    if not model["base_model_dir"]:
        raise ValueError("model.base_model_dir is required.")
    if not data["manifest"]:
        raise ValueError("data.manifest is required.")
    if not training["output_dir"]:
        raise ValueError("training.output_dir is required.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config and resolve all filesystem paths relative to it."""
    source = Path(path).expanduser().resolve()
    config = _merge(DEFAULT_CONFIG, _read(source))
    base = source.parent
    config["model"]["base_model_dir"] = _resolve_path(
        str(config["model"].get("base_model_dir") or ""), base
    )
    config["data"]["manifest"] = _resolve_path(
        str(config["data"].get("manifest") or ""), base
    )
    config["data"]["prepared_cache_dir"] = _resolve_path(
        str(config["data"].get("prepared_cache_dir") or ""), base
    )
    for key in ("output_dir", "status_path", "cancel_path"):
        config["training"][key] = _resolve_path(
            str(config["training"].get(key) or ""), base
        )
    config["_config_path"] = str(source)
    validate_config(config)
    return config
