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
            # nnInteractive is trained on the nnU-Net augmentation family.
            # Keep these values aligned with nnUNetTrainer defaults while
            # implementing them locally so the offline environment does not
            # need batchgeneratorsv2.
            "profile": "nninteractive_nnunet",
            "flip_probability": 0.5,
            "mirror_axes": [0, 1, 2],
            "rotation_probability": 0.2,
            "rotation_degrees": [-30.0, 30.0],
            "scaling_probability": 0.2,
            "scaling_range": [0.7, 1.4],
            "noise_probability": 0.1,
            "noise_variance_range": [0.0, 0.1],
            "blur_probability": 0.2,
            "blur_channel_probability": 0.5,
            "blur_sigma_range": [0.5, 1.0],
            "brightness_probability": 0.15,
            "brightness_multiplier_range": [0.75, 1.25],
            "contrast_probability": 0.15,
            "contrast_range": [0.75, 1.25],
            "low_resolution_probability": 0.25,
            "low_resolution_channel_probability": 0.5,
            "low_resolution_scale_range": [0.5, 1.0],
            "gamma_invert_probability": 0.1,
            "gamma_probability": 0.3,
            "gamma_range": [0.7, 1.5],
        },
    },
    "prompts": {
        "mode": "clicks",
        # Each correction step follows the CLoPA paper: one foreground and
        # one background correction where the corresponding error exists.
        # The moderate long tail exposes late editing without the cost of the
        # paper's 100-step research-only evaluation protocol.
        "training_goal": "general",
        "correction_policy": "clopa_paired",
        "interaction_steps": 8,
        "min_interaction_steps": 1,
        "max_interaction_steps": 8,
        "interaction_step_weights": [
            0.28,
            0.24,
            0.17,
            0.12,
            0.08,
            0.05,
            0.035,
            0.025,
        ],
        "short_interaction_probability": 0.65,
        "validation_interaction_steps": [1, 3, 5, 8],
        "initial_mask_probability": 0.5,
        "validate_initial_masks": True,
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
    if "provided_initial_mask_probability" in prompts:
        raise ValueError(
            "prompts.provided_initial_mask_probability was removed because "
            "synthetic Initial Masks are no longer generated."
        )
    training = config["training"]

    if model["strategy"] not in {"clopa_in", "clopa_conv", "full"}:
        raise ValueError("model.strategy must be clopa_in, clopa_conv, or full.")
    if prompts["mode"] != "clicks":
        raise ValueError(
            "Only prompts.mode=clicks is enabled in the validated first release. "
            "Mixed prompt training requires a separate equivalence study."
        )
    if prompts.get("training_goal", "general") not in {
        "general",
        "start_empty",
        "refine_existing",
        "legacy",
    }:
        raise ValueError(
            "prompts.training_goal must be general, start_empty, "
            "refine_existing, or legacy."
        )
    if prompts.get("correction_policy", "clopa_paired") not in {
        "official_single",
        "clopa_paired",
    }:
        raise ValueError(
            "prompts.correction_policy must be official_single or clopa_paired."
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
    profile = str(augmentation.get("profile") or "legacy").strip().lower()
    if profile not in {"legacy", "nninteractive_nnunet"}:
        raise ValueError(
            "data.augmentation.profile must be legacy or nninteractive_nnunet."
        )
    if not 0.0 <= float(augmentation["flip_probability"]) <= 1.0:
        raise ValueError("data.augmentation.flip_probability must be in [0, 1].")
    probability_keys = (
        "rotation_probability",
        "scaling_probability",
        "noise_probability",
        "blur_probability",
        "blur_channel_probability",
        "brightness_probability",
        "contrast_probability",
        "low_resolution_probability",
        "low_resolution_channel_probability",
        "gamma_invert_probability",
        "gamma_probability",
    )
    for key in probability_keys:
        if key in augmentation and not 0.0 <= float(augmentation[key]) <= 1.0:
            raise ValueError(
                "data.augmentation.{} must be in [0, 1].".format(key)
            )
    range_keys = (
        "rotation_degrees",
        "scaling_range",
        "noise_variance_range",
        "blur_sigma_range",
        "brightness_multiplier_range",
        "contrast_range",
        "low_resolution_scale_range",
        "gamma_range",
    )
    if profile == "legacy":
        range_keys = range_keys + (
            "intensity_scale_range",
            "intensity_shift_range",
            "noise_std_range",
        )
    for key in range_keys:
        if key not in augmentation:
            continue
        values = augmentation[key]
        if len(values) != 2 or float(values[0]) > float(values[1]):
            raise ValueError(
                "data.augmentation.{} must be an ordered pair.".format(key)
            )
    positive_range_keys = (
        "scaling_range",
        "blur_sigma_range",
        "brightness_multiplier_range",
        "contrast_range",
        "low_resolution_scale_range",
        "gamma_range",
    )
    for key in positive_range_keys:
        if key in augmentation and float(augmentation[key][0]) <= 0:
            raise ValueError(
                "data.augmentation.{} must be positive.".format(key)
            )
    if (
        "noise_variance_range" in augmentation
        and float(augmentation["noise_variance_range"][0]) < 0
    ):
        raise ValueError(
            "data.augmentation.noise_variance_range cannot be negative."
        )
    mirror_axes = [int(value) for value in augmentation.get("mirror_axes", [])]
    if any(value not in (0, 1, 2) for value in mirror_axes) or len(
        mirror_axes
    ) != len(set(mirror_axes)):
        raise ValueError(
            "data.augmentation.mirror_axes must contain unique axes from 0, 1, 2."
        )

    positive_ints = (
        ("prompts.interaction_steps", prompts["interaction_steps"]),
        (
            "prompts.min_interaction_steps",
            prompts["min_interaction_steps"],
        ),
        (
            "prompts.max_interaction_steps",
            prompts["max_interaction_steps"],
        ),
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
    if int(prompts["min_interaction_steps"]) > int(
        prompts["max_interaction_steps"]
    ):
        raise ValueError(
            "prompts.min_interaction_steps cannot exceed "
            "prompts.max_interaction_steps."
        )
    validation_steps = [
        int(value)
        for value in prompts.get("validation_interaction_steps") or []
    ]
    if (
        not validation_steps
        or any(value < 1 for value in validation_steps)
        or validation_steps != sorted(set(validation_steps))
        or validation_steps[-1] > int(prompts["max_interaction_steps"])
    ):
        raise ValueError(
            "prompts.validation_interaction_steps must be unique increasing "
            "values within the configured interaction range."
        )
    for name in (
        "short_interaction_probability",
        "initial_mask_probability",
    ):
        if not 0.0 <= float(prompts[name]) <= 1.0:
            raise ValueError("prompts.{} must be in [0, 1].".format(name))
    weights = prompts.get("interaction_step_weights")
    if weights is not None:
        expected = (
            int(prompts["max_interaction_steps"])
            - int(prompts["min_interaction_steps"])
            + 1
        )
        if len(weights) != expected:
            raise ValueError(
                "prompts.interaction_step_weights must contain one value for "
                "each configured interaction count."
            )
        if (
            any(float(value) < 0 for value in weights)
            or sum(float(value) for value in weights) <= 0
        ):
            raise ValueError(
                "prompts.interaction_step_weights must be non-negative and "
                "contain a positive value."
            )
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
    override = _read(source)
    prompt_override = override.get("prompts") or {}
    if "provided_initial_mask_probability" in prompt_override:
        raise ValueError(
            "prompts.provided_initial_mask_probability was removed because "
            "synthetic Initial Masks are no longer generated. Use real Initial "
            "Masks with prompts.initial_mask_probability, or start empty."
        )
    config = _merge(DEFAULT_CONFIG, override)
    if (
        "interaction_steps" in prompt_override
        and "max_interaction_steps" not in prompt_override
    ):
        legacy_horizon = int(prompt_override["interaction_steps"])
        config["prompts"]["max_interaction_steps"] = legacy_horizon
        # Preserve historical exact-N experiments. New Mimics requests write
        # min/max explicitly and use a variable interaction distribution.
        config["prompts"]["min_interaction_steps"] = legacy_horizon
        config["prompts"]["training_goal"] = "legacy"
        config["prompts"]["interaction_step_weights"] = None
        if "correction_policy" not in prompt_override:
            config["prompts"]["correction_policy"] = "clopa_paired"
        if "initial_mask_probability" not in prompt_override:
            config["prompts"]["initial_mask_probability"] = 0.0
        if "validate_initial_masks" not in prompt_override:
            config["prompts"]["validate_initial_masks"] = False
        checkpoints = [
            value
            for value in config["prompts"]["validation_interaction_steps"]
            if int(value) <= legacy_horizon
        ]
        if legacy_horizon not in checkpoints:
            checkpoints.append(legacy_horizon)
        config["prompts"]["validation_interaction_steps"] = sorted(
            set(int(value) for value in checkpoints)
        )
    elif (
        (
            "min_interaction_steps" in prompt_override
            or "max_interaction_steps" in prompt_override
        )
        and "interaction_step_weights" not in prompt_override
    ):
        # Custom ranges written by earlier releases used the short/long
        # sampler and have no explicit categorical distribution.
        config["prompts"]["interaction_step_weights"] = None
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
