"""Validated, composable DINOv3 training policies shared by UI and pipeline."""

from __future__ import annotations

import copy


STRATEGY_SCHEMA_VERSION = "mimics_dinov3_strategy.v2"

DEFAULT_OPTIONS = {
    "sampling_mode": "adaptive",
    "patch_size_mode": "fingerprint",
    "patch_size_zyx": "64,192,192",
    "patches_per_case": 2,
    "patch_focus": "foreground",
    "channel_policy": "repeat",
    "slice_axis": "axial",
    "neighbor_distance_mm": 3.0,
    "loss_type": "auto",
    "keep_largest_component": False,
}

PRESETS = {
    "adaptive": {
        "label": "Adaptive from training data",
        "summary": "Uses the training-data fingerprint to choose full-volume or patch sampling. Other controls remain editable.",
        "options": {},
    },
    "full_volume": {
        "label": "Full-volume baseline",
        "summary": "For targets that occupy enough of the field of view. Uses Dice-Focal and whole-volume inference.",
        "options": {"sampling_mode": "full", "loss_type": "auto"},
    },
    "patch_focused": {
        "label": "Patch-focused sparse target",
        "summary": "Raises foreground exposure for small targets with fingerprint-sized patches and sliding-window inference.",
        "options": {"sampling_mode": "patch", "patch_focus": "foreground", "loss_type": "auto"},
    },
}

# Backward-compatible name for callers that imported STRATEGIES.
STRATEGIES = PRESETS


def strategy_ids():
    return list(PRESETS.keys())


def strategy_label(strategy_id):
    return PRESETS.get(strategy_id, PRESETS["adaptive"])["label"]


def strategy_summary(strategy_id):
    return PRESETS.get(strategy_id, PRESETS["adaptive"])["summary"]


def strategy_defaults(strategy_id):
    values = copy.deepcopy(DEFAULT_OPTIONS)
    values.update(copy.deepcopy(PRESETS.get(strategy_id, PRESETS["adaptive"])["options"]))
    return values


def suggested_strategy(_organ=None):
    return "adaptive"


def _parse_patch_size(value):
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 3:
        raise ValueError("Patch size must be formatted as z,y,x")
    result = [int(value) for value in parts]
    if any(value <= 0 for value in result):
        raise ValueError("Patch dimensions must be positive")
    return result


def _patch_sampling(name):
    if name == "boundary":
        return {"interior": 0.20, "boundary": 0.50, "near_negative": 0.20, "random": 0.10,
                "boundary_width": 2, "near_negative_width": 8}
    if name == "negative_balanced":
        return {"interior": 0.20, "boundary": 0.20, "near_negative": 0.35, "random": 0.25,
                "boundary_width": 2, "near_negative_width": 8}
    return None


def normalize_strategy_options(options, fingerprint=None, policy=None, preset="adaptive"):
    if preset not in PRESETS:
        raise ValueError("Unknown DINOv3 preset: {}".format(preset))
    values = strategy_defaults(preset)
    values.update(copy.deepcopy(options or {}))
    if values.get("sampling_mode") == "adaptive" and policy is not None:
        values["sampling_mode"] = "patch" if bool(policy.get("patch", {}).get("enabled")) else "full"

    enums = {
        "sampling_mode": ("adaptive", "full", "patch"),
        "patch_size_mode": ("fingerprint", "custom"),
        "patch_focus": ("foreground", "boundary", "negative_balanced"),
        "channel_policy": ("repeat", "2_5d"),
        "slice_axis": ("axial", "coronal", "sagittal"),
        "loss_type": ("auto", "dice_ce", "dice_focal"),
    }
    for key, allowed in enums.items():
        values[key] = str(values.get(key, ""))
        if values[key] not in allowed:
            raise ValueError("Unsupported {}: {}".format(key, values[key]))
    for key, minimum, maximum in (
        ("neighbor_distance_mm", 0.1, 20.0),
    ):
        values[key] = float(values[key])
        if values[key] < minimum or values[key] > maximum:
            raise ValueError("{} must be between {} and {}".format(key, minimum, maximum))
    values["patches_per_case"] = int(values["patches_per_case"])
    if values["patches_per_case"] < 1:
        raise ValueError("Patch count must be positive")
    values["keep_largest_component"] = bool(values.get("keep_largest_component"))
    if values["sampling_mode"] == "patch" and values["patch_size_mode"] == "custom":
        _parse_patch_size(values["patch_size_zyx"])
    return values


def compile_strategy(strategy_id, fingerprint=None, policy=None, user_options=None):
    values = normalize_strategy_options(user_options, fingerprint, policy, strategy_id)
    if values["sampling_mode"] == "adaptive":
        raise ValueError("Adaptive sampling requires a training-data fingerprint policy")
    patch = {"enabled": values["sampling_mode"] == "patch"}
    if patch["enabled"]:
        if values["patch_size_mode"] == "fingerprint":
            patch["size_zyx"] = list((policy or {}).get("patch", {}).get("size_zyx") or _parse_patch_size(values["patch_size_zyx"]))
        else:
            patch["size_zyx"] = _parse_patch_size(values["patch_size_zyx"])
        patch.update({
            "patches_per_case_per_epoch": values["patches_per_case"],
            "foreground_probability": 0.75,
            "inference_sliding_window": True,
            "inference_overlap": 0.5,
        })
        sampling = _patch_sampling(values["patch_focus"])
        if sampling:
            patch["sampling"] = sampling

    if values["loss_type"] == "dice_ce":
        loss = {"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5}
    else:
        loss = {"type": "dice_focal", "dice_weight": 0.7, "focal_weight": 0.3,
                "focal_alpha": 0.75, "focal_gamma": 2.0}
    model = {"slice_axis": values["slice_axis"], "slice_batch_size": 1,
             "channel_policy": values["channel_policy"]}
    if values["channel_policy"] == "2_5d":
        model["neighbor_distance_mm"] = values["neighbor_distance_mm"]
    return {
        "strategy": {"schema_version": STRATEGY_SCHEMA_VERSION, "preset": strategy_id, "options": values},
        "model": model,
        "data": {"patch": patch, "roi": {"enabled": False}},
        "loss": loss,
        "inference": {"threshold": 0.5,
                      "keep_largest_component": values["keep_largest_component"]},
    }
