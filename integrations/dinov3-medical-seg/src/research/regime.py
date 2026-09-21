"""Tier-0 sampling+loss regime cells derived from train-only fingerprints."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from .fingerprint import derive_policy

# Four regime cells: {full, patch} sampling x {dice_ce, dice_focal} loss.
REGIME_CELL_IDS = ("full_dice_ce", "full_dice_focal", "patch_dice_ce", "patch_dice_focal")

# Foreground focal alpha for the sparse CT targets in this study, matching the
# adrenal_dice_focal candidate in the study plan.
_DICE_FOCAL_LOSS = {
    "type": "dice_focal",
    "dice_weight": 0.7,
    "focal_weight": 0.3,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
}
_DICE_CE_LOSS = {"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5}


def regime_cells(fingerprint: Mapping) -> dict:
    """Return the four regime override dicts for one task's support fingerprint.

    Patch geometry comes from ``derive_policy`` (train-support-only), but the
    patch cells enable patch sampling unconditionally: for a slice-wise
    pseudo-3D model on whole-body CT, foreground-dense training is needed
    regardless of the ``small_target`` gate.
    """
    policy = derive_policy(fingerprint)
    patch_size = [int(v) for v in policy["patch"]["size_zyx"]]
    foreground_probability = float(policy["patch"]["foreground_probability"])

    def _full_data():
        return {"patch": {"enabled": False}}

    def _patch_data():
        return {
            "patch": {
                "enabled": True,
                "size_zyx": patch_size,
                "foreground_probability": foreground_probability,
                "inference_sliding_window": True,
                "inference_overlap": 0.5,
            }
        }

    return {
        "full_dice_ce": {"data": _full_data(), "loss": deepcopy(_DICE_CE_LOSS)},
        "full_dice_focal": {"data": _full_data(), "loss": deepcopy(_DICE_FOCAL_LOSS)},
        "patch_dice_ce": {"data": _patch_data(), "loss": deepcopy(_DICE_CE_LOSS)},
        "patch_dice_focal": {"data": _patch_data(), "loss": deepcopy(_DICE_FOCAL_LOSS)},
    }


def apply_regime(base_config: Mapping, regime_override: Mapping) -> dict:
    """Merge a regime override into a base config.

    The loss block is REPLACED wholesale (not deep-merged) so switching
    dice_ce -> dice_focal never leaves stale keys like ce_weight. The data
    block is deep-merged so the regime only touches ``data.patch`` and leaves
    intensity/img_size/modality intact.
    """
    result = deepcopy(dict(base_config))
    regime = deepcopy(dict(regime_override))
    data = dict(result.get("data", {}))
    regime_data = dict(regime.get("data", {}))
    if "patch" in regime_data:
        merged_patch = dict(data.get("patch", {}))
        merged_patch.update(regime_data["patch"])
        data["patch"] = merged_patch
    for key, value in regime_data.items():
        if key != "patch":
            data[key] = value
    result["data"] = data
    if "loss" in regime:
        result["loss"] = regime["loss"]
    return result
