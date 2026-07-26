"""Serializable DINOv3 medical input preprocessing contracts."""

from __future__ import annotations

from copy import deepcopy


VOLUME_CONTRACT_VERSION = "dinov3_volume_input.v1"
CACHED_SLICE_CONTRACT_VERSION = "dinov3_cached_slice_input.v1"


def input_contract_for_config(config: dict) -> dict:
    data = dict(config.get("data") or {})
    model = dict(config.get("model") or {})
    training = dict(config.get("training") or {})
    pipeline = str(training.get("pipeline") or "volume")
    if pipeline == "cached_slices":
        return {
            "schema_version": CACHED_SLICE_CONTRACT_VERSION,
            "source_intensity_space": "source_physical_values",
            "spatial_orientation": "native_source_grid",
            "target_spacing": deepcopy(data.get("target_spacing")),
            "normalization": str(
                data.get("slice_normalization") or "timeslice_casewise"
            ),
            "image_size": list(data.get("img_size") or [256, 256]),
            "encoder_input_normalization": str(
                model.get("input_normalization") or "none"
            ),
        }
    return {
        "schema_version": VOLUME_CONTRACT_VERSION,
        "source_intensity_space": "source_physical_values",
        "spatial_orientation": "canonical_ras",
        "target_spacing": deepcopy(data.get("target_spacing")),
        "modality": str(data.get("modality") or "other").lower(),
        "intensity": deepcopy(data.get("intensity") or {}),
        "channel_policy": str(model.get("channel_policy") or "repeat"),
        "slice_axis": str(model.get("slice_axis") or "axial"),
        "image_size": list(data.get("img_size") or [224, 224]),
        "encoder_input_normalization": str(
            model.get("input_normalization") or "none"
        ),
        "image_mean": list(model.get("image_mean") or []),
        "image_std": list(model.get("image_std") or []),
    }


def validate_input_contract(config: dict, *, required: bool = False) -> dict:
    training = dict(config.get("training") or {})
    data = dict(config.get("data") or {})
    if (
        str(training.get("pipeline") or "volume") == "cached_slices"
        and data.get("target_spacing") not in (None, [], "")
    ):
        raise RuntimeError(
            "Frozen Feature 2D models preserve the source NIfTI grid and cannot "
            "use data.target_spacing. This model configuration would apply "
            "different geometry during training and inference."
        )
    stored = dict((config.get("runtime") or {}).get("input_contract") or {})
    if not stored:
        if required:
            raise RuntimeError(
                "The model configuration has no DINOv3 input preprocessing contract."
            )
        return {}
    expected = input_contract_for_config(config)
    if stored != expected:
        raise RuntimeError(
            "The DINOv3 input preprocessing configuration differs from the "
            "contract saved for this model. Refusing distribution-shifted inference."
        )
    return expected
