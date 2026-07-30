"""Converged DINOv3 architecture and resource planning.

The public training UI chooses an architecture family, not an implementation
class.  This module resolves that intent after the selected training cases have
been fingerprinted.  Legacy decoder names remain loadable for existing models
and retry records, but are not used for new automatic plans.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


ARCHITECTURE_SCHEMA_VERSION = "mimics_dinov3_architecture.v1"

TRAINING_DIMENSIONS = ("auto", "2d", "3d")
QUALITY_MODES = ("standard", "high_detail")
BACKBONE_TUNING = ("frozen", "lora")

PUBLIC_DECODERS = {
    ("2d", "standard"): "scale_aware2d",
    ("2d", "high_detail"): "scale_aware2d",
    ("3d", "standard"): "context3d_lite",
    ("3d", "high_detail"): "context3d_hybrid",
}

LEGACY_2D_DECODERS = {
    "feature_unet2d",
    "conv2d",
    "conv2d_unet",
    "conv2d_deeplab",
    "conv2d_2_5d",
}
LEGACY_3D_DECODERS = {
    "linear3d",
    "mlp_probe",
    "segformer3d",
    "token_pyramid3d",
    "dpt3d",
    "context3d_multiscale",
}
LEGACY_DECODERS = LEGACY_2D_DECODERS | LEGACY_3D_DECODERS
SUPPORTED_DECODERS = set(PUBLIC_DECODERS.values()) | LEGACY_DECODERS


def intermediate_layer_indices(num_hidden_layers: int) -> list[int]:
    """Choose four semantic depths for a DINOv3 ViT of any supported scale.

    The formula reproduces the layer choices used by SegDINO for 12-layer
    ViT-S/B and 24-layer ViT-L models, then extends the same relative depths to
    deeper backbones.
    """
    depth = int(num_hidden_layers)
    if depth < 4:
        raise ValueError("A multi-level DINOv3 decoder requires at least four layers.")
    indices = [
        depth // 5,
        depth // 2 - 1,
        (3 * depth) // 4 - 1,
        depth - 1,
    ]
    if indices != sorted(set(indices)) or indices[0] < 0:
        indices = [
            int(round(index * (depth - 1) / 3.0))
            for index in range(4)
        ]
    if indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError(
            "Could not derive four unique DINOv3 feature layers from depth {}.".format(
                depth
            )
        )
    return indices


def inspect_dinov3_vit_weights(model_root) -> dict:
    """Validate one local Hugging Face DINOv3 ViT backbone directory."""
    root = Path(model_root).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "DINOv3 weights must be a folder containing config.json: {}".format(
                root
            )
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle) or {}
    except Exception as exc:
        raise ValueError(
            "Could not read DINOv3 config.json at {}: {}".format(
                config_path, exc
            )
        ) from exc
    model_type = str(config.get("model_type") or "").strip().lower()
    architectures = [
        str(value)
        for value in (config.get("architectures") or [])
    ]
    if model_type != "dinov3_vit" and not any(
        "dinov3vit" in value.replace("_", "").lower()
        for value in architectures
    ):
        raise ValueError(
            "Unsupported pretrained architecture '{}'. New Mimics 2D/3D "
            "plans require Hugging Face DINOv3 ViT weights, not an arbitrary "
            "vision backbone.".format(model_type or "unknown")
        )
    depth = int(config.get("num_hidden_layers") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    patch_size = int(config.get("patch_size") or 0)
    if depth < 4 or hidden_size < 1 or patch_size < 1:
        raise ValueError(
            "DINOv3 config.json must define positive hidden_size and patch_size "
            "and at least four transformer layers."
        )
    processor_path = root / "preprocessor_config.json"
    single_weights = root / "model.safetensors"
    sharded_index = root / "model.safetensors.index.json"
    if not single_weights.is_file() and not sharded_index.is_file():
        raise ValueError(
            "No model.safetensors or model.safetensors.index.json was found "
            "under {}.".format(root)
        )
    if not processor_path.is_file():
        raise ValueError(
            "No preprocessor_config.json was found under {}.".format(root)
        )
    try:
        with processor_path.open("r", encoding="utf-8") as handle:
            processor = json.load(handle) or {}
    except Exception as exc:
        raise ValueError(
            "Could not read DINOv3 preprocessor_config.json at {}: {}".format(
                processor_path, exc
            )
        ) from exc
    image_mean = processor.get("image_mean") or [0.485, 0.456, 0.406]
    image_std = processor.get("image_std") or [0.229, 0.224, 0.225]
    if (
        len(image_mean) != 3
        or len(image_std) != 3
        or any(float(value) <= 0.0 for value in image_std)
    ):
        raise ValueError(
            "DINOv3 preprocessor_config.json must define three image_mean and "
            "three positive image_std values."
        )
    weight_files = [single_weights]
    if sharded_index.is_file() and not single_weights.is_file():
        try:
            with sharded_index.open("r", encoding="utf-8") as handle:
                weight_index = json.load(handle) or {}
            shard_names = sorted(
                set((weight_index.get("weight_map") or {}).values())
            )
        except Exception as exc:
            raise ValueError(
                "Could not read sharded DINOv3 weight index {}: {}".format(
                    sharded_index, exc
                )
            ) from exc
        if not shard_names:
            raise ValueError(
                "The sharded DINOv3 weight index contains no weight files."
            )
        weight_files = [root / str(name) for name in shard_names]
        missing = [str(path) for path in weight_files if not path.is_file()]
        if missing:
            raise ValueError(
                "The sharded DINOv3 weight set is incomplete; missing: {}".format(
                    ", ".join(missing[:8])
                )
            )
    return {
        "model_root": str(root),
        "config_path": str(config_path),
        "processor_path": str(processor_path),
        "model_type": model_type or "dinov3_vit",
        "architectures": architectures,
        "num_hidden_layers": depth,
        "hidden_size": hidden_size,
        "patch_size": patch_size,
        "image_size": config.get("image_size"),
        "image_mean": [float(value) for value in image_mean],
        "image_std": [float(value) for value in image_std],
        "weight_files": [str(path) for path in weight_files],
        "out_indices": intermediate_layer_indices(depth),
        "weights_format": (
            "safetensors"
            if single_weights.is_file()
            else "sharded_safetensors"
        ),
    }


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def dimension_for_decoder(decoder: str) -> str:
    name = str(decoder or "").strip().lower()
    if name in LEGACY_2D_DECODERS or name == "scale_aware2d":
        return "2d"
    if name in LEGACY_3D_DECODERS or name.startswith("context3d_"):
        return "3d"
    return "auto"


def normalize_architecture_options(options: Mapping | None) -> dict:
    values = dict(options or {})
    decoder = str(values.get("decoder") or "").strip().lower()
    dimension_was_explicit = bool(str(values.get("training_dimension") or "").strip())
    dimension = str(values.get("training_dimension") or "").strip().lower()
    if not dimension:
        dimension = dimension_for_decoder(decoder)
    if dimension not in TRAINING_DIMENSIONS:
        raise ValueError(
            "Training dimension must be Auto, 2D, or 3D."
        )
    quality = str(values.get("quality_mode") or "standard").strip().lower()
    if quality not in QUALITY_MODES:
        raise ValueError("Quality mode must be Standard or High detail.")
    tuning = str(
        values.get("finetune_method") or values.get("backbone_tuning") or "frozen"
    ).strip().lower()
    if tuning in {"decoder_only", "decode_only", "decoder-only", "decode-only"}:
        tuning = "frozen"
    # Old jobs may contain adapter/full. They remain executable through their
    # recorded decoder, but new family-based configurations only expose the two
    # bounded modes below.
    if dimension_was_explicit and tuning not in BACKBONE_TUNING:
        raise ValueError("Backbone tuning must be Frozen or LoRA.")
    if tuning not in BACKBONE_TUNING and decoder not in LEGACY_DECODERS:
        raise ValueError("Backbone tuning must be Frozen or LoRA.")
    return {
        "training_dimension": dimension,
        "quality_mode": quality,
        "finetune_method": tuning,
        "legacy_decoder": decoder if decoder in LEGACY_DECODERS else "",
        "dimension_was_explicit": dimension_was_explicit,
    }


def _summary_value(fingerprint: Mapping, key: str, default):
    return (fingerprint.get("summary") or {}).get(key, default)


def resolve_architecture(
    options: Mapping,
    fingerprint: Mapping,
    *,
    gpu_memory_gb: float = 0.0,
) -> dict:
    normalized = normalize_architecture_options(options)
    requested = normalized["training_dimension"]
    requested_quality = normalized["quality_mode"]
    quality = requested_quality
    legacy_decoder = normalized["legacy_decoder"]

    # Retry and imported configurations without the new family field preserve
    # the old decoder exactly. New UI requests always resolve through the public
    # architecture families.
    if legacy_decoder and not normalized["dimension_was_explicit"]:
        resolved_dimension = dimension_for_decoder(legacy_decoder)
        decoder = legacy_decoder
        reason = "legacy decoder preserved for model and retry compatibility"
        legacy = True
    else:
        spacing = [
            float(value)
            for value in _summary_value(
                fingerprint, "median_spacing_zyx", [1.0, 1.0, 1.0]
            )
        ]
        finest = max(1e-6, min(spacing))
        anisotropy = float(max(spacing) / finest)
        if requested == "auto":
            resolved_dimension = "2d" if anisotropy >= 2.5 else "3d"
            reason = (
                "strong through-plane anisotropy favors slice-wise training"
                if resolved_dimension == "2d"
                else "near-isotropic spacing supports volumetric context"
            )
        else:
            resolved_dimension = requested
            reason = "user selected the training dimension"
        if resolved_dimension == "2d":
            quality = "standard"
        decoder = PUBLIC_DECODERS[(resolved_dimension, quality)]
        legacy = False

    plan = {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "requested_dimension": requested,
        "resolved_dimension": resolved_dimension,
        "requested_quality_mode": requested_quality,
        "quality_mode": quality,
        "decoder": decoder,
        "backbone_tuning": normalized["finetune_method"],
        "gpu_memory_gb": float(max(0.0, gpu_memory_gb)),
        "legacy_compatibility": legacy,
        "reason": reason,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def seal_architecture_plan(plan: Mapping) -> dict:
    """Hash the final plan after resource-dependent fields are resolved."""
    result = dict(plan)
    result.pop("plan_sha256", None)
    result["plan_sha256"] = _digest(result)
    return result


def recommended_batch_size(
    architecture_plan: Mapping,
    policy: Mapping,
    *,
    gpu_memory_gb: float,
) -> int:
    """Return a conservative real batch size for the selected hardware.

    The recommendation is deliberately conservative. Users may request a
    larger batch, including on remote GPUs; the runtime performs the actual
    allocation and reports CUDA OOM without silently changing the request.
    """
    memory = float(max(0.0, gpu_memory_gb))
    if str(architecture_plan.get("backbone_tuning") or "frozen") == "lora":
        # LoRA parameters are small, but gradients through the transformer
        # retain substantially more activations than a frozen backbone.
        memory *= 0.65
    decoder = str(architecture_plan.get("decoder") or "")
    patch_enabled = bool((policy.get("patch") or {}).get("enabled"))
    input_size = [
        int(value)
        for value in (policy.get("input_size") or [224, 224])
    ]
    high_resolution = max(input_size) >= 320
    if bool(architecture_plan.get("sub_volume")):
        return 1
    if decoder == "feature_unet2d":
        if memory >= 48:
            return 16
        if memory >= 24:
            return 8
        return 4
    if decoder == "scale_aware2d":
        if high_resolution:
            return 2 if patch_enabled and memory >= 48 else 1
        if patch_enabled and memory >= 48:
            return 4
        if patch_enabled and memory >= 20:
            return 2
        if memory >= 48:
            return 2
        return 1
    if decoder == "context3d_lite":
        if high_resolution:
            return 2 if patch_enabled and memory >= 48 else 1
        if patch_enabled and memory >= 48:
            return 4
        if patch_enabled and memory >= 24:
            return 2
        if not patch_enabled and memory >= 48:
            return 2
        return 1
    if decoder in ("context3d_hybrid", "context3d_multiscale"):
        if patch_enabled and memory >= 48 and not high_resolution:
            return 2
        return 1
    return 1
