#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External DINOv3 few-shot training setup UI.

This tool is intentionally outside the foreground Mimics Python process.  The
Mimics entry starts it with Popen and returns immediately.  The UI writes a
setup/job status JSON file so Mimics can keep monitoring without blocking.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue

# Ensure the project root and tools/ directory are importable.
# The embeddable Python (nninteractive_env) uses a ._pth file that
# ignores PYTHONPATH, so we must inject paths directly into sys.path.
_here = os.path.dirname(os.path.abspath(__file__))
_tools_dir = _here
_project_root = os.path.dirname(_tools_dir)
for _candidate in (_tools_dir, _project_root):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    from fewshot_strategies import DEFAULT_OPTIONS, STRATEGIES, normalize_strategy_options, strategy_defaults, strategy_ids, strategy_label, strategy_summary, suggested_strategy
except ImportError:
    from tools.fewshot_strategies import DEFAULT_OPTIONS, STRATEGIES, normalize_strategy_options, strategy_defaults, strategy_ids, strategy_label, strategy_summary, suggested_strategy
try:
    from fewshot_architecture import (
        LEGACY_DECODERS,
        PUBLIC_DECODERS,
        SUPPORTED_DECODERS,
        inspect_dinov3_vit_weights,
        normalize_architecture_options,
    )
except ImportError:
    from tools.fewshot_architecture import (
        LEGACY_DECODERS,
        PUBLIC_DECODERS,
        SUPPORTED_DECODERS,
        inspect_dinov3_vit_weights,
        normalize_architecture_options,
    )
from ui_theme import (
    choose_existing_directory_async,
    configure_application,
    stylesheet as shared_stylesheet,
)
from ui_preferences import load_preferences, save_preferences
try:
    from remote_compute_ui import RemoteComputeSelector
except Exception:
    # Remote compute is optional. A missing optional module must never prevent
    # the existing local training window from opening.
    RemoteComputeSelector = None
from training_data_ui import (
    LABEL_SOURCE_CHOICES,
    label_source_hint,
    normalized_source_mode,
)
TITLE = "DINOv3 Few-Shot Training"
STRATEGY_DATA_KEYS = set(DEFAULT_OPTIONS.keys())
DECODER_CHOICES = tuple(sorted(set(PUBLIC_DECODERS.values())))
FEATURE_ENCODER_BACKEND_CHOICES = (
    ("Compatibility ONNX", "onnx"),
    ("Native PyTorch weights", "pytorch"),
)
TRAINING_DIMENSION_ITEMS = (
    ("Automatic from training data", "auto"),
    ("2D slice training", "2d"),
    ("3D context training", "3d"),
)
QUALITY_MODE_ITEMS = (
    ("Standard", "standard"),
    ("High detail", "high_detail"),
)
MODALITY_ITEMS = (
    ("Automatic (metadata, otherwise generic)", "auto"),
    ("CT", "ct"),
    ("MRI", "mr"),
    ("Other / unknown", "other"),
)

MODEL_SCALE_BY_DIRECTORY = {
    "dinov3-vits16": "vits16",
    "dinov3-vitb16": "vitb16",
    "dinov3-vitl16": "vitl16",
    "dinov3-vith16plus": "vith16plus",
}
MODEL_FAMILY_LABELS = {
    "vits16": "DINOv3 ViT-S/16",
    "vitb16": "DINOv3 ViT-B/16",
    "vitl16": "DINOv3 ViT-L/16",
    "vith16plus": "DINOv3 ViT-H+/16",
}


def model_scale_for_path(model_path, fallback="custom"):
    """Return the stable scale id for a model installed in the fixed repository."""
    name = Path(str(model_path or "")).expanduser().name.lower()
    return MODEL_SCALE_BY_DIRECTORY.get(name, str(fallback or "custom").lower())


def discover_pretrained_models(dinov3_root, current_model_path="", current_scale=""):
    """List validated Hugging Face DINOv3 ViT weights under ``models/``.

    The setup UI intentionally does not browse arbitrary directories. New
    weights become selectable by installing one complete model folder under the
    project's fixed model repository. A valid external path is retained only
    when editing an older saved job, so retries do not silently switch weights.
    """
    root = Path(str(dinov3_root or "")).expanduser()
    models_root = root / "models"
    records = []
    seen = set()

    def add_record(path, *, legacy_external=False):
        try:
            info = inspect_dinov3_vit_weights(path)
        except ValueError:
            return
        resolved = str(Path(info["model_root"]).resolve())
        key = os.path.normcase(resolved)
        if key in seen:
            return
        seen.add(key)
        scale = model_scale_for_path(resolved)
        family = MODEL_FAMILY_LABELS.get(scale, Path(resolved).name)
        suffix = " · existing external path" if legacy_external else ""
        records.append({
            "key": resolved,
            "path": resolved,
            "scale": scale,
            "label": "{} · {}d · {} layers{}".format(
                family,
                int(info["hidden_size"]),
                int(info["num_hidden_layers"]),
                suffix,
            ),
            "info": info,
            "legacy_external": bool(legacy_external),
        })

    if models_root.is_dir():
        for path in sorted(
            (item for item in models_root.iterdir() if item.is_dir()),
            key=lambda item: item.name.lower(),
        ):
            add_record(path)

    raw_current = str(current_model_path or "").strip()
    if raw_current:
        current = Path(os.path.expandvars(os.path.expanduser(raw_current)))
        if not current.is_absolute():
            current = root / current
        try:
            inside_repository = (
                os.path.commonpath(
                    [str(current.resolve()), str(models_root.resolve())]
                )
                == str(models_root.resolve())
            )
        except (OSError, ValueError):
            inside_repository = False
        add_record(current, legacy_external=not inside_repository)

    preferred_scale = str(current_scale or "").strip().lower()
    records.sort(
        key=lambda item: (
            0 if item["scale"] == preferred_scale else 1,
            item["label"].lower(),
        )
    )
    return records


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def write_json_best_effort(path, payload):
    try:
        write_json_atomic(path, payload, retries=8, max_sleep=0.15)
        return True
    except Exception:
        return False


def safe_slug(value):
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", ".", " "):
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unnamed"


def split_csv(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def configured_mask_names(config, organ):
    names = [str(organ or "").strip()]
    aliases = config.get("organ_mask_aliases") or {}
    if isinstance(aliases, dict):
        organ_key = safe_slug(organ)
        for key, values in aliases.items():
            if safe_slug(key) != organ_key:
                continue
            names.extend(split_csv(values))
    result = []
    seen = set()
    for name in names:
        key = safe_slug(name)
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def hidden_process_kwargs():
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def default_training_options(config, profile_name=None):
    profiles = config.get("training_profiles") or {}
    default_profile = profile_name or config.get("default_training_profile") or config.get("default_profile")
    values = {}
    if default_profile and isinstance(profiles, dict):
        values.update(profiles.get(default_profile, {}) or {})
    profile_decoder = str(values.get("decoder") or "").strip().lower()
    if profile_decoder in SUPPORTED_DECODERS and "training_dimension" not in values:
        values["preserve_legacy_decoder"] = True
    values.setdefault("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))
    values.setdefault("strategy", config.get("default_strategy", "adaptive"))
    values.setdefault("epochs", config.get("default_epochs", 20))
    values.setdefault("batch_size", config.get("default_batch_size", 0))
    values.setdefault("grad_accumulation", config.get("default_grad_accumulation", 1))
    values.setdefault("lr", config.get("default_lr", 0.001))
    values.setdefault("weight_decay", config.get("default_weight_decay", 0.01))
    values.setdefault("lr_scheduler", config.get("default_lr_scheduler", "cosine"))
    values.setdefault("warmup_epochs", config.get("default_warmup_epochs", 3))
    values.setdefault("validation_interval", config.get("default_validation_interval", 2))
    values.setdefault("img_size", config.get("default_img_size", "256,256"))
    values.setdefault("modality", config.get("default_modality", "auto"))
    values.setdefault("min_samples", config.get("default_min_samples", 1))
    values.setdefault("max_samples", config.get("default_max_samples", 0))
    values.setdefault("sample_mode", config.get("default_sample_mode", "all"))
    values.setdefault("val_fraction", config.get("default_val_fraction", 0.2))
    values.setdefault("min_val_samples", config.get("default_min_val_samples", 1))
    values.setdefault("finetune_method", config.get("default_finetune_method", "lora"))
    values.setdefault("training_dimension", config.get("default_training_dimension", "auto"))
    values.setdefault("quality_mode", config.get("default_quality_mode", "standard"))
    values.setdefault("decoder", config.get("default_decoder", "auto"))
    values.setdefault("model_scale", config.get("default_model_scale", "vitb16"))
    values.setdefault("model_path", config.get("default_model_path", ""))
    values.setdefault("model_sha256", config.get("default_model_sha256", ""))
    values.setdefault(
        "safetensors_sha256",
        config.get("default_safetensors_sha256", ""),
    )
    values.setdefault("lora_rank", config.get("default_lora_rank", 8))
    values.setdefault("lora_alpha", config.get("default_lora_alpha", 16))
    values.setdefault("adapter_bottleneck", config.get("default_adapter_bottleneck", 64))
    values.setdefault("mixed_precision", config.get("default_mixed_precision", False))
    values.setdefault("sub_volume", config.get("default_sub_volume", False))
    values.setdefault("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))
    values.setdefault("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2))
    values.setdefault("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))
    values.setdefault("export_labels_before_training", config.get("default_export_labels_before_training", True))
    values.setdefault(
        "label_source",
        config.get("default_label_source")
        or (
            "mcs_refresh"
            if _bool(values.get("export_labels_before_training", True))
            else "source_dataset"
        ),
    )
    values.setdefault("label_root", "")
    values.setdefault(
        "encoder_backend",
        config.get("default_feature_encoder_backend", "onnx"),
    )
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault("gpu_memory_gb", config.get("default_gpu_memory_gb", 0.0))
    values.setdefault(
        "background_mimics_lock_timeout_seconds",
        config.get("background_mimics_lock_timeout_seconds", 1800),
    )
    return values


def training_options_for_organ(config, organ, profile_name=None):
    """Resolve one coherent initial strategy instead of mixing it with a legacy profile."""
    values = default_training_options(config, profile_name)
    if not config.get("default_strategy"):
        values["strategy"] = suggested_strategy(organ)
    values.update(strategy_defaults(values.get("strategy")))
    if (
        str(values.get("decoder")) == "feature_unet2d"
        and _bool(values.get("preserve_legacy_decoder", False))
    ):
        values["strategy"] = "full_volume"
        values.update(strategy_defaults("full_volume"))
    values["mask_names"] = ",".join(configured_mask_names(config, organ))
    return values


def _float(value, name):
    try:
        return float(value)
    except Exception:
        raise ValueError("{0} must be a number.".format(name))


def _int(value, name, minimum=None):
    try:
        result = int(value)
    except Exception:
        raise ValueError("{0} must be an integer.".format(name))
    if minimum is not None and result < minimum:
        raise ValueError("{0} must be at least {1}.".format(name, minimum))
    return result


def validate_options(options):
    normalized = dict(options)
    if str(normalized.get("strategy", "adaptive")) not in strategy_ids():
        raise ValueError("Unknown DINOv3 training strategy.")
    normalized["epochs"] = _int(normalized.get("epochs", 20), "Epochs", 1)
    normalized["batch_size"] = _int(
        normalized.get("batch_size", 0),
        "Batch size",
        0,
    )
    normalized["gpu_memory_gb"] = _float(
        normalized.get("gpu_memory_gb", 0.0),
        "GPU memory budget",
    )
    if normalized["gpu_memory_gb"] < 0.0:
        raise ValueError("GPU memory budget cannot be negative.")
    modality = str(normalized.get("modality") or "auto").strip().lower()
    if modality == "mri":
        modality = "mr"
    if modality not in ("auto", "ct", "mr", "other"):
        raise ValueError("Image modality must be Automatic, CT, MRI, or Other.")
    normalized["modality"] = modality
    decoder = str(normalized.get("decoder", "auto")).strip().lower()
    preserve_legacy = _bool(
        normalized.get("preserve_legacy_decoder", False)
    ) or (
        decoder in LEGACY_DECODERS
        and not str(normalized.get("training_dimension") or "").strip()
    )
    normalized["preserve_legacy_decoder"] = preserve_legacy
    architecture_input = dict(normalized)
    if preserve_legacy:
        architecture_input.pop("training_dimension", None)
    architecture = normalize_architecture_options(architecture_input)
    normalized.update({
        "training_dimension": architecture["training_dimension"],
        "quality_mode": architecture["quality_mode"],
        "finetune_method": architecture["finetune_method"],
    })
    cached_slices = decoder == "feature_unet2d" and preserve_legacy
    normalized["grad_accumulation"] = _int(normalized.get("grad_accumulation", 1), "Grad accumulation", 1)
    if (
        not cached_slices
        and
        _bool(normalized.get("sub_volume", False))
        and normalized["batch_size"] > 1
    ):
        raise ValueError(
            "Sub-volume training requires Batch size Auto or 1. Use gradient "
            "accumulation for a larger effective batch."
        )
    normalized["min_samples"] = _int(normalized.get("min_samples", 1), "Min train samples", 1)
    normalized["max_samples"] = _int(normalized.get("max_samples", 0), "Max samples", 0)
    normalized["min_val_samples"] = _int(normalized.get("min_val_samples", 1), "Min validation samples", 0)
    normalized["lora_rank"] = _int(normalized.get("lora_rank", 8), "LoRA rank", 1)
    normalized["lora_alpha"] = _int(normalized.get("lora_alpha", 16), "LoRA alpha", 1)
    normalized["adapter_bottleneck"] = _int(normalized.get("adapter_bottleneck", 64), "Adapter bottleneck", 1)
    normalized["keep_last_checkpoints"] = _int(normalized.get("keep_last_checkpoints", 2), "Keep last checkpoints", 0)
    normalized["lr"] = _float(normalized.get("lr", 0.001), "Learning rate")
    normalized["weight_decay"] = _float(normalized.get("weight_decay", 0.01), "Weight decay")
    normalized["lr_scheduler"] = str(normalized.get("lr_scheduler", "cosine")).strip().lower()
    if normalized["lr_scheduler"] not in ("constant", "constant_warmup", "cosine"):
        raise ValueError("Learning-rate schedule must be constant, constant with warmup, or cosine.")
    normalized["warmup_epochs"] = _int(normalized.get("warmup_epochs", 3), "Warmup epochs", 0)
    normalized["validation_interval"] = _int(
        normalized.get("validation_interval", 2),
        "Validation interval",
        1,
    )
    if decoder not in SUPPORTED_DECODERS and decoder != "auto":
        raise ValueError("Unsupported decoder: {0}".format(normalized.get("decoder")))
    normalized["val_fraction"] = _float(normalized.get("val_fraction", 0.2), "Validation fraction")
    normalized["mixed_precision"] = _bool(normalized.get("mixed_precision", False))
    normalized["sub_volume"] = _bool(normalized.get("sub_volume", False))
    normalized["keep_materialized_dataset"] = _bool(normalized.get("keep_materialized_dataset", False))
    label_source = str(normalized.get("label_source") or "").strip().lower()
    legacy_export_requested = _bool(
        normalized.get("export_labels_before_training", True)
    )
    if not label_source:
        label_source = (
            "mcs_refresh"
            if legacy_export_requested
            else "source_dataset"
        )
    elif label_source == "mcs_refresh" and not legacy_export_requested:
        # Preserve saved configurations from the former checkbox-only UI.
        label_source = "source_dataset"
    if label_source not in {value for _label, value in LABEL_SOURCE_CHOICES}:
        raise ValueError("Unknown training label source.")
    normalized["label_source"] = label_source
    normalized["export_labels_before_training"] = label_source == "mcs_refresh"
    label_root = str(normalized.get("label_root", "") or "").strip()
    normalized["label_root"] = (
        os.path.abspath(os.path.expanduser(os.path.expandvars(label_root)))
        if label_root
        else ""
    )
    if label_source == "exported_masks":
        if not normalized["label_root"]:
            raise ValueError("Choose the exported masks folder.")
        if not os.path.isdir(normalized["label_root"]):
            raise ValueError("The exported masks folder does not exist.")
    normalized["mask_names"] = ",".join(split_csv(normalized.get("mask_names", "")))
    normalized["model_sha256"] = str(
        normalized.get("model_sha256", "") or ""
    ).strip().lower()
    if normalized["model_sha256"] and (
        len(normalized["model_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in normalized["model_sha256"]
        )
    ):
        raise ValueError("Model SHA-256 must contain exactly 64 hexadecimal characters.")
    model_scale = str(
        normalized.get("model_scale") or "vits16"
    ).strip().lower()
    model_path = str(normalized.get("model_path") or "").strip()
    selected_backbone = None
    if model_scale == "custom" or model_path:
        if not model_path:
            raise ValueError(
                "Choose the custom DINOv3 weights folder."
            )
        model_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(model_path))
        )
        cached_onnx = (
            cached_slices
            and str(
                normalized.get("encoder_backend") or "onnx"
            ).strip().lower()
            == "onnx"
        )
        if cached_onnx:
            onnx_path = (
                model_path
                if model_path.lower().endswith(".onnx")
                else os.path.join(model_path, "model.onnx")
            )
            if not os.path.isfile(onnx_path):
                raise ValueError(
                    "Custom ONNX weights were not found: {0}".format(
                        onnx_path
                    )
                )
        else:
            try:
                selected_backbone = inspect_dinov3_vit_weights(model_path)
            except ValueError as exc:
                raise ValueError(
                    "Custom pretrained weights are not compatible: {0}".format(
                        exc
                    )
                )
            normalized["model_scale"] = model_scale_for_path(
                model_path,
                fallback=model_scale,
            )
            # A configured checksum only applies to its original fixed model.
            # Unknown and legacy external folders must not inherit it.
            if normalized["model_scale"] == "custom":
                normalized["model_sha256"] = ""
        normalized["model_path"] = model_path
    else:
        normalized["model_path"] = ""
    normalized["mcs_output_dir"] = os.path.abspath(os.path.expanduser(
        str(normalized.get("mcs_output_dir", "") or "")
    )) if str(normalized.get("mcs_output_dir", "") or "").strip() else ""
    strategy_values = {key: normalized.get(key, value) for key, value in DEFAULT_OPTIONS.items()}
    normalized.update(normalize_strategy_options(strategy_values, preset=str(normalized.get("strategy", "adaptive"))))
    if normalized["val_fraction"] < 0.0 or normalized["val_fraction"] > 0.9:
        raise ValueError("Validation fraction must be between 0.0 and 0.9.")
    parts = [part.strip() for part in str(normalized.get("img_size", "224,224")).split(",")]
    if len(parts) != 2:
        raise ValueError("Image size must be formatted as height,width.")
    image_size = [_int(part, "Image size", 16) for part in parts]
    model_patch_size = int(
        (selected_backbone or {}).get("patch_size") or 16
    )
    if any(value % model_patch_size for value in image_size):
        raise ValueError(
            "Image height and width must be multiples of the selected DINOv3 "
            "patch size ({0}).".format(model_patch_size)
        )
    sv_parts = [part.strip() for part in str(normalized.get("sub_volume_size", "32,256,256")).split(",")]
    if len(sv_parts) != 3:
        raise ValueError("Sub-volume size must be formatted as z,y,x.")
    [_int(part, "Sub-volume size", 1) for part in sv_parts]
    if cached_slices:
        encoder_backend = str(
            normalized.get("encoder_backend") or "onnx"
        ).strip().lower()
        if encoder_backend not in {"onnx", "pytorch"}:
            raise ValueError("Frozen Feature 2D encoder must be ONNX or PyTorch.")
        normalized["encoder_backend"] = encoder_backend
        if normalized["lr_scheduler"] == "constant_warmup":
            raise ValueError(
                "Frozen feature slice training supports cosine or constant learning rate."
            )
        adjustments = []
        cached_overrides = {
            "strategy": ("full_volume", "Frozen Feature 2D only supports full-volume sampling"),
            "finetune_method": ("frozen", "Frozen Feature 2D requires a frozen DINO encoder"),
            "model_scale": ("vits16", "Frozen Feature 2D uses the bundled ViT-S/16 encoder"),
            "sampling_mode": ("full", "Frozen Feature 2D does not use patch sampling"),
            "channel_policy": ("repeat", "Frozen Feature 2D uses single-channel repeat"),
            "slice_axis": ("axial", "Frozen Feature 2D preserves native axial slice order"),
            "grad_accumulation": (1, "Frozen Feature 2D uses real slice batches"),
            "mixed_precision": (False, "Frozen Feature 2D disables mixed precision"),
            "sub_volume": (False, "Frozen Feature 2D does not use 3D sub-volumes"),
            "warmup_epochs": (0, "Frozen Feature 2D disables warmup"),
        }
        for key, (target_value, reason) in cached_overrides.items():
            current = normalized.get(key)
            if current != target_value and str(current) != str(target_value):
                adjustments.append({
                    "field": key,
                    "from": current,
                    "to": target_value,
                    "reason": reason,
                })
            normalized[key] = target_value
        if adjustments:
            normalized["_compatibility_adjustments"] = adjustments
        if (
            encoder_backend == "onnx"
            and not str(normalized.get("model_path", "") or "").strip()
        ):
            if str(normalized.get("img_size", "")) != "256,256":
                if not normalized.get("_compatibility_adjustments"):
                    normalized["_compatibility_adjustments"] = []
                normalized["_compatibility_adjustments"].append({
                    "field": "img_size",
                    "from": normalized.get("img_size"),
                    "to": "256,256",
                    "reason": "The bundled ONNX encoder declares a fixed 256x256 input",
                })
            normalized["img_size"] = "256,256"
        elif encoder_backend == "pytorch":
            if not str(normalized.get("model_path", "") or "").strip():
                normalized["model_sha256"] = str(
                    normalized.get("safetensors_sha256")
                    or ""
                ).strip().lower()
    else:
        normalized["encoder_backend"] = "pytorch"
        if not str(normalized.get("model_path", "") or "").strip():
            if str(normalized.get("model_scale", "")).lower() == "vits16":
                normalized["model_sha256"] = str(
                    normalized.get("safetensors_sha256") or ""
                ).strip().lower()
            else:
                normalized["model_sha256"] = ""
        normalized["preserve_legacy_decoder"] = False
    return normalized


def append_training_args(cmd, config, options):
    strategy_options = {key: options.get(key, value) for key, value in DEFAULT_OPTIONS.items()}
    cmd.extend([
        "--strategy",
        str(options.get("strategy", config.get("default_strategy", "adaptive"))),
        "--strategy-options-json",
        json.dumps(strategy_options, sort_keys=True),
        "--base-config",
        str(options.get("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))),
        "--epochs",
        str(int(options.get("epochs", config.get("default_epochs", 20)))),
        "--batch-size",
        str(int(options.get("batch_size", config.get("default_batch_size", 0)))),
        "--grad-accumulation",
        str(int(options.get("grad_accumulation", config.get("default_grad_accumulation", 1)))),
        "--lr",
        str(float(options.get("lr", config.get("default_lr", 0.001)))),
        "--weight-decay",
        str(float(options.get("weight_decay", config.get("default_weight_decay", 0.01)))),
        "--lr-scheduler",
        str(options.get("lr_scheduler", config.get("default_lr_scheduler", "cosine"))),
        "--warmup-epochs",
        str(int(options.get("warmup_epochs", config.get("default_warmup_epochs", 3)))),
        "--validation-interval",
        str(int(options.get(
            "validation_interval",
            config.get("default_validation_interval", 2),
        ))),
        "--img-size",
        str(options.get("img_size", config.get("default_img_size", "256,256"))),
        "--modality",
        str(options.get("modality", config.get("default_modality", "auto"))),
        "--min-samples",
        str(int(options.get("min_samples", config.get("default_min_samples", 1)))),
        "--max-samples",
        str(int(options.get("max_samples", config.get("default_max_samples", 0)))),
        "--sample-mode",
        str(options.get("sample_mode", config.get("default_sample_mode", "all"))),
        "--val-fraction",
        str(float(options.get("val_fraction", config.get("default_val_fraction", 0.2)))),
        "--min-val-samples",
        str(int(options.get("min_val_samples", config.get("default_min_val_samples", 1)))),
        "--finetune-method",
        str(options.get("finetune_method", config.get("default_finetune_method", "lora"))),
        "--model-scale",
        str(options.get("model_scale", config.get("default_model_scale", "vitb16"))),
        "--encoder-backend",
        str(options.get("encoder_backend", "auto")),
        "--lora-rank",
        str(int(options.get("lora_rank", config.get("default_lora_rank", 8)))),
        "--lora-alpha",
        str(int(options.get("lora_alpha", config.get("default_lora_alpha", 16)))),
        "--adapter-bottleneck",
        str(int(options.get("adapter_bottleneck", config.get("default_adapter_bottleneck", 64)))),
        "--gpu-lock-timeout-seconds",
        str(float(options.get("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400)))),
        "--gpu-memory-gb",
        str(float(options.get("gpu_memory_gb", config.get("default_gpu_memory_gb", 0.0)))),
        "--background-mimics-lock-timeout-seconds",
        str(float(options.get(
            "background_mimics_lock_timeout_seconds",
            config.get("background_mimics_lock_timeout_seconds", 1800),
        ))),
        "--keep-last-checkpoints",
        str(int(options.get("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2)))),
    ])
    if _bool(options.get("preserve_legacy_decoder", False)):
        cmd.extend([
            "--decoder",
            str(options.get("decoder") or "feature_unet2d"),
        ])
    else:
        cmd.extend([
            "--training-dimension",
            str(options.get("training_dimension") or "auto"),
            "--quality-mode",
            str(options.get("quality_mode") or "standard"),
        ])
    model_path = str(options.get("model_path", config.get("default_model_path", "")) or "")
    if model_path:
        cmd.extend(["--model-path", model_path])
    model_sha256 = str(
        options.get("model_sha256", config.get("default_model_sha256", "")) or ""
    ).strip()
    if model_sha256:
        cmd.extend(["--model-sha256", model_sha256])
    val_cases = split_csv(options.get("val_cases", ""))
    if val_cases:
        cmd.extend(["--val-cases", ",".join(val_cases)])
    cases = split_csv(options.get("cases", ""))
    if cases:
        cmd.extend(["--cases", ",".join(cases)])
    if bool(options.get("mixed_precision", config.get("default_mixed_precision", False))):
        cmd.append("--mixed-precision")
    if bool(options.get("sub_volume", config.get("default_sub_volume", False))):
        cmd.append("--sub-volume")
    cmd.extend([
        "--sub-volume-size",
        str(options.get("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))),
    ])
    if bool(options.get("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))):
        cmd.append("--keep-materialized-dataset")
    label_root = str(options.get("label_root", "") or "").strip()
    if label_root:
        cmd.extend(["--label-root", label_root])
    mask_names = split_csv(options.get("mask_names", ""))
    if mask_names:
        cmd.extend(["--mask-names", ",".join(mask_names)])


def prepare_training_launch(context, options, run_id=None):
    config = context.get("config") or {}
    organ = context["organ"]
    options = dict(options)
    if (
        str(options.get("label_source") or "mcs_refresh") == "mcs_refresh"
        and not split_csv(options.get("mask_names", ""))
    ):
        options["mask_names"] = ",".join(configured_mask_names(config, organ))
    options = validate_options(options)
    ts_root = os.path.abspath(
        os.path.expanduser(os.path.expandvars(str(context.get("ts_root") or "")))
    ) if str(context.get("ts_root") or "").strip() else ""
    if not ts_root or not os.path.isdir(ts_root):
        raise RuntimeError("Choose an existing source image root before starting training.")
    workspace = os.path.join(ts_root, "fewshot_models")
    context["ts_root"] = ts_root
    context["workspace"] = workspace
    python_exe = context.get("python_exe") or sys.executable
    run_id = run_id or "train_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    organ_slug = safe_slug(organ)
    cancel_path = os.path.join(workspace, "runs", organ_slug, run_id, "cancel.request")
    status_path = os.path.join(workspace, "jobs", run_id + ".json")
    cmd = [
        python_exe,
        context["pipeline_script"],
        "train",
        "--ts-root",
        ts_root,
        "--organ",
        organ,
        "--dinov3-root",
        context["dinov3_root"],
        "--python",
        python_exe,
        "--run-id",
        run_id,
    ]
    if str(options.get("label_source") or "mcs_refresh") == "mcs_refresh":
        if "mimics_exe" in context and not str(context.get("mimics_exe") or "").strip():
            raise RuntimeError(
                "Refreshing labels requires a separate background Mimics executable. "
                "Configure MIMICS_BACKGROUND_EXE, or turn off label refresh and use already exported NIfTI labels."
            )
        cmd.append("--export-labels")
        mcs_output_dir = str(options.get("mcs_output_dir") or context.get("mcs_output_dir") or "").strip()
        if mcs_output_dir:
            cmd.extend(["--mcs-output-dir", os.path.abspath(mcs_output_dir)])
    append_training_args(cmd, config, options)
    mimics_exe = context.get("mimics_exe")
    if mimics_exe:
        cmd.extend(["--mimics-exe", mimics_exe])
    job_payload = {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": run_id,
        "kind": "train",
        "status": "launching",
        "organ": organ,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "cancel_path": cancel_path,
        "training_options": options,
        "retry_context": {
            key: context.get(key)
            for key in (
                "project_root",
                "pipeline_script",
                "dinov3_root",
                "python_exe",
                "mimics_exe",
                "mcs_output_dir",
                "ts_root",
                "workspace",
                "organ",
                "case_ids",
                "config",
            )
        },
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "launched_by": "external_advanced_ui",
    }
    return {
        "cmd": cmd,
        "run_id": run_id,
        "status_path": status_path,
        "cancel_path": cancel_path,
        "job_payload": job_payload,
        "options": options,
    }


def launch_training(context, options):
    if str(options.get("execution_backend") or "local") == "remote":
        return launch_remote_training(context, options)
    launch = prepare_training_launch(context, options)
    write_json_atomic(launch["status_path"], launch["job_payload"])
    try:
        process = subprocess.Popen(
            launch["cmd"],
            cwd=context.get("project_root") or None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_process_kwargs()
        )
    except Exception:
        payload = read_json(launch["status_path"], launch["job_payload"]) or launch["job_payload"]
        payload["status"] = "failed"
        payload["error"] = "Could not start DINOv3 training process."
        payload["traceback"] = traceback.format_exc()
        payload["updated_at_epoch"] = time.time()
        write_json_best_effort(launch["status_path"], payload)
        raise
    payload = read_json(launch["status_path"], launch["job_payload"]) or launch["job_payload"]
    payload["launcher_pid"] = process.pid
    payload["updated_at_epoch"] = time.time()
    write_json_best_effort(launch["status_path"], payload)
    setup_status_path = context.get("setup_status_path")
    if setup_status_path:
        write_json_best_effort(setup_status_path, {
            "schema_version": "mimics_fewshot_setup.v1",
            "job_id": context.get("setup_id"),
            "kind": "train_setup",
            "status": "training_started",
            "organ": context.get("organ"),
            "ts_root": str(context.get("ts_root") or ""),
            "workspace": context.get("workspace"),
            "training_job_id": launch["run_id"],
            "training_status_path": launch["status_path"],
            "launcher_pid": process.pid,
            "updated_at_epoch": time.time(),
        })
    return launch["run_id"], launch["status_path"], process.pid


def launch_remote_training(context, options):
    profile_id = str(options.get("remote_profile_id") or "").strip()
    if not profile_id:
        raise RuntimeError(
            "Choose a saved remote server or switch Compute back to This workstation."
        )
    launch = prepare_training_launch(context, options)
    payload = dict(launch["job_payload"])
    payload.update({
        "execution_backend": "remote",
        "remote_profile_id": profile_id,
        "status": "preparing_remote",
        "phase": "preparing_remote_data",
        "progress_percent": 0,
    })
    write_json_atomic(launch["status_path"], payload)
    spec_path = launch["status_path"] + ".remote.json"
    serializable_context = json.loads(json.dumps(context, default=str))
    write_json_atomic(spec_path, {
        "schema_version": "mimics_remote_launch.v1",
        "kind": "dinov3",
        "run_id": launch["run_id"],
        "status_path": launch["status_path"],
        "cancel_path": launch["cancel_path"],
        "remote_profile_id": profile_id,
        "context": serializable_context,
        "options": launch["options"],
        "created_at_epoch": time.time(),
    })
    controller = os.path.join(
        context.get("project_root") or _project_root,
        "tools",
        "remote_training_controller.py",
    )
    controller_log_path = launch["status_path"] + ".remote_controller.log"
    payload["controller_log_path"] = controller_log_path
    payload["diagnostic_log_paths"] = [controller_log_path]
    write_json_best_effort(launch["status_path"], payload)
    controller_log = None
    try:
        os.makedirs(os.path.dirname(controller_log_path), exist_ok=True)
        controller_log = open(
            controller_log_path, "a", encoding="utf-8", errors="replace"
        )
        process = subprocess.Popen(
            [
                str(context.get("python_exe") or sys.executable),
                controller,
                "run",
                "--spec",
                spec_path,
            ],
            cwd=context.get("project_root") or None,
            stdin=subprocess.DEVNULL,
            stdout=controller_log,
            stderr=subprocess.STDOUT,
            **hidden_process_kwargs()
        )
    except Exception:
        payload["status"] = "failed"
        payload["phase"] = "remote_controller_launch_failed"
        payload["error"] = "Could not start the remote training controller."
        payload["traceback"] = traceback.format_exc()
        payload["updated_at_epoch"] = time.time()
        write_json_best_effort(launch["status_path"], payload)
        raise
    finally:
        if controller_log is not None:
            try:
                controller_log.close()
            except Exception:
                pass
    latest = read_json(launch["status_path"], payload) or payload
    latest.update({
        "controller_pid": process.pid,
        "launcher_pid": process.pid,
        "remote_launch_spec": spec_path,
        "updated_at_epoch": time.time(),
    })
    write_json_best_effort(launch["status_path"], latest)
    setup_status_path = context.get("setup_status_path")
    if setup_status_path:
        write_json_best_effort(setup_status_path, {
            "schema_version": "mimics_fewshot_setup.v1",
            "job_id": context.get("setup_id"),
            "kind": "train_setup",
            "status": "training_started",
            "execution_backend": "remote",
            "remote_profile_id": profile_id,
            "organ": context.get("organ"),
            "ts_root": str(context.get("ts_root") or ""),
            "workspace": context.get("workspace"),
            "training_job_id": launch["run_id"],
            "training_status_path": launch["status_path"],
            "launcher_pid": process.pid,
            "updated_at_epoch": time.time(),
        })
    return launch["run_id"], launch["status_path"], process.pid


def _bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def format_status_line(job):
    if not isinstance(job, dict):
        return ""
    parts = []
    job_id = job.get("job_id")
    status = job.get("status")
    if job_id or status:
        parts.append("{0} | {1}".format(job_id or "job", status or "unknown"))
    if str(job.get("execution_backend") or "") == "remote":
        parts.append(
            "{0}, GPU {1}".format(
                job.get("profile_name") or "remote",
                job.get("remote_gpu_device") or "automatic",
            )
        )
        if job.get("dataset_cache_hit") is True:
            parts.append("training data reused")
        elif job.get("transfer_percent") is not None and status == "uploading":
            parts.append("upload {0}%".format(job.get("transfer_percent")))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        parts.append("train {0}, val {1}".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        ))
    export_progress = job.get("label_export_progress") or {}
    if isinstance(export_progress, dict) and export_progress:
        index = int(export_progress.get("index", 0) or 0)
        total = int(export_progress.get("total", 0) or 0)
        phase = str(export_progress.get("phase") or export_progress.get("status") or "running")
        parts.append("labels {0}/{1} {2}".format(index, total, phase) if total else "labels " + phase)
    progress = job.get("training_progress") or {}
    if isinstance(progress, dict):
        if progress.get("latest_epoch_line"):
            parts.append(str(progress.get("latest_epoch_line")))
        else:
            progress_parts = []
            if progress.get("epoch") is not None and progress.get("epochs") is not None:
                progress_parts.append("epoch {0}/{1}".format(progress.get("epoch"), progress.get("epochs")))
            if progress.get("phase"):
                progress_parts.append(str(progress.get("phase")))
            if progress.get("batch") is not None and progress.get("batches") is not None:
                progress_parts.append("batch {0}/{1}".format(progress.get("batch"), progress.get("batches")))
            metrics = progress.get("metrics") or {}
            if metrics.get("loss") is not None:
                try:
                    progress_parts.append("loss {0:.4f}".format(float(metrics.get("loss"))))
                except Exception:
                    progress_parts.append("loss {0}".format(metrics.get("loss")))
            if metrics.get("mean_dsc") is not None:
                try:
                    progress_parts.append("val_dice {0:.4f}".format(float(metrics.get("mean_dsc"))))
                except Exception:
                    progress_parts.append("val_dice {0}".format(metrics.get("mean_dsc")))
            if progress.get("best_dsc") is not None:
                try:
                    progress_parts.append("best {0:.4f}".format(float(progress.get("best_dsc"))))
                except Exception:
                    progress_parts.append("best {0}".format(progress.get("best_dsc")))
            if progress_parts:
                parts.append(", ".join(progress_parts))
    if job.get("error"):
        parts.append("error: {0}".format(job.get("error")))
    return " | ".join(parts)


def _option_label(value, labels):
    for label, item in labels:
        if str(item) == str(value):
            return label
    return "Custom ({0})".format(value)


def _labels_with_current(labels, current_label):
    values = [label for label, _item in labels]
    if current_label not in values:
        values.append(current_label)
    return values


def _choice_value(label, labels):
    for item_label, value in labels:
        if item_label == label:
            return value
    return None


def window_layout_for_screen(screen_width, screen_height):
    """Return geometry/minsize values that keep the action footer visible."""
    try:
        screen_width = int(screen_width)
        screen_height = int(screen_height)
    except Exception:
        screen_width, screen_height = 1280, 900
    width = min(1220, max(980, screen_width - 80))
    # Keep a real margin for the OS title bar/taskbar on shorter screens.  The
    # setup body is scrollable/compressible; a too-tall window hides the action
    # footer and feels broken.
    height = min(900, max(560, screen_height - 120))
    min_width = min(960, width)
    min_height = min(560, height)
    return width, height, min_width, min_height


def resolve_mimics_output_dir_for_ui(ts_root, project_root):
    default_dir = os.path.abspath(os.path.join(ts_root, "mcs_output"))
    merged = {}
    for name in ("mimics_io_config.json", "nninteractive_config.json"):
        path = os.path.join(project_root or "", name)
        loaded = read_json(path, {}) or {}
        if isinstance(loaded, dict):
            merged.update(loaded)
    configured = merged.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if os.path.isabs(configured):
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(ts_root, configured))


def case_ids_from_dataset_for_ui(ts_root, project_root=""):
    rows = set()
    output_dir = resolve_mimics_output_dir_for_ui(ts_root, project_root)
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.lower().endswith(".mcs"):
                rows.add(name[:-4])
    if os.path.isdir(ts_root):
        for name in os.listdir(ts_root):
            path = os.path.join(ts_root, name)
            if os.path.isdir(path) and name not in ("mcs_output", "segmentations", "fewshot_models"):
                rows.add(name)
    return sorted(rows)


def case_rows_from_dataset_for_ui(
    ts_root,
    project_root="",
    label_source="mcs_refresh",
    label_root="",
    mask_names=None,
    mcs_output_dir="",
):
    """Return a lightweight, label-aware case inventory for the setup window."""
    from fewshot_pipeline import find_image, find_label

    root = Path(ts_root).expanduser().resolve()
    source = normalized_source_mode(label_source)
    names = list(mask_names or [])
    mcs_root = (
        Path(mcs_output_dir).expanduser().resolve()
        if str(mcs_output_dir or "").strip()
        else Path(resolve_mimics_output_dir_for_ui(str(root), project_root))
    )
    if source == "mcs_refresh":
        case_ids = (
            sorted(path.stem for path in mcs_root.glob("*.mcs") if path.is_file())
            if mcs_root.is_dir()
            else []
        )
    else:
        excluded_paths = {mcs_root}
        if str(label_root or "").strip():
            excluded_paths.add(Path(label_root).expanduser().resolve())
        case_ids = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and path.name not in ("mcs_output", "segmentations", "fewshot_models")
            and path.resolve() not in excluded_paths
        )

    rows = []
    for case_id in case_ids:
        case_dir = root / case_id
        image = find_image(case_dir) if case_dir.is_dir() else None
        label = None
        mcs_path = mcs_root / (case_id + ".mcs")
        if source == "source_dataset" and case_dir.is_dir():
            label = find_label(
                case_dir,
                names[0] if names else "",
                fallback_to_case_labels=True,
                mask_names=names,
            )
        elif source == "exported_masks" and case_dir.is_dir():
            label = find_label(
                case_dir,
                names[0] if names else "",
                label_root=label_root,
                fallback_to_case_labels=False,
                mask_names=names,
            )

        if not image:
            state = "image_missing"
            detail = "Source image not found"
        elif source == "mcs_refresh" and mcs_path.is_file():
            state = "ready"
            detail = "Saved project found; Mask checked at training start"
        elif source == "mcs_refresh":
            state = "mcs_missing"
            detail = "Saved .mcs project not found"
        elif label:
            state = "ready"
            detail = "Matching Mask found"
        else:
            state = "mask_missing"
            detail = "Matching Mask not found"
        rows.append(
            {
                "case_id": case_id,
                "state": state,
                "detail": detail,
                "image": str(image or ""),
                "label": str(label or ""),
                "mcs_path": str(mcs_path) if mcs_path.is_file() else "",
            }
        )
    return rows


# UI choice tables shared by the setup window. The Tkinter backend
# (TrainingSetupApp) was removed: the PySide6 window (QtTrainingSetupApp) is the
# only supported external UI and PySide6 ships with the bundled python_env on
# every machine.
EPOCH_CHOICES = [("Fast check (3)", 3), ("Quick (5)", 5), ("Standard (10)", 10), ("More training (20)", 20)]
VAL_CHOICES = [("No validation", 0.0), ("Small validation (10%)", 0.1), ("Standard validation (20%)", 0.2), ("Larger validation (30%)", 0.3)]
MEMORY_CHOICES = [
    ("Balanced", "balanced"),
    ("Low GPU memory", "low_memory"),
    ("Higher quality", "quality"),
    ("Custom", "custom"),
]
LR_CHOICES = ["0.001", "0.0005", "0.0002", "0.0001"]
WEIGHT_DECAY_CHOICES = ["0.01", "0.001", "0.0001", "0.0"]
IMG_SIZE_CHOICES = [
    ("Fast (192 x 192)", "192,192"),
    ("Balanced (224 x 224)", "224,224"),
    ("Detailed (256 x 256)", "256,256"),
    ("High detail (320 x 320)", "320,320"),
]
IMG_SIZE_CUSTOM_LABEL = "Custom"
SUB_VOLUME_DEPTH_CHOICES = ["16", "24", "32", "48", "64"]


def _load_pyside6():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class QtTrainingSetupApp(object):
    """PySide6 implementation of the external training setup window."""

    EPOCH_CHOICES = EPOCH_CHOICES
    VAL_CHOICES = VAL_CHOICES
    MEMORY_CHOICES = MEMORY_CHOICES
    LR_CHOICES = LR_CHOICES
    WEIGHT_DECAY_CHOICES = WEIGHT_DECAY_CHOICES
    IMG_SIZE_CHOICES = IMG_SIZE_CHOICES
    IMG_SIZE_CUSTOM_LABEL = IMG_SIZE_CUSTOM_LABEL
    SUB_VOLUME_DEPTH_CHOICES = SUB_VOLUME_DEPTH_CHOICES

    def __init__(self, window, context, qt_modules):
        self.window = window
        self.context = context
        remembered_paths = load_preferences("dinov3_training")
        if not str(self.context.get("ts_root") or "").strip():
            self.context["ts_root"] = remembered_paths.get("dataset_root", "")
        if not str(self.context.get("mcs_output_dir") or "").strip():
            self.context["mcs_output_dir"] = remembered_paths.get("mcs_dir", "")
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        self.config = context.get("config") or {}
        self.profiles = self.config.get("training_profiles") or {}
        self.profile_names = sorted(self.profiles.keys()) if isinstance(self.profiles, dict) else []
        default_profile = self.config.get("default_training_profile") or (self.profile_names[0] if self.profile_names else "")
        self.values = training_options_for_organ(
            self.config, self.context.get("organ"), default_profile,
        )
        if isinstance(self.context.get("initial_options"), dict):
            initial_options = self.context.get("initial_options") or {}
            self.values.update(initial_options)
            if (
                "export_labels_before_training" in initial_options
                and "label_source" not in initial_options
            ):
                self.values["label_source"] = (
                    "mcs_refresh"
                    if _bool(initial_options["export_labels_before_training"])
                    else "source_dataset"
                )
        if not str(self.values.get("label_root") or "").strip():
            self.values["label_root"] = remembered_paths.get("label_root", "")
        self.model_records = discover_pretrained_models(
            self.context.get("dinov3_root") or "",
            self.values.get("model_path", ""),
            self.values.get("model_scale", "vitb16"),
        )
        self.widgets = {}
        self.quick_widgets = {}
        self.case_list = None
        self.case_rows = []
        self.case_filter_edit = None
        self.show_unavailable_cases = None
        self.choose_specific_cases = None
        self.manual_case_widget = None
        self.case_inventory_label = None
        self.mcs_path_widget = None
        self.exported_masks_path_widget = None
        self.data_source_hint = None
        self.manual_cases = None
        self.dataset_edit = None
        self.mcs_folder_edit = None
        self.mask_names_edit = None
        self.label_root_edit = None
        self.label_root_browse = None
        self.scan_data_button = None
        self.scan_progress = None
        self.status_label = None
        self.status_text = None
        self.status_group = None
        self.details_button = None
        self.start_button = None
        self.open_log_button = None
        self.close_button = None
        self.training_status_path = None
        self.training_log_dir = ""
        self.last_status_line = ""
        self.started = False
        self._closing = False
        self._syncing_quick = False
        self._dataset_scan_generation = 0
        self._dataset_scan_pending = False
        self._dataset_scan_deadline = 0.0
        self._active_dataset_scan_signature = None
        self._dataset_scan_results = Queue()
        self._dataset_scan_timer = self.QtCore.QTimer(self.window)
        self._dataset_scan_timer.timeout.connect(self._poll_dataset_scan)
        self._dataset_scan_timer.start(80)
        self._build()

    def _build(self):
        QtCore, QtGui, QtWidgets = self.QtCore, self.QtGui, self.QtWidgets
        self.window.setWindowTitle(TITLE)
        width, height, min_width, min_height = window_layout_for_screen(1280, 900)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            size = screen.availableGeometry().size()
            width, height, min_width, min_height = window_layout_for_screen(size.width(), size.height())
        self.window.resize(width, height)
        self.window.setMinimumSize(min_width, min_height)
        self.window.setStyleSheet(self._stylesheet())

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        title = QtWidgets.QLabel("DINOv3 Few-Shot Training")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("Organ: {0}".format(self.context.get("organ", "?")))
        subtitle.setObjectName("subtitleLabel")
        warning = QtWidgets.QLabel(
            "When using saved .mcs projects, save current edits before training. "
            "Data preparation and training run outside Mimics."
        )
        warning.setObjectName("warningLabel")
        warning.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addWidget(warning)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_setup_tab(), "Data and samples")
        self.tabs.addTab(self._build_expert_tab(), "Model and policy")
        outer.addWidget(self.tabs, 1)

        self.status_group = QtWidgets.QGroupBox("Setup details")
        status_layout = QtWidgets.QVBoxLayout(self.status_group)
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(96)
        status_layout.addWidget(self.status_text)
        self.status_group.setVisible(False)
        outer.addWidget(self.status_group)
        self._append_log(
            "Ready. Choose the data source and training options, then start background training."
        )

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Configure samples and parameters, then start background training.")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.close_button = QtWidgets.QPushButton("Cancel")
        self.close_button.clicked.connect(self.close)
        footer.addWidget(self.close_button)
        self.details_button = QtWidgets.QPushButton("Show Details")
        self.details_button.setToolTip(
            "Show setup and background handoff details."
        )
        self.details_button.clicked.connect(self._toggle_setup_details)
        footer.addWidget(self.details_button)
        self.open_log_button = QtWidgets.QPushButton("Open Log Folder")
        self.open_log_button.setEnabled(False)
        self.open_log_button.clicked.connect(self.open_log_folder)
        footer.addWidget(self.open_log_button)
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_training)
        footer.addWidget(self.start_button)
        outer.addLayout(footer)
        self.window.setCentralWidget(central)

        shortcut_start = QtGuiShortcut(QtGui, "Return", self.window)
        shortcut_start.activated.connect(self.start_training)
        shortcut_close = QtGuiShortcut(QtGui, "Escape", self.window)
        shortcut_close.activated.connect(self.close)
        self.shortcuts = [shortcut_start, shortcut_close]
        self._refresh_quick_labels()
        if self.context.get("ts_root"):
            self.QtCore.QTimer.singleShot(
                0,
                lambda: self.apply_dataset_root(self.context.get("ts_root", "")),
            )

    def _toggle_setup_details(self):
        if self.status_group is None:
            return
        visible = not self.status_group.isVisible()
        self.status_group.setVisible(visible)
        if self.details_button is not None:
            self.details_button.setText(
                "Hide Details" if visible else "Show Details"
            )

    def _stylesheet(self):
        return shared_stylesheet()

    def _build_setup_tab(self):
        QtWidgets = self.QtWidgets

        def folder_button(callback, tooltip):
            button = QtWidgets.QPushButton("Browse...")
            button.setIcon(
                self.window.style().standardIcon(
                    QtWidgets.QStyle.SP_DirOpenIcon
                )
            )
            button.setMinimumWidth(104)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            return button

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(body)
        tab_layout.addWidget(scroll)

        if RemoteComputeSelector is not None:
            self.remote_selector = RemoteComputeSelector(
                self.window,
                (self.QtCore, self.QtGui, self.QtWidgets),
            )
            layout.addWidget(self.remote_selector.group)

        profile_group = QtWidgets.QGroupBox("Training data")
        profile_layout = QtWidgets.QGridLayout(profile_group)
        profile_layout.setHorizontalSpacing(12)
        profile_layout.setVerticalSpacing(10)
        profile_layout.setColumnMinimumWidth(0, 172)
        profile_layout.setColumnStretch(1, 1)

        profile_layout.addWidget(QtWidgets.QLabel("Label source"), 0, 0)
        self.widgets["label_source"] = self._data_combo(
            LABEL_SOURCE_CHOICES,
            self.values.get("label_source", "mcs_refresh"),
        )
        self.widgets["label_source"].currentIndexChanged.connect(
            self._refresh_label_source_enabled
        )
        profile_layout.addWidget(self.widgets["label_source"], 0, 1, 1, 2)

        profile_layout.addWidget(QtWidgets.QLabel("Image modality"), 1, 0)
        self.widgets["modality"] = self._data_combo(
            MODALITY_ITEMS,
            self.values.get("modality", "auto"),
        )
        self.widgets["modality"].setToolTip(
            "Automatic uses reliable imported/DICOM metadata when every selected "
            "case can be classified; when none can be classified it uses the "
            "generic robust policy. Partial or mixed metadata asks you to choose "
            "CT, MRI, or Other. The saved choice is reused for 2D/3D, full/patch "
            "training, and inference."
        )
        profile_layout.addWidget(self.widgets["modality"], 1, 1, 1, 2)

        self.data_source_hint = QtWidgets.QLabel()
        self.data_source_hint.setObjectName("hint")
        self.data_source_hint.setWordWrap(True)
        profile_layout.addWidget(self.data_source_hint, 2, 0, 1, 3)

        dataset_caption = QtWidgets.QLabel("Original image dataset *")
        dataset_caption.setMinimumWidth(172)
        profile_layout.addWidget(dataset_caption, 3, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.context.get("ts_root") or ""))
        self.dataset_edit.editingFinished.connect(self.apply_dataset_from_field)
        self.dataset_edit.setPlaceholderText("Folder containing one subfolder per case")
        profile_layout.addWidget(self.dataset_edit, 3, 1)
        browse = folder_button(self.browse_dataset, "Choose original image dataset")
        profile_layout.addWidget(browse, 3, 2)

        self.mcs_path_widget = QtWidgets.QWidget()
        mcs_row = QtWidgets.QHBoxLayout(self.mcs_path_widget)
        mcs_row.setContentsMargins(0, 0, 0, 0)
        mcs_caption = QtWidgets.QLabel("Saved .mcs folder *")
        mcs_caption.setMinimumWidth(172)
        mcs_row.addWidget(mcs_caption)
        self.mcs_folder_edit = QtWidgets.QLineEdit(
            str(self.context.get("mcs_output_dir", ""))
        )
        self.mcs_folder_edit.setToolTip(
            "Required only when labels come from saved Mimics projects."
        )
        mcs_row.addWidget(self.mcs_folder_edit, 1)
        browse_mcs = folder_button(
            self.browse_mcs_folder,
            "Choose folder containing saved .mcs projects",
        )
        mcs_row.addWidget(browse_mcs)
        profile_layout.addWidget(self.mcs_path_widget, 4, 0, 1, 3)

        self.exported_masks_path_widget = QtWidgets.QWidget()
        export_row = QtWidgets.QHBoxLayout(self.exported_masks_path_widget)
        export_row.setContentsMargins(0, 0, 0, 0)
        export_caption = QtWidgets.QLabel("Exported masks folder *")
        export_caption.setMinimumWidth(172)
        export_row.addWidget(export_caption)
        self.label_root_edit = QtWidgets.QLineEdit(
            str(self.values.get("label_root", "") or "")
        )
        self.label_root_edit.setPlaceholderText(
            "Contains <case>/segmentations/<mask>.nii.gz"
        )
        export_row.addWidget(self.label_root_edit, 1)
        self.label_root_browse = folder_button(
            self.browse_label_root,
            "Choose previously exported masks folder",
        )
        export_row.addWidget(self.label_root_browse)
        profile_layout.addWidget(self.exported_masks_path_widget, 5, 0, 1, 3)

        mask_caption = QtWidgets.QLabel("Target Mask name(s) *")
        mask_caption.setMinimumWidth(172)
        profile_layout.addWidget(mask_caption, 6, 0)
        self.mask_names_edit = QtWidgets.QLineEdit(str(self.values.get("mask_names", "")))
        self.mask_names_edit.setPlaceholderText("Example: liver, liver_seg")
        self.mask_names_edit.setToolTip(
            "Comma-separated aliases for the same training target. One name must "
            "match in each usable case."
        )
        profile_layout.addWidget(self.mask_names_edit, 6, 1)
        self.scan_data_button = QtWidgets.QPushButton("Scan Data")
        self.scan_data_button.clicked.connect(self.apply_dataset_from_field)
        profile_layout.addWidget(self.scan_data_button, 6, 2)
        mask_source_hint = QtWidgets.QLabel(
            "Filled from the Mask selected in Mimics: {0}. Edit only to add "
            "alternative names used by other cases.".format(
                self.context.get("organ") or "(none)"
            )
        )
        mask_source_hint.setObjectName("hint")
        mask_source_hint.setWordWrap(True)
        profile_layout.addWidget(mask_source_hint, 7, 1, 1, 2)
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setRange(0, 100)
        self.scan_progress.setValue(0)
        self.scan_progress.setTextVisible(True)
        self.scan_progress.setFormat("Dataset has not been scanned")
        profile_layout.addWidget(self.scan_progress, 8, 1, 1, 2)
        layout.addWidget(profile_group)

        sample_group = QtWidgets.QGroupBox("Cases")
        sample_layout = QtWidgets.QVBoxLayout(sample_group)
        self.choose_specific_cases = QtWidgets.QCheckBox(
            "Choose specific cases instead of using every matching Mask"
        )
        self.choose_specific_cases.setChecked(False)
        self.choose_specific_cases.toggled.connect(
            self._refresh_manual_case_selection
        )
        sample_layout.addWidget(self.choose_specific_cases)
        self.case_inventory_label = QtWidgets.QLabel(
            "Scan the data to find cases with usable labels."
        )
        self.case_inventory_label.setObjectName("hint")
        self.case_inventory_label.setWordWrap(True)
        sample_layout.addWidget(self.case_inventory_label)

        sample_controls = QtWidgets.QHBoxLayout()
        sample_controls.addWidget(QtWidgets.QLabel("Sample order"))
        self.widgets["sample_mode"] = self._combo(["all", "latest"], self.values.get("sample_mode", "all"))
        sample_controls.addWidget(self.widgets["sample_mode"])
        sample_controls.addSpacing(16)
        sample_controls.addWidget(QtWidgets.QLabel("Max samples"))
        self.widgets["max_samples"] = self._spin(0, 100000, self.values.get("max_samples", 0))
        sample_controls.addWidget(self.widgets["max_samples"])
        sample_controls.addStretch(1)
        sample_layout.addLayout(sample_controls)

        self.manual_case_widget = QtWidgets.QWidget()
        manual_case_layout = QtWidgets.QVBoxLayout(self.manual_case_widget)
        manual_case_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QtWidgets.QHBoxLayout()
        self.case_filter_edit = QtWidgets.QLineEdit()
        self.case_filter_edit.setPlaceholderText("Filter cases")
        self.case_filter_edit.textChanged.connect(self._refresh_case_list)
        filter_row.addWidget(self.case_filter_edit, 1)
        self.show_unavailable_cases = QtWidgets.QCheckBox("Show unavailable")
        self.show_unavailable_cases.toggled.connect(self._refresh_case_list)
        filter_row.addWidget(self.show_unavailable_cases)
        manual_case_layout.addLayout(filter_row)

        middle = QtWidgets.QHBoxLayout()
        self.case_list = QtWidgets.QListWidget()
        self.case_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.case_list.setMinimumHeight(160)
        middle.addWidget(self.case_list, 1)
        side = QtWidgets.QVBoxLayout()
        select_all = QtWidgets.QPushButton("Select All")
        select_all.clicked.connect(self._select_all_cases)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.case_list.clearSelection)
        side.addWidget(select_all)
        side.addWidget(clear)
        side.addSpacing(12)
        hint = QtWidgets.QLabel(
            "Manual selection is used only when 'Choose specific cases' is enabled."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addStretch(1)
        middle.addLayout(side)
        manual_case_layout.addLayout(middle, 1)

        manual = QtWidgets.QHBoxLayout()
        manual.addWidget(QtWidgets.QLabel("Manual cases"))
        self.manual_cases = QtWidgets.QLineEdit()
        manual.addWidget(self.manual_cases, 1)
        manual_case_layout.addLayout(manual)
        sample_layout.addWidget(self.manual_case_widget)

        mins = QtWidgets.QHBoxLayout()
        mins.addWidget(QtWidgets.QLabel("Min train"))
        self.widgets["min_samples"] = self._spin(1, 100000, self.values.get("min_samples", 1))
        mins.addWidget(self.widgets["min_samples"])
        mins.addWidget(QtWidgets.QLabel("Min val"))
        self.widgets["min_val_samples"] = self._spin(0, 100000, self.values.get("min_val_samples", 1))
        mins.addWidget(self.widgets["min_val_samples"])
        mins.addStretch(1)
        sample_layout.addLayout(mins)
        layout.addWidget(sample_group, 1)
        layout.addStretch(1)
        self._refresh_label_source_enabled()
        self._refresh_manual_case_selection()
        return tab

    def _build_expert_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        scroll.setWidget(body)
        tab_layout.addWidget(scroll)

        strategy_group = QtWidgets.QGroupBox("Training strategy")
        strategy_layout = QtWidgets.QGridLayout(strategy_group)
        strategy_layout.setColumnStretch(1, 1)
        self.widgets["strategy"] = QtWidgets.QComboBox()
        for strategy_id in strategy_ids():
            self.widgets["strategy"].addItem(strategy_label(strategy_id), strategy_id)
        initial_strategy = str(self.values.get("strategy", "adaptive"))
        initial_index = self.widgets["strategy"].findData(initial_strategy)
        self.widgets["strategy"].setCurrentIndex(max(0, initial_index))
        strategy_layout.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
        strategy_layout.addWidget(self.widgets["strategy"], 0, 1)
        self.strategy_summary = QtWidgets.QLabel()
        self.strategy_summary.setWordWrap(True)
        self.strategy_summary.setObjectName("strategySummary")
        strategy_layout.addWidget(self.strategy_summary, 1, 0, 1, 2)
        self.widgets["strategy"].currentTextChanged.connect(self._apply_strategy_defaults)
        layout.addWidget(strategy_group)

        data_group = QtWidgets.QGroupBox("Data policy")
        prediction_group = QtWidgets.QGroupBox("Objective and prediction")
        data_form = QtWidgets.QFormLayout(data_group)
        prediction_form = QtWidgets.QFormLayout(prediction_group)
        data_form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        prediction_form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        self.data_policy_note = QtWidgets.QLabel()
        self.data_policy_note.setObjectName("hint")
        self.data_policy_note.setWordWrap(True)
        data_form.addRow(self.data_policy_note)

        self.widgets["sampling_mode"] = self._data_combo([
            ("Adaptive", "adaptive"), ("Full volume", "full"), ("Patch", "patch"),
        ], self.values.get("sampling_mode"))
        self.widgets["patch_size_mode"] = self._data_combo([
            ("From data fingerprint", "fingerprint"), ("Custom Z,Y,X", "custom"),
        ], self.values.get("patch_size_mode"))
        self.widgets["patch_size_zyx"] = QtWidgets.QLineEdit(str(self.values.get("patch_size_zyx")))
        self.widgets["patch_focus"] = self._data_combo([
            ("Foreground", "foreground"), ("Boundary", "boundary"),
            ("Foreground + nearby negatives", "negative_balanced"),
        ], self.values.get("patch_focus"))
        self.widgets["patches_per_case"] = self._spin(1, 32, self.values.get("patches_per_case", 2))
        self.widgets["channel_policy"] = self._data_combo([
            ("Automatic from spacing", "auto"),
            ("Single slice (repeat grayscale)", "repeat"),
            ("2.5D neighboring slices", "2_5d"),
        ], self.values.get("channel_policy"))
        self.widgets["slice_axis"] = self._data_combo([
            ("Automatic from spacing", "auto"),
            ("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal"),
        ], self.values.get("slice_axis"))
        self.widgets["neighbor_distance_mm"] = self._double_spin(
            0.1, 20.0, self.values.get("neighbor_distance_mm", 3.0), 0.5,
        )
        for label, key in (
            ("Sampling", "sampling_mode"), ("Patch size", "patch_size_mode"),
            ("Custom patch", "patch_size_zyx"), ("Patch focus", "patch_focus"),
            ("Patches / case", "patches_per_case"),
            ("Slice input", "channel_policy"), ("View plane", "slice_axis"),
            ("Neighbor distance (mm)", "neighbor_distance_mm"),
        ):
            data_form.addRow(label, self.widgets[key])

        self.widgets["loss_type"] = self._data_combo([
            ("Automatic (Dice + Focal)", "auto"),
            ("Dice + Focal", "dice_focal"),
            ("Dice + Cross Entropy", "dice_ce"),
        ], self.values.get("loss_type"))
        self.widgets["keep_largest_component"] = QtWidgets.QCheckBox("Keep largest connected component")
        self.widgets["keep_largest_component"].setChecked(_bool(self.values.get("keep_largest_component", False)))
        self.widgets["mirror_tta"] = QtWidgets.QCheckBox("Mirror averaging (slower)")
        self.widgets["mirror_tta"].setChecked(_bool(self.values.get("mirror_tta", False)))
        self.widgets["mirror_tta"].setToolTip(
            "Average the original prediction with laterality-safe in-plane "
            "mirrors. Left/right mirroring is excluded. The same safe axes are "
            "used for training augmentation."
        )
        for label, key in (("Loss", "loss_type"),):
            prediction_form.addRow(label, self.widgets[key])
        inference_note = QtWidgets.QLabel("Inference is automatic: full-volume training uses whole-volume inference; patch training uses sliding windows.")
        inference_note.setWordWrap(True)
        prediction_form.addRow(inference_note)
        prediction_form.addRow(self.widgets["keep_largest_component"])
        prediction_form.addRow(self.widgets["mirror_tta"])
        policy_row = QtWidgets.QWidget()
        policy_layout = QtWidgets.QHBoxLayout(policy_row)
        policy_layout.setContentsMargins(0, 0, 0, 0)
        policy_layout.setSpacing(12)
        for group in (data_group, prediction_group):
            group.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Maximum,
            )
        policy_layout.addWidget(data_group, 1)
        policy_layout.addWidget(prediction_group, 1)
        policy_layout.setAlignment(data_group, self.QtCore.Qt.AlignTop)
        policy_layout.setAlignment(prediction_group, self.QtCore.Qt.AlignTop)
        layout.addWidget(policy_row)

        left = QtWidgets.QGroupBox("Model and task")
        right = QtWidgets.QGroupBox("Training and resources")
        left_form = QtWidgets.QFormLayout(left)
        right_form = QtWidgets.QFormLayout(right)
        left_form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        right_form.setLabelAlignment(self.QtCore.Qt.AlignRight)

        model_hint = QtWidgets.QLabel(
            "Choose the training dimension. The final decoder, slice context, "
            "and inference path are resolved as one compatible plan."
        )
        model_hint.setWordWrap(True)
        left_form.addRow(model_hint)
        self.widgets["training_dimension"] = self._data_combo(
            TRAINING_DIMENSION_ITEMS,
            self.values.get("training_dimension", "auto"),
        )
        self.widgets["quality_mode"] = self._data_combo(
            QUALITY_MODE_ITEMS,
            self.values.get("quality_mode", "standard"),
        )
        self.widgets["finetune_method"] = self._combo(
            ["frozen", "lora"],
            self.values.get("finetune_method", "frozen"),
        )
        self.widgets["finetune_method"].currentTextChanged.connect(self._refresh_method_enabled)
        selected_model_key = self._initial_model_key()
        model_items = [
            (item["label"], item["key"])
            for item in self.model_records
        ]
        if not model_items:
            model_items = [("No compatible DINOv3 ViT weights installed", "")]
        self.widgets["model_selection"] = self._data_combo(
            model_items,
            selected_model_key,
        )
        self.widgets["model_selection"].setEnabled(bool(self.model_records))
        self.widgets["model_selection"].setToolTip(
            "Lists validated Hugging Face DINOv3 ViT folders installed under "
            "external/dinov3-medical-seg/models. The same selected backbone is "
            "supported by both the 2D and 3D training plans."
        )
        self.widgets["model_selection"].currentIndexChanged.connect(
            self._model_selection_changed
        )
        self.widgets["lora_rank"] = self._spin(1, 128, self.values.get("lora_rank", 8))
        self.widgets["lora_alpha"] = self._spin(1, 512, self.values.get("lora_alpha", 16))
        for label, key in [
            ("Training dimension", "training_dimension"),
            ("3D quality (when used)", "quality_mode"),
            ("Backbone adaptation", "finetune_method"),
            ("Pretrained weights", "model_selection"),
            ("LoRA rank", "lora_rank"),
            ("LoRA alpha", "lora_alpha"),
        ]:
            left_form.addRow(label, self.widgets[key])
        self.dimensionality_summary = QtWidgets.QLabel()
        self.dimensionality_summary.setWordWrap(True)
        self.dimensionality_summary.setObjectName("strategySummary")
        left_form.addRow(self.dimensionality_summary)
        backend_hint = QtWidgets.QLabel(
            "Install another complete Hugging Face DINOv3 ViT folder in the fixed "
            "models directory to add it here. New 2D/3D plans use multi-level "
            "PyTorch features; saved ONNX compatibility models remain loadable."
        )
        backend_hint.setWordWrap(True)
        left_form.addRow(backend_hint)

        self.resource_hint = QtWidgets.QLabel(
            "Auto selects a conservative batch from the selected GPU. Set a "
            "larger real batch for a remote GPU; variable-depth volumes are "
            "padded with ignored label voxels."
        )
        resource_hint = self.resource_hint
        resource_hint.setWordWrap(True)
        right_form.addRow(resource_hint)
        self.widgets["epochs"] = self._spin(1, 10000, self.values.get("epochs", 20))
        self.widgets["batch_size"] = self._spin(
            0,
            64,
            self.values.get("batch_size", 0),
        )
        self.widgets["batch_size"].setSpecialValueText("Auto")
        self.widgets["batch_size"].setToolTip(
            "0 selects a hardware-aware recommendation. A larger value is "
            "passed unchanged to local or remote training."
        )
        self.widgets["gpu_memory_gb"] = self._double_spin(
            0.0,
            256.0,
            self.values.get("gpu_memory_gb", 0.0),
            1.0,
        )
        self.widgets["gpu_memory_gb"].setSpecialValueText("Auto detect")
        self.widgets["gpu_memory_gb"].setSuffix(" GB")
        self.widgets["grad_accumulation"] = self._spin(1, 1024, self.values.get("grad_accumulation", 1))
        self.widgets["lr"] = self._combo(self.LR_CHOICES, self.values.get("lr", "0.001"), editable=True)
        self.widgets["weight_decay"] = self._combo(self.WEIGHT_DECAY_CHOICES, self.values.get("weight_decay", "0.01"), editable=True)
        self.widgets["lr_scheduler"] = self._combo(["cosine", "constant_warmup", "constant"], self.values.get("lr_scheduler", "cosine"))
        self.widgets["warmup_epochs"] = self._spin(0, 1000, self.values.get("warmup_epochs", 3))
        self.widgets["validation_interval"] = self._spin(
            1,
            1000,
            self.values.get("validation_interval", 2),
        )
        img_label = self._img_size_choice_from_value(self.values.get("img_size", "224,224"))
        self.quick_widgets["img_size_choice"] = self._combo([label for label, _ in self.IMG_SIZE_CHOICES] + [self.IMG_SIZE_CUSTOM_LABEL], img_label)
        self.widgets["img_size"] = QtWidgets.QLineEdit(str(self.values.get("img_size", "224,224")))
        self.widgets["img_size"].setPlaceholderText("height,width")
        self.widgets["sub_volume_depth"] = self._combo(self.SUB_VOLUME_DEPTH_CHOICES, self._sub_volume_depth_from_values(self.values))
        self.widgets["sub_volume_depth"].setToolTip(
            "Advanced memory option: split each 3D case into z-depth chunks during training."
        )
        self.widgets["keep_last_checkpoints"] = self._spin(0, 1000, self.values.get("keep_last_checkpoints", 2))
        self.widgets["val_fraction"] = self._double_spin(0.0, 0.9, self.values.get("val_fraction", 0.2), 0.05)
        self.widgets["mixed_precision"] = QtWidgets.QCheckBox("Mixed precision")
        self.widgets["mixed_precision"].setChecked(_bool(self.values.get("mixed_precision", False)))
        self.widgets["sub_volume"] = QtWidgets.QCheckBox("Sub-volume training")
        self.widgets["sub_volume"].setChecked(_bool(self.values.get("sub_volume", False)))

        for label, key in [
            ("Epochs", "epochs"),
            ("Batch size", "batch_size"),
            ("GPU memory budget", "gpu_memory_gb"),
            ("Grad accumulation", "grad_accumulation"),
            ("Learning rate", "lr"),
            ("Weight decay", "weight_decay"),
            ("LR schedule", "lr_scheduler"),
            ("Warmup epochs", "warmup_epochs"),
            ("Validation every N epochs", "validation_interval"),
        ]:
            right_form.addRow(label, self.widgets[key])
        right_form.addRow("Image detail", self.quick_widgets["img_size_choice"])
        right_form.addRow("Custom size", self.widgets["img_size"])
        right_form.addRow("Validation fraction", self.widgets["val_fraction"])
        check_row = QtWidgets.QHBoxLayout()
        check_row.addWidget(self.widgets["mixed_precision"])
        check_row.addStretch(1)
        right_form.addRow(check_row)

        self.quick_widgets["img_size_choice"].currentTextChanged.connect(self._sync_image_size_choice)
        self.widgets["img_size"].textChanged.connect(self._sync_custom_image_size)
        for key in ("epochs", "batch_size", "grad_accumulation", "val_fraction", "sub_volume_depth"):
            self._connect_change(key, self._refresh_quick_labels)
        self.widgets["sub_volume"].stateChanged.connect(self._refresh_quick_labels)
        self.widgets["sub_volume"].stateChanged.connect(self._refresh_method_enabled)
        self.widgets["lr_scheduler"].currentTextChanged.connect(self._refresh_method_enabled)
        self.widgets["training_dimension"].currentTextChanged.connect(
            self._architecture_selection_changed
        )
        self.widgets["quality_mode"].currentTextChanged.connect(
            self._update_dimensionality_summary
        )
        self.widgets["sampling_mode"].currentTextChanged.connect(self._update_dimensionality_summary)
        self.widgets["channel_policy"].currentTextChanged.connect(self._update_dimensionality_summary)
        self._set_custom_size_entry_state()
        self._refresh_method_enabled()
        self._refresh_model_source_enabled()
        self._update_dimensionality_summary()
        epoch_label = _option_label(self.values.get("epochs", 10), self.EPOCH_CHOICES)
        val_label = _option_label(self.values.get("val_fraction", 0.2), self.VAL_CHOICES)
        memory_label = self._memory_mode_from_values(self.values)
        self.quick_widgets["epochs_choice"] = self._combo(_labels_with_current(self.EPOCH_CHOICES, epoch_label), epoch_label)
        self.quick_widgets["val_fraction_choice"] = self._combo(_labels_with_current(self.VAL_CHOICES, val_label), val_label)
        self.quick_widgets["memory_mode"] = self._combo([label for label, _ in self.MEMORY_CHOICES], memory_label)
        self.quick_widgets["img_size_choice"].setToolTip("Choose a preset or enter any positive height,width. Values should be multiples of 16.")
        parameter_row = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QHBoxLayout(parameter_row)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(12)
        for group in (left, right):
            group.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Maximum,
            )
        parameter_layout.addWidget(left, 1)
        parameter_layout.addWidget(right, 1)
        parameter_layout.setAlignment(left, self.QtCore.Qt.AlignTop)
        parameter_layout.setAlignment(right, self.QtCore.Qt.AlignTop)
        layout.addWidget(parameter_row)
        layout.addStretch(1)
        self._update_strategy_summary()
        for key in STRATEGY_DATA_KEYS:
            self._connect_change(key, self._strategy_option_changed)
        self._refresh_strategy_enabled()
        return tab

    def _combo(self, values, current="", editable=False):
        combo = self.QtWidgets.QComboBox()
        combo.setEditable(bool(editable))
        text_values = [str(value) for value in values if str(value)]
        combo.addItems(text_values)
        current = str(current or "")
        if current and combo.findText(current) < 0:
            combo.addItem(current)
        if current:
            combo.setCurrentText(current)
        return combo

    def _data_combo(self, items, current=None):
        combo = self.QtWidgets.QComboBox()
        for label, value in items:
            combo.addItem(str(label), str(value))
        index = combo.findData(str(current))
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _spin(self, minimum, maximum, value, step=1):
        widget = self.QtWidgets.QSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setSingleStep(int(step))
        try:
            widget.setValue(int(value))
        except Exception:
            widget.setValue(int(minimum))
        return widget

    def _double_spin(self, minimum, maximum, value, step):
        widget = self.QtWidgets.QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(float(step))
        widget.setDecimals(4)
        try:
            widget.setValue(float(value))
        except Exception:
            widget.setValue(float(minimum))
        return widget

    def _connect_change(self, key, callback):
        widget = self.widgets.get(key)
        if widget is None:
            return
        if hasattr(widget, "valueChanged"):
            widget.valueChanged.connect(callback)
        elif hasattr(widget, "currentTextChanged"):
            widget.currentTextChanged.connect(callback)
        elif hasattr(widget, "textChanged"):
            widget.textChanged.connect(callback)
        elif hasattr(widget, "stateChanged"):
            widget.stateChanged.connect(callback)

    def _select_all_cases(self):
        for idx in range(self.case_list.count()):
            item = self.case_list.item(idx)
            if item.flags() & self.QtCore.Qt.ItemIsEnabled:
                item.setSelected(True)

    def _refresh_manual_case_selection(self):
        manual = bool(
            self.choose_specific_cases
            and self.choose_specific_cases.isChecked()
        )
        if self.manual_case_widget is not None:
            self.manual_case_widget.setVisible(manual)
        for control in (
            self.case_filter_edit,
            self.show_unavailable_cases,
            self.case_list,
            self.manual_cases,
        ):
            if control is not None:
                control.setEnabled(manual)
        if not manual:
            if self.case_list is not None:
                self.case_list.clear()
            return
        self._refresh_case_list()

    def _refresh_case_list(self):
        if self.case_list is None:
            return
        if self.choose_specific_cases is not None and not self.choose_specific_cases.isChecked():
            self.case_list.clear()
            return
        query = (
            str(self.case_filter_edit.text()).strip().lower()
            if self.case_filter_edit is not None
            else ""
        )
        show_unavailable = bool(
            self.show_unavailable_cases
            and self.show_unavailable_cases.isChecked()
        )
        selected_ids = {
            str(item.data(self.QtCore.Qt.UserRole) or item.text())
            for item in self.case_list.selectedItems()
        }
        self.case_list.setUpdatesEnabled(False)
        try:
            self.case_list.clear()
            for row in self.case_rows:
                ready = row.get("state") == "ready"
                case_id = str(row.get("case_id") or "")
                if not ready and not show_unavailable:
                    continue
                if query and query not in case_id.lower():
                    continue
                text = case_id if ready else "{}  -  {}".format(
                    case_id,
                    row.get("detail") or "Unavailable",
                )
                item = self.QtWidgets.QListWidgetItem(text)
                item.setData(self.QtCore.Qt.UserRole, case_id)
                if not ready:
                    item.setFlags(item.flags() & ~self.QtCore.Qt.ItemIsEnabled)
                    item.setForeground(self.QtGui.QColor("#98a2b3"))
                elif case_id in selected_ids:
                    item.setSelected(True)
                self.case_list.addItem(item)
        finally:
            self.case_list.setUpdatesEnabled(True)

    def browse_dataset(self):
        choose_existing_directory_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Original image dataset",
            self.dataset_edit.text() or self.context.get("ts_root", "") or str(Path.home()),
            self.apply_dataset_root,
        )

    def browse_mcs_folder(self):
        current = str(self.mcs_folder_edit.text()).strip() if self.mcs_folder_edit is not None else ""
        def selected(path):
            if path and self.mcs_folder_edit is not None:
                self.mcs_folder_edit.setText(os.path.abspath(path))
                self.context["mcs_output_dir"] = os.path.abspath(path)

        choose_existing_directory_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Saved .mcs folder",
            current or str(Path.home()),
            selected,
        )

    def browse_label_root(self):
        current = (
            str(self.label_root_edit.text()).strip()
            if self.label_root_edit is not None
            else ""
        )
        def selected(path):
            if path and self.label_root_edit is not None:
                self.label_root_edit.setText(os.path.abspath(path))

        choose_existing_directory_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Exported masks folder",
            current or str(Path.home()),
            selected,
        )

    def _initial_model_key(self):
        raw_path = str(self.values.get("model_path") or "").strip()
        if raw_path:
            candidate = Path(os.path.expandvars(os.path.expanduser(raw_path)))
            if not candidate.is_absolute():
                candidate = (
                    Path(str(self.context.get("dinov3_root") or ""))
                    / candidate
                )
            try:
                resolved = os.path.normcase(str(candidate.resolve()))
            except OSError:
                resolved = os.path.normcase(str(candidate))
            for record in self.model_records:
                if os.path.normcase(record["path"]) == resolved:
                    return record["key"]
        scale = str(self.values.get("model_scale") or "vitb16").lower()
        for record in self.model_records:
            if record["scale"] == scale:
                return record["key"]
        return self.model_records[0]["key"] if self.model_records else ""

    def _selected_model_record(self):
        widget = self.widgets.get("model_selection")
        key = str(widget.currentData() or "") if widget is not None else ""
        for record in self.model_records:
            if record["key"] == key:
                return record
        return None

    def _select_model_scale(self, scale):
        widget = self.widgets.get("model_selection")
        if widget is None:
            return False
        for record in self.model_records:
            if record["scale"] != str(scale).lower():
                continue
            index = widget.findData(record["key"])
            if index >= 0:
                widget.setCurrentIndex(index)
                return True
        return False

    def _model_selection_changed(self, *_args):
        record = self._selected_model_record()
        if record is None:
            self.values["model_path"] = ""
            return
        self.values["model_path"] = record["path"]
        self.values["model_scale"] = record["scale"]
        self.values["model_sha256"] = ""
        self._refresh_method_enabled()
        self._update_dimensionality_summary()

    def _refresh_label_source_enabled(self):
        widget = self.widgets.get("label_source")
        source = (
            str(widget.currentData() or "mcs_refresh")
            if widget is not None
            else str(self.values.get("label_source", "mcs_refresh"))
        )
        use_mcs = source == "mcs_refresh"
        use_export = source == "exported_masks"
        self.values["label_source"] = source
        self.values["export_labels_before_training"] = use_mcs
        if self.mcs_path_widget is not None:
            self.mcs_path_widget.setVisible(use_mcs)
        if self.exported_masks_path_widget is not None:
            self.exported_masks_path_widget.setVisible(use_export)
        if self.data_source_hint is not None:
            self.data_source_hint.setText(label_source_hint(source))
        if self.case_rows:
            self.case_inventory_label.setText(
                "Data source changed. Scan again before starting training."
            )

    def _current_data_signature(self):
        def normalized_path(widget):
            value = str(widget.text()).strip() if widget is not None else ""
            return os.path.normcase(os.path.abspath(value)) if value else ""

        return (
            normalized_source_mode(self._widget_value("label_source")),
            normalized_path(self.dataset_edit),
            normalized_path(self.mcs_folder_edit),
            normalized_path(self.label_root_edit),
            tuple(
                sorted(
                    safe_slug(value)
                    for value in split_csv(
                        self.mask_names_edit.text()
                        if self.mask_names_edit is not None
                        else ""
                    )
                )
            ),
        )

    def apply_dataset_from_field(self):
        if self.dataset_edit is None:
            return
        path = str(self.dataset_edit.text()).strip()
        if path:
            self.apply_dataset_root(path)

    def _set_scan_busy(self):
        if self.scan_progress is not None:
            self.scan_progress.setRange(0, 0)
            self.scan_progress.setFormat("Scanning dataset...")
        if self.scan_data_button is not None:
            self.scan_data_button.setText("Scanning...")
            self.scan_data_button.setEnabled(False)

    def _finish_scan_progress(self, text, success):
        self._dataset_scan_deadline = 0.0
        if self.scan_progress is not None:
            self.scan_progress.setRange(0, 100)
            self.scan_progress.setValue(100 if success else 0)
            self.scan_progress.setFormat(str(text))
        if self.scan_data_button is not None:
            self.scan_data_button.setText("Scan Data")
            self.scan_data_button.setEnabled(True)

    def apply_dataset_root(self, path):
        raw_path = str(path or "").strip()
        if not raw_path:
            self._dataset_scan_pending = False
            self._set_status("Choose the source image root to continue.")
            self._finish_scan_progress("Dataset path is required", False)
            return
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(raw_path)))
        if self.dataset_edit is not None and self.dataset_edit.text() != path:
            self.dataset_edit.setText(path)
        self._dataset_scan_generation += 1
        self._dataset_scan_pending = True
        self._dataset_scan_deadline = time.time() + float(
            self.config.get("dataset_scan_timeout_seconds", 120)
        )
        generation = self._dataset_scan_generation
        project_root = self.context.get("project_root", "")
        label_source = self._widget_value("label_source")
        label_root = (
            str(self.label_root_edit.text()).strip()
            if self.label_root_edit is not None
            else ""
        )
        mcs_output_dir = (
            str(self.mcs_folder_edit.text()).strip()
            if self.mcs_folder_edit is not None
            else ""
        )
        mask_names = split_csv(
            self.mask_names_edit.text() if self.mask_names_edit is not None else ""
        )
        if not mask_names:
            self._dataset_scan_pending = False
            self._set_status("Enter at least one Target Mask name before scanning.")
            self._finish_scan_progress("Target Mask name is required", False)
            return
        if label_source == "mcs_refresh" and not mcs_output_dir:
            self._dataset_scan_pending = False
            self._set_status("Choose the folder containing saved .mcs projects.")
            self._finish_scan_progress("Saved .mcs folder is required", False)
            return
        if label_source == "exported_masks" and not label_root:
            self._dataset_scan_pending = False
            self._set_status("Choose the previously exported masks folder.")
            self._finish_scan_progress("Exported masks folder is required", False)
            return
        scan_signature = self._current_data_signature()
        self._set_status("Checking dataset in the background...")
        self._set_scan_busy()
        if self.case_list is not None:
            self.case_list.setEnabled(False)
        if self.start_button is not None:
            self.start_button.setEnabled(False)

        def scan_dataset():
            try:
                if not os.path.isdir(path):
                    raise RuntimeError("Dataset folder does not exist: {0}".format(path))
                if label_source == "mcs_refresh" and not os.path.isdir(
                    mcs_output_dir
                ):
                    raise RuntimeError(
                        "Saved .mcs folder does not exist or is unavailable: "
                        "{0}".format(mcs_output_dir)
                    )
                if label_source == "exported_masks" and not os.path.isdir(
                    label_root
                ):
                    raise RuntimeError(
                        "Exported masks folder does not exist or is unavailable: "
                        "{0}".format(label_root)
                    )
                output_dir = (
                    os.path.abspath(mcs_output_dir)
                    if mcs_output_dir
                    else resolve_mimics_output_dir_for_ui(path, project_root)
                )
                rows = case_rows_from_dataset_for_ui(
                    path,
                    project_root=project_root,
                    label_source=label_source,
                    label_root=label_root,
                    mask_names=mask_names,
                    mcs_output_dir=output_dir,
                )
                self._dataset_scan_results.put(
                    (generation, path, output_dir, rows, scan_signature, "")
                )
            except Exception as exc:
                self._dataset_scan_results.put(
                    (generation, path, "", [], scan_signature, str(exc))
                )

        worker = threading.Thread(target=scan_dataset, name="fewshot-dataset-scan")
        worker.daemon = True
        worker.start()

    def _poll_dataset_scan(self):
        if self._closing:
            return
        while True:
            try:
                generation, path, output_dir, rows, scan_signature, error = (
                    self._dataset_scan_results.get_nowait()
                )
            except Empty:
                if (
                    self._dataset_scan_pending
                    and self._dataset_scan_deadline
                    and time.time() >= self._dataset_scan_deadline
                ):
                    self._dataset_scan_generation += 1
                    self._dataset_scan_pending = False
                    self._dataset_scan_deadline = 0.0
                    self._active_dataset_scan_signature = None
                    message = (
                        "Dataset scan timed out. The selected location may be "
                        "offline or responding too slowly; choose another path "
                        "or retry."
                    )
                    self._set_status(message)
                    self._append_log(message)
                    self._finish_scan_progress("Scan timed out", False)
                    if self.case_list is not None:
                        self.case_list.setEnabled(True)
                    if self.start_button is not None and not self.started:
                        self.start_button.setEnabled(True)
                return
            if generation != self._dataset_scan_generation:
                continue
            if error:
                self._dataset_scan_pending = False
                self._dataset_scan_deadline = 0.0
                self._set_status(error)
                self._append_log(error)
                self._finish_scan_progress("Scan failed", False)
                if self.case_list is not None:
                    self.case_list.setEnabled(True)
                if self.start_button is not None and not self.started:
                    self.start_button.setEnabled(True)
                continue
            self._dataset_scan_pending = False
            self._dataset_scan_deadline = 0.0
            self._active_dataset_scan_signature = scan_signature
            self.context["ts_root"] = path
            self.context["workspace"] = os.path.join(path, "fewshot_models")
            self.context["mcs_output_dir"] = output_dir
            if self.mcs_folder_edit is not None and not self.mcs_folder_edit.text().strip():
                self.mcs_folder_edit.setText(output_dir)
            self.case_rows = list(rows)
            ready_rows = [row for row in rows if row.get("state") == "ready"]
            unavailable = len(rows) - len(ready_rows)
            self.context["case_ids"] = [
                str(row.get("case_id")) for row in ready_rows
            ]
            self._refresh_manual_case_selection()
            if self.start_button is not None and not self.started:
                self.start_button.setEnabled(True)
            if self.case_inventory_label is not None:
                if scan_signature[0] == "mcs_refresh":
                    self.case_inventory_label.setText(
                        "{} saved project candidate(s) found; {} case(s) are "
                        "missing a source image. Target Masks are verified in the "
                        "background when training starts, and projects without a "
                        "match are skipped automatically.".format(
                            len(ready_rows),
                            unavailable,
                        )
                    )
                else:
                    self.case_inventory_label.setText(
                        "{} usable case(s) found; {} unavailable case(s) hidden. "
                        "Only cases with both an image and matching Mask will be "
                        "used.".format(
                            len(ready_rows),
                            unavailable,
                        )
                    )
            message = "Dataset checked: {} usable, {} unavailable.".format(
                len(ready_rows),
                unavailable,
            )
            self._finish_scan_progress(
                "Scan complete: {} usable, {} unavailable".format(
                    len(ready_rows), unavailable
                ),
                True,
            )
            self._set_status(message)
            self._append_log(message)

    def _available_model_scales(self):
        records = discover_pretrained_models(
            self.context.get("dinov3_root") or "",
            self.values.get("model_path", ""),
            self.values.get("model_scale", "vitb16"),
        )
        choices = [
            item["scale"]
            for item in records
            if item["scale"] != "custom"
        ]
        current = str(self.values.get("model_scale", "vitb16") or "vitb16")
        if current not in choices:
            choices.insert(0, current)
        return choices or ["vitb16"]

    def _sub_volume_depth_from_values(self, values):
        parts = str(values.get("sub_volume_size", "32,256,256")).split(",")
        depth = parts[0].strip() if parts else "32"
        return depth if depth in self.SUB_VOLUME_DEPTH_CHOICES else "32"

    def _img_size_choice_from_value(self, value):
        for label, item in self.IMG_SIZE_CHOICES:
            if str(item) == str(value):
                return label
        return self.IMG_SIZE_CUSTOM_LABEL

    def _memory_mode_from_values(self, values):
        sub_volume = _bool(values.get("sub_volume", False))
        try:
            grad_accumulation = int(values.get("grad_accumulation", 1) or 1)
            batch_size = int(values.get("batch_size", 1) or 1)
        except Exception:
            return "Custom"
        img_size = str(values.get("img_size", "") or "")
        depth = self._sub_volume_depth_from_values(values)
        if sub_volume and batch_size == 1 and grad_accumulation == 2 and img_size == "192,192" and depth == "24":
            return "Low GPU memory"
        if (not sub_volume) and batch_size == 1 and grad_accumulation == 2 and img_size == "256,256":
            return "Higher quality"
        if (not sub_volume) and batch_size == 1 and grad_accumulation == 1 and img_size == "224,224":
            return "Balanced"
        return "Custom"

    def _widget_value(self, key):
        widget = self.widgets.get(key)
        if widget is None:
            return self.values.get(key)
        if isinstance(widget, self.QtWidgets.QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, self.QtWidgets.QSpinBox):
            return int(widget.value())
        if isinstance(widget, self.QtWidgets.QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, self.QtWidgets.QComboBox):
            if key in (
                "decoder",
                "label_source",
                "encoder_backend",
                "training_dimension",
                "quality_mode",
                "modality",
            ):
                return str(widget.currentData() or widget.currentText())
            if key == "strategy" or key in STRATEGY_DATA_KEYS:
                return str(widget.currentData() or "adaptive")
            return str(widget.currentText())
        if isinstance(widget, self.QtWidgets.QLineEdit):
            return str(widget.text()).strip()
        return self.values.get(key)

    def _set_widget_value(self, key, value):
        widget = self.widgets.get(key)
        if widget is None:
            self.values[key] = value
            return
        if isinstance(widget, self.QtWidgets.QCheckBox):
            widget.setChecked(_bool(value))
        elif isinstance(widget, self.QtWidgets.QSpinBox):
            try:
                widget.setValue(int(value))
            except Exception:
                pass
        elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
            try:
                widget.setValue(float(value))
            except Exception:
                pass
        elif isinstance(widget, self.QtWidgets.QComboBox):
            if key in (
                "strategy",
                "decoder",
                "label_source",
                "encoder_backend",
                "training_dimension",
                "quality_mode",
                "modality",
            ) or key in STRATEGY_DATA_KEYS:
                index = widget.findData(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                self.values[key] = value
                return
            text = str(value)
            if widget.findText(text) < 0:
                widget.addItem(text)
            widget.setCurrentText(text)
        elif isinstance(widget, self.QtWidgets.QLineEdit):
            widget.setText(str(value))
        self.values[key] = value

    def _set_combo_value(self, widget, value):
        text = str(value)
        if widget.findText(text) < 0:
            widget.addItem(text)
        widget.setCurrentText(text)

    def _current_values(self):
        values = dict(self.values)
        if self.mcs_folder_edit is not None:
            values["mcs_output_dir"] = str(self.mcs_folder_edit.text()).strip()
        if self.mask_names_edit is not None:
            values["mask_names"] = str(self.mask_names_edit.text()).strip()
        if self.label_root_edit is not None:
            values["label_root"] = str(self.label_root_edit.text()).strip()
        for key in list(self.widgets.keys()):
            if key == "sub_volume_depth":
                continue
            values[key] = self._widget_value(key)
        depth = self._widget_value("sub_volume_depth") or self._sub_volume_depth_from_values(values)
        img_size = str(values.get("img_size", "224,224"))
        parts = [part.strip() for part in img_size.split(",")]
        if len(parts) == 2:
            values["sub_volume_size"] = "{0},{1},{2}".format(depth, parts[0], parts[1])
        return values

    def _label_for_widget(self, key):
        """Return the QLabel paired with a form widget, if any."""
        widget = self.widgets.get(key)
        if widget is None:
            return None
        parent = widget.parentWidget()
        if parent is None:
            return None
        for child in parent.findChildren(self.QtWidgets.QLabel):
            buddy = child.buddy()
            if buddy is widget:
                return child
        return None

    def _sync_image_size_choice(self):
        choice = self.quick_widgets["img_size_choice"].currentText()
        if choice == self.IMG_SIZE_CUSTOM_LABEL:
            self._set_custom_size_entry_state()
            self.widgets["img_size"].setFocus()
            self._refresh_quick_labels()
            return
        else:
            value = _choice_value(choice, self.IMG_SIZE_CHOICES)
        if value:
            self.widgets["img_size"].setText(str(value))
        self._set_custom_size_entry_state()
        self._refresh_quick_labels()

    def _sync_custom_image_size(self):
        choice = self.quick_widgets.get("img_size_choice")
        if choice is not None:
            label = self._img_size_choice_from_value(self.widgets["img_size"].text().strip())
            if choice.currentText() != label:
                self._set_combo_value(choice, label)
        self._refresh_quick_labels()

    def _set_custom_size_entry_state(self):
        widget = self.widgets.get("img_size")
        choice = self.quick_widgets.get("img_size_choice")
        if widget is not None and choice is not None:
            fixed_default_encoder = (
                _bool(self.values.get("preserve_legacy_decoder", False))
                and str(self.values.get("decoder") or "") == "feature_unet2d"
                and str(self.values.get("encoder_backend") or "onnx") == "onnx"
                and not str(self.values.get("model_path", "") or "").strip()
            )
            choice.setEnabled(not fixed_default_encoder)
            widget.setEnabled(not fixed_default_encoder)

    @staticmethod
    def _set_combo_item_enabled(widget, text, enabled):
        if widget is None:
            return
        index = widget.findText(text)
        model = widget.model()
        item = (
            model.item(index)
            if index >= 0 and hasattr(model, "item")
            else None
        )
        if item is not None:
            item.setEnabled(bool(enabled))

    def _refresh_method_enabled(self):
        cached_slices = bool(
            _bool(self.values.get("preserve_legacy_decoder", False))
            and str(self.values.get("decoder") or "") == "feature_unet2d"
        )
        method_widget = self.widgets.get("finetune_method")
        if cached_slices and method_widget is not None and method_widget.currentText() != "frozen":
            self._set_widget_value("finetune_method", "frozen")
        method = str(method_widget.currentText() if method_widget is not None else "").lower()
        lora_enabled = method == "lora"
        for key in ("lora_rank", "lora_alpha"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(lora_enabled and not cached_slices)
        dimension = str(
            self._widget_value("training_dimension") or "auto"
        )
        quality = self.widgets.get("quality_mode")
        if quality is not None:
            quality.setEnabled(not cached_slices and dimension != "2d")
        sub_volume = self.widgets.get("sub_volume")
        depth = self.widgets.get("sub_volume_depth")
        if sub_volume is not None and depth is not None:
            batch_size = int(self._widget_value("batch_size") or 0)
            sub_volume_allowed = (
                not cached_slices
                and dimension != "2d"
                and batch_size <= 1
            )
            sub_volume.setEnabled(sub_volume_allowed)
            if not sub_volume_allowed and sub_volume.isChecked():
                sub_volume.setChecked(False)
            sub_volume.setToolTip(
                "Sub-volume training is disabled for explicit batch sizes "
                "above 1; use gradient accumulation instead."
                if batch_size > 1
                else "Split long 3D cases into depth chunks to reduce memory."
            )
            depth.setEnabled(
                sub_volume_allowed
                and bool(sub_volume.isChecked())
            )
        scheduler = self.widgets.get("lr_scheduler")
        warmup = self.widgets.get("warmup_epochs")
        if scheduler is not None and warmup is not None:
            self._set_combo_item_enabled(
                scheduler,
                "constant_warmup",
                not cached_slices,
            )
            if cached_slices and str(scheduler.currentText()) == "constant_warmup":
                self._set_widget_value("lr_scheduler", "cosine")
            warmup.setEnabled(
                not cached_slices
                and str(scheduler.currentText()) in ("cosine", "constant_warmup")
            )
        if cached_slices:
            self._select_model_scale("vits16")
            strategy_widget = self.widgets.get("strategy")
            if strategy_widget is not None:
                blocked = strategy_widget.blockSignals(True)
                try:
                    self._set_widget_value("strategy", "full_volume")
                finally:
                    strategy_widget.blockSignals(blocked)
                self._update_strategy_summary()
            if (
                str(self.values.get("encoder_backend") or "onnx") == "onnx"
                and not str(self.values.get("model_path", "") or "").strip()
            ):
                self.widgets["img_size"].setText("256,256")
                self._set_combo_value(
                    self.quick_widgets["img_size_choice"],
                    self._img_size_choice_from_value("256,256"),
                )
                self.values["model_sha256"] = str(
                    self.config.get("default_model_sha256", "") or ""
                )
            for key, value in (
                ("model_scale", "vits16"),
                ("sampling_mode", "full"),
                ("channel_policy", "repeat"),
                ("slice_axis", "axial"),
                ("grad_accumulation", 1),
            ):
                self._set_widget_value(key, value)
            self.widgets["mixed_precision"].setChecked(False)
            self.widgets["sub_volume"].setChecked(False)
            current_loss = self._widget_value("loss_type")
            if current_loss in ("auto", "dice_ce", "dice_focal"):
                self._set_widget_value("loss_type", "ce")
        else:
            self.values["encoder_backend"] = "pytorch"
            if (
                str(self._widget_value("model_scale") or "").lower() == "vits16"
                and not str(self.values.get("model_path", "") or "").strip()
            ):
                self.values["model_sha256"] = str(
                    self.config.get("default_safetensors_sha256", "") or ""
                )
        self._set_custom_size_entry_state()
        for key in (
            "finetune_method",
            "model_selection",
            "channel_policy",
            "slice_axis",
            "grad_accumulation",
            "mixed_precision",
        ):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(not cached_slices)
        if hasattr(self, "resource_hint"):
            self.resource_hint.setText(
                "Slices are shuffled across cases and trained in real batches; "
                "the frozen encoder is computed once."
                if cached_slices
                else "Batch size 0 is Auto. Larger volume batches are padded "
                "to the deepest case and padded labels are excluded from loss."
            )

    def _architecture_selection_changed(self, *_args):
        # Selecting a family is an explicit migration away from a legacy
        # compatibility profile. Existing model manifests remain untouched.
        self.values["preserve_legacy_decoder"] = False
        self.values["decoder"] = "auto"
        self._refresh_method_enabled()
        self._refresh_model_source_enabled()
        self._refresh_strategy_enabled()
        self._update_dimensionality_summary()

    def _refresh_model_source_enabled(self, *_args):
        record = self._selected_model_record()
        if record is not None:
            self.values["model_path"] = record["path"]
            self.values["model_scale"] = record["scale"]

    def _update_strategy_summary(self):
        widget = self.widgets.get("strategy")
        if widget is None or not hasattr(self, "strategy_summary"):
            return
        strategy_id = str(widget.currentData() or "adaptive")
        self.strategy_summary.setText(strategy_summary(strategy_id))

    def _apply_strategy_defaults(self, _text=None):
        widget = self.widgets.get("strategy")
        if widget is None:
            return
        strategy_id = str(widget.currentData() or "adaptive")
        self._applying_strategy = True
        try:
            self.values["strategy"] = strategy_id
            for key, value in strategy_defaults(strategy_id).items():
                self._set_widget_value(key, value)
            self._update_strategy_summary()
            self._refresh_method_enabled()
            self._refresh_strategy_enabled()
            self._set_status("Preset applied: {0}".format(strategy_label(strategy_id)))
        finally:
            self._applying_strategy = False

    def _strategy_option_changed(self, *_args):
        if self._applying_strategy:
            return
        self.strategy_summary.setText(strategy_summary(str(self.widgets["strategy"].currentData())) + " Manual overrides are applied below.")
        self._refresh_strategy_enabled()

    def _refresh_strategy_enabled(self):
        cached_slices = bool(
            _bool(self.values.get("preserve_legacy_decoder", False))
            and str(self.values.get("decoder") or "") == "feature_unet2d"
        )
        strategy_widget = self.widgets.get("strategy")
        if strategy_widget is not None:
            strategy_widget.setEnabled(not cached_slices)
        sampling = str(self._widget_value("sampling_mode") or "full")
        patch_enabled = sampling in ("adaptive", "patch")
        for key in ("patch_size_mode", "patch_focus", "patches_per_case"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(patch_enabled)
        custom_patch = patch_enabled and self._widget_value("patch_size_mode") == "custom"
        self.widgets["patch_size_zyx"].setEnabled(custom_patch)
        self.widgets["neighbor_distance_mm"].setEnabled(self._widget_value("channel_policy") == "2_5d")
        data_policy_keys = (
            "sampling_mode",
            "patch_size_mode",
            "patch_size_zyx",
            "patch_focus",
            "patches_per_case",
            "channel_policy",
            "slice_axis",
            "neighbor_distance_mm",
        )
        if cached_slices:
            self.data_policy_note.setText(
                "Frozen Feature 2D uses a fixed native-slice policy. The values "
                "below are shown for clarity and are locked to compatible settings."
            )
            for key in data_policy_keys:
                widget = self.widgets.get(key)
                if widget is None:
                    continue
                widget.setEnabled(False)
        else:
            self.data_policy_note.setText(
                "Sampling and slice context are applied consistently during "
                "training and prediction."
            )
            for key in data_policy_keys:
                widget = self.widgets.get(key)
                if widget is None:
                    continue
                widget.setVisible(True)
                label = self._label_for_widget(key)
                if label is not None:
                    label.setVisible(True)
        mirror_tta = self.widgets.get("mirror_tta")
        if mirror_tta is not None:
            mirror_tta.setVisible(not cached_slices)
            mirror_tta.setEnabled(not cached_slices)
        self._update_dimensionality_summary()

    def _update_dimensionality_summary(self, *_args):
        if not hasattr(self, "dimensionality_summary"):
            return
        dimension = str(
            self._widget_value("training_dimension") or "auto"
        )
        quality = str(self._widget_value("quality_mode") or "standard")
        sampling = str(self._widget_value("sampling_mode") or "adaptive")
        channels = str(self._widget_value("channel_policy") or "repeat")
        region = "a 3D sub-volume" if sampling == "patch" else ("a fingerprint-selected 3D region" if sampling == "adaptive" else "the full 3D volume")
        context = (
            "spacing-selected slice context"
            if channels == "auto"
            else (
                "neighboring-slice (2.5D) encoder context"
                if channels == "2_5d"
                else "single-slice encoder context"
            )
        )
        if (
            _bool(self.values.get("preserve_legacy_decoder", False))
            and str(self.values.get("decoder") or "") == "feature_unet2d"
        ):
            region = "native-grid axial slices with no 3D resampling"
            context = "one cached frozen DINO feature map plus raw grayscale skips"
            decoding = "independent 2D decoding followed by ordered 3D stacking"
        elif dimension == "2d":
            decoding = "multi-level scale-aware 2D decoding followed by ordered 3D stacking"
        elif dimension == "3d":
            decoding = (
                "high-detail multi-scale 3D context decoding"
                if quality == "high_detail"
                else "lightweight volumetric 3D context decoding"
            )
        else:
            decoding = "fingerprint-selected 2D or 3D decoding"
        self.dimensionality_summary.setText("Uses {0}, {1}, and {2}.".format(region, context, decoding))

    def _sync_quick_settings(self):
        if self._syncing_quick:
            return
        self._syncing_quick = True
        try:
            epoch_value = _choice_value(self.quick_widgets["epochs_choice"].currentText(), self.EPOCH_CHOICES)
            if epoch_value is not None:
                self._set_widget_value("epochs", epoch_value)
            val_value = _choice_value(self.quick_widgets["val_fraction_choice"].currentText(), self.VAL_CHOICES)
            if val_value is not None:
                self._set_widget_value("val_fraction", val_value)
            mode = _choice_value(self.quick_widgets["memory_mode"].currentText(), self.MEMORY_CHOICES)
            if mode == "low_memory":
                self._set_widget_value("grad_accumulation", 2)
                self.widgets["img_size"].setText("192,192")
                self.widgets["sub_volume"].setChecked(True)
                self._set_combo_value(self.widgets["sub_volume_depth"], "24")
                self.widgets["mixed_precision"].setChecked(False)
            elif mode == "quality":
                self._set_widget_value("grad_accumulation", 2)
                self.widgets["img_size"].setText("256,256")
                self.widgets["sub_volume"].setChecked(False)
                self._set_combo_value(self.widgets["sub_volume_depth"], "32")
            elif mode == "balanced":
                self._set_widget_value("grad_accumulation", 1)
                self.widgets["img_size"].setText("224,224")
                self.widgets["sub_volume"].setChecked(False)
                self._set_combo_value(self.widgets["sub_volume_depth"], "32")
        finally:
            self._syncing_quick = False
        self._refresh_quick_labels()

    def _refresh_quick_labels(self):
        if self._syncing_quick:
            return
        self._syncing_quick = True
        try:
            values = self._current_values()
            self._set_combo_value(self.quick_widgets["epochs_choice"], _option_label(values.get("epochs", 10), self.EPOCH_CHOICES))
            self._set_combo_value(self.quick_widgets["val_fraction_choice"], _option_label(values.get("val_fraction", 0.2), self.VAL_CHOICES))
            self._set_combo_value(self.quick_widgets["memory_mode"], self._memory_mode_from_values(values))
            img_label = self._img_size_choice_from_value(values.get("img_size", "224,224"))
            if self.quick_widgets["img_size_choice"].currentText() != self.IMG_SIZE_CUSTOM_LABEL:
                self._set_combo_value(self.quick_widgets["img_size_choice"], img_label)
            self._set_custom_size_entry_state()
            self._refresh_method_enabled()
        finally:
            self._syncing_quick = False

    def apply_profile(self):
        profile_widget = self.quick_widgets.get("profile")
        profile_name = profile_widget.currentText() if profile_widget is not None else ""
        values = default_training_options(self.config, profile_name)
        for key, value in values.items():
            self._set_widget_value(key, value)
        self._set_combo_value(self.widgets["sub_volume_depth"], self._sub_volume_depth_from_values(values))
        self._refresh_quick_labels()
        self._set_status("Applied profile: {0}".format(profile_name or "default"))
        self._append_log("Applied profile: {0}".format(profile_name or "default"))

    def collect_options(self):
        if self._dataset_scan_pending:
            raise RuntimeError("Wait for the dataset check to finish.")
        selected_root = str(self.dataset_edit.text() if self.dataset_edit is not None else "").strip()
        selected_root = os.path.abspath(
            os.path.expanduser(os.path.expandvars(selected_root))
        ) if selected_root else ""
        active_root = str(self.context.get("ts_root") or "").strip()
        if not selected_root:
            raise RuntimeError("Choose an existing source image root.")
        if os.path.normcase(selected_root) != os.path.normcase(active_root):
            raise RuntimeError(
                "The dataset path changed. Press Enter or leave the field, then wait for its check to finish."
            )
        if self._active_dataset_scan_signature != self._current_data_signature():
            raise RuntimeError(
                "The data source, path, or Target Mask names changed. Scan Data "
                "again before starting training."
            )
        self._sync_image_size_choice()
        options = self._current_values()
        selected_model = self._selected_model_record()
        if selected_model is None:
            raise RuntimeError(
                "No compatible DINOv3 ViT weights are installed. Add one complete "
                "Hugging Face model folder under external/dinov3-medical-seg/models "
                "and reopen this window."
            )
        options["model_path"] = selected_model["path"]
        options["model_scale"] = selected_model["scale"]
        options["model_sha256"] = ""
        options.pop("model_selection", None)
        selected = []
        choose_specific = bool(
            self.choose_specific_cases
            and self.choose_specific_cases.isChecked()
        )
        if choose_specific and self.case_list is not None:
            for item in self.case_list.selectedItems():
                selected.append(
                    str(item.data(self.QtCore.Qt.UserRole) or item.text())
                )
        elif not choose_specific:
            selected = [
                str(row.get("case_id"))
                for row in self.case_rows
                if row.get("state") == "ready"
            ]
        manual = (
            split_csv(self.manual_cases.text())
            if choose_specific and self.manual_cases is not None
            else []
        )
        cases = selected + [case for case in manual if case not in selected]
        if not cases:
            raise RuntimeError(
                "No usable case is available. Check the paths, Target Mask names, "
                "and scan the data again."
            )
        if cases:
            options["cases"] = cases
            options["sample_mode"] = "all"
        else:
            options.pop("cases", None)
        if self.remote_selector is not None:
            backend, profile_id = self.remote_selector.selection()
            options["execution_backend"] = backend
            options["remote_profile_id"] = profile_id
        return validate_options(options)

    def start_training(self):
        if self.started:
            return
        try:
            options = self.collect_options()
        except Exception as exc:
            self._set_status("Cannot start training: {0}".format(exc))
            self._append_log("Cannot start training: {0}".format(exc))
            return
        try:
            save_preferences("dinov3_training", {
                "dataset_root": self.context.get("ts_root") or "",
                "mcs_dir": options.get("mcs_output_dir") or self.context.get("mcs_output_dir") or "",
                "label_root": options.get("label_root") or "",
            })
        except Exception:
            pass
        try:
            run_id, status_path, pid = launch_training(self.context, options)
        except Exception as exc:
            self._write_setup_failure(exc)
            self._set_status("Could not start training: {0}".format(exc))
            self._append_log("Could not start training: {0}".format(exc))
            return
        self.started = True
        self.training_status_path = status_path
        self.training_log_dir = os.path.dirname(os.path.dirname(status_path))
        self.start_button.setEnabled(False)
        self.open_log_button.setEnabled(True)
        self.close_button.setText("Close")
        self._set_status("Training started: {0} (PID {1})".format(run_id, pid))
        self._append_log("Training started in the background: {0} (PID {1})".format(run_id, pid))
        self._append_log("Status file: {0}".format(status_path))
        self.QtCore.QTimer.singleShot(500, self.poll_training_status)

    def poll_training_status(self):
        if self._closing or not self.training_status_path:
            return
        status = read_json(self.training_status_path, {}) or {}
        line = format_status_line(status)
        if line and line != self.last_status_line:
            self.last_status_line = line
            self._set_status(line)
            self._append_log(line)
        if status.get("train_log"):
            self.training_log_dir = os.path.dirname(status.get("train_log"))
        if status.get("status") in ("completed", "failed", "cancelled", "abandoned"):
            return
        self.QtCore.QTimer.singleShot(1500, self.poll_training_status)

    def _append_log(self, text):
        if self.status_text is None:
            return
        stamp = time.strftime("%H:%M:%S")
        self.status_text.append("[{0}] {1}".format(stamp, text))

    def _set_status(self, text):
        if self.status_label is not None:
            self.status_label.setText(str(text))

    def open_log_folder(self):
        folder = self.training_log_dir or os.path.join(self.context.get("workspace", ""), "jobs")
        if not folder or not os.path.isdir(folder):
            self._append_log("Log folder is not available yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            self._append_log("Could not open log folder: {0}".format(exc))

    def _write_setup_failure(self, exc):
        path = self.context.get("setup_status_path")
        if not path:
            return
        write_json_best_effort(path, {
            "schema_version": "mimics_fewshot_setup.v1",
            "job_id": self.context.get("setup_id"),
            "kind": "train_setup",
            "status": "failed",
            "organ": self.context.get("organ"),
            "ts_root": str(self.context.get("ts_root") or ""),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "updated_at_epoch": time.time(),
        })

    def mark_closed_if_needed(self):
        if self._closing:
            return
        self._closing = True
        try:
            self._dataset_scan_timer.stop()
        except Exception:
            pass
        if self.started:
            return
        path = self.context.get("setup_status_path")
        if path:
            payload = read_json(path, {}) or {}
            payload.update({
                "status": "closed",
                "updated_at_epoch": time.time(),
            })
            write_json_best_effort(path, payload)

    def close(self):
        self.mark_closed_if_needed()
        self.window.close()


def QtGuiShortcut(QtGui, key, parent):
    return QtGui.QShortcut(QtGui.QKeySequence(key), parent)


def run_pyside6_ui(context):
    qt_modules = _load_pyside6()
    QtCore, _QtGui, QtWidgets = qt_modules

    class CloseAwareMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            QtWidgets.QMainWindow.__init__(self)
            self.controller = None

        def closeEvent(self, event):
            if self.controller is not None:
                self.controller.mark_closed_if_needed()
            QtWidgets.QMainWindow.closeEvent(self, event)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, TITLE)
    window = CloseAwareMainWindow()
    controller = QtTrainingSetupApp(window, context, qt_modules)
    window.controller = controller
    window.show()
    return app.exec()


def run_ui(context):
    """Open the external training setup window (PySide6 only).

    The Tkinter fallback backend was removed: PySide6 ships with the bundled
    python_env on every machine. If PySide6 cannot start, the failure is
    recorded in the setup status JSON so the annotator sees why no window
    appeared instead of a raw traceback.
    """
    try:
        return run_pyside6_ui(context)
    except Exception as exc:
        message = (
            "The external DINOv3 training setup window could not open because PySide6 "
            "is unavailable in the configured external Python environment. Training was not "
            "started. Re-run Admin > Setup/Repair Environment to restore the bundled UI "
            "packages, or use the default Train/Update Model entry."
        )
        setup_path = context.get("setup_status_path")
        if setup_path:
            write_json_best_effort(setup_path, {
                "schema_version": "mimics_fewshot_setup.v1",
                "job_id": context.get("setup_id"),
                "kind": "train_setup",
                "status": "failed",
                "organ": context.get("organ"),
                "ts_root": os.path.abspath(context.get("ts_root", "")),
                "error": message,
                "details": "PySide6: {0}".format(exc),
                "updated_at_epoch": time.time(),
            })
        raise RuntimeError("{0} (PySide6: {1})".format(message, exc))


def generate_preview(path, tab="setup"):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required only for --preview PNG generation. Runtime DINOv3 setup UI does not "
            "depend on Pillow. Install pillow in the preview-generation Python environment or skip --preview. "
            "Import error: {0}".format(exc)
        )

    width, height = 1280, 960
    image = Image.new("RGB", (width, height), "#f4f5f7")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 30)
        head_font = ImageFont.truetype("Arial.ttf", 19)
        font = ImageFont.truetype("Arial.ttf", 17)
        small = ImageFont.truetype("Arial.ttf", 15)
    except Exception:
        title_font = head_font = font = small = ImageFont.load_default()

    draw.rectangle((0, 0, width, 72), fill="#1f2937")
    draw.text((32, 20), "DINOv3 Few-Shot Training", fill="white", font=title_font)
    draw.text((32, 92), "Organ: liver    Dataset: D:\\Dataset\\TotalSegmentator", fill="#1f2937", font=head_font)
    draw.text((32, 126), "Save edited .mcs projects before starting. Training exports labels from saved projects in the background.", fill="#9a5b00", font=font)

    draw.rectangle((30, 166, width - 30, height - 180), outline="#d1d5db", width=2, fill="#ffffff")
    draw.rectangle((30, 166, 300, 212), fill="#e8eef7" if tab == "setup" else "#ffffff", outline="#d1d5db")
    draw.text((60, 180), "Data and samples", fill="#111827", font=head_font)
    draw.rectangle((300, 166, 570, 212), fill="#e8eef7" if tab == "expert" else "#ffffff", outline="#d1d5db")
    draw.text((330, 180), "Model and policy", fill="#111827", font=head_font)

    def draw_field(x, y, label, value, kind="select", w=230):
        draw.text((x, y), label, fill="#374151", font=small)
        draw.rectangle((x, y + 22, x + w, y + 54), outline="#9ca3af", fill="#ffffff")
        draw.text((x + 10, y + 29), value, fill="#111827", font=small)
        if kind == "select":
            draw.polygon([(x + w - 20, y + 34), (x + w - 10, y + 34), (x + w - 15, y + 42)], fill="#4b5563")
        elif kind == "spin":
            draw.line((x + w - 30, y + 22, x + w - 30, y + 54), fill="#d1d5db")
            draw.polygon([(x + w - 20, y + 31), (x + w - 10, y + 31), (x + w - 15, y + 25)], fill="#4b5563")
            draw.polygon([(x + w - 20, y + 44), (x + w - 10, y + 44), (x + w - 15, y + 50)], fill="#4b5563")

    def draw_row(x, y, label, value, kind="select", w=300, label_w=168):
        draw.text((x, y + 8), label, fill="#374151", font=small)
        field_x = x + label_w
        disabled = kind == "disabled"
        draw.rectangle(
            (field_x, y, field_x + w, y + 30),
            outline="#d1d5db" if disabled else "#9ca3af",
            fill="#f3f4f6" if disabled else "#ffffff",
        )
        draw.text(
            (field_x + 10, y + 7),
            value,
            fill="#9ca3af" if disabled else "#111827",
            font=small,
        )
        if kind == "select":
            draw.polygon([(field_x + w - 20, y + 11), (field_x + w - 10, y + 11), (field_x + w - 15, y + 19)], fill="#4b5563")
        elif kind == "spin":
            draw.line((field_x + w - 30, y, field_x + w - 30, y + 30), fill="#d1d5db")
            draw.polygon([(field_x + w - 20, y + 9), (field_x + w - 10, y + 9), (field_x + w - 15, y + 3)], fill="#4b5563")
            draw.polygon([(field_x + w - 20, y + 21), (field_x + w - 10, y + 21), (field_x + w - 15, y + 27)], fill="#4b5563")

    if tab == "expert":
        draw.text((60, 240), "Training strategy and model policy", fill="#111827", font=head_font)
        draw.text((365, 244), "Architecture families keep incompatible combinations out of the workflow.", fill="#6b7280", font=small)
        left_x, right_x = 60, 660
        card_y, card_bottom = 280, 780
        card_w = 560
        draw.rectangle((left_x, card_y, left_x + card_w, card_bottom), outline="#d1d5db", fill="#f9fafb")
        draw.text((left_x + 22, card_y + 22), "Model and task", fill="#111827", font=head_font)
        draw.text((left_x + 22, card_y + 54), "Public choices resolve to one compatible model plan.", fill="#6b7280", font=small)
        model_rows = [
            ("Training dimension", "Automatic from data", "select"),
            ("3D quality", "Standard", "select"),
            ("Fine-tuning", "frozen", "select"),
            ("Pretrained weights", "DINOv3 ViT-S/16", "select"),
            ("LoRA rank", "8", "disabled"),
            ("LoRA alpha", "16", "disabled"),
        ]
        y = card_y + 88
        for label, value, kind in model_rows:
            draw_row(left_x + 22, y, label, value, kind, w=330, label_w=170)
            y += 36
        draw.text((left_x + 22, y + 12), "Auto resolves 2D or 3D from the training-data fingerprint;", fill="#6b7280", font=small)
        draw.text((left_x + 22, y + 34), "existing ONNX compatibility models remain loadable.", fill="#6b7280", font=small)

        draw.rectangle((right_x, card_y, right_x + card_w, card_bottom), outline="#d1d5db", fill="#f9fafb")
        draw.text((right_x + 22, card_y + 22), "Training and resources", fill="#111827", font=head_font)
        draw.text((right_x + 22, card_y + 54), "Runtime, memory, validation, and retention.", fill="#6b7280", font=small)
        opt_rows = [
            ("Epochs", "20", "spin"),
            ("Batch size", "Auto", "spin"),
            ("GPU memory budget", "Auto detect", "spin"),
            ("Grad accumulation", "1", "spin"),
            ("Learning rate", "0.001", "select"),
            ("Weight decay", "0.0001", "select"),
            ("LR schedule", "cosine", "select"),
            ("Warmup epochs", "0", "spin"),
            ("Validation interval", "5", "spin"),
            ("Image detail", "Detailed (256 x 256)", "select"),
            ("Custom size", "256,256", "disabled"),
        ]
        y = card_y + 88
        for label, value, kind in opt_rows:
            draw_row(right_x + 22, y, label, value, kind, w=250, label_w=172)
            y += 36
        y += 10
        check_x = right_x + 22
        for label, checked in (("Mixed precision", False),):
            draw.rectangle((check_x, y, check_x + 16, y + 16), outline="#d1d5db", fill="#f3f4f6")
            if checked:
                draw.line((check_x + 3, y + 8, check_x + 8, y + 13), fill="#2563eb", width=2)
                draw.line((check_x + 8, y + 13, check_x + 16, y + 3), fill="#2563eb", width=2)
            draw.text((check_x + 28, y - 2), label, fill="#9ca3af", font=small)
            check_x += 190
    else:
        draw.text((60, 236), "Training data", fill="#111827", font=head_font)
        draw.rectangle((60, 270, width - 60, 490), outline="#d1d5db", fill="#f9fafb")
        dataset_rows = [
            ("Label source", "Saved Mimics projects (.mcs)", "select"),
            ("Original image dataset *", "D:\\Dataset\\TotalSegmentator", "text"),
            ("Saved .mcs folder *", "D:\\Dataset\\TotalSegmentator\\mcs_output", "text"),
            ("Target Mask name(s) *", "liver, liver_seg", "text"),
        ]
        y = 282
        for label, value, kind in dataset_rows:
            draw_row(82, y, label, value, kind, w=760, label_w=210)
            if "dataset" in label.lower() or ".mcs" in label:
                draw.rectangle((1040, y, 1145, y + 30), outline="#9ca3af", fill="#ffffff")
                draw.text((1066, y + 7), "Browse", fill="#111827", font=small)
            y += 36
        draw.text(
            (292, 430),
            "Projects without a matching saved Mask are checked and skipped automatically.",
            fill="#667085",
            font=small,
        )
        draw.rectangle((1040, 426, 1145, 458), outline="#9ca3af", fill="#ffffff")
        draw.text((1060, 434), "Scan Data", fill="#111827", font=small)

        draw.text((60, 520), "Cases", fill="#111827", font=head_font)
        draw.rectangle((60, 554, width - 60, 700), outline="#d1d5db", fill="#f9fafb")
        draw.rectangle((84, 578, 100, 594), outline="#9ca3af", fill="#ffffff")
        draw.text(
            (116, 574),
            "Choose specific cases instead of using every matching Mask",
            fill="#344054",
            font=font,
        )
        draw.text(
            (84, 616),
            "12 usable cases found; 604 unavailable cases hidden.",
            fill="#667085",
            font=small,
        )
        draw.text((84, 660), "Sample order", fill="#374151", font=small)
        draw.rectangle((190, 652, 300, 684), outline="#9ca3af", fill="#ffffff")
        draw.text((202, 660), "all", fill="#111827", font=small)
        draw.text((340, 660), "Max samples", fill="#374151", font=small)
        draw.rectangle((444, 652, 544, 684), outline="#9ca3af", fill="#ffffff")
        draw.text((456, 660), "0", fill="#111827", font=small)
        draw.text((610, 660), "Min train  1    Min val  1", fill="#374151", font=small)

    status_top = height - 155
    draw.rectangle((30, status_top, width - 30, height - 75), outline="#d1d5db", fill="#ffffff")
    draw.text((50, status_top + 8), "Status", fill="#111827", font=head_font)
    log_lines = [
        "[14:20:03] Ready. Choose data, samples, and model policy, then start background training.",
        "[14:20:29] Training progress will appear here after Start Training.",
    ]
    y = status_top + 35
    for line in log_lines:
        draw.text((50, y), line, fill="#374151", font=small)
        y += 22

    draw.text((32, height - 40), "Training progress stays visible here; Mimics remains usable.", fill="#374151", font=font)
    draw.rectangle((width - 450, height - 55, width - 300, height - 14), fill="#ffffff", outline="#9ca3af")
    draw.text((width - 424, height - 44), "Open Log Folder", fill="#111827", font=font)
    draw.rectangle((width - 285, height - 55, width - 175, height - 14), fill="#ffffff", outline="#9ca3af")
    draw.text((width - 248, height - 44), "Cancel", fill="#111827", font=font)
    draw.rectangle((width - 160, height - 55, width - 32, height - 14), fill="#2563eb", outline="#1d4ed8")
    draw.text((width - 145, height - 44), "Start Training", fill="#ffffff", font=font)

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", help="Path to the setup context JSON written by Mimics.")
    parser.add_argument("--preview", help="Write a static PNG preview of the UI and exit.")
    parser.add_argument("--preview-tab", choices=("setup", "expert"), default="setup")
    args = parser.parse_args(argv)
    if args.preview:
        generate_preview(args.preview, tab=args.preview_tab)
        print(args.preview)
        return 0
    if not args.context:
        parser.error("--context is required unless --preview is used")
    context = read_json(args.context, None)
    if not context:
        raise RuntimeError("Could not read setup context: {0}".format(args.context))
    run_ui(context)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
