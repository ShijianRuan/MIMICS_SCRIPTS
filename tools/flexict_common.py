#!/usr/bin/env python3
"""Shared contracts for the FlexiCT few-shot Mimics integration.

Everything FlexiCT-specific that the pipeline, UIs, and runtime entry need:
config loading (flexict_config.json), workspace paths, the independent model
registry (flexict_models/registry.json, schema flexict_model_registry.v1,
dataset-id range 750-799 kept apart from nnU-Net's 701+), pretrained-weight
discovery, and request normalization.

The flexict-finetune repo under integrations/ is deliberately standalone (no
Mimics knowledge); this module is the host-side adapter and must stay the
only place that knows about both worlds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "flexict_config.json"

SCHEMA_VERSION = "flexict_model_registry.v1"
MODEL_SCHEMA_VERSION = "flexict_model.v1"
TERMINAL_STATES = {"completed", "failed", "cancelled", "abandoned"}
JOB_SCHEMA_VERSION = "flexict_job.v1"

DATASET_ID_FIRST = 750
DATASET_ID_LAST = 799
CONFIGURATIONS = ("2d", "3d_fullres", "pair", "auto")

DEFAULT_CONFIG: dict[str, Any] = {
    "workspace_dir": "flexict_models",
    "flexict_dir": "integrations/flexict-finetune",
    "pretrained_weights_dir": "",
    "default_configuration": "auto",
    "default_epochs": 150,
    "default_mirror_disable_axes": "",
    "default_val_cases": 3,
    "dataset_id_first": 750,
    "default_uncertainty_method": "disagreement",
    "gpu_lock_timeout_seconds": 86400,
    "label_export_timeout_seconds": 7200,
    "job_retention_days": 30,
    "status_poll_seconds": 1.0,
}


def load_config(explicit_path: str | Path | None = None) -> dict[str, Any]:
    """Merge flexict_config.json over defaults; unknown keys are kept."""
    config = dict(DEFAULT_CONFIG)
    source = Path(explicit_path) if explicit_path else CONFIG_PATH
    payload = None
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = None
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key, value in payload.items():
            config[str(key)] = value
    return config


def workspace_root(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    workspace_dir = str(config.get("workspace_dir") or "").strip()
    if not workspace_dir:
        workspace_dir = str(DEFAULT_CONFIG["workspace_dir"])
    candidate = Path(workspace_dir)
    if not candidate.is_absolute():
        candidate = ROOT / workspace_dir
    return candidate.resolve()


def workspace_paths(workspace: str | Path | None = None,
                    config: dict[str, Any] | None = None) -> dict[str, Path]:
    root = (
        Path(workspace).expanduser().resolve()
        if workspace
        else workspace_root(config)
    )
    return {
        "root": root,
        "jobs": root / "jobs",
        "raw": root / "runtime" / "nnUNet_raw",
        "preprocessed": root / "runtime" / "nnUNet_preprocessed",
        "results": root / "runtime" / "nnUNet_results",
        "registry": root / "registry.json",
    }


def flexict_repo_dir(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    flexict_dir = str(config.get("flexict_dir") or "").strip()
    if not flexict_dir:
        flexict_dir = str(DEFAULT_CONFIG["flexict_dir"])
    candidate = Path(flexict_dir)
    if not candidate.is_absolute():
        candidate = ROOT / flexict_dir
    return candidate.resolve()


def resolve_pretrained_dir(config: dict[str, Any] | None = None) -> tuple[Path, str]:
    """Locate the pretrained FlexiCT backbone weights.

    Order: config pretrained_weights_dir (absolute override) > the repo's own
    weights/ > the read-only research share R:\\flexict-finetune (legacy
    fallback). Returns (path, source) where source is one of
    "config"/"repo"/"fallback" so callers can warn when using the fallback.
    """
    config = config or load_config()
    override = str(config.get("pretrained_weights_dir") or "").strip()
    if override:
        candidate = Path(override)
        if (candidate / "flexict_2d" / "model.safetensors").is_file() and (
            candidate / "flexict_3d" / "model.safetensors"
        ).is_file():
            return candidate.resolve(), "config"
        raise FileNotFoundError(
            "pretrained_weights_dir is set but incomplete (needs "
            "flexict_2d/model.safetensors and flexict_3d/model.safetensors "
            "under it): {}".format(candidate)
        )
    repo = flexict_repo_dir(config)
    if (repo / "weights" / "flexict_2d" / "model.safetensors").is_file() and (
        repo / "weights" / "flexict_3d" / "model.safetensors"
    ).is_file():
        return repo / "weights", "repo"
    fallback = Path("R:/flexict-finetune/weights")
    if (fallback / "flexict_2d" / "model.safetensors").is_file() and (
        fallback / "flexict_3d" / "model.safetensors"
    ).is_file():
        return fallback, "fallback"
    raise FileNotFoundError(
        "FlexiCT pretrained weights not found: set pretrained_weights_dir in "
        "flexict_config.json, or place flexict_2d/flexict_3d under "
        "{}/weights/.".format(repo)
    )


def _dataset_ids_in(paths: dict[str, Path], used: set[int]) -> None:
    for key in ("raw", "preprocessed", "results"):
        base = paths[key]
        if not base.is_dir():
            continue
        for candidate in base.glob("Dataset[0-9][0-9][0-9]_*"):
            try:
                used.add(int(candidate.name[7:10]))
            except (TypeError, ValueError):
                continue


def flexict_suggest_dataset_id(
    workspace: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    """Pick an unused dataset id inside the FlexiCT 750-799 band."""
    paths = workspace_paths(workspace, config)
    used: set[int] = set()
    _dataset_ids_in(paths, used)
    for model in load_models(workspace or paths["root"], config):
        try:
            dataset_id = int(model.get("dataset_id"))
            used.add(dataset_id)
        except (TypeError, ValueError):
            continue
    first = int(config.get("dataset_id_first") or DATASET_ID_FIRST) \
        if config else DATASET_ID_FIRST
    first = min(max(first, DATASET_ID_FIRST), DATASET_ID_LAST)
    for value in range(first, DATASET_ID_LAST + 1):
        if value not in used:
            return value
    raise RuntimeError(
        "All FlexiCT dataset ids from {} to {} are in use.".format(
            DATASET_ID_FIRST, DATASET_ID_LAST
        )
    )


def load_models(workspace: str | Path | None = None,
                config: dict[str, Any] | None = None,
                include_missing: bool = False) -> list[dict[str, Any]]:
    """Load the FlexiCT model registry (flexict_models/registry.json)."""
    paths = workspace_paths(workspace, config)
    payload = None
    try:
        with paths["registry"].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = None
    models: dict[str, dict[str, Any]] = {}
    for row in (payload or {}).get("models") or []:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "")
        if not model_id:
            continue
        model_dir = Path(str(row.get("model_dir") or "")).expanduser()
        if not model_dir.is_dir():
            if include_missing:
                relocated = dict(row)
                relocated["status"] = "missing_path"
                models[model_id] = relocated
            continue
        models[model_id] = dict(row)
    return sorted(
        models.values(),
        key=lambda row: float(row.get("created_at_epoch") or 0),
        reverse=True,
    )


def save_registry(workspace: str | Path | None,
                  config: dict[str, Any] | None,
                  models: list[dict[str, Any]],
                  recommended_model_id: str = "") -> None:
    from mimics_label_export import write_json_atomic  # host helper, same folder

    paths = workspace_paths(workspace, config)
    existing = {}
    try:
        with paths["registry"].open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except Exception:
        existing = {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recommended_model_id": str(
            recommended_model_id
            or (existing or {}).get("recommended_model_id")
            or ""
        ),
        "models": models,
        "updated_at_epoch": time.time(),
    }
    write_json_atomic(paths["registry"], payload)


def register_model(manifest: dict[str, Any],
                   workspace: str | Path | None = None,
                   config: dict[str, Any] | None = None) -> None:
    model = dict(manifest)
    model_id = str(model.get("model_id") or "")
    if not model_id:
        raise ValueError("Model manifest has no model_id.")
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        model["schema_version"] = MODEL_SCHEMA_VERSION
    if not model.get("created_at_epoch"):
        model["created_at_epoch"] = time.time()
    models = [
        row for row in load_models(workspace, config)
        if str(row.get("model_id")) != model_id
    ]
    models.append(model)
    models.sort(
        key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True
    )
    paths = workspace_paths(workspace, config)
    save_registry(paths["root"], config, models[:200],
                  recommended_model_id=str(model.get("model_id")))


def model_usability(model: dict[str, Any]) -> tuple[bool, str]:
    """A registered FlexiCT model is usable when its trainer output folder
    carries plans.json + dataset.json + fold_0/checkpoint_best.pth (fold 0 with
    a real train/val split — checkpoint_best is selected by validation Dice)."""
    model_dir = Path(str(model.get("model_dir") or "")).expanduser()
    if not model_dir.is_dir():
        return False, "model folder is missing"
    if not (model_dir / "plans.json").is_file():
        return False, "plans.json is missing"
    if not (model_dir / "dataset.json").is_file():
        return False, "dataset.json is missing"
    if not (model_dir / "fold_0" / "checkpoint_best.pth").is_file():
        return False, "fold_0/checkpoint_best.pth is missing"
    return True, ""


def recommended_model(workspace: str | Path | None = None,
                      config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The registry's recommended model, if present and usable."""
    paths = workspace_paths(workspace, config)
    payload = {}
    try:
        with paths["registry"].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    model_id = str((payload or {}).get("recommended_model_id") or "")
    if not model_id:
        return None
    for row in load_models(paths["root"], config):
        if str(row.get("model_id")) == model_id:
            usable, _ = model_usability(row)
            return row if usable else None
    return None


def load_pair(workspace: str | Path | None = None,
              config: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (model_2d, model_3d) sharing a pair_id, or (None, None)."""
    models = load_models(workspace, config)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in models:
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            by_pair.setdefault(pair_id, []).append(row)
    for pair_id, rows in by_pair.items():
        model_2d = None
        model_3d = None
        for row in rows:
            configuration = str(row.get("configuration") or "")
            if configuration == "2d" and model_2d is None:
                model_2d = row
            elif configuration == "3d_fullres" and model_3d is None:
                model_3d = row
        if model_2d is not None and model_3d is not None:
            usable_2d, _ = model_usability(model_2d)
            usable_3d, _ = model_usability(model_3d)
            if usable_2d and usable_3d:
                return model_2d, model_3d
    return None, None


def normalize_request(values: dict[str, Any]) -> dict[str, Any]:
    """Validate + fill defaults for a training request from the UI/runtime."""
    out = dict(values)
    label_name = str(out.get("label_name") or "").strip()
    if not label_name:
        raise ValueError("label_name (organ/target name) is required.")
    out["label_name"] = label_name
    out["cases"] = [str(c).strip() for c in (out.get("cases") or [])
                    if str(c).strip()]
    if not out["cases"]:
        raise ValueError("At least one training case is required.")
    configuration = str(out.get("configuration") or "auto").strip().lower()
    if configuration not in CONFIGURATIONS:
        raise ValueError(
            "configuration must be one of {}".format(", ".join(CONFIGURATIONS))
        )
    out["configuration"] = configuration
    try:
        epochs = int(out.get("epochs") or 0)
    except (TypeError, ValueError):
        epochs = 0
    if epochs <= 0:
        epochs = int(DEFAULT_CONFIG["default_epochs"])
    out["epochs"] = epochs
    mirror = str(out.get("mirror_disable_axes") or "").strip()
    out["mirror_disable_axes"] = mirror
    try:
        val_cases = int(out.get("val_cases") or 0)
    except (TypeError, ValueError):
        val_cases = 0
    if val_cases <= 0:
        val_cases = int(DEFAULT_CONFIG["default_val_cases"])
    if val_cases >= len(out["cases"]):
        if len(out["cases"]) <= 1:
            raise ValueError(
                "Need at least 2 cases (1 train + 1 validation)."
            )
        val_cases = max(1, len(out["cases"]) - 1)
    out["val_cases"] = val_cases
    return out
