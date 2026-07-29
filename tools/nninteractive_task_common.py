#!/usr/bin/env python3
"""Shared external-runtime helpers for nnInteractive task models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from runtime_py35 import dataset_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "nninteractive_finetune_config.json"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "paused"}
ACTIVE_STATUSES = {
    "created",
    "validating_cases",
    "exporting_labels",
    "preparing_data",
    "preparing_remote",
    "connecting_remote",
    "uploading",
    "starting_remote",
    "reconnecting_remote",
    "waiting_for_remote_gpu",
    "remote_control_unavailable",
    "finalizing_remote",
    "downloading",
    "waiting_for_gpu",
    "training",
    "validating",
    "registering",
    "pausing",
    "stopping",
}
MODEL_METADATA_FILES = (
    "plans.json",
    "dataset.json",
    "inference_info.json",
    "inference_session_class.json",
)
NNINTERACTIVE_INPUT_CONTRACT = {
    "schema_version": "nninteractive_input_contract.v1",
    "spatial_orientation": "canonical_ras",
    "source_grid_policy": "mimics_dicom_compatible_minimal_resample",
    "intensity_space": "source_physical_values",
    "dicom_rescale": "apply_rescale_slope_and_intercept",
    "normalization": "nonzero_spatial_bbox_zscore",
    "normalization_channel": 0,
    "standard_deviation_correction": 1,
}
DEFAULT_VALIDATED_PROMPT_TYPES = ("point",)
UNUSABLE_MODEL_STATES = {"corrupt", "failed", "incompatible", "cancelled"}


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error: OSError | None = None
    for attempt in range(20):
        temporary = target.with_name(
            "{}.{}.{}.tmp".format(target.name, os.getpid(), uuid.uuid4().hex)
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(target))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(0.25, 0.02 * (attempt + 1)))
    try:
        with target.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    except OSError as exc:
        last_error = exc
    raise OSError("Could not update {}: {}".format(target, last_error))


def append_log(path: str | Path, message: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with target.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("[{}] {}\n".format(stamp, str(message).rstrip()))


def safe_slug(value: object, fallback: str = "task") -> str:
    text = str(value or "").strip().lower()
    output = []
    for character in text:
        if character.isalnum() or character in ("_", "-", "."):
            output.append(character)
        else:
            output.append("_")
    return re.sub(r"_+", "_", "".join(output)).strip("._-") or fallback


def load_config() -> dict[str, Any]:
    payload = read_json(CONFIG_PATH, {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    return payload


def resolve_path(value: object, base: Path = ROOT) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    path = Path(text) if text else base
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def workspace_root(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    configured = (
        os.environ.get("NNINTERACTIVE_TASK_MODELS_DIR", "")
        or config.get("workspace_dir", "")
        or "nninteractive_task_models"
    )
    root = resolve_path(configured)
    root.mkdir(parents=True, exist_ok=True)
    return root


def jobs_dir(workspace: Path) -> Path:
    path = workspace / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_dir(workspace: Path, task_id: str) -> Path:
    path = workspace / "tasks" / safe_slug(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_path(workspace: Path) -> Path:
    return workspace / "registry.json"


def relative_model_path(workspace: Path, model_dir: str | Path) -> str:
    """Return a portable model path when the model lives in its workspace."""
    try:
        return Path(model_dir).expanduser().resolve().relative_to(
            workspace.expanduser().resolve()
        ).as_posix()
    except (OSError, RuntimeError, ValueError):
        return ""


def resolve_registered_model_dir(
    workspace: Path,
    model: dict[str, Any],
    task: dict[str, Any] | None = None,
) -> Path:
    """Resolve new portable records and legacy absolute-path records."""
    root = workspace.expanduser().resolve()
    candidates: list[Path] = []
    relative = str(model.get("model_relpath") or "").strip()
    if relative:
        candidates.append((root / Path(relative)).resolve())
    legacy = str(model.get("model_dir") or "").strip()
    if legacy:
        legacy_path = Path(os.path.expandvars(os.path.expanduser(legacy)))
        if not legacy_path.is_absolute():
            legacy_path = root / legacy_path
        candidates.append(legacy_path.resolve())
    task_id = safe_slug((task or {}).get("task_id"))
    model_id = safe_slug(model.get("model_id"), fallback="model")
    candidates.append(
        (root / "tasks" / task_id / "models" / model_id).resolve()
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def bindings_path(workspace: Path) -> Path:
    return workspace / "project_bindings.json"


def find_environment_python() -> Path:
    configured = str(
        os.environ.get("NNINTERACTIVE_ENV_PYTHON") or ""
    ).strip()
    candidates = (
        Path(configured) if configured else None,
        ROOT / "nninteractive_env" / "python.exe",
        ROOT / "nninteractive_env" / "Scripts" / "python.exe",
        ROOT / "nninteractive_env" / "python" / "python.exe",
        ROOT / "nninteractive_env" / "bin" / "python3",
        ROOT / "nninteractive_env" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "The nninteractive_env Python was not found. Run Setup Environment first."
    )


def official_model_dir(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    configured = (
        os.environ.get("NNINTERACTIVE_MODEL_DIR", "")
        or config.get("official_model_dir", "")
    )
    if configured:
        return resolve_path(configured)
    return (ROOT / "nninteractive_env" / "models" / "nnInteractive_v1.0").resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model_dir(
    path: str | Path, *, include_checksum: bool = True
) -> dict[str, Any]:
    model_dir = Path(path).expanduser().resolve()
    missing = [
        name for name in MODEL_METADATA_FILES if not (model_dir / name).is_file()
    ]
    checkpoints = sorted(model_dir.glob("fold_*/checkpoint_final.pth"))
    if not checkpoints:
        missing.append("fold_*/checkpoint_final.pth")
    result: dict[str, Any] = {
        "model_dir": str(model_dir),
        "compatible": not missing,
        "missing": missing,
    }
    if checkpoints:
        result["checkpoint"] = str(checkpoints[0])
        if include_checksum:
            result["checkpoint_sha256"] = sha256_file(checkpoints[0])
        result["fold"] = checkpoints[0].parent.name.replace("fold_", "", 1)
    manifest = read_json(model_dir / "finetune_manifest.json", {}) or {}
    verification = manifest.get("runtime_verification") or {}
    result["runtime_verified"] = bool(
        verification.get("verified")
        and verification.get("expected_parameter_fingerprint")
        == verification.get("loaded_parameter_fingerprint")
    )
    result["effective_model_fingerprint"] = str(
        verification.get("loaded_parameter_fingerprint") or ""
    )
    return result


def load_registry(workspace: Path) -> dict[str, Any]:
    payload = read_json(registry_path(workspace), {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("schema_version", "nninteractive_task_registry.v1")
    payload.setdefault("tasks", [])
    return payload


def save_registry(workspace: Path, payload: dict[str, Any]) -> None:
    for task in payload.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for model in task.get("models") or []:
            if not isinstance(model, dict):
                continue
            resolved = resolve_registered_model_dir(workspace, model, task)
            relative = relative_model_path(workspace, resolved)
            if relative:
                model["model_relpath"] = relative
                model.pop("model_dir", None)
    payload["schema_version"] = "nninteractive_task_registry.v2"
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(registry_path(workspace), payload)


def task_rows(workspace: Path) -> list[dict[str, Any]]:
    rows = load_registry(workspace).get("tasks") or []
    return [row for row in rows if isinstance(row, dict)]


def find_task(workspace: Path, task_id: str) -> dict[str, Any] | None:
    wanted = safe_slug(task_id)
    for row in task_rows(workspace):
        if safe_slug(row.get("task_id")) == wanted:
            return row
    return None


def model_rows(workspace: Path, task_id: str) -> list[dict[str, Any]]:
    task = find_task(workspace, task_id) or {}
    rows = task.get("models") or []
    result = [row for row in rows if isinstance(row, dict)]
    result.sort(key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True)
    return result


def model_is_usable(model: dict[str, Any] | None) -> bool:
    if not isinstance(model, dict) or not model.get("compatible", True):
        return False
    return str(model.get("state") or "").strip().lower() not in UNUSABLE_MODEL_STATES


def strategy_display_name(value: object) -> str:
    strategy = str(value or "").strip().lower()
    if strategy == "clopa_in":
        return "CLoPA-IN"
    if strategy in {"clopa_conv", "clopa_cn"}:
        return "CLoPA-CN"
    if strategy == "full":
        return "Full adaptation"
    return strategy or "Unknown"


def selected_model(
    workspace: Path, task_id: str, model_id: str = ""
) -> dict[str, Any] | None:
    task = find_task(workspace, task_id)
    if not task:
        return None
    wanted = str(model_id or task.get("recommended_model_id") or "").strip()
    for row in task.get("models") or []:
        if (
            isinstance(row, dict)
            and str(row.get("model_id") or "") == wanted
            and model_is_usable(row)
        ):
            selected = dict(row)
            selected["_workspace"] = str(workspace.expanduser().resolve())
            return selected
    return None


def model_profile(
    task: dict[str, Any],
    model: dict[str, Any],
    *,
    workspace: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    if not model_is_usable(model):
        raise RuntimeError("The selected task model is not available for annotation.")
    workspace = workspace or (
        Path(str(model.get("_workspace")))
        if model.get("_workspace")
        else workspace_root()
    )
    model_dir = resolve_registered_model_dir(workspace, model, task)
    audit = audit_model_dir(
        model_dir,
        include_checksum=verify_checksum,
    )
    if not audit.get("compatible"):
        raise RuntimeError(
            "The selected task model is incomplete: {}".format(
                ", ".join(audit.get("missing") or [])
            )
        )
    expected = str(model.get("checkpoint_sha256") or "").lower()
    actual = str(audit.get("checkpoint_sha256") or expected).lower()
    if verify_checksum and expected and expected != actual:
        raise RuntimeError(
            "The selected task model checkpoint does not match its registered checksum."
        )
    if model.get("runtime_verified") and not audit.get("runtime_verified"):
        raise RuntimeError(
            "The selected task model lost its effective runtime verification metadata."
        )
    return {
        "source": "task_model",
        "profile_id": "{}:{}".format(task.get("task_id"), model.get("model_id")),
        "task_id": str(task.get("task_id") or ""),
        "task_name": str(task.get("task_name") or task.get("task_id") or ""),
        "model_id": str(model.get("model_id") or ""),
        "model_dir": str(model_dir),
        "checkpoint_sha256": actual,
        "strategy": str(model.get("strategy") or ""),
        "validated_prompt_types": list(
            model.get("validated_prompt_types")
            or DEFAULT_VALIDATED_PROMPT_TYPES
        ),
        "effective_model_fingerprint": str(
            model.get("effective_model_fingerprint")
            or audit.get("effective_model_fingerprint")
            or ""
        ),
        "runtime_verified": bool(
            model.get("runtime_verified") or audit.get("runtime_verified")
        ),
        "input_contract": dict(
            model.get("input_contract") or NNINTERACTIVE_INPUT_CONTRACT
        ),
    }


def normalize_mask_name(value: object) -> str:
    text = str(value or "").strip().lower()
    output = []
    for character in text:
        if character.isalnum() or character in ("_", "-", "."):
            output.append(character)
        else:
            output.append("_")
    return re.sub(r"_+", "_", "".join(output)).strip("._-")


_LABEL_SUFFIXES = (
    ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd.gz", ".nrrd",
)


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_case_image(case_dir: str | Path) -> Path | None:
    root = Path(case_dir)
    preferred = (
        "ct.nii.gz",
        "mr.nii.gz",
        "image.nii.gz",
        "ct.nii",
        "mr.nii",
        "image.nii",
        "image.mha",
        "image.mhd",
        "image.nrrd",
    )
    by_lower = {
        path.name.lower(): path
        for path in root.iterdir()
        if path.is_file()
    } if root.is_dir() else {}
    for name in preferred:
        if name in by_lower:
            return by_lower[name].resolve()
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_file():
                continue
            lower = path.name.lower()
            if (
                _is_nifti(path)
                or lower.endswith((".mha", ".mhd", ".nrrd", ".nrrd.gz"))
            ) and not any(token in lower for token in ("label", "mask", "seg")):
                candidates.append(path)
    if len(candidates) == 1:
        return candidates[0].resolve()
    if root.is_dir():
        try:
            from mimics_bridge import is_dicom_folder

            if is_dicom_folder(str(root)):
                return root.resolve()
        except Exception:
            if any(
                path.suffix.lower() == ".dcm"
                for path in root.iterdir()
                if path.is_file()
            ):
                return root.resolve()
    return None


def _label_stem(path: Path) -> str:
    lower = path.name.lower()
    for suffix in _LABEL_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[:-len(suffix)]
    return path.stem


def find_prepared_label(case_dir: str | Path, mask_names: list[str]) -> Path | None:
    root = Path(case_dir)
    normalized = {normalize_mask_name(name) for name in mask_names if name}
    candidates = []
    for folder in (root, root / "segmentations"):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file() or not any(
                path.name.lower().endswith(suffix)
                for suffix in _LABEL_SUFFIXES
            ):
                continue
            norm = normalize_mask_name(_label_stem(path))
            if norm in normalized or norm in {"label", "mask", "segmentation"}:
                candidates.append(path)
    unique = sorted(set(path.resolve() for path in candidates))
    return unique[0] if len(unique) == 1 else None


def discover_mcs_cases(
    mcs_dir: str | Path, image_root: str | Path | None = None
) -> list[dict[str, Any]]:
    root = Path(mcs_dir).expanduser()
    source_root = Path(image_root).expanduser() if image_root else None
    if not root.is_dir():
        return []
    result = []
    manifest_file = root / dataset_manifest.MANIFEST_FILENAME
    manifest = (
        dataset_manifest.load_manifest(manifest_file)
        if manifest_file.is_file()
        else {"cases": {}}
    )
    for mcs_path in sorted(root.glob("*.mcs")):
        case_id = mcs_path.stem
        case_dir = source_root / case_id if source_root else None
        image = find_case_image(case_dir) if case_dir and case_dir.is_dir() else None
        manifest_row = dataset_manifest.find_case(manifest, case_id) or {}
        if image is None and manifest_row:
            resolved = dataset_manifest.resolve_case_path(
                manifest_file, manifest_row, "image"
            )
            image = Path(resolved) if resolved else None
            resolved_case_dir = dataset_manifest.resolve_case_path(
                manifest_file, manifest_row, "source_case_dir"
            )
            if resolved_case_dir:
                case_dir = Path(resolved_case_dir)
        result.append(
            {
                "case_id": case_id,
                "mcs_path": str(mcs_path.resolve()),
                "case_dir": str(case_dir.resolve()) if case_dir and case_dir.exists() else "",
                "image": str(image) if image else "",
                "state": "ready" if image else "image_missing",
            }
        )
    return result


def discover_prepared_cases(
    dataset_root: str | Path,
    mask_names: list[str],
    label_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return []
    label_base = Path(label_root).expanduser() if label_root else None
    label_manifest = None
    if label_base:
        candidate = (
            label_base
            if label_base.name.lower() == dataset_manifest.MANIFEST_FILENAME
            else label_base / dataset_manifest.MANIFEST_FILENAME
        )
        if candidate.is_file():
            label_manifest = candidate
    label_manifest_payload = (
        dataset_manifest.load_manifest(label_manifest)
        if label_manifest
        else {"cases": {}}
    )
    result = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image = find_case_image(case_dir)
        label = None
        label_source = "case_folder"
        if label_manifest:
            manifest_row = (
                dataset_manifest.find_case(
                    label_manifest_payload, case_dir.name
                )
                or {}
            )
            resolved = dataset_manifest.resolve_case_label(
                label_manifest, manifest_row, mask_names
            )
            if resolved:
                label = Path(resolved[1])
                label_source = "dataset_manifest"
        if label is None and label_base and label_base.is_dir():
            label = find_prepared_label(label_base / case_dir.name, mask_names)
            label_source = "separate_label_root"
        if label is None and not label_base:
            label = find_prepared_label(case_dir, mask_names)
        state = "ready"
        if not image:
            state = "image_missing"
        elif not label:
            state = "mask_missing"
        result.append(
            {
                "case_id": case_dir.name,
                "case_dir": str(case_dir.resolve()),
                "image": str(image) if image else "",
                "label": str(label) if label else "",
                "label_source": label_source if label else "",
                "state": state,
            }
        )
    return result


def status_summary(status: dict[str, Any]) -> str:
    phase = str(status.get("status") or "created")
    if phase == "exporting_labels":
        progress = status.get("label_export_progress") or {}
        return "Preparing labels, {} of {} cases".format(
            progress.get("index") or progress.get("completed") or 0,
            progress.get("total") or status.get("case_total") or "?",
        )
    if phase == "waiting_for_gpu":
        return "Waiting for the GPU"
    if phase == "training":
        training_phase = str(status.get("phase") or "").lower()
        if "verif" in training_phase or training_phase.startswith("final"):
            return "Verifying the trained model"
        return "Training, epoch {} of {}".format(
            status.get("epoch") or 0, status.get("epochs") or "?"
        )
    if phase == "validating":
        return "Checking model quality"
    if phase == "preparing_data":
        index = int(status.get("preparation_index") or 0)
        total = int(status.get("preparation_total") or 0)
        if total > 0:
            return "Preparing data {}/{} ({:.0f}%)".format(
                index,
                total,
                100.0 * index / total,
            )
        return "Preparing training data"
    if phase == "preparing_remote":
        return "Preparing verified data for the remote server"
    if phase == "connecting_remote":
        return "Connecting to the remote server"
    if phase == "uploading":
        if str(status.get("phase") or "") == "remote_dataset_cache_hit":
            return "Verified training data reused on the remote server"
        return "Uploading training data, {}%".format(
            int(status.get("transfer_percent") or 0)
        )
    if phase == "starting_remote":
        return "Starting the remote GPU container"
    if phase == "reconnecting_remote":
        return "Remote connection interrupted; training continues on the server"
    if phase == "waiting_for_remote_gpu":
        return "Waiting for remote GPU {}".format(
            status.get("remote_gpu_device") or "automatic"
        )
    if phase == "remote_control_unavailable":
        return "Remote training continues; Docker status is temporarily unavailable"
    if phase in {"remote_training_completed", "finalizing_remote"}:
        return "Remote training completed; preparing the model for local use"
    if phase == "downloading":
        return "Downloading the trained model, {}%".format(
            int(status.get("transfer_percent") or 0)
        )
    mapping = {
        "created": "Preparing training",
        "validating_cases": "Checking selected cases",
        "registering": "Saving the best model",
        "completed": "Training completed",
        "paused": "Training paused; GPU released",
        "cancelled": "Training stopped",
        "failed": "Training needs attention",
        "pausing": "Pausing training and releasing the GPU",
        "stopping": "Stopping training",
    }
    return mapping.get(phase, phase.replace("_", " ").title())
