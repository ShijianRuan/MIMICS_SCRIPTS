#!/usr/bin/env python3
"""Shared contracts for the local, remote, and Mimics nnU-Net integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "integrations" / "nnunet_segmentation_workflow"
TRAINER_ROOT = WORKFLOW_ROOT / "trainers"
SCHEMA_VERSION = "mimics_nnunet_job.v1"
MODEL_SCHEMA_VERSION = "mimics_nnunet_model.v1"
TERMINAL_STATES = {"completed", "failed", "cancelled", "abandoned"}

# Directory-signature cache: scanning a large DICOM/export folder turns a
# fingerprint check into thousands of stat calls. The cached manifest is
# keyed by the directory's own mtime/ctime, so adding, removing, or
# renaming entries invalidates it immediately. Editing an existing file in
# place does not change any directory mtime, so such a change is picked up
# at the latest when the weekly revalidation window expires — the same
# trade-off already accepted for the remote dataset archive cache. In
# practice these directories hold exported DICOM series that are written
# once; mutable label files go through the per-file signature path.
PATH_SIGNATURE_CACHE_SECONDS = 7 * 24 * 60 * 60
PATH_SIGNATURE_CACHE_VERSION = "path_signature_cache.v1"


def safe_identifier(value: object, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text[:96] or fallback


def medical_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    for suffix in (".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        "{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for attempt in range(20):
            try:
                os.replace(str(temporary), str(destination))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25, 0.02 * (attempt + 1)))
        raise OSError(
            "Could not publish {}: {}".format(destination, last_error)
        )
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def append_log(path: str | Path, message: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            "[{}] {}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), str(message).rstrip()
            )
        )


def compact_completed_log(
    path: str | Path,
    max_bytes: int = 32 * 1024 * 1024,
    head_bytes: int = 1024 * 1024,
    tail_bytes: int = 15 * 1024 * 1024,
) -> dict[str, Any]:
    """Bound a completed task log while retaining startup and recent diagnostics."""
    destination = Path(path)
    try:
        original_size = int(destination.stat().st_size)
    except OSError:
        return {"compacted": False, "original_bytes": 0, "retained_bytes": 0}
    if original_size <= int(max_bytes):
        return {
            "compacted": False,
            "original_bytes": original_size,
            "retained_bytes": original_size,
        }
    head_count = max(0, min(int(head_bytes), original_size))
    tail_count = max(0, min(int(tail_bytes), original_size - head_count))
    with destination.open("rb") as handle:
        head = handle.read(head_count)
        handle.seek(max(head_count, original_size - tail_count))
        tail = handle.read()
    marker = (
        "\n[Log compacted after task completion: {} bytes omitted.]\n".format(
            max(0, original_size - len(head) - len(tail))
        )
    ).encode("utf-8")
    temporary = destination.with_name(
        "{}.{}.compact.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(head)
            handle.write(marker)
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return {
        "compacted": True,
        "original_bytes": original_size,
        "retained_bytes": int(destination.stat().st_size),
    }


def update_status(path: str | Path, **values: Any) -> dict[str, Any]:
    payload = read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)
    return payload


def _path_signature_cache_dir() -> Path | None:
    configured = os.environ.get("MIMICS_PATH_SIGNATURE_CACHE_DIR", "").strip()
    if configured.lower() in {"off", "disabled", "none"}:
        return None
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    if os.name == "nt" and (
        os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    ):
        return (
            Path(os.environ.get("LOCALAPPDATA") or os.environ["APPDATA"])
            / "Mimics-Script"
            / "path_signature_cache"
        )
    return (
        Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        / "mimics-script"
        / "path_signature_cache"
    )


def _load_cached_path_signature(
    source: Path, stat: Any
) -> dict[str, Any] | None:
    try:
        cache_dir = _path_signature_cache_dir()
        if cache_dir is None:
            return None
        key_material = json.dumps(
            [
                PATH_SIGNATURE_CACHE_VERSION,
                str(source),
                int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9)),
                int(getattr(stat, "st_dev", 0) or 0),
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(key_material).hexdigest()[:32]
        entry_path = cache_dir / (key + ".json")
        entry = read_json(entry_path, {}) or {}
        if not isinstance(entry, dict):
            return None
        if (
            entry.get("version") != PATH_SIGNATURE_CACHE_VERSION
            or entry.get("path") != str(source)
            or not isinstance(entry.get("signature"), dict)
            or time.time() - float(entry.get("verified_at_epoch") or 0)
            > PATH_SIGNATURE_CACHE_SECONDS
        ):
            return None
        return entry["signature"]
    except Exception:
        return None


def _store_cached_path_signature(
    source: Path, stat: Any, signature: dict[str, Any]
) -> None:
    try:
        cache_dir = _path_signature_cache_dir()
        if cache_dir is None:
            return
        key_material = json.dumps(
            [
                PATH_SIGNATURE_CACHE_VERSION,
                str(source),
                int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9)),
                int(getattr(stat, "st_dev", 0) or 0),
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(key_material).hexdigest()[:32]
        entry_path = cache_dir / (key + ".json")
        write_json_atomic(
            entry_path,
            {
                "version": PATH_SIGNATURE_CACHE_VERSION,
                "path": str(source),
                "verified_at_epoch": time.time(),
                "signature": signature,
            },
        )
        cutoff = time.time() - PATH_SIGNATURE_CACHE_SECONDS * 2
        for stale in entry_path.parent.glob("*.json"):
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                pass
    except Exception:
        # A read-only or unavailable cache location must never fail a
        # fingerprint check.
        pass


def path_signature(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    if source.is_file():
        return {
            "kind": "file",
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9)),
            "ctime_ns": int(getattr(stat, "st_ctime_ns", stat.st_ctime * 1e9)),
            "device": int(getattr(stat, "st_dev", 0) or 0),
            "file_id": int(getattr(stat, "st_ino", 0) or 0),
        }
    cached = _load_cached_path_signature(source, stat)
    if cached is not None:
        return cached
    total = 0
    latest = int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9))
    count = 0
    manifest = hashlib.sha256()
    for child in sorted(source.rglob("*")):
        if not child.is_file():
            continue
        child_stat = child.stat()
        size = int(child_stat.st_size)
        modified = int(
            getattr(child_stat, "st_mtime_ns", child_stat.st_mtime * 1e9)
        )
        changed = int(
            getattr(child_stat, "st_ctime_ns", child_stat.st_ctime * 1e9)
        )
        total += size
        latest = max(
            latest,
            modified,
        )
        count += 1
        manifest.update(
            json.dumps(
                [
                    child.relative_to(source).as_posix(),
                    size,
                    modified,
                    changed,
                    int(getattr(child_stat, "st_dev", 0) or 0),
                    int(getattr(child_stat, "st_ino", 0) or 0),
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        manifest.update(b"\n")
    signature = {
        "kind": "directory",
        "files": count,
        "size": total,
        "mtime_ns": latest,
        "ctime_ns": int(getattr(stat, "st_ctime_ns", stat.st_ctime * 1e9)),
        "manifest_sha256": manifest.hexdigest(),
    }
    _store_cached_path_signature(source, stat, signature)
    return signature


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _labels_from_mapping(mapping: object) -> list[dict[str, Any]] | None:
    if not isinstance(mapping, dict) or not mapping:
        return None
    flat_rows = []
    grouped_rows = []
    for name, value in mapping.items():
        if str(name).strip().lower() == "background":
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            flat_rows.append(
                {
                    "name": str(name),
                    "id": int(value),
                    "aliases": [str(name)],
                    "source_mode": "alternatives",
                }
            )
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("label"), int)
            and not isinstance(value.get("label"), bool)
            and isinstance(value.get("organs"), list)
        ):
            grouped_rows.append(
                {
                    "name": str(name),
                    "id": int(value["label"]),
                    "aliases": [str(item) for item in value["organs"]],
                    "source_mode": "union",
                }
            )
            continue
        return None
    if flat_rows and not grouped_rows:
        by_id: dict[int, dict[str, Any]] = {}
        for row in flat_rows:
            label_id = int(row["id"])
            existing = by_id.get(label_id)
            if existing is None:
                by_id[label_id] = dict(row)
                continue
            existing["aliases"].extend(row["aliases"])
            existing["source_mode"] = "union"
        flat_rows = list(by_id.values())
    rows = grouped_rows if grouped_rows and not flat_rows else flat_rows
    if not rows or (grouped_rows and flat_rows):
        return None
    return normalize_labels(rows)


def load_label_sets(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load multi-class task definitions from ModelMap TOML or dataset JSON."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Label map does not exist: {}".format(source))
    if source.suffix.lower() == ".json":
        payload = read_json(source, None)
    else:
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Python tomllib or the tomli package is required to read ModelMap TOML files."
                ) from exc
        with source.open("rb") as handle:
            payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Label map must contain a JSON/TOML object.")
    candidates: dict[str, object] = {}
    if isinstance(payload.get("labels"), dict):
        candidates[source.stem] = payload["labels"]
    else:
        direct = _labels_from_mapping(payload)
        if direct is not None:
            return {source.stem: direct}
        candidates.update(
            (str(name), value)
            for name, value in payload.items()
            if isinstance(value, dict)
        )
    result = {}
    errors = []
    for name, mapping in candidates.items():
        try:
            rows = _labels_from_mapping(mapping)
        except ValueError as exc:
            errors.append("{}: {}".format(name, exc))
            continue
        if rows:
            result[name] = rows
    if not result:
        detail = " " + "; ".join(errors[:5]) if errors else ""
        raise ValueError(
            "No valid multi-class label set was found in {}.{}".format(
                source, detail
            )
        )
    return result


def normalize_labels(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("At least one nnU-Net label must be configured.")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    aliases_seen: dict[str, str] = {}
    ids: set[int] = set()
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Label row {} is not an object.".format(index))
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("Label row {} has no output name.".format(index))
        label_id = int(raw.get("id") or index)
        if label_id <= 0 or label_id > 255:
            raise ValueError("Label '{}' ID must be between 1 and 255.".format(name))
        aliases_value = raw.get("aliases") or [name]
        source_mode = str(raw.get("source_mode") or "alternatives").strip().lower()
        if source_mode not in {"alternatives", "union"}:
            raise ValueError(
                "Label '{}' source mode must be 'alternatives' or 'union'.".format(
                    name
                )
            )
        if isinstance(aliases_value, str):
            aliases = [
                item.strip()
                for item in aliases_value.replace(";", ",").split(",")
                if item.strip()
            ]
        else:
            aliases = [str(item).strip() for item in aliases_value if str(item).strip()]
        if name not in aliases:
            aliases.insert(0, name)
        unique_aliases = []
        local_aliases = set()
        for alias in aliases:
            canonical_alias = safe_identifier(alias).lower()
            if canonical_alias in local_aliases:
                continue
            owner = aliases_seen.get(canonical_alias)
            if owner is not None:
                raise ValueError(
                    "Mask alias '{}' is assigned to both '{}' and '{}'.".format(
                        alias, owner, name
                    )
                )
            local_aliases.add(canonical_alias)
            unique_aliases.append(alias)
        canonical_name = safe_identifier(name).lower()
        if canonical_name in names:
            raise ValueError("Duplicate label name: {}".format(name))
        if label_id in ids:
            raise ValueError("Duplicate label ID: {}".format(label_id))
        names.add(canonical_name)
        ids.add(label_id)
        for canonical_alias in local_aliases:
            aliases_seen[canonical_alias] = name
        rows.append(
            {
                "name": name,
                "id": label_id,
                "aliases": unique_aliases,
                "source_mode": source_mode,
            }
        )
    rows.sort(key=lambda row: int(row["id"]))
    expected_ids = list(range(1, len(rows) + 1))
    actual_ids = [int(row["id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            "nnU-Net label IDs must be consecutive starting at 1; received {}.".format(
                ", ".join(str(value) for value in actual_ids)
            )
        )
    return rows


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    values = dict(request or {})
    operation = str(values.get("operation") or "train").lower()
    if operation not in {"train", "infer"}:
        raise ValueError("Unsupported nnU-Net operation: {}".format(operation))
    values["operation"] = operation
    values["job_id"] = safe_identifier(values.get("job_id"), operation)
    values["workspace"] = str(
        Path(values.get("workspace") or Path.home() / ".mimics_script" / "nnunet")
        .expanduser()
        .resolve()
    )
    if operation == "train":
        values["task_name"] = str(values.get("task_name") or "Segmentation task").strip()
        values["task_id"] = safe_identifier(values.get("task_id") or values["task_name"])
        values["labels"] = normalize_labels(values.get("labels"))
        values["dataset_id"] = int(values.get("dataset_id") or 701)
        if not 1 <= values["dataset_id"] <= 999:
            raise ValueError("Dataset ID must be between 1 and 999.")
        values["configuration"] = str(values.get("configuration") or "3d_fullres")
        if values["configuration"] not in {"2d", "3d_fullres", "3d_lowres"}:
            raise ValueError(
                "Supported configurations are 2d, 3d_fullres, and 3d_lowres."
            )
        values["modality"] = str(values.get("modality") or "CT").strip() or "CT"
        values["fold"] = str(values.get("fold") if values.get("fold") is not None else "0")
        if values["fold"] not in {"0", "1", "2", "3", "4", "all"}:
            raise ValueError("Fold must be 0-4 or all.")
        values["epochs"] = int(values.get("epochs") or 1000)
        if not 1 <= values["epochs"] <= 10000:
            raise ValueError("Epochs must be between 1 and 10000.")
        values["num_gpus"] = int(values.get("num_gpus") or 1)
        if not 1 <= values["num_gpus"] <= 16:
            raise ValueError("GPU count must be between 1 and 16.")
        values["preprocess_workers"] = int(values.get("preprocess_workers") or 4)
        if not 1 <= values["preprocess_workers"] <= 64:
            raise ValueError("Preprocessing workers must be between 1 and 64.")
        raw_validation_fraction = values.get("validation_fraction")
        values["validation_fraction"] = (
            0.2
            if raw_validation_fraction in (None, "")
            else float(raw_validation_fraction)
        )
        if not 0.0 <= values["validation_fraction"] < 0.9:
            raise ValueError("Validation fraction must be in [0, 0.9).")
        if values["fold"] != "all" and values["validation_fraction"] <= 0:
            raise ValueError(
                "Validation fraction must be greater than zero for folds 0-4. "
                "Choose fold 'all' to train without a validation split."
            )
        values["split_seed"] = int(values.get("split_seed") or 2026)
        values["trainer"] = str(values.get("trainer") or "MimicsNNUNetTrainer").strip()
        values["plans"] = str(values.get("plans") or "nnUNetPlans").strip()
        values["gpu_id"] = str(values.get("gpu_id") or "").strip()
        if values["gpu_id"]:
            devices = [item.strip() for item in values["gpu_id"].split(",") if item.strip()]
            if not devices or any(not item.isdigit() for item in devices):
                raise ValueError("Local GPU devices must be comma-separated numeric IDs.")
            if len(devices) != values["num_gpus"]:
                raise ValueError(
                    "GPU count must match the number of local GPU device IDs."
                )
            values["gpu_id"] = ",".join(devices)
        values["gpu_lock_timeout_seconds"] = float(
            values.get("gpu_lock_timeout_seconds") or 86400
        )
        if values["gpu_lock_timeout_seconds"] <= 0:
            raise ValueError("GPU wait timeout must be positive.")
        values["label_source"] = str(values.get("label_source") or "dataset_masks")
        if values["label_source"] not in {
            "dataset_masks",
            "exported_masks",
            "mcs_refresh",
            "prepared",
        }:
            raise ValueError("Unsupported label source: {}".format(values["label_source"]))
        values["missing_label_policy"] = str(
            values.get("missing_label_policy") or "require_all"
        ).strip().lower()
        if values["missing_label_policy"] not in {"require_all", "background"}:
            raise ValueError(
                "Missing-label policy must be 'require_all' or 'background'."
            )
        for key, expected_length in (("spacing", 3), ("patch_size", 2 if values["configuration"] == "2d" else 3)):
            raw = values.get(key)
            if raw in (None, "", []):
                values[key] = None
                continue
            parsed = [float(item) if key == "spacing" else int(item) for item in raw]
            if len(parsed) != expected_length or any(item <= 0 for item in parsed):
                raise ValueError(
                    "{} must contain {} positive values for {}.".format(
                        key, expected_length, values["configuration"]
                    )
                )
            values[key] = parsed
        if values["configuration"] == "2d" and values["spacing"] is not None:
            raise ValueError(
                "Manual target spacing is not supported for nnU-Net's 2D plan. "
                "Leave spacing automatic or select a 3D configuration."
            )
        batch_size = values.get("batch_size")
        values["batch_size"] = int(batch_size) if batch_size not in (None, "", 0, "0") else None
        if values["batch_size"] is not None and values["batch_size"] < 1:
            raise ValueError("Batch size must be positive.")
        if bool(values.get("continue_training")) and str(
            values.get("pretrained_weights") or ""
        ).strip():
            raise ValueError(
                "Continuing a fold and loading pretrained weights are mutually exclusive."
            )
    else:
        modality = str(values.get("source_modality") or "").strip().upper()
        values["source_modality"] = "MR" if modality == "MRI" else modality
    return values


def workspace_paths(workspace: str | Path) -> dict[str, Path]:
    root = Path(workspace).expanduser().resolve()
    return {
        "root": root,
        "jobs": root / "jobs",
        "models": root / "models",
        "cache": root / "cache",
        "runtime": root / "runtime",
        "registry": root / "model_registry.json",
    }


def suggest_dataset_id(workspace: str | Path, preferred: int = 701) -> int:
    """Choose an unused nnU-Net dataset number in one managed model library."""
    paths = workspace_paths(workspace)
    used = set()
    for root in (
        paths["runtime"] / "nnUNet_raw",
        paths["runtime"] / "nnUNet_preprocessed",
        paths["runtime"] / "nnUNet_results",
    ):
        if not root.is_dir():
            continue
        for candidate in root.glob("Dataset[0-9][0-9][0-9]_*"):
            try:
                used.add(int(candidate.name[7:10]))
            except (TypeError, ValueError):
                pass
    for model in load_models(workspace):
        try:
            used.add(int(model.get("dataset_id")))
        except (TypeError, ValueError):
            pass
    order = list(range(max(1, int(preferred)), 1000)) + list(
        range(1, max(1, int(preferred)))
    )
    for value in order:
        if value not in used:
            return value
    raise RuntimeError("All nnU-Net Dataset IDs from 1 to 999 are in use.")


def model_registry_paths(workspace: str | Path) -> list[Path]:
    local = workspace_paths(workspace)["registry"]
    global_path = Path.home() / ".mimics_script" / "nnunet_model_registry.json"
    return [local, global_path]


def load_models(
    workspace: str | Path, include_missing: bool = False
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for registry_path in model_registry_paths(workspace):
        payload = read_json(registry_path, {}) or {}
        for row in payload.get("models") or []:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("model_id") or "")
            model_dir = Path(str(row.get("model_dir") or "")).expanduser()
            manifest_path = Path(str(row.get("manifest_path") or "")).expanduser()
            if not model_id:
                continue
            if not model_dir.is_dir() or not manifest_path.is_file():
                # Flag instead of drop: a registry row whose recorded absolute
                # path is dead (e.g. after moving the checkout to another
                # machine) stays visible so the user can repair or re-import
                # it instead of the model silently disappearing.
                if include_missing:
                    row = dict(row)
                    row["status"] = "missing_path"
                    merged[model_id] = row
                continue
            merged[model_id] = dict(row)
    models_root = workspace_paths(workspace)["models"]
    if models_root.is_dir():
        for manifest_path in models_root.glob("*/*/mimics_model_manifest.json"):
            row = read_json(manifest_path, {}) or {}
            model_id = str(row.get("model_id") or "")
            if not model_id or row.get("schema_version") != MODEL_SCHEMA_VERSION:
                continue
            relocated = dict(row)
            relocated["model_dir"] = str(manifest_path.parent.resolve())
            relocated["manifest_path"] = str(manifest_path.resolve())
            merged[model_id] = relocated
    return sorted(
        merged.values(),
        key=lambda row: float(row.get("created_at_epoch") or 0),
        reverse=True,
    )


def model_usability(model: dict[str, Any]) -> tuple[bool, str]:
    model_dir = Path(str(model.get("model_dir") or "")).expanduser()
    if not model_dir.is_dir():
        return False, "model folder is missing"
    folds = [str(value) for value in model.get("folds") or []]
    if not folds:
        folds = [
            path.name[5:]
            for path in model_dir.glob("fold_*")
            if path.is_dir()
        ]
    if not folds:
        return False, "no trained fold is recorded"
    missing = [
        value
        for value in folds
        if not (model_dir / ("fold_" + value) / "checkpoint_final.pth").is_file()
    ]
    if missing:
        return False, "checkpoint_final.pth is missing for fold(s): {}".format(
            ", ".join(missing)
        )
    if not (model_dir / "plans.json").is_file():
        return False, "plans.json is missing"
    if not (model_dir / "dataset.json").is_file():
        return False, "dataset.json is missing"
    return True, ""


def register_model(workspace: str | Path, manifest: dict[str, Any]) -> None:
    model = dict(manifest)
    model_id = str(model.get("model_id") or "")
    if not model_id:
        raise ValueError("Model manifest has no model_id.")
    for registry_path in model_registry_paths(workspace):
        payload = read_json(registry_path, {}) or {}
        rows = [
            dict(row)
            for row in payload.get("models") or []
            if isinstance(row, dict) and str(row.get("model_id") or "") != model_id
        ]
        rows.append(model)
        rows.sort(
            key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True
        )
        write_json_atomic(
            registry_path,
            {
                "schema_version": "mimics_nnunet_registry.v1",
                "models": rows[:500],
                "updated_at_epoch": time.time(),
            },
        )


def selected_case_ids(value: object) -> set[str] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        rows = value.replace(";", ",").split(",")
    elif isinstance(value, Iterable):
        rows = value
    else:
        rows = [value]
    selected = {str(item).strip() for item in rows if str(item).strip()}
    return selected or None
