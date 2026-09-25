#!/usr/bin/env python3
"""Managed nnU-Net training and inference for Mimics-Script.

This module deliberately owns data discovery and spatial validation instead
of calling the legacy TotalSegmentator converter. The existing nnU-Net
planning, training, and prediction functions remain the execution backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from nnunet_common import (  # noqa: E402
    MODEL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TRAINER_ROOT,
    append_log,
    compact_completed_log,
    medical_stem,
    normalize_request,
    path_signature,
    read_json,
    register_model,
    safe_identifier,
    selected_case_ids,
    stable_digest,
    update_status,
    workspace_paths,
    write_json_atomic,
)
from resource_locks import (  # noqa: E402
    FileResourceLock,
    ResourceLockCancelled,
    default_resource_lock_dir,
    process_start_marker,
    register_process,
    unregister_process,
)


SUPPORTED_SUFFIXES = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd.gz",
    ".nrrd",
)

KNOWN_EPOCH_TRAINERS = {
    "MimicsNNUNetTrainer",
    "MimicsNNUNetTrainerNoMirroring",
}


def _cancelled(control_path: Path) -> bool:
    control = read_json(control_path, {}) or {}
    return str(control.get("action") or "").lower() in {"cancel", "stop"}


def _raise_if_cancelled(control_path: Path) -> None:
    if _cancelled(control_path):
        raise InterruptedError("cancel")


def _acquire_local_gpu(
    request: dict[str, Any],
    status_path: Path,
    control_path: Path,
    log_path: Path,
    operation: str,
) -> FileResourceLock | None:
    if bool(request.get("use_cpu")):
        return None
    if bool(request.get("gpu_lock_managed_externally")) or os.environ.get(
        "MIMICS_REMOTE_GPU_LOCK"
    ):
        return None
    if os.environ.get("MIMICS_DISABLE_GPU_LOCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    lock = FileResourceLock(
        default_resource_lock_dir(ROOT) / "gpu.lock",
        "GPU",
        "nnU-Net {}".format(operation),
    )
    last_notice = [0.0]

    def on_wait(holder: dict[str, Any]) -> None:
        try:
            from tools.mimics_label_export import (
                request_nninteractive_server_release_on_contention,
            )

            request_nninteractive_server_release_on_contention(holder)
        except Exception:
            pass
        update_status(
            status_path,
            status="waiting_for_gpu",
            phase="waiting_for_gpu",
            message="Waiting for the GPU. Mimics remains available.",
            resource_wait={
                "resource": "GPU",
                "owner": str((holder or {}).get("owner") or "another AI task"),
                "pid": (holder or {}).get("pid"),
            },
        )
        now = time.time()
        if now - last_notice[0] >= 30.0:
            append_log(
                log_path,
                "Waiting for the GPU held by {} (pid {}).".format(
                    str((holder or {}).get("owner") or "another AI task"),
                    str((holder or {}).get("pid") or "unknown"),
                ),
            )
            last_notice[0] = now

    try:
        lock.acquire(
            wait_seconds=float(request.get("gpu_lock_timeout_seconds") or 86400),
            poll_seconds=2.0,
            on_wait=on_wait,
            should_cancel=lambda: _cancelled(control_path),
        )
    except ResourceLockCancelled as exc:
        raise InterruptedError("cancel") from exc
    lock.update_pid(
        os.getpid(),
        kind="nnunet_{}".format(operation),
        job_id=str(request.get("job_id") or ""),
        stop_path=str(control_path),
    )
    # on_wait() overwrote status/phase with waiting_for_gpu while blocked;
    # restore the active status so viewers do not show a running job as
    # waiting until the caller's next update_status (which for training only
    # arrives after the whole worker finishes).
    try:
        current = read_json(status_path, {}) or {}
    except Exception:
        current = {}
    if current.get("status") == "waiting_for_gpu":
        active = "running" if operation != "training" else "training"
        update_status(
            status_path,
            status=active,
            phase=active,
            resource_wait=None,
        )
    else:
        update_status(status_path, resource_wait=None)
    append_log(log_path, "GPU resource acquired for nnU-Net {}.".format(operation))
    return lock


def _acquire_dataset_lock(
    request: dict[str, Any],
    status_path: Path,
    control_path: Path,
    log_path: Path,
) -> FileResourceLock:
    identity = stable_digest(
        {
            "workspace": str(Path(request["workspace"]).expanduser().resolve()),
            "dataset_id": int(request["dataset_id"]),
        }
    )[:20]
    lock = FileResourceLock(
        default_resource_lock_dir(ROOT) / ("nnunet_dataset_{}.lock".format(identity)),
        "nnU-Net dataset",
        "nnU-Net Dataset{:03d}".format(int(request["dataset_id"])),
    )
    last_notice = [0.0]

    def on_wait(holder: dict[str, Any]) -> None:
        owner = str((holder or {}).get("owner") or "another nnU-Net task")
        update_status(
            status_path,
            status="waiting_for_dataset",
            phase="waiting_for_dataset",
            message=(
                "Waiting for another nnU-Net task using this Dataset ID. "
                "Mimics remains available."
            ),
            resource_wait={
                "resource": "nnU-Net dataset",
                "owner": owner,
                "pid": (holder or {}).get("pid"),
                "cancel_action": "Stop Running Task",
            },
        )
        now = time.time()
        if now - last_notice[0] >= 30.0:
            append_log(
                log_path,
                "Waiting for Dataset{:03d}, currently used by {} (pid {}). "
                "Use Stop Running Task to cancel this wait.".format(
                    int(request["dataset_id"]),
                    owner,
                    str((holder or {}).get("pid") or "unknown"),
                ),
            )
            last_notice[0] = now

    try:
        lock.acquire(
            wait_seconds=float(request.get("dataset_lock_timeout_seconds") or 7200),
            poll_seconds=1.0,
            on_wait=on_wait,
            should_cancel=lambda: _cancelled(control_path),
        )
    except ResourceLockCancelled as exc:
        raise InterruptedError("cancel") from exc
    lock.update_pid(
        os.getpid(),
        kind="nnunet_dataset",
        job_id=str(request.get("job_id") or ""),
        stop_path=str(control_path),
    )
    update_status(status_path, resource_wait=None)
    append_log(
        log_path,
        "Dataset{:03d} preparation resource acquired.".format(
            int(request["dataset_id"])
        ),
    )
    return lock


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    src = Path(source).resolve()
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except OSError:
        shutil.copy2(str(src), str(dst))
        return "copy"


def _is_medical_file(path: Path) -> bool:
    lower = path.name.lower()
    return path.is_file() and any(lower.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _find_case_image(case_dir: Path) -> Path | None:
    from tools.mimics_label_export import find_image

    return find_image(case_dir)


def _case_directories(dataset_root: Path, requested: set[str] | None) -> list[Path]:
    if not dataset_root.is_dir():
        raise RuntimeError("Image dataset does not exist: {}".format(dataset_root))
    rows = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in {"mcs_output", "flexict_models", "nnunet_models"}:
            continue
        if requested and child.name not in requested:
            continue
        if _find_case_image(child) is not None:
            rows.append(child)
    if not rows and _find_case_image(dataset_root) is not None:
        if requested and dataset_root.name not in requested:
            return []
        rows.append(dataset_root)
    return rows


def _candidate_label_dirs(label_root: Path, case_id: str) -> list[Path]:
    candidates = [
        label_root / case_id / "segmentations",
        label_root / case_id,
        label_root / "segmentations" / case_id,
    ]
    return [path for path in candidates if path.is_dir()]


def _find_label_files(
    label_root: Path,
    case_id: str,
    aliases: list[str],
    source_mode: str = "alternatives",
) -> list[Path]:
    wanted = {safe_identifier(value).lower() for value in aliases}
    matches: dict[str, Path] = {}
    for directory in _candidate_label_dirs(label_root, case_id):
        for path in sorted(directory.iterdir()):
            if not _is_medical_file(path):
                continue
            if safe_identifier(medical_stem(path)).lower() in wanted:
                matches[os.path.normcase(str(path.resolve()))] = path.resolve()
    if label_root.is_dir():
        for path in sorted(label_root.iterdir()):
            if not _is_medical_file(path):
                continue
            stem = safe_identifier(medical_stem(path)).lower()
            for alias in wanted:
                if stem in {
                    "{}_{}".format(safe_identifier(case_id).lower(), alias),
                    "{}__{}".format(safe_identifier(case_id).lower(), alias),
                }:
                    matches[os.path.normcase(str(path.resolve()))] = path.resolve()
    paths = sorted(matches.values(), key=lambda path: os.path.normcase(str(path)))
    if source_mode == "alternatives" and len(paths) > 1:
        raise RuntimeError(
            "Case '{}' has more than one file matching aliases {}: {}".format(
                case_id,
                ", ".join(aliases),
                ", ".join(path.name for path in paths),
            )
        )
    return paths


def _find_label_file(label_root: Path, case_id: str, aliases: list[str]) -> Path | None:
    """Compatibility helper for callers that expect one alternative label file."""
    matches = _find_label_files(label_root, case_id, aliases, "alternatives")
    return matches[0] if matches else None


def _source_label_root(request: dict[str, Any]) -> Path:
    source = request["label_source"]
    if source == "dataset_masks":
        return Path(request["dataset_root"]).expanduser().resolve()
    if source in {"exported_masks", "prepared"}:
        path = Path(str(request.get("label_root") or "")).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError("Label folder does not exist: {}".format(path))
        return path
    raise RuntimeError("Label root is not available before .mcs export.")


def _export_mcs_labels(
    request: dict[str, Any],
    case_ids: list[str],
    staging_root: Path,
    status_path: Path,
    control_path: Path,
) -> Path:
    from tools import mimics_label_export

    mcs_dir = Path(str(request.get("mcs_dir") or "")).expanduser().resolve()
    if not mcs_dir.is_dir():
        raise RuntimeError("Saved .mcs folder does not exist: {}".format(mcs_dir))
    aliases = []
    for row in request["labels"]:
        aliases.extend(row["aliases"])
    update_status(
        status_path,
        status="exporting_labels",
        phase="exporting_mcs_labels",
        message="Exporting selected Masks from saved Mimics projects.",
        progress_percent=3,
    )
    result = mimics_label_export.launch_mimics_export(
        Path(request["dataset_root"]).expanduser().resolve(),
        set(case_ids),
        request.get("mimics_exe"),
        Path(request["workspace"]).expanduser().resolve(),
        float(request.get("label_export_timeout_seconds") or 7200),
        status_path=status_path,
        cancel_path=control_path,
        lock_timeout_seconds=float(request.get("background_mimics_lock_timeout_seconds") or 30),
        label_staging_dir=staging_root,
        export_space="source_image",
        mask_names=aliases,
        target_mask_name=None,
        mcs_output_dir=mcs_dir,
        skip_projects_without_requested_mask=False,
        skip_invalid_projects=True,
    )
    batch = result.get("batch_status") or {}
    failed = (
        not result.get("launched")
        or result.get("timed_out")
        or int(result.get("returncode", 0) or 0) != 0
        or str(batch.get("status") or "").lower() in {"failed", "stopping"}
    )
    if failed:
        raise RuntimeError(
            "Mimics label export did not complete. Diagnostics: {}".format(
                result.get("job_runtime") or result.get("log") or staging_root
            )
        )
    _raise_if_cancelled(control_path)
    return staging_root


def _case_source_signature(
    image: Path, labels: list[list[Path]], request: dict[str, Any]
) -> str:
    return stable_digest(
        {
            "image": path_signature(image),
            "labels": [
                [path_signature(path) for path in source_group]
                for source_group in labels
            ],
            "label_mapping": request["labels"],
            "missing_label_policy": request["missing_label_policy"],
            "overlap_policy": str(request.get("overlap_policy") or "fail"),
            "contract": "nnunet_source_grid_case.v3",
        }
    )


def _materialize_case(
    case_id: str,
    image_source: Path,
    label_sources: list[list[Path]],
    request: dict[str, Any],
    cache_root: Path,
    control_path: Path | None = None,
) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np
    from mimics_bridge import (
        _affine_close,
        _validate_resampled_mask_foreground,
        read_nifti_mask_with_affine,
        resample_mask_to_image_grid,
    )
    from tools.mimics_label_export import materialize_source_image as _materialize_source_image

    fingerprint = _case_source_signature(image_source, label_sources, request)
    case_cache = cache_root / safe_identifier(case_id)
    metadata_path = case_cache / "metadata.json"
    image_path = case_cache / "image.nii.gz"
    label_path = case_cache / "label.nii.gz"
    metadata = read_json(metadata_path, {}) or {}
    cache_candidate = (
        metadata.get("fingerprint") == fingerprint
        and image_path.is_file()
        and label_path.is_file()
    )
    if cache_candidate:
        try:
            cached_image = nib.load(str(image_path))
            cached_label = nib.load(str(label_path))
            cached_shape = tuple(int(value) for value in metadata.get("source_shape") or [])
            cached_affine = np.asarray(metadata.get("source_affine") or [], dtype=float)
            cache_candidate = (
                len(cached_shape) == 3
                and cached_affine.shape == (4, 4)
                and cached_shape == tuple(cached_image.shape[:3])
                and tuple(cached_label.shape[:3]) == tuple(cached_image.shape[:3])
                and _affine_close(cached_affine, cached_image.affine)
                and _affine_close(cached_label.affine, cached_image.affine)
                and len(metadata.get("labels") or []) == len(request["labels"])
            )
        except Exception:
            cache_candidate = False
    if cache_candidate:
        return {
            "case_id": case_id,
            "image": image_path,
            "label": label_path,
            "cache_hit": True,
            "fingerprint": fingerprint,
            "foreground_voxels": int(metadata.get("foreground_voxels") or 0),
            "overlap_voxels": int(metadata.get("overlap_voxels") or 0),
        }

    staging = case_cache.with_name(
        "{}.publishing_{}".format(case_cache.name, uuid.uuid4().hex)
    )
    shutil.rmtree(str(staging), ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        if control_path is not None:
            _raise_if_cancelled(control_path)
        _materialize_source_image(image_source, staging / "image.nii.gz")
        source_image = nib.load(str(staging / "image.nii.gz"))
        source_shape = tuple(int(value) for value in source_image.shape[:3])
        source_affine = np.asarray(source_image.affine, dtype=float)
        combined = np.zeros(source_shape, dtype=np.uint8)
        overlap_total = 0
        per_label = []
        for label_spec, source_group in zip(request["labels"], label_sources):
            if control_path is not None:
                _raise_if_cancelled(control_path)
            aligned_union = np.zeros(source_shape, dtype=np.uint8)
            source_count_total = 0
            all_geometry_matched = True
            source_files = []
            for label_source in source_group:
                if control_path is not None:
                    _raise_if_cancelled(control_path)
                binary, affine = read_nifti_mask_with_affine(str(label_source))
                matched = (
                    tuple(binary.shape) == source_shape
                    and _affine_close(affine, source_affine)
                )
                if not matched:
                    aligned = resample_mask_to_image_grid(
                        binary, affine, source_shape, source_affine
                    )
                else:
                    aligned = np.asarray(binary, dtype=np.uint8)
                source_count, _ = _validate_resampled_mask_foreground(
                    binary,
                    aligned,
                    "Preparing nnU-Net label '{}' from '{}' for case '{}'".format(
                        label_spec["name"], label_source.name, case_id
                    ),
                )
                source_count_total += source_count
                all_geometry_matched = all_geometry_matched and matched
                aligned_union[aligned != 0] = 1
                source_files.append(str(label_source))
            target_count = int(np.count_nonzero(aligned_union))
            overlap = int(np.count_nonzero((combined != 0) & (aligned_union != 0)))
            overlap_total += overlap
            if overlap and str(request.get("overlap_policy") or "fail") == "fail":
                raise RuntimeError(
                    "Case '{}' has {} overlapping voxels while adding label '{}'. "
                    "nnU-Net multiclass labels cannot represent overlapping Masks. "
                    "Train them as separate tasks or explicitly allow ordered replacement."
                    .format(case_id, overlap, label_spec["name"])
                )
            combined[aligned_union != 0] = int(label_spec["id"])
            per_label.append(
                {
                    "name": label_spec["name"],
                    "id": int(label_spec["id"]),
                    "source_mode": label_spec["source_mode"],
                    "source_files": source_files,
                    "source_foreground_voxels": source_count_total,
                    "target_foreground_voxels": target_count,
                    "geometry_matched": bool(all_geometry_matched),
                }
            )
        foreground = int(np.count_nonzero(combined))
        if control_path is not None:
            _raise_if_cancelled(control_path)
        output = nib.Nifti1Image(combined, source_affine)
        output.set_qform(source_affine, code=1)
        output.set_sform(source_affine, code=1)
        nib.save(output, str(staging / "label.nii.gz"))
        metadata = {
            "schema_version": "mimics_nnunet_case_cache.v2",
            "case_id": case_id,
            "fingerprint": fingerprint,
            "source_shape": list(source_shape),
            "source_affine": source_affine.tolist(),
            "foreground_voxels": foreground,
            "overlap_voxels": overlap_total,
            "labels": per_label,
            "updated_at_epoch": time.time(),
        }
        write_json_atomic(staging / "metadata.json", metadata)
        if case_cache.exists():
            shutil.rmtree(str(case_cache), ignore_errors=False)
        os.replace(str(staging), str(case_cache))
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)
    return {
        "case_id": case_id,
        "image": image_path,
        "label": label_path,
        "cache_hit": False,
        "fingerprint": fingerprint,
        "foreground_voxels": foreground,
        "overlap_voxels": overlap_total,
    }


def prepare_source_grid_cases(
    request: dict[str, Any],
    output_root: Path,
    status_path: Path,
    control_path: Path,
) -> list[dict[str, Any]]:
    request = normalize_request(request)
    dataset_root = Path(str(request.get("dataset_root") or "")).expanduser().resolve()
    requested = selected_case_ids(request.get("cases"))
    case_dirs = _case_directories(dataset_root, requested)
    if not case_dirs:
        raise RuntimeError("No supported image cases were found in {}.".format(dataset_root))
    label_root: Path
    export_root = output_root / "fresh_mcs_labels"
    if request["label_source"] == "mcs_refresh":
        label_root = _export_mcs_labels(
            request,
            [path.name for path in case_dirs],
            export_root,
            status_path,
            control_path,
        )
    else:
        label_root = _source_label_root(request)
    cache_root = (
        Path(request["workspace"]).expanduser().resolve()
        / "cache"
        / "source_grid"
        / safe_identifier(request["task_id"])
    )
    rows = []
    skipped = []
    cache_hits = 0
    total = len(case_dirs)
    for index, case_dir in enumerate(case_dirs, start=1):
        _raise_if_cancelled(control_path)
        case_id = case_dir.name
        image = _find_case_image(case_dir)
        if image is None:
            skipped.append({"case_id": case_id, "reason": "image_not_found"})
            continue
        label_files = []
        missing = []
        for label in request["labels"]:
            found = _find_label_files(
                label_root,
                case_id,
                label["aliases"],
                label["source_mode"],
            )
            if not found and request["missing_label_policy"] == "require_all":
                missing.append(label["name"])
            label_files.append(found)
        if missing:
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "labels_not_found",
                    "labels": missing,
                }
            )
            update_status(
                status_path,
                status="preparing_data",
                phase="preparing_source_grid",
                current_case=case_id,
                preparation_index=index,
                preparation_total=total,
                skipped_cases=len(skipped),
                progress_percent=5 + int(20 * index / max(1, total)),
            )
            continue
        row = _materialize_case(
            case_id,
            image,
            label_files,
            request,
            cache_root,
            control_path=control_path,
        )
        if row["foreground_voxels"] == 0 and str(request.get("empty_case_policy") or "skip") == "skip":
            skipped.append({"case_id": case_id, "reason": "empty_label"})
        else:
            image_dst = output_root / "input" / safe_identifier(case_id) / "image.nii.gz"
            label_dst = output_root / "labels" / safe_identifier(case_id) / "label.nii.gz"
            _link_or_copy(row["image"], image_dst)
            _link_or_copy(row["label"], label_dst)
            row["image"] = image_dst
            row["label"] = label_dst
            rows.append(row)
            cache_hits += int(bool(row["cache_hit"]))
        update_status(
            status_path,
            status="preparing_data",
            phase=("reusing_source_grid_cache" if row["cache_hit"] else "preparing_source_grid"),
            message="Preparing training case {} of {}.".format(index, total),
            current_case=case_id,
            preparation_index=index,
            preparation_total=total,
            prepared_cases=len(rows),
            skipped_cases=len(skipped),
            source_grid_cache_reused=cache_hits,
            progress_percent=5 + int(20 * index / max(1, total)),
        )
    shutil.rmtree(str(export_root), ignore_errors=True)
    minimum = int(request.get("minimum_cases") or 2)
    if len(rows) < minimum:
        details = "; ".join(
            "{}: {}".format(row["case_id"], row["reason"])
            for row in skipped[:10]
        )
        raise RuntimeError(
            "nnU-Net needs at least {} usable cases; found {}. {}".format(
                minimum, len(rows), details
            )
        )
    update_status(status_path, skipped_case_details=skipped[:100])
    return rows


def _split_cases(rows: list[dict[str, Any]], request: dict[str, Any]) -> tuple[list[str], list[str]]:
    case_ids = sorted(str(row["case_id"]) for row in rows)
    random.Random(int(request["split_seed"])).shuffle(case_ids)
    if len(case_ids) < 2 or float(request["validation_fraction"]) <= 0:
        return case_ids, []
    count = max(1, int(round(len(case_ids) * float(request["validation_fraction"]))))
    count = min(count, len(case_ids) - 1)
    validation = sorted(case_ids[-count:])
    return sorted(case_ids[:-count]), validation


def _split_folds(rows: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, list[str]]]:
    case_ids = sorted(str(row["case_id"]) for row in rows)
    random.Random(int(request["split_seed"])).shuffle(case_ids)
    if len(case_ids) < 2 or float(request["validation_fraction"]) <= 0:
        return [{"train": sorted(case_ids), "val": []} for _ in range(5)]
    count = max(1, int(round(len(case_ids) * float(request["validation_fraction"]))))
    count = min(count, len(case_ids) - 1)
    folds: list[dict[str, list[str]]] = []
    for fold_index in range(5):
        start = (fold_index * count) % len(case_ids)
        validation = {
            case_ids[(start + offset) % len(case_ids)] for offset in range(count)
        }
        folds.append(
            {
                "train": sorted(value for value in case_ids if value not in validation),
                "val": sorted(validation),
            }
        )
    return folds


def _dataset_name(request: dict[str, Any]) -> str:
    return "Dataset{:03d}_{}".format(
        int(request["dataset_id"]), safe_identifier(request["task_id"])
    )


def _assert_dataset_id_available(
    roots: dict[str, Path], request: dict[str, Any]
) -> None:
    expected = _dataset_name(request)
    prefix = "Dataset{:03d}_".format(int(request["dataset_id"]))
    conflicts = set()
    for root in (roots["raw"], roots["preprocessed"], roots["results"]):
        if not root.is_dir():
            continue
        for candidate in root.glob(prefix + "*"):
            if candidate.name != expected:
                conflicts.add(candidate.name)
    if conflicts:
        raise RuntimeError(
            "Dataset ID {} is already used by {} in this model library. "
            "Choose a different Dataset ID for task '{}'.".format(
                int(request["dataset_id"]),
                ", ".join(sorted(conflicts)),
                request["task_name"],
            )
        )


def _materialize_nnunet_raw(
    rows: list[dict[str, Any]], request: dict[str, Any], raw_root: Path, preprocessed_root: Path
) -> tuple[Path, str]:
    dataset_name = _dataset_name(request)
    dataset_dir = raw_root / dataset_name
    staging = dataset_dir.with_name(
        "{}.publishing_{}".format(dataset_name, uuid.uuid4().hex)
    )
    shutil.rmtree(str(staging), ignore_errors=True)
    (staging / "imagesTr").mkdir(parents=True)
    (staging / "labelsTr").mkdir(parents=True)
    folds = _split_folds(rows, request)
    case_key_by_id = {}
    for row in rows:
        case_key = safe_identifier(row["case_id"])
        case_key_by_id[str(row["case_id"])] = case_key
        _link_or_copy(row["image"], staging / "imagesTr" / (case_key + "_0000.nii.gz"))
        _link_or_copy(row["label"], staging / "labelsTr" / (case_key + ".nii.gz"))
    labels = {"background": 0}
    labels.update({str(row["name"]): int(row["id"]) for row in request["labels"]})
    dataset_json = {
        "name": request["task_name"],
        "description": "Prepared by Mimics-Script on the original source-image grid.",
        "channel_names": {"0": request["modality"]},
        "labels": labels,
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": str(
            request.get("image_reader_writer") or "NibabelIOWithReorient"
        ),
    }
    write_json_atomic(staging / "dataset.json", dataset_json)
    fingerprint = stable_digest(
        {
            "cases": sorted(
                (str(row["case_id"]), str(row["fingerprint"])) for row in rows
            ),
            "dataset": dataset_json,
            "contract": "mimics_nnunet_raw.v1",
        }
    )
    write_json_atomic(
        staging / "mimics_dataset_manifest.json",
        {
            "schema_version": "mimics_nnunet_dataset.v1",
            "dataset_fingerprint": fingerprint,
            "task_id": request["task_id"],
            "cases": [
                {
                    "case_id": str(row["case_id"]),
                    "nnunet_case_id": case_key_by_id[str(row["case_id"])],
                    "fingerprint": row["fingerprint"],
                }
                for row in rows
            ],
            "created_at_epoch": time.time(),
        },
    )
    if dataset_dir.exists():
        shutil.rmtree(str(dataset_dir), ignore_errors=False)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staging), str(dataset_dir))
    split_dir = preprocessed_root / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        split_dir / "splits_final.json",
        [
            {
                "train": [case_key_by_id[value] for value in fold["train"]],
                "val": [case_key_by_id[value] for value in fold["val"]],
            }
            for fold in folds
        ],
    )
    return dataset_dir, fingerprint


def _worker_environment(
    request: dict[str, Any], roots: dict[str, Path], stage: str = "train"
) -> dict[str, str]:
    environment = {
        "nnUNet_raw": str(roots["raw"]),
        "nnUNet_preprocessed": str(roots["preprocessed"]),
        "nnUNet_results": str(roots["results"]),
        "nnUNet_extTrainer": str(TRAINER_ROOT),
        "MIMICS_NNUNET_EPOCHS": str(int(request.get("epochs") or 1000)),
    }
    if os.name == "nt" and stage == "train":
        # Windows training: nnU-Net's spawn'd data-augmentation workers each
        # import torch/OpenBLAS and have a history of allocation-failure
        # crashes ("One or more background workers are no longer alive") on
        # RAM-tight workstations. Single-process DA is slower but stable —
        # the same fix the FlexiCT pipeline applies on Windows. Train stage
        # ONLY: the preprocess stage feeds this value to torch.set_num_threads
        # and nnunetv2 2.8.0 rejects 0 there. Linux (remote containers) keeps
        # nnU-Net's multiprocessing default.
        environment["nnUNet_n_proc_DA"] = "0"
    # Every stage, every platform: the preprocess stage's spawn.Pool workers
    # each import torch/OpenBLAS, which starts one BLAS thread per core (~20
    # here); under virtual-memory pressure OpenBLAS dies with "Memory
    # allocation still failed after 10 retries" and nnU-Net's Pool silently
    # respawns the corpse, hanging the stage at 0 CPU forever. Capping BLAS
    # threads matches nnU-Net's own run_training.py entrypoint and the
    # mimics_batch_cli precedent; the shared remote server wants the cap too.
    for blas_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS"):
        environment.setdefault(blas_var, "1")
    gpu_id = str(request.get("gpu_id") if request.get("gpu_id") not in (None, "") else "").strip()
    if gpu_id:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def _spawn_worker(
    stage: str,
    params: dict[str, Any],
    request: dict[str, Any],
    roots: dict[str, Path],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
    resource_lock: FileResourceLock | None = None,
) -> dict[str, Any]:
    spec_path = job_dir / (stage + "_spec.json")
    result_path = job_dir / (stage + "_result.json")
    start_gate = job_dir / (
        ".{}_start_{}.ready".format(stage, uuid.uuid4().hex)
    )
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    environment = _worker_environment(request, roots, stage)
    if os.name == "nt":
        # Long CPU/GPU stages (a 2D slice-model inference runs >1h) must not
        # keep scratch under the system %TEMP%: periodic IT cleanup on
        # managed workstations deletes it mid-run and the export step then
        # fails with FileNotFoundError on its own output. Point TMP/TEMP at
        # a directory inside the job folder, which lives as long as the job.
        job_temp = job_dir / "worker_tmp"
        job_temp.mkdir(parents=True, exist_ok=True)
        environment.setdefault("TMP", str(job_temp))
        environment.setdefault("TEMP", str(job_temp))
    write_json_atomic(
        spec_path,
        {
            "stage": stage,
            "params": params,
            "environment": environment,
            "start_gate": str(start_gate),
            "start_gate_timeout_seconds": 120,
            "control_path": str(control_path),
        },
    )
    command = [
        sys.executable,
        str(ROOT / "tools" / "nnunet_stage_worker.py"),
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        ) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        ownership_token = ""
        try:
            # Register the stage worker so the health panel and kill-background
            # see it (mirrors the finetune pipelines' trainer registration).
            record = register_process(
                ROOT,
                "nnunet_{}".format(stage),
                process.pid,
                job_id=str(request.get("job_id") or ""),
                state_path=str(status_path),
            )
            ownership_token = (record or {}).get("ownership_token") or ""
        except Exception:
            ownership_token = ""
        try:
            if resource_lock is not None:
                transferred = resource_lock.update_pid(
                    process.pid,
                    kind="nnunet_{}".format(stage),
                    job_id=str(request.get("job_id") or ""),
                    stop_path=str(control_path),
                )
                if not transferred:
                    raise RuntimeError(
                        "The GPU lock could not be transferred to the nnU-Net {} worker."
                        .format(stage)
                    )
            update_status(
                status_path,
                worker_pid=process.pid,
                worker_start_marker=process_start_marker(process.pid),
                active_stage=stage,
            )
            write_json_atomic(
                start_gate,
                {"ready_at_epoch": time.time(), "worker_pid": process.pid},
            )
        except Exception:
            from tools.mimics_label_export import terminate_process_tree

            terminate_process_tree(process.pid)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        try:
            last_epoch = -1
            while process.poll() is None:
                if _cancelled(control_path):
                    from tools.mimics_label_export import terminate_process_tree

                    terminate_process_tree(process.pid)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise InterruptedError("cancel")
                if stage == "train":
                    try:
                        with log_path.open("rb") as reader:
                            reader.seek(0, os.SEEK_END)
                            reader.seek(max(0, reader.tell() - 131072), os.SEEK_SET)
                            tail = reader.read().decode("utf-8", "replace")
                        matches = re.findall(r"(?i)\bepoch\s*[:#]?\s*(\d+)\b", tail)
                        if matches:
                            epoch = max(int(value) for value in matches)
                            if epoch != last_epoch:
                                trainer = str(request.get("trainer") or "")
                                total_epochs = (
                                    int(request.get("epochs") or 1000)
                                    if trainer in KNOWN_EPOCH_TRAINERS
                                    else None
                                )
                                completed_epoch = (
                                    min(total_epochs, epoch + 1)
                                    if total_epochs is not None
                                    else epoch + 1
                                )
                                if total_epochs is None:
                                    message = (
                                        "Training epoch {}. The selected trainer controls "
                                        "the total duration."
                                    ).format(completed_epoch)
                                    progress_percent = 50
                                else:
                                    message = "Training epoch {} of {}.".format(
                                        completed_epoch, total_epochs
                                    )
                                    progress_percent = min(
                                        95,
                                        50
                                        + int(
                                            45
                                            * completed_epoch
                                            / max(1, total_epochs)
                                        ),
                                    )
                                update_status(
                                    status_path,
                                    status="training",
                                    phase="training",
                                    message=message,
                                    current_epoch=completed_epoch,
                                    total_epochs=total_epochs,
                                    progress_percent=progress_percent,
                                )
                                last_epoch = epoch
                    except OSError:
                        pass
                time.sleep(0.5)
        finally:
            try:
                start_gate.unlink()
            except OSError:
                pass
            try:
                if ownership_token:
                    unregister_process(
                        ROOT,
                        "nnunet_{}".format(stage),
                        process.pid,
                        ownership_token=ownership_token,
                    )
            except Exception:
                pass
    result = read_json(result_path, {}) or {}
    if process.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(
            "nnU-Net {} failed: {}. Log: {}".format(
                stage, result.get("error") or "worker exit {}".format(process.returncode), log_path
            )
        )
    return dict(result.get("result") or {})


def _runtime_roots(request: dict[str, Any]) -> dict[str, Path]:
    workspace = Path(request["workspace"]).expanduser().resolve()
    custom = request.get("runtime_roots") or {}
    raw = Path(str(custom.get("raw") or workspace / "runtime" / "nnUNet_raw")).resolve()
    preprocessed = Path(
        str(custom.get("preprocessed") or workspace / "runtime" / "nnUNet_preprocessed")
    ).resolve()
    results = Path(
        str(custom.get("results") or workspace / "runtime" / "nnUNet_results")
    ).resolve()
    for path in (raw, preprocessed, results):
        path.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "preprocessed": preprocessed, "results": results}


def _preprocess_cache_valid(
    preprocessed_root: Path, dataset_name: str, fingerprint: str, request: dict[str, Any]
) -> tuple[bool, str]:
    manifest_path = preprocessed_root / dataset_name / "mimics_preprocess_manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    planning = {
        "dataset_fingerprint": fingerprint,
        "configuration": request["configuration"],
        "spacing": request.get("spacing"),
        "patch_size": request.get("patch_size"),
        "batch_size": request.get("batch_size"),
        "plans": request.get("plans") or "nnUNetPlans",
    }
    identity = stable_digest(planning)
    expected = preprocessed_root / dataset_name / request["configuration"]
    return (
        manifest.get("identity") == identity and expected.is_dir(),
        identity,
    )


def _prepare_and_preprocess_dataset(
    rows: list[dict[str, Any]],
    request: dict[str, Any],
    roots: dict[str, Path],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
) -> tuple[Path, str, str]:
    dataset_lock = _acquire_dataset_lock(
        request, status_path, control_path, log_path
    )
    try:
        _assert_dataset_id_available(roots, request)
        dataset_dir, dataset_fingerprint = _materialize_nnunet_raw(
            rows, request, roots["raw"], roots["preprocessed"]
        )
        dataset_name = dataset_dir.name
        cache_valid, preprocess_identity = _preprocess_cache_valid(
            roots["preprocessed"], dataset_name, dataset_fingerprint, request
        )
        plans = str(request.get("plans") or "nnUNetPlans")
        if cache_valid and not bool(request.get("force_preprocess")):
            update_status(
                status_path,
                status="preprocessing",
                phase="reusing_preprocessed_data",
                message="Verified preprocessed nnU-Net data reused.",
                progress_percent=45,
                preprocess_cache_reused=True,
            )
        else:
            # A dataset ID is reusable, but generated arrays from an older
            # fingerprint must not survive deleted or renamed cases.
            split_dir = roots["preprocessed"] / dataset_name
            split_payload = read_json(split_dir / "splits_final.json", []) or []
            shutil.rmtree(str(split_dir), ignore_errors=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(split_dir / "splits_final.json", split_payload)
            update_status(
                status_path,
                status="preprocessing",
                phase="planning_and_preprocessing",
                message="Fingerprinting, planning, and preprocessing the dataset.",
                progress_percent=30,
            )
            result = _spawn_worker(
                "preprocess",
                {
                    "dataset_id": int(request["dataset_id"]),
                    "configuration": request["configuration"],
                    "num_processes": int(request["preprocess_workers"]),
                    "target_spacing": request.get("spacing"),
                    "target_patch_size": request.get("patch_size"),
                    "target_batch_size": request.get("batch_size"),
                    "verify_integrity": bool(request.get("verify_integrity", True)),
                },
                request,
                roots,
                job_dir,
                status_path,
                control_path,
                log_path,
            )
            plans = str(result.get("plans") or plans)
            write_json_atomic(
                roots["preprocessed"]
                / dataset_name
                / "mimics_preprocess_manifest.json",
                {
                    "schema_version": "mimics_nnunet_preprocess.v1",
                    "identity": preprocess_identity,
                    "dataset_fingerprint": dataset_fingerprint,
                    "plans": plans,
                    "configuration": request["configuration"],
                    "updated_at_epoch": time.time(),
                },
            )
        return dataset_dir, dataset_fingerprint, plans
    finally:
        dataset_lock.release()


def _model_directory(roots: dict[str, Path], request: dict[str, Any], plans: str) -> Path:
    return (
        roots["results"]
        / _dataset_name(request)
        / "{}__{}__{}".format(request["trainer"], plans, request["configuration"])
    )


def _copy_model_bundle(
    source_model: Path,
    request: dict[str, Any],
    dataset_fingerprint: str,
    plans: str,
    training_data_profile: dict[str, Any],
) -> dict[str, Any]:
    if not source_model.is_dir():
        raise RuntimeError("nnU-Net model folder was not produced: {}".format(source_model))
    fold_name = "fold_{}".format(request["fold"])
    checkpoint = source_model / fold_name / "checkpoint_final.pth"
    if not checkpoint.is_file():
        raise RuntimeError("Training completed without checkpoint: {}".format(checkpoint))
    model_id = str(request.get("model_id") or "nnunet_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    ))
    workspace = Path(request["workspace"]).expanduser().resolve()
    destination = workspace / "models" / safe_identifier(request["task_id"]) / safe_identifier(model_id)
    if source_model.resolve() != destination.resolve():
        staging = destination.with_name(destination.name + ".publishing_" + uuid.uuid4().hex)
        shutil.rmtree(str(staging), ignore_errors=True)
        shutil.copytree(str(source_model), str(staging))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(str(destination), ignore_errors=False)
        os.replace(str(staging), str(destination))
    manifest_path = destination / "mimics_model_manifest.json"
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "task_id": request["task_id"],
        "task_name": request["task_name"],
        "labels": request["labels"],
        "modality": request["modality"],
        "dataset_id": int(request["dataset_id"]),
        "dataset_name": _dataset_name(request),
        "configuration": request["configuration"],
        "trainer": request["trainer"],
        "plans": plans,
        "folds": [request["fold"]],
        "epochs": int(request["epochs"]),
        "dataset_fingerprint": dataset_fingerprint,
        "training_data_profile": training_data_profile,
        "model_dir": str(destination),
        "manifest_path": str(manifest_path),
        "execution_backend": str(request.get("execution_backend") or "local"),
        "created_at_epoch": time.time(),
    }
    write_json_atomic(manifest_path, manifest)
    register_model(workspace, manifest)
    return manifest


def build_training_data_profile(
    rows: list[dict[str, Any]], modality: str
) -> dict[str, Any]:
    """Describe compatible input properties without binding a model to one case grid."""
    import nibabel as nib
    import numpy as np

    if not rows:
        raise RuntimeError("No validated training cases are available for profiling.")
    shapes = []
    spacings = []
    fields_of_view = []
    orientations = set()
    for row in rows:
        image = nib.load(str(row["image"]))
        shape = tuple(int(value) for value in image.shape[:3])
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise RuntimeError(
                "Training case '{}' is not a valid 3D image.".format(
                    row.get("case_id") or row["image"]
                )
            )
        spacing = np.linalg.norm(np.asarray(image.affine, dtype=float)[:3, :3], axis=0)
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise RuntimeError(
                "Training case '{}' has invalid physical spacing.".format(
                    row.get("case_id") or row["image"]
                )
            )
        sorted_spacing = sorted(float(value) for value in spacing)
        sorted_fov = sorted(float(spacing[index]) * shape[index] for index in range(3))
        shapes.append(sorted(shape))
        spacings.append(sorted_spacing)
        fields_of_view.append(sorted_fov)
        orientations.add("".join(str(value) for value in nib.aff2axcodes(image.affine)))

    def bounds(values: list[list[float]]) -> dict[str, list[float]]:
        array = np.asarray(values, dtype=float)
        return {
            "min": [float(value) for value in np.min(array, axis=0)],
            "max": [float(value) for value in np.max(array, axis=0)],
            "median": [float(value) for value in np.median(array, axis=0)],
        }

    normalized_modality = str(modality or "").strip().upper()
    if normalized_modality == "MRI":
        normalized_modality = "MR"
    return {
        "schema_version": "mimics_nnunet_training_data_profile.v1",
        "case_count": len(rows),
        "modality": normalized_modality,
        "channel_count": 1,
        "spatial_dimensions": 3,
        "shape_sorted": bounds(shapes),
        "spacing_mm_sorted": bounds(spacings),
        "field_of_view_mm_sorted": bounds(fields_of_view),
        "orientation_codes": sorted(orientations),
    }


def validate_model_input_compatibility(
    image_path: str | Path,
    manifest: dict[str, Any],
    request: dict[str, Any],
) -> list[str]:
    """Reject hard incompatibilities and report soft distribution shifts."""
    import nibabel as nib
    import numpy as np

    request_task = str(request.get("task_id") or "").strip()
    model_task = str(manifest.get("task_id") or "").strip()
    if request_task and model_task and request_task != model_task:
        raise RuntimeError(
            "The selected model belongs to task '{}', not task '{}'.".format(
                model_task, request_task
            )
        )
    model_modality = str(manifest.get("modality") or "").strip().upper()
    source_modality = str(request.get("source_modality") or "").strip().upper()
    model_modality = "MR" if model_modality == "MRI" else model_modality
    source_modality = "MR" if source_modality == "MRI" else source_modality
    if (
        model_modality in {"CT", "MR"}
        and source_modality in {"CT", "MR"}
        and model_modality != source_modality
    ):
        raise RuntimeError(
            "The selected model was trained for {}, but the active image is {}."
            .format(model_modality, source_modality)
        )

    image = nib.load(str(image_path))
    if len(image.shape) < 3 or any(int(value) <= 0 for value in image.shape[:3]):
        raise RuntimeError("The prediction input is not a valid 3D medical image.")
    if len(image.shape) > 3 and any(int(value) != 1 for value in image.shape[3:]):
        raise RuntimeError(
            "This integration supports one image channel, but the prediction input "
            "contains non-singleton extra dimensions: {}.".format(image.shape)
        )
    spacing = np.linalg.norm(np.asarray(image.affine, dtype=float)[:3, :3], axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise RuntimeError("The prediction image has invalid physical spacing.")

    profile = manifest.get("training_data_profile") or {}
    if not isinstance(profile, dict) or not profile:
        return [
            "This legacy model has no training-data profile; task and spatial "
            "distribution compatibility could not be fully checked."
        ]
    if int(profile.get("channel_count") or 1) != 1:
        raise RuntimeError(
            "The selected model requires {} input channels; this integration provides one."
            .format(profile.get("channel_count"))
        )
    if int(profile.get("spatial_dimensions") or 3) != 3:
        raise RuntimeError("The selected model is not a 3D medical-volume model.")

    warnings = []

    def outside_distribution(name: str, values: list[float], profile_key: str) -> None:
        limits = profile.get(profile_key) or {}
        lower = limits.get("min") or []
        upper = limits.get("max") or []
        if len(lower) != 3 or len(upper) != 3:
            return
        for value, low, high in zip(sorted(values), lower, upper):
            if float(value) < float(low) / 4.0 or float(value) > float(high) * 4.0:
                warnings.append(
                    "Input {} is far outside the training range; verify that the "
                    "selected model and source image are appropriate.".format(name)
                )
                return

    outside_distribution(
        "voxel spacing",
        [float(value) for value in spacing],
        "spacing_mm_sorted",
    )
    outside_distribution(
        "physical field of view",
        [float(spacing[index]) * int(image.shape[index]) for index in range(3)],
        "field_of_view_mm_sorted",
    )
    return warnings


def register_downloaded_model(
    request: dict[str, Any], model_dir: str | Path, remote_values: dict[str, Any] | None = None
) -> dict[str, Any]:
    destination = Path(model_dir).expanduser().resolve()
    manifest_path = destination / "mimics_model_manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise RuntimeError("Downloaded nnU-Net model manifest is missing or invalid.")
    manifest["model_dir"] = str(destination)
    manifest["manifest_path"] = str(manifest_path)
    manifest["execution_backend"] = "remote"
    manifest.update(dict(remote_values or {}))
    write_json_atomic(manifest_path, manifest)
    register_model(request["workspace"], manifest)
    return manifest


def validate_materialized_source_geometry(
    image_path: str | Path, expected_geometry: dict[str, Any] | None
) -> None:
    expected = dict(expected_geometry or {})
    if not expected:
        return
    import nibabel as nib
    import numpy as np

    image = nib.load(str(image_path))
    expected_shape = tuple(int(value) for value in expected.get("source_shape") or [])
    expected_affine = expected.get("source_voxel_to_ras_matrix")
    if (
        len(expected_shape) != 3
        or tuple(image.shape[:3]) != expected_shape
        or expected_affine is None
        or not np.allclose(
            image.affine,
            np.asarray(expected_affine, dtype=float),
            atol=1e-4,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "The source image on disk no longer matches the geometry recorded "
            "in the open Mimics project. Relink the source image before prediction."
        )


def _record_log_compaction(status_path: Path, log_path: Path) -> None:
    try:
        result = compact_completed_log(log_path)
        if result.get("compacted"):
            update_status(status_path, log_compaction=result)
    except OSError:
        # Logging maintenance must never change a completed task into failure.
        pass


def run_training(job_dir: Path) -> int:
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = normalize_request(read_json(request_path, {}) or {})
    write_json_atomic(request_path, request)
    roots = _runtime_roots(request)
    staging = job_dir / "prepared"
    shutil.rmtree(str(staging), ignore_errors=True)
    staging.mkdir(parents=True)
    update_status(
        status_path,
        schema_version=SCHEMA_VERSION,
        job_id=request["job_id"],
        kind="train",
        task_id=request["task_id"],
        task_name=request["task_name"],
        status="preparing_data",
        phase="discovering_cases",
        message="Discovering images and labels.",
        log_path=str(log_path),
        control_path=str(control_path),
        controller_pid=os.getpid(),
        controller_start_marker=process_start_marker(os.getpid()),
        progress_percent=1,
    )
    try:
        if request["label_source"] == "prepared":
            rows = []
            for raw in request.get("prepared_cases") or []:
                image = Path(raw["image"])
                label = Path(raw["label"])
                if not image.is_file() or not label.is_file():
                    raise RuntimeError("Prepared remote case is incomplete: {}".format(raw))
                rows.append(
                    {
                        "case_id": str(raw["case_id"]),
                        "image": image,
                        "label": label,
                        "fingerprint": str(raw.get("fingerprint") or stable_digest({
                            "image": path_signature(image), "label": path_signature(label)
                        })),
                        "cache_hit": True,
                    }
                )
        else:
            rows = prepare_source_grid_cases(
                request, staging, status_path, control_path
            )
        training_data_profile = build_training_data_profile(
            rows, request.get("modality") or ""
        )
        _raise_if_cancelled(control_path)
        update_status(
            status_path,
            status="preparing_data",
            phase="building_nnunet_dataset",
            message="Building the nnU-Net dataset without changing the source grid.",
            progress_percent=27,
        )
        dataset_dir, dataset_fingerprint, plans = _prepare_and_preprocess_dataset(
            rows,
            request,
            roots,
            job_dir,
            status_path,
            control_path,
            log_path,
        )
        _raise_if_cancelled(control_path)
        update_status(
            status_path,
            status="training",
            phase="training",
            message="nnU-Net training is running. Epoch details are available in the log.",
            progress_percent=50,
            plans=plans,
            dataset_fingerprint=dataset_fingerprint,
            training_curve_path=str(
                _model_directory(roots, request, plans)
                / "fold_{}".format(request["fold"])
                / "progress.png"
            ),
        )
        trainer_params = {
            "dataset_id": int(request["dataset_id"]),
            "configuration": request["configuration"],
            "fold": request["fold"],
            "trainer": request["trainer"],
            "plans": plans,
            "pretrained_weights": request.get("pretrained_weights") or None,
            "num_gpus": int(request["num_gpus"]),
            "continue_training": bool(request.get("continue_training")),
            "only_run_validation": False,
            "export_validation_probabilities": bool(
                request.get("export_validation_probabilities")
            ),
            "disable_checkpointing": False,
            "val_with_best": False,
            "gpu_id": None,
        }
        if request["trainer"] in KNOWN_EPOCH_TRAINERS:
            trainer_params["epochs"] = int(request["epochs"])
        gpu_lock = _acquire_local_gpu(
            request, status_path, control_path, log_path, "training"
        )
        try:
            _spawn_worker(
                "train",
                trainer_params,
                request,
                roots,
                job_dir,
                status_path,
                control_path,
                log_path,
                resource_lock=gpu_lock,
            )
        finally:
            if gpu_lock is not None:
                gpu_lock.release()
        update_status(
            status_path,
            status="finalizing",
            phase="registering_model",
            message="Verifying and registering the trained model.",
            progress_percent=96,
        )
        source_model = _model_directory(roots, request, plans)
        model = _copy_model_bundle(
            source_model,
            request,
            dataset_fingerprint,
            plans,
            training_data_profile,
        )
        update_status(
            status_path,
            status="completed",
            phase="completed",
            message="nnU-Net training completed.",
            progress_percent=100,
            model=model,
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except InterruptedError:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="nnU-Net training was cancelled.",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except Exception as exc:
        append_log(log_path, traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 1
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)
        _record_log_compaction(status_path, log_path)


def run_inference(job_dir: Path) -> int:
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = normalize_request(read_json(request_path, {}) or {})
    model_manifest_path = Path(str(request.get("model_manifest") or "")).expanduser().resolve()
    manifest = read_json(model_manifest_path, {}) or {}
    model_dir = Path(
        str(
            request.get("model_dir_override")
            or manifest.get("model_dir")
            or model_manifest_path.parent
        )
    ).expanduser().resolve()
    image_path = Path(str(request.get("image_path") or "")).expanduser().resolve()
    output_path = Path(str(request.get("output_path") or job_dir / "prediction.nii.gz")).resolve()
    roots = _runtime_roots(request)
    update_status(
        status_path,
        schema_version=SCHEMA_VERSION,
        job_id=request["job_id"],
        kind="infer",
        task_id=manifest.get("task_id") or request.get("task_id") or "",
        task_name=manifest.get("task_name") or request.get("task_name") or "",
        labels=manifest.get("labels") or [],
        status="running",
        phase="loading_model",
        message="Loading the selected nnU-Net model.",
        model_id=manifest.get("model_id"),
        model_manifest=str(model_manifest_path),
        image_path=str(image_path),
        output_path=str(output_path),
        log_path=str(log_path),
        control_path=str(control_path),
        controller_pid=os.getpid(),
        controller_start_marker=process_start_marker(os.getpid()),
        progress_percent=5,
    )
    try:
        if not manifest or not model_dir.is_dir():
            raise RuntimeError("Selected nnU-Net model is missing or invalid.")
        if not image_path.exists():
            raise RuntimeError("Prediction image does not exist: {}".format(image_path))
        from tools.mimics_label_export import materialize_source_image as _materialize_source_image

        inference_input = job_dir / "input" / "source_image.nii.gz"
        inference_input.parent.mkdir(parents=True, exist_ok=True)
        _materialize_source_image(image_path, inference_input)
        validate_materialized_source_geometry(
            inference_input, request.get("source_geometry_expected")
        )
        compatibility_warnings = validate_model_input_compatibility(
            inference_input, manifest, request
        )
        if compatibility_warnings:
            update_status(
                status_path,
                compatibility_warnings=compatibility_warnings,
            )
            for warning in compatibility_warnings:
                append_log(log_path, "Compatibility warning: {}".format(warning))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        update_status(
            status_path,
            status="running",
            phase="predicting",
            message="nnU-Net inference is running.",
            progress_percent=20,
        )
        gpu_lock = _acquire_local_gpu(
            request, status_path, control_path, log_path, "inference"
        )
        try:
            _spawn_worker(
                "infer",
                {
                    "model_folder": str(model_dir),
                    "input_path": str(inference_input),
                    "output_path": str(output_path),
                    "disable_tta": bool(request.get("disable_tta", True)),
                    "use_cpu": bool(request.get("use_cpu", False)),
                    "enable_stats": False,
                    "gpu_device_id": 0,
                    "num_processes_preprocessing": int(request.get("inference_workers") or 2),
                    "num_processes_segmentation_export": int(request.get("inference_workers") or 2),
                },
                request,
                roots,
                job_dir,
                status_path,
                control_path,
                log_path,
                resource_lock=gpu_lock,
            )
        finally:
            if gpu_lock is not None:
                gpu_lock.release()
        if not output_path.is_file():
            raise RuntimeError("nnU-Net inference produced no output: {}".format(output_path))
        import nibabel as nib
        import numpy as np

        source = nib.load(str(inference_input))
        prediction = nib.load(str(output_path))
        if tuple(source.shape[:3]) != tuple(prediction.shape[:3]) or not np.allclose(
            source.affine, prediction.affine, atol=1e-4, rtol=0.0
        ):
            raise RuntimeError(
                "Prediction grid does not match the source image. The result was not offered to Mimics."
            )
        update_status(
            status_path,
            status="completed",
            phase="completed",
            message="nnU-Net inference completed and passed spatial validation.",
            progress_percent=100,
            labels=manifest.get("labels") or [],
            inference_image_path=str(inference_input),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except InterruptedError:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="nnU-Net inference was cancelled.",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except Exception as exc:
        append_log(log_path, traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 1
    finally:
        _record_log_compaction(status_path, log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--job-dir", required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    job_dir = Path(args.job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "infer":
        return run_inference(job_dir)
    return run_training(job_dir)


if __name__ == "__main__":
    raise SystemExit(main())
