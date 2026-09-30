#!/usr/bin/env python3
"""External training controller for Mimics nnInteractive task models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
TOOLS_DIR = ROOT / "tools"
FINETUNE_SRC = ROOT / "integrations" / "nninteractive-finetune" / "src"
for candidate in (str(ROOT), str(TOOLS_DIR), str(FINETUNE_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from nninteractive_task_common import (  # noqa: E402
    ACTIVE_STATUSES,
    NNINTERACTIVE_INPUT_CONTRACT,
    append_log,
    audit_model_dir,
    find_environment_python,
    find_prepared_label,
    load_config,
    load_registry,
    model_rows,
    official_model_dir,
    read_json,
    relative_model_path,
    safe_slug,
    save_registry,
    sha256_file,
    task_dir,
    write_json_atomic,
    workspace_root,
)
from nnunet_common import replace_with_retry  # noqa: E402
from resource_locks import (  # noqa: E402
    FileResourceLock,
    ResourceLockCancelled,
    ResourceLockTimeout,
    default_resource_lock_dir,
    register_process,
)

from pipeline_common import (  # noqa: E402
    spawn_background_mimics_export as _spawn_bg_mimics,
    terminate_popen_tree as _pipeline_terminate_tree,
)


def _finetune_runner_script() -> str:
    """Build a child script that imports the fine-tune CLI explicitly.

    Windows embeddable Python can ignore PYTHONPATH when a ``._pth`` file is
    active, so the child must insert the repository paths into its own
    ``sys.path`` before importing the package.
    """
    inserts = [
        "sys.path.insert(0,{!r})".format(path)
        for path in (str(FINETUNE_SRC), str(ROOT))
    ]
    return (
        "import sys;"
        + ";".join(inserts)
        + ";from nninteractive_finetune.__main__ import main;sys.exit(main())"
    )


def update_status(path: Path, **values: Any) -> dict[str, Any]:
    payload = read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)
    return payload


def _hidden_process_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {"start_new_session": True}
    return {
        "creationflags": int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
        )
    }


def _process_exists(pid: object) -> bool:
    try:
        from resource_locks import process_exists

        return process_exists(pid)
    except Exception:
        return False


def _terminate_process_tree(process: subprocess.Popen[Any], grace: float = 10.0) -> bool:
    # Shared kill ladder from pipeline_common (terminate -> grace poll ->
    # taskkill /T /F -> final wait).
    return _pipeline_terminate_tree(process, grace=grace)


def _control_action(control_path: Path) -> str:
    payload = read_json(control_path, {}) or {}
    action = str(payload.get("action") or "").strip().lower()
    if action:
        return action
    if str(payload.get("status") or "").strip().lower() == "stop_requested":
        return "stop"
    return ""


def _write_cancel_marker(path: Path, action: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "{} requested at {}\n".format(action, time.strftime("%Y-%m-%d %H:%M:%S")),
        encoding="utf-8",
    )


def _find_mimics_exe(request: dict[str, Any]) -> str:
    configured = str(request.get("mimics_exe") or "").strip()
    if configured and Path(configured).is_file():
        return configured
    try:
        from tools.mimics_label_export import find_mimics_exe

        return str(find_mimics_exe(configured) or "")
    except Exception:
        return ""


def _run_label_export(
    request: dict[str, Any],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
) -> Path:
    cases = [row for row in request.get("cases") or [] if row.get("split") in ("train", "val")]
    case_ids = [str(row["case_id"]) for row in cases]
    if not case_ids:
        raise RuntimeError("No training or validation cases were selected.")
    mimics_exe = _find_mimics_exe(request)
    if not mimics_exe:
        raise RuntimeError(
            "A background-capable Mimics executable was not found. "
            "Label export from saved .mcs projects cannot start."
        )
    staging = job_dir / "staging" / "labels"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    export_root = job_dir / "mimics_export"
    export_root.mkdir(parents=True, exist_ok=True)
    batch_status = export_root / "status.json"
    stop_path = export_root / "stop.request"
    config_path = export_root / "export_config.json"
    runner_path = export_root / "run_export.py"
    runtime_log = export_root / "process.log"
    task_label_name = safe_slug(request.get("task_name") or request.get("task_id"))
    mask_names = [
        str(value).strip()
        for value in request.get("mask_names") or []
        if str(value).strip()
    ]
    for stale_control in (stop_path,):
        try:
            stale_control.unlink()
        except FileNotFoundError:
            pass
    export_config = {
        "ts_root": str(Path(request["image_root"]).resolve()),
        "output_dir": str(Path(request["mcs_dir"]).resolve()),
        "export_root": str(export_root),
        "job_runtime": str(export_root),
        "status_path": str(batch_status),
        "stop_path": str(stop_path),
        "label_staging_dir": str(staging),
        "export_space": "source_image",
        "cases": case_ids,
        "case_dirs": {
            str(row["case_id"]): str(Path(row["case_dir"]).resolve())
            for row in cases
        },
        "mcs_paths": {
            str(row["case_id"]): str(Path(row["mcs_path"]).resolve())
            for row in cases
        },
        "source_image_paths": {
            str(row["case_id"]): str(Path(row["image"]).resolve())
            for row in cases
        },
        "mask_names": mask_names,
        "target_mask_name": task_label_name,
        "skip_projects_without_requested_mask": True,
    }
    write_json_atomic(config_path, export_config)
    runner_path.write_text(
        "\n".join(
            [
                "import sys",
                "sys.path.insert(0, {!r})".format(str(ROOT / "runtime_py35")),
                "import mimics_export",
                "raise SystemExit(mimics_export.run_background_batch_export({!r}))".format(
                    str(config_path)
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    update_status(
        status_path,
        status="exporting_labels",
        phase="exporting_labels",
        progress_percent=2,
        case_total=len(cases),
        label_export_status=str(batch_status),
        label_export_stop_path=str(stop_path),
    )
    # Scoped background-Mimics lock (shared protocol): serialize on the
    # resources the job actually touches — the .mcs source directory and
    # the label staging destination.  A per-job unique lock would provide
    # no mutual exclusion between concurrent jobs reading the same .mcs
    # folder, so it was replaced by the unified scoped protocol.
    from pipeline_common import acquire_scoped_background_mimics_locks

    output_dir = Path(request["mcs_dir"]).expanduser().resolve()
    locks = acquire_scoped_background_mimics_locks(
        [output_dir, staging],
        owner="nnInteractive task label export",
        wait_seconds=0,
        cancel_path=control_path,
    )
    process = None
    try:
        with runtime_log.open("ab") as log_handle:
            process = _spawn_bg_mimics(
                mimics_exe,
                runner_path,
                export_root,
                status_path=str(status_path),
                log_handle=log_handle,
                mimics_log_path=export_root / "mimics_application.log",
            )
        for lock in locks:
            if not lock.update_pid(
                process.pid,
                kind="nninteractive_finetune_export",
                controller_pid=os.getpid(),
                stop_path=str(stop_path),
                control_path=str(control_path),
                status_path=str(status_path),
                job_id=job_dir.name,
            ):
                raise RuntimeError(
                    "Background Mimics lock ownership was lost while recording PID {}.".format(
                        process.pid
                    )
                )
        append_log(log_path, "Background Mimics label export started (PID {}).".format(process.pid))
        timeout = float(load_config().get("label_export_timeout_seconds", 3600))
        deadline = time.time() + timeout
        last_progress = None
        while process.poll() is None:
            action = _control_action(control_path)
            if action in ("pause", "stop", "cancel") and not stop_path.exists():
                write_json_atomic(
                    stop_path,
                    {
                        "action": action,
                        "requested_at_epoch": time.time(),
                    },
                )
                update_status(
                    status_path,
                    status="pausing" if action == "pause" else "stopping",
                )
            progress = read_json(batch_status, {}) or {}
            signature = (
                progress.get("status"),
                progress.get("phase"),
                progress.get("case_id"),
                progress.get("index"),
                progress.get("completed"),
                progress.get("failed"),
            )
            if signature != last_progress:
                last_progress = signature
                total = int(progress.get("total") or len(cases) or 1)
                completed = int(
                    progress.get("index")
                    or progress.get("completed")
                    or 0
                )
                update_status(
                    status_path,
                    label_export_progress=progress,
                    progress_percent=min(
                        10,
                        2 + int(8.0 * completed / max(1, total)),
                    ),
                )
            if time.time() >= deadline:
                write_json_atomic(
                    stop_path,
                    {"action": "timeout", "requested_at_epoch": time.time()},
                )
                if not _terminate_process_tree(process):
                    update_status(
                        status_path,
                        status="stopping",
                        termination_pending=True,
                        worker_pid=process.pid,
                    )
                    raise RuntimeError(
                        "Label export timed out and the background Mimics process did not exit."
                    )
                raise TimeoutError("Label export exceeded {} seconds.".format(int(timeout)))
            time.sleep(1.0)
        final = read_json(batch_status, {}) or {}
        action = _control_action(control_path)
        if action in ("pause", "stop", "cancel"):
            raise InterruptedError(action)
        if process.returncode != 0 or str(final.get("status") or "").lower() == "failed":
            raise RuntimeError(
                final.get("error")
                or "Background Mimics label export failed with code {}.".format(
                    process.returncode
                )
            )
        if int(final.get("failed") or 0) > 0:
            raise RuntimeError(
                "Label export failed for {} selected case(s).".format(final.get("failed"))
            )
        append_log(
            log_path,
            "Label export completed: {} exported, {} without the target Mask skipped.".format(
                int(final.get("completed") or 0),
                int(final.get("skipped") or 0),
            ),
        )
        return staging
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        for lock in reversed(locks):
            lock.release()


def _ensure_nifti_image(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared_source = source
    temporary_source = None
    if source.is_dir():
        try:
            import SimpleITK as sitk
        except Exception as exc:
            raise RuntimeError(
                "SimpleITK is required to convert DICOM source images: {}".format(
                    exc
                )
            )
        series_ids = list(
            sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source)) or []
        )
        if not series_ids:
            raise RuntimeError(
                "No readable DICOM series was found in {}.".format(source)
            )
        if len(series_ids) > 1:
            raise RuntimeError(
                "More than one DICOM series was found in {}. Prepare one series per case.".format(
                    source
                )
            )
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(source), series_ids[0]
        )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        image = reader.Execute()
        temporary_source = destination.with_name("source_dicom_input.nii.gz")
        sitk.WriteImage(image, str(temporary_source), True)
        prepared_source = temporary_source
    elif not source.is_file():
        raise RuntimeError("Source image was not found: {}.".format(source))

    from mimics_bridge import prepare_source_fastpath_nifti

    try:
        prepare_source_fastpath_nifti(str(prepared_source), str(destination))
    finally:
        if temporary_source is not None:
            try:
                temporary_source.unlink()
            except OSError:
                pass
    return destination.resolve()


def _ensure_binary_nifti_label(
    source: Path,
    destination: Path,
    reference_image: Path,
    *,
    allow_empty: bool = False,
) -> Path:
    """Normalize a label onto the Mimics-compatible training image grid."""
    import nibabel as nib
    import numpy as np
    from mimics_bridge import (
        _affine_close,
        _mask_file_declares_spatial_geometry,
        _validate_resampled_mask_foreground,
        read_nifti_mask_with_affine,
        resample_mask_to_image_grid,
    )

    label, affine = read_nifti_mask_with_affine(str(source))
    foreground = int(np.count_nonzero(label))
    if foreground <= 0 and not allow_empty:
        raise RuntimeError("The selected label is empty: {}.".format(source))
    reference = nib.load(str(reference_image))
    reference_shape = tuple(int(value) for value in reference.shape[:3])
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    if (
        tuple(int(value) for value in label.shape[:3]) != reference_shape
        or not _affine_close(affine, reference_affine)
    ):
        aligned = resample_mask_to_image_grid(
            label,
            affine,
            reference_shape,
            reference_affine,
            allow_voxel_aligned_fallback=(
                not _mask_file_declares_spatial_geometry(str(source))
            ),
        )
        _validate_resampled_mask_foreground(
            label,
            aligned,
            "Preparing nnInteractive fine-tuning label '{}'".format(source),
        )
        label = aligned
        affine = reference_affine
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.asarray(label, dtype=np.uint8),
        np.asarray(affine, dtype=float),
    )
    image.set_qform(np.asarray(affine, dtype=float), code=1)
    image.set_sform(np.asarray(affine, dtype=float), code=1)
    nib.save(image, str(destination))
    return destination.resolve()


def _source_grid_path_signature(path: Path) -> dict[str, Any]:
    """Return a cheap invalidation signature without reading image voxels."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    if resolved.is_file():
        return {
            "kind": "file",
            "name": resolved.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    if resolved.is_dir():
        # DICOM series may contain thousands of files. The source directory is
        # treated as immutable after selection, so a directory-level signature
        # avoids turning a cache lookup back into a full dataset scan.
        return {
            "kind": "directory",
            "name": resolved.name,
            "mtime_ns": int(stat.st_mtime_ns),
        }
    raise FileNotFoundError("Training input does not exist: {}.".format(resolved))


def _source_grid_case_fingerprint(
    image_source: Path,
    label_source: Path,
    initial_source: Path | None,
) -> str:
    payload = {
        "image": _source_grid_path_signature(image_source),
        "label": _source_grid_path_signature(label_source),
        "initial": (
            _source_grid_path_signature(initial_source)
            if initial_source is not None
            else None
        ),
        "contract": "nninteractive_source_grid_inputs.v2",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepare_source_grid_case_cache(
    workspace: Path,
    task_id: str,
    case_id: str,
    image_source: Path,
    label_source: Path,
    initial_source: Path | None,
) -> tuple[Path, Path, Path | None, bool, str, str]:
    """Materialize one source-grid case once and reuse it across jobs.

    The entry is content-addressed by cheap source signatures. Each completed
    entry is immutable, so concurrent local/remote launches cannot overwrite a
    usable case while another job is reading it.
    """
    fingerprint = _source_grid_case_fingerprint(
        image_source, label_source, initial_source
    )
    entry = (
        workspace
        / "cache"
        / "source_grid_inputs"
        / safe_slug(task_id)
        / safe_slug(case_id)
        / fingerprint
    )
    image_path = entry / "image.nii.gz"
    label_path = entry / "label.nii.gz"
    initial_path = entry / "initial_mask.nii.gz"
    metadata_path = entry / "metadata.json"
    metadata = read_json(metadata_path, {}) or {}
    expects_initial = initial_source is not None
    initial_checked = bool(metadata.get("initial_mask_checked"))
    has_initial = bool(metadata.get("has_initial_mask"))
    initial_rejected_reason = str(
        metadata.get("initial_mask_rejected_reason") or ""
    )
    if (
        metadata.get("fingerprint") == fingerprint
        and image_path.is_file()
        and label_path.is_file()
        and (
            not expects_initial
            or (
                initial_checked
                and (not has_initial or initial_path.is_file())
            )
        )
    ):
        return (
            image_path.resolve(),
            label_path.resolve(),
            initial_path.resolve() if has_initial else None,
            True,
            initial_rejected_reason,
            fingerprint,
        )

    entry.parent.mkdir(parents=True, exist_ok=True)
    staging = entry.with_name("{}.{}.tmp".format(entry.name, uuid.uuid4().hex))
    try:
        staging.mkdir(parents=True)
        staged_image = _ensure_nifti_image(
            image_source, staging / "image.nii.gz"
        )
        staged_label = _ensure_binary_nifti_label(
            label_source,
            staging / "label.nii.gz",
            staged_image,
        )
        staged_initial = None
        has_initial = False
        initial_rejected_reason = ""
        if initial_source is not None:
            staged_initial = _ensure_binary_nifti_label(
                initial_source,
                staging / "initial_mask.nii.gz",
                staged_image,
                allow_empty=True,
            )
            try:
                import nibabel as nib
                import numpy as np

                final_values = np.asarray(
                    nib.load(str(staged_label)).dataobj
                ) != 0
                initial_values = np.asarray(
                    nib.load(str(staged_initial)).dataobj
                ) != 0
                if not initial_values.any():
                    initial_rejected_reason = "empty"
                elif np.array_equal(final_values, initial_values):
                    initial_rejected_reason = "identical_to_target"
                else:
                    has_initial = True
                if not has_initial:
                    try:
                        Path(staged_initial).unlink()
                    except OSError:
                        pass
            except Exception as exc:
                raise RuntimeError(
                    "Could not verify the Initial Mask for {}: {}".format(
                        case_id, exc
                    )
                )
        write_json_atomic(
            staging / "metadata.json",
            {
                "schema_version": "nninteractive_source_grid_input_cache.v2",
                "fingerprint": fingerprint,
                "case_id": str(case_id),
                "source_image": str(image_source),
                "source_label": str(label_source),
                "source_initial_mask": (
                    str(initial_source) if initial_source is not None else ""
                ),
                "initial_mask_checked": bool(expects_initial),
                "has_initial_mask": bool(has_initial),
                "initial_mask_rejected_reason": initial_rejected_reason,
                "updated_at_epoch": time.time(),
            },
        )
        try:
            replace_with_retry(str(staging), str(entry))
        except OSError:
            # A concurrent job may have published the same immutable entry.
            metadata = read_json(metadata_path, {}) or {}
            if not (
                metadata.get("fingerprint") == fingerprint
                and image_path.is_file()
                and label_path.is_file()
                and (
                    not expects_initial
                    or (
                        bool(metadata.get("initial_mask_checked"))
                        and (
                            not bool(metadata.get("has_initial_mask"))
                            or initial_path.is_file()
                        )
                    )
                )
            ):
                raise
            has_initial = bool(metadata.get("has_initial_mask"))
            initial_rejected_reason = str(
                metadata.get("initial_mask_rejected_reason") or ""
            )
        return (
            image_path.resolve(),
            label_path.resolve(),
            initial_path.resolve() if has_initial else None,
            False,
            initial_rejected_reason,
            fingerprint,
        )
    finally:
        if staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)


def _find_exported_label(staging: Path, case_id: str) -> Path | None:
    folder = staging / case_id / "segmentations"
    candidates = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
    candidates = sorted(set(path.resolve() for path in candidates if path.is_file()))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one exported label for {}, found {} in {}.".format(
                case_id, len(candidates), folder
            )
        )
    return candidates[0]


def _mcs_export_fingerprint(
    row: dict[str, Any], mask_names: list[str]
) -> str:
    digest = hashlib.sha256()
    for key in ("mcs_path", "image"):
        path = Path(str(row.get(key) or "")).expanduser().resolve()
        stat = path.stat()
        digest.update(key.encode("ascii"))
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(
        json.dumps(
            sorted(str(value).strip().lower() for value in mask_names),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"nninteractive_mcs_source_grid_export.v2")
    return digest.hexdigest()


def _copy_file_atomic(source: Path, destination: Path) -> None:
    # Shared staged-rename copy from pipeline_common.
    from pipeline_common import copy_file_atomic

    copy_file_atomic(source, destination)


def _prepare_cached_mcs_labels(
    request: dict[str, Any],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
    *,
    mask_names_override: list[str] | None = None,
    cache_role: str = "target",
    output_name: str = "labels",
) -> Path:
    selected = [
        row
        for row in request.get("cases") or []
        if row.get("split") in ("train", "val")
    ]
    mask_names = [
        str(value).strip()
        for value in (
            mask_names_override
            if mask_names_override is not None
            else request.get("mask_names") or []
        )
        if str(value).strip()
    ]
    if not mask_names:
        raise RuntimeError(
            "No Mask name was configured for the {} export.".format(
                cache_role
            )
        )
    workspace = Path(request["workspace"]).expanduser().resolve()
    task_id = safe_slug(request.get("task_id") or request.get("task_name"))
    cache_bucket = (
        "mcs_labels"
        if cache_role == "target"
        else "mcs_initial_masks"
    )
    cache_root = workspace / "cache" / cache_bucket / task_id
    output_root = job_dir / "staging" / output_name
    if output_root.exists():
        shutil.rmtree(str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    reusable = []
    changed = []
    signatures = {}
    for row in selected:
        case_id = str(row["case_id"])
        try:
            signature = _mcs_export_fingerprint(row, mask_names)
        except OSError:
            changed.append(row)
            continue
        signatures[case_id] = signature
        case_cache = cache_root / safe_slug(case_id)
        metadata = read_json(case_cache / "metadata.json", {}) or {}
        cached_labels = sorted(
            (case_cache / "segmentations").glob("*.nii.gz")
        ) + sorted((case_cache / "segmentations").glob("*.nii"))
        if (
            metadata.get("fingerprint") == signature
            and len(cached_labels) == 1
            and cached_labels[0].is_file()
        ):
            reusable.append((row, cached_labels[0]))
        else:
            changed.append(row)
    for row, cached_label in reusable:
        destination = (
            output_root
            / str(row["case_id"])
            / "segmentations"
            / cached_label.name
        )
        _copy_file_atomic(cached_label, destination)
    append_log(
        log_path,
        "Label cache: reused {} unchanged case(s); {} case(s) require fresh "
        "Mimics export for {} Masks.".format(
            len(reusable), len(changed), cache_role
        ),
    )
    cache_status = {
        "{}_mask_cache_reused".format(cache_role): len(reusable),
        "{}_mask_cache_refresh".format(cache_role): len(changed),
    }
    if cache_role == "target":
        cache_status.update(
            label_cache_reused=len(reusable),
            label_cache_refresh=len(changed),
        )
    update_status(status_path, **cache_status)
    if changed:
        export_request = dict(request)
        export_request["cases"] = changed
        export_request["mask_names"] = list(mask_names)
        export_request["task_name"] = "{}_{}".format(
            request.get("task_name") or request.get("task_id") or "task",
            cache_role,
        )
        export_job_dir = job_dir / "_changed_{}_export".format(
            safe_slug(cache_role)
        )
        fresh = _run_label_export(
            export_request,
            export_job_dir,
            status_path,
            control_path,
            log_path,
        )
        for row in changed:
            case_id = str(row["case_id"])
            exported = _find_exported_label(fresh, case_id)
            if exported is None:
                continue
            destination = (
                output_root
                / case_id
                / "segmentations"
                / exported.name
            )
            _copy_file_atomic(exported, destination)
            signature = signatures.get(case_id)
            if not signature:
                signature = _mcs_export_fingerprint(row, mask_names)
            cache_case = cache_root / safe_slug(case_id)
            cache_label = cache_case / "segmentations" / exported.name
            try:
                _copy_file_atomic(exported, cache_label)
                write_json_atomic(
                    cache_case / "metadata.json",
                    {
                        "schema_version": "nninteractive_mcs_label_cache.v1",
                        "case_id": case_id,
                        "fingerprint": signature,
                        "mask_names": mask_names,
                        "cache_role": cache_role,
                        "updated_at_epoch": time.time(),
                    },
                )
            except OSError as exc:
                append_log(
                    log_path,
                    "Warning: label cache could not publish case {}: {}. "
                    "The current training run will continue with its staged "
                    "label.".format(case_id, exc),
                )
                update_status(
                    status_path,
                    label_cache_warning={
                        "case_id": case_id,
                        "error": str(exc),
                    },
                )
        shutil.rmtree(str(export_job_dir), ignore_errors=True)
    return output_root


def _prepare_manifest(
    request: dict[str, Any],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
) -> tuple[Path, Path | None]:
    manifest_path = job_dir / "dataset_manifest.json"
    validation_path = job_dir / "validation_manifest.json"
    if manifest_path.is_file():
        return manifest_path, validation_path if validation_path.is_file() else None
    workspace = Path(request["workspace"]).expanduser().resolve()
    source_mode = str(request.get("source_mode") or "mcs").lower()
    initial_mask_source = str(
        request.get("initial_mask_source") or "none"
    ).strip().lower()
    training_goal = str(
        request.get("training_goal") or "general"
    ).strip().lower()
    if training_goal not in {
        "general",
        "start_empty",
        "refine_existing",
    }:
        raise RuntimeError(
            "Unsupported nnInteractive training goal: {}".format(
                training_goal
            )
        )
    if training_goal == "start_empty":
        initial_mask_source = "none"
    initial_mask_names = [
        str(value).strip()
        for value in request.get("initial_mask_names") or []
        if str(value).strip()
    ]
    if (
        initial_mask_source in ("mcs", "exported_masks")
        and {
            safe_slug(value)
            for value in request.get("mask_names") or []
            if str(value).strip()
        }
        & {safe_slug(value) for value in initial_mask_names}
    ):
        raise RuntimeError(
            "Initial Mask names overlap the final Target Mask names. "
            "Choose a distinct draft or partial Mask."
        )
    staging = None
    if source_mode == "mcs":
        staging = _prepare_cached_mcs_labels(
            request, job_dir, status_path, control_path, log_path
        )
    initial_staging = None
    if initial_mask_source == "mcs":
        if not initial_mask_names:
            raise RuntimeError(
                "Initial Mask source is saved .mcs projects, but no Initial "
                "Mask name was configured."
            )
        initial_request = dict(request)
        initial_cases = []
        for row in request.get("cases") or []:
            item = dict(row)
            initial_mcs_path = str(
                item.get("initial_mcs_path") or item.get("mcs_path") or ""
            ).strip()
            if initial_mcs_path:
                item["mcs_path"] = initial_mcs_path
                initial_cases.append(item)
        initial_request["cases"] = initial_cases
        initial_staging = _prepare_cached_mcs_labels(
            initial_request,
            job_dir,
            status_path,
            control_path,
            log_path,
            mask_names_override=initial_mask_names,
            cache_role="initial",
            output_name="initial_masks",
        )
    elif initial_mask_source not in ("none", "exported_masks"):
        raise RuntimeError(
            "Unsupported Initial Mask source: {}".format(
                initial_mask_source
            )
        )
    update_status(
        status_path,
        status="preparing_data",
        phase="preparing_data",
        progress_percent=10,
    )
    rows = []
    validation_rows = []
    real_initial_count = 0
    source_grid_cache_hits = 0
    selected = [
        row for row in request.get("cases") or [] if row.get("split") in ("train", "val")
    ]
    for index, row in enumerate(selected):
        action = _control_action(control_path)
        if action in ("pause", "stop", "cancel"):
            raise InterruptedError(action)
        case_id = str(row["case_id"])
        image_source = Path(row["image"])
        if staging is not None:
            label_source = _find_exported_label(staging, case_id)
            if label_source is None:
                append_log(
                    log_path,
                    "Skipped {} because its saved .mcs project has no matching target Mask.".format(
                        case_id
                    ),
                )
                continue
        else:
            label_source = Path(row["label"]).resolve()
        initial_source = None
        if initial_staging is not None:
            initial_source = _find_exported_label(
                initial_staging, case_id
            )
        elif initial_mask_source == "exported_masks":
            configured = str(row.get("initial_mask") or "").strip()
            if configured and Path(configured).is_file():
                initial_source = Path(configured).resolve()
            else:
                initial_root = Path(
                    str(request.get("initial_mask_root") or "")
                ).expanduser()
                if initial_root.is_dir():
                    initial_source = find_prepared_label(
                        initial_root / case_id,
                        initial_mask_names,
                    )
        (
            image_path,
            label_path,
            initial_path,
            cache_hit,
            initial_rejected_reason,
            source_grid_fingerprint,
        ) = (
            _prepare_source_grid_case_cache(
                workspace,
                str(
                    request.get("prepared_cache_namespace")
                    or request.get("task_id")
                    or request.get("task_name")
                    or "task"
                ),
                case_id,
                image_source,
                Path(label_source),
                Path(initial_source) if initial_source is not None else None,
            )
        )
        if cache_hit:
            source_grid_cache_hits += 1
        if initial_path is not None:
            real_initial_count += 1
        elif initial_rejected_reason:
            reason = {
                "empty": "it is empty",
                "identical_to_target": "it is identical to the final Target Mask",
            }.get(initial_rejected_reason, initial_rejected_reason)
            append_log(
                log_path,
                "Ignored the Initial Mask for {} because {}. This case will "
                "start empty unless the training goal requires an existing "
                "Mask.".format(case_id, reason),
            )
        if training_goal == "refine_existing" and initial_path is None:
            append_log(
                log_path,
                "Skipped {} because refine-existing training requires a real "
                "Initial Mask distinct from the final Target Mask.".format(
                    case_id
                ),
            )
            continue
        item = {
            "case_id": case_id,
            "image": str(image_path),
            "label": str(label_path),
            "source_label": str(label_source),
            "source_image": str(image_source.resolve()),
            "source_mcs": (
                str(Path(row["mcs_path"]).resolve())
                if source_mode == "mcs" and row.get("mcs_path")
                else ""
            ),
            "initial_mask": str(initial_path) if initial_path else "",
            "source_initial_mask": (
                str(Path(initial_source).resolve())
                if initial_source is not None and initial_path is not None
                else ""
            ),
            "source_initial_mcs": (
                str(
                    Path(
                        row.get("initial_mcs_path")
                        or row.get("mcs_path")
                    ).resolve()
                )
                if (
                    initial_mask_source == "mcs"
                    and initial_path is not None
                    and (
                        row.get("initial_mcs_path")
                        or row.get("mcs_path")
                    )
                )
                else ""
            ),
            "initial_mask_source_type": (
                {
                    "mcs": "mimics_saved_mask",
                    "exported_masks": "exported_nifti_mask",
                }.get(initial_mask_source, "none")
                if initial_path is not None
                else "none"
            ),
            "initial_mask_source_model": str(
                row.get("initial_mask_source_model") or ""
            ),
            "initial_mask_source_name": (
                ", ".join(initial_mask_names) if initial_path is not None else ""
            ),
            "source_grid_cache_fingerprint": source_grid_fingerprint,
            "split": str(row.get("split") or "train"),
        }
        rows.append(item)
        if item["split"] == "val":
            validation_rows.append(dict(item))
        update_status(
            status_path,
            preparation_index=index + 1,
            preparation_total=len(selected),
            current_case=case_id,
            source_grid_case_cache_hit=cache_hit,
            phase=(
                "reusing_local_prepared_data"
                if cache_hit
                else "preparing_data"
            ),
            progress_percent=min(
                15,
                10 + int(5.0 * (index + 1) / max(1, len(selected))),
            ),
        )
        append_log(
            log_path,
            "[{}/{}] {} source-grid data for {}.".format(
                index + 1,
                len(selected),
                "Reused" if cache_hit else "Prepared",
                case_id,
            ),
        )
    if not rows:
        if training_goal == "refine_existing":
            raise RuntimeError(
                "None of the selected cases has a usable real Initial Mask. "
                "Refine-existing training requires an Initial Mask that is "
                "non-empty, aligned to the source image, and distinct from "
                "the final Target Mask. Review the Initial Mask source and "
                "names, or choose empty-start/general training."
            )
        raise RuntimeError(
            "No selected case has both a source image and a matching target Mask."
        )
    append_log(
        log_path,
        "Local source-grid cache: reused {} of {} selected case(s).".format(
            source_grid_cache_hits, len(rows)
        ),
    )
    update_status(
        status_path,
        source_grid_cache_reused=source_grid_cache_hits,
        source_grid_cache_total=len(rows),
        phase=(
            "reusing_local_prepared_data"
            if source_grid_cache_hits == len(rows)
            else "preparing_data"
        ),
    )
    if not any(row["split"] == "val" for row in rows) and len(rows) >= 2:
        rows[-1]["split"] = "val"
        validation_rows = [dict(rows[-1])]
        append_log(
            log_path,
            "The originally assigned validation cases had no matching Mask; "
            "{} was reassigned to validation.".format(rows[-1]["case_id"]),
        )
    write_json_atomic(manifest_path, {"cases": rows})
    if validation_rows:
        write_json_atomic(validation_path, {"cases": validation_rows})
    if training_goal == "start_empty":
        append_log(
            log_path,
            "Prepared {} training and {} validation case(s). Initial Masks "
            "are disabled for the empty-start training goal.".format(
                sum(row["split"] == "train" for row in rows),
                len(validation_rows),
            ),
        )
    else:
        append_log(
            log_path,
            "Prepared {} training and {} validation case(s); {} case(s) have "
            "a real Initial Mask. Cases without one start from an empty "
            "Mask.".format(
                sum(row["split"] == "train" for row in rows),
                len(validation_rows),
                real_initial_count,
            ),
        )
    update_status(
        status_path,
        real_initial_mask_cases=real_initial_count,
        empty_start_cases=max(0, len(rows) - real_initial_count),
        initial_mask_policy=(
            "disabled"
            if training_goal == "start_empty"
            else (
                "real_and_empty"
                if real_initial_count
                else "empty_only"
            )
        ),
    )
    return manifest_path, validation_path if validation_rows else None


_LATERALITY_TOKENS = {
    "l",
    "r",
    "left",
    "right",
    "lhs",
    "rhs",
    "lt",
    "rt",
}


def _request_is_left_right_sensitive(request: dict[str, Any]) -> bool:
    values = [
        request.get("task_id"),
        request.get("task_name"),
        *(request.get("mask_names") or []),
    ]
    for raw in values:
        text = str(raw or "").strip().lower()
        if not text:
            continue
        if "左" in text or "右" in text:
            return True
        tokens = set(re.findall(r"[a-z0-9]+", text))
        if tokens & _LATERALITY_TOKENS:
            return True
    return False


def _resolve_mirror_plan(request: dict[str, Any]) -> dict[str, Any]:
    requested = str(request.get("mirror_policy") or "auto").strip().lower()
    if requested not in {"auto", "preserve_lr", "all_axes"}:
        raise RuntimeError(
            "Unknown nnInteractive mirroring policy: {}".format(requested)
        )
    sensitive = _request_is_left_right_sensitive(request)
    resolved = (
        "preserve_lr" if requested == "auto" and sensitive else requested
    )
    if resolved == "auto":
        resolved = "all_axes"
    return {
        "requested": requested,
        "resolved": resolved,
        "left_right_sensitive": sensitive,
        # Prepared arrays are canonical RAS, so spatial axis 0 is left-right.
        "mirror_axes": [1, 2] if resolved == "preserve_lr" else [0, 1, 2],
    }


def _training_config(
    request: dict[str, Any],
    job_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    model_dir = Path(request["output_model_dir"]).resolve()
    trainer_status = job_dir / "trainer_status.json"
    trainer_cancel = job_dir / "trainer_cancel.request"
    config = load_config()
    mirror_plan = _resolve_mirror_plan(request)
    legacy_interaction_profiles = {
        "quick": {
            "interaction_steps": 3,
            "min_interaction_steps": 1,
            "max_interaction_steps": 3,
            "interaction_step_weights": None,
            "short_interaction_probability": 1.0,
            "validation_interaction_steps": [1, 3],
            "initial_mask_probability": 0.6,
            "validate_initial_masks": True,
            "training_goal": "legacy",
            "correction_policy": "clopa_paired",
        },
        "typical": {
            "interaction_steps": 10,
            "min_interaction_steps": 1,
            "max_interaction_steps": 10,
            "interaction_step_weights": None,
            "short_interaction_probability": 0.7,
            "validation_interaction_steps": [1, 3, 5, 10],
            "initial_mask_probability": 0.35,
            "validate_initial_masks": True,
            "training_goal": "legacy",
            "correction_policy": "clopa_paired",
        },
        "extended": {
            "interaction_steps": 15,
            "min_interaction_steps": 1,
            "max_interaction_steps": 15,
            "interaction_step_weights": None,
            "short_interaction_probability": 0.55,
            "validation_interaction_steps": [1, 3, 5, 10, 15],
            "initial_mask_probability": 0.45,
            "validate_initial_masks": True,
            "training_goal": "legacy",
            "correction_policy": "clopa_paired",
        },
    }
    goal_plans = {
        "general": {
            "initial_mask_probability": 0.3,
            "validate_initial_masks": True,
        },
        "start_empty": {
            "initial_mask_probability": 0.0,
            "validate_initial_masks": False,
        },
        "refine_existing": {
            "initial_mask_probability": 1.0,
            "validate_initial_masks": True,
        },
    }
    requested_goal = str(request.get("training_goal") or "").strip().lower()
    legacy_profile = str(
        request.get("interaction_profile") or ""
    ).strip().lower()
    if requested_goal:
        goal_values = goal_plans.get(requested_goal)
        if goal_values is None:
            raise RuntimeError(
                "Unknown nnInteractive training goal: {}".format(
                    requested_goal
                )
            )
        prompt_plan = {
            "training_goal": requested_goal,
            "correction_policy": "clopa_paired",
            "interaction_steps": 8,
            "min_interaction_steps": 1,
            "max_interaction_steps": 8,
            "interaction_step_weights": [
                0.28, 0.24, 0.17, 0.12, 0.08, 0.05, 0.035, 0.025
            ],
            "short_interaction_probability": 0.65,
            "validation_interaction_steps": [1, 3, 5, 8],
            **goal_values,
        }
        interaction_profile = ""
    elif legacy_profile:
        prompt_plan = legacy_interaction_profiles.get(legacy_profile)
        if prompt_plan is None:
            raise RuntimeError(
                "Unknown legacy nnInteractive interaction profile: {}".format(
                    legacy_profile
                )
            )
        interaction_profile = legacy_profile
    else:
        requested_goal = "general"
        prompt_plan = {
            "training_goal": requested_goal,
            "correction_policy": "clopa_paired",
            "interaction_steps": 8,
            "min_interaction_steps": 1,
            "max_interaction_steps": 8,
            "interaction_step_weights": [
                0.28, 0.24, 0.17, 0.12, 0.08, 0.05, 0.035, 0.025
            ],
            "short_interaction_probability": 0.65,
            "validation_interaction_steps": [1, 3, 5, 8],
            **goal_plans[requested_goal],
        }
        interaction_profile = ""
    return {
        "model": {
            "base_model_dir": str(Path(request["base_model_dir"]).resolve()),
            "fold": str(request.get("fold") or "0"),
            "checkpoint_name": "checkpoint_final.pth",
            "strategy": str(request.get("strategy") or "clopa_in"),
        },
        "data": {
            "manifest": str(manifest_path),
            "label_values": [1],
            "validation_fraction": 0.0,
            "patch_size": [128, 128, 128],
            "foreground_patch_probability": 0.5,
            "num_workers": 0,
            "prepared_cache_dir": str(
                Path(
                    request.get("workspace") or job_dir.parent
                ).expanduser().resolve()
                / "cache"
                / "prepared_cases"
                / safe_slug(
                    request.get("prepared_cache_namespace")
                    or request.get("task_id")
                    or request.get("task_name")
                )
            ),
            "keep_prepared_cache": True,
            "augmentation": {
                "enabled": True,
                "profile": "nninteractive_nnunet",
                "flip_probability": 0.5,
                "mirror_axes": mirror_plan["mirror_axes"],
                "mirror_policy_requested": mirror_plan["requested"],
                "mirror_policy_resolved": mirror_plan["resolved"],
                "left_right_sensitive": mirror_plan["left_right_sensitive"],
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
            **prompt_plan,
            "interaction_profile": interaction_profile,
            "point_radius": 4,
            "center_bias": 8.0,
            "interaction_decay": 0.9,
        },
        "training": {
            "output_dir": str(model_dir),
            "epochs": int(request.get("epochs") or config.get("default_epochs", 10)),
            "steps_per_epoch": int(config.get("training_steps_per_epoch", 50)),
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "mixed_precision": True,
            "seed": int(request.get("seed") or 20260724),
            "validation_batches": int(config.get("validation_batches", 8)),
            "device": "auto",
            "resume": True,
            "status_path": str(trainer_status),
            "job_status_path": str(job_dir / "status.json"),
            "cancel_path": str(trainer_cancel),
        },
    }


def _gpu_lock(
    status_path: Path,
    control_path: Path,
    job_id: str,
) -> FileResourceLock:
    config = load_config()
    lock = FileResourceLock(
        default_resource_lock_dir(ROOT) / "gpu.lock",
        "GPU",
        "nnInteractive task fine-tuning",
    )
    last_notice = [0.0]

    def on_wait(holder: dict[str, Any]) -> None:
        owner = str((holder or {}).get("owner") or "another AI task")
        update_status(
            status_path,
            status="waiting_for_gpu",
            phase="waiting_for_gpu",
            message=(
                "Waiting for the GPU held by {}. Training can be paused or "
                "stopped from Task Models; Mimics remains available."
            ).format(owner),
            gpu_holder=holder,
            resource_wait={
                "resource": "GPU",
                "owner": owner,
                "pid": (holder or {}).get("pid"),
                "cancel_action": "Pause and Release GPU or Stop Training",
            },
        )
        try:
            from tools.mimics_label_export import (
                request_nninteractive_server_release_on_contention,
            )

            request_nninteractive_server_release_on_contention(holder)
        except Exception:
            pass
        now = time.time()
        if now - last_notice[0] >= 30.0:
            append_log(
                status_path.parent / "job.log",
                "Waiting for the GPU held by {} (pid {}). Use Pause and "
                "Release GPU or Stop Training to release the wait.".format(
                    owner, str((holder or {}).get("pid") or "unknown")
                ),
            )
            last_notice[0] = now

    acquired = lock.acquire(
        wait_seconds=float(config.get("gpu_lock_timeout_seconds", 86400)),
        poll_seconds=2.0,
        on_wait=on_wait,
        should_cancel=lambda: _control_action(control_path)
        in ("pause", "stop", "cancel"),
    )
    acquired.update_pid(
        os.getpid(),
        kind="nninteractive_finetune",
        stop_path=str(control_path),
        job_id=str(job_id),
    )
    update_status(status_path, resource_wait=None)
    append_log(
        status_path.parent / "job.log",
        "GPU resource acquired for nnInteractive task fine-tuning.",
    )
    return acquired


def _copy_trainer_status(job_status: Path, trainer_status: dict[str, Any]) -> None:
    raw_status = str(trainer_status.get("status") or "")
    mapped = {
        "initializing": "training",
        "preparing": "preparing_data",
        "training": "training",
        "validating": "training",
        "finalizing": "training",
        # The trainer subprocess has finished exporting, but the controller
        # still has model comparison and registry publication to complete.
        "completed": "training",
    }.get(raw_status, raw_status or "training")
    values = {
        key: value
        for key, value in trainer_status.items()
        if key
        in {
            "phase",
            "epoch",
            "epochs",
            "update",
            "updates_per_epoch",
            "loss",
            "train_final_dice",
            "elapsed_seconds",
            "latest_epoch",
            "best_score",
            "validation_batch",
            "validation_batches",
            "completed_cases",
            "total_cases",
            "prepared_cache_reused",
            "prepared_cache_total",
            "reused",
            "error",
        }
    }
    if raw_status == "completed":
        values["phase"] = "checkpoint_runtime_verified"
    epochs = max(1, int(trainer_status.get("epochs") or 1))
    epoch = max(0, int(trainer_status.get("epoch") or 0))
    progress_percent = 15
    if raw_status == "preparing":
        completed_cases = int(trainer_status.get("completed_cases") or 0)
        total_cases = max(1, int(trainer_status.get("total_cases") or 1))
        progress_percent = 10 + int(
            5.0 * completed_cases / total_cases
        )
    elif raw_status in ("training", "validating", "finalizing", "completed"):
        completed_epochs = max(0, min(epochs, epoch - 1))
        epoch_fraction = 0.0
        if raw_status == "training":
            updates = max(1, int(trainer_status.get("updates_per_epoch") or 1))
            epoch_fraction = 0.85 * min(
                1.0,
                float(trainer_status.get("update") or 0) / updates,
            )
            if str(trainer_status.get("phase") or "") == "epoch_completed":
                epoch_fraction = 1.0
        elif raw_status == "validating":
            batches = max(1, int(trainer_status.get("validation_batches") or 1))
            epoch_fraction = 0.85 + 0.15 * min(
                1.0,
                float(trainer_status.get("validation_batch") or 0) / batches,
            )
        else:
            completed_epochs = epochs
            epoch_fraction = 0.0
        training_fraction = min(
            1.0,
            (completed_epochs + epoch_fraction) / epochs,
        )
        progress_percent = 15 + int(70.0 * training_fraction)
    current = read_json(job_status, {}) or {}
    history = list(current.get("metrics_history") or [])
    latest = trainer_status.get("latest_epoch") or {}
    latest_epoch = int(latest.get("epoch") or 0)
    if latest_epoch:
        validation = latest.get("validation") or {}
        point = {
            "epoch": latest_epoch,
            "train_loss": latest.get("train_loss"),
            "validation_auc": validation.get("trajectory_auc"),
        }
        replaced = False
        for index, row in enumerate(history):
            if int(row.get("epoch") or 0) == latest_epoch:
                history[index] = point
                replaced = True
                break
        if not replaced:
            history.append(point)
        history.sort(key=lambda row: int(row.get("epoch") or 0))
        signature = "{}|{}|{}".format(
            latest_epoch,
            point.get("train_loss"),
            point.get("validation_auc"),
        )
        if signature != str(current.get("metric_log_signature") or ""):
            train_loss = point.get("train_loss")
            val_auc = validation.get("trajectory_auc")
            val_dice = validation.get("dice", {})
            val_fg = val_dice.get("foreground") if isinstance(val_dice, dict) else None
            lr = trainer_status.get("learning_rate")
            elapsed = trainer_status.get("elapsed_seconds", 0)
            pieces = [
                "Epoch {}/{}".format(
                    latest_epoch,
                    trainer_status.get("epochs") or "?",
                )
            ]
            if train_loss is not None:
                pieces.append("loss {:.4f}".format(float(train_loss)))
            if val_auc is not None:
                pieces.append(
                    "validation AUC {:.4f}".format(float(val_auc))
                )
            if val_fg is not None:
                pieces.append("Dice {:.4f}".format(float(val_fg)))
            if lr is not None:
                pieces.append("lr {:.2e}".format(float(lr)))
            elapsed_min = elapsed / 60
            if elapsed_min > 1:
                pieces.append("{:.1f}m".format(elapsed_min))
            append_log(job_status.parent / "job.log", " | ".join(pieces))
    else:
        signature = str(current.get("metric_log_signature") or "")
    update_status(
        job_status,
        status=mapped,
        trainer_status=trainer_status,
        metrics_history=history[-500:],
        metric_log_signature=signature,
        progress_percent=min(85, max(10, progress_percent)),
        **values
    )


def _run_training(
    request: dict[str, Any],
    job_dir: Path,
    status_path: Path,
    control_path: Path,
    log_path: Path,
    manifest_path: Path,
) -> None:
    python_exe = find_environment_python()
    config_path = job_dir / "training_config.json"
    training_config = _training_config(request, job_dir, manifest_path)
    write_json_atomic(config_path, training_config)
    augmentation = training_config["data"]["augmentation"]
    update_status(
        status_path,
        mirror_policy=augmentation.get("mirror_policy_resolved"),
        mirror_axes=augmentation.get("mirror_axes"),
        left_right_sensitive=augmentation.get("left_right_sensitive"),
    )
    append_log(
        log_path,
        "Spatial mirroring: {} (canonical RAS axes {}).".format(
            augmentation.get("mirror_policy_resolved"),
            augmentation.get("mirror_axes"),
        ),
    )
    trainer_status_path = Path(training_config["training"]["status_path"])
    trainer_cancel_path = Path(training_config["training"]["cancel_path"])
    try:
        trainer_cancel_path.unlink()
    except FileNotFoundError:
        pass
    gpu_lock = _gpu_lock(status_path, control_path, job_dir.name)
    process = None
    lock_releasable = True
    try:
        update_status(status_path, status="training", phase="starting_training")
        with (job_dir / "trainer.log").open("ab") as handle:
            process = subprocess.Popen(
                [
                    str(python_exe),
                    "-c",
                    _finetune_runner_script(),
                    "train",
                    "--config",
                    str(config_path),
                ],
                cwd=str(ROOT),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                **_hidden_process_kwargs()
            )
        gpu_lock.update_pid(
            process.pid,
            kind="nninteractive_finetune",
            controller_pid=os.getpid(),
            worker_pid=process.pid,
            control_path=str(control_path),
            cancel_path=str(trainer_cancel_path),
            status_path=str(status_path),
            job_id=job_dir.name,
        )
        try:
            register_process(
                ROOT, "trainer", process.pid,
                parent_pid=os.getpid(),
                state_path=str(status_path),
            )
        except Exception:
            pass
        append_log(log_path, "Training process started (PID {}).".format(process.pid))
        last_signature = None
        while process.poll() is None:
            action = _control_action(control_path)
            if action in ("pause", "stop", "cancel") and not trainer_cancel_path.exists():
                _write_cancel_marker(trainer_cancel_path, action)
                update_status(
                    status_path,
                    status="pausing" if action == "pause" else "stopping",
                    requested_action=action,
                )
            trainer_status = read_json(trainer_status_path, {}) or {}
            signature = (
                trainer_status.get("status"),
                trainer_status.get("phase"),
                trainer_status.get("epoch"),
                trainer_status.get("update"),
                trainer_status.get("validation_batch"),
            )
            if trainer_status and signature != last_signature:
                last_signature = signature
                _copy_trainer_status(status_path, trainer_status)
            time.sleep(1.0)
        trainer_status = read_json(trainer_status_path, {}) or {}
        if trainer_status:
            _copy_trainer_status(status_path, trainer_status)
        action = _control_action(control_path)
        if action in ("pause", "stop", "cancel"):
            raise InterruptedError(action)
        if process.returncode != 0:
            raise RuntimeError(
                trainer_status.get("error")
                or "The fine-tuning process exited with code {}.".format(
                    process.returncode
                )
            )
        append_log(
            log_path,
            "Training epochs finished and the best checkpoint was exported. "
            "Quality comparison and model registration are still running.",
        )
    finally:
        if process is not None and process.poll() is None:
            lock_releasable = _terminate_process_tree(process)
        if lock_releasable:
            gpu_lock.release()
        else:
            update_status(
                status_path,
                status="stopping",
                termination_pending=True,
                worker_pid=process.pid if process else 0,
            )


def _run_evaluation(
    model_dir: Path,
    manifest: Path,
    output: Path,
    status_path: Path,
    control_path: Path,
    label: str,
    training_goal: str,
    initial_mask_probability: float,
) -> dict[str, Any]:
    python_exe = find_environment_python()
    gpu_lock = _gpu_lock(status_path, control_path, status_path.stem)
    process = None
    try:
        evaluation_progress = (
            86 if safe_slug(label) == "current_model" else 93
        )
        update_status(
            status_path,
            status="validating",
            phase="validating_{}".format(safe_slug(label)),
            validation_label=label,
            progress_percent=evaluation_progress,
        )
        with output.with_suffix(".log").open("ab") as handle:
            process = subprocess.Popen(
                [
                    str(python_exe),
                    "-c",
                    _finetune_runner_script(),
                    "evaluate",
                    "--model-dir",
                    str(model_dir),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--label-values",
                    "1",
                    "--clicks",
                    "8",
                    "--device",
                    "auto",
                    "--training-goal",
                    str(training_goal),
                    "--initial-mask-probability",
                    str(float(initial_mask_probability)),
                    "--correction-policy",
                    "clopa_paired",
                ],
                cwd=str(ROOT),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                **_hidden_process_kwargs()
            )
        gpu_lock.update_pid(
            process.pid,
            kind="nninteractive_finetune_evaluation",
            controller_pid=os.getpid(),
            worker_pid=process.pid,
            control_path=str(control_path),
            status_path=str(status_path),
        )
        try:
            register_process(
                ROOT, "trainer", process.pid,
                parent_pid=os.getpid(),
                state_path=str(status_path),
            )
        except Exception:
            pass
        while process.poll() is None:
            action = _control_action(control_path)
            if action in ("pause", "stop", "cancel"):
                _terminate_process_tree(process)
                raise InterruptedError(action)
            time.sleep(1.0)
        if process.returncode != 0:
            raise RuntimeError(
                "{} evaluation exited with code {}. See {}.".format(
                    label, process.returncode, output.with_suffix(".log")
                )
            )
        report = read_json(output, {}) or {}
        if "trajectory_auc" not in report:
            raise RuntimeError("{} evaluation did not produce a valid report.".format(label))
        update_status(
            status_path,
            progress_percent=92 if evaluation_progress == 86 else 98,
        )
        return report
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        gpu_lock.release()


def _case_auc(
    report: dict[str, Any], trajectory_key: str = "dice_by_click"
) -> dict[str, float]:
    result = {}
    for row in report.get("cases") or []:
        trajectory = [
            float(value) for value in row.get(trajectory_key) or []
        ]
        if trajectory:
            result[str(row.get("case_id"))] = sum(trajectory) / len(trajectory)
    return result


def _quality_result(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_auc = float(baseline.get("trajectory_auc") or 0.0)
    candidate_auc = float(candidate.get("trajectory_auc") or 0.0)
    baseline_cases = _case_auc(baseline)
    candidate_cases = _case_auc(candidate)
    regression_limit = float(config.get("maximum_severe_case_regression", 0.2))
    severe = []
    for case_id, value in candidate_cases.items():
        if case_id in baseline_cases and baseline_cases[case_id] - value > regression_limit:
            severe.append(
                {
                    "case_id": case_id,
                    "baseline_auc": baseline_cases[case_id],
                    "candidate_auc": value,
                    "delta": value - baseline_cases[case_id],
                }
            )
    delta = candidate_auc - baseline_auc
    comparable_case_ids = set(baseline_cases) & set(candidate_cases)
    mode_comparisons = {}
    mode_severe = []
    mode_not_improved = []
    minimum_improvement = float(
        config.get("minimum_mean_auc_improvement", 0.0)
    )
    for mode, report_key, case_key, starting_key in (
        (
            "empty_mask",
            "empty_mask_trajectory_auc",
            "empty_mask_dice_by_click",
            "empty_mask_baseline_dice",
        ),
        (
            "real_initial_mask",
            "real_initial_mask_trajectory_auc",
            "real_initial_mask_dice_by_click",
            "real_initial_mask_baseline_dice",
        ),
    ):
        baseline_mode = baseline.get(report_key)
        candidate_mode = candidate.get(report_key)
        if baseline_mode is None and candidate_mode is None:
            continue
        baseline_starting = baseline.get(starting_key)
        candidate_starting = candidate.get(starting_key)
        starting_matches = (
            baseline_starting is None
            and candidate_starting is None
        ) or (
            baseline_starting is not None
            and candidate_starting is not None
            and abs(float(baseline_starting) - float(candidate_starting)) <= 1e-8
        )
        comparison = {
            "baseline_auc": baseline_mode,
            "candidate_auc": candidate_mode,
            "starting_mask_dice": candidate_starting,
            "starting_mask_baseline_matches": starting_matches,
            "baseline_auc_gain_vs_start": (
                float(baseline_mode) - float(baseline_starting)
                if baseline_mode is not None and baseline_starting is not None
                else None
            ),
            "candidate_auc_gain_vs_start": (
                float(candidate_mode) - float(candidate_starting)
                if candidate_mode is not None and candidate_starting is not None
                else None
            ),
            "delta_auc": (
                float(candidate_mode) - float(baseline_mode)
                if baseline_mode is not None and candidate_mode is not None
                else None
            ),
        }
        mode_comparisons[mode] = comparison
        if (
            comparison["delta_auc"] is not None
            and float(comparison["delta_auc"]) < minimum_improvement
        ):
            mode_not_improved.append(
                {
                    "mode": mode,
                    "delta_auc": comparison["delta_auc"],
                    "required_delta_auc": minimum_improvement,
                }
            )
        if not starting_matches:
            mode_severe.append(
                {
                    "mode": mode,
                    "reason": "starting_mask_baseline_changed",
                    "baseline_starting_dice": baseline_starting,
                    "candidate_starting_dice": candidate_starting,
                }
            )
        baseline_mode_cases = _case_auc(baseline, case_key)
        candidate_mode_cases = _case_auc(candidate, case_key)
        for case_id, value in candidate_mode_cases.items():
            if (
                case_id in baseline_mode_cases
                and baseline_mode_cases[case_id] - value > regression_limit
            ):
                mode_severe.append(
                    {
                        "mode": mode,
                        "case_id": case_id,
                        "baseline_auc": baseline_mode_cases[case_id],
                        "candidate_auc": value,
                        "delta": value - baseline_mode_cases[case_id],
                    }
                )
    qualifies = (
        bool(comparable_case_ids)
        and
        delta >= minimum_improvement
        and not severe
        and not mode_severe
        and not mode_not_improved
    )
    return {
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "delta_auc": delta,
        "comparable_cases": len(comparable_case_ids),
        "severe_regressions": severe,
        "mode_comparisons": mode_comparisons,
        "mode_severe_regressions": mode_severe,
        "mode_not_improved": mode_not_improved,
        "qualifies": qualifies,
    }


def _register_model(
    request: dict[str, Any],
    workspace: Path,
    model_dir: Path,
    quality: dict[str, Any] | None,
    validation_count: int,
) -> tuple[dict[str, Any], bool]:
    config = load_config()
    audit = audit_model_dir(model_dir)
    if not audit.get("compatible"):
        raise RuntimeError(
            "The exported model failed compatibility audit: {}".format(
                ", ".join(audit.get("missing") or [])
            )
        )
    task_id = safe_slug(request["task_id"])
    registry = load_registry(workspace)
    tasks = registry.get("tasks") or []
    task = next(
        (row for row in tasks if safe_slug(row.get("task_id")) == task_id),
        None,
    )
    if task is None:
        task = {
            "task_id": task_id,
            "task_name": str(request.get("task_name") or task_id),
            "mask_names": list(request.get("mask_names") or []),
            "models": [],
        }
        tasks.append(task)
    model_id = str(request["model_id"])
    rows = [row for row in task.get("models") or [] if row.get("model_id") != model_id]
    minimum_validation = int(
        config.get("minimum_validation_cases_for_auto_selection", 2)
    )
    automatically_selected = bool(
        quality
        and quality.get("qualifies")
        and validation_count >= minimum_validation
    )
    state = "validated" if quality else "unverified"
    if quality and not quality.get("qualifies"):
        state = "not_improved"
    finetune_manifest = read_json(model_dir / "finetune_manifest.json", {}) or {}
    input_contract = finetune_manifest.get("input_contract")
    if input_contract != NNINTERACTIVE_INPUT_CONTRACT:
        raise RuntimeError(
            "The exported task model does not declare the supported "
            "nnInteractive input contract. Training output was not registered."
        )
    runtime_verification = finetune_manifest.get("runtime_verification") or {}
    if (
        not runtime_verification.get("verified")
        or runtime_verification.get("expected_parameter_fingerprint")
        != runtime_verification.get("loaded_parameter_fingerprint")
    ):
        raise RuntimeError(
            "The exported task model did not pass effective runtime-weight "
            "verification and was not registered."
        )
    validated_prompt_types = list(
        finetune_manifest.get("validated_prompt_types") or ["point"]
    )
    training_summary = finetune_manifest.get("training") or {}
    actual_train_cases = list(training_summary.get("train_cases") or [])
    model = {
        "model_id": model_id,
        "model_relpath": relative_model_path(workspace, model_dir),
        "checkpoint_sha256": audit.get("checkpoint_sha256"),
        "fold": audit.get("fold", "0"),
        "strategy": str(request.get("strategy") or ""),
        "training_goal": str(
            training_summary.get("training_goal")
            or request.get("training_goal")
            or "legacy"
        ),
        "correction_policy": str(
            training_summary.get("correction_policy")
            or "clopa_paired"
        ),
        "parent_model_id": str(request.get("parent_model_id") or "official"),
        "created_at_epoch": time.time(),
        "train_case_count": (
            len(actual_train_cases)
            if actual_train_cases
            else sum(
                row.get("split") == "train"
                for row in request.get("cases") or []
            )
        ),
        "validation_case_count": validation_count,
        "quality": quality or {},
        "state": state,
        "compatible": True,
        "source_mode": str(request.get("source_mode") or ""),
        "input_contract": input_contract,
        "validated_prompt_types": validated_prompt_types,
        "effective_model_fingerprint": runtime_verification.get(
            "loaded_parameter_fingerprint"
        ),
        "runtime_verified": True,
    }
    rows.append(model)
    rows.sort(key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True)
    task["models"] = rows
    task["task_name"] = str(request.get("task_name") or task_id)
    task["mask_names"] = list(request.get("mask_names") or [])
    task["updated_at_epoch"] = time.time()
    if automatically_selected:
        task["recommended_model_id"] = model_id
    registry["tasks"] = tasks
    save_registry(workspace, registry)
    write_json_atomic(task_dir(workspace, task_id) / "task.json", task)
    write_json_atomic(model_dir / "integration_manifest.json", model)
    finetune_manifest["model_dir"] = "."
    finetune_manifest["checkpoint"] = "fold_{}/checkpoint_final.pth".format(
        audit.get("fold", "0")
    )
    finetune_manifest["base_model_id"] = str(
        request.get("parent_model_id") or "official"
    )
    finetune_manifest.pop("base_model_dir", None)
    finetune_manifest.pop("base_checkpoint", None)
    write_json_atomic(model_dir / "finetune_manifest.json", finetune_manifest)
    return model, automatically_selected


def _cleanup_terminal_artifacts(
    request: dict[str, Any],
    job_dir: Path,
    *,
    remove_partial_model: bool,
    reset_resume_state: bool = False,
) -> dict[str, Any]:
    """Remove large rebuildable files while preserving job diagnostics."""
    workspace = Path(request.get("workspace") or job_dir.parent.parent).resolve()
    paths = [
        job_dir / "staging",
        job_dir / "prepared_cache",
        job_dir / "mimics_export" / "work",
    ]
    if remove_partial_model:
        paths.append(Path(request.get("output_model_dir") or ""))
    report: dict[str, Any] = {"removed": [], "not_removed": []}
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if not str(path) or resolved in (ROOT, job_dir, job_dir.parent):
            continue
        try:
            resolved.relative_to(job_dir)
            safe = True
        except ValueError:
            try:
                resolved.relative_to(workspace)
                safe = True
            except ValueError:
                safe = False
        if not safe:
            report["not_removed"].append(
                {"path": str(resolved), "reason": "outside job workspace"}
            )
            continue
        if not resolved.exists():
            continue
        last_error = ""
        for attempt in range(8):
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
                last_error = ""
                break
            except OSError as exc:
                last_error = str(exc)
                time.sleep(min(0.4, 0.04 * (attempt + 1)))
        if resolved.exists():
            report["not_removed"].append(
                {"path": str(resolved), "reason": last_error or "still exists"}
            )
        else:
            report["removed"].append(str(resolved))
    if reset_resume_state:
        for name in (
            "dataset_manifest.json",
            "validation_manifest.json",
            "training_config.json",
            "trainer_status.json",
            "trainer_cancel.request",
        ):
            path = job_dir / name
            try:
                path.unlink()
                report["removed"].append(str(path))
            except FileNotFoundError:
                pass
            except OSError as exc:
                report["not_removed"].append(
                    {"path": str(path), "reason": str(exc)}
                )
    report["completed_at_epoch"] = time.time()
    try:
        _sweep_expired_jobs(workspace)
        _sweep_expired_prepared_cache(workspace)
    except Exception:
        pass
    return report


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}

DEFAULT_PREPARED_CACHE_RETENTION_DAYS = 30


def _sweep_expired_prepared_cache(
    workspace: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prune prepared source-grid cache buckets past their retention window.

    The cache reuses materialized per-case inputs across jobs and can grow to
    tens of GB; everything in it is rebuildable from the source .mcs files.
    ``prepared_cache_retention_days`` (0 disables) controls the window.
    """
    if config is None:
        config = load_config()
    try:
        retention_days = float(
            config.get("prepared_cache_retention_days",
                       DEFAULT_PREPARED_CACHE_RETENTION_DAYS) or 0
        )
    except (TypeError, ValueError):
        retention_days = 0.0
    report: dict[str, Any] = {
        "removed_buckets": [],
        "retention_days": retention_days,
    }
    if retention_days <= 0:
        return report
    root = workspace / "cache" / "source_grid_inputs"
    if not root.is_dir():
        return report
    cutoff = time.time() - retention_days * 86400
    for task_dir in _safe_iterdir(root):
        if task_dir is None or not task_dir.is_dir():
            continue
        for case_dir in _safe_iterdir(task_dir):
            if case_dir is None or not case_dir.is_dir():
                continue
            for bucket in _safe_iterdir(case_dir):
                if bucket is None or not bucket.is_dir():
                    continue
                try:
                    if bucket.stat().st_mtime >= cutoff:
                        continue
                    shutil.rmtree(bucket, ignore_errors=True)
                    report["removed_buckets"].append(
                        str(bucket.relative_to(root))
                    )
                except OSError:
                    continue
        # Drop case dirs that became empty so the tree stays tidy.
        try:
            if task_dir.is_dir() and not any(task_dir.iterdir()):
                task_dir.rmdir()
        except OSError:
            pass
    return report


def _safe_iterdir(path: Path):
    """iterdir that swallows races (dir vanished mid-sweep)."""
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _sweep_expired_jobs(workspace: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prune terminal finetune job dirs older than ``job_retention_days``.

    Keeps ``status.json`` (Model Center history needs it) and deletes the rest
    of each expired job dir (job.log, trainer.log, staging residue). Jobs in
    non-terminal or resumable states are never touched.
    """
    if config is None:
        config = load_config()
    try:
        retention_days = float(config.get("job_retention_days", 0) or 0)
    except (TypeError, ValueError):
        retention_days = 0.0
    report: dict[str, Any] = {"removed_jobs": [], "kept_jobs": 0, "retention_days": retention_days}
    if retention_days <= 0:
        return report
    jobs_path = workspace / "jobs"
    if not jobs_path.is_dir():
        return report
    cutoff = time.time() - retention_days * 86400
    for entry in jobs_path.iterdir():
        if not entry.is_dir():
            continue
        status = read_json(entry / "status.json", {}) or {}
        if str(status.get("status") or "") not in TERMINAL_JOB_STATUSES:
            continue
        completed = float(status.get("completed_at_epoch") or 0)
        if not completed:
            try:
                completed = (entry / "status.json").stat().st_mtime
            except OSError:
                continue
        if completed >= cutoff:
            report["kept_jobs"] += 1
            continue
        for child in entry.iterdir():
            if child.name == "status.json":
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                pass
        report["removed_jobs"].append(entry.name)
    return report


def run_job(job_dir_value: str) -> int:
    job_dir = Path(job_dir_value).expanduser().resolve()
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = read_json(request_path, {}) or {}
    if not request:
        raise RuntimeError("Training request is missing: {}".format(request_path))
    workspace = Path(request["workspace"]).resolve()
    update_status(
        status_path,
        schema_version="nninteractive_task_job.v1",
        job_id=job_dir.name,
        task_id=request.get("task_id"),
        task_name=request.get("task_name"),
        status="validating_cases",
        phase="validating_cases",
        progress_percent=0,
        controller_pid=os.getpid(),
        control_path=str(control_path),
        log_path=str(log_path),
    )
    append_log(log_path, "nnInteractive task training controller started.")
    try:
        manifest_path, validation_path = _prepare_manifest(
            request, job_dir, status_path, control_path, log_path
        )
        action = _control_action(control_path)
        if action in ("pause", "stop", "cancel"):
            raise InterruptedError(action)
        _run_training(
            request,
            job_dir,
            status_path,
            control_path,
            log_path,
            manifest_path,
        )
        model_dir = Path(request["output_model_dir"]).resolve()
        quality = None
        validation_count = 0
        if validation_path is not None:
            validation_manifest = read_json(validation_path, {}) or {}
            validation_count = len(validation_manifest.get("cases") or [])
            baseline_dir = Path(request["base_model_dir"]).resolve()
            evaluation_goal = str(
                request.get("training_goal") or "general"
            )
            initial_probability = {
                "start_empty": 0.0,
                "refine_existing": 1.0,
            }.get(evaluation_goal, 0.3)
            baseline_report = _run_evaluation(
                baseline_dir,
                validation_path,
                job_dir / "baseline_evaluation.json",
                status_path,
                control_path,
                "current model",
                evaluation_goal,
                initial_probability,
            )
            candidate_report = _run_evaluation(
                model_dir,
                validation_path,
                job_dir / "candidate_evaluation.json",
                status_path,
                control_path,
                "new model",
                evaluation_goal,
                initial_probability,
            )
            quality = _quality_result(baseline_report, candidate_report, load_config())
        update_status(
            status_path,
            status="registering",
            phase="registering",
            progress_percent=99,
        )
        model, selected = _register_model(
            request, workspace, model_dir, quality, validation_count
        )
        outcome = (
            "new_model_selected"
            if selected
            else (
                "current_model_retained"
                if quality
                else "model_saved_without_automatic_selection"
            )
        )
        update_status(
            status_path,
            status="completed",
            phase="completed",
            model=model,
            quality=quality or {},
            selection_outcome=outcome,
            completed_at_epoch=time.time(),
            progress_percent=100,
        )
        append_log(log_path, "Training completed: {}.".format(outcome))
        # Log a human-readable summary
        trainer_status = read_json(job_dir / "trainer_status.json", {}) or {}
        current = read_json(status_path, {}) or {}
        best_epoch = None
        hist = trainer_status.get("metrics_history", current.get("metrics_history") or [])
        if hist:
            best_epoch = max(hist, key=lambda x: float(x.get("validation_auc") or 0))
        best_msg = (
            "Best epoch {}/{} AUC {:.4f}".format(
                best_epoch.get("epoch"), best_epoch.get("epochs") or "?",
                float(best_epoch.get("validation_auc") or 0),
            ) if best_epoch else ""
        )
        dur = int(trainer_status.get("elapsed_seconds", current.get("elapsed_seconds", 0)))
        selected_tag = "NEW MODEL SELECTED" if selected else "current model retained"
        append_log(log_path, "Summary: {} epochs, {:.1f}m, best AUC {:.4f} | {} | {}".format(
            trainer_status.get("epochs") or current.get("epochs") or "?",
            dur / 60,
            trainer_status.get("best_score", current.get("best_score", 0)) or 0,
            best_msg,
            selected_tag,
        ))
        cleanup = _cleanup_terminal_artifacts(
            request,
            job_dir,
            remove_partial_model=False,
        )
        update_status(status_path, artifact_cleanup=cleanup)
        return 0
    except InterruptedError as exc:
        action = str(exc) or _control_action(control_path)
        final = "paused" if action == "pause" else "cancelled"
        update_status(
            status_path,
            status=final,
            phase=final,
            completed_at_epoch=time.time(),
        )
        append_log(log_path, "Training {} by user request.".format(final))
        if final == "cancelled":
            cleanup = _cleanup_terminal_artifacts(
                request,
                job_dir,
                remove_partial_model=True,
            )
            update_status(status_path, artifact_cleanup=cleanup)
        return 0
    except (ResourceLockCancelled,):
        action = _control_action(control_path)
        final = "paused" if action == "pause" else "cancelled"
        update_status(status_path, status=final, phase=final)
        if final == "cancelled":
            cleanup = _cleanup_terminal_artifacts(
                request,
                job_dir,
                remove_partial_model=True,
            )
            update_status(status_path, artifact_cleanup=cleanup)
        return 0
    except Exception as exc:
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
        )
        append_log(log_path, "Training failed: {}: {}.".format(type(exc).__name__, exc))
        if not bool(load_config().get("keep_failed_training_artifacts", False)):
            cleanup = _cleanup_terminal_artifacts(
                request,
                job_dir,
                remove_partial_model=True,
                reset_resume_state=True,
            )
            update_status(status_path, artifact_cleanup=cleanup)
            if cleanup.get("not_removed"):
                append_log(
                    log_path,
                    "Some failed-job artifacts could not be removed; see artifact_cleanup in status.json.",
                )
            else:
                append_log(
                    log_path,
                    "Removed rebuildable failed-job data; logs and status were retained.",
                )
        return 1


def resume_job(job_dir_value: str) -> int:
    job_dir = Path(job_dir_value).expanduser().resolve()
    status_path = job_dir / "status.json"
    status = read_json(status_path, {}) or {}
    if str(status.get("status") or "") not in {"paused", "failed"}:
        raise RuntimeError("Only paused or failed training can be resumed.")
    for path in (
        job_dir / "control.json",
        job_dir / "trainer_cancel.request",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    update_status(
        status_path,
        status="created",
        phase="resuming",
        error="",
        traceback="",
        resumed_at_epoch=time.time(),
    )
    return run_job(str(job_dir))


def _tail_text(path: Path, max_bytes: int = 2048) -> str:
    """Return the last ``max_bytes`` of a text file, or '' when missing."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(max(0, size - max_bytes))
                handle.readline()  # drop the partial first line
            text = handle.read().decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
    except OSError:
        return ""


def diagnose_job(job_dir_value: str) -> dict[str, Any]:
    """Aggregate everything needed to understand a failed training job.

    Read by the Model Center failed-state panel.  Pure inspection: nothing
    is written, and unreadable pieces degrade to empty strings instead of
    raising, so a half-cleaned job directory still yields a diagnosis.
    """
    job_dir = Path(job_dir_value).expanduser().resolve()
    status = read_json(job_dir / "status.json", {}) or {}
    request = read_json(job_dir / "request.json", {}) or {}

    error = str(status.get("error") or "").strip()
    error_line = ""
    if error:
        # The full traceback lives in status.json; surface the last frame,
        # which names the failing function and line.
        frames = [
            line.strip()
            for line in str(status.get("traceback") or "").splitlines()
            if line.strip().startswith('File "')
        ]
        error_line = frames[-1] if frames else ""

    stage_chain = []
    for source, key in (
        ("status", "status"),
        ("status", "phase"),
        ("trainer_status", "status"),
    ):
        if source == "status":
            value = str(status.get(key) or "").strip()
        else:
            value = str(
                (read_json(job_dir / "trainer_status.json", {}) or {}).get(key) or ""
            ).strip()
        if value and value not in stage_chain:
            stage_chain.append(value)

    artifacts = {}
    for name in (
        "request.json",
        "status.json",
        "job.log",
        "trainer.log",
        "trainer_status.json",
        "dataset_manifest.json",
        "validation_manifest.json",
        "training_config.json",
        "control.json",
    ):
        artifacts[name] = (job_dir / name).is_file()

    gpu_wait = bool(status.get("gpu_lock_wait_seconds"))
    label_export = bool(
        status.get("label_export_progress")
        and status.get("status") == "exporting_labels"
    )
    if not error:
        if status.get("status") in ("paused", "cancelled"):
            hint = "Training was stopped by user request; no failure occurred."
        else:
            hint = "No error was recorded for this job."
    elif gpu_wait or status.get("status") == "waiting_for_gpu":
        hint = "Failed while waiting for the GPU; another task may have held the lock."
    elif label_export:
        hint = "Failed while reading saved Masks; check that every case still contains the target Mask."
    elif "out of memory" in error.lower() or "cuda" in error.lower():
        hint = "Failed with a GPU memory error; close other GPU programs or reduce concurrent tasks."
    elif artifacts.get("job.log") is False:
        hint = "The job log is missing; the job directory may have been cleaned."
    else:
        hint = "See the log tail below for the last activity before the failure."

    return {
        "schema_version": "nninteractive_job_diagnosis.v1",
        "job_dir": str(job_dir),
        "status": str(status.get("status") or ""),
        "error": error,
        "error_line": error_line,
        "stage_chain": stage_chain,
        "execution_backend": str(status.get("execution_backend") or ""),
        "epoch": status.get("epoch"),
        "epochs": status.get("epochs"),
        "artifacts": artifacts,
        "artifact_cleanup": status.get("artifact_cleanup") or {},
        "job_log_tail": _tail_text(job_dir / "job.log"),
        "trainer_log_tail": _tail_text(job_dir / "trainer.log"),
        "hint": hint,
        "request_summary": {
            "task_id": request.get("task_id"),
            "epochs": request.get("epochs"),
            "strategy": request.get("strategy"),
            "case_count": len(request.get("cases") or []),
        },
        "generated_at_epoch": time.time(),
    }


def request_control(job_dir_value: str, action: str) -> int:
    job_dir = Path(job_dir_value).expanduser().resolve()
    action = str(action).strip().lower()
    if action not in {"pause", "stop", "cancel"}:
        raise ValueError("Unsupported control action: {}".format(action))
    write_json_atomic(
        job_dir / "control.json",
        {"action": action, "requested_at_epoch": time.time()},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--job-dir", required=True)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--job-dir", required=True)
    control_parser = subparsers.add_parser("control")
    control_parser.add_argument("--job-dir", required=True)
    control_parser.add_argument("--action", required=True, choices=("pause", "stop", "cancel"))
    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("--job-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Startup sweep: prune terminal jobs and stale prepared-cache buckets
    # past their retention windows no matter which subcommand runs (both are
    # no-ops when the config keys are absent or zero).
    try:
        _sweep_expired_jobs(workspace_root())
        _sweep_expired_prepared_cache(workspace_root())
    except Exception:
        pass
    if args.command == "run":
        return run_job(args.job_dir)
    if args.command == "resume":
        return resume_job(args.job_dir)
    if args.command == "diagnose":
        print(json.dumps(diagnose_job(args.job_dir), indent=2))
        return 0
    return request_control(args.job_dir, args.action)


if __name__ == "__main__":
    raise SystemExit(main())
