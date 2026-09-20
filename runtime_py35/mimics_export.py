# -*- coding: utf-8 -*-
"""Mimics-internal driver for .mcs mask -> dataset export.

Runs inside Mimics Python 3.5.2. Only uses stdlib + mimics API.
Calls mimics_bridge.py (in nninteractive_env) via subprocess for NIfTI writing.

Uses Win32 SetTimer / PyQt5 QTimer for non-blocking async polling,
matching nnInteractive's pattern; no manual second click needed.

Flow:
    1. Export all masks from current Mimics project as .u8 buffers
    2. Launch mimics_bridge.py "convert" in background (non-blocking)
    3. Timer polls every 0.5s until bridge completes
    4. Auto-apply result: clean up buffers, show completion
    5. For batch: timer processes cases one-by-one automatically
"""

from __future__ import print_function

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import mimics

import dataset_manifest
import runtime_common


# -- Global async monitor state ----------------------------------------
_EXPORT_MONITORS = {}
_EXPORT_LAUNCH_THREADS = []
EXPORT_STOP_FILE = "_export_stop.json"
_LOG_ROTATE_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_BACKUPS = 3
_CONFIG_CACHE = None
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
# Grid snapshot stored at import time (see create_mcs_batch.py). When the
# source image file is missing, these two let the bridge resample back to the
# exact original grid without the file itself.
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
# Track the background_mimics lock taken by the launcher so the status monitor
# can release it when the child Mimics crashes without writing a status (which
# otherwise leaves a stale lock blocking later writes to that destination).
_ACTIVE_BG_MIMICS_LOCKS = {}
_ACTIVE_BG_MIMICS_LOCK_MUTEX = threading.RLock()


def _record_exported_labels(
    manifest_root, case_id, result, source_image_path="", mcs_path=""
):
    """Record only labels that exist and were successfully exported."""
    labels = []
    for row in (result or {}).get("exported") or []:
        if row.get("action") not in ("new", "overwritten", "unchanged"):
            continue
        path = str(row.get("path") or "")
        if not path or not os.path.isfile(path):
            continue
        labels.append(
            {
                "mask_name": row.get("name") or "",
                "output_name": os.path.basename(path).replace(
                    ".nii.gz", ""
                ).replace(".nii", ""),
                "path": path,
                "shape": (result or {}).get("export_shape") or [],
                "voxel_to_ras_matrix": (
                    (result or {}).get("export_voxel_to_ras_matrix") or []
                ),
                "export_space": (result or {}).get("export_space") or "",
                "source": "mimics_export",
                "mask_source": row.get("mask_source") or "mcs_mask_buffer",
                "source_mask_path": row.get("source_mask_path") or "",
                "source_mask_sha256": row.get("source_mask_sha256") or "",
                "mcs_mask_bypassed": bool(row.get("mcs_mask_bypassed", False)),
            }
        )
    if not labels:
        return ""
    return dataset_manifest.update_case(
        manifest_root,
        case_id,
        image_path=source_image_path,
        mcs_path=mcs_path,
        labels=labels,
        source_geometry={
            "shape": (result or {}).get("export_shape") or [],
            "voxel_to_ras_matrix": (
                (result or {}).get("export_voxel_to_ras_matrix") or []
            ),
            "world_coordinate_system": "RAS",
        },
        mimics_geometry={
            "voxel_to_ras_matrix": (
                (result or {}).get("mimics_voxel_to_ras_matrix") or []
            ),
            "world_coordinate_system": "RAS",
        },
        provenance={
            "last_operation": "mimics_export",
            "export_space": (result or {}).get("export_space") or "",
            "mcs_mask_bypassed": bool((result or {}).get("mcs_mask_bypassed", False)),
            "source_mask_paths": (result or {}).get("source_mask_paths") or {},
        },
    )


def _write_export_task_status(monitor, values):
    path = monitor.get("task_status_path") if monitor else ""
    if not path:
        return
    try:
        payload = runtime_common.read_json(path, {}) or {}
        payload.update(values or {})
        payload["updated_at_epoch"] = time.time()
        runtime_common.write_json_atomic(path, payload)
    except Exception:
        # Progress telemetry must never fail or delay the export itself.
        pass


def _export_task_stop_requested(monitor):
    path = monitor.get("task_stop_path") if monitor else ""
    return bool(path and os.path.isfile(path))


def _set_active_background_export_lock(lock_path, lock_token):
    with _ACTIVE_BG_MIMICS_LOCK_MUTEX:
        _ACTIVE_BG_MIMICS_LOCKS[str(lock_token)] = str(lock_path)


def _active_background_export_lock():
    with _ACTIVE_BG_MIMICS_LOCK_MUTEX:
        if not _ACTIVE_BG_MIMICS_LOCKS:
            return (None, None)
        token, path = next(iter(_ACTIVE_BG_MIMICS_LOCKS.items()))
        return (path, token)


def _clear_active_background_export_lock(lock_token=None):
    with _ACTIVE_BG_MIMICS_LOCK_MUTEX:
        if lock_token is None:
            _ACTIVE_BG_MIMICS_LOCKS.clear()
            return True
        if str(lock_token) not in _ACTIVE_BG_MIMICS_LOCKS:
            return False
        _ACTIVE_BG_MIMICS_LOCKS.pop(str(lock_token), None)
        return True


def _release_background_export_lock():
    """Release the background_mimics lock if its holder process is dead.

    Called when the background Mimics child exits without writing a status.
    A crash (common with Mimics 21 when a -b instance loses the license to the
    interactive window) leaves the lock file behind with a dead PID, blocking
    later exports to that destination. Only remove the lock when the recorded
    PID is gone, so a genuinely running export is never disturbed.
    """
    with _ACTIVE_BG_MIMICS_LOCK_MUTEX:
        locks = list(_ACTIVE_BG_MIMICS_LOCKS.items())
    for token, lock_path in locks:
        payload = runtime_common.read_json(lock_path, {}) or {}
        pid = payload.get("pid")
        if pid and runtime_common.process_exists(pid):
            continue
        try:
            runtime_common.release_resource_lock(lock_path, token)
        except Exception:
            pass
        _clear_active_background_export_lock(token)


def _finalize_background_export_process_status(
    process, status_path=None, stop_path=None, handshake_path=None
):
    """Write a terminal export state after the child has definitely exited."""
    status_path = (
        status_path or getattr(process, "_mimics_status_path", "") or ""
    )
    if not status_path:
        return {}
    status = runtime_common.read_json(status_path, {}) or {}
    if status.get("status") in ("closed", "cancelled", "failed"):
        return status
    try:
        exit_code = process.poll()
    except Exception:
        exit_code = getattr(process, "returncode", None)
    stop_path = stop_path or getattr(process, "_mimics_stop_path", "") or ""
    stopped = bool(stop_path and os.path.isfile(stop_path))
    handshake_path = (
        handshake_path
        or getattr(process, "_mimics_handshake_path", "")
        or ""
    )
    runner_started = bool(handshake_path and os.path.isfile(handshake_path))
    status.update({
        "status": "cancelled" if stopped else "failed",
        "phase": "cancelled" if stopped else "process_exited",
        "pid": getattr(process, "pid", 0),
        "exit_code": exit_code,
        "runner_started": runner_started,
        "updated_at_epoch": time.time(),
    })
    if stopped:
        status["error"] = "Background mask export stopped by user request."
    elif runner_started:
        status["error"] = (
            "The export runner started, but the export script stopped before "
            "recording a terminal state."
        )
    else:
        status["error"] = (
            "Background Mimics exited before executing the export runner."
        )
    try:
        runtime_common.write_json_atomic(status_path, status)
    except Exception:
        pass
    return status


def _watch_background_export_process(
    process, lock_path=None, lock_token=None, lock_records=None
):
    """Finalize status, then release the shared background-Mimics slot."""
    while True:
        try:
            try:
                process.wait(timeout=30.0)
            except TypeError:
                process.wait()
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            try:
                if process.poll() is not None:
                    break
            except Exception:
                pass
            time.sleep(5.0)
            continue
        break
    _finalize_background_export_process_status(process)
    records = list(lock_records or [])
    if not records and lock_path and lock_token:
        records = [(lock_path, lock_token)]
    for current_path, current_token in records:
        try:
            runtime_common.release_resource_lock(current_path, current_token)
        except Exception:
            pass
        _clear_active_background_export_lock(current_token)

# All intermediate / scratch files are placed under this subdirectory inside
# the user-visible output folder so they do not clutter exported segmentations.
_RUNTIME_SUBDIR = ".mimics_runtime"


def _rt(output_dir, *parts):
    """Return output_dir/.mimics_runtime/<joined parts>.

    Keeps work dirs, job state, logs, and failure records out of the
    user-visible output folder that only contains segmentation files.
    """
    return os.path.join(output_dir, _RUNTIME_SUBDIR, *parts)


def _prune_local_export_jobs(max_age_days=14, max_jobs=100):
    root = os.path.join(_project_root(), ".mimics_runtime", "export_jobs")
    if not os.path.isdir(root):
        return
    now = time.time()
    rows = []
    try:
        names = os.listdir(root)
    except Exception:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            rows.append((os.path.getmtime(path), path))
        except OSError:
            continue
    rows.sort(reverse=True)
    for index, (modified, path) in enumerate(rows):
        if index < int(max_jobs) and now - modified <= float(max_age_days) * 86400.0:
            continue
        if _local_export_job_is_active(path):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _local_export_job_is_active(path):
    absolute = os.path.abspath(path)
    for monitor in list(_EXPORT_MONITORS.values()):
        try:
            if monitor.get("done"):
                continue
            work_dir = monitor.get("work_dir") or monitor.get("job_runtime")
            if work_dir and os.path.abspath(work_dir) == absolute:
                return True
        except Exception:
            continue
    for relative in (
        "status.json",
        "runner_started.json",
        os.path.join("bridge", "job_state.json"),
    ):
        payload = runtime_common.read_json(os.path.join(path, relative), {}) or {}
        status = str(payload.get("status") or "").lower()
        if status in ("closed", "completed", "failed", "cancelled", "error"):
            continue
        pid = payload.get("pid")
        if pid and runtime_common.process_exists(pid):
            return True
    return False


_write_json_atomic = runtime_common.write_json_atomic
_safe_case_filename = runtime_common.safe_filename
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return True
    except Exception:
        try:
            print(message)
        except Exception:
            pass
        return False


def _rotate_log_file(path, max_bytes=_LOG_ROTATE_BYTES, backups=_LOG_ROTATE_BACKUPS):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < max_bytes:
            return
        backups = int(backups)
        if backups <= 0:
            os.remove(path)
            return
        oldest = "{0}.{1}".format(path, backups)
        if os.path.isfile(oldest):
            os.remove(oldest)
        for index in range(backups - 1, 0, -1):
            src = "{0}.{1}".format(path, index)
            dst = "{0}.{1}".format(path, index + 1)
            if os.path.isfile(src):
                os.rename(src, dst)
        os.rename(path, path + ".1")
    except Exception:
        pass


def _append_export_log(root_dir, message):
    text = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(text)
    try:
        if root_dir and not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        path = os.path.join(root_dir or os.getcwd(), _RUNTIME_SUBDIR, "mimics_export.log")
        _rotate_log_file(path)
        with open(path, "a") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _record_failed_case(output_dir_or_export_root, case_id, phase, error):
    try:
        failed_dir = _rt(output_dir_or_export_root or os.getcwd(), "_failed_exports")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir)
        payload = {
            "case_id": str(case_id or "unknown"),
            "phase": str(phase or "unknown"),
            "error": str(error or ""),
            "failed_at_epoch": time.time(),
        }
        filename = "{0}_{1}.json".format(_safe_case_filename(case_id), _safe_case_filename(phase))
        _write_json_atomic(os.path.join(failed_dir, filename), payload)
    except Exception:
        pass


def _await_degraded_confirmation(status_path, confirm_path, request_id, stop_path,
                                 export_root, progress, poll_seconds=2.0,
                                 timeout_seconds=7 * 86400):
    """Pause the batch until the foreground Mimics answers the degraded-export
    question for this batch (one prompt total).

    Returns True (export on the Mimics grid), False (declined / timed out) or
    None (stop requested while waiting). The foreground writes confirm_path in
    response to the "waiting_degraded_confirm" status written here.
    """
    deadline = time.time() + timeout_seconds
    _write_json_atomic(
        status_path,
        {
            "status": "waiting_degraded_confirm",
            "phase": "waiting_degraded_confirm",
            "pid": os.getpid(),
            "case_id": progress.get("case_id", ""),
            "index": progress.get("index", 0),
            "total": progress.get("total", 0),
            "completed": progress.get("completed", 0),
            "failed": progress.get("failed", 0),
            "request_id": request_id,
            "confirm_path": confirm_path,
            "updated_at_epoch": time.time(),
        },
    )
    _append_export_log(
        export_root,
        "Waiting for user confirmation of degraded Mimics-grid export for this batch.",
    )
    while time.time() < deadline:
        if os.path.isfile(stop_path):
            _append_export_log(export_root, "Mask export stopped while waiting for the degraded-export confirmation.")
            return None
        try:
            response = runtime_common.read_json(confirm_path, {})
        except Exception:
            response = {}
        if response.get("request_id") == request_id and response.get("answered"):
            accepted = bool(response.get("accepted"))
            _append_export_log(
                export_root,
                "User {0} the degraded Mimics-grid export for this batch.".format(
                    "accepted" if accepted else "declined"
                ),
            )
            return accepted
        time.sleep(poll_seconds)
    _append_export_log(export_root, "No degraded-export confirmation arrived in time; treating as declined.")
    return False


# -- Path helpers (shared with mimics_import.py) ------------------------


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def _load_data_io_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    merged = {}
    io_path = os.path.join(_project_root(), "mimics_io_config.json")
    nn_path = os.path.join(_project_root(), "nninteractive_config.json")
    for path in (io_path, nn_path):
        try:
            with open(path, "r") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                merged.update(loaded)
        except Exception:
            pass
    _CONFIG_CACHE = merged
    return _CONFIG_CACHE


def _resolve_export_output_dir(ts_root):
    default_dir = os.path.join(ts_root, "mcs_output")
    config = _load_data_io_config()
    configured = config.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if not os.path.isabs(configured):
        configured = os.path.abspath(os.path.join(ts_root, configured))
    else:
        configured = os.path.abspath(configured)
    try:
        if not os.path.isdir(configured):
            os.makedirs(configured)
        return configured
    except Exception:
        return default_dir


def _resolve_buffer_mapping():
    config = _load_data_io_config()
    axes = config.get("mimics_buffer_axes", [0, 1, 2])
    flips = config.get("mimics_buffer_flips", [False, False, False])
    try:
        axes = [int(value) for value in axes]
        flips = [bool(value) for value in flips]
        if sorted(axes) != [0, 1, 2] or len(flips) != 3:
            raise ValueError("invalid mapping")
        return axes, flips
    except Exception:
        return [0, 1, 2], [False, False, False]


def _resource_lock_path(name):
    return runtime_common.resource_lock_path(_project_root(), name)


def _environment_root():
    root = _project_root()
    candidates = [
        os.path.join(root, "python_env"),
        os.path.join(os.path.dirname(root), "python_env"),
        os.path.join(root, "nninteractive_env"),
        os.path.join(os.path.dirname(root), "nninteractive_env"),
        root,
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "python.exe")):
            return candidate
        if os.path.isfile(os.path.join(candidate, "Scripts", "python.exe")):
            return candidate
        if os.path.isfile(os.path.join(candidate, "python", "python.exe")):
            return candidate
    return candidates[0]


def _bridge_script():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "mimics_bridge.py"),
        os.path.join(here, "mimics_bridge.py"),
        os.path.join(_project_root(), "adapters", "mimics", "mimics_bridge.py"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return os.path.abspath(candidates[0])


def _background_env(extra=None):
    return runtime_common.background_env(extra)


def _python_exe():
    env_root = _environment_root()
    candidates = [
        os.path.join(env_root, "python.exe"),
        os.path.join(env_root, "Scripts", "python.exe"),
        os.path.join(env_root, "python", "python.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def _find_mimics_exe():
    """Find MimicsResearch.exe installation path."""
    return runtime_common.find_mimics_exe()


def _launch_background_batch_export(ts_root, cases_filter, axes, flips, label_output_root=None,
                                    overwrite_existing=False, case_dirs=None, mcs_output_dir=None,
                                    mcs_paths=None, source_image_paths=None, mask_names=None,
                                    export_formats=None):
    """Launch batch export in a separate background Mimics process."""
    output_dir = os.path.abspath(mcs_output_dir) if mcs_output_dir else _resolve_export_output_dir(ts_root)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    export_root = os.path.abspath(label_output_root or output_dir)
    mimics_exe = _find_mimics_exe()
    if not mimics_exe:
        _append_export_log(
            output_dir,
            "A separate background Mimics executable was not found. Configure "
            "MIMICS_BACKGROUND_EXE; the open MimicsMedical.exe is not reused automatically.",
        )
        return None

    _prune_local_export_jobs()
    here = os.path.dirname(os.path.abspath(__file__))
    job_id = "export_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_runtime = os.path.join(
        _project_root(), ".mimics_runtime", "export_jobs", job_id
    )
    if not os.path.isdir(job_runtime):
        os.makedirs(job_runtime)
    config_path = os.path.join(job_runtime, "export_config.json")
    runner_path = os.path.join(job_runtime, "run_export.py")
    status_path = os.path.join(job_runtime, "status.json")
    handshake_path = os.path.join(job_runtime, "runner_started.json")
    mimics_log_path = os.path.join(job_runtime, "mimics_application.log")
    stop_path = os.path.join(job_runtime, EXPORT_STOP_FILE)
    _write_json_atomic(
        config_path,
        {
            "ts_root": os.path.abspath(ts_root),
            "output_dir": os.path.abspath(output_dir),
            "cases": sorted(list(cases_filter)) if cases_filter else None,
            "axes": axes,
            "flips": flips,
            "label_output_root": os.path.abspath(label_output_root) if label_output_root else "",
            "export_root": job_runtime,
            "display_output_root": export_root,
            "overwrite_existing": bool(overwrite_existing),
            "case_dirs": dict(case_dirs or {}),
            "mcs_paths": dict(mcs_paths or {}),
            "source_image_paths": dict(source_image_paths or {}),
            "mask_names": list(mask_names or []),
            "export_formats": [
                str(fmt).lower().lstrip(".")
                for fmt in (export_formats or [])
                if str(fmt).strip()
            ],
            "status_path": status_path,
            "job_runtime": job_runtime,
            "stop_path": stop_path,
        },
    )
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated runner for background Mimics batch export\n")
        f.write("import sys, os, json, time\n")
        f.write(
            "open({0}, 'w').write(json.dumps({{'pid': os.getpid(), "
            "'started_at_epoch': time.time()}}))\n".format(json.dumps(handshake_path))
        )
        f.write("sys.path.insert(0, {0})\n".format(json.dumps(here)))
        f.write("import mimics_export\n")
        f.write(
            "mimics_export.run_background_batch_export({0})\n".format(
                json.dumps(config_path)
            )
        )

    log_path = os.path.join(job_runtime, "process.log")
    _rotate_log_file(log_path)
    # Protect both sides of the operation: import may still be replacing a
    # project in output_dir, while another export may be writing the final
    # labels. When no safe-copy root is supplied, do_convert writes under
    # ts_root regardless of where the .mcs projects are stored.
    destination_scope = os.path.abspath(label_output_root or ts_root)
    lock_paths = sorted(set([
        runtime_common.background_mimics_lock_path(_project_root(), output_dir),
        runtime_common.background_mimics_lock_path(_project_root(), destination_scope),
    ]))
    lock_records = []
    for lock_path in lock_paths:
        lock_token = runtime_common.acquire_resource_lock(
            lock_path,
            "background_mimics",
            "batch label export",
            wait_seconds=0.0,
        )
        if not lock_token:
            for held_path, held_token in reversed(lock_records):
                runtime_common.release_resource_lock(held_path, held_token)
                _clear_active_background_export_lock(held_token)
            _append_export_log(
                output_dir,
                "The .mcs source or label destination is busy; batch export was not started.",
            )
            return None
        lock_records.append((lock_path, lock_token))
        _set_active_background_export_lock(lock_path, lock_token)
    try:
        if os.path.isfile(stop_path):
            os.remove(stop_path)
    except OSError:
        pass
    log_handle = None
    process = None
    try:
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe,
                runner_path,
                mimics_log_path=mimics_log_path,
            ),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_background_process_kwargs()
        )
        process._mimics_status_path = status_path
        process._mimics_handshake_path = handshake_path
        process._mimics_log_path = mimics_log_path
        process._mimics_process_log_path = log_path
        process._mimics_job_runtime = job_runtime
        process._mimics_stop_path = stop_path
        for lock_path, lock_token in lock_records:
            if not runtime_common.update_resource_lock_pid(
                lock_path,
                lock_token,
                process.pid,
                {
                    "kind": "export_labels",
                    "ts_root": os.path.abspath(ts_root),
                    "mcs_source": output_dir,
                    "export_root": export_root,
                    "destination_scope": destination_scope,
                    "stop_path": stop_path,
                },
            ):
                raise RuntimeError(
                    "Background export started, but lock ownership could not be transferred to PID {0}.".format(
                        process.pid
                    )
                )
        _append_export_log(
            export_root,
            "Background Mimics launch requested (PID={0}) for mask export; "
            "waiting for runner handshake. Job: {1}".format(process.pid, job_id),
        )
        watcher = threading.Thread(
            target=_watch_background_export_process,
            args=(process, None, None, lock_records),
            name="MimicsBackgroundExportWatcher",
        )
        watcher.daemon = True
        watcher.start()
        return process
    except Exception as exc:
        process_stopped = process is None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5.0)
                except Exception:
                    pass
            try:
                process_stopped = process.poll() is not None
            except Exception:
                process_stopped = not runtime_common.process_exists(
                    getattr(process, "pid", 0)
                )
        if process_stopped:
            for lock_path, lock_token in lock_records:
                runtime_common.release_resource_lock(lock_path, lock_token)
                _clear_active_background_export_lock(lock_token)
        else:
            for lock_path, lock_token in lock_records:
                try:
                    runtime_common.update_resource_lock_pid(
                        lock_path,
                        lock_token,
                        process.pid,
                        {
                            "kind": "export_labels",
                            "ts_root": os.path.abspath(ts_root),
                            "mcs_source": output_dir,
                            "export_root": export_root,
                            "destination_scope": destination_scope,
                            "stop_path": stop_path,
                            "termination_pending": True,
                        },
                    )
                except Exception:
                    pass
            _append_export_log(
                output_dir,
                "Background Mimics PID {0} did not exit after launch failure; "
                "the shared lock was retained.".format(getattr(process, "pid", "?")),
            )
        _append_export_log(output_dir, "Could not start background batch export: {0}".format(exc))
        return None
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


def call_bridge(params, extra_env=None):
    """Call mimics_bridge.py via subprocess, return parsed JSON result."""
    python_exe = _python_exe()
    bridge = _bridge_script()

    process = subprocess.Popen(
        [python_exe, bridge],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_background_env(extra_env),
        **_background_process_kwargs()
    )
    stdin_data = json.dumps(params).encode("utf-8")
    try:
        stdout, stderr = process.communicate(input=stdin_data, timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("mimics_bridge.py timed out")

    if process.returncode != 0:
        err = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError("mimics_bridge.py failed (exit {}): {}".format(process.returncode, err))

    try:
        return json.loads(stdout.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            "mimics_bridge.py returned invalid JSON: {}\nstdout: {}\nstderr: {}".format(
                e, stdout.decode("utf-8", "replace")[:500], stderr.decode("utf-8", "replace")[:500]
            )
        )


# -- Async bridge helpers ----------------------------------------------

def _launch_bridge_background(bridge_params, job_dir):
    """Launch mimics_bridge.py in background. Don't wait for completion."""
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)

    input_file = os.path.join(job_dir, "bridge_input.json")
    result_file = os.path.join(job_dir, "bridge_result.json")
    error_file = os.path.join(job_dir, "bridge_error.log")

    with open(input_file, "w") as f:
        json.dump(bridge_params, f)

    python_exe = _python_exe()
    bridge = _bridge_script()

    stdin_handle = open(input_file, "r")
    stdout_handle = open(result_file, "w")
    stderr_handle = open(error_file, "w")
    try:
        process = subprocess.Popen(
            [python_exe, bridge],
            stdin=stdin_handle,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=_background_env(),
            **_background_process_kwargs()
        )
    finally:
        stdin_handle.close()
        stdout_handle.close()
        stderr_handle.close()
    return process


def _is_pid_alive(pid):
    """Check if a process with given PID is still running.

    Delegates to runtime_common.process_exists, which declares pointer-sized
    HANDLE ctypes signatures so the 64-bit process handle is not truncated to
    c_int (which intermittently reports a live process as dead)."""
    return runtime_common.process_exists(pid)


def _check_job_status(job_dir):
    """Check status of a background bridge job.

    Returns ("running", None) | ("done", result_dict) | ("error", error_msg).
    """
    result_file = os.path.join(job_dir, "bridge_result.json")
    state_file = os.path.join(job_dir, "job_state.json")
    error_file = os.path.join(job_dir, "bridge_error.log")

    # Try to read result
    if os.path.isfile(result_file):
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
            if result.get("status") == "ok":
                return ("done", result)
            else:
                return ("error", result.get("error", "bridge returned non-ok status"))
        except (ValueError, IOError):
            pass  # File incomplete - process may still be writing

    # No valid result - check if process is alive
    pid = None
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            pid = state.get("pid")
        except (ValueError, IOError):
            pass

    if pid and _is_pid_alive(pid):
        return ("running", None)

    # Process dead but no valid result
    err_msg = "bridge process exited unexpectedly"
    if os.path.isfile(error_file):
        try:
            with open(error_file, "r") as f:
                err_msg = f.read()[:500]
        except Exception:
            pass
    return ("error", err_msg)


def _cleanup_job_dir(job_dir):
    """Remove a job directory after processing."""
    if not job_dir or not os.path.isdir(job_dir):
        return
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass


# -- Timer-based async monitor (same pattern as nnInteractive) ----------

def _allow_windows_event_monitor():
    """Opt-in switch for Mimics event-timer usage on Windows.

    Certain Mimics versions emit ``Subscription.__del__ AttributeError:
    'Subscription' object has no attribute 'subscription'`` when the RAII
    subscription object is collected after an explicit ``unsubscribe()``.
    Prefer Win32 SetTimer / PyQt QTimer by default and only enable Mimics
    event subscriptions when this flag is set, matching mimics_import and
    io_setup_mimics.
    """
    value = os.environ.get("MIMICS_USE_EVENT_TIMER", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _stop_export_monitor(monitor_key):
    """Stop and clean up a running export monitor."""
    monitor = _EXPORT_MONITORS.pop(monitor_key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    try:
        if timer is not None and timer.isActive():
            timer.stop()
    except Exception:
        pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        user32, timer_id = win32_timer
        try:
            user32.KillTimer(None, timer_id)
        except Exception:
            pass

    subscription = monitor.get("event_subscription")
    if subscription is not None:
        try:
            subscription.unsubscribe()
        except Exception:
            pass


def _start_mimics_event_export_monitor(monitor, tick, poll_seconds, error_context):
    """Run export polling through the documented Mimics timer notification."""
    try:
        events = getattr(mimics, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return False
        interval = max(0.1, float(poll_seconds))
        key = monitor.get("monitor_key")
        _stop_export_monitor(key)
        monitor["event_last_tick"] = 0.0

        def callback(*_args, **_kwargs):
            now = time.time()
            if now - float(monitor.get("event_last_tick", 0.0)) < interval:
                return
            monitor["event_last_tick"] = now
            try:
                tick()
            except Exception as exc:
                _mimics_log(logging.ERROR, "{0}: {1}".format(error_context, exc))

        subscription = subscribe("timer", callback)
        if subscription is None:
            return False
        monitor["event_callback"] = callback
        monitor["event_subscription"] = subscription
        _EXPORT_MONITORS[key] = monitor
        return True
    except Exception:
        return False


def _export_monitor_tick(monitor):
    """Timer callback: check if bridge finished, apply result if done."""
    if monitor.get("done"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_export_monitor(monitor_key)
        mimics.dialogs.message_box(
            title="Export Timeout",
            message="Data export timed out. Please retry.",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    monitor["done"] = True
    _stop_export_monitor(monitor_key)

    if status == "error":
        mimics.dialogs.message_box(title="Export Error", message="Export failed: {0}".format(result))
        _cleanup_job_dir(job_dir)
        if monitor.get("batch_queue"):
            _start_next_batch_export(monitor)
        return

    # status == "done"
    try:
        work_dir = monitor.get("work_dir")
        total_new, total_overwritten, total_unchanged = _apply_export_result(result, work_dir)
        _cleanup_job_dir(job_dir)
        print("  export complete")
    except Exception as e:
        print("  export failed: {0}".format(e))
        traceback.print_exc()
        mimics.dialogs.message_box(title="Export Error", message="Export failed: {0}".format(e))
        _cleanup_job_dir(job_dir)
        if monitor.get("batch_queue"):
            _start_next_batch_export(monitor)
        return

    # Single case: show completion
    if not monitor.get("batch_queue"):
        case_dir = monitor.get("case_dir")
        mimics.dialogs.message_box(
            title="Export Complete",
            message="Exported to: {0}\nNew: {1}, overwritten: {2}, unchanged: {3}".format(
                os.path.join(case_dir, "segmentations"),
                total_new, total_overwritten, total_unchanged
            ),
        )
        return

    # Batch: update progress and start next case
    monitor["completed"] = monitor.get("completed", 0) + 1
    _start_next_batch_export(monitor)


def _start_next_batch_export(monitor):
    """Start bridge for the next case in the batch queue."""
    queue = monitor.get("batch_queue")
    if not queue:
        completed = monitor.get("completed", 0)
        failed = monitor.get("failed", 0)
        total = monitor.get("total", 0)
        mimics.dialogs.message_box(
            title="Batch Export Complete",
            message="Exported {0}/{1} case(s); {2} failed.".format(completed, total, failed),
        )
        return

    monitor["done"] = True
    mimics.dialogs.message_box(
        title="Batch Export Disabled",
        message=(
            "Foreground batch export is disabled because opening .mcs files "
            "in the current Mimics process can block the GUI. Start batch "
            "export again to run it in a background Mimics process."
        ),
    )
    return

def _start_win32_export_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for export result polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_export_monitor(monitor_key)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )

    def _timer_proc(hwnd, message, timer_id, tick_count):
        _export_monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _EXPORT_MONITORS[monitor_key] = monitor
    return True


def _start_export_monitor(job_dir, work_dir, case_dir=None, timeout_seconds=600,
                          poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer that polls for bridge completion and auto-applies."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "work_dir": work_dir,
        "case_dir": case_dir,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "batch_queue": batch_queue,
        "completed": 0,
        "failed": 0,
    }
    if batch_info:
        monitor.update(batch_info)

    if _allow_windows_event_monitor() and _start_mimics_event_export_monitor(
        monitor,
        lambda: _export_monitor_tick(monitor),
        poll_seconds,
        "Export monitor failed",
    ):
        return True

    # Try PyQt5 QTimer first (works inside Mimics GUI event loop)
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        # Fall back to Win32 SetTimer
        if _start_win32_export_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Export Running",
            message="Data export has started, but progress cannot be applied automatically. Run export again later to check progress.",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_export_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Export Running",
            message="Data export has started, but progress cannot be applied automatically. Run export again later to check progress.",
        )
        return False

    timer = QTimer()
    _stop_export_monitor(job_dir)
    monitor["timer"] = timer
    _EXPORT_MONITORS[job_dir] = monitor

    def _tick():
        _export_monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


# -- Export masks from current Mimics project ---------------------------

def _sanitize_name(name):
    safe = name
    for ch in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


def _canonical_mask_name(name):
    text = str(name or "").strip().lower()
    for ch in (" ", "-", "/", "\\", "."):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _source_mask_stem(path):
    """Return a comparable Mask name for a supported NIfTI filename."""
    name = os.path.basename(str(path or ""))
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        name = name[:-7]
    elif lower.endswith(".nii"):
        name = name[:-4]
    else:
        return ""
    return _canonical_mask_name(name)


def _source_mask_search_dirs(source_mask_root, case_id):
    """Return canonical source-label directories for one case."""
    root = os.path.abspath(os.path.expanduser(str(source_mask_root or "")))
    case_id = str(case_id or "").strip()
    candidates = []
    case_root = os.path.join(root, case_id)
    candidates.append(os.path.join(case_root, "segmentations"))
    if os.path.basename(root).lower() == case_id.lower():
        candidates.append(os.path.join(root, "segmentations"))
        candidates.append(root)
    if os.path.basename(root).lower() == "segmentations":
        candidates.append(root)
    result = []
    seen = set()
    for path in candidates:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen or not os.path.isdir(path):
            continue
        seen.add(normalized)
        result.append(path)
    return result


def _find_source_mask_path(source_mask_root, case_id, mask_name):
    """Find one unambiguous canonical source Mask for a saved Mask name."""
    wanted = _canonical_mask_name(mask_name)
    if not wanted:
        return None
    matches = []
    for directory in _source_mask_search_dirs(source_mask_root, case_id):
        try:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if not os.path.isfile(path) or not filename.lower().endswith((".nii", ".nii.gz")):
                    continue
                if _source_mask_stem(filename) == wanted:
                    matches.append(os.path.abspath(path))
        except OSError:
            continue
    unique = []
    seen = set()
    for path in matches:
        normalized = os.path.normcase(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    if len(unique) > 1:
        raise RuntimeError(
            "More than one canonical source Mask matches '{}' for case '{}': {}".format(
                mask_name, case_id, ", ".join(unique)
            )
        )
    return unique[0] if unique else None


def _resolve_source_mask_paths(source_mask_root, case_id, masks):
    """Resolve every selected saved Mask to a canonical source file."""
    root = str(source_mask_root or "").strip()
    if not root:
        return {}
    if not os.path.isdir(root):
        raise RuntimeError("Canonical source-mask root was not found: {}".format(root))
    resolved = {}
    missing = []
    for a_mask in masks:
        name = str(
            getattr(a_mask, "name", a_mask) if a_mask is not None else ""
        ).strip()
        path = _find_source_mask_path(root, case_id, name)
        if path is None:
            missing.append(name or "(unnamed)")
        else:
            resolved[name] = path
    if missing:
        raise RuntimeError(
            "Canonical source Mask(s) were not found for case '{}': {}. Expected "
            "<source-mask-root>/<case>/segmentations/<mask>.nii or .nii.gz.".format(
                case_id, ", ".join(missing)
            )
        )
    return resolved


def _current_mask_names():
    """Return mask names without reading their potentially large voxel buffers."""
    names = []
    try:
        masks = mimics.data.masks
    except Exception:
        return names
    for a_mask in masks:
        name = str(getattr(a_mask, "name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _get_voxel_buffer_bytes(mask):
    """Get mask voxel buffer as raw bytes, handling various return types."""
    buf = mask.get_voxel_buffer()
    if hasattr(buf, "tobytes"):
        return buf.tobytes()
    elif hasattr(buf, "tostring"):
        return buf.tostring()
    elif isinstance(buf, (bytes, bytearray)):
        return bytes(buf)
    else:
        try:
            import numpy as np
            arr = np.asarray(buf)
            return arr.astype(np.uint8).tobytes()
        except Exception:
            return bytes(buf)


def _metadata_get(obj, name, default=""):
    try:
        item = obj.metadata.find(name)
        if item is not None:
            return str(getattr(item, "value", default))
    except Exception:
        pass
    return default


def _parse_matrix_metadata(value):
    try:
        matrix = json.loads(value) if isinstance(value, str) else value
        if not matrix or len(matrix) != 4:
            return None
        parsed = []
        for row in matrix:
            if len(row) != 4:
                return None
            parsed.append([float(item) for item in row])
        return parsed
    except Exception:
        return None


def _lps_point_to_ras(point):
    return [-float(point[0]), -float(point[1]), float(point[2])]


def _point_values(point):
    if point is None:
        raise RuntimeError("Mimics returned an empty voxel center.")
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [float(getattr(point, names[0])), float(getattr(point, names[1])), float(getattr(point, names[2]))]
        except Exception:
            pass
    return [float(point[0]), float(point[1]), float(point[2])]


def _voxel_center(image, index):
    getter = getattr(image, "get_voxel_center", None)
    if not callable(getter):
        raise RuntimeError("Mimics image does not expose get_voxel_center().")
    values = [int(value) for value in index]
    try:
        return _point_values(getter(values))
    except TypeError:
        pass
    try:
        return _point_values(getter(tuple(values)))
    except TypeError:
        pass
    return _point_values(getter(values[0], values[1], values[2]))


def _derive_mimics_voxel_to_ras_matrix(image, image_shape):
    """Derive the open Mimics image voxel grid in RAS coordinates."""
    if not image_shape:
        return None
    try:
        origin = _lps_point_to_ras(_voxel_center(image, [0, 0, 0]))
        matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
        for axis in range(3):
            if int(image_shape[axis]) > 1:
                index = [0, 0, 0]
                index[axis] = 1
                point = _lps_point_to_ras(_voxel_center(image, index))
                vector = [point[i] - origin[i] for i in range(3)]
            else:
                vector = [0.0, 0.0, 0.0]
                vector[axis] = 1.0
            for row in range(3):
                matrix[row][axis] = vector[row]
        for row in range(3):
            matrix[row][3] = origin[row]
        matrix[3] = [0.0, 0.0, 0.0, 1.0]
        return matrix
    except Exception:
        return None


def export_masks_to_buffers(
    buffers_dir, mask_names=None, target_mask_name=None, skip_voxel_data=False
):
    """Export all masks from current Mimics project as .u8 files.

    Returns manifest dict.
    """
    if not os.path.isdir(buffers_dir):
        os.makedirs(buffers_dir)

    # Clean old buffers
    for fname in os.listdir(buffers_dir):
        fpath = os.path.join(buffers_dir, fname)
        if os.path.isfile(fpath):
            os.unlink(fpath)

    masks = mimics.data.masks
    print("found {0} mask(s)".format(len(masks)))
    wanted = None
    if mask_names:
        wanted = set()
        for raw in mask_names:
            name = str(raw or "").strip()
            if not name:
                continue
            wanted.add(_canonical_mask_name(name))

    selected = []
    for a_mask in masks:
        if wanted:
            original = str(getattr(a_mask, "name", "") or "")
            if _canonical_mask_name(original) not in wanted and _canonical_mask_name(_sanitize_name(original)) not in wanted:
                continue
        selected.append(a_mask)

    if wanted:
        print("selected {0} mask(s) by filter".format(len(selected)))
    if target_mask_name and len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one saved mask for training target {0}, but matched {1}: {2}".format(
                target_mask_name,
                len(selected),
                ", ".join(str(getattr(item, "name", "") or "") for item in selected) or "(none)",
            )
        )

    manifest = {
        "masks": [],
        "mimics_shape": None,
        "mimics_voxel_to_ras_matrix": None,
        "mimics_voxel_to_ras_matrix_source": "",
    }

    # Get image shape from first mask's image
    if len(masks) > 0:
        try:
            img = masks[0].image
            dims = [int(v) for v in img.logical_dimensions]
            manifest["mimics_shape"] = dims
            matrix = _parse_matrix_metadata(
                _metadata_get(img, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
            )
            if matrix is not None:
                manifest["mimics_voxel_to_ras_matrix"] = matrix
                manifest["mimics_voxel_to_ras_matrix_source"] = "image_metadata"
            else:
                matrix = _derive_mimics_voxel_to_ras_matrix(img, dims)
                if matrix is not None:
                    manifest["mimics_voxel_to_ras_matrix"] = matrix
                    manifest["mimics_voxel_to_ras_matrix_source"] = "voxel_centers"
            print("image dimensions: {0}".format(dims))
        except Exception as e:
            print("could not read image dimensions: {0}".format(e))

    for a_mask in selected:
        name = str(a_mask.name)
        print("exporting mask: {0}".format(name))

        safe_name = _sanitize_name(target_mask_name or name)
        u8_path = os.path.join(buffers_dir, safe_name + ".u8")

        if not skip_voxel_data:
            try:
                try:
                    mimics.update_gui()
                except Exception:
                    pass
                raw = _get_voxel_buffer_bytes(a_mask)
                try:
                    mimics.update_gui()
                except Exception:
                    pass
            except Exception as e:
                print("  could not read data for {0}: {1}".format(name, e))
                continue

            with open(u8_path, "wb") as f:
                f.write(raw)
        else:
            print("  canonical source Mask configured; skipping MCS voxel buffer")

        mask_entry = {
            "original_name": name,
            "safe_name": safe_name,
        }
        if not skip_voxel_data:
            mask_entry["u8_filename"] = safe_name + ".u8"
        manifest["masks"].append(mask_entry)
        print("  -> {0}".format(safe_name))

    # Save manifest
    manifest_path = os.path.join(buffers_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Manifest saved: {0}".format(manifest_path))

    return manifest


def _selected_masks(mask_names=None):
    wanted = set(
        _canonical_mask_name(name)
        for name in (mask_names or [])
        if str(name or "").strip()
    )
    selected = []
    for a_mask in mimics.data.masks:
        name = str(getattr(a_mask, "name", "") or "")
        if wanted and _canonical_mask_name(name) not in wanted:
            continue
        selected.append(a_mask)
    return selected


def _object_identity(value):
    for attr in ("guid", "id", "identifier"):
        try:
            current = getattr(value, attr, None)
        except Exception:
            current = None
        if current is not None:
            return str(current)
    return str(id(value)) if value is not None else ""


def _foreground_export_target_is_open(monitor):
    launch_project = str(monitor.get("launch_project_path") or "")
    current_project = _current_project_path()
    if launch_project:
        try:
            same_project = os.path.normcase(os.path.abspath(launch_project)) == os.path.normcase(
                os.path.abspath(current_project)
            )
        except Exception:
            same_project = False
        if not same_project:
            return False
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    expected_image = str(monitor.get("launch_image_id") or "")
    if active_image is not None and _object_identity(active_image) == expected_image:
        return True
    try:
        for image in mimics.data.images:
            if _object_identity(image) == expected_image:
                return True
    except Exception:
        pass
    return False


def _new_export_manifest(selected):
    manifest = {
        "masks": [],
        "mimics_shape": None,
        "mimics_voxel_to_ras_matrix": None,
        "mimics_voxel_to_ras_matrix_source": "",
    }
    if not selected:
        return manifest
    image = getattr(selected[0], "image", None)
    if image is None:
        raise RuntimeError("The selected mask is not attached to the active image.")
    dims = [int(value) for value in image.logical_dimensions]
    manifest["mimics_shape"] = dims
    matrix = _parse_matrix_metadata(
        _metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
    )
    if matrix is not None:
        manifest["mimics_voxel_to_ras_matrix"] = matrix
        manifest["mimics_voxel_to_ras_matrix_source"] = "image_metadata"
    else:
        matrix = _derive_mimics_voxel_to_ras_matrix(image, dims)
        if matrix is None:
            raise RuntimeError(
                "Could not derive the active Mimics image geometry. Export was stopped "
                "instead of writing an ambiguously oriented mask."
            )
        manifest["mimics_voxel_to_ras_matrix"] = matrix
        manifest["mimics_voxel_to_ras_matrix_source"] = "voxel_centers"
    return manifest


def _finish_foreground_export(monitor, error=None, result=None, cancelled=False):
    monitor["done"] = True
    _stop_export_monitor(monitor.get("monitor_key"))
    work_dir = monitor.get("work_dir")
    if cancelled:
        _write_export_task_status(
            monitor,
            {
                "status": "cancelled",
                "phase": "cancelled",
                "completed": int(monitor.get("mask_index", 0) or 0),
                "failed": 0,
                "total": len(monitor.get("selected_masks") or []),
            },
        )
        runtime_common.release_local_operation(
            "mask_buffer_access", monitor.get("operation_token")
        )
        _mimics_log(logging.INFO, "Mask export stopped by user.")
        _append_export_log(
            monitor.get("output_root") or "",
            "Current-project mask export stopped by user.",
        )
        _cleanup_work_dir(work_dir)
        try:
            mimics.dialogs.message_box(
                title="Export Stopped",
                message="Mask export stopped. Temporary buffers were removed.",
                ui_blocking=False,
            )
        except Exception:
            pass
        return
    if error:
        _write_export_task_status(
            monitor,
            {
                "status": "failed",
                "phase": "failed",
                "completed": int(monitor.get("mask_index", 0) or 0),
                "failed": 1,
                "total": len(monitor.get("selected_masks") or []),
                "error": str(error),
            },
        )
        runtime_common.release_local_operation(
            "mask_buffer_access", monitor.get("operation_token")
        )
        _mimics_log(logging.ERROR, "Mask export failed: {0}".format(error))
        _append_export_log(
            monitor.get("output_root") or "",
            "Current-project mask export failed: {0}".format(error),
        )
        _cleanup_work_dir(work_dir)
        try:
            mimics.dialogs.message_box(
                title="Export Failed",
                message="Mask export failed.\n\n{0}\n\nLog/output: {1}".format(
                    error, monitor.get("output_root")
                ),
                ui_blocking=False,
            )
        except Exception:
            pass
        return
    try:
        total_new, total_overwritten, total_unchanged = _apply_export_result(result or {}, work_dir)
    except Exception as exc:
        _finish_foreground_export(monitor, error=exc)
        return
    try:
        _record_exported_labels(
            monitor.get("output_root") or "",
            monitor.get("case_id") or "",
            result or {},
            source_image_path=monitor.get("source_image_path") or "",
            mcs_path=monitor.get("mcs_path") or "",
        )
        mcs_path = str(monitor.get("mcs_path") or "")
        if mcs_path:
            mcs_root = os.path.dirname(os.path.abspath(mcs_path))
            if os.path.normcase(mcs_root) != os.path.normcase(
                os.path.abspath(monitor.get("output_root") or "")
            ):
                _record_exported_labels(
                    mcs_root,
                    monitor.get("case_id") or "",
                    result or {},
                    source_image_path=monitor.get("source_image_path") or "",
                    mcs_path=mcs_path,
                )
    except Exception as manifest_exc:
        _mimics_log(
            logging.WARNING,
            "Masks were exported, but dataset manifest update failed: {0}".format(
                manifest_exc
            ),
        )
    degraded_note = ""
    if (result or {}).get("degraded_export") or monitor.get("degraded_export"):
        degraded_note = (
            "\n\nNote: exported on the current Mimics grid because the original "
            "source image could not be resolved. The grid differs from the "
            "original data."
        )
    message = (
        "Mask export complete: {0} new, {1} overwritten, {2} unchanged, "
        "{3} existing files preserved. Output: {4}{5}"
    ).format(
        total_new,
        total_overwritten,
        total_unchanged,
        int((result or {}).get("total_skipped_existing", 0) or 0),
        (result or {}).get("output_seg_dir") or monitor.get("output_root"),
        degraded_note,
    )
    _mimics_log(logging.INFO, message)
    _write_export_task_status(
        monitor,
        {
            "status": "completed",
            "phase": "completed",
            "completed": len(monitor.get("selected_masks") or []),
            "failed": 0,
            "total": len(monitor.get("selected_masks") or []),
        },
    )
    runtime_common.release_local_operation(
        "mask_buffer_access", monitor.get("operation_token")
    )
    _cleanup_work_dir(work_dir)
    try:
        mimics.dialogs.message_box(
            title="Export Complete",
            message=message,
            ui_blocking=False,
        )
    except Exception:
        pass


def _finish_foreground_export_after_process(
    monitor, error=None, cancelled=False
):
    """Reap the external bridge before terminal status and work cleanup."""
    process = monitor.get("process")
    if process is None or process.poll() is not None:
        _finish_foreground_export(
            monitor, error=error, cancelled=cancelled
        )
        return
    monitor["done"] = True
    _stop_export_monitor(monitor.get("monitor_key"))
    _write_export_task_status(
        monitor,
        {
            "status": "cancelling" if cancelled else "stopping",
            "phase": "cancelling" if cancelled else "stopping_after_error",
            "completed": int(monitor.get("mask_index", 0) or 0),
            "failed": 0 if cancelled else 1,
            "total": len(monitor.get("selected_masks") or []),
        },
    )

    def _complete():
        _finish_foreground_export(
            monitor, error=error, cancelled=cancelled
        )

    if not runtime_common.terminate_process_async(
        process=process,
        graceful_seconds=2.0,
        on_complete=_complete,
    ):
        _complete()


def _foreground_export_tick(monitor):
    if monitor.get("done") or monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        if monitor.get("cancel_requested") or _export_task_stop_requested(monitor):
            _finish_foreground_export_after_process(
                monitor, cancelled=True
            )
            return
        if time.time() > monitor.get("deadline", 0):
            _finish_foreground_export_after_process(
                monitor, error="Export timed out."
            )
            return
        phase = monitor.get("phase")
        if phase == "buffers":
            if not _foreground_export_target_is_open(monitor):
                _finish_foreground_export(
                    monitor,
                    error=(
                        "The active Mimics project or image changed while mask buffers were being read. "
                        "Export stopped before conversion; reopen the original project and retry."
                    ),
                )
                return
            selected = monitor["selected_masks"]
            index = int(monitor.get("mask_index", 0))
            if index < len(selected):
                a_mask = selected[index]
                name = str(getattr(a_mask, "name", "") or "mask")
                _mimics_log(
                    logging.INFO,
                    "Mask export: reading {0}/{1}: {2}.".format(
                        index + 1, len(selected), name
                    ),
                )
                _write_export_task_status(
                    monitor,
                    {
                        "status": "running",
                        "phase": "reading_masks",
                        "completed": index,
                        "failed": 0,
                        "total": len(selected),
                        "mask_name": name,
                    },
                )
                try:
                    mimics.update_gui()
                except Exception:
                    pass
                raw = _get_voxel_buffer_bytes(a_mask)
                expected = 1
                for dim in monitor["manifest"]["mimics_shape"]:
                    expected *= int(dim)
                if len(raw) != expected:
                    raise RuntimeError(
                        "Mask {0} returned {1} voxels; expected {2}.".format(
                            name, len(raw), expected
                        )
                    )
                safe_name = _sanitize_name(name)
                with open(os.path.join(monitor["buffers_dir"], safe_name + ".u8"), "wb") as handle:
                    handle.write(raw)
                monitor["manifest"]["masks"].append({
                    "original_name": name,
                    "safe_name": safe_name,
                    "u8_filename": safe_name + ".u8",
                })
                monitor["mask_index"] = index + 1
                return
            manifest_path = os.path.join(monitor["buffers_dir"], "manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(monitor["manifest"], handle, indent=2, ensure_ascii=False)
            # The source snapshot is now immutable on disk. Release Mimics'
            # Mask-buffer lease before the external geometry conversion so AI
            # results and Mask import do not wait for unrelated file I/O.
            operation_token = monitor.pop("operation_token", None)
            if operation_token:
                runtime_common.release_local_operation(
                    "mask_buffer_access", operation_token
                )
            params = {
                "action": "convert",
                "buffers_dir": monitor["buffers_dir"],
                "manifest_path": manifest_path,
                "case_dir": monitor["case_dir"],
                "axes": monitor["axes"],
                "flips": monitor["flips"],
                "export_space": "source_image",
                "require_source_geometry": not monitor.get("degraded_export"),
                "source_image_path": monitor["source_image_path"],
                "output_seg_dir": monitor["output_seg_dir"],
                "overwrite_existing": monitor["overwrite_existing"],
                "export_formats": monitor.get("export_formats") or [],
                "mimics_voxel_to_ras_matrix": monitor["manifest"]["mimics_voxel_to_ras_matrix"],
            }
            # When the source image file is gone, pipeline-created projects
            # still carry its exact grid in mcs metadata; the bridge can
            # resample to that grid without the file. Forward it so the
            # export lands on the original grid rather than the Mimics grid.
            for bridge_key, meta_key in (
                ("source_image_shape", SOURCE_IMAGE_SHAPE_METADATA),
                ("source_voxel_to_ras_matrix", SOURCE_VOXEL_TO_RAS_MATRIX_METADATA),
            ):
                meta_value = monitor.get("source_grid_metadata", {}).get(meta_key, "")
                if meta_value:
                    params[bridge_key] = meta_value
            process = _launch_bridge_background(params, monitor["bridge_job_dir"])
            _write_json_atomic(
                os.path.join(monitor["bridge_job_dir"], "job_state.json"),
                {"pid": process.pid, "started_at_epoch": time.time()},
            )
            monitor["process"] = process
            monitor["phase"] = "bridge"
            _write_export_task_status(
                monitor,
                {
                    "status": "running",
                    "phase": "converting_to_source_grid",
                    "completed": len(selected),
                    "failed": 0,
                    "total": len(selected),
                },
            )
            _mimics_log(logging.INFO, "Mask buffers captured. Converting to the original image grid in the background.")
            return
        if phase == "bridge":
            status, payload = _check_job_status(monitor["bridge_job_dir"])
            if status == "running":
                due, elapsed = runtime_common.progress_notice_due(
                    monitor,
                    "current_export_convert",
                    detail="converting_to_source_grid",
                    interval_seconds=60.0,
                    initial_delay_seconds=30.0,
                )
                if due:
                    _mimics_log(
                        logging.INFO,
                        "Mask export is still converting to the original "
                        "image grid ({0}s). Mimics remains available. Use "
                        "Stop Mask Export to cancel.".format(int(elapsed)),
                    )
                return
            runtime_common.clear_progress_notice(
                monitor, "current_export_convert"
            )
            if status == "error":
                _finish_foreground_export(monitor, error=payload)
                return
            _cleanup_job_dir(monitor["bridge_job_dir"])
            _finish_foreground_export(monitor, result=payload)
    except Exception as exc:
        _finish_foreground_export(monitor, error=exc)
    finally:
        monitor["busy"] = False


def _start_foreground_export_monitor(monitor, poll_seconds=0.15):
    key = monitor["monitor_key"]
    _stop_export_monitor(key)
    if _allow_windows_event_monitor() and _start_mimics_event_export_monitor(
        monitor,
        lambda: _foreground_export_tick(monitor),
        poll_seconds,
        "Current-project export monitor failed",
    ):
        return True
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(
                None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint
            )

            def _timer_proc(hwnd, message, timer_id, tick_count):
                _foreground_export_tick(monitor)

            callback = TIMERPROC(_timer_proc)
            user32.SetTimer.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC,
            ]
            user32.SetTimer.restype = ctypes.c_size_t
            user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            timer_id = user32.SetTimer(None, 0, max(100, int(poll_seconds * 1000)), callback)
            if timer_id:
                monitor["callback"] = callback
                monitor["win32_timer"] = (user32, timer_id)
                _EXPORT_MONITORS[key] = monitor
                return True
        except Exception:
            pass
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: _foreground_export_tick(monitor))
        timer.start(max(100, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        _EXPORT_MONITORS[key] = monitor
        return True
    except Exception:
        return False


def _read_source_grid_metadata(selected):
    """Read the import-time grid snapshot (shape + voxel-to-RAS matrix) from
    the active image's metadata. Returns a dict of raw metadata values or {}
    when the project carries no usable snapshot (e.g. hand-created projects).
    """
    try:
        image = getattr(selected[0], "image", None)
        if image is None:
            return {}
        shape = _metadata_get(image, SOURCE_IMAGE_SHAPE_METADATA, "")
        matrix = _metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, "")
        if not str(shape or "").strip() or not str(matrix or "").strip():
            return {}
        if _parse_matrix_metadata(matrix) is None:
            return {}
        return {
            SOURCE_IMAGE_SHAPE_METADATA: str(shape),
            SOURCE_VOXEL_TO_RAS_MATRIX_METADATA: str(matrix),
        }
    except Exception:
        return {}


def _confirm_degraded_export(case_count=1):
    """Ask once before exporting on the Mimics grid without source geometry.

    Returns True when the user accepted the degraded export. The decision is
    caller-scoped: batch export asks once per batch, single-case export once
    per export action.
    """
    scope = "this mask" if case_count <= 1 else "{0} case(s)".format(case_count)
    try:
        answer = mimics.dialogs.question_box(
            title="Export Without Original Image",
            message=(
                "The original source image could not be resolved for {0}, so the mask "
                "cannot be resampled back to the original grid.\n\n"
                "It will be exported on the current Mimics grid instead. The resulting "
                "file's grid differs from the original data; do not use it where "
                "alignment with the original image is required.\n\n"
                "Export on the Mimics grid anyway?"
            ).format(scope),
            buttons="Export on Mimics Grid;Cancel",
            ui_blocking=True,
        )
        return str(answer or "").strip().lower().startswith("export")
    except Exception:
        return False


def _start_current_project_export(case_dir, source_image_path, output_root, axes,
                                  flips, mask_names=None, overwrite_existing=False,
                                  export_formats=None):
    _prune_local_export_jobs()
    token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "current-project mask export"
    )
    if not token:
        raise RuntimeError(
            "Another Mimics-Script operation is currently reading or applying mask voxels. "
            "Wait for it to finish or stop that task before exporting."
        )
    try:
        selected = _selected_masks(mask_names)
        if not selected:
            raise RuntimeError("No matching masks were found in the current project.")
        source_image_path = str(source_image_path or "").strip()
        degraded_export = False
        source_grid_metadata = {}
        if not source_image_path:
            # The export can still land on the original grid when the project
            # carries the grid snapshot from import time (pipeline-created
            # projects always do). Only ask about the Mimics-grid fallback
            # when even that snapshot is missing.
            source_grid_metadata = _read_source_grid_metadata(selected)
            if not source_grid_metadata:
                if not _confirm_degraded_export(len(selected)):
                    raise RuntimeError(
                        "The original source image or case folder is required so the exported mask "
                        "can be resampled to the original shape and orientation."
                    )
                degraded_export = True
                source_image_path = ""
        case_id = _current_project_case_id() or os.path.basename(os.path.abspath(case_dir))
        output_seg_dir = os.path.join(output_root, case_id, "segmentations")
        runtime_dir = os.path.join(
            _project_root(), ".mimics_runtime", "export_jobs",
            "current_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]),
        )
        buffers_dir = os.path.join(runtime_dir, "buffers")
        bridge_job_dir = os.path.join(runtime_dir, "bridge")
        task_id = os.path.basename(runtime_dir)
        task_dir = os.path.join(_project_root(), ".mimics_runtime", "ui_tasks")
        if not os.path.isdir(task_dir):
            os.makedirs(task_dir)
        task_status_path = os.path.join(task_dir, task_id + ".json")
        task_stop_path = os.path.join(task_dir, task_id + "_stop.json")
        try:
            if os.path.isfile(task_stop_path):
                os.remove(task_stop_path)
        except Exception:
            pass
        os.makedirs(buffers_dir)
        manifest = _new_export_manifest(selected)
        monitor = {
            "monitor_key": runtime_dir,
            "work_dir": runtime_dir,
            "buffers_dir": buffers_dir,
            "bridge_job_dir": bridge_job_dir,
            "case_dir": os.path.abspath(case_dir),
            "case_id": case_id,
            "mcs_path": _current_project_path() or "",
            "launch_project_path": _current_project_path() or "",
            "launch_image_id": _object_identity(
                getattr(selected[0], "image", None)
            ),
            "source_image_path": source_image_path,
            "output_root": os.path.abspath(output_root),
            "output_seg_dir": output_seg_dir,
            "axes": list(axes),
            "flips": list(flips),
            "overwrite_existing": bool(overwrite_existing),
            "degraded_export": degraded_export,
            "source_grid_metadata": dict(source_grid_metadata),
            "export_formats": [
                str(fmt).lower().lstrip(".")
                for fmt in (export_formats or [])
                if str(fmt).strip()
            ],
            "selected_masks": selected,
            "manifest": manifest,
            "mask_index": 0,
            "phase": "buffers",
            "deadline": time.time() + 3600.0,
            "done": False,
            "busy": False,
            "operation_token": token,
            "task_status_path": task_status_path,
            "task_stop_path": task_stop_path,
        }
        _write_export_task_status(
            monitor,
            {
                "status": "running",
                "phase": "reading_masks",
                "completed": 0,
                "failed": 0,
                "total": len(selected),
            },
        )
        task_descriptor = {
            "kind": "export",
            "title": "Export current project masks",
            "status_path": task_status_path,
            "stop_path": task_stop_path,
            "log_path": os.path.join(
                os.path.abspath(output_root), _RUNTIME_SUBDIR, "mimics_export.log"
            ),
            "output_path": os.path.abspath(output_root),
        }
        if not _start_foreground_export_monitor(monitor):
            raise RuntimeError("Mimics does not expose a usable GUI timer for incremental export.")
        token = None  # The monitor now owns and releases the local-operation token.
        _mimics_log(
            logging.INFO,
            "Mask export started for {0} mask(s). Mimics remains available between mask reads.".format(
                len(selected)
            ),
        )
        try:
            mimics.view.show_log_panel()
        except Exception:
            pass
        return task_descriptor
    except Exception:
        if token:
            runtime_common.release_local_operation("mask_buffer_access", token)
        raise


def cancel_current_project_exports():
    """Request cancellation for exports owned by this Mimics session."""
    count = 0
    for monitor in list(_EXPORT_MONITORS.values()):
        if monitor.get("done"):
            continue
        monitor["cancel_requested"] = True
        stop_path = (
            monitor.get("task_stop_path")
            or monitor.get("stop_path")
            or ""
        )
        if stop_path:
            try:
                runtime_common.write_json_atomic(
                    stop_path,
                    {
                        "status": "stop_requested",
                        "requested_at_epoch": time.time(),
                        "requested_by_pid": os.getpid(),
                    },
                )
            except Exception:
                pass
        process = monitor.get("process")
        if monitor.get("operation_token") and not monitor.get("busy"):
            _finish_foreground_export_after_process(
                monitor, cancelled=True
            )
            count += 1
            continue
        if process is not None and monitor.get("launch_finished") and not stop_path:
            runtime_common.terminate_process_async(
                process=process,
                graceful_seconds=2.0,
            )
        count += 1
    return count


# -- Disk space helpers -------------------------------------------------

def _cleanup_work_dir(work_dir):
    """Remove intermediate work directory (.u8 buffers)."""
    if not work_dir or not os.path.isdir(work_dir):
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        print("  cleaned work dir: {0}".format(work_dir))
    except Exception as e:
        print("  warning: could not clean work dir {0}: {1}".format(work_dir, e))


def _check_disk_space(path, required_mb):
    """Check if the drive containing *path* has at least *required_mb* MB free."""
    try:
        if os.name == "nt":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(os.path.abspath(path)),
                ctypes.pointer(free_bytes),
                None,
                None,
            )
            free_mb = free_bytes.value / (1024 * 1024)
        else:
            stat = os.statvfs(path)
            free_mb = stat.f_bavail * stat.f_frsize / (1024 * 1024)
        return (free_mb >= required_mb, free_mb)
    except Exception:
        return (True, -1)


def _pick_directory(title):
    """Show a native folder-picker dialog. Returns path or None.

    Uses PyQt5 QFileDialog when available (shares Mimics' Qt event loop,
    no GUI freeze). Falls back to Tkinter only if PyQt5 is unavailable.
    """
    # Try PyQt5 first; Mimics is a Qt app, so QFileDialog integrates
    # with Mimics' event loop and won't cause GUI freeze.
    # Note: we don't check QApplication.instance() because Mimics may
    # not expose it to Python, but QFileDialog can still work if PyQt5
    # is importable (Mimics' Qt runtime is already running).
    try:
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(None, title, "")
        return str(path) if path else None
    except Exception:
        pass

    # Fallback: Tkinter (may cause GUI freeze in Mimics)
    try:
        import Tkinter as tk
        import tkFileDialog
    except ImportError:
        try:
            import tkinter as tk
            from tkinter import filedialog as tkFileDialog
        except ImportError:
            return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = tkFileDialog.askdirectory(parent=root, title=title)
    root.destroy()
    return path if path else None


def _current_project_case_id():
    try:
        info = mimics.file.get_project_information()
    except Exception:
        return None
    for attr in ("filename", "file_name", "path", "project_path", "project_file"):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if value:
            name = os.path.basename(str(value))
            return name[:-4] if name.lower().endswith(".mcs") else name
    return None


def _launch_background_batch_export_async(*args, **kwargs):
    """Move output-path I/O and background Mimics startup off the GUI thread."""
    _prune_export_launch_threads()
    ts_root = args[0] if args else ""
    mcs_scope = kwargs.get("mcs_output_dir")
    if not mcs_scope and ts_root:
        mcs_scope = _resolve_export_output_dir(ts_root)
    destination_scope = kwargs.get("label_output_root") or ts_root or mcs_scope
    holder = None
    seen_lock_names = set()
    for scope in (mcs_scope, destination_scope):
        if not scope:
            continue
        lock_name = runtime_common.background_mimics_lock_name(scope)
        if lock_name in seen_lock_names:
            continue
        seen_lock_names.add(lock_name)
        holder = runtime_common.active_resource_lock(_project_root(), lock_name)
        if holder:
            break
    if holder:
        try:
            mimics.dialogs.message_box(
                title="Export Waiting",
                message=(
                    "Mask export was not started because its saved .mcs source or label destination is busy.\n\n"
                    "Current task: {0}\n\n"
                    "Wait for the import/export to finish, or stop it from its own Stop entry before retrying."
                ).format(runtime_common.resource_lock_summary(holder)),
                ui_blocking=False,
            )
        except TypeError:
            mimics.dialogs.message_box(
                title="Export Waiting",
                message="The saved .mcs source or selected label destination is currently busy.",
            )
        return None
    export_root = kwargs.get("label_output_root") or kwargs.get("mcs_output_dir") or ""
    launch_state = {
        "monitor_key": "background_export_" + uuid.uuid4().hex,
        "export_root": os.path.abspath(export_root) if export_root else "",
        "launch_finished": False,
        "launch_error": "",
        "process": None,
        "started_at_epoch": time.time(),
        "deadline": time.time() + 7 * 86400,
        "done": False,
    }

    def worker():
        try:
            process = _launch_background_batch_export(*args, **kwargs)
            launch_state["process"] = process
            if process is None:
                launch_state["launch_error"] = "Background Mimics could not be started."
            else:
                launch_state["status_path"] = getattr(process, "_mimics_status_path", "")
                launch_state["handshake_path"] = getattr(process, "_mimics_handshake_path", "")
                launch_state["mimics_log_path"] = getattr(process, "_mimics_log_path", "")
                launch_state["process_log_path"] = getattr(process, "_mimics_process_log_path", "")
                launch_state["job_runtime"] = getattr(process, "_mimics_job_runtime", "")
                launch_state["stop_path"] = getattr(process, "_mimics_stop_path", "")
                if launch_state.get("cancel_requested"):
                    stop_path = launch_state.get("stop_path")
                    if stop_path:
                        runtime_common.write_json_atomic(
                            stop_path,
                            {
                                "status": "stop_requested",
                                "requested_at_epoch": time.time(),
                                "requested_by_pid": os.getpid(),
                            },
                        )
                    else:
                        runtime_common.terminate_process_async(
                            process=process,
                            graceful_seconds=2.0,
                        )
        except Exception as exc:
            root = kwargs.get("mcs_output_dir") or kwargs.get("label_output_root") or ""
            _append_export_log(root, "Could not start background mask export: {0}".format(exc))
            launch_state["launch_error"] = str(exc)
        finally:
            launch_state["launch_finished"] = True
            try:
                _EXPORT_LAUNCH_THREADS.remove(thread)
            except Exception:
                pass

    thread = threading.Thread(target=worker, name="MimicsMaskExportLaunch")
    thread.daemon = True
    _EXPORT_LAUNCH_THREADS.append(thread)
    try:
        thread.start()
    except Exception as exc:
        try:
            _EXPORT_LAUNCH_THREADS.remove(thread)
        except Exception:
            pass
        _mimics_log(logging.ERROR, "Mask export launcher could not start: {0}".format(exc))
        return None
    if not _start_background_export_status_monitor(launch_state):
        launch_state["cancel_requested"] = True
        process = launch_state.get("process")
        if process is not None:
            stop_path = launch_state.get("stop_path")
            if stop_path:
                try:
                    runtime_common.write_json_atomic(
                        stop_path,
                        {
                            "status": "stop_requested",
                            "requested_at_epoch": time.time(),
                            "requested_by_pid": os.getpid(),
                        },
                    )
                except Exception:
                    pass
            else:
                runtime_common.terminate_process_async(
                    process=process,
                    graceful_seconds=2.0,
                )
        _mimics_log(
            logging.ERROR,
            "Mask export was stopped because Mimics could not create a GUI status timer.",
        )
        return None
    _mimics_log(
        logging.INFO,
        "Mask export requested. Progress will appear here; output: {0}".format(
            launch_state.get("export_root") or "(resolving output path)"
        ),
    )
    try:
        mimics.view.show_log_panel()
    except Exception:
        pass
    return thread


def _prune_export_launch_threads():
    """Drop completed launcher threads from long-running Mimics sessions."""
    live = []
    for thread in list(_EXPORT_LAUNCH_THREADS):
        try:
            if thread.is_alive():
                live.append(thread)
        except Exception:
            pass
    _EXPORT_LAUNCH_THREADS[:] = live


def _background_export_status_tick(monitor):
    if monitor.get("done") or monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        key = monitor.get("monitor_key")
        if monitor.get("launch_finished") and monitor.get("launch_error"):
            monitor["done"] = True
            _stop_export_monitor(key)
            _mimics_log(
                logging.ERROR,
                "Mask export could not start: {0}".format(monitor.get("launch_error")),
            )
            mimics.dialogs.message_box(
                title="Export Could Not Start",
                message=str(monitor.get("launch_error")),
                ui_blocking=False,
            )
            return
        root = monitor.get("export_root")
        status_path = monitor.get("status_path") or (_rt(root, "_export_batch_status.json") if root else "")
        status = runtime_common.read_json(status_path, {}) if status_path else {}
        status = status or {}
        process = monitor.get("process")
        if (
            monitor.get("launch_finished")
            and process is not None
            and process.poll() is not None
            and status.get("status") not in ("closed", "cancelled", "failed")
        ):
            status = _finalize_background_export_process_status(
                process,
                status_path=status_path,
                stop_path=monitor.get("stop_path"),
                handshake_path=monitor.get("handshake_path"),
            )
        try:
            if float(status.get("updated_at_epoch", 0.0) or 0.0) < float(monitor.get("started_at_epoch", 0.0) or 0.0):
                status = {}
        except Exception:
            status = {}
        signature = (
            status.get("status"),
            status.get("phase"),
            status.get("case_id"),
            status.get("index"),
            status.get("total"),
            status.get("completed"),
            status.get("failed"),
        )
        if (
            status.get("status") == "waiting_degraded_confirm"
            and monitor.get("degraded_request_id") != status.get("request_id")
        ):
            # The background export hit a case without source geometry and is
            # paused. Answer once for the whole batch via the confirm file.
            monitor["degraded_request_id"] = status.get("request_id")
            remaining = 0
            try:
                remaining = max(0, int(status.get("total", 0) or 0) - int(status.get("index", 0) or 0))
            except Exception:
                remaining = 0
            accepted = _confirm_degraded_export(max(1, remaining))
            confirm_path = status.get("confirm_path") or ""
            if confirm_path:
                runtime_common.write_json_atomic(
                    confirm_path,
                    {
                        "request_id": status.get("request_id"),
                        "answered": True,
                        "accepted": bool(accepted),
                        "answered_at_epoch": time.time(),
                        "answered_by_pid": os.getpid(),
                    },
                )
            _mimics_log(
                logging.INFO,
                "User {0} the degraded Mimics-grid export for the running batch.".format(
                    "accepted" if accepted else "declined"
                ),
            )
        if status and signature != monitor.get("last_signature"):
            monitor["last_signature"] = signature
            try:
                index = int(status.get("index", 0) or 0)
                total = int(status.get("total", 0) or 0)
                progress = "{0}/{1}".format(index, total) if total else "starting"
                phase = str(status.get("phase") or status.get("status") or "running")
                mimics.logging.log_user_message(
                    level=logging.INFO,
                    message="Mask export [{0}] {1}: completed {2}, failed {3}{4}.".format(
                        progress,
                        phase,
                        int(status.get("completed", 0) or 0),
                        int(status.get("failed", 0) or 0),
                        "; current " + str(status.get("case_id")) if status.get("case_id") else "",
                    ),
                )
            except Exception:
                pass
        if status and status.get("status") not in (
            "closed", "cancelled", "failed"
        ):
            detail = "{0}|{1}|{2}".format(
                status.get("phase") or status.get("status") or "running",
                status.get("case_id") or "",
                status.get("index") or status.get("completed") or 0,
            )
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "background_export_progress",
                detail=detail,
                interval_seconds=60.0,
                initial_delay_seconds=60.0,
            )
            if due:
                _mimics_log(
                    logging.INFO,
                    "Mask export is still running ({0}s in the current "
                    "stage): {1}. Use Stop Mask Export to cancel.".format(
                        int(elapsed),
                        status.get("phase")
                        or status.get("status")
                        or "running",
                    ),
                )
        if status.get("status") in ("closed", "cancelled", "failed"):
            monitor["done"] = True
            _stop_export_monitor(key)
            failed = int(status.get("failed", 0) or 0)
            if status.get("status") == "failed":
                message = (
                    "Mask export failed after exporting {0} case(s).\n\n{1}\n\nDiagnostics: {2}"
                ).format(
                    int(status.get("completed", 0) or 0),
                    status.get("error", "The background export script stopped unexpectedly."),
                    monitor.get("job_runtime") or root,
                )
                _mimics_log(logging.ERROR, message)
                mimics.dialogs.message_box(
                    title="Export Failed",
                    message=message,
                    ui_blocking=False,
                )
                return
            if status.get("status") == "cancelled":
                mimics.dialogs.message_box(
                    title="Export Stopped",
                    message="Mask export stopped.\n\nExported before stop: {0}\nFailed: {1}\nOutput: {2}".format(
                        int(status.get("completed", 0) or 0), failed, root,
                    ),
                    ui_blocking=False,
                )
                return
            # P1 skip summary: categorized counts instead of a bare total.
            batch_totals = status.get("batch_totals") or {}
            summary_lines = [
                "New: {0}".format(int(batch_totals.get("new", 0) or 0)),
                "Overwritten: {0}".format(int(batch_totals.get("overwritten", 0) or 0)),
                "Unchanged: {0}".format(int(batch_totals.get("unchanged", 0) or 0)),
                "Existing files preserved (not updated): {0}".format(
                    int(batch_totals.get("skipped_existing", 0) or 0)
                ),
            ]
            not_updated = int(batch_totals.get("skipped_existing", 0) or 0)
            if status.get("degraded_export"):
                summary_lines.append(
                    "\nNote: some cases were exported on the current Mimics grid "
                    "because their original source image could not be resolved."
                )
            if not_updated > 0:
                summary_lines.append(
                    "\nCases with files that were not updated are listed in:\n{0}\n"
                    "Re-run those cases with the overwrite option to refresh them.".format(
                        status.get("skipped_exports_path") or "(list unavailable)"
                    )
                )
            message = (
                "Mask export finished.\n\nExported: {0}\nFailed: {1}\n{2}\n\nOutput: {3}{4}".format(
                    int(status.get("completed", 0) or 0),
                    failed,
                    "\n".join(summary_lines),
                    root,
                    "\n\nReview mimics_export.log before retrying failed cases." if failed else "",
                )
            )
            mimics.dialogs.message_box(
                title="Export Completed with Errors" if failed else "Export Complete",
                message=message,
                # When files were left not-updated, the dialog must stay for
                # the user to read the list path; do not auto-dismiss.
                ui_blocking=bool(not_updated > 0),
            )
            return
    finally:
        monitor["busy"] = False


def _start_background_export_status_monitor(monitor, poll_seconds=2.0):
    key = monitor["monitor_key"]
    _stop_export_monitor(key)
    if _allow_windows_event_monitor() and _start_mimics_event_export_monitor(
        monitor,
        lambda: _background_export_status_tick(monitor),
        poll_seconds,
        "Background export status monitor failed",
    ):
        return True
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint)

            def _timer_proc(hwnd, message, timer_id, tick_count):
                try:
                    _background_export_status_tick(monitor)
                except Exception:
                    pass

            callback = TIMERPROC(_timer_proc)
            user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
            user32.SetTimer.restype = ctypes.c_size_t
            user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            timer_id = user32.SetTimer(None, 0, int(poll_seconds * 1000), callback)
            if timer_id:
                monitor["callback"] = callback
                monitor["win32_timer"] = (user32, timer_id)
                _EXPORT_MONITORS[key] = monitor
                return True
        except Exception:
            pass
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: _background_export_status_tick(monitor))
        timer.start(max(500, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        _EXPORT_MONITORS[key] = monitor
        return True
    except Exception:
        return False


def _current_project_path():
    try:
        info = mimics.file.get_project_information()
    except Exception:
        return ""
    for attr in ("filename", "file_name", "path", "project_path", "project_file"):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if value:
            return os.path.abspath(str(value))
    return ""


def _active_source_path():
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    source = _metadata_get(image, SOURCE_IMAGE_PATH_METADATA, "") if image is not None else ""
    if not source:
        return ""
    return os.path.abspath(str(source))


def _run_main_with_args(args, source_info_override=None):
    previous = list(sys.argv)
    try:
        sys.argv = [previous[0]] + list(args)
        return main(source_info_override=source_info_override)
    finally:
        sys.argv = previous


def _launch_external_export_setup():
    import io_setup_mimics

    project_path = _current_project_path()
    if not project_path or not project_path.lower().endswith(".mcs"):
        raise RuntimeError("Save the current Mimics project before exporting masks.")
    case_id = _current_project_case_id() or os.path.splitext(os.path.basename(project_path))[0]
    source_initial = _active_source_path()

    def submitted(selection):
        source = str(selection.get("source_path", "") or "")
        export_root = str(selection.get("output_path", "") or "")
        source_info = selection.get("case_info") or {}
        if not source or not export_root or not source_info.get("image"):
            raise RuntimeError("The source image and export destination are required.")
        mask_selection = str(selection.get("mask_selection") or "all").strip()
        mask_names = None
        if mask_selection.lower() != "all":
            mask_names = [
                value.strip() for value in mask_selection.split(",") if value.strip()
            ]
        axes, flips = _resolve_buffer_mapping()
        return _start_current_project_export(
            str(source_info.get("case_dir") or source),
            str(source_info.get("image") or source),
            export_root,
            axes,
            flips,
            mask_names=mask_names,
            overwrite_existing=selection.get("conflict_policy") == "overwrite",
            export_formats=selection.get("export_formats") or [],
        )

    return io_setup_mimics.launch(
        "export_masks",
        _python_exe(),
        {
            "case_id": case_id,
            "source_initial": source_initial,
            # mimics_output_dir controls .mcs storage only. It must never
            # silently redirect exported label files.
            "configured_output": "",
            "mask_names": _current_mask_names(),
        },
        submitted,
    )


# -- Single case export -------------------------------------------------

def _export_masks_and_build_params(
    case_dir,
    axes,
    flips,
    work_dir,
    output_seg_dir=None,
    export_space="source_image",
    mask_names=None,
    target_mask_name=None,
    overwrite_existing=None,
    source_image_path=None,
    source_mask_root=None,
    case_id=None,
    export_formats=None,
):
    """Export masks to .u8 buffers and build bridge params. Returns (bridge_params, manifest) or None."""
    print("Exporting masks to: {0}".format(case_dir))

    buffers_dir = os.path.join(work_dir, "export_buffers")
    canonical_root = str(source_mask_root or "").strip()
    normalized_export_space = str(export_space or "source_image").lower()
    if canonical_root and normalized_export_space not in (
        "source", "source_image", "source_grid"
    ):
        raise RuntimeError(
            "Canonical source Mask export requires export_space=source_image; "
            "it cannot be used as a raw Mimics-grid export."
        )

    # Step 1: Export masks to .u8 buffers
    manifest = export_masks_to_buffers(
        buffers_dir,
        mask_names=mask_names,
        target_mask_name=target_mask_name,
        skip_voxel_data=bool(canonical_root),
    )

    if not manifest["masks"]:
        print("No masks to export")
        return None

    mimics_shape = manifest.get("mimics_shape")
    if not mimics_shape:
        print("Error: could not determine mimics_shape")
        return None

    if canonical_root:
        canonical_case_id = str(case_id or os.path.basename(os.path.abspath(case_dir)))
        source_mask_paths = _resolve_source_mask_paths(
            canonical_root,
            canonical_case_id,
            [row.get("original_name", "") for row in manifest["masks"]],
        )
    else:
        source_mask_paths = {}

    # Step 2: Build bridge params for convert action
    bridge_params = {
        "action": "convert",
        "buffers_dir": buffers_dir,
        "manifest_path": os.path.join(buffers_dir, "manifest.json"),
        "case_dir": case_dir,
        "axes": axes,
        "flips": flips,
        "export_space": str(export_space or "source_image"),
        "require_source_geometry": str(export_space or "source_image").lower() in (
            "source", "source_image", "source_grid",
        ),
        "source_mask_paths": source_mask_paths,
        "source_mask_root": canonical_root,
    }
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    if active_image is None:
        try:
            active_image = mimics.data.masks[0].image if len(mimics.data.masks) else None
        except Exception:
            active_image = None
    source_image_path = str(source_image_path or "")
    if not source_image_path and active_image is not None:
        source_image_path = _metadata_get(active_image, SOURCE_IMAGE_PATH_METADATA, "")
    if source_image_path:
        bridge_params["source_image_path"] = str(source_image_path)
    else:
        # Source file unavailable: forward the import-time grid snapshot so
        # the bridge resamples to the original grid (shape + affine) instead
        # of the Mimics grid. Hand-created projects carry no snapshot, in
        # which case the bridge reports source_geometry_unavailable and the
        # caller decides on the degraded path.
        try:
            shape_snapshot = _metadata_get(active_image, SOURCE_IMAGE_SHAPE_METADATA, "")
            matrix_snapshot = _metadata_get(active_image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, "")
        except Exception:
            shape_snapshot = ""
            matrix_snapshot = ""
        if str(shape_snapshot or "").strip() and str(matrix_snapshot or "").strip():
            if _parse_matrix_metadata(matrix_snapshot) is not None:
                bridge_params["source_image_shape"] = str(shape_snapshot)
                bridge_params["source_voxel_to_ras_matrix"] = str(matrix_snapshot)
    if output_seg_dir:
        bridge_params["output_seg_dir"] = output_seg_dir
    if export_formats:
        # P1 multi-format export: the bridge writes one file per format with
        # identical geometry; nothing is set here when only the historical
        # .nii.gz default is wanted.
        bridge_params["export_formats"] = list(export_formats)
    if overwrite_existing is not None:
        bridge_params["overwrite_existing"] = bool(overwrite_existing)
    if manifest.get("mimics_voxel_to_ras_matrix"):
        bridge_params["mimics_voxel_to_ras_matrix"] = manifest.get("mimics_voxel_to_ras_matrix")
    return (bridge_params, manifest)


def _apply_export_result(result, work_dir):
    """Process bridge convert result and clean up."""
    total_new = result.get("total_new", 0)
    total_overwritten = result.get("total_overwritten", 0)
    total_unchanged = result.get("total_unchanged", 0)
    total_skipped = result.get("total_skipped_existing", 0)
    print("Export complete: {0} new, {1} overwritten, {2} unchanged, {3} existing preserved".format(
        total_new, total_overwritten, total_unchanged, total_skipped))

    # Clean up intermediate .u8 buffers
    _cleanup_work_dir(work_dir)

    return total_new, total_overwritten, total_unchanged


# -- TS case discovery (for batch mode) --------------------------------

def discover_ts_cases(ts_root, case_filter=None, exclude_dirs=None):
    """Find all cases in a TS-like dataset."""
    cases = []
    excluded = set(os.path.abspath(d) for d in (exclude_dirs or []) if d)
    for name in sorted(os.listdir(ts_root)):
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name in ("mcs_output", "segmentations"):
            continue
        # Skip output/work directories that happen to live under ts_root and
        # contain stray image files (e.g. mask_exports with a leftover .nii.gz),
        # otherwise they get treated as a case and produce "Missing .mcs".
        if os.path.abspath(case_dir) in excluded:
            continue
        if case_filter and name not in case_filter:
            continue

        has_image = False
        for img_name in ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii", "ct.mhd", "mri.mhd", "ct.mha", "mri.mha", "ct.nrrd", "mri.nrrd"):
            if os.path.isfile(os.path.join(case_dir, img_name)):
                has_image = True
                break
        if not has_image:
            dicom_dir = os.path.join(case_dir, "dicom")
            if os.path.isdir(dicom_dir):
                has_image = True
        if not has_image:
            for fname in sorted(os.listdir(case_dir)):
                if fname.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
                    has_image = True
                    break
        if not has_image and case_filter and name in case_filter and os.listdir(case_dir):
            # A specifically requested case may be a flat DICOM directory.
            # The bridge validates headers outside the foreground Mimics GUI.
            has_image = True

        if has_image:
            cases.append({"case_id": name, "case_dir": case_dir})
    return cases


def _matching_current_mask_names(mask_names):
    wanted = set(
        _canonical_mask_name(name)
        for name in (mask_names or [])
        if str(name or "").strip()
    )
    available = _current_mask_names()
    if not wanted:
        return available, list(available)
    matched = [
        name for name in available
        if _canonical_mask_name(name) in wanted
    ]
    return available, matched


def _preflight_batch_mask_names(cases, output_dir, config, export_root, status_path, stop_path):
    requested = config.get("mask_names") or []
    if not isinstance(requested, list):
        requested = [requested]
    requested = [str(name).strip() for name in requested if str(name).strip()]
    target_name = str(config.get("target_mask_name") or "").strip()
    if not requested:
        return []

    failures = []
    total = len(cases)
    _append_export_log(
        export_root,
        "Checking saved mask names in {0} project(s) before reading voxel data. Accepted names: {1}".format(
            total,
            ", ".join(requested),
        ),
    )
    for index, case_info in enumerate(cases):
        if os.path.isfile(stop_path):
            return [{"case_id": "", "reason": "cancelled"}]
        case_id = case_info["case_id"]
        mcs_path = str(
            (config.get("mcs_paths") or {}).get(case_id)
            or os.path.join(output_dir, case_id + ".mcs")
        )
        _write_json_atomic(
            status_path,
            {
                "status": "checking_masks",
                "phase": "mask_name_preflight",
                "pid": os.getpid(),
                "case_id": case_id,
                "index": index + 1,
                "total": total,
                "completed": 0,
                "failed": len(failures),
                "updated_at_epoch": time.time(),
            },
        )
        if not os.path.isfile(mcs_path):
            failures.append({
                "case_id": case_id,
                "reason": ".mcs file not found",
                "mcs_path": mcs_path,
                "available_masks": [],
            })
            continue
        project_opened = False
        try:
            mimics.file.open_project(mcs_path)
            project_opened = True
            available, matched = _matching_current_mask_names(requested)
            if not matched:
                failures.append({
                    "case_id": case_id,
                    "reason": "no saved mask matched",
                    "mcs_path": mcs_path,
                    "available_masks": available,
                })
            elif target_name and len(matched) != 1:
                failures.append({
                    "case_id": case_id,
                    "reason": "multiple saved masks matched one training target",
                    "mcs_path": mcs_path,
                    "available_masks": available,
                    "matched_masks": matched,
                })
        except Exception as exc:
            failures.append({
                "case_id": case_id,
                "reason": "could not inspect project: {0}".format(exc),
                "mcs_path": mcs_path,
                "available_masks": [],
            })
            # Mimics may have partially replaced the active project before
            # reporting a parsing error. Attempt a close even when open_project
            # did not return, so the next case does not inherit that state.
            try:
                mimics.file.close_project()
            except Exception:
                pass
        finally:
            if project_opened:
                try:
                    mimics.file.close_project()
                except Exception:
                    pass
    return failures


def _partition_mask_preflight_failures(
    failures, allow_unlabeled_projects, allow_invalid_projects=False
):
    skipped = []
    hard_failures = []
    for row in failures or []:
        reason = str(row.get("reason") or "")
        skippable = (
            allow_unlabeled_projects
            and reason == "no saved mask matched"
        )
        if allow_invalid_projects and (
            reason == ".mcs file not found"
            or reason.startswith("could not inspect project:")
        ):
            skippable = True
        if skippable:
            skipped.append(row)
        else:
            hard_failures.append(row)
    return skipped, hard_failures


def _acquire_export_lock(job_runtime):
    if not os.path.isdir(job_runtime):
        try:
            os.makedirs(job_runtime)
        except OSError:
            if not os.path.isdir(job_runtime):
                raise
    lock_path = os.path.join(job_runtime, "_export_batch.lock")
    guard = runtime_common._open_resource_guard(lock_path, wait_seconds=2.0)
    if guard is None:
        return None
    try:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            try:
                with open(lock_path, "r") as handle:
                    pid = int((handle.read() or "0").strip() or "0")
                if pid and _is_pid_alive(pid):
                    return None
            except Exception:
                pass
            try:
                os.remove(lock_path)
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError:
                return None
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return lock_path
    finally:
        runtime_common._close_resource_guard(guard)


def run_background_batch_export(config_path):
    """Run batch export inside a background Mimics process."""
    try:
        with open(config_path, "r") as handle:
            config = json.load(handle)
    except Exception as exc:
        print("Could not read export batch config: {0}".format(exc))
        return 1

    ts_root = config.get("ts_root")
    output_dir = config.get("output_dir")
    output_dir = os.path.abspath(output_dir) if output_dir else _resolve_export_output_dir(ts_root)
    # Emit a startup marker as early as possible. If the per-job Mimics export
    # log is empty, this function never executed — meaning the MimicsResearch
    # -b -run_script child process exited without running the script, which is
    # the real cause of empty fresh_labels / "found 0 labels" in training.
    export_root_early = config.get("export_root") or output_dir
    _append_export_log(export_root_early, "run_background_batch_export entered (pid={0}).".format(os.getpid()))
    # Logs and failed-records go into the dedicated export_root (under workspace),
    # not under mcs_output where .mcs files live.
    export_root = config.get("export_root") or output_dir
    status_path = config.get("status_path") or _rt(export_root, "_export_batch_status.json")
    label_staging_dir = config.get("label_staging_dir") or ""
    label_output_root = config.get("label_output_root") or ""
    overwrite_existing = bool(config.get("overwrite_existing", False))
    export_space = str(config.get("export_space") or "source_image")
    mask_names = config.get("mask_names") or []
    if not isinstance(mask_names, list):
        mask_names = [mask_names]
    source_mask_root = str(config.get("source_mask_root") or "").strip()
    target_mask_name = str(config.get("target_mask_name") or "").strip()
    export_formats = config.get("export_formats") or []
    if isinstance(export_formats, str):
        export_formats = [export_formats]
    export_formats = [
        str(fmt).lower().lstrip(".") for fmt in export_formats if str(fmt).strip()
    ]
    cases_filter = config.get("cases")
    cases_filter = set(cases_filter) if cases_filter else None
    axes = config.get("axes") or [0, 1, 2]
    flips = config.get("flips") or [False, False, False]
    mask_resample_method = str(
        config.get("mask_resample_method") or "nearest"
    ).strip().lower()
    if mask_resample_method not in ("nearest", "distance"):
        _append_export_log(
            export_root,
            "Invalid mask_resample_method {0!r}; using nearest.".format(
                mask_resample_method
            ),
        )
        mask_resample_method = "nearest"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    job_runtime = config.get("job_runtime") or export_root
    lock_path = _acquire_export_lock(job_runtime)
    if not lock_path:
        _append_export_log(export_root, "Another background export process is already running; exiting.")
        # Return non-zero so the caller (fewshot training) does not mistake a
        # skipped export for a successful one and then fail with a misleading
        # "found 0 labels". A return code of 0 here previously let training
        # proceed on an empty fresh_labels directory.
        return 1

    completed = 0
    failed = 0
    cancelled = False
    stop_path = config.get("stop_path") or _rt(export_root, EXPORT_STOP_FILE)
    try:
        _append_export_log(export_root, "Background batch export started.")
        _write_json_atomic(
            status_path,
            {
                "status": "discovering",
                "phase": "discovering_cases",
                "pid": os.getpid(),
                "updated_at_epoch": time.time(),
            },
        )
        # Exclude output/work directories under ts_root so they are not mistaken
        # for cases (mask_exports with a stray .nii.gz would otherwise show up
        # as a "Missing .mcs" case).
        exclude_dirs = [export_root, output_dir, label_staging_dir, label_output_root]
        cases = discover_ts_cases(ts_root, cases_filter, exclude_dirs=exclude_dirs)
        known = set(row.get("case_id") for row in cases)
        for mapped_id, mapped_dir in (config.get("case_dirs") or {}).items():
            if mapped_id not in known and os.path.isdir(mapped_dir):
                cases.append({"case_id": str(mapped_id), "case_dir": os.path.abspath(mapped_dir)})
        cases.sort(key=lambda row: row.get("case_id", ""))
        total = len(cases)
        _append_export_log(export_root, "Discovered {0} case(s) for export.".format(total))
        if total == 0:
            error = "No dataset cases matched the export selection. Check the dataset root and selected case names."
            _write_json_atomic(
                status_path,
                {
                    "status": "failed",
                    "phase": "discovering_cases",
                    "pid": os.getpid(),
                    "completed": 0,
                    "failed": 0,
                    "total": 0,
                    "error": error,
                    "updated_at_epoch": time.time(),
                },
            )
            _append_export_log(export_root, error)
            return 1
        preflight_failures = _preflight_batch_mask_names(
            cases,
            output_dir,
            config,
            export_root,
            status_path,
            stop_path,
        )
        skipped_mask_cases, preflight_failures = _partition_mask_preflight_failures(
            preflight_failures,
            bool(config.get("skip_projects_without_requested_mask", False)),
            bool(config.get("skip_invalid_projects", False)),
        )
        if skipped_mask_cases:
            skipped_ids = set(
                row.get("case_id")
                for row in skipped_mask_cases
                if row.get("case_id")
            )
            cases = [
                row for row in cases if row.get("case_id") not in skipped_ids
            ]
            for row in skipped_mask_cases:
                _append_export_log(
                    export_root,
                    "{0}: skipped during saved-mask preflight: {1}.".format(
                        row.get("case_id") or "(unknown case)",
                        row.get("reason") or "project is not usable",
                    ),
                )
            if not cases and not preflight_failures:
                error = (
                    "None of the selected .mcs projects is usable for the "
                    "requested saved Mask: {0}."
                ).format(", ".join(str(name) for name in mask_names))
                _write_json_atomic(
                    status_path,
                    {
                        "status": "failed",
                        "phase": "mask_name_preflight",
                        "pid": os.getpid(),
                        "completed": 0,
                        "failed": 0,
                        "skipped": len(skipped_mask_cases),
                        "total": total,
                        "error": error,
                        "mask_validation_skipped": skipped_mask_cases[:100],
                        "updated_at_epoch": time.time(),
                    },
                )
                _append_export_log(export_root, error)
                return 1
        if preflight_failures:
            if len(preflight_failures) == 1 and preflight_failures[0].get("reason") == "cancelled":
                _append_export_log(export_root, "Mask export stopped during saved-mask preflight.")
                _write_json_atomic(
                    status_path,
                    {
                        "status": "cancelled",
                        "phase": "mask_name_preflight",
                        "pid": os.getpid(),
                        "completed": 0,
                        "failed": 0,
                        "total": total,
                        "updated_at_epoch": time.time(),
                    },
                )
                return 0
            failed = len(preflight_failures)
            for row in preflight_failures:
                available = row.get("available_masks") or []
                matched = row.get("matched_masks") or []
                detail = "{0}: {1}; available: {2}".format(
                    row.get("case_id") or "(unknown case)",
                    row.get("reason") or "mask validation failed",
                    ", ".join(available) if available else "(none)",
                )
                if matched:
                    detail += "; matched: " + ", ".join(matched)
                _append_export_log(export_root, detail)
                _record_failed_case(
                    export_root,
                    row.get("case_id") or "unknown",
                    "mask_preflight",
                    detail,
                )
            error = (
                "Saved-mask preflight failed for {0} of {1} project(s). No voxel buffers were read and "
                "no labels were exported. Accepted names: {2}. Review the per-case diagnostics."
            ).format(failed, total, ", ".join(str(name) for name in mask_names))
            _write_json_atomic(
                status_path,
                {
                    "status": "failed",
                    "phase": "mask_name_preflight",
                    "pid": os.getpid(),
                    "completed": 0,
                    "failed": failed,
                    "total": total,
                    "error": error,
                    "mask_validation": preflight_failures[:100],
                    "updated_at_epoch": time.time(),
                },
            )
            _append_export_log(export_root, error)
            return 1
        export_total = len(cases)
        # "unanswered" -> "accepted"/"declined" once the user answers the
        # batch-level degraded-export confirmation (if it is ever needed).
        degraded_decision = "unanswered"
        batch_totals = {"new": 0, "overwritten": 0, "unchanged": 0, "skipped_existing": 0}
        skipped_exports = []
        if mask_names:
            _append_export_log(
                export_root,
                "Saved-mask preflight found {0} usable project(s); {1} project(s) "
                "were skipped with case-level diagnostics.".format(
                    export_total,
                    len(skipped_mask_cases),
                ),
            )
        for index, case_info in enumerate(cases):
            if os.path.isfile(stop_path):
                cancelled = True
                _append_export_log(export_root, "Mask export stop requested; no additional cases will be opened.")
                break
            case_id = case_info["case_id"]
            case_dir = case_info["case_dir"]
            mcs_path = str((config.get("mcs_paths") or {}).get(case_id) or os.path.join(output_dir, case_id + ".mcs"))
            work_dir = os.path.join(
                job_runtime, "work", _safe_case_filename(case_id)
            )
            _write_json_atomic(
                status_path,
                {
                    "status": "exporting",
                    "phase": "exporting_voxels",
                    "pid": os.getpid(),
                    "case_id": case_id,
                    "index": index + 1,
                    "total": export_total,
                    "requested_total": total,
                    "skipped": len(skipped_mask_cases),
                    "completed": completed,
                    "failed": failed,
                    "updated_at_epoch": time.time(),
                },
            )
            if not os.path.isfile(mcs_path):
                failed += 1
                _record_failed_case(export_root, case_id, "open_project", ".mcs file not found: {0}".format(mcs_path))
                _append_export_log(export_root, "[{0}/{1}] Missing .mcs: {2}".format(index + 1, export_total, mcs_path))
                continue
            project_opened = False
            try:
                _append_export_log(export_root, "[{0}/{1}] Exporting: {2}".format(index + 1, export_total, case_id))
                mimics.file.open_project(mcs_path)
                project_opened = True
                output_seg_dir = None
                if label_staging_dir:
                    output_seg_dir = os.path.join(label_staging_dir, case_id, "segmentations")
                elif label_output_root:
                    output_seg_dir = os.path.join(label_output_root, case_id, "segmentations")
                built = _export_masks_and_build_params(
                    case_dir,
                    axes,
                    flips,
                    work_dir,
                    output_seg_dir=output_seg_dir,
                    export_space=export_space,
                    mask_names=mask_names,
                    target_mask_name=target_mask_name,
                    overwrite_existing=(True if label_staging_dir else overwrite_existing),
                    source_image_path=(config.get("source_image_paths") or {}).get(case_id),
                    source_mask_root=source_mask_root,
                    case_id=case_id,
                    export_formats=export_formats,
                )
                try:
                    mimics.file.close_project()
                    project_opened = False
                except Exception:
                    pass
                if built is None:
                    _cleanup_work_dir(work_dir)
                    if mask_names:
                        failed += 1
                        error = "No mask matched the requested name(s): {0}".format(
                            ", ".join(str(name) for name in mask_names)
                        )
                        _record_failed_case(export_root, case_id, "mask_selection", error)
                        _append_export_log(export_root, "{0}: {1}".format(case_id, error))
                    else:
                        _append_export_log(export_root, "No masks to export: {0}".format(case_id))
                        completed += 1
                    continue
                bridge_params, manifest = built
                result = call_bridge(
                    bridge_params,
                    extra_env={
                        "MIMICS_MASK_RESAMPLE_METHOD": mask_resample_method,
                    },
                )
                if (
                    result.get("status") != "ok"
                    and result.get("error_code") == "source_geometry_unavailable"
                ):
                    # P1 degraded path: ask once for the whole batch. If the
                    # user declines, the batch stops without silently
                    # switching export spaces mid-run.
                    if degraded_decision == "unanswered":
                        request_id = uuid.uuid4().hex
                        confirm_path = os.path.join(job_runtime, "degraded_confirm.json")
                        try:
                            if os.path.isfile(confirm_path):
                                os.remove(confirm_path)
                        except Exception:
                            pass
                        answer = _await_degraded_confirmation(
                            status_path, confirm_path, request_id, stop_path,
                            export_root,
                            {
                                "case_id": case_id,
                                "index": index + 1,
                                "total": export_total,
                                "completed": completed,
                                "failed": failed,
                            },
                        )
                        if answer is None:
                            cancelled = True
                            break
                        degraded_decision = "accepted" if answer else "declined"
                    if degraded_decision != "accepted":
                        failed += 1
                        _record_failed_case(
                            export_root, case_id, "export",
                            "Source-image export was requested, but the original image geometry "
                            "could not be resolved, and the Mimics-grid fallback was not confirmed.",
                        )
                        _append_export_log(
                            export_root,
                            "{0}: source geometry unavailable; degraded export not confirmed.".format(case_id),
                        )
                        _cleanup_work_dir(work_dir)
                        continue
                    # Confirmed: retry this case on the Mimics grid.
                    bridge_params = dict(bridge_params)
                    bridge_params["export_space"] = "mimics_grid"
                    bridge_params["require_source_geometry"] = False
                    result = call_bridge(
                        bridge_params,
                        extra_env={
                            "MIMICS_MASK_RESAMPLE_METHOD": mask_resample_method,
                        },
                    )
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("error", "bridge returned non-ok status"))
                total_new, total_overwritten, total_unchanged = _apply_export_result(result, work_dir)
                # P1 skip summary: aggregate per-case classifications so the
                # completion dialog can show what actually happened, and dump
                # the not-updated list for one-click retry.
                batch_totals["new"] += int(total_new)
                batch_totals["overwritten"] += int(total_overwritten)
                batch_totals["unchanged"] += int(total_unchanged)
                batch_totals["skipped_existing"] += int(
                    result.get("total_skipped_existing", 0) or 0
                )
                for row in result.get("exported", []) or []:
                    if row.get("action") == "skipped_existing":
                        skipped_exports.append(
                            {
                                "case_id": case_id,
                                "mask_name": row.get("name", ""),
                                "path": row.get("path", ""),
                                "reason": row.get("reason", ""),
                                "per_format_action": row.get("per_format_action", {}),
                            }
                        )
                if not label_staging_dir:
                    try:
                        _record_exported_labels(
                            label_output_root or ts_root,
                            case_id,
                            result,
                            source_image_path=(
                                (config.get("source_image_paths") or {}).get(
                                    case_id
                                )
                                or ""
                            ),
                            mcs_path=mcs_path,
                        )
                        manifest_root = label_output_root or ts_root
                        # A custom label_output_root is an isolated export
                        # destination. Do not update dataset_manifest.json
                        # beside the read-only .mcs store merely because the
                        # MCS directory is also used as output_dir.
                        if (
                            not label_output_root
                            and os.path.normcase(os.path.abspath(output_dir)) != os.path.normcase(
                                os.path.abspath(manifest_root)
                            )
                        ):
                            _record_exported_labels(
                                output_dir,
                                case_id,
                                result,
                                source_image_path=(
                                    (config.get("source_image_paths") or {}).get(
                                        case_id
                                    )
                                    or ""
                                ),
                                mcs_path=mcs_path,
                            )
                    except Exception as manifest_exc:
                        _append_export_log(
                            export_root,
                            "Exported {0}, but dataset manifest update failed: {1}".format(
                                case_id, manifest_exc
                            ),
                        )
                completed += 1
                _append_export_log(
                    export_root,
                    "Exported {0}: new={1}, overwritten={2}, unchanged={3}, existing_preserved={4}".format(
                        case_id,
                        total_new,
                        total_overwritten,
                        total_unchanged,
                        result.get("total_skipped_existing", 0),
                    ),
                )
                if os.path.isfile(stop_path):
                    cancelled = True
                    _append_export_log(export_root, "Mask export stop requested after the current case completed.")
                    break
            except Exception as exc:
                failed += 1
                _append_export_log(export_root, "Export failed for {0}: {1}".format(case_id, exc))
                _record_failed_case(export_root, case_id, "export", exc)
                traceback.print_exc()
                if project_opened:
                    try:
                        mimics.file.close_project()
                    except Exception:
                        pass
                _cleanup_work_dir(work_dir)

        # P1 skip summary: persist the not-updated list next to the labels so
        # it can be retried directly (only these cases, with overwrite on).
        try:
            skipped_path = os.path.join(export_root, "_skipped_exports.json")
            if skipped_exports:
                _write_json_atomic(
                    skipped_path,
                    {
                        "generated_at_epoch": time.time(),
                        "job_runtime": job_runtime,
                        "label_output_root": label_output_root or ts_root,
                        "overwrite_existing": bool(overwrite_existing),
                        "count": len(skipped_exports),
                        "skipped": skipped_exports,
                    },
                )
            elif os.path.isfile(skipped_path):
                os.remove(skipped_path)
        except Exception:
            pass
        _write_json_atomic(
            status_path,
            {
                "status": "cancelled" if cancelled else "closed",
                "phase": "cancelled" if cancelled else "completed",
                "pid": os.getpid(),
                "completed": completed,
                "failed": failed,
                "skipped": len(skipped_mask_cases),
                "total": export_total,
                "requested_total": total,
                "mask_validation_skipped": skipped_mask_cases[:100],
                "batch_totals": dict(batch_totals),
                "degraded_export": degraded_decision == "accepted",
                "skipped_exports_path": (
                    os.path.join(export_root, "_skipped_exports.json")
                    if skipped_exports
                    else ""
                ),
                "updated_at_epoch": time.time(),
            },
        )
        _append_export_log(
            export_root,
            "Background batch export {0}: {1} succeeded, {2} skipped, {3} failed.".format(
                "stopped" if cancelled else "finished",
                completed,
                len(skipped_mask_cases),
                failed,
            ),
        )
        return 0
    except Exception as exc:
        _append_export_log(export_root, "Background batch export stopped unexpectedly: {0}".format(exc))
        _write_json_atomic(
            status_path,
            {
                "status": "failed",
                "phase": "failed",
                "pid": os.getpid(),
                "completed": completed,
                "failed": failed,
                "error": str(exc),
                "updated_at_epoch": time.time(),
            },
        )
        traceback.print_exc()
        return 1
    finally:
        try:
            if os.path.isfile(stop_path):
                os.remove(stop_path)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except Exception:
            pass


# -- Main entry ---------------------------------------------------------

def _explicit_mcs_path_map(mcs_path=None, mcs_list_arg=None, mcs_list_file=None):
    paths = []
    if mcs_path:
        paths.append(str(mcs_path).strip())
    if mcs_list_arg:
        paths.extend(
            item.strip()
            for item in str(mcs_list_arg).replace(";", ",").split(",")
            if item.strip()
        )
    if mcs_list_file:
        with open(mcs_list_file, "r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    paths.append(value)
    result = {}
    for value in paths:
        absolute = os.path.abspath(os.path.expandvars(os.path.expanduser(value)))
        if not os.path.isfile(absolute):
            raise RuntimeError("Explicit .mcs file was not found: {0}".format(absolute))
        if not absolute.lower().endswith(".mcs"):
            raise RuntimeError("Explicit project is not an .mcs file: {0}".format(absolute))
        case_id = os.path.basename(absolute)[:-4]
        if case_id in result and os.path.normcase(result[case_id]) != os.path.normcase(absolute):
            raise RuntimeError(
                "More than one explicit .mcs file has the same case name '{0}'.".format(
                    case_id
                )
            )
        result[case_id] = absolute
    return result


def main(source_info_override=None):
    """Entry point.

    Usage:
        mimics_export.py --case-dir <dir> [--axes 0,1,2] [--flips false,false,false]
        mimics_export.py --ts-root <dir> [--cases s0000,s0001] [--axes 0,1,2] [--flips false,false,false]
    """
    # Cleanup is opt-in. Use Stop_Background_Services.py for manual cleanup.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        from mimics_import import _cleanup_stale_processes
        if runtime_common.auto_cleanup_enabled():
            cleanup_thread = threading.Thread(target=_cleanup_stale_processes)
            cleanup_thread.daemon = True
            cleanup_thread.start()
    except Exception:
        pass

    ts_root = None
    case_dir = None
    cases_filter = None
    axes = [0, 1, 2]
    flips = [False, False, False]
    label_output_root = None
    overwrite_existing = False
    mcs_dir = None
    mcs_path_override = None
    external_setup = False
    mask_names = None
    # Batch export of an explicit list of .mcs files, bypassing the TS-style
    # directory/image discovery in discover_ts_cases. Either a comma-separated
    # list or a text file with one path per line.
    mcs_list_arg = None
    mcs_list_file = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--ts-root" and i + 1 < len(args):
            ts_root = args[i + 1]
            i += 2
        elif arg == "--case-dir" and i + 1 < len(args):
            case_dir = args[i + 1]
            i += 2
        elif arg == "--cases" and i + 1 < len(args):
            cases_filter = set(c.strip() for c in args[i + 1].split(","))
            i += 2
        elif arg == "--axes" and i + 1 < len(args):
            axes = [int(v.strip()) for v in args[i + 1].split(",")]
            i += 2
        elif arg == "--flips" and i + 1 < len(args):
            parts = [v.strip().lower() for v in args[i + 1].split(",")]
            flips = [p in ("true", "1", "yes") for p in parts]
            i += 2
        elif arg == "--label-output-root" and i + 1 < len(args):
            label_output_root = args[i + 1]
            i += 2
        elif arg == "--mcs-dir" and i + 1 < len(args):
            mcs_dir = args[i + 1]
            i += 2
        elif arg == "--mcs-path" and i + 1 < len(args):
            mcs_path_override = args[i + 1]
            i += 2
        elif arg == "--overwrite-source":
            overwrite_existing = True
            i += 1
        elif arg == "--external-setup":
            external_setup = True
            i += 1
        elif arg == "--mask-names" and i + 1 < len(args):
            mask_names = [value.strip() for value in args[i + 1].split(",") if value.strip()]
            i += 2
        elif arg == "--mcs-list" and i + 1 < len(args):
            mcs_list_arg = args[i + 1]
            i += 2
        elif arg == "--mcs-list-file" and i + 1 < len(args):
            mcs_list_file = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        explicit_mcs_paths = _explicit_mcs_path_map(
            mcs_path=mcs_path_override,
            mcs_list_arg=mcs_list_arg,
            mcs_list_file=mcs_list_file,
        )
    except Exception as exc:
        _mimics_log(logging.ERROR, "Mask export arguments are invalid: {0}".format(exc))
        mimics.dialogs.message_box(
            title="Export Arguments Invalid",
            message=str(exc),
            ui_blocking=False,
        )
        return 1

    # Interactive setup runs in external PySide6; Mimics remains responsive.
    if not ts_root and not case_dir and not explicit_mcs_paths:
        return _launch_external_export_setup()

    if explicit_mcs_paths and not ts_root and not case_dir:
        message = (
            "Explicit .mcs export also requires --ts-root so each project can be "
            "matched to its original image grid."
        )
        _mimics_log(logging.ERROR, message)
        mimics.dialogs.message_box(
            title="Dataset Root Required",
            message=message,
            ui_blocking=False,
        )
        return 1

    # -- Single case mode ----------------------------------------------

    if case_dir and not explicit_mcs_paths:
        case_id = _current_project_case_id() or os.path.basename(os.path.abspath(case_dir))
        ts_root = os.path.dirname(os.path.abspath(case_dir))
        if not label_output_root and not overwrite_existing:
            label_output_root = os.path.join(ts_root, "mask_exports")
        source_image_path = str((source_info_override or {}).get("image") or case_dir)
        try:
            _start_current_project_export(
                case_dir,
                source_image_path,
                label_output_root,
                axes,
                flips,
                mask_names=mask_names,
                overwrite_existing=overwrite_existing,
            )
        except Exception as exc:
            _mimics_log(logging.ERROR, "Mask export could not start: {0}".format(exc))
            mimics.dialogs.message_box(
                title="Export Could Not Start",
                message=str(exc),
                ui_blocking=False,
            )
            return 1
        return 0

    # -- Batch mode -----------------------------------------------------
    case_dirs = None
    source_image_paths = None
    if case_dir and explicit_mcs_paths:
        case_id = os.path.basename(os.path.abspath(case_dir))
        if len(explicit_mcs_paths) != 1:
            message = "--case-dir can be paired with only one explicit .mcs file."
            _mimics_log(logging.ERROR, message)
            mimics.dialogs.message_box(
                title="Export Arguments Invalid",
                message=message,
                ui_blocking=False,
            )
            return 1
        explicit_path = list(explicit_mcs_paths.values())[0]
        explicit_mcs_paths = {case_id: explicit_path}
        cases_filter = set([case_id])
        case_dirs = {case_id: os.path.abspath(case_dir)}
        source_image_paths = {
            case_id: str((source_info_override or {}).get("image") or case_dir)
        }
        ts_root = os.path.dirname(os.path.abspath(case_dir))
    elif explicit_mcs_paths:
        cases_filter = set(explicit_mcs_paths)

    launch_thread = _launch_background_batch_export_async(
        ts_root, cases_filter, axes, flips,
        label_output_root=label_output_root,
        overwrite_existing=overwrite_existing,
        case_dirs=case_dirs,
        mcs_output_dir=mcs_dir,
        mcs_paths=explicit_mcs_paths,
        source_image_paths=source_image_paths,
        mask_names=mask_names,
    )
    if launch_thread is None:
        return 1
    return 0


# -- Quick export -------------------------------------------------------

def quick_export_main():
    """Open the supported external setup and incremental export workflow."""
    return _launch_external_export_setup()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        try:
            mimics.dialogs.message_box(title="Fatal Error", message=str(error))
        except Exception:
            pass
        raise
