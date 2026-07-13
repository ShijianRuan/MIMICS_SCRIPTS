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
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import mimics

import runtime_common


# -- Global async monitor state ----------------------------------------
_EXPORT_MONITORS = {}
_LOG_ROTATE_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_BACKUPS = 3
_CONFIG_CACHE = None
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"


_write_json_atomic = runtime_common.write_json_atomic
_safe_case_filename = runtime_common.safe_filename
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs


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
        path = os.path.join(root_dir or os.getcwd(), "mimics_export.log")
        _rotate_log_file(path)
        with open(path, "a") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _record_failed_case(output_dir_or_export_root, case_id, phase, error):
    try:
        failed_dir = os.path.join(output_dir_or_export_root or os.getcwd(), "_failed_exports")
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


# -- Path helpers (shared with mimics_import.py) ------------------------


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "nninteractive_env", ".git"),
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


def _resource_lock_path(name):
    return runtime_common.resource_lock_path(_project_root(), name)


def _environment_root():
    root = _project_root()
    candidates = [
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
    return runtime_common.background_env(extra, include_itk=True)


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
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Materialise", "Mimics Research 21.0", "MimicsResearch.exe"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Mimics Research 21.0", "MimicsResearch.exe"),
        "D:\\Mimics Research 21.0\\MimicsResearch.exe",
        "C:\\Mimics Research 21.0\\MimicsResearch.exe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    for drive in ["C:", "D:", "E:"]:
        for name in ["Mimics Research 21.0", "MimicsResearch 21.0"]:
            path = os.path.join(drive + "\\", name, "MimicsResearch.exe")
            if os.path.isfile(path):
                return path
    return None


def _launch_background_batch_export(ts_root, cases_filter, axes, flips, label_output_root=None, overwrite_existing=False, case_dirs=None):
    """Launch batch export in a separate background Mimics process."""
    output_dir = _resolve_export_output_dir(ts_root)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    mimics_exe = _find_mimics_exe()
    if not mimics_exe:
        _append_export_log(output_dir, "MimicsResearch.exe was not found; background export cannot start.")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(output_dir, "_export_batch_config.json")
    runner_path = os.path.join(output_dir, "_run_export_batch.py")
    _write_json_atomic(
        config_path,
        {
            "ts_root": os.path.abspath(ts_root),
            "output_dir": os.path.abspath(output_dir),
            "cases": sorted(list(cases_filter)) if cases_filter else None,
            "axes": axes,
            "flips": flips,
            "label_output_root": os.path.abspath(label_output_root) if label_output_root else "",
            "overwrite_existing": bool(overwrite_existing),
            "case_dirs": dict(case_dirs or {}),
        },
    )
    with open(runner_path, "w") as f:
        f.write("# Auto-generated runner for background Mimics batch export\n")
        f.write("import sys, os\n")
        f.write("sys.path.insert(0, r'{0}')\n".format(here))
        f.write("import mimics_export\n")
        f.write("mimics_export.run_background_batch_export(r'{0}')\n".format(config_path))

    log_path = os.path.join(output_dir, "_background_export_mimics.log")
    _rotate_log_file(log_path)
    lock_path = _resource_lock_path("background_mimics.lock")
    lock_token = runtime_common.acquire_resource_lock(
        lock_path,
        "background_mimics",
        "batch label export",
        wait_seconds=0.0,
    )
    if not lock_token:
        _append_export_log(
            output_dir,
            "Background Mimics is already running for another Mimics-Script task; batch export was not started.",
        )
        return None
    log_handle = None
    try:
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            [mimics_exe, "-b", "-run_script", runner_path],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_background_process_kwargs()
        )
        runtime_common.update_resource_lock_pid(
            lock_path,
            lock_token,
            process.pid,
            {"kind": "export_labels", "ts_root": os.path.abspath(ts_root)},
        )
        _append_export_log(output_dir, "Background Mimics started (PID={0}) for batch export.".format(process.pid))
        return process
    except Exception as exc:
        runtime_common.release_resource_lock(lock_path, lock_token)
        _append_export_log(output_dir, "Could not start background batch export: {0}".format(exc))
        return None
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


def call_bridge(params):
    """Call mimics_bridge.py via subprocess, return parsed JSON result."""
    python_exe = _python_exe()
    bridge = _bridge_script()

    process = subprocess.Popen(
        [python_exe, bridge],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_background_env(),
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
    """Check if a process with given PID is still running."""
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_INFORMATION = 0x0400
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong(0)
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == STILL_ACTIVE
        else:
            os.kill(int(pid), 0)
            return True
    except Exception:
        return False


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


def export_masks_to_buffers(buffers_dir, mask_names=None):
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

        safe_name = _sanitize_name(name)
        u8_path = os.path.join(buffers_dir, safe_name + ".u8")

        with open(u8_path, "wb") as f:
            f.write(raw)

        manifest["masks"].append({
            "original_name": name,
            "safe_name": safe_name,
            "u8_filename": safe_name + ".u8",
        })
        print("  -> {0}".format(safe_name))

    # Save manifest
    manifest_path = os.path.join(buffers_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Manifest saved: {0}".format(manifest_path))

    return manifest


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


# -- Single case export -------------------------------------------------

def _export_masks_and_build_params(case_dir, axes, flips, work_dir, output_seg_dir=None, export_space="source_image", mask_names=None, overwrite_existing=None):
    """Export masks to .u8 buffers and build bridge params. Returns (bridge_params, manifest) or None."""
    print("Exporting masks to: {0}".format(case_dir))

    buffers_dir = os.path.join(work_dir, "export_buffers")

    # Step 1: Export masks to .u8 buffers
    manifest = export_masks_to_buffers(buffers_dir, mask_names=mask_names)

    if not manifest["masks"]:
        print("No masks to export")
        return None

    mimics_shape = manifest.get("mimics_shape")
    if not mimics_shape:
        print("Error: could not determine mimics_shape")
        return None

    # Step 2: Build bridge params for convert action
    bridge_params = {
        "action": "convert",
        "buffers_dir": buffers_dir,
        "manifest_path": os.path.join(buffers_dir, "manifest.json"),
        "case_dir": case_dir,
        "axes": axes,
        "flips": flips,
        "export_space": str(export_space or "source_image"),
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
    source_image_path = _metadata_get(active_image, SOURCE_IMAGE_PATH_METADATA, "") if active_image is not None else ""
    if source_image_path:
        bridge_params["source_image_path"] = str(source_image_path)
    if output_seg_dir:
        bridge_params["output_seg_dir"] = output_seg_dir
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

def discover_ts_cases(ts_root, case_filter=None):
    """Find all cases in a TS-like dataset."""
    cases = []
    for name in sorted(os.listdir(ts_root)):
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name in ("mcs_output", "segmentations"):
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


def _acquire_export_lock(output_dir):
    lock_path = os.path.join(output_dir, "_export_batch.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return lock_path
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
        except Exception:
            return None
        return _acquire_export_lock(output_dir)


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
    # Logs and failed-records go into the dedicated export_root (under workspace),
    # not under mcs_output where .mcs files live.
    export_root = config.get("export_root") or output_dir
    label_staging_dir = config.get("label_staging_dir") or ""
    label_output_root = config.get("label_output_root") or ""
    overwrite_existing = bool(config.get("overwrite_existing", False))
    export_space = str(config.get("export_space") or "source_image")
    mask_names = config.get("mask_names") or []
    if not isinstance(mask_names, list):
        mask_names = [mask_names]
    cases_filter = config.get("cases")
    cases_filter = set(cases_filter) if cases_filter else None
    axes = config.get("axes") or [0, 1, 2]
    flips = config.get("flips") or [False, False, False]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    lock_path = _acquire_export_lock(export_root)
    if not lock_path:
        _append_export_log(export_root, "Another background export process is already running; exiting.")
        return 0

    completed = 0
    failed = 0
    try:
        _append_export_log(export_root, "Background batch export started.")
        _write_json_atomic(
            os.path.join(export_root, "_export_batch_status.json"),
            {"status": "discovering", "pid": os.getpid(), "updated_at_epoch": time.time()},
        )
        cases = discover_ts_cases(ts_root, cases_filter)
        known = set(row.get("case_id") for row in cases)
        for mapped_id, mapped_dir in (config.get("case_dirs") or {}).items():
            if mapped_id not in known and os.path.isdir(mapped_dir):
                cases.append({"case_id": str(mapped_id), "case_dir": os.path.abspath(mapped_dir)})
        cases.sort(key=lambda row: row.get("case_id", ""))
        total = len(cases)
        _append_export_log(export_root, "Discovered {0} case(s) for export.".format(total))
        for index, case_info in enumerate(cases):
            case_id = case_info["case_id"]
            case_dir = case_info["case_dir"]
            mcs_path = os.path.join(output_dir, case_id + ".mcs")
            work_dir = os.path.join(output_dir, case_id + "_export_work")
            _write_json_atomic(
                os.path.join(export_root, "_export_batch_status.json"),
                {
                    "status": "exporting",
                    "pid": os.getpid(),
                    "case_id": case_id,
                    "index": index + 1,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "updated_at_epoch": time.time(),
                },
            )
            if not os.path.isfile(mcs_path):
                failed += 1
                _record_failed_case(export_root, case_id, "open_project", ".mcs file not found: {0}".format(mcs_path))
                _append_export_log(export_root, "[{0}/{1}] Missing .mcs: {2}".format(index + 1, total, mcs_path))
                continue
            try:
                _append_export_log(export_root, "[{0}/{1}] Exporting: {2}".format(index + 1, total, case_id))
                mimics.file.open_project(mcs_path)
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
                    overwrite_existing=(True if label_staging_dir else overwrite_existing),
                )
                try:
                    mimics.file.close_project()
                except Exception:
                    pass
                if built is None:
                    _cleanup_work_dir(work_dir)
                    _append_export_log(export_root, "No masks to export: {0}".format(case_id))
                    completed += 1
                    continue
                bridge_params, manifest = built
                result = call_bridge(bridge_params)
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("error", "bridge returned non-ok status"))
                total_new, total_overwritten, total_unchanged = _apply_export_result(result, work_dir)
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
            except Exception as exc:
                failed += 1
                _append_export_log(export_root, "Export failed for {0}: {1}".format(case_id, exc))
                _record_failed_case(export_root, case_id, "export", exc)
                traceback.print_exc()
                try:
                    mimics.file.close_project()
                except Exception:
                    pass
                _cleanup_work_dir(work_dir)

        _write_json_atomic(
            os.path.join(export_root, "_export_batch_status.json"),
            {
                "status": "closed",
                "pid": os.getpid(),
                "completed": completed,
                "failed": failed,
                "updated_at_epoch": time.time(),
            },
        )
        _append_export_log(export_root, "Background batch export finished: {0} succeeded, {1} failed.".format(completed, failed))
        return 0
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


# -- Main entry ---------------------------------------------------------

def main():
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
        elif arg == "--overwrite-source":
            overwrite_existing = True
            i += 1
        else:
            i += 1

    # Interactive: if no args, ask for case dir or TS root
    if not ts_root and not case_dir:
        # Always let the annotator confirm the source case directory. Project
        # metadata may point to an old location after a dataset/project copy.
        case_dir = _pick_directory("Select source case directory")
        if not case_dir or not os.path.isdir(case_dir):
            mimics.dialogs.message_box(
                "No valid directory selected.",
                title="Export Masks",
                ui_blocking=True,
            )
            return 1

    # -- Single case mode ----------------------------------------------

    if case_dir:
        case_id = _current_project_case_id() or os.path.basename(os.path.abspath(case_dir))
        ts_root = os.path.dirname(os.path.abspath(case_dir))
        output_dir = _resolve_export_output_dir(ts_root)
        mcs_path = os.path.join(output_dir, case_id + ".mcs")
        if not os.path.isfile(mcs_path):
            mimics.dialogs.message_box(
                title="Export Error",
                message=(
                    "Foreground mask export is disabled to keep Mimics responsive.\n"
                    "Save the project first, then run export again.\n\n"
                    "Expected .mcs path:\n{0}"
                ).format(mcs_path),
            )
            return 1
        if not label_output_root and not overwrite_existing:
            destination = mimics.dialogs.question_box(
                title="Export Masks",
                message=(
                    "Export all masks from the saved project.\n\n"
                    "Safe Copy writes to a folder you select and preserves existing files.\n"
                    "Overwrite Original updates <case>/segmentations in place."
                ),
                buttons="Safe Copy;Overwrite Original;Cancel",
                ui_blocking=True,
            )
            if destination == "Safe Copy":
                label_output_root = _pick_directory("Select mask export destination")
                if not label_output_root:
                    return 1
            elif destination == "Overwrite Original":
                overwrite_existing = True
            else:
                return 1
        process = _launch_background_batch_export(
            ts_root, set([case_id]), axes, flips,
            label_output_root=label_output_root,
            overwrite_existing=overwrite_existing,
            case_dirs={case_id: os.path.abspath(case_dir)},
        )
        if process is None:
            mimics.dialogs.message_box(
                title="Export Error",
                message="Could not start background export. See mimics_export.log in output directory.",
            )
            return 1
        mimics.dialogs.message_box(
            title="Export Started",
            message=(
                "Export for {0} is running in a background Mimics process.\n"
                "The current Mimics window remains available.\n\n"
                "Status/logs: {1}"
            ).format(case_id, output_dir),
        )
        return 0

    # -- Batch mode -----------------------------------------------------
    process = _launch_background_batch_export(
        ts_root, cases_filter, axes, flips,
        label_output_root=label_output_root,
        overwrite_existing=overwrite_existing,
    )
    if process is None:
        output_dir = _resolve_export_output_dir(ts_root)
        mimics.dialogs.message_box(
            title="Export Error",
            message="Could not start background batch export. See mimics_export.log in {0}.".format(output_dir),
        )
        return 1
    output_dir = _resolve_export_output_dir(ts_root)
    mimics.dialogs.message_box(
        title="Export Started",
        message=(
            "Batch export is running in a background Mimics process.\n"
            "The current Mimics window remains available.\n\n"
            "Status/logs: {0}"
        ).format(output_dir),
    )
    return 0


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
