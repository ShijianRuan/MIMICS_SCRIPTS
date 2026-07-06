# -*- coding: utf-8 -*-
"""Mimics-internal driver for dataset -> .mcs import.

Runs inside Mimics Python 3.5.2. Only uses stdlib + mimics API.
Calls mimics_bridge.py (in nninteractive_env) via subprocess for NIfTI/DICOM work.

Uses Win32 SetTimer / PyQt5 QTimer for non-blocking async polling,
matching nnInteractive's pattern; no manual second click needed.

Flow:
    1. Annotator picks a case directory
    2. Launch mimics_bridge.py "prepare" in background (non-blocking)
    3. Timer polls every 0.5s until bridge completes
    4. Save a prepare manifest and let background Mimics create .mcs
    5. For batch: timer processes cases one-by-one automatically
"""

from __future__ import print_function

import json
import logging
import os
import hashlib
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
_IMPORT_MONITORS = {}

# Track background Mimics process for .mcs creation
_BG_MIMICS_PID = None
_BG_MIMICS_BUSY_NOTICE_AT = 0.0
_BG_MIMICS_RETRY_ACTIVE = False
_MCS_QUEUE_ACTIVE = "_mcs_queue_active.json"
_MCS_QUEUE_DONE = "_mcs_queue_done.json"
_MCS_QUEUE_STOP = "_mcs_queue_stop.json"
_LOG_ROTATE_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_BACKUPS = 3


def _auto_open_mcs_enabled():
    return os.environ.get("MIMICS_IMPORT_AUTO_OPEN_MCS", "").strip().lower() in ("1", "true", "yes")


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


def _append_import_log(root_dir, message):
    text = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    logged_to_mimics = _mimics_log(logging.INFO, text)
    if not logged_to_mimics:
        print(text)
    try:
        if root_dir and not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        path = os.path.join(root_dir or os.getcwd(), "logs", "mimics_import.log")
        _rotate_log_file(path)
        with open(path, "a") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _queue_active_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_ACTIVE)


def _queue_done_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_DONE)


def _queue_stop_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_STOP)


def _mcs_queue_registry_dir():
    return os.path.join(_project_root(), ".mimics_runtime", "mcs_queues")


def _register_mcs_queue(output_dir, total_count=0):
    try:
        registry_dir = _mcs_queue_registry_dir()
        if not os.path.isdir(registry_dir):
            os.makedirs(registry_dir)
        digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8", "replace")).hexdigest()[:16]
        base = _safe_case_filename(os.path.basename(os.path.abspath(output_dir))).strip("._") or "queue"
        name = "{0}_{1}".format(base, digest)
        path = os.path.join(registry_dir, name + ".json")
        _write_json_atomic(
            path,
            {
                "output_dir": os.path.abspath(output_dir),
                "total_count": int(total_count or 0),
                "updated_at_epoch": time.time(),
            },
        )
    except Exception:
        pass


def _mark_mcs_queue_active(output_dir, total_count=0):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    done_path = _queue_done_path(output_dir)
    stop_path = _queue_stop_path(output_dir)
    try:
        if os.path.isfile(done_path):
            os.remove(done_path)
    except Exception:
        pass
    try:
        if os.path.isfile(stop_path):
            os.remove(stop_path)
    except Exception:
        pass
    _write_json_atomic(
        _queue_active_path(output_dir),
        {
            "status": "active",
            "total_count": int(total_count or 0),
            "updated_at_epoch": time.time(),
        },
    )
    _register_mcs_queue(output_dir, total_count)


def _mark_mcs_queue_done(output_dir, completed=0, failed=0):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    _write_json_atomic(
        _queue_done_path(output_dir),
        {
            "status": "done",
            "completed": int(completed or 0),
            "failed": int(failed or 0),
            "updated_at_epoch": time.time(),
        },
    )
    try:
        active = _queue_active_path(output_dir)
        if os.path.isfile(active):
            os.remove(active)
    except Exception:
        pass
    _register_mcs_queue(output_dir, completed + failed)


def _background_stop_requested(output_dir, since_epoch=0.0):
    try:
        stop_path = _queue_stop_path(output_dir)
        if not os.path.isfile(stop_path):
            return False
        if since_epoch and os.path.getmtime(stop_path) < float(since_epoch):
            return False
        return True
    except Exception:
        return False


def _save_prepare_manifest(work_dir, result, output_mcs=None):
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    manifest = dict(result)
    if output_mcs:
        manifest["output_mcs"] = os.path.abspath(output_mcs)
    path = os.path.join(work_dir, "prepare_manifest.json")
    _write_json_atomic(path, manifest)
    return path


def _record_failed_case(output_dir, case_id, phase, error):
    try:
        failed_dir = os.path.join(output_dir or os.getcwd(), "_failed_cases")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir)
        payload = {
            "case_id": str(case_id or "unknown"),
            "phase": str(phase or "unknown"),
            "error": str(error or ""),
            "failed_at_epoch": time.time(),
        }
        filename = "{0}_{1}.json".format(
            _safe_case_filename(case_id),
            _safe_case_filename(phase),
        )
        _write_json_atomic(os.path.join(failed_dir, filename), payload)
    except Exception:
        pass


# -- Path helpers (same pattern as nninteractive_mimics.py) -------------


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "nninteractive_env", ".git"),
    )


def _resource_lock_path(name):
    return runtime_common.resource_lock_path(_project_root(), name)


def _resource_lock_dir():
    return runtime_common.resource_lock_dir(_project_root())


def _aggressive_auto_cleanup_enabled():
    return runtime_common.aggressive_auto_cleanup_enabled()


def _environment_root():
    root = _project_root()
    candidates = [
        os.path.join(root, "nninteractive_env"),
        os.path.join(os.path.dirname(root), "nninteractive_env"),
        root,
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "Scripts", "python.exe")):
            return candidate
        if os.path.isfile(os.path.join(candidate, "python", "python.exe")):
            return candidate
    return candidates[0]


def _bridge_script():
    """Find mimics_bridge.py."""
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
        os.path.join(env_root, "Scripts", "python.exe"),
        os.path.join(env_root, "python", "python.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


# -- Stale process / temp cleanup --------------------------------------

def _cleanup_stale_processes():
    """Clean safe stale runtime state from a previous crashed session.

    By default this only removes Mimics-Script resource lock files whose PID
    no longer exists. Process termination is deliberately opt-in via
    MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START=1 or the explicit Stop Background
    Services entry, because killing live bridge/background Mimics processes
    can interrupt a valid async workflow.

    Uses a single hidden batch PowerShell call instead of per-process calls
    to avoid popping up visible console windows that freeze Mimics.
    """
    killed = []
    locks_removed = runtime_common.cleanup_stale_resource_locks(_resource_lock_dir())
    if not _aggressive_auto_cleanup_enabled():
        if locks_removed:
            print("Startup cleanup: removed {0} stale resource lock file(s)".format(locks_removed))
        return

    # Substrings that identify nnInteractive server / watchdog processes
    # which must NOT be killed during cleanup.
    _SERVER_PROTECT_MARKERS = (
        "nninteractive.inference.server.main",
        "--watchdog",
    )

    # 1. Kill stale bridge python processes (nninteractive_env python.exe)
    #    and background Mimics processes (-b flag).
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_TERMINATE = 0x0001

        # Single hidden PowerShell call to get all python.exe and
        # mimicsresearch.exe PIDs and command lines at once.
        proc = subprocess.Popen(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'mimicsresearch.exe' } | "
                "Select-Object ProcessId,Name,CommandLine | "
                "ConvertTo-Json -Compress",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_hidden_process_kwargs()
        )
        stdout, _ = proc.communicate(timeout=15)
        if proc.returncode == 0 and stdout and stdout.strip():
            try:
                records = json.loads(stdout.decode("utf-8", errors="replace"))
            except ValueError:
                records = []
            if isinstance(records, dict):
                records = [records]

            env_root = _environment_root().lower()
            for record in records or []:
                try:
                    pid = int(record.get("ProcessId", 0))
                except (TypeError, ValueError):
                    continue
                if not pid:
                    continue
                name = str(record.get("Name") or "").lower()
                cmdline = str(record.get("CommandLine") or "").lower()
                is_protected = any(
                    marker in cmdline
                    for marker in _SERVER_PROTECT_MARKERS
                )
                if is_protected:
                    continue
                should_kill = False
                if name == "python.exe" and env_root in cmdline:
                    should_kill = True
                elif name == "mimicsresearch.exe" and "-b" in cmdline:
                    should_kill = True
                if should_kill:
                    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        killed.append((name, pid))
    except Exception:
        pass  # Best-effort; don't crash if cleanup fails

    # 2. Remove stale Mimics temp lock files
    try:
        temp_dir = os.environ.get("TEMP", "")
        if temp_dir and os.path.isdir(temp_dir):
            for root, dirs, files in os.walk(temp_dir):
                for fname in files:
                    if fname == "__fold__.lock":
                        # Only remove locks in Mimics-related directories
                        if "mimics" in root.lower() or "materialise" in root.lower():
                            try:
                                os.remove(os.path.join(root, fname))
                                locks_removed += 1
                            except Exception:
                                pass
    except Exception:
        pass

    if killed or locks_removed:
        parts = []
        if killed:
            parts.append("terminated {0} stale process(es)".format(len(killed)))
        if locks_removed:
            parts.append("removed {0} lock file(s)".format(locks_removed))
        print("Startup cleanup: " + ", ".join(parts))


# -- TS case discovery (runs in Mimics Python 3.5, stdlib only) ---------

def discover_ts_cases(ts_root, case_filter=None):
    """Find all cases in a TS-like dataset. Returns list of dicts."""
    cases = []
    for name in sorted(os.listdir(ts_root)):
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name in ("mcs_output", "segmentations"):
            continue
        if case_filter and name not in case_filter:
            continue

        image_path = None
        image_type = None

        for img_name in ("ct.nii.gz", "mri.nii.gz"):
            candidate = os.path.join(case_dir, img_name)
            if os.path.isfile(candidate):
                image_path = candidate
                image_type = "nifti"
                break

        if image_path is None:
            for fname in sorted(os.listdir(case_dir)):
                if fname.endswith(".nii.gz") and fname not in ("ct.nii.gz", "mri.nii.gz"):
                    candidate = os.path.join(case_dir, fname)
                    if os.path.isfile(candidate):
                        image_path = candidate
                        image_type = "nifti"
                        break

        if image_path is None:
            dicom_dir = os.path.join(case_dir, "dicom")
            if os.path.isdir(dicom_dir):
                image_path = dicom_dir
                image_type = "dicom"

        if image_path is None:
            continue

        masks = []
        seg_dir = os.path.join(case_dir, "segmentations")
        if os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                if fname.endswith(".nii.gz"):
                    organ = fname.replace(".nii.gz", "")
                    masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

        cases.append({
            "case_id": name,
            "image": image_path,
            "image_type": image_type,
            "masks": masks,
            "case_dir": case_dir,
        })
    return cases


def _discover_single_case(case_dir):
    """Discover image and masks for a single case directory.

    Returns case_info dict, or None if no valid image found.
    Used for lazy discovery; each case is scanned only when it's
    about to be converted, avoiding a blocking full-dataset scan.
    """
    name = os.path.basename(case_dir)
    if not os.path.isdir(case_dir):
        return None
    if name in ("mcs_output", "segmentations"):
        return None

    image_path = None
    image_type = None

    for img_name in ("ct.nii.gz", "mri.nii.gz"):
        candidate = os.path.join(case_dir, img_name)
        if os.path.isfile(candidate):
            image_path = candidate
            image_type = "nifti"
            break

    if image_path is None:
        for fname in sorted(os.listdir(case_dir)):
            if fname.endswith(".nii.gz") and fname not in ("ct.nii.gz", "mri.nii.gz"):
                candidate = os.path.join(case_dir, fname)
                if os.path.isfile(candidate):
                    image_path = candidate
                    image_type = "nifti"
                    break

    if image_path is None:
        dicom_dir = os.path.join(case_dir, "dicom")
        if os.path.isdir(dicom_dir):
            image_path = dicom_dir
            image_type = "dicom"

    if image_path is None:
        return None

    masks = []
    seg_dir = os.path.join(case_dir, "segmentations")
    if os.path.isdir(seg_dir):
        for fname in sorted(os.listdir(seg_dir)):
            if fname.endswith(".nii.gz"):
                organ = fname.replace(".nii.gz", "")
                masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

    return {
        "case_id": name,
        "image": image_path,
        "image_type": image_type,
        "masks": masks,
        "case_dir": case_dir,
    }


# -- Call mimics_bridge.py ---------------------------------------------

def call_bridge(params):
    """Call mimics_bridge.py via subprocess, return parsed JSON result."""
    python_exe = _python_exe()
    bridge = _bridge_script()

    process = subprocess.Popen(
        [python_exe, bridge],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_hidden_process_kwargs()
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

    _write_json_atomic(input_file, bridge_params)

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


def _launch_bridge_job_thread(bridge_params, job_dir, output_dir, phase, case_id=None):
    """Start bridge from a worker thread so Mimics GUI can repaint first."""
    def _run():
        try:
            process = _launch_bridge_background(bridge_params, job_dir)
            state = {
                "phase": phase,
                "pid": process.pid,
                "started_at": time.time(),
            }
            if case_id:
                state["case_id"] = case_id
            _write_json_atomic(os.path.join(job_dir, "job_state.json"), state)
            _append_import_log(output_dir, "Bridge process started (PID={0}) for {1}.".format(process.pid, phase))
        except Exception as exc:
            _append_import_log(output_dir, "Could not start bridge process for {0}: {1}".format(phase, exc))
            try:
                _write_json_atomic(
                    os.path.join(job_dir, "bridge_result.json"),
                    {"status": "error", "error": str(exc)},
                )
            except Exception:
                pass

    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)
    _write_json_atomic(
        os.path.join(job_dir, "job_state.json"),
        {
            "phase": "launching",
            "requested_phase": phase,
            "case_id": case_id or "",
            "started_at": time.time(),
        },
    )
    thread = threading.Thread(target=_run)
    thread.daemon = True
    thread.start()
    return thread


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
    phase = ""
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            pid = state.get("pid")
            phase = state.get("phase", "")
        except (ValueError, IOError):
            pass
    if phase == "launching":
        return ("running", None)

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


def _cleanup_work_dir(work_dir):
    """Remove an incomplete or already-consumed work directory."""
    if not work_dir or not os.path.isdir(work_dir):
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass


def _terminate_job_process(job_dir):
    """Best-effort termination for a timed-out bridge process."""
    state_file = os.path.join(job_dir or "", "job_state.json")
    pid = None
    try:
        with open(state_file, "r") as handle:
            state = json.load(handle)
        pid = int(state.get("pid") or 0)
    except Exception:
        pid = None
    if not pid or not _is_pid_alive(pid):
        return
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_hidden_process_kwargs()
            )
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


# -- Timer-based async monitor (same pattern as nnInteractive) ----------

def _stop_import_monitor(monitor_key):
    """Stop and clean up a running import monitor."""
    monitor = _IMPORT_MONITORS.pop(monitor_key, None)
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


def _import_monitor_tick(monitor):
    """Timer callback: check if bridge finished, then queue .mcs creation."""
    if monitor.get("done"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        output_mcs = monitor.get("output_mcs")
        output_dir = os.path.dirname(os.path.abspath(output_mcs)) if output_mcs else ""
        _terminate_job_process(job_dir)
        _record_failed_case(output_dir, monitor.get("case_id"), "prepare_timeout", "Dataset preparation timed out.")
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        if not monitor.get("batch_queue") and output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
        mimics.dialogs.message_box(
            title="Import Timeout",
            message="Dataset preparation timed out. Please retry.",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    monitor["done"] = True
    _stop_import_monitor(monitor_key)

    if status == "error":
        output_mcs = monitor.get("output_mcs")
        output_dir = os.path.dirname(os.path.abspath(output_mcs)) if output_mcs else ""
        _record_failed_case(output_dir, monitor.get("case_id"), "prepare", result)
        mimics.dialogs.message_box(title="Import Error", message="Preparation failed: {0}".format(result))
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        elif output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
        return

    # status == "done"
    try:
        output_mcs = monitor.get("output_mcs")
        work_dir = monitor.get("work_dir")
        _save_prepare_manifest(work_dir, result, output_mcs=output_mcs)
        _cleanup_job_dir(job_dir)
        output_dir = os.path.dirname(os.path.abspath(output_mcs))
        _mark_mcs_queue_active(output_dir, monitor.get("total", 1))
        _ensure_bg_mimics_running(output_dir, monitor.get("total", 1), mark_active=False)
        _mark_mcs_queue_done(output_dir, completed=1, failed=0)
        _start_first_mcs_monitor(
            output_dir,
            target_mcs=output_mcs,
            notify_only=not _auto_open_mcs_enabled(),
        )
        _append_import_log(
            output_dir,
            "Data preparation finished; background Mimics is creating the .mcs file: {0}".format(output_mcs),
        )
    except Exception as e:
        output_mcs = monitor.get("output_mcs")
        output_dir = os.path.dirname(os.path.abspath(output_mcs)) if output_mcs else ""
        _append_import_log(output_dir, "Import queueing failed: {0}".format(e))
        _record_failed_case(output_dir, monitor.get("case_id"), "prepare_manifest", e)
        traceback.print_exc()
        mimics.dialogs.message_box(title="Import Error", message="Import queueing failed: {0}".format(e))
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        elif output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
        return

    # Single case: show queued status.
    if not monitor.get("batch_queue"):
        mimics.dialogs.message_box(
            title="Import Queued",
            message=(
                "Data preparation is complete.\n"
                "A background Mimics process is creating:\n{0}\n\n"
                "Mimics will notify you when it is ready."
            ).format(output_mcs),
        )
        return

    # Batch: update progress and start next case
    monitor["completed"] = monitor.get("completed", 0) + 1
    _start_next_batch_case(monitor)


def _start_next_batch_case(monitor):
    """Start bridge for the next case in the batch queue."""
    queue = monitor.get("batch_queue")
    if not queue:
        # All done
        completed = monitor.get("completed", 0)
        failed = monitor.get("failed", 0)
        total = monitor.get("total", 0)
        mimics.dialogs.message_box(
            title="Batch Import Complete",
            message="Queued {0}/{1} case(s); {2} failed.".format(completed, total, failed),
        )
        return

    case_info = queue.pop(0)
    case_id = case_info["case_id"]
    output_dir = monitor.get("output_dir")
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    output_mcs = os.path.join(output_dir, case_id + ".mcs")
    work_dir = os.path.join(output_dir, case_id + "_work")
    job_dir = os.path.join(jobs_dir, case_id)

    _append_import_log(monitor.get("output_dir", ""), "\n[{0}/{1}] Preparing: {2}".format(
        monitor.get("completed", 0) + monitor.get("failed", 0) + 1,
        monitor.get("total", 0),
        case_id))

    bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)

    # Update monitor for next case
    monitor["job_dir"] = job_dir
    monitor["output_mcs"] = output_mcs
    monitor["work_dir"] = work_dir
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 1800)
    monitor["done"] = False

    _write_json_atomic(
        os.path.join(job_dir, "job_state.json"),
        {
            "phase": "launching",
            "requested_phase": "preparing",
            "case_id": case_id,
            "started_at": time.time(),
        },
    )
    _launch_bridge_job_thread(bridge_params, job_dir, output_dir, "preparing", case_id=case_id)

    _IMPORT_MONITORS[job_dir] = monitor


# -- Batch prepare-only flow (no Mimics API, GUI stays responsive) ------

def _batch_prepare_tick(monitor):
    """Timer callback for batch prepare: bridge done -> manifest -> next case."""
    if monitor.get("done"):
        return

    # Prevent re-entrancy: while we process a completed case (which may
    # show a blocking message_box), the timer can fire again and re-enter
    # this function.  The busy flag prevents double-processing.
    if monitor.get("busy"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["busy"] = True
        output_dir = monitor.get("output_dir")
        case_id = monitor.get("case_id", "")
        _append_import_log(output_dir, "Conversion timed out: {0}".format(case_id))
        _terminate_job_process(job_dir)
        _record_failed_case(output_dir, case_id, "prepare_timeout", "Dataset conversion timed out.")
        monitor["failed"] = monitor.get("failed", 0) + 1
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        monitor["busy"] = False
        _start_next_batch_prepare(monitor)
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    # Case finished (done or error).
    # Set busy flag to prevent re-entrancy during message_box etc.
    monitor["busy"] = True

    if status == "error":
        output_dir = monitor.get("output_dir", "")
        case_id = monitor.get("case_id", "")
        _append_import_log(output_dir, "Conversion failed for {0}: {1}".format(case_id, result))
        _record_failed_case(output_dir, case_id, "prepare", result)
        monitor["failed"] = monitor.get("failed", 0) + 1
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        monitor["busy"] = False
        _start_next_batch_prepare(monitor)
        return

    # status == "done" - save manifest for later background .mcs creation.
    work_dir = monitor.get("work_dir")
    case_id = result.get("case_id", monitor.get("case_id", ""))
    output_dir = monitor.get("output_dir")
    completed_count = monitor.get("completed", 0) + 1
    total = monitor.get("total", 0)

    output_mcs = os.path.join(output_dir, case_id + ".mcs")
    queued = False
    try:
        _save_prepare_manifest(work_dir, result, output_mcs=output_mcs)
        monitor["completed"] = completed_count
        queued = True
        _append_import_log(
            output_dir,
            "[{0}/{1}] Prepared {2}; background Mimics will create the .mcs file.".format(
                completed_count,
                total,
                case_id,
            ),
        )
    except Exception as e:
        _append_import_log(output_dir, "Failed to save prepare manifest for {0}: {1}".format(case_id, e))
        _record_failed_case(output_dir, case_id, "prepare_manifest", e)
        monitor["failed"] = monitor.get("failed", 0) + 1
        _cleanup_work_dir(work_dir)
    _cleanup_job_dir(job_dir)
    if queued:
        _ensure_bg_mimics_running(output_dir, total)
        if not monitor.get("first_mcs_monitor_started"):
            monitor["first_mcs_monitor_started"] = True
            _start_first_mcs_monitor(
                output_dir,
                target_mcs=None,
                timeout_seconds=1800,
                poll_seconds=2.0,
                after_epoch=monitor.get("batch_started_epoch") or (time.time() - 1.0),
                skip_if_project_open=True,
                notify_only=not _auto_open_mcs_enabled(),
            )

    # Check if more cases to prepare; delegate to _start_next_batch_prepare
    # which handles both "start next" and "all done" cases.
    _start_next_batch_prepare(monitor)
    monitor["busy"] = False


def _start_next_batch_prepare(monitor):
    """Start bridge prepare for the next case in the batch queue.

    Queue items are (name, case_dir) tuples; lazy discovery: each case
    is scanned only when it's about to be converted.
    """
    queue = monitor.get("batch_queue")
    if not queue:
        # All cases converted; stop timer.
        # Background Mimics should already be running (started after 2nd
        # case).  Just ensure it's alive for any remaining manifests.
        monitor["done"] = True
        monitor["busy"] = False
        _stop_import_monitor(monitor.get("monitor_key"))
        completed = monitor.get("completed", 0)
        failed = monitor.get("failed", 0)
        total = monitor.get("total", 0)
        output_dir = monitor.get("output_dir")
        _append_import_log(
            output_dir,
            "All {0} case(s) prepared; {1} failed. Background Mimics is still creating .mcs files from prepared data.".format(
                completed,
                failed,
            ),
        )
        _mark_mcs_queue_done(output_dir, completed=completed, failed=failed)
        _ensure_bg_mimics_running(output_dir, total, mark_active=False)
        return

    # Pop next (name, case_dir) from queue
    item = queue.pop(0)
    case_info = None
    if isinstance(item, tuple):
        case_name, case_dir = item
    else:
        # If the bridge already returned full case metadata, use it directly
        # and avoid a second filesystem scan in the Mimics GUI process.
        if item.get("image"):
            case_info = item
        case_name = item.get("case_id", "")
        case_dir = item.get("case_dir", "")

    if case_info is None:
        case_info = _discover_single_case(case_dir)
    if case_info is None:
        output_dir = monitor.get("output_dir", "")
        _append_import_log(output_dir, "Skipping case without image data: {0}".format(case_name))
        _record_failed_case(output_dir, case_name, "discover_case", "No image data found.")
        monitor["failed"] = monitor.get("failed", 0) + 1
        _start_next_batch_prepare(monitor)
        return

    case_id = case_info["case_id"]
    output_dir = monitor.get("output_dir")
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    work_dir = os.path.join(output_dir, case_id + "_work")
    job_dir = os.path.join(jobs_dir, case_id)

    completed = monitor.get("completed", 0)
    failed = monitor.get("failed", 0)
    total = monitor.get("total", 0)
    _append_import_log(
        output_dir,
        "[{0}/{1}] Preparing: {2}".format(completed + failed + 1, total, case_id),
    )

    bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)

    # Update monitor for next case
    old_monitor_key = monitor.get("monitor_key")
    monitor["job_dir"] = job_dir
    monitor["work_dir"] = work_dir
    monitor["case_id"] = case_id
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 1800)
    monitor["done"] = False

    _write_json_atomic(
        os.path.join(job_dir, "job_state.json"),
        {
            "phase": "launching",
            "requested_phase": "preparing",
            "case_id": case_id,
            "started_at": time.time(),
        },
    )
    _launch_bridge_job_thread(bridge_params, job_dir, output_dir, "preparing", case_id=case_id)

    # Re-register monitor under new key (timer keeps running)
    _IMPORT_MONITORS.pop(old_monitor_key, None)
    _IMPORT_MONITORS[job_dir] = monitor


def _start_batch_prepare_monitor(job_dir, work_dir, timeout_seconds=1800,
                                  poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer for batch prepare-only flow."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "work_dir": work_dir,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "batch_queue": batch_queue,
        "completed": 0,
        "failed": 0,
    }
    if batch_info:
        monitor.update(batch_info)

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Conversion Running",
            message="Dataset conversion has started, but progress cannot be monitored automatically.",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Conversion Running",
            message="Dataset conversion has started, but progress cannot be monitored automatically.",
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        _batch_prepare_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


def _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for batch prepare polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
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
        _batch_prepare_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _find_mimics_exe():
    """Find MimicsResearch.exe installation path."""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Materialise", "Mimics Research 21.0", "MimicsResearch.exe"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Mimics Research 21.0", "MimicsResearch.exe"),
        "D:\\Mimics Research 21.0\\MimicsResearch.exe",
        "C:\\Mimics Research 21.0\\MimicsResearch.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try to find via registry or common locations
    for drive in ["C:", "D:", "E:"]:
        for name in ["Mimics Research 21.0", "MimicsResearch 21.0"]:
            path = os.path.join(drive + "\\", name, "MimicsResearch.exe")
            if os.path.isfile(path):
                return path
    return None


def _ensure_bg_mimics_running(output_dir, total_count=0, mark_active=True):
    """Ensure a background Mimics is running to create .mcs files.

    Called after each case's manifest is saved.  If no background Mimics
    is alive, launch one.  This way .mcs files are created continuously
    as manifests become available, rather than waiting for all conversions
    to finish.
    """
    global _BG_MIMICS_PID
    if mark_active:
        _mark_mcs_queue_active(output_dir, total_count)
    if _background_stop_requested(output_dir):
        _append_import_log(output_dir, "Background .mcs creation is stopped by user request.")
        return
    # Check if existing background Mimics is still alive
    if _BG_MIMICS_PID and _is_pid_alive(_BG_MIMICS_PID):
        return  # still running, it will pick up new manifests

    # Launch a new one
    _launch_background_mimics(output_dir, total_count=total_count)


def _schedule_bg_mimics_retry(output_dir, total_count=0):
    global _BG_MIMICS_RETRY_ACTIVE
    if _BG_MIMICS_RETRY_ACTIVE:
        return
    _BG_MIMICS_RETRY_ACTIVE = True

    def _retry():
        global _BG_MIMICS_RETRY_ACTIVE
        retry_started = time.time()
        deadline = time.time() + 21600.0
        try:
            while time.time() < deadline:
                time.sleep(30.0)
                if _background_stop_requested(output_dir, retry_started):
                    _append_import_log(output_dir, "Background Mimics retry stopped by user request.")
                    return
                if _BG_MIMICS_PID and _is_pid_alive(_BG_MIMICS_PID):
                    return
                process = _launch_background_mimics(
                    output_dir,
                    total_count=total_count,
                    schedule_retry=False,
                )
                if process is not None:
                    return
        finally:
            _BG_MIMICS_RETRY_ACTIVE = False

    thread = threading.Thread(target=_retry)
    thread.daemon = True
    thread.start()


def _launch_background_mimics(output_dir, total_count=0, schedule_retry=True):
    """Launch Mimics in background mode to create .mcs files from prepared data.

    Mimics runs without GUI (-b flag), executing create_mcs_batch.py which
    reads prepare manifests and creates .mcs files one by one.
    Returns the Popen object, or None on failure.
    """
    global _BG_MIMICS_PID
    global _BG_MIMICS_BUSY_NOTICE_AT
    if _background_stop_requested(output_dir):
        _append_import_log(output_dir, "Background .mcs creation was not started because stop was requested.")
        return None
    mimics_exe = _find_mimics_exe()
    if not mimics_exe:
        _append_import_log(output_dir, "MimicsResearch.exe was not found; background .mcs creation cannot start.")
        return None

    # Find the create_mcs_batch.py script
    here = os.path.dirname(os.path.abspath(__file__))
    script_candidates = [
        os.path.join(here, "create_mcs_batch.py"),
        os.path.join(here, "..", "runtime_py35", "create_mcs_batch.py"),
        os.path.join(_project_root(), "adapters", "mimics", "runtime_py35", "create_mcs_batch.py"),
    ]
    script_path = None
    for c in script_candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            script_path = c
            break
    if not script_path:
        script_path = os.path.abspath(script_candidates[0])

    # Write a runner script that Mimics can execute
    runner_path = os.path.join(output_dir, "_run_create_mcs.py")
    script_dir = os.path.dirname(script_path)
    with open(runner_path, "w") as f:
        f.write("# Auto-generated runner for background Mimics .mcs creation\n")
        f.write("import sys, os\n")
        f.write("sys.path.insert(0, r'{0}')\n".format(script_dir))
        f.write("import create_mcs_batch\n")
        f.write("create_mcs_batch.main(r'{0}')\n".format(output_dir))

    # Launch Mimics in background mode
    cmd = [mimics_exe, "-b", "-run_script", runner_path]
    log_path = os.path.join(output_dir, "_background_mimics.log")
    _rotate_log_file(log_path)
    lock_path = _resource_lock_path("background_mimics.lock")
    lock_token = runtime_common.acquire_resource_lock(
        lock_path,
        "background_mimics",
        "import .mcs creation",
        wait_seconds=0.0,
    )
    if not lock_token:
        now = time.time()
        if now - _BG_MIMICS_BUSY_NOTICE_AT >= 60.0:
            _BG_MIMICS_BUSY_NOTICE_AT = now
            _append_import_log(
                output_dir,
                "Background Mimics is already running for another Mimics-Script task; .mcs creation will continue when that process exits.",
            )
        if schedule_retry:
            _schedule_bg_mimics_retry(output_dir, total_count=total_count)
        return None
    log_handle = None
    try:
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_background_process_kwargs()
        )
        _BG_MIMICS_PID = process.pid
        runtime_common.update_resource_lock_pid(
            lock_path,
            lock_token,
            process.pid,
            {"kind": "create_mcs", "output_dir": os.path.abspath(output_dir)},
        )
        _append_import_log(
            output_dir,
            "Background Mimics started (PID={0}) for .mcs creation.".format(process.pid),
        )
        return process
    except Exception as e:
        runtime_common.release_resource_lock(lock_path, lock_token)
        _append_import_log(output_dir, "Could not start background Mimics: {0}".format(e))
        return None
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


# -- Auto-open first .mcs monitor --------------------------------------

def _first_mcs_monitor_tick(monitor):
    """Timer callback: check if first .mcs file exists and is stable."""
    if monitor.get("done"):
        return

    output_dir = monitor.get("output_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        return

    target_mcs = monitor.get("target_mcs")
    after_epoch = monitor.get("after_epoch")
    candidates = []
    if target_mcs:
        candidates = [target_mcs]
    else:
        try:
            items = sorted(os.listdir(output_dir))
        except Exception:
            return
        candidates = [os.path.join(output_dir, item) for item in items if item.endswith(".mcs")]

    for mcs_path in candidates:
        if not os.path.isfile(mcs_path):
            continue
        if after_epoch:
            try:
                if os.path.getmtime(mcs_path) < after_epoch:
                    continue
            except Exception:
                continue
        try:
            size = os.path.getsize(mcs_path)
        except Exception:
            continue
        stable_key = os.path.abspath(mcs_path)
        previous = monitor.get("last_file")
        previous_size = monitor.get("last_size")
        if previous != stable_key or previous_size != size:
            monitor["last_file"] = stable_key
            monitor["last_size"] = size
            return
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        try:
            if monitor.get("notify_only"):
                _append_import_log(output_dir, ".mcs is ready: {0}".format(mcs_path))
                try:
                    mimics.dialogs.message_box(
                        title="MCS Ready",
                        message="A converted .mcs file is ready:\n{0}".format(mcs_path),
                        ui_blocking=False,
                    )
                except TypeError:
                    mimics.dialogs.message_box(
                        title="MCS Ready",
                        message="A converted .mcs file is ready:\n{0}".format(mcs_path),
                    )
                return
            if monitor.get("skip_if_project_open"):
                try:
                    if len(mimics.data.images) > 0:
                        _append_import_log(output_dir, "Skipped automatic .mcs open because a project is already open: {0}".format(mcs_path))
                        return
                except Exception:
                    pass
            mimics.file.open_project(filename=mcs_path)
            _append_import_log(output_dir, "Automatically opened .mcs: {0}".format(mcs_path))
        except Exception as exc:
            _append_import_log(output_dir, "Could not automatically open .mcs {0}: {1}".format(mcs_path, exc))
        return


def _start_first_mcs_monitor(output_dir, target_mcs=None, timeout_seconds=900, poll_seconds=2.0,
                             after_epoch=None, skip_if_project_open=False, notify_only=False):
    """Start a timer that polls for the first stable .mcs file."""
    monitor = {
        "monitor_key": "first_mcs_" + (target_mcs or output_dir),
        "output_dir": output_dir,
        "target_mcs": target_mcs,
        "after_epoch": after_epoch,
        "skip_if_project_open": skip_if_project_open,
        "notify_only": notify_only,
        "done": False,
        "deadline": time.time() + timeout_seconds,
    }

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        _stop_import_monitor(monitor["monitor_key"])
        monitor["timer"] = timer
        _IMPORT_MONITORS[monitor["monitor_key"]] = monitor

        def _tick():
            _first_mcs_monitor_tick(monitor)

        timer.timeout.connect(_tick)
        timer.start(int(poll_seconds * 1000))
        return True
    except Exception:
        pass

    # Fall back to Win32 SetTimer
    if os.name != "nt":
        return False
    try:
        import ctypes
        monitor_key = monitor["monitor_key"]
        _stop_import_monitor(monitor_key)
        user32 = ctypes.windll.user32
        timer_interval_ms = int(poll_seconds * 1000)
        TIMERPROC = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_uint,
        )

        def _timer_proc(hwnd, message, timer_id, tick_count):
            _first_mcs_monitor_tick(monitor)

        callback = TIMERPROC(_timer_proc)
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
        if not timer_id:
            return False
        monitor["callback"] = callback
        monitor["win32_timer"] = (user32, timer_id)
        _IMPORT_MONITORS[monitor_key] = monitor
        return True
    except Exception:
        return False


def _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for import result polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
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
        _import_monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _start_import_monitor(job_dir, output_mcs, work_dir, timeout_seconds=1800,
                          poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer that polls for bridge completion and auto-applies."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "output_mcs": output_mcs,
        "work_dir": work_dir,
        "case_id": os.path.splitext(os.path.basename(output_mcs or ""))[0],
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
        if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Import Running",
            message=(
                "Dataset conversion has started, but the result cannot be "
                "queued automatically. Run import again later to check progress."
            ),
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Import Running",
            message=(
                "Dataset conversion has started, but the result cannot be "
                "queued automatically. Run import again later to check progress."
            ),
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        _import_monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


# -- Disk space helpers -------------------------------------------------


def _check_disk_space(path, required_mb):
    """Check if the drive containing *path* has at least *required_mb* MB free.

    Returns (ok, free_mb). On platforms where statvfs is unavailable,
    returns (True, -1) so the caller can proceed.
    """
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


# -- Discover monitor (batch mode: discover -> import chain) ------------

def _discover_monitor_tick(monitor):
    """Timer callback for discover phase: check if bridge discover finished."""
    if monitor.get("done"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        mimics.dialogs.message_box(
            title="Scan Timeout",
            message="Dataset scan timed out. Please retry.",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still discovering

    monitor["done"] = True
    _stop_import_monitor(monitor_key)

    if status == "error":
        mimics.dialogs.message_box(title="Scan Error", message="Dataset scan failed: {0}".format(result))
        _cleanup_job_dir(job_dir)
        return

    # Discover done; result contains cases list.
    cases = result.get("cases", [])
    count = result.get("count", len(cases))
    _cleanup_job_dir(job_dir)

    if not cases:
        mimics.dialogs.message_box(
            title="Import",
            message="No case data was found in the selected folder.",
        )
        return

    output_dir = monitor.get("output_dir")
    _append_import_log(output_dir, "Discovered {0} case(s); starting preparation.".format(count))

    # Disk space check (quick, non-blocking)
    estimated_mb = count * 500
    ok, free_mb = _check_disk_space(output_dir, estimated_mb)
    if not ok:
        mimics.dialogs.message_box(
            title="Insufficient Disk Space",
            message=(
                "Insufficient disk space. Estimated requirement: {0} MB; "
                "available: {1} MB. Please free disk space and retry."
            ).format(estimated_mb, int(free_mb)),
        )
        return

    _mark_mcs_queue_active(output_dir, count)

    # Auto-start prepare; user already chose the folder.

    # Start batch prepare-only flow (no Mimics API calls, GUI stays responsive)
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    first_case = cases[0]
    first_case_id = first_case["case_id"]
    first_work_dir = os.path.join(output_dir, first_case_id + "_work")
    first_job_dir = os.path.join(jobs_dir, first_case_id)

    _append_import_log(output_dir, "[1/{0}] Preparing: {1}".format(count, first_case_id))
    bridge_params = _build_bridge_params(first_case, axes, flips, first_work_dir)
    _write_json_atomic(
        os.path.join(first_job_dir, "job_state.json"),
        {
            "phase": "launching",
            "requested_phase": "preparing",
            "case_id": first_case_id,
            "started_at": time.time(),
        },
    )
    _launch_bridge_job_thread(bridge_params, first_job_dir, output_dir, "preparing", case_id=first_case_id)

    # Bridge launched, timer will auto-chain remaining cases

    batch_info = {
        "total": count,
        "output_dir": output_dir,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
        "batch_started_epoch": time.time(),
    }
    _start_batch_prepare_monitor(
        first_job_dir, first_work_dir,
        batch_queue=cases[1:],
        batch_info=batch_info,
    )


def _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for discover result polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
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
        _discover_monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _start_import_discover_monitor(job_dir, ts_root, output_dir, axes, flips, jobs_dir,
                                    poll_seconds=0.5, timeout_seconds=900):
    """Start a non-blocking timer that polls for discover completion, then auto-starts import."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "ts_root": ts_root,
        "output_dir": output_dir,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
    }

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Scan Running",
            message="Dataset scan has started, but progress cannot be monitored automatically.",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="Scan Running",
            message="Dataset scan has started, but progress cannot be monitored automatically.",
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        _discover_monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


# -- Single case import -------------------------------------------------

def _build_bridge_params(case_info, axes, flips, work_dir):
    """Build bridge params for the prepare action."""
    buffers_dir = os.path.join(work_dir, "buffers")
    if not os.path.isdir(buffers_dir):
        os.makedirs(buffers_dir)
    dicom_out = os.path.join(work_dir, "derived_dicom")

    return {
        "action": "prepare",
        "image_path": case_info["image"],
        "masks": case_info.get("masks", []),
        "dicom_out": dicom_out,
        "buffers_out": buffers_dir,
        "axes": axes,
        "flips": flips,
        "case_id": case_info["case_id"],
        "case_dir": case_info.get("case_dir", ""),
    }


# -- Main entry ---------------------------------------------------------

def main():
    """Entry point. Reads config from argv or interactive dialog.

    Usage:
        mimics_import.py --ts-root <dir> [--cases s0000,s0001] [--output-dir <dir>] [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --case-dir <dir> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
    """
    # Safe cleanup runs in a daemon thread. By default it only removes stale
    # resource locks; process killing is explicit or aggressive opt-in.
    if runtime_common.auto_cleanup_enabled():
        cleanup_thread = threading.Thread(target=_cleanup_stale_processes)
        cleanup_thread.daemon = True
        cleanup_thread.start()

    # Parse args (simple, Python 3.5 compatible)
    ts_root = None
    case_dir = None
    output = None
    output_dir = None
    cases_filter = None
    axes = [0, 1, 2]
    flips = [False, False, False]

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
        elif arg == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif arg == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
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
        else:
            i += 1

    # Interactive: if no args, ask for TS root or case dir
    if not ts_root and not case_dir:
        ts_root = _pick_directory("Select dataset folder")
        if not ts_root or not os.path.isdir(ts_root):
            mimics.dialogs.message_box(
                "No valid folder was selected.",
                title="Import Dataset",
                ui_blocking=True,
            )
            return 1

    # -- Single case mode ----------------------------------------------

    if case_dir:
        if not output:
            output = os.path.join(case_dir, os.path.basename(case_dir) + ".mcs")
        case_info = {
            "case_id": os.path.basename(case_dir),
            "image": None,
            "image_type": None,
            "masks": [],
            "case_dir": case_dir,
        }
        # Find image
        for img_name in ("ct.nii.gz", "mri.nii.gz"):
            candidate = os.path.join(case_dir, img_name)
            if os.path.isfile(candidate):
                case_info["image"] = candidate
                case_info["image_type"] = "nifti"
                break
        if case_info["image"] is None:
            dicom_dir = os.path.join(case_dir, "dicom")
            if os.path.isdir(dicom_dir):
                case_info["image"] = dicom_dir
                case_info["image_type"] = "dicom"
        if case_info["image"] is None:
            for fname in sorted(os.listdir(case_dir)):
                if fname.endswith(".nii.gz"):
                    case_info["image"] = os.path.join(case_dir, fname)
                    case_info["image_type"] = "nifti"
                    break
        if case_info["image"] is None:
            mimics.dialogs.message_box(title="Error", message="No image data found: {0}".format(case_dir))
            return 1
        # Find masks
        seg_dir = os.path.join(case_dir, "segmentations")
        if os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                if fname.endswith(".nii.gz"):
                    organ = fname.replace(".nii.gz", "")
                    case_info["masks"].append({"name": organ, "path": os.path.join(seg_dir, fname)})

        work_dir = os.path.dirname(os.path.abspath(output))
        jobs_dir = os.path.join(work_dir, "_import_jobs")
        job_dir = os.path.join(jobs_dir, case_info["case_id"])

        # Launch bridge in background + start timer to queue .mcs creation.
        _mark_mcs_queue_active(os.path.dirname(os.path.abspath(output)), 1)
        _append_import_log(os.path.dirname(os.path.abspath(output)), "Preparing: {0}".format(case_info["case_id"]))
        bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
        _write_json_atomic(
            os.path.join(job_dir, "job_state.json"),
            {
                "phase": "launching",
                "requested_phase": "preparing",
                "case_id": case_info["case_id"],
                "started_at": time.time(),
            },
        )
        _launch_bridge_job_thread(bridge_params, job_dir, os.path.dirname(os.path.abspath(output)), "preparing", case_id=case_info["case_id"])

        # Bridge launched, timer will queue the result for background Mimics.
        _start_import_monitor(job_dir, output, work_dir)
        return 0

    # -- Batch mode: discover and prepare cases without blocking Mimics GUI.

    # No confirmation dialog; user already chose the folder, just start.
    if not output_dir:
        output_dir = os.path.join(ts_root, "mcs_output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    jobs_dir = os.path.join(output_dir, "_import_jobs")

    _append_import_log(output_dir, "Starting dataset discovery in the bridge process.")
    discover_job_dir = os.path.join(jobs_dir, "_discover")
    bridge_params = {
        "action": "discover",
        "ts_root": ts_root,
        "cases_filter": list(cases_filter) if cases_filter else None,
    }
    _launch_bridge_job_thread(bridge_params, discover_job_dir, output_dir, "discovering")
    _start_import_discover_monitor(
        discover_job_dir,
        ts_root,
        output_dir,
        axes,
        flips,
        jobs_dir,
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
            mimics.dialogs.message_box(title="Fatal Error", message="Error: {0}".format(error))
        except Exception:
            pass
        raise
