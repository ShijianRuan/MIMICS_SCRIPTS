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
_CONFIG_CACHE = None


def _auto_open_mcs_enabled():
    return os.environ.get("MIMICS_IMPORT_AUTO_OPEN_MCS", "").strip().lower() in ("1", "true", "yes")


_write_json_atomic = runtime_common.write_json_atomic
_safe_case_filename = runtime_common.safe_filename
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs


def _write_json_quick(path, value):
    """Best-effort lightweight JSON write.

    Uses direct write (no temp file + replace + fsync loop) to avoid possible
    native instability in host callbacks while still recording state.
    """
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _mimics_log(level, message):
    # Some Mimics builds are unstable when log_user_message is called during
    # import workflows. Keep it disabled by default and allow explicit opt-in.
    enabled = os.environ.get("MIMICS_IMPORT_USE_MIMICS_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if not enabled:
        return False
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return True
    except Exception:
        return False


def _safe_message_box(title, message, ui_blocking=True):
    """Show a Mimics dialog, catching any API failures gracefully."""
    try:
        mimics.dialogs.message_box(title=title, message=message, ui_blocking=ui_blocking)
    except TypeError:
        try:
            mimics.dialogs.message_box(title=title, message=message)
        except Exception:
            pass
    except Exception:
        pass


def _user_progress(level, message):
    """Publish concise lifecycle transitions to the Mimics log panel."""
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return
    except Exception:
        print(message)


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


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
    try:
        thread_name = threading.current_thread().name
    except Exception:
        thread_name = "unknown"
    text = "[{0}] [pid={1}] [thread={2}] {3}".format(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.getpid(),
        thread_name,
        message,
    )
    # Mimics API calls are not thread-safe. Only attempt Mimics logging on the
    # main thread and only when explicitly enabled by environment variable.
    logged_to_mimics = False
    try:
        if threading.current_thread() is threading.main_thread():
            logged_to_mimics = _mimics_log(logging.INFO, text)
    except Exception:
        logged_to_mimics = False
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


def _verbose_log_enabled():
    return os.environ.get("MIMICS_IMPORT_VERBOSE_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _verbose_log(root_dir, message):
    """Log detailed debug info only when verbose logging is explicitly enabled."""
    if _verbose_log_enabled():
        _append_import_log(root_dir, message)


def _append_import_exception(root_dir, context, exc=None):
    try:
        if exc is None:
            detail = traceback.format_exc()
        else:
            detail = "{0}\n{1}".format(repr(exc), traceback.format_exc())
        _append_import_log(root_dir, "{0} | exception={1}".format(context, detail))
    except Exception:
        pass


def _summarize_bridge_params(params):
    try:
        action = params.get("action", "")
        summary = {
            "action": action,
            "case_id": params.get("case_id", ""),
            "ts_root": params.get("ts_root", ""),
            "case_dir": params.get("case_dir", ""),
            "image_path": params.get("image_path", ""),
            "mask_count": len(params.get("masks", []) or []),
            "axes": params.get("axes", []),
            "flips": params.get("flips", []),
        }
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "<bridge-param-summary-unavailable>"


def _queue_active_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_ACTIVE)


def _queue_done_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_DONE)


def _queue_stop_path(output_dir):
    return os.path.join(output_dir, _MCS_QUEUE_STOP)


def _mcs_queue_registry_dir():
    return os.path.join(_project_root(), ".mimics_runtime", "mcs_queues")


def _register_mcs_queue(output_dir, total_count=0):
    # Queue registry is optional metadata only. It is not required for import.
    # Keep it disabled by default because some environments crash in this path.
    enabled = os.environ.get("MIMICS_IMPORT_ENABLE_QUEUE_REGISTRY", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if not enabled:
        return
    try:
        registry_dir = _mcs_queue_registry_dir()
        if not os.path.isdir(registry_dir):
            os.makedirs(registry_dir)
        digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8", "replace")).hexdigest()[:16]
        base = _safe_case_filename(os.path.basename(os.path.abspath(output_dir))).strip("._") or "queue"
        name = "{0}_{1}".format(base, digest)
        path = os.path.join(registry_dir, name + ".json")
        _write_json_quick(
            path,
            {
                "output_dir": os.path.abspath(output_dir),
                "total_count": int(total_count or 0),
                "updated_at_epoch": time.time(),
            },
        )
    except Exception:
        # Registry failures must never affect import flow.
        pass


def _mark_mcs_queue_active(output_dir, total_count=0):
    _verbose_log(output_dir, "Queue active | output_dir={0} | total_count={1}".format(output_dir, int(total_count or 0)))
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
    _write_json_quick(
        _queue_active_path(output_dir),
        {
            "status": "active",
            "total_count": int(total_count or 0),
            "updated_at_epoch": time.time(),
        },
    )
    _verbose_log(output_dir, "Queue active written | path={0}".format(_queue_active_path(output_dir)))
    _register_mcs_queue(output_dir, total_count)
    _verbose_log(output_dir, "Queue active complete")


def _mark_mcs_queue_done(output_dir, completed=0, failed=0):
    _verbose_log(output_dir, "Queue done | output_dir={0} | completed={1} | failed={2}".format(output_dir, int(completed or 0), int(failed or 0)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    _write_json_quick(
        _queue_done_path(output_dir),
        {
            "status": "done",
            "completed": int(completed or 0),
            "failed": int(failed or 0),
            "updated_at_epoch": time.time(),
        },
    )
    _verbose_log(output_dir, "Queue done written | path={0}".format(_queue_done_path(output_dir)))
    try:
        active = _queue_active_path(output_dir)
        if os.path.isfile(active):
            os.remove(active)
    except Exception:
        pass
    _register_mcs_queue(output_dir, completed + failed)
    _verbose_log(output_dir, "Queue done complete")


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


def _resolve_import_output_dir(base_dir, create=True):
    default_dir = os.path.join(base_dir, "mcs_output")
    config = _load_data_io_config()
    configured = config.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if not os.path.isabs(configured):
        configured = os.path.abspath(os.path.join(base_dir, configured))
    else:
        configured = os.path.abspath(configured)
    if not create:
        return configured
    try:
        if not os.path.isdir(configured):
            os.makedirs(configured)
        return configured
    except Exception:
        return default_dir


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
        if os.path.isfile(os.path.join(candidate, "python.exe")):
            return candidate
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
        os.path.join(env_root, "python.exe"),
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

_MEDICAL_IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
_MASK_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nrrd.gz", ".seg.nii", ".seg.nii.gz")


def _is_medical_image_file(path):
    name = os.path.basename(path).lower()
    return os.path.isfile(path) and any(name.endswith(suffix) for suffix in _MEDICAL_IMAGE_SUFFIXES)


def _image_stem(path):
    name = os.path.basename(path)
    lower = name.lower()
    for suffix in _MEDICAL_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[:-len(suffix)] or "case"
    return os.path.splitext(name)[0] or "case"


def _find_case_image(case_dir, allow_direct_dicom=False):
    for img_name in ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii", "ct.mhd", "mri.mhd", "ct.mha", "mri.mha"):
        candidate = os.path.join(case_dir, img_name)
        if _is_medical_image_file(candidate):
            return candidate, "medical_image"
    for fname in sorted(os.listdir(case_dir)):
        candidate = os.path.join(case_dir, fname)
        if _is_medical_image_file(candidate):
            return candidate, "medical_image"
    dicom_dir = os.path.join(case_dir, "dicom")
    if os.path.isdir(dicom_dir):
        return dicom_dir, "dicom"
    # A directly selected folder may itself be a flat DICOM series. Validation
    # happens in the external bridge so Mimics never parses all DICOM headers.
    if allow_direct_dicom and os.path.isdir(case_dir) and os.listdir(case_dir):
        return case_dir, "dicom_candidate"
    return None, None

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

        image_path, image_type = _find_case_image(case_dir)

        if image_path is None:
            continue

        masks = []
        seg_dir = os.path.join(case_dir, "segmentations")
        if os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                lower = fname.lower()
                if any(lower.endswith(s) for s in _MASK_SUFFIXES):
                    organ = fname
                    for s in _MASK_SUFFIXES:
                        if organ.lower().endswith(s):
                            organ = organ[:-len(s)]
                            break
                    if not organ:
                        organ = "mask"
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
    if _is_medical_image_file(case_dir):
        image_file = os.path.abspath(case_dir)
        case_dir = os.path.dirname(image_file)
        name = _image_stem(image_file)
    else:
        image_file = None
        name = os.path.basename(case_dir)
    if not os.path.isdir(case_dir):
        return None
    if name in ("mcs_output", "segmentations"):
        return None

    if image_file:
        image_path, image_type = image_file, "medical_image"
    else:
        image_path, image_type = _find_case_image(case_dir, allow_direct_dicom=True)

    if image_path is None:
        return None

    masks = []
    seg_dir = os.path.join(case_dir, "segmentations")
    allow_masks = True
    if image_file:
        sibling_images = [name for name in os.listdir(case_dir) if _is_medical_image_file(os.path.join(case_dir, name))]
        allow_masks = len(sibling_images) <= 1
    if allow_masks and os.path.isdir(seg_dir):
        for fname in sorted(os.listdir(seg_dir)):
            lower = fname.lower()
            if any(lower.endswith(s) for s in _MASK_SUFFIXES):
                organ = fname
                # Strip all known suffixes to get the organ name
                for s in _MASK_SUFFIXES:
                    if organ.lower().endswith(s):
                        organ = organ[:-len(s)]
                        break
                if not organ:
                    organ = "mask"
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

    _verbose_log(
        os.path.dirname(job_dir),
        "Bridge launch | job_dir={0} | params={1}".format(job_dir, _summarize_bridge_params(bridge_params)),
    )
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
        _verbose_log(
            os.path.dirname(job_dir),
            "Bridge subprocess pid={0}".format(process.pid),
        )
    finally:
        stdin_handle.close()
        stdout_handle.close()
        stderr_handle.close()
    return process


def _launch_bridge_job_thread(bridge_params, job_dir, output_dir, phase, case_id=None):
    """Start bridge from a worker thread so Mimics GUI can repaint first."""
    def _run():
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
        _verbose_log(
            output_dir,
            "Job state | phase={0} | case_id={1} | params={2}".format(
                phase, case_id or "", _summarize_bridge_params(bridge_params),
            ),
        )
        _verbose_log(
            output_dir,
            "Worker thread | phase={0} | case_id={1}".format(phase, case_id or ""),
        )
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
            _append_import_log(output_dir, "Bridge PID={0} | phase={1}".format(process.pid, phase))
        except Exception as exc:
            _append_import_log(output_dir, "Could not start bridge process for {0}: {1}".format(phase, exc))
            _append_import_exception(output_dir, "Bridge worker thread failed", exc)
            try:
                _write_json_atomic(
                    os.path.join(job_dir, "bridge_result.json"),
                    {"status": "error", "error": str(exc)},
                )
            except Exception:
                pass

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
                _verbose_log(
                    os.path.dirname(job_dir),
                    "Job done | keys={0}".format(sorted(result.keys())),
                )
                return ("done", result)
            else:
                _verbose_log(
                    os.path.dirname(job_dir),
                    "Job error | {0}".format(result.get("error", "non-ok status")),
                )
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

    # Single case: completion is reported by the .mcs status monitor. Avoid an
    # intermediate dialog that makes the user acknowledge the same operation
    # twice; the Mimics log remains available for progress inspection.
    if not monitor.get("batch_queue"):
        _user_progress(
            logging.INFO,
            "Import preparation finished. Background Mimics is creating: {0}".format(output_mcs),
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
        try:
            _batch_prepare_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_batch_prepare_monitor qtimer callback failed", exc)

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
        try:
            _batch_prepare_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_win32_batch_prepare_monitor timer callback failed", exc)

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
        f.write("sys.path.insert(0, {0})\n".format(json.dumps(script_dir)))
        f.write("os.environ['MIMICS_BRIDGE_PYTHON'] = {0}\n".format(json.dumps(_python_exe())))
        f.write("os.environ['MIMICS_BRIDGE_SCRIPT'] = {0}\n".format(json.dumps(_bridge_script())))
        f.write("import create_mcs_batch\n")
        f.write("create_mcs_batch.main({0})\n".format(json.dumps(output_dir)))

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
            holder = runtime_common.active_resource_lock(_project_root(), "background_mimics.lock")
            _append_import_log(
                output_dir,
                "Background Mimics is already running for another Mimics-Script task; .mcs creation will continue when that process exits.",
            )
            _safe_message_box(
                "Import Waiting",
                (
                    "Prepared data is waiting for the background Mimics license.\n\n"
                    "Current task: {0}\n\n"
                    "This import will continue automatically. Use 04 Stop Import Queue "
                    "to cancel this import, or wait for the current task to finish."
                ).format(runtime_common.resource_lock_summary(holder)),
                ui_blocking=False,
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

    if monitor.get("first_notified"):
        status = runtime_common.read_json(os.path.join(output_dir, "_mcs_batch_status.json"), {}) or {}
        signature = (status.get("status"), status.get("case_id"), status.get("completed"), status.get("failed"))
        if signature != monitor.get("last_batch_signature"):
            monitor["last_batch_signature"] = signature
            state = str(status.get("status") or "waiting")
            _user_progress(
                logging.INFO,
                "Import .mcs creation: {0}; completed {1}, failed {2}{3}.".format(
                    state,
                    int(status.get("completed", 0) or 0),
                    int(status.get("failed", 0) or 0),
                    "; current " + str(status.get("case_id")) if status.get("case_id") else "",
                ),
            )
        if status.get("status") == "closed":
            monitor["done"] = True
            _stop_import_monitor(monitor_key)
            completed = int(status.get("completed", 0) or 0)
            failed = int(status.get("failed", 0) or 0)
            _safe_message_box(
                "Import Completed with Errors" if failed else "Import Complete",
                "Background .mcs creation finished.\n\nCreated: {0}\nFailed: {1}\nOutput: {2}{3}".format(
                    completed,
                    failed,
                    output_dir,
                    "\n\nReview mimics_import.log and _failed_cases.json before retrying failed cases." if failed else "",
                ),
                ui_blocking=False,
            )
        elif status:
            try:
                stale_seconds = time.time() - float(status.get("updated_at_epoch", 0.0) or 0.0)
            except Exception:
                stale_seconds = 0.0
            holder = runtime_common.active_resource_lock(_project_root(), "background_mimics.lock")
            if stale_seconds > 90.0 and not holder:
                monitor["done"] = True
                _stop_import_monitor(monitor_key)
                _safe_message_box(
                    "Import Stopped Unexpectedly",
                    (
                        "Background .mcs creation stopped before reporting completion.\n\n"
                        "Prepared files were kept for retry. Run Import again or inspect:\n{0}"
                    ).format(os.path.join(output_dir, "logs", "_create_mcs_batch.log")),
                    ui_blocking=False,
                )
        return

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
        keep_for_batch = not bool(target_mcs)
        if not keep_for_batch:
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
                if keep_for_batch:
                    monitor["first_notified"] = True
                    monitor["deadline"] = time.time() + 7 * 86400
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
            if keep_for_batch:
                monitor["first_notified"] = True
                monitor["deadline"] = time.time() + 7 * 86400
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
        "deadline": time.time() + (timeout_seconds if target_mcs else max(timeout_seconds, 7 * 86400)),
    }

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        _stop_import_monitor(monitor["monitor_key"])
        monitor["timer"] = timer
        _IMPORT_MONITORS[monitor["monitor_key"]] = monitor

        def _tick():
            try:
                _first_mcs_monitor_tick(monitor)
            except Exception as exc:
                _append_import_exception(monitor.get("output_dir", ""), "_start_first_mcs_monitor qtimer callback failed", exc)

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
            try:
                _first_mcs_monitor_tick(monitor)
            except Exception as exc:
                _append_import_exception(monitor.get("output_dir", ""), "_start_first_mcs_monitor win32 timer callback failed", exc)

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
        try:
            _import_monitor_tick(monitor)
        except Exception as exc:
            out = ""
            try:
                out = os.path.dirname(os.path.abspath(monitor.get("output_mcs") or ""))
            except Exception:
                out = ""
            _append_import_exception(out, "_start_win32_import_monitor timer callback failed", exc)

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

    # Prefer the native Windows message-loop timer. Importing PyQt5 after a
    # folder dialog closes can stall Mimics while Qt scans plugins and paths.
    if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
        return True

    # Non-Windows fallback: use the existing Mimics Qt event loop.
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
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
        try:
            _import_monitor_tick(monitor)
        except Exception as exc:
            out = ""
            try:
                out = os.path.dirname(os.path.abspath(monitor.get("output_mcs") or ""))
            except Exception:
                out = ""
            _append_import_exception(out, "_start_import_monitor qtimer callback failed", exc)

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


def _pick_image_file(title):
    """Pick one supported volume without reading it in the Mimics process."""
    file_filter = "Medical volumes (*.nii *.nii.gz *.mha *.mhd *.nrrd)"
    try:
        from PyQt5.QtWidgets import QFileDialog
        path, _selected = QFileDialog.getOpenFileName(None, title, "", file_filter)
        return str(path) if path else None
    except Exception:
        pass
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
    path = tkFileDialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("Medical volumes", "*.nii *.nii.gz *.mha *.mhd *.nrrd"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


def _mask_name_from_path(path):
    """Derive a mask name from a file path, stripping common suffixes."""
    name = os.path.basename(path)
    lower = name.lower()
    # Strip compression suffix first
    if lower.endswith(".gz"):
        name = name[:-3]
        lower = name.lower()
    # Strip format suffix
    for suffix in (".nii", ".mha", ".mhd", ".nrrd", ".seg"):
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name or "mask"


# -- Discover monitor (batch mode: discover -> import chain) ------------

def _discover_monitor_tick(monitor):
    """Timer callback for discover phase: check if bridge discover finished."""
    try:
        if monitor.get("done"):
            return

        job_dir = monitor.get("job_dir")
        monitor_key = monitor.get("monitor_key")

        # Timeout check
        if time.time() > monitor.get("deadline", 0):
            monitor["done"] = True
            _stop_import_monitor(monitor_key)
            _safe_message_box("Scan Timeout", "Dataset scan timed out. Please retry.")
            return

        status, result = _check_job_status(job_dir)
        previous_status = monitor.get("_last_discover_status", "")
        if status != previous_status:
            monitor["_last_discover_status"] = status
            _verbose_log(
                monitor.get("output_dir", ""),
                "Discover status | {0} -> {1}".format(previous_status or "<none>", status),
            )

        if status == "running":
            return  # still discovering

        monitor["done"] = True
        _stop_import_monitor(monitor_key)

        if status == "error":
            _append_import_log(
                monitor.get("output_dir", ""),
                "Discover failed | {0}".format(result),
            )
            _safe_message_box("Scan Error", "Dataset scan failed: {0}".format(result))
            _cleanup_job_dir(job_dir)
            return

        # Discover done; result contains cases list.
        cases = result.get("cases", [])
        count = result.get("count", len(cases))
        _verbose_log(
            monitor.get("output_dir", ""),
            "Discover done | case_count={0}".format(count),
        )
        _cleanup_job_dir(job_dir)

        if not cases:
            _safe_message_box("Import", "No case data was found in the selected folder.")
            return

        output_dir = monitor.get("output_dir")
        _append_import_log(output_dir, "Discovered {0} case(s); starting preparation.".format(count))

        # Disk space check (quick, non-blocking)
        estimated_mb = count * 500
        ok, free_mb = _check_disk_space(output_dir, estimated_mb)
        _verbose_log(
            output_dir,
            "Disk space | mb_est={0} ok={1} free={2}".format(
                estimated_mb, ok,
                int(free_mb) if free_mb is not None else "?",
            ),
        )
        if not ok:
            _safe_message_box(
                "Insufficient Disk Space",
                "Insufficient disk space. Estimated requirement: {0} MB; "
                "available: {1} MB. Please free disk space and retry.".format(
                    estimated_mb, int(free_mb)),
            )
            return

        _mark_mcs_queue_active(output_dir, count)

        # Start batch prepare-only flow (no Mimics API calls, GUI stays responsive)
        axes = monitor.get("axes")
        flips = monitor.get("flips")
        jobs_dir = monitor.get("jobs_dir")

        first_case = cases[0]
        first_case_id = first_case["case_id"]
        first_work_dir = os.path.join(output_dir, first_case_id + "_work")
        first_job_dir = os.path.join(jobs_dir, first_case_id)

        _verbose_log(
            output_dir,
            "First case | case_id={0}".format(first_case_id),
        )
        _append_import_log(output_dir, "[1/{0}] Preparing: {1}".format(count, first_case_id))
        bridge_params = _build_bridge_params(first_case, axes, flips, first_work_dir)
        _verbose_log(output_dir, "Prepare params={0}".format(_summarize_bridge_params(bridge_params)))
        _launch_bridge_job_thread(bridge_params, first_job_dir, output_dir, "preparing", case_id=first_case_id)

        batch_info = {
            "total": count,
            "output_dir": output_dir,
            "axes": axes,
            "flips": flips,
            "jobs_dir": jobs_dir,
            "batch_started_epoch": time.time(),
        }
        _verbose_log(output_dir, "Batch prepare | remaining={0}".format(max(0, len(cases) - 1)))
        _start_batch_prepare_monitor(
            first_job_dir, first_work_dir,
            batch_queue=cases[1:],
            batch_info=batch_info,
        )
    except Exception as exc:
        output_dir = monitor.get("output_dir", "")
        _append_import_exception(output_dir, "_discover_monitor_tick fatal", exc)
        monitor["done"] = True
        try:
            _stop_import_monitor(monitor.get("monitor_key"))
        except Exception:
            pass
        _safe_message_box("Import Error", "Discover callback failed. Please check mimics_import.log for details.")


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
        try:
            _discover_monitor_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_win32_discover_monitor timer callback failed", exc)

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
    _verbose_log(
        output_dir,
        "Discover monitor | ts_root={0} | poll={1}s | timeout={2}s".format(
            ts_root, poll_seconds, timeout_seconds,
        ),
    )

    # Prefer Win32 SetTimer so returning from the native folder picker does not
    # immediately trigger a potentially slow PyQt5/plugin import in Mimics.
    if _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
        return True

    # Non-Windows fallback: use the existing Qt event loop.
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        mimics.dialogs.message_box(
            title="Scan Running",
            message="Dataset scan has started, but progress cannot be monitored automatically.",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
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
        try:
            _discover_monitor_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_import_discover_monitor qtimer callback failed", exc)

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


def _run_main_with_args(args, import_mode=None, case_info_override=None):
    previous = list(sys.argv)
    try:
        sys.argv = [previous[0]] + list(args)
        return main(import_mode=import_mode, case_info_override=case_info_override)
    finally:
        sys.argv = previous


def _launch_external_import_setup(import_mode):
    import io_setup_mimics

    mode = "import_single" if import_mode == "single_case" else "import_batch"
    configured = str(_load_data_io_config().get("mimics_output_dir", "") or "")

    def submitted(selection):
        source = str(selection.get("source_path", "") or "")
        output_dir = str(selection.get("output_path", "") or "")
        if not source or not output_dir:
            raise RuntimeError("The source and output paths were not returned by the path window.")
        if mode == "import_single":
            source_arg = "--image-file" if os.path.isfile(source) else "--case-dir"
            args = [source_arg, source, "--output-dir", output_dir]
        else:
            args = ["--ts-root", source, "--output-dir", output_dir]
        _run_main_with_args(
            args,
            import_mode=import_mode,
            case_info_override=selection.get("case_info"),
        )

    return io_setup_mimics.launch(
        mode,
        _python_exe(),
        {"configured_output": configured},
        submitted,
    )


# -- Main entry ---------------------------------------------------------

def main(import_mode=None, case_info_override=None):
    """Entry point. Reads config from argv or interactive dialog.

    Usage:
        mimics_import.py --ts-root <dir> [--cases s0000,s0001] [--output-dir <dir>] [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --case-dir <dir> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --image-file <volume.mhd> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --image-file <volume.mhd> --mask-files mask1.nii.gz,mask2.nii.gz --output <file.mcs>

    import_mode:
        None         — interactive (ask)
        "single_case" — skip dialog, single-case mode directly
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
    image_file = None
    output = None
    output_dir = None
    cases_filter = None
    mask_files = None
    axes = [0, 1, 2]
    flips = [False, False, False]

    args = sys.argv[1:]
    _verbose_log("", "main() argv={0}".format(sys.argv))
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--ts-root" and i + 1 < len(args):
            ts_root = args[i + 1]
            i += 2
        elif arg == "--case-dir" and i + 1 < len(args):
            case_dir = args[i + 1]
            i += 2
        elif arg == "--image-file" and i + 1 < len(args):
            image_file = args[i + 1]
            case_dir = image_file
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
        elif arg == "--mask-files" and i + 1 < len(args):
            mask_files = [p.strip() for p in args[i + 1].split(",") if p.strip()]
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

    _verbose_log(
        "",
        "Args | ts_root={0} | case_dir={1} | output={2} | output_dir={3} | axes={4} | flips={5}".format(
            ts_root,
            case_dir,
            output,
            output_dir,
            axes,
            flips,
        ),
    )

    # Interactive path selection is hosted in external PySide6. Mimics only
    # launches it and polls a tiny status JSON through a GUI timer.
    if not ts_root and not case_dir:
        return _launch_external_import_setup(import_mode)

    # -- Single case mode ----------------------------------------------

    if case_dir:
        _verbose_log("", "Mode: single-case | source={0}".format(case_dir))
        selected_source = os.path.abspath(case_dir)
        source_is_file = _is_medical_image_file(selected_source)
        source_case_dir = os.path.dirname(selected_source) if source_is_file else selected_source
        source_case_id = _image_stem(selected_source) if source_is_file else os.path.basename(selected_source)
        case_info = case_info_override or _discover_single_case(selected_source)
        if case_info is None:
            mimics.dialogs.message_box(title="Error", message="No supported image data found: {0}".format(selected_source))
            return 1
        if not output:
            dataset_root = source_case_dir if source_is_file else os.path.dirname(source_case_dir)
            output_dir_default = output_dir or _resolve_import_output_dir(dataset_root)
            output = os.path.join(output_dir_default, case_info["case_id"] + ".mcs")

        # Merge --mask-files from command line into case_info.
        # For interactive mask import, use the dedicated "Import Masks" entry.
        if mask_files:
            extra_masks = []
            for p in mask_files:
                if os.path.isfile(p):
                    extra_masks.append({
                        "name": _mask_name_from_path(p),
                        "path": os.path.abspath(p),
                    })
            existing_names = {m["name"] for m in case_info.get("masks", [])}
            for em in extra_masks:
                if em["name"] in existing_names:
                    case_info["masks"] = [m for m in case_info["masks"] if m["name"] != em["name"]]
                case_info["masks"].append(em)

        output_dir_abs = os.path.dirname(os.path.abspath(output))
        work_dir = os.path.join(output_dir_abs, case_info["case_id"] + "_work")
        jobs_dir = os.path.join(output_dir_abs, "_import_jobs")
        job_dir = os.path.join(jobs_dir, case_info["case_id"])

        # Launch bridge in background + start timer to queue .mcs creation.
        bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
        _launch_bridge_job_thread(bridge_params, job_dir, output_dir_abs, "preparing", case_id=case_info["case_id"])

        # Bridge launched, timer will queue the result for background Mimics.
        _start_import_monitor(job_dir, output, work_dir)
        return 0

    # -- Batch mode: discover and prepare cases without blocking Mimics GUI.

    # No confirmation dialog; user already chose the folder, just start.
    if not output_dir:
        # Resolve only the path on the GUI thread. Directory creation and all
        # logging happen in the launch worker to avoid slow/network-drive I/O
        # immediately after the folder picker closes.
        output_dir = _resolve_import_output_dir(ts_root, create=False)

    jobs_dir = os.path.join(output_dir, "_import_jobs")

    discover_job_dir = os.path.join(jobs_dir, "_discover")
    bridge_params = {
        "action": "discover",
        "ts_root": ts_root,
        "cases_filter": list(cases_filter) if cases_filter else None,
    }
    _update_gui()
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
