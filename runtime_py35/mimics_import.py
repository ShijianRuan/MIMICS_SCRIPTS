# -*- coding: utf-8 -*-
"""mimics_import.py — Mimics-internal driver for dataset → .mcs import.

Runs inside Mimics Python 3.5.2. Only uses stdlib + mimics API.
Calls mimics_bridge.py (in nninteractive_env) via subprocess for NIfTI/DICOM work.

Uses Win32 SetTimer / PyQt5 QTimer for non-blocking async polling,
matching nnInteractive's pattern — no manual second click needed.

Flow:
    1. Annotator picks a case directory
    2. Launch mimics_bridge.py "prepare" in background (non-blocking)
    3. Timer polls every 0.5s until bridge completes
    4. Auto-apply result: DICOM import, mask creation, .mcs save
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

import mimics


# ── Global async monitor state ───────────────────────────────────────
_IMPORT_MONITORS = {}

# Track background Mimics process for .mcs creation
_BG_MIMICS_PID = None


# ── Path helpers (same pattern as nninteractive_mimics.py) ─────────────

def _find_root(start_dir, sentinel_files, max_depth=6):
    current = os.path.abspath(start_dir)
    for _ in range(max_depth):
        for sentinel in sentinel_files:
            if os.path.isfile(os.path.join(current, sentinel)):
                return current
            if os.path.isdir(os.path.join(current, sentinel)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "nninteractive_env", ".git"),
    )


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


def _hidden_process_kwargs():
    """Return subprocess kwargs that suppress console windows on Windows.

    Mimics embeds Python 3.5.2 where ``subprocess.CREATE_NO_WINDOW`` may not
    exist.  We use hardcoded Win32 constants so the flag is always applied.
    """
    if os.name != "nt":
        return {}
    import ctypes

    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE

    # CREATE_NO_WINDOW (0x08000000) | CREATE_NEW_PROCESS_GROUP (0x00000200)
    creationflags = 0x08000000 | 0x00000200

    return {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
    }


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


# ── Stale process / temp cleanup ─────────────────────────────────────

def _cleanup_stale_processes():
    """Kill leftover bridge python and background Mimics processes from a
    previous crashed session.  Also removes Mimics temp lock files.

    Called at the start of main() so every import begins with a clean slate.

    .. note::
        nnInteractive server and watchdog processes are **excluded** from
        killing so that a running inference server is not disrupted.

    Uses a single hidden batch PowerShell call instead of per-process calls
    to avoid popping up visible console windows that freeze Mimics.
    """
    killed = []

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
    locks_removed = 0
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
            parts.append("清理 {0} 个残留进程".format(len(killed)))
        if locks_removed:
            parts.append("删除 {0} 个锁文件".format(locks_removed))
        print("启动清理: " + ", ".join(parts))


# ── TS case discovery (runs in Mimics Python 3.5, stdlib only) ────────

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
    Used for lazy discovery — each case is scanned only when it's
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


# ── Call mimics_bridge.py ─────────────────────────────────────────────

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


# ── Async bridge helpers ─────────────────────────────────────────────

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

    process = subprocess.Popen(
        [python_exe, bridge],
        stdin=open(input_file, "r"),
        stdout=open(result_file, "w"),
        stderr=open(error_file, "w"),
        **_hidden_process_kwargs()
    )
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


# ── Timer-based async monitor (same pattern as nnInteractive) ────────

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
    """Timer callback: check if bridge finished, apply result if done."""
    if monitor.get("done"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        mimics.dialogs.message_box(
            title="导入超时",
            message="数据转换超时，请重试。",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    monitor["done"] = True
    _stop_import_monitor(monitor_key)

    if status == "error":
        mimics.dialogs.message_box(title="导入错误", message="转换出错: {0}".format(result))
        _cleanup_job_dir(job_dir)
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        return

    # status == "done"
    try:
        output_mcs = monitor.get("output_mcs")
        work_dir = monitor.get("work_dir")
        mcs_path = _apply_import_result(result, output_mcs, work_dir)
        _cleanup_job_dir(job_dir)
        print("  导入完成: {0}".format(mcs_path))
    except Exception as e:
        print("  导入失败: {0}".format(e))
        traceback.print_exc()
        mimics.dialogs.message_box(title="导入错误", message="导入失败: {0}".format(e))
        _cleanup_job_dir(job_dir)
        try:
            mimics.file.close_project()
        except Exception:
            pass
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        return

    # Single case: show completion
    if not monitor.get("batch_queue"):
        mimics.dialogs.message_box(
            title="导入完成",
            message="已创建: {0}".format(mcs_path),
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
            title="批量导入完成",
            message="成功导入 {0}/{1} 个病例，{2} 个失败。".format(completed, total, failed),
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

    print("\n[{0}/{1}] 正在导入: {2}".format(
        monitor.get("completed", 0) + monitor.get("failed", 0) + 1,
        monitor.get("total", 0),
        case_id))

    bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
    process = _launch_bridge_background(bridge_params, job_dir)

    # Update monitor for next case
    monitor["job_dir"] = job_dir
    monitor["output_mcs"] = output_mcs
    monitor["work_dir"] = work_dir
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 600)
    monitor["done"] = False

    state = {
        "phase": "preparing",
        "case_id": case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    _IMPORT_MONITORS[job_dir] = monitor


# ── Batch prepare-only flow (no Mimics API, GUI stays responsive) ──

def _batch_prepare_tick(monitor):
    """Timer callback for batch prepare: bridge done → save manifest → next case."""
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
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        mimics.dialogs.message_box(
            title="转换超时",
            message="数据转换超时，请重试。",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    # ── Case finished (done or error) ──────────────────────────────
    # Set busy flag to prevent re-entrancy during message_box etc.
    monitor["busy"] = True

    # Remove old job_dir from _IMPORT_MONITORS (timer stays alive)
    _IMPORT_MONITORS.pop(monitor_key, None)

    if status == "error":
        print("  转换失败: {0}".format(result))
        monitor["failed"] = monitor.get("failed", 0) + 1
        _cleanup_job_dir(job_dir)
        monitor["busy"] = False
        _start_next_batch_prepare(monitor)
        return

    # status == "done" — save manifest for later .mcs creation
    work_dir = monitor.get("work_dir")
    case_id = result.get("case_id", monitor.get("case_id", ""))
    output_dir = monitor.get("output_dir")
    completed_count = monitor.get("completed", 0) + 1
    total = monitor.get("total", 0)

    # Update completed count BEFORE doing heavy work / showing dialog,
    # so re-entrant timer ticks see consistent state.
    monitor["completed"] = completed_count

    # First case: open immediately in the current Mimics instance so the
    # user can start annotating while remaining cases convert in background.
    if completed_count == 1:
        try:
            output_mcs = os.path.join(output_dir, case_id + ".mcs")
            mcs_path = _apply_import_result(result, output_mcs, work_dir)
            print("  [{0}/{1}] {2} 已打开，可以开始标注".format(completed_count, total, case_id))
        except Exception as exc:
            print("  打开第一例失败: {0}".format(exc))
            traceback.print_exc()
            # Fall back to manifest-based approach for this case
            manifest_path = os.path.join(work_dir, "prepare_manifest.json")
            try:
                with open(manifest_path, "w") as f:
                    json.dump(result, f)
            except Exception:
                pass
            _cleanup_job_dir(job_dir)
            # Start background Mimics to create .mcs from this manifest
            _ensure_bg_mimics_running(output_dir)
    else:
        # Subsequent cases: save manifest, then ensure background Mimics
        # is creating .mcs files from accumulated manifests.
        manifest_path = os.path.join(work_dir, "prepare_manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(result, f)
            print("  [{0}/{1}] {2} 转换完成".format(completed_count, total, case_id))
        except Exception as e:
            print("  保存清单失败: {0}".format(e))
        _cleanup_job_dir(job_dir)
        # Ensure background Mimics is running to create .mcs files
        _ensure_bg_mimics_running(output_dir)

    # Check if more cases to prepare — delegate to _start_next_batch_prepare
    # which handles both "start next" and "all done" cases.
    _start_next_batch_prepare(monitor)
    monitor["busy"] = False


def _start_next_batch_prepare(monitor):
    """Start bridge prepare for the next case in the batch queue.

    Queue items are (name, case_dir) tuples — lazy discovery: each case
    is scanned only when it's about to be converted.
    """
    queue = monitor.get("batch_queue")
    if not queue:
        # All cases converted — stop timer.
        # Background Mimics should already be running (started after 2nd
        # case).  Just ensure it's alive for any remaining manifests.
        monitor["done"] = True
        monitor["busy"] = False
        _stop_import_monitor(monitor.get("monitor_key"))
        completed = monitor.get("completed", 0)
        failed = monitor.get("failed", 0)
        total = monitor.get("total", 0)
        output_dir = monitor.get("output_dir")
        print("全部 {0} 个病例转换完成（{1} 个失败）。后台正在创建 .mcs 文件...".format(completed, failed))
        _ensure_bg_mimics_running(output_dir)
        return

    # Pop next (name, case_dir) from queue
    item = queue.pop(0)
    if isinstance(item, tuple):
        case_name, case_dir = item
    else:
        # Backward compat: old-style case_info dict
        case_name = item.get("case_id", "")
        case_dir = item.get("case_dir", "")

    # Lazy discovery: scan this case's image/masks now (fast — single dir)
    case_info = _discover_single_case(case_dir)
    if case_info is None:
        print("  跳过（无图像数据）: {0}".format(case_name))
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
    print("[{0}/{1}] 正在转换: {2}".format(completed + failed + 1, total, case_id))

    bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
    process = _launch_bridge_background(bridge_params, job_dir)

    # Update monitor for next case
    old_monitor_key = monitor.get("monitor_key")
    monitor["job_dir"] = job_dir
    monitor["work_dir"] = work_dir
    monitor["case_id"] = case_id
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 600)
    monitor["done"] = False

    # Write job state so _check_job_status can find the pid
    state = {
        "phase": "preparing",
        "case_id": case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    # Re-register monitor under new key (timer keeps running)
    _IMPORT_MONITORS.pop(old_monitor_key, None)
    _IMPORT_MONITORS[job_dir] = monitor


def _start_batch_prepare_monitor(job_dir, work_dir, timeout_seconds=600,
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
            title="转换进行中",
            message="数据转换已开始，但无法自动监控进度。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="转换进行中",
            message="数据转换已开始，但无法自动监控进度。",
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


def _ensure_bg_mimics_running(output_dir):
    """Ensure a background Mimics is running to create .mcs files.

    Called after each case's manifest is saved.  If no background Mimics
    is alive, launch one.  This way .mcs files are created continuously
    as manifests become available, rather than waiting for all conversions
    to finish.
    """
    global _BG_MIMICS_PID
    # Check if existing background Mimics is still alive
    if _BG_MIMICS_PID and _is_pid_alive(_BG_MIMICS_PID):
        return  # still running, it will pick up new manifests

    # Launch a new one
    _launch_background_mimics(output_dir)


def _launch_background_mimics(output_dir, total_count=0):
    """Launch Mimics in background mode to create .mcs files from prepared data.

    Mimics runs without GUI (-b flag), executing create_mcs_batch.py which
    reads prepare manifests and creates .mcs files one by one.
    Returns the Popen object, or None on failure.
    """
    global _BG_MIMICS_PID
    mimics_exe = _find_mimics_exe()
    if not mimics_exe:
        print("  未找到 MimicsResearch.exe，无法启动后台创建")
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
    try:
        process = subprocess.Popen(cmd, **_hidden_process_kwargs())
        _BG_MIMICS_PID = process.pid
        print("  后台 Mimics 已启动 (PID={0})，正在创建 .mcs 文件...".format(process.pid))
        return process
    except Exception as e:
        print("  无法启动后台 Mimics: {0}".format(e))
        return None


# ── Auto-open first .mcs monitor ────────────────────────────────────

def _first_mcs_monitor_tick(monitor):
    """Timer callback: check if first .mcs file exists, open it when found."""
    if monitor.get("done"):
        return

    output_dir = monitor.get("output_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        return

    # Check if any .mcs file exists in output_dir
    try:
        items = sorted(os.listdir(output_dir))
    except Exception:
        return

    for item in items:
        if item.endswith(".mcs"):
            mcs_path = os.path.join(output_dir, item)
            if os.path.isfile(mcs_path):
                monitor["done"] = True
                _stop_import_monitor(monitor_key)
                try:
                    mimics.file.open_project(filename=mcs_path)
                    print("已自动打开第一个病例: {0}".format(item))
                except Exception:
                    pass
                return


def _start_first_mcs_monitor(output_dir, timeout_seconds=300, poll_seconds=2.0):
    """Start a timer that polls for the first .mcs file and auto-opens it."""
    monitor = {
        "monitor_key": "first_mcs_" + output_dir,
        "output_dir": output_dir,
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


def _start_import_monitor(job_dir, output_mcs, work_dir, timeout_seconds=600,
                          poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer that polls for bridge completion and auto-applies."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "output_mcs": output_mcs,
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

    # Try PyQt5 QTimer first (works inside Mimics GUI event loop)
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        # Fall back to Win32 SetTimer
        if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="导入进行中",
            message="数据转换已开始，但无法自动应用结果。\n请稍后再次运行导入查看结果。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="导入进行中",
            message="数据转换已开始，但无法自动应用结果。\n请稍后再次运行导入查看结果。",
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


# ── Mask buffer injection (same as sp_common.set_mask_buffer_from_u8) ─

def inject_buffer(mask, buffer_path, mimics_shape):
    with open(buffer_path, "rb") as f:
        raw = f.read()
    expected = 1
    for dim in mimics_shape:
        expected *= int(dim)
    if len(raw) != expected:
        raise RuntimeError(
            "buffer byte count mismatch: {} != {}  path={}".format(len(raw), expected, buffer_path)
        )
    try:
        import numpy as np
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(mimics_shape)).astype(np.bool_)
        mask.set_voxel_buffer(pixels)
        return "numpy"
    except ImportError:
        view = memoryview(bytearray(raw)).cast("?", shape=list(mimics_shape))
        mask.set_voxel_buffer(view)
        return "memoryview"


# ── Disk space helpers ────────────────────────────────────────────────

def _cleanup_work_dir(work_dir):
    """Remove intermediate work directory (DICOM + .u8 buffers)."""
    if not work_dir or not os.path.isdir(work_dir):
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        print("  清理临时文件")
    except Exception as e:
        print("  清理临时文件失败: {0}".format(e))


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
    # Try PyQt5 first — Mimics is a Qt app, so QFileDialog integrates
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


# ── Discover monitor (batch mode: discover → confirm → import chain) ─

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
            title="扫描超时",
            message="扫描数据集超时，请重试。",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still discovering

    monitor["done"] = True
    _stop_import_monitor(monitor_key)

    if status == "error":
        mimics.dialogs.message_box(title="扫描错误", message="扫描出错: {0}".format(result))
        _cleanup_job_dir(job_dir)
        return

    # Discover done — result contains cases list
    cases = result.get("cases", [])
    count = result.get("count", len(cases))
    _cleanup_job_dir(job_dir)

    if not cases:
        mimics.dialogs.message_box(
            title="导入",
            message="所选文件夹中没有找到病例数据。",
        )
        return

    print("发现 {0} 个病例，开始转换...".format(count))

    # Disk space check (quick, non-blocking)
    output_dir = monitor.get("output_dir")
    estimated_mb = count * 500
    ok, free_mb = _check_disk_space(output_dir, estimated_mb)
    if not ok:
        mimics.dialogs.message_box(
            title="磁盘空间不足",
            message="磁盘空间不足，需要约 {0} MB，仅剩 {1} MB。\n请清理磁盘后重试。".format(estimated_mb, int(free_mb)),
        )
        return

    # Auto-start prepare (no confirmation needed — user already chose the folder)

    # Start batch prepare-only flow (no Mimics API calls, GUI stays responsive)
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    first_case = cases[0]
    first_case_id = first_case["case_id"]
    first_work_dir = os.path.join(output_dir, first_case_id + "_work")
    first_job_dir = os.path.join(jobs_dir, first_case_id)

    print("[1/{0}] 正在转换: {1}".format(count, first_case_id))
    bridge_params = _build_bridge_params(first_case, axes, flips, first_work_dir)
    process = _launch_bridge_background(bridge_params, first_job_dir)

    state = {
        "phase": "preparing",
        "case_id": first_case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(first_job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    # Bridge launched, timer will auto-chain remaining cases

    batch_info = {
        "total": count,
        "output_dir": output_dir,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
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
                                    poll_seconds=0.5, timeout_seconds=120):
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
            title="扫描进行中",
            message="正在扫描数据集，但无法自动监控进度。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="扫描进行中",
            message="正在扫描数据集，但无法自动监控进度。",
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


# ── Single case import ────────────────────────────────────────────────

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
    }


def _apply_import_result(result, output_mcs, work_dir):
    """Apply bridge prepare result in Mimics (DICOM import, masks, save).

    Runs in the Mimics Python thread. Uses mimics API.
    Disables GUI updates during mask creation to prevent progressive
    mask display and potential crashes from rapid UI refreshes.
    """
    dicom_folder = result["dicom_folder"]
    mask_results = result["masks"]
    print("  正在导入 DICOM 数据...")
    mimics.file.import_dicom_images(source_folder=dicom_folder)

    if len(mimics.data.images) == 0:
        raise RuntimeError("DICOM import produced no images")

    image = mimics.data.images[0]
    mimics_shape = [int(v) for v in image.logical_dimensions]
    mimics.data.images.set_active(image)

    # Disable GUI updates during mask creation to prevent progressive
    # display and crashes from rapid UI refreshes in timer callbacks.
    gui_was_enabled = True
    try:
        mimics.disable_update_gui()
    except Exception:
        gui_was_enabled = False

    try:
        # Create masks and inject buffers.
        # Note: mask.image is read-only — masks are automatically linked
        # to the active image at creation time.  Do NOT set mask.image.
        # Set visible=False BEFORE injecting buffer data so Mimics never
        # shows the mask even briefly during set_voxel_buffer.
        for mr in mask_results:
            name = mr["name"]
            u8_path = mr["u8_path"]
            buf_shape = mr["mimics_shape"]

            mask = mimics.segment.create_mask()
            mask.name = name
            # Hide immediately, before any data is injected
            try:
                mask.visible = False
            except Exception:
                pass

            method = inject_buffer(mask, u8_path, buf_shape)
            print("    -> {0}".format(name))
    finally:
        # Re-enable GUI updates (must always run, even on error)
        if gui_was_enabled:
            try:
                mimics.enable_update_gui()
            except Exception:
                pass

    # Save .mcs
    mcs_path = os.path.abspath(output_mcs)
    mcs_dir = os.path.dirname(mcs_path)
    if mcs_dir and not os.path.isdir(mcs_dir):
        os.makedirs(mcs_dir)
    print("  正在保存 .mcs 文件...")
    mimics.file.save_project(filename=mcs_path, save_as_type="Mimics Project Files")
    print("  保存完成: {0}".format(mcs_path))

    # Clean up intermediate work dir
    _cleanup_work_dir(work_dir)

    return mcs_path


# ── Main entry ────────────────────────────────────────────────────────

def main():
    """Entry point. Reads config from argv or interactive dialog.

    Usage:
        mimics_import.py --ts-root <dir> [--cases s0000,s0001] [--output-dir <dir>] [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --case-dir <dir> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
    """
    # Clean up any leftover processes / lock files from a previous crashed session.
    # Run in background so Mimics UI does not freeze during PowerShell queries.
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
        ts_root = _pick_directory("请选择数据集文件夹")
        if not ts_root or not os.path.isdir(ts_root):
            mimics.dialogs.message_box(
                "未选择有效的文件夹。",
                title="导入数据集",
                ui_blocking=True,
            )
            return 1

    # ── Single case mode ──────────────────────────────────────────────

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
            mimics.dialogs.message_box(title="错误", message="未找到图像数据: {0}".format(case_dir))
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

        # Launch bridge in background + start timer to auto-apply result
        print("正在准备: {0}".format(case_info["case_id"]))
        bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
        process = _launch_bridge_background(bridge_params, job_dir)

        state = {
            "phase": "preparing",
            "case_id": case_info["case_id"],
            "pid": process.pid,
            "started_at": time.time(),
        }
        with open(os.path.join(job_dir, "job_state.json"), "w") as f:
            json.dump(state, f)

        # Bridge launched, timer will auto-apply result
        _start_import_monitor(job_dir, output, work_dir)
        return 0

    # ── Batch mode: prepare all cases directly ──────────────────────

    # No confirmation dialog — user already chose the folder, just start.
    if not output_dir:
        output_dir = os.path.join(ts_root, "mcs_output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    jobs_dir = os.path.join(output_dir, "_import_jobs")

    # Lazy discovery: only do a quick os.listdir to get directory names.
    # Each case's image/mask details are scanned on-the-fly when it's
    # about to be converted, so the user sees progress immediately
    # instead of waiting for a full dataset scan.
    print("正在扫描数据集...")
    all_names = sorted(os.listdir(ts_root))
    case_names = []
    for name in all_names:
        if name in ("mcs_output", "segmentations"):
            continue
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if cases_filter and name not in cases_filter:
            continue
        case_names.append(name)

    count = len(case_names)
    if count == 0:
        mimics.dialogs.message_box(
            title="导入",
            message="所选文件夹中没有找到病例数据。",
        )
        return 1

    print("发现 {0} 个病例目录，开始转换...".format(count))

    # Disk space check
    estimated_mb = count * 500
    ok, free_mb = _check_disk_space(output_dir, estimated_mb)
    if not ok:
        mimics.dialogs.message_box(
            title="磁盘空间不足",
            message="磁盘空间不足，需要约 {0} MB，仅剩 {1} MB。".format(estimated_mb, int(free_mb)),
        )
        return 1

    # Discover first case in detail (fast — single directory scan)
    first_name = case_names[0]
    first_case_dir = os.path.join(ts_root, first_name)
    first_case = _discover_single_case(first_case_dir)
    if first_case is None:
        mimics.dialogs.message_box(
            title="导入",
            message="第一个病例目录中没有找到图像数据: {0}".format(first_name),
        )
        return 1

    # Remaining cases: store (name, case_dir) tuples for lazy discovery
    remaining = [(n, os.path.join(ts_root, n)) for n in case_names[1:]]

    first_case_id = first_case["case_id"]
    first_work_dir = os.path.join(output_dir, first_case_id + "_work")
    first_job_dir = os.path.join(jobs_dir, first_case_id)

    print("[1/{0}] 正在转换: {1}".format(count, first_case_id))
    bridge_params = _build_bridge_params(first_case, axes, flips, first_work_dir)
    process = _launch_bridge_background(bridge_params, first_job_dir)

    state = {
        "phase": "preparing",
        "case_id": first_case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(first_job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    batch_info = {
        "total": count,
        "output_dir": output_dir,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
        "ts_root": ts_root,
    }
    _start_batch_prepare_monitor(
        first_job_dir, first_work_dir,
        batch_queue=remaining,
        batch_info=batch_info,
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
            mimics.dialogs.message_box(title="严重错误", message="发生错误: {0}".format(error))
        except Exception:
            pass
        raise
