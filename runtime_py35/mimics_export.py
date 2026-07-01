# -*- coding: utf-8 -*-
"""mimics_export.py — Mimics-internal driver for .mcs mask → dataset export.

Runs inside Mimics Python 3.5.2. Only uses stdlib + mimics API.
Calls mimics_bridge.py (in nninteractive_env) via subprocess for NIfTI writing.

Uses Win32 SetTimer / PyQt5 QTimer for non-blocking async polling,
matching nnInteractive's pattern — no manual second click needed.

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

import mimics


# ── Global async monitor state ───────────────────────────────────────
_EXPORT_MONITORS = {}


# ── Path helpers (shared with mimics_import.py) ──────────────────────

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
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    result = {"startupinfo": startupinfo}
    if flags:
        result["creationflags"] = flags
    return result


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
            title="导出超时",
            message="数据导出超时，请重试。",
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        return  # still running, next tick will check again

    monitor["done"] = True
    _stop_export_monitor(monitor_key)

    if status == "error":
        mimics.dialogs.message_box(title="导出错误", message="导出出错: {0}".format(result))
        _cleanup_job_dir(job_dir)
        if monitor.get("batch_queue"):
            _start_next_batch_export(monitor)
        return

    # status == "done"
    try:
        work_dir = monitor.get("work_dir")
        total_new, total_overwritten, total_unchanged = _apply_export_result(result, work_dir)
        _cleanup_job_dir(job_dir)
        print("  导出完成")
    except Exception as e:
        print("  导出失败: {0}".format(e))
        traceback.print_exc()
        mimics.dialogs.message_box(title="导出错误", message="导出失败: {0}".format(e))
        _cleanup_job_dir(job_dir)
        if monitor.get("batch_queue"):
            _start_next_batch_export(monitor)
        return

    # Single case: show completion
    if not monitor.get("batch_queue"):
        case_dir = monitor.get("case_dir")
        mimics.dialogs.message_box(
            title="导出完成",
            message="已导出到: {0}\n新建: {1}, 覆盖: {2}, 未变: {3}".format(
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
            title="批量导出完成",
            message="成功导出 {0}/{1} 个病例，{2} 个失败。".format(completed, total, failed),
        )
        return

    case_info = queue.pop(0)
    case_id = case_info["case_id"]
    c_dir = case_info["case_dir"]
    ts_root = monitor.get("ts_root")
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    mcs_path = os.path.join(ts_root, "mcs_output", case_id + ".mcs")
    if not os.path.isfile(mcs_path):
        print("  跳过: .mcs 文件不存在: {0}".format(mcs_path))
        monitor["failed"] = monitor.get("failed", 0) + 1
        _start_next_batch_export(monitor)
        return

    work_dir = os.path.join(ts_root, "mcs_output", case_id + "_work")
    job_dir = os.path.join(jobs_dir, case_id)

    print("\n[{0}/{1}] 正在导出: {2}".format(
        monitor.get("completed", 0) + monitor.get("failed", 0) + 1,
        monitor.get("total", 0),
        case_id))

    # Open .mcs, export masks, close .mcs
    try:
        print("  正在打开: {0}".format(mcs_path))
        mimics.file.open_project(mcs_path)
        built = _export_masks_and_build_params(c_dir, axes, flips, work_dir)
        mimics.file.close_project()
    except Exception as e:
        print("  准备失败 {0}: {1}".format(case_id, e))
        traceback.print_exc()
        monitor["failed"] = monitor.get("failed", 0) + 1
        try:
            mimics.file.close_project()
        except Exception:
            pass
        _start_next_batch_export(monitor)
        return

    if built is None:
        print("  {0} 无可导出的 mask".format(case_id))
        _start_next_batch_export(monitor)
        return

    bridge_params, manifest = built
    process = _launch_bridge_background(bridge_params, job_dir)

    # Update monitor for next case
    monitor["job_dir"] = job_dir
    monitor["work_dir"] = work_dir
    monitor["case_dir"] = c_dir
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 600)
    monitor["done"] = False

    state = {
        "phase": "converting",
        "case_id": case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    _EXPORT_MONITORS[job_dir] = monitor


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
            title="导出进行中",
            message="数据导出已开始，但无法自动应用结果。\n请稍后再次运行导出查看结果。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_export_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="导出进行中",
            message="数据导出已开始，但无法自动应用结果。\n请稍后再次运行导出查看结果。",
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


# ── Export masks from current Mimics project ──────────────────────────

def _sanitize_name(name):
    safe = name
    for ch in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


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


def export_masks_to_buffers(buffers_dir):
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
    print("发现 {0} 个 mask".format(len(masks)))

    manifest = {"masks": [], "mimics_shape": None}

    # Get image shape from first mask's image
    if len(masks) > 0:
        try:
            img = masks[0].image
            dims = [int(v) for v in img.logical_dimensions]
            manifest["mimics_shape"] = dims
            print("图像尺寸: {0}".format(dims))
        except Exception as e:
            print("无法获取图像尺寸: {0}".format(e))

    for a_mask in masks:
        name = str(a_mask.name)
        print("正在导出 mask: {0}".format(name))

        try:
            raw = _get_voxel_buffer_bytes(a_mask)
        except Exception as e:
            print("  获取 {0} 数据出错: {1}".format(name, e))
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


# ── Disk space helpers ────────────────────────────────────────────────

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


# ── Single case export ────────────────────────────────────────────────

def _export_masks_and_build_params(case_dir, axes, flips, work_dir):
    """Export masks to .u8 buffers and build bridge params. Returns (bridge_params, manifest) or None."""
    print("Exporting masks to: {0}".format(case_dir))

    buffers_dir = os.path.join(work_dir, "export_buffers")

    # Step 1: Export masks to .u8 buffers
    manifest = export_masks_to_buffers(buffers_dir)

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
    }
    return (bridge_params, manifest)


def _apply_export_result(result, work_dir):
    """Process bridge convert result and clean up."""
    total_new = result.get("total_new", 0)
    total_overwritten = result.get("total_overwritten", 0)
    total_unchanged = result.get("total_unchanged", 0)
    print("Export complete: {0} new, {1} overwritten, {2} unchanged".format(
        total_new, total_overwritten, total_unchanged))

    # Clean up intermediate .u8 buffers
    _cleanup_work_dir(work_dir)

    return total_new, total_overwritten, total_unchanged


# ── TS case discovery (for batch mode) ────────────────────────────────

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
        for img_name in ("ct.nii.gz", "mri.nii.gz"):
            if os.path.isfile(os.path.join(case_dir, img_name)):
                has_image = True
                break
        if not has_image:
            dicom_dir = os.path.join(case_dir, "dicom")
            if os.path.isdir(dicom_dir):
                has_image = True
        if not has_image:
            for fname in sorted(os.listdir(case_dir)):
                if fname.endswith(".nii.gz"):
                    has_image = True
                    break

        if has_image:
            cases.append({"case_id": name, "case_dir": case_dir})
    return cases


# ── Main entry ────────────────────────────────────────────────────────

def main():
    """Entry point.

    Usage:
        mimics_export.py --case-dir <dir> [--axes 0,1,2] [--flips false,false,false]
        mimics_export.py --ts-root <dir> [--cases s0000,s0001] [--axes 0,1,2] [--flips false,false,false]
    """
    # Clean up any leftover processes / lock files from a previous crashed session.
    # Run in background so Mimics UI does not freeze during PowerShell queries.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        from mimics_import import _cleanup_stale_processes
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
        else:
            i += 1

    # Interactive: if no args, ask for case dir or TS root
    if not ts_root and not case_dir:
        case_dir = _pick_directory("Select case directory")
        if not case_dir or not os.path.isdir(case_dir):
            mimics.dialogs.message_box(
                "No valid directory selected.",
                title="Export Masks",
                ui_blocking=True,
            )
            return 1

    # ── Single case mode ──────────────────────────────────────────────

    if case_dir:
        work_dir = os.path.join(case_dir, "mcs_work")
        jobs_dir = os.path.join(case_dir, "_export_jobs")
        job_dir = os.path.join(jobs_dir, os.path.basename(case_dir))

        # Export masks to buffers, then launch bridge + start timer
        try:
            built = _export_masks_and_build_params(case_dir, axes, flips, work_dir)
        except Exception as e:
            mimics.dialogs.message_box(title="Export Error", message=str(e))
            raise

        if built is None:
            mimics.dialogs.message_box(title="Export", message="No masks to export.")
            return 0

        bridge_params, manifest = built
        process = _launch_bridge_background(bridge_params, job_dir)

        state = {
            "phase": "converting",
            "case_id": os.path.basename(case_dir),
            "pid": process.pid,
            "started_at": time.time(),
        }
        with open(os.path.join(job_dir, "job_state.json"), "w") as f:
            json.dump(state, f)

        print("  launched bridge (pid={0}), starting auto-poll timer...".format(process.pid))
        _start_export_monitor(job_dir, work_dir, case_dir=case_dir)
        return 0

    # ── Batch mode ────────────────────────────────────────────────────
    cases = discover_ts_cases(ts_root, cases_filter)
    if not cases:
        mimics.dialogs.message_box(
            title="Error",
            message="No cases found in: {0}".format(ts_root),
        )
        return 1

    print("Found {0} cases".format(len(cases)))

    # Disk space pre-check: estimate ~200 MB per case (.u8 buffers + NIfTI)
    estimated_mb = len(cases) * 200
    ok, free_mb = _check_disk_space(ts_root, estimated_mb)
    if not ok:
        mimics.dialogs.message_box(
            title="Disk Space Warning",
            message=(
                "Insufficient disk space on {0}.\n"
                "Estimated: {1} MB, Available: {2} MB\n"
                "Aborting to avoid partial failure."
            ).format(os.path.splitdrive(ts_root)[0], estimated_mb, int(free_mb)),
        )
        return 1
    if free_mb >= 0:
        print("Disk space: {0} MB free (estimated {1} MB needed)".format(int(free_mb), estimated_mb))

    # Confirm
    try:
        proceed = mimics.dialogs.question_box(
            title="Confirm Export",
            message="Export masks for {0} cases?\nCases will be processed one-by-one automatically.".format(len(cases)),
        )
        if not proceed:
            print("Cancelled")
            return 0
    except Exception:
        pass

    jobs_dir = os.path.join(ts_root, "mcs_output", "_export_jobs")

    # Start first case, timer will auto-chain to next cases
    first_case = cases[0]
    first_case_id = first_case["case_id"]
    first_c_dir = first_case["case_dir"]
    first_mcs_path = os.path.join(ts_root, "mcs_output", first_case_id + ".mcs")

    if not os.path.isfile(first_mcs_path):
        mimics.dialogs.message_box(title="Error", message=".mcs not found: {0}".format(first_mcs_path))
        return 1

    first_work_dir = os.path.join(ts_root, "mcs_output", first_case_id + "_work")
    first_job_dir = os.path.join(jobs_dir, first_case_id)

    # Open .mcs, export masks, close .mcs
    try:
        print("[1/{0}] {1}".format(len(cases), first_case_id))
        print("  opening: {0}".format(first_mcs_path))
        mimics.file.open_project(first_mcs_path)
        built = _export_masks_and_build_params(first_c_dir, axes, flips, first_work_dir)
        mimics.file.close_project()
    except Exception as e:
        mimics.dialogs.message_box(title="Export Error", message=str(e))
        try:
            mimics.file.close_project()
        except Exception:
            pass
        raise

    if built is None:
        mimics.dialogs.message_box(title="Export", message="No masks to export for first case.")
        return 0

    bridge_params, manifest = built
    process = _launch_bridge_background(bridge_params, first_job_dir)

    state = {
        "phase": "converting",
        "case_id": first_case_id,
        "pid": process.pid,
        "started_at": time.time(),
    }
    with open(os.path.join(first_job_dir, "job_state.json"), "w") as f:
        json.dump(state, f)

    print("  launched bridge (pid={0}), starting auto-poll timer...".format(process.pid))

    batch_info = {
        "total": len(cases),
        "ts_root": ts_root,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
    }
    _start_export_monitor(
        first_job_dir, first_work_dir, case_dir=first_c_dir,
        batch_queue=cases[1:],  # remaining cases
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
            mimics.dialogs.message_box(title="Fatal Error", message=str(error))
        except Exception:
            pass
        raise
