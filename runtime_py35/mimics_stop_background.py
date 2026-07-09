# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services.

Runs inside Mimics and submits a hidden PowerShell cleanup request without
waiting for completion, so the foreground Mimics GUI remains responsive.
"""

from __future__ import print_function

import glob
import os
import logging
import subprocess
import threading
import time

import mimics

import runtime_common


# Command-line substrings that identify Mimics-Script owned processes.
# These are matched against the full command line of every running process.
MARKERS = (
    # Bridge processes (external Python)
    "mimics_bridge.py",
    "nninteractive_bridge.py",
    "--async-worker",
    "--watchdog",

    # Auto-generated runner scripts (run inside background Mimics)
    "_run_create_mcs.py",
    "_run_export_batch.py",

    # Actual runtime module names (found via -run_script or import)
    "create_mcs_batch.py",
    "mimics_export.py",
    "mimics_import.py",

    # DINOv3 few-shot training/inference
    "fewshot_pipeline.py",
    "fewshot_mimics.py",
    "fewshot_model_chooser.py",

    # nnInteractive inference server
    "nninteractive.inference.server.main",

    # Environment setup / packaging tools
    "setup_env.py",
    "package_portable.py",
)


_hidden_process_kwargs = runtime_common.hidden_process_kwargs

# Keep async timer/callback references alive until a monitor finishes.
_CACHE_CLEAR_MONITORS = []
_STOP_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "fewshot_config.json", "mimics_bridge.py", ".git"),
    )


def _owned_roots():
    """Return all directory roots that may appear in process command lines."""
    root = _project_root()
    result = [root]
    for child in ("nninteractive_env", "external", "tools", "runtime_py35"):
        path = os.path.join(root, child)
        if os.path.exists(path):
            result.append(path)
    return [os.path.abspath(p) for p in result if p]


def _clear_resource_locks():
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    for name in ("gpu.lock", "background_mimics.lock"):
        try:
            os.remove(os.path.join(lock_dir, name))
        except OSError:
            pass


def _runtime_dir():
    path = os.path.join(_project_root(), ".mimics_runtime")
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


# -- Queue / output directory discovery ---------------------------------

def _scan_filesystem_for_queue_dirs():
    """Walk common locations to find active queue directories.

    This is the fallback when the lock-file and registry approaches
    miss directories (e.g. queue registry is disabled, lock was cleaned
    up, or the output directory is outside the project tree).
    """
    result = []
    root = _project_root()

    # 1. Scan for _mcs_queue_active.json files under the project root
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip deep vendor/env trees
        if any(skip in dirpath.replace(os.sep, "/") for skip in (
            "nninteractive_env", ".git", "__pycache__", "external/dinov3",
        )):
            continue
        if "_mcs_queue_active.json" in filenames:
            result.append(dirpath)
        # Limit depth to avoid scanning huge model directories
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 6:
            _dirnames[:] = []

    # 2. Also check the runtime directory for any stored queue info
    runtime = os.path.join(root, ".mimics_runtime")
    if os.path.isdir(runtime):
        for name in os.listdir(runtime):
            if name.startswith("mcs_") and name.endswith(".json"):
                payload = runtime_common.read_json(os.path.join(runtime, name), {}) or {}
                output_dir = payload.get("output_dir")
                if output_dir:
                    result.append(output_dir)

    return result


def _queue_dirs_from_runtime_state():
    """Discover queue directories from lock files and registries."""
    result = []
    root = _project_root()
    lock_dir = runtime_common.resource_lock_dir(root)

    # Read ALL lock files, not just background_mimics.lock
    if os.path.isdir(lock_dir):
        for name in os.listdir(lock_dir):
            if not name.endswith(".lock"):
                continue
            lock_path = os.path.join(lock_dir, name)
            payload = runtime_common.read_json(lock_path, {}) or {}
            if not payload:
                continue
            # output_dir / ts_root may be at top level or nested under "details"
            details = payload.get("details") or {}
            output_dir = payload.get("output_dir") or details.get("output_dir")
            if output_dir:
                result.append(output_dir)
            ts_root = payload.get("ts_root") or details.get("ts_root")
            if ts_root:
                result.append(ts_root)
                result.append(os.path.join(ts_root, "mcs_output"))

    # Queue registry (may be disabled by env var, but check anyway)
    registry_dir = os.path.join(_runtime_dir(), "mcs_queues")
    if os.path.isdir(registry_dir):
        for name in os.listdir(registry_dir):
            if not name.endswith(".json"):
                continue
            payload = runtime_common.read_json(os.path.join(registry_dir, name), {}) or {}
            output_dir = payload.get("output_dir")
            if output_dir:
                result.append(output_dir)

    # Filesystem scan as aggressive fallback
    result.extend(_scan_filesystem_for_queue_dirs())

    # Deduplicate
    unique = []
    seen = set()
    for path in result:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _request_queue_stop():
    """Write stop markers to every discovered queue directory."""
    stopped = []
    payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": "Stop Background Services",
    }
    for output_dir in _queue_dirs_from_runtime_state():
        if not os.path.isdir(output_dir):
            continue
        try:
            runtime_common.write_json_atomic(
                os.path.join(output_dir, "_mcs_queue_stop.json"), payload
            )
            # Also remove the active marker so the background process
            # doesn't think a producer is still sending work.
            active = os.path.join(output_dir, "_mcs_queue_active.json")
            if os.path.isfile(active):
                os.remove(active)
            stopped.append(output_dir)
        except Exception:
            pass
    return stopped


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


# -- Cache directories --------------------------------------------------

_CACHE_DIRS = (
    ".mimics_runtime",
    "logs",
    "__pycache__",
)

_CACHE_GLOBS = (
    "*_work",           # Bridge prepare work directories
    "_import_jobs",     # Bridge job directories
    "source_fastpath_cache",  # nnInteractive on-demand axial NIfTI cache
)


def _find_cache_paths():
    """Return all cache/temp paths that can be safely cleared."""
    root = _project_root()
    paths = []

    # Direct cache directories under project root
    for name in _CACHE_DIRS:
        p = os.path.join(root, name)
        if os.path.exists(p):
            paths.append(p)

    # Glob patterns (match in root and common output locations)
    for pattern in _CACHE_GLOBS:
        for p in glob.glob(os.path.join(root, pattern)):
            paths.append(p)
        for p in glob.glob(os.path.join(root, "*", pattern)):
            paths.append(p)
        for p in glob.glob(os.path.join(root, "*", "*", pattern)):
            paths.append(p)

    # __pycache__ directories recursively (skip env and .git)
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (
            "nninteractive_env", ".git", "external",
        )]
        if os.path.basename(dirpath) == "__pycache__":
            paths.append(dirpath)

    return sorted(set(os.path.abspath(p) for p in paths if os.path.exists(p)))


def clear_all_caches():
    """Remove all intermediate files, temp directories, and caches.

    Only deletes non-critical, auto-generated files. Project source code,
    config files, model weights, and the nninteractive_env are NOT touched.

    Files that are currently in use (locked by another process) are skipped
    and reported separately so the user knows what was retained.
    """
    paths = _find_cache_paths()
    removed = []
    failed = []
    skipped_in_use = []
    now = time.time()

    for p in paths:
        try:
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p, ignore_errors=False)
            else:
                os.remove(p)
            removed.append(p)
        except OSError as e:
            errno = getattr(e, "winerror", 0) or getattr(e, "errno", 0)
            # Windows ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33)
            # or POSIX EACCES (13) / EPERM (1) — file is likely in use.
            if errno in (1, 13, 32, 33):
                skipped_in_use.append((p, str(e)))
            else:
                failed.append((p, str(e)))
        except Exception as e:
            failed.append((p, str(e)))

    # Also clear stale state files in .mimics_runtime (but keep the dir itself,
    # active locks, and queue registries).
    runtime_dir = os.path.join(_project_root(), ".mimics_runtime")
    if os.path.isdir(runtime_dir):
        _clear_stale_runtime_files(runtime_dir, removed, failed, skipped_in_use)

    # Only clear resource locks when no background process is holding them.
    _clear_resource_locks_safe()

    return {
        "removed": removed,
        "failed": failed,
        "skipped_in_use": skipped_in_use,
    }


def _clear_stale_runtime_files(runtime_dir, removed, failed, skipped_in_use):
    """Remove stale files from .mimics_runtime/, keeping active locks."""
    # Locks that are currently held by a running process
    active_pids = set()
    lock_dir = os.path.join(runtime_dir, "locks")
    if os.path.isdir(lock_dir):
        for name in os.listdir(lock_dir):
            if not name.endswith(".lock"):
                continue
            payload = runtime_common.read_json(os.path.join(lock_dir, name), {}) or {}
            pid = payload.get("pid")
            if pid and runtime_common.process_exists(pid):
                active_pids.add(int(pid))

    for name in os.listdir(runtime_dir):
        filepath = os.path.join(runtime_dir, name)
        if not os.path.isfile(filepath):
            continue
        # Skip active lock files and queue registries
        if name.endswith(".lock") or name == "mcs_queues":
            continue
        try:
            os.remove(filepath)
            removed.append(filepath)
        except OSError as e:
            errno = getattr(e, "winerror", 0) or getattr(e, "errno", 0)
            if errno in (1, 13, 32, 33):
                skipped_in_use.append((filepath, str(e)))
            else:
                failed.append((filepath, str(e)))
        except Exception as e:
            failed.append((filepath, str(e)))


def _clear_resource_locks_safe():
    """Clear resource locks only when no process is holding them."""
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    for name in ("gpu.lock", "background_mimics.lock"):
        path = os.path.join(lock_dir, name)
        if not os.path.isfile(path):
            continue
        payload = runtime_common.read_json(path, {}) or {}
        pid = payload.get("pid")
        if pid and runtime_common.process_exists(int(pid)):
            # Lock holder is still running — keep the lock
            continue
        try:
            os.remove(path)
        except OSError:
            pass


# -- Async cache clearing (non-blocking Mimics GUI) -----------------

def _clear_cache_tick(monitor):
    """Timer callback: check if the background clearing thread finished."""
    if monitor.get("done"):
        return
    thread = monitor.get("thread")
    if thread is None or thread.is_alive():
        now = time.time()
        last = float(monitor.get("last_progress_log_epoch", 0.0) or 0.0)
        if now - last >= 5.0:
            started = float(monitor.get("started_at_epoch", now) or now)
            _mimics_log(
                logging.INFO,
                "Cache clearing is still running... elapsed {0:.1f}s".format(max(0.0, now - started)),
            )
            monitor["last_progress_log_epoch"] = now
        return  # still working

    monitor["done"] = True
    result = monitor.get("result") or {}
    removed_count = len(result.get("removed", []))
    failed_count = len(result.get("failed", []))
    in_use_count = len(result.get("skipped_in_use", []))

    lines = ["Cache cleared."]
    if removed_count:
        lines.append("{0} path(s) removed.".format(removed_count))
    if in_use_count:
        lines.append("{0} path(s) in use — skipped.".format(in_use_count))
    if failed_count:
        lines.append("{0} path(s) could not be removed.".format(failed_count))
        for path, err in result.get("failed", [])[:5]:
            lines.append("  {0}: {1}".format(path, err))

    message = "\n".join(lines)
    try:
        mimics.dialogs.message_box(title="Clear Cache", message=message, ui_blocking=False)
    except TypeError:
        mimics.dialogs.message_box(title="Clear Cache", message=message)
    _mimics_log(logging.INFO, message)
    _dispose_cache_monitor(monitor)


def clear_cache_main():
    """Entry point for the Clear Cache scripting library button.

    All heavy disk I/O runs in a daemon thread. Mimics GUI stays responsive.
    A timer polls the thread every 0.5s and shows results when done.
    """
    monitor = {
        "thread": None,
        "result": None,
        "done": False,
        "started_at_epoch": time.time(),
        "last_progress_log_epoch": 0.0,
    }

    def _run():
        try:
            monitor["result"] = clear_all_caches()
        except Exception as exc:
            monitor["result"] = {
                "removed": [],
                "failed": [("(fatal)", str(exc))],
                "skipped_in_use": [],
            }

    thread = threading.Thread(target=_run)
    thread.daemon = True
    monitor["thread"] = thread
    thread.start()

    _mimics_log(logging.INFO, "Cache clearing started in background...")

    # Timer to poll for completion (non-blocking)
    _start_timer(_clear_cache_tick, monitor, poll_seconds=0.5)
    _CACHE_CLEAR_MONITORS.append(monitor)
    return 0


def _dispose_cache_monitor(monitor):
    """Stop timer handles and release monitor references after completion."""
    _stop_timer_handles(monitor)
    try:
        _CACHE_CLEAR_MONITORS.remove(monitor)
    except ValueError:
        pass


def _stop_timer_handles(monitor):
    """Stop any Qt/Win32 timer handles stored in monitor."""
    timer = monitor.get("qt_timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        try:
            user32, timer_id = win32_timer
            user32.KillTimer(None, timer_id)
        except Exception:
            pass


def _start_timer(tick_fn, monitor, poll_seconds=0.5):
    """Start a non-blocking timer that calls tick_fn(monitor) every poll_seconds.

    Tries PyQt5 QTimer → Win32 SetTimer → daemon thread (in that order).
    """
    # 1. PyQt5 QTimer (best — integrates with Mimics Qt event loop)
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.setSingleShot(False)

        def _tick():
            try:
                tick_fn(monitor)
            except Exception:
                pass

        timer.timeout.connect(_tick)
        timer.start(max(100, int(poll_seconds * 1000)))
        monitor["qt_timer"] = timer
        return
    except Exception:
        pass

    # 2. Win32 SetTimer (runs on the message pump thread)
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_uint, ctypes.c_ulong)

            def _timer_proc(hwnd, msg, timer_id, tick_count):
                try:
                    tick_fn(monitor)
                except Exception:
                    pass

            callback = TIMERPROC(_timer_proc)
            user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_uint, ctypes.c_void_p]
            user32.SetTimer.restype = ctypes.c_size_t
            user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            user32.KillTimer.restype = ctypes.c_int
            timer_id = user32.SetTimer(None, 0, int(poll_seconds * 1000), ctypes.cast(callback, ctypes.c_void_p))
            if timer_id:
                monitor["win32_timer"] = (user32, timer_id)
                monitor["win32_callback"] = callback
            return
        except Exception:
            pass

    # 3. Daemon thread (last resort — always works, slightly more CPU)
    def _thread_poll():
        while not monitor.get("done"):
            try:
                tick_fn(monitor)
            except Exception:
                pass
            time.sleep(poll_seconds)

    t = threading.Thread(target=_thread_poll)
    t.daemon = True
    t.start()


# -- Stop background processes ------------------------------------------

def stop_background_processes():
    if os.name != "nt":
        return False

    # 1. Write queue stop markers first (graceful shutdown signal)
    stopped_queues = _request_queue_stop()

    stop_log = os.path.join(_runtime_dir(), "stop_background_last.json")

    # 2. Build PowerShell command to forcefully kill remaining processes
    markers = "@(" + ",".join(
        "'{}'".format(marker.replace("'", "''")) for marker in MARKERS
    ) + ")"
    owned_roots = _owned_roots() + _queue_dirs_from_runtime_state()
    roots = "@(" + ",".join(
        "'{}'".format(root.replace("'", "''")) for root in owned_roots
    ) + ")"
    queues = "@(" + ",".join(
        "'{}'".format(path.replace("'", "''")) for path in stopped_queues
    ) + ")"
    lock_paths = [
        os.path.join(runtime_common.resource_lock_dir(_project_root()), "gpu.lock"),
        os.path.join(runtime_common.resource_lock_dir(_project_root()), "background_mimics.lock"),
    ]
    locks = "@(" + ",".join(
        "'{}'".format(path.replace("'", "''")) for path in lock_paths
    ) + ")"

    # PowerShell script: find and kill all owned processes
    command = (
        "$markers={0};"
        "$roots={1};"
        "$queues={2};"
        "$locks={3};"
        "$out='{4}';"
        # Find processes whose command line references both a root and a marker
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "  $cmd = $_.CommandLine;"
        "  if (-not $cmd) {{ return $false }};"
        "  $cmd = $cmd -replace '/', '\\';"
        "  $inRoot = ($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "  $hasMarker = ($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "  $inRoot -and $hasMarker"
        "}};"
        # Also find ANY process with markers regardless of root (broad sweep)
        "$broad=Get-CimInstance Win32_Process | Where-Object {{"
        "  $cmd = $_.CommandLine;"
        "  if (-not $cmd) {{ return $false }};"
        "  $cmd = $cmd -replace '/', '\\';"
        "  ($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "}};"
        # Merge and deduplicate by ProcessId
        "$all = @($matched) + @($broad) | Sort-Object ProcessId -Unique;"
        "$records = @($all | Select-Object ProcessId, Name, CommandLine);"
        "$killed = @();"
        "$all | ForEach-Object {{"
        "  $procId = $_.ProcessId;"
        "  taskkill /PID $procId /T /F 2>$null 1>$null;"
        "  $killed += [PSCustomObject]@{{"
        "    ProcessId = $procId;"
        "    Name = $_.Name;"
        "    ExitCode = $LASTEXITCODE;"
        "    CommandLine = $_.CommandLine"
        "  }};"
        "}};"
        "Start-Sleep -Milliseconds 500;"
        "$locks | ForEach-Object {{ Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }};"
        "$report = [PSCustomObject]@{{"
        "  RequestedAt = (Get-Date).ToString('s');"
        "  QueueStopDirs = $queues;"
        "  OwnedRoots = $roots;"
        "  Matched = $records;"
        "  Killed = $killed"
        "}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(markers, roots, queues, locks, stop_log.replace("'", "''"))

    process = None
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_hidden_process_kwargs()
        )
        process = True
    except Exception as exc:
        _mimics_log(logging.WARNING, "Could not launch PowerShell stop command: {0}".format(exc))
        # Queue stop markers were already written — the background processes
        # will pick them up on their own poll cycle.

    _mimics_log(
        logging.INFO,
        "Stop request submitted. Queue stop markers: {0}. Owned roots: {1}. Report: {2}".format(
            len(stopped_queues),
            len(owned_roots),
            stop_log,
        ),
    )
    return {
        "ok": True,
        "stop_log": stop_log,
        "stopped_queues": stopped_queues,
        "owned_roots": owned_roots,
        "launched": bool(process),
    }


def _stop_background_tick(monitor):
    if monitor.get("done"):
        return
    report_path = monitor.get("stop_log")
    if report_path and os.path.isfile(report_path):
        report = runtime_common.read_json(report_path, {}) or {}
        killed = report.get("Killed") or []
        matched = report.get("Matched") or []
        queue_dirs = report.get("QueueStopDirs") or []
        monitor["done"] = True
        _stop_timer_handles(monitor)
        key = monitor.get("monitor_key")
        if key in _STOP_MONITORS:
            del _STOP_MONITORS[key]
        lines = ["Stop Background Services completed."]
        lines.append("Matched process(es): {0}".format(len(matched)))
        lines.append("Kill request(s): {0}".format(len(killed)))
        lines.append("Queue stop marker dir(s): {0}".format(len(queue_dirs)))
        lines.append("Report: {0}".format(report_path))
        msg = "\n".join(lines)
        _mimics_log(logging.INFO, msg)
        try:
            mimics.dialogs.message_box(title="Stop Background Services", message=msg, ui_blocking=False)
        except TypeError:
            mimics.dialogs.message_box(title="Stop Background Services", message=msg)
        return
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_timer_handles(monitor)
        key = monitor.get("monitor_key")
        if key in _STOP_MONITORS:
            del _STOP_MONITORS[key]
        msg = (
            "Stop Background Services is still running or did not produce a report yet.\n\n"
            "Check: {0}"
        ).format(monitor.get("stop_log", "(unknown)"))
        _mimics_log(logging.WARNING, msg)
        try:
            mimics.dialogs.message_box(title="Stop Background Services", message=msg, ui_blocking=False)
        except TypeError:
            mimics.dialogs.message_box(title="Stop Background Services", message=msg)


def _start_stop_monitor(monitor, poll_seconds=0.5, timeout_seconds=90.0):
    monitor["done"] = False
    monitor["deadline"] = time.time() + float(timeout_seconds)
    key = monitor.get("monitor_key")
    _STOP_MONITORS[key] = monitor
    _start_timer(_stop_background_tick, monitor, poll_seconds=float(poll_seconds))
    return True


# -- Main entry points --------------------------------------------------

def main():
    result = stop_background_processes()
    ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
    if isinstance(result, dict) and result.get("stop_log"):
        _start_stop_monitor(
            {
                "monitor_key": "stop_bg_{0}".format(int(time.time() * 1000)),
                "stop_log": result.get("stop_log"),
            },
            poll_seconds=0.5,
            timeout_seconds=90.0,
        )
    try:
        mimics.dialogs.message_box(
            title="Stop Background Services",
            message=(
                "Stop request submitted for Mimics-Script background processes.\n"
                "A completion message will appear when finished."
                if ok else
                "Background cleanup is only implemented for Windows Mimics workstations."
            ),
            ui_blocking=False,
        )
    except TypeError:
        mimics.dialogs.message_box(
            title="Stop Background Services",
            message=(
                "Stop request submitted for Mimics-Script background processes."
                if ok else
                "Background cleanup is only implemented for Windows Mimics workstations."
            ),
        )
    return 0


if __name__ == "__main__":
    main()
