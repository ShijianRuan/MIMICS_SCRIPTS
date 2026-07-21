# -*- coding: utf-8 -*-
"""Mimics-side orchestrator for environment setup.

Runs inside Mimics Python 3.5.2. Launches the external setup worker
(Python 3.10+) in the background and polls its state file using a
Win32 timer or PyQt5 QTimer — Mimics GUI stays responsive.

Flow:
    1. User picks an action (check / install / setup-from-scratch)
    2. External setup_env.py is launched as a hidden subprocess
    3. Timer ticks every 2s, reading .mimics_runtime/setup_env_state.json
    4. When state transitions to "ok" / "error", dialog shows the result
"""

from __future__ import print_function

import json
import logging
import os
import subprocess
import sys
import threading
import time

import mimics

import runtime_common


_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_env = runtime_common.background_env
_find_root = runtime_common.find_root
_read_json = runtime_common.read_json

TITLE = "Setup Environment"
_MONITORS = {}


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "fewshot_config.json", "mimics_bridge.py", ".git"),
    )


def _find_external_python():
    root = _project_root()
    for rel in (
        "nninteractive_env/python.exe",
        "nninteractive_env/Scripts/python.exe",
        "nninteractive_env/python/python.exe",
        "python/python.exe",
        "nninteractive_env/bin/python3",
        "nninteractive_env/bin/python",
    ):
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            return path
    # Try system Python before giving up.
    try:
        import shutil
        for cmd in ("python3", "python"):
            found = shutil.which(cmd)
            if found:
                return found
    except Exception:
        pass
    # Do NOT fall back to sys.executable here. Inside Mimics this may point to
    # MimicsResearch.exe, which would start a new Mimics instance.
    return ""


def _state_file():
    return os.path.join(_project_root(), ".mimics_runtime", "setup_env_state.json")


def _setup_script():
    return os.path.join(_project_root(), "tools", "setup_env.py")


def _log_file():
    return os.path.join(_project_root(), ".mimics_runtime", "setup_env.log")


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


# -- Background launch ---------------------------------------------------

def _find_portable_archive():
    """Search common locations for the portable archive."""
    root = _project_root()
    candidates = [
        os.path.join(os.path.dirname(root), "mimics_script_portable.zip"),
        os.path.join(root, "mimics_script_portable.zip"),
        os.path.join(os.path.expanduser("~"), "Desktop", "mimics_script_portable.zip"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def _is_offline_bundle():
    """Return True if this looks like an offline bundle directory."""
    root = _project_root()
    return (
        os.path.isdir(os.path.join(root, "wheels"))
        and os.path.isdir(os.path.join(root, "python"))
    )


def _launch_setup_worker(action, extra_arg=None):
    """Launch the external setup_env.py in the background."""
    python_exe = _find_external_python()
    script = _setup_script()
    state_path = _state_file()

    if not python_exe:
        _mimics_log(logging.ERROR, "No external Python interpreter was found for setup worker.")
        return None

    # Write initial state so the timer has something to read
    try:
        state_dir = os.path.dirname(state_path)
        if state_dir and not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        with open(state_path, "w") as f:
            json.dump({
                "status": "launching",
                "message": "Starting setup worker...",
                "updated_at_epoch": time.time(),
            }, f, indent=2, sort_keys=True)
    except Exception as exc:
        _mimics_log(logging.WARNING, "Could not write initial state file: {0}".format(exc))
        # Continue anyway — the worker may still complete and write its own state

    cmd = [python_exe, script, action]
    if extra_arg:
        cmd.append(extra_arg)

    log_handle = open(_log_file(), "ab")
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_hidden_process_kwargs()
        )
        return process
    except Exception as exc:
        _mimics_log(logging.ERROR, "Could not start setup worker: {0}".format(exc))
        return None
    finally:
        log_handle.close()


# -- Timer-based monitor -------------------------------------------------

def _poll_setup_state(monitor):
    """Called by the timer. Check if the external worker has finished."""
    if monitor.get("done"):
        return

    state_path = monitor.get("state_path")
    state = _read_json(state_path, {}) or {}
    status = state.get("status", "")
    terminal = ("ok", "error", "incomplete", "extracted")

    # If the worker already wrote a terminal state, handle it first.
    # This avoids false "worker stopped" warnings for short-lived jobs.
    if status in terminal:
        monitor["done"] = True
        _stop_monitor(monitor)
        _show_result(monitor, state)
        return

    # If the subprocess was killed (e.g. by Stop_Background_Services),
    # detect it and clean up rather than polling forever.
    pid = monitor.get("pid")
    if pid and not runtime_common.process_exists(int(pid)):
        elapsed = time.time() - monitor.get("started_at", time.time())
        # Only conclude "killed" after a short grace period — the process
        # may have legitimately exited after writing its final state.
        if elapsed > 3.0:
            monitor["done"] = True
            _stop_monitor(monitor)
            _mimics_log(
                logging.WARNING,
                "[Setup] Worker process (PID={0}) is no longer running. It may have been stopped.".format(pid),
            )
            try:
                mimics.dialogs.message_box(
                    title=TITLE,
                    message="Setup worker was stopped.\n\n"
                            "The background process (PID={0}) is no longer running.\n"
                            "Check {1} for details.".format(
                                pid, _log_file()),
                    ui_blocking=False,
                )
            except Exception:
                pass
            return
    message = state.get("message", "")

    # Show progress whenever message changes, even if status stays the same.
    if message and message != monitor.get("_last_message", ""):
        monitor["_last_message"] = message
        _mimics_log(logging.INFO, "[Setup] {0}".format(message))

    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_monitor(monitor)
        runtime_common.terminate_process_async(
            process=monitor.get("process"),
            pid=monitor.get("pid"),
            graceful_seconds=5.0,
        )
        _mimics_log(logging.WARNING, "[Setup] Worker timed out.")
        try:
            mimics.dialogs.message_box(
                title=TITLE,
                message="Setup timed out after {0}s.\n\n"
                        "The worker may still be running. Check the log:\n{1}".format(
                            int(monitor.get("timeout_seconds", 1800)), _log_file()),
                ui_blocking=False,
            )
        except Exception:
            pass
        return

    monitor["_last_check_time"] = time.time()

    if status == monitor.get("_last_status", ""):
        return
    monitor["_last_status"] = status

    if status in terminal:
        monitor["done"] = True
        _stop_monitor(monitor)
        _show_result(monitor, state)


def _show_result(monitor, state):
    """Display the final result in a Mimics dialog."""
    status = state.get("status", "error")
    message = state.get("message", "Unknown result")
    all_ok = state.get("all_ok", False)
    detail = state.get("detail", {})
    missing = state.get("missing_packages", [])
    error = state.get("error", "")
    action = monitor.get("action", "check")
    _mimics_log(logging.INFO, "[Setup] Completed action={0}, status={1}.".format(action, status))

    lines = []
    if status == "ok":
        lines.append("Environment is ready.")
    elif status == "incomplete":
        lines.append("Environment setup incomplete.")
    else:
        lines.append("Setup encountered an error.")

    if message:
        lines.append(message)

    if missing:
        lines.append("\nMissing packages ({0}):".format(len(missing)))
        lines.append(", ".join(missing[:10]))

    if error:
        lines.append("\nError detail: {0}".format(error[:300]))

    if detail:
        pkg_status = detail.get("packages", {})
        if pkg_status:
            lines.append("\nPackage status:")
            for pkg, ok in sorted(pkg_status.items()):
                lines.append("  {0}: {1}".format(pkg, "OK" if ok else "MISSING"))
        cuda = detail.get("cuda_available", False)
        devices = detail.get("cuda_device_count", 0)
        lines.append("\nCUDA: {0} ({1} device(s))".format(
            "Available" if cuda else "Not available", devices))

    lines.append("\nFull log: {0}".format(_log_file()))

    msg = "\n".join(lines)
    try:
        mimics.dialogs.message_box(
            title=TITLE,
            message=msg,
            ui_blocking=True if action == "check" else False,
        )
    except TypeError:
        mimics.dialogs.message_box(
            title=TITLE,
            message=msg,
        )


# -- Monitor management (matching mimics_import.py pattern) -------------

def _stop_monitor(monitor):
    key = monitor.get("monitor_key")
    if key and key in _MONITORS:
        del _MONITORS[key]
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


def _start_monitor(monitor, poll_seconds=2.0, timeout_seconds=1800):
    """Start a timer to poll the external worker."""
    monitor.setdefault("deadline", time.time() + timeout_seconds)
    monitor.setdefault("_last_status", "")
    monitor.setdefault("_last_message", "")
    monitor.setdefault("timeout_seconds", timeout_seconds)
    monitor["done"] = False
    key = monitor.get("monitor_key")
    _MONITORS[key] = monitor

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.setSingleShot(False)

        def _tick():
            try:
                _poll_setup_state(monitor)
            except Exception as exc:
                _mimics_log(logging.WARNING, "[Setup] Monitor tick failed: {0}".format(exc))

        timer.timeout.connect(_tick)
        timer.start(max(100, int(poll_seconds * 1000)))
        monitor["qt_timer"] = timer
        return True
    except Exception:
        pass

    # Fallback: Win32 SetTimer
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_uint, ctypes.c_ulong)

            def _timer_proc(hwnd, msg, timer_id, tick_count):
                try:
                    _poll_setup_state(monitor)
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
            return True
        except Exception:
            pass

    # Last resort: daemon thread
    def _thread_poll():
        while not monitor.get("done"):
            try:
                _poll_setup_state(monitor)
            except Exception:
                pass
            time.sleep(poll_seconds)

    t = threading.Thread(target=_thread_poll)
    t.daemon = True
    t.start()
    return True


# -- Main entry ---------------------------------------------------------

def main(action=None):
    """Launch the setup flow.

    action: "check" / "install" / "extract" / "setup-from-scratch" / None (ask user)
    """
    active = [item for item in _MONITORS.values() if item and not item.get("done")]
    if active:
        current = active[0]
        answer = mimics.dialogs.question_box(
            title=TITLE,
            message=(
                "Environment setup is already running.\n\n"
                "Action: {0}\nPID: {1}\n\nKeep it running, or stop this setup process."
            ).format(current.get("action", "setup"), current.get("pid", "?")),
            buttons="Keep Running;Stop Current Setup",
            ui_blocking=True,
        )
        if answer == "Stop Current Setup":
            current["done"] = True
            _stop_monitor(current)
            runtime_common.terminate_process_async(
                process=current.get("process"),
                pid=current.get("pid"),
                graceful_seconds=5.0,
            )
            _mimics_log(logging.INFO, "[Setup] Stop requested; environment worker is shutting down.")
        return 0

    if not action:
        # Build the menu dynamically — show Offline Install only if bundle is detected
        if _is_offline_bundle():
            buttons = "Offline Install;Check;Repair (Install Missing);Setup From Scratch;Cancel"
            message = (
                "Offline Install — install everything from the offline bundle (no internet needed).\n"
                "Check — validate Python, CUDA, packages and models.\n"
                "Repair — check and install missing packages.\n"
                "Setup From Scratch — create a new environment from scratch."
            )
        else:
            buttons = "Extract Archive;Check;Repair (Install Missing);Setup From Scratch;Cancel"
            message = (
                "Extract Archive — extract a portable .zip archive.\n"
                "Check — validate Python, CUDA, packages and models.\n"
                "Repair — check and install missing packages.\n"
                "Setup From Scratch — create a new environment from scratch."
            )
        answer = mimics.dialogs.question_box(
            title=TITLE,
            message=message,
            buttons=buttons,
            ui_blocking=True,
        )
        if answer == "Extract Archive":
            action = "extract"
        elif answer == "Offline Install":
            action = "offline-install"
        elif answer == "Check":
            action = "check"
        elif answer == "Repair (Install Missing)":
            action = "install"
        elif answer == "Setup From Scratch":
            action = "setup-from-scratch"
        else:
            return 1

    if action not in ("check", "install", "extract", "setup-from-scratch", "offline-install"):
        _mimics_log(logging.WARNING, "Unknown action: {0}".format(action))
        return 1

    # For extract, find the archive file
    extra_args = None
    if action == "extract":
        archive_path = _find_portable_archive()
        if not archive_path:
            mimics.dialogs.message_box(
                title=TITLE,
                message=(
                    "Portable archive not found.\n\n"
                    "Place mimcs_script_portable.zip in one of these locations:\n"
                    "  {0}\n"
                    "  {1}\n"
                    "  Your Desktop".format(
                        os.path.abspath(os.path.join(_project_root(), "..")),
                        os.path.abspath(_project_root()),
                    )
                ),
                ui_blocking=True,
            )
            return 1
        extra_args = archive_path

    # For offline-install, check that the bundle is present
    if action == "offline-install":
        if not _is_offline_bundle():
            mimics.dialogs.message_box(
                title=TITLE,
                message=(
                    "Offline bundle not found.\n\n"
                    "This requires the offline bundle with python/ and wheels/ directories.\n"
                    "Run: python tools/package_portable.py offline-bundle\n"
                    "to create the bundle first."
                ),
                ui_blocking=True,
            )
            return 1

    _mimics_log(logging.INFO, "Starting environment setup: {0}".format(action))
    process = _launch_setup_worker(action, extra_args)
    if not process:
        _mimics_log(logging.ERROR, "Failed to launch setup worker.")
        try:
            mimics.dialogs.message_box(
                title=TITLE,
                message="Could not start the setup worker.\n\n"
                        "Check that a Python interpreter is available at:\n{0}".format(
                            _find_external_python()),
            )
        except Exception:
            pass
        return 1

    monitor = {
        "monitor_key": "setup_env_" + action,
        "state_path": _state_file(),
        "pid": process.pid,
        "process": process,
        "action": action,
        "started_at": time.time(),
    }

    _start_monitor(monitor, poll_seconds=2.0, timeout_seconds=1800)
    _mimics_log(
        logging.INFO,
        "Setup worker PID={0} running in background. Results will appear when complete.".format(
            process.pid),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
