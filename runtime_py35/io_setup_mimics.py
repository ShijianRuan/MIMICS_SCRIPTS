# -*- coding: utf-8 -*-
"""Launch external PySide6 I/O setup windows without blocking Mimics."""

from __future__ import print_function

import json
import logging
import os
import subprocess
import time
import uuid

import mimics

import runtime_common


_IO_SETUP_MONITORS = {}
# Track which monitors have already shown an alert, so a dying process does
# not trigger a stack of non-blocking message boxes ("endless popups").
_ALERTED = set()


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "nninteractive_env", ".git"),
    )


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _message(title, message):
    try:
        mimics.dialogs.message_box(title=title, message=message, ui_blocking=False)
    except Exception:
        pass


def _alert_once(key, title, message):
    """Show a non-blocking message at most once per monitor key.

    The external window is polled on a timer; if the child process dies while
    the user keeps clicking around Mimics, every tick would otherwise spawn
    another message box, producing the "endless popups" symptom.
    """
    if key in _ALERTED:
        return
    _ALERTED.add(key)
    _message(title, message)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _write_status_quick(path, value):
    """Atomically write UI hand-off state without blocking the GUI on retries."""
    try:
        runtime_common.write_json_atomic(path, value)
        return True
    except Exception:
        return False


def _read_stderr_tail(path, max_bytes=8192):
    """Read a bounded diagnostic tail without allowing logs to fill a dialog."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - int(max_bytes)), os.SEEK_SET)
            data = handle.read(int(max_bytes))
        return data.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _external_failure_message(base_message, monitor, reported_error=""):
    parts = [str(base_message).strip()]
    if reported_error:
        parts.append("Reported error:\n{0}".format(str(reported_error).strip()))
    stderr_log = monitor.get("stderr_log", "")
    stderr_tail = _read_stderr_tail(stderr_log)
    if stderr_tail:
        parts.append("Diagnostic output:\n{0}".format(stderr_tail))
    if stderr_log:
        parts.append("Diagnostic log: {0}".format(stderr_log))
    return "\n\n".join(item for item in parts if item)


def _report_external_failure(key, monitor, base_message, reported_error=""):
    message = _external_failure_message(base_message, monitor, reported_error)
    try:
        mimics.logging.log_user_message(level=logging.ERROR, message=message)
    except Exception:
        pass
    _alert_once(key, "Path Setup", message)


def _stop_monitor(key):
    monitor = _IO_SETUP_MONITORS.pop(key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        try:
            win32_timer[0].KillTimer(None, win32_timer[1])
        except Exception:
            pass


def _stop_all_monitors():
    """Tear down every active monitor and kill its child process.

    Without this, each call to launch() stacks another Win32 timer on Mimics'
    UI thread and another pythonw process; a few clicks in and there are half
    a dozen QApplication instances fighting for the message loop, which is the
    real cause of the "freezes, crashes, endless popups" symptom.
    """
    for key in list(_IO_SETUP_MONITORS.keys()):
        monitor = _IO_SETUP_MONITORS.get(key)
        if not monitor:
            continue
        process = monitor.get("process")
        if process is not None and process.poll() is None:
            runtime_common.terminate_process_async(process=process, graceful_seconds=2.0)
        _stop_monitor(key)
    _ALERTED.clear()


def _tick(monitor):
    # Guard against re-entry. _start_win32_monitor runs this on Mimics' UI
    # thread via a Win32 timer callback; a bare exception here would unwind
    # through ctypes and crash Mimics, so the whole body is guarded.
    if monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        key = monitor["key"]
        if time.time() > monitor["deadline"]:
            _stop_monitor(key)
            _report_external_failure(
                key,
                monitor,
                "The external path window timed out. No task was started.",
            )
            return
        status = _read_json(monitor["status_path"])
        if not status or status.get("status") in ("opening", "configuring"):
            process = monitor.get("process")
            if process is not None and process.poll() is not None:
                exit_code = process.poll()
                _stop_monitor(key)
                # Distinguish "user closed the window" (exit 0) from a crash.
                if exit_code == 0:
                    try:
                        mimics.logging.log_user_message(
                            level=logging.INFO,
                            message="Path selection was cancelled; no task was started.",
                        )
                    except Exception:
                        pass
                else:
                    _report_external_failure(
                        key,
                        monitor,
                        "The external path window exited unexpectedly (code {0}). No task was started.".format(exit_code),
                    )
            return
        state = status.get("status")
        if state in ("cancelled", "closed"):
            _stop_monitor(key)
            return
        if state == "failed":
            _stop_monitor(key)
            _report_external_failure(
                key,
                monitor,
                "The external path window failed. No task was started.",
                status.get("error", ""),
            )
            return
        if state != "submitted":
            return
        try:
            task = monitor["on_submit"](status.get("selection") or {})
            if not isinstance(task, dict):
                task = {}
            written = _write_status_quick(
                monitor["status_path"],
                {
                    "status": "launched",
                    "selection": status.get("selection") or {},
                    "task": task,
                    "updated_at_epoch": time.time(),
                },
            )
            if not written:
                try:
                    mimics.logging.log_user_message(
                        level=logging.WARNING,
                        message=(
                            "The task started, but the external progress window could not be attached "
                            "because its local hand-off status could not be written. Progress remains "
                            "available in the Mimics log."
                        ),
                    )
                except Exception:
                    pass
                process = monitor.get("process")
                if process is not None and process.poll() is None:
                    runtime_common.terminate_process_async(process=process, graceful_seconds=1.0)
            _stop_monitor(key)
            _ALERTED.discard(key)
        except Exception as exc:
            try:
                _write_status_quick(
                    monitor["status_path"],
                    {
                        "status": "failed",
                        "error": str(exc),
                        "updated_at_epoch": time.time(),
                    },
                )
            except Exception:
                pass
            _stop_monitor(key)
            _alert_once(key, "Could Not Start", str(exc))
    except Exception as exc:
        # Never let an exception escape into the Win32 timer callback.
        try:
            _stop_monitor(monitor.get("key"))
        except Exception:
            pass
        try:
            _report_external_failure(
                monitor.get("key", "io_setup"),
                monitor,
                "The external path-window monitor failed. No task was started.",
                str(exc),
            )
        except Exception:
            pass
    finally:
        monitor["busy"] = False


def _start_win32_monitor(monitor, poll_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        callback_type = ctypes.WINFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint
        )

        def callback(_hwnd, _message, _timer_id, _ticks):
            _tick(monitor)

        callback_ref = callback_type(callback)
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, callback_type]
        user32.SetTimer.restype = ctypes.c_size_t
        timer_id = user32.SetTimer(None, 0, max(250, int(poll_seconds * 1000)), callback_ref)
        if not timer_id:
            return False
        monitor["callback"] = callback_ref
        monitor["win32_timer"] = (user32, timer_id)
        return True
    except Exception:
        return False


def _start_monitor(monitor, poll_seconds=2.0):
    _IO_SETUP_MONITORS[monitor["key"]] = monitor
    if _start_win32_monitor(monitor, poll_seconds):
        return True
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: _tick(monitor))
        timer.start(max(1000, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        return True
    except Exception:
        _IO_SETUP_MONITORS.pop(monitor["key"], None)
        return False


def launch(mode, python_exe, context, on_submit, timeout_seconds=3600, ui_script=None):
    # Reuse is single-window: tear down any previous path window (and its
    # pythonw + Win32 timer) before starting a new one so they never pile up.
    _stop_all_monitors()
    root = _project_root()
    runtime_dir = os.path.join(root, ".mimics_runtime", "io_setup")
    if not os.path.isdir(runtime_dir):
        os.makedirs(runtime_dir)
    setup_id = "io_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    context_path = os.path.join(runtime_dir, setup_id + "_context.json")
    status_path = os.path.join(runtime_dir, setup_id + ".json")
    payload = dict(context or {})
    payload.update({
        "schema_version": "mimics_io_setup.v1",
        "mode": mode,
        "status_path": status_path,
        "state_path": os.path.join(root, ".mimics_runtime", "ui_state", "io_paths.json"),
    })
    runtime_common.write_json_atomic(context_path, payload)
    runtime_common.write_json_atomic(status_path, {
        "status": "opening", "mode": mode, "created_at_epoch": time.time()
    })
    script = ui_script or os.path.join(root, "tools", "io_path_setup_ui.py")
    if not os.path.isabs(script):
        script = os.path.join(root, script)
    if not os.path.isfile(script):
        raise RuntimeError("External path UI was not found: {0}".format(script))
    command = [python_exe, script, "--context", context_path]
    if os.name == "nt" and os.path.basename(str(command[0])).lower() == "python.exe":
        pythonw = os.path.join(os.path.dirname(command[0]), "pythonw.exe")
        if os.path.isfile(pythonw):
            command[0] = pythonw
    # Launch as a visible GUI process. background_process_kwargs() sets
    # CREATE_NO_WINDOW + SW_HIDE, which is meant for hidden background scripts
    # and would suppress the PySide window, leaving status stuck on "opening".
    env = runtime_common.background_env(include_itk=False)
    stderr_log = os.path.join(runtime_dir, setup_id + "_stderr.log")
    try:
        stderr_handle = open(stderr_log, "w", encoding="utf-8")
    except Exception:
        stderr_handle = None
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle if stderr_handle is not None else subprocess.DEVNULL,
            env=env,
        )
    finally:
        if stderr_handle is not None:
            try:
                stderr_handle.close()
            except Exception:
                pass
    monitor = {
        "key": setup_id,
        "status_path": status_path,
        "process": process,
        "stderr_log": stderr_log,
        "on_submit": on_submit,
        "deadline": time.time() + float(timeout_seconds),
        "busy": False,
    }
    if not _start_monitor(monitor):
        try:
            process.terminate()
        except Exception:
            pass
        raise RuntimeError("Mimics did not expose a timer API for the external path window.")
    _update_gui()
    return 0
