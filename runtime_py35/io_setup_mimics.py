# -*- coding: utf-8 -*-
"""Launch external PySide6 I/O setup windows without blocking Mimics."""

from __future__ import print_function

import json
import os
import subprocess
import time
import uuid

import mimics

import runtime_common


_IO_SETUP_MONITORS = {}


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


def _read_json(path):
    try:
        with open(path, "r") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


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


def _tick(monitor):
    if monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        key = monitor["key"]
        if time.time() > monitor["deadline"]:
            _stop_monitor(key)
            _message("Path Setup", "The external path window timed out. No task was started.")
            return
        status = _read_json(monitor["status_path"])
        if not status or status.get("status") in ("opening", "configuring"):
            process = monitor.get("process")
            if process is not None and process.poll() is not None:
                _stop_monitor(key)
                _message("Path Setup", "The external path window closed unexpectedly. No task was started.")
            return
        state = status.get("status")
        if state in ("cancelled", "closed"):
            _stop_monitor(key)
            return
        if state == "failed":
            _stop_monitor(key)
            _message("Path Setup", status.get("error", "The external path window failed."))
            return
        if state != "submitted":
            return
        _stop_monitor(key)
        try:
            monitor["on_submit"](status.get("selection") or {})
        except Exception as exc:
            _message("Could Not Start", str(exc))
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


def _start_monitor(monitor, poll_seconds=0.5):
    _IO_SETUP_MONITORS[monitor["key"]] = monitor
    if _start_win32_monitor(monitor, poll_seconds):
        return True
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: _tick(monitor))
        timer.start(max(250, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        return True
    except Exception:
        _IO_SETUP_MONITORS.pop(monitor["key"], None)
        return False


def launch(mode, python_exe, context, on_submit, timeout_seconds=3600):
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
    command = [python_exe, os.path.join(root, "tools", "io_path_setup_ui.py"), "--context", context_path]
    if os.name == "nt" and os.path.basename(str(command[0])).lower() == "python.exe":
        pythonw = os.path.join(os.path.dirname(command[0]), "pythonw.exe")
        if os.path.isfile(pythonw):
            command[0] = pythonw
    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=runtime_common.background_env(include_itk=False),
        **runtime_common.background_process_kwargs()
    )
    monitor = {
        "key": setup_id,
        "status_path": status_path,
        "process": process,
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
