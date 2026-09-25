# -*- coding: utf-8 -*-
"""Collect a redacted diagnostics bundle for support (inside Mimics).

Runs tools/collect_diagnostics.py in the external Python and writes the zip
next to the project root, then tells the annotator where it is. Paths inside
the bundle are redacted to their last two components, so it is safe to share.

The collection itself runs in a daemon thread with a timer polling for
completion (the same pattern as mimics_stop_background.clear_cache_main):
collecting a bundle can take minutes, and blocking the Mimics GUI thread
while the annotator is already troubleshooting is unacceptable.
"""

from __future__ import print_function

import logging
import os
import threading
import time

import mimics

import runtime_common

# One collection at a time; a second click joins the first run instead of
# stacking zips.
_ACTIVE_MONITOR = None


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _message_box(message):
    try:
        mimics.dialogs.message_box(
            message,
            title="Collect Diagnostics",
            ui_blocking=False,
        )
    except Exception:
        pass


def _collect_in_background(python_exe, script, output, root, monitor):
    """Worker thread body: run the external collector, record the outcome."""
    import subprocess

    try:
        result = subprocess.run(
            [python_exe, script, "--output", output],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        monitor["result"] = {
            "returncode": result.returncode,
            "output": output,
            "tail": result.stdout.decode("utf-8", errors="replace")[-800:],
        }
    except Exception as exc:
        monitor["result"] = {"returncode": -1, "output": output, "tail": str(exc)}


def _diagnostics_tick(monitor):
    """Timer tick: report and stop polling once the worker finishes."""
    global _ACTIVE_MONITOR
    if not monitor.get("done"):
        return
    _stop_timer(monitor)
    if _ACTIVE_MONITOR is monitor:
        _ACTIVE_MONITOR = None
    result = monitor.get("result") or {}
    output = result.get("output")
    if result.get("returncode") == 0 and output and os.path.isfile(output):
        _message_box(
            "Diagnostics bundle written to:\n{0}\n\n"
            "Paths and tokens inside are redacted; it is safe to share.".format(
                output
            )
        )
        _mimics_log(
            logging.INFO,
            "Diagnostics bundle collected: {0}".format(output),
        )
    else:
        _message_box(
            "Diagnostics collection failed:\n{0}".format(result.get("tail") or "")
        )
        _mimics_log(
            logging.WARNING,
            "Diagnostics collection failed: {0}".format(
                result.get("tail") or "unknown error"
            ),
        )


def _start_timer(tick_fn, monitor, poll_seconds=0.5):
    """Non-blocking timer: Win32 SetTimer, then Qt, then a daemon thread.

    Mirrors mimics_stop_background._start_timer so the Mimics message pump
    drives the polling instead of a second GUI binding.
    """
    if os.name == "nt":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(
                None, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint, ctypes.c_ulong,
            )

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
            timer_id = user32.SetTimer(
                None, 0, int(poll_seconds * 1000),
                ctypes.cast(callback, ctypes.c_void_p),
            )
            if timer_id:
                monitor["win32_timer"] = (user32, timer_id)
                monitor["win32_callback"] = callback
                return
        except Exception:
            pass

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

    def _thread_poll():
        while not monitor.get("done"):
            try:
                tick_fn(monitor)
            except Exception:
                pass
            time.sleep(poll_seconds)

    thread = threading.Thread(target=_thread_poll)
    thread.daemon = True
    thread.start()


def _stop_timer(monitor):
    """Release whichever timer handle _start_timer installed."""
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        try:
            user32, timer_id = win32_timer
            user32.KillTimer(None, timer_id)
        except Exception:
            pass
    timer = monitor.get("qt_timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass


def collect_bundle():
    """Start diagnostics collection in the background and return at once."""
    global _ACTIVE_MONITOR

    if _ACTIVE_MONITOR is not None and not _ACTIVE_MONITOR.get("done"):
        _message_box("Diagnostics collection is already running.")
        return 0

    root = _project_root()
    python_exe = runtime_common.find_external_python(root, allow_system_python=False)
    if not python_exe or not os.path.isfile(python_exe):
        _message_box(
            "The external tools Python was not found. Run Admin > Setup/Repair "
            "Environment first."
        )
        return 1
    script = os.path.join(root, "tools", "collect_diagnostics.py")
    if not os.path.isfile(script):
        _message_box("The diagnostics tool was not found: {0}".format(script))
        return 1
    output = os.path.join(
        root, "diagnostics_{0}.zip".format(time.strftime("%Y%m%dT%H%M%S"))
    )

    monitor = {
        "thread": None,
        "result": None,
        "done": False,
        "started_at_epoch": time.time(),
    }

    def _run():
        _collect_in_background(python_exe, script, output, root, monitor)

    thread = threading.Thread(target=_run)
    thread.daemon = True
    monitor["thread"] = thread
    thread.start()

    _mimics_log(
        logging.INFO,
        "Diagnostics collection started in the background: {0}".format(output),
    )
    _message_box(
        "Collecting diagnostics in the background...\n"
        "You can keep working; a message will appear when the bundle is "
        "ready at:\n{0}".format(output)
    )

    def _tick_when_done(current):
        current["done"] = current.get("thread") is None or not current[
            "thread"
        ].is_alive()
        _diagnostics_tick(current)

    _start_timer(_tick_when_done, monitor, poll_seconds=0.5)
    _ACTIVE_MONITOR = monitor
    return 0


def main():
    return collect_bundle()


if __name__ == "__main__":
    main()
