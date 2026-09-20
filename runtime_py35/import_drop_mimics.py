# -*- coding: utf-8 -*-
"""Open the floating drop-to-import window (inside Mimics).

Mimics' scripting API cannot receive native drag events, so the drop zone
lives in an external always-on-top PySide6 window (tools/import_drop_window.py).
This launcher starts that window as a registered external_ui process and
returns immediately; the window is fully self-contained: it recognizes the
dropped payload against the dataset profile, confirms with the user, and
submits to the same import workers the path-setup UI uses. It exits on its
own after an idle timeout, so there is no resident service to manage.
"""

from __future__ import print_function

import logging
import os
import subprocess
import time

import mimics

import runtime_common


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


_DROP_WINDOW_SCRIPT = "import_drop_window.py"


def _drop_window_script():
    root = _project_root()
    candidates = [
        os.path.join(root, "tools", _DROP_WINDOW_SCRIPT),
        os.path.join(os.path.dirname(root), "tools", _DROP_WINDOW_SCRIPT),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0])


def _launch_gui_process(cmd, cwd=None, stderr_log=None):
    # Visible GUI: no CREATE_NO_WINDOW, prefer pythonw.exe on Windows.
    launch_cmd = list(cmd)
    if os.name == "nt" and launch_cmd:
        exe = os.path.abspath(str(launch_cmd[0]))
        if os.path.basename(exe).lower() == "python.exe":
            pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            if os.path.isfile(pythonw):
                launch_cmd[0] = pythonw
    env = runtime_common.background_env()
    stderr_dest = subprocess.DEVNULL
    stderr_file_handle = None
    if stderr_log:
        try:
            parent = os.path.dirname(os.path.abspath(stderr_log))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            stderr_file_handle = open(stderr_log, "w", encoding="utf-8")
            stderr_dest = stderr_file_handle
        except Exception:
            stderr_dest = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            launch_cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_dest,
            env=env,
        )
    finally:
        if stderr_file_handle is not None:
            try:
                stderr_file_handle.close()
            except Exception:
                pass
    return proc


def open_drop_window():
    """Start the drop-to-import window. Returns 0 on success."""
    root = _project_root()
    python_exe = runtime_common.find_external_python(root, allow_system_python=False)
    if not python_exe or not os.path.isfile(python_exe):
        mimics.dialogs.message_box(
            "The external tools Python was not found. Run Admin > Setup/Repair "
            "Environment first.",
            title="Drop Import",
            ui_blocking=False,
        )
        return 1
    script = _drop_window_script()
    if not os.path.isfile(script):
        mimics.dialogs.message_box(
            "The drop window script was not found: {0}".format(script),
            title="Drop Import",
            ui_blocking=False,
        )
        return 1
    runtime_dir = os.path.join(root, ".mimics_runtime", "drop_import")
    if not os.path.isdir(runtime_dir):
        os.makedirs(runtime_dir)
    stderr_log = os.path.join(
        runtime_dir, "drop_window_{0}.log".format(time.strftime("%Y%m%dT%H%M%S"))
    )
    try:
        process = _launch_gui_process(
            [python_exe, script], cwd=root, stderr_log=stderr_log
        )
    except Exception as exc:
        mimics.dialogs.message_box(
            "Could not start the drop window:\n{0}".format(exc),
            title="Drop Import",
            ui_blocking=False,
        )
        return 1
    runtime_common.register_process(
        root,
        "external_ui",
        process.pid,
        parent_pid=os.getpid(),
        cleanup_policy="idle_timeout_s:1800",
    )
    mimics.logging.log_user_message(
        level=logging.INFO,
        message=(
            "Drop-to-import window started in an external process (PID {0}). "
            "It closes automatically when idle.".format(process.pid)
        ),
    )
    return 0


def main():
    return open_drop_window()


if __name__ == "__main__":
    main()
