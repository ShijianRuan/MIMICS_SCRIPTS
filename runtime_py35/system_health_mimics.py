# -*- coding: utf-8 -*-
"""Open the system health overview panel (inside Mimics).

The panel is an external PySide6 window (tools/system_health_panel.py): it
aggregates the process registry, resource locks, import queues, and the
nnInteractive server state into one read-only page with explicit action
buttons. This launcher starts it as a registered external_ui process and
returns immediately.
"""

from __future__ import print_function

import logging
import os
import time

import mimics

import runtime_common


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


_HEALTH_PANEL_SCRIPT = "system_health_panel.py"


def _health_panel_script():
    root = _project_root()
    candidates = [
        os.path.join(root, "tools", _HEALTH_PANEL_SCRIPT),
        os.path.join(os.path.dirname(root), "tools", _HEALTH_PANEL_SCRIPT),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0])


def open_health_panel():
    """Start the health panel window. Returns 0 on success."""
    root = _project_root()
    python_exe = runtime_common.find_external_python(root, allow_system_python=False)
    if not python_exe or not os.path.isfile(python_exe):
        mimics.dialogs.message_box(
            "The external tools Python was not found. Run Admin > Setup/Repair "
            "Environment first.",
            title="System Health",
            ui_blocking=False,
        )
        return 1
    script = _health_panel_script()
    if not os.path.isfile(script):
        mimics.dialogs.message_box(
            "The health panel script was not found: {0}".format(script),
            title="System Health",
            ui_blocking=False,
        )
        return 1
    runtime_dir = os.path.join(root, ".mimics_runtime", "health_panel")
    if not os.path.isdir(runtime_dir):
        os.makedirs(runtime_dir)
    stderr_log = os.path.join(
        runtime_dir,
        "health_panel_{0}.log".format(time.strftime("%Y%m%dT%H%M%S")),
    )
    try:
        process = runtime_common.launch_external_gui_process(
            [python_exe, script], cwd=root, stderr_log=stderr_log
        )
    except Exception as exc:
        mimics.dialogs.message_box(
            "Could not start the health panel:\n{0}".format(exc),
            title="System Health",
            ui_blocking=False,
        )
        return 1
    runtime_common.register_process(
        root,
        "external_ui",
        process.pid,
        parent_pid=os.getpid(),
        state_path=stderr_log,
    )
    mimics.logging.log_user_message(
        level=logging.INFO,
        message=(
            "System health panel started in an external process (PID {0}).".format(
                process.pid
            )
        ),
    )
    return 0


def main():
    return open_health_panel()


if __name__ == "__main__":
    main()
