# -*- coding: utf-8 -*-
"""Collect a redacted diagnostics bundle for support (inside Mimics).

Runs tools/collect_diagnostics.py in the external Python and writes the zip
next to the project root, then tells the annotator where it is. Paths inside
the bundle are redacted to their last two components, so it is safe to share.
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


def collect_bundle():
    """Collect diagnostics and report the bundle path to the user."""
    root = _project_root()
    python_exe = runtime_common.find_external_python(root, allow_system_python=False)
    if not python_exe or not os.path.isfile(python_exe):
        mimics.dialogs.message_box(
            "The external tools Python was not found. Run Admin > Setup/Repair "
            "Environment first.",
            title="Collect Diagnostics",
            ui_blocking=False,
        )
        return 1
    script = os.path.join(root, "tools", "collect_diagnostics.py")
    if not os.path.isfile(script):
        mimics.dialogs.message_box(
            "The diagnostics tool was not found: {0}".format(script),
            title="Collect Diagnostics",
            ui_blocking=False,
        )
        return 1
    output = os.path.join(
        root, "diagnostics_{0}.zip".format(time.strftime("%Y%m%dT%H%M%S"))
    )
    import subprocess
    try:
        result = subprocess.run(
            [python_exe, script, "--output", output],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    except Exception as exc:
        mimics.dialogs.message_box(
            "Could not collect diagnostics:\n{0}".format(exc),
            title="Collect Diagnostics",
            ui_blocking=False,
        )
        return 1
    if result.returncode != 0 or not os.path.isfile(output):
        mimics.dialogs.message_box(
            "Diagnostics collection failed:\n{0}".format(
                result.stdout.decode("utf-8", errors="replace")[-800:]
            ),
            title="Collect Diagnostics",
            ui_blocking=False,
        )
        return 1
    mimics.dialogs.message_box(
        "Diagnostics bundle written to:\n{0}\n\n"
        "Paths and tokens inside are redacted; it is safe to share.".format(
            output
        ),
        title="Collect Diagnostics",
        ui_blocking=False,
    )
    mimics.logging.log_user_message(
        level=logging.INFO,
        message="Diagnostics bundle collected: {0}".format(output),
    )
    return 0


def main():
    return collect_bundle()


if __name__ == "__main__":
    main()
