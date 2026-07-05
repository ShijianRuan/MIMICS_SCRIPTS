# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services.

Runs inside Mimics and submits a hidden PowerShell cleanup request without
waiting for completion, so the foreground Mimics GUI remains responsive.
"""

from __future__ import print_function

import os
import logging
import subprocess

import mimics

import runtime_common


MARKERS = (
    "mimics_bridge.py",
    "nninteractive_bridge.py",
    "--async-worker",
    "_run_create_mcs.py",
    "_run_export_batch.py",
    "fewshot_pipeline.py",
    "nninteractive.inference.server.main",
    "--watchdog",
)


_hidden_process_kwargs = runtime_common.hidden_process_kwargs


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "fewshot_config.json", "mimics_bridge.py", ".git"),
    )


def _owned_roots():
    root = _project_root()
    result = [root]
    for child in ("nninteractive_env", "external", "tools", "runtime_py35"):
        path = os.path.join(root, child)
        if os.path.exists(path):
            result.append(path)
    return [os.path.abspath(path) for path in result if path]


def _clear_resource_locks():
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    for name in ("gpu.lock", "background_mimics.lock"):
        try:
            os.remove(os.path.join(lock_dir, name))
        except OSError:
            pass


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def stop_background_processes():
    if os.name != "nt":
        return False
    markers = "@(" + ",".join("'{}'".format(marker.replace("'", "''")) for marker in MARKERS) + ")"
    roots = "@(" + ",".join("'{}'".format(root.replace("'", "''")) for root in _owned_roots()) + ")"
    command = (
        "$markers={};"
        "$roots={};"
        "Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; "
        "$cmd -and "
        "($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }}) -and "
        "($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }})"
        "}} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
    ).format(markers, roots)
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_hidden_process_kwargs()
    )
    _clear_resource_locks()
    _mimics_log(logging.INFO, "Stop request submitted for Mimics-Script owned background processes only.")
    return True


def main():
    ok = stop_background_processes()
    try:
        mimics.dialogs.message_box(
            title="Stop Background Services",
            message=(
                "Stop request submitted for Mimics-Script background processes."
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
