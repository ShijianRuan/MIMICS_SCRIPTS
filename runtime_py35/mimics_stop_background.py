# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services.

Runs inside Mimics and submits a hidden PowerShell cleanup request without
waiting for completion, so the foreground Mimics GUI remains responsive.
"""

from __future__ import print_function

import os
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


def stop_background_processes():
    if os.name != "nt":
        return False
    markers = "@(" + ",".join("'{}'".format(marker.replace("'", "''")) for marker in MARKERS) + ")"
    command = (
        "$markers={};"
        "Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; $cmd -and ($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }})"
        "}} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
    ).format(markers)
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_hidden_process_kwargs()
    )
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
