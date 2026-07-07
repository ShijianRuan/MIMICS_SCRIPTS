# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services.

Runs inside Mimics and submits a hidden PowerShell cleanup request without
waiting for completion, so the foreground Mimics GUI remains responsive.
"""

from __future__ import print_function

import os
import logging
import subprocess
import time

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


def _runtime_dir():
    path = os.path.join(_project_root(), ".mimics_runtime")
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def _queue_registry_dir():
    return os.path.join(_runtime_dir(), "mcs_queues")


def _queue_dirs_from_runtime_state():
    result = []
    lock_path = os.path.join(runtime_common.resource_lock_dir(_project_root()), "background_mimics.lock")
    for path in (lock_path,):
        payload = runtime_common.read_json(path, {}) or {}
        # output_dir / ts_root may be at top level (resource lock schema) or
        # nested under "details" (older background_mimics.lock schema).
        details = payload.get("details") or {}
        output_dir = payload.get("output_dir") or details.get("output_dir")
        if output_dir:
            result.append(output_dir)
        ts_root = payload.get("ts_root") or details.get("ts_root")
        if ts_root:
            result.append(ts_root)
            result.append(os.path.join(ts_root, "mcs_output"))
    registry = _queue_registry_dir()
    if os.path.isdir(registry):
        for name in os.listdir(registry):
            if not name.endswith(".json"):
                continue
            payload = runtime_common.read_json(os.path.join(registry, name), {}) or {}
            output_dir = payload.get("output_dir")
            if output_dir:
                result.append(output_dir)
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
            runtime_common.write_json_atomic(os.path.join(output_dir, "_mcs_queue_stop.json"), payload)
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


def stop_background_processes():
    if os.name != "nt":
        return False
    stopped_queues = _request_queue_stop()
    stop_log = os.path.join(_runtime_dir(), "stop_background_last.json")
    markers = "@(" + ",".join("'{}'".format(marker.replace("'", "''")) for marker in MARKERS) + ")"
    owned_roots = _owned_roots() + _queue_dirs_from_runtime_state()
    roots = "@(" + ",".join("'{}'".format(root.replace("'", "''")) for root in owned_roots) + ")"
    queues = "@(" + ",".join("'{}'".format(path.replace("'", "''")) for path in stopped_queues) + ")"
    lock_paths = [
        os.path.join(runtime_common.resource_lock_dir(_project_root()), "gpu.lock"),
        os.path.join(runtime_common.resource_lock_dir(_project_root()), "background_mimics.lock"),
    ]
    locks = "@(" + ",".join("'{}'".format(path.replace("'", "''")) for path in lock_paths) + ")"
    command = (
        "$markers={};"
        "$roots={};"
        "$queues={};"
        "$locks={};"
        "$out='{}';"
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; "
        "if ($cmd) {{ $cmd = $cmd -replace '/', '\\' }}; "
        "$cmd -and "
        "($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }}) -and "
        "($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }})"
        "}};"
        "$records=@($matched | Select-Object ProcessId,Name,CommandLine);"
        "$killed=@();"
        "$matched | ForEach-Object {{"
        "  $procId=$_.ProcessId;"
        "  taskkill /PID $procId /T /F 2>$null 1>$null;"
        "  $killed += [PSCustomObject]@{{ProcessId=$procId;Name=$_.Name;ExitCode=$LASTEXITCODE;CommandLine=$_.CommandLine}};"
        "}};"
        "Start-Sleep -Milliseconds 500;"
        "$locks | ForEach-Object {{ Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }};"
        "$report=[PSCustomObject]@{{RequestedAt=(Get-Date).ToString('s');QueueStopDirs=$queues;OwnedRoots=$roots;Matched=$records;Killed=$killed}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(markers, roots, queues, locks, stop_log.replace("'", "''"))
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_hidden_process_kwargs()
    )
    _mimics_log(
        logging.INFO,
        "Stop request submitted for Mimics-Script owned background processes only. Queue stop markers: {0}. Owned roots checked: {1}. The stop report will list matched/killed PIDs: {2}".format(
            len(stopped_queues),
            len(owned_roots),
            stop_log,
        ),
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
