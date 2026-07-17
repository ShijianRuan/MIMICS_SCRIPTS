# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services.

Runs inside Mimics and submits a hidden PowerShell cleanup request without
waiting for completion, so the foreground Mimics GUI remains responsive.
"""

from __future__ import print_function

import glob
import os
import logging
import subprocess
import sys
import threading
import time

import mimics

import runtime_common


# Command-line substrings that identify Mimics-Script owned processes.
# These are matched against the full command line of every running process.
MARKERS = (
    # Bridge processes (external Python)
    "mimics_bridge.py",
    "nninteractive_bridge.py",
    "--async-worker",
    "--watchdog",

    # Auto-generated runner scripts (run inside background Mimics)
    "_run_create_mcs.py",
    "_run_export_batch.py",

    # Actual runtime module names (found via -run_script or import)
    "create_mcs_batch.py",
    "mimics_export.py",
    "mimics_import.py",

    # DINOv3 few-shot training/inference
    "fewshot_pipeline.py",
    "fewshot_mimics.py",
    "fewshot_model_chooser.py",
    "fewshot_training_setup_ui.py",
    "fewshot_status_viewer.py",
    "io_path_setup_ui.py",
    "mask_file_picker_ui.py",

    # nnInteractive inference server
    "nninteractive.inference.server.main",

    # Environment setup / packaging tools
    "setup_env.py",
    "package_portable.py",
)


_hidden_process_kwargs = runtime_common.hidden_process_kwargs

# Keep async timer/callback references alive until a monitor finishes.
_CACHE_CLEAR_MONITORS = []
_STOP_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "fewshot_config.json", "mimics_bridge.py", ".git"),
    )


def _owned_roots():
    """Return all directory roots that may appear in process command lines."""
    root = _project_root()
    result = [root]
    for child in ("nninteractive_env", "external", "tools", "runtime_py35"):
        path = os.path.join(root, child)
        if os.path.exists(path):
            result.append(path)
    return [os.path.abspath(p) for p in result if p]


def _clear_resource_locks():
    """Remove only locks whose recorded owner process is no longer alive."""
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    return runtime_common.cleanup_stale_resource_locks(lock_dir)


def _stop_inprocess_monitors():
    """Detach Mimics timers before their owned children are terminated."""
    stopped = 0
    module = sys.modules.get("io_setup_mimics")
    if module is not None:
        try:
            module._stop_all_monitors()
            stopped += 1
        except Exception:
            pass
    for module_name, collection_name, stop_name in (
        ("mimics_import", "_IMPORT_MONITORS", "_stop_import_monitor"),
        ("mimics_export", "_EXPORT_MONITORS", "_stop_export_monitor"),
        ("mask_import", "_MASK_IMPORT_MONITORS", "_stop_mask_import_monitor"),
        ("fix_source_affine_metadata", "_MONITORS", "_stop_monitor"),
        ("fewshot_mimics", "_MONITORS", "_stop_monitor"),
        ("nninteractive_mimics", "_ASYNC_MONITORS", "_stop_async_monitor"),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        collection = getattr(module, collection_name, {}) or {}
        stopper = getattr(module, stop_name, None)
        if stopper is None:
            continue
        for key in list(collection.keys()):
            try:
                stopper(key)
                stopped += 1
            except Exception:
                pass
    fewshot = sys.modules.get("fewshot_mimics")
    if fewshot is not None:
        for process in list(getattr(fewshot, "_GUI_PROCESSES", {}).values()):
            try:
                runtime_common.terminate_process_async(process=process, graceful_seconds=2.0)
            except Exception:
                pass
        try:
            fewshot._GUI_PROCESSES.clear()
        except Exception:
            pass
    nninteractive = sys.modules.get("nninteractive_mimics")
    if nninteractive is not None:
        try:
            nninteractive._close_all_async_image_workers()
        except Exception:
            pass
    return stopped


def _stop_import_inprocess_monitors():
    """Stop import bridge work and detach its callbacks before queue cleanup."""
    module = sys.modules.get("mimics_import")
    if module is None:
        return []
    output_dirs = []
    monitors = getattr(module, "_IMPORT_MONITORS", {}) or {}
    stopper = getattr(module, "_stop_import_monitor", None)
    terminator = getattr(module, "_terminate_job_process", None)
    cleanup_work = getattr(module, "_cleanup_work_dir", None)
    cancel_monitor = getattr(module, "_cancel_import_monitor", None)
    for key, monitor in list(monitors.items()):
        try:
            output_dir = monitor.get("output_dir")
            if not output_dir and monitor.get("output_mcs"):
                output_dir = os.path.dirname(os.path.abspath(monitor.get("output_mcs")))
            if output_dir:
                output_dirs.append(output_dir)
            if cancel_monitor is not None:
                cancel_monitor(monitor, reason="Import stopped from Mimics.")
                continue
            monitor["done"] = True
            work_dir = monitor.get("work_dir")
            if terminator is not None and monitor.get("job_dir"):
                terminator(
                    monitor.get("job_dir"),
                    on_complete=(lambda path=work_dir: cleanup_work(path)) if cleanup_work is not None else None,
                )
            if stopper is not None:
                stopper(key)
            if cleanup_work is not None and terminator is None:
                cleanup_work(monitor.get("work_dir"))
        except Exception:
            pass
    return output_dirs


def _runtime_dir():
    path = os.path.join(_project_root(), ".mimics_runtime")
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


# -- Queue / output directory discovery ---------------------------------

def _scan_filesystem_for_queue_dirs():
    """Walk common locations to find active queue directories.

    This is the fallback when the lock-file and registry approaches
    miss directories (e.g. queue registry is disabled, lock was cleaned
    up, or the output directory is outside the project tree).
    """
    result = []
    root = _project_root()

    # 1. Scan for _mcs_queue_active.json files under the project root
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip deep vendor/env trees
        if any(skip in dirpath.replace(os.sep, "/") for skip in (
            "nninteractive_env", ".git", "__pycache__", "external/dinov3",
        )):
            continue
        if "_mcs_queue_active.json" in filenames:
            state = runtime_common.read_json(os.path.join(dirpath, "_mcs_queue_active.json"), {}) or {}
            if state.get("output_dir"):
                result.append(state.get("output_dir"))
            elif os.path.basename(dirpath) == ".mimics_runtime":
                result.append(os.path.dirname(dirpath))
            else:
                result.append(dirpath)
        # Limit depth to avoid scanning huge model directories
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 6:
            _dirnames[:] = []

    # 2. Also check the runtime directory for any stored queue info
    runtime = os.path.join(root, ".mimics_runtime")
    if os.path.isdir(runtime):
        for name in os.listdir(runtime):
            if name.startswith("mcs_") and name.endswith(".json"):
                payload = runtime_common.read_json(os.path.join(runtime, name), {}) or {}
                output_dir = payload.get("output_dir")
                if output_dir:
                    result.append(output_dir)

    return result


def _queue_dirs_from_runtime_state():
    """Discover queue directories from lock files and registries."""
    result = []
    root = _project_root()
    lock_dir = runtime_common.resource_lock_dir(root)

    # Read ALL lock files, not just background_mimics.lock
    if os.path.isdir(lock_dir):
        for name in os.listdir(lock_dir):
            if not name.endswith(".lock"):
                continue
            lock_path = os.path.join(lock_dir, name)
            payload = runtime_common.read_json(lock_path, {}) or {}
            if not payload:
                continue
            # output_dir / ts_root may be at top level or nested under "details"
            details = payload.get("details") or {}
            output_dir = payload.get("output_dir") or details.get("output_dir")
            if output_dir:
                result.append(output_dir)
            ts_root = payload.get("ts_root") or details.get("ts_root")
            if ts_root:
                result.append(ts_root)
                result.append(os.path.join(ts_root, "mcs_output"))

    # Queue registry (may be disabled by env var, but check anyway)
    registry_dir = os.path.join(runtime_common.import_runtime_base(_project_root()), "mcs_queues")
    if os.path.isdir(registry_dir):
        for name in os.listdir(registry_dir):
            if not name.endswith(".json"):
                continue
            payload = runtime_common.read_json(os.path.join(registry_dir, name), {}) or {}
            output_dir = payload.get("output_dir")
            if output_dir:
                result.append(output_dir)

    # Filesystem scan as aggressive fallback
    result.extend(_scan_filesystem_for_queue_dirs())

    # Deduplicate
    unique = []
    seen = set()
    for path in result:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _request_queue_stop(reason="Stop Background Services"):
    """Write stop markers to every discovered queue directory."""
    stopped = []
    payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": reason,
    }
    for output_dir in _queue_dirs_from_runtime_state():
        if not os.path.isdir(output_dir):
            continue
        try:
            queue_runtime = runtime_common.import_queue_runtime_dir(_project_root(), output_dir)
            runtime_common.write_json_atomic(
                os.path.join(queue_runtime, "_mcs_queue_stop.json"), payload
            )
            # Also remove the active marker so the background process
            # doesn't think a producer is still sending work.
            active = os.path.join(queue_runtime, "_mcs_queue_active.json")
            if os.path.isfile(active):
                os.remove(active)
            stopped.append(output_dir)
        except Exception:
            pass
    return stopped


def _background_mimics_lock_path():
    return os.path.join(runtime_common.resource_lock_dir(_project_root()), "background_mimics.lock")


def _lock_is_import_creation(payload):
    if not isinstance(payload, dict):
        return False
    owner = str(payload.get("owner") or "").lower()
    kind = str(payload.get("kind") or "").lower()
    return kind == "create_mcs" or "import .mcs creation" in owner


def _lock_is_mask_export(payload):
    if not isinstance(payload, dict):
        return False
    owner = str(payload.get("owner") or "").lower()
    kind = str(payload.get("kind") or "").lower()
    return kind == "export_labels" or "label export" in owner


def _request_lock_owned_stop_markers(reason):
    """Signal lock-owned jobs without mutating or deleting their resource lock."""
    written = []
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    for name in ("background_mimics.lock", "gpu.lock"):
        payload = runtime_common.read_json(os.path.join(lock_dir, name), {}) or {}
        if not payload:
            continue
        candidates = []
        if payload.get("stop_path"):
            candidates.append(payload.get("stop_path"))
        if payload.get("cancel_path"):
            candidates.append(payload.get("cancel_path"))
        for path in candidates:
            if not path:
                continue
            try:
                runtime_common.write_json_atomic(
                    path,
                    {
                        "status": "stop_requested",
                        "requested_at_epoch": time.time(),
                        "reason": reason,
                    },
                )
                written.append(path)
            except Exception:
                pass
    return written


def _stop_export_inprocess_monitors():
    """Detach export callbacks and cancel a launcher that is still racing."""
    module = sys.modules.get("mimics_export")
    if module is None:
        return 0
    stopped = 0
    monitors = getattr(module, "_EXPORT_MONITORS", {}) or {}
    stopper = getattr(module, "_stop_export_monitor", None)
    for key, monitor in list(monitors.items()):
        try:
            monitor["cancel_requested"] = True
            process = monitor.get("process")
            if process is not None and process.poll() is None:
                runtime_common.terminate_process_async(process=process, graceful_seconds=2.0)
            if stopper is not None:
                stopper(key)
            stopped += 1
        except Exception:
            pass
    return stopped


def stop_background_import():
    """Request only the import/.mcs creation queue to stop.

    This is intentionally narrower than Stop_Background_Services: it writes
    import queue stop markers and only targets the background Mimics process
    whose resource lock says it is doing import .mcs creation.
    """
    stopped_queues = _request_queue_stop(reason="Stop Background Import")
    inprocess_outputs = _stop_import_inprocess_monitors()
    payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": "Stop Background Import",
    }
    for output_dir in inprocess_outputs:
        try:
            queue_runtime = runtime_common.import_queue_runtime_dir(_project_root(), output_dir)
            runtime_common.write_json_atomic(
                os.path.join(queue_runtime, "_mcs_queue_stop.json"), payload
            )
            if output_dir not in stopped_queues:
                stopped_queues.append(output_dir)
        except Exception:
            pass
    lock_path = _background_mimics_lock_path()
    lock_payload = runtime_common.read_json(lock_path, {}) or {}
    lock_token = str(lock_payload.get("token") or "")
    target_pid = None
    if _lock_is_import_creation(lock_payload):
        try:
            target_pid = int(lock_payload.get("pid") or 0)
        except Exception:
            target_pid = None
    stop_log = os.path.join(_runtime_dir(), "stop_import_last.json")
    report = {
        "RequestedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "QueueStopDirs": stopped_queues,
        "TargetPid": target_pid,
        "TargetKind": lock_payload.get("kind", ""),
        "TargetOwner": lock_payload.get("owner", ""),
        "LockPath": lock_path,
        "Killed": [],
        "Message": "",
    }
    if not target_pid:
        report["Message"] = (
            "Import stop markers were written. No active background Mimics import process was found."
        )
        try:
            runtime_common.write_json_atomic(stop_log, report)
        except Exception:
            pass
        return {
            "ok": True,
            "stop_log": stop_log,
            "stopped_queues": stopped_queues,
            "target_pid": None,
            "launched": False,
        }

    if os.name != "nt":
        report["Message"] = (
            "Import stop markers were written. Process termination is only implemented for Windows Mimics workstations."
        )
        try:
            runtime_common.write_json_atomic(stop_log, report)
        except Exception:
            pass
        return {
            "ok": True,
            "stop_log": stop_log,
            "stopped_queues": stopped_queues,
            "target_pid": target_pid,
            "launched": False,
        }

    command = (
        "$pidToStop={0};"
        "$lock='{1}';"
        "$out='{2}';"
        "$expectedToken='{3}';"
        "$markers=@('_run_create_mcs.py','create_mcs_batch.py');"
        "$current=$null;"
        "try {{$current=Get-Content -Raw -Path $lock -ErrorAction Stop | ConvertFrom-Json}} catch {{}};"
        "$lockMatches=[bool]($current -and ([string]$current.token -eq $expectedToken) -and ([int64]$current.pid -eq $pidToStop));"
        "$record = Get-CimInstance Win32_Process -Filter \"ProcessId=$pidToStop\";"
        "$killed = @();"
        "$matched = $false;"
        "if ($record -and $record.CommandLine) {{"
        "  $cmd = $record.CommandLine -replace '/', '\\';"
        "  $matched = [bool]($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "}};"
        "if ($record -and $matched -and $lockMatches) {{"
        "  taskkill /PID $pidToStop /T /F 2>$null 1>$null;"
        "  $killed += [PSCustomObject]@{{"
        "    ProcessId = $pidToStop;"
        "    Name = $record.Name;"
        "    ExitCode = $LASTEXITCODE;"
        "    CommandLine = $record.CommandLine"
        "  }};"
        "}};"
        "$message = if (-not $lockMatches) {{ 'Import lock ownership changed; no process was killed.' }} elseif ($matched) {{ 'Background import stop was requested.' }} else {{ 'Import lock PID did not match an import command line; no process was killed.' }};"
        "$report = [PSCustomObject]@{{"
        "  RequestedAt = (Get-Date).ToString('s');"
        "  QueueStopDirs = @({4});"
        "  TargetPid = $pidToStop;"
        "  CommandLineMatched = $matched;"
        "  Killed = $killed;"
        "  LockPath = $lock;"
        "  Message = $message"
        "}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(
        int(target_pid),
        lock_path.replace("'", "''"),
        stop_log.replace("'", "''"),
        lock_token.replace("'", "''"),
        ",".join("'{}'".format(path.replace("'", "''")) for path in stopped_queues),
    )
    launched = False
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_hidden_process_kwargs()
        )
        launched = True
    except Exception as exc:
        report["Message"] = "Could not launch PowerShell import stop command: {0}".format(exc)
        try:
            runtime_common.write_json_atomic(stop_log, report)
        except Exception:
            pass
        _mimics_log(logging.WARNING, report["Message"])
    return {
        "ok": True,
        "stop_log": stop_log,
        "stopped_queues": stopped_queues,
        "target_pid": target_pid,
        "launched": launched,
    }


def stop_background_export():
    """Stop only the Mimics-Script mask-export worker."""
    module = sys.modules.get("mimics_export")
    if module is not None:
        try:
            module.cancel_current_project_exports()
        except Exception:
            pass
    lock_path = _background_mimics_lock_path()
    lock_payload = runtime_common.read_json(lock_path, {}) or {}
    lock_token = str(lock_payload.get("token") or "")
    target_pid = None
    export_root = str(lock_payload.get("export_root") or "")
    if _lock_is_mask_export(lock_payload):
        try:
            target_pid = int(lock_payload.get("pid") or 0)
        except Exception:
            target_pid = None
    stop_path = str(lock_payload.get("stop_path") or "")
    if not stop_path and export_root:
        stop_path = os.path.join(export_root, ".mimics_runtime", "_export_stop.json")
    if stop_path:
        try:
            runtime_common.write_json_atomic(
                stop_path,
                {
                    "status": "stop_requested",
                    "requested_at_epoch": time.time(),
                    "reason": "Stop Mask Export",
                },
            )
        except Exception:
            pass
    stop_log = os.path.join(_runtime_dir(), "stop_export_last.json")
    try:
        if os.path.isfile(stop_log):
            os.remove(stop_log)
    except OSError:
        pass
    report = {
        "RequestedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "TargetPid": target_pid,
        "TargetKind": lock_payload.get("kind", ""),
        "TargetOwner": lock_payload.get("owner", ""),
        "ExportRoot": export_root,
        "StopMarker": stop_path,
        "Matched": [],
        "Killed": [],
        "Message": "",
    }
    if not target_pid:
        report["Message"] = "No active Mimics-Script mask export process was found."
        runtime_common.write_json_atomic(stop_log, report)
        return {"ok": True, "stop_log": stop_log, "target_pid": None, "launched": False}
    if os.name != "nt":
        report["Message"] = "Export stop marker written; process termination is only available on Windows."
        runtime_common.write_json_atomic(stop_log, report)
        return {"ok": True, "stop_log": stop_log, "target_pid": target_pid, "launched": False}

    command = (
        "$pidToStop={0};"
        "$lock='{1}';"
        "$out='{2}';"
        "$expectedToken='{3}';"
        "$markers=@('_run_export_batch.py','mimics_export.py');"
        "Start-Sleep -Seconds 2;"
        "$current=$null;"
        "try {{$current=Get-Content -Raw -Path $lock -ErrorAction Stop | ConvertFrom-Json}} catch {{}};"
        "$lockMatches=[bool]($current -and ([string]$current.token -eq $expectedToken) -and ([int64]$current.pid -eq $pidToStop));"
        "$record=Get-CimInstance Win32_Process -Filter \"ProcessId=$pidToStop\";"
        "$matched=$false;$killed=@();$records=@();"
        "if ($record -and $record.CommandLine) {{"
        " $matched=[bool]($markers | Where-Object {{ $record.CommandLine -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "}};"
        "if ($record -and $matched -and $lockMatches) {{"
        " $records=@($record | Select-Object ProcessId,Name,CommandLine);"
        " taskkill /PID $pidToStop /T /F 2>$null 1>$null;"
        " $killed+=@([PSCustomObject]@{{ProcessId=$pidToStop;Name=$record.Name;ExitCode=$LASTEXITCODE;CommandLine=$record.CommandLine}});"
        "}};"
        "$message=if (-not $lockMatches) {{'Mask-export lock ownership changed; no process was killed.'}} elseif (-not $record) {{'Mask export stopped gracefully.'}} elseif ($matched) {{'Mask export process was stopped.'}} else {{'Lock PID did not match a mask-export command; no process was killed.'}};"
        "$report=[PSCustomObject]@{{RequestedAt=(Get-Date).ToString('s');TargetPid=$pidToStop;Matched=$records;Killed=$killed;Message=$message}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(
        int(target_pid),
        lock_path.replace("'", "''"),
        stop_log.replace("'", "''"),
        lock_token.replace("'", "''"),
    )
    launched = False
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_hidden_process_kwargs()
        )
        launched = True
    except Exception as exc:
        report["Message"] = "Could not launch mask-export stop command: {0}".format(exc)
        runtime_common.write_json_atomic(stop_log, report)
    return {"ok": True, "stop_log": stop_log, "target_pid": target_pid, "launched": launched}


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


# -- Cache directories --------------------------------------------------

_CACHE_DIRS = (
    "logs",
    "__pycache__",
)

_CACHE_GLOBS = (
    "*_work",           # Bridge prepare work directories
    "_import_jobs",     # Bridge job directories
    "source_fastpath_cache",  # nnInteractive on-demand axial NIfTI cache
)


def _find_cache_paths():
    """Return all cache/temp paths that can be safely cleared."""
    root = _project_root()
    paths = []

    # Direct cache directories under project root
    for name in _CACHE_DIRS:
        p = os.path.join(root, name)
        if os.path.exists(p):
            paths.append(p)

    # Glob patterns (match in root and common output locations)
    for pattern in _CACHE_GLOBS:
        for p in glob.glob(os.path.join(root, pattern)):
            paths.append(p)
        for p in glob.glob(os.path.join(root, "*", pattern)):
            paths.append(p)
        for p in glob.glob(os.path.join(root, "*", "*", pattern)):
            paths.append(p)

    # __pycache__ directories recursively (skip env and .git)
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (
            "nninteractive_env", ".git", "external",
        )]
        if os.path.basename(dirpath) == "__pycache__":
            paths.append(dirpath)

    # Runtime control state is not a cache. Only include explicitly
    # regenerable nnInteractive data, and only terminal async job folders.
    nn_runtime = os.path.join(root, ".mimics_runtime", "nninteractive")
    source_cache = os.path.join(nn_runtime, "source_fastpath_cache")
    if os.path.isdir(source_cache):
        paths.append(source_cache)
    async_root = os.path.join(nn_runtime, "async_jobs")
    if os.path.isdir(async_root):
        for name in os.listdir(async_root):
            job_dir = os.path.join(async_root, name)
            if not os.path.isdir(job_dir):
                continue
            status = runtime_common.read_json(os.path.join(job_dir, "worker_status.json"), {}) or {}
            if str(status.get("status", "")) in ("closed", "failed", "expired"):
                paths.append(job_dir)

    return sorted(set(os.path.abspath(p) for p in paths if os.path.exists(p)))


def _active_cache_cleanup_blockers():
    """Return active tasks whose scratch files must not be removed."""
    blockers = []
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    for name in ("background_mimics.lock", "gpu.lock"):
        path = os.path.join(lock_dir, name)
        payload = runtime_common.read_json(path, {}) or {}
        pid = payload.get("pid")
        if pid and runtime_common.process_exists(pid):
            blockers.append(
                "{0}: {1}".format(name, runtime_common.resource_lock_summary(payload))
            )

    for module_name, collection_name in (
        ("mimics_import", "_IMPORT_MONITORS"),
        ("mimics_export", "_EXPORT_MONITORS"),
        ("mask_import", "_MASK_IMPORT_MONITORS"),
        ("fewshot_mimics", "_MONITORS"),
        ("nninteractive_mimics", "_ASYNC_MONITORS"),
    ):
        module = sys.modules.get(module_name)
        collection = getattr(module, collection_name, {}) if module is not None else {}
        if any(item and not item.get("done") for item in (collection or {}).values()):
            blockers.append("{0} has an active Mimics monitor".format(module_name))

    task_dir = os.path.join(_runtime_dir(), "ui_tasks")
    if os.path.isdir(task_dir):
        for name in os.listdir(task_dir):
            if not name.endswith(".json") or name.endswith("_stop.json"):
                continue
            payload = runtime_common.read_json(os.path.join(task_dir, name), {}) or {}
            status = str(payload.get("status") or "").lower()
            if status in ("completed", "closed", "failed", "cancelled", "expired"):
                continue
            pid = payload.get("pid") or payload.get("controller_pid")
            if pid and runtime_common.process_exists(pid):
                blockers.append("active task status: {0}".format(name))
    return blockers


def clear_all_caches():
    """Remove all intermediate files, temp directories, and caches.

    Only deletes non-critical, auto-generated files. Project source code,
    config files, model weights, and the nninteractive_env are NOT touched.

    Files that are currently in use (locked by another process) are skipped
    and reported separately so the user knows what was retained.
    """
    blockers = _active_cache_cleanup_blockers()
    if blockers:
        return {
            "removed": [],
            "failed": [],
            "skipped_in_use": [
                (_runtime_dir(), "Cache cleanup skipped while tasks are active: " + "; ".join(blockers))
            ],
        }

    paths = _find_cache_paths()
    removed = []
    failed = []
    skipped_in_use = []
    now = time.time()

    for p in paths:
        try:
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p, ignore_errors=False)
            else:
                os.remove(p)
            removed.append(p)
        except OSError as e:
            errno = getattr(e, "winerror", 0) or getattr(e, "errno", 0)
            # Windows ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33)
            # or POSIX EACCES (13) / EPERM (1) — file is likely in use.
            if errno in (1, 13, 32, 33):
                skipped_in_use.append((p, str(e)))
            else:
                failed.append((p, str(e)))
        except Exception as e:
            failed.append((p, str(e)))

    # Only clear resource locks when no background process is holding them.
    _clear_resource_locks_safe()

    return {
        "removed": removed,
        "failed": failed,
        "skipped_in_use": skipped_in_use,
    }


def _clear_stale_runtime_files(runtime_dir, removed, failed, skipped_in_use):
    """Remove stale files from .mimics_runtime/, keeping active locks."""
    # Locks that are currently held by a running process
    active_pids = set()
    lock_dir = os.path.join(runtime_dir, "locks")
    if os.path.isdir(lock_dir):
        for name in os.listdir(lock_dir):
            if not name.endswith(".lock"):
                continue
            payload = runtime_common.read_json(os.path.join(lock_dir, name), {}) or {}
            pid = payload.get("pid")
            if pid and runtime_common.process_exists(pid):
                active_pids.add(int(pid))

    for name in os.listdir(runtime_dir):
        filepath = os.path.join(runtime_dir, name)
        if not os.path.isfile(filepath):
            continue
        # Skip active lock files and queue registries
        if name.endswith(".lock") or name == "mcs_queues":
            continue
        try:
            os.remove(filepath)
            removed.append(filepath)
        except OSError as e:
            errno = getattr(e, "winerror", 0) or getattr(e, "errno", 0)
            if errno in (1, 13, 32, 33):
                skipped_in_use.append((filepath, str(e)))
            else:
                failed.append((filepath, str(e)))
        except Exception as e:
            failed.append((filepath, str(e)))


def _clear_resource_locks_safe():
    """Clear resource locks only when no process is holding them."""
    lock_dir = runtime_common.resource_lock_dir(_project_root())
    runtime_common.cleanup_stale_resource_locks(lock_dir)


# -- Async cache clearing (non-blocking Mimics GUI) -----------------

def _clear_cache_tick(monitor):
    """Timer callback: check if the background clearing thread finished."""
    if monitor.get("done"):
        return
    thread = monitor.get("thread")
    if thread is None or thread.is_alive():
        now = time.time()
        last = float(monitor.get("last_progress_log_epoch", 0.0) or 0.0)
        if now - last >= 5.0:
            started = float(monitor.get("started_at_epoch", now) or now)
            _mimics_log(
                logging.INFO,
                "Cache clearing is still running... elapsed {0:.1f}s".format(max(0.0, now - started)),
            )
            monitor["last_progress_log_epoch"] = now
        return  # still working

    monitor["done"] = True
    result = monitor.get("result") or {}
    removed_count = len(result.get("removed", []))
    failed_count = len(result.get("failed", []))
    in_use_count = len(result.get("skipped_in_use", []))

    lines = ["Cache cleared."]
    if removed_count:
        lines.append("{0} path(s) removed.".format(removed_count))
    if in_use_count:
        lines.append("{0} path(s) in use — skipped.".format(in_use_count))
    if failed_count:
        lines.append("{0} path(s) could not be removed.".format(failed_count))
        for path, err in result.get("failed", [])[:5]:
            lines.append("  {0}: {1}".format(path, err))

    message = "\n".join(lines)
    try:
        mimics.dialogs.message_box(title="Clear Cache", message=message, ui_blocking=False)
    except TypeError:
        mimics.dialogs.message_box(title="Clear Cache", message=message)
    _mimics_log(logging.INFO, message)
    _dispose_cache_monitor(monitor)


def clear_cache_main():
    """Entry point for the Clear Cache scripting library button.

    All heavy disk I/O runs in a daemon thread. Mimics GUI stays responsive.
    A timer polls the thread every 0.5s and shows results when done.
    """
    for active in list(_CACHE_CLEAR_MONITORS):
        thread = active.get("thread") if active else None
        if thread is not None and thread.is_alive():
            _mimics_log(logging.INFO, "Cache clearing is already running.")
            try:
                mimics.dialogs.message_box(
                    title="Clear Cache",
                    message="Cache clearing is already running.",
                    ui_blocking=False,
                )
            except Exception:
                pass
            return 0

    monitor = {
        "thread": None,
        "result": None,
        "done": False,
        "started_at_epoch": time.time(),
        "last_progress_log_epoch": 0.0,
    }

    def _run():
        try:
            monitor["result"] = clear_all_caches()
        except Exception as exc:
            monitor["result"] = {
                "removed": [],
                "failed": [("(fatal)", str(exc))],
                "skipped_in_use": [],
            }

    thread = threading.Thread(target=_run)
    thread.daemon = True
    monitor["thread"] = thread
    thread.start()

    _mimics_log(logging.INFO, "Cache clearing started in background...")

    # Timer to poll for completion (non-blocking)
    _start_timer(_clear_cache_tick, monitor, poll_seconds=0.5)
    _CACHE_CLEAR_MONITORS.append(monitor)
    return 0


def _dispose_cache_monitor(monitor):
    """Stop timer handles and release monitor references after completion."""
    _stop_timer_handles(monitor)
    try:
        _CACHE_CLEAR_MONITORS.remove(monitor)
    except ValueError:
        pass


def _stop_timer_handles(monitor):
    """Stop any Qt/Win32 timer handles stored in monitor."""
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


def _start_timer(tick_fn, monitor, poll_seconds=0.5):
    """Start a non-blocking timer that calls tick_fn(monitor) every poll_seconds.

    Uses Win32 SetTimer inside Mimics, then Qt or a daemon-thread fallback.
    """
    # 1. Use Mimics' existing Windows message pump without importing a second
    # Qt binding into the host process.
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_uint, ctypes.c_ulong)

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
            timer_id = user32.SetTimer(None, 0, int(poll_seconds * 1000), ctypes.cast(callback, ctypes.c_void_p))
            if timer_id:
                monitor["win32_timer"] = (user32, timer_id)
                monitor["win32_callback"] = callback
            return
        except Exception:
            pass

    # 2. Qt fallback for non-Windows hosts or a failed native timer.
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

    # 3. Daemon thread (last resort).
    def _thread_poll():
        while not monitor.get("done"):
            try:
                tick_fn(monitor)
            except Exception:
                pass
            time.sleep(poll_seconds)

    t = threading.Thread(target=_thread_poll)
    t.daemon = True
    t.start()


# -- Stop background processes ------------------------------------------

def stop_background_processes():
    if os.name != "nt":
        return False

    # 1. Publish graceful stop signals before detaching monitors. A task that
    # reaches its safe boundary during this window can write its own terminal
    # state and release its token-owned lock normally.
    stopped_queues = _request_queue_stop()
    stop_markers = _request_lock_owned_stop_markers("Stop Background Services")
    _stop_inprocess_monitors()

    stop_log = os.path.join(_runtime_dir(), "stop_background_last.json")

    # 2. Build PowerShell command to forcefully kill remaining processes
    markers = "@(" + ",".join(
        "'{}'".format(marker.replace("'", "''")) for marker in MARKERS
    ) + ")"
    owned_roots = _owned_roots() + _queue_dirs_from_runtime_state()
    roots = "@(" + ",".join(
        "'{}'".format(root.replace("'", "''")) for root in owned_roots
    ) + ")"
    queues = "@(" + ",".join(
        "'{}'".format(path.replace("'", "''")) for path in stopped_queues
    ) + ")"
    cutoff_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # PowerShell script: find and kill all owned processes
    command = (
        "$markers={0};"
        "$roots={1};"
        "$queues={2};"
        "$out='{3}';"
        "$foregroundPid={4};"
        "$cutoff=[DateTime]::Parse('{5}').ToUniversalTime();"
        # Find processes whose command line references both a root and a marker
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "  $cmd = $_.CommandLine;"
        "  if (-not $cmd -or $_.ProcessId -eq $PID -or $_.ProcessId -eq $foregroundPid) {{ return $false }};"
        "  if ($_.CreationDate -and $_.CreationDate.ToUniversalTime() -gt $cutoff) {{ return $false }};"
        "  $cmd = $cmd -replace '/', '\\';"
        "  $inRoot = ($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "  $hasMarker = ($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }} | Select-Object -First 1);"
        "  $inRoot -and $hasMarker"
        "}};"
        # Require both an owned root and a dedicated marker. A broad marker-only
        # sweep can terminate unrelated installations that use the same filenames.
        "$all = @($matched) | Sort-Object ProcessId -Unique;"
        "$records = @($all | Select-Object ProcessId, Name, CommandLine);"
        "$killed = @();"
        "$all | ForEach-Object {{"
        "  $procId = $_.ProcessId;"
        "  taskkill /PID $procId /T /F 2>$null 1>$null;"
        "  $killed += [PSCustomObject]@{{"
        "    ProcessId = $procId;"
        "    Name = $_.Name;"
        "    ExitCode = $LASTEXITCODE;"
        "    CommandLine = $_.CommandLine"
        "  }};"
        "}};"
        "Start-Sleep -Milliseconds 500;"
        "$report = [PSCustomObject]@{{"
        "  RequestedAt = (Get-Date).ToString('s');"
        "  QueueStopDirs = $queues;"
        "  OwnedRoots = $roots;"
        "  Matched = $records;"
        "  Killed = $killed"
        "}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(
        markers,
        roots,
        queues,
        stop_log.replace("'", "''"),
        int(os.getpid()),
        cutoff_utc,
    )

    process = None
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_hidden_process_kwargs()
        )
        process = True
    except Exception as exc:
        _mimics_log(logging.WARNING, "Could not launch PowerShell stop command: {0}".format(exc))
        # Queue stop markers were already written — the background processes
        # will pick them up on their own poll cycle.

    _mimics_log(
        logging.INFO,
        "Stop request submitted. Queue stop markers: {0}. Owned roots: {1}. Report: {2}".format(
            len(stopped_queues),
            len(owned_roots),
            stop_log,
        ),
    )
    return {
        "ok": True,
        "stop_log": stop_log,
        "stopped_queues": stopped_queues,
        "owned_roots": owned_roots,
        "launched": bool(process),
    }


def _stop_background_tick(monitor):
    if monitor.get("done"):
        return
    report_path = monitor.get("stop_log")
    if report_path and os.path.isfile(report_path):
        report = runtime_common.read_json(report_path, {}) or {}
        killed = report.get("Killed") or []
        matched = report.get("Matched") or []
        queue_dirs = report.get("QueueStopDirs") or []
        if isinstance(killed, dict):
            killed = [killed]
        if isinstance(matched, dict):
            matched = [matched]
        if isinstance(queue_dirs, (str, bytes)):
            queue_dirs = [queue_dirs]
        monitor["done"] = True
        _stop_timer_handles(monitor)
        key = monitor.get("monitor_key")
        if key in _STOP_MONITORS:
            del _STOP_MONITORS[key]
        title = monitor.get("title", "Stop Background Services")
        lines = [monitor.get("completion_text", title + " completed.")]
        lines.append("Matched process(es): {0}".format(len(matched)))
        lines.append("Kill request(s): {0}".format(len(killed)))
        if queue_dirs:
            lines.append("Queue stop marker dir(s): {0}".format(len(queue_dirs)))
        if report.get("Message"):
            lines.append(str(report.get("Message")))
        lines.append("Report: {0}".format(report_path))
        msg = "\n".join(lines)
        _mimics_log(logging.INFO, msg)
        try:
            mimics.dialogs.message_box(title=title, message=msg, ui_blocking=False)
        except TypeError:
            mimics.dialogs.message_box(title=title, message=msg)
        return
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_timer_handles(monitor)
        key = monitor.get("monitor_key")
        if key in _STOP_MONITORS:
            del _STOP_MONITORS[key]
        title = monitor.get("title", "Stop Background Services")
        msg = (
            "{0} is still running or did not produce a report yet.\n\n"
            "Check: {1}"
        ).format(title, monitor.get("stop_log", "(unknown)"))
        _mimics_log(logging.WARNING, msg)
        try:
            mimics.dialogs.message_box(title=title, message=msg, ui_blocking=False)
        except TypeError:
            mimics.dialogs.message_box(title=title, message=msg)


def _start_stop_monitor(monitor, poll_seconds=0.5, timeout_seconds=90.0):
    monitor["done"] = False
    monitor["deadline"] = time.time() + float(timeout_seconds)
    key = monitor.get("monitor_key")
    _STOP_MONITORS[key] = monitor
    _start_timer(_stop_background_tick, monitor, poll_seconds=float(poll_seconds))
    return True


# -- Main entry points --------------------------------------------------

def main():
    result = stop_background_processes()
    ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
    if isinstance(result, dict) and result.get("stop_log"):
        _start_stop_monitor(
            {
                "monitor_key": "stop_bg_{0}".format(int(time.time() * 1000)),
                "stop_log": result.get("stop_log"),
            },
            poll_seconds=0.5,
            timeout_seconds=90.0,
        )
    if not ok:
        try:
            mimics.dialogs.message_box(
                title="Stop Background Services",
                message="Background cleanup is only implemented for Windows Mimics workstations.",
                ui_blocking=False,
            )
        except TypeError:
            mimics.dialogs.message_box(
                title="Stop Background Services",
                message="Background cleanup is only implemented for Windows Mimics workstations.",
            )
    return 0


def main_stop_import():
    result = stop_background_import()
    target = result.get("target_pid") if isinstance(result, dict) else None
    queues = result.get("stopped_queues", []) if isinstance(result, dict) else []
    report = result.get("stop_log", "") if isinstance(result, dict) else ""
    if target:
        message = (
            "Background import stop requested.\n"
            "Target background Mimics PID: {0}\n"
            "Queue stop marker dir(s): {1}\n"
            "Report: {2}"
        ).format(target, len(queues), report)
    else:
        message = (
            "Background import stop markers were written.\n"
            "No active background Mimics import process was found.\n"
            "Queue stop marker dir(s): {0}\n"
            "Report: {1}"
        ).format(len(queues), report)
    _mimics_log(logging.INFO, message)
    try:
        mimics.dialogs.message_box(
            title="Stop Background Import",
            message=message,
            ui_blocking=False,
        )
    except TypeError:
        mimics.dialogs.message_box(
            title="Stop Background Import",
            message=message,
        )
    return 0


def main_stop_export():
    foreground_count = 0
    try:
        import mimics_export
        foreground_count = mimics_export.cancel_current_project_exports()
    except Exception:
        foreground_count = 0
    result = stop_background_export()
    report = result.get("stop_log", "") if isinstance(result, dict) else ""
    if report:
        _start_stop_monitor(
            {
                "monitor_key": "stop_export_{0}".format(int(time.time() * 1000)),
                "stop_log": report,
                "title": "Stop Mask Export",
                "completion_text": "Mask export stop completed.",
            },
            poll_seconds=0.5,
            timeout_seconds=30.0,
        )
    _mimics_log(
        logging.INFO,
        "Mask export stop requested. Foreground exports signalled: {0}. "
        "Completion will be reported when resources are released.".format(foreground_count),
    )
    return 0


if __name__ == "__main__":
    main()
