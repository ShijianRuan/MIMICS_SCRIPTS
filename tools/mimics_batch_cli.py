#!/usr/bin/env python3
"""External batch utilities for Mimics-Script.

This script runs outside the foreground Mimics GUI.

Commands:
  prepare-import  Convert dataset cases to prepare manifests and optionally
                  launch background Mimics to create .mcs files.
  export-labels   Launch background Mimics to export labels from saved .mcs.
  kill-background Stop integration-created bridge, background Mimics, and
                  nnInteractive service processes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import (
    FileResourceLock,
    ResourceLockTimeout,
    default_resource_lock_dir,
)

BRIDGE = ROOT / "mimics_bridge.py"
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
import runtime_common
RESOURCE_LOCK_DIR = default_resource_lock_dir(ROOT)
BACKGROUND_MIMICS_LOCK_PATH = RESOURCE_LOCK_DIR / "background_mimics.lock"


def project_python_candidates():
    return [
        ROOT / "nninteractive_env" / "python.exe",
        ROOT / "nninteractive_env" / "Scripts" / "python.exe",
        ROOT / "nninteractive_env" / "python" / "python.exe",
        ROOT / "nninteractive_env" / "bin" / "python3",
        ROOT / "nninteractive_env" / "bin" / "python",
    ]


def resolve_bridge_python(explicit=None):
    candidates = []
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        candidates.append(path)
    candidates.extend(project_python_candidates())
    env_value = os.environ.get("MIMICS_BRIDGE_PYTHON") or os.environ.get("MIMICS_FEWSHOT_PYTHON")
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = ROOT / path
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run setup_offline.bat or pass --python explicitly."
    )


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    # SMB servers can allow create/write but deny rename/replace. Control JSON
    # readers tolerate a short incomplete interval, so direct overwrite is a
    # better final fallback than aborting a long import.
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError as exc:
        last_error = exc
    if last_error is not None:
        raise last_error


def run_bridge(python_exe, params):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    proc = subprocess.run(
        [python_exe, str(BRIDGE)],
        input=json.dumps(params).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:2000])
    try:
        result = json.loads(proc.stdout.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("bridge returned invalid JSON: {}".format(exc))
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "bridge returned non-ok status"))
    return result


def find_mimics_exe(explicit=None):
    """Find MimicsResearch.exe, with optional explicit override."""
    if explicit and Path(explicit).is_file():
        return explicit
    # Delegate to runtime_common which has the full search logic
    return runtime_common.find_mimics_exe()


def discover_cases(ts_root, cases, python_exe, mask_selection="all"):
    result = run_bridge(
        python_exe,
        {
            "action": "discover",
            "ts_root": str(ts_root),
            "cases_filter": sorted(cases) if cases else None,
            "mask_selection": mask_selection,
        },
    )
    if result.get("mask_mode") == "named" and int(result.get("mask_count", 0) or 0) == 0:
        raise RuntimeError("No segmentation files matched --masks={0}".format(mask_selection))
    if result.get("mask_mode") == "named":
        print(
            "Mask filter matched {} file(s); {} case(s) have no matching mask.".format(
                int(result.get("mask_count", 0) or 0),
                int(result.get("cases_without_selected_masks", 0) or 0),
            )
        )
    return list(result.get("cases", []))


def _acquire_background_mimics_lock(owner, wait_seconds=0.0):
    lock = FileResourceLock(BACKGROUND_MIMICS_LOCK_PATH, "background_mimics", owner)
    lock.acquire(wait_seconds=float(wait_seconds), poll_seconds=5.0)
    return lock


def launch_create_mcs(output_dir, mimics_exe, bridge_python, lock_timeout_seconds=0.0):
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(output_dir)))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runner = runtime_dir / "_run_create_mcs.py"
    runner.write_text(
        "\n".join([
            "# Auto-generated runner for background Mimics .mcs creation",
            "import sys, os",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "os.environ['MIMICS_BRIDGE_PYTHON'] = r'{}'".format(str(bridge_python)),
            "os.environ['MIMICS_BRIDGE_SCRIPT'] = r'{}'".format(str(ROOT / "mimics_bridge.py")),
            "import create_mcs_batch",
            "create_mcs_batch.main(r'{}', runtime_dir=r'{}')".format(str(output_dir), str(runtime_dir)),
            "",
        ]),
        encoding="utf-8",
    )
    lock = _acquire_background_mimics_lock("batch .mcs creation", lock_timeout_seconds)
    log = open(str(runtime_dir / "_background_mimics.log"), "ab")
    mimics_log = runtime_dir / "_background_mimics_application.log"
    try:
        proc = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe, str(runner), mimics_log_path=str(mimics_log)
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not lock.update_pid(
            proc.pid,
            kind="create_mcs",
            output_dir=str(output_dir.resolve()),
        ):
            raise RuntimeError(
                "Background Mimics started, but the background-Mimics lock "
                "could not be transferred to PID {}.".format(proc.pid)
            )
        return proc
    except Exception:
        poll = getattr(proc, "poll", None) if "proc" in locals() else None
        process_running = False
        if "proc" in locals():
            try:
                process_running = poll is None or poll() is None
            except Exception:
                process_running = True
        if process_running:
            try:
                proc.terminate()
                if hasattr(proc, "wait"):
                    proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    if hasattr(proc, "wait"):
                        proc.wait(timeout=10)
                except Exception:
                    pass
        process_stopped = not process_running
        if process_running:
            try:
                process_stopped = proc.poll() is not None
            except Exception:
                process_stopped = not runtime_common.process_exists(
                    getattr(proc, "pid", 0)
                )
        if process_stopped:
            lock.release()
        else:
            try:
                lock.update_pid(
                    proc.pid,
                    kind="create_mcs",
                    output_dir=str(output_dir.resolve()),
                    termination_pending=True,
                )
            except Exception:
                pass
        raise
    finally:
        log.close()


def launch_export_labels(ts_root, cases, mimics_exe, axes, flips, lock_timeout_seconds=0.0,
                         mcs_dir=None, label_output_root=None, overwrite_source=False,
                         mask_names=None):
    output_dir = Path(mcs_dir).resolve() if mcs_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = "export_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_dir = Path(ROOT) / ".mimics_runtime" / "export_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config = job_dir / "export_config.json"
    runner = job_dir / "run_export_batch.py"
    status_path = job_dir / "status.json"
    stop_path = job_dir / "_export_stop.json"
    write_json_atomic(
        config,
        {
            "ts_root": str(ts_root),
            "cases": sorted(cases) if cases else None,
            "axes": axes,
            "flips": flips,
            "output_dir": str(output_dir),
            "export_root": str(job_dir),
            "display_output_root": str(Path(label_output_root).resolve()) if label_output_root else str(ts_root),
            "label_output_root": str(Path(label_output_root).resolve()) if label_output_root else "",
            "overwrite_existing": bool(overwrite_source),
            "mask_names": list(mask_names or []),
            "status_path": str(status_path),
            "stop_path": str(stop_path),
            "job_runtime": str(job_dir),
        },
    )
    runner.write_text(
        "\n".join([
            "# Auto-generated runner for background Mimics batch export",
            "import sys, os",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "import mimics_export",
            "mimics_export.run_background_batch_export(r'{}')".format(str(config)),
            "",
        ]),
        encoding="utf-8",
    )
    lock = _acquire_background_mimics_lock("batch label export", lock_timeout_seconds)
    log = open(str(job_dir / "process.log"), "ab")
    mimics_log = job_dir / "mimics_application.log"
    try:
        proc = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe, str(runner), mimics_log_path=str(mimics_log)
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not lock.update_pid(
            proc.pid,
            kind="export_labels",
            ts_root=str(ts_root.resolve()),
            export_root=str(Path(label_output_root).resolve()) if label_output_root else str(ts_root),
            stop_path=str(stop_path),
        ):
            raise RuntimeError(
                "Background Mimics started, but the background-Mimics lock "
                "could not be transferred to PID {}.".format(proc.pid)
            )
        return proc
    except Exception:
        poll = getattr(proc, "poll", None) if "proc" in locals() else None
        process_running = False
        if "proc" in locals():
            try:
                process_running = poll is None or poll() is None
            except Exception:
                process_running = True
        if process_running:
            try:
                proc.terminate()
                if hasattr(proc, "wait"):
                    proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    if hasattr(proc, "wait"):
                        proc.wait(timeout=10)
                except Exception:
                    pass
        process_stopped = not process_running
        if process_running:
            try:
                process_stopped = proc.poll() is not None
            except Exception:
                process_stopped = not runtime_common.process_exists(
                    getattr(proc, "pid", 0)
                )
        if process_stopped:
            lock.release()
        else:
            try:
                lock.update_pid(
                    proc.pid,
                    kind="export_labels",
                    ts_root=str(ts_root.resolve()),
                    export_root=(
                        str(Path(label_output_root).resolve())
                        if label_output_root else str(ts_root)
                    ),
                    stop_path=str(stop_path),
                    termination_pending=True,
                )
            except Exception:
                pass
        raise
    finally:
        log.close()


def cmd_prepare_import(args):
    ts_root = Path(args.ts_root).resolve()
    bridge_python = resolve_bridge_python(args.python)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_filter = set(args.cases.split(",")) if args.cases else None
    cases = discover_cases(ts_root, cases_filter, bridge_python, args.masks)
    print("Discovered {} case(s).".format(len(cases)))
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(output_dir)))
    run_root = Path(runtime_common.import_runtime_base(str(ROOT))) / "import_runs" / (
        time.strftime("%Y%m%dT%H%M%S") + "_cli_" + uuid.uuid4().hex[:10]
    )
    queue_dir = runtime_dir / "prepared_queue"
    write_json_atomic(runtime_dir / "_mcs_queue_active.json", {"total": len(cases), "created_at_epoch": time.time()})
    completed = 0
    failed = 0
    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        work_dir = run_root / "work" / case_id
        print("[{}/{}] Preparing {}".format(index, len(cases), case_id))
        params = {
            "action": "prepare",
            "image_path": case["image"],
            "masks": case.get("masks", []),
            "dicom_out": str(work_dir / "derived_dicom"),
            "buffers_out": str(work_dir / "buffers"),
            "axes": args.axes,
            "flips": args.flips,
            "case_id": case_id,
        }
        try:
            result = run_bridge(bridge_python, params)
            result["output_mcs"] = str(output_dir / (case_id + ".mcs"))
            write_json_atomic(work_dir / "prepare_manifest.json", result)
            write_json_atomic(
                queue_dir / (case_id + "_" + uuid.uuid4().hex + ".json"),
                {
                    "case_id": case_id,
                    "work_dir": str(work_dir.resolve()),
                    "output_mcs": result["output_mcs"],
                    "created_at_epoch": time.time(),
                },
            )
            completed += 1
        except Exception as exc:
            failed += 1
            fail_dir = runtime_dir / "_failed_cases"
            write_json_atomic(fail_dir / (case_id + "_prepare.json"), {"case_id": case_id, "error": str(exc)})
            print("  failed: {}".format(exc))
    active = runtime_dir / "_mcs_queue_active.json"
    if active.exists():
        active.unlink()
    write_json_atomic(runtime_dir / "_mcs_queue_done.json", {"completed": completed, "failed": failed, "updated_at_epoch": time.time()})
    if args.no_create_mcs:
        return 0
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found. Manifests are ready; run .mcs creation later.", file=sys.stderr)
        return 2
    try:
        proc = launch_create_mcs(output_dir, mimics_exe, bridge_python, args.background_mimics_lock_timeout_seconds)
    except ResourceLockTimeout as exc:
        print("Background Mimics is busy: {}".format(exc), file=sys.stderr)
        return 75
    print("Background Mimics started for .mcs creation, PID={}".format(proc.pid))
    return 0


def cmd_export_labels(args):
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found.", file=sys.stderr)
        return 2
    cases = set(args.cases.split(",")) if args.cases else None
    try:
        proc = launch_export_labels(
            Path(args.ts_root).resolve(),
            cases,
            mimics_exe,
            args.axes,
            args.flips,
            args.background_mimics_lock_timeout_seconds,
            mcs_dir=args.mcs_dir,
            label_output_root=args.output_dir,
            overwrite_source=args.overwrite_source,
            mask_names=[
                value.strip() for value in str(args.masks or "all").split(",")
                if value.strip() and value.strip().lower() != "all"
            ],
        )
    except ResourceLockTimeout as exc:
        print("Background Mimics is busy: {}".format(exc), file=sys.stderr)
        return 75
    print("Background Mimics started for label export, PID={}".format(proc.pid))
    return 0


def _runtime_owned_roots():
    result = []
    payload = {}
    try:
        payload = json.loads(BACKGROUND_MIMICS_LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    details = payload.get("details") or {}
    output_dir = details.get("output_dir")
    if output_dir:
        result.append(Path(output_dir))
    ts_root = details.get("ts_root")
    if ts_root:
        ts_root_path = Path(ts_root)
        result.append(ts_root_path)
        result.append(ts_root_path / "mcs_output")
    registry = Path(runtime_common.import_runtime_base(str(ROOT))) / "mcs_queues"
    if registry.is_dir():
        for path in registry.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            output_dir = row.get("output_dir")
            if output_dir:
                result.append(Path(output_dir))
    unique = []
    seen = set()
    for path in result:
        try:
            normalized = str(path.resolve())
        except Exception:
            normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def cmd_kill_background(args):
    if os.name != "nt":
        print("Process cleanup is implemented for Windows Mimics workstations.")
        return 0
    stopped_queues = []
    stop_payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": "tools/mimics_batch_cli.py kill-background",
    }
    for queue_dir in _runtime_owned_roots():
        queue_path = Path(queue_dir)
        if not queue_path.is_dir():
            continue
        try:
            runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(queue_path)))
            runtime_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(runtime_dir / "_mcs_queue_stop.json", stop_payload)
            active = runtime_dir / "_mcs_queue_active.json"
            if active.is_file():
                active.unlink()
            stopped_queues.append(str(queue_path))
        except Exception:
            pass
    stop_markers = []
    for lock_path in (BACKGROUND_MIMICS_LOCK_PATH, RESOURCE_LOCK_DIR / "gpu.lock"):
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            lock_payload = {}
        details = lock_payload.get("details") or {}
        for key in ("stop_path", "cancel_path"):
            marker = lock_payload.get(key) or details.get(key)
            if not marker:
                continue
            try:
                write_json_atomic(Path(marker), stop_payload)
                stop_markers.append(str(marker))
            except Exception:
                pass
    markers = [
        "mimics_bridge.py",
        "nninteractive_bridge.py",
        "--async-worker",
        "_run_create_mcs.py",
        "_run_export_batch.py",
        "fewshot_pipeline.py",
        "nninteractive.inference.server.main",
        "--watchdog",
    ]
    ps_markers = "@(" + ",".join("'{}'".format(m.replace("'", "''")) for m in markers) + ")"
    owned_roots = [
        str(ROOT),
        str(ROOT / "nninteractive_env"),
        str(ROOT / "external"),
        str(ROOT / "tools"),
        str(ROOT / "runtime_py35"),
    ]
    owned_roots.extend(_runtime_owned_roots())
    ps_roots = "@(" + ",".join("'{}'".format(str(r).replace("'", "''")) for r in owned_roots) + ")"
    ps_queues = "@(" + ",".join("'{}'".format(str(r).replace("'", "''")) for r in stopped_queues) + ")"
    ps_stop_markers = "@(" + ",".join(
        "'{}'".format(str(r).replace("'", "''")) for r in stop_markers
    ) + ")"
    stop_log = ROOT / ".mimics_runtime" / "stop_background_last.json"
    stop_log.parent.mkdir(parents=True, exist_ok=True)
    cutoff_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    command = (
        "$markers={};"
        "$roots={};"
        "$queues={};"
        "$stopMarkers={};"
        "$out='{}';"
        "$foregroundPid={};"
        "$cutoff=[DateTime]::Parse('{}').ToUniversalTime();"
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; "
        "$cmd -and $_.ProcessId -ne $PID -and $_.ProcessId -ne $foregroundPid -and "
        "(-not $_.CreationDate -or $_.CreationDate.ToUniversalTime() -le $cutoff) -and "
        "($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }}) -and "
        "($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }})"
        "}};"
        "$records=@($matched | Select-Object ProcessId,Name,CommandLine);"
        "$killed=@();"
        "$matched | ForEach-Object {{"
        "  $procId=$_.ProcessId;"
        "  Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue;"
        "  $killed += [PSCustomObject]@{{ProcessId=$procId;Name=$_.Name;ExitCode=0;CommandLine=$_.CommandLine}};"
        "}};"
        "Start-Sleep -Milliseconds 500;"
        "$report=[PSCustomObject]@{{RequestedAt=(Get-Date).ToString('s');QueueStopDirs=$queues;StopMarkers=$stopMarkers;OwnedRoots=$roots;Matched=$records;Killed=$killed}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(
        ps_markers,
        ps_roots,
        ps_queues,
        ps_stop_markers,
        str(stop_log).replace("'", "''"),
        int(os.getpid()),
        cutoff_utc,
    )
    subprocess.Popen(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Stop request submitted for integration background processes. Queue stop markers: {}. Details: {}".format(len(stopped_queues), stop_log))
    return 0


def parse_axes(value):
    result = [int(part.strip()) for part in value.split(",")]
    if sorted(result) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("axes must be a permutation like 0,1,2")
    return result


def parse_flips(value):
    parts = [part.strip().lower() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("flips must have three values")
    return [part in ("1", "true", "yes") for part in parts]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-import", help="Prepare dataset import and optionally create .mcs in background Mimics")
    p.add_argument("--ts-root", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--cases")
    p.add_argument("--masks", default="all", help="all, none, or comma-separated mask names")
    p.add_argument("--python")
    p.add_argument("--mimics-exe")
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument("--no-create-mcs", action="store_true")
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_prepare_import)

    p = sub.add_parser("export-labels", help="Export labels from saved .mcs files in background Mimics")
    p.add_argument("--ts-root", required=True)
    p.add_argument("--cases")
    p.add_argument("--mimics-exe")
    p.add_argument("--mcs-dir", help="Folder containing the saved .mcs projects")
    p.add_argument("--masks", default="all", help="all or comma-separated mask names")
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="Safe export root; writes <root>/<case>/segmentations without overwriting")
    destination.add_argument("--overwrite-source", action="store_true", help="Explicitly overwrite <case>/segmentations")
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_export_labels)

    p = sub.add_parser("kill-background", help="Stop integration-created background processes")
    p.set_defaults(func=cmd_kill_background)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
