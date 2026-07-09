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

from resource_locks import FileResourceLock, ResourceLockTimeout

BRIDGE = ROOT / "mimics_bridge.py"
RUNTIME = ROOT / "runtime_py35"
RESOURCE_LOCK_DIR = ROOT / ".mimics_runtime" / "locks"
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
    if explicit and Path(explicit).is_file():
        return explicit
    candidates = [
        Path(os.environ.get("MIMICS_EXE", "")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Materialise" / "Mimics Research 21.0" / "MimicsResearch.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Mimics Research 21.0" / "MimicsResearch.exe",
        Path(r"D:\Mimics Research 21.0\MimicsResearch.exe"),
        Path(r"C:\Mimics Research 21.0\MimicsResearch.exe"),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return str(candidate)
    return None


def discover_cases(ts_root, cases, python_exe):
    result = run_bridge(
        python_exe,
        {
            "action": "discover",
            "ts_root": str(ts_root),
            "cases_filter": sorted(cases) if cases else None,
        },
    )
    return list(result.get("cases", []))


def _acquire_background_mimics_lock(owner, wait_seconds=0.0):
    lock = FileResourceLock(BACKGROUND_MIMICS_LOCK_PATH, "background_mimics", owner)
    lock.acquire(wait_seconds=float(wait_seconds), poll_seconds=5.0)
    return lock


def launch_create_mcs(output_dir, mimics_exe, bridge_python, lock_timeout_seconds=0.0):
    runner = output_dir / "_run_create_mcs.py"
    runner.write_text(
        "\n".join([
            "# Auto-generated runner for background Mimics .mcs creation",
            "import sys, os",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "os.environ['MIMICS_BRIDGE_PYTHON'] = r'{}'".format(str(bridge_python)),
            "os.environ['MIMICS_BRIDGE_SCRIPT'] = r'{}'".format(str(ROOT / "mimics_bridge.py")),
            "import create_mcs_batch",
            "create_mcs_batch.main(r'{}')".format(str(output_dir)),
            "",
        ]),
        encoding="utf-8",
    )
    lock = _acquire_background_mimics_lock("batch .mcs creation", lock_timeout_seconds)
    log = open(str(output_dir / "_background_mimics.log"), "ab")
    try:
        proc = subprocess.Popen(
            [mimics_exe, "-b", "-run_script", str(runner)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        lock.update_pid(proc.pid, kind="create_mcs", output_dir=str(output_dir.resolve()))
        return proc
    except Exception:
        lock.release()
        raise
    finally:
        log.close()


def launch_export_labels(ts_root, cases, mimics_exe, axes, flips, lock_timeout_seconds=0.0):
    output_dir = ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "_export_batch_config.json"
    runner = output_dir / "_run_export_batch.py"
    write_json_atomic(
        config,
        {
            "ts_root": str(ts_root),
            "cases": sorted(cases) if cases else None,
            "axes": axes,
            "flips": flips,
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
    log = open(str(output_dir / "_background_export_mimics.log"), "ab")
    try:
        proc = subprocess.Popen(
            [mimics_exe, "-b", "-run_script", str(runner)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        lock.update_pid(proc.pid, kind="export_labels", ts_root=str(ts_root.resolve()))
        return proc
    except Exception:
        lock.release()
        raise
    finally:
        log.close()


def cmd_prepare_import(args):
    ts_root = Path(args.ts_root).resolve()
    bridge_python = resolve_bridge_python(args.python)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_filter = set(args.cases.split(",")) if args.cases else None
    cases = discover_cases(ts_root, cases_filter, bridge_python)
    print("Discovered {} case(s).".format(len(cases)))
    write_json_atomic(output_dir / "_mcs_queue_active.json", {"total": len(cases), "created_at_epoch": time.time()})
    completed = 0
    failed = 0
    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        work_dir = output_dir / (case_id + "_work")
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
            completed += 1
        except Exception as exc:
            failed += 1
            fail_dir = output_dir / "_failed_cases"
            write_json_atomic(fail_dir / (case_id + "_prepare.json"), {"case_id": case_id, "error": str(exc)})
            print("  failed: {}".format(exc))
    active = output_dir / "_mcs_queue_active.json"
    if active.exists():
        active.unlink()
    write_json_atomic(output_dir / "_mcs_queue_done.json", {"completed": completed, "failed": failed, "updated_at_epoch": time.time()})
    if args.no_create_mcs:
        return 0
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("MimicsResearch.exe was not found. Manifests are ready; run .mcs creation later.", file=sys.stderr)
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
        print("MimicsResearch.exe was not found.", file=sys.stderr)
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
    registry = ROOT / ".mimics_runtime" / "mcs_queues"
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
        if not queue_dir.is_dir():
            continue
        try:
            (queue_dir / "_mcs_queue_stop.json").write_text(json.dumps(stop_payload, indent=2, sort_keys=True), encoding="utf-8")
            active = queue_dir / "_mcs_queue_active.json"
            if active.is_file():
                active.unlink()
            stopped_queues.append(str(queue_dir))
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
    lock_paths = [str(BACKGROUND_MIMICS_LOCK_PATH), str(RESOURCE_LOCK_DIR / "gpu.lock")]
    ps_locks = "@(" + ",".join("'{}'".format(str(r).replace("'", "''")) for r in lock_paths) + ")"
    stop_log = ROOT / ".mimics_runtime" / "stop_background_last.json"
    stop_log.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "$markers={};"
        "$roots={};"
        "$queues={};"
        "$locks={};"
        "$out='{}';"
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; "
        "$cmd -and "
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
        "$locks | ForEach-Object {{ Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }};"
        "$report=[PSCustomObject]@{{RequestedAt=(Get-Date).ToString('s');QueueStopDirs=$queues;OwnedRoots=$roots;Matched=$records;Killed=$killed}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(ps_markers, ps_roots, ps_queues, ps_locks, str(stop_log).replace("'", "''"))
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-import", help="Prepare dataset import and optionally create .mcs in background Mimics")
    p.add_argument("--ts-root", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--cases")
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
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_export_labels)

    p = sub.add_parser("kill-background", help="Stop integration-created background processes")
    p.set_defaults(func=cmd_kill_background)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
