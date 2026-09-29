#!/usr/bin/env python3
"""Launch a read-only MCS inspection in a background Mimics instance.

Usage:
    python_env/python.exe tools/inspect_mcs_projects.py \
        --mcs-dir R:/label_task_mr \
        --report E:/MIMICS_SCRIPTS/.mimics_runtime/mcs_inspect_report.json \
        [--cases s0001,s0002] [--limit N] [--timeout-seconds 7200]

Acquires the same background-Mimics resource locks the official append/export
jobs use (so it never races a real import/export job), writes a job config,
launches runtime_py35/inspect_mcs_batch.py inside background Mimics, waits for
the report to reach a terminal status, and prints a summary.  The .mcs files
are only opened and closed - never saved, never modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime_py35"
for path in (str(ROOT), str(ROOT / "tools"), str(RUNTIME)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mimics_batch_cli import (  # noqa: E402
    _acquire_background_mimics_locks,
    find_mimics_exe,
)
import runtime_common  # noqa: E402
import mimics_label_export  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cases", help="Comma-separated case IDs (default: all)")
    parser.add_argument("--limit", type=int, help="Only inspect the first N cases")
    parser.add_argument("--mimics-exe")
    parser.add_argument("--job-dir")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found.", file=sys.stderr)
        return 2

    mcs_root = Path(args.mcs_dir).expanduser().resolve()
    if not mcs_root.is_dir():
        print("MCS directory not found: {}".format(mcs_root), file=sys.stderr)
        return 2
    mcs_files = sorted(mcs_root.glob("*.mcs"))
    if not mcs_files:
        print("No .mcs files found under: {}".format(mcs_root), file=sys.stderr)
        return 2
    requested = None
    if args.cases:
        requested = set(v.strip() for v in args.cases.split(",") if v.strip())
        missing = sorted(requested - {p.stem for p in mcs_files})
        if missing:
            print("Requested cases not found: {}".format(", ".join(missing)), file=sys.stderr)
            return 2
        mcs_files = [p for p in mcs_files if p.stem in requested]
    if args.limit:
        mcs_files = mcs_files[: args.limit]

    stamp = time.strftime("%Y%m%dT%H%M%S")
    job_dir = (
        Path(args.job_dir).resolve() if args.job_dir
        else ROOT / ".mimics_runtime" / "inspect_jobs" / stamp
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = job_dir / "inspect_config.json"
    runner_path = job_dir / "run_inspect.py"
    config_path.write_text(
        __import__("json").dumps({
            "schema_version": "inspect_mcs_job.v1",
            "job_dir": str(job_dir),
            "report_path": str(report_path),
            "cases": [
                {"case_id": p.stem, "mcs_path": str(p)} for p in mcs_files
            ],
        }, indent=2),
        encoding="utf-8",
    )
    runner_path.write_text(
        "\n".join([
            "# Auto-generated read-only MCS inspection runner",
            "import sys",
            "sys.path.insert(0, r'{}')".format(str(ROOT / "runtime_py35")),
            "import inspect_mcs_batch",
            "_result = inspect_mcs_batch.main(r'{}')".format(str(config_path)),
            "if _result:\n    raise SystemExit(_result)",
            "",
        ]),
        encoding="utf-8",
    )

    scopes = [mcs_root, report_path.parent]
    locks = _acquire_background_mimics_locks(
        "read-only MCS inspection",
        scopes,
        args.background_mimics_lock_timeout_seconds,
    )
    log = open(str(job_dir / "process.log"), "ab")
    mimics_log = job_dir / "mimics_application.log"
    try:
        proc = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe, str(runner_path), mimics_log_path=str(mimics_log)
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for lock in locks:
            lock.update_pid(
                proc.pid,
                kind="inspect_mcs",
                mcs_source=str(mcs_root),
                report_path=str(report_path),
            )
    except Exception:
        for lock in locks:
            try:
                lock.release()
            except Exception:
                pass
        raise
    finally:
        log.close()

    print("Background Mimics started for inspection, PID={}.".format(proc.pid))
    print("Report: {}".format(report_path))
    print("Job directory: {}".format(job_dir))

    deadline = time.time() + args.timeout_seconds
    last_status = None
    try:
        while True:
            if proc.poll() is not None:
                break
            payload = mimics_label_export.read_json(str(report_path), {}) or {}
            status = payload.get("status")
            if status != last_status and status == "running":
                print(
                    "  progress: {}/{} done, {} failed".format(
                        payload.get("completed", 0),
                        payload.get("total", 0),
                        payload.get("failed", 0),
                    ),
                    flush=True,
                )
                last_status = status
            if time.time() > deadline:
                print("Timed out waiting for the inspection job.", file=sys.stderr)
                mimics_label_export.terminate_process_tree(proc.pid)
                return 75
            time.sleep(5.0)
    finally:
        pass

    payload = mimics_label_export.read_json(str(report_path), {}) or {}
    print(
        "Inspection {status}: completed={completed}, failed={failed}, total={total}".format(
            status=payload.get("status", "unknown"),
            completed=payload.get("completed", "?"),
            failed=payload.get("failed", "?"),
            total=payload.get("total", "?"),
        )
    )
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
