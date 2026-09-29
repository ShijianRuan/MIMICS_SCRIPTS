#!/usr/bin/env python3
"""Launch per-case check-and-append of missing masks in background Mimics.

Usage:
    python_env/python.exe tools/sync_missing_masks.py \
        --mcs-dir R:/label_task_mr \
        --source-root "Z:/ImageAnalysisData/1-CT/Segmentation/data/TotalsegmentatorMRI_dataset_v200" \
        [--cases s0001,s0002] [--in-place | --output-dir DIR] [--dry-run]

For every case the background Mimics job opens the .mcs, lists the existing
Mask names, and immediately appends every source mask that is missing - one
open/append/save cycle per case, no scan-everything-first pass.  Source masks
come from <source-root>/<case>/segmentations/*.nii.gz.  With --dry-run only
the case list is validated and nothing is launched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime_py35"
for path in (str(ROOT), str(ROOT / "tools"), str(RUNTIME)):
    if path not in sys.path:
        sys.path.insert(0, path)

import mimics_label_export  # noqa: E402
from mimics_batch_cli import (  # noqa: E402
    _acquire_background_mimics_locks,
    find_mimics_exe,
)
import runtime_common  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cases", help="Comma-separated case IDs (default: all)")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--in-place", action="store_true",
                             help="Atomically replace each source .mcs after a successful save")
    destination.add_argument("--output-dir", help="Write updated .mcs files to this folder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing output MCS files")
    parser.add_argument("--job-dir", help="Job directory (default: local .mimics_runtime)")
    parser.add_argument("--scratch-root",
                        help="Buffer scratch root (default: <job-dir>/work)")
    parser.add_argument("--mimics-exe")
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    return parser


def _source_case_root(source_root: Path, case_id: str) -> Path:
    for candidate in (source_root / case_id / "segmentations", source_root / case_id):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Source case directory not found for {}: {}".format(case_id, source_root / case_id)
    )


def _source_masks(source_root: Path, case_id: str) -> list[dict]:
    seg_dir = _source_case_root(source_root, case_id)
    masks = []
    for path in sorted(seg_dir.glob("*.nii.gz")):
        name = path.name[:-7]
        if name.lower().endswith(".nii"):
            name = name[:-4]
        masks.append({"name": name, "mask_path": str(path)})
    if not masks:
        raise RuntimeError("No source masks found under: {}".format(seg_dir))
    return masks


def _read_source_foreground(mask_path: str) -> int:
    raise NotImplementedError  # unused; kept out deliberately

def main(argv=None):
    args = build_parser().parse_args(argv)
    mcs_root = Path(args.mcs_dir).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    if not mcs_root.is_dir():
        print("MCS directory not found: {}".format(mcs_root), file=sys.stderr)
        return 2
    if not source_root.is_dir():
        print("Source root not found: {}".format(source_root), file=sys.stderr)
        return 2
    mcs_files = sorted(mcs_root.glob("*.mcs"))
    if not mcs_files:
        print("No .mcs files found under: {}".format(mcs_root), file=sys.stderr)
        return 2
    if args.cases:
        requested = [v.strip() for v in args.cases.split(",") if v.strip()]
        missing = sorted(set(requested) - {p.stem for p in mcs_files})
        if missing:
            print("Requested cases not found: {}".format(", ".join(missing)), file=sys.stderr)
            return 2
        mcs_files = [p for p in mcs_files if p.stem in set(requested)]

    cases = []
    for mcs_path in mcs_files:
        case_id = mcs_path.stem
        source_masks = _source_masks(source_root, case_id)
        output_mcs = (
            str(mcs_path) if args.in_place
            else str(Path(args.output_dir).expanduser().resolve() / (case_id + ".mcs"))
        )
        cases.append({
            "case_id": case_id,
            "mcs_path": str(mcs_path),
            "output_mcs_path": output_mcs,
            "source_masks": source_masks,
        })

    if args.dry_run:
        print("{} case(s) would be checked; existing masks are listed live per case.".format(len(cases)))
        return 0

    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found.", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%S")
    job_dir = (
        Path(args.job_dir).expanduser().resolve() if args.job_dir
        else ROOT / ".mimics_runtime" / "sync_missing_masks_jobs" / stamp
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "sync_config.json"
    runner_path = job_dir / "run_sync.py"
    status_path = job_dir / "status.json"
    stop_path = job_dir / "stop.json"
    config_path.write_text(json.dumps({
        "schema_version": "sync_missing_masks_job.v1",
        "job_dir": str(job_dir),
        "cases": cases,
        "force": bool(args.force),
        "status_path": str(status_path),
        "stop_path": str(stop_path),
    }, indent=2), encoding="utf-8")
    runner_path.write_text(
        "\n".join([
            "# Auto-generated per-case check-and-append runner",
            "import sys",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "import sync_missing_masks_batch",
            "_result = sync_missing_masks_batch.main(r'{}')".format(str(config_path)),
            "if _result:\n    raise SystemExit(_result)",
            "",
        ]),
        encoding="utf-8",
    )

    scopes = [mcs_root]
    if not args.in_place:
        scopes.append(Path(args.output_dir).expanduser().resolve())
    scopes.append(source_root)
    locks = _acquire_background_mimics_locks(
        "sync missing masks",
        scopes,
        args.background_mimics_lock_timeout_seconds,
    )
    log = open(str(job_dir / "process.log"), "ab")
    mimics_log = job_dir / "mimics_application.log"
    # Per-case scratch: batches of 5 buffers peak at ~1.5 GB; the job drive
    # must have that much headroom or a --scratch-root on a roomier drive
    # should be passed.  Wiped per case either way.
    scratch_root = (
        Path(args.scratch_root).expanduser().resolve() if args.scratch_root
        else job_dir / "work"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MIMICS_SYNC_SCRATCH_ROOT"] = str(scratch_root)
    try:
        proc = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe, str(runner_path), mimics_log_path=str(mimics_log)
            ),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for lock in locks:
            lock.update_pid(
                proc.pid,
                kind="sync_missing_masks",
                mcs_source=str(mcs_root),
                status_path=str(status_path),
                stop_path=str(stop_path),
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

    print("Background Mimics started for mask sync, PID={}.".format(proc.pid))
    print("Job directory: {}".format(job_dir))
    print("Stop file: {}".format(stop_path))
    if args.in_place:
        print("In-place update: each .mcs is atomically replaced after a successful save.")

    deadline = time.time() + args.timeout_seconds
    last_index = -1
    while True:
        if proc.poll() is not None:
            break
        payload = mimics_label_export.read_json(str(status_path), {}) or {}
        index = int(payload.get("index", 0) or 0)
        if index != last_index:
            print(
                "  [{}/{}] {} (completed={}, failed={}, skipped={})".format(
                    index, payload.get("total", "?"), payload.get("case_id", ""),
                    payload.get("completed", 0), payload.get("failed", 0),
                    payload.get("skipped", 0),
                ),
                flush=True,
            )
            last_index = index
        if str(payload.get("status")) in ("completed", "completed_with_errors", "cancelled"):
            break
        if time.time() > deadline:
            print("Timed out waiting for the sync job.", file=sys.stderr)
            mimics_label_export.terminate_process_tree(proc.pid)
            return 75
        time.sleep(args.poll_seconds)

    payload = mimics_label_export.read_json(str(status_path), {}) or {}
    print(
        "Sync {status}: completed={completed}, skipped={skipped}, failed={failed}".format(
            status=payload.get("status", "unknown"),
            completed=payload.get("completed", "?"),
            skipped=payload.get("skipped", "?"),
            failed=payload.get("failed", "?"),
        )
    )
    failed_cases = payload.get("failed_cases") or []
    for row in failed_cases[:20]:
        print("  FAILED {}: {}".format(row.get("case_id"), row.get("error")))
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
