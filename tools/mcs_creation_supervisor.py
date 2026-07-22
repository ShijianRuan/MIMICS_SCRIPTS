#!/usr/bin/env python3
"""Keep a prepared .mcs queue moving across background Mimics crashes.

This process runs outside Mimics. Python exceptions are handled per case by
create_mcs_batch.py. If Mimics itself terminates natively, the current-case
marker identifies only that case for quarantine, then a fresh Mimics process
continues the remaining queue.
"""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import runtime_common


CURRENT_CASE_FILE = "_mcs_current_case.json"
STATUS_FILE = "_mcs_batch_status.json"
STOP_FILE = "_mcs_queue_stop.json"
DONE_FILE = "_mcs_queue_done.json"
ACTIVE_FILE = "_mcs_queue_active.json"


def _write_status(runtime_dir, status, **details):
    payload = runtime_common.read_json(str(runtime_dir / STATUS_FILE), {}) or {}
    payload.update(details)
    payload.update({
        "status": status,
        "supervisor_pid": os.getpid(),
        "updated_at_epoch": time.time(),
    })
    runtime_common.write_json_atomic(str(runtime_dir / STATUS_FILE), payload)


def _preparation_progress(runtime_dir):
    active = runtime_common.read_json(str(runtime_dir / ACTIVE_FILE), {}) or {}
    done = runtime_common.read_json(str(runtime_dir / DONE_FILE), {}) or {}
    source = active if str(active.get("status") or "").lower() == "active" else done
    completed = int(source.get("completed", 0) or 0)
    failed = int(source.get("failed", 0) or 0)
    total = int(
        source.get("total", 0)
        or source.get("total_count", 0)
        or (completed + failed)
        or 0
    )
    return {"completed": completed, "failed": failed, "total": total}


def _record_failure(runtime_dir, marker, exit_code):
    try:
        failed_dir = runtime_dir / "_failed_cases"
        failed_dir.mkdir(parents=True, exist_ok=True)
        case_id = str(marker.get("case_id") or "unknown")
        payload = {
            "case_id": case_id,
            "phase": "create_mcs_native_exit",
            "error": (
                "Background Mimics exited while creating this case "
                "(exit code {}). The case was isolated so the batch could continue."
            ).format(exit_code),
            "worker_pid": int(marker.get("worker_pid") or 0),
            "failed_at_epoch": time.time(),
        }
        path = failed_dir / "{}_create_mcs_native_exit_{}.json".format(
            runtime_common.safe_filename(case_id), uuid.uuid4().hex[:8]
        )
        runtime_common.write_json_atomic(str(path), payload)
        return str(path)
    except Exception as exc:
        print(
            "Warning: native-exit failure details could not be written: {}".format(exc),
            file=sys.stderr,
            flush=True,
        )
        return ""


def _remove_path(path, is_dir=False, retries=12):
    if not path:
        return True
    path = str(path)
    for attempt in range(max(1, int(retries))):
        try:
            if is_dir:
                shutil.rmtree(path, ignore_errors=True)
                if not os.path.exists(path):
                    return True
            elif os.path.isfile(path):
                os.remove(path)
            if not os.path.exists(path):
                return True
        except OSError:
            pass
        if attempt + 1 < max(1, int(retries)):
            time.sleep(min(0.25, 0.025 * (attempt + 1)))
    return not os.path.exists(path)


def quarantine_interrupted_case(runtime_dir, child_pid, exit_code):
    """Quarantine only a marker written by the child that just exited."""
    marker_path = runtime_dir / CURRENT_CASE_FILE
    marker = runtime_common.read_json(str(marker_path), {}) or {}
    if not marker:
        return None
    marker_pid = int(marker.get("worker_pid") or 0)
    if child_pid and marker_pid and marker_pid != int(child_pid):
        return None

    status = runtime_common.read_json(str(runtime_dir / STATUS_FILE), {}) or {}
    creation_completed = max(
        int(status.get("creation_completed", status.get("completed", 0)) or 0),
        int(marker.get("completed_before_case", 0) or 0),
    )
    creation_failed = max(
        int(status.get("creation_failed", status.get("failed", 0)) or 0),
        int(marker.get("failed_before_case", 0) or 0),
    ) + 1
    preparation = _preparation_progress(runtime_dir)
    _record_failure(runtime_dir, marker, exit_code)
    _remove_path(marker.get("descriptor_path"))
    _remove_path(marker.get("staging_mcs"))
    # Removing the manifest first makes the descriptor non-runnable even when
    # antivirus software temporarily prevents deleting the whole work tree.
    if marker.get("work_dir"):
        _remove_path(Path(marker.get("work_dir")) / "prepare_manifest.json")
    _remove_path(marker.get("work_dir"), is_dir=True)
    _remove_path(marker_path)
    try:
        _write_status(
            runtime_dir,
            "recovering",
            phase="recovering_after_case_crash",
            case_id=str(marker.get("case_id") or "unknown"),
            completed=creation_completed,
            failed=creation_failed + int(preparation.get("failed", 0) or 0),
            creation_completed=creation_completed,
            creation_failed=creation_failed,
            preparation_failed=int(preparation.get("failed", 0) or 0),
            total=int(preparation.get("total", 0) or 0),
            error=(
                "A background Mimics process stopped while creating case {}. "
                "That case was isolated; remaining cases will continue."
            ).format(marker.get("case_id") or "unknown"),
        )
    except Exception as exc:
        # Telemetry must not strand all later cases. The descriptor/work item
        # has already been quarantined, so a fresh worker can safely continue.
        print(
            "Warning: recovery status could not be written; continuing the queue: {}".format(exc),
            file=sys.stderr,
            flush=True,
        )
    return marker


def _queue_has_work(runtime_dir, output_dir):
    queue_dir = runtime_dir / "prepared_queue"
    try:
        for path in queue_dir.iterdir():
            if path.suffix.lower() != ".json":
                continue
            descriptor = runtime_common.read_json(str(path), {}) or {}
            work_dir = str(descriptor.get("work_dir") or "")
            if work_dir and os.path.isfile(
                os.path.join(work_dir, "prepare_manifest.json")
            ):
                return True
    except OSError:
        pass
    for root in (runtime_dir, output_dir):
        try:
            for path in root.iterdir():
                if path.is_dir() and path.name.endswith("_work"):
                    if (path / "prepare_manifest.json").is_file():
                        return True
        except OSError:
            continue
    return False


def _producer_active(runtime_dir):
    payload = runtime_common.read_json(str(runtime_dir / ACTIVE_FILE), {}) or {}
    if str(payload.get("status") or "").lower() != "active":
        return False
    try:
        updated = float(payload.get("updated_at_epoch", 0.0) or 0.0)
    except Exception:
        updated = 0.0
    return bool(updated and time.time() - updated <= 180.0)


def _queue_finished(runtime_dir, output_dir):
    return (runtime_dir / DONE_FILE).is_file() and not _queue_has_work(
        runtime_dir, output_dir
    )


def _stop_child(child):
    """Stop the supervised Mimics process without leaving a child behind."""
    if child is None or child.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.call(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        try:
            child.terminate()
        except Exception:
            pass
    try:
        child.wait(timeout=15.0)
    except Exception:
        try:
            child.kill()
            child.wait(timeout=5.0)
        except Exception:
            pass


def _wait_for_child(child, runtime_dir):
    while child.poll() is None:
        if (runtime_dir / STOP_FILE).is_file():
            _stop_child(child)
            return child.poll(), True
        time.sleep(1.0)
    return child.returncode, False


def _run(args):
    runtime_dir = Path(args.runtime_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    command = runtime_common.background_mimics_command(
        args.mimics_exe,
        args.runner,
        mimics_log_path=args.mimics_log or None,
    )
    consecutive_start_failures = 0

    while True:
        if (runtime_dir / STOP_FILE).is_file():
            _write_status(runtime_dir, "cancelled", error="Import queue stopped by user request.")
            return 0

        handshake = Path(args.handshake) if args.handshake else None
        if handshake:
            _remove_path(handshake)
        started_at = time.time()
        print("Starting background Mimics queue worker.", flush=True)
        child = None
        launch_error = None
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
            exit_code, stopped = _wait_for_child(child, runtime_dir)
        except Exception as exc:
            launch_error = exc
            exit_code = "launch_error: {}".format(exc)
            stopped = False
        runtime_seconds = time.time() - started_at

        if stopped or (runtime_dir / STOP_FILE).is_file():
            _write_status(runtime_dir, "cancelled", error="Import queue stopped by user request.")
            return 0

        status = runtime_common.read_json(str(runtime_dir / STATUS_FILE), {}) or {}
        state = str(status.get("status") or "").lower()
        if state in ("closed", "cancelled") and not _queue_has_work(runtime_dir, output_dir):
            return 0

        recovered = quarantine_interrupted_case(
            runtime_dir, getattr(child, "pid", 0), exit_code
        )
        if recovered:
            consecutive_start_failures = 0
            print(
                "Background Mimics stopped during case {}; continuing the remaining queue.".format(
                    recovered.get("case_id") or "unknown"
                ),
                flush=True,
            )
            continue

        if _queue_finished(runtime_dir, output_dir) and not _producer_active(runtime_dir):
            preparation = _preparation_progress(runtime_dir)
            _write_status(
                runtime_dir,
                "closed",
                completed=int(status.get("completed", 0) or 0),
                failed=int(status.get("failed", 0) or 0),
                total=int(preparation.get("total", 0) or status.get("total", 0) or 0),
            )
            return 0

        consecutive_start_failures += 1
        if consecutive_start_failures >= max(1, int(args.max_start_retries)):
            reason = (
                "Background Mimics exited before a case could be identified after {} attempts "
                "(last result {}, runtime {:.1f}s). Prepared cases remain queued."
            ).format(
                consecutive_start_failures,
                launch_error or exit_code,
                runtime_seconds,
            )
            _write_status(runtime_dir, "failed", error=reason, exit_code=exit_code)
            print(reason, file=sys.stderr, flush=True)
            return 1

        _write_status(
            runtime_dir,
            "restarting",
            phase="restarting_background_mimics",
            error=(
                "Background Mimics exited before completing the queue; retrying "
                "({}/{})."
            ).format(consecutive_start_failures, args.max_start_retries),
        )
        delay = min(30.0, max(1.0, float(args.retry_delay)) * consecutive_start_failures)
        time.sleep(delay)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimics-exe", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--handshake", default="")
    parser.add_argument("--mimics-log", default="")
    parser.add_argument("--max-start-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        try:
            _write_status(
                Path(args.runtime_dir).resolve(),
                "failed",
                error="MCS creation supervisor failed: {}".format(exc),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
