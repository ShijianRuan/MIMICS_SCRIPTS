#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External worker for single-case import preparation.

Runs outside foreground Mimics to avoid host instability. It prepares one case
through mimics_bridge, publishes a prepared-queue descriptor, and launches
background Mimics for .mcs creation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime_py35"
for candidate in (str(ROOT), str(RUNTIME), str(Path(__file__).resolve().parent)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import runtime_common
import tools.mimics_batch_cli as batch_cli
from resource_locks import FileResourceLock, ResourceLockTimeout, default_resource_lock_dir


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write_status(path: Path, status: str, phase: str, **payload) -> None:
    record = {
        "status": str(status),
        "phase": str(phase),
        "updated_at_epoch": time.time(),
    }
    record.update(payload)
    runtime_common.write_json_atomic(str(path), record)


def _append_log(path: Path, message: str) -> None:
    text = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _stop_requested(stop_path: Path) -> bool:
    return stop_path.is_file()


def _queue_stop_requested(runtime_dir: Path) -> bool:
    return (runtime_dir / "_mcs_queue_stop.json").is_file()


def _emit_queue_stop(runtime_dir: Path, reason: str) -> None:
    payload = {
        "status": "stop_requested",
        "reason": str(reason),
        "requested_at_epoch": time.time(),
    }
    runtime_common.write_json_atomic(str(runtime_dir / "_mcs_queue_stop.json"), payload)


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))
    except Exception:
        return False


def _filter_masks(masks: list[dict], selection: str) -> list[dict]:
    text = str(selection or "all").strip()
    if not text or text.lower() == "all":
        return list(masks)
    if text.lower() in ("none", "no", "off"):
        return []
    wanted = {item.strip().lower() for item in text.split(",") if item.strip()}
    return [
        item for item in masks
        if str(item.get("name") or "").strip().lower() in wanted
    ]


def _background_holder_output(project_root: str, output_dir: Path) -> str:
    holder = runtime_common.active_resource_lock(
        project_root,
        runtime_common.background_mimics_lock_name(str(output_dir)),
    ) or {}
    details = holder.get("details") or {}
    return str(details.get("output_dir") or holder.get("output_dir") or "")


def _write_producer_done(runtime_dir: Path) -> None:
    active = runtime_dir / "_mcs_queue_active.json"
    try:
        if active.is_file():
            active.unlink()
    except OSError:
        pass
    runtime_common.write_json_atomic(
        str(runtime_dir / "_mcs_queue_done.json"),
        {
            "status": "done",
            "completed": 1,
            "failed": 0,
            "updated_at_epoch": time.time(),
        },
    )


def _active_producer_exists(runtime_dir: Path, stale_seconds: float = 180.0) -> bool:
    active_path = runtime_dir / "_mcs_queue_active.json"
    if not active_path.is_file():
        return False
    payload = _read_json(active_path)
    try:
        updated = float(
            payload.get("updated_at_epoch", active_path.stat().st_mtime) or 0.0
        )
    except (OSError, TypeError, ValueError):
        updated = time.time()
    return bool(time.time() - updated <= max(1.0, float(stale_seconds)))


def _withdraw_descriptor(descriptor_path: Path) -> None:
    try:
        if descriptor_path.is_file():
            descriptor_path.unlink()
    except OSError:
        pass


def _wait_for_mcs(
    output_mcs: Path,
    runtime_dir: Path,
    process,
    status_path: Path,
    stop_path: Path,
    log_path: Path,
    case_id: str,
    timeout_seconds: float,
    baseline_mtime_ns: int | None,
    allow_existing: bool,
    descriptor_path: Path,
    project_root: str,
    output_dir: Path,
    mimics_exe: str,
    bridge_python: str,
    status_not_before_epoch: float,
) -> int:
    batch_status_path = runtime_dir / "_mcs_batch_status.json"
    deadline = time.time() + max(60.0, float(timeout_seconds))
    last_size = -1
    stable_count = 0
    stop_forwarded = False

    while time.time() < deadline:
        if (
            _stop_requested(stop_path) or _queue_stop_requested(runtime_dir)
        ) and not stop_forwarded:
            stop_forwarded = True
            _emit_queue_stop(runtime_dir, "Single-case import was stopped by the user.")
            descriptor_was_pending = descriptor_path.is_file()
            _withdraw_descriptor(descriptor_path)
            _write_status(
                status_path,
                "cancelling",
                "stopping_background_mimics",
                completed=0,
                failed=0,
                total=1,
                progress_percent=70,
                case_id=case_id,
            )
            if descriptor_was_pending:
                _write_status(
                    status_path,
                    "cancelled",
                    "cancelled",
                    error="Single-case import was stopped before .mcs creation started.",
                    completed=0,
                    failed=0,
                    total=1,
                    progress_percent=70,
                    case_id=case_id,
                )
                return 0

        if output_mcs.is_file():
            try:
                stat = output_mcs.stat()
                current_size = int(stat.st_size)
                current_mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            except OSError:
                current_size = -1
                current_mtime_ns = -1
            output_is_current = bool(
                (allow_existing and not descriptor_path.exists())
                or baseline_mtime_ns is None
                or current_mtime_ns != baseline_mtime_ns
            )
            if output_is_current and current_size > 0 and current_size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_size = current_size
            if stable_count >= 2:
                _write_status(
                    status_path,
                    "completed",
                    "completed",
                    completed=1,
                    failed=0,
                    total=1,
                    progress_percent=100,
                    case_id=case_id,
                    output_mcs=str(output_mcs),
                )
                _append_log(log_path, "single-case import completed: {0}".format(output_mcs))
                return 0

        batch_status = _read_json(batch_status_path)
        batch_state = str(batch_status.get("status") or "").lower()
        try:
            batch_status_is_current = float(batch_status.get("updated_at_epoch", 0.0) or 0.0) >= float(status_not_before_epoch)
        except (TypeError, ValueError):
            batch_status_is_current = False
        batch_case_id = str(batch_status.get("case_id") or "")
        failed_case_matches = not batch_case_id or batch_case_id == case_id
        batch_terminal = batch_state in ("closed", "failed", "error", "cancelled", "canceled")
        failed_descriptor_consumed = batch_terminal and not descriptor_path.exists()
        if batch_status_is_current and (
            batch_state in ("failed", "error")
            or int(batch_status.get("failed", 0) or 0) > 0
        ) and (failed_case_matches or failed_descriptor_consumed):
            error = str(batch_status.get("error") or "Background Mimics could not create the .mcs file.")
            _withdraw_descriptor(descriptor_path)
            _write_status(
                status_path,
                "failed",
                "create_mcs_failed",
                error=error,
                completed=0,
                failed=1,
                total=1,
                progress_percent=70,
                case_id=case_id,
            )
            return 1
        if (
            batch_status_is_current
            and batch_state in ("cancelled", "canceled")
            and not output_mcs.is_file()
        ):
            _withdraw_descriptor(descriptor_path)
            _write_status(
                status_path,
                "cancelled",
                "cancelled",
                error="Single-case import was stopped before the .mcs file was completed.",
                completed=0,
                failed=0,
                total=1,
                progress_percent=70,
                case_id=case_id,
            )
            return 0

        # If this worker joined an existing queue and that owner disappeared
        # before consuming our descriptor, restart the creator for the pending
        # work instead of leaving the GUI waiting until timeout.
        if (
            process is None
            and descriptor_path.exists()
            and not _background_holder_output(project_root, output_dir)
            and not stop_forwarded
        ):
            try:
                process = batch_cli.launch_create_mcs(
                    output_dir,
                    mimics_exe,
                    bridge_python,
                    0.0,
                )
                _append_log(
                    log_path,
                    "restarted background Mimics for an unconsumed queue item; pid={0}".format(
                        getattr(process, "pid", "?")
                    ),
                )
            except ResourceLockTimeout:
                pass
            except Exception as exc:
                _withdraw_descriptor(descriptor_path)
                _write_status(
                    status_path,
                    "failed",
                    "background_restart_failed",
                    error=str(exc),
                    completed=0,
                    failed=1,
                    total=1,
                    progress_percent=70,
                    case_id=case_id,
                )
                return 1

        if process is not None and process.poll() is not None and not output_mcs.is_file():
            error = (
                "Background Mimics exited before creating the .mcs file "
                "(exit code {0}). Review {1}."
            ).format(process.returncode, str(runtime_dir / "_background_mimics.log"))
            _withdraw_descriptor(descriptor_path)
            _write_status(
                status_path,
                "failed",
                "background_mimics_exited",
                error=error,
                completed=0,
                failed=1,
                total=1,
                progress_percent=70,
                case_id=case_id,
            )
            return 1
        time.sleep(0.5)

    _emit_queue_stop(runtime_dir, "Single-case import timed out.")
    _withdraw_descriptor(descriptor_path)
    _write_status(
        status_path,
        "failed",
        "create_mcs_timeout",
        error="Timed out while waiting for the background Mimics process to create the .mcs file.",
        completed=0,
        failed=1,
        total=1,
        progress_percent=70,
        case_id=case_id,
    )
    return 1


def run(selection_path: Path, status_path: Path, stop_path: Path, log_path: Path) -> int:
    selection = _read_json(selection_path)
    source_path = str(selection.get("source_path") or "").strip()
    output_dir = str(selection.get("output_path") or "").strip()
    project_root = str(selection.get("project_root") or ROOT)

    if not source_path or not output_dir:
        _write_status(status_path, "failed", "invalid_selection", error="Missing source_path or output_path in selection JSON.")
        return 2

    case_info = selection.get("case_info") or {}
    case_id = str(case_info.get("case_id") or "case")
    if not case_info:
        _write_status(status_path, "failed", "invalid_selection", error="Selection payload did not include case_info.")
        return 2

    mask_selection = str(selection.get("mask_selection") or "all").strip().lower()
    masks = _filter_masks(list(case_info.get("masks") or []), mask_selection)
    if mask_selection not in ("all", "none", "no", "off") and not masks:
        _write_status(
            status_path,
            "failed",
            "no_matching_masks",
            error="No segmentation file matched: {0}.".format(mask_selection),
            completed=0,
            failed=1,
            total=1,
        )
        return 2

    output_dir_path = Path(output_dir).resolve()
    run_root = status_path.parent
    work_dir = run_root / "work" / case_id
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(project_root, str(output_dir_path)))
    queue_dir = runtime_dir / "prepared_queue"
    output_mcs = output_dir_path / (case_id + ".mcs")
    timeout_seconds = float(selection.get("timeout_seconds") or 7200.0)
    baseline_mtime_ns = None
    if output_mcs.is_file():
        try:
            baseline_stat = output_mcs.stat()
            baseline_mtime_ns = int(
                getattr(baseline_stat, "st_mtime_ns", int(baseline_stat.st_mtime * 1e9))
            )
        except OSError:
            baseline_mtime_ns = None

    lock_digest = hashlib.sha1(str(output_mcs).lower().encode("utf-8", "replace")).hexdigest()[:20]
    case_lock = FileResourceLock(
        default_resource_lock_dir(Path(project_root)) / ("import_case_{0}.lock".format(lock_digest)),
        "single_case_import",
        "single-case import {0}".format(case_id),
    )
    try:
        case_lock.acquire(wait_seconds=0.0)
    except ResourceLockTimeout as exc:
        _write_status(
            status_path,
            "failed",
            "case_already_running",
            error="This case is already being imported: {0}".format(exc),
            completed=0,
            failed=1,
            total=1,
        )
        return 75

    producer_lock = FileResourceLock(
        Path(runtime_common.import_producer_lock_path(project_root, str(output_dir_path))),
        "import_producer",
        "single-case preparation for {0}".format(output_dir_path),
    )
    producer_lock_held = False
    try:
        try:
            producer_lock.acquire(wait_seconds=0.0)
            producer_lock_held = True
            if not producer_lock.update_pid(
                os.getpid(),
                kind="prepare_import",
                output_dir=str(output_dir_path),
            ):
                raise RuntimeError("Import producer lock ownership changed during startup.")
            batch_cli.clear_stale_import_queue_stop(output_dir_path)
        except ResourceLockTimeout as exc:
            _write_status(
                status_path,
                "failed",
                "output_queue_busy",
                error="Another import is preparing data for this output folder: {0}".format(exc),
                completed=0,
                failed=1,
                total=1,
            )
            return 75

        _append_log(log_path, "single-case worker started")
        if _active_producer_exists(runtime_dir):
            error = (
                "Another import is still preparing cases for this output folder. "
                "Wait for that import to finish or stop its queue before starting this single case."
            )
            _write_status(
                status_path,
                "failed",
                "output_queue_busy",
                error=error,
                completed=0,
                failed=1,
                total=1,
                progress_percent=1,
                case_id=case_id,
            )
            return 75
        _write_status(
            status_path,
            "running",
            "preparing",
            completed=0,
            failed=0,
            total=1,
            progress_percent=5,
            case_id=case_id,
        )

        if _stop_requested(stop_path):
            _write_status(status_path, "cancelled", "cancelled", error="Stop requested before preparation started.")
            return 0

        bridge_python = batch_cli.resolve_bridge_python(None)
        params = {
            "action": "prepare",
            "image_path": str(case_info.get("image") or source_path),
            "masks": masks,
            "dicom_out": str(work_dir / "derived_dicom"),
            "buffers_out": str(work_dir / "buffers"),
            "axes": selection.get("axes") or [0, 1, 2],
            "flips": selection.get("flips") or [False, False, False],
            "case_id": case_id,
            "case_dir": str(case_info.get("case_dir") or ""),
        }

        try:
            result = batch_cli.run_bridge(bridge_python, params)
        except Exception as exc:
            _append_log(log_path, "prepare failed: {0}".format(exc))
            _write_status(status_path, "failed", "prepare_failed", error=str(exc), completed=0, failed=1, total=1, case_id=case_id)
            return 1

        if _stop_requested(stop_path) or _queue_stop_requested(runtime_dir):
            _write_status(
                status_path,
                "cancelled",
                "cancelled",
                error="Import stop requested before the prepared case was queued.",
                completed=0,
                failed=0,
                total=1,
                case_id=case_id,
            )
            return 0

        result["output_mcs"] = str(output_mcs)
        runtime_common.write_json_atomic(str(work_dir / "prepare_manifest.json"), result)
        fingerprint_path = runtime_dir / "fingerprints" / (
            runtime_common.safe_filename(case_id) + ".fingerprint"
        )
        stored_fingerprint = ""
        try:
            stored_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        allow_existing = bool(
            output_mcs.is_file()
            and result.get("source_fingerprint")
            and stored_fingerprint == result.get("source_fingerprint")
        )

        if _active_producer_exists(runtime_dir):
            error = (
                "Another import is still preparing cases for this output folder. "
                "Wait for that import to finish or stop its queue before starting this single case."
            )
            _write_status(
                status_path,
                "failed",
                "output_queue_busy",
                error=error,
                completed=0,
                failed=1,
                total=1,
                progress_percent=55,
                case_id=case_id,
            )
            return 75

        queue_dir.mkdir(parents=True, exist_ok=True)
        descriptor = queue_dir / (runtime_common.safe_filename(case_id) + "_" + uuid.uuid4().hex + ".json")
        queue_committed_epoch = time.time()
        runtime_common.write_json_atomic(
            str(descriptor),
            {
                "case_id": case_id,
                "work_dir": str(work_dir.resolve()),
                "output_mcs": str(output_mcs),
                "created_at_epoch": time.time(),
            },
        )

        runtime_dir.mkdir(parents=True, exist_ok=True)
        _write_producer_done(runtime_dir)
        producer_lock.release()
        producer_lock_held = False

        if _stop_requested(stop_path):
            _emit_queue_stop(runtime_dir, "Stop requested after preparation.")
            _withdraw_descriptor(descriptor)
            _write_status(status_path, "cancelled", "cancelled", error="Stop requested after preparation.", completed=0, failed=0, total=1, case_id=case_id)
            return 0

        mimics_exe = batch_cli.find_mimics_exe(None)
        if not mimics_exe:
            error = "A separate background Mimics executable was not found. Configure MIMICS_BACKGROUND_EXE."
            _append_log(log_path, error)
            _withdraw_descriptor(descriptor)
            _write_status(status_path, "failed", "background_launch_failed", error=error, completed=0, failed=1, total=1, case_id=case_id)
            return 1

        _write_status(
            status_path,
            "running",
            "waiting_for_background_mimics",
            completed=0,
            failed=0,
            total=1,
            progress_percent=60,
            case_id=case_id,
        )
        proc = None
        launch_deadline = time.time() + min(timeout_seconds, 3600.0)
        while proc is None:
            if _stop_requested(stop_path):
                _emit_queue_stop(runtime_dir, "Stop requested while waiting for background Mimics.")
                _withdraw_descriptor(descriptor)
                _write_status(status_path, "cancelled", "cancelled", error="Stopped while waiting for background Mimics.", completed=0, failed=0, total=1, case_id=case_id)
                return 0
            holder_output = _background_holder_output(project_root, output_dir_path)
            if holder_output and _same_path(holder_output, output_dir_path):
                _append_log(log_path, "joined the existing background Mimics queue for this output folder")
                break
            try:
                proc = batch_cli.launch_create_mcs(output_dir_path, mimics_exe, bridge_python, 0.0)
                _append_log(log_path, "background Mimics launched pid={0}".format(getattr(proc, "pid", "?")))
            except ResourceLockTimeout:
                if time.time() >= launch_deadline:
                    error = "Timed out waiting for another Mimics-Script background Mimics task to release its resource lock."
                    _withdraw_descriptor(descriptor)
                    _write_status(status_path, "failed", "background_mimics_busy_timeout", error=error, completed=0, failed=1, total=1, case_id=case_id)
                    return 1
                time.sleep(2.0)
            except Exception as exc:
                _append_log(log_path, "background Mimics launch failed: {0}".format(exc))
                _withdraw_descriptor(descriptor)
                _write_status(status_path, "failed", "background_launch_failed", error=str(exc), completed=0, failed=1, total=1, case_id=case_id)
                return 1

        _write_status(
            status_path,
            "running",
            "creating_mcs",
            completed=0,
            failed=0,
            total=1,
            progress_percent=70,
            case_id=case_id,
            output_mcs=str(output_mcs),
        )
        return _wait_for_mcs(
            output_mcs,
            runtime_dir,
            proc,
            status_path,
            stop_path,
            log_path,
            case_id,
            timeout_seconds,
            baseline_mtime_ns,
            allow_existing,
            descriptor,
            project_root,
            output_dir_path,
            mimics_exe,
            bridge_python,
            queue_committed_epoch,
        )
    finally:
        if producer_lock_held:
            producer_lock.release()
        case_lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--stop-path", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args(argv)

    selection_path = Path(args.selection_json)
    status_path = Path(args.status_path)
    stop_path = Path(args.stop_path)
    log_path = Path(args.log_path)

    try:
        return int(run(selection_path, status_path, stop_path, log_path))
    except Exception:
        detail = traceback.format_exc()
        try:
            _append_log(log_path, "fatal error:\n{0}".format(detail))
            _write_status(
                status_path,
                "failed",
                "worker_failed",
                error=detail[-3000:],
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
