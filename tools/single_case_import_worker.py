#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External worker for single-case import preparation.

Runs outside foreground Mimics to avoid host instability. It prepares one case
through mimics_bridge, publishes a prepared-queue descriptor, and launches
background Mimics for .mcs creation.
"""

from __future__ import annotations

import argparse
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


def _emit_queue_stop(runtime_dir: Path, reason: str) -> None:
    payload = {
        "status": "stop_requested",
        "reason": str(reason),
        "requested_at_epoch": time.time(),
    }
    runtime_common.write_json_atomic(str(runtime_dir / "_mcs_queue_stop.json"), payload)


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

    masks = list(case_info.get("masks") or [])
    mask_selection = str(selection.get("mask_selection") or "all").strip().lower()
    if mask_selection in ("none", "no", "off"):
        masks = []

    output_dir_path = Path(output_dir).resolve()
    run_root = status_path.parent
    work_dir = run_root / "work" / case_id
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(project_root, str(output_dir_path)))
    queue_dir = runtime_dir / "prepared_queue"
    output_mcs = output_dir_path / (case_id + ".mcs")

    _append_log(log_path, "single-case worker started")
    _write_status(
        status_path,
        "running",
        "preparing",
        completed=0,
        failed=0,
        total=1,
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

    result["output_mcs"] = str(output_mcs)
    runtime_common.write_json_atomic(str(work_dir / "prepare_manifest.json"), result)

    queue_dir.mkdir(parents=True, exist_ok=True)
    descriptor = queue_dir / (runtime_common.safe_filename(case_id) + "_" + uuid.uuid4().hex + ".json")
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
    done_marker = runtime_dir / "_mcs_queue_done.json"
    if done_marker.is_file():
        try:
            done_marker.unlink()
        except Exception:
            pass
    runtime_common.write_json_atomic(
        str(runtime_dir / "_mcs_queue_active.json"),
        {
            "status": "active",
            "output_dir": str(output_dir_path),
            "total_count": 1,
            "updated_at_epoch": time.time(),
        },
    )

    if _stop_requested(stop_path):
        _emit_queue_stop(runtime_dir, "Stop requested after preparation.")
        _write_status(status_path, "cancelled", "cancelled", error="Stop requested after preparation.", completed=0, failed=0, total=1, case_id=case_id)
        return 0

    try:
        mimics_exe = batch_cli.find_mimics_exe(None)
        proc = batch_cli.launch_create_mcs(output_dir_path, mimics_exe, bridge_python, 0.0)
        _append_log(log_path, "background Mimics launched pid={0}".format(getattr(proc, "pid", "?")))
    except Exception as exc:
        _append_log(log_path, "background Mimics launch failed: {0}".format(exc))
        _write_status(
            status_path,
            "failed",
            "background_launch_failed",
            error=str(exc),
            completed=1,
            failed=0,
            total=1,
            case_id=case_id,
        )
        return 1

    _write_status(
        status_path,
        "running",
        "creating_mcs",
        completed=1,
        failed=0,
        total=1,
        case_id=case_id,
        output_mcs=str(output_mcs),
    )
    return 0


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
