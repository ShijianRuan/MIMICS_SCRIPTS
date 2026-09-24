#!/usr/bin/env python3
"""Create, launch, enumerate, and stop managed nnU-Net jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from nnunet_common import (  # noqa: E402
    SCHEMA_VERSION,
    TERMINAL_STATES,
    normalize_request,
    read_json,
    safe_identifier,
    update_status,
    workspace_paths,
    write_json_atomic,
)
from resource_locks import (  # noqa: E402
    process_exists,
    process_matches,
    process_start_marker,
)


def hidden_process_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {"start_new_session": True}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startup,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000),
    }


def _launch(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **hidden_process_kwargs(),
    )


def _status_process_matches(status: dict[str, Any], key: str) -> bool:
    try:
        pid = int(status.get(key) or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    return process_matches(pid, status.get("{}_start_marker".format(key)))


def create_job(request: dict[str, Any]) -> dict[str, Any]:
    values = dict(request or {})
    values.setdefault(
        "job_id",
        "{}_{}_{}".format(
            str(values.get("operation") or "train"),
            time.strftime("%Y%m%dT%H%M%S"),
            uuid.uuid4().hex[:8],
        ),
    )
    values = normalize_request(values)
    paths = workspace_paths(values["workspace"])
    job_dir = paths["jobs"] / safe_identifier(values["job_id"])
    if job_dir.exists():
        raise RuntimeError("nnU-Net job already exists: {}".format(job_dir))
    job_dir.mkdir(parents=True)
    if values["operation"] == "infer" and not str(values.get("output_path") or "").strip():
        values["output_path"] = str(job_dir / "prediction.nii.gz")
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    write_json_atomic(request_path, values)
    write_json_atomic(control_path, {"action": "run", "updated_at_epoch": time.time()})
    status = {
        "schema_version": SCHEMA_VERSION,
        "job_id": values["job_id"],
        "kind": values["operation"],
        "task_id": values.get("task_id", ""),
        "task_name": values.get("task_name", ""),
        "status": "launching",
        "phase": "launching",
        "message": "Starting the nnU-Net background task.",
        "workspace": values["workspace"],
        "job_dir": str(job_dir),
        "request_path": str(request_path),
        "control_path": str(control_path),
        "log_path": str(job_dir / "job.log"),
        "execution_backend": str(values.get("execution_backend") or "local"),
        "remote_profile_id": str(values.get("remote_profile_id") or ""),
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "progress_percent": 0,
    }
    write_json_atomic(status_path, status)
    if status["execution_backend"] == "remote":
        if not status["remote_profile_id"]:
            raise RuntimeError("Remote training server was not selected.")
        spec_path = job_dir / "remote_spec.json"
        write_json_atomic(
            spec_path,
            {
                "schema_version": "mimics_remote_nnunet_spec.v1",
                "kind": "nnunet" if values["operation"] == "train" else "nnunet_infer",
                "job_id": values["job_id"],
                "job_dir": str(job_dir),
                "status_path": str(status_path),
                "request_path": str(request_path),
                "remote_profile_id": status["remote_profile_id"],
            },
        )
        command = [
            sys.executable,
            str(ROOT / "tools" / "remote_training_controller.py"),
            "run",
            "--spec",
            str(spec_path),
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "tools" / "nnunet_pipeline.py"),
            "infer" if values["operation"] == "infer" else "run",
            "--job-dir",
            str(job_dir),
        ]
    try:
        process = _launch(command)
    except Exception as exc:
        update_status(
            status_path,
            status="failed",
            phase="launch_failed",
            error="Could not start nnU-Net background task: {}".format(exc),
            completed_at_epoch=time.time(),
        )
        raise
    update_status(
        status_path,
        launcher_pid=process.pid,
        launcher_start_marker=process_start_marker(process.pid),
        message="nnU-Net task started in an external process.",
    )
    return {
        "job_id": values["job_id"],
        "job_dir": str(job_dir),
        "status_path": str(status_path),
        "control_path": str(control_path),
        "launcher_pid": process.pid,
    }


def reconcile_job_status(status_path: str | Path) -> dict[str, Any]:
    path = Path(status_path)
    status = read_json(path, {}) or {}
    state = str(status.get("status") or "").lower()
    if state in TERMINAL_STATES or state in {"orphaned_remote", "attention_required"}:
        return status
    try:
        age = time.time() - float(
            status.get("updated_at_epoch") or status.get("created_at_epoch") or 0
        )
    except Exception:
        age = 0.0
    if age < 10.0:
        return status
    pids = []
    for key in ("worker_pid", "controller_pid", "launcher_pid"):
        try:
            pid = int(status.get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in pids:
            pids.append(pid)
    if not pids or any(
        _status_process_matches(status, key)
        for key in ("worker_pid", "controller_pid", "launcher_pid")
    ):
        return status
    remote = str(status.get("execution_backend") or "local") == "remote"
    remote_launch_possible = bool(str(status.get("remote_job_dir") or "").strip())
    if remote and remote_launch_possible:
        values = {
            "status": "orphaned_remote",
            "phase": "controller_stopped",
            "message": (
                "The local remote-task controller stopped unexpectedly. "
                "Use Stop to clean up the remote container before retrying."
            ),
            "error": "No local nnU-Net controller process is running.",
        }
    else:
        values = {
            "status": "cancelled" if state == "cancelling" else "failed",
            "phase": "controller_stopped",
            "message": (
                "The nnU-Net background process stopped before recording completion."
            ),
            "error": (
                "The remote controller stopped before a remote job was launched."
                if remote
                else "No nnU-Net controller or worker process is running."
            ),
            "completed_at_epoch": time.time(),
        }
    return update_status(path, **values)


def list_jobs(workspace: str | Path) -> list[dict[str, Any]]:
    jobs_dir = workspace_paths(workspace)["jobs"]
    rows = []
    if not jobs_dir.is_dir():
        return rows
    for status_path in jobs_dir.glob("*/status.json"):
        status = reconcile_job_status(status_path)
        if status:
            status["status_path"] = str(status_path)
            rows.append(status)
    rows.sort(
        key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True
    )
    return rows


def _stop_orphaned_local_worker(status_path: Path, status: dict[str, Any]) -> None:
    try:
        worker_pid = int(status.get("worker_pid") or 0)
    except Exception:
        worker_pid = 0
    worker_marker = str(status.get("worker_start_marker") or "").strip()
    if worker_pid <= 0 or not worker_marker:
        return

    controller_alive = False
    for key in ("controller_pid", "launcher_pid"):
        if _status_process_matches(status, key):
            controller_alive = True
            break
    if controller_alive or not process_matches(worker_pid, worker_marker):
        return

    from tools.mimics_label_export import terminate_process_tree

    terminate_process_tree(worker_pid)
    deadline = time.time() + 20.0
    while time.time() < deadline and process_matches(worker_pid, worker_marker):
        time.sleep(0.25)
    if process_matches(worker_pid, worker_marker) and os.name != "nt":
        try:
            os.kill(worker_pid, 9)
        except OSError:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline and process_matches(worker_pid, worker_marker):
            time.sleep(0.1)

    if process_matches(worker_pid, worker_marker):
        update_status(
            status_path,
            status="stopping",
            phase="termination_pending",
            message=(
                "The orphaned nnU-Net worker did not exit yet. Its GPU lock remains "
                "owned by that worker, so another local AI task cannot take the GPU."
            ),
            termination_pending=True,
        )
    else:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="The orphaned nnU-Net worker was stopped and its resources were released.",
            worker_pid=None,
            worker_start_marker=None,
            termination_pending=False,
            completed_at_epoch=time.time(),
        )


def stop_job(status_path: str | Path) -> bool:
    path = Path(status_path).expanduser().resolve()
    status = read_json(path, {}) or {}
    if str(status.get("status") or "").lower() in TERMINAL_STATES:
        return False
    control_path = Path(str(status.get("control_path") or path.parent / "control.json"))
    write_json_atomic(
        control_path,
        {"action": "cancel", "requested_at_epoch": time.time()},
    )
    update_status(
        path,
        status="cancelling",
        phase="cancelling",
        message="Stop requested. Waiting for the worker to release resources.",
    )
    if str(status.get("execution_backend") or "") == "remote":
        _launch(
            [
                sys.executable,
                str(ROOT / "tools" / "remote_training_controller.py"),
                "cancel",
                "--status",
                str(path),
            ]
        )
    else:
        _stop_orphaned_local_worker(path, status)
    return True


def abandon_job(status_path: str | Path) -> bool:
    path = Path(status_path).expanduser().resolve()
    status = read_json(path, {}) or {}
    if str(status.get("execution_backend") or "") != "remote":
        return False
    if str(status.get("status") or "").lower() in TERMINAL_STATES:
        return False
    from tools.remote_training_controller import abandon

    return abandon(path) == 0


def reattach_job(status_path: str | Path) -> bool:
    """Re-launch the remote controller for an orphaned remote task."""
    path = Path(status_path).expanduser().resolve()
    status = read_json(path, {}) or {}
    if str(status.get("execution_backend") or "") != "remote":
        return False
    if not str(status.get("remote_job_dir") or "").strip():
        return False
    state = str(status.get("status") or "").lower()
    if state in TERMINAL_STATES:
        return False
    try:
        process = _launch(
            [
                sys.executable,
                str(ROOT / "tools" / "remote_training_controller.py"),
                "reattach",
                "--status",
                str(path),
            ]
        )
    except Exception as exc:
        update_status(
            path,
            status="orphaned_remote",
            phase="reattach_failed",
            error="Could not start the re-attach controller: {}".format(exc),
        )
        raise
    update_status(
        path,
        status="reattaching",
        phase="reattaching_remote",
        message="Re-attaching to the remote container in an external process.",
        launcher_pid=process.pid,
        controller_pid=process.pid,
        remote_state_unknown=False,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--status", required=True)
    abandon_parser = sub.add_parser("abandon")
    abandon_parser.add_argument("--status", required=True)
    reattach_parser = sub.add_parser("reattach")
    reattach_parser.add_argument("--status", required=True)
    args = parser.parse_args()
    if args.command == "abandon":
        abandon_job(args.status)
        return 0
    if args.command == "reattach":
        reattach_job(args.status)
        return 0
    if args.command == "stop":
        stop_job(args.status)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
