#!/usr/bin/env python3
"""Entrypoint executed inside the unified remote GPU container."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


APP_ROOT = Path(os.environ.get("MIMICS_AI_APP_ROOT") or "/app").resolve()
REMOTE_LOG_MAX_BYTES = max(
    1024 * 1024,
    int(os.environ.get("MIMICS_REMOTE_LOG_MAX_BYTES") or 32 * 1024 * 1024),
)
REMOTE_LOG_BACKUP_COUNT = max(
    1, min(10, int(os.environ.get("MIMICS_REMOTE_LOG_BACKUP_COUNT") or 3))
)


class _RotatingBinaryLog:
    """Bound remote subprocess output while preserving recent diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("ab")

    def _rotate(self) -> None:
        self.handle.close()
        oldest = self.path.with_name(
            "{}.{}".format(self.path.name, REMOTE_LOG_BACKUP_COUNT)
        )
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass
        for index in range(REMOTE_LOG_BACKUP_COUNT - 1, 0, -1):
            source = self.path.with_name("{}.{}".format(self.path.name, index))
            target = self.path.with_name(
                "{}.{}".format(self.path.name, index + 1)
            )
            if source.exists():
                os.replace(str(source), str(target))
        if self.path.exists():
            os.replace(str(self.path), str(self.path) + ".1")
        self.handle = self.path.open("ab")

    def write(self, data: bytes) -> None:
        if not data:
            return
        try:
            current = int(self.handle.tell())
        except Exception:
            current = int(self.path.stat().st_size) if self.path.exists() else 0
        if current > 0 and current + len(data) > REMOTE_LOG_MAX_BYTES:
            self._rotate()
        self.handle.write(data)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _append_worker_log(job_dir: Path, text: str) -> None:
    writer = _RotatingBinaryLog(job_dir / "remote_worker.log")
    try:
        writer.write(str(text).encode("utf-8", "replace"))
    finally:
        writer.close()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _worker_status(job_dir: Path, **values: Any) -> None:
    path = job_dir / "worker_status.json"
    payload = _read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    _write_json(path, payload)


def _run(command: list[str], job_dir: Path) -> int:
    log_path = job_dir / "remote_worker.log"
    _worker_status(
        job_dir,
        status="running",
        command=command,
        worker_pid=os.getpid(),
        log_path="/job/remote_worker.log",
    )
    log = _RotatingBinaryLog(log_path)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(APP_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            bufsize=0,
        )
        _worker_status(job_dir, child_pid=process.pid)
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(1024 * 1024)
            if not chunk:
                break
            log.write(chunk)
        returncode = int(process.wait())
    finally:
        log.close()
    _worker_status(
        job_dir,
        status="completed" if returncode == 0 else "failed",
        returncode=returncode,
        completed_at_epoch=time.time(),
    )
    return returncode


@contextlib.contextmanager
def _remote_gpu_lock(job_dir: Path):
    import fcntl

    global_lock_path = Path(
        os.environ.get("MIMICS_REMOTE_GPU_GLOBAL_LOCK")
        or "/remote-locks/gpu-all.lock"
    )
    device_lock_path = Path(
        os.environ.get("MIMICS_REMOTE_GPU_LOCK")
        or "/remote-locks/gpu-automatic.lock"
    )
    scope = str(
        os.environ.get("MIMICS_REMOTE_GPU_LOCK_SCOPE") or "all"
    ).lower()
    gpu_device = str(
        os.environ.get("MIMICS_REMOTE_GPU_DEVICE") or "auto"
    )
    global_lock_path.parent.mkdir(parents=True, exist_ok=True)
    device_lock_path.parent.mkdir(parents=True, exist_ok=True)
    handles = [global_lock_path.open("a+", encoding="utf-8")]
    if scope == "device":
        handles.append(device_lock_path.open("a+", encoding="utf-8"))
    acquired = []
    try:
        while True:
            try:
                fcntl.flock(
                    handles[0].fileno(),
                    (
                        fcntl.LOCK_SH
                        if scope == "device"
                        else fcntl.LOCK_EX
                    )
                    | fcntl.LOCK_NB,
                )
                acquired.append(handles[0])
                break
            except BlockingIOError:
                _worker_status(
                    job_dir,
                    status="waiting_for_remote_gpu",
                    phase="waiting_for_remote_gpu",
                    gpu_device=gpu_device,
                    message="Waiting for the remote GPU scheduler.",
                )
                time.sleep(2.0)
        if scope == "device":
            while True:
                try:
                    fcntl.flock(
                        handles[1].fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    acquired.append(handles[1])
                    break
                except BlockingIOError:
                    _worker_status(
                        job_dir,
                        status="waiting_for_remote_gpu",
                        phase="waiting_for_remote_gpu",
                        gpu_device=gpu_device,
                        message="Waiting for remote GPU {}.".format(gpu_device),
                    )
                    time.sleep(2.0)
        _worker_status(
            job_dir,
            status="starting",
            phase="remote_gpu_acquired",
            gpu_device=gpu_device,
            message="Remote GPU {} acquired.".format(gpu_device),
        )
        try:
            yield
        finally:
            for handle in reversed(acquired):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        for handle in handles:
            handle.close()


def run_nninteractive(job_dir: Path, request: dict[str, Any]) -> int:
    pipeline = APP_ROOT / "tools" / "nninteractive_finetune_pipeline.py"
    if not pipeline.is_file():
        raise RuntimeError(
            "nnInteractive fine-tuning pipeline is missing from the runtime image."
        )
    pipeline_job = job_dir / "pipeline_job"
    pipeline_job.mkdir(parents=True, exist_ok=True)
    pipeline_request = request.get("pipeline_request")
    if not isinstance(pipeline_request, dict):
        raise RuntimeError("Remote nnInteractive training request is missing.")
    _write_json(pipeline_job / "request.json", pipeline_request)
    _write_json(
        pipeline_job / "status.json",
        {
            "schema_version": "nninteractive_task_job.v1",
            "job_id": str(pipeline_request.get("job_id") or job_dir.name),
            "task_id": str(pipeline_request.get("task_id") or ""),
            "task_name": str(pipeline_request.get("task_name") or ""),
            "status": "created",
            "phase": "created",
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    _write_json(
        pipeline_job / "control.json",
        {"action": "run", "updated_at_epoch": time.time()},
    )
    return _run(
        [
            sys.executable,
            str(pipeline),
            "run",
            "--job-dir",
            str(pipeline_job),
        ],
        job_dir,
    )


def run_nnunet(job_dir: Path, request: dict[str, Any]) -> int:
    pipeline = APP_ROOT / "tools" / "nnunet_pipeline.py"
    if not pipeline.is_file():
        raise RuntimeError("nnU-Net pipeline is missing from the runtime image.")
    pipeline_job = job_dir / "pipeline_job"
    pipeline_job.mkdir(parents=True, exist_ok=True)
    pipeline_request = request.get("pipeline_request")
    if not isinstance(pipeline_request, dict):
        raise RuntimeError("Remote nnU-Net request is missing.")
    _write_json(pipeline_job / "request.json", pipeline_request)
    _write_json(
        pipeline_job / "control.json",
        {"action": "run", "updated_at_epoch": time.time()},
    )
    _write_json(
        pipeline_job / "status.json",
        {
            "schema_version": "mimics_nnunet_job.v1",
            "job_id": str(pipeline_request.get("job_id") or job_dir.name),
            "status": "created",
            "phase": "created",
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    command = "infer" if str(pipeline_request.get("operation")) == "infer" else "run"
    return _run(
        [sys.executable, str(pipeline), command, "--job-dir", str(pipeline_job)],
        job_dir,
    )


def preflight(models_dir: Path) -> int:
    import torch

    result: dict[str, Any] = {
        "ok": bool(torch.cuda.is_available()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "models_dir": str(models_dir),
        "nninteractive_model": str(
            models_dir / "nninteractive" / "nnInteractive_v1.0"
        ),
        "offline_environment": {
            key: str(os.environ.get(key) or "")
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "WANDB_MODE",
            )
        },
    }
    result["offline_mode"] = bool(
        result["offline_environment"].get("HF_HUB_OFFLINE") == "1"
        and result["offline_environment"].get("TRANSFORMERS_OFFLINE") == "1"
        and result["offline_environment"].get("HF_DATASETS_OFFLINE") == "1"
        and result["offline_environment"].get("WANDB_MODE") == "offline"
    )
    nninteractive_root = models_dir / "nninteractive" / "nnInteractive_v1.0"
    result["nninteractive_weights"] = bool(
        nninteractive_root.is_dir()
        and any(
            path.is_file()
            for pattern in ("*.pth", "*.safetensors")
            for path in nninteractive_root.rglob(pattern)
        )
    )
    try:
        import nnInteractive  # noqa: F401
        import nninteractive_finetune  # noqa: F401

        result["nninteractive_import"] = True
    except Exception as exc:
        result["nninteractive_import"] = False
        result["nninteractive_error"] = str(exc)
    try:
        import nnunetv2  # noqa: F401

        trainer_root = (
            APP_ROOT
            / "integrations"
            / "nnunet_segmentation_workflow"
            / "trainers"
        )
        if str(trainer_root) not in sys.path:
            sys.path.insert(0, str(trainer_root))
        from MimicsNNUNetTrainer import MimicsNNUNetTrainer  # noqa: F401

        result["nnunet_import"] = True
        result["nnunet_custom_trainer"] = True
    except Exception as exc:
        result["nnunet_import"] = False
        result["nnunet_custom_trainer"] = False
        result["nnunet_error"] = str(exc)
    result["ok"] = bool(
        result["ok"]
        and result.get("offline_mode")
        and result.get("nninteractive_weights")
        and result.get("nninteractive_import")
        and result.get("nnunet_import")
        and result.get("nnunet_custom_trainer")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


def main() -> int:
    os.environ.setdefault("NNINTERACTIVE_ENV_PYTHON", sys.executable)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--job-dir", default="/job")
    check = sub.add_parser("preflight")
    check.add_argument("--models-dir", default="/models")
    args = parser.parse_args()
    if args.command == "preflight":
        return preflight(Path(args.models_dir))
    job_dir = Path(args.job_dir).resolve()
    request = _read_json(job_dir / "remote_request.json", {}) or {}
    kind = str(request.get("kind") or "")
    _worker_status(
        job_dir,
        schema_version="mimics_remote_worker.v1",
        status="starting",
        kind=kind,
        started_at_epoch=time.time(),
    )
    try:
        with _remote_gpu_lock(job_dir):
            if kind == "nninteractive_train":
                return run_nninteractive(job_dir, request)
            if kind in {"nnunet_train", "nnunet_infer"}:
                return run_nnunet(job_dir, request)
            raise RuntimeError("Unsupported remote job kind: {}".format(kind))
    except Exception as exc:
        _worker_status(
            job_dir,
            status="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
        )
        _append_worker_log(job_dir, traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
