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
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=str(APP_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        _worker_status(job_dir, child_pid=process.pid)
        returncode = int(process.wait())
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


def run_dino(job_dir: Path, request: dict[str, Any]) -> int:
    pipeline = APP_ROOT / "tools" / "fewshot_pipeline.py"
    if not pipeline.is_file():
        raise RuntimeError("DINOv3 pipeline is missing from the runtime image.")
    arguments = request.get("pipeline_args") or []
    if not isinstance(arguments, list) or not arguments:
        raise RuntimeError("Remote DINOv3 pipeline arguments are missing.")
    resolved = [
        sys.executable if str(value) == "__REMOTE_PYTHON__" else str(value)
        for value in arguments
    ]
    return _run([sys.executable, str(pipeline)] + resolved, job_dir)


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


def preflight(models_dir: Path) -> int:
    import torch

    result: dict[str, Any] = {
        "ok": bool(torch.cuda.is_available()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "models_dir": str(models_dir),
        "dinov3_models": str(models_dir / "dinov3"),
        "nninteractive_model": str(
            models_dir / "nninteractive" / "nnInteractive_v1.0"
        ),
    }
    try:
        import nnInteractive  # noqa: F401
        import nninteractive_finetune  # noqa: F401

        result["nninteractive_import"] = True
    except Exception as exc:
        result["nninteractive_import"] = False
        result["nninteractive_error"] = str(exc)
    try:
        import onnxruntime
        from transformers import DINOv3ViTBackbone  # noqa: F401

        result["onnx_providers"] = onnxruntime.get_available_providers()
        result["dinov3_import"] = True
    except Exception as exc:
        result["onnx_providers"] = []
        result["dinov3_import"] = False
        result["onnx_error"] = str(exc)
    result["ok"] = bool(
        result["ok"]
        and result.get("nninteractive_import")
        and result.get("dinov3_import")
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
            if kind == "dinov3_train":
                return run_dino(job_dir, request)
            if kind == "nninteractive_train":
                return run_nninteractive(job_dir, request)
            raise RuntimeError("Unsupported remote job kind: {}".format(kind))
    except Exception as exc:
        _worker_status(
            job_dir,
            status="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
        )
        with (job_dir / "remote_worker.log").open(
            "a", encoding="utf-8", errors="replace"
        ) as handle:
            handle.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
