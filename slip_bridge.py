# SLIP bridge: runs SLIP (Segmentation with Low-latency Interactive
# Prompting) interactive segmentation on the python_env stack.
#
# Two modes:
# - default:          single-shot, JSON request on stdin -> JSON result on
#                     stdout (one predict command; for tests and debugging)
# - --async-worker:   persistent file-queue worker used by the non-blocking
#                     Mimics mode (see runtime_py35/slip_mimics.py)
#
# SLIP itself is vendored at integrations/slip (unmodified) and its optional
# dependencies (torchio, cupy, ...) live in slip_models/deps, so nothing is
# installed into python_env. Both paths are prepended to sys.path before the
# SLIP import, mirroring the layout proven by the smoke test.
#
# Coordinate contract: the image buffer is the Mimics voxel array in index
# order (Z, Y, X); clicks are [z, y, x] indices into that same array; the
# predicted mask is written back in the same order. SLIP reorders axes
# internally and maps clicks/masks back to the original array itself, so no
# platform buffer-mapping machinery is needed here (unlike nninteractive_bridge).
#
# Protocol schema "slip_async_job.v1" (all files inside job_dir):
#   initialize.json           worker startup request (see _SlipSession)
#   commands/command_NNNNNN.json   {sequence, command_id, action: predict|undo|reset,
#                                   click: [z, y, x], label: 0|1}
#   results/result_NNNNNN.json     {status: refined|reset|error, sequence, ...}
#   results/prediction_NNNNNN.u8   uint8 0/1 mask, one byte per voxel
#   worker_status.json        worker heartbeat (schema slip_worker_status.v1)
#   close.json                {reason} - ask the worker to exit
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Embeddable Python may not include script directory on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from resource_locks import (
    FileResourceLock,
    ResourceLockTimeout,
    default_resource_lock_dir,
    process_exists as resource_process_exists,
    register_process,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "slip_models" / "SLIP_ckpt.pth"
DEFAULT_DEPS_DIR = PROJECT_ROOT / "slip_models" / "deps"
VENDORED_SLIP_ROOT = PROJECT_ROOT / "integrations" / "slip"
JOB_SCHEMA = "slip_async_job.v1"
STATUS_SCHEMA = "slip_worker_status.v1"
LOG_ROTATE_BYTES = 10 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3
# SLIP with propagation needs ~10 GB VRAM (paper recommendation); refuse to
# start when less than this is free so the user gets an actionable message
# instead of a CUDA OOM three minutes into model load.
DEFAULT_MINIMUM_FREE_GPU_MEMORY_GB = 8.0

SUPPORTED_IMAGE_DTYPES = {
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
}


# ---------------------------------------------------------------------------
# Small shared helpers (same conventions as nninteractive_bridge.py)
# ---------------------------------------------------------------------------

def _write_text_atomic(path: Path, text: str, retries: int = 12) -> None:
    """Publish text reliably without deleting a readable previous state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        for attempt in range(max(1, int(retries))):
            try:
                os.replace(temporary, path)
                return
            except OSError:
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(min(0.15, 0.02 * (attempt + 1)))
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(
        path, json.dumps(value, indent=2, ensure_ascii=False, default=str)
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _gpu_lock_path() -> Path:
    """Resolve the GPU lock path per call, honoring env overrides."""
    return default_resource_lock_dir(PROJECT_ROOT) / "gpu.lock"


def _gpu_lock_enabled(device: str) -> bool:
    if os.environ.get("MIMICS_DISABLE_GPU_LOCK", "").strip().lower() in (
        "1", "true", "yes",
    ):
        return False
    return str(device or "").lower().startswith("cuda")


def _check_free_gpu_memory(device: str, minimum_gb: float) -> None:
    """Refuse to start on CUDA when free VRAM is below the floor.

    Runs before the GPU lock so the user gets an actionable message instead
    of a lock wait. Skipped silently when CUDA is unavailable or the probe
    itself errors (never blocks a working start on a probe failure).
    """
    if not str(device or "").lower().startswith("cuda"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < float(minimum_gb):
        raise RuntimeError(
            "Not enough free GPU memory to start the SLIP worker.\n\n"
            "Free: {0:.1f} GB of {1:.1f} GB "
            "(minimum required: {2:.0f} GB).\n\n"
            "Close other GPU programs or stop running AI jobs, "
            "then try again.".format(
                free_gb, total_bytes / (1024 ** 3), float(minimum_gb)
            )
        )


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _load_image_volume(image: dict[str, Any]) -> np.ndarray:
    """Read the exported Mimics voxel buffer and convert GV to HU.

    Returns a float32 (Z, Y, X) array in Mimics index order. SLIP expects
    HU-scale CT input (its preprocessing clamps to [-1000, 1000] then
    z-normalises), so the slope/intercept pair from the Mimics export must
    map grey values to HU.
    """
    if not isinstance(image, dict):
        raise RuntimeError("Request is missing the 'image' object.")
    buffer_path = str(image.get("buffer_path") or "").strip()
    if not buffer_path:
        raise RuntimeError("Request is missing image.buffer_path.")
    shape_value = image.get("shape")
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 3:
        raise RuntimeError(
            "image.shape must be [z, y, x]; got: {0!r}".format(shape_value)
        )
    shape = tuple(int(value) for value in shape_value)
    if min(shape) < 1:
        raise RuntimeError("image.shape must be positive; got: {0}".format(shape))
    dtype_name = str(image.get("dtype") or "").strip().lower()
    if dtype_name not in SUPPORTED_IMAGE_DTYPES:
        raise RuntimeError(
            "Unsupported image dtype {0!r}; expected one of: {1}".format(
                dtype_name, ", ".join(sorted(SUPPORTED_IMAGE_DTYPES))
            )
        )
    dtype = SUPPORTED_IMAGE_DTYPES[dtype_name]
    try:
        raw = Path(buffer_path).read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "Could not read image buffer {0}: {1}".format(buffer_path, exc)
        ) from exc
    expected_bytes = int(np.prod(shape)) * dtype().itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(
            "Image buffer byte count mismatch for shape {0} dtype {1}: "
            "expected {2} bytes, file has {3}.".format(
                shape, dtype_name, expected_bytes, len(raw)
            )
        )
    slope = float(image.get("to_hu_slope", 1.0) or 1.0)
    intercept = float(image.get("to_hu_intercept", 0.0) or 0.0)
    volume = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if slope == 1.0 and intercept == 0.0:
        return np.ascontiguousarray(volume, dtype=np.float32)
    return np.ascontiguousarray(
        volume.astype(np.float32) * slope + intercept, dtype=np.float32
    )


def _write_mask_u8(mask: np.ndarray, output_path: str) -> dict[str, Any]:
    """Write a bool mask as raw uint8 bytes (tmp file + atomic replace)."""
    binary = np.asarray(mask, dtype=bool)
    data = binary.astype(np.uint8).tobytes()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return {"output_path": str(path), "voxels": int(binary.sum())}


def _validate_click(click: Any, label: Any, shape: tuple[int, int, int]) -> list[int]:
    if not isinstance(click, (list, tuple)) or len(click) != 3:
        raise RuntimeError(
            "A predict command needs exactly one click: [z, y, x]; got: {0!r}".format(click)
        )
    values: list[int] = []
    for axis, value in enumerate(click):
        try:
            index = int(value)
        except (TypeError, ValueError):
            raise RuntimeError(
                "Click coordinate {0!r} is not an integer.".format(value)
            ) from None
        if not 0 <= index < shape[axis]:
            raise RuntimeError(
                "Click coordinate {0} is out of bounds for axis {1} "
                "(size {2}).".format(index, axis, shape[axis])
            )
        values.append(index)
    try:
        label_value = int(label)
    except (TypeError, ValueError):
        raise RuntimeError("Click label must be 0 (exclude) or 1 (include).") from None
    if label_value not in (0, 1):
        raise RuntimeError("Click label must be 0 (exclude) or 1 (include).")
    values.append(label_value)
    return values


# ---------------------------------------------------------------------------
# Session: owns the SLIP module, GPU lock, and the redirect log
# ---------------------------------------------------------------------------

class _SlipSession:
    """One SLIP session on one image: load model, embed image, serve clicks."""

    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.device = _resolve_device(request.get("device"))
        self.checkpoint = str(
            request.get("checkpoint_path") or DEFAULT_CHECKPOINT
        )
        self.deps_dir = str(request.get("deps_dir") or DEFAULT_DEPS_DIR)
        self.activate_propagation = bool(request.get("activate_propagation", True))
        self.gpu_lock_timeout_seconds = float(
            request.get("gpu_lock_timeout_seconds", 30.0)
        )
        self.minimum_free_gpu_memory_gb = float(
            request.get(
                "minimum_free_gpu_memory_gb",
                os.environ.get(
                    "SLIP_MINIMUM_FREE_GPU_MEMORY_GB",
                    DEFAULT_MINIMUM_FREE_GPU_MEMORY_GB,
                ),
            )
        )
        log_dir = str(request.get("log_dir") or "").strip()
        self.log_path = (
            Path(log_dir) if log_dir else PROJECT_ROOT / ".mimics_runtime" / "slip" / "logs"
        ) / "slip_worker.log"
        self._log_handle = None
        self._gpu_lock: FileResourceLock | None = None
        self._registered = False
        self.module = None  # SLIP Inference_module
        self.image_shape: tuple[int, int, int] | None = None

    # -- output redirect ----------------------------------------------------
    # SLIP prints progress (with emoji) on every call; a GBK console would
    # crash on it and the worker's stdout belongs to the caller, so all SLIP
    # output goes to the session log instead.

    @contextlib.contextmanager
    def _redirect(self) -> Iterator[None]:
        with contextlib.redirect_stdout(self._log_handle or sys.stdout), \
                contextlib.redirect_stderr(self._log_handle or sys.stderr):
            yield

    def _log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = "[{0}] {1}\n".format(stamp, message)
        if self._log_handle is not None:
            self._log_handle.write(line)
            self._log_handle.flush()
        else:
            sys.stderr.write(line)

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> None:
        import torch  # fail fast with a clear error if python_env lacks it

        checkpoint = Path(self.checkpoint)
        if not checkpoint.is_file():
            raise RuntimeError(
                "SLIP checkpoint not found: {0}".format(checkpoint)
            )
        deps_dir = Path(self.deps_dir)
        if not deps_dir.is_dir():
            raise RuntimeError(
                "SLIP deps directory not found: {0}".format(deps_dir)
            )
        if not VENDORED_SLIP_ROOT.is_dir():
            raise RuntimeError(
                "Vendored SLIP source not found: {0}".format(VENDORED_SLIP_ROOT)
            )

        # All SLIP output goes to the session log, never the console.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._log("session open: device={0} checkpoint={1}".format(
            self.device, checkpoint))

        # GPU: VRAM floor first, then the shared lock (mutual exclusion with
        # nnInteractive and any other Mimics-Script AI job).
        if _gpu_lock_enabled(self.device):
            _check_free_gpu_memory(self.device, self.minimum_free_gpu_memory_gb)
            self._gpu_lock = FileResourceLock(
                _gpu_lock_path(),
                "gpu",
                "SLIP worker (pid {0})".format(os.getpid()),
            )
            try:
                self._gpu_lock.acquire(
                    wait_seconds=self.gpu_lock_timeout_seconds, poll_seconds=2.0
                )
            except ResourceLockTimeout as exc:
                self._gpu_lock = None
                raise RuntimeError(
                    "GPU is already in use by another Mimics-Script AI job. "
                    "Stop the running job or wait for it to finish. {0}".format(exc)
                ) from exc

        volume = _load_image_volume(self.request.get("image") or {})
        self.image_shape = tuple(int(value) for value in volume.shape)

        # Isolated deps + vendored source, both prepended before the import.
        for path in (str(deps_dir), str(VENDORED_SLIP_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        with self._redirect():
            from SLIP_model.inference_module import Inference_module

            self._log("loading checkpoint (this includes CUDA warm-up, ~30s)")
            started = time.time()
            self.module = Inference_module(
                device=self.device,
                checkpoint=str(checkpoint),
                activate_propagation=self.activate_propagation,
                torch_compile=False,
                vis=False,
            )
            load_seconds = time.time() - started
            self._log("patching image (shape {0})".format(self.image_shape))
            started = time.time()
            self.module.process_image(
                image_path=None,
                gt_path=None,
                target_label=1,
                numpy_image=volume,
            )
            process_seconds = time.time() - started
        patch_count = len(getattr(self.module, "bbox_data", None) or {})
        if patch_count < 1:
            raise RuntimeError(
                "SLIP could not extract any patch from the image; "
                "the volume may be degenerate."
            )
        self._log(
            "session ready: model_load={0:.1f}s process_image={1:.1f}s "
            "patches={2}".format(load_seconds, process_seconds, patch_count)
        )
        self.load_seconds = load_seconds
        self.process_seconds = process_seconds
        self.patch_count = patch_count

    def register(self, state_path: Path, parent_pid: int) -> None:
        """Register with the shared process registry (worker mode only)."""
        try:
            register_process(
                str(PROJECT_ROOT),
                "async_worker",
                os.getpid(),
                parent_pid=int(parent_pid or 0) or None,
                state_path=str(state_path),
            )
            self._registered = True
        except Exception:
            pass  # best-effort: worker_status.json stays the source of truth

    # -- commands ---------------------------------------------------------------

    def predict(self, click: Any, label: Any, output_path: str) -> dict[str, Any]:
        if self.module is None or self.image_shape is None:
            raise RuntimeError("SLIP session is not open.")
        z, y, x, label_value = _validate_click(click, label, self.image_shape)
        started = time.time()
        with self._redirect():
            mask = self.module.click_inference([[z, y, x]], [label_value])
        elapsed = time.time() - started
        binary = mask.squeeze().cpu().numpy().astype(bool)
        info = _write_mask_u8(binary, output_path)
        return {
            "schema_version": JOB_SCHEMA,
            "status": "refined",
            "action": "predict",
            "interaction_seconds": round(elapsed, 3),
            "action_history_len": len(self.module.action_history),
            **info,
        }

    def undo(self, output_path: str) -> dict[str, Any]:
        if self.module is None:
            raise RuntimeError("SLIP session is not open.")
        if len(self.module.action_history) < 1:
            raise RuntimeError(
                "Nothing to undo: no click has been applied in this session."
            )
        started = time.time()
        with self._redirect():
            mask = self.module.undo(self.module.action_history)
        elapsed = time.time() - started
        binary = mask.squeeze().cpu().numpy().astype(bool)
        info = _write_mask_u8(binary, output_path)
        return {
            "schema_version": JOB_SCHEMA,
            "status": "refined",
            "action": "undo",
            "interaction_seconds": round(elapsed, 3),
            "action_history_len": len(self.module.action_history),
            **info,
        }

    def reset(self) -> dict[str, Any]:
        if self.module is None:
            raise RuntimeError("SLIP session is not open.")
        with self._redirect():
            self.module.reset()
        return {
            "schema_version": JOB_SCHEMA,
            "status": "reset",
            "action": "reset",
            "interaction_seconds": 0.0,
            "action_history_len": 0,
            "patches_cached": len(self.module.bbox_data or {}),
        }

    def close(self) -> None:
        self.module = None
        if self._gpu_lock is not None:
            try:
                self._gpu_lock.release()
            except Exception:
                pass
            self._gpu_lock = None
        if self._log_handle is not None:
            self._log("session closed")
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _error_result(exc: BaseException, stage: str) -> dict[str, Any]:
    return {
        "schema_version": JOB_SCHEMA,
        "status": "error",
        "stage": stage,
        "error": "{0}: {1}".format(type(exc).__name__, exc),
        "traceback": traceback.format_exc(),
    }


def _run_single_shot(input_data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One request on stdin, one JSON result on stdout: single predict."""
    output_path = str(input_data.get("output_path") or "").strip()
    if not output_path:
        return (
            {"status": "error",
             "error": "Single-shot mode needs 'output_path' for the mask."},
            2,
        )
    click = input_data.get("click")
    label = input_data.get("label", 1)
    session = _SlipSession(input_data)
    try:
        session.open()
        result = session.predict(click, label, output_path)
        return result, 0
    except Exception as exc:
        return _error_result(exc, "single_shot"), 2
    finally:
        session.close()


def _worker_status(job_dir: Path, status: str, **details: Any) -> None:
    payload = {
        "schema_version": STATUS_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
    }
    payload.update(details)
    _write_json_atomic(job_dir / "worker_status.json", payload)


def _run_command(session: _SlipSession, command: dict[str, Any],
                 output_path: str) -> dict[str, Any]:
    action = str(command.get("action") or "").strip().lower()
    if action == "predict":
        return session.predict(command.get("click"), command.get("label", 1),
                               output_path)
    if action == "undo":
        return session.undo(output_path)
    if action == "reset":
        return session.reset()
    raise RuntimeError(
        "Unknown command action {0!r}; expected predict, undo or reset.".format(
            command.get("action")
        )
    )


def _async_worker_main(job_dir_value: str) -> int:
    """Persistent file-queue worker used by the non-blocking Mimics mode."""
    job_dir = Path(job_dir_value).resolve()
    request = _load_json(job_dir / "initialize.json")
    if request is None:
        _worker_status(job_dir, "failed", stage="read_initialize",
                       error="initialize.json missing or invalid")
        return 2
    idle_timeout = float(request.get("idle_timeout_seconds", 600))
    poll_seconds = max(0.1, float(request.get("poll_seconds", 0.5)))
    parent_pid = int(request.get("parent_pid", 0) or 0)
    session = None
    last_sequence = 0
    try:
        _worker_status(job_dir, "initializing", stage="load_model_and_image")
        session = _SlipSession(request)
        session.open()
        session.register(job_dir / "worker_status.json", parent_pid)
        _worker_status(
            job_dir,
            "ready",
            stage="waiting_for_prompt",
            device=session.device,
            patches=session.patch_count,
            model_load_seconds=round(session.load_seconds, 1),
            process_image_seconds=round(session.process_seconds, 1),
            worker_log=str(session.log_path),
        )
        last_activity = time.time()

        while True:
            if (job_dir / "close.json").is_file():
                _worker_status(job_dir, "closing",
                               stage="close_requested")
                return 0

            # If the parent (Mimics) process has died, shut down immediately
            # instead of holding GPU memory for the full idle timeout.
            if parent_pid and not resource_process_exists(parent_pid):
                _worker_status(job_dir, "expired",
                               stage="parent_process_gone")
                return 0

            pending = []
            for path in sorted((job_dir / "commands").glob("command_*.json")):
                try:
                    sequence = int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if sequence > last_sequence:
                    pending.append((sequence, path))

            if pending:
                sequence, command_path = pending[0]
                command = _load_json(command_path) or {}
                result_path = job_dir / "results" / "result_{0:06d}.json".format(sequence)
                output_path = str(
                    job_dir / "results" / "prediction_{0:06d}.u8".format(sequence)
                )
                _worker_status(job_dir, "running", stage="interaction",
                               sequence=sequence)
                try:
                    result = _run_command(session, command, output_path)
                except Exception as exc:
                    result = _error_result(exc, "interaction")
                result["sequence"] = sequence
                result["command_id"] = command.get("command_id")
                _write_json_atomic(result_path, result)
                last_sequence = sequence
                last_activity = time.time()
                _worker_status(job_dir, "ready", stage="waiting_for_prompt",
                               sequence=sequence,
                               result_status=result.get("status"),
                               error=result.get("error"))
                continue

            if time.time() - last_activity >= idle_timeout:
                _worker_status(job_dir, "expired", stage="idle_timeout",
                               idle_timeout_seconds=idle_timeout)
                return 0
            time.sleep(poll_seconds)
    except Exception as exc:
        try:
            _worker_status(job_dir, "failed", stage="initialize_or_poll",
                           error=str(exc), traceback=traceback.format_exc())
        except Exception:
            pass
        return 2
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        status = _load_json(job_dir / "worker_status.json")
        if status and status.get("status") == "closing":
            _worker_status(job_dir, "closed", stage="closed")


def main() -> int:
    # SLIP prints emoji; a GBK console would crash before any JSON appears.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if len(sys.argv) == 3 and sys.argv[1] == "--async-worker":
        return _async_worker_main(sys.argv[2])
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "error",
                          "error": "Invalid JSON input: {0}".format(exc)}))
        return 2
    if not isinstance(input_data, dict):
        print(json.dumps({"status": "error", "error": "Input must be a JSON object."}))
        return 2
    result, code = _run_single_shot(input_data)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
