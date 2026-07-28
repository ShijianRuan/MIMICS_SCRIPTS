"""Small runtime primitives shared by the external CLI commands."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Replace JSON with retries and a flushed SMB/Windows-compatible fallback."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error: OSError | None = None
    for attempt in range(20):
        temporary = target.with_name(
            "{}.{}.{}.tmp".format(target.name, os.getpid(), uuid.uuid4().hex)
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(target))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(0.25, 0.02 * (attempt + 1)))
    # SMB servers and some Windows readers permit writes but deny replace.
    # There is only one status writer, and readers already need to tolerate a
    # transiently incomplete JSON document, so do not fail training solely
    # because atomic replacement is unavailable.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    except OSError as exc:
        last_error = exc
    raise OSError("Could not update {}: {}".format(target, last_error))


class StatusReporter:
    def __init__(self, path: str | Path | None, job_id: str) -> None:
        self.path = Path(path) if path else None
        self.job_id = job_id
        self.payload: dict[str, Any] = {
            "schema_version": "nninteractive_finetune_status.v1",
            "job_id": job_id,
            "status": "created",
            "created_at_epoch": time.time(),
        }

    def update(self, **values: Any) -> None:
        self.payload.update(values)
        self.payload["updated_at_epoch"] = time.time()
        if self.path:
            write_json_atomic(self.path, self.payload)


def cancellation_requested(path: str | Path | None) -> bool:
    return bool(path and Path(path).is_file())


class CancelledError(RuntimeError):
    pass


def process_exists(pid: int) -> bool:
    """Return whether a process is alive without relying on exit code 259."""
    if int(pid) <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import ctypes

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
    if not handle:
        # Access denied means the process exists but is owned by a more
        # privileged account. Treating it as dead could steal a live lock.
        return int(ctypes.windll.kernel32.GetLastError()) == 5
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


@contextmanager
def exclusive_job_lock(path: str | Path, job_id: str):
    """Prevent two trainers from writing the same output/work directory."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "nninteractive_finetune_job_lock.v1",
        "job_id": job_id,
        "pid": os.getpid(),
        "created_at_epoch": time.time(),
    }
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(
                str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            try:
                os.write(descriptor, json.dumps(payload).encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            acquired = True
            break
        except FileExistsError:
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
                holder_pid = int(current.get("pid") or 0)
            except (OSError, ValueError, TypeError):
                holder_pid = 0
            if holder_pid and process_exists(holder_pid):
                raise RuntimeError(
                    "Another fine-tuning process is already using this output "
                    "directory (PID {}).".format(holder_pid)
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    "A stale training lock could not be removed: {}".format(lock_path)
                ) from exc
    if not acquired:
        raise RuntimeError("Could not acquire training lock: {}".format(lock_path))
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            current = {}
        if (
            current.get("job_id") == job_id
            and int(current.get("pid") or 0) == os.getpid()
        ):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
