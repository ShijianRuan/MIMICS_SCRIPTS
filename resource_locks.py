"""Cross-process resource locks for Mimics-Script background workers."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


INVALID_LOCK_GRACE_SECONDS = 5.0


class ResourceLockTimeout(RuntimeError):
    pass


class ResourceLockCancelled(RuntimeError):
    pass


def _write_json_atomic(path: Path, payload: dict, retries: int = 20, max_sleep: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def process_exists(pid: object) -> bool:
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(value, 0)
            return True
        except Exception:
            return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, value)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


class FileResourceLock:
    """Exclusive lock backed by an atomically-created JSON file."""

    def __init__(self, path: os.PathLike[str] | str, resource: str, owner: str):
        self.path = Path(path)
        self.resource = resource
        self.owner = owner
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(
        self,
        wait_seconds: float = 0.0,
        poll_seconds: float = 2.0,
        on_wait: Optional[Callable[[dict], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "FileResourceLock":
        deadline = time.time() + max(0.0, float(wait_seconds))
        last_notice = 0.0
        while True:
            if should_cancel and should_cancel():
                raise ResourceLockCancelled(f"Cancelled while waiting for {self.resource} lock")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._payload(os.getpid())
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
                finally:
                    os.close(fd)
                self.acquired = True
                return self
            except FileExistsError:
                current = self.read()
                if self._is_stale(current):
                    self._unlink_any()
                    continue
                if on_wait and time.time() - last_notice >= max(1.0, float(poll_seconds)):
                    on_wait(current)
                    last_notice = time.time()
                now = time.time()
                if now >= deadline:
                    holder = current.get("owner", "unknown") if isinstance(current, dict) else "unknown"
                    pid = current.get("pid", "?") if isinstance(current, dict) else "?"
                    raise ResourceLockTimeout(f"{self.resource} is busy: {holder} (pid {pid})")
                self._sleep_between_attempts(deadline, poll_seconds, should_cancel)

    def read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update_pid(self, pid: int, **extra: object) -> None:
        if not self.acquired:
            return
        payload = self._payload(pid)
        payload.update(extra)
        _write_json_atomic(self.path, payload)

    def release(self) -> None:
        release_lock(self.path, self.token)
        self.acquired = False

    def _payload(self, pid: int) -> dict:
        return {
            "schema_version": "mimics_script_resource_lock.v1",
            "resource": self.resource,
            "owner": self.owner,
            "path": str(self.path),
            "pid": int(pid),
            "token": self.token,
            "created_at_epoch": time.time(),
        }

    def _is_stale(self, payload: dict) -> bool:
        if not payload:
            return self._invalid_lock_file_is_old()
        return not process_exists(payload.get("pid"))

    def _unlink_any(self) -> None:
        for attempt in range(8):
            try:
                self.path.unlink()
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(min(0.15, 0.03 * (attempt + 1)))

    def _invalid_lock_file_is_old(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except Exception:
            return False
        return age >= INVALID_LOCK_GRACE_SECONDS

    def _sleep_between_attempts(
        self,
        deadline: float,
        poll_seconds: float,
        should_cancel: Optional[Callable[[], bool]],
    ) -> None:
        remaining_poll = max(0.25, float(poll_seconds))
        while remaining_poll > 0:
            if should_cancel and should_cancel():
                raise ResourceLockCancelled(f"Cancelled while waiting for {self.resource} lock")
            remaining_deadline = deadline - time.time()
            if remaining_deadline <= 0:
                return
            step = min(0.25, remaining_poll, remaining_deadline)
            time.sleep(max(0.01, step))
            remaining_poll -= step


def release_lock(path: os.PathLike[str] | str, token: str | None = None) -> bool:
    lock_path = Path(path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except Exception:
        payload = {}
    if token and payload.get("token") != token:
        return False
    for attempt in range(8):
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(min(0.15, 0.03 * (attempt + 1)))
    return False
