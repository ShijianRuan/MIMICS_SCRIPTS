"""Cross-process resource locks for Mimics-Script background workers."""

from __future__ import annotations

import json
import errno
import hashlib
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
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError as exc:
        last_error = exc
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
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            if exc.errno == errno.EPERM:
                return True
            return False
        except Exception:
            return False
    try:
        import ctypes

        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102
        ERROR_ACCESS_DENIED = 5
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(
            SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, value
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_PARAMETER:
                return False
            if error == ERROR_ACCESS_DENIED:
                return True
            return True
        try:
            result = int(kernel32.WaitForSingleObject(handle, 0))
            if result == WAIT_TIMEOUT:
                return True
            if result == WAIT_OBJECT_0:
                return False
            return True
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def process_start_marker(pid: object) -> str:
    try:
        value = int(pid)
    except Exception:
        return ""
    if value <= 0:
        return ""
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", ctypes.c_uint32),
                    ("dwHighDateTime", ctypes.c_uint32),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetProcessTimes.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]
            kernel32.GetProcessTimes.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, value)
            if not handle:
                return ""
            try:
                created = FILETIME()
                exited = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                if kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ) == 0:
                    return ""
                return str(
                    (int(created.dwHighDateTime) << 32)
                    | int(created.dwLowDateTime)
                )
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return ""
    try:
        fields = Path(f"/proc/{value}/stat").read_text(encoding="ascii").split()
        if len(fields) > 21:
            return str(fields[21])
    except Exception:
        pass
    return ""


def process_matches(pid: object, expected_start_marker: object = None) -> bool:
    if not process_exists(pid):
        return False
    expected = str(expected_start_marker or "").strip()
    if not expected:
        return True
    current = process_start_marker(pid)
    return not current or current == expected


def default_resource_lock_dir(project_root: os.PathLike[str] | str) -> Path:
    configured = os.environ.get("MIMICS_RESOURCE_LOCK_DIR", "").strip()
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    project_text = os.path.abspath(str(project_root))
    project = Path(project_text)
    if os.name == "nt" and project_text.startswith("\\\\"):
        local_base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("TEMP")
            or os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
        )
        if local_base:
            normalized = os.path.normcase(project_text)
            digest = hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()[:16]
            return Path(local_base) / "Mimics-Script" / digest / "locks"
    return project / ".mimics_runtime" / "locks"


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if not written:
            raise OSError("Could not finish writing the resource lock file.")
        offset += int(written)


class _MutationGuard:
    def __init__(self, lock_path: Path, wait_seconds: float = 2.0):
        # The guard is the stable OS-lock identity for this resource path.
        # It is intentionally persistent: unlinking it while another process
        # can open or hold it may split contenders across different inodes and
        # destroy mutual exclusion. The files are one byte and bounded by the
        # number of distinct resource lock paths.
        self.path = Path(str(lock_path) + ".guard")
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        deadline = time.time() + self.wait_seconds
        while True:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        self.handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                return self
            except (IOError, OSError):
                if time.time() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise ResourceLockTimeout(
                        f"Could not inspect resource lock {self.path}"
                    )
                time.sleep(0.02)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        self.handle.close()
        self.handle = None


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
            try:
                with _MutationGuard(self.path, wait_seconds=min(0.5, max(0.05, deadline - time.time())) if wait_seconds else 0.25):
                    payload = self._payload(os.getpid())
                    try:
                        fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        try:
                            _write_all(
                                fd,
                                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
                            )
                            try:
                                os.fsync(fd)
                            except Exception:
                                pass
                        finally:
                            os.close(fd)
                        self.acquired = True
                        return self
                    except FileExistsError:
                        current = self.read()
                        if self._is_stale(current):
                            verify = self.read()
                            if verify.get("token") == current.get("token"):
                                self._unlink_any()
                            continue
            except ResourceLockTimeout:
                current = self.read()
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

    def update_pid(self, pid: int, **extra: object) -> bool:
        if not self.acquired:
            return False
        with _MutationGuard(self.path):
            current = self.read()
            if current.get("token") != self.token:
                self.acquired = False
                return False
            payload = self._payload(pid)
            payload["created_at_epoch"] = current.get(
                "created_at_epoch", payload["created_at_epoch"]
            )
            payload["updated_at_epoch"] = time.time()
            payload.update(extra)
            _write_json_atomic(self.path, payload)
            return True

    def release(self) -> None:
        if release_lock(self.path, self.token):
            self.acquired = False

    def _payload(self, pid: int) -> dict:
        payload = {
            "schema_version": "mimics_script_resource_lock.v1",
            "resource": self.resource,
            "owner": self.owner,
            "path": str(self.path),
            "pid": int(pid),
            "token": self.token,
            "created_at_epoch": time.time(),
        }
        marker = process_start_marker(pid)
        if marker:
            payload["process_start_marker"] = marker
        return payload

    def _is_stale(self, payload: dict) -> bool:
        if not payload:
            return self._invalid_lock_file_is_old()
        return not process_matches(
            payload.get("pid"), payload.get("process_start_marker")
        )

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
    if not token:
        return not lock_path.exists()
    try:
        with _MutationGuard(lock_path):
            for attempt in range(8):
                try:
                    payload = json.loads(lock_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    return True
                except Exception:
                    return False
                if payload.get("token") != token:
                    return False
                try:
                    lock_path.unlink()
                    return True
                except FileNotFoundError:
                    return True
                except OSError:
                    time.sleep(min(0.15, 0.03 * (attempt + 1)))
    except ResourceLockTimeout:
        return False
    return False
