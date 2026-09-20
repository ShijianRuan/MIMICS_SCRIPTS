"""Cross-process resource locks and process registry for Mimics-Script.

This module is shared by the external Python 3.13 tooling and the Python
3.5 scripts that run inside Mimics, so it must stay compatible with
Python 3.5 syntax: no f-strings, no ``X | Y`` annotations, no
``from __future__ import annotations``.
"""

import json
import errno
import hashlib
import os
import signal
import subprocess
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
        fields = Path("/proc/{0}/stat".format(value)).read_text(encoding="ascii").split()
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


def default_resource_lock_dir(project_root) -> Path:
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


PROCESS_RECORD_SCHEMA = "mimics_process_record.v1"

# Cleanup policies for registered processes.
#   kill_on_sweep  (default) the sweep may terminate the process when its
#                  parent is gone or it is explicitly targeted
#   leave_alive    never terminate via sweep; only the owner stops it
#                  (foreground-owned UIs that outlive a Mimics restart)
#   idle_timeout_s:<seconds>  like kill_on_sweep, but the process also
#                  self-exits after idling; sweep treats it as kill_on_sweep
CLEANUP_KILL_ON_SWEEP = "kill_on_sweep"
CLEANUP_LEAVE_ALIVE = "leave_alive"

VALID_PROCESS_ROLES = (
    "nninteractive_server",
    "nninteractive_watchdog",
    "async_worker",
    "scribble_worker",
    "background_mimics",
    "mcs_supervisor",
    "single_case_import_worker",
    "trainer",
    "training_controller",
    "remote_stop_helper",
    "external_ui",
)


def default_process_registry_dir(project_root) -> Path:
    """Directory holding one small JSON record per registered process."""
    base = default_resource_lock_dir(project_root).parent
    return base / "processes"


def _process_record_path(registry_dir, record_id: str) -> Path:
    safe = str(record_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return Path(registry_dir) / (safe + ".json")


def _process_record_id(role: str, pid: int, start_marker: str) -> str:
    # pid + start marker uniquely identifies one process lifetime; the role
    # prefix keeps records human-readable in the registry directory.
    digest = hashlib.sha1(
        ("{0}|{1}".format(int(pid), str(start_marker))).encode("utf-8", "replace")
    ).hexdigest()[:12]
    return "{0}-{1}-{2}".format(str(role), int(pid), digest)


def register_process(
    project_root,
    role: str,
    pid: int,
    cmdline_signature: str = "",
    ownership_token: str = "",
    parent_pid=None,
    state_path: str = "",
    cleanup_policy: str = CLEANUP_KILL_ON_SWEEP,
    extra=None,
) -> dict:
    """Atomically write a process record into the registry.

    Returns the written record.  The record is a manifest entry, not a
    status store: ``state_path`` points at the subsystem's own detailed
    state file (server.json, worker_status.json, job status), which stays
    where it is.
    """
    role = str(role)
    if role not in VALID_PROCESS_ROLES:
        raise ValueError("Unknown process role: {0}".format(role))
    pid = int(pid)
    start_marker = process_start_marker(pid)
    record_id = _process_record_id(role, pid, start_marker)
    registry_dir = default_process_registry_dir(project_root)
    record = {
        "schema_version": PROCESS_RECORD_SCHEMA,
        "record_id": record_id,
        "role": role,
        "pid": pid,
        "start_marker": start_marker,
        "cmdline_signature": str(cmdline_signature or "")[:512],
        "ownership_token": str(ownership_token or "") or uuid.uuid4().hex,
        "parent_pid": int(parent_pid) if parent_pid else 0,
        "state_path": str(state_path or ""),
        "cleanup_policy": str(cleanup_policy or CLEANUP_KILL_ON_SWEEP),
        "created_at_epoch": time.time(),
    }
    if extra:
        # extra keys are merged but cannot overwrite the identity fields
        identity = set(record.keys())
        for key, value in dict(extra).items():
            if str(key) not in identity:
                record[str(key)] = value
    _write_json_atomic(_process_record_path(registry_dir, record_id), record)
    return record


def unregister_process(project_root, role: str, pid: int, ownership_token: str = "") -> bool:
    """Remove a process record.  Ownership token must match when given."""
    pid = int(pid)
    path = _find_process_record_path(project_root, role, pid)
    if path is None:
        return False
    payload = _read_process_record(path)
    if not payload:
        return False
    if ownership_token and payload.get("ownership_token") != str(ownership_token):
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _read_process_record(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _find_process_record_path(project_root, role: str, pid: int):
    registry_dir = default_process_registry_dir(project_root)
    # Fast path: the deterministic record id.
    start_marker = process_start_marker(pid)
    if start_marker:
        path = _process_record_path(
            registry_dir, _process_record_id(role, pid, start_marker)
        )
        if path.is_file():
            return path
    # Fallback: scan for role+pid (covers start-marker lookup failures).
    prefix = "{0}-{1}-".format(str(role), int(pid))
    try:
        for entry in registry_dir.iterdir():
            if entry.name.startswith(prefix) and entry.suffix == ".json":
                return entry
    except OSError:
        pass
    return None


def snapshot_processes(project_root, include_dead=False):
    """List registry records, newest first, with a live flag attached."""
    registry_dir = default_process_registry_dir(project_root)
    records = []
    try:
        entries = sorted(registry_dir.iterdir())
    except OSError:
        return records
    for entry in entries:
        if entry.suffix != ".json":
            continue
        payload = _read_process_record(entry)
        if not payload:
            continue
        payload["_live"] = process_matches(
            payload.get("pid"), payload.get("start_marker")
        )
        payload["_record_path"] = str(entry)
        records.append(payload)
    records.sort(key=lambda item: float(item.get("created_at_epoch") or 0.0), reverse=True)
    if not include_dead:
        records = [item for item in records if item.get("_live")]
    return records


def process_is_live(project_root, role: str, pid: int) -> bool:
    """True when a live registry record exists for this role+pid."""
    path = _find_process_record_path(project_root, role, pid)
    if path is None:
        return False
    payload = _read_process_record(path)
    if not payload:
        return False
    return process_matches(payload.get("pid"), payload.get("start_marker"))


def terminate_process(project_root, role: str, pid: int, graceful_seconds: float = 5.0) -> bool:
    """The single kill ladder: terminate -> grace poll -> taskkill /T /F.

    Liveness is checked against the registry record's start marker so a
    recycled PID can never be killed by mistake.  Returns True when the
    process was terminated (or already gone) by the end of the call.
    """
    path = _find_process_record_path(project_root, role, pid)
    payload = _read_process_record(path) if path is not None else {}
    expected_marker = (payload or {}).get("start_marker")
    if not process_matches(pid, expected_marker):
        return True
    try:
        if os.name != "nt":
            os.kill(int(pid), signal.SIGTERM)
        else:
            # Windows graceful step: taskkill without /F posts WM_CLOSE to
            # windowed processes (and their trees).  Console-only processes
            # reject it, which is harmless - the force step follows.
            try:
                subprocess.call(
                    ["taskkill", "/PID", str(int(pid)), "/T"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass
        deadline = time.time() + max(0.0, float(graceful_seconds))
        while time.time() < deadline:
            if not process_matches(pid, expected_marker):
                return True
            time.sleep(0.25)
    except OSError:
        pass
    if not process_matches(pid, expected_marker):
        return True
    # Force: taskkill the whole tree (children included).
    if os.name == "nt":
        try:
            subprocess.call(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
    else:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            return False
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not process_matches(pid, expected_marker):
            return True
        time.sleep(0.25)
    return process_matches(pid, expected_marker) is False


def sweep_processes(
    project_root,
    protect_roles=(),
    include_cmdline_scan=False,
):
    """Bring the runtime back to a consistent state.

    Invariant: after the foreground Mimics (re)starts and calls this once,
    the registry is consistent - dead records are cleared, orphaned
    processes with ``kill_on_sweep`` policy are terminated, and locks they
    held are released (token-verified against the record's ownership
    token).

    ``include_cmdline_scan`` adds a legacy fallback for processes started
    before the registry existed: a command-line census finds unregistered
    env-root python.exe processes.  The registry is the primary path.
    """
    protect = set(str(role) for role in protect_roles)
    summary = {
        "removed_dead_records": 0,
        "terminated_orphans": [],
        "released_locks": [],
        "cmdline_scan": include_cmdline_scan,
    }
    for record in snapshot_processes(project_root, include_dead=True):
        role = str(record.get("role") or "")
        pid = record.get("pid")
        path = record.get("_record_path")
        live = process_matches(pid, record.get("start_marker"))
        if not live:
            # Dead: clear the record; release locks it still holds.
            if _remove_record_file(path):
                summary["removed_dead_records"] += 1
            released = _release_locks_for_record(project_root, record)
            summary["released_locks"].extend(released)
            continue
        if role in protect:
            continue
        policy = str(record.get("cleanup_policy") or CLEANUP_KILL_ON_SWEEP)
        if policy == CLEANUP_LEAVE_ALIVE:
            continue
        parent_pid = record.get("parent_pid")
        if parent_pid and not process_exists(parent_pid):
            # Orphan: parent died, policy says kill.
            token = str(record.get("ownership_token") or "")
            if terminate_process(project_root, role, pid):
                summary["terminated_orphans"].append(
                    {"role": role, "pid": pid, "reason": "parent_gone"}
                )
                summary["released_locks"].extend(
                    _release_locks_for_record(project_root, record)
                )
                # The process is now dead; drop its record so the registry
                # reflects reality after a single sweep.
                if _remove_record_file(record.get("_record_path")):
                    summary["removed_dead_records"] += 1
    return summary


def _remove_record_file(path) -> bool:
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _release_locks_for_record(project_root, record):
    """Release resource locks still held by a dead/terminated process.

    Identity is proven by PID + start marker: the lock payload carries the
    holder's ``process_start_marker``, and a dead PID with a matching start
    marker cannot be a different live process.  Locks whose holder is still
    alive, or whose identity does not match the record, are never touched.
    """
    lock_dir = default_resource_lock_dir(project_root)
    target_pid = record.get("pid")
    target_marker = str(record.get("start_marker") or "")
    released = []
    try:
        entries = sorted(lock_dir.iterdir())
    except OSError:
        return released
    for entry in entries:
        if entry.suffix != ".lock":
            continue
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        lock_pid = payload.get("pid")
        if not lock_pid or int(lock_pid) != int(target_pid):
            continue
        if process_exists(lock_pid):
            # Still alive (or PID got reused by a live process): never
            # release a lock owned by a live process.
            continue
        lock_marker = str(payload.get("process_start_marker") or "")
        if lock_marker and target_marker and lock_marker != target_marker:
            # The lock belonged to an earlier process with the same PID.
            continue
        if release_lock(entry, str(payload.get("token") or "") or None):
            released.append(str(entry))
    return released


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
                        "Could not inspect resource lock {0}".format(self.path)
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

    def __init__(self, path, resource: str, owner: str):
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
                raise ResourceLockCancelled(
                    "Cancelled while waiting for {0} lock".format(self.resource)
                )
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
                raise ResourceLockTimeout(
                    "{0} is busy: {1} (pid {2})".format(self.resource, holder, pid)
                )
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
                raise ResourceLockCancelled(
                    "Cancelled while waiting for {0} lock".format(self.resource)
                )
            remaining_deadline = deadline - time.time()
            if remaining_deadline <= 0:
                return
            step = min(0.25, remaining_poll, remaining_deadline)
            time.sleep(max(0.01, step))
            remaining_poll -= step


def release_lock(path, token=None) -> bool:
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
