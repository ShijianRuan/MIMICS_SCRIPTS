# -*- coding: utf-8 -*-
"""Shared helpers for Mimics-side Python 3.5 runtime scripts."""

from __future__ import print_function

import json
import os
import subprocess
import time
import uuid


INVALID_LOCK_GRACE_SECONDS = 5.0


def write_json_atomic(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text)


def write_text_atomic(path, text):
    parent = os.path.dirname(path)
    last_error = None
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    for attempt in range(12):
        temporary = path + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(temporary, "w") as handle:
                handle.write(str(text))
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.isfile(temporary):
                    os.remove(temporary)
            except Exception:
                pass
            time.sleep(min(0.15, 0.02 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def read_json(path, default=None):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return default


def safe_filename(value):
    text = str(value or "unknown")
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"


def safe_slug(value):
    return safe_filename(value).strip("._") or "unknown"


def find_root(start_dir, sentinel_files, max_depth=6):
    current = os.path.abspath(start_dir)
    for _ in range(max_depth):
        for sentinel in sentinel_files:
            if os.path.isfile(os.path.join(current, sentinel)):
                return current
            if os.path.isdir(os.path.join(current, sentinel)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hidden_process_kwargs():
    if os.name != "nt":
        return {}

    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE

    return {
        "startupinfo": startupinfo,
        "creationflags": CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    }


def background_process_kwargs(low_priority=True):
    kwargs = hidden_process_kwargs()
    if os.name == "nt" and low_priority:
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | BELOW_NORMAL_PRIORITY_CLASS
    return kwargs


def background_env(extra=None, include_itk=False):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    if include_itk:
        env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    if extra:
        env.update(extra)
    return env


def auto_cleanup_enabled():
    value = os.environ.get("MIMICS_AUTO_CLEANUP_ON_START", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def aggressive_auto_cleanup_enabled():
    value = os.environ.get("MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def resource_lock_dir(project_root):
    return os.path.join(project_root, ".mimics_runtime", "locks")


def resource_lock_path(project_root, name):
    return os.path.join(resource_lock_dir(project_root), name)


def process_exists(pid):
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


def _resource_lock_payload(resource, owner, pid, token):
    return {
        "schema_version": "mimics_script_resource_lock.v1",
        "resource": resource,
        "owner": owner,
        "pid": int(pid),
        "token": token,
        "created_at_epoch": time.time(),
    }


def acquire_resource_lock(path, resource, owner, wait_seconds=0.0, poll_seconds=2.0):
    token = uuid.uuid4().hex
    deadline = time.time() + max(0.0, float(wait_seconds))
    while True:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = _resource_lock_payload(resource, owner, os.getpid(), token)
        payload["path"] = path
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            finally:
                os.close(fd)
            return token
        except OSError:
            current = read_json(path, {}) or {}
            if (not current and _invalid_lock_file_is_old(path)) or (current and not process_exists(current.get("pid"))):
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue
            if time.time() >= deadline:
                return None
            time.sleep(max(0.25, float(poll_seconds)))


def update_resource_lock_pid(path, token, pid, extra=None):
    current = read_json(path, {}) or {}
    if current.get("token") != token:
        return False
    current["pid"] = int(pid)
    current["updated_at_epoch"] = time.time()
    if extra:
        current.update(extra)
    write_json_atomic(path, current)
    return True


def release_resource_lock(path, token):
    current = read_json(path, {}) or {}
    if token and current.get("token") != token:
        return False
    for attempt in range(8):
        try:
            os.remove(path)
            return True
        except OSError:
            if not os.path.exists(path):
                return True
            time.sleep(min(0.15, 0.03 * (attempt + 1)))
    return False


def cleanup_stale_resource_locks(lock_dir):
    removed = 0
    if not lock_dir or not os.path.isdir(lock_dir):
        return removed
    try:
        names = os.listdir(lock_dir)
    except Exception:
        return removed
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(lock_dir, name)
        payload = read_json(path, {}) or {}
        if not payload and not _invalid_lock_file_is_old(path):
            continue
        if payload and process_exists(payload.get("pid")):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def _invalid_lock_file_is_old(path):
    try:
        age = time.time() - os.path.getmtime(path)
    except Exception:
        return False
    return age >= INVALID_LOCK_GRACE_SECONDS
