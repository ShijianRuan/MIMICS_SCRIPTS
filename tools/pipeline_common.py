#!/usr/bin/env python3
"""Shared primitives for the external training pipelines.

The few-shot, nnInteractive fine-tune, and nnU-Net pipelines each grew
identical copies of atomic writes, cancel markers, and process-tree
termination.  This module is the single home for those helpers so a fix
lands once.

Scope rules (decided during the 2026-09 refactor):
- Only genuinely identical logic lives here.  ``safe_slug`` variants stay
  where they are because their fallbacks and underscore-collapsing differ
  and those differences are baked into on-disk paths.
- The background-Mimics lock protocol is scoped (output directory +
  destination), matching the few-shot pipeline.  The fine-tune pipeline
  previously used a per-job unique lock, which provided no mutual
  exclusion between concurrent jobs reading the same .mcs folder — that
  divergence is fixed here.
- ``nninteractive_bridge.py`` keeps its own copies of these helpers: it
  must stay a single file that can be copied to a standalone deployment.

This module never imports Mimics and runs on Python 3.10+.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import runtime_common  # noqa: E402
from resource_locks import (  # noqa: E402
    FileResourceLock,
    ResourceLockCancelled,
    ResourceLockTimeout,
    register_process,
)


def write_text_atomic(path, text, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = str(text)
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
    # Last resort: a plain non-atomic write is better than losing the
    # status update entirely (mirrors nninteractive_task_common).
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


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text, retries=retries, max_sleep=max_sleep)


def read_json(path, default=None):
    """Read a JSON file; return ``default`` on any error (missing, invalid)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_cancel_marker(cancel_path):
    """Best-effort cancel marker; returns an error string or None."""
    if not cancel_path:
        return None
    try:
        write_text_atomic(
            cancel_path,
            "cancel requested at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")),
            retries=8,
            max_sleep=0.15,
        )
        return None
    except Exception as exc:
        return str(exc)


def copy_file_atomic(source, destination):
    """Copy one file via a staged rename so readers never see a partial file."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        "{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        shutil.copy2(str(source), str(temporary))
        last_error = None
        for attempt in range(20):
            try:
                os.replace(str(temporary), str(destination))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25, 0.02 * (attempt + 1)))
        raise OSError(
            "Could not publish {} after bounded replace retries: {}".format(
                destination, last_error
            )
        )
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def terminate_process_tree(pid):
    """Fire-and-forget tree kill by PID (no grace period)."""
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def terminate_popen_tree(process, grace=10.0):
    """Full ladder for an owned Popen: terminate, wait, force-kill tree."""
    if process is None:
        return True
    try:
        if process.poll() is not None:
            return True
    except Exception:
        pass
    try:
        process.terminate()
    except Exception:
        pass
    deadline = time.time() + max(0.0, float(grace))
    while time.time() < deadline:
        try:
            if process.poll() is not None:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    terminate_process_tree(process.pid)
    try:
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=10)
        except Exception:
            pass
    try:
        return process.poll() is not None
    except Exception:
        return False


def hidden_process_kwargs():
    """Standard flags for hidden, low-priority background subprocesses."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
    return {"startupinfo": startupinfo, "creationflags": flags}


def acquire_scoped_background_mimics_locks(
    scopes,
    owner,
    wait_seconds=0.0,
    cancel_path=None,
    on_wait=None,
    poll_seconds=5.0,
):
    """Acquire one scoped background-Mimics lock per scope, all-or-nothing.

    This is the unified lock protocol for every pipeline that spawns a
    background Mimics to export labels.  Locks are keyed by the shared
    resources (the .mcs output directory and the label destination), so
    concurrent jobs touching the same directories serialize while
    disjoint jobs proceed in parallel.  On timeout or cancellation every
    lock acquired so far is released before the error propagates.
    """
    unique = {}
    for scope in scopes:
        lock_path = runtime_common.background_mimics_lock_path(str(ROOT), str(scope or "default"))
        unique[str(lock_path)] = scope
    ordered = [unique[key] for key in sorted(unique)]
    deadline = time.time() + max(0.0, float(wait_seconds))

    def _cancelled():
        return bool(cancel_path) and Path(cancel_path).is_file()

    while True:
        locks = []
        try:
            for scope in ordered:
                lock_path = Path(
                    runtime_common.background_mimics_lock_path(str(ROOT), str(scope))
                )
                lock = FileResourceLock(lock_path, "background_mimics", owner)
                lock.acquire(
                    wait_seconds=0.0,
                    poll_seconds=poll_seconds,
                    on_wait=on_wait,
                    should_cancel=_cancelled,
                )
                locks.append(lock)
            return locks
        except (ResourceLockTimeout, ResourceLockCancelled):
            for lock in reversed(locks):
                lock.release()
            raise
        except Exception:
            for lock in reversed(locks):
                lock.release()
            now = time.time()
            if now >= deadline:
                raise
            remaining = min(2.0, max(0.0, deadline - now))
            while remaining > 0:
                if _cancelled():
                    raise ResourceLockCancelled(
                        "Cancelled while waiting for background Mimics resources"
                    )
                step = min(0.25, remaining)
                time.sleep(step)
                remaining -= step


def spawn_background_mimics_export(
    mimics_exe,
    runner_path,
    export_root,
    status_path="",
    log_handle=None,
    mimics_log_path=None,
    state_path="",
):
    """Spawn background Mimics for a label export and register it.

    Uses runtime_common.background_mimics_command so the documented
    command line is built in exactly one place, and records the process
    in the process registry (best-effort — registration never blocks
    the spawn).
    """
    command = runtime_common.background_mimics_command(
        mimics_exe, runner_path, mimics_log_path=mimics_log_path
    )
    stdout = log_handle if log_handle is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        **hidden_process_kwargs()
    )
    try:
        register_process(
            ROOT,
            "background_mimics",
            process.pid,
            parent_pid=os.getpid(),
            state_path=str(state_path or status_path or ""),
        )
    except Exception:
        pass
    return process


__all__ = [
    "write_text_atomic",
    "write_json_atomic",
    "write_cancel_marker",
    "copy_file_atomic",
    "terminate_process_tree",
    "terminate_popen_tree",
    "hidden_process_kwargs",
    "acquire_scoped_background_mimics_locks",
    "spawn_background_mimics_export",
]
