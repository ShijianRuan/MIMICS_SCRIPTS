#!/usr/bin/env python3
"""Framework-agnostic Mimics label-export and resource-lock plumbing.

Every AI integration (nnU-Net, nnInteractive fine-tune, FlexiCT) needs the
same machinery around the Mimics side of training: locate the background
MimicsResearch.exe, export labels from saved .mcs projects, materialize
image/label NIfTI pairs on the source grid, serialize GPU and background
Mimics access, and reap child process trees. This module holds that shared
machinery so no integration depends on another integration's pipeline module.

Extracted verbatim from the original AI-pipeline module; function bodies
are unchanged except for neutralized naming and framework-specific owner
strings.

This script never imports Mimics. It is safe to run from the foreground Mimics
process through subprocess.Popen because all long-running work happens here or
in child Python/Mimics background processes.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import runtime_common

import pipeline_common

from resource_locks import (
    FileResourceLock,
    ResourceLockCancelled,
    ResourceLockTimeout,
    default_resource_lock_dir,
    process_exists as resource_process_exists,
    register_process,
    release_lock,
)

LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3


def cancel_requested(cancel_path) -> bool:
    """True when the caller asked to stop.

    Two control conventions reach this module: job pipelines pass their
    control.json (present from job creation; only its ``action`` payload
    says "stopped"), while older marker-file callers pass a path that
    exists only once cancelled. Treat a JSON file as cancelled by content
    and anything else (missing, or non-JSON marker) by existence.
    """
    if not cancel_path:
        return False
    path = Path(cancel_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except Exception:
        return True  # non-JSON marker file: its existence is the signal
    if isinstance(payload, dict):
        return str(payload.get("action") or "").lower() in {"cancel", "stop"}
    return True


def _gpu_lock_path():
    """Resolve the GPU lock path per call, honoring env overrides.

    A module-level constant would freeze the path at import time; test
    suites that isolate locks via MIMICS_RESOURCE_LOCK_DIR after import
    would still touch the production lock directory.
    """
    return default_resource_lock_dir(ROOT) / "gpu.lock"


def detect_gpu_memory_gb(requested=0.0):
    """Resolve the planning memory budget without allocating a CUDA context."""
    try:
        value = float(requested or 0.0)
    except Exception:
        value = 0.0
    if value > 0:
        return value
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
        values = [
            float(line.strip()) / 1024.0
            for line in output.splitlines()
            if line.strip()
        ]
        if values:
            # Training uses the first visible CUDA device by default. Taking
            # the largest card on a multi-GPU host can over-plan a job that is
            # actually bound to a smaller device.
            return values[0]
    except Exception:
        pass
    return 12.0


def cleanup_local_export_jobs(max_age_days=14, max_jobs=100):
    root = ROOT / ".mimics_runtime" / "export_jobs"
    if not root.is_dir():
        return 0
    now = time.time()
    rows = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            rows.append((path.stat().st_mtime, path))
        except OSError:
            continue
    rows.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for index, (modified, path) in enumerate(rows):
        if index < int(max_jobs) and now - modified <= float(max_age_days) * 86400.0:
            continue
        if local_export_job_is_active(path):
            continue
        if _remove_tree_quietly(path):
            removed += 1
    return removed


def local_export_job_is_active(path):
    path = Path(path)
    for relative in ("status.json", "runner_started.json"):
        payload = read_json(path / relative, {}) or {}
        status = str(payload.get("status") or "").lower()
        if status in ("closed", "completed", "failed", "cancelled", "error"):
            continue
        pid = payload.get("pid")
        if pid and process_exists(pid):
            return True
    return False


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    pipeline_common.write_json_atomic(path, payload, retries=retries, max_sleep=max_sleep)


def write_text_atomic(path, text, retries=20, max_sleep=0.25):
    pipeline_common.write_text_atomic(path, text, retries=retries, max_sleep=max_sleep)


def append_text(path, text, retries=8, max_sleep=0.15):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(str(text))
                try:
                    handle.flush()
                except Exception:
                    pass
            return True
        except OSError as exc:
            last_error = exc
            time.sleep(min(float(max_sleep), 0.03 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def read_json(path, default=None):
    return pipeline_common.read_json(path, default)


def rotate_log(path):
    return pipeline_common.rotate_log(path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_BACKUPS)


def append_log(workspace, message):
    text = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(text, flush=True)
    try:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / "pipeline.log"
        rotate_log(path)
        append_text(path, text + "\n")
    except Exception as exc:
        print(
            "[{}] Warning: could not write pipeline log: {}".format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                exc,
            ),
            file=sys.stderr,
            flush=True,
        )


def unlink_with_retry(path, retries=8, max_sleep=0.15):
    path = Path(path)
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_error = exc
            time.sleep(min(float(max_sleep), 0.03 * (attempt + 1)))
    if last_error is not None:
        raise last_error
    return True


def rmtree_with_retry(path, retries=8, max_sleep=0.25):
    path = Path(path)
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            shutil.rmtree(str(path))
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_error = exc
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    if last_error is not None:
        raise last_error
    return True


def write_cancel_marker(cancel_path):
    return pipeline_common.write_cancel_marker(cancel_path)


TEMP_LOG_RETENTION_DAYS = 30


def prune_temp_log_root(retention_days=TEMP_LOG_RETENTION_DAYS):
    """Delete fallback subprocess logs past the retention window (best effort).

    The %TEMP% fallback root is only used when the primary log is locked, but
    files written there live outside the project tree and would otherwise
    accumulate forever.
    """
    temp_root = Path(tempfile.gettempdir()) / "mimics_script_pipeline_logs"
    if not temp_root.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    try:
        for entry in temp_root.iterdir():
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                continue
    except OSError:
        pass


def open_subprocess_log(path, workspace, label):
    path = Path(path)
    workspace = Path(workspace)
    prune_temp_log_root()
    temp_root = Path(tempfile.gettempdir()) / "mimics_script_pipeline_logs"
    candidates = [
        path,
        path.with_name(path.stem + "_" + uuid.uuid4().hex[:8] + path.suffix),
        workspace / "logs" / "subprocess" / (path.stem + "_" + uuid.uuid4().hex[:8] + path.suffix),
        temp_root / (path.stem + "_" + uuid.uuid4().hex[:8] + path.suffix),
    ]
    errors = []
    primary_error = None
    for index, candidate in enumerate(candidates):
        attempts = 30 if index == 0 else 8
        max_sleep = 0.20 if index == 0 else 0.15
        for attempt in range(attempts):
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                handle = candidate.open("ab")
                warning = None
                if candidate != path:
                    warning = (
                        "{} primary log was not writable. Using fallback log: {}.".format(
                            label,
                            candidate,
                        )
                    )
                    if primary_error:
                        warning += " Primary error: {}".format(primary_error)
                    append_log(workspace, warning)
                return handle, candidate, warning
            except OSError as exc:
                if candidate == path and primary_error is None:
                    primary_error = str(exc)
                errors.append("{}: {}".format(candidate, exc))
                time.sleep(min(max_sleep, 0.03 * (attempt + 1)))
    append_log(
        workspace,
        "{} could not be opened in any fallback location; subprocess output will be discarded. {}".format(
            label,
            " | ".join(errors[-3:]),
        ),
    )
    return open(os.devnull, "ab"), None, "{} unavailable; subprocess output was discarded. {}".format(
        label,
        " | ".join(errors[-3:]),
    )


def safe_slug(value):
    text = str(value or "unknown").strip()
    out = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("._") or "unknown"


def _remove_tree_quietly(path):
    try:
        path = Path(path)
        if path.is_dir():
            rmtree_with_retry(path)
        elif path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def load_mimics_io_config():
    merged = {}
    for path in (ROOT / "mimics_io_config.json", ROOT / "nninteractive_config.json"):
        if path.is_file():
            loaded = read_json(path, {}) or {}
            if isinstance(loaded, dict):
                merged.update(loaded)
    return merged


def resolve_mimics_output_dir(ts_root, config=None):
    default_dir = Path(ts_root).resolve() / "mcs_output"
    config = load_mimics_io_config() if config is None else (config or {})
    configured = config.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if not os.path.isabs(configured):
        path = Path(ts_root).resolve() / configured
    else:
        path = Path(configured)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    except Exception:
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir


def _parse_axes_value(value, default=None):
    if default is None:
        default = [0, 1, 2]
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    axes = [int(part) for part in value]
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("mimics buffer axes must be a permutation of 0,1,2: {}".format(value))
    return axes


def _parse_flips_value(value, default=None):
    if default is None:
        default = [False, False, False]
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(value) != 3:
        raise ValueError("mimics buffer flips must contain three values: {}".format(value))
    truthy = ("1", "true", "yes", "y", "on")
    return [bool(part) if isinstance(part, bool) else str(part).strip().lower() in truthy for part in value]


def resolve_mimics_buffer_mapping(config=None):
    config = load_mimics_io_config() if config is None else (config or {})
    axes_value = (
        config.get("mimics_buffer_axes")
        if "mimics_buffer_axes" in config
        else config.get("platform_to_mimics_axes", config.get("axes", [0, 1, 2]))
    )
    flips_value = (
        config.get("mimics_buffer_flips")
        if "mimics_buffer_flips" in config
        else config.get("platform_to_mimics_flips", config.get("flips", [False, False, False]))
    )
    return _parse_axes_value(axes_value), _parse_flips_value(flips_value)


def resolve_path(value, base):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def find_mimics_exe(explicit=None):
    """Find MimicsResearch.exe, with optional explicit override."""
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))
    # Delegate to runtime_common which has the full search logic
    return runtime_common.find_mimics_exe()


def hidden_process_kwargs():
    return pipeline_common.hidden_process_kwargs()


def background_env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def process_exists(pid):
    return resource_process_exists(pid)


def _env_flag_disabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _lock_wait_payload(resource, current):
    if not isinstance(current, dict):
        current = {}
    payload = {
        "resource": resource,
        "owner": current.get("owner", "unknown"),
        "pid": current.get("pid", ""),
        "created_at_epoch": current.get("created_at_epoch"),
    }
    owner = str(payload["owner"] or "").lower()
    if resource == "background_mimics" and "import" in owner:
        payload["user_action"] = (
            "Import and label export share one background Mimics license. "
            "Wait for import to finish, or stop it in Task Status "
            "(01 Data menu) if training is more urgent."
        )
    elif resource == "gpu":
        payload["user_action"] = (
            "The job will start automatically when the active AI operation releases the GPU. "
            "Use the integration's Stop AI Task entry only when the running task should be cancelled."
        )
    return payload


def _nninteractive_operation_is_active(state):
    try:
        operation_pid = int(state.get("active_operation_pid", 0) or 0)
    except Exception:
        operation_pid = 0
    if not state.get("active_operation") or not operation_pid or not process_exists(operation_pid):
        return False
    try:
        started = float(state.get("active_operation_started_at_epoch", time.time()) or time.time())
        timeout = max(60.0, float(state.get("active_operation_timeout_seconds", 3600) or 3600))
    except Exception:
        return True
    return time.time() - started <= timeout


def _nninteractive_worker_has_pending_work(state):
    """Return whether an async nnInteractive worker still has queued compute.

    The server-level ``active_operation`` flag covers requests already inside
    prediction.  This check closes the small gap between a command being
    queued and prediction setting that flag, while allowing a completed result
    waiting for Mimics to be applied to yield the GPU immediately.
    """
    control_dir = str(state.get("client_control_dir") or "").strip()
    if not control_dir:
        return False
    root = Path(control_dir)
    worker_status_path = root / "worker_status.json"
    worker_status = read_json(worker_status_path, {}) or {}
    if (
        not worker_status_path.is_file()
        and str(state.get("schema_version") or "") == "nninteractive_owned_server.v3"
    ):
        try:
            client_pid = int(state.get("client_pid", 0) or 0)
        except Exception:
            client_pid = 0
        # State publication can precede the first worker-status write. Treat
        # that short initialization window as active instead of terminating a
        # model process which is about to receive its first command.
        if client_pid and process_exists(client_pid):
            return True
    if str(worker_status.get("status") or "").lower() in (
        "initializing", "running", "closing"
    ):
        return True
    try:
        completed_sequence = int(worker_status.get("sequence", 0) or 0)
    except Exception:
        completed_sequence = 0
    try:
        for command_path in (root / "commands").glob("command_*.json"):
            try:
                sequence = int(command_path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if sequence > completed_sequence:
                return True
    except OSError:
        return True
    return False


def _is_owned_nninteractive_state(state):
    return str(state.get("schema_version") or "") in (
        "nninteractive_owned_server.v2",
        "nninteractive_owned_server.v3",
    )


def cleanup_idle_nninteractive_server_lock(current):
    if not isinstance(current, dict):
        return False
    if current.get("resource") != "gpu":
        return False
    state_path = current.get("state_path")
    if not state_path:
        return False
    state = read_json(state_path, {}) or {}
    if not _is_owned_nninteractive_state(state):
        return False
    if state.get("gpu_lock_token") != current.get("token"):
        return False
    try:
        pid = int(state.get("pid", 0))
    except Exception:
        pid = 0
    if not pid:
        return False
    if _nninteractive_operation_is_active(state):
        return False
    try:
        watchdog_pid = int(state.get("watchdog_pid", 0) or 0)
    except Exception:
        watchdog_pid = 0
    if watchdog_pid and process_exists(watchdog_pid):
        return False
    try:
        idle_timeout = float(state.get("service_idle_timeout_seconds", 1800))
        last_activity = float(state.get("last_activity_epoch", time.time()))
    except Exception:
        idle_timeout = 1800.0
        last_activity = time.time()
    if time.time() - last_activity < idle_timeout + 60.0:
        return False
    if process_exists(pid):
        if not terminate_process_tree(pid):
            return False
        deadline = time.time() + 15.0
        while process_exists(pid) and time.time() < deadline:
            time.sleep(0.25)
        if process_exists(pid):
            return False
    latest_state = read_json(state_path, {}) or {}
    if latest_state.get("gpu_lock_token") != current.get("token"):
        return False
    try:
        Path(state_path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return bool(release_lock(
        current.get("path") or _gpu_lock_path(),
        current.get("token"),
    ))


def request_nninteractive_server_release_on_contention(current):
    """Ask an idle nnInteractive server to release GPU when another job is waiting.

    Instead of killing the server immediately, this marks its activity as expired
    so the owned watchdog shuts it down and releases the GPU lock.
    """
    if not isinstance(current, dict):
        return False
    if current.get("resource") != "gpu":
        return False
    state_path = current.get("state_path")
    if not state_path:
        return False
    state = read_json(state_path, {}) or {}
    if not _is_owned_nninteractive_state(state):
        return False
    if state.get("gpu_lock_token") != current.get("token"):
        return False
    try:
        pid = int(state.get("pid", 0))
    except Exception:
        pid = 0
    if not pid or not process_exists(pid):
        return False
    if _nninteractive_operation_is_active(state):
        return False
    if _nninteractive_worker_has_pending_work(state):
        return False
    now = time.time()

    try:
        requested_at = float(state.get("contention_release_requested_epoch", 0.0) or 0.0)
    except Exception:
        requested_at = 0.0
    if requested_at and now - requested_at < 10.0:
        return False

    state["contention_release_requested_epoch"] = now
    state["contention_release_requested_by"] = "shared_gpu_scheduler"
    control_dir = str(state.get("client_control_dir") or "").strip()
    try:
        client_pid = int(state.get("client_pid", 0) or 0)
    except Exception:
        client_pid = 0
    if control_dir and client_pid and process_exists(client_pid):
        # The long-lived image worker owns the graceful shutdown protocol. It
        # remains usable even if the auxiliary watchdog has already failed.
        write_json_atomic(state_path, state)
        write_json_atomic(Path(control_dir) / "close.json", {
            "reason": "gpu_contention",
            "requested_at_epoch": now,
            "requested_by": "shared_gpu_scheduler",
        })
    else:
        try:
            watchdog_pid = int(state.get("watchdog_pid", 0) or 0)
        except Exception:
            watchdog_pid = 0
        if not watchdog_pid or not process_exists(watchdog_pid):
            return False
        # Legacy one-shot bridges have no worker control channel. Expire the
        # server directly; the watchdog owns termination and lock release.
        state["last_activity_epoch"] = 0.0
        write_json_atomic(state_path, state)
    return True


def acquire_background_mimics_lock(
    workspace, status_path, cancel_path, owner, timeout_seconds, scope=None
):
    lock_path = Path(runtime_common.background_mimics_lock_path(
        str(ROOT), str(scope or workspace)
    ))
    lock = FileResourceLock(lock_path, "background_mimics", owner)
    last_log = {"epoch": 0.0}

    def on_wait(current):
        update_status(status_path, {
            "status": "waiting_for_background_mimics",
            "resource_wait": _lock_wait_payload("background_mimics", current),
        })
        now = time.time()
        if now - last_log["epoch"] >= 60:
            holder = current.get("owner", "unknown") if isinstance(current, dict) else "unknown"
            pid = current.get("pid", "?") if isinstance(current, dict) else "?"
            append_log(workspace, "Waiting for background Mimics held by {} (pid {}).".format(holder, pid))
            last_log["epoch"] = now

    update_status(status_path, {
        "status": "waiting_for_background_mimics",
        "resource_wait": {"resource": "background_mimics"},
    })
    lock.acquire(
        wait_seconds=float(timeout_seconds),
        poll_seconds=5.0,
        on_wait=on_wait,
        should_cancel=lambda: cancel_requested(cancel_path),
    )
    update_status(status_path, {"resource_wait": None})
    return lock


def acquire_background_mimics_locks(
    workspace, status_path, cancel_path, owner, timeout_seconds, scopes
):
    """Acquire all resources together without holding one while waiting."""
    unique_scopes = {}
    for scope in scopes:
        lock_path = runtime_common.background_mimics_lock_path(
            str(ROOT), str(scope or workspace)
        )
        unique_scopes[str(lock_path)] = scope
    ordered_scopes = [unique_scopes[path] for path in sorted(unique_scopes)]
    deadline = time.time() + max(0.0, float(timeout_seconds))
    last_notice = 0.0
    while True:
        locks = []
        try:
            for scope in ordered_scopes:
                locks.append(
                    acquire_background_mimics_lock(
                        workspace,
                        status_path,
                        cancel_path,
                        owner,
                        0.0,
                        scope=scope,
                    )
                )
            return locks
        except ResourceLockTimeout as exc:
            for lock in reversed(locks):
                lock.release()
            now = time.time()
            if now >= deadline:
                raise
            if now - last_notice >= 60.0:
                append_log(
                    workspace,
                    "Waiting until the saved .mcs source and label destination "
                    "are both available: {}.".format(exc),
                )
                last_notice = now
            remaining = min(2.0, max(0.0, deadline - now))
            while remaining > 0:
                if cancel_requested(cancel_path):
                    raise ResourceLockCancelled(
                        "Cancelled while waiting for background Mimics resources"
                    )
                step = min(0.25, remaining)
                time.sleep(step)
                remaining -= step
        except Exception:
            for lock in reversed(locks):
                lock.release()
            raise


def terminate_process_tree(pid):
    return pipeline_common.terminate_process_tree(pid)


def terminate_and_reap_process(process, timeout_seconds=15.0):
    """Stop an owned child and confirm it exited before releasing resources."""
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
    try:
        process.wait(timeout=min(5.0, max(0.0, float(timeout_seconds))))
    except Exception:
        try:
            terminate_process_tree(process.pid)
        except Exception:
            pass
        try:
            process.wait(timeout=max(0.0, float(timeout_seconds)))
        except Exception:
            try:
                process.kill()
                process.wait(timeout=10.0)
            except Exception:
                pass
    try:
        return process.poll() is not None
    except Exception:
        return not process_exists(getattr(process, "pid", 0))


_IMAGE_SUFFIXES = (
    ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd.gz", ".nrrd",
)


def _has_suffix(path, suffixes):
    return any(str(path.name).lower().endswith(suffix) for suffix in suffixes)


def find_image(case_dir, profile_id=None):
    """Find the source image in a case directory.

    Preferred names come from the dataset profile (dataset_profiles.json);
    historical variants (mr./image.) are kept as built-in extras so
    behaviour is unchanged.
    """
    from dataset_profiles import load_profile

    profile = load_profile(profile_id)
    preferred = list(profile["image_candidates"])
    for extra in (
        "mr.nii.gz", "image.nii.gz", "mr.nii", "image.nii",
        "mr.mha", "image.mha", "mr.mhd", "image.mhd",
        "mr.nrrd", "image.nrrd",
    ):
        if extra not in preferred:
            preferred.append(extra)
    mask_dirs = {d.lower() for d in profile["mask_dirs"]}
    root = Path(case_dir)
    candidates = []
    if not root.is_dir():
        return None
    for path in root.iterdir():
        if not path.is_file():
            continue
        if _has_suffix(path, _IMAGE_SUFFIXES):
            candidates.append(path)
    for wanted in preferred:
        for path in sorted(candidates):
            if path.name.lower() == wanted:
                return path
    for path in sorted(candidates):
        if not any(part.lower() in mask_dirs for part in path.parts):
            return path
    try:
        from mimics_bridge import is_dicom_folder

        if is_dicom_folder(str(root)):
            return root
    except Exception:
        pass
    return None


def copy_or_link(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        unlink_with_retry(dst)
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copy"


def _write_nifti(array, affine, destination):
    import nibabel as nib
    import numpy as np

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(array), np.asarray(affine, dtype=float))
    image.set_qform(np.asarray(affine, dtype=float), code=1)
    image.set_sform(np.asarray(affine, dtype=float), code=1)
    nib.save(image, str(destination))
    return destination


def materialize_source_image(image_src, image_dst):
    """Convert any supported source image to NIfTI without changing its grid."""
    image_src = Path(image_src)
    image_dst = Path(image_dst)
    lower = image_src.name.lower()
    if image_src.is_file() and lower.endswith(".nii.gz"):
        return copy_or_link(image_src, image_dst)
    if image_src.is_file() and lower.endswith(".nii"):
        import nibabel as nib
        import numpy as np

        image = nib.load(str(image_src))
        _write_nifti(
            np.asanyarray(image.dataobj),
            image.affine,
            image_dst,
        )
        return "nifti_repacked"

    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError(
            "SimpleITK is required to convert source image {}: {}".format(
                image_src, exc
            )
        )
    if image_src.is_dir():
        series_ids = list(
            sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(image_src)) or []
        )
        if len(series_ids) != 1:
            raise RuntimeError(
                "Expected one DICOM series in {}, found {}.".format(
                    image_src, len(series_ids)
                )
            )
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(image_src), series_ids[0]
        )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(image_src))
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(image_dst), True)
    return "converted_to_nifti"


def launch_mimics_export(
    ts_root,
    cases,
    mimics_exe,
    workspace,
    timeout_seconds,
    status_path=None,
    cancel_path=None,
    lock_timeout_seconds=None,
    label_staging_dir=None,
    export_space="source_image",
    mask_names=None,
    target_mask_name=None,
    mcs_output_dir=None,
    skip_projects_without_requested_mask=False,
    skip_invalid_projects=False,
):
    mimics_exe = find_mimics_exe(mimics_exe)
    if not mimics_exe:
        append_log(
            workspace,
            "A separate background Mimics executable was not found. Fresh .mcs "
            "label export cannot start. Set the background Mimics executable "
            "in the configuration (mimics_background_exe), or select an "
            "existing exported masks folder.",
        )
        return {"launched": False, "reason": "mimics_not_found"}
    # The .mcs files live in the configured Mimics output directory, while
    # training control artifacts stay in the caller's workspace.
    output_dir = Path(mcs_output_dir).expanduser().resolve() if mcs_output_dir else resolve_mimics_output_dir(ts_root)
    axes, flips = resolve_mimics_buffer_mapping()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_job_id = safe_slug(Path(status_path).stem if status_path else "export_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"),
        uuid.uuid4().hex[:8],
    ))
    cleanup_local_export_jobs()
    export_root = ROOT / ".mimics_runtime" / "export_jobs" / export_job_id
    export_root.mkdir(parents=True, exist_ok=True)
    if label_staging_dir:
        label_staging_dir = Path(label_staging_dir)
        if label_staging_dir.exists():
            rmtree_with_retry(label_staging_dir)
        label_staging_dir.mkdir(parents=True, exist_ok=True)
    log_path = export_root / "process.log"
    mimics_log_path = export_root / "mimics_application.log"
    batch_status_path = export_root / "status.json"
    handshake_path = export_root / "runner_started.json"
    stop_path = export_root / "_export_stop.json"
    config_path_write = export_root / "export_config.json"
    export_config = {
        "ts_root": str(Path(ts_root).resolve()),
        "cases": sorted(cases) if cases else None,
        "axes": axes,
        "flips": flips,
        "export_space": str(export_space or "source_image"),
        "output_dir": str(output_dir),
        "export_root": str(export_root),
        "status_path": str(batch_status_path),
        "job_runtime": str(export_root),
        "stop_path": str(stop_path),
        "skip_projects_without_requested_mask": bool(
            skip_projects_without_requested_mask
        ),
        "skip_invalid_projects": bool(skip_invalid_projects),
    }
    if mask_names:
        export_config["mask_names"] = [str(name) for name in mask_names if str(name).strip()]
    if target_mask_name:
        export_config["target_mask_name"] = str(target_mask_name).strip()
    if label_staging_dir:
        export_config["label_staging_dir"] = str(label_staging_dir)
    write_json_atomic(config_path_write, export_config)
    runner = export_root / "run_export_batch.py"
    write_text_atomic(
        runner,
        "\n".join([
            "# Auto-generated runner for Mimics label export",
            "import sys, os, json, time",
            "open({}, 'w').write(json.dumps({{'pid': os.getpid(), 'started_at_epoch': time.time()}}))".format(
                json.dumps(str(handshake_path))
            ),
            "sys.path.insert(0, {})".format(json.dumps(str(ROOT / "runtime_py35"))),
            "import mimics_export",
            "mimics_export.run_background_batch_export({})".format(
                json.dumps(str(config_path_write))
            ),
            "",
        ]),
    )
    locks = []
    if status_path and cancel_path:
        locks = acquire_background_mimics_locks(
            workspace,
            status_path,
            cancel_path,
            "label export",
            lock_timeout_seconds if lock_timeout_seconds is not None else timeout_seconds,
            scopes=[output_dir, label_staging_dir or Path(ts_root).resolve()],
        )
    proc = None
    lock_releasable = True
    try:
        log_handle, actual_log_path, log_warning = open_subprocess_log(log_path, workspace, "Background Mimics export log")
        with log_handle as log:
            proc = subprocess.Popen(
                [
                    mimics_exe,
                    "-background_mode",
                    "-save_log",
                    str(mimics_log_path),
                    "-run_script",
                    str(runner),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **hidden_process_kwargs()
            )
        for lock in locks:
            if not lock.update_pid(
                proc.pid,
                kind="label_export",
                ts_root=str(Path(ts_root).resolve()),
                mcs_source=str(output_dir),
                destination_scope=str(label_staging_dir or Path(ts_root).resolve()),
                stop_path=str(stop_path),
                cancel_path=str(cancel_path or ""),
                status_path=str(status_path or ""),
            ):
                raise RuntimeError(
                    "Background Mimics lock ownership was lost while recording PID {}.".format(
                        proc.pid
                    )
                )
        try:
            register_process(
                ROOT, "background_mimics", proc.pid,
                parent_pid=os.getpid(),
                state_path=str(status_path or ""),
            )
        except Exception:
            pass
        append_log(workspace, "Background Mimics export started, pid={}.".format(proc.pid))
        if status_path:
            update_status(status_path, {
                "status": "exporting_labels",
                "pid": proc.pid,
                "controller_pid": os.getpid(),
                "label_export_status": str(batch_status_path),
                "label_export_stop_path": str(stop_path),
            })
        deadline = time.time() + float(timeout_seconds)
        last_progress_signature = None
        stop_requested = False
        while time.time() < deadline:
            if cancel_path and cancel_requested(cancel_path) and not stop_requested:
                stop_requested = True
                write_json_atomic(stop_path, {
                    "status": "stop_requested",
                    "requested_at_epoch": time.time(),
                    "reason": "Training was cancelled during label export",
                })
                if status_path:
                    update_status(status_path, {"status": "cancelling"})
            if status_path and batch_status_path.is_file():
                live_status = read_json(batch_status_path, {}) or {}
                signature = (
                    live_status.get("status"),
                    live_status.get("phase"),
                    live_status.get("case_id"),
                    live_status.get("index"),
                    live_status.get("total"),
                    live_status.get("completed"),
                    live_status.get("failed"),
                )
                if signature != last_progress_signature:
                    last_progress_signature = signature
                    update_status(status_path, {"label_export_progress": live_status})
            if proc.poll() is not None:
                result = {"launched": True, "returncode": proc.returncode}
                batch_status = read_json(batch_status_path, {}) or {}
                if str(batch_status.get("status") or "").lower() not in (
                    "closed", "cancelled", "failed",
                ):
                    batch_status.update({
                        "status": "cancelled" if stop_requested else "failed",
                        "phase": "cancelled" if stop_requested else "process_exited",
                        "pid": proc.pid,
                        "returncode": proc.returncode,
                        "error": (
                            "Label export stopped by cancellation request."
                            if stop_requested
                            else "Background Mimics exited before recording a terminal export state."
                        ),
                        "updated_at_epoch": time.time(),
                    })
                    write_json_atomic(batch_status_path, batch_status)
                if batch_status:
                    result["batch_status"] = batch_status
                result["runner_started"] = handshake_path.is_file()
                result["job_runtime"] = str(export_root)
                result["mimics_log"] = str(mimics_log_path)
                if label_staging_dir:
                    result["label_staging_dir"] = str(label_staging_dir)
                if actual_log_path is not None:
                    result["log"] = str(actual_log_path)
                else:
                    result["log_unavailable"] = True
                if log_warning:
                    result["log_warning"] = log_warning
                return result
            time.sleep(2.0)
        write_json_atomic(stop_path, {
            "status": "stop_requested",
            "requested_at_epoch": time.time(),
            "reason": "Label export timed out",
        })
        lock_releasable = terminate_and_reap_process(proc)
        timed_out_status = read_json(batch_status_path, {}) or {}
        if str(timed_out_status.get("status") or "").lower() not in (
            "closed", "cancelled", "failed",
        ):
            timed_out_status.update({
                "status": "failed" if lock_releasable else "stopping",
                "phase": "timeout" if lock_releasable else "stopping_after_timeout",
                "pid": proc.pid,
                "error": (
                    "Label export timed out."
                    if lock_releasable else
                    "Label export timed out, but the background Mimics "
                    "process has not exited; its resource lock was retained."
                ),
                "updated_at_epoch": time.time(),
            })
            write_json_atomic(batch_status_path, timed_out_status)
        result = {
            "launched": True,
            "timed_out": True,
            "pid": proc.pid,
            "runner_started": handshake_path.is_file(),
            "job_runtime": str(export_root),
            "mimics_log": str(mimics_log_path),
            "batch_status": timed_out_status,
        }
        if label_staging_dir:
            result["label_staging_dir"] = str(label_staging_dir)
        if actual_log_path is not None:
            result["log"] = str(actual_log_path)
        else:
            result["log_unavailable"] = True
        if log_warning:
            result["log_warning"] = log_warning
        return result
    except Exception:
        if proc is not None and proc.poll() is None:
            lock_releasable = terminate_and_reap_process(proc)
        if not lock_releasable and status_path:
            update_status(status_path, {
                "status": "stopping",
                "pid": getattr(proc, "pid", 0),
                "controller_pid": os.getpid(),
                "label_export_status": str(batch_status_path),
                "label_export_stop_path": str(stop_path),
                "termination_pending": True,
            })
        raise
    finally:
        if lock_releasable:
            for lock in reversed(locks):
                lock.release()


def _status_workspace(path):
    path = Path(path)
    if path.parent.name == "jobs":
        return path.parent.parent
    return path.parent


def update_status(path, payload, raise_on_failure=False):
    existing = read_json(path, {}) or {}
    existing.update(payload)
    existing["updated_at_epoch"] = time.time()
    try:
        write_json_atomic(path, existing, retries=8, max_sleep=0.15)
        return True
    except Exception as exc:
        try:
            append_log(
                _status_workspace(path),
                "Warning: could not update job status file {}: {}. The background job continues.".format(
                    Path(path),
                    exc,
                ),
            )
        except Exception:
            pass
        if raise_on_failure:
            raise
        return False
