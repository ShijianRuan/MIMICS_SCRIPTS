# -*- coding: utf-8 -*-
"""Standalone nnInteractive bridge for Mimics segmentation.

This script runs in the nnInteractive virtual environment (Python 3.10+ with
PyTorch). It receives interaction data via JSON stdin, communicates with the
nnInteractive server (auto-launched if needed), and writes the refined mask
as a .u8 buffer.

The bridge auto-manages the nnInteractive server lifecycle:
  - On first call: starts the server as a background subprocess.
  - Subsequent calls: reuse the running server.
  - An owned watchdog stops the service after a configurable idle timeout.
  - No manual server management needed by the annotator.

Protocol (JSON stdin -> JSON stdout):
    Input keys:
        image_path          : str   - Optional NIfTI image or DICOM folder in platform coordinates
        image_source_kind   : str   - Optional "nifti" or "dicom_folder"
        image_expected_shape: [int, int, int] - Optional source-vs-Mimics shape guard
        image_buffer_path   : str   - Optional raw image buffer exported from MIMICS
        image_buffer_shape  : [int, int, int] - Required with image_buffer_path
        image_buffer_dtype  : str   - Optional NumPy dtype, default int16
        interactions        : list  - Ordered point/scribble/box/lasso prompts
        [initial_seg_path]  : str   - Optional starting Mask in MIMICS coordinates
        [initial_seg_shape] : [int, int, int]
        buffer_mapping      : dict  - { platform_to_mimics_axes, platform_to_mimics_flips }
        output_path         : str   - Where to write the refined .u8 buffer
        model_dir           : str   - Path to nnInteractive checkpoint folder
        [bg_interaction_path] : str - Optional background scribble .u8 buffer
        [server_url]        : str   - Optional existing nnInteractive server URL
        [device]            : str   - "auto" (default), "cuda:0", or "cpu"

    Output keys:
        status              : str   - "refined" | "skipped" | "error"
        output_path         : str
        elapsed_seconds     : float
        mode                : str   - "remote" | "local"
        first_call          : bool  - True if server was just started (model loading)
        [error]             : str   - Present only when status == "error"
"""

from __future__ import annotations

import itertools
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    process_matches as resource_process_matches,
    process_start_marker as resource_process_start_marker,
    register_process,
    release_lock,
)


# ---------------------------------------------------------------------------
#  Server lifecycle management
# ---------------------------------------------------------------------------

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1527
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 600  # first CPU startup can take several minutes
HEALTHZ_RETRY_INTERVAL = 0.5   # seconds between healthz checks
SERVER_IDLE_TIMEOUT = 1800
SERVER_HEARTBEAT_STALE_SECONDS = 90.0
SERVER_IDENTITY_STARTUP_GRACE_SECONDS = SERVER_STARTUP_TIMEOUT + 60.0
LOG_ROTATE_BYTES = 10 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3
PROJECT_ROOT = Path(__file__).resolve().parent


def _gpu_lock_path() -> Path:
    """Resolve the GPU lock path per call, honoring env overrides.

    A module-level constant would freeze the path at import time; test
    suites that isolate locks via MIMICS_RESOURCE_LOCK_DIR after import
    would still touch the production lock directory.
    """
    return default_resource_lock_dir(PROJECT_ROOT) / "gpu.lock"


def _rotate_log_file(path: Path, max_bytes: int = LOG_ROTATE_BYTES, backups: int = LOG_ROTATE_BACKUPS) -> None:
    try:
        if not path.is_file() or path.stat().st_size < max_bytes:
            return
        backups = int(backups)
        if backups <= 0:
            path.unlink()
            return
        oldest = path.with_name(f"{path.name}.{backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{index}")
            dst = path.with_name(f"{path.name}.{index + 1}")
            if src.exists():
                src.rename(dst)
        path.rename(path.with_name(path.name + ".1"))
    except Exception:
        pass


def _gpu_lock_enabled(device: str) -> bool:
    if os.environ.get("MIMICS_DISABLE_GPU_LOCK", "").strip().lower() in ("1", "true", "yes"):
        return False
    return str(device or "").lower().startswith("cuda")


def _check_free_gpu_memory(device: str) -> None:
    """Refuse to start a CUDA server when free VRAM is below the floor.

    Prevents the classic "server starts, model load dies with CUDA OOM
    three minutes in" failure. Runs before the GPU lock so the user gets
    an actionable message instead of a lock timeout. Skipped silently
    when CUDA is unavailable or the check itself errors (never blocks a
    working start on a probe failure).
    """
    if not str(device or "").lower().startswith("cuda"):
        return
    minimum_gb = float(os.environ.get("NNINTERACTIVE_MINIMUM_FREE_GPU_MEMORY_GB", "4"))
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            "Not enough free GPU memory to start the nnInteractive server.\n\n"
            f"Free: {free_gb:.1f} GB of {total_gb:.1f} GB "
            f"(minimum required: {minimum_gb:.0f} GB).\n\n"
            "Close other GPU programs or stop running AI training jobs, "
            "then try again."
        )


def _release_gpu_lock_from_state(state: dict[str, Any] | None) -> bool:
    if not state:
        return True
    lock_path = state.get("gpu_lock_path")
    token = state.get("gpu_lock_token")
    if not lock_path or not token:
        return True
    for attempt in range(3):
        if release_lock(lock_path, str(token)):
            return True
        time.sleep(0.05 * (attempt + 1))
    return False


def _runtime_work_dir(model_dir: str, runtime_work_dir: str | None = None) -> Path:
    if runtime_work_dir:
        root = Path(runtime_work_dir)
    else:
        root = Path(model_dir).parent
    root.mkdir(parents=True, exist_ok=True)
    return root


def _server_state_path(model_dir: str, runtime_work_dir: str | None = None) -> Path:
    """Path to the owned-server state file."""
    return _runtime_work_dir(model_dir, runtime_work_dir) / ".nninteractive_server.json"


def _server_log_path(model_dir: str, runtime_work_dir: str | None = None) -> Path:
    """Path to the server log file."""
    return _runtime_work_dir(model_dir, runtime_work_dir) / ".nninteractive_server.log"


def _bridge_log_path(model_dir: str, log_dir: str | None = None) -> Path:
    root = Path(log_dir) if log_dir else Path(model_dir).parent / "logs"
    return root / "nninteractive_bridge.jsonl"


def _append_bridge_log(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_log_file(path)
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
        "pid": os.getpid(),
    }
    payload.update(details)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _hidden_process_kwargs(*, detached: bool = False) -> dict[str, Any]:
    """Create subprocess options that do not open a Windows console."""
    if os.name != "nt":
        return {"start_new_session": True} if detached else {}

    # Use hardcoded Win32 constants for reliability across Python versions.
    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0
    CREATE_NO_WINDOW = 0x08000000
    CREATE_NEW_PROCESS_GROUP = 0x00000200

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE
    flags = CREATE_NO_WINDOW
    if detached:
        flags |= CREATE_NEW_PROCESS_GROUP
    return {"startupinfo": startupinfo, "creationflags": flags}


def _server_address(server_url: str) -> tuple[str, int]:
    normalized = str(server_url or "").strip()
    # Accept common malformed forms from config/env, e.g. "http:///127.0.0.1:1527".
    if normalized.startswith("http:///"):
        normalized = "http://" + normalized[len("http:///"):]
    if normalized and "://" not in normalized:
        normalized = "http://" + normalized

    parsed = urlparse(normalized)
    if parsed.scheme == "http" and not parsed.hostname and parsed.path:
        # Recover URLs where host:port was parsed as path due to extra slash.
        candidate = parsed.path.lstrip("/")
        reparsed = urlparse("http://" + candidate)
        if reparsed.hostname:
            parsed = reparsed

    if parsed.scheme != "http" or not parsed.hostname:
        raise RuntimeError(f"Unsupported nnInteractive server URL: {server_url}")
    return parsed.hostname, int(parsed.port or 80)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _available_server_url(preferred_url: str) -> str:
    """Return a server URL whose port is free.

    If the preferred port is already in use, bind an ephemeral socket to find
    a free local port.  This prevents port conflicts from blocking the user
    while still avoiding the memory-exhaustion risk of launching multiple
    model servers (the caller ensures only one owned server exists at a time
    via ``_ensure_server``).
    """
    normalized = str(preferred_url or "").strip()
    if normalized.startswith("http:///"):
        normalized = "http://" + normalized[len("http:///"):]
    if normalized and "://" not in normalized:
        normalized = "http://" + normalized
    host, port = _server_address(normalized)
    if not _port_open(host, port):
        return normalized
    # Preferred port is occupied; find a free one via ephemeral socket.
    free_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        free_sock.bind((host, 0))
        free_port = free_sock.getsockname()[1]
    finally:
        free_sock.close()
    fallback = f"http://{host}:{free_port}"
    try:
        _append_bridge_log(
            _bridge_log_path("."),  # best-effort log; path may not be set yet
            "port_conflict_fallback",
            preferred_url=normalized,
            fallback_url=fallback,
        )
    except Exception:
        pass
    return fallback


def _resolve_device(requested: str, allow_cpu_fallback: bool = True) -> tuple[str, str | None]:
    normalized = str(requested or "auto").strip().lower()
    if normalized == "cpu":
        return "cpu", None
    import torch

    if normalized in ("", "auto"):
        return ("cuda:0", None) if torch.cuda.is_available() else (
            "cpu",
            "CUDA is unavailable; nnInteractive will run on CPU.",
        )
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise RuntimeError(
                f"nnInteractive requested device {requested!r}, but torch.cuda.is_available() is false."
            )
        return (
            "cpu",
            f"Requested device {requested!r} is unavailable; falling back to CPU.",
        )
    return requested, None


def _resolve_fold(model_dir: str, requested: Any) -> str | None:
    folds = sorted(
        path.name.split("_", 1)[1]
        for path in Path(model_dir).glob("fold_*")
        if path.is_dir() and (path / "checkpoint_final.pth").is_file()
    )
    if not folds:
        raise RuntimeError(
            f"No fold_*/checkpoint_final.pth was found under model directory: {model_dir}"
        )
    if requested in (None, "", "auto"):
        return None
    value = str(requested)
    if value == "all":
        if len(folds) == 1:
            return folds[0]
        return "all"
    if value not in folds:
        raise RuntimeError(
            f"Requested fold {value!r} is unavailable. Available folds: {', '.join(folds)}"
        )
    return value


def _selected_checkpoint_identity(model_dir: str, fold: str | None) -> str:
    """Return the exact checkpoint identity that the server will load.

    Task models currently register one checkpoint SHA256. When more than one
    fold is selected, use a deterministic set identity rather than pretending
    that one member checksum identifies the ensemble.
    """
    root = Path(model_dir)
    checkpoints = sorted(
        path
        for path in root.glob("fold_*/checkpoint_final.pth")
        if path.is_file()
    )
    if fold not in (None, "", "auto", "all"):
        checkpoints = [
            root / "fold_{}".format(fold) / "checkpoint_final.pth"
        ]
    if not checkpoints or any(not path.is_file() for path in checkpoints):
        raise RuntimeError(
            "The selected nnInteractive checkpoint is missing under: {}".format(
                model_dir
            )
        )

    checksums = []
    for path in checkpoints:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksums.append((path.parent.name, digest.hexdigest()))
    if len(checksums) == 1:
        return checksums[0][1]

    combined = hashlib.sha256()
    for fold_name, checksum in checksums:
        combined.update(fold_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(checksum.encode("ascii"))
        combined.update(b"\0")
    return "set:{}".format(combined.hexdigest())


def _write_text_atomic(path: Path, text: str, retries: int = 12) -> None:
    """Publish text reliably without deleting a readable previous state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    last_error: OSError | None = None
    direct_write_succeeded = False
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
            except OSError as exc:
                last_error = exc
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(min(0.15, 0.02 * (attempt + 1)))

        # Some SMB shares allow writes but reject replace. Readers already
        # tolerate a brief incomplete JSON document, so use a flushed direct
        # write only after bounded atomic-replace retries.
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        direct_write_succeeded = True
    except OSError as exc:
        last_error = exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    if not direct_write_succeeded and last_error is not None:
        raise last_error


def _write_server_state(path: Path, state: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(state, indent=2))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
    )


def _load_server_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _process_exists(pid: int) -> bool:
    """Use the shared conservative process check, including Windows exit code 259."""
    return resource_process_exists(pid)


def _windows_process_command_line(pid: int) -> str | None:
    try:
        import psutil

        values = psutil.Process(pid).cmdline()
        if values:
            return subprocess.list2cmdline([str(value) for value in values])
    except Exception:
        pass
    commands = [
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "(Get-CimInstance Win32_Process -Filter "
                "\"ProcessId = {0}\").CommandLine"
            ).format(pid),
        ],
        [
            "wmic",
            "process",
            "where",
            "ProcessId={0}".format(pid),
            "get",
            "CommandLine",
            "/value",
        ],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=6,
                check=False,
                **_hidden_process_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = result.stdout.strip()
        if not output:
            continue
        if command[0].lower() == "wmic":
            for line in output.splitlines():
                if line.strip().lower().startswith("commandline="):
                    output = line.split("=", 1)[1].strip()
                    break
        if output:
            return output
    return None


def _process_command_line(pid: int) -> str | None:
    if os.name == "nt":
        return _windows_process_command_line(pid)
    try:
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.is_file():
            return proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _server_heartbeat_is_recent(
    state: dict[str, Any],
    *,
    now: float | None = None,
) -> bool:
    try:
        heartbeat = float(state.get("server_heartbeat_epoch", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    return heartbeat > 0.0 and 0.0 <= current - heartbeat <= SERVER_HEARTBEAT_STALE_SECONDS


def _server_is_within_startup_grace(
    state: dict[str, Any],
    *,
    now: float | None = None,
) -> bool:
    try:
        started = float(state.get("started_at_epoch", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    return started > 0.0 and 0.0 <= current - started <= SERVER_IDENTITY_STARTUP_GRACE_SECONDS


def _server_health_matches_state(state: dict[str, Any]) -> bool:
    server_url = str(state.get("server_url") or "").strip()
    ownership_token = str(state.get("ownership_token") or "").strip()
    return bool(
        server_url
        and ownership_token
        and _server_running(server_url, ownership_token)
    )


def _owned_server_operation_is_active(
    state: dict[str, Any],
    *,
    now: float | None = None,
) -> bool:
    if state.get("active_operation") != "prediction":
        return False
    try:
        pid = int(state.get("active_operation_pid", 0) or 0)
        started = float(
            state.get("active_operation_started_at_epoch", 0.0) or 0.0
        )
        timeout = max(
            60.0,
            float(state.get("active_operation_timeout_seconds", 3600.0) or 3600.0),
        )
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    if started <= 0.0 or not 0.0 <= current - started <= timeout:
        return False
    return resource_process_matches(
        pid,
        state.get("active_operation_process_start_marker"),
    )


def _process_matches_server(state: dict[str, Any]) -> bool:
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if not _process_exists(pid):
        return False
    command_line = _process_command_line(pid)
    if not command_line:
        # PID alone is not identity: Windows can recycle it after a crash.
        # An unreadable command line needs a second, bounded liveness signal.
        return (
            _server_health_matches_state(state)
            or _server_heartbeat_is_recent(state)
            or _server_is_within_startup_grace(state)
            or _owned_server_operation_is_active(state)
        )
    normalized = command_line.lower() if os.name == "nt" else command_line
    required = [
        "nnInteractive.inference.server.main",
        str(Path(str(state.get("model_dir", ""))).resolve()),
        str(state.get("ownership_token", "")),
    ]
    if os.name == "nt":
        required = [value.lower() for value in required]
    return all(value and value in normalized for value in required)


def _server_running(server_url: str, api_key: str | None = None) -> bool:
    """Check whether the expected server answers its health endpoint."""
    import urllib.request

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        req = urllib.request.Request(f"{server_url}/healthz", headers=headers)
        resp = urllib.request.urlopen(req, timeout=3)
        return resp.status == 200
    except Exception:
        return False


def _remove_server_state(path: Path, ownership_token: str | None = None) -> bool:
    """Retire owned state before releasing its GPU lock.

    Keeping the lock until the state file is gone prevents a replacement
    server from acquiring the GPU, writing a new state file, and then having
    that new state removed by an older cleanup path. If Windows temporarily
    refuses deletion, first try to rename the state out of the canonical path.
    As a final fallback, release only after the recorded server is confirmed
    dead; the ownership-token check prevents this cleanup from touching a
    replacement server.
    """
    state = _load_server_state(path)
    if ownership_token and (
        not state or state.get("ownership_token") != ownership_token
    ):
        return False
    for attempt in range(8):
        try:
            path.unlink()
            return _release_gpu_lock_from_state(state)
        except FileNotFoundError:
            return _release_gpu_lock_from_state(state)
        except OSError:
            time.sleep(min(0.2, 0.025 * (attempt + 1)))

    retired_path = path.with_name(
        "{0}.retired.{1}.{2}".format(
            path.name,
            str(ownership_token or state.get("ownership_token") or "unknown")[:12],
            uuid.uuid4().hex,
        )
    )
    try:
        path.replace(retired_path)
    except FileNotFoundError:
        return _release_gpu_lock_from_state(state)
    except OSError:
        latest = _load_server_state(path)
        if ownership_token and (
            not latest or latest.get("ownership_token") != ownership_token
        ):
            return False
        if state and _process_matches_server(state):
            return False
        return _release_gpu_lock_from_state(state)

    try:
        retired_path.unlink()
    except OSError:
        pass
    return _release_gpu_lock_from_state(state)


def _terminate_owned_server(state: dict[str, Any]) -> bool:
    """Terminate only a process whose command line matches our ownership record."""
    if not _process_matches_server(state):
        return False
    pid = int(state["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.25)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            **_hidden_process_kwargs(),
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.25)
    return not _process_exists(pid)


def _stop_spawned_server(process: subprocess.Popen, timeout_seconds: float = 10.0) -> bool:
    """Stop a server started by this process and confirm it has exited."""
    try:
        if process.poll() is not None:
            return True
    except Exception:
        pass
    try:
        process.terminate()
        process.wait(timeout=timeout_seconds)
        return True
    except Exception:
        pass
    try:
        process.kill()
        process.wait(timeout=timeout_seconds)
        return True
    except Exception:
        pass
    try:
        return not _process_exists(int(process.pid))
    except Exception:
        return False


def _terminate_model_servers(model_dir: str) -> int:
    """Best-effort cleanup of old nnInteractive servers for this model.

    Loading several CPU model servers at once can exhaust RAM and make Mimics,
    VS Code, and the server all fail in unrelated-looking ways. Only processes
    whose command line explicitly names nnInteractive.inference.server.main and
    this model directory are terminated.
    """
    if os.name != "nt":
        return 0
    model_path = str(Path(model_dir).resolve()).lower()
    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*nnInteractive.inference.server.main*' } | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            ),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
            **_hidden_process_kwargs(),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0
        records = json.loads(result.stdout)
        if isinstance(records, dict):
            records = [records]
        stopped = 0
        for record in records or []:
            try:
                pid = int(record.get("ProcessId"))
            except (TypeError, ValueError):
                continue
            command_line = str(record.get("CommandLine") or "")
            if pid == os.getpid() or model_path not in command_line.lower():
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
                **_hidden_process_kwargs(),
            )
            stopped += 1
        return stopped
    except Exception:
        return 0


def _touch_server_activity(state_path: Path, ownership_token: str) -> None:
    state = _load_server_state(state_path)
    if not state or state.get("ownership_token") != ownership_token:
        return
    state["last_activity_epoch"] = time.time()
    _write_server_state(state_path, state)


def _record_server_heartbeat(state_path: Path, ownership_token: str) -> None:
    """Record a successful token-authenticated server health check."""
    state = _load_server_state(state_path)
    if not state or state.get("ownership_token") != ownership_token:
        return
    state["server_heartbeat_epoch"] = time.time()
    _write_server_state(state_path, state)


def _set_server_operation_active(
    state_path: Path,
    ownership_token: str,
    active: bool,
) -> None:
    """Publish whether an owned server is serving an inference operation."""
    state = _load_server_state(state_path)
    if not state or state.get("ownership_token") != ownership_token:
        return
    state["last_activity_epoch"] = time.time()
    if active:
        state["active_operation"] = "prediction"
        state["active_operation_pid"] = os.getpid()
        state["active_operation_started_at_epoch"] = time.time()
        marker = resource_process_start_marker(os.getpid())
        if marker:
            state["active_operation_process_start_marker"] = marker
    else:
        state.pop("active_operation", None)
        state.pop("active_operation_pid", None)
        state.pop("active_operation_started_at_epoch", None)
        state.pop("active_operation_process_start_marker", None)
    _write_server_state(state_path, state)


def _expire_server_activity(state_path: Path, ownership_token: str) -> None:
    """Set last_activity_epoch far in the past so the watchdog shuts down the server on its next check."""
    state = _load_server_state(state_path)
    if not state or state.get("ownership_token") != ownership_token:
        return
    state["last_activity_epoch"] = 0.0
    _write_server_state(state_path, state)


def _watchdog_main(state_path_value: str, ownership_token: str) -> int:
    state_path = Path(state_path_value)
    while True:
        state = _load_server_state(state_path)
        if not state or state.get("ownership_token") != ownership_token:
            return 0
        if not _process_matches_server(state):
            if _remove_server_state(state_path, ownership_token):
                return 0
            time.sleep(5.0)
            continue
        state_url = str(state.get("server_url") or "").strip()
        if state_url and _server_running(state_url, ownership_token):
            _record_server_heartbeat(state_path, ownership_token)
            state = _load_server_state(state_path) or state
        try:
            operation_pid = int(state.get("active_operation_pid", 0) or 0)
        except (TypeError, ValueError):
            operation_pid = 0
        if state.get("active_operation") and operation_pid and _process_exists(operation_pid):
            started = float(state.get("active_operation_started_at_epoch", time.time()) or time.time())
            timeout = max(60.0, float(state.get("active_operation_timeout_seconds", 3600) or 3600))
            if time.time() - started <= timeout:
                time.sleep(5.0)
                continue
        idle_timeout = float(state.get("service_idle_timeout_seconds", SERVER_IDLE_TIMEOUT))
        last_activity = float(state.get("last_activity_epoch", time.time()))
        remaining = idle_timeout - (time.time() - last_activity)
        if remaining <= 0:
            if _terminate_owned_server(state):
                if _remove_server_state(state_path, ownership_token):
                    return 0
            time.sleep(5.0)
            continue
        time.sleep(max(1.0, min(30.0, remaining)))


def _start_watchdog(state_path: Path, ownership_token: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--watchdog",
            str(state_path),
            ownership_token,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **_hidden_process_kwargs(detached=True),
    )


def _registry_project_root() -> str:
    """Project root for the process registry (parent of the lock dir)."""
    return str(PROJECT_ROOT)


def _start_server(
    model_dir: str,
    device: str,
    service_idle_timeout_seconds: float,
    server_url: str,
    fold: str | None,
    gpu_lock_timeout_seconds: float = 30.0,
    runtime_work_dir: str | None = None,
    checkpoint_identity: str = "",
) -> tuple[subprocess.Popen, dict[str, Any]]:
    """Start the nnInteractive server as a background subprocess.

    Returns the Popen object. The server processes requests after the model
    is loaded (~10-30 seconds on first start).
    """
    # Use the same Python that's running this bridge (guaranteed to have the
    # nnInteractive package available, whether from venv or portable bundle).
    python_exe = sys.executable
    log_path = _server_log_path(model_dir, runtime_work_dir)
    state_path = _server_state_path(model_dir, runtime_work_dir)
    ownership_token = uuid.uuid4().hex
    host, port = _server_address(server_url)

    # Build env: ensure site-packages from the portable bundle are on PYTHONPATH.
    env = os.environ.copy()
    bundle_site = os.path.join(os.path.dirname(python_exe), "..", "Lib", "site-packages")
    bundle_site = os.path.normpath(bundle_site)
    if os.path.isdir(bundle_site):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bundle_site + (";" + existing if existing else "")

    cmd = [
        python_exe,
        "-m", "nnInteractive.inference.server.main",
        "--model-dir", str(model_dir),
        "--host", host,
        "--port", str(port),
        "--device", device,
        "--idle-timeout-seconds", str(int(service_idle_timeout_seconds)),
        "--liveness-timeout-seconds", "120",
        "--max-sessions", "1",
        "--api-key", ownership_token,
        "--no-torch-compile",
    ]
    if fold is not None:
        cmd.extend(["--fold", fold])

    gpu_lock = None
    if _gpu_lock_enabled(device):
        _check_free_gpu_memory(device)
        gpu_lock = FileResourceLock(
            _gpu_lock_path(),
            "gpu",
            "nnInteractive server ({0})".format(Path(model_dir).name),
        )
        try:
            gpu_lock.acquire(wait_seconds=float(gpu_lock_timeout_seconds), poll_seconds=2.0)
        except ResourceLockTimeout as exc:
            raise RuntimeError(
                "GPU is already in use by another Mimics-Script AI job. "
                "Stop the running job or wait for it to finish. {0}".format(exc)
            ) from exc

    _rotate_log_file(log_path)
    try:
        with open(log_path, "a") as log_fh:
            log_fh.write(f"\n{'='*60}\n")
            log_fh.write(f"Starting nnInteractive server at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logged_cmd = ["<redacted>" if value == ownership_token else value for value in cmd]
            log_fh.write(f"Command: {' '.join(logged_cmd)}\n")
            log_fh.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                **_hidden_process_kwargs(detached=True),
            )
    except Exception:
        if gpu_lock is not None:
            gpu_lock.release()
        raise
    try:
        if gpu_lock is not None and not gpu_lock.update_pid(
            proc.pid,
            server_url=server_url,
            model_dir=str(Path(model_dir).resolve()),
            state_path=str(state_path),
        ):
            raise RuntimeError(
                "nnInteractive server started, but GPU lock ownership could "
                "not be transferred to PID {}.".format(proc.pid)
            )

        state = {
            "schema_version": "nninteractive_owned_server.v3",
            "pid": proc.pid,
            "server_url": server_url,
            "model_dir": str(Path(model_dir).resolve()),
            "device": device,
            "fold": fold or "auto",
            "checkpoint_sha256": str(checkpoint_identity or ""),
            "ownership_token": ownership_token,
            "started_at_epoch": time.time(),
            "last_activity_epoch": time.time(),
            "server_heartbeat_epoch": 0.0,
            "service_idle_timeout_seconds": float(service_idle_timeout_seconds),
        }
        if gpu_lock is not None:
            state["gpu_lock_path"] = str(gpu_lock.path)
            state["gpu_lock_token"] = gpu_lock.token
        _write_server_state(state_path, state)
        watchdog = _start_watchdog(state_path, ownership_token)
        state["watchdog_pid"] = watchdog.pid
        _write_server_state(state_path, state)
        # Register both processes with the unified registry so the startup
        # sweep and the health panel can see them. Best-effort only: the
        # server state JSON above stays the source of truth.
        try:
            register_process(
                _registry_project_root(),
                "nninteractive_server",
                proc.pid,
                ownership_token=ownership_token,
                parent_pid=os.getpid(),
                state_path=str(state_path),
            )
            register_process(
                _registry_project_root(),
                "nninteractive_watchdog",
                watchdog.pid,
                ownership_token=ownership_token,
                parent_pid=proc.pid,
                state_path=str(state_path),
            )
        except Exception:
            pass
        return proc, state
    except Exception:
        process_stopped = _stop_spawned_server(proc)
        if process_stopped:
            _remove_server_state(state_path, ownership_token)
        if gpu_lock is not None and process_stopped:
            # Release through the lock object even when an older, undeletable
            # state file occupies state_path. Token validation prevents this
            # from releasing a replacement owner's lock.
            gpu_lock.release()
        raise


def _wait_for_server(
    server_url: str,
    api_key: str,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    process: subprocess.Popen | None = None,
) -> bool:
    """Block until the server responds to healthz or timeout.

    Returns True if the server is ready.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_running(server_url, api_key):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(HEALTHZ_RETRY_INTERVAL)
    return False


def _ensure_server(
    model_dir: str,
    device: str,
    service_idle_timeout_seconds: float = SERVER_IDLE_TIMEOUT,
    startup_timeout_seconds: float = SERVER_STARTUP_TIMEOUT,
    preferred_server_url: str = SERVER_URL,
    fold: str | None = None,
    gpu_lock_timeout_seconds: float = 30.0,
    runtime_work_dir: str | None = None,
    expected_checkpoint_sha256: str = "",
) -> tuple[bool, str, str]:
    """Ensure the nnInteractive server is running; start it if needed.

    Returns (first_call, server_url, api_key).
      - first_call=True means the server was just started (model loading).
    """
    state_path = _server_state_path(model_dir, runtime_work_dir)
    state = _load_server_state(state_path)
    expected_model = str(Path(model_dir).resolve())
    fold = _resolve_fold(model_dir, fold)
    expected_checkpoint = str(expected_checkpoint_sha256 or "").strip().lower()
    if state and _process_matches_server(state):
        state_url = str(state.get("server_url") or preferred_server_url)
        matches_request = (
            str(state.get("model_dir")) == expected_model
            and str(state.get("device")) == str(device)
            and str(state.get("fold") or "auto") == str(fold or "auto")
            and (
                not expected_checkpoint
                or str(state.get("checkpoint_sha256") or "").strip().lower()
                == expected_checkpoint
            )
        )
        api_key = str(state.get("ownership_token") or "")
        if matches_request and _server_running(state_url, api_key):
            _touch_server_activity(state_path, api_key)
            return False, state_url, api_key
        if not _terminate_owned_server(state):
            raise RuntimeError(
                "The previous owned nnInteractive server did not exit; its GPU "
                "lock was retained and a replacement server was not started."
            )
    if state:
        if not _remove_server_state(
            state_path,
            str(state.get("ownership_token") or ""),
        ):
            raise RuntimeError(
                "The previous nnInteractive server stopped, but its owned "
                "state/GPU lock could not be retired safely. Retry after the "
                "file lock is released."
            )

    legacy_pid = Path(model_dir).parent / ".nninteractive_server.pid"
    try:
        legacy_pid.unlink()
    except FileNotFoundError:
        pass

    _terminate_model_servers(model_dir)
    checkpoint_identity = ""
    if expected_checkpoint:
        checkpoint_identity = _selected_checkpoint_identity(model_dir, fold)
        if checkpoint_identity.lower() != expected_checkpoint:
            raise RuntimeError(
                "The selected nnInteractive checkpoint does not match the "
                "registered model identity. Refusing to start a different or "
                "partially replaced model."
            )
    server_url = _available_server_url(preferred_server_url)
    proc, state = _start_server(
        model_dir,
        device,
        service_idle_timeout_seconds,
        server_url,
        fold,
        gpu_lock_timeout_seconds,
        runtime_work_dir,
        checkpoint_identity,
    )
    api_key = str(state["ownership_token"])

    # Wait for the server to be ready.
    ready = _wait_for_server(
        server_url,
        api_key,
        startup_timeout_seconds,
        process=proc,
    )
    if not ready:
        # Server failed to start. Clean up.
        process_stopped = _stop_spawned_server(proc, timeout_seconds=5.0)
        if process_stopped:
            _remove_server_state(state_path, api_key)
        else:
            raise RuntimeError(
                "nnInteractive server did not become ready and could not be "
                "stopped. Its GPU lock was retained to prevent another job "
                "from using the same GPU concurrently."
            )
        exit_detail = (
            f" The server process exited with code {proc.returncode}."
            if proc.returncode is not None
            else ""
        )
        raise RuntimeError(
            "nnInteractive server did not become ready within {0}s.{1} "
            "Check log: {2}".format(
                startup_timeout_seconds,
                exit_detail,
                _server_log_path(model_dir, runtime_work_dir),
            )
        )

    _record_server_heartbeat(state_path, api_key)

    return True, server_url, api_key


# ---------------------------------------------------------------------------
#  Buffer mapping (platform <-> MIMICS)
# ---------------------------------------------------------------------------

def _apply_mapping(array: np.ndarray, axes: list[int], flips: list[bool]) -> np.ndarray:
    """Transpose + flip a numpy array according to the buffer mapping."""
    result = np.transpose(array, axes)
    for axis, flip in enumerate(flips):
        if flip:
            result = np.flip(result, axis=axis)
    return result.copy()


def _invert_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Compute the inverse of a buffer mapping."""
    axes: list[int] = list(mapping["platform_to_mimics_axes"])
    flips: list[bool] = list(mapping["platform_to_mimics_flips"])
    inv_axes = [0] * len(axes)
    for i, a in enumerate(axes):
        inv_axes[a] = i
    inv_flips = [flips[inv_axes[i]] for i in range(len(flips))]
    return {"platform_to_mimics_axes": inv_axes, "platform_to_mimics_flips": inv_flips}


def mimics_to_platform(array: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    """Transform a MIMICS-coordinate array into platform coordinates."""
    inv = _invert_mapping(mapping)
    return _apply_mapping(array, inv["platform_to_mimics_axes"], inv["platform_to_mimics_flips"])


def mimics_shape_to_platform_shape(shape: list[int], mapping: dict[str, Any]) -> list[int]:
    inv = _invert_mapping(mapping)
    mimics_shape = [int(value) for value in shape]
    return [mimics_shape[int(axis)] for axis in inv["platform_to_mimics_axes"]]


def platform_to_mimics(array: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    """Transform a platform-coordinate array into MIMICS coordinates."""
    return _apply_mapping(
        array,
        list(mapping["platform_to_mimics_axes"]),
        list(mapping["platform_to_mimics_flips"]),
    )


def canonical_ras_buffer_mapping(voxel_to_ras: Any) -> dict[str, Any]:
    """Return a lossless Mimics-array mapping to closest canonical RAS."""
    value = voxel_to_ras
    if isinstance(value, str):
        value = json.loads(value)
    affine = np.asarray(value, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("Mimics voxel-to-RAS matrix is missing or invalid")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1.0e-8:
        raise ValueError("Mimics voxel-to-RAS matrix is singular")

    # Lazy import: nibabel pulls in pydicom and is slow to import (it
    # dominated cold-start of every async worker that touched this
    # module). mimics_bridge.py already follows this pattern.
    import nibabel as nib

    orientation = nib.orientations.io_orientation(affine)
    probe = np.arange(2 * 3 * 4, dtype=np.int16).reshape((2, 3, 4))
    expected = nib.orientations.apply_orientation(probe, orientation)
    for axes in itertools.permutations((0, 1, 2)):
        for flips in itertools.product((False, True), repeat=3):
            mapping = {
                "platform_to_mimics_axes": list(axes),
                "platform_to_mimics_flips": list(flips),
            }
            candidate = mimics_to_platform(probe, mapping)
            if candidate.shape == expected.shape and np.array_equal(
                candidate, expected
            ):
                return mapping
    raise RuntimeError(
        "Could not express the Mimics-to-canonical-RAS orientation as an "
        "axis permutation and flip."
    )


# ---------------------------------------------------------------------------
#  nnInteractive session
# ---------------------------------------------------------------------------

def _connect_remote(
    server_url: str,
    api_key: str | None = None,
    *,
    prediction_timeout_seconds: float = 1800,
    set_image_timeout_seconds: float = 1800,
) -> Any:
    """Connect to a running nnInteractive server."""
    from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession

    if api_key is None:
        api_key = os.environ.get("NN_INTERACTIVE_API_KEY")
    return nnInteractiveRemoteInferenceSession(
        server_url=server_url,
        api_key=api_key,
        read_timeout=prediction_timeout_seconds,
        set_image_read_timeout=set_image_timeout_seconds,
        write_timeout=max(120.0, min(set_image_timeout_seconds, 600.0)),
    )


def _connect_local(model_dir: str, device: str) -> Any:
    """Load the nnInteractive model locally (in-process)."""
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count() or 4,
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(str(model_dir))
    return session


# ---------------------------------------------------------------------------
#  Image loading
# ---------------------------------------------------------------------------

def load_image_nifti(path: str) -> np.ndarray:
    """Load a NIfTI image as a float32 4D array with shape (1, X, Y, Z)."""
    # Lazy import — see canonical_ras_buffer_mapping.
    import nibabel as nib

    nii = nib.load(path)
    data = np.asarray(nii.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[None]
    elif data.ndim == 4:
        data = data[0:1]
    else:
        raise RuntimeError(f"Unexpected image dimensions: {data.ndim}")
    return data


def load_image_medical_file(path: str) -> np.ndarray:
    """Load MHD/MHA/NRRD source physical values in XYZ array order."""
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError(
            "SimpleITK is required to read this source medical image"
        ) from exc
    image = sitk.ReadImage(str(path))
    data_zyx = sitk.GetArrayFromImage(image)
    if data_zyx.ndim != 3:
        raise RuntimeError(
            "Unexpected medical image dimensions: {}".format(data_zyx.ndim)
        )
    return np.transpose(
        np.asarray(data_zyx, dtype=np.float32), (2, 1, 0)
    )[None]


def _parse_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        value = json.loads(value)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _affine_close(left: np.ndarray, right: np.ndarray, tolerance: float = 1.0e-5) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=tolerance))


def _resample_image_to_mimics_grid(
    data: np.ndarray,
    source_voxel_to_ras: np.ndarray,
    mimics_voxel_to_ras: np.ndarray,
    expected_shape: list[int],
) -> np.ndarray:
    source_to_mimics = np.linalg.inv(source_voxel_to_ras) @ mimics_voxel_to_ras
    expected = [int(value) for value in expected_shape]
    if list(data.shape) == expected and _affine_close(source_to_mimics, np.eye(4)):
        return data.astype(np.float32, copy=False)

    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError(
            "Source image orientation differs from the open Mimics image, but scipy "
            "is not available in the external nnInteractive environment for "
            "background resampling."
        ) from exc

    output = np.empty(tuple(expected), dtype=np.float32)
    max_slab_voxels = 4_000_000
    slab_depth = max(1, min(expected[2], max_slab_voxels // max(1, expected[0] * expected[1])))
    data = data.astype(np.float32, copy=False)

    for z0 in range(0, expected[2], slab_depth):
        z1 = min(expected[2], z0 + slab_depth)
        grid = np.indices((expected[0], expected[1], z1 - z0), dtype=np.float64)
        grid[2] += float(z0)
        flat = grid.reshape(3, -1)
        homogeneous = np.vstack([flat, np.ones((1, flat.shape[1]), dtype=np.float64)])
        source_coords = (source_to_mimics @ homogeneous)[:3]
        sampled = ndimage.map_coordinates(
            data,
            source_coords,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        output[:, :, z0:z1] = sampled.reshape((expected[0], expected[1], z1 - z0))
    return output


def _align_source_image_to_mimics_grid(data: np.ndarray, input_data: dict[str, Any]) -> np.ndarray:
    expected_shape = input_data.get("image_expected_shape") or input_data.get("interaction_shape")
    if not expected_shape:
        return data
    expected = [int(value) for value in expected_shape]
    source_affine = _parse_matrix(input_data.get("image_source_voxel_to_ras_matrix"))
    mimics_affine = _parse_matrix(input_data.get("image_mimics_voxel_to_ras_matrix"))
    if source_affine is not None and mimics_affine is not None:
        return _resample_image_to_mimics_grid(data, source_affine, mimics_affine, expected)
    loaded_shape = [int(value) for value in data.shape]
    if loaded_shape != expected:
        raise RuntimeError(
            f"Loaded source image shape does not match the open Mimics image and no affine resampling metadata was provided: {loaded_shape} != {expected}"
        )
    return data.astype(np.float32, copy=False)


def _apply_source_intensity_transform(data: np.ndarray, input_data: dict[str, Any]) -> np.ndarray:
    slope = input_data.get("image_source_to_mimics_gv_slope")
    intercept = input_data.get("image_source_to_mimics_gv_intercept")
    if slope is None or intercept is None:
        return data.astype(np.float32, copy=False)
    try:
        slope_f = float(slope)
        intercept_f = float(intercept)
    except (TypeError, ValueError):
        return data.astype(np.float32, copy=False)
    if slope_f == 1.0 and intercept_f == 0.0:
        return data.astype(np.float32, copy=False)
    return data.astype(np.float32, copy=False) * slope_f + intercept_f


def _apply_buffer_model_intensity_transform(
    data: np.ndarray, input_data: dict[str, Any]
) -> np.ndarray:
    slope = float(input_data.get("image_buffer_to_model_slope", 1.0) or 1.0)
    intercept = float(
        input_data.get("image_buffer_to_model_intercept", 0.0) or 0.0
    )
    if slope == 1.0 and intercept == 0.0:
        result = data
    else:
        result = data.astype(np.float32, copy=False) * slope + intercept
    zero_tolerance = input_data.get("image_buffer_model_zero_tolerance")
    if zero_tolerance is not None:
        tolerance = abs(float(zero_tolerance))
        if tolerance > 0.0:
            result = np.asarray(result, dtype=np.float32).copy()
            result[np.abs(result) <= tolerance] = 0.0
    return result


def _dicom_sort_key(record: tuple[Path, Any], normal: np.ndarray | None) -> tuple[float, float, str]:
    path, ds = record
    if normal is not None and hasattr(ds, "ImagePositionPatient"):
        try:
            ipp = np.array([float(value) for value in ds.ImagePositionPatient], dtype=float)
            return (float(np.dot(ipp, normal)), float(getattr(ds, "InstanceNumber", 0) or 0), str(path))
        except Exception:
            pass
    try:
        return (float(getattr(ds, "InstanceNumber", 0) or 0), 0.0, str(path))
    except Exception:
        return (0.0, 0.0, str(path))


def _dicom_normal(records: list[tuple[Path, Any]]) -> np.ndarray | None:
    for _path, ds in records:
        if not hasattr(ds, "ImageOrientationPatient"):
            continue
        try:
            iop = [float(value) for value in ds.ImageOrientationPatient]
            row = np.array(iop[:3], dtype=float)
            column = np.array(iop[3:], dtype=float)
            normal = np.cross(row, column)
            norm = float(np.linalg.norm(normal))
            if np.isfinite(norm) and norm > 0:
                return normal / norm
        except Exception:
            continue
    return None


def _dicom_attr_part(ds: Any, name: str) -> str:
    # None and absent must map to the same sentinel, but a present value of
    # 0 (e.g. SeriesNumber=0) must stay a real, distinct part.
    value = getattr(ds, name, None)
    return "" if value is None else str(value)


def _dicom_group_key(ds: Any) -> str:
    uid = getattr(ds, "SeriesInstanceUID", None)
    if uid:
        return str(uid)
    # Without a SeriesInstanceUID (de-identified or legacy export), the
    # standard remaining discriminator is (StudyInstanceUID, SeriesNumber).
    # Only when those are missing too do slices share the total-absence
    # bucket, where the duplicate-position check below guards against
    # silently interleaving two distinct series.
    return "__missing_uid__|{}|{}".format(
        _dicom_attr_part(ds, "StudyInstanceUID"),
        _dicom_attr_part(ds, "SeriesNumber"),
    )


def _reject_interleaved_missing_uid_groups(
    groups: dict[str, list[tuple[Path, Any]]],
) -> None:
    """Fail closed when a UID-less group provably mixes two series.

    Applies to groups not named by a real SeriesInstanceUID: the
    (StudyInstanceUID, SeriesNumber) pair is a heuristic, so two distinct
    series stripped of UIDs can still land in one bucket. Two records at
    the same (rounded) ImagePositionPatient are proof of interleaving —
    two series covering the same grid, or copy residue — and must not be
    silently stacked into a geometrically wrong volume. Records without a
    parseable IPP (e.g. multi-frame objects) are skipped.
    """
    for key, group in groups.items():
        if key and not key.startswith("__missing_uid__"):
            continue
        seen_positions: set[tuple[float, float, float]] = set()
        for _path, ds in group:
            if not hasattr(ds, "ImagePositionPatient"):
                continue
            try:
                ipp = tuple(
                    round(float(value), 2)
                    for value in ds.ImagePositionPatient
                )
            except Exception:
                continue
            if ipp in seen_positions:
                raise RuntimeError(
                    "The DICOM source folder contains multiple series that "
                    "cannot be told apart (no SeriesInstanceUID). "
                    "Use a source folder that contains only the intended series."
                )
            seen_positions.add(ipp)


def _select_dicom_records(
    records: list[tuple[Path, Any]],
    expected_shape: list[int] | None,
    allow_shape_mismatch: bool = False,
) -> list[tuple[Path, Any]]:
    if not records:
        raise RuntimeError("No readable DICOM image slices were found")

    groups: dict[str, list[tuple[Path, Any]]] = {}
    for record in records:
        groups.setdefault(_dicom_group_key(record[1]), []).append(record)

    _reject_interleaved_missing_uid_groups(groups)

    def group_shape(group: list[tuple[Path, Any]]) -> list[int] | None:
        if not group:
            return None
        first = group[0][1]
        return [int(first.Columns), int(first.Rows), int(len(group))]

    if expected_shape:
        expected = [int(value) for value in expected_shape]
        matches = [group for group in groups.values() if group_shape(group) == expected]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple DICOM series match the open Mimics image shape. "
                "Use a source folder that contains only the intended series."
            )
        all_shape = group_shape(records)
        if len(groups) == 1 and all_shape == expected:
            return records
        if allow_shape_mismatch and len(groups) == 1:
            return next(iter(groups.values()))
        if allow_shape_mismatch:
            raise RuntimeError(
                "No DICOM series shape matches the open Mimics image, and the source folder contains multiple series. "
                "Use a source folder that contains only the intended series."
            )

    return max(groups.values(), key=len)


def load_image_dicom_folder(
    path: str,
    expected_shape: list[int] | None = None,
    allow_shape_mismatch: bool = False,
) -> np.ndarray:
    """Load a DICOM folder as a float32 4D array with shape (1, Columns, Rows, Slices).

    Mimics voxel indexes are exposed as (x, y, z). For a DICOM slice, x maps to
    Columns and y maps to Rows, so each pixel_array (Rows, Columns) is
    transposed before stacking.
    """
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError(
            "DICOM source fast path requires pydicom in the external nnInteractive environment"
        ) from exc

    root = Path(path)
    if not root.is_dir():
        raise RuntimeError(f"DICOM source path is not a folder: {path}")

    records: list[tuple[Path, Any]] = []
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(child), stop_before_pixels=True, force=False)
        except Exception:
            continue
        if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
            records.append((child, ds))

    selected = _select_dicom_records(records, expected_shape, allow_shape_mismatch)
    normal = _dicom_normal(selected)
    selected = sorted(selected, key=lambda record: _dicom_sort_key(record, normal))

    slices: list[np.ndarray] = []
    for path_obj, _meta in selected:
        ds = pydicom.dcmread(str(path_obj), force=False)
        array = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            array = array * slope + intercept
        slices.append(array.T)

    if not slices:
        raise RuntimeError(f"No readable DICOM pixel data was found: {path}")

    data = np.stack(slices, axis=2).astype(np.float32, copy=False)
    if expected_shape and not allow_shape_mismatch:
        expected = [int(value) for value in expected_shape]
        loaded_shape = [int(value) for value in data.shape]
        if loaded_shape != expected:
            raise RuntimeError(
                f"DICOM source shape does not match the open Mimics image: {loaded_shape} != {expected}"
            )
    return data[None]


def load_image_source(input_data: dict[str, Any]) -> np.ndarray:
    path = str(input_data["image_path"])
    source_kind = str(input_data.get("image_source_kind") or "").lower()
    expected_shape = input_data.get("image_expected_shape") or input_data.get("interaction_shape")
    has_affine_resample_metadata = (
        _parse_matrix(input_data.get("image_source_voxel_to_ras_matrix")) is not None
        and _parse_matrix(input_data.get("image_mimics_voxel_to_ras_matrix")) is not None
    )
    try:
        if source_kind == "dicom_folder" or Path(path).is_dir():
            image = load_image_dicom_folder(
                path,
                expected_shape,
                allow_shape_mismatch=has_affine_resample_metadata,
            )
        elif source_kind == "medical_image":
            image = load_image_medical_file(path)
        else:
            image = load_image_nifti(path)
    except PermissionError as exc:
        raise RuntimeError(
            "Cannot read the source image file (access denied). The file may be "
            "locked by antivirus software, another process, or restricted "
            "permissions.\n\nPath: {0}\nError: {1}\n\n"
            "Workaround: set image_input_mode to \"mimics\" in nninteractive_config.json "
            "to use the Mimics image buffer instead.".format(path, exc)
        ) from exc
    aligned = _align_source_image_to_mimics_grid(image[0], input_data)
    aligned = _apply_source_intensity_transform(aligned, input_data)
    return aligned[None]


def load_image_raw(
    path: str,
    shape: list[int],
    dtype: str,
    *,
    buffer_mapping: dict[str, Any],
    coordinates: str = "mimics",
) -> np.ndarray:
    """Load a raw image buffer as a float32 4D platform-coordinate array."""
    raw_dtype = np.dtype(dtype)
    raw = Path(path).read_bytes()
    expected = int(np.prod(shape)) * raw_dtype.itemsize
    if len(raw) != expected:
        raise RuntimeError(
            f"Image buffer byte count mismatch: {len(raw)} != {expected}"
        )
    data = np.frombuffer(raw, dtype=raw_dtype).reshape(tuple(shape))
    if coordinates == "mimics":
        data = mimics_to_platform(data, buffer_mapping)
    elif coordinates != "platform":
        raise RuntimeError(f"Unsupported image buffer coordinates: {coordinates}")
    return data.astype(np.float32, copy=False)[None]


def load_interaction_u8(path: str, shape: list[int]) -> np.ndarray:
    """Load a .u8 buffer as a bool 3D array with the given MIMICS shape."""
    raw = Path(path).read_bytes()
    expected = int(shape[0]) * int(shape[1]) * int(shape[2])
    if len(raw) != expected:
        raise RuntimeError(
            f"Interaction buffer byte count mismatch: {len(raw)} != {expected}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).astype(bool)


def _nonzero_bbox(mask: np.ndarray) -> list[list[int]] | None:
    nonzero = np.argwhere(mask)
    if len(nonzero) == 0:
        return None
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    return [[int(mins[d]), int(maxs[d])] for d in range(3)]


def _iter_2d_interaction_crops(mask: np.ndarray) -> list[tuple[np.ndarray, list[list[int]]]]:
    """Split a sparse 3D edit mask into nnInteractive-compatible 2D crops."""
    if not np.any(mask):
        return []
    candidates: list[tuple[int, np.ndarray]] = []
    for axis in range(3):
        reduce_axes = tuple(i for i in range(3) if i != axis)
        indices = np.where(mask.any(axis=reduce_axes))[0]
        if len(indices):
            candidates.append((axis, indices))
    if not candidates:
        return []

    axis, indices = min(candidates, key=lambda item: len(item[1]))
    result: list[tuple[np.ndarray, list[list[int]]]] = []
    for index in indices:
        selector = [slice(None), slice(None), slice(None)]
        selector[axis] = slice(int(index), int(index) + 1)
        slice_mask = mask[tuple(selector)]
        local_bbox = _nonzero_bbox(slice_mask)
        if local_bbox is None:
            continue

        bbox = [list(item) for item in local_bbox]
        bbox[axis] = [int(index), int(index) + 1]
        crop = mask[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1],
        ].astype(np.uint8)
        result.append((crop, bbox))
    return result


def _polyline_to_mask(
    shape: list[int],
    points: list[list[int]],
    *,
    closed: bool = False,
) -> np.ndarray:
    """Rasterize voxel-index polyline points into a sparse 3D prompt mask."""
    result = np.zeros(tuple(shape), dtype=bool)
    if not points:
        return result
    if len(points) == 1:
        return _point_mask(shape, points[0])
    path_points = [list(_bounded_voxel_point(shape, point)) for point in points]
    if closed and path_points[-1] != path_points[0]:
        path_points.append(path_points[0])
    previous = np.asarray(path_points[0], dtype=float)
    for current_value in path_points[1:]:
        current = np.asarray(current_value, dtype=float)
        steps = max(int(np.max(np.abs(current - previous))), 1) + 1
        samples = np.rint(
            np.linspace(previous, current, num=steps, endpoint=True)
        ).astype(int)
        for sample in samples:
            if all(0 <= int(sample[axis]) < int(shape[axis]) for axis in range(3)):
                result[tuple(int(value) for value in sample)] = True
        previous = current
    return result


def _filled_region_boundary(mask: np.ndarray) -> np.ndarray:
    """Convert a filled Mimics Lasso region into per-slice closed contours."""
    if not np.any(mask):
        return mask.astype(bool)
    candidates: list[tuple[int, np.ndarray]] = []
    for axis in range(3):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slices = np.where(mask.any(axis=reduce_axes))[0]
        if len(slices):
            candidates.append((axis, slices))
    axis, _ = min(candidates, key=lambda item: len(item[1]))
    moved = np.moveaxis(mask.astype(bool), axis, 0)
    boundary = np.zeros_like(moved, dtype=bool)
    for index, plane in enumerate(moved):
        if not np.any(plane):
            continue
        interior = plane.copy()
        interior[0, :] = False
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False
        interior[1:-1, 1:-1] &= (
            plane[:-2, 1:-1]
            & plane[2:, 1:-1]
            & plane[1:-1, :-2]
            & plane[1:-1, 2:]
        )
        boundary[index] = plane & ~interior
    return np.moveaxis(boundary, 0, axis)


def _bounded_voxel_point(
    shape: list[int],
    point: list[int],
) -> tuple[int, int, int]:
    """Validate a point and clamp only a one-voxel edge rounding overshoot."""
    if len(point) != 3:
        raise RuntimeError(f"Point must have three indexes: {point}")
    if len(shape) != 3 or any(int(value) <= 0 for value in shape):
        raise RuntimeError(f"Invalid image shape for point interaction: {shape}")
    try:
        values = tuple(float(value) for value in point)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Point indexes must be numeric: {point}") from exc
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError(f"Point indexes must be finite: {point}")
    if any(
        values[axis] < -1.0 or values[axis] > float(shape[axis])
        for axis in range(3)
    ):
        raise RuntimeError(f"Point is outside image bounds: {point} vs {shape}")
    return tuple(
        min(max(int(round(values[axis])), 0), int(shape[axis]) - 1)
        for axis in range(3)
    )


def _point_mask(shape: list[int], point: list[int]) -> np.ndarray:
    result = np.zeros(tuple(shape), dtype=bool)
    indexes = _bounded_voxel_point(shape, point)
    result[indexes] = True
    return result


def _legacy_interactions(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the first prototype protocol to the ordered interaction form."""
    interaction_path = input_data.get("interaction_path")
    if not interaction_path:
        return []
    result = [
        {
            "interaction_type": input_data.get("interaction_type", "scribble"),
            "include_interaction": input_data.get("include_interaction", True),
            "mask_path": interaction_path,
            "mask_shape": input_data.get("interaction_shape"),
            "coordinates": "mimics",
        }
    ]
    if input_data.get("bg_interaction_path"):
        result.append(
            {
                "interaction_type": input_data.get("interaction_type", "scribble"),
                "include_interaction": False,
                "mask_path": input_data["bg_interaction_path"],
                "mask_shape": input_data.get("interaction_shape"),
                "coordinates": "mimics",
            }
        )
    return result


def _interaction_mask(
    interaction: dict[str, Any],
    *,
    mimics_shape: list[int],
    platform_shape: list[int],
    buffer_mapping: dict[str, Any],
) -> np.ndarray:
    coordinates = interaction.get("coordinates", "mimics")
    shape = mimics_shape if coordinates == "mimics" else platform_shape
    if interaction.get("point") is not None:
        mask = _point_mask(shape, interaction["point"])
    elif interaction.get("polyline_points") is not None:
        mask = _polyline_to_mask(
            shape,
            interaction["polyline_points"],
            closed=bool(interaction.get("polyline_closed", False)),
        )
    elif interaction.get("bbox") is not None:
        mask = np.zeros(tuple(shape), dtype=bool)
        bbox = interaction["bbox"]
        mask[
            int(bbox[0][0]):int(bbox[0][1]),
            int(bbox[1][0]):int(bbox[1][1]),
            int(bbox[2][0]):int(bbox[2][1]),
        ] = True
    elif interaction.get("mask_path"):
        mask_shape = interaction.get("mask_shape") or shape
        loaded = load_interaction_u8(interaction["mask_path"], mask_shape)
        bbox = interaction.get("interaction_bbox")
        if bbox:
            mask = np.zeros(tuple(shape), dtype=bool)
            expected_shape = tuple(
                int(bbox[axis][1]) - int(bbox[axis][0])
                for axis in range(3)
            )
            if tuple(loaded.shape) != expected_shape:
                raise RuntimeError(
                    f"Interaction crop shape mismatch: {loaded.shape} != {expected_shape}"
                )
            mask[
                int(bbox[0][0]):int(bbox[0][1]),
                int(bbox[1][0]):int(bbox[1][1]),
                int(bbox[2][0]):int(bbox[2][1]),
            ] = loaded
        else:
            mask = loaded
    else:
        raise RuntimeError(
            f"Interaction has no point, polyline, bbox, or mask: {interaction}"
        )
    if coordinates == "mimics":
        return mimics_to_platform(mask, buffer_mapping)
    if coordinates == "platform":
        return mask
    raise RuntimeError(f"Unsupported interaction coordinates: {coordinates}")


def _apply_interaction(
    session: Any,
    interaction_mask: np.ndarray,
    interaction_type: str,
    include_interaction: bool,
) -> int:
    if interaction_type == "point":
        nonzero = np.argwhere(interaction_mask)
        if len(nonzero):
            session.add_point_interaction(
                tuple(int(value) for value in nonzero[0]),
                include_interaction=include_interaction,
                run_prediction=True,
            )
            return 1
        return 0
    if interaction_type == "box":
        bbox = _nonzero_bbox(interaction_mask)
        if bbox is not None:
            session.add_bbox_interaction(
                bbox,
                include_interaction=include_interaction,
                run_prediction=True,
            )
            return 1
        return 0
    if interaction_type == "lasso":
        interaction_mask = _filled_region_boundary(interaction_mask)
        add_method = session.add_lasso_interaction
    elif interaction_type == "scribble":
        add_method = session.add_scribble_interaction
    else:
        raise RuntimeError(f"Unsupported interaction_type: {interaction_type}")
    crops = _iter_2d_interaction_crops(interaction_mask)
    for crop, bbox in crops:
        # nnInteractive performs patch-centred 3D inference. Every newly
        # collected prompt must therefore receive its own prediction so it
        # becomes the centre of a patch and the resulting prev_seg is
        # available to the following prompt. Mimics receives only the final
        # buffer after this ordered sequence has completed.
        add_method(
            crop,
            include_interaction=include_interaction,
            run_prediction=True,
            interaction_bbox=bbox,
        )
    return len(crops)


def _interaction_fingerprint(interaction: dict[str, Any]) -> str:
    """Stable fingerprint used to detect append-only prompt updates."""
    try:
        return json.dumps(interaction, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(interaction)


def _apply_point_set(
    session: Any,
    interaction: dict[str, Any],
    *,
    mimics_shape: list[int],
    platform_shape: list[int],
    buffer_mapping: dict[str, Any],
) -> int:
    points = []
    coordinates = interaction.get("coordinates", "mimics")
    for item in interaction.get("points", []):
        point_mask = _interaction_mask(
            {
                "point": item["point"],
                "coordinates": item.get("coordinates", coordinates),
            },
            mimics_shape=mimics_shape,
            platform_shape=platform_shape,
            buffer_mapping=buffer_mapping,
        )
        nonzero = np.argwhere(point_mask)
        if len(nonzero):
            points.append(
                (
                    tuple(int(value) for value in nonzero[0]),
                    bool(item.get("include_interaction", True)),
                )
            )

    if not points:
        return 0

    prediction_policy = str(
        interaction.get("prediction_policy") or "sequential"
    ).strip().lower()
    if prediction_policy not in {"sequential", "initial_empty_batch"}:
        raise RuntimeError(
            f"Unsupported point-set prediction policy: {prediction_policy}"
        )

    # An empty-Mask first prompt may accumulate several points and predict
    # once. Corrections remain sequential so every new prompt sees the prior
    # prediction through nnInteractive's previous-segmentation channel.
    for index, (point, include) in enumerate(points):
        session.add_point_interaction(
            point,
            include_interaction=include,
            run_prediction=(
                prediction_policy == "sequential" or index == len(points) - 1
            ),
        )
    return len(points)


def _apply_scribble_set(
    session: Any,
    interaction: dict[str, Any],
    *,
    mimics_shape: list[int],
    platform_shape: list[int],
    buffer_mapping: dict[str, Any],
) -> int:
    scribbles = interaction.get("scribbles") or []
    accepted = 0
    prepared: list[tuple[np.ndarray, bool]] = []
    coordinates = interaction.get("coordinates", "mimics")
    for item in scribbles:
        item_value = dict(item)
        item_value.setdefault("coordinates", item.get("coordinates", coordinates))
        mask = _interaction_mask(
            item_value,
            mimics_shape=mimics_shape,
            platform_shape=platform_shape,
            buffer_mapping=buffer_mapping,
        )
        if np.any(mask):
            prepared.append((mask, bool(item.get("include_interaction", True))))

    for mask, include in prepared:
        crops = _iter_2d_interaction_crops(mask)
        for crop, bbox in crops:
            session.add_scribble_interaction(
                crop,
                include_interaction=include,
                run_prediction=True,
                interaction_bbox=bbox,
            )
            accepted += 1
    return accepted


def _is_capacity_error(exc: BaseException) -> bool:
    text = str(exc).lower().replace(" ", "")
    return "serverisatcapacity" in text or "atcapacity" in text


# ---------------------------------------------------------------------------
#  Main bridge entry point
# ---------------------------------------------------------------------------

class _BridgeSessionContext:
    """Keep one remote session and one preprocessed image for a Mimics tool run."""

    def __init__(self, input_data: dict[str, Any]):
        context_started = time.time()
        self.input_data = input_data
        self.model_dir = str(input_data["model_dir"])
        self.runtime_work_dir = input_data.get("runtime_work_dir")
        self.requested_device = str(input_data.get("device", "auto"))
        self.log_path = _bridge_log_path(self.model_dir, input_data.get("log_dir"))
        self.keep_server_warm_after_session = bool(
            input_data.get("keep_server_warm_after_session", True)
        )
        self.buffer_mapping = input_data.get("buffer_mapping") or {
            "platform_to_mimics_axes": [0, 1, 2],
            "platform_to_mimics_flips": [False, False, False],
        }
        self.model_input_space = str(
            input_data.get("model_input_space") or "mimics"
        ).strip().lower()
        if self.model_input_space == "canonical_ras":
            self.buffer_mapping = canonical_ras_buffer_mapping(
                input_data.get("image_mimics_voxel_to_ras_matrix")
            )
        self.device, self.device_warning = _resolve_device(
            self.requested_device,
            bool(input_data.get("allow_cpu_fallback", True)),
        )
        _append_bridge_log(
            self.log_path,
            "session_initializing",
            python=sys.executable,
            requested_device=self.requested_device,
            resolved_device=self.device,
            device_warning=self.device_warning,
            model_dir=self.model_dir,
            image_source=input_data.get("image_source"),
            image_source_kind=input_data.get("image_source_kind"),
            image_source_index_space=input_data.get("image_source_index_space"),
            image_source_modality=input_data.get("image_source_modality"),
            image_source_world_coordinate_system=input_data.get("image_source_world_coordinate_system"),
            image_mimics_world_coordinate_system=input_data.get("image_mimics_world_coordinate_system"),
            image_source_to_mimics_world_matrix=input_data.get("image_source_to_mimics_world_matrix"),
            image_source_voxel_to_ras_matrix=input_data.get("image_source_voxel_to_ras_matrix"),
            image_mimics_voxel_to_ras_matrix=input_data.get("image_mimics_voxel_to_ras_matrix"),
            image_mimics_to_source_index_matrix=input_data.get("image_mimics_to_source_index_matrix"),
            image_expected_shape=input_data.get("image_expected_shape"),
            image_source_intensity_space=input_data.get("image_source_intensity_space"),
            image_source_to_mimics_gv_slope=input_data.get("image_source_to_mimics_gv_slope"),
            image_source_to_mimics_gv_intercept=input_data.get("image_source_to_mimics_gv_intercept"),
            model_input_space=self.model_input_space,
            model_input_intensity_space=input_data.get(
                "model_input_intensity_space", ""
            ),
            image_input_provenance=input_data.get("image_input_provenance", ""),
            image_intensity_compatibility=input_data.get(
                "image_intensity_compatibility", ""
            ),
            image_source_intensity_encoding=input_data.get(
                "image_source_intensity_encoding", ""
            ),
            image_source_intensity_recovery_basis=input_data.get(
                "image_source_intensity_recovery_basis", ""
            ),
            effective_buffer_mapping=self.buffer_mapping,
        )

        if input_data.get("image_buffer_shape"):
            self.mimics_shape = [int(value) for value in input_data["image_buffer_shape"]]
        elif input_data.get("interaction_shape"):
            self.mimics_shape = [int(value) for value in input_data["interaction_shape"]]
        elif input_data.get("image_expected_shape"):
            self.mimics_shape = [int(value) for value in input_data["image_expected_shape"]]
        else:
            self.mimics_shape = []

        if input_data.get("image_buffer_path"):
            self.image_np = load_image_raw(
                input_data["image_buffer_path"],
                input_data["image_buffer_shape"],
                input_data.get("image_buffer_dtype", "int16"),
                buffer_mapping=self.buffer_mapping,
                coordinates=input_data.get("image_buffer_coordinates", "mimics"),
            )
            self.image_np = _apply_buffer_model_intensity_transform(
                self.image_np, input_data
            )
        elif input_data.get("image_path"):
            image_mimics = load_image_source(input_data)
            self.image_np = mimics_to_platform(image_mimics[0], self.buffer_mapping)[None]
        else:
            raise RuntimeError("Missing image_path or image_buffer_path")
        self.image_load_seconds = round(time.time() - context_started, 2)

        self.platform_shape = [int(value) for value in self.image_np.shape[1:]]
        if not self.mimics_shape:
            self.mimics_shape = platform_to_mimics(self.image_np[0], self.buffer_mapping).shape
            self.mimics_shape = [int(value) for value in self.mimics_shape]
        expected_platform_shape = mimics_shape_to_platform_shape(self.mimics_shape, self.buffer_mapping)
        if expected_platform_shape and self.platform_shape != expected_platform_shape:
            raise RuntimeError(
                "Loaded source image shape does not match the open Mimics image: "
                f"{self.platform_shape} != {expected_platform_shape}"
            )

        self.server_url = str(input_data.get("server_url") or SERVER_URL)
        self.auto_start_server = bool(input_data.get("auto_start_server", True))
        self.server_api_key = os.environ.get("NN_INTERACTIVE_API_KEY")
        # Surface the request's free-VRAM floor for _check_free_gpu_memory
        # (env is the channel _start_server reads; the request payload is
        # authoritative when present).
        try:
            _requested_min_free_gb = float(
                input_data.get("minimum_free_gpu_memory_gb", 0)
            )
        except (TypeError, ValueError):
            _requested_min_free_gb = 0.0
        if _requested_min_free_gb > 0:
            os.environ["NNINTERACTIVE_MINIMUM_FREE_GPU_MEMORY_GB"] = str(
                _requested_min_free_gb
            )
        self.owned_state_path: Path | None = None
        self.owned_token: str | None = None
        server_started = time.time()
        if self.auto_start_server and self.server_url == SERVER_URL:
            self.first_call, self.server_url, self.server_api_key = _ensure_server(
                self.model_dir,
                self.device,
                float(input_data.get("server_idle_timeout_seconds", SERVER_IDLE_TIMEOUT)),
                float(
                    input_data.get(
                        "server_startup_timeout_seconds",
                        SERVER_STARTUP_TIMEOUT,
                    )
                ),
                self.server_url,
                input_data.get("fold", "auto"),
                float(input_data.get("gpu_lock_timeout_seconds", 30)),
                self.runtime_work_dir,
                str(input_data.get("checkpoint_sha256") or ""),
            )
            self.owned_state_path = _server_state_path(self.model_dir, self.runtime_work_dir)
            self.owned_token = self.server_api_key
        else:
            self.first_call = False
        self.server_ready_seconds = round(time.time() - server_started, 2)

        if self.owned_state_path is not None and self.owned_token:
            _touch_server_activity(self.owned_state_path, self.owned_token)
            state = _load_server_state(self.owned_state_path)
            if state and state.get("ownership_token") == self.owned_token:
                state["client_pid"] = os.getpid()
                state["active_operation_timeout_seconds"] = max(
                    60.0,
                    float(input_data.get("prediction_timeout_seconds", 1800)) + 60.0,
                )
                control_dir = str(input_data.get("async_worker_control_dir") or "").strip()
                if control_dir:
                    state["client_control_dir"] = control_dir
                _write_server_state(self.owned_state_path, state)
        self.session = None

        try:
            self._connect_and_upload()
        except Exception as exc:
            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = None
            can_restart_owned_server = (
                self.auto_start_server
                and self.server_url.startswith("http://127.0.0.1:")
                and self.owned_state_path is not None
                and _is_capacity_error(exc)
            )
            if not can_restart_owned_server:
                raise
            state = _load_server_state(self.owned_state_path)
            if state:
                if not _terminate_owned_server(state):
                    raise RuntimeError(
                        "The existing nnInteractive server could not be stopped "
                        "for the capacity-recovery restart."
                    )
                if not _remove_server_state(
                    self.owned_state_path,
                    str(state.get("ownership_token") or ""),
                ):
                    raise RuntimeError(
                        "The stopped nnInteractive server state could not be "
                        "retired safely for the capacity-recovery restart."
                    )
            _append_bridge_log(
                self.log_path,
                "server_capacity_restart",
                server_url=self.server_url,
                error=str(exc),
            )
            self.first_call, self.server_url, self.server_api_key = _ensure_server(
                self.model_dir,
                self.device,
                float(input_data.get("server_idle_timeout_seconds", SERVER_IDLE_TIMEOUT)),
                float(
                    input_data.get(
                        "server_startup_timeout_seconds",
                        SERVER_STARTUP_TIMEOUT,
                    )
                ),
                SERVER_URL,
                input_data.get("fold", "auto"),
                float(input_data.get("gpu_lock_timeout_seconds", 30)),
                self.runtime_work_dir,
                str(input_data.get("checkpoint_sha256") or ""),
            )
            self.owned_state_path = _server_state_path(self.model_dir, self.runtime_work_dir)
            self.owned_token = self.server_api_key
            self._connect_and_upload()
        self.initial_platform = self._load_initial_platform(
            input_data.get("initial_seg_path"),
            input_data.get("initial_seg_shape"),
        )
        self.incremental_interaction_replay = bool(
            input_data.get("incremental_interaction_replay", True)
        )
        self._applied_initial_key: str | None = None
        self._applied_interaction_fingerprints: list[str] = []
        _append_bridge_log(
            self.log_path,
            "session_ready",
            device=self.device,
            server_url=self.server_url,
            first_call=self.first_call,
            image_shape=self.platform_shape,
            image_load_seconds=self.image_load_seconds,
            server_ready_seconds=self.server_ready_seconds,
            set_image_seconds=self.set_image_seconds,
            set_target_seconds=self.set_target_seconds,
            image_source_intensity_space=input_data.get("image_source_intensity_space"),
            image_input_provenance=input_data.get("image_input_provenance", ""),
            image_intensity_compatibility=input_data.get(
                "image_intensity_compatibility", ""
            ),
            image_source_intensity_encoding=input_data.get(
                "image_source_intensity_encoding", ""
            ),
            image_source_intensity_recovery_basis=input_data.get(
                "image_source_intensity_recovery_basis", ""
            ),
        )

    def _connect_and_upload(self) -> None:
        upload_started = time.time()
        self.session = _connect_remote(
            self.server_url,
            self.server_api_key,
            prediction_timeout_seconds=float(
                self.input_data.get("prediction_timeout_seconds", 1800)
            ),
            set_image_timeout_seconds=float(
                self.input_data.get("set_image_timeout_seconds", 1800)
            ),
        )
        self.session.set_image(self.image_np)
        self.set_image_seconds = round(time.time() - upload_started, 2)
        target_started = time.time()
        self.target = np.zeros(self.image_np.shape[1:], dtype=np.uint8)
        self.session.set_target_buffer(self.target)
        self.set_target_seconds = round(time.time() - target_started, 2)

    @staticmethod
    def _server_failure_is_recoverable(exc: BaseException) -> bool:
        text = str(exc).lower()
        if any(
            value in text
            for value in (
                "out of memory",
                "cuda error",
                "cudnn error",
                "device-side assert",
                "model file not found",
                "checkpoint",
                "corrupt",
                "outside image bounds",
                "out of bounds",
                "indexerror",
                "index error",
            )
        ):
            return False
        return any(
            value in text
            for value in (
                "10061",
                "connection refused",
                "failed to establish a new connection",
                "remote end closed connection",
                "server disconnected",
                "server is not running",
                "status code 500",
                "http 500",
                "internal server error",
            )
        )

    def _restart_owned_server(self, exc: BaseException) -> None:
        if not (
            self.auto_start_server
            and self.server_url.startswith("http://127.0.0.1:")
            and self.owned_state_path is not None
        ):
            raise RuntimeError(
                "The nnInteractive server connection failed and this client "
                "does not own a local server that can be restarted: {}".format(exc)
            ) from exc
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
        state = _load_server_state(self.owned_state_path)
        if state and state.get("ownership_token") == self.owned_token:
            if _process_matches_server(state) and not _terminate_owned_server(state):
                raise RuntimeError(
                    "The failed nnInteractive server could not be stopped safely."
                ) from exc
            if not _remove_server_state(self.owned_state_path, self.owned_token):
                raise RuntimeError(
                    "The failed nnInteractive server state could not be retired safely."
                ) from exc
        _append_bridge_log(
            self.log_path,
            "server_recovery_restart",
            error=str(exc),
            server_url=self.server_url,
        )
        self.first_call, self.server_url, self.server_api_key = _ensure_server(
            self.model_dir,
            self.device,
            float(
                self.input_data.get(
                    "server_idle_timeout_seconds", SERVER_IDLE_TIMEOUT
                )
            ),
            float(
                self.input_data.get(
                    "server_startup_timeout_seconds", SERVER_STARTUP_TIMEOUT
                )
            ),
            SERVER_URL,
            self.input_data.get("fold", "auto"),
            float(self.input_data.get("gpu_lock_timeout_seconds", 30)),
            self.runtime_work_dir,
            str(self.input_data.get("checkpoint_sha256") or ""),
        )
        self.owned_state_path = _server_state_path(
            self.model_dir, self.runtime_work_dir
        )
        self.owned_token = self.server_api_key
        self._connect_and_upload()
        state = _load_server_state(self.owned_state_path)
        if state and state.get("ownership_token") == self.owned_token:
            state["client_pid"] = os.getpid()
            state["active_operation_timeout_seconds"] = max(
                60.0,
                float(
                    self.input_data.get("prediction_timeout_seconds", 1800)
                )
                + 60.0,
            )
            control_dir = str(
                self.input_data.get("async_worker_control_dir") or ""
            ).strip()
            if control_dir:
                state["client_control_dir"] = control_dir
            _write_server_state(self.owned_state_path, state)
            _set_server_operation_active(
                self.owned_state_path, self.owned_token, True
            )
        self._applied_initial_key = None
        self._applied_interaction_fingerprints = []

    def _load_initial_platform(
        self,
        initial_seg_path: str | None,
        initial_seg_shape: list[int] | None,
    ) -> np.ndarray | None:
        if not initial_seg_path:
            return None
        initial_shape = initial_seg_shape or self.mimics_shape
        initial_mimics = load_interaction_u8(initial_seg_path, initial_shape)
        if not np.any(initial_mimics):
            return None
        return mimics_to_platform(initial_mimics, self.buffer_mapping).astype(np.uint8)

    def _apply_initial_segmentation(self, initial_platform: np.ndarray | None) -> None:
        if initial_platform is None:
            return
        try:
            self.session.add_initial_seg_interaction(
                initial_platform,
                run_prediction=False,
            )
        except TypeError:
            print(
                "nninteractive_bridge: add_initial_seg_interaction has no run_prediction "
                "kwarg; using legacy API (may trigger an extra inference pass)",
                file=sys.stderr,
            )
            self.session.add_initial_seg_interaction(self.initial_platform)

    def predict(
        self,
        interactions: list[dict[str, Any]],
        output_path: str,
        *,
        initial_seg_path: str | None = None,
        initial_seg_shape: list[int] | None = None,
        use_context_initial_seg: bool = True,
    ) -> dict[str, Any]:
        owned_state_path = getattr(self, "owned_state_path", None)
        owned_token = getattr(self, "owned_token", None)
        if owned_state_path is not None and owned_token:
            _set_server_operation_active(owned_state_path, owned_token, True)
        try:
            try:
                return self._predict_impl(
                    interactions,
                    output_path,
                    initial_seg_path=initial_seg_path,
                    initial_seg_shape=initial_seg_shape,
                    use_context_initial_seg=use_context_initial_seg,
                )
            except Exception as exc:
                if not self._server_failure_is_recoverable(exc):
                    raise
                self._restart_owned_server(exc)
                return self._predict_impl(
                    interactions,
                    output_path,
                    initial_seg_path=initial_seg_path,
                    initial_seg_shape=initial_seg_shape,
                    use_context_initial_seg=use_context_initial_seg,
                )
        finally:
            current_path = getattr(self, "owned_state_path", None) or owned_state_path
            current_token = getattr(self, "owned_token", None) or owned_token
            if current_path is not None and current_token:
                _set_server_operation_active(current_path, current_token, False)

    def _predict_impl(
        self,
        interactions: list[dict[str, Any]],
        output_path: str,
        *,
        initial_seg_path: str | None = None,
        initial_seg_shape: list[int] | None = None,
        use_context_initial_seg: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        if not interactions:
            return {
                "status": "skipped",
                "output_path": output_path,
                "elapsed_seconds": 0.0,
                "reason": "no_interactions",
            }
        if initial_seg_path is not None or not use_context_initial_seg:
            initial_platform = self._load_initial_platform(initial_seg_path, initial_seg_shape)
        else:
            initial_platform = self.initial_platform

        interaction_fingerprints = [
            _interaction_fingerprint(interaction) for interaction in interactions
        ]
        initial_key = json.dumps(
            {
                "initial_seg_path": initial_seg_path if not use_context_initial_seg else "__context__",
                "initial_seg_shape": initial_seg_shape or self.mimics_shape,
                "use_context_initial_seg": bool(use_context_initial_seg),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        def _apply_one_interaction(interaction: dict[str, Any]) -> int:
            interaction_type = str(interaction.get("interaction_type", "scribble"))
            if interaction_type == "point_set":
                return _apply_point_set(
                    self.session,
                    interaction,
                    mimics_shape=self.mimics_shape,
                    platform_shape=self.platform_shape,
                    buffer_mapping=self.buffer_mapping,
                )
            if interaction_type == "scribble_set":
                return _apply_scribble_set(
                    self.session,
                    interaction,
                    mimics_shape=self.mimics_shape,
                    platform_shape=self.platform_shape,
                    buffer_mapping=self.buffer_mapping,
                )
            interaction_platform = _interaction_mask(
                interaction,
                mimics_shape=self.mimics_shape,
                platform_shape=self.platform_shape,
                buffer_mapping=self.buffer_mapping,
            )
            if not np.any(interaction_platform):
                return 0
            return _apply_interaction(
                self.session,
                interaction_platform,
                interaction_type,
                bool(interaction.get("include_interaction", True)),
            )

        def _incremental_start_index(force_reset: bool) -> int:
            if force_reset or not self.incremental_interaction_replay:
                return 0
            previous = self._applied_interaction_fingerprints
            if not previous:
                return 0
            if self._applied_initial_key != initial_key:
                return 0
            if len(interaction_fingerprints) <= len(previous):
                return 0
            if interaction_fingerprints[:len(previous)] != previous:
                return 0
            return len(previous)

        def _apply_interactions(force_reset: bool = False) -> tuple[int, int, float]:
            apply_started = time.time()
            start_index = _incremental_start_index(force_reset)
            if start_index <= 0:
                self.session.reset_interactions()
                self._apply_initial_segmentation(initial_platform)
                start_index = 0
            applied_count = 0
            for interaction in interactions[start_index:]:
                applied_count += _apply_one_interaction(interaction)
            return applied_count, start_index, round(time.time() - apply_started, 2)

        applied, replay_start_index, prompt_apply_seconds = _apply_interactions()
        if applied == 0:
            return {
                "status": "skipped",
                "output_path": output_path,
                "elapsed_seconds": round(time.time() - started, 2),
                "reason": "interactions_empty",
            }

        result_platform = np.asarray(self.target, dtype=np.uint8)
        warmup_retry = False
        if self.first_call and not np.any(result_platform):
            # On a fresh server, the very first interaction can occasionally return
            # an empty mask despite valid prompts; retry once in the same session.
            print(
                "nninteractive_bridge: first-call empty prediction detected; retrying once.",
                file=sys.stderr,
            )
            applied, replay_start_index, prompt_apply_seconds = _apply_interactions(force_reset=True)
            result_platform = np.asarray(self.target, dtype=np.uint8)
            warmup_retry = True

        if not np.any(result_platform):
            print(
                "nninteractive_bridge: prediction output is empty (foreground_voxels=0) "
                "after applying {0} interaction(s).".format(applied),
                file=sys.stderr,
            )
        result_mimics = platform_to_mimics(result_platform, self.buffer_mapping)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(result_mimics.tobytes(order="C"))

        self._applied_initial_key = initial_key
        self._applied_interaction_fingerprints = interaction_fingerprints

        result = {
            "status": "refined",
            "output_path": output_path,
            "elapsed_seconds": round(time.time() - started, 2),
            "mode": "remote",
            "first_call": self.first_call,
            "requested_device": self.requested_device,
            "device": self.device,
            "device_warning": self.device_warning,
            "server_url": self.server_url,
            "bridge_log": str(self.log_path),
            "server_log": str(_server_log_path(self.model_dir, getattr(self, "runtime_work_dir", None))),
            "interaction_count": len(interactions),
            "prediction_steps": applied,
            "model_license": getattr(self.session, "license", None),
            "result_shape": list(result_mimics.shape),
            "foreground_voxels": int(np.count_nonzero(result_mimics)),
            "warmup_retry": warmup_retry,
            "incremental_replay": replay_start_index > 0,
            "replayed_interactions": len(interactions) - replay_start_index,
            "skipped_replay_interactions": replay_start_index,
            "prompt_apply_seconds": prompt_apply_seconds,
            "image_load_seconds": self.image_load_seconds,
            "server_ready_seconds": self.server_ready_seconds,
            "set_image_seconds": self.set_image_seconds,
            "set_target_seconds": self.set_target_seconds,
        }
        _append_bridge_log(
            self.log_path,
            "prediction_completed",
            elapsed_seconds=result["elapsed_seconds"],
            interaction_count=len(interactions),
            prediction_steps=applied,
            foreground_voxels=result["foreground_voxels"],
            warmup_retry=warmup_retry,
            incremental_replay=result["incremental_replay"],
            replayed_interactions=result["replayed_interactions"],
            skipped_replay_interactions=result["skipped_replay_interactions"],
            prompt_apply_seconds=prompt_apply_seconds,
        )
        return result

    def close(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self.owned_state_path is not None and self.owned_token:
                if self.keep_server_warm_after_session:
                    _touch_server_activity(self.owned_state_path, self.owned_token)
                else:
                    # A competing GPU workflow must not wait for the normal
                    # idle watchdog interval after this client has finished.
                    # Stop only the token-owned server and release its lock;
                    # fall back to watchdog expiry if termination cannot be
                    # confirmed safely.
                    state = _load_server_state(self.owned_state_path)
                    retired = False
                    if state and state.get("ownership_token") == self.owned_token:
                        if _terminate_owned_server(state):
                            retired = _remove_server_state(
                                self.owned_state_path, self.owned_token
                            )
                    if not retired:
                        _expire_server_activity(
                            self.owned_state_path, self.owned_token
                        )
            _append_bridge_log(
                self.log_path,
                "session_closed",
                keep_server_warm_after_session=self.keep_server_warm_after_session,
            )


def _error_result(
    exc: Exception,
    *,
    stage: str,
    started: float,
    model_dir: str,
    log_path: Path,
    runtime_work_dir: str | None = None,
) -> dict[str, Any]:
    trace = traceback.format_exc()
    try:
        _append_bridge_log(
            log_path,
            "bridge_failed",
            stage=stage,
            error=str(exc),
            traceback=trace,
            python=sys.executable,
        )
    except Exception:
        pass
    return {
        "status": "error",
        "error": str(exc),
        "stage": stage,
        "traceback": trace,
        "bridge_log": str(log_path),
        "server_log": str(_server_log_path(model_dir, runtime_work_dir)),
        "python": sys.executable,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def run_bridge(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute one refinement while using the managed server lifecycle."""
    started = time.time()
    model_dir = str(input_data.get("model_dir") or "")
    output_path = str(input_data.get("output_path") or "")
    if not model_dir or not output_path:
        return {
            "status": "error",
            "error": "Missing required keys: model_dir and output_path",
        }
    log_path = _bridge_log_path(model_dir, input_data.get("log_dir"))
    interactions = input_data.get("interactions") or _legacy_interactions(input_data)
    if not interactions:
        return {
            "status": "skipped",
            "output_path": output_path,
            "elapsed_seconds": 0.0,
            "reason": "no_interactions",
        }
    context = None
    try:
        context = _BridgeSessionContext(input_data)
        result = context.predict(interactions, output_path)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        return result
    except Exception as exc:
        return _error_result(
            exc,
            stage="initialize_or_predict",
            started=started,
            model_dir=model_dir,
            log_path=log_path,
            runtime_work_dir=input_data.get("runtime_work_dir"),
        )
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass


# Error messages whose substrings indicate a non-recoverable failure.
_FATAL_ERROR_SUBSTRINGS = [
    "out of memory",
    "cuda error",
    "cudnn error",
    "runtimeerror: cuda",
    "no kernel image is available",
    "device-side assert triggered",
    "model file not found",
    "checkpoint",
    "corrupt",
    "connection refused",
    "server is not running",
    "server died",
    "status code 500",
    "http 500",
    "internal server error",
]


def _is_recoverable_worker_error(exc: Exception) -> bool:
    """Return True if *exc* is a transient error that might succeed on retry."""
    message = str(exc).lower()
    for fatal in _FATAL_ERROR_SUBSTRINGS:
        if fatal in message:
            return False
    return True


def _worker_main() -> int:
    """JSON-lines worker that reuses one preprocessed image across prompts."""
    context = None
    model_dir = ""
    # A relative default would write early errors (before the first
    # initialize request sets the real path) into whatever CWD the worker
    # happened to start in. Park them in the temp dir until then.
    log_path = Path(tempfile.gettempdir()) / "nninteractive_bridge.jsonl"
    for raw in sys.stdin:
        started = time.time()
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw)
            action = request.get("action")
            if action == "initialize":
                if context is not None:
                    context.close()
                model_dir = str(request.get("model_dir") or "")
                if not model_dir:
                    raise RuntimeError("Worker initialize request is missing model_dir")
                log_path = _bridge_log_path(model_dir, request.get("log_dir"))
                context = _BridgeSessionContext(request)
                result = {
                    "status": "ready",
                    "device": context.device,
                    "device_warning": context.device_warning,
                    "server_url": context.server_url,
                    "first_call": context.first_call,
                    "mode": "remote",
                    "bridge_log": str(context.log_path),
                    "server_log": str(_server_log_path(model_dir, request.get("runtime_work_dir"))),
                    "image_load_seconds": context.image_load_seconds,
                    "server_ready_seconds": context.server_ready_seconds,
                    "set_image_seconds": context.set_image_seconds,
                    "set_target_seconds": context.set_target_seconds,
                }
            elif action == "predict":
                if context is None:
                    raise RuntimeError("Worker has not been initialized")
                result = context.predict(
                    request.get("interactions") or [],
                    str(request["output_path"]),
                    initial_seg_path=request.get("initial_seg_path"),
                    initial_seg_shape=request.get("initial_seg_shape"),
                    use_context_initial_seg="initial_seg_path" not in request,
                )
            elif action == "close":
                if context is not None:
                    context.close()
                    context = None
                result = {"status": "closed"}
                print(json.dumps(result, separators=(",", ":")), flush=True)
                return 0
            else:
                raise RuntimeError(f"Unsupported worker action: {action!r}")
        except Exception as exc:
            result = _error_result(
                exc,
                stage=f"worker_{request.get('action', 'request')}",
                started=started,
                model_dir=model_dir,
                log_path=log_path,
                runtime_work_dir=request.get("runtime_work_dir"),
            )
            print(json.dumps(result, separators=(",", ":")), flush=True)
            # Fatal errors (OOM, model corruption, server death) cannot be
            # recovered by retrying on the same stdin loop.  Exit so the
            # caller (Mimics side) can detect the dead worker and restart
            # cleanly instead of looping on every subsequent request.
            if context is None or not _is_recoverable_worker_error(exc):
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                    context = None
                return 2
        else:
            print(json.dumps(result, separators=(",", ":")), flush=True)
    if context is not None:
        context.close()
    return 0


def _async_worker_status(
    job_dir: Path,
    status: str,
    **details: Any,
) -> None:
    payload = {
        "schema_version": "nninteractive_async_status.v1",
        "status": status,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
    }
    payload.update(details)
    _write_json_atomic(job_dir / "worker_status.json", payload)


def _async_worker_main(job_dir_value: str) -> int:
    """Persistent file-queue worker used by the non-blocking Mimics mode."""
    job_dir = Path(job_dir_value).resolve()
    request_path = job_dir / "initialize.json"
    context = None
    last_sequence = 0
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        idle_timeout = float(request.get("async_worker_idle_timeout_seconds", 600))
        poll_seconds = max(0.1, float(request.get("async_poll_seconds", 0.5)))
        _async_worker_status(job_dir, "initializing", stage="load_model_and_image")
        context = _BridgeSessionContext(request)
        _async_worker_status(
            job_dir,
            "ready",
            stage="waiting_for_prompt",
            device=context.device,
            device_warning=context.device_warning,
            server_url=context.server_url,
            first_call=context.first_call,
            image_load_seconds=context.image_load_seconds,
            server_ready_seconds=context.server_ready_seconds,
            set_image_seconds=context.set_image_seconds,
            set_target_seconds=context.set_target_seconds,
            bridge_log=str(context.log_path),
            server_log=str(_server_log_path(context.model_dir, context.runtime_work_dir)),
        )
        last_activity = time.time()
        parent_pid = int(request.get("parent_pid", 0) or 0)

        while True:
            close_path = job_dir / "close.json"
            if close_path.is_file():
                close_request = _load_server_state(close_path)
                if str(close_request.get("reason", "")) == "gpu_contention":
                    context.keep_server_warm_after_session = False
                _async_worker_status(job_dir, "closing", stage="close_requested")
                return 0

            # If the parent (Mimics) process has died, shut down immediately
            # instead of holding GPU memory for the full idle timeout.
            if parent_pid and not _process_exists(parent_pid):
                _async_worker_status(
                    job_dir,
                    "expired",
                    stage="parent_process_gone",
                )
                return 0

            commands = sorted((job_dir / "commands").glob("command_*.json"))
            pending = []
            for path in commands:
                try:
                    sequence = int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if sequence > last_sequence:
                    pending.append((sequence, path))

            if pending:
                sequence, command_path = pending[0]
                command = json.loads(command_path.read_text(encoding="utf-8"))
                result_path = job_dir / "results" / f"result_{sequence:06d}.json"
                output_path = str(job_dir / "results" / f"prediction_{sequence:06d}.u8")
                _async_worker_status(
                    job_dir,
                    "running",
                    stage="prediction",
                    sequence=sequence,
                    prediction_steps_hint=int(
                        command.get("prediction_steps_hint") or 1
                    ),
                    command_path=str(command_path),
                )
                try:
                    result = context.predict(
                        command.get("interactions") or [],
                        output_path,
                        initial_seg_path=command.get("initial_seg_path"),
                        initial_seg_shape=command.get("initial_seg_shape"),
                        use_context_initial_seg=False,
                    )
                    result["sequence"] = sequence
                    result["command_id"] = command.get("command_id")
                    result["expected_target_sha256"] = command.get(
                        "expected_target_sha256"
                    )
                    result["model_identity"] = command.get("model_identity", "official")
                    result["model_profile_id"] = command.get("model_profile_id", "official")
                    result["model_id"] = command.get("model_id", "official")
                    result["checkpoint_sha256"] = command.get("checkpoint_sha256", "")
                    result["effective_model_fingerprint"] = command.get(
                        "effective_model_fingerprint", ""
                    )
                    result["task_id"] = command.get("task_id", "")
                except Exception as exc:
                    result = _error_result(
                        exc,
                        stage="async_prediction",
                        started=time.time(),
                        model_dir=context.model_dir,
                        log_path=context.log_path,
                        runtime_work_dir=context.runtime_work_dir,
                    )
                    result["sequence"] = sequence
                    result["command_id"] = command.get("command_id")
                    result["expected_target_sha256"] = command.get(
                        "expected_target_sha256"
                    )
                    result["model_identity"] = command.get("model_identity", "official")
                    result["model_profile_id"] = command.get("model_profile_id", "official")
                    result["model_id"] = command.get("model_id", "official")
                    result["checkpoint_sha256"] = command.get("checkpoint_sha256", "")
                    result["effective_model_fingerprint"] = command.get(
                        "effective_model_fingerprint", ""
                    )
                    result["task_id"] = command.get("task_id", "")
                _write_json_atomic(result_path, result)
                last_sequence = sequence
                last_activity = time.time()
                _async_worker_status(
                    job_dir,
                    "result_ready",
                    stage="waiting_for_mimics",
                    sequence=sequence,
                    result_status=result.get("status"),
                    prediction_steps=result.get("prediction_steps"),
                    result_path=str(result_path),
                    output_path=result.get("output_path"),
                    error=result.get("error"),
                )
                continue

            if time.time() - last_activity >= idle_timeout:
                _async_worker_status(
                    job_dir,
                    "expired",
                    stage="idle_timeout",
                    idle_timeout_seconds=idle_timeout,
                )
                return 0
            time.sleep(poll_seconds)
    except Exception as exc:
        trace = traceback.format_exc()
        try:
            _async_worker_status(
                job_dir,
                "failed",
                stage="initialize_or_poll",
                error=str(exc),
                traceback=trace,
            )
        except Exception:
            pass
        return 2
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        status_path = job_dir / "worker_status.json"
        status = _load_server_state(status_path)
        if status and status.get("status") == "closing":
            _async_worker_status(job_dir, "closed", stage="closed")


def main() -> int:
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    if len(sys.argv) == 4 and sys.argv[1] == "--watchdog":
        return _watchdog_main(sys.argv[2], sys.argv[3])
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        return _worker_main()
    if len(sys.argv) == 3 and sys.argv[1] == "--async-worker":
        return _async_worker_main(sys.argv[2])
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        result = {"status": "error", "error": f"Invalid JSON input: {exc}"}
        print(json.dumps(result, indent=2))
        return 2

    result = run_bridge(input_data)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "error" else 2


if __name__ == "__main__":
    sys.exit(main())
