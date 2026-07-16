#!/usr/bin/env python3
"""External DINOv3 few-shot pipeline for Mimics-Script.

This script never imports Mimics. It is safe to run from the foreground Mimics
process through subprocess.Popen because all long-running work happens here or
in child Python/Mimics background processes.
"""

import argparse
import atexit
import hashlib
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

from resource_locks import FileResourceLock, ResourceLockCancelled, ResourceLockTimeout, release_lock
from tools.fewshot_strategies import compile_strategy, normalize_strategy_options, strategy_ids

DEFAULT_WORKSPACE = "fewshot_models"
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3
RESOURCE_LOCK_DIR = ROOT / ".mimics_runtime" / "locks"
GPU_LOCK_PATH = RESOURCE_LOCK_DIR / "gpu.lock"
BACKGROUND_MIMICS_LOCK_PATH = RESOURCE_LOCK_DIR / "background_mimics.lock"


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
        if _remove_tree_quietly(path):
            removed += 1
    return removed


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text, retries=retries, max_sleep=max_sleep)


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
    if last_error is not None:
        raise last_error


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
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def rotate_log(path):
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size < LOG_ROTATE_BYTES:
            return
        backups = int(LOG_ROTATE_BACKUPS)
        if backups <= 0:
            path.unlink()
            return
        oldest = path.with_name(path.name + "." + str(backups))
        if oldest.is_file():
            oldest.unlink()
        for index in range(backups - 1, 0, -1):
            src = path.with_name(path.name + "." + str(index))
            dst = path.with_name(path.name + "." + str(index + 1))
            if src.is_file():
                src.rename(dst)
        path.rename(path.with_name(path.name + ".1"))
    except Exception:
        pass


def append_log(workspace, message):
    text = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(text, flush=True)
    try:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / "fewshot_pipeline.log"
        rotate_log(path)
        append_text(path, text + "\n")
    except Exception as exc:
        print(
            "[{}] Warning: could not write DINOv3 pipeline log: {}".format(
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


def open_subprocess_log(path, workspace, label):
    path = Path(path)
    workspace = Path(workspace)
    temp_root = Path(tempfile.gettempdir()) / "mimics_script_fewshot_logs"
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


def _status_signature(payload):
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return repr(payload)


def maybe_update_status(path, payload, state, heartbeat_seconds=5.0):
    now = time.time()
    signature = _status_signature(payload)
    if signature == state.get("signature") and now - float(state.get("epoch", 0.0) or 0.0) < float(heartbeat_seconds):
        return False
    update_status(path, payload)
    state["signature"] = signature
    state["epoch"] = now
    return True


def global_registry_path():
    return Path.home() / ".mimics_script" / "fewshot_model_index.json"


def parse_case_list(value):
    if not value:
        return None
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def load_repo_config():
    path = ROOT / "fewshot_config.json"
    if path.is_file():
        return read_json(path, {})
    return {}


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_JOB_STATUSES = {
    "launching", "preparing", "exporting_labels", "waiting_for_background_mimics",
    "waiting_for_gpu", "training", "running", "cancelling",
}


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


def _register_transient_cleanup(path):
    path = str(Path(path))
    atexit.register(lambda: _remove_tree_quietly(path))


def cleanup_workspace_artifacts(workspace, dinov3_root=None, config=None):
    """Bound disposable few-shot history without deleting registered models."""
    workspace = Path(workspace)
    config = load_repo_config() if config is None else (config or {})
    now = time.time()
    job_days = max(1, int(config.get("terminal_job_retention_days", 30)))
    max_jobs = max(1, int(config.get("max_terminal_job_records", 100)))
    failed_days = max(1, int(config.get("failed_run_retention_days", 7)))
    run_log_days = max(1, int(config.get("completed_run_log_retention_days", 30)))
    setup_days = max(1, int(config.get("setup_context_retention_days", 7)))
    report = {"job_records": 0, "datasets": 0, "failed_runs": 0, "run_logs": 0, "contexts": 0, "experiments": 0}

    jobs_dir = workspace / "jobs"
    terminal = []
    statuses = {}
    if jobs_dir.is_dir():
        for path in jobs_dir.glob("*.json"):
            payload = read_json(path, {}) or {}
            status = str(payload.get("status", "")).lower()
            job_id = str(payload.get("job_id") or path.stem)
            pid = payload.get("pid") or payload.get("controller_pid") or payload.get("launcher_pid")
            try:
                pid = int(pid or 0)
            except Exception:
                pid = 0
            if status in ACTIVE_JOB_STATUSES and pid and not process_exists(pid):
                status = "failed"
                payload.update({
                    "status": "failed",
                    "error": "background controller exited before recording a terminal state",
                    "orphaned": True,
                    "updated_at_epoch": min(
                        float(payload.get("updated_at_epoch") or payload.get("created_at_epoch") or path.stat().st_mtime),
                        path.stat().st_mtime,
                    ),
                })
                try:
                    write_json_atomic(path, payload)
                except Exception:
                    pass
            if status in TERMINAL_JOB_STATUSES:
                updated = float(payload.get("updated_at_epoch") or payload.get("created_at_epoch") or path.stat().st_mtime)
                terminal.append((updated, path, payload))
                statuses[job_id] = payload
            elif path.name.endswith("_context.json") and now - path.stat().st_mtime > setup_days * 86400:
                if _remove_tree_quietly(path):
                    report["contexts"] += 1
        terminal.sort(key=lambda item: item[0], reverse=True)
        for index, (updated, path, _payload) in enumerate(terminal):
            if index < max_jobs and now - updated <= job_days * 86400:
                continue
            if _remove_tree_quietly(path):
                report["job_records"] += 1

    for job_id, payload in statuses.items():
        status = str(payload.get("status", "")).lower()
        updated = float(payload.get("updated_at_epoch") or payload.get("created_at_epoch") or now)
        dataset_path = payload.get("dataset_dir")
        retained = bool((payload.get("model") or {}).get("dataset_retained", False))
        if dataset_path and not retained and _remove_tree_quietly(dataset_path):
            report["datasets"] += 1
        if str(payload.get("kind", "")) == "infer" and now - updated > run_log_days * 86400:
            for key in ("log", "cancel_path"):
                path = payload.get(key)
                if path and _remove_tree_quietly(path):
                    report["run_logs"] += 1
        run_dir = workspace / "runs" / safe_slug(payload.get("organ", "")) / job_id
        if status in ("failed", "cancelled") and now - updated > failed_days * 86400:
            if _remove_tree_quietly(run_dir):
                report["failed_runs"] += 1
        elif status == "completed" and now - updated > run_log_days * 86400:
            for name in ("train.log", "train.log.1", "train.log.2", "train.log.3"):
                path = run_dir / name
                if path.is_file() and _remove_tree_quietly(path):
                    report["run_logs"] += 1

        if dinov3_root and status in TERMINAL_JOB_STATUSES and not bool(config.get("keep_training_experiment_artifacts", False)):
            experiment = Path(dinov3_root) / "experiments" / "mimics_fewshot_{}_{}".format(
                safe_slug(payload.get("organ", "")), job_id
            )
            if _remove_tree_quietly(experiment):
                report["experiments"] += 1
    return report


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


def workspace_for(ts_root, workspace=None):
    if workspace:
        return Path(workspace).resolve()
    return Path(ts_root).resolve() / DEFAULT_WORKSPACE


def dinov3_root_from_args(args):
    repo_cfg = load_repo_config()
    value = args.dinov3_root or os.environ.get("MIMICS_FEWSHOT_DINOV3_ROOT") or repo_cfg.get("dinov3_project")
    if value:
        path = resolve_path(value, ROOT)
    else:
        path = (ROOT / "external" / "dinov3-medical-seg").resolve()
    if not (path / "scripts" / "train.py").is_file():
        raise RuntimeError("DINOv3 project was not found: {}".format(path))
    return path


def project_python_candidates():
    return [
        ROOT / "nninteractive_env" / "python.exe",
        ROOT / "nninteractive_env" / "Scripts" / "python.exe",
        ROOT / "nninteractive_env" / "python" / "python.exe",
        ROOT / "nninteractive_env" / "bin" / "python3",
        ROOT / "nninteractive_env" / "bin" / "python",
    ]


def append_python_candidate(candidates, value, base=ROOT):
    if not value:
        return
    text = str(value)
    if text in ("python", "python3"):
        return
    path = Path(text)
    if not path.is_absolute():
        path = Path(base) / path
    candidates.append(path)


def python_from_args(args, dinov3_root):
    repo_cfg = load_repo_config()
    candidates = []
    if args.python:
        append_python_candidate(candidates, args.python)
    candidates.extend(project_python_candidates())
    if not args.python:
        append_python_candidate(candidates, os.environ.get("MIMICS_FEWSHOT_PYTHON"))
        append_python_candidate(candidates, repo_cfg.get("python"))
    candidates.extend([
        dinov3_root / ".venv" / "Scripts" / "python.exe",
        dinov3_root / ".venv" / "bin" / "python",
        dinov3_root / "venv" / "Scripts" / "python.exe",
        dinov3_root / "venv" / "bin" / "python",
    ])
    current = Path(sys.executable)
    try:
        current_resolved = current.resolve()
        env_root = (ROOT / "nninteractive_env").resolve()
        if str(current_resolved).startswith(str(env_root)):
            candidates.append(current)
    except Exception:
        pass
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run Setup Environment or setup_offline.bat before using DINOv3."
    )


def base_config_from_args(args, dinov3_root):
    repo_cfg = load_repo_config()
    value = args.base_config or repo_cfg.get("base_config") or "config/research/ct_fewshot_fast.yaml"
    path = resolve_path(value, dinov3_root)
    if not path.is_file():
        raise RuntimeError("base config was not found: {}".format(path))
    return path


def find_mimics_exe(explicit=None):
    """Find MimicsResearch.exe, with optional explicit override."""
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))
    # Delegate to runtime_common which has the full search logic
    sys.path.insert(0, str(ROOT / "runtime_py35"))
    import runtime_common
    return runtime_common.find_mimics_exe()


def hidden_process_kwargs():
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"startupinfo": startupinfo, "creationflags": flags}


def background_env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def process_exists(pid):
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
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


def _env_flag_disabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def gpu_lock_enabled():
    if _env_flag_disabled("MIMICS_DISABLE_GPU_LOCK"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        # If torch cannot be imported here, the child training process will
        # fail soon anyway. Keep the lock conservative so a broken environment
        # does not start competing with nnInteractive.
        return True


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
            "Wait for import to finish, or use 01 Data > 04 Stop Import Queue if training is more urgent."
        )
    elif resource == "gpu":
        payload["user_action"] = (
            "The job will start automatically when the active AI operation releases the GPU. "
            "Use 02 AI > DINOv3 > 05 Stop AI Task only when the running task should be cancelled."
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


def cleanup_idle_nninteractive_server_lock(current):
    if not isinstance(current, dict):
        return False
    if current.get("resource") != "gpu":
        return False
    state_path = current.get("state_path")
    if not state_path:
        return False
    state = read_json(state_path, {}) or {}
    if state.get("schema_version") != "nninteractive_owned_server.v2":
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
        terminate_process_tree(pid)
    release_lock(current.get("path") or GPU_LOCK_PATH, current.get("token"))
    try:
        Path(state_path).unlink()
    except Exception:
        pass
    return True


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
    if state.get("schema_version") != "nninteractive_owned_server.v2":
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
    try:
        watchdog_pid = int(state.get("watchdog_pid", 0) or 0)
    except Exception:
        watchdog_pid = 0
    if not watchdog_pid or not process_exists(watchdog_pid):
        return False

    now = time.time()
    min_idle_seconds = 15.0
    try:
        last_activity = float(state.get("last_activity_epoch", now))
    except Exception:
        last_activity = now
    if now - last_activity < min_idle_seconds:
        return False

    try:
        requested_at = float(state.get("contention_release_requested_epoch", 0.0) or 0.0)
    except Exception:
        requested_at = 0.0
    if requested_at and now - requested_at < 10.0:
        return False

    state["contention_release_requested_epoch"] = now
    state["contention_release_requested_by"] = "fewshot_pipeline"
    control_dir = str(state.get("client_control_dir") or "").strip()
    try:
        client_pid = int(state.get("client_pid", 0) or 0)
    except Exception:
        client_pid = 0
    if control_dir and client_pid and process_exists(client_pid):
        write_json_atomic(state_path, state)
        write_json_atomic(Path(control_dir) / "close.json", {
            "reason": "gpu_contention",
            "requested_at_epoch": now,
            "requested_by": "fewshot_pipeline",
        })
    else:
        # Legacy one-shot bridges have no worker control channel. Expire the
        # server directly; the watchdog owns termination and lock release.
        state["last_activity_epoch"] = 0.0
        write_json_atomic(state_path, state)
    return True


def acquire_gpu_lock_for_job(workspace, status_path, cancel_path, owner, timeout_seconds):
    if not gpu_lock_enabled():
        return None
    lock = FileResourceLock(GPU_LOCK_PATH, "gpu", owner)
    cleanup_idle_nninteractive_server_lock(lock.read())
    last_log = {"epoch": 0.0}

    def on_wait(current):
        if cleanup_idle_nninteractive_server_lock(current):
            append_log(workspace, "Cleaned an idle nnInteractive server whose watchdog was no longer running.")
            return
        if request_nninteractive_server_release_on_contention(current):
            append_log(workspace, "Requested nnInteractive server to release GPU due to lock contention.")
            return
        update_status(status_path, {
            "status": "waiting_for_gpu",
            "resource_wait": _lock_wait_payload("gpu", current),
        })
        now = time.time()
        if now - last_log["epoch"] >= 60:
            holder = current.get("owner", "unknown") if isinstance(current, dict) else "unknown"
            pid = current.get("pid", "?") if isinstance(current, dict) else "?"
            append_log(workspace, "Waiting for GPU resource held by {} (pid {}).".format(holder, pid))
            last_log["epoch"] = now

    update_status(status_path, {
        "status": "waiting_for_gpu",
        "resource_wait": {"resource": "gpu"},
    })
    lock.acquire(
        wait_seconds=float(timeout_seconds),
        poll_seconds=2.0,
        on_wait=on_wait,
        should_cancel=lambda: Path(cancel_path).is_file(),
    )
    update_status(status_path, {"resource_wait": None})
    append_log(workspace, "GPU resource acquired for {}.".format(owner))
    return lock


def acquire_background_mimics_lock(workspace, status_path, cancel_path, owner, timeout_seconds):
    lock = FileResourceLock(BACKGROUND_MIMICS_LOCK_PATH, "background_mimics", owner)
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
        should_cancel=lambda: Path(cancel_path).is_file(),
    )
    update_status(status_path, {"resource_wait": None})
    return lock


def terminate_process_tree(pid):
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
                **hidden_process_kwargs()
            )
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def latest_running_job(workspace):
    jobs_dir = Path(workspace) / "jobs"
    if not jobs_dir.is_dir():
        return None, None
    active_statuses = set([
        "launching",
        "preparing",
        "exporting_labels",
        "waiting_for_background_mimics",
        "waiting_for_gpu",
        "training",
        "running",
        "cancelling",
    ])
    rows = []
    for path in jobs_dir.glob("*.json"):
        payload = read_json(path, {}) or {}
        if payload.get("status") not in active_statuses:
            continue
        pid = payload.get("pid") or payload.get("launcher_pid") or payload.get("controller_pid")
        if pid and not process_exists(pid):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        rows.append((mtime, path, payload))
    if not rows:
        return None, None
    rows.sort(reverse=True)
    return rows[0][1], rows[0][2]


def case_dirs(ts_root, cases=None):
    root = Path(ts_root)
    wanted = set(cases or [])
    result = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ("mcs_output", "segmentations", DEFAULT_WORKSPACE):
            continue
        if wanted and child.name not in wanted:
            continue
        result.append(child)
    return result


def find_image(case_dir):
    root = Path(case_dir)
    preferred = ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii")
    candidates = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            candidates.append(path)
    for wanted in preferred:
        for path in sorted(candidates):
            if path.name.lower() == wanted:
                return path
    for path in sorted(candidates):
        if "segmentations" not in [part.lower() for part in path.parts]:
            return path
    return None


def _find_label_in_seg_dir(seg_dir, organ):
    seg_dir = Path(seg_dir)
    if not seg_dir.is_dir():
        return None
    names = [
        organ + ".nii.gz",
        organ + ".nii",
        safe_slug(organ) + ".nii.gz",
        safe_slug(organ) + ".nii",
    ]
    for name in names:
        path = seg_dir / name
        if path.is_file():
            return path
    organ_lower = organ.lower()
    for path in sorted(seg_dir.glob("*.nii.gz")) + sorted(seg_dir.glob("*.nii")):
        stem = path.name.replace(".nii.gz", "").replace(".nii", "")
        if stem.lower() == organ_lower:
            return path
    return None


def find_label(case_dir, organ, label_root=None, fallback_to_case_labels=True):
    case_dir = Path(case_dir)
    search_dirs = []
    if label_root:
        root = Path(label_root)
        search_dirs.extend([
            root / case_dir.name / "segmentations",
            root / case_dir.name,
        ])
    if fallback_to_case_labels:
        search_dirs.append(case_dir / "segmentations")
    seen = set()
    for seg_dir in search_dirs:
        resolved = str(Path(seg_dir))
        if resolved in seen:
            continue
        seen.add(resolved)
        label = _find_label_in_seg_dir(seg_dir, organ)
        if label:
            return label
    return None


def _label_skip_reason(label_path):
    """Return a skip reason for unusable labels, otherwise an empty string."""
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(str(label_path))
        if len(img.shape) < 3 or any(int(value) <= 0 for value in img.shape[:3]):
            return "label is not a valid 3D NIfTI"
        data = np.asanyarray(img.dataobj)
        if not np.any(data):
            return "label is empty (all zeros)"
        return ""
    except Exception as exc:
        return "label could not be read: {}".format(exc)


def discover_samples(ts_root, organ, cases=None, label_root=None, fallback_to_case_labels=True):
    samples = []
    skipped = []
    for case_dir in case_dirs(ts_root, cases):
        image = find_image(case_dir)
        label = find_label(
            case_dir,
            organ,
            label_root=label_root,
            fallback_to_case_labels=fallback_to_case_labels,
        )
        if image and label:
            label_skip_reason = _label_skip_reason(label)
            if label_skip_reason:
                skipped.append({
                    "case_id": case_dir.name,
                    "has_image": True,
                    "has_label": True,
                    "reason": label_skip_reason,
                })
                continue
            samples.append({
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "image": str(image),
                "label": str(label),
                "label_root": str(label_root) if label_root else "",
                "label_source": "fresh_export" if label_root else "case_segmentations",
                "label_mtime": float(label.stat().st_mtime),
            })
        else:
            skipped.append({
                "case_id": case_dir.name,
                "has_image": bool(image),
                "has_label": bool(label),
            })
    return samples, skipped


def select_samples(samples, mode, max_samples):
    selected = list(samples)
    if mode == "latest":
        selected.sort(key=lambda item: item.get("label_mtime", 0.0), reverse=True)
    else:
        selected.sort(key=lambda item: item["case_id"])
    if max_samples and max_samples > 0:
        selected = selected[:max_samples]
    selected.sort(key=lambda item: item["case_id"])
    return selected


def split_train_validation(samples, val_fraction=0.0, val_cases=None, min_train_samples=1, min_val_samples=1):
    selected = list(samples)
    selected.sort(key=lambda item: item["case_id"])
    val_case_set = set(val_cases or [])
    if val_case_set:
        train_samples = [item for item in selected if item["case_id"] not in val_case_set]
        val_samples = [item for item in selected if item["case_id"] in val_case_set]
        if len(train_samples) < int(min_train_samples):
            raise RuntimeError(
                "validation case selection leaves only {} training samples; at least {} required".format(
                    len(train_samples),
                    int(min_train_samples),
                )
            )
        return train_samples, val_samples

    fraction = float(val_fraction or 0.0)
    if fraction <= 0.0:
        return selected, []
    if len(selected) <= int(min_train_samples):
        return selected, []

    val_count = int(round(len(selected) * fraction))
    if val_count <= 0 and len(selected) >= int(min_train_samples) + int(min_val_samples):
        val_count = int(min_val_samples)
    val_count = max(0, min(val_count, len(selected) - int(min_train_samples)))
    if val_count <= 0:
        return selected, []
    return selected[:-val_count], selected[-val_count:]


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


def _materialize_image_aligned_to_label(image_src, label_src, image_dst):
    """Materialize image so it shares the label/Mimics grid."""
    import nibabel as nib
    from mimics_bridge import _affine_close, resample_image_to_grid

    image_img = nib.load(str(image_src))
    label_img = nib.load(str(label_src))
    image_shape = tuple(int(value) for value in image_img.shape[:3])
    label_shape = tuple(int(value) for value in label_img.shape[:3])
    image_affine = image_img.affine
    label_affine = label_img.affine
    if image_shape == label_shape and _affine_close(image_affine, label_affine):
        return copy_or_link(image_src, image_dst), True

    image_dst = Path(image_dst)
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    resample_image_to_grid(str(image_src), label_shape, label_affine, str(image_dst))
    return "resampled_to_label_grid", False


def validate_fresh_export_geometry(samples):
    """Fail closed when a fresh Mimics export is not on its source-image grid."""
    import nibabel as nib
    from mimics_bridge import _affine_close

    checked = []
    for sample in samples:
        if sample.get("label_source") != "fresh_export":
            continue
        image = nib.load(str(sample["image"]))
        label = nib.load(str(sample["label"]))
        image_shape = tuple(int(value) for value in image.shape[:3])
        label_shape = tuple(int(value) for value in label.shape[:3])
        if image_shape != label_shape or not _affine_close(image.affine, label.affine):
            raise RuntimeError(
                "fresh Mimics label export is not aligned to the source image for case {0}; "
                "image shape {1}, label shape {2}. Training was stopped instead of silently "
                "resampling the image onto a different grid.".format(
                    sample.get("case_id", "?"), image_shape, label_shape,
                )
            )
        checked.append(sample.get("case_id"))
    return checked


def _materialize_split(samples, image_dir, label_dir, split_name):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for sample in samples:
        case_id = safe_slug(sample["case_id"])
        image_src = Path(sample["image"])
        label_src = Path(sample["label"])
        image_ext = ".nii.gz" if image_src.name.endswith(".nii.gz") else ".nii"
        label_ext = ".nii.gz" if label_src.name.endswith(".nii.gz") else ".nii"
        image_dst = image_dir / (case_id + image_ext)
        label_dst = label_dir / (case_id + label_ext)
        image_method, geometry_matched = _materialize_image_aligned_to_label(image_src, label_src, image_dst)
        label_method = copy_or_link(label_src, label_dst)
        row = dict(sample)
        row.update({
            "split": split_name,
            "source_image": str(image_src),
            "source_label": str(label_src),
            "dataset_image": str(image_dst),
            "dataset_label": str(label_dst),
            "image_materialization": image_method,
            "label_materialization": label_method,
            "image_label_geometry_matched": bool(geometry_matched),
        })
        materialized.append(row)
    return materialized


def _estimate_materialize_bytes(samples):
    total = 0
    seen = set()
    for sample in samples or []:
        for key in ("image", "label"):
            try:
                path = Path(sample[key]).resolve()
            except Exception:
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                total += int(path.stat().st_size)
            except Exception:
                pass
    return total


def _check_materialize_disk_space(train_samples, val_samples, dataset_dir):
    required = int(_estimate_materialize_bytes(list(train_samples or []) + list(val_samples or [])) * 1.5)
    required = max(required, 50 * 1024 * 1024)
    target = Path(dataset_dir)
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(str(probe))
    if usage.free < required:
        raise RuntimeError(
            "not enough free disk space to materialize the few-shot dataset: "
            "required about {:.1f} MB, free {:.1f} MB at {}".format(
                required / (1024.0 * 1024.0),
                usage.free / (1024.0 * 1024.0),
                probe,
            )
        )


def materialize_dataset(train_samples, dataset_dir, val_samples=None):
    dataset_dir = Path(dataset_dir)
    _check_materialize_disk_space(train_samples, val_samples or [], dataset_dir)
    if dataset_dir.exists():
        rmtree_with_retry(dataset_dir)
    train_rows = _materialize_split(
        train_samples,
        dataset_dir / "imagesTr",
        dataset_dir / "labelsTr",
        "train",
    )
    val_rows = _materialize_split(
        val_samples or [],
        dataset_dir / "imagesVal",
        dataset_dir / "labelsVal",
        "validation",
    )
    return train_rows, val_rows


def validate_materialized_dataset(rows):
    """Validate image/label pairs that will be passed to DINOv3 training."""
    import nibabel as nib
    import numpy as np
    from mimics_bridge import _affine_close

    issues = []
    checked = []
    for row in rows or []:
        case_id = row.get("case_id", "")
        image_path = row.get("dataset_image")
        label_path = row.get("dataset_label")
        try:
            image_img = nib.load(str(image_path))
            label_img = nib.load(str(label_path))
            image_shape = tuple(int(value) for value in image_img.shape[:3])
            label_shape = tuple(int(value) for value in label_img.shape[:3])
            label_data = np.asanyarray(label_img.dataobj)
            foreground_voxels = int(np.count_nonzero(label_data))
            affine_close = bool(_affine_close(image_img.affine, label_img.affine))
            affine_max_abs_diff = float(np.max(np.abs(image_img.affine - label_img.affine)))
            row["dataset_validation"] = {
                "image_shape": list(image_shape),
                "label_shape": list(label_shape),
                "affine_close": affine_close,
                "affine_max_abs_diff": affine_max_abs_diff,
                "foreground_voxels": foreground_voxels,
            }
            if image_shape != label_shape:
                issues.append("{}: image shape {} != label shape {}".format(case_id, image_shape, label_shape))
            if not affine_close:
                issues.append("{}: image/label affine mismatch (max abs diff {:.6g})".format(
                    case_id,
                    affine_max_abs_diff,
                ))
            if foreground_voxels <= 0:
                issues.append("{}: label is empty after materialization".format(case_id))
            checked.append(row["dataset_validation"])
        except Exception as exc:
            issues.append("{}: could not validate materialized image/label pair: {}".format(case_id, exc))
    if issues:
        raise RuntimeError("few-shot dataset validation failed: " + "; ".join(issues[:10]))
    return {
        "checked_pairs": len(checked),
        "issues": [],
    }


def yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(item) for item in value) + "]"
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return '"' + text + '"'


def compute_class_weights(materialized_rows, max_weight=20.0):
    """Compute [bg_weight, fg_weight] from materialized label files.

    Scans every training label, counts background vs foreground voxels,
    and returns inverse-frequency weights normalised so that bg=1.0.
    The foreground weight is clamped to *max_weight* to avoid instability
    when the organ occupies a tiny fraction of the volume.
    """
    import nibabel as nib
    import numpy as np

    bg_total = 0
    fg_total = 0
    for row in materialized_rows or []:
        label_path = row.get("dataset_label")
        if not label_path:
            continue
        try:
            data = np.asanyarray(nib.load(str(label_path)).dataobj)
            fg = int(np.count_nonzero(data))
            bg = int(data.size - fg)
            bg_total += bg
            fg_total += fg
        except Exception:
            continue
    if fg_total <= 0:
        return [1.0, 1.0]
    ratio = bg_total / fg_total
    fg_weight = min(float(ratio), float(max_weight))
    return [1.0, round(fg_weight, 2)]


def write_training_config(
    path,
    base_config,
    dataset_dir,
    exp_name,
    args,
    status_path=None,
    cancel_path=None,
    metrics_history_path=None,
    validation_enabled=True,
    class_weights=None,
    strategy_overrides=None,
):
    path = Path(path)
    if int(args.batch_size) != 1:
        raise RuntimeError(
            "Batch size must stay 1 for variable-depth 3D Mimics cases. "
            "Use Grad accumulation to increase the effective batch size."
        )
    img_size = [int(part.strip()) for part in str(args.img_size).split(",")]
    model_path = args.model_path
    if not model_path:
        model_scale = str(args.model_scale or "vitb16").lower()
        if model_scale in ("vitl16", "vit_l", "large"):
            model_path = "./models/dinov3-vitl16"
        elif model_scale in ("vith16plus", "vit_h", "huge"):
            model_path = "./models/dinov3-vith16plus"
        else:
            model_path = "./models/dinov3-vitb16"
    finetune_method = str(args.finetune_method or "lora").lower()
    if finetune_method in ("decoder_only", "decode_only", "decoder-only", "decode-only"):
        finetune_method = "frozen"
    decoder_type = str(args.decoder or "segformer3d")
    config = {
        "_base_": [str(base_config)],
        "exp_name": exp_name,
        "model": {
            "model_path": model_path, "num_classes": 2, "input_normalization": "imagenet",
            "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
        },
        "finetune": {
            "method": finetune_method, "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha), "adapter_bottleneck": int(args.adapter_bottleneck),
        },
        "decoder": {"type": decoder_type},
        "data": {
            "name": "mimics_fewshot_" + safe_slug(args.organ), "data_root": str(dataset_dir),
            "img_size": img_size, "k_shot": -1, "fold": 0, "modality": args.modality,
        },
        "training": {
            "epochs": int(args.epochs), "batch_size": int(args.batch_size),
            "grad_accumulation": int(args.grad_accumulation), "mixed_precision": bool(args.mixed_precision),
            "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            "scheduler": None if getattr(args, "lr_scheduler", "cosine") == "constant" else getattr(args, "lr_scheduler", "cosine"),
            "warmup_epochs": int(getattr(args, "warmup_epochs", 3)),
            "keep_last_checkpoints": int(args.keep_last_checkpoints),
            "validation_enabled": bool(validation_enabled),
            "sub_volume": {"enabled": bool(args.sub_volume),
                           "size": [int(part.strip()) for part in str(args.sub_volume_size).split(",")]},
        },
        "loss": {"type": "dice_focal", "dice_weight": 0.7, "focal_weight": 0.3,
                 "focal_alpha": 0.75, "focal_gamma": 2.0,
                 "class_weights": class_weights if class_weights else [1.0, 1.0]},
    }

    def merge(target, update):
        for key, value in (update or {}).items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
        return target

    merge(config, strategy_overrides or {})
    if status_path or cancel_path or metrics_history_path:
        config["runtime"] = {
            "status_path": str(status_path or ""), "cancel_path": str(cancel_path or ""),
            "metrics_history_path": str(metrics_history_path or ""), "status_interval_seconds": 2.0,
        }
    import yaml
    write_text_atomic(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=False))
    return config


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
    mcs_output_dir=None,
):
    mimics_exe = find_mimics_exe(mimics_exe)
    if not mimics_exe:
        append_log(workspace, "A separate background Mimics executable was not found; using existing exported labels only.")
        return {"launched": False, "reason": "mimics_not_found"}
    # The .mcs files live in the configured Mimics output directory, while
    # few-shot control artifacts stay in the fewshot workspace.
    output_dir = Path(mcs_output_dir).expanduser().resolve() if mcs_output_dir else resolve_mimics_output_dir(ts_root)
    axes, flips = resolve_mimics_buffer_mapping()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_job_id = safe_slug(Path(status_path).stem if status_path else "export_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"),
        uuid.uuid4().hex[:8],
    ))
    cleanup_local_export_jobs()
    export_root = ROOT / ".mimics_runtime" / "export_jobs" / ("fewshot_" + export_job_id)
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
    }
    if mask_names:
        export_config["mask_names"] = [str(name) for name in mask_names if str(name).strip()]
    if label_staging_dir:
        export_config["label_staging_dir"] = str(label_staging_dir)
    write_json_atomic(config_path_write, export_config)
    runner = export_root / "run_export_batch.py"
    write_text_atomic(
        runner,
        "\n".join([
            "# Auto-generated runner for Mimics few-shot label export",
            "import sys, os, json, time",
            "open(r'{}', 'w').write(json.dumps({{'pid': os.getpid(), 'started_at_epoch': time.time()}}))".format(
                str(handshake_path)
            ),
            "sys.path.insert(0, r'{}')".format(str(ROOT / "runtime_py35")),
            "import mimics_export",
            "mimics_export.run_background_batch_export(r'{}')".format(str(config_path_write)),
            "",
        ]),
    )
    lock = None
    if status_path and cancel_path:
        lock = acquire_background_mimics_lock(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 label export",
            lock_timeout_seconds if lock_timeout_seconds is not None else timeout_seconds,
        )
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
        if lock is not None:
            lock.update_pid(proc.pid, kind="fewshot_label_export", ts_root=str(Path(ts_root).resolve()))
    except Exception:
        if lock is not None:
            lock.release()
        raise
    append_log(workspace, "Background Mimics export started, pid={}.".format(proc.pid))
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if proc.poll() is not None:
            if lock is not None:
                lock.release()
            result = {"launched": True, "returncode": proc.returncode}
            batch_status = read_json(batch_status_path, {}) or {}
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
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    if lock is not None:
        lock.release()
    result = {
        "launched": True,
        "timed_out": True,
        "pid": proc.pid,
        "runner_started": handshake_path.is_file(),
        "job_runtime": str(export_root),
        "mimics_log": str(mimics_log_path),
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


def latest_epoch_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    best = ckpt_dir / "best_model.pth"
    if best.is_file():
        return best
    candidates = sorted(ckpt_dir.glob("epoch_*.pth"))
    return candidates[-1] if candidates else None


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


def register_global_model(manifest):
    path = global_registry_path()
    payload = read_json(path, {}) or {}
    models = payload.get("models") or []
    manifest_path = str(Path(manifest["checkpoint"]).parent / "manifest.json")
    row = {
        "organ": manifest.get("organ", ""),
        "organ_slug": manifest.get("organ_slug", safe_slug(manifest.get("organ", ""))),
        "model_id": manifest.get("model_id", ""),
        "manifest_path": manifest_path,
        "checkpoint": manifest.get("checkpoint", ""),
        "config": manifest.get("config", ""),
        "sample_count": manifest.get("sample_count", 0),
        "train_sample_count": manifest.get("train_sample_count", 0),
        "validation_sample_count": manifest.get("validation_sample_count", 0),
        "created_at_epoch": manifest.get("created_at_epoch", time.time()),
        "ts_root": manifest.get("ts_root", ""),
        "workspace": manifest.get("workspace", ""),
    }
    models = [
        item for item in models
        if not (
            item.get("organ_slug") == row["organ_slug"]
            and item.get("model_id") == row["model_id"]
            and item.get("manifest_path") == row["manifest_path"]
        )
    ]
    models.append(row)
    models.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    write_json_atomic(path, {"schema_version": "mimics_fewshot_global_models.v1", "models": models[:200]})


def global_model_rows(organ=None):
    payload = read_json(global_registry_path(), {}) or {}
    rows = []
    wanted = safe_slug(organ) if organ else None
    for row in payload.get("models") or []:
        if wanted and row.get("organ_slug") != wanted:
            continue
        manifest_path = Path(row.get("manifest_path", ""))
        if not manifest_path.is_file():
            continue
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    return rows


def training_progress_line(progress):
    if not isinstance(progress, dict):
        return ""
    if progress.get("latest_epoch_line"):
        return str(progress.get("latest_epoch_line"))
    parts = []
    if progress.get("epoch") is not None and progress.get("epochs") is not None:
        parts.append("epoch {}/{}".format(progress.get("epoch"), progress.get("epochs")))
    if progress.get("phase"):
        parts.append(str(progress.get("phase")))
    if progress.get("batch") is not None and progress.get("batches") is not None:
        parts.append("batch {}/{}".format(progress.get("batch"), progress.get("batches")))
    metrics = progress.get("metrics") or {}
    if metrics.get("loss") is not None:
        try:
            parts.append("loss {:.4f}".format(float(metrics.get("loss"))))
        except Exception:
            parts.append("loss {}".format(metrics.get("loss")))
    if metrics.get("mean_dsc") is not None:
        try:
            parts.append("val_dice {:.4f}".format(float(metrics.get("mean_dsc"))))
        except Exception:
            parts.append("val_dice {}".format(metrics.get("mean_dsc")))
    if progress.get("lr") is not None:
        try:
            parts.append("lr {:.2e}".format(float(progress.get("lr"))))
        except Exception:
            pass
    if progress.get("best_dsc") is not None:
        try:
            parts.append("best {:.4f}".format(float(progress.get("best_dsc"))))
        except Exception:
            pass
    return ", ".join(parts)


def cmd_train(args):
    strategy_id = str(getattr(args, "strategy", "adaptive") or "adaptive")
    try:
        strategy_options = json.loads(str(getattr(args, "strategy_options_json", "") or "{}"))
    except Exception as exc:
        raise RuntimeError("invalid strategy options JSON: {}".format(exc))
    if not isinstance(strategy_options, dict):
        raise RuntimeError("strategy options JSON must contain an object")
    # Validate all combinations that do not depend on the materialized-data
    # fingerprint before launching background Mimics or allocating a GPU.
    normalize_strategy_options(strategy_options, preset=strategy_id)
    ts_root = Path(args.ts_root).resolve()
    dinov3_root = dinov3_root_from_args(args)
    python_exe = python_from_args(args, dinov3_root)
    base_config = base_config_from_args(args, dinov3_root)
    workspace = workspace_for(ts_root, args.workspace)
    organ_slug = safe_slug(args.organ)
    run_id = args.run_id or "train_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_dir = workspace / "jobs"
    status_path = job_dir / (run_id + ".json")
    run_dir = workspace / "runs" / organ_slug / run_id
    dataset_dir = workspace / "datasets" / organ_slug / run_id
    config_path = run_dir / "config.yaml"
    train_log = run_dir / "train.log"
    train_status = run_dir / "train_status.json"
    metrics_history = run_dir / "metrics_history.json"
    cancel_path = run_dir / "cancel.request"
    exp_name = "mimics_fewshot_{}_{}".format(organ_slug, run_id)

    workspace.mkdir(parents=True, exist_ok=True)
    repo_config = load_repo_config()
    maintenance = cleanup_workspace_artifacts(workspace, dinov3_root, repo_config)
    if any(maintenance.values()):
        append_log(workspace, "Automatic storage maintenance removed disposable artifacts: {}.".format(maintenance))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(status_path, {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": run_id,
        "kind": "train",
        "status": "preparing",
        "organ": args.organ,
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "controller_pid": os.getpid(),
        "cancel_path": str(cancel_path),
        "metrics_history": str(metrics_history),
        "created_at_epoch": time.time(),
    })
    append_log(workspace, "Training job {} started for organ {}.".format(run_id, args.organ))

    cases = set(parse_case_list(args.cases) or [])
    if not cases:
        cases = None
    fresh_label_root = None
    fresh_label_cleanup = {"enabled": False}
    if args.export_labels:
        update_status(status_path, {"status": "exporting_labels"})
        fresh_label_root = run_dir / "fresh_labels"
        _register_transient_cleanup(fresh_label_root)
        try:
            export_result = launch_mimics_export(
                ts_root,
                cases,
                args.mimics_exe,
                workspace,
                args.export_timeout_seconds,
                status_path=status_path,
                cancel_path=cancel_path,
                lock_timeout_seconds=args.background_mimics_lock_timeout_seconds,
                label_staging_dir=fresh_label_root,
                export_space="source_image",
                mask_names=[args.organ],
                mcs_output_dir=args.mcs_output_dir,
            )
        except ResourceLockCancelled:
            update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for background Mimics"})
            return 130
        except ResourceLockTimeout as exc:
            update_status(status_path, {"status": "failed", "error": str(exc)})
            return 75
        if not export_result.get("launched"):
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": "fresh label export was requested but background Mimics export could not be started",
            })
            return 75
        if export_result.get("timed_out") or int(export_result.get("returncode", 0) or 0) != 0:
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": "fresh label export did not finish successfully; training was not started with stale labels",
            })
            return 75
        batch_status = export_result.get("batch_status") or {}
        if (
            str(batch_status.get("status") or "").lower() == "failed"
            or int(batch_status.get("failed", 0) or 0) > 0
        ):
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": (
                    "Fresh label export failed for {0} selected case(s). Training was not started "
                    "with an incomplete dataset. Review the job-scoped export diagnostics: {1}"
                ).format(
                    int(batch_status.get("failed", 0) or 0),
                    export_result.get("job_runtime") or export_result.get("log") or workspace,
                ),
            })
            return 75
        update_status(status_path, {"label_export": export_result})

        # Guard against the background Mimics export exiting with code 0 without
        # actually running the script (e.g. MimicsResearch -b -run_script started
        # but never executed the runner). In that case fresh_labels is empty and
        # the generic "found 0 labels" error below is misleading. Detect it here
        # and point at the real cause.
        if fresh_label_root is not None and not any(fresh_label_root.rglob("*")):
            export_log = export_result.get("log")
            log_size = -1
            if export_log and Path(export_log).is_file():
                log_size = Path(export_log).stat().st_size
            batch_status = export_result.get("batch_status") or {}
            runner_started = bool(export_result.get("runner_started"))
            if not runner_started:
                diagnosis = (
                    "Mimics did not execute the job runner. Check the dedicated Mimics application "
                    "log for license, startup, or single-instance redirection details."
                )
            elif not batch_status:
                diagnosis = (
                    "The runner started, but the export script stopped before writing its first status."
                )
            else:
                diagnosis = (
                    "The export script finished without producing the requested mask. Check whether "
                    "the selected .mcs projects contain a mask whose name matches the training organ."
                )
            error = (
                "Label export produced no files in {1}. {4} "
                "(returncode={0}; process log size={2} bytes; export status={3}). "
                "Diagnostics: {5}. Training was not started with missing or stale labels."
            ).format(
                export_result.get("returncode"),
                fresh_label_root,
                log_size,
                batch_status.get("status") or "missing",
                diagnosis,
                export_result.get("job_runtime") or export_result.get("log") or workspace,
            )
            update_status(status_path, {
                "status": "failed",
                "error": error,
                "samples_found": 0,
                "label_export": export_result,
            })
            return 2

    samples, skipped = discover_samples(
        ts_root,
        args.organ,
        cases,
        label_root=fresh_label_root,
        fallback_to_case_labels=not bool(fresh_label_root),
    )
    empty_labels = [s for s in skipped if s.get("reason") == "label is empty (all zeros)"]
    if empty_labels:
        append_log(
            workspace,
            "Skipped {} case(s) with empty labels for {}: {}".format(
                len(empty_labels), args.organ,
                ", ".join(s["case_id"] for s in empty_labels),
            ),
        )
    no_labels = [s for s in skipped if not s.get("has_label")]
    if no_labels:
        append_log(
            workspace,
            "{} case(s) have no {} label and will not be used.".format(
                len(no_labels), args.organ,
            ),
        )
    selected = select_samples(samples, args.sample_mode, int(args.max_samples or 0))
    if len(selected) < int(args.min_samples):
        if args.export_labels:
            error = (
                "not enough freshly exported labels for organ {0}; found {1}, required {2}. "
                "Check that the selected cases have saved .mcs files and that each .mcs contains a Mask named {0}. "
                "If you want to use existing NIfTI labels instead of refreshing from .mcs, disable label export before training."
            ).format(args.organ, len(selected), args.min_samples)
        else:
            error = "not enough labeled samples for organ {}; found {}, required {}".format(
                args.organ,
                len(selected),
                args.min_samples,
            )
        update_status(status_path, {
            "status": "failed",
            "error": error,
            "samples_found": len(samples),
            "skipped": skipped[:100],
        })
        return 2

    train_samples, val_samples = split_train_validation(
        selected,
        val_fraction=args.val_fraction,
        val_cases=parse_case_list(args.val_cases),
        min_train_samples=args.min_samples,
        min_val_samples=args.min_val_samples,
    )
    fresh_geometry_checked = validate_fresh_export_geometry(train_samples + val_samples)
    if not args.keep_materialized_dataset:
        _register_transient_cleanup(dataset_dir)
    materialized_train, materialized_val = materialize_dataset(train_samples, dataset_dir, val_samples)
    dataset_validation = validate_materialized_dataset(materialized_train + materialized_val)
    if fresh_label_root:
        fresh_label_cleanup = {
            "enabled": True,
            "path": str(fresh_label_root),
            "cleaned": False,
        }
        try:
            rmtree_with_retry(fresh_label_root)
            fresh_label_cleanup["cleaned"] = True
            fresh_label_cleanup["cleaned_at_epoch"] = time.time()
        except Exception as exc:
            fresh_label_cleanup["error"] = str(exc)
            append_log(workspace, "Could not clean fresh label staging {}: {}".format(fresh_label_root, exc))
    validation_enabled = bool(materialized_val)
    class_weight_cap = float(repo_config.get("default_class_weight_cap", 5.0))
    class_weights = compute_class_weights(materialized_train, max_weight=class_weight_cap)
    append_log(workspace, "Auto class weights for {}: {}".format(args.organ, class_weights))
    if str(dinov3_root) not in sys.path:
        sys.path.insert(0, str(dinov3_root))
    from src.research.fingerprint import build_training_fingerprint, derive_policy
    support_case_ids = [item["case_id"] for item in materialized_train]
    training_fingerprint = build_training_fingerprint(dataset_dir, support_case_ids, args.organ)
    derived_policy = derive_policy(training_fingerprint, gpu_memory_gb=12.0)
    strategy_overrides = compile_strategy(
        strategy_id, training_fingerprint, derived_policy, strategy_options,
    )
    write_training_config(
        config_path,
        base_config,
        dataset_dir,
        exp_name,
        args,
        status_path=train_status,
        cancel_path=cancel_path,
        metrics_history_path=metrics_history,
        validation_enabled=validation_enabled,
        class_weights=class_weights,
        strategy_overrides=strategy_overrides,
    )
    from src.utils.config import load_config as load_dinov3_config
    effective_config = load_dinov3_config(str(config_path))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    write_json_atomic(
        run_dir / "samples.json",
        {
            "samples": materialized_train + materialized_val,
            "train_samples": materialized_train,
            "validation_samples": materialized_val,
            "skipped": skipped,
            "dataset_validation": dataset_validation,
            "fresh_label_cleanup": fresh_label_cleanup,
            "fresh_source_geometry_checked_cases": fresh_geometry_checked,
            "training_fingerprint": training_fingerprint,
            "derived_policy": derived_policy,
            "strategy": strategy_overrides.get("strategy", {}),
            "selection": {
                "cases": sorted(cases) if cases else None,
                "sample_mode": args.sample_mode,
                "max_samples": int(args.max_samples or 0),
                "val_fraction": float(args.val_fraction or 0.0),
                "val_cases": parse_case_list(args.val_cases),
            },
        },
    )
    update_status(status_path, {
        "status": "waiting_for_gpu" if gpu_lock_enabled() else "training",
        "sample_count": len(materialized_train) + len(materialized_val),
        "train_sample_count": len(materialized_train),
        "validation_sample_count": len(materialized_val),
        "train_cases": [item["case_id"] for item in materialized_train],
        "validation_cases": [item["case_id"] for item in materialized_val],
        "dataset_dir": str(dataset_dir),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "strategy": strategy_overrides.get("strategy", {}),
        "train_log": str(train_log),
        "training_status": str(train_status),
        "metrics_history": str(metrics_history),
        "cancel_path": str(cancel_path),
        "dataset_validation": dataset_validation,
        "fresh_label_cleanup": fresh_label_cleanup,
        "fresh_source_geometry_checked_cases": fresh_geometry_checked,
    })

    cmd = [python_exe, str(dinov3_root / "scripts" / "train.py"), "--config", str(config_path)]
    append_log(workspace, "Launching DINOv3 training: {}".format(" ".join(cmd)))
    proc = None
    gpu_lock = None
    try:
        gpu_lock = acquire_gpu_lock_for_job(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 training {}".format(run_id),
            args.gpu_lock_timeout_seconds,
        )
        update_status(status_path, {"status": "training"})
        log_handle, actual_train_log, train_log_warning = open_subprocess_log(
            train_log,
            workspace,
            "DINOv3 training log",
        )
        with log_handle as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(dinov3_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=background_env(),
            )
            if gpu_lock is not None:
                gpu_lock.update_pid(proc.pid, kind="fewshot_train", job_id=run_id)
            status_payload = {"pid": proc.pid, "command": cmd}
            if actual_train_log is not None and actual_train_log != train_log:
                train_log = actual_train_log
            if actual_train_log is not None:
                status_payload["train_log"] = str(train_log)
            else:
                status_payload["train_log_unavailable"] = True
            if train_log_warning:
                status_payload["train_log_warning"] = train_log_warning
            update_status(status_path, status_payload)
            cancel_started = None
            last_progress_line = ""
            loop_status_state = {"signature": _status_signature(status_payload), "epoch": time.time()}
            while proc.poll() is None:
                progress = read_json(train_status, {}) or {}
                payload = {"status": "training", "pid": proc.pid}
                if progress:
                    payload["training_progress"] = progress
                    progress_line = training_progress_line(progress)
                    if progress_line and progress_line != last_progress_line:
                        append_log(workspace, "Training progress: {}".format(progress_line))
                        last_progress_line = progress_line
                if cancel_path.is_file():
                    payload["status"] = "cancelling"
                    if cancel_started is None:
                        cancel_started = time.time()
                    elif time.time() - cancel_started > 30:
                        terminate_process_tree(proc.pid)
                maybe_update_status(status_path, payload, loop_status_state, heartbeat_seconds=5.0)
                time.sleep(1.0)
    except ResourceLockCancelled:
        update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for GPU"})
        append_log(workspace, "Training job {} cancelled while waiting for GPU.".format(run_id))
        return 130
    except ResourceLockTimeout as exc:
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Training job {} could not acquire GPU: {}.".format(run_id, exc))
        return 75
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            terminate_process_tree(proc.pid)
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Training job {} failed before completion: {}.".format(run_id, exc))
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.release()

    final_progress = read_json(train_status, {}) or {}
    if cancel_path.is_file() or final_progress.get("status") == "cancelled":
        update_status(status_path, {
            "status": "cancelled",
            "returncode": proc.returncode,
            "training_progress": final_progress,
            "metrics_history": str(metrics_history),
        })
        append_log(workspace, "Training job {} cancelled.".format(run_id))
        return 130

    if proc.returncode != 0:
        error = "training process failed; see train_log"
        try:
            if actual_train_log is None:
                error = "training process failed; training log is unavailable"
        except Exception:
            pass
        update_status(status_path, {
            "status": "failed",
            "returncode": proc.returncode,
            "error": error,
            "metrics_history": str(metrics_history),
        })
        return proc.returncode or 1

    exp_dir = dinov3_root / "experiments" / exp_name
    if not bool(repo_config.get("keep_training_experiment_artifacts", False)):
        _register_transient_cleanup(exp_dir)
    ckpt = latest_epoch_checkpoint(exp_dir / "checkpoints")
    if not ckpt:
        update_status(status_path, {"status": "failed", "error": "no checkpoint was produced"})
        return 1

    model_dir = workspace / "models" / organ_slug / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_ckpt = model_dir / "model.pth"
    shutil.copy2(str(ckpt), str(registry_ckpt))
    shutil.copy2(str(config_path), str(model_dir / "config.yaml"))
    best_dsc_value = final_progress.get("best_dsc")
    manifest = {
        "schema_version": "mimics_fewshot_model.v2",
        "model_id": run_id,
        "organ": args.organ,
        "organ_slug": organ_slug,
        "best_dsc": best_dsc_value,
        "checkpoint": str(registry_ckpt),
        "config": str(model_dir / "config.yaml"),
        "config_sha256": config_sha256,
        "strategy": strategy_overrides.get("strategy", {}),
        "effective_config": effective_config,
        "spatial_convention": "canonical_ras_array_xyz__model_tensor_zyx",
        "source_checkpoint": str(registry_ckpt),
        "training_source_checkpoint": str(ckpt),
        "experiment_dir": str(exp_dir),
        "experiment_artifacts_retained": bool(repo_config.get("keep_training_experiment_artifacts", False)),
        "sample_count": len(materialized_train) + len(materialized_val),
        "train_sample_count": len(materialized_train),
        "validation_sample_count": len(materialized_val),
        "samples": materialized_train + materialized_val,
        "train_samples": materialized_train,
        "validation_samples": materialized_val,
        "dataset_dir": str(dataset_dir),
        "dataset_retained": bool(args.keep_materialized_dataset),
        "dataset_validation": dataset_validation,
        "fresh_label_cleanup": fresh_label_cleanup,
        "fresh_source_geometry_checked_cases": fresh_geometry_checked,
        "training_progress": final_progress,
        "metrics_history": str(metrics_history),
        "training_parameters": {
            "base_config": str(base_config),
            "strategy": strategy_id,
            "finetune_method": str(args.finetune_method),
            "decoder": str(args.decoder),
            "model_scale": str(args.model_scale),
            "model_path": str(args.model_path or ""),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accumulation": int(args.grad_accumulation),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "img_size": str(args.img_size),
            "modality": str(args.modality),
            "mixed_precision": bool(args.mixed_precision),
            "sub_volume": bool(args.sub_volume),
            "sub_volume_size": str(args.sub_volume_size),
            "keep_last_checkpoints": int(args.keep_last_checkpoints),
            "keep_materialized_dataset": bool(args.keep_materialized_dataset),
        },
        "created_at_epoch": time.time(),
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "dinov3_root": str(dinov3_root),
        "base_config": str(base_config),
    }
    write_json_atomic(model_dir / "manifest.json", manifest)
    write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
    try:
        register_global_model(manifest)
    except Exception as exc:
        manifest["global_registry_error"] = str(exc)
        write_json_atomic(model_dir / "manifest.json", manifest)
        write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
        append_log(workspace, "Could not update global DINOv3 model registry: {}.".format(exc))
    if not args.keep_materialized_dataset:
        try:
            rmtree_with_retry(dataset_dir)
            manifest["dataset_retained"] = False
            manifest["dataset_cleanup_at_epoch"] = time.time()
            write_json_atomic(model_dir / "manifest.json", manifest)
            write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
        except Exception as exc:
            manifest["dataset_retained"] = True
            manifest["dataset_cleanup_error"] = str(exc)
            write_json_atomic(model_dir / "manifest.json", manifest)
            write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
            append_log(workspace, "Could not clean materialized dataset {}: {}".format(dataset_dir, exc))
    update_status(status_path, {
        "status": "completed",
        "model": manifest,
        "returncode": 0,
        "training_progress": final_progress,
        "metrics_history": str(metrics_history),
    })
    best = final_progress.get("best_dsc")
    if best is not None:
        append_log(workspace, "Training job {} completed. Best DSC: {:.4f}. Model: {}".format(
            run_id, float(best), registry_ckpt))
    else:
        append_log(workspace, "Training job {} completed. Model: {}".format(run_id, registry_ckpt))
    return 0


def load_model_manifest(workspace, organ, model_id=None, model_manifest=None):
    manifest_source = model_manifest
    if model_manifest:
        manifest = read_json(model_manifest)
        if not manifest:
            raise RuntimeError("model manifest could not be read: {}".format(model_manifest))
        return validate_model_manifest(manifest, model_manifest)
    organ_slug = safe_slug(organ)
    if model_id and model_id != "latest":
        path = Path(workspace) / "models" / organ_slug / model_id / "manifest.json"
    else:
        path = Path(workspace) / "models" / organ_slug / "latest.json"
    manifest_source = path
    manifest = read_json(path)
    if not manifest:
        raise RuntimeError("no model manifest found for organ {} at {}".format(organ, path))
    return validate_model_manifest(manifest, manifest_source)


def validate_model_manifest(manifest, source=""):
    if not isinstance(manifest, dict):
        raise RuntimeError("model manifest is invalid: {}".format(source))
    checked = dict(manifest)
    if source:
        checked["_manifest_path"] = str(source)
    for key in ("checkpoint", "config"):
        value = manifest.get(key, "")
        if not value:
            raise RuntimeError("model manifest is missing '{}': {}".format(key, source))
        path = Path(value)
        if not path.is_file():
            raise RuntimeError("model manifest points to a missing {}: {} ({})".format(key, path, source))
        try:
            size = int(path.stat().st_size)
            checked[key + "_size_bytes"] = size
        except Exception:
            size = None
            checked[key + "_size_bytes"] = None
        if size is not None and size <= 0:
            raise RuntimeError("model manifest points to an empty {}: {} ({})".format(key, path, source))
    expected_config_sha256 = str(manifest.get("config_sha256") or "").strip().lower()
    if expected_config_sha256:
        actual_config_sha256 = hashlib.sha256(Path(manifest["config"]).read_bytes()).hexdigest()
        checked["config_sha256_verified"] = actual_config_sha256 == expected_config_sha256
        if actual_config_sha256 != expected_config_sha256:
            raise RuntimeError(
                "model configuration was changed after training; refusing inconsistent inference: {} ({})".format(
                    manifest["config"], source,
                )
            )
    checkpoint_path = Path(manifest.get("checkpoint", ""))
    try:
        with open(str(checkpoint_path), "rb") as handle:
            header = handle.read(4)
        if not (header.startswith(b"PK") or header.startswith(b"\x80")):
            raise RuntimeError(
                "model checkpoint does not look like a torch checkpoint: {} ({})".format(
                    checkpoint_path,
                    source,
                )
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("model checkpoint could not be inspected: {} ({})".format(exc, source))
    return checked


def _parse_json_shape(value):
    if not value:
        return None
    try:
        shape = json.loads(value)
        shape = [int(shape[0]), int(shape[1]), int(shape[2])]
    except Exception:
        raise RuntimeError("invalid expected source shape: {}".format(value))
    if not all(item > 0 for item in shape):
        raise RuntimeError("invalid expected source shape: {}".format(value))
    return shape


def _parse_json_matrix(value):
    if not value:
        return None
    try:
        matrix = json.loads(value)
        if len(matrix) != 4:
            raise ValueError("not 4x4")
        parsed = []
        for row in matrix:
            if len(row) != 4:
                raise ValueError("not 4x4")
            parsed.append([float(item) for item in row])
        return parsed
    except Exception:
        raise RuntimeError("invalid expected source voxel-to-RAS matrix")


def _normalize_path_text(path):
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return os.path.normcase(str(Path(text).resolve()))
    except Exception:
        return os.path.normcase(os.path.abspath(text))


def validate_inference_source_geometry(
    image_path,
    expected_shape=None,
    expected_affine=None,
    expected_image_path="",
):
    result = {
        "image_path": str(image_path),
        "checked": False,
    }
    if not expected_shape and not expected_affine and not expected_image_path:
        result["reason"] = "no expected source geometry was provided"
        return result
    import nibabel as nib
    import numpy as np
    from mimics_bridge import _affine_close

    img = nib.load(str(image_path))
    actual_shape = [int(value) for value in img.shape[:3]]
    result["checked"] = True
    result["actual_shape"] = actual_shape
    if expected_image_path:
        expected_image_path = str(expected_image_path).strip()
        result["expected_image_path"] = expected_image_path
        normalized_expected = _normalize_path_text(expected_image_path)
        normalized_actual = _normalize_path_text(image_path)
        result["actual_image_path"] = str(Path(image_path).resolve())
        result["expected_image_path_matched"] = bool(
            normalized_expected and normalized_actual and normalized_expected == normalized_actual
        )
        if normalized_expected and normalized_actual and normalized_expected != normalized_actual:
            raise RuntimeError(
                "source image path does not match the open Mimics project: {} != {}".format(
                    result["actual_image_path"],
                    expected_image_path,
                )
            )
    if expected_shape:
        result["expected_shape"] = [int(value) for value in expected_shape]
        if actual_shape != result["expected_shape"]:
            raise RuntimeError(
                "source image shape does not match the open Mimics project: {} != {}".format(
                    actual_shape,
                    result["expected_shape"],
                )
            )
    if expected_affine:
        expected_affine_np = np.asarray(expected_affine, dtype=float)
        result["expected_voxel_to_ras_matrix"] = expected_affine_np.tolist()
        result["actual_voxel_to_ras_matrix"] = np.asarray(img.affine, dtype=float).tolist()
        result["affine_max_abs_diff"] = float(np.max(np.abs(np.asarray(img.affine, dtype=float) - expected_affine_np)))
        if not _affine_close(img.affine, expected_affine_np):
            raise RuntimeError(
                "source image affine does not match the open Mimics project (max abs diff {:.6g}): {}".format(
                    result["affine_max_abs_diff"],
                    result.get("actual_image_path", str(Path(image_path).resolve())),
                )
            )
    return result


def cmd_infer(args):
    ts_root = Path(args.ts_root).resolve()
    workspace = workspace_for(ts_root, args.workspace)
    try:
        maintenance_root = dinov3_root_from_args(args)
    except Exception:
        maintenance_root = None
    maintenance = cleanup_workspace_artifacts(workspace, maintenance_root)
    if any(maintenance.values()):
        append_log(workspace, "Automatic storage maintenance removed disposable artifacts: {}.".format(maintenance))
    organ_slug = safe_slug(args.organ)
    job_id = args.job_id or "infer_{}_{}_{}".format(
        safe_slug(args.case_id),
        organ_slug,
        uuid.uuid4().hex[:8],
    )
    status_path = workspace / "jobs" / (job_id + ".json")
    output_dir = workspace / "predictions" / safe_slug(args.case_id) / organ_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    cancel_path = output_dir / (job_id + ".cancel")
    status_base = {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": job_id,
        "kind": "infer",
        "status": "preparing",
        "organ": args.organ,
        "case_id": args.case_id,
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "controller_pid": os.getpid(),
        "cancel_path": str(cancel_path),
        "created_at_epoch": time.time(),
    }
    write_json_atomic(status_path, status_base)
    case_dir = ts_root / args.case_id
    try:
        if not case_dir.is_dir():
            raise RuntimeError("case directory was not found: {}".format(case_dir))
        image = find_image(case_dir)
        if not image:
            raise RuntimeError("no NIfTI image found for case: {}".format(args.case_id))
        dinov3_root = dinov3_root_from_args(args)
        python_exe = python_from_args(args, dinov3_root)
        model = load_model_manifest(workspace, args.organ, args.model_id, args.model_manifest)
        expected_shape = _parse_json_shape(getattr(args, "expected_source_shape", ""))
        expected_affine = _parse_json_matrix(getattr(args, "expected_source_voxel_to_ras_matrix", ""))
        expected_image_path = str(getattr(args, "expected_source_image_path", "") or "").strip()
        source_validation = validate_inference_source_geometry(
            image,
            expected_shape=expected_shape,
            expected_affine=expected_affine,
            expected_image_path=expected_image_path,
        )
    except Exception as exc:
        update_status(status_path, {
            "status": "failed",
            "error": str(exc),
            "model_id": getattr(args, "model_id", "latest"),
            "model_manifest": getattr(args, "model_manifest", "") or "",
            "expected_source_shape": _parse_json_shape(getattr(args, "expected_source_shape", "")),
            "expected_source_voxel_to_ras_matrix": _parse_json_matrix(
                getattr(args, "expected_source_voxel_to_ras_matrix", "")
            ),
            "expected_source_image_path": str(getattr(args, "expected_source_image_path", "") or ""),
        })
        append_log(workspace, "Inference job {} failed during preflight: {}.".format(job_id, exc))
        return 1

    output_model_id = args.model_id or model.get("model_id", "latest")
    if args.model_manifest and output_model_id == "latest":
        output_model_id = model.get("model_id", "external_model")
    output_path = output_dir / (safe_slug(output_model_id) + ".nii.gz")
    log_path = output_dir / (job_id + ".log")

    status_running = dict(status_base)
    status_running.update({
        "status": "waiting_for_gpu" if gpu_lock_enabled() else "running",
        "image_path": str(image),
        "output_path": str(output_path),
        "model_id": model.get("model_id", args.model_id or "latest"),
        "model_manifest": model.get("_manifest_path", args.model_manifest or ""),
        "model": model,
        "source_validation": source_validation,
        "updated_at_epoch": time.time(),
    })
    write_json_atomic(status_path, status_running)
    cmd = [
        python_exe,
        str(dinov3_root / "scripts" / "infer.py"),
        "--config",
        str(model["config"]),
        "--checkpoint",
        str(model["checkpoint"]),
        "--input",
        str(image),
        "--output",
        str(output_path),
    ]
    append_log(workspace, "Launching DINOv3 inference: {}".format(" ".join(cmd)))
    proc = None
    gpu_lock = None
    try:
        gpu_lock = acquire_gpu_lock_for_job(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 inference {}".format(job_id),
            args.gpu_lock_timeout_seconds,
        )
        update_status(status_path, {"status": "running"})
        log_handle, actual_log_path, inference_log_warning = open_subprocess_log(
            log_path,
            workspace,
            "DINOv3 inference log",
        )
        with log_handle as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(dinov3_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=background_env(),
            )
            if gpu_lock is not None:
                gpu_lock.update_pid(proc.pid, kind="fewshot_infer", job_id=job_id)
            status_payload = {"pid": proc.pid, "command": cmd}
            if actual_log_path is not None and actual_log_path != log_path:
                log_path = actual_log_path
            if actual_log_path is not None:
                status_payload["log"] = str(log_path)
            else:
                status_payload["log_unavailable"] = True
            if inference_log_warning:
                status_payload["log_warning"] = inference_log_warning
            update_status(status_path, status_payload)
            loop_status_state = {"signature": _status_signature(status_payload), "epoch": time.time()}
            while proc.poll() is None:
                if cancel_path.is_file():
                    maybe_update_status(
                        status_path,
                        {"status": "cancelling", "pid": proc.pid},
                        loop_status_state,
                        heartbeat_seconds=5.0,
                    )
                    terminate_process_tree(proc.pid)
                else:
                    maybe_update_status(
                        status_path,
                        {"status": "running", "pid": proc.pid},
                        loop_status_state,
                        heartbeat_seconds=5.0,
                    )
                time.sleep(2.0)
    except ResourceLockCancelled:
        update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for GPU"})
        append_log(workspace, "Inference job {} cancelled while waiting for GPU.".format(job_id))
        return 130
    except ResourceLockTimeout as exc:
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Inference job {} could not acquire GPU: {}.".format(job_id, exc))
        return 75
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            terminate_process_tree(proc.pid)
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Inference job {} failed before completion: {}.".format(job_id, exc))
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.release()
    if cancel_path.is_file():
        update_status(status_path, {
            "status": "cancelled",
            "returncode": proc.returncode,
        })
        append_log(workspace, "Inference job {} cancelled.".format(job_id))
        return 130
    if proc.returncode != 0:
        error = "inference process failed; see log"
        try:
            if actual_log_path is None:
                error = "inference process failed; inference log is unavailable"
        except Exception:
            pass
        update_status(status_path, {
            "status": "failed",
            "returncode": proc.returncode,
            "error": error,
        })
        return proc.returncode or 1
    update_status(status_path, {
        "status": "completed",
        "returncode": 0,
        "output_path": str(output_path),
        "image_path": str(image),
    })
    append_log(workspace, "Inference job {} completed. Output: {}".format(job_id, output_path))
    return 0


def cmd_list_models(args):
    workspace = workspace_for(Path(args.ts_root).resolve(), args.workspace)
    root = workspace / "models"
    rows = []
    if root.is_dir():
        if args.all:
            manifests = sorted(root.glob("*/*/manifest.json"))
        else:
            manifests = sorted(root.glob("*/latest.json"))
        for manifest_path in manifests:
            manifest = read_json(manifest_path, {})
            organ = manifest.get("organ", manifest_path.parent.name)
            if args.organ and safe_slug(organ) != safe_slug(args.organ):
                continue
            rows.append({
                "scope": "dataset",
                "organ": organ,
                "model_id": manifest.get("model_id", ""),
                "best_dsc": manifest.get("best_dsc"),
                "sample_count": manifest.get("sample_count", 0),
                "train_sample_count": manifest.get("train_sample_count", 0),
                "validation_sample_count": manifest.get("validation_sample_count", 0),
                "checkpoint": manifest.get("checkpoint", ""),
                "manifest_path": str(manifest_path),
                "created_at_epoch": manifest.get("created_at_epoch", 0.0),
            })
    if args.include_global:
        for row in global_model_rows(args.organ):
            global_row = dict(row)
            global_row["scope"] = "global"
            rows.append(global_row)
    rows.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    print(json.dumps({"models": rows}, indent=2, sort_keys=True))
    return 0


def cmd_discover(args):
    cases = set(parse_case_list(args.cases) or [])
    samples, skipped = discover_samples(Path(args.ts_root).resolve(), args.organ, cases or None)
    selected = select_samples(samples, args.sample_mode, int(args.max_samples or 0))
    print(json.dumps({
        "organ": args.organ,
        "sample_count": len(samples),
        "selected_count": len(selected),
        "samples": selected,
        "skipped": skipped,
    }, indent=2, sort_keys=True))
    return 0


def cmd_cancel(args):
    workspace = workspace_for(Path(args.ts_root).resolve(), args.workspace)
    if args.job_id:
        status_path = workspace / "jobs" / (args.job_id + ".json")
        status = read_json(status_path, {}) or {}
        if not status:
            print("No job was found: {}".format(args.job_id), file=sys.stderr)
            return 2
    else:
        status_path, status = latest_running_job(workspace)
        if not status:
            print("No running few-shot job was found.")
            return 0
    cancel_path = status.get("cancel_path")
    if cancel_path:
        cancel_error = write_cancel_marker(cancel_path)
        if cancel_error:
            print("Warning: could not write cancel marker {}: {}".format(cancel_path, cancel_error), file=sys.stderr)
            update_status(status_path, {"cancel_marker_error": cancel_error})
    grace_seconds = max(0.0, float(getattr(args, "grace_seconds", 30.0)))
    pids = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        pid = status.get(key)
        try:
            pid = int(pid)
        except Exception:
            pid = 0
        if pid > 0 and pid not in pids:
            pids.append(pid)
    deadline = time.time() + grace_seconds
    while pids and time.time() < deadline:
        if all(not process_exists(pid) for pid in pids):
            break
        update_status(status_path, {
            "status": "cancelling",
            "cancel_requested_at_epoch": time.time(),
            "grace_seconds": grace_seconds,
        })
        time.sleep(0.5)
    killed = []
    for pid in pids:
        if process_exists(pid) and terminate_process_tree(pid):
            killed.append(int(pid))
    update_status(status_path, {
        "status": "cancelled",
        "cancel_requested_at_epoch": time.time(),
        "cancelled_pids": killed,
    })
    print("Cancel request submitted for job {}.".format(status.get("job_id", status_path.stem)))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Export labels, build a few-shot dataset, and train a DINOv3 model")
    train.add_argument("--ts-root", required=True)
    train.add_argument("--organ", required=True)
    train.add_argument("--workspace")
    train.add_argument("--dinov3-root")
    train.add_argument("--python")
    train.add_argument("--base-config")
    train.add_argument("--strategy", choices=tuple(strategy_ids()), default="adaptive")
    train.add_argument("--strategy-options-json", default="{}")
    train.add_argument("--cases")
    train.add_argument("--sample-mode", choices=("all", "latest"), default="all")
    train.add_argument("--min-samples", type=int, default=1)
    train.add_argument("--max-samples", type=int, default=0)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--img-size", default="224,224")
    train.add_argument("--modality", default="ct")
    train.add_argument("--val-fraction", type=float, default=0.0)
    train.add_argument("--val-cases")
    train.add_argument("--min-val-samples", type=int, default=1)
    train.add_argument("--finetune-method", choices=("frozen", "decoder_only", "decode_only", "lora", "adapter", "full"), default="lora")
    train.add_argument("--decoder", choices=(
        "linear3d", "mlp_probe", "segformer3d", "token_pyramid3d", "dpt3d",
        "conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d",
    ), default="segformer3d")
    train.add_argument("--lr-scheduler", choices=("constant", "constant_warmup", "cosine"), default="cosine")
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--model-scale", choices=("vitb16", "vitl16", "vith16plus"), default="vitb16")
    train.add_argument("--model-path")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--adapter-bottleneck", type=int, default=64)
    train.add_argument("--mixed-precision", action="store_true")
    train.add_argument("--sub-volume", action="store_true")
    train.add_argument("--sub-volume-size", default="32,256,256")
    train.add_argument("--export-labels", action="store_true")
    train.add_argument("--mimics-exe")
    train.add_argument("--mcs-output-dir", help="Folder containing saved .mcs projects for this training run")
    train.add_argument("--export-timeout-seconds", type=float, default=1800)
    train.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=1800)
    train.add_argument("--gpu-lock-timeout-seconds", type=float, default=86400)
    train.add_argument("--keep-last-checkpoints", type=int, default=2)
    train.add_argument("--keep-materialized-dataset", action="store_true")
    train.add_argument("--run-id")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run inference for one case with the latest or selected organ model")
    infer.add_argument("--ts-root", required=True)
    infer.add_argument("--case-id", required=True)
    infer.add_argument("--organ", required=True)
    infer.add_argument("--workspace")
    infer.add_argument("--dinov3-root")
    infer.add_argument("--python")
    infer.add_argument("--model-id", default="latest")
    infer.add_argument("--model-manifest")
    infer.add_argument("--expected-source-shape")
    infer.add_argument("--expected-source-voxel-to-ras-matrix")
    infer.add_argument("--expected-source-image-path")
    infer.add_argument("--gpu-lock-timeout-seconds", type=float, default=3600)
    infer.add_argument("--job-id")
    infer.set_defaults(func=cmd_infer)

    models = sub.add_parser("list-models", help="List latest registered models")
    models.add_argument("--ts-root", required=True)
    models.add_argument("--workspace")
    models.add_argument("--organ")
    models.add_argument("--all", action="store_true")
    models.add_argument("--include-global", action="store_true")
    models.set_defaults(func=cmd_list_models)

    discover = sub.add_parser("discover", help="Discover exported image/label pairs for one organ")
    discover.add_argument("--ts-root", required=True)
    discover.add_argument("--organ", required=True)
    discover.add_argument("--cases")
    discover.add_argument("--sample-mode", choices=("all", "latest"), default="all")
    discover.add_argument("--max-samples", type=int, default=0)
    discover.set_defaults(func=cmd_discover)

    cancel = sub.add_parser("cancel", help="Cancel the latest running job or a selected job")
    cancel.add_argument("--ts-root", required=True)
    cancel.add_argument("--workspace")
    cancel.add_argument("--job-id")
    cancel.add_argument("--grace-seconds", type=float, default=30.0)
    cancel.set_defaults(func=cmd_cancel)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except Exception as exc:
        ts_root = getattr(args, "ts_root", None)
        if ts_root:
            try:
                append_log(workspace_for(ts_root, getattr(args, "workspace", None)), "Failed: {}".format(exc))
            except Exception:
                pass
        print("Error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
