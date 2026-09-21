#!/usr/bin/env python3
"""External DINOv3 few-shot pipeline for Mimics-Script.

This script never imports Mimics. It is safe to run from the foreground Mimics
process through subprocess.Popen because all long-running work happens here or
in child Python/Mimics background processes.
"""

import argparse
import atexit
import copy
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
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import runtime_common
import dataset_manifest

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
from tools.fewshot_strategies import compile_strategy, normalize_strategy_options, strategy_ids
from tools.fewshot_architecture import (
    SUPPORTED_DECODERS,
    inspect_dinov3_vit_weights,
    recommended_batch_size,
    resolve_architecture,
    seal_architecture_plan,
)

DEFAULT_WORKSPACE = "fewshot_models"
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3
RESOURCE_LOCK_DIR = default_resource_lock_dir(ROOT)
GPU_LOCK_PATH = RESOURCE_LOCK_DIR / "gpu.lock"


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
    return pipeline_common.write_cancel_marker(cancel_path)


TEMP_LOG_RETENTION_DAYS = 30


def prune_temp_log_root(retention_days=TEMP_LOG_RETENTION_DAYS):
    """Delete fallback subprocess logs past the retention window (best effort).

    The %TEMP% fallback root is only used when the primary log is locked, but
    files written there live outside the project tree and would otherwise
    accumulate forever.
    """
    temp_root = Path(tempfile.gettempdir()) / "mimics_script_fewshot_logs"
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


def resolve_manifest_artifact(value, manifest_path, manifest=None):
    """Resolve portable relative artifacts and relocate legacy manifests."""
    text = str(value or "").strip()
    if not text:
        return Path()
    source = Path(manifest_path).expanduser().resolve() if manifest_path else None
    path = Path(text).expanduser()
    foreign_absolute = (
        len(text) >= 3
        and text[1] == ":"
        and text[2] in ("\\", "/")
    ) or text.startswith("\\\\")
    if not path.is_absolute() and not foreign_absolute:
        return ((source.parent if source else Path.cwd()) / path).resolve()
    if path.exists() or source is None:
        return path.resolve()
    # Old manifests stored an absolute model path. After copying the model
    # directory, the artifact still has the same basename beside manifest.json.
    relocated = source.parent / path.name
    if relocated.exists():
        return relocated.resolve()
    model_id = safe_slug((manifest or {}).get("model_id", ""))
    versioned = source.parent / model_id / path.name
    return versioned.resolve() if versioned.exists() else path.resolve()


def resolved_manifest_payload(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    payload = read_json(path, {}) or {}
    if not isinstance(payload, dict):
        return {}
    resolved = dict(payload)
    for key in ("checkpoint", "config"):
        if payload.get(key):
            resolved[key] = str(
                resolve_manifest_artifact(payload[key], path, payload)
            )
    resolved["_manifest_path"] = str(path)
    return resolved


def parse_case_list(value):
    if not value:
        return None
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def load_repo_config():
    path = ROOT / "fewshot_config.json"
    if path.is_file():
        return read_json(path, {})
    return {}


def resolve_training_mask_names(config, organ, value=None):
    requested = parse_case_list(value) or []
    if not requested:
        requested = [str(organ or "").strip()]
        aliases = (config or {}).get("organ_mask_aliases") or {}
        if isinstance(aliases, dict):
            organ_key = safe_slug(organ)
            for key, names in aliases.items():
                if safe_slug(key) == organ_key:
                    requested.extend(parse_case_list(names) or [])
    result = []
    seen = set()
    for name in requested:
        text = str(name or "").strip()
        key = safe_slug(text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "abandoned"}
ACTIVE_JOB_STATUSES = {
    "launching", "preparing", "exporting_labels", "waiting_for_background_mimics",
    "waiting_for_gpu", "training", "running", "cancelling", "stopping", "finalizing",
    "preparing_remote", "connecting_remote", "uploading", "starting_remote",
    "reconnecting_remote", "downloading",
    "waiting_for_remote_gpu",
    "remote_control_unavailable",
    "finalizing_remote",
}
CANCELLABLE_JOB_STATUSES = ACTIVE_JOB_STATUSES - {"cancelling", "stopping", "finalizing"}


def job_process_ids(payload):
    result = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        try:
            pid = int((payload or {}).get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in result:
            result.append(pid)
    return result


def job_has_live_process(payload):
    return any(process_exists(pid) for pid in job_process_ids(payload))


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
            pids = job_process_ids(payload)
            if status in ACTIVE_JOB_STATUSES and pids and not job_has_live_process(payload):
                cancelled = bool(payload.get("cancel_path") and Path(payload["cancel_path"]).is_file())
                status = "cancelled" if cancelled else "failed"
                payload.update({
                    "status": status,
                    "error": (
                        "task processes stopped after cancellation was requested"
                        if cancelled
                        else "background controller exited before recording a terminal state"
                    ),
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
            experiment = Path(
                payload.get("experiment_dir")
                or (
                    Path(dinov3_root)
                    / "experiments"
                    / "mimics_fewshot_{}_{}".format(
                        safe_slug(payload.get("organ", "")),
                        job_id,
                    )
                )
            )
            if _remove_tree_quietly(experiment):
                report["experiments"] += 1
    return report


def cleanup_terminal_training_artifacts(args, status_path):
    """Immediately bound failed/cancelled training storage without losing logs."""
    status_path = Path(status_path)
    payload = read_json(status_path, {}) or {}
    status = str(payload.get("status") or "").lower()
    if status not in ("failed", "cancelled"):
        return {}
    config = load_repo_config()
    if bool(config.get("keep_failed_training_artifacts", False)):
        report = {
            "skipped": True,
            "reason": "keep_failed_training_artifacts is enabled",
            "completed_at_epoch": time.time(),
        }
        update_status(status_path, {"artifact_cleanup": report})
        return report

    workspace = Path(payload.get("workspace") or status_path.parent.parent).resolve()
    run_id = str(payload.get("job_id") or getattr(args, "run_id", "") or "")
    organ_slug = safe_slug(payload.get("organ") or getattr(args, "organ", ""))
    run_dir = workspace / "runs" / organ_slug / run_id
    candidates = [
        Path(payload.get("dataset_dir") or (workspace / "datasets" / organ_slug / run_id)),
        run_dir / "fresh_labels",
    ]
    model_dir = workspace / "models" / organ_slug / run_id
    if not (model_dir / "manifest.json").is_file():
        candidates.append(model_dir)

    experiment_value = str(payload.get("experiment_dir") or "").strip()
    if experiment_value:
        candidates.append(Path(experiment_value))
    else:
        try:
            dinov3_root = dinov3_root_from_args(args)
            candidates.append(
                dinov3_root
                / "experiments"
                / "mimics_fewshot_{}_{}".format(organ_slug, run_id)
            )
        except Exception:
            pass

    report = {"removed": [], "not_removed": []}
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved in (workspace, run_dir, run_dir.parent) or not resolved.exists():
            continue
        safe = False
        try:
            resolved.relative_to(workspace)
            safe = True
        except ValueError:
            safe = bool(
                run_id
                and run_id in resolved.name
                and resolved.name.startswith("mimics_fewshot_")
            )
        if not safe:
            report["not_removed"].append(
                {"path": str(resolved), "reason": "outside the job workspace"}
            )
            continue
        try:
            if resolved.is_dir():
                rmtree_with_retry(resolved)
            else:
                resolved.unlink()
        except Exception as exc:
            report["not_removed"].append(
                {"path": str(resolved), "reason": str(exc)}
            )
        else:
            report["removed"].append(str(resolved))
    report["completed_at_epoch"] = time.time()
    update_status(status_path, {"artifact_cleanup": report})
    append_log(
        workspace,
        (
            "Failed-job storage cleanup removed {} rebuildable path(s); "
            "{} path(s) could not be removed. Logs and status were retained."
        ).format(len(report["removed"]), len(report["not_removed"])),
    )
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
        path = (ROOT / "integrations" / "dinov3-medical-seg").resolve()
    if not (path / "scripts" / "train.py").is_file():
        raise RuntimeError("DINOv3 project was not found: {}".format(path))
    return path


def project_python_candidates():
    return [
        ROOT / "python_env" / "python.exe",
        ROOT / "python_env" / "Scripts" / "python.exe",
        ROOT / "python_env" / "python" / "python.exe",
        ROOT / "python_env" / "bin" / "python3",
        ROOT / "python_env" / "bin" / "python",
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
    # Explicit --python always wins; otherwise use the canonical discovery
    # (env overrides + python_env/nninteractive_env layouts) before falling
    # back to the fewshot-specific extras (repo config, DINOv3 venvs, and
    # the running interpreter when it already lives inside an env).
    if args.python:
        candidates = []
        append_python_candidate(candidates, args.python)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    found = runtime_common.find_external_python(str(ROOT))
    if found:
        return found
    candidates = []
    candidates.extend(project_python_candidates())
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
        env_roots = [(ROOT / name).resolve() for name in ("python_env", "nninteractive_env")]
        if any(str(current_resolved).startswith(str(env_root)) for env_root in env_roots):
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
        current.get("path") or GPU_LOCK_PATH,
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
    try:
        update_status(status_path, {"resource_wait": None})
        append_log(workspace, "GPU resource acquired for {}.".format(owner))
    except Exception:
        lock.release()
        raise
    return lock


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
        should_cancel=lambda: Path(cancel_path).is_file(),
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
                if Path(cancel_path).is_file():
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


def latest_running_job(workspace):
    jobs_dir = Path(workspace) / "jobs"
    if not jobs_dir.is_dir():
        return None, None
    rows = []
    for path in jobs_dir.glob("*.json"):
        payload = read_json(path, {}) or {}
        if payload.get("status") not in ACTIVE_JOB_STATUSES:
            continue
        pids = job_process_ids(payload)
        if pids and not job_has_live_process(payload):
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


_IMAGE_SUFFIXES = (
    ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd.gz", ".nrrd",
)
_LABEL_SUFFIXES = (
    ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd.gz", ".nrrd",
)


def _has_suffix(path, suffixes):
    return any(str(path.name).lower().endswith(suffix) for suffix in suffixes)


def find_image(case_dir, profile_id=None):
    """Find the source image in a case directory.

    Preferred names come from the dataset profile (dataset_profiles.json);
    the few-shot pipeline historically also accepted mr./image. variants,
    which are kept as built-in extras so behaviour is unchanged.
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


def _medical_stem(path):
    lower = path.name.lower()
    for suffix in _LABEL_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[:-len(suffix)]
    return path.stem


def _find_label_in_seg_dir(seg_dir, mask_names):
    seg_dir = Path(seg_dir)
    if not seg_dir.is_dir():
        return None
    wanted = set(
        safe_slug(name)
        for name in (mask_names or [])
        if str(name or "").strip()
    )
    matches = []
    for path in sorted(seg_dir.iterdir()):
        if not path.is_file() or not _has_suffix(path, _LABEL_SUFFIXES):
            continue
        if safe_slug(_medical_stem(path)) in wanted:
            matches.append(path)
    matches = sorted(
        dict(
            (os.path.normcase(os.path.abspath(str(path))), path)
            for path in matches
        ).values()
    )
    if len(matches) > 1:
        raise RuntimeError(
            "More than one label matched the target aliases in {}: {}. "
            "Keep one label or narrow Target Mask names.".format(
                seg_dir, ", ".join(path.name for path in matches)
            )
        )
    return matches[0] if matches else None


def find_label(
    case_dir,
    organ,
    label_root=None,
    fallback_to_case_labels=True,
    mask_names=None,
):
    case_dir = Path(case_dir)
    accepted_names = list(mask_names or [organ])
    if organ and safe_slug(organ) not in set(safe_slug(v) for v in accepted_names):
        accepted_names.insert(0, organ)
    if label_root:
        manifest_file = Path(label_root)
        if manifest_file.name.lower() != dataset_manifest.MANIFEST_FILENAME:
            manifest_file = manifest_file / dataset_manifest.MANIFEST_FILENAME
        if manifest_file.is_file():
            payload = dataset_manifest.load_manifest(manifest_file)
            case_row = dataset_manifest.find_case(payload, case_dir.name) or {}
            resolved = dataset_manifest.resolve_case_label(
                manifest_file, case_row, accepted_names
            )
            if resolved:
                return Path(resolved[1])
    search_dirs = []
    if label_root:
        root = Path(label_root)
        if root.name.lower() == dataset_manifest.MANIFEST_FILENAME:
            root = root.parent
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
        label = _find_label_in_seg_dir(seg_dir, accepted_names)
        if label:
            return label
    return None


def _label_skip_reason(label_path):
    """Return a skip reason for unusable labels, otherwise an empty string."""
    try:
        import numpy as np
        from mimics_bridge import read_mask_labels_with_affine
        data, _affine, _labels = read_mask_labels_with_affine(str(label_path))
        if len(data.shape) < 3 or any(int(value) <= 0 for value in data.shape[:3]):
            return "label is not a valid 3D medical image"
        if not np.any(data):
            return "label is empty (all zeros)"
        return ""
    except Exception as exc:
        return "label could not be read: {}".format(exc)


def discover_samples(
    ts_root,
    organ,
    cases=None,
    label_root=None,
    fallback_to_case_labels=True,
    label_source=None,
    mask_names=None,
):
    samples = []
    skipped = []
    for case_dir in case_dirs(ts_root, cases):
        image = find_image(case_dir)
        label = find_label(
            case_dir,
            organ,
            label_root=label_root,
            fallback_to_case_labels=fallback_to_case_labels,
            mask_names=mask_names,
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
                "label_source": (
                    str(label_source)
                    if label_source
                    else ("fresh_export" if label_root else "case_segmentations")
                ),
                "label_mtime": float(label.stat().st_mtime),
            })
        else:
            skipped.append({
                "case_id": case_dir.name,
                "has_image": bool(image),
                "has_label": bool(label),
            })
    return samples, skipped


def _normalized_modality(value):
    text = str(value or "").strip().lower()
    if text in ("ct", "computed_tomography"):
        return "ct"
    if text in ("mr", "mri", "magnetic_resonance"):
        return "mr"
    if text in ("other", "generic", "unknown"):
        return "other"
    return ""


def _manifest_modality(case_id, roots):
    for root in roots or []:
        if not root:
            continue
        manifest_file = Path(root)
        if manifest_file.name.lower() != dataset_manifest.MANIFEST_FILENAME:
            manifest_file = manifest_file / dataset_manifest.MANIFEST_FILENAME
        if not manifest_file.is_file():
            continue
        payload = dataset_manifest.load_manifest(str(manifest_file))
        row = dataset_manifest.find_case(payload, case_id) or {}
        provenance = row.get("provenance") or {}
        value = _normalized_modality(
            provenance.get("source_modality")
            or row.get("source_modality")
            or row.get("modality")
        )
        if value:
            return value, "dataset manifest"
    return "", ""


def _source_image_modality(image_path):
    path = Path(image_path)
    if path.is_dir():
        try:
            from mimics_bridge import infer_dicom_modality

            value = _normalized_modality(infer_dicom_modality(str(path)))
            if value:
                return value, "DICOM metadata"
        except Exception:
            pass
    name = path.name.lower()
    for suffix in _IMAGE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name in ("ct", "computed_tomography") or name.startswith("ct_"):
        return "ct", "standard image filename"
    if name in ("mr", "mri") or name.startswith(("mr_", "mri_")):
        return "mr", "standard image filename"
    return "", ""


def resolve_training_modality(requested, samples, manifest_roots=None):
    """Resolve one label-free intensity policy for the complete training run."""
    requested_value = str(requested or "auto").strip().lower()
    if requested_value != "auto":
        explicit = _normalized_modality(requested_value)
        if not explicit:
            raise RuntimeError(
                "Image modality must be Auto, CT, MRI, or Other."
            )
        for sample in samples or []:
            sample["source_modality"] = explicit
            sample["modality_source"] = "user selection"
        return {
            "requested": requested_value,
            "resolved": explicit,
            "source": "user selection",
            "unresolved_cases": [],
        }

    detected = {}
    unresolved = []
    for sample in samples or []:
        case_id = str(sample.get("case_id") or "")
        value, source = _manifest_modality(case_id, manifest_roots)
        if not value:
            value, source = _source_image_modality(sample.get("image"))
        if value:
            detected.setdefault(value, []).append(case_id)
            sample["source_modality"] = value
            sample["modality_source"] = source
        else:
            unresolved.append(case_id)
    if len(detected) > 1:
        details = "; ".join(
            "{}: {}".format(key.upper(), ", ".join(values[:8]))
            for key, values in sorted(detected.items())
        )
        raise RuntimeError(
            "Automatic modality detection found mixed CT and MR data ({}). "
            "Train them separately or choose the intended modality explicitly.".format(
                details
            )
        )
    if detected and unresolved:
        raise RuntimeError(
            "Automatic modality detection could not classify {} of {} selected "
            "case(s) (for example: {}). Choose CT, MRI, or Other explicitly so "
            "one intensity policy is not silently applied to an unknown "
            "modality.".format(
                len(unresolved),
                len(samples or []),
                ", ".join(unresolved[:8]),
            )
        )
    resolved = next(iter(detected), "other")
    for sample in samples or []:
        sample.setdefault("source_modality", resolved)
        sample.setdefault(
            "modality_source",
            "robust image percentiles" if resolved == "other" else "run consensus",
        )
    return {
        "requested": "auto",
        "resolved": resolved,
        "source": (
            "dataset metadata"
            if detected
            else "no reliable metadata; robust image percentiles"
        ),
        "unresolved_cases": unresolved,
    }


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


def _materialize_source_image(image_src, image_dst):
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


def _materialize_label_on_source_grid(label_src, image_dst, label_dst):
    """Keep the source image fixed and map the binary label onto that grid."""
    import nibabel as nib
    import numpy as np
    from mimics_bridge import (
        _affine_close,
        _validate_resampled_mask_foreground,
        read_nifti_mask_with_affine,
        resample_mask_to_image_grid,
    )

    image_img = nib.load(str(image_dst))
    label_array, label_affine = read_nifti_mask_with_affine(str(label_src))
    image_shape = tuple(int(value) for value in image_img.shape[:3])
    image_affine = image_img.affine
    geometry_matched = (
        image_shape == tuple(int(value) for value in label_array.shape[:3])
        and _affine_close(image_affine, label_affine)
    )
    if geometry_matched:
        target_label = np.asarray(label_array, dtype=np.uint8)
        method = "copied_on_source_grid"
    else:
        target_label = resample_mask_to_image_grid(
            label_array,
            label_affine,
            image_shape,
            image_affine,
        )
        method = "label_resampled_to_source_image_grid"
    source_foreground, target_foreground = _validate_resampled_mask_foreground(
        label_array,
        target_label,
        "Preparing DINOv3 label '{}'".format(label_src),
    )
    _write_nifti(target_label.astype(np.uint8), image_affine, label_dst)
    return method, geometry_matched, source_foreground, target_foreground



def validate_fresh_export_geometry(samples):
    """Fail closed when a fresh Mimics export is not on its source-image grid."""
    from mimics_bridge import _affine_close
    from mimics_bridge import get_source_image_geometry, read_nifti_mask_with_affine

    checked = []
    for sample in samples:
        if str(sample.get("label_source") or "") not in (
            "fresh_export",
            "fresh_mcs_export",
        ):
            continue
        image = get_source_image_geometry(str(sample["image"]))
        if not image:
            raise RuntimeError(
                "source image geometry could not be read for case {}".format(
                    sample.get("case_id", "?")
                )
            )
        label_array, label_affine = read_nifti_mask_with_affine(
            str(sample["label"])
        )
        image_shape = tuple(int(value) for value in image["shape"])
        label_shape = tuple(int(value) for value in label_array.shape[:3])
        if image_shape != label_shape or not _affine_close(
            image["affine"], label_affine
        ):
            raise RuntimeError(
                "fresh Mimics label export is not aligned to the source image for case {0}; "
                "image shape {1}, label shape {2}. Training was stopped instead of silently "
                "resampling the image onto a different grid.".format(
                    sample.get("case_id", "?"), image_shape, label_shape,
                )
            )
        checked.append(sample.get("case_id"))
    return checked


def _materialization_fingerprint(sample):
    payload = {
        "image": _path_stat_signature(sample["image"]),
        "label": _path_stat_signature(sample["label"]),
        "contract": "dinov3_source_grid_materialization.v2",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _materialized_cache_case(sample, cache_dir):
    """Return a source-grid cache entry, rebuilding only the changed case."""
    case_id = safe_slug(sample["case_id"])
    cache_case = Path(cache_dir) / case_id
    fingerprint = _materialization_fingerprint(sample)
    metadata_path = cache_case / "metadata.json"
    image_path = cache_case / "image.nii.gz"
    label_path = cache_case / "label.nii.gz"
    metadata = read_json(metadata_path, {}) or {}
    if (
        metadata.get("fingerprint") == fingerprint
        and image_path.is_file()
        and label_path.is_file()
    ):
        return {
            "cache_hit": True,
            "image": image_path,
            "label": label_path,
            "metadata": metadata,
        }

    cache_case.parent.mkdir(parents=True, exist_ok=True)
    staging = cache_case.with_name(
        "{}.publishing_{}".format(cache_case.name, uuid.uuid4().hex)
    )
    try:
        staging.mkdir(parents=True)
        image_method = _materialize_source_image(
            sample["image"], staging / "image.nii.gz"
        )
        (
            label_method,
            geometry_matched,
            source_foreground,
            target_foreground,
        ) = _materialize_label_on_source_grid(
            sample["label"],
            staging / "image.nii.gz",
            staging / "label.nii.gz",
        )
        metadata = {
            "schema_version": "dinov3_materialized_case_cache.v1",
            "case_id": str(sample["case_id"]),
            "fingerprint": fingerprint,
            "image_materialization": image_method,
            "label_materialization": label_method,
            "image_label_geometry_matched": bool(geometry_matched),
            "source_label_foreground_voxels": source_foreground,
            "dataset_label_foreground_voxels": target_foreground,
            "updated_at_epoch": time.time(),
        }
        write_json_atomic(staging / "metadata.json", metadata)
        if cache_case.exists():
            rmtree_with_retry(cache_case)
        last_error = None
        for attempt in range(20):
            try:
                os.replace(str(staging), str(cache_case))
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25, 0.02 * (attempt + 1)))
        if last_error is not None:
            raise OSError(
                "Could not publish materialized cache for {}: {}".format(
                    sample["case_id"], last_error
                )
            )
    finally:
        if staging.exists():
            try:
                rmtree_with_retry(staging)
            except OSError:
                pass
    return {
        "cache_hit": False,
        "image": cache_case / "image.nii.gz",
        "label": cache_case / "label.nii.gz",
        "metadata": metadata,
    }


def _materialize_split(
    samples, image_dir, label_dir, split_name, cache_dir=None
):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for sample in samples:
        case_id = safe_slug(sample["case_id"])
        image_src = Path(sample["image"])
        label_src = Path(sample["label"])
        image_dst = image_dir / (case_id + ".nii.gz")
        label_dst = label_dir / (case_id + ".nii.gz")
        cache_entry = None
        cache_error = ""
        if cache_dir:
            try:
                cache_entry = _materialized_cache_case(sample, cache_dir)
            except Exception as exc:
                cache_error = str(exc)
        if cache_entry:
            image_method = copy_or_link(cache_entry["image"], image_dst)
            label_method = copy_or_link(cache_entry["label"], label_dst)
            metadata = cache_entry["metadata"]
            geometry_matched = bool(
                metadata.get("image_label_geometry_matched", True)
            )
            source_foreground = int(
                metadata.get("source_label_foreground_voxels", 0) or 0
            )
            target_foreground = int(
                metadata.get("dataset_label_foreground_voxels", 0) or 0
            )
        else:
            image_method = _materialize_source_image(image_src, image_dst)
            (
                label_method,
                geometry_matched,
                source_foreground,
                target_foreground,
            ) = _materialize_label_on_source_grid(
                label_src, image_dst, label_dst
            )
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
            "source_label_foreground_voxels": source_foreground,
            "dataset_label_foreground_voxels": target_foreground,
            "materialization_cache_hit": bool(
                cache_entry and cache_entry["cache_hit"]
            ),
            "materialization_cache_error": cache_error,
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


def materialize_dataset(
    train_samples, dataset_dir, val_samples=None, cache_dir=None
):
    dataset_dir = Path(dataset_dir)
    _check_materialize_disk_space(train_samples, val_samples or [], dataset_dir)
    if dataset_dir.exists():
        rmtree_with_retry(dataset_dir)
    train_rows = _materialize_split(
        train_samples,
        dataset_dir / "imagesTr",
        dataset_dir / "labelsTr",
        "train",
        cache_dir=cache_dir,
    )
    val_rows = _materialize_split(
        val_samples or [],
        dataset_dir / "imagesVal",
        dataset_dir / "labelsVal",
        "validation",
        cache_dir=cache_dir,
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


def _model_root_for_config(model_path, base_config):
    path = Path(str(model_path)).expanduser()
    if path.is_absolute():
        return path
    base_path = Path(base_config).expanduser().resolve()
    candidates = []
    for parent in (base_path.parent,) + tuple(base_path.parents):
        if (parent / "src").is_dir() and (parent / "models").is_dir():
            candidates.append(parent / path)
            break
    configured = str(load_repo_config().get("dinov3_project") or "").strip()
    if configured:
        candidates.append(Path(configured) / path)
    candidates.append(
        ROOT / "integrations" / "dinov3-medical-seg" / path
    )
    candidates.append(base_path.parent / path)
    for candidate in candidates:
        if candidate.is_dir() or candidate.is_file():
            return candidate
    return candidates[0]


def _backbone_feature_plan(model_path, base_config, model_scale):
    """Resolve semantic feature layers from the actual local model config."""
    model_root = _model_root_for_config(model_path, base_config)
    try:
        return inspect_dinov3_vit_weights(model_root)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


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
    decoder_type = str(args.decoder or "segformer3d")
    cached_slice_pipeline = decoder_type == "feature_unet2d"
    if not cached_slice_pipeline and str(args.modality or "auto").lower() == "auto":
        raise RuntimeError(
            "Training modality must be resolved before writing the portable "
            "model configuration."
        )
    if int(args.batch_size) < 1:
        raise RuntimeError("Resolved batch size must be at least one.")
    img_size = [int(part.strip()) for part in str(args.img_size).split(",")]
    model_path = args.model_path
    if not model_path:
        model_scale = str(args.model_scale or "vitb16").lower()
        if model_scale in ("vits16", "vit_s", "small"):
            model_path = "./models/dinov3-vits16"
        elif model_scale in ("vitl16", "vit_l", "large"):
            model_path = "./models/dinov3-vitl16"
        elif model_scale in ("vith16plus", "vit_h", "huge"):
            model_path = "./models/dinov3-vith16plus"
        else:
            model_path = "./models/dinov3-vitb16"
    finetune_method = str(args.finetune_method or "lora").lower()
    if finetune_method in ("decoder_only", "decode_only", "decoder-only", "decode-only"):
        finetune_method = "frozen"
    encoder_backend = str(
        getattr(args, "encoder_backend", "auto") or "auto"
    ).strip().lower()
    if cached_slice_pipeline and encoder_backend == "auto":
        encoder_path = Path(model_path)
        if not encoder_path.is_absolute():
            encoder_path = Path(base_config).resolve().parents[2] / encoder_path
        onnx_path = (
            encoder_path
            if encoder_path.suffix.lower() == ".onnx"
            else encoder_path / "model.onnx"
        )
        encoder_backend = "onnx" if onnx_path.is_file() else "pytorch"
    compatibility_onnx = cached_slice_pipeline and encoder_backend == "onnx"
    backbone_plan = (
        {
            "num_hidden_layers": 12,
            "hidden_size": 384,
            "patch_size": 16,
            "out_indices": [11],
            "config_path": "",
            "model_type": "dinov3_vit",
            "weights_format": "onnx",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        if cached_slice_pipeline
        else _backbone_feature_plan(model_path, base_config, args.model_scale)
    )
    patch_size = int(backbone_plan.get("patch_size") or 16)
    if any(value % patch_size for value in img_size):
        raise RuntimeError(
            "Image height and width {} must be divisible by the selected "
            "DINOv3 patch size {}.".format(img_size, patch_size)
        )
    config = {
        "_base_": [str(base_config)],
        "exp_name": exp_name,
        "model": {
            "model_path": model_path,
            "num_classes": 2,
            # The compatibility ONNX model consumes the reference casewise
            # z-score values directly. The optional HuggingFace/PyTorch encoder
            # is a different model and retains its ImageNet input contract.
            "input_normalization": "none" if compatibility_onnx else "imagenet",
            "image_mean": list(backbone_plan["image_mean"]),
            "image_std": list(backbone_plan["image_std"]),
            "encoder_backend": encoder_backend,
            "out_indices": list(backbone_plan["out_indices"]),
        },
        "finetune": {
            "method": finetune_method, "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            # Q/V is the bounded LoRA policy used by every supported DINOv3
            # ViT scale. Recording it removes an otherwise hidden architecture
            # decision from model manifests and keeps inference reproducible.
            "target_modules": ["q_proj", "v_proj"],
            "adapter_bottleneck": int(args.adapter_bottleneck),
        },
        "decoder": {
            "type": decoder_type,
            "architecture_family": (
                getattr(args, "architecture_plan", {}) or {}
            ).get("resolved_dimension"),
            "quality_mode": (
                getattr(args, "architecture_plan", {}) or {}
            ).get("quality_mode"),
            "anisotropic_context": bool(
                (strategy_overrides or {}).get("model", {}).get(
                    "channel_policy"
                )
                == "2_5d"
            ),
        },
        "data": {
            "name": "mimics_fewshot_" + safe_slug(args.organ), "data_root": str(dataset_dir),
            "img_size": img_size, "k_shot": -1, "fold": 0, "modality": args.modality,
            "intensity": (
                {"window": [-1024.0, 1024.0], "percentiles": None}
                if str(args.modality).lower() == "ct"
                else {"window": None, "percentiles": [0.5, 99.5]}
            ),
            "resize_mode": "stretch" if cached_slice_pipeline else "fit_pad",
            "normalization_scope": (
                "slice_case"
                if cached_slice_pipeline
                else "case_before_roi_or_patch"
            ),
            "slice_normalization": (
                "timeslice_casewise"
                if compatibility_onnx
                else "percentile_minmax"
            ),
        },
        "training": {
            "epochs": int(args.epochs), "batch_size": int(args.batch_size),
            "grad_accumulation": int(args.grad_accumulation), "mixed_precision": bool(args.mixed_precision),
            "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            "scheduler": None if getattr(args, "lr_scheduler", "cosine") == "constant" else getattr(args, "lr_scheduler", "cosine"),
            "warmup_epochs": int(getattr(args, "warmup_epochs", 3)),
            "validation_interval": int(getattr(args, "validation_interval", 2)),
            # The Mimics UI exposes an explicit epoch budget. Do not inherit
            # hidden early stopping from a research base configuration.
            "early_stopping": {
                "min_epochs": int(args.epochs),
                "patience": 0,
                "min_delta": 0.0,
            },
            "keep_last_checkpoints": int(args.keep_last_checkpoints),
            "validation_enabled": bool(validation_enabled),
            "gpu_memory_budget_gb": float(
                getattr(args, "gpu_memory_gb", 0.0) or 0.0
            ),
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
    architecture_plan = copy.deepcopy(
        getattr(args, "architecture_plan", {}) or {}
    )
    if architecture_plan:
        architecture_plan["backbone_depth"] = int(
            backbone_plan["num_hidden_layers"]
        )
        architecture_plan["backbone_hidden_size"] = int(
            backbone_plan.get("hidden_size") or 0
        )
        architecture_plan["backbone_patch_size"] = patch_size
        architecture_plan["backbone_model_type"] = str(
            backbone_plan.get("model_type") or ""
        )
        architecture_plan["backbone_image_mean"] = list(
            backbone_plan.get("image_mean") or []
        )
        architecture_plan["backbone_image_std"] = list(
            backbone_plan.get("image_std") or []
        )
        architecture_plan["backbone_out_indices"] = list(
            backbone_plan["out_indices"]
        )
        architecture_plan = seal_architecture_plan(architecture_plan)
        args.architecture_plan = architecture_plan
        config.setdefault("runtime", {})["architecture_plan"] = architecture_plan
    modality_resolution = copy.deepcopy(
        getattr(args, "modality_resolution", {}) or {}
    )
    if modality_resolution:
        config.setdefault("runtime", {})[
            "modality_resolution"
        ] = modality_resolution
    model_sha256 = str(getattr(args, "model_sha256", "") or "").strip().lower()
    if model_sha256:
        config["model"]["expected_sha256"] = model_sha256
    if cached_slice_pipeline:
        scheduler = str(getattr(args, "lr_scheduler", "cosine") or "cosine")
        if scheduler == "constant_warmup":
            raise RuntimeError(
                "Frozen feature slice training supports cosine or constant learning rate; "
                "warmup is not part of this verified training path."
            )
        config["model"].update({
            "out_indices": [11],
            "slice_axis": "axial",
            "slice_batch_size": max(1, int(args.batch_size)),
            "channel_policy": "repeat",
        })
        config["finetune"]["method"] = "frozen"
        config["data"].update({
            "img_size": img_size,
            "patch": {"enabled": False},
            "roi": {"enabled": False},
            "native_grid": True,
            "target_spacing": None,
        })
        config["augmentation"] = {"enabled": False}
        config["training"].update({
            "pipeline": "cached_slices",
            "experiment_root": str(path.parent / "training_artifacts"),
            "batch_size": max(1, int(args.batch_size)),
            "grad_accumulation": 1,
            "mixed_precision": False,
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "scheduler": None if scheduler == "constant" else "cosine_epoch",
            "warmup_epochs": 0,
            "cosine_min_lr_ratio": 0.1,
            "keep_feature_cache": False,
            # Each source affine determines a non-left/right in-plane axis.
            # The cache records that axis per case and the trainer mirrors only
            # along that case-specific safe direction.
            "cached_slice_flip_probability": 0.5,
        })
        strategy_options = (config.get("strategy") or {}).get("options") or {}
        cached_loss_type = str(strategy_options.get("loss_type", "auto"))
        if cached_loss_type not in ("auto", "ce"):
            config.setdefault("runtime", {}).setdefault(
                "compatibility_adjustments", []
            ).append({
                "field": "loss_type",
                "from": cached_loss_type,
                "to": "ce",
                "reason": (
                    "Frozen Feature 2D uses the verified cross-entropy "
                    "training objective"
                ),
            })
        # Strategy compilation may have resolved "auto" to a volumetric loss.
        # The final pipeline contract is authoritative for this decoder.
        config["loss"] = {"type": "ce"}
        config.setdefault("inference", {}).update({
            "tta_axes": [],
            "scales": [],
        })
        strategy = config.get("strategy")
        if isinstance(strategy, dict) and isinstance(strategy.get("options"), dict):
            strategy["options"].update({
                "sampling_mode": "full",
                "channel_policy": "repeat",
                "slice_axis": "axial",
            })
        compatibility_adjustments = getattr(args, "_compatibility_adjustments", None)
        if not compatibility_adjustments:
            compatibility_adjustments = []
            original_finetune = str(getattr(args, "finetune_method", "frozen") or "frozen").lower()
            if original_finetune in ("decoder_only", "decode_only", "decoder-only", "decode-only"):
                compatibility_adjustments.append({
                    "field": "finetune_method",
                    "from": original_finetune,
                    "to": "frozen",
                    "reason": "decoder_only is an alias for frozen (decoder-only training with frozen encoder)",
                })
            original_strategy = str(getattr(args, "strategy", "full_volume") or "full_volume")
            if original_strategy != "full_volume":
                compatibility_adjustments.append({
                    "field": "strategy",
                    "from": original_strategy,
                    "to": "full_volume",
                    "reason": "Frozen Feature 2D only supports full-volume sampling",
                })
            original_grad = int(getattr(args, "grad_accumulation", 1) or 1)
            if original_grad != 1:
                compatibility_adjustments.append({
                    "field": "grad_accumulation",
                    "from": original_grad,
                    "to": 1,
                    "reason": "Frozen Feature 2D uses real slice batches; grad_accumulation must be 1",
                })
            original_mixed = bool(getattr(args, "mixed_precision", False))
            if original_mixed:
                compatibility_adjustments.append({
                    "field": "mixed_precision",
                    "from": True,
                    "to": False,
                    "reason": "Frozen Feature 2D disables mixed precision for feature-cache parity",
                })
        if compatibility_adjustments:
            recorded = config.setdefault("runtime", {}).setdefault(
                "compatibility_adjustments", []
            )
            for adjustment in compatibility_adjustments:
                if adjustment not in recorded:
                    recorded.append(adjustment)
    if status_path or cancel_path or metrics_history_path:
        config.setdefault("runtime", {}).update({
            "status_path": str(status_path or ""), "cancel_path": str(cancel_path or ""),
            "metrics_history_path": str(metrics_history_path or ""), "status_interval_seconds": 2.0,
        })
    try:
        from src.data.input_contract import input_contract_for_config
    except ImportError:
        # The DINOv3 project may live outside this repo (see fewshot_config.json
        # "dinov3_project"). Try the bundled copy first, then the configured path.
        candidate_roots = [ROOT / "integrations" / "dinov3-medical-seg"]
        configured = str(load_repo_config().get("dinov3_project") or "").strip()
        if configured:
            candidate_roots.append(Path(configured))
        for candidate_root in candidate_roots:
            if (candidate_root / "src" / "data" / "input_contract.py").is_file():
                if str(candidate_root) not in sys.path:
                    sys.path.insert(0, str(candidate_root))
                break
        from src.data.input_contract import input_contract_for_config
    config.setdefault("runtime", {})["input_contract"] = input_contract_for_config(
        config
    )
    import yaml
    write_text_atomic(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=False))
    return config


def portable_inference_config(effective_config, dinov3_root):
    """Strip run-local paths while preserving the trained model architecture."""
    config = copy.deepcopy(effective_config)
    config.pop("_base_", None)
    data = config.setdefault("data", {})
    data.pop("data_root", None)
    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        for key in (
            "status_path",
            "cancel_path",
            "metrics_history_path",
        ):
            runtime.pop(key, None)
    training = config.get("training")
    if isinstance(training, dict):
        training.pop("experiment_root", None)
    model = config.get("model")
    if isinstance(model, dict):
        raw = str(model.get("model_path") or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(dinov3_root) / path
            try:
                model["model_path"] = path.resolve().relative_to(
                    Path(dinov3_root).resolve()
                ).as_posix()
            except (OSError, RuntimeError, ValueError):
                # An external encoder path remains explicit; the model audit on
                # the target machine will report it instead of silently changing it.
                model["model_path"] = str(path.resolve())
    return config


def validate_training_encoder_assets(config, dinov3_root):
    """Fail before GPU acquisition when the configured frozen encoder is unusable."""
    model = dict(config.get("model") or {})
    raw_path = str(model.get("model_path") or "").strip()
    if not raw_path:
        raise RuntimeError("The training configuration does not define model.model_path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(dinov3_root) / path
    path = path.resolve()
    onnx_path = path if path.suffix.lower() == ".onnx" else path / "model.onnx"
    backend = str(model.get("encoder_backend") or "auto").strip().lower()
    if backend not in ("auto", "onnx", "pytorch"):
        raise RuntimeError("Unknown encoder backend: {}".format(backend))
    if backend == "pytorch" and path.suffix.lower() == ".onnx":
        raise RuntimeError(
            "The PyTorch encoder backend requires a HuggingFace model directory, "
            "not an ONNX file."
        )
    use_onnx = backend == "onnx" or (
        backend == "auto" and onnx_path.is_file()
    )
    expected_sha256 = str(model.get("expected_sha256") or "").strip().lower()
    if use_onnx and not onnx_path.is_file():
        raise RuntimeError(
            "The verified default encoder is required but was not found: {}".format(
                onnx_path
            )
        )
    if use_onnx:
        finetune_method = str(
            config.get("finetune", {}).get("method") or "frozen"
        ).lower()
        if finetune_method != "frozen":
            raise RuntimeError(
                "The ONNX encoder is frozen and cannot use {} fine-tuning. "
                "Choose Frozen Feature 2D or a local PyTorch model.".format(
                    finetune_method
                )
            )
        out_indices = list(model.get("out_indices") or [])
        if out_indices != [11]:
            raise RuntimeError(
                "The ONNX encoder exposes only its final feature map; model.out_indices "
                "must be [11], not {}.".format(out_indices)
            )
        if expected_sha256:
            digest = hashlib.sha256()
            with onnx_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "The configured ViT-S encoder checksum does not match the verified "
                    "model. Expected {}, found {}.".format(
                        expected_sha256,
                        actual_sha256,
                    )
                )
        try:
            import onnxruntime
        except ImportError:
            raise RuntimeError(
                "The configured ViT-S encoder is ONNX, but onnxruntime is not installed "
                "in nninteractive_env. Install the offline onnxruntime-gpu wheel first."
            )
        if os.name == "nt" and hasattr(onnxruntime, "preload_dlls"):
            try:
                onnxruntime.preload_dlls()
            except Exception:
                pass
        available_providers = list(onnxruntime.get_available_providers())
        try:
            import torch
            torch_cuda_ready = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda_ready = False
        if os.name == "nt" and torch_cuda_ready and "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "PyTorch can use CUDA, but ONNX Runtime cannot. Install the matching "
                "onnxruntime-gpu Windows wheel and CUDA/cuDNN runtime, then rerun "
                "environment setup. Available ONNX providers: {}".format(
                    ", ".join(available_providers) or "none"
                )
            )
        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_shape = list(session.get_inputs()[0].shape)
        configured_size = [
            int(value)
            for value in (config.get("data", {}).get("img_size") or [])
        ]
        fixed_size = (
            [int(value) for value in input_shape[-2:]]
            if all(
                isinstance(value, int) and value > 0
                for value in input_shape[-2:]
            )
            else None
        )
        if fixed_size is not None and configured_size != fixed_size:
            raise RuntimeError(
                "The local ViT-S ONNX encoder expects image size {}, but training is "
                "configured for {}. Select the matching Image detail value.".format(
                    fixed_size,
                    configured_size,
                )
            )
        if (
            fixed_size is None
            and (
                len(configured_size) != 2
                or any(value <= 0 or value % 16 for value in configured_size)
            )
        ):
            raise RuntimeError(
                "A dynamic ONNX encoder requires two configured image dimensions "
                "that are positive multiples of 16."
            )
        input_size = fixed_size or configured_size
        return {
            "backend": "onnx",
            "path": str(onnx_path),
            "input_size": input_size,
            "dynamic_input": fixed_size is None,
            "available_providers": available_providers,
        }
    try:
        backbone_spec = inspect_dinov3_vit_weights(path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    config_path = Path(backbone_spec["config_path"])
    weights_path = path / "model.safetensors"
    if expected_sha256:
        if not weights_path.is_file():
            raise RuntimeError(
                "A single-file SHA-256 was configured, but the selected "
                "DINOv3 model uses sharded safetensors."
            )
        digest = hashlib.sha256()
        with weights_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "The configured PyTorch encoder checksum does not match "
                "model.safetensors. Expected {}, found {}.".format(
                    expected_sha256,
                    actual_sha256,
                )
            )
    try:
        hidden_size = int(backbone_spec["hidden_size"])
        patch_size = int(backbone_spec["patch_size"])
    except Exception as exc:
        raise RuntimeError("Could not validate DINO encoder config {}: {}".format(config_path, exc))
    if str(config.get("decoder", {}).get("type")) == "feature_unet2d":
        if hidden_size != 384 or patch_size != 16:
            raise RuntimeError(
                "feature_unet2d requires ViT-S/16 (hidden_size=384, patch_size=16); "
                "configured encoder reports hidden_size={}, patch_size={}.".format(
                    hidden_size,
                    patch_size,
                )
            )
    return {
        "backend": "pytorch",
        "path": str(path),
        "hidden_size": hidden_size,
        "patch_size": patch_size,
    }


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
            "label export cannot start. Set MIMICS_BACKGROUND_EXE or "
            "mimics_background_exe, or select an existing exported masks folder.",
        )
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
            "# Auto-generated runner for Mimics few-shot label export",
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
            "DINOv3 label export",
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
                kind="fewshot_label_export",
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
            if cancel_path and Path(cancel_path).is_file() and not stop_requested:
                stop_requested = True
                write_json_atomic(stop_path, {
                    "status": "stop_requested",
                    "requested_at_epoch": time.time(),
                    "reason": "DINOv3 training was cancelled during label export",
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
            "reason": "DINOv3 label export timed out",
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
                    "DINOv3 label export timed out."
                    if lock_releasable else
                    "DINOv3 label export timed out, but the background Mimics "
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


def _path_stat_signature(path):
    path = Path(path)
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "name": path.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    if path.is_dir():
        count = 0
        total_size = 0
        latest_mtime = 0
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            stat = child.stat()
            count += 1
            total_size += int(stat.st_size)
            latest_mtime = max(latest_mtime, int(stat.st_mtime_ns))
        return {
            "kind": "directory",
            "name": path.name,
            "file_count": count,
            "total_size": total_size,
            "latest_mtime_ns": latest_mtime,
        }
    raise OSError("path does not exist: {}".format(path))


def _dino_mcs_label_fingerprint(mcs_path, image_path, mask_names):
    payload = {
        "mcs": _path_stat_signature(mcs_path),
        "image": _path_stat_signature(image_path),
        "mask_names": sorted(str(value).strip().lower() for value in mask_names),
        "export_contract": "dinov3_source_grid_mcs_export.v2",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cached_case_label(cache_root, case_id, fingerprint):
    case_root = Path(cache_root) / safe_slug(case_id)
    metadata = read_json(case_root / "metadata.json", {}) or {}
    labels = sorted((case_root / "segmentations").glob("*.nii.gz"))
    labels += sorted((case_root / "segmentations").glob("*.nii"))
    if (
        metadata.get("fingerprint") == fingerprint
        and len(labels) == 1
        and labels[0].is_file()
    ):
        return labels[0]
    return None


def _copy_label_atomic(source, destination):
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


def _plan_dino_mcs_label_cache(
    ts_root,
    workspace,
    organ_slug,
    cases,
    mask_names,
    mcs_output_dir=None,
):
    mcs_root = (
        Path(mcs_output_dir).expanduser().resolve()
        if mcs_output_dir
        else resolve_mimics_output_dir(ts_root)
    )
    requested = set(cases or [])
    if requested:
        case_ids = sorted(requested)
    else:
        case_ids = sorted(path.stem for path in mcs_root.glob("*.mcs"))
    cache_root = Path(workspace) / "cache" / "mcs_labels" / organ_slug
    reusable = {}
    changed = []
    fingerprints = {}
    for case_id in case_ids:
        mcs_path = mcs_root / (case_id + ".mcs")
        image_path = find_image(Path(ts_root) / case_id)
        if not mcs_path.is_file() or not image_path:
            changed.append(case_id)
            continue
        try:
            fingerprint = _dino_mcs_label_fingerprint(
                mcs_path, image_path, mask_names
            )
        except OSError:
            changed.append(case_id)
            continue
        fingerprints[case_id] = fingerprint
        cached = _cached_case_label(cache_root, case_id, fingerprint)
        if cached:
            reusable[case_id] = cached
        else:
            changed.append(case_id)
    return {
        "cache_root": cache_root,
        "reusable": reusable,
        "changed": changed,
        "fingerprints": fingerprints,
        "requested": case_ids,
    }


def _publish_dino_mcs_label_cache(
    plan, staging_root, mask_names, output_root=None
):
    available = set(plan["reusable"])
    warnings = []
    cache_root = Path(plan["cache_root"])
    staging_root = Path(staging_root)
    for case_id in plan["changed"]:
        source_dir = staging_root / case_id / "segmentations"
        labels = sorted(source_dir.glob("*.nii.gz"))
        labels += sorted(source_dir.glob("*.nii"))
        if len(labels) != 1:
            continue
        if output_root is not None:
            try:
                _copy_label_atomic(
                    labels[0],
                    Path(output_root)
                    / case_id
                    / "segmentations"
                    / labels[0].name,
                )
            except OSError as exc:
                warnings.append(
                    {
                        "case_id": case_id,
                        "stage": "training_staging",
                        "error": str(exc),
                    }
                )
                continue
        available.add(case_id)
        fingerprint = plan["fingerprints"].get(case_id)
        if not fingerprint:
            continue
        cache_case = cache_root / safe_slug(case_id)
        cached_label = cache_case / "segmentations" / labels[0].name
        try:
            _copy_label_atomic(labels[0], cached_label)
            write_json_atomic(
                cache_case / "metadata.json",
                {
                    "schema_version": "dinov3_mcs_label_cache.v1",
                    "case_id": case_id,
                    "fingerprint": fingerprint,
                    "mask_names": list(mask_names),
                    "updated_at_epoch": time.time(),
                },
            )
        except OSError as exc:
            warnings.append(
                {
                    "case_id": case_id,
                    "stage": "cache_publish",
                    "error": str(exc),
                }
            )
    return available, warnings


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


def register_global_model(manifest, manifest_path=None):
    path = global_registry_path()
    payload = read_json(path, {}) or {}
    models = payload.get("models") or []
    manifest_path = str(
        Path(manifest_path).resolve()
        if manifest_path
        else Path(manifest["checkpoint"]).parent / "manifest.json"
    )
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
        manifest_path = Path(row.get("manifest_path", "")).expanduser()
        if not manifest_path.is_file():
            continue
        resolved = dict(row)
        resolved["manifest_path"] = str(manifest_path.resolve())
        rows.append(resolved)
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


def _cmd_train_impl(args):
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
    mask_names = resolve_training_mask_names(repo_config, args.organ, getattr(args, "mask_names", None))
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
        "mask_names": mask_names,
        "created_at_epoch": time.time(),
    })
    append_log(workspace, "Training job {} started for organ {}.".format(run_id, args.organ))

    cases = set(parse_case_list(args.cases) or [])
    if not cases:
        cases = None
    fresh_label_root = None
    reusable_label_root = None
    if args.export_labels and getattr(args, "label_root", None):
        update_status(status_path, {
            "status": "failed",
            "error": (
                "Choose either fresh export from saved .mcs or an existing "
                "exported masks folder, not both."
            ),
        })
        return 2
    if getattr(args, "label_root", None):
        reusable_label_root = Path(args.label_root).expanduser().resolve()
        if not reusable_label_root.is_dir():
            update_status(status_path, {
                "status": "failed",
                "error": "exported masks folder does not exist: {}".format(
                    reusable_label_root
                ),
            })
            return 2
    fresh_label_cleanup = {"enabled": False}
    if args.export_labels:
        update_status(status_path, {"status": "exporting_labels"})
        fresh_label_root = run_dir / "fresh_labels"
        _register_transient_cleanup(fresh_label_root)
        label_cache_plan = _plan_dino_mcs_label_cache(
            ts_root,
            workspace,
            organ_slug,
            cases,
            mask_names,
            mcs_output_dir=args.mcs_output_dir,
        )
        fresh_label_root.mkdir(parents=True, exist_ok=True)
        for case_id, cached_label in label_cache_plan["reusable"].items():
            _copy_label_atomic(
                cached_label,
                fresh_label_root
                / case_id
                / "segmentations"
                / cached_label.name,
        )
        changed_label_root = run_dir / "fresh_labels_changed"
        _register_transient_cleanup(changed_label_root)
        update_status(
            status_path,
            {
                "label_cache_reused": len(label_cache_plan["reusable"]),
                "label_cache_refresh": len(label_cache_plan["changed"]),
            },
        )
        append_log(
            workspace,
            "MCS label cache: reused {} unchanged case(s); {} case(s) "
            "require fresh export.".format(
                len(label_cache_plan["reusable"]),
                len(label_cache_plan["changed"]),
            ),
        )
        try:
            if label_cache_plan["changed"]:
                export_result = launch_mimics_export(
                    ts_root,
                    set(label_cache_plan["changed"]),
                    args.mimics_exe,
                    workspace,
                    args.export_timeout_seconds,
                    status_path=status_path,
                    cancel_path=cancel_path,
                    lock_timeout_seconds=args.background_mimics_lock_timeout_seconds,
                    label_staging_dir=changed_label_root,
                    export_space="source_image",
                    mask_names=mask_names,
                    target_mask_name=organ_slug,
                    mcs_output_dir=args.mcs_output_dir,
                    skip_projects_without_requested_mask=True,
                    skip_invalid_projects=True,
                )
            else:
                export_result = {
                    "launched": True,
                    "returncode": 0,
                    "runner_started": False,
                    "cache_only": True,
                    "batch_status": {
                        "status": "completed",
                        "completed": len(label_cache_plan["reusable"]),
                        "failed": 0,
                        "skipped": 0,
                    },
                }
        except ResourceLockCancelled:
            update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for background Mimics"})
            return 130
        except ResourceLockTimeout as exc:
            update_status(status_path, {"status": "failed", "error": str(exc)})
            return 75
        if not export_result.get("launched"):
            launch_reason = str(export_result.get("reason") or "")
            if launch_reason == "mimics_not_found":
                launch_error = (
                    "Fresh label export requires MimicsResearch.exe, but it could "
                    "not be found. Set MIMICS_BACKGROUND_EXE or "
                    "mimics_background_exe in mimics_io_config.json, then retry. "
                    "Alternatively, choose an existing exported masks folder."
                )
            else:
                launch_error = (
                    "Fresh label export was requested but background Mimics "
                    "export could not be started."
                )
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": launch_error,
            })
            return 75
        batch_status = export_result.get("batch_status") or {}
        if str(batch_status.get("status") or "").lower() == "stopping":
            update_status(status_path, {
                "status": "stopping",
                "label_export": export_result,
                "termination_pending": True,
                "error": batch_status.get("error") or (
                    "The background Mimics label export is still stopping; "
                    "its resource lock was retained."
                ),
            })
            return 75
        if export_result.get("timed_out") or int(export_result.get("returncode", 0) or 0) != 0:
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": "fresh label export did not finish successfully; training was not started with stale labels",
            })
            return 75
        if str(batch_status.get("status") or "").lower() == "cancelled":
            update_status(status_path, {
                "status": "cancelled",
                "label_export": export_result,
                "error": "training was cancelled during fresh label export",
            })
            return 130
        if cancel_path.is_file():
            update_status(status_path, {
                "status": "cancelled",
                "label_export": export_result,
                "error": "training was cancelled after fresh label export",
            })
            return 130
        if str(batch_status.get("status") or "").lower() == "failed":
            update_status(status_path, {
                "status": "failed",
                "label_export": export_result,
                "error": (
                    "Fresh label export could not complete. Training was not started. "
                    "Review the job-scoped export diagnostics: {0}"
                ).format(
                    export_result.get("job_runtime") or export_result.get("log") or workspace,
                ),
            })
            return 75
        if label_cache_plan["changed"]:
            available_cases, cache_warnings = (
                _publish_dino_mcs_label_cache(
                    label_cache_plan,
                    changed_label_root,
                    mask_names,
                    output_root=fresh_label_root,
                )
            )
            if cache_warnings:
                append_log(
                    workspace,
                    "Warning: {} MCS label cache publication operation(s) "
                    "failed. Current training labels remain available; these "
                    "cases may be exported again next run.".format(
                        len(cache_warnings)
                    ),
                )
                update_status(
                    status_path,
                    {"label_cache_warnings": cache_warnings[:10]},
                )
            rmtree_with_retry(changed_label_root)
        else:
            available_cases = set(label_cache_plan["reusable"])
        cases = set(available_cases)
        export_failed = int(batch_status.get("failed", 0) or 0)
        export_skipped = int(batch_status.get("skipped", 0) or 0)
        if export_failed or export_skipped:
            warning = (
                "Fresh label export isolated {0} failed and {1} skipped case(s). "
                "Training will continue only if the remaining exported labels "
                "satisfy the requested sample count. Diagnostics: {2}"
            ).format(
                export_failed,
                export_skipped,
                export_result.get("job_runtime") or export_result.get("log") or workspace,
            )
            append_log(workspace, warning)
            update_status(status_path, {
                "label_export_warning": warning,
                "label_export": export_result,
            })
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
                    "the selected .mcs projects contain exactly one matching saved mask."
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

    selected_label_root = fresh_label_root or reusable_label_root
    resolved_label_source = (
        "fresh_mcs_export"
        if fresh_label_root
        else ("exported_masks" if reusable_label_root else "source_dataset")
    )
    update_status(status_path, {
        "label_source": resolved_label_source,
        "label_root": str(selected_label_root) if selected_label_root else "",
    })
    samples, skipped = discover_samples(
        ts_root,
        args.organ,
        cases,
        label_root=selected_label_root,
        fallback_to_case_labels=not bool(selected_label_root),
        label_source=resolved_label_source,
        mask_names=mask_names,
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
                "Check that the selected cases have saved .mcs files and that each .mcs contains exactly one "
                "matching Mask ({3}). "
                "If you want to use existing NIfTI labels instead of refreshing from .mcs, disable label export before training."
            ).format(args.organ, len(selected), args.min_samples, ", ".join(mask_names))
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
    modality_resolution = resolve_training_modality(
        args.modality,
        train_samples + val_samples,
        manifest_roots=[
            args.mcs_output_dir,
            selected_label_root,
            resolve_mimics_output_dir(ts_root),
            ts_root,
        ],
    )
    args.modality = modality_resolution["resolved"]
    args.modality_resolution = modality_resolution
    modality_message = (
        "Image modality: requested {requested}, resolved {resolved} "
        "({source})."
    ).format(**modality_resolution)
    if modality_resolution["unresolved_cases"]:
        modality_message += (
            " {} case(s) had no modality metadata and use the same resolved "
            "run policy."
        ).format(len(modality_resolution["unresolved_cases"]))
    append_log(workspace, modality_message)
    update_status(
        status_path,
        {
            "modality": args.modality,
            "modality_resolution": modality_resolution,
        },
    )
    generalization_warnings = []
    if len(train_samples) < 5:
        generalization_warnings.append(
            "Only {} training case(s) were selected; decoder overfitting is likely.".format(
                len(train_samples)
            )
        )
    if val_samples and len(val_samples) < 2:
        generalization_warnings.append(
            "Only one validation case was selected; validation Dice is not a stable "
            "estimate of generalization."
        )
    if generalization_warnings:
        message = " ".join(generalization_warnings)
        append_log(workspace, "Training data warning: " + message)
        update_status(status_path, {"training_data_warning": message})
    fresh_geometry_checked = validate_fresh_export_geometry(train_samples + val_samples)
    if not args.keep_materialized_dataset:
        _register_transient_cleanup(dataset_dir)
    materialized_train, materialized_val = materialize_dataset(
        train_samples,
        dataset_dir,
        val_samples,
        cache_dir=(
            Path(args.materialization_cache_dir).expanduser().resolve()
            if getattr(args, "materialization_cache_dir", None)
            else workspace / "cache" / "materialized" / organ_slug
        ),
    )
    materialized_rows = materialized_train + materialized_val
    materialization_cache_hits = sum(
        bool(row.get("materialization_cache_hit"))
        for row in materialized_rows
    )
    materialization_cache_errors = [
        {
            "case_id": row.get("case_id"),
            "error": row.get("materialization_cache_error"),
        }
        for row in materialized_rows
        if row.get("materialization_cache_error")
    ]
    append_log(
        workspace,
        "Source-grid materialization cache: reused {} of {} case(s).".format(
            materialization_cache_hits, len(materialized_rows)
        ),
    )
    update_status(
        status_path,
        {
            "materialization_cache_reused": materialization_cache_hits,
            "materialization_cache_total": len(materialized_rows),
            "materialization_cache_warnings": materialization_cache_errors[:10],
        },
    )
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
    training_fingerprint = build_training_fingerprint(
        dataset_dir,
        support_case_ids,
        args.organ,
        modality=args.modality,
    )
    gpu_memory_gb = detect_gpu_memory_gb(
        getattr(args, "gpu_memory_gb", 0.0)
    )
    args.gpu_memory_gb = gpu_memory_gb
    derived_policy = derive_policy(
        training_fingerprint,
        gpu_memory_gb=gpu_memory_gb,
    )
    strategy_overrides = compile_strategy(
        strategy_id,
        training_fingerprint,
        derived_policy,
        strategy_options,
    )
    effective_resource_policy = copy.deepcopy(derived_policy)
    effective_resource_policy["patch"] = copy.deepcopy(
        (strategy_overrides.get("data") or {}).get("patch") or {}
    )
    effective_resource_policy["input_size"] = [
        int(part.strip())
        for part in str(args.img_size).split(",")
    ]
    architecture_plan = resolve_architecture(
        {
            "training_dimension": getattr(args, "training_dimension", None),
            "quality_mode": getattr(args, "quality_mode", "standard"),
            "finetune_method": args.finetune_method,
            "decoder": args.decoder,
        },
        training_fingerprint,
        gpu_memory_gb=gpu_memory_gb,
    )
    architecture_plan["sub_volume"] = bool(args.sub_volume)
    requested_batch_size = int(args.batch_size)
    if bool(args.sub_volume) and requested_batch_size > 1:
        raise RuntimeError(
            "Sub-volume training cannot be combined with volume batch size "
            "above 1. Use batch size 1 with gradient accumulation, or disable "
            "sub-volume training."
        )
    if requested_batch_size == 0:
        args.batch_size = recommended_batch_size(
            architecture_plan,
            effective_resource_policy,
            gpu_memory_gb=gpu_memory_gb,
        )
        architecture_plan["batch_size_source"] = "hardware_recommendation"
    else:
        args.batch_size = requested_batch_size
        architecture_plan["batch_size_source"] = "user"
    architecture_plan["requested_batch_size"] = requested_batch_size
    architecture_plan["resolved_batch_size"] = int(args.batch_size)
    architecture_plan = seal_architecture_plan(architecture_plan)
    args.decoder = architecture_plan["decoder"]
    args.architecture_plan = architecture_plan
    if not architecture_plan.get("legacy_compatibility"):
        args.encoder_backend = "pytorch"
        if (
            str(args.model_scale or "").lower() == "vits16"
            and not str(args.model_path or "").strip()
        ):
            args.model_sha256 = str(
                repo_config.get("default_safetensors_sha256") or ""
            ).strip()
    append_log(
        workspace,
        (
            "Architecture plan: requested {requested_dimension}, resolved "
            "{resolved_dimension}/{quality_mode} with {decoder}; batch size "
            "{resolved_batch_size} on a {gpu_memory_gb:.1f} GB planning budget."
        ).format(**architecture_plan),
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
    architecture_plan = copy.deepcopy(
        (effective_config.get("runtime") or {}).get("architecture_plan")
        or architecture_plan
    )
    append_log(
        workspace,
        "DINOv3 feature layers: {} of {} transformer blocks.".format(
            architecture_plan.get("backbone_out_indices"),
            architecture_plan.get("backbone_depth"),
        ),
    )
    try:
        encoder_assets = validate_training_encoder_assets(effective_config, dinov3_root)
    except Exception as exc:
        update_status(status_path, {
            "status": "failed",
            "error": str(exc),
            "config_path": str(config_path),
        })
        append_log(workspace, "Training preflight failed: {}.".format(exc))
        return 2
    experiment_root = Path(
        effective_config.get("training", {}).get("experiment_root")
        or (dinov3_root / "experiments")
    )
    if not experiment_root.is_absolute():
        experiment_root = (dinov3_root / experiment_root).resolve()
    exp_dir = experiment_root / exp_name
    if not bool(repo_config.get("keep_training_experiment_artifacts", False)):
        _register_transient_cleanup(exp_dir)
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
            "architecture_plan": architecture_plan,
            "strategy": strategy_overrides.get("strategy", {}),
            "encoder_assets": encoder_assets,
            "selection": {
                "cases": sorted(cases) if cases else None,
                "sample_mode": args.sample_mode,
                "max_samples": int(args.max_samples or 0),
                "val_fraction": float(args.val_fraction or 0.0),
                "val_cases": parse_case_list(args.val_cases),
                "label_source": resolved_label_source,
                "label_root": str(selected_label_root) if selected_label_root else "",
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
        "architecture_plan": architecture_plan,
        "training_seed": int(
            effective_config.get("training", {}).get("seed", 0)
        ),
        "training_fold": int(
            effective_config.get("data", {}).get("fold", 0)
        ),
        "encoder_assets": encoder_assets,
        "experiment_dir": str(exp_dir),
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
    final_progress = {}
    gpu_lock_releasable = True
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
            if gpu_lock is not None and not gpu_lock.update_pid(
                proc.pid,
                kind="fewshot_train",
                job_id=run_id,
                cancel_path=str(cancel_path),
                status_path=str(status_path),
            ):
                raise RuntimeError(
                    "Training started, but GPU lock ownership could not be "
                    "transferred to PID {}.".format(proc.pid)
                )
            try:
                register_process(
                    ROOT, "trainer", proc.pid,
                    parent_pid=os.getpid(),
                    state_path=str(status_path),
                )
            except Exception:
                pass
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
        final_progress = read_json(train_status, {}) or {}
        worker_cancelled = str(final_progress.get("status") or "").lower() == "cancelled"
        cancel_forced_exit = cancel_started is not None and proc.returncode != 0
        if worker_cancelled or cancel_forced_exit:
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
            if actual_train_log is None:
                error = "training process failed; training log is unavailable"
            update_status(status_path, {
                "status": "failed",
                "returncode": proc.returncode,
                "error": error,
                "metrics_history": str(metrics_history),
            })
            return proc.returncode or 1
        update_status(status_path, {
            "status": "finalizing",
            "returncode": 0,
            "training_progress": final_progress,
            "metrics_history": str(metrics_history),
            "late_cancel_ignored": bool(cancel_path.is_file() and cancel_started is None),
        })
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
            gpu_lock_releasable = terminate_and_reap_process(proc)
        update_status(status_path, {
            "status": "failed" if gpu_lock_releasable else "stopping",
            "error": str(exc),
            "termination_pending": not gpu_lock_releasable,
        })
        append_log(workspace, "Training job {} failed before completion: {}.".format(run_id, exc))
        if not gpu_lock_releasable:
            append_log(
                workspace,
                "Training process {} did not exit after termination; the GPU lock was retained.".format(
                    getattr(proc, "pid", "?")
                ),
            )
        return 1
    finally:
        if gpu_lock is not None and gpu_lock_releasable:
            gpu_lock.release()

    ckpt = latest_epoch_checkpoint(exp_dir / "checkpoints")
    if not ckpt:
        update_status(status_path, {"status": "failed", "error": "no checkpoint was produced"})
        return 1

    model_dir = workspace / "models" / organ_slug / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_ckpt = model_dir / "model.pth"
    shutil.copy2(str(ckpt), str(registry_ckpt))
    registered_config = model_dir / "config.yaml"
    import yaml
    write_text_atomic(
        registered_config,
        yaml.safe_dump(
            portable_inference_config(effective_config, dinov3_root),
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    registered_config_sha256 = hashlib.sha256(
        registered_config.read_bytes()
    ).hexdigest()
    best_dsc_value = final_progress.get("best_dsc")
    manifest = {
        "schema_version": "mimics_fewshot_model.v2",
        "model_id": run_id,
        "organ": args.organ,
        "organ_slug": organ_slug,
        "source_mask_names": mask_names,
        "best_dsc": best_dsc_value,
        "checkpoint": "model.pth",
        "config": "config.yaml",
        "config_sha256": registered_config_sha256,
        "strategy": strategy_overrides.get("strategy", {}),
        "architecture_plan": architecture_plan,
        "effective_config": effective_config,
        "input_contract": (
            effective_config.get("runtime", {}).get("input_contract") or {}
        ),
        "spatial_convention": "canonical_ras_array_xyz__model_tensor_zyx",
        "source_checkpoint": "model.pth",
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
            "training_dimension": str(
                architecture_plan.get("requested_dimension") or "auto"
            ),
            "resolved_dimension": str(
                architecture_plan.get("resolved_dimension") or ""
            ),
            "quality_mode": str(
                architecture_plan.get("quality_mode") or "standard"
            ),
            "model_scale": str(args.model_scale),
            "model_path": str(args.model_path or ""),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "requested_batch_size": int(requested_batch_size),
            "gpu_memory_gb": float(gpu_memory_gb),
            "seed": int(effective_config.get("training", {}).get("seed", 0)),
            "fold": int(effective_config.get("data", {}).get("fold", 0)),
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
            "source_mask_names": mask_names,
            "label_source": resolved_label_source,
            "label_root": str(selected_label_root) if selected_label_root else "",
        },
        "created_at_epoch": time.time(),
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "dinov3_root": str(dinov3_root),
        "base_config": str(base_config),
    }
    model_manifest_path = model_dir / "manifest.json"
    write_json_atomic(model_manifest_path, manifest)
    latest_manifest = dict(manifest)
    latest_manifest["checkpoint"] = "{}/model.pth".format(run_id)
    latest_manifest["config"] = "{}/config.yaml".format(run_id)
    latest_manifest["model_manifest"] = "{}/manifest.json".format(run_id)
    latest_path = workspace / "models" / organ_slug / "latest.json"
    write_json_atomic(latest_path, latest_manifest)
    try:
        register_global_model(manifest, model_manifest_path)
    except Exception as exc:
        manifest["global_registry_error"] = str(exc)
        write_json_atomic(model_manifest_path, manifest)
        latest_manifest.update({"global_registry_error": str(exc)})
        write_json_atomic(latest_path, latest_manifest)
        append_log(workspace, "Could not update global DINOv3 model registry: {}.".format(exc))
    if not args.keep_materialized_dataset:
        try:
            rmtree_with_retry(dataset_dir)
            manifest["dataset_retained"] = False
            manifest["dataset_cleanup_at_epoch"] = time.time()
            write_json_atomic(model_manifest_path, manifest)
            latest_manifest["dataset_retained"] = False
            latest_manifest["dataset_cleanup_at_epoch"] = manifest["dataset_cleanup_at_epoch"]
            write_json_atomic(latest_path, latest_manifest)
        except Exception as exc:
            manifest["dataset_retained"] = True
            manifest["dataset_cleanup_error"] = str(exc)
            write_json_atomic(model_manifest_path, manifest)
            latest_manifest["dataset_retained"] = True
            latest_manifest["dataset_cleanup_error"] = str(exc)
            write_json_atomic(latest_path, latest_manifest)
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


def cmd_train(args):
    """Run training and fail a created job record closed on finalization errors."""
    if not getattr(args, "run_id", None):
        args.run_id = "train_{}_{}".format(
            time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
    try:
        return _cmd_train_impl(args)
    except Exception as exc:
        try:
            workspace = workspace_for(
                Path(args.ts_root).resolve(), getattr(args, "workspace", None)
            )
            status_path = workspace / "jobs" / (str(args.run_id) + ".json")
            if status_path.is_file():
                current = read_json(status_path, {}) or {}
                if (
                    str(current.get("status") or "").lower() == "stopping"
                    and job_has_live_process(current)
                ):
                    update_status(status_path, {
                        "phase": "termination_pending",
                        "error": str(exc),
                        "termination_pending": True,
                    })
                    append_log(
                        workspace,
                        "Training job {} could not finish cleanup because an owned "
                        "process is still running; the resource lock was retained: {}.".format(
                            args.run_id, exc
                        ),
                    )
                    return 1
                update_status(status_path, {
                    "status": "failed",
                    "phase": "finalization_failed",
                    "error": str(exc),
                })
                append_log(
                    workspace,
                    "Training job {} failed while finalizing model artifacts: {}.".format(
                        args.run_id, exc
                    ),
                )
                return 1
        except Exception:
            pass
        raise
    finally:
        try:
            workspace = workspace_for(
                Path(args.ts_root).resolve(), getattr(args, "workspace", None)
            )
            status_path = workspace / "jobs" / (str(args.run_id) + ".json")
            if status_path.is_file():
                cleanup_terminal_training_artifacts(args, status_path)
        except Exception as cleanup_exc:
            try:
                append_log(
                    workspace,
                    "Automatic failed-job storage cleanup could not finish: {}.".format(
                        cleanup_exc
                    ),
                )
            except Exception:
                pass


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
        path = resolve_manifest_artifact(value, source, manifest)
        if not path.is_file():
            raise RuntimeError("model manifest points to a missing {}: {} ({})".format(key, path, source))
        checked[key] = str(path)
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
        actual_config_sha256 = hashlib.sha256(Path(checked["config"]).read_bytes()).hexdigest()
        checked["config_sha256_verified"] = actual_config_sha256 == expected_config_sha256
        if actual_config_sha256 != expected_config_sha256:
            raise RuntimeError(
                "model configuration was changed after training; refusing inconsistent inference: {} ({})".format(
                    checked["config"], source,
                )
            )
    checkpoint_path = Path(checked.get("checkpoint", ""))
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
    import numpy as np
    from mimics_bridge import _affine_close, get_source_image_geometry

    geometry = get_source_image_geometry(str(image_path))
    if not geometry:
        raise RuntimeError(
            "source image geometry could not be read: {}".format(image_path)
        )
    actual_shape = [int(value) for value in geometry["shape"][:3]]
    actual_affine = np.asarray(geometry["affine"], dtype=float)
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
        result["actual_voxel_to_ras_matrix"] = actual_affine.tolist()
        result["affine_max_abs_diff"] = float(
            np.max(np.abs(actual_affine - expected_affine_np))
        )
        if not _affine_close(actual_affine, expected_affine_np):
            raise RuntimeError(
                "source image affine does not match the open Mimics project (max abs diff {:.6g}): {}".format(
                    result["affine_max_abs_diff"],
                    result.get("actual_image_path", str(Path(image_path).resolve())),
                )
            )
    return result


def _inference_nifti_input(source, destination):
    source = Path(source)
    lower = source.name.lower()
    if source.is_file() and (
        lower.endswith(".nii") or lower.endswith(".nii.gz")
    ):
        return source.resolve(), False
    _materialize_source_image(source, destination)
    return Path(destination).resolve(), True


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
        explicit_image = str(getattr(args, "image_path", "") or "").strip()
        if not case_dir.is_dir() and not explicit_image:
            raise RuntimeError("case directory was not found: {}".format(case_dir))
        source_image = (
            Path(explicit_image).expanduser().resolve()
            if explicit_image
            else find_image(case_dir)
        )
        if explicit_image and not source_image.exists():
            raise RuntimeError(
                "the explicitly resolved source image was not found: {}".format(
                    source_image
                )
            )
        if not source_image:
            raise RuntimeError(
                "no supported medical image was found for case: {}".format(
                    args.case_id
                )
            )
        dinov3_root = dinov3_root_from_args(args)
        python_exe = python_from_args(args, dinov3_root)
        model = load_model_manifest(workspace, args.organ, args.model_id, args.model_manifest)
        if str(dinov3_root) not in sys.path:
            sys.path.insert(0, str(dinov3_root))
        from src.data.input_contract import validate_input_contract
        from src.utils.config import load_config as load_dinov3_config
        inference_config = load_dinov3_config(str(model["config"]))
        effective_input_contract = validate_input_contract(inference_config)
        registered_input_contract = model.get("input_contract") or {}
        if (
            registered_input_contract
            and registered_input_contract != effective_input_contract
        ):
            raise RuntimeError(
                "The selected model registry and config disagree about DINOv3 "
                "input preprocessing. Inference was stopped before GPU startup."
            )
        expected_shape = _parse_json_shape(getattr(args, "expected_source_shape", ""))
        expected_affine = _parse_json_matrix(getattr(args, "expected_source_voxel_to_ras_matrix", ""))
        expected_image_path = str(getattr(args, "expected_source_image_path", "") or "").strip()
        source_validation = validate_inference_source_geometry(
            source_image,
            expected_shape=expected_shape,
            expected_affine=expected_affine,
            expected_image_path=expected_image_path,
        )
        image, remove_inference_input = _inference_nifti_input(
            source_image,
            output_dir / (job_id + "_input.nii.gz"),
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
        "image_path": str(source_image),
        "inference_input_path": str(image),
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
    gpu_lock_releasable = True
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
            if gpu_lock is not None and not gpu_lock.update_pid(
                proc.pid,
                kind="fewshot_infer",
                job_id=job_id,
                cancel_path=str(cancel_path),
                status_path=str(status_path),
            ):
                raise RuntimeError(
                    "Inference started, but GPU lock ownership could not be "
                    "transferred to PID {}.".format(proc.pid)
                )
            try:
                register_process(
                    ROOT, "trainer", proc.pid,
                    parent_pid=os.getpid(),
                    state_path=str(status_path),
                )
            except Exception:
                pass
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
        if proc.returncode != 0:
            if cancel_path.is_file():
                update_status(status_path, {
                    "status": "cancelled",
                    "returncode": proc.returncode,
                })
                append_log(workspace, "Inference job {} cancelled.".format(job_id))
                return 130
            error = "inference process failed; see log"
            if actual_log_path is None:
                error = "inference process failed; inference log is unavailable"
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
            "image_path": str(source_image),
            "inference_input_path": (
                "" if remove_inference_input else str(image)
            ),
            "late_cancel_ignored": bool(cancel_path.is_file()),
        })
        append_log(workspace, "Inference job {} completed. Output: {}".format(job_id, output_path))
        return 0
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
            gpu_lock_releasable = terminate_and_reap_process(proc)
        update_status(status_path, {
            "status": "failed" if gpu_lock_releasable else "stopping",
            "error": str(exc),
            "termination_pending": not gpu_lock_releasable,
        })
        append_log(workspace, "Inference job {} failed before completion: {}.".format(job_id, exc))
        if not gpu_lock_releasable:
            append_log(
                workspace,
                "Inference process {} did not exit after termination; the GPU lock was retained.".format(
                    getattr(proc, "pid", "?")
                ),
            )
        return 1
    finally:
        if gpu_lock is not None and gpu_lock_releasable:
            gpu_lock.release()
        if locals().get("remove_inference_input"):
            try:
                unlink_with_retry(image)
            except Exception:
                pass


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
            manifest = resolved_manifest_payload(manifest_path)
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
    current_status = str(status.get("status") or "").lower()
    if current_status in TERMINAL_JOB_STATUSES:
        print("Job {} is already {}.".format(status.get("job_id", status_path.stem), current_status))
        return 0
    if current_status not in CANCELLABLE_JOB_STATUSES:
        print(
            "Job {} is {} and cannot be cancelled at this stage.".format(
                status.get("job_id", status_path.stem),
                current_status or "not running",
            )
        )
        return 0

    cancel_path = status.get("cancel_path")
    cancel_error = None
    if cancel_path:
        cancel_error = write_cancel_marker(cancel_path)
        if cancel_error:
            print("Warning: could not write cancel marker {}: {}".format(cancel_path, cancel_error), file=sys.stderr)
    grace_seconds = max(0.0, float(getattr(args, "grace_seconds", 30.0)))
    pids = job_process_ids(status)
    deadline = time.time() + grace_seconds
    while pids and time.time() < deadline:
        if all(not process_exists(pid) for pid in pids):
            break
        time.sleep(0.5)
    killed = []
    for pid in pids:
        if process_exists(pid) and terminate_process_tree(pid):
            killed.append(int(pid))
    reap_deadline = time.time() + 10.0
    while pids and time.time() < reap_deadline:
        if all(not process_exists(pid) for pid in pids):
            break
        time.sleep(0.25)

    latest = read_json(status_path, {}) or {}
    latest_status = str(latest.get("status") or "").lower()
    if latest_status not in TERMINAL_JOB_STATUSES and not job_has_live_process(latest):
        latest.update({
            "status": "cancelled",
            "cancel_requested_at_epoch": time.time(),
            "cancelled_pids": killed,
            "updated_at_epoch": time.time(),
        })
        if cancel_error:
            latest["cancel_marker_error"] = cancel_error
        write_json_atomic(status_path, latest)
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
    train.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Real training batch; 0 selects a hardware-aware recommendation",
    )
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--img-size", default="224,224")
    train.add_argument(
        "--modality",
        choices=("auto", "ct", "mr", "mri", "other"),
        default="auto",
    )
    train.add_argument("--val-fraction", type=float, default=0.0)
    train.add_argument("--val-cases")
    train.add_argument("--min-val-samples", type=int, default=1)
    train.add_argument("--finetune-method", choices=("frozen", "decoder_only", "decode_only", "lora", "adapter", "full"), default="frozen")
    train.add_argument(
        "--training-dimension",
        choices=("auto", "2d", "3d"),
        default=None,
    )
    train.add_argument(
        "--quality-mode",
        choices=("standard", "high_detail"),
        default="standard",
    )
    train.add_argument("--decoder", choices=(
        "scale_aware2d", "context3d_lite", "context3d_hybrid",
        "context3d_multiscale",
        "linear3d", "mlp_probe", "segformer3d", "token_pyramid3d", "dpt3d",
        "conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d", "feature_unet2d",
    ), default=None)
    train.add_argument("--lr-scheduler", choices=("constant", "constant_warmup", "cosine"), default="cosine")
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--model-scale", choices=("vits16", "vitb16", "vitl16", "vith16plus"), default="vitb16")
    train.add_argument(
        "--encoder-backend",
        choices=("auto", "onnx", "pytorch"),
        default="auto",
    )
    train.add_argument("--validation-interval", type=int, default=2)
    train.add_argument("--model-path")
    train.add_argument("--model-sha256")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--adapter-bottleneck", type=int, default=64)
    train.add_argument("--mixed-precision", action="store_true")
    train.add_argument("--sub-volume", action="store_true")
    train.add_argument("--sub-volume-size", default="32,256,256")
    train.add_argument("--export-labels", action="store_true")
    train.add_argument(
        "--label-root",
        help=(
            "Reusable exported masks root containing "
            "<case>/segmentations/<mask>.nii.gz or <case>/<mask>.nii.gz"
        ),
    )
    train.add_argument("--mimics-exe")
    train.add_argument("--mcs-output-dir", help="Folder containing saved .mcs projects for this training run")
    train.add_argument(
        "--mask-names",
        help="Comma-separated saved .mcs mask names accepted for this training target",
    )
    train.add_argument("--export-timeout-seconds", type=float, default=1800)
    train.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=1800)
    train.add_argument("--gpu-lock-timeout-seconds", type=float, default=86400)
    train.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=0.0,
        help="Planning budget; 0 detects the selected local/remote GPU",
    )
    train.add_argument("--keep-last-checkpoints", type=int, default=2)
    train.add_argument("--keep-materialized-dataset", action="store_true")
    train.add_argument(
        "--materialization-cache-dir",
        help=(
            "Persistent source-grid materialization cache. Remote workers use "
            "this to reuse unchanged image/label preparation across containers."
        ),
    )
    train.add_argument("--run-id")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run inference for one case with the latest or selected organ model")
    infer.add_argument("--ts-root", required=True)
    infer.add_argument("--case-id", required=True)
    infer.add_argument("--organ", required=True)
    infer.add_argument("--workspace")
    infer.add_argument(
        "--image-path",
        help=(
            "Explicit source image resolved from the open Mimics project or "
            "its relocatable dataset manifest"
        ),
    )
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
