#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External read-only DINOv3 few-shot status viewer.

The foreground Mimics process starts this tool with Popen and returns
immediately.  This UI only reads JSON/log files and can request cancellation
through the existing cancel markers recorded in job status files.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for _candidate in (str(TOOLS_DIR), str(PROJECT_ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from resource_locks import process_exists as resource_process_exists

from ui_theme import (
    choose_open_file_async,
    choose_save_file_async,
    configure_application,
    stylesheet as shared_stylesheet,
)


TITLE = "DINOv3 Few-Shot Status"
ACTIVE_STATUSES = set([
    "launching",
    "preparing",
    "exporting_labels",
    "waiting_for_background_mimics",
    "waiting_for_gpu",
    "training",
    "running",
    "cancelling",
    "stopping",
    "finalizing",
    "configuring",
    "selecting_model",
    "training_started",
    "preparing_remote",
    "connecting_remote",
    "uploading",
    "starting_remote",
    "reconnecting_remote",
    "waiting_for_remote_gpu",
    "remote_control_unavailable",
    "finalizing_remote",
    "downloading",
])


def safe_slug(value):
    text = str(value or "").strip().lower()
    result = []
    for character in text:
        result.append(character if character.isalnum() or character in "-_." else "_")
    return re.sub(r"_+", "_", "".join(result)).strip("._-") or "model"
CANCELLABLE_STATUSES = ACTIVE_STATUSES - {
    "cancelling", "stopping", "finalizing", "training_started",
}


def hidden_process_kwargs():
    if os.name != "nt":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startup,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


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


def write_json_best_effort(path, payload):
    try:
        write_json_atomic(path, payload, retries=8, max_sleep=0.15)
        return True
    except Exception:
        return False


def write_cancel_marker(cancel_path):
    if not cancel_path:
        return None
    try:
        write_text_atomic(
            cancel_path,
            "cancel requested at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")),
            retries=8,
            max_sleep=0.15,
        )
        return None
    except Exception as exc:
        return str(exc)


def request_job_cancel_async(job, status_path, grace_seconds=10.0):
    """Request cancellation without overwriting live progress from the worker."""
    cancel_error = write_cancel_marker(job.get("cancel_path"))
    remote = str(job.get("execution_backend") or "") == "remote"
    if remote and status_path:
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "remote_training_controller.py"),
                    "cancel",
                    "--status",
                    str(status_path),
                ],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            grace_seconds = max(float(grace_seconds), 45.0)
        except Exception as exc:
            cancel_error = "Could not start remote stop helper: {}".format(exc)
    pids = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        try:
            pid = int(job.get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in pids:
            pids.append(pid)

    def worker():
        deadline = time.time() + max(0.0, float(grace_seconds))
        while pids and time.time() < deadline:
            if all(not process_exists(pid) for pid in pids):
                break
            time.sleep(0.25)
        if remote and status_path:
            latest = read_json(status_path, {}) or {}
            if not bool(latest.get("remote_stop_confirmed")):
                latest.update({
                    "status": "stopping",
                    "phase": "remote_termination_pending",
                    "error": (
                        latest.get("error")
                        or "Waiting to reconnect and confirm the remote container stop."
                    ),
                    "updated_at_epoch": time.time(),
                })
                write_json_best_effort(status_path, latest)
                return
        killed = []
        for pid in pids:
            if process_exists(pid) and terminate_process_tree(pid):
                killed.append(pid)
        reap_deadline = time.time() + 10.0
        while pids and time.time() < reap_deadline:
            if all(not process_exists(pid) for pid in pids):
                break
            time.sleep(0.25)
        if not status_path:
            return
        latest = read_json(status_path, {}) or {}
        if str(latest.get("status") or "").lower() in (
            "completed",
            "failed",
            "cancelled",
            "abandoned",
        ):
            return
        latest_pids = []
        for key in ("pid", "controller_pid", "launcher_pid"):
            try:
                pid = int(latest.get(key) or 0)
            except Exception:
                pid = 0
            if pid > 0 and pid not in latest_pids:
                latest_pids.append(pid)
        if any(process_exists(pid) for pid in latest_pids):
            return
        if remote and not bool(latest.get("remote_stop_confirmed")):
            latest.update({
                "status": "stopping",
                "phase": "remote_termination_pending",
                "error": (
                    latest.get("error")
                    or "The remote container stop has not been confirmed."
                ),
                "updated_at_epoch": time.time(),
            })
            write_json_best_effort(status_path, latest)
            return
        latest.update({
            "status": "cancelled",
            "cancel_requested_at_epoch": time.time(),
            "cancelled_pids": killed,
            "updated_at_epoch": time.time(),
        })
        if cancel_error:
            latest["cancel_marker_error"] = cancel_error
        write_json_best_effort(status_path, latest)

    thread = threading.Thread(target=worker, name="fewshot-cancel")
    thread.daemon = True
    thread.start()
    return cancel_error


def can_abandon_remote_job(job):
    return bool(
        str(job.get("execution_backend") or "") == "remote"
        and job.get("remote_state_unknown")
        and str(job.get("status") or "").lower() not in {
            "completed", "failed", "cancelled", "paused", "abandoned"
        }
    )


def request_job_abandon_async(status_path):
    if not status_path:
        return "This task has no local status path."
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "remote_training_controller.py"),
                "abandon",
                "--status",
                str(status_path),
            ],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_process_kwargs()
        )
        return None
    except Exception as exc:
        return "Could not start local abandon helper: {}".format(exc)


def format_time(epoch):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except Exception:
        return "unknown"


def display_status(value):
    labels = {
        "launching": "Launching",
        "preparing": "Preparing data",
        "exporting_labels": "Exporting labels",
        "waiting_for_background_mimics": "Waiting for background Mimics",
        "waiting_for_gpu": "Waiting for GPU",
        "training": "Training",
        "running": "Running inference",
        "cancelling": "Cancelling",
        "stopping": "Stopping after an error",
        "finalizing": "Finalizing",
        "configuring": "Configuring training",
        "selecting_model": "Selecting model",
        "training_started": "Training started",
        "preparing_remote": "Preparing remote data",
        "connecting_remote": "Connecting to remote server",
        "uploading": "Uploading training data",
        "starting_remote": "Starting remote training",
        "waiting_for_remote_gpu": "Waiting for remote GPU",
        "reconnecting_remote": "Reconnecting to remote server",
        "remote_control_unavailable": "Remote Docker status unavailable",
        "finalizing_remote": "Preparing remote model for local use",
        "downloading": "Downloading trained model",
        "closed": "Closed",
        "cancelled": "Cancelled",
        "abandoned": "Abandoned locally",
        "failed": "Failed",
        "completed": "Completed",
    }
    return labels.get(str(value or ""), str(value or "Unknown").replace("_", " "))


def display_resource(value):
    labels = {
        "gpu": "GPU",
        "background_mimics": "background Mimics",
    }
    return labels.get(str(value or ""), str(value or "resource").replace("_", " "))


def display_kind(value):
    labels = {
        "train": "training",
        "infer": "prediction",
        "model_choice": "model selection",
        "train_setup": "training setup",
    }
    return labels.get(str(value or ""), str(value or "?").replace("_", " "))


def resource_wait_text(job):
    resource_wait = job.get("resource_wait") or {}
    if not resource_wait:
        return ""
    resource = display_resource(resource_wait.get("resource", "resource"))
    holder = resource_wait.get("owner", "unknown")
    pid = resource_wait.get("pid", "")
    if pid:
        text = "Waiting for {0}: {1} (PID {2})".format(resource, holder, pid)
    else:
        text = "Waiting for {0}".format(resource)
    action = str(resource_wait.get("user_action") or "").strip()
    return text + ("\n" + action if action else "")


def progress_line(progress):
    if not isinstance(progress, dict):
        return ""
    if progress.get("latest_epoch_line"):
        return str(progress.get("latest_epoch_line"))
    parts = []
    if progress.get("epoch") is not None and progress.get("epochs") is not None:
        parts.append("epoch {0}/{1}".format(progress.get("epoch"), progress.get("epochs")))
    if progress.get("phase"):
        parts.append(str(progress.get("phase")))
    if progress.get("batch") is not None and progress.get("batches") is not None:
        parts.append("batch {0}/{1}".format(progress.get("batch"), progress.get("batches")))
    metrics = progress.get("metrics") or {}
    if metrics.get("loss") is not None:
        try:
            parts.append("loss {0:.4f}".format(float(metrics.get("loss"))))
        except Exception:
            parts.append("loss {0}".format(metrics.get("loss")))
    if metrics.get("mean_dsc") is not None:
        try:
            parts.append("val_dice {0:.4f}".format(float(metrics.get("mean_dsc"))))
        except Exception:
            parts.append("val_dice {0}".format(metrics.get("mean_dsc")))
    if progress.get("best_dsc") is not None:
        try:
            parts.append("best {0:.4f}".format(float(progress.get("best_dsc"))))
        except Exception:
            parts.append("best {0}".format(progress.get("best_dsc")))
    if progress.get("lr") is not None:
        try:
            parts.append("lr {0:.2e}".format(float(progress.get("lr"))))
        except Exception:
            pass
    return ", ".join(parts)


def architecture_status_lines(job):
    plan = job.get("architecture_plan") or {}
    if not isinstance(plan, dict) or not plan:
        return []
    requested = str(plan.get("requested_dimension") or "auto").strip().lower()
    resolved = str(plan.get("resolved_dimension") or "").strip().upper()
    quality = str(plan.get("quality_mode") or "standard").strip().lower()
    requested_label = "Auto" if requested == "auto" else requested.upper()
    quality_label = "High detail" if quality == "high_detail" else "Standard"
    architecture = requested_label
    if resolved:
        architecture += " \u2192 {0}".format(resolved)
        if resolved == "3D":
            architecture += " {0}".format(quality_label)

    lines = ["Architecture: {0}".format(architecture)]
    resolved_batch = plan.get("resolved_batch_size")
    if resolved_batch is not None:
        source = str(plan.get("batch_size_source") or "").strip().lower()
        batch_text = "Batch size: {0}".format(resolved_batch)
        if source == "hardware_recommendation":
            memory = plan.get("gpu_memory_gb")
            if memory is not None:
                try:
                    batch_text += " (Auto for {0:g} GB GPU budget)".format(
                        float(memory)
                    )
                except Exception:
                    batch_text += " (Auto)"
            else:
                batch_text += " (Auto)"
        else:
            batch_text += " (user setting)"
        lines.append(batch_text)
    return lines


def user_status_lines(job):
    """Build the concise status summary shown to annotators."""
    lines = [
        "{0} - {1}".format(display_status(job.get("status")), display_kind(job.get("kind"))),
        "Organ: {0}".format(job.get("organ", "?")),
    ]
    if job.get("case_id"):
        lines.append("Case: {0}".format(job.get("case_id")))
    if str(job.get("execution_backend") or "") == "remote":
        lines.append(
            "Compute: {0}, GPU {1}".format(
                job.get("profile_name") or "remote server",
                job.get("remote_gpu_device") or "automatic",
            )
        )
        if job.get("dataset_cache_hit") is True:
            lines.append("Data transfer: reused verified remote cache.")
        elif job.get("dataset_cache_hit") is False and job.get("transfer_percent") is not None:
            speed = float(job.get("transfer_bytes_per_second") or 0.0)
            eta = job.get("transfer_eta_seconds")
            detail = ""
            if speed > 0:
                detail += " at {:.1f} MiB/s".format(speed / float(1024 ** 2))
            if eta is not None:
                detail += ", about {} min remaining".format(
                    max(1, int(round(float(eta) / 60.0)))
                )
            lines.append(
                "Data transfer: {0}%{1}.".format(
                    job.get("transfer_percent"), detail
                )
            )
        local_reused = job.get("local_materialization_cache_reused")
        local_total = job.get("local_materialization_cache_total")
        if local_reused is not None and local_total is not None:
            lines.append(
                "Local preparation: reused {0}/{1} source-grid cases.".format(
                    local_reused, local_total
                )
            )
        archive_reused = job.get("local_dataset_archive_cache_reused")
        archive_total = job.get("local_dataset_archive_cache_total")
        if archive_reused is not None and archive_total is not None:
            lines.append(
                "Local packaging: reused {0}/{1} case archives.".format(
                    archive_reused, archive_total
                )
            )
        remote_reused = job.get("materialization_cache_reused")
        remote_total = job.get("materialization_cache_total")
        if remote_reused is not None and remote_total is not None:
            lines.append(
                "Remote preparation: reused {0}/{1} prepared cases.".format(
                    remote_reused, remote_total
                )
            )
        if str(job.get("phase") or "") == "verifying_remote_dataset_cache":
            lines.append(
                "Remote cache check: {0}/{1} cases.".format(
                    job.get("dataset_case_completed") or 0,
                    job.get("dataset_case_total") or 0,
                )
            )
        if str(job.get("phase") or "") == "extracting_remote_dataset":
            lines.append(
                "Remote data access: {0}/{1} case archives ready.".format(
                    job.get("remote_dataset_extract_index") or 0,
                    job.get("remote_dataset_extract_total") or 0,
                )
            )
    strategy = job.get("strategy") or (job.get("training_options") or {}).get("strategy") or {}
    strategy_id = (strategy.get("preset") or strategy.get("id")) if isinstance(strategy, dict) else strategy
    if strategy_id:
        lines.append("Strategy: {0}".format(str(strategy_id).replace("_", " ")))
    lines.extend(architecture_status_lines(job))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        lines.append("Samples: {0} training, {1} validation".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        ))
    wait = resource_wait_text(job)
    if wait:
        lines.append(wait)
    progress = progress_line(job.get("training_progress") or {})
    if progress:
        lines.append("Progress: {0}".format(progress))
    export_progress = job.get("label_export_progress") or {}
    if isinstance(export_progress, dict) and export_progress:
        index = int(export_progress.get("index", 0) or 0)
        total = int(export_progress.get("total", 0) or 0)
        phase = str(export_progress.get("phase") or export_progress.get("status") or "running")
        lines.append(
            "Label export: {0}/{1} - {2}".format(index, total, phase)
            if total else "Label export: " + phase
        )
    application_message = str(job.get("application_message") or "").strip()
    if application_message:
        lines.append("Result: {0}".format(application_message))
    if job.get("error"):
        lines.append("Error: {0}".format(job.get("error")))
    elif job.get("status") == "completed":
        lines.append("Finished successfully.")
    return lines


def technical_status_lines(job):
    """Return diagnostics that are useful for support but noisy in the main view."""
    lines = [
        "Job ID: {0}".format(job.get("job_id", "?")),
        "Created: {0}".format(format_time(job.get("created_at_epoch"))),
        "Updated: {0}".format(format_time(job.get("updated_at_epoch"))),
    ]
    for label, key in (
        ("Training log", "train_log"),
        ("Remote controller log", "controller_log_path"),
        ("Metrics history", "metrics_history"),
        ("Inference log", "log"),
        ("Prediction", "output_path"),
    ):
        if job.get(key):
            lines.append("{0}: {1}".format(label, job.get(key)))
    model = job.get("model") or job.get("selected_model") or {}
    if isinstance(model, dict):
        if model.get("model_id"):
            lines.append("Model ID: {0}".format(model.get("model_id")))
        if model.get("checkpoint"):
            lines.append("Checkpoint: {0}".format(model.get("checkpoint")))
    architecture_plan = job.get("architecture_plan") or {}
    if isinstance(architecture_plan, dict) and architecture_plan.get("decoder"):
        lines.append(
            "Resolved decoder: {0}".format(architecture_plan.get("decoder"))
        )
        if architecture_plan.get("plan_sha256"):
            lines.append(
                "Architecture plan hash: {0}".format(
                    architecture_plan.get("plan_sha256")
                )
            )
    if job.get("training_seed") is not None:
        lines.append("Training seed: {0}".format(job.get("training_seed")))
    if job.get("training_fold") is not None:
        lines.append("Training fold: {0}".format(job.get("training_fold")))
    for label, key in (
        ("Training log warning", "train_log_warning"),
        ("Inference log warning", "log_warning"),
        ("Cancel marker warning", "cancel_marker_error"),
    ):
        if job.get(key):
            lines.append("{0}: {1}".format(label, job.get(key)))
    if job.get("train_log_unavailable"):
        lines.append("Training log is unavailable.")
    if job.get("log_unavailable"):
        lines.append("Inference log is unavailable.")
    return lines


def format_job_line(job):
    pieces = [
        job.get("job_id", "?"),
        display_kind(job.get("kind")),
        job.get("organ", "?"),
        display_status(job.get("status", "?")),
    ]
    if job.get("case_id"):
        pieces.append("case {0}".format(job.get("case_id")))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        pieces.append("train {0}, val {1}".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        ))
    wait = resource_wait_text(job)
    if wait:
        pieces.append(wait)
    progress = progress_line(job.get("training_progress") or {})
    if progress:
        pieces.append(progress)
    export_progress = job.get("label_export_progress") or {}
    if isinstance(export_progress, dict) and export_progress:
        index = int(export_progress.get("index", 0) or 0)
        total = int(export_progress.get("total", 0) or 0)
        phase = str(export_progress.get("phase") or export_progress.get("status") or "running")
        pieces.append(
            "labels {0}/{1} {2}".format(index, total, phase)
            if total else "labels " + phase
        )
    if job.get("error"):
        pieces.append("error: {0}".format(job.get("error")))
    if job.get("train_log_warning") or job.get("log_warning"):
        pieces.append("log warning")
    if job.get("train_log_unavailable") or job.get("log_unavailable"):
        pieces.append("log unavailable")
    if job.get("cancel_marker_error"):
        pieces.append("cancel marker warning: {0}".format(job.get("cancel_marker_error")))
    return " | ".join([str(part) for part in pieces if str(part)])


def filter_jobs(rows, filter_text, limit=80):
    filtered = []
    now = time.time()
    for mtime, payload in rows:
        status = payload.get("status")
        kind = payload.get("kind")
        is_active = status in ACTIVE_STATUSES
        if filter_text == "Training" and kind != "train":
            continue
        if filter_text == "Inference" and kind != "infer":
            continue
        if filter_text == "Failed / cancelled" and status not in ("failed", "cancelled", "cancelling"):
            continue
        if filter_text == "Active + recent":
            age_days = (now - float(payload.get("updated_at_epoch", mtime) or mtime)) / 86400.0
            if not is_active and status in ("failed", "cancelled") and age_days > 7.0:
                continue
            if not is_active and len(filtered) >= 25:
                continue
        filtered.append((mtime, payload))
    return [item[1] for item in filtered[:limit]]


def select_current_task(rows, organ="", job_id="", limit=25):
    """Return jobs for the selected organ, with the current task first.

    Previously this returned only a single job, which caused completed training
    progress and logs to disappear from the GUI as soon as a newer job (e.g.
    inference) became active.  Now all matching jobs are returned so the user
    can select any job in the list and inspect its progress, log, and curve.
    The most relevant job (active work > active > newest) is placed first.
    """
    organ = str(organ or "").strip().lower()
    job_id = str(job_id or "").strip()
    candidates = []
    for mtime, payload in rows:
        if job_id and str(payload.get("job_id") or "") != job_id:
            continue
        if organ and str(payload.get("organ") or "").strip().lower() != organ:
            continue
        candidates.append((mtime, payload))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Identify the "current" job to pin at the top of the list.
    active = [item for item in candidates if item[1].get("status") in ACTIVE_STATUSES]
    active_work = [
        item
        for item in active
        if item[1].get("kind") in ("train", "infer")
    ]
    primary = (active_work or active or candidates)[0]
    # Return primary first, then the rest in mtime order, deduplicating.
    seen_ids = set()
    result = []
    for item in [primary] + candidates:
        jid = item[1].get("job_id")
        if jid in seen_ids:
            continue
        seen_ids.add(jid)
        result.append(item[1])
        if len(result) >= limit:
            break
    return result


def process_exists(pid):
    return resource_process_exists(pid)


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


def open_path(path):
    if not path:
        return
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def launch_visible_gui_process(command, cwd=None, stderr_path=None):
    """Start an external GUI without Windows console-hiding flags."""
    launch_command = list(command)
    if os.name == "nt" and launch_command:
        executable = os.path.abspath(str(launch_command[0]))
        if os.path.basename(executable).lower() == "python.exe":
            pythonw = os.path.join(os.path.dirname(executable), "pythonw.exe")
            if os.path.isfile(pythonw):
                launch_command[0] = pythonw

    stderr_handle = None
    stderr_target = subprocess.DEVNULL
    if stderr_path:
        try:
            parent = os.path.dirname(os.path.abspath(stderr_path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            stderr_handle = open(stderr_path, "w", encoding="utf-8")
            stderr_target = stderr_handle
        except Exception:
            stderr_handle = None
            stderr_target = subprocess.DEVNULL
    try:
        return subprocess.Popen(
            launch_command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_target,
        )
    finally:
        if stderr_handle is not None:
            try:
                stderr_handle.close()
            except Exception:
                pass


def read_log_text(path, max_bytes=2 * 1024 * 1024, retries=3):
    if not path or not os.path.isfile(path):
        return ""
    for attempt in range(max(1, int(retries))):
        try:
            with open(path, "rb") as handle:
                try:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if size > int(max_bytes):
                        handle.seek(-int(max_bytes), os.SEEK_END)
                    else:
                        handle.seek(0)
                except Exception:
                    pass
                data = handle.read(int(max_bytes))
            for encoding in ("utf-8-sig", "utf-8", "mbcs", "cp936", "gbk"):
                try:
                    return data.decode(encoding)
                except Exception:
                    pass
            return data.decode("utf-8", "replace")
        except Exception:
            time.sleep(min(0.10, 0.03 * (attempt + 1)))
    return ""


def tail_text_from_text(text, max_lines=80):
    if not text:
        return ""
    lines = str(text).splitlines(True)
    return "".join(lines[-max_lines:])


def tail_text(path, max_lines=80):
    return tail_text_from_text(read_log_text(path), max_lines=max_lines)


def filter_log_for_job(text, job):
    job_id = str((job or {}).get("job_id") or "").strip()
    if not text or not job_id:
        return ""
    return "".join(line for line in str(text).splitlines(True) if job_id in line)


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s*train_loss=([0-9eE+\-.]+)(?:.*?val_dice=([0-9eE+\-.]+))?"
)


def parse_epoch_metrics_from_texts(*texts):
    rows = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for match in EPOCH_RE.finditer(str(text)):
            epoch = int(match.group(1))
            key = (epoch, match.group(2), match.group(3), match.group(4))
            if key in seen:
                continue
            seen.add(key)
            row = {
                "epoch": epoch,
                "epochs": int(match.group(2)),
                "train_loss": float(match.group(3)),
                "val_dice": None,
            }
            if match.group(4) is not None:
                row["val_dice"] = float(match.group(4))
            rows.append(row)
    rows.sort(key=lambda item: item["epoch"])
    return rows


def _number_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_epoch_metrics_from_history(path):
    payload = read_json(path, None)
    if isinstance(payload, list):
        history = payload
        epoch_count = None
    elif isinstance(payload, dict):
        history = payload.get("history") or payload.get("rows") or []
        epoch_count = payload.get("epoch_count")
    else:
        return []

    rows = []
    seen = set()
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            epoch = int(item.get("epoch"))
        except Exception:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        train_loss = _number_or_none(item.get("train_loss"))
        if train_loss is None:
            train_loss = _number_or_none(metrics.get("train_loss"))
        if train_loss is None:
            train_loss = _number_or_none(metrics.get("loss"))
        if train_loss is None:
            continue
        val_dice = _number_or_none(item.get("val_dice"))
        if val_dice is None:
            val_dice = _number_or_none(metrics.get("val_dice"))
        if val_dice is None:
            val_dice = _number_or_none(metrics.get("mean_dsc"))
        try:
            epochs = int(item.get("epochs") or epoch_count or epoch)
        except Exception:
            epochs = epoch
        key = (epoch, train_loss, val_dice)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "val_dice": val_dice,
        })
    rows.sort(key=lambda item: item["epoch"])
    return rows


def training_curve_rows(job, log_text="", pipeline_text=""):
    job = job or {}
    progress = job.get("training_progress") if isinstance(job.get("training_progress"), dict) else {}
    history_path = (
        job.get("metrics_history")
        or job.get("metrics_history_path")
        or progress.get("metrics_history")
        or progress.get("metrics_history_path")
    )
    rows = parse_epoch_metrics_from_history(history_path)
    if rows:
        return rows
    return parse_epoch_metrics_from_texts(log_text, pipeline_text)


def parse_epoch_metrics(*paths):
    texts = []
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        texts.append(read_log_text(path, max_bytes=4 * 1024 * 1024))
    return parse_epoch_metrics_from_texts(*texts)


class StatusViewerApp(object):
    def __init__(self, root, context):
        self.root = root
        self.context = context
        self.ts_root = os.path.abspath(context.get("ts_root", ""))
        self.workspace = os.path.abspath(context.get("workspace") or os.path.join(self.ts_root, "fewshot_models"))
        self.jobs_dir = os.path.join(self.workspace, "jobs")
        self.selected_job_id = ""
        self.jobs = []
        self.job_paths = {}
        self.listbox = None
        self.summary_var = None
        self.detail_text = None
        self.log_text = None
        self.technical_text = None
        self.chart = None
        self.stop_button = None
        self.retry_button = None
        self.edit_retry_button = None
        self.open_log_button = None
        self.filter_var = None
        self._refresh_after_id = None
        self._build()
        self.refresh()

    def _build(self):
        import tkinter as tk
        from tkinter import ttk

        self.root.title(TITLE)
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)
        try:
            self.root.attributes("-topmost", False)
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="DINOv3 Few-Shot Status", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        organ = self.context.get("selected_organ") or "selected organ"
        ttk.Label(header, text="Dataset: {0}    Task: {1}".format(self.ts_root, organ)).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(toolbar, text="Open Workspace", command=lambda: open_path(self.workspace)).pack(side="left", padx=(8, 0))
        self.open_log_button = ttk.Button(toolbar, text="Open Log Folder", command=self.open_log_folder)
        self.open_log_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(toolbar, text="Request Stop", command=self.request_stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Close", command=self.root.destroy).pack(side="right")

        self.summary_var = tk.StringVar(value="Loading jobs...")
        ttk.Label(outer, textvariable=self.summary_var).pack(fill="x")

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        body.add(left, weight=1)
        ttk.Label(left, text="Current task").pack(anchor="w")
        self.listbox = tk.Listbox(left, exportselection=False, height=28)
        job_scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=job_scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        job_scroll.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(body)
        body.add(right, weight=3)

        detail = ttk.LabelFrame(right, text="Selected activity", padding=8)
        detail.pack(fill="x")
        self.detail_text = tk.Text(detail, height=9, wrap="word", state="disabled")
        self.detail_text.pack(fill="x", expand=False)

        chart_box = ttk.LabelFrame(right, text="Training curve", padding=8)
        chart_box.pack(fill="x", pady=(8, 0))
        self.chart = tk.Canvas(chart_box, height=240, background="#f8fafc", highlightthickness=1, highlightbackground="#d1d5db")
        self.chart.pack(fill="x", expand=False)

        log_box = ttk.LabelFrame(right, text="Recent log", padding=8)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_box, height=14, wrap="none", state="disabled")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="left", fill="y")

    def refresh(self):
        old_selected = self.selected_job_id
        self.jobs, self.job_paths = self._load_jobs()
        self.listbox.delete(0, "end")
        selected_index = 0
        for index, job in enumerate(self.jobs):
            progress = progress_line(job.get("training_progress") or {})
            secondary = progress or job.get("application_message") or display_kind(job.get("kind"))
            label = "{0}   {1}\n{2}".format(
                display_status(job.get("status")),
                job.get("organ", "?"),
                secondary,
            )
            self.listbox.insert("end", label)
            if job.get("job_id") == old_selected:
                selected_index = index
        if self.jobs:
            self.listbox.selection_set(selected_index)
            self.listbox.activate(selected_index)
            self.selected_job_id = self.jobs[selected_index].get("job_id", "")
            self.show_job(self.jobs[selected_index])
            self.summary_var.set("Current task for {0}. Auto-refresh every 2 seconds.".format(
                self.context.get("selected_organ") or self.jobs[selected_index].get("organ", "selected organ")
            ))
        else:
            self.selected_job_id = ""
            self.summary_var.set("No DINOv3 few-shot jobs were found.")
            self._set_text(self.detail_text, "No jobs were found under:\n{0}".format(self.jobs_dir))
            self._set_text(self.log_text, "")
            self.draw_chart([])
        if self._refresh_after_id is not None:
            try:
                self.root.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self._refresh_after_id = self.root.after(2000, self.refresh)

    def _load_jobs(self):
        rows = []
        paths = {}
        root = Path(self.jobs_dir)
        if not root.is_dir():
            return [], {}
        for path in root.glob("*.json"):
            if path.name.endswith("_context.json"):
                continue
            payload = read_json(path, {}) or {}
            if not payload:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            rows.append((mtime, payload))
            if payload.get("job_id"):
                paths[payload.get("job_id")] = str(path)
        rows.sort(key=lambda item: item[0], reverse=True)
        return select_current_task(
            rows,
            self.context.get("selected_organ"),
            self.context.get("selected_job_id"),
        ), paths

    def on_select(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index < 0 or index >= len(self.jobs):
            return
        job = self.jobs[index]
        self.selected_job_id = job.get("job_id", "")
        self.show_job(job)

    def show_job(self, job):
        self._set_text(self.detail_text, "\n".join(user_status_lines(job)))

        log_path = job.get("train_log") or job.get("log")
        controller_log = job.get("controller_log_path")
        pipeline_log = os.path.join(self.workspace, "fewshot_pipeline.log")
        log_text = read_log_text(log_path, max_bytes=2 * 1024 * 1024)
        pipeline_text = ""
        log_tail = "\n".join(technical_status_lines(job))
        recent_log = tail_text_from_text(log_text, 80)
        if recent_log:
            log_tail += "\n\n---- recent task log ----\n" + recent_log
        if pipeline_log and pipeline_log != log_path:
            pipeline_text = filter_log_for_job(
                read_log_text(pipeline_log, max_bytes=1024 * 1024), job,
            )
            pipeline_tail = tail_text_from_text(pipeline_text, 60)
            if pipeline_tail:
                log_tail = (log_tail + "\n" if log_tail else "") + "---- pipeline log ----\n" + pipeline_tail
        controller_tail = tail_text(controller_log, 40)
        if controller_tail:
            log_tail += (
                "\n\n---- remote connection log ----\n" + controller_tail
            )
        self._set_text(self.log_text, log_tail, preserve_scroll=True)
        rows = training_curve_rows(job, log_text, pipeline_text)
        self.draw_chart(rows)
        try:
            abandonable = can_abandon_remote_job(job)
            self.stop_button.configure(
                text="Abandon Locally" if abandonable else "Request Stop"
            )
            state = (
                ["!disabled"]
                if abandonable or job.get("status") in CANCELLABLE_STATUSES
                else ["disabled"]
            )
            self.stop_button.state(state)
        except Exception:
            pass

    def _set_text(self, widget, text, preserve_scroll=False):
        old_position = None
        at_bottom = True
        if preserve_scroll:
            try:
                old_position = widget.yview()
                at_bottom = old_position[1] >= 0.995
            except Exception:
                old_position = None
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text or "")
        widget.configure(state="disabled")
        if preserve_scroll:
            try:
                if at_bottom:
                    widget.yview_moveto(1.0)
                elif old_position is not None:
                    widget.yview_moveto(old_position[0])
            except Exception:
                pass

    def draw_chart(self, rows):
        canvas = self.chart
        canvas.delete("all")
        width = max(320, int(canvas.winfo_width() or 900))
        height = 240
        canvas.create_rectangle(0, 0, width, height, fill="#f8fafc", outline="")

        margin_l, margin_r, margin_t, margin_b = 64, 54, 52, 40
        x0, y0 = margin_l, height - margin_b
        x1, y1 = width - margin_r, margin_t
        plot_w = max(1, x1 - x0)
        plot_h = max(1, y0 - y1)

        canvas.create_rectangle(x0, y1, x1, y0, fill="#ffffff", outline="#d1d5db")
        canvas.create_text(16, 18, anchor="w", text="Training Progress", fill="#111827", font=("Segoe UI", 10, "bold"))
        if not rows:
            canvas.create_text(width / 2, height / 2, text="No epoch metrics yet", fill="#6b7280", font=("Segoe UI", 10))
            return

        epochs = [row["epoch"] for row in rows]
        min_epoch, max_epoch = min(epochs), max(epochs)
        if min_epoch == max_epoch:
            max_epoch = min_epoch + 1
        losses = [row["train_loss"] for row in rows if row.get("train_loss") is not None]
        min_loss = min(losses) if losses else 0.0
        max_loss = max(losses) if losses else 1.0
        if min_loss == max_loss:
            pad = max(0.01, abs(min_loss) * 0.1)
            min_loss -= pad
            max_loss += pad
        else:
            pad = (max_loss - min_loss) * 0.12
            min_loss = max(0.0, min_loss - pad)
            max_loss += pad

        tick_count = 4
        for i in range(tick_count + 1):
            frac = i / float(tick_count)
            y = y0 - frac * plot_h
            loss_value = min_loss + frac * (max_loss - min_loss)
            dice_value = frac
            canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            canvas.create_text(x0 - 10, y, anchor="e", text="{0:.3g}".format(loss_value), fill="#991b1b", font=("Segoe UI", 8))
            canvas.create_text(x1 + 10, y, anchor="w", text="{0:.2f}".format(dice_value), fill="#1d4ed8", font=("Segoe UI", 8))

        epoch_ticks = sorted(set([min(epochs), max(epochs)] + [row["epoch"] for row in rows]))
        if len(epoch_ticks) > 6:
            step = max(1, int(round(len(epoch_ticks) / 5.0)))
            epoch_ticks = epoch_ticks[::step]
            if max(epochs) not in epoch_ticks:
                epoch_ticks.append(max(epochs))

        def x_at(epoch):
            return x0 + (float(epoch) - min_epoch) / float(max_epoch - min_epoch) * plot_w

        def y_loss(value):
            return y0 - (float(value) - min_loss) / float(max_loss - min_loss) * plot_h

        def y_dice(value):
            value = max(0.0, min(1.0, float(value)))
            return y0 - value * plot_h

        for epoch in epoch_ticks:
            x = x_at(epoch)
            canvas.create_line(x, y0, x, y0 + 4, fill="#9ca3af")
            canvas.create_text(x, y0 + 17, text=str(epoch), fill="#4b5563", font=("Segoe UI", 8))

        loss_points = []
        dice_points = []
        for row in rows:
            if row.get("train_loss") is not None:
                loss_points.extend([x_at(row["epoch"]), y_loss(row["train_loss"])])
            if row.get("val_dice") is not None:
                dice_points.extend([x_at(row["epoch"]), y_dice(row["val_dice"])])
        if len(loss_points) >= 4:
            canvas.create_line(*loss_points, fill="#dc2626", width=2, smooth=True)
        elif len(loss_points) == 2:
            canvas.create_oval(loss_points[0] - 3, loss_points[1] - 3, loss_points[0] + 3, loss_points[1] + 3, fill="#dc2626")
        if len(dice_points) >= 4:
            canvas.create_line(*dice_points, fill="#2563eb", width=2, smooth=True)
        elif len(dice_points) == 2:
            canvas.create_oval(dice_points[0] - 3, dice_points[1] - 3, dice_points[0] + 3, dice_points[1] + 3, fill="#2563eb")

        if len(loss_points) >= 2:
            lx, ly = loss_points[-2], loss_points[-1]
            canvas.create_oval(lx - 4, ly - 4, lx + 4, ly + 4, fill="#dc2626", outline="#ffffff", width=1)
        if len(dice_points) >= 2:
            dx, dy = dice_points[-2], dice_points[-1]
            canvas.create_oval(dx - 4, dy - 4, dx + 4, dy + 4, fill="#2563eb", outline="#ffffff", width=1)

        canvas.create_text(x0, height - 12, anchor="w", text="Epoch", fill="#4b5563", font=("Segoe UI", 8))
        canvas.create_text(x0 - 42, y1 - 18, anchor="w", text="Loss", fill="#991b1b", font=("Segoe UI", 8, "bold"))
        canvas.create_text(max(x0 + 40, x1 - 28), y1 - 18, anchor="w", text="Dice", fill="#1d4ed8", font=("Segoe UI", 8, "bold"))

        latest_loss = None
        latest_dice = None
        latest_epoch = epochs[-1]
        for row in reversed(rows):
            if latest_loss is None and row.get("train_loss") is not None:
                latest_loss = row.get("train_loss")
            if latest_dice is None and row.get("val_dice") is not None:
                latest_dice = row.get("val_dice")
        badges = [
            ("Epoch {0}/{1}".format(latest_epoch, max(epochs)), "#374151", "#f3f4f6"),
        ]
        if latest_loss is not None:
            badges.append(("loss {0:.4f}".format(float(latest_loss)), "#991b1b", "#fee2e2"))
        if latest_dice is not None:
            badges.append(("val dice {0:.4f}".format(float(latest_dice)), "#1d4ed8", "#dbeafe"))

        bx = min(150, max(16, width // 4))
        for text, color, fill in badges:
            tw = min(max(76, len(text) * 7 + 20), max(80, width - bx - 16))
            canvas.create_rectangle(bx, 12, bx + tw, 36, fill=fill, outline="#e5e7eb")
            canvas.create_text(bx + 10, 24, anchor="w", text=text, fill=color, font=("Segoe UI", 9))
            bx += tw + 8

        legend_x = max(x0 + 120, x1 - 178)
        canvas.create_line(legend_x, 24, legend_x + 24, 24, fill="#dc2626", width=2)
        canvas.create_text(legend_x + 32, 24, anchor="w", text="train loss", fill="#374151", font=("Segoe UI", 8))
        canvas.create_line(legend_x + 104, 24, legend_x + 128, 24, fill="#2563eb", width=2)
        canvas.create_text(legend_x + 136, 24, anchor="w", text="val dice", fill="#374151", font=("Segoe UI", 8))

    def selected_job(self):
        if not self.selected_job_id:
            return None
        for job in self.jobs:
            if job.get("job_id") == self.selected_job_id:
                return job
        return None

    def open_log_folder(self):
        job = self.selected_job()
        path = ""
        if job:
            path = job.get("train_log") or job.get("log") or ""
        if path and os.path.isfile(path):
            open_path(os.path.dirname(path))
            return
        open_path(self.workspace)

    def request_stop(self):
        job = self.selected_job()
        if not job:
            return
        status_path = self.job_paths.get(job.get("job_id"))
        if can_abandon_remote_job(job):
            from tkinter import messagebox

            confirmed = messagebox.askyesno(
                "Abandon Remote Task Locally",
                "Stop waiting on this workstation?\n\nThe server cannot confirm "
                "whether the container stopped. It may still use GPU or disk "
                "resources. An administrator must inspect the recorded "
                "container name.",
                parent=self.root,
            )
            if not confirmed:
                return
            error = request_job_abandon_async(status_path)
            if error:
                self.summary_var.set(error)
            return
        if job.get("status") not in CANCELLABLE_STATUSES:
            return
        cancel_error = request_job_cancel_async(job, status_path)
        job["status"] = "cancelling"
        job["cancel_requested_at_epoch"] = time.time()
        if cancel_error:
            job["cancel_marker_error"] = cancel_error
        job["updated_at_epoch"] = time.time()
        self.show_job(job)


def _load_pyside6():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class QtStatusViewerApp(object):
    """PySide6 implementation of the external read-only status viewer."""

    def __init__(self, window, context, qt_modules, curve_widget_cls):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        self.curve_widget_cls = curve_widget_cls
        self.ts_root = os.path.abspath(context.get("ts_root", ""))
        self.workspace = os.path.abspath(context.get("workspace") or os.path.join(self.ts_root, "fewshot_models"))
        self.jobs_dir = os.path.join(self.workspace, "jobs")
        self.selected_job_id = ""
        self.jobs = []
        self.job_paths = {}
        self.job_combo = None
        self.summary_label = None
        self.detail_text = None
        self.log_text = None
        self.chart = None
        self.stop_button = None
        self.open_log_button = None
        self.filter_combo = None
        self.auto_refresh = None
        self.live_label = None
        self.model_process = None
        self.model_action = ""
        self._refresh_running = False
        self._refresh_results = Queue()
        self.timer = self.QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.refresh)
        self.result_timer = self.QtCore.QTimer(self.window)
        self.result_timer.timeout.connect(self._poll_refresh_result)
        self._build()
        self.refresh()
        self.timer.start(2000)
        self.result_timer.start(80)

    def _build(self):
        QtWidgets = self.QtWidgets
        self.window.setWindowTitle(TITLE)
        self.window.resize(1060, 720)
        self.window.setMinimumSize(840, 600)
        self.window.setStyleSheet(self._stylesheet())
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QtWidgets.QLabel("DINOv3 Few-Shot Status")
        title.setObjectName("titleLabel")
        organ = self.context.get("selected_organ") or "selected organ"
        subtitle = QtWidgets.QLabel("Dataset: {0}    Task: {1}".format(self.ts_root, organ))
        subtitle.setObjectName("subtitleLabel")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        toolbar = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        toolbar.addWidget(refresh)
        open_workspace = QtWidgets.QPushButton("Open Workspace")
        open_workspace.clicked.connect(lambda: open_path(self.workspace))
        toolbar.addWidget(open_workspace)
        self.open_log_button = QtWidgets.QPushButton("Open Log Folder")
        self.open_log_button.clicked.connect(self.open_log_folder)
        toolbar.addWidget(self.open_log_button)
        self.stop_button = QtWidgets.QPushButton("Request Stop")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.request_stop)
        toolbar.addWidget(self.stop_button)
        self.retry_button = QtWidgets.QPushButton("Retry")
        self.retry_button.clicked.connect(self.retry_same_settings)
        toolbar.addWidget(self.retry_button)
        self.edit_retry_button = QtWidgets.QPushButton("Edit and Retry")
        self.edit_retry_button.clicked.connect(self.edit_and_retry)
        toolbar.addWidget(self.edit_retry_button)
        toolbar.addStretch(1)
        self.auto_refresh = QtWidgets.QCheckBox("Auto-refresh")
        self.auto_refresh.setChecked(True)
        self.auto_refresh.toggled.connect(self._toggle_auto_refresh)
        toolbar.addWidget(self.auto_refresh)
        self.live_label = QtWidgets.QLabel("Live")
        self.live_label.setObjectName("liveLabel")
        toolbar.addWidget(self.live_label)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.window.close)
        toolbar.addWidget(close)
        outer.addLayout(toolbar)

        model_toolbar = QtWidgets.QHBoxLayout()
        model_toolbar.addWidget(QtWidgets.QLabel("Model portability"))
        self.import_model_button = QtWidgets.QPushButton("Import Model Package")
        self.import_model_button.setToolTip(
            "Install a portable DINOv3 model package from another workstation."
        )
        self.import_model_button.clicked.connect(self.import_model_package)
        model_toolbar.addWidget(self.import_model_button)
        self.export_model_button = QtWidgets.QPushButton("Export Selected Model")
        self.export_model_button.setToolTip(
            "Export the model used by the selected task as a portable package."
        )
        self.export_model_button.clicked.connect(self.export_selected_model)
        model_toolbar.addWidget(self.export_model_button)
        model_toolbar.addStretch(1)
        outer.addLayout(model_toolbar)

        self.summary_label = QtWidgets.QLabel("Loading jobs...")
        outer.addWidget(self.summary_label)

        # Job selector combo so the user can switch between completed and
        # active jobs (e.g. review training progress after inference starts).
        selector_row = QtWidgets.QHBoxLayout()
        selector_label = QtWidgets.QLabel("Task history:")
        selector_row.addWidget(selector_label)
        self.job_combo = QtWidgets.QComboBox()
        self.job_combo.currentIndexChanged.connect(self.on_combo_select)
        selector_row.addWidget(self.job_combo, 1)
        outer.addLayout(selector_row)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        detail_group = QtWidgets.QGroupBox("Current status")
        detail_layout = QtWidgets.QVBoxLayout(detail_group)
        self.detail_text = QtWidgets.QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(150)
        detail_layout.addWidget(self.detail_text)
        right_layout.addWidget(detail_group)

        detail_tabs = QtWidgets.QTabWidget()
        progress_tab = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(progress_tab)
        self.chart = self.curve_widget_cls()
        self.chart.setMinimumHeight(230)
        chart_layout.addWidget(self.chart)
        detail_tabs.addTab(progress_tab, "Progress")

        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_tab)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        try:
            self.log_text.setFont(self.QtGui.QFont("Consolas", 9))
        except Exception:
            pass
        log_layout.addWidget(self.log_text)
        detail_tabs.addTab(log_tab, "Activity log")

        technical_tab = QtWidgets.QWidget()
        technical_layout = QtWidgets.QVBoxLayout(technical_tab)
        self.technical_text = QtWidgets.QTextEdit()
        self.technical_text.setReadOnly(True)
        self.technical_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        technical_layout.addWidget(self.technical_text)
        detail_tabs.addTab(technical_tab, "Technical details")
        right_layout.addWidget(detail_tabs, 1)
        outer.addWidget(right, 1)
        self.window.setCentralWidget(central)

    def _stylesheet(self):
        return shared_stylesheet()

    def _toggle_auto_refresh(self, enabled):
        if enabled:
            self.timer.start(2000)
            self.live_label.setText("Live")
            self.live_label.setObjectName("liveLabel")
        else:
            self.timer.stop()
            self.live_label.setText("Paused")
            self.live_label.setObjectName("subtitleLabel")
        self.live_label.style().unpolish(self.live_label)
        self.live_label.style().polish(self.live_label)

    def refresh(self):
        if self._refresh_running:
            return
        self._refresh_running = True

        def load_jobs():
            try:
                jobs, paths = self._load_jobs()
                self._refresh_results.put((jobs, paths, ""))
            except Exception as exc:
                self._refresh_results.put(([], {}, str(exc)))

        worker = threading.Thread(target=load_jobs, name="fewshot-status-refresh")
        worker.daemon = True
        worker.start()

    def _poll_refresh_result(self):
        try:
            jobs, paths, error = self._refresh_results.get_nowait()
        except Empty:
            return
        self._refresh_running = False
        if error:
            self.summary_label.setText("Could not refresh status: {0}".format(error))
            self.live_label.setText("Refresh failed")
            self.live_label.setObjectName("errorLabel")
            self.live_label.style().unpolish(self.live_label)
            self.live_label.style().polish(self.live_label)
            return
        self._apply_loaded_jobs(jobs, paths)
        if self.auto_refresh.isChecked():
            self.live_label.setText("Live - updated {0}".format(time.strftime("%H:%M:%S")))
            self.live_label.setObjectName("liveLabel")
            self.live_label.style().unpolish(self.live_label)
            self.live_label.style().polish(self.live_label)

    def _apply_loaded_jobs(self, jobs, paths):
        old_selected = self.selected_job_id
        self.jobs, self.job_paths = jobs, paths
        self.job_combo.blockSignals(True)
        self.job_combo.clear()
        selected_index = 0
        for index, job in enumerate(self.jobs):
            progress = progress_line(job.get("training_progress") or {})
            secondary = progress or job.get("application_message") or display_kind(job.get("kind"))
            label = "{0}  {1}  {2}".format(
                display_status(job.get("status")),
                job.get("organ", "?"),
                secondary,
            )
            self.job_combo.addItem(label)
            if job.get("job_id") == old_selected:
                selected_index = index
        self.job_combo.blockSignals(False)
        if self.jobs:
            self.job_combo.setCurrentIndex(selected_index)
            self.selected_job_id = self.jobs[selected_index].get("job_id", "")
            self.show_job(self.jobs[selected_index])
            self.summary_label.setText("Current task for {0}.".format(
                self.context.get("selected_organ") or self.jobs[selected_index].get("organ", "selected organ")
            ))
        else:
            self.selected_job_id = ""
            self.summary_label.setText("No DINOv3 few-shot jobs were found.")
            self.detail_text.setPlainText("No jobs were found under:\n{0}".format(self.jobs_dir))
            self.log_text.setPlainText("")
            self.technical_text.setPlainText("")
            self.chart.set_rows([])

    def _load_jobs(self):
        rows = []
        paths = {}
        root = Path(self.jobs_dir)
        if not root.is_dir():
            return [], {}
        for path in root.glob("*.json"):
            if path.name.endswith("_context.json"):
                continue
            payload = read_json(path, {}) or {}
            if not payload:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            rows.append((mtime, payload))
            if payload.get("job_id"):
                paths[payload.get("job_id")] = str(path)
        rows.sort(key=lambda item: item[0], reverse=True)
        selected = select_current_task(
            rows,
            self.context.get("selected_organ"),
            self.context.get("selected_job_id"),
        )
        return selected, paths

    def on_select(self, row):
        if row < 0 or row >= len(self.jobs):
            return
        job = self.jobs[row]
        self.selected_job_id = job.get("job_id", "")
        self.show_job(job)

    def on_combo_select(self, index):
        if index < 0 or index >= len(self.jobs):
            return
        job = self.jobs[index]
        self.selected_job_id = job.get("job_id", "")
        self.show_job(job)

    def selected_job(self):
        if not self.selected_job_id:
            return None
        for job in self.jobs:
            if job.get("job_id") == self.selected_job_id:
                return job
        return None

    def _start_model_process(self, arguments, action):
        if self.model_process is not None:
            return
        process = self.QtCore.QProcess(self.window)
        process.setProgram(sys.executable)
        process.setArguments([str(value) for value in arguments])
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(self.QtCore.QProcess.MergedChannels)
        process.finished.connect(self._model_process_finished)
        self.model_process = process
        self.model_action = action
        self.import_model_button.setEnabled(False)
        self.export_model_button.setEnabled(False)
        self.summary_label.setText(
            "{} is running outside Mimics. Other annotation work can continue.".format(
                action
            )
        )
        process.start()

    def _model_process_finished(self, exit_code, _exit_status):
        process = self.model_process
        output = ""
        if process is not None:
            output = bytes(process.readAllStandardOutput()).decode(
                "utf-8", "replace"
            ).strip()
            process.deleteLater()
        action = self.model_action
        self.model_process = None
        self.model_action = ""
        self.import_model_button.setEnabled(True)
        self.export_model_button.setEnabled(True)
        if int(exit_code) == 0:
            self.summary_label.setText(
                "{} completed. {}".format(
                    action, output.splitlines()[-1] if output else ""
                )
            )
            self.refresh()
        else:
            self.summary_label.setText(
                "{} failed. {}".format(action, output or "No diagnostic output.")
            )

    def import_model_package(self):
        choose_open_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Import DINOv3 Model Package",
            str(Path.home()),
            "AI model packages (*.zip)",
            self._import_model_package_path,
            button=self.import_model_button,
        )

    def _import_model_package_path(self, path):
        if not path:
            return
        self._start_model_process(
            [
                str(PROJECT_ROOT / "tools" / "ai_model_bundle.py"),
                "import-dinov3",
                "--workspace",
                self.workspace,
                "--bundle",
                path,
                "--set-latest",
            ],
            "Model import",
        )

    def export_selected_model(self):
        job = self.selected_job() or {}
        model = job.get("model") or job.get("selected_model") or {}
        model_id = str(model.get("model_id") or job.get("model_id") or "")
        organ = str(model.get("organ") or job.get("organ") or "")
        manifest_path = str(
            model.get("_manifest_path")
            or job.get("model_manifest")
            or ""
        )
        if not manifest_path and model_id and organ:
            manifest_path = os.path.join(
                self.workspace,
                "models",
                safe_slug(organ),
                safe_slug(model_id),
                "manifest.json",
            )
        if not manifest_path or not os.path.isfile(manifest_path):
            self.summary_label.setText(
                "The selected task does not reference an exportable trained model."
            )
            return
        suggested = Path.home() / "{}_{}.zip".format(
            safe_slug(organ), safe_slug(model_id or "model")
        )
        choose_save_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Export DINOv3 Model Package",
            str(suggested),
            "AI model packages (*.zip)",
            lambda output: self._export_selected_model_path(
                output, manifest_path
            ),
            button=self.export_model_button,
        )

    def _export_selected_model_path(self, output, manifest_path):
        if not output:
            return
        self._start_model_process(
            [
                str(PROJECT_ROOT / "tools" / "ai_model_bundle.py"),
                "export-dinov3",
                "--manifest",
                manifest_path,
                "--output",
                output,
            ],
            "Model export",
        )

    def show_job(self, job):
        self.detail_text.setPlainText("\n".join(user_status_lines(job)))

        log_path = job.get("train_log") or job.get("log")
        controller_log = job.get("controller_log_path")
        pipeline_log = os.path.join(self.workspace, "fewshot_pipeline.log")
        log_text = read_log_text(log_path, max_bytes=2 * 1024 * 1024)
        pipeline_text = ""
        log_tail = tail_text_from_text(log_text, 120)
        technical = "\n".join(technical_status_lines(job))
        if pipeline_log and pipeline_log != log_path:
            pipeline_text = filter_log_for_job(
                read_log_text(pipeline_log, max_bytes=1024 * 1024), job,
            )
            pipeline_tail = tail_text_from_text(pipeline_text, 60)
            if pipeline_tail:
                technical += ("\n\n" if technical else "") + "---- pipeline log ----\n" + pipeline_tail
        controller_tail = tail_text(controller_log, 40)
        if controller_tail:
            technical += (
                "\n\n---- remote connection log ----\n" + controller_tail
            )
        self._set_log_text(log_tail)
        self.technical_text.setPlainText(technical)
        rows = training_curve_rows(job, log_text, pipeline_text)
        self.chart.set_rows(rows)
        abandonable = can_abandon_remote_job(job)
        self.stop_button.setText(
            "Abandon Locally" if abandonable else "Request Stop"
        )
        self.stop_button.setEnabled(
            abandonable or job.get("status") in CANCELLABLE_STATUSES
        )
        has_retry = bool(job.get("retry_context") and job.get("training_options"))
        another_active = any(
            item is not job and item.get("status") in ACTIVE_STATUSES for item in self.jobs
        )
        retryable = job.get("kind") == "train" and job.get("status") in (
            "failed", "cancelled", "completed",
        )
        self.retry_button.setEnabled(bool(has_retry and retryable and not another_active))
        self.edit_retry_button.setEnabled(bool(has_retry and retryable and not another_active))
        if another_active:
            hint = "Wait for the active DINOv3 task to finish or stop it before retrying."
        elif not has_retry:
            hint = "This older task does not contain the setup context required for one-click retry."
        else:
            hint = ""
        self.retry_button.setToolTip(hint)
        self.edit_retry_button.setToolTip(hint)

    def _set_log_text(self, text):
        if self.log_text.toPlainText() == (text or ""):
            return
        scrollbar = self.log_text.verticalScrollBar()
        old_value = scrollbar.value()
        at_bottom = old_value >= scrollbar.maximum() - 3
        self.log_text.setPlainText(text or "")
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(old_value, scrollbar.maximum()))

    def open_log_folder(self):
        job = self.selected_job()
        path = ""
        if job:
            path = job.get("train_log") or job.get("log") or ""
        if path and os.path.isfile(path):
            open_path(os.path.dirname(path))
            return
        open_path(self.workspace)

    def request_stop(self):
        job = self.selected_job()
        if not job:
            return
        status_path = self.job_paths.get(job.get("job_id"))
        if can_abandon_remote_job(job):
            answer = self.QtWidgets.QMessageBox.warning(
                self.window,
                "Abandon Remote Task Locally",
                "Stop waiting on this workstation?\n\nThe server cannot confirm "
                "whether the container stopped. It may still use GPU or disk "
                "resources. An administrator must inspect the recorded "
                "container name.",
                self.QtWidgets.QMessageBox.Yes
                | self.QtWidgets.QMessageBox.No,
                self.QtWidgets.QMessageBox.No,
            )
            if answer != self.QtWidgets.QMessageBox.Yes:
                return
            error = request_job_abandon_async(status_path)
            self.summary_label.setText(
                error or "Local monitoring is being abandoned."
            )
            return
        if job.get("status") not in CANCELLABLE_STATUSES:
            return
        cancel_error = request_job_cancel_async(job, status_path)
        job["status"] = "cancelling"
        job["cancel_requested_at_epoch"] = time.time()
        if cancel_error:
            job["cancel_marker_error"] = cancel_error
        job["updated_at_epoch"] = time.time()
        self.show_job(job)

    def _retry_context(self, job):
        context = dict(job.get("retry_context") or {})
        context["organ"] = job.get("organ") or context.get("organ")
        context["ts_root"] = job.get("ts_root") or context.get("ts_root")
        context["workspace"] = job.get("workspace") or context.get("workspace")
        return context

    def retry_same_settings(self):
        job = self.selected_job()
        if not job:
            return
        answer = self.QtWidgets.QMessageBox.question(
            self.window,
            "Retry Training",
            "Start a new training run with the same saved data and parameters?",
            self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
            self.QtWidgets.QMessageBox.No,
        )
        if answer != self.QtWidgets.QMessageBox.Yes:
            return
        try:
            try:
                import fewshot_training_setup_ui as setup_ui
            except ImportError:
                from tools import fewshot_training_setup_ui as setup_ui
            run_id, _status_path, _pid = setup_ui.launch_training(
                self._retry_context(job),
                dict(job.get("training_options") or {}),
            )
            self.summary_label.setText("Retry started: {0}".format(run_id))
            self.refresh()
        except Exception as exc:
            self.QtWidgets.QMessageBox.warning(
                self.window, "Retry Could Not Start", str(exc),
            )

    def edit_and_retry(self):
        job = self.selected_job()
        if not job:
            return
        try:
            context = self._retry_context(job)
            context["initial_options"] = dict(job.get("training_options") or {})
            setup_id = "setup_retry_{0}_{1}".format(
                time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8],
            )
            jobs_dir = os.path.join(context.get("workspace") or self.workspace, "jobs")
            os.makedirs(jobs_dir, exist_ok=True)
            context_path = os.path.join(jobs_dir, setup_id + "_context.json")
            context["setup_id"] = setup_id
            context["setup_status_path"] = os.path.join(jobs_dir, setup_id + ".json")
            context["setup_stderr_path"] = os.path.join(
                jobs_dir, setup_id + "_stderr.log",
            )
            write_json_atomic(context_path, context)
            write_json_atomic(
                context["setup_status_path"],
                {
                    "schema_version": "mimics_fewshot_setup.v1",
                    "job_id": setup_id,
                    "kind": "train_setup",
                    "status": "configuring",
                    "organ": context.get("organ"),
                    "ts_root": context.get("ts_root"),
                    "workspace": context.get("workspace"),
                    "context_path": context_path,
                    "stderr_log": context.get("setup_stderr_path"),
                    "created_at_epoch": time.time(),
                    "updated_at_epoch": time.time(),
                },
            )
            script = str(Path(__file__).with_name("fewshot_training_setup_ui.py"))
            launch_visible_gui_process(
                [sys.executable, script, "--context", context_path],
                cwd=context.get("project_root") or None,
                stderr_path=context.get("setup_stderr_path"),
            )
            self.summary_label.setText("Training settings opened with the previous run prefilled.")
        except Exception as exc:
            self.QtWidgets.QMessageBox.warning(
                self.window, "Settings Could Not Open", str(exc),
            )


def run_pyside6_ui(context):
    qt_modules = _load_pyside6()
    QtCore, QtGui, QtWidgets = qt_modules

    class CurveWidget(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.rows = []

        def sizeHint(self):
            return QtCore.QSize(620, 260)

        def minimumSizeHint(self):
            return QtCore.QSize(420, 220)

        def set_rows(self, rows):
            self.rows = list(rows or [])
            self.update()

        def paintEvent(self, _event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            rect = self.rect()
            painter.setClipRect(rect)
            width = max(320, rect.width())
            height = max(190, rect.height())
            painter.fillRect(rect, QtGui.QColor("#ffffff"))
            margin_l, margin_r, margin_t, margin_b = 60, 52, 62, 40
            x0, y0 = margin_l, height - margin_b
            x1, y1 = width - margin_r, margin_t
            plot_w = max(1, x1 - x0)
            plot_h = max(1, y0 - y1)
            painter.setPen(QtGui.QPen(QtGui.QColor("#d1d5db"), 1))
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawRect(QtCore.QRectF(x0, y1, plot_w, plot_h))
            if not self.rows:
                painter.setPen(QtGui.QColor("#6b7280"))
                painter.drawText(rect, QtCore.Qt.AlignCenter, "No epoch metrics yet")
                return
            epochs = [row["epoch"] for row in self.rows]
            min_epoch, max_epoch = min(epochs), max(epochs)
            if min_epoch == max_epoch:
                max_epoch = min_epoch + 1
            losses = [row["train_loss"] for row in self.rows if row.get("train_loss") is not None]
            min_loss = min(losses) if losses else 0.0
            max_loss = max(losses) if losses else 1.0
            if min_loss == max_loss:
                pad = max(0.01, abs(min_loss) * 0.1)
                min_loss -= pad
                max_loss += pad
            else:
                pad = (max_loss - min_loss) * 0.12
                min_loss = max(0.0, min_loss - pad)
                max_loss += pad

            def x_at(epoch):
                return x0 + (float(epoch) - min_epoch) / float(max_epoch - min_epoch) * plot_w

            def y_loss(value):
                return y0 - (float(value) - min_loss) / float(max_loss - min_loss) * plot_h

            def y_dice(value):
                return y0 - max(0.0, min(1.0, float(value))) * plot_h

            for i in range(5):
                frac = i / 4.0
                y = y0 - frac * plot_h
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 1))
                painter.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
                painter.setPen(QtGui.QColor("#991b1b"))
                painter.drawText(8, int(y + 4), "{0:.3g}".format(min_loss + frac * (max_loss - min_loss)))
                painter.setPen(QtGui.QColor("#1d4ed8"))
                painter.drawText(int(x1 + 8), int(y + 4), "{0:.2f}".format(frac))

            epoch_ticks = sorted(set([min(epochs), max(epochs)] + [row["epoch"] for row in self.rows]))
            if len(epoch_ticks) > 6:
                step = max(1, int(round(len(epoch_ticks) / 5.0)))
                epoch_ticks = epoch_ticks[::step]
                if max(epochs) not in epoch_ticks:
                    epoch_ticks.append(max(epochs))
            painter.setPen(QtGui.QColor("#4b5563"))
            for epoch in epoch_ticks:
                x = x_at(epoch)
                painter.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, y0 + 4))
                painter.drawText(int(x - 8), y0 + 22, str(epoch))

            loss_points = []
            dice_points = []
            for row in self.rows:
                if row.get("train_loss") is not None:
                    loss_points.append(QtCore.QPointF(x_at(row["epoch"]), y_loss(row["train_loss"])))
                if row.get("val_dice") is not None:
                    dice_points.append(QtCore.QPointF(x_at(row["epoch"]), y_dice(row["val_dice"])))
            if len(loss_points) >= 2:
                painter.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 2))
                painter.drawPolyline(QtGui.QPolygonF(loss_points))
            if len(dice_points) >= 2:
                painter.setPen(QtGui.QPen(QtGui.QColor("#2563eb"), 2))
                painter.drawPolyline(QtGui.QPolygonF(dice_points))
            for points, color in ((loss_points, "#dc2626"), (dice_points, "#2563eb")):
                if points:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
                    painter.setBrush(QtGui.QColor(color))
                    p = points[-1]
                    painter.drawEllipse(p, 4, 4)
            latest_loss = next((row.get("train_loss") for row in reversed(self.rows) if row.get("train_loss") is not None), None)
            latest_dice = next((row.get("val_dice") for row in reversed(self.rows) if row.get("val_dice") is not None), None)
            summary = ["Epoch {0}".format(epochs[-1])]
            if latest_loss is not None:
                summary.append("train loss {0:.4f}".format(float(latest_loss)))
            if latest_dice is not None:
                summary.append("validation Dice {0:.4f}".format(float(latest_dice)))
            painter.setPen(QtGui.QColor("#344054"))
            painter.drawText(
                QtCore.QRectF(16, 10, max(1, width - 32), 22),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                "   |   ".join(summary),
            )
            legend_x = 16
            painter.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 2))
            painter.drawLine(legend_x, 44, legend_x + 24, 44)
            painter.setPen(QtGui.QColor("#374151"))
            painter.drawText(legend_x + 32, 49, "train loss")
            painter.setPen(QtGui.QPen(QtGui.QColor("#2563eb"), 2))
            painter.drawLine(legend_x + 112, 44, legend_x + 136, 44)
            painter.setPen(QtGui.QColor("#374151"))
            painter.drawText(legend_x + 144, 49, "validation Dice")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, TITLE)
    window = QtWidgets.QMainWindow()
    QtStatusViewerApp(window, context, qt_modules, CurveWidget)
    window.show()
    return app.exec()


def run_ui(context):
    backend = os.environ.get("MIMICS_DINOV3_GUI_BACKEND", "auto").strip().lower()
    if backend in ("", "auto", "pyside6", "qt"):
        try:
            return run_pyside6_ui(context)
        except Exception:
            if backend in ("pyside6", "qt"):
                raise
    try:
        import tkinter as tk
    except Exception as exc:
        raise RuntimeError(
            "The external DINOv3 status window could not open because neither PySide6 nor Tkinter is "
            "available in the configured external Python environment: {0}".format(exc)
        )
    root = tk.Tk()
    StatusViewerApp(root, context)
    root.mainloop()


def generate_preview(path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required only for --preview PNG generation. Runtime status viewing does not "
            "depend on Pillow. Import error: {0}".format(exc)
        )

    width, height = 1280, 900
    image = Image.new("RGB", (width, height), "#f4f5f7")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 30)
        head_font = ImageFont.truetype("Arial.ttf", 19)
        font = ImageFont.truetype("Arial.ttf", 16)
        small = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        title_font = head_font = font = small = ImageFont.load_default()

    draw.rectangle((0, 0, width, 70), fill="#1f2937")
    draw.text((32, 20), "DINOv3 Few-Shot Status", fill="white", font=title_font)
    draw.text((32, 92), "Dataset: D:\\Dataset\\TotalSegmentator    Organ filter: liver", fill="#1f2937", font=head_font)
    draw.text((32, 143), "Show", fill="#374151", font=font)
    draw.rectangle((82, 132, 250, 170), outline="#9ca3af", fill="#ffffff")
    draw.text((98, 143), "Active + recent", fill="#111827", font=font)
    for idx, label in enumerate(["Refresh", "Open Workspace", "Open Log Folder", "Request Stop"]):
        x = 270 + idx * 155
        draw.rectangle((x, 132, x + 140, 170), outline="#9ca3af", fill="#ffffff")
        draw.text((x + 16, 143), label, fill="#111827", font=font)

    draw.text((32, 194), "12 item(s), 1 active. Auto-refresh every 2 seconds.", fill="#374151", font=font)
    draw.rectangle((32, 225, 420, height - 35), outline="#d1d5db", fill="#ffffff")
    draw.text((50, 244), "Recent activity", fill="#111827", font=head_font)
    rows = [
        ("Training", "training", "liver", "train_20260708_01", True),
        ("Selecting model", "model selection", "liver", "choose_model_s0401", False),
        ("Completed", "training", "spleen", "train_20260707_03", False),
        ("Completed", "prediction", "liver", "infer_s0401_liver", False),
        ("Failed", "training", "kidney_left", "train_20260706_02", False),
    ]
    y = 282
    for status, kind, organ, job_id, selected in rows:
        if selected:
            draw.rectangle((44, y - 6, 408, y + 31), fill="#dbeafe")
        draw.text((58, y), "{0}   {1}   {2}".format(status, kind, organ), fill="#111827", font=font)
        draw.text((58, y + 18), job_id, fill="#6b7280", font=small)
        y += 52

    right_x = 450
    draw.rectangle((right_x, 225, width - 32, 392), outline="#d1d5db", fill="#ffffff")
    draw.text((right_x + 18, 244), "Selected activity", fill="#111827", font=head_font)
    details = [
        "ID: train_20260708_01",
        "Type: training",
        "Organ: liver",
        "Status: Training",
        "Samples: train 8, validation 2",
        "Progress: Epoch 6/10: train_loss=0.3182, val_dice=0.7421",
        "Train log: ...\\fewshot_models\\runs\\liver\\train_20260708_01\\train.log",
    ]
    y = 276
    for line in details:
        draw.text((right_x + 18, y), line, fill="#374151", font=font)
        y += 18

    chart_y = 420
    draw.rectangle((right_x, chart_y, width - 32, chart_y + 240), outline="#d1d5db", fill="#f8fafc")
    draw.text((right_x + 18, chart_y + 16), "Training curve", fill="#111827", font=head_font)
    badge_x = right_x + 185
    for label, color, fill, w in [
        ("Epoch 6/10", "#374151", "#f3f4f6", 100),
        ("loss 0.3182", "#991b1b", "#fee2e2", 110),
        ("val dice 0.7421", "#1d4ed8", "#dbeafe", 135),
    ]:
        draw.rectangle((badge_x, chart_y + 11, badge_x + w, chart_y + 37), outline="#e5e7eb", fill=fill)
        draw.text((badge_x + 10, chart_y + 17), label, fill=color, font=small)
        badge_x += w + 8

    gx0, gy0, gx1, gy1 = right_x + 78, chart_y + 190, width - 92, chart_y + 70
    draw.rectangle((gx0, gy1, gx1, gy0), outline="#d1d5db", fill="#ffffff")
    for i in range(5):
        y_tick = gy0 - i * (gy0 - gy1) / 4.0
        draw.line((gx0, y_tick, gx1, y_tick), fill="#e5e7eb")
        draw.text((gx0 - 48, y_tick - 7), ["0.26", "0.31", "0.36", "0.41", "0.46"][i], fill="#991b1b", font=small)
        draw.text((gx1 + 10, y_tick - 7), ["0.00", "0.25", "0.50", "0.75", "1.00"][i], fill="#1d4ed8", font=small)
    draw.text((gx0 - 42, gy1 - 24), "Loss", fill="#991b1b", font=small)
    draw.text((gx1 + 18, gy1 - 24), "Dice", fill="#1d4ed8", font=small)
    loss_points = [(gx0, gy1 + 22), (gx0 + 120, gy1 + 42), (gx0 + 240, gy1 + 60), (gx0 + 360, gy1 + 78), (gx0 + 480, gy1 + 96)]
    dice_points = [(gx0, gy0 - 30), (gx0 + 120, gy0 - 55), (gx0 + 240, gy0 - 74), (gx0 + 360, gy0 - 88), (gx0 + 480, gy0 - 104)]
    draw.line(loss_points, fill="#dc2626", width=3)
    draw.line(dice_points, fill="#2563eb", width=3)
    for point, color in [(loss_points[-1], "#dc2626"), (dice_points[-1], "#2563eb")]:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#ffffff")
    draw.text((gx0, gy0 + 14), "1", fill="#4b5563", font=small)
    draw.text((gx0 + 240, gy0 + 14), "3", fill="#4b5563", font=small)
    draw.text((gx0 + 480, gy0 + 14), "6", fill="#4b5563", font=small)
    draw.line((gx1 - 170, chart_y + 28, gx1 - 146, chart_y + 28), fill="#dc2626", width=3)
    draw.text((gx1 - 138, chart_y + 20), "train loss", fill="#374151", font=small)
    draw.line((gx1 - 58, chart_y + 28, gx1 - 34, chart_y + 28), fill="#2563eb", width=3)
    draw.text((gx1 - 26, chart_y + 20), "val dice", fill="#374151", font=small)

    log_y = 690
    draw.rectangle((right_x, log_y, width - 32, height - 35), outline="#d1d5db", fill="#ffffff")
    draw.text((right_x + 18, log_y + 16), "Recent log", fill="#111827", font=head_font)
    logs = [
        "[2026-07-08 14:20:03] Training job train_20260708_01 started for organ liver.",
        "[2026-07-08 14:22:11] Training progress: Epoch 4/10: train_loss=0.3921, val_dice=0.7015",
        "[2026-07-08 14:24:38] Training progress: Epoch 5/10: train_loss=0.3480, val_dice=0.7288",
        "[2026-07-08 14:27:02] Training progress: Epoch 6/10: train_loss=0.3182, val_dice=0.7421",
    ]
    y = log_y + 52
    for line in logs:
        draw.text((right_x + 18, y), line, fill="#374151", font=small)
        y += 24

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", help="Path to the status viewer context JSON written by Mimics.")
    parser.add_argument("--ts-root")
    parser.add_argument("--workspace")
    parser.add_argument("--organ")
    parser.add_argument("--preview", help="Write a static PNG preview of the status UI and exit.")
    args = parser.parse_args(argv)
    if args.preview:
        generate_preview(args.preview)
        print(args.preview)
        return 0
    if args.context:
        context = read_json(args.context, None)
        if not context:
            raise RuntimeError("Could not read status viewer context: {0}".format(args.context))
    elif args.ts_root:
        context = {
            "ts_root": os.path.abspath(args.ts_root),
            "workspace": os.path.abspath(args.workspace) if args.workspace else os.path.join(os.path.abspath(args.ts_root), "fewshot_models"),
            "selected_organ": args.organ or "",
        }
    else:
        parser.error("--context or --ts-root is required unless --preview is used")
    run_ui(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
