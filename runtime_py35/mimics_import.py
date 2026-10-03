# -*- coding: utf-8 -*-
"""Mimics-internal driver for dataset -> .mcs import.

Runs inside Mimics Python 3.5.2. Only uses stdlib + mimics API.
Calls mimics_bridge.py (in nninteractive_env) via subprocess for NIfTI/DICOM work.

Uses Win32 SetTimer / PyQt5 QTimer for non-blocking async polling,
matching nnInteractive's pattern; no manual second click needed.

Flow:
    1. Annotator picks a case directory
    2. Launch mimics_bridge.py "prepare" in background (non-blocking)
    3. Timer polls every 0.5s until bridge completes
    4. Save a prepare manifest and let background Mimics create .mcs
    5. For batch: timer processes cases one-by-one automatically
"""

from __future__ import print_function

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import mimics

import runtime_common


# -- Global async monitor state ----------------------------------------
_IMPORT_MONITORS = {}
_BRIDGE_LAUNCH_ERRORS = {}
_BRIDGE_PROCESSES = {}
_BRIDGE_EXIT_SEEN = {}

# Track background Mimics process for .mcs creation
_BG_MIMICS_BUSY_NOTICE_AT = 0.0
_BG_MIMICS_RETRY_OUTPUTS = set()
_BG_MIMICS_PROCESSES = {}
_BG_MIMICS_LAUNCH_OUTPUTS = set()
_BG_MIMICS_FAILED_OUTPUTS = set()
_BG_MIMICS_STATE_LOCK = threading.RLock()
_MCS_QUEUE_ACTIVE = "_mcs_queue_active.json"
_MCS_QUEUE_DONE = "_mcs_queue_done.json"
_MCS_QUEUE_STOP = "_mcs_queue_stop.json"
_LOG_ROTATE_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_BACKUPS = 3
_CONFIG_CACHE = None
_LAST_TASK_DESCRIPTOR = {}


def _background_output_key(output_dir):
    return os.path.normcase(os.path.abspath(output_dir))


def _bg_state_snapshot(output_dir):
    key = _background_output_key(output_dir)
    with _BG_MIMICS_STATE_LOCK:
        state = dict(_BG_MIMICS_PROCESSES.get(key) or {})
        state.setdefault("pid", None)
        state.setdefault("output_dir", os.path.abspath(output_dir))
        state["launch_active"] = key in _BG_MIMICS_LAUNCH_OUTPUTS
        return state


def _bg_try_begin_launch(output_dir):
    key = _background_output_key(output_dir)
    with _BG_MIMICS_STATE_LOCK:
        if key in _BG_MIMICS_LAUNCH_OUTPUTS:
            return False
        _BG_MIMICS_LAUNCH_OUTPUTS.add(key)
        return True


def _bg_end_launch(output_dir):
    key = _background_output_key(output_dir)
    with _BG_MIMICS_STATE_LOCK:
        _BG_MIMICS_LAUNCH_OUTPUTS.discard(key)


def _bg_set_process(pid, output_dir):
    key = _background_output_key(output_dir)
    with _BG_MIMICS_STATE_LOCK:
        _BG_MIMICS_PROCESSES[key] = {
            "pid": pid,
            "output_dir": os.path.abspath(output_dir),
        }


def _bg_clear_process(pid):
    with _BG_MIMICS_STATE_LOCK:
        for key, state in list(_BG_MIMICS_PROCESSES.items()):
            if state.get("pid") == pid:
                _BG_MIMICS_PROCESSES.pop(key, None)
                return True
    return False


def _bg_failed_contains(output_key):
    with _BG_MIMICS_STATE_LOCK:
        return output_key in _BG_MIMICS_FAILED_OUTPUTS


def _bg_mark_failed(output_key):
    with _BG_MIMICS_STATE_LOCK:
        _BG_MIMICS_FAILED_OUTPUTS.add(output_key)


def _bg_retry_add(output_key):
    with _BG_MIMICS_STATE_LOCK:
        if output_key in _BG_MIMICS_RETRY_OUTPUTS:
            return False
        _BG_MIMICS_RETRY_OUTPUTS.add(output_key)
        return True


def _bg_retry_discard(output_key):
    with _BG_MIMICS_STATE_LOCK:
        _BG_MIMICS_RETRY_OUTPUTS.discard(output_key)


def _rt(output_dir, *parts):
    """Return the local control directory for an output queue.

    No queue/status JSON is written to the selected output share. This avoids
    SMB rename failures and prevents network latency from stalling Mimics timer
    callbacks. The output folder contains only final .mcs files.
    """
    return os.path.join(
        runtime_common.import_queue_runtime_dir(_project_root(), output_dir),
        *parts
    )


def _new_import_run_root():
    """Create an isolated local directory for one import invocation.

    Bridge control JSON and large derived DICOM/buffer files must not live on
    the selected output share. SMB rename semantics and reused case names made
    separate imports interfere with each other there.
    """
    root = os.path.join(
        runtime_common.import_runtime_base(_project_root()),
        "import_runs",
        "{0}_{1}_{2}".format(time.strftime("%Y%m%dT%H%M%S"), os.getpid(), uuid.uuid4().hex[:10]),
    )
    os.makedirs(root)
    return root


def _write_import_task_status(path, values):
    if not path:
        return
    try:
        payload = runtime_common.read_json(path, {}) or {}
        payload.update(values or {})
        payload["updated_at_epoch"] = time.time()
        runtime_common.write_json_atomic(path, payload)
    except Exception:
        # Progress telemetry must never fail or delay the import itself.
        pass


def _import_task_stopped(monitor):
    path = monitor.get("task_stop_path") if monitor else ""
    return bool(path and os.path.isfile(path))


def _clear_stale_queue_stop(output_dir):
    """Clear a prior queue stop only after its producer and creator exited."""
    stop_path = _queue_stop_path(output_dir)
    if not os.path.isfile(stop_path):
        return
    # Some Mimics Python builds can become unstable around short-lived Win32
    # lock probes (msvcrt.locking) during UI-driven startup. For this stale
    # stop cleanup, a non-locking read is sufficient and avoids that code path.
    holder = {}
    try:
        lock_path = runtime_common.background_mimics_lock_path(
            _project_root(), output_dir
        )
        payload = runtime_common.read_json(lock_path, {}) or {}
        if isinstance(payload, dict):
            holder = payload
    except Exception:
        holder = {}
    producer = {}
    try:
        producer_path = runtime_common.import_producer_lock_path(
            _project_root(), output_dir
        )
        payload = runtime_common.read_json(producer_path, {}) or {}
        if isinstance(payload, dict):
            producer = payload
    except Exception:
        producer = {}
    if producer and runtime_common.process_matches(
        producer.get("pid"), producer.get("process_start_marker")
    ):
        raise RuntimeError(
            "The previous import is still responding to its stop request. "
            "Wait for preparation to exit before starting another import into "
            "this output folder."
        )
    if holder and str(holder.get("kind") or "").lower() == "create_mcs":
        holder_output = str(holder.get("output_dir") or "")
        if (
            holder_output
            and os.path.normcase(os.path.abspath(holder_output))
            == os.path.normcase(os.path.abspath(output_dir))
            and runtime_common.process_matches(
                holder.get("pid"), holder.get("process_start_marker")
            )
        ):
            raise RuntimeError(
                "The previous import queue is still stopping. Wait until its "
                "background Mimics process exits before starting another import "
                "into this output folder."
            )
    try:
        os.remove(stop_path)
    except OSError:
        if os.path.isfile(stop_path):
            raise RuntimeError(
                "The previous import stop marker could not be cleared: {0}".format(
                    stop_path
                )
            )


def _set_last_import_task(run_root, output_dir, title):
    global _LAST_TASK_DESCRIPTOR
    _checkpoint_record("set_last_task_enter", run_root=run_root, output_dir=output_dir, title=title)
    status_path = os.path.join(run_root, "status.json")
    task_stop_path = os.path.join(run_root, "stop.json")
    _checkpoint_record("set_last_task_paths_ready", status_path=status_path, task_stop_path=task_stop_path)
    queue_stop_path = _queue_stop_path(output_dir)
    _checkpoint_record("set_last_task_queue_stop_ready", queue_stop_path=queue_stop_path)
    _clear_stale_queue_stop(output_dir)
    _checkpoint_record("set_last_task_after_clear_stale", output_dir=output_dir)
    _LAST_TASK_DESCRIPTOR = {
        "kind": "import",
        "title": str(title),
        "status_path": status_path,
        "secondary_status_path": _rt(output_dir, "_mcs_batch_status.json"),
        "stop_path": task_stop_path,
        "stop_paths": [task_stop_path, queue_stop_path],
        "log_path": _rt(output_dir, "logs", "mimics_import.log"),
        "output_path": os.path.abspath(output_dir),
    }
    _checkpoint_record("set_last_task_done", descriptor_keys=sorted(_LAST_TASK_DESCRIPTOR.keys()))
    return status_path, task_stop_path


def _prepared_queue_dir(output_dir):
    return _rt(output_dir, "prepared_queue")


def _publish_prepared_work(output_dir, case_id, work_dir, output_mcs):
    """Publish a small queue descriptor after the local manifest is durable."""
    queue_dir = _prepared_queue_dir(output_dir)
    if not os.path.isdir(queue_dir):
        os.makedirs(queue_dir)
    descriptor = os.path.join(
        queue_dir,
        "{0}_{1}.json".format(_safe_case_filename(case_id), uuid.uuid4().hex),
    )
    _write_json_atomic(
        descriptor,
        {
            "case_id": str(case_id or "case"),
            "work_dir": os.path.abspath(work_dir),
            "output_mcs": os.path.abspath(output_mcs),
            "created_at_epoch": time.time(),
        },
    )
    return descriptor


def _auto_open_mcs_enabled():
    return os.environ.get("MIMICS_IMPORT_AUTO_OPEN_MCS", "").strip().lower() in ("1", "true", "yes")


_write_json_atomic = runtime_common.write_json_atomic
_safe_case_filename = runtime_common.safe_filename
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs


def _allow_windows_event_monitor():
    """Opt-in switch for Mimics event timer usage on Windows.

    Certain Mimics versions can emit Subscription.__del__ AttributeError after
    explicit unsubscribe. Prefer Win32 SetTimer by default and only enable
    Mimics event subscriptions when this flag is set.
    """
    value = os.environ.get("MIMICS_USE_EVENT_TIMER", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _checkpoint_enabled():
    # Crash breadcrumbs are cheap and purely diagnostic; always keep them on.
    return True


def _prune_old_checkpoints(out_dir, retention_days=14):
    """Delete breadcrumb files older than ``retention_days`` (best effort).

    Every Mimics import PID leaves a checkpoint pair under debug_out; without
    pruning these accumulate forever. Failure here must never affect the
    import workflow.
    """
    try:
        cutoff = time.time() - retention_days * 86400
        for name in os.listdir(out_dir):
            if not name.startswith("mimics_import_checkpoint_"):
                continue
            path = os.path.join(out_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                continue
    except Exception:
        pass


def _checkpoint_record(stage, **fields):
    """Write last-stage breadcrumbs for crash diagnosis.

    Mimics host crashes can terminate this script before UI-facing status files
    update. Keep a best-effort local breadcrumb trail under debug_out.
    """
    if not _checkpoint_enabled():
        return
    try:
        root = runtime_common.import_runtime_base(_project_root())
        out_dir = os.path.join(root, "debug_out")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        _prune_old_checkpoints(out_dir)
        record = {
            "stage": str(stage),
            "pid": int(os.getpid()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(),
        }
        for key, value in (fields or {}).items():
            try:
                json.dumps(value)
                record[str(key)] = value
            except Exception:
                record[str(key)] = str(value)
        runtime_common.write_json_atomic(
            os.path.join(out_dir, "mimics_import_checkpoint_{0}.json".format(os.getpid())),
            record,
        )
        with open(
            os.path.join(out_dir, "mimics_import_checkpoint_{0}.jsonl".format(os.getpid())),
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _startup_cleanup_enabled():
    """Whether to run startup stale cleanup from import entry.

    This cleanup touches Win32 lock probing and process checks. Keep it opt-in
    for import startup to avoid host instability on certain Mimics builds.
    """
    value = os.environ.get("MIMICS_IMPORT_STARTUP_CLEANUP", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _write_json_quick(path, value):
    """Best-effort atomic JSON write for local queue control state."""
    runtime_common.write_json_atomic(path, value)


def _mimics_log(level, message):
    # Some Mimics builds are unstable when log_user_message is called during
    # import workflows. Keep it disabled by default and allow explicit opt-in.
    enabled = os.environ.get("MIMICS_IMPORT_USE_MIMICS_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if not enabled:
        return False
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return True
    except Exception:
        return False


def _safe_message_box(title, message, ui_blocking=True):
    """Show a Mimics dialog, catching any API failures gracefully."""
    try:
        mimics.dialogs.message_box(title=title, message=message, ui_blocking=ui_blocking)
    except TypeError:
        try:
            mimics.dialogs.message_box(title=title, message=message)
        except Exception:
            pass
    except Exception:
        pass


def _user_progress(level, message):
    """Publish concise lifecycle transitions to the Mimics log panel."""
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return
    except Exception:
        print(message)


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _rotate_log_file(path, max_bytes=_LOG_ROTATE_BYTES, backups=_LOG_ROTATE_BACKUPS):
    return runtime_common.rotate_log_file(path, max_bytes=max_bytes, backups=backups)


def _append_import_log(root_dir, message):
    try:
        thread_name = threading.current_thread().name
    except Exception:
        thread_name = "unknown"
    text = "[{0}] [pid={1}] [thread={2}] {3}".format(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.getpid(),
        thread_name,
        message,
    )
    # Mimics API calls are not thread-safe. Only attempt Mimics logging on the
    # main thread and only when explicitly enabled by environment variable.
    logged_to_mimics = False
    try:
        if threading.current_thread() is threading.main_thread():
            logged_to_mimics = _mimics_log(logging.INFO, text)
    except Exception:
        logged_to_mimics = False
    if not logged_to_mimics:
        print(text)
    try:
        if root_dir and not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        path = _rt(root_dir or os.getcwd(), "logs", "mimics_import.log")
        _rotate_log_file(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _verbose_log_enabled():
    return os.environ.get("MIMICS_IMPORT_VERBOSE_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _verbose_log(root_dir, message):
    """Log detailed debug info only when verbose logging is explicitly enabled."""
    if _verbose_log_enabled():
        _append_import_log(root_dir, message)


def _append_import_exception(root_dir, context, exc=None):
    try:
        if exc is None:
            detail = traceback.format_exc()
        else:
            detail = "{0}\n{1}".format(repr(exc), traceback.format_exc())
        _append_import_log(root_dir, "{0} | exception={1}".format(context, detail))
    except Exception:
        pass


def _summarize_bridge_params(params):
    try:
        action = params.get("action", "")
        summary = {
            "action": action,
            "case_id": params.get("case_id", ""),
            "ts_root": params.get("ts_root", ""),
            "case_dir": params.get("case_dir", ""),
            "image_path": params.get("image_path", ""),
            "mask_count": len(params.get("masks", []) or []),
            "axes": params.get("axes", []),
            "flips": params.get("flips", []),
        }
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "<bridge-param-summary-unavailable>"


def _queue_active_path(output_dir):
    return _rt(output_dir, _MCS_QUEUE_ACTIVE)


def _queue_done_path(output_dir):
    return _rt(output_dir, _MCS_QUEUE_DONE)


def _queue_stop_path(output_dir):
    return _rt(output_dir, _MCS_QUEUE_STOP)


def _acquire_import_producer_lease(output_dir, owner):
    """Own preparation writes for one shared output queue without waiting."""
    lock_path = runtime_common.import_producer_lock_path(
        _project_root(), output_dir
    )
    token = runtime_common.acquire_resource_lock(
        lock_path,
        "import_producer",
        owner,
        wait_seconds=0.0,
    )
    if not token:
        return None
    return {
        "producer_lock_path": lock_path,
        "producer_lock_token": token,
        "producer_output_dir": os.path.abspath(output_dir),
    }


def _release_import_producer_lease(state):
    """Release a producer lease stored on a task/monitor dict exactly once."""
    if not isinstance(state, dict):
        return False
    lock_path = str(state.get("producer_lock_path") or "")
    token = str(state.get("producer_lock_token") or "")
    state["producer_lock_token"] = ""
    if not lock_path or not token:
        return False
    return bool(runtime_common.release_resource_lock(lock_path, token))


def _finish_import_producer(state, completed=0, failed=0):
    """Close an active queue producer and release its cross-entry lock."""
    output_dir = str(
        (state or {}).get("producer_output_dir")
        or (state or {}).get("output_dir")
        or ""
    )
    if output_dir and (state or {}).get("queue_marked_active"):
        _mark_mcs_queue_done(output_dir, completed=completed, failed=failed)
        state["queue_marked_active"] = False
    return _release_import_producer_lease(state)


def _mcs_queue_registry_dir():
    return os.path.join(runtime_common.import_runtime_base(_project_root()), "mcs_queues")


def _register_mcs_queue(output_dir, total_count=0):
    # This small local registry makes per-row Stop in Task Status work even while a
    # prepared queue is waiting for the background Mimics license.
    try:
        registry_dir = _mcs_queue_registry_dir()
        if not os.path.isdir(registry_dir):
            os.makedirs(registry_dir)
        digest = runtime_common.stable_digest_hex(os.path.abspath(output_dir))[:16]
        base = _safe_case_filename(os.path.basename(os.path.abspath(output_dir))).strip("._") or "queue"
        name = "{0}_{1}".format(base, digest)
        path = os.path.join(registry_dir, name + ".json")
        _write_json_quick(
            path,
            {
                "output_dir": os.path.abspath(output_dir),
                "runtime_dir": _rt(output_dir),
                "total_count": int(total_count or 0),
                "updated_at_epoch": time.time(),
            },
        )
    except Exception:
        # Registry failures must never affect import flow.
        pass


IMPORT_QUEUE_RETENTION_DAYS = 14


def _queue_dir_has_live_consumer(queue_dir):
    """True when any lock JSON in the queue dir names a live pid."""
    try:
        for name in os.listdir(queue_dir):
            if not name.endswith(".lock"):
                continue
            payload = runtime_common.read_json(
                os.path.join(queue_dir, name), {}
            ) or {}
            pid = payload.get("pid") or payload.get("acquiring_pid")
            if pid and runtime_common.process_exists(pid):
                return True
    except OSError:
        return True  # unreadable: assume busy, never delete
    return False


def _prune_import_queues(max_age_days=IMPORT_QUEUE_RETENTION_DAYS):
    """Delete control directories of long-idle import queues.

    Mirrors the export-side _prune_local_export_jobs precedent. A queue
    dir is removable when (a) its active-marker heartbeat is older than
    max_age_days, or it has no heartbeat at all AND the dir mtime itself
    is that old (the fallback covers dirs caught between creation and the
    first heartbeat write), and (b) no consumer lock inside it names a
    live pid, and (c) any *.lock.guard inside it is uncontended
    (non-blocking try-acquire; a contender disqualifies the dir). The
    matching mcs_queues registry entry is removed with it, so the Stop
    Import Queue path cannot resurrect the directory.
    """
    base = os.path.join(
        runtime_common.import_runtime_base(_project_root()), "import_queues"
    )
    if not os.path.isdir(base):
        return 0
    now = time.time()
    cutoff = now - float(max_age_days) * 86400.0
    registry_dir = _mcs_queue_registry_dir()
    removed = 0
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    for name in names:
        queue_dir = os.path.join(base, name)
        if not os.path.isdir(queue_dir):
            continue
        try:
            active = runtime_common.read_json(
                os.path.join(queue_dir, _MCS_QUEUE_ACTIVE), {}
            ) or {}
            heartbeat = float(active.get("updated_at_epoch") or 0)
        except Exception:
            heartbeat = 0.0
        try:
            dir_age = now - os.path.getmtime(queue_dir)
        except OSError:
            continue
        stale = (
            heartbeat and now - heartbeat >= float(max_age_days) * 86400.0
        ) or (
            not heartbeat and dir_age >= float(max_age_days) * 86400.0
        )
        if not stale:
            continue
        if _queue_dir_has_live_consumer(queue_dir):
            continue
        # Same try-acquire rule the guard sweep applies: never delete a
        # guard someone may be opening right now.
        try:
            import resource_locks
        except Exception:
            continue
        contended = False
        for entry in os.listdir(queue_dir):
            if not entry.endswith(".guard"):
                continue
            guard_path = os.path.join(queue_dir, entry)
            try:
                handle = open(guard_path, "a+b")
                try:
                    # Seek to byte 0: "a+b" opens at end-of-file and
                    # msvcrt.locking/flock lock from the current position.
                    handle.seek(0)
                    resource_locks._try_lock_byte(handle)
                    resource_locks._unlock_byte(handle)
                finally:
                    handle.close()
            except Exception:
                contended = True
                break
        if contended:
            continue
        shutil.rmtree(queue_dir, ignore_errors=True)
        if not os.path.isdir(queue_dir):
            removed += 1
        # Drop the matching registry entry (same digest naming as
        # _register_mcs_queue) so stale registrations cannot resurrect
        # the directory via a stop marker.
        try:
            for reg_name in os.listdir(registry_dir):
                if not reg_name.endswith(".json"):
                    continue
                payload = runtime_common.read_json(
                    os.path.join(registry_dir, reg_name), {}
                ) or {}
                if os.path.normcase(
                    str(payload.get("runtime_dir") or "")
                ) == os.path.normcase(queue_dir):
                    try:
                        os.remove(os.path.join(registry_dir, reg_name))
                    except OSError:
                        pass
        except OSError:
            pass
    return removed


def _mark_mcs_queue_active(output_dir, total_count=0):
    _verbose_log(output_dir, "Queue active | output_dir={0} | total_count={1}".format(output_dir, int(total_count or 0)))
    # Import launch is the same cheap user-action moment the export side
    # uses for its prune: keep the queues directory bounded without any
    # background timer of its own.
    try:
        _prune_import_queues()
    except Exception:
        pass
    rt_dir = _rt(output_dir)
    if not os.path.isdir(rt_dir):
        os.makedirs(rt_dir)
    done_path = _queue_done_path(output_dir)
    try:
        if os.path.isfile(done_path):
            os.remove(done_path)
    except Exception:
        pass
    _write_json_quick(
        _queue_active_path(output_dir),
        {
            "status": "active",
            "output_dir": os.path.abspath(output_dir),
            "total_count": int(total_count or 0),
            "updated_at_epoch": time.time(),
        },
    )
    _verbose_log(output_dir, "Queue active written | path={0}".format(_queue_active_path(output_dir)))
    _register_mcs_queue(output_dir, total_count)
    _verbose_log(output_dir, "Queue active complete")


def _heartbeat_mcs_queue_active(output_dir):
    """Refresh an existing producer marker from a worker thread."""
    active_path = _queue_active_path(output_dir)
    if not os.path.isfile(active_path):
        return False
    try:
        payload = runtime_common.read_json(active_path, {}) or {}
        if str(payload.get("status") or "").lower() != "active":
            return False
        payload["updated_at_epoch"] = time.time()
        _write_json_quick(active_path, payload)
        return True
    except Exception:
        return False


def _mark_mcs_queue_done(output_dir, completed=0, failed=0):
    _verbose_log(output_dir, "Queue done | output_dir={0} | completed={1} | failed={2}".format(output_dir, int(completed or 0), int(failed or 0)))
    rt_dir = _rt(output_dir)
    if not os.path.isdir(rt_dir):
        os.makedirs(rt_dir)
    _write_json_quick(
        _queue_done_path(output_dir),
        {
            "status": "done",
            "output_dir": os.path.abspath(output_dir),
            "completed": int(completed or 0),
            "failed": int(failed or 0),
            "updated_at_epoch": time.time(),
        },
    )
    _verbose_log(output_dir, "Queue done written | path={0}".format(_queue_done_path(output_dir)))
    try:
        active = _queue_active_path(output_dir)
        if os.path.isfile(active):
            os.remove(active)
    except Exception:
        pass
    _register_mcs_queue(output_dir, completed + failed)
    _verbose_log(output_dir, "Queue done complete")


def _background_stop_requested(output_dir, since_epoch=0.0):
    try:
        stop_path = _queue_stop_path(output_dir)
        if not os.path.isfile(stop_path):
            return False
        if since_epoch and os.path.getmtime(stop_path) < float(since_epoch):
            return False
        return True
    except Exception:
        return False


def _save_prepare_manifest(work_dir, result, output_mcs=None):
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    manifest = dict(result)
    if output_mcs:
        manifest["output_mcs"] = os.path.abspath(output_mcs)
    path = os.path.join(work_dir, "prepare_manifest.json")
    _write_json_atomic(path, manifest)
    return path


def _record_failed_case(output_dir, case_id, phase, error, source_image=""):
    """Record one failed case; source_image is the user's original input
    path so the failure record names the real data location (R61-5)."""
    try:
        failed_dir = _rt(output_dir or os.getcwd(), "_failed_cases")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir)
        payload = {
            "case_id": str(case_id or "unknown"),
            "phase": str(phase or "unknown"),
            "error": str(error or ""),
            "failed_at_epoch": time.time(),
        }
        if source_image:
            payload["source_image"] = str(source_image)
        filename = "{0}_{1}.json".format(
            _safe_case_filename(case_id),
            _safe_case_filename(phase),
        )
        _write_json_atomic(os.path.join(failed_dir, filename), payload)
    except Exception:
        pass


def _error_guidance(error_text, phase=""):
    """Map a raw import error to a plain-English category and action.

    Returns (category, message, suggested_action) - all strings. category is
    one of: environment_broken, disk_full, network_unavailable,
    source_data_invalid, background_mimics_failed, unknown. Used by the
    failure dialogs so every error the annotator can see comes with a
    concrete next step.
    """
    text = str(error_text or "").lower()
    phase_text = str(phase or "").lower()
    combined = text + " " + phase_text
    if (
        "no module named" in combined
        or "importerror" in combined
        or "modulenotfounderror" in combined
        or "python_env" in combined and "missing" in combined
        or "environment" in combined and "broken" in combined
    ):
        return (
            "environment_broken",
            "数据导入 Python 环境不完整或已损坏。",
            "运行 管理菜单 > 环境设置/修复，然后重试导入。",
        )
    if (
        "no space left" in combined
        or "disk full" in combined
        or "not enough disk" in combined
        or "errno 28" in combined
    ):
        return (
            "disk_full",
            "导入过程中磁盘空间已耗尽。",
            "释放磁盘空间（在导入输出磁盘上，或选择其他输出目录）后重试。"
            "失败病例逐例记录在本次导入的 _failed_cases 目录中；"
            "可通过任务状态窗口（01_Data > 04）打开查看。",
        )
    if (
        "network path not found" in combined
        or "network name cannot be found" in combined
        or "unavailable" in combined and "network" in combined
        or "the network" in combined
        or "share" in combined and "not accessible" in combined
    ):
        return (
            "network_unavailable",
            "存放输入数据的网络驱动器或共享无法访问。",
            "重新连接网络驱动器并确认可在资源管理器中打开，"
            "然后重试导入。",
        )
    if (
        "not a valid" in combined and ("nifti" in combined or "image" in combined)
        or "corrupt" in combined
        or "empty" in combined and ("image" in combined or "mask" in combined)
        or "geometry" in combined and "mismatch" in combined
    ):
        return (
            "source_data_invalid",
            "某个输入文件已损坏，或几何信息不一致。",
            "检查失败病例（其原始源路径已记录在失败详情中）；"
            "重新导出或排除该病例，然后重试其余病例。",
        )
    if (
        "background mimics" in combined
        or "mimics.exe" in combined
        or "could not start" in combined and "mimics" in combined
    ):
        return (
            "background_mimics_failed",
            "后台 Mimics 实例无法创建 .mcs 文件。",
            "重试导入。若再次失败，请关闭其他 Mimics 窗口，运行 "
            "管理菜单 > 环境设置/修复，或查看消息中显示的后台 Mimics 日志。",
        )
    return (
        "unknown",
        str(error_text or "未知错误"),
        "重试导入。若持续失败，请查看导入输出目录的 logs 子目录以及"
        "为失败病例保留的诊断文件，或联系支持人员。",
    )


# -- Path helpers (same pattern as nninteractive_mimics.py) -------------


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def _load_data_io_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    merged = {}
    io_path = os.path.join(_project_root(), "mimics_io_config.json")
    nn_path = os.path.join(_project_root(), "nninteractive_config.json")
    for path in (io_path, nn_path):
        try:
            with open(path, "r") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                merged.update(loaded)
        except Exception:
            pass
    _CONFIG_CACHE = merged
    return _CONFIG_CACHE


def _resolve_import_output_dir(base_dir, create=True):
    default_dir = os.path.join(base_dir, "mcs_output")
    config = _load_data_io_config()
    configured = config.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if not os.path.isabs(configured):
        configured = os.path.abspath(os.path.join(base_dir, configured))
    else:
        configured = os.path.abspath(configured)
    if not create:
        return configured
    try:
        if not os.path.isdir(configured):
            os.makedirs(configured)
        return configured
    except Exception:
        return default_dir


def _resource_lock_path(name):
    return runtime_common.resource_lock_path(_project_root(), name)


def _resource_lock_dir():
    return runtime_common.resource_lock_dir(_project_root())


def _environment_root():
    root = _project_root()
    candidates = [
        os.path.join(root, "python_env"),
        os.path.join(os.path.dirname(root), "python_env"),
        os.path.join(root, "nninteractive_env"),
        os.path.join(os.path.dirname(root), "nninteractive_env"),
        root,
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "python.exe")):
            return candidate
        if os.path.isfile(os.path.join(candidate, "Scripts", "python.exe")):
            return candidate
        if os.path.isfile(os.path.join(candidate, "python", "python.exe")):
            return candidate
    return candidates[0]


def _bridge_script():
    """Find mimics_bridge.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "mimics_bridge.py"),
        os.path.join(here, "mimics_bridge.py"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return os.path.abspath(candidates[0])


def _background_env(extra=None):
    return runtime_common.background_env(extra)


def _python_exe():
    found = runtime_common.find_external_python(_project_root())
    if found:
        return found
    # Preserve the historical behavior: return the default path even when
    # missing so callers can report a clear setup error.
    return os.path.join(_environment_root(), "python.exe")


# -- Stale process / temp cleanup --------------------------------------

def _cleanup_stale_processes():
    """Clean safe stale runtime state from a previous crashed session.

    The process registry sweep runs first: it clears dead records,
    terminates registered orphans (parent gone, kill-on-sweep policy) and
    releases locks they still hold - proving ownership via PID + start
    marker, so no aggressive flag is needed for those.

    The registry sweep is the only terminator here. Killing other live
    bridge/background Mimics processes by command-line heuristics was
    retired (it could interrupt a valid async workflow); use the
    explicit Stop All Owned Services entry for that.
    """
    locks_removed = runtime_common.cleanup_stale_resource_locks(_resource_lock_dir())
    registry_summary = runtime_common.sweep_processes(
        runtime_common.project_root()
    )
    registry_killed = len(registry_summary.get("terminated_orphans") or [])
    locks_removed += len(registry_summary.get("released_locks") or [])
    if locks_removed or registry_killed:
        print(
            "Startup cleanup: removed {0} stale resource lock file(s); "
            "terminated {1} orphaned registered process(es)".format(
                locks_removed, registry_killed
            )
        )


# -- TS case discovery (runs in Mimics Python 3.5, stdlib only) ---------
# Layout rules (preferred image names, mask directories, excluded output
# folders) come from dataset_profiles.json via the shared profile loader.

import dataset_profiles as _dataset_profiles

_MEDICAL_IMAGE_SUFFIXES = tuple(_dataset_profiles.load_profile()["fallback_image_suffixes"])
_MASK_SUFFIXES = tuple(_dataset_profiles.load_profile()["mask_suffixes"])


def _is_medical_image_file(path):
    name = os.path.basename(path).lower()
    return os.path.isfile(path) and any(name.endswith(suffix) for suffix in _MEDICAL_IMAGE_SUFFIXES)


def _image_stem(path):
    name = os.path.basename(path)
    lower = name.lower()
    for suffix in _MEDICAL_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[:-len(suffix)] or "case"
    return os.path.splitext(name)[0] or "case"


def _find_case_image(case_dir, allow_direct_dicom=False, profile=None):
    profile = profile or _dataset_profiles.load_profile()
    for img_name in profile["image_candidates"]:
        candidate = os.path.join(case_dir, img_name)
        if _is_medical_image_file(candidate):
            return candidate, "medical_image"
    nonempty = False
    try:
        with os.scandir(case_dir) as entries:
            for index, entry in enumerate(entries):
                nonempty = True
                lower = entry.name.lower()
                if any(lower.endswith(suffix) for suffix in _MEDICAL_IMAGE_SUFFIXES):
                    try:
                        if entry.is_file():
                            return entry.path, "medical_image"
                    except OSError:
                        pass
                if allow_direct_dicom and (
                    lower.endswith(".dcm") or index >= 511
                ):
                    return case_dir, "dicom_candidate"
    except OSError:
        pass
    for dicom_name in profile["dicom_dirs"]:
        dicom_dir = os.path.join(case_dir, dicom_name)
        if os.path.isdir(dicom_dir):
            return dicom_dir, "dicom"
    # A directly selected folder may itself be a flat DICOM series. Validation
    # happens in the external bridge so Mimics never parses all DICOM headers.
    if allow_direct_dicom and nonempty:
        return case_dir, "dicom_candidate"
    return None, None

def discover_ts_cases(ts_root, case_filter=None, profile_id=None):
    """Find all cases in a TS-like dataset. Returns list of dicts."""
    profile = _dataset_profiles.load_profile(profile_id)
    mask_dirs = list(profile["mask_dirs"]) or ["segmentations"]
    cases = []
    for name in sorted(os.listdir(ts_root)):
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name in profile["exclude_dirs"]:
            continue
        if case_filter and name not in case_filter:
            continue

        image_path = None
        image_type = None

        image_path, image_type = _find_case_image(case_dir, profile=profile)

        if image_path is None:
            continue

        masks = []
        for mask_dir_name in mask_dirs:
            seg_dir = os.path.join(case_dir, mask_dir_name)
            if os.path.isdir(seg_dir):
                for fname in sorted(os.listdir(seg_dir)):
                    lower = fname.lower()
                    if any(lower.endswith(s) for s in _MASK_SUFFIXES):
                        organ = fname
                        for s in _MASK_SUFFIXES:
                            if organ.lower().endswith(s):
                                organ = organ[:-len(s)]
                                break
                        if not organ:
                            organ = "mask"
                        masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

        cases.append({
            "case_id": name,
            "image": image_path,
            "image_type": image_type,
            "masks": masks,
            "case_dir": case_dir,
        })
    return cases


def _discover_single_case(case_dir, profile_id=None):
    """Discover image and masks for a single case directory.

    Returns case_info dict, or None if no valid image found.
    Used for lazy discovery; each case is scanned only when it's
    about to be converted, avoiding a blocking full-dataset scan.
    """
    profile = _dataset_profiles.load_profile(profile_id)
    if _is_medical_image_file(case_dir):
        image_file = os.path.abspath(case_dir)
        case_dir = os.path.dirname(image_file)
        name = _image_stem(image_file)
    else:
        image_file = None
        name = os.path.basename(case_dir)
    if not os.path.isdir(case_dir):
        return None
    if name in profile["exclude_dirs"]:
        return None

    if image_file:
        image_path, image_type = image_file, "medical_image"
    else:
        image_path, image_type = _find_case_image(
            case_dir, allow_direct_dicom=True, profile=profile
        )

    if image_path is None:
        return None

    masks = []
    allow_masks = True
    if image_file:
        # Count sibling medical images with an early stop: a case directory
        # can hold tens of thousands of DICOM slices, and os.path.isfile on
        # every entry hangs the import. We only need to know if >1 volume is
        # present, so stop as soon as a second candidate appears.
        sibling_images = 0
        try:
            with os.scandir(case_dir) as entries:
                for entry in entries:
                    lower = entry.name.lower()
                    if any(lower.endswith(suffix) for suffix in _MEDICAL_IMAGE_SUFFIXES):
                        sibling_images += 1
                        if sibling_images > 1:
                            break
        except OSError:
            pass
        allow_masks = sibling_images <= 1
    if allow_masks:
        for mask_dir_name in (list(profile["mask_dirs"]) or ["segmentations"]):
            seg_dir = os.path.join(case_dir, mask_dir_name)
            if os.path.isdir(seg_dir):
                for fname in sorted(os.listdir(seg_dir)):
                    lower = fname.lower()
                    if any(lower.endswith(s) for s in _MASK_SUFFIXES):
                        organ = fname
                        # Strip all known suffixes to get the organ name
                        for s in _MASK_SUFFIXES:
                            if organ.lower().endswith(s):
                                organ = organ[:-len(s)]
                                break
                        if not organ:
                            organ = "mask"
                        masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

    return {
        "case_id": name,
        "image": image_path,
        "image_type": image_type,
        "masks": masks,
        "case_dir": case_dir,
    }


# -- Call mimics_bridge.py ---------------------------------------------
# Synchronous bridge calls were removed from the Mimics GUI process: every
# long-running bridge action now goes through _launch_bridge_background so
# the GUI thread never blocks on a subprocess.communicate(timeout=600).

# -- Async bridge helpers ----------------------------------------------

def _launch_bridge_background(bridge_params, job_dir):
    """Launch mimics_bridge.py in background. Don't wait for completion."""
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)

    input_file = os.path.join(job_dir, "bridge_input.json")
    result_file = os.path.join(job_dir, "bridge_result.json")
    error_file = os.path.join(job_dir, "bridge_error.log")

    _verbose_log(
        os.path.dirname(job_dir),
        "Bridge launch | job_dir={0} | params={1}".format(job_dir, _summarize_bridge_params(bridge_params)),
    )
    _write_json_atomic(input_file, bridge_params)

    python_exe = _python_exe()
    bridge = _bridge_script()

    # Open result/error files ONLY after Popen succeeds, so a launch
    # failure does not leave a stale empty bridge_result.json that
    # confuses _check_job_status into thinking the bridge ran and failed.
    stdin_handle = open(input_file, "r")
    stdout_handle = None
    stderr_handle = None
    try:
        stdout_handle = open(result_file, "w")
        stderr_handle = open(error_file, "w")
        process = subprocess.Popen(
            [python_exe, bridge],
            stdin=stdin_handle,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=_background_env(),
            **_background_process_kwargs()
        )
        _verbose_log(
            os.path.dirname(job_dir),
            "Bridge subprocess pid={0}".format(process.pid),
        )
    finally:
        stdin_handle.close()
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
    return process


def _launch_bridge_job_thread(bridge_params, job_dir, output_dir, phase, case_id=None):
    """Start bridge from a worker thread so Mimics GUI can repaint first."""
    def _run():
        try:
            if not os.path.isdir(job_dir):
                os.makedirs(job_dir)
            _write_json_atomic(
                os.path.join(job_dir, "job_state.json"),
                {
                    "phase": "launching",
                    "requested_phase": phase,
                    "case_id": case_id or "",
                    "started_at": time.time(),
                },
            )
            _verbose_log(
                output_dir,
                "Job state | phase={0} | case_id={1} | params={2}".format(
                    phase, case_id or "", _summarize_bridge_params(bridge_params),
                ),
            )
        except Exception as exc:
            _BRIDGE_LAUNCH_ERRORS[job_dir] = (
                "Could not create or write the import output directory {0}: {1}".format(
                    os.path.abspath(output_dir), exc,
                )
            )
            _append_import_log(output_dir, "Could not initialize bridge job for {0}: {1}".format(phase, exc))
            _append_import_exception(output_dir, "Bridge job initialization failed", exc)
            return
        # Retry bridge launch up to 3 times with a short delay.
        # subprocess.Popen can fail intermittently inside Mimics due to
        # GIL contention, handle inheritance races, or transient OS errors.
        last_exc = None
        for _attempt in range(3):
            try:
                process = _launch_bridge_background(bridge_params, job_dir)
                _BRIDGE_PROCESSES[job_dir] = process
                state = {
                    "phase": phase,
                    "pid": process.pid,
                    "started_at": time.time(),
                }
                if case_id:
                    state["case_id"] = case_id
                _write_json_atomic(os.path.join(job_dir, "job_state.json"), state)
                _append_import_log(output_dir, "Bridge PID={0} | phase={1}".format(process.pid, phase))
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                _append_import_log(output_dir, "Bridge launch attempt {0} failed for {1}: {2}".format(_attempt + 1, phase, exc))
                if _attempt < 2:
                    time.sleep(1.0)
        if last_exc is not None:
            _BRIDGE_LAUNCH_ERRORS[job_dir] = "Could not start the import bridge process: {0}".format(last_exc)
            _append_import_log(output_dir, "Could not start bridge process for {0}: {1}".format(phase, last_exc))
            _append_import_exception(output_dir, "Bridge worker thread failed", last_exc)
            try:
                _write_json_atomic(
                    os.path.join(job_dir, "bridge_result.json"),
                    {"status": "error", "error": str(last_exc)},
                )
            except Exception:
                pass
            return

        # Keep the queue-producer lease current while one large image is being
        # prepared. This runs in the launcher thread, never on the Mimics GUI
        # timer, and prevents a second import from treating active work as stale.
        while process.poll() is None:
            _heartbeat_mcs_queue_active(output_dir)
            time.sleep(15.0)

    thread = threading.Thread(target=_run)
    thread.daemon = True
    thread.start()
    return thread


def _is_pid_alive(pid):
    """Check if a process with given PID is still running.

    Delegates to runtime_common.process_exists, which uses
    PROCESS_QUERY_LIMITED_INFORMATION (lower privilege, works across users /
    integrity levels where PROCESS_QUERY_INFORMATION is denied) with correct
    pointer-sized HANDLE ctypes signatures. A denied OpenProcess returns a
    zero handle and makes a running bridge process look dead, producing the
    "exited unexpectedly [no_result_file, no_error_file]" error."""
    return runtime_common.process_exists(pid)


def _check_job_status(job_dir):
    """Check status of a background bridge job.

    Returns ("running", None) | ("done", result_dict) | ("error", error_msg).
    """
    launch_error = _BRIDGE_LAUNCH_ERRORS.pop(job_dir, None)
    if launch_error:
        _BRIDGE_PROCESSES.pop(job_dir, None)
        _BRIDGE_EXIT_SEEN.pop(job_dir, None)
        return ("error", launch_error)
    result_file = os.path.join(job_dir, "bridge_result.json")
    state_file = os.path.join(job_dir, "job_state.json")
    error_file = os.path.join(job_dir, "bridge_error.log")

    # Try to read result
    if os.path.isfile(result_file):
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
            if result.get("status") == "ok":
                _verbose_log(
                    os.path.dirname(job_dir),
                    "Job done | keys={0}".format(sorted(result.keys())),
                )
                _BRIDGE_PROCESSES.pop(job_dir, None)
                _BRIDGE_EXIT_SEEN.pop(job_dir, None)
                return ("done", result)
            else:
                _verbose_log(
                    os.path.dirname(job_dir),
                    "Job error | {0}".format(result.get("error", "non-ok status")),
                )
                _BRIDGE_PROCESSES.pop(job_dir, None)
                _BRIDGE_EXIT_SEEN.pop(job_dir, None)
                return ("error", result.get("error", "bridge returned non-ok status"))
        except (ValueError, IOError):
            pass  # File incomplete - process may still be writing

    # Prefer the Popen object retained by this Mimics session. This avoids a
    # cross-integrity OpenProcess denial being mistaken for process exit.
    process = _BRIDGE_PROCESSES.get(job_dir)
    if process is not None:
        try:
            if process.poll() is None:
                _BRIDGE_EXIT_SEEN.pop(job_dir, None)
                return ("running", None)
        except Exception:
            pass

    # No valid result - check if process is alive
    pid = None
    phase = ""
    started_at = 0.0
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            pid = state.get("pid")
            phase = state.get("phase", "")
            started_at = float(state.get("started_at") or 0.0)
        except (ValueError, IOError, TypeError):
            pass
    if phase == "launching":
        return ("running", None)

    if pid and _is_pid_alive(pid):
        _BRIDGE_EXIT_SEEN.pop(job_dir, None)
        return ("running", None)

    # Startup grace period: the bridge Python imports numpy/SimpleITK etc.
    # before writing any output, and process-alive checks (OpenProcess /
    # GetExitCodeProcess) can intermittently report a freshly-started child as
    # dead on Mimics' embedded Python. For the first ~10s after launch, treat
    # a "dead" reading as "still starting" unless an error/result file already
    # proves otherwise (those are checked above). This stops the false
    # "exited unexpectedly [result_file_size=0]" during the import warm-up.
    if started_at and (time.time() - started_at) < 10.0:
        return ("running", None)

    # The child can exit just before its atomic result becomes visible. Give
    # the next timer tick a chance to read it instead of sleeping on Mimics'
    # GUI thread. The function starts by reading the result on every tick.
    first_exit_seen = _BRIDGE_EXIT_SEEN.setdefault(job_dir, time.time())
    if time.time() - first_exit_seen < 0.5:
        return ("running", None)
    _BRIDGE_EXIT_SEEN.pop(job_dir, None)

    err_msg = "bridge process exited unexpectedly"
    if os.path.isfile(error_file):
        try:
            with open(error_file, "r") as f:
                err_text = f.read().strip()
            if err_text:
                err_msg = err_text[:500]
        except Exception:
            pass
    # Append diagnostics so the user-visible error is actionable.
    _diag_parts = []
    if pid:
        _diag_parts.append("pid={0}".format(pid))
    if phase:
        _diag_parts.append("phase={0}".format(phase))
    if os.path.isfile(result_file):
        try:
            _rs = os.path.getsize(result_file)
            _diag_parts.append("result_file_size={0}".format(_rs))
        except Exception:
            pass
    else:
        _diag_parts.append("no_result_file")
    if os.path.isfile(error_file):
        try:
            _es = os.path.getsize(error_file)
            _diag_parts.append("error_file_size={0}".format(_es))
        except Exception:
            pass
    else:
        _diag_parts.append("no_error_file")
    if _diag_parts:
        err_msg = "{0} [{1}]".format(err_msg, ", ".join(_diag_parts))
    _BRIDGE_PROCESSES.pop(job_dir, None)
    return ("error", err_msg)


def _cleanup_job_dir(job_dir):
    """Remove a job directory after processing."""
    _BRIDGE_PROCESSES.pop(job_dir, None)
    if not job_dir or not os.path.isdir(job_dir):
        return
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass
    _prune_empty_parents(job_dir, 2)


def _cleanup_work_dir(work_dir):
    """Remove an incomplete or already-consumed work directory."""
    if not work_dir or not os.path.isdir(work_dir):
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass
    _prune_empty_parents(work_dir, 2)


def _prune_empty_parents(path, levels):
    current = os.path.dirname(path or "")
    for _ in range(max(0, int(levels))):
        if not current or not os.path.isdir(current):
            return
        try:
            os.rmdir(current)
        except OSError:
            return
        current = os.path.dirname(current)


def _terminate_job_process(job_dir, on_complete=None):
    """Best-effort termination for a timed-out bridge process."""
    def _complete_later(delay=0.0, target_pid=None):
        if on_complete is None:
            return
        def _run_complete():
            if delay:
                time.sleep(delay)
            if target_pid:
                deadline = time.time() + 15.0
                while _is_pid_alive(target_pid) and time.time() < deadline:
                    time.sleep(0.25)
                if _is_pid_alive(target_pid):
                    return
            try:
                on_complete()
            except Exception:
                pass
        thread = threading.Thread(target=_run_complete)
        thread.daemon = True
        thread.start()

    process = _BRIDGE_PROCESSES.pop(job_dir, None)
    if process is not None:
        try:
            if process.poll() is None:
                runtime_common.terminate_process_async(
                    process=process,
                    graceful_seconds=2.0,
                    on_complete=on_complete,
                )
                return
        except Exception:
            pass
    state_file = os.path.join(job_dir or "", "job_state.json")
    pid = None
    try:
        with open(state_file, "r") as handle:
            state = json.load(handle)
        pid = int(state.get("pid") or 0)
    except Exception:
        pid = None
    if not pid or not _is_pid_alive(pid):
        _complete_later()
        return
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_hidden_process_kwargs()
            )
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    _complete_later(0.0, target_pid=pid)


# -- Timer-based async monitor (same pattern as nnInteractive) ----------

def _stop_import_monitor(monitor_key):
    """Stop and clean up a running import monitor."""
    monitor = _IMPORT_MONITORS.pop(monitor_key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    try:
        if timer is not None and timer.isActive():
            timer.stop()
    except Exception:
        pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        user32, timer_id = win32_timer
        try:
            user32.KillTimer(None, timer_id)
        except Exception:
            pass
    subscription = monitor.get("event_subscription")
    if subscription is not None:
        try:
            subscription.unsubscribe()
        except Exception:
            pass


def _start_mimics_event_monitor(monitor, tick, poll_seconds, error_context):
    """Register a throttled callback through the documented Mimics timer event."""
    try:
        events = getattr(mimics, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return False
        interval = max(0.1, float(poll_seconds))
        monitor_key = monitor.get("monitor_key")
        _stop_import_monitor(monitor_key)
        monitor["event_last_tick"] = 0.0

        def callback(*_args, **_kwargs):
            now = time.time()
            if now - float(monitor.get("event_last_tick", 0.0)) < interval:
                return
            monitor["event_last_tick"] = now
            try:
                tick()
            except Exception as exc:
                output_dir = monitor.get("output_dir", "")
                if not output_dir and monitor.get("output_mcs"):
                    output_dir = os.path.dirname(os.path.abspath(monitor.get("output_mcs")))
                _append_import_exception(output_dir, error_context, exc)

        subscription = subscribe("timer", callback)
        if subscription is None:
            return False
        monitor["event_callback"] = callback
        monitor["event_subscription"] = subscription
        _IMPORT_MONITORS[monitor_key] = monitor
        return True
    except Exception:
        return False


def _cancel_import_monitor(monitor, reason="Import stopped by user request."):
    """Stop one bridge task and publish cancelled only after it is reaped."""
    if not monitor or monitor.get("_cancel_finalizer_started"):
        return False
    monitor["_cancel_finalizer_started"] = True
    monitor["done"] = True
    monitor_key = monitor.get("monitor_key")
    job_dir = monitor.get("job_dir")
    work_dir = monitor.get("work_dir")
    _stop_import_monitor(monitor_key)
    progress = {
        "status": "cancelling",
        "phase": "cancelling",
        "completed": monitor.get("completed", 0),
        "failed": monitor.get("failed", 0),
        "total": monitor.get("total", 1),
    }
    _write_import_task_status(monitor.get("task_status_path"), progress)

    def _finalize():
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(work_dir)
        _finish_import_producer(
            monitor,
            completed=monitor.get("completed", 0),
            failed=monitor.get("failed", 0),
        )
        final = dict(progress)
        final.update({
            "status": "cancelled",
            "phase": "cancelled",
            "error": str(reason or "Import stopped by user request."),
        })
        _write_import_task_status(monitor.get("task_status_path"), final)

    _terminate_job_process(job_dir, on_complete=_finalize)
    return True


def _fail_import_monitor_after_process(
    monitor, phase, error, message_title=None, message=None
):
    """Publish a timeout/error terminal only after the bridge has exited."""
    if not monitor or monitor.get("_failure_finalizer_started"):
        return False
    monitor["_failure_finalizer_started"] = True
    monitor["done"] = True
    job_dir = monitor.get("job_dir")
    work_dir = monitor.get("work_dir")
    output_mcs = monitor.get("output_mcs")
    output_dir = (
        os.path.dirname(os.path.abspath(output_mcs))
        if output_mcs else monitor.get("output_dir", "")
    )
    _stop_import_monitor(monitor.get("monitor_key"))
    _write_import_task_status(
        monitor.get("task_status_path"),
        {
            "status": "stopping",
            "phase": "stopping_after_{0}".format(phase),
            "error": str(error),
            "completed": monitor.get("completed", 0),
            "failed": monitor.get("failed", 0),
            "total": monitor.get("total", 1),
        },
    )
    if message_title and message:
        _safe_message_box(message_title, message)

    def _finalize():
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(work_dir)
        if monitor.get("case_id"):
            _record_failed_case(
                output_dir, monitor.get("case_id"), phase, str(error),
                source_image=monitor.get("case_image", ""),
            )
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "failed",
                "phase": phase,
                "error": str(error),
                "completed": monitor.get("completed", 0),
                "failed": max(1, int(monitor.get("failed", 0) or 0)),
                "total": monitor.get("total", 1),
            },
        )
        if not monitor.get("batch_queue") and output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
        _finish_import_producer(
            monitor,
            completed=monitor.get("completed", 0),
            failed=max(1, int(monitor.get("failed", 0) or 0)),
        )

    _terminate_job_process(job_dir, on_complete=_finalize)
    return True


def _import_monitor_tick(monitor):
    """Timer callback: check if bridge finished, then queue .mcs creation."""
    if monitor.get("done"):
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    if _import_task_stopped(monitor):
        _cancel_import_monitor(monitor)
        return

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        _fail_import_monitor_after_process(
            monitor,
            "prepare_timeout",
            "Dataset preparation timed out.",
            message_title="导入超时",
            message=(
                "数据集准备已超时。外部转换器正在被停止；"
                "其退出后任务将变为失败。"
            ),
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        detail = str(monitor.get("case_id") or "case")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "case_preparation",
            detail=detail,
            interval_seconds=60.0,
            initial_delay_seconds=30.0,
        )
        if due:
            _user_progress(
                logging.INFO,
                "Import is still preparing {0} ({1}s). Mimics remains "
                "available. Use Task Status (01 Data menu) to cancel.".format(
                    detail, int(elapsed)
                ),
            )
        return  # still running, next tick will check again

    runtime_common.clear_progress_notice(monitor, "case_preparation")

    monitor["done"] = True
    _stop_import_monitor(monitor_key)

    if _import_task_stopped(monitor):
        _cancel_import_monitor(monitor)
        return

    if status == "error":
        output_mcs = monitor.get("output_mcs")
        output_dir = os.path.dirname(os.path.abspath(output_mcs)) if output_mcs else ""
        _record_failed_case(
            output_dir, monitor.get("case_id"), "prepare", result,
            source_image=monitor.get("case_image", ""),
        )
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "failed",
                "phase": "prepare_failed",
                "error": str(result),
                "case_id": monitor.get("case_id", ""),
                "total": monitor.get("total", 1),
            },
        )
        # Keep the job directory on failure so bridge_error.log / job_state.json
        # survive for diagnosis. The "bridge process exited unexpectedly
        # [no_result_file, no_error_file]" symptom leaves no trace otherwise.
        _append_import_log(
            output_dir,
            "Preparation failed for {0}: {1}. Job dir preserved at {2}.".format(
                monitor.get("case_id"), result, job_dir,
            ),
        )
        _category, _guidance_message, guidance_action = _error_guidance(result, "prepare")
        source_image = monitor.get("case_image", "")
        mimics.dialogs.message_box(
            title="导入错误",
            message=(
                "病例 '{0}' 准备失败：\n{1}\n\n来源：{2}\n\n"
                "建议操作：{3}\n\n诊断文件保留在：\n{4}"
            ).format(
                monitor.get("case_id") or "未知",
                result,
                source_image or "请查看导入日志",
                guidance_action,
                job_dir,
            ),
        )
        _cleanup_work_dir(monitor.get("work_dir"))
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        elif output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
            _release_import_producer_lease(monitor)
        return

    # status == "done"
    try:
        output_mcs = monitor.get("output_mcs")
        work_dir = monitor.get("work_dir")
        output_dir = os.path.dirname(os.path.abspath(output_mcs))
        _save_prepare_manifest(work_dir, result, output_mcs=output_mcs)
        _publish_prepared_work(output_dir, monitor.get("case_id"), work_dir, output_mcs)
        _cleanup_job_dir(job_dir)
        # A single-case producer has already committed its only descriptor.
        # Marking the queue done lets the consumer exit immediately after that
        # case instead of waiting for a stale producer heartbeat.
        _mark_mcs_queue_done(output_dir, completed=1, failed=0)
        _release_import_producer_lease(monitor)
        _ensure_bg_mimics_running(output_dir, monitor.get("total", 1), mark_active=False)
        _start_first_mcs_monitor(
            output_dir,
            target_mcs=output_mcs,
            skip_if_project_open=True,
            notify_only=not _auto_open_mcs_enabled(),
        )
        _append_import_log(
            output_dir,
            "Data preparation finished; background Mimics is creating the .mcs file: {0}".format(output_mcs),
        )
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "creating_mcs",
                "completed": 1,
                "failed": 0,
                "total": monitor.get("total", 1),
                "case_id": monitor.get("case_id", ""),
            },
        )
    except Exception as e:
        output_mcs = monitor.get("output_mcs")
        output_dir = os.path.dirname(os.path.abspath(output_mcs)) if output_mcs else ""
        _append_import_log(output_dir, "Import queueing failed: {0}".format(e))
        _record_failed_case(
            output_dir, monitor.get("case_id"), "prepare_manifest", e,
            source_image=monitor.get("case_image", ""),
        )
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "failed",
                "phase": "queue_failed",
                "error": str(e),
                "total": monitor.get("total", 1),
            },
        )
        traceback.print_exc()
        _category, _guidance_message, guidance_action = _error_guidance(e, "queue")
        mimics.dialogs.message_box(
            title="导入错误",
            message="导入排队失败：{0}\n\n建议操作：{1}".format(e, guidance_action),
        )
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        # In batch mode, continue to next case
        if monitor.get("batch_queue"):
            _start_next_batch_case(monitor)
        elif output_dir:
            _mark_mcs_queue_done(output_dir, completed=0, failed=1)
            _release_import_producer_lease(monitor)
        return

    # Single case: completion is reported by the .mcs status monitor. Avoid an
    # intermediate dialog that makes the user acknowledge the same operation
    # twice; the Mimics log remains available for progress inspection.
    if not monitor.get("batch_queue"):
        _user_progress(
            logging.INFO,
            "Import preparation finished. Background Mimics is creating: {0}".format(output_mcs),
        )
        return

    # Batch: update progress and start next case
    monitor["completed"] = monitor.get("completed", 0) + 1
    _start_next_batch_case(monitor)


def _start_next_batch_case(monitor):
    """Start bridge for the next case in the batch queue."""
    queue = monitor.get("batch_queue")
    if not queue:
        # All done
        completed = monitor.get("completed", 0)
        failed = monitor.get("failed", 0)
        total = monitor.get("total", 0)
        mimics.dialogs.message_box(
            title="批量导入完成",
            message="已排队 {0}/{1} 例；失败 {2} 例。".format(completed, total, failed),
        )
        return

    case_info = queue.pop(0)
    case_id = case_info["case_id"]
    output_dir = monitor.get("output_dir")
    axes = monitor.get("axes")
    flips = monitor.get("flips")
    jobs_dir = monitor.get("jobs_dir")

    output_mcs = os.path.join(output_dir, case_id + ".mcs")
    work_dir = os.path.join(monitor.get("work_root"), case_id)
    job_dir = os.path.join(jobs_dir, case_id)

    _append_import_log(monitor.get("output_dir", ""), "\n[{0}/{1}] Preparing: {2}".format(
        monitor.get("completed", 0) + monitor.get("failed", 0) + 1,
        monitor.get("total", 0),
        case_id))

    bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)

    # Update monitor for next case
    old_monitor_key = monitor.get("monitor_key")
    monitor["job_dir"] = job_dir
    monitor["output_mcs"] = output_mcs
    monitor["work_dir"] = work_dir
    monitor["case_id"] = case_id
    monitor["monitor_key"] = job_dir
    monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 1800)
    monitor["done"] = False

    _launch_bridge_job_thread(bridge_params, job_dir, output_dir, "preparing", case_id=case_id)

    _IMPORT_MONITORS.pop(old_monitor_key, None)
    _IMPORT_MONITORS[job_dir] = monitor


# -- Batch prepare-only flow (no Mimics API, GUI stays responsive) ------

def _batch_prepare_tick_impl(monitor):
    """Timer callback for batch prepare: bridge done -> manifest -> next case."""
    if monitor.get("done"):
        return

    if _import_task_stopped(monitor):
        _cancel_import_monitor(monitor)
        return

    timeout_pending = monitor.get("_timeout_pending")
    if timeout_pending:
        if not monitor.pop("_timed_out_process_stopped", False):
            return
        output_dir = monitor.get("output_dir")
        case_id = timeout_pending.get("case_id", "")
        error = timeout_pending.get("error", "Dataset conversion timed out.")
        _record_failed_case(
            output_dir, case_id, "prepare_timeout", error,
            source_image=monitor.get("case_image", ""),
        )
        monitor["failed"] = monitor.get("failed", 0) + 1
        monitor.pop("_timeout_pending", None)
        monitor["busy"] = False
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "preparing",
                "completed": monitor.get("completed", 0),
                "failed": monitor.get("failed", 0),
                "total": monitor.get("total", 0),
                "case_id": case_id,
            },
        )
        _start_next_batch_prepare(monitor)
        return

    # Prevent re-entrancy: while we process a completed case (which may
    # show a blocking message_box), the timer can fire again and re-enter
    # this function.  The busy flag prevents double-processing.
    if monitor.get("busy"):
        return

    if monitor.pop("selecting_next", False):
        monitor["busy"] = True
        _start_next_batch_prepare(monitor)
        return

    job_dir = monitor.get("job_dir")
    monitor_key = monitor.get("monitor_key")

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["busy"] = True
        output_dir = monitor.get("output_dir")
        case_id = monitor.get("case_id", "")
        error = "Dataset conversion timed out."
        _append_import_log(
            output_dir,
            "Conversion timed out for {0}; waiting for the bridge process to exit before continuing.".format(
                case_id
            ),
        )
        monitor["_timeout_pending"] = {
            "case_id": case_id,
            "error": error,
        }
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "stopping_case",
                "completed": monitor.get("completed", 0),
                "failed": monitor.get("failed", 0),
                "total": monitor.get("total", 0),
                "case_id": case_id,
            },
        )

        def _timed_out_process_stopped():
            _cleanup_job_dir(job_dir)
            _cleanup_work_dir(monitor.get("work_dir"))
            monitor["_timed_out_process_stopped"] = True

        _terminate_job_process(
            job_dir,
            on_complete=_timed_out_process_stopped,
        )
        return

    status, result = _check_job_status(job_dir)

    if status == "running":
        detail = str(monitor.get("case_id") or "case")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "case_preparation",
            detail=detail,
            interval_seconds=60.0,
            initial_delay_seconds=30.0,
        )
        if due:
            _user_progress(
                logging.INFO,
                "Import is still preparing {0} ({1}s); completed {2}/{3}, "
                "failed {4}. Use Task Status"
                " (01 Data menu) to cancel.".format(
                    detail,
                    int(elapsed),
                    int(monitor.get("completed", 0) or 0),
                    int(monitor.get("total", 0) or 0),
                    int(monitor.get("failed", 0) or 0),
                ),
            )
        return  # still running, next tick will check again

    runtime_common.clear_progress_notice(monitor, "case_preparation")

    # Case finished (done or error).
    # Set busy flag to prevent re-entrancy during message_box etc.
    monitor["busy"] = True

    if _import_task_stopped(monitor):
        monitor["busy"] = False
        _cancel_import_monitor(monitor)
        return

    if status == "error":
        output_dir = monitor.get("output_dir", "")
        case_id = monitor.get("case_id", "")
        _append_import_log(output_dir, "Conversion failed for {0}: {1}".format(case_id, result))
        _record_failed_case(
            output_dir, case_id, "prepare", result,
            source_image=monitor.get("case_image", ""),
        )
        monitor["failed"] = monitor.get("failed", 0) + 1
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "preparing",
                "completed": monitor.get("completed", 0),
                "failed": monitor.get("failed", 0),
                "total": monitor.get("total", 0),
                "case_id": case_id,
            },
        )
        _cleanup_job_dir(job_dir)
        _cleanup_work_dir(monitor.get("work_dir"))
        monitor["busy"] = False
        _start_next_batch_prepare(monitor)
        return

    # status == "done" - save manifest for later background .mcs creation.
    work_dir = monitor.get("work_dir")
    case_id = result.get("case_id", monitor.get("case_id", ""))
    output_dir = monitor.get("output_dir")
    completed_count = monitor.get("completed", 0) + 1
    total = monitor.get("total", 0)

    output_mcs = os.path.join(output_dir, case_id + ".mcs")
    queued = False
    try:
        _save_prepare_manifest(work_dir, result, output_mcs=output_mcs)
        _publish_prepared_work(output_dir, case_id, work_dir, output_mcs)
        monitor["completed"] = completed_count
        queued = True
        _append_import_log(
            output_dir,
            "[{0}/{1}] Prepared {2}; background Mimics will create the .mcs file.".format(
                completed_count,
                total,
                case_id,
            ),
        )
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "preparing",
                "completed": completed_count,
                "failed": monitor.get("failed", 0),
                "total": total,
                "case_id": case_id,
            },
        )
    except Exception as e:
        _append_import_log(output_dir, "Failed to save prepare manifest for {0}: {1}".format(case_id, e))
        _record_failed_case(
            output_dir, case_id, "prepare_manifest", e,
            source_image=monitor.get("case_image", ""),
        )
        monitor["failed"] = monitor.get("failed", 0) + 1
        _cleanup_work_dir(work_dir)
    _cleanup_job_dir(job_dir)
    if queued:
        _ensure_bg_mimics_running(output_dir, total)
        if not monitor.get("first_mcs_monitor_started"):
            monitor["first_mcs_monitor_started"] = True
            _start_first_mcs_monitor(
                output_dir,
                target_mcs=None,
                timeout_seconds=1800,
                poll_seconds=2.0,
                after_epoch=monitor.get("batch_started_epoch") or (time.time() - 1.0),
                skip_if_project_open=True,
                notify_only=not _auto_open_mcs_enabled(),
            )

    # Check if more cases to prepare; delegate to _start_next_batch_prepare
    # which handles both "start next" and "all done" cases.
    _start_next_batch_prepare(monitor)
    monitor["busy"] = False


def _batch_prepare_tick(monitor):
    """Run one prepare tick and never leave a stale re-entrancy flag."""
    was_busy = bool(monitor.get("busy"))
    try:
        return _batch_prepare_tick_impl(monitor)
    finally:
        if not was_busy:
            monitor["busy"] = False


def _finish_batch_prepare_queue(monitor):
    monitor["done"] = True
    monitor["busy"] = False
    _stop_import_monitor(monitor.get("monitor_key"))
    completed = monitor.get("completed", 0)
    failed = monitor.get("failed", 0)
    total = monitor.get("total", 0)
    output_dir = monitor.get("output_dir")
    _append_import_log(
        output_dir,
        "All {0} case(s) prepared; {1} failed. Background Mimics is still creating .mcs files from prepared data.".format(
            completed,
            failed,
        ),
    )
    _mark_mcs_queue_done(output_dir, completed=completed, failed=failed)
    monitor["queue_marked_active"] = False
    _release_import_producer_lease(monitor)
    _ensure_bg_mimics_running(output_dir, total, mark_active=False)
    _write_import_task_status(
        monitor.get("task_status_path"),
        {
            "status": "running",
            "phase": "waiting_for_mcs",
            "completed": completed,
            "failed": failed,
            "total": total,
        },
    )


def _start_next_batch_prepare(monitor):
    """Start the next case; quarantine setup failures without stopping the batch."""
    skipped_this_tick = 0
    while True:
        queue = monitor.get("batch_queue")
        if not queue:
            _finish_batch_prepare_queue(monitor)
            return
        item = queue.pop(0)
        case_info = None
        if isinstance(item, tuple):
            case_name, case_dir = item
        else:
            try:
                if item.get("image"):
                    case_info = item
                case_name = item.get("case_id", "")
                case_dir = item.get("case_dir", "")
            except Exception:
                case_name = "unknown_case"
                case_dir = ""
        try:
            if case_info is None:
                case_info = _discover_single_case(case_dir)
        except Exception as exc:
            output_dir = monitor.get("output_dir", "")
            _append_import_log(
                output_dir,
                "Case discovery failed for {0}; continuing with the next case: {1}".format(
                    case_name, exc
                ),
            )
            _record_failed_case(output_dir, case_name, "discover_case", exc, source_image=case_dir)
            monitor["failed"] = monitor.get("failed", 0) + 1
            skipped_this_tick += 1
            if skipped_this_tick >= 8 and monitor.get("batch_queue"):
                monitor["selecting_next"] = True
                return
            continue
        if case_info is None:
            output_dir = monitor.get("output_dir", "")
            _append_import_log(
                output_dir,
                "Skipping case without image data: {0}".format(case_name),
            )
            _record_failed_case(
                output_dir,
                case_name,
                "discover_case",
                "No image data found.",
                source_image=case_dir,
            )
            monitor["failed"] = monitor.get("failed", 0) + 1
            skipped_this_tick += 1
            if skipped_this_tick >= 8 and monitor.get("batch_queue"):
                monitor["selecting_next"] = True
                return
            continue

        try:
            work_dir = None
            job_dir = None
            case_id = str(case_info["case_id"])
            if not case_id:
                raise RuntimeError("Discovered case has an empty case_id.")
            output_dir = monitor.get("output_dir")
            axes = monitor.get("axes")
            flips = monitor.get("flips")
            jobs_dir = monitor.get("jobs_dir")

            work_dir = os.path.join(monitor.get("work_root"), case_id)
            job_dir = os.path.join(jobs_dir, case_id)

            completed = monitor.get("completed", 0)
            failed = monitor.get("failed", 0)
            total = monitor.get("total", 0)
            _append_import_log(
                output_dir,
                "[{0}/{1}] Preparing: {2}".format(completed + failed + 1, total, case_id),
            )
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "running",
                    "phase": "preparing",
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "case_id": case_id,
                },
            )

            bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
            old_monitor_key = monitor.get("monitor_key")
            _launch_bridge_job_thread(
                bridge_params,
                job_dir,
                output_dir,
                "preparing",
                case_id=case_id,
            )
        except Exception as exc:
            output_dir = monitor.get("output_dir", "")
            case_mapping = case_info if isinstance(case_info, dict) else {}
            failed_case_id = str(
                case_mapping.get("case_id") or case_name or "unknown_case"
            )
            _append_import_log(
                output_dir,
                "Could not start preparation for {0}; continuing with the next case: {1}".format(
                    failed_case_id, exc
                ),
            )
            _record_failed_case(
                output_dir, failed_case_id, "prepare_start", exc,
                source_image=str(
                    (case_mapping or {}).get("image")
                    or (case_mapping or {}).get("case_dir")
                    or case_dir
                ),
            )
            monitor["failed"] = monitor.get("failed", 0) + 1
            try:
                _cleanup_job_dir(job_dir)
                _cleanup_work_dir(work_dir)
            except Exception:
                pass
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "running",
                    "phase": "preparing",
                    "completed": monitor.get("completed", 0),
                    "failed": monitor.get("failed", 0),
                    "total": monitor.get("total", 0),
                    "case_id": failed_case_id,
                },
            )
            skipped_this_tick += 1
            if skipped_this_tick >= 8 and monitor.get("batch_queue"):
                monitor["selecting_next"] = True
                return
            continue

        # Register only after the bridge launch has succeeded. A failed launch
        # must not replace the active monitor key with a job that never ran.
        monitor["job_dir"] = job_dir
        monitor["work_dir"] = work_dir
        monitor["case_id"] = case_id
        # The user's original input path, so failure records point at the
        # real data location rather than internal work dirs (R61-5).
        monitor["case_image"] = str(case_info.get("image") or case_info.get("case_dir") or "")
        monitor["monitor_key"] = job_dir
        monitor["deadline"] = time.time() + monitor.get("timeout_seconds", 1800)
        monitor["done"] = False
        _IMPORT_MONITORS.pop(old_monitor_key, None)
        _IMPORT_MONITORS[job_dir] = monitor
        return


def _start_batch_prepare_monitor(job_dir, work_dir, timeout_seconds=1800,
                                  poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer for batch prepare-only flow."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "work_dir": work_dir,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "batch_queue": batch_queue,
        "completed": 0,
        "failed": 0,
    }
    if batch_info:
        monitor.update(batch_info)

    if os.name == "nt":
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        if _allow_windows_event_monitor() and _start_mimics_event_monitor(
            monitor,
            lambda: _batch_prepare_tick(monitor),
            poll_seconds,
            "_start_batch_prepare_monitor Mimics timer callback failed",
        ):
            return True
    elif _start_mimics_event_monitor(
        monitor,
        lambda: _batch_prepare_tick(monitor),
        poll_seconds,
        "_start_batch_prepare_monitor Mimics timer callback failed",
    ):
        return True

    # Try PyQt5 QTimer first
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="转换进行中",
            message="数据集转换已开始，但进度无法自动监控。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            title="转换进行中",
            message="数据集转换已开始，但进度无法自动监控。",
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        try:
            _batch_prepare_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_batch_prepare_monitor qtimer callback failed", exc)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


def _start_win32_batch_prepare_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for batch prepare polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )

    def _timer_proc(hwnd, message, timer_id, tick_count):
        try:
            _batch_prepare_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_win32_batch_prepare_monitor timer callback failed", exc)

    callback = TIMERPROC(_timer_proc)
    callback_ptr = ctypes.cast(callback, ctypes.c_void_p)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_ptr)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _find_mimics_exe():
    """Find MimicsResearch.exe installation path."""
    return runtime_common.find_mimics_exe()


def _ensure_bg_mimics_running(output_dir, total_count=0, mark_active=True):
    """Ensure a background Mimics is running to create .mcs files.

    Called after each case's manifest is saved.  If no background Mimics
    is alive, launch one.  This way .mcs files are created continuously
    as manifests become available, rather than waiting for all conversions
    to finish.
    """
    if mark_active:
        _mark_mcs_queue_active(output_dir, total_count)
    if _background_stop_requested(output_dir):
        _append_import_log(output_dir, "Background .mcs creation is stopped by user request.")
        return
    output_key = os.path.normcase(os.path.abspath(output_dir))
    if _bg_failed_contains(output_key):
        return
    # Check if existing background Mimics is still alive
    state = _bg_state_snapshot(output_dir)
    if state["pid"] and _is_pid_alive(state["pid"]):
        return  # same queue; the worker will pick up the new descriptor

    # Executable discovery, lock acquisition, and Popen can touch slow disks.
    # Keep all of it off the Mimics GUI timer callback.
    if not _bg_try_begin_launch(output_dir):
        _schedule_bg_mimics_retry(output_dir, total_count=total_count)
        return

    def _launch():
        try:
            _launch_background_mimics(output_dir, total_count=total_count)
        finally:
            _bg_end_launch(output_dir)

    thread = threading.Thread(target=_launch, name="MimicsBackgroundLauncher")
    thread.daemon = True
    try:
        thread.start()
    except Exception:
        _bg_end_launch(output_dir)
        raise


def _schedule_bg_mimics_retry(output_dir, total_count=0):
    key = os.path.normcase(os.path.abspath(output_dir))
    if _bg_failed_contains(key):
        return
    if not _bg_retry_add(key):
        return

    def _retry():
        retry_started = time.time()
        deadline = time.time() + 21600.0
        try:
            while time.time() < deadline:
                time.sleep(30.0)
                if _background_stop_requested(output_dir, retry_started):
                    _append_import_log(output_dir, "Background Mimics retry stopped by user request.")
                    return
                if _bg_failed_contains(key):
                    return
                state = _bg_state_snapshot(output_dir)
                if state["pid"] and _is_pid_alive(state["pid"]):
                    return
                if not _bg_try_begin_launch(output_dir):
                    continue
                try:
                    process = _launch_background_mimics(
                        output_dir,
                        total_count=total_count,
                        schedule_retry=False,
                    )
                finally:
                    _bg_end_launch(output_dir)
                if process is not None:
                    return
        finally:
            _bg_retry_discard(key)

    thread = threading.Thread(target=_retry)
    thread.daemon = True
    try:
        thread.start()
    except Exception:
        _bg_retry_discard(key)
        raise


def _watch_background_mimics(process, output_dir, handshake_path, mimics_log_path,
                              process_log_path, lock_path, lock_token):
    """Classify the child exit, persist terminal status, then release its lock."""
    exit_code = None
    forced_reason = ""
    started_at = time.time()
    deadline = started_at + 24.0 * 3600.0
    while exit_code is None:
        try:
            exit_code = process.wait(timeout=30.0)
            break
        except subprocess.TimeoutExpired:
            if _background_stop_requested(output_dir, started_at):
                forced_reason = "stop requested"
            elif time.time() >= deadline:
                forced_reason = "24-hour background Mimics safety timeout"
            else:
                continue
            try:
                process.terminate()
                exit_code = process.wait(timeout=10.0)
            except Exception:
                try:
                    process.kill()
                    exit_code = process.wait(timeout=10.0)
                except Exception:
                    exit_code = process.poll()
            if exit_code is None:
                time.sleep(5.0)
                continue
            break
        except Exception:
            exit_code = process.poll()
            if exit_code is None:
                time.sleep(5.0)
                continue
            break
    status_path = _rt(output_dir, "_mcs_batch_status.json")
    try:
        status = runtime_common.read_json(status_path, {}) or {}
        handshake_exists = os.path.isfile(handshake_path)
        if handshake_exists and status.get("status") in ("closed", "cancelled", "failed"):
            if status.get("status") == "failed":
                _bg_mark_failed(os.path.normcase(os.path.abspath(output_dir)))
            return

        if not handshake_exists or status.get("status") not in ("closed", "cancelled", "failed"):
            output_key = os.path.normcase(os.path.abspath(output_dir))
            if forced_reason == "stop requested":
                runtime_common.write_json_atomic(
                    status_path,
                    {
                        "status": "cancelled",
                        "pid": getattr(process, "pid", 0),
                        "exit_code": exit_code,
                        "error": "Background .mcs creation stopped by user request.",
                        "runner_started": bool(handshake_exists),
                        "updated_at_epoch": time.time(),
                    },
                )
                return
            _bg_mark_failed(output_key)
            mimics_tail = runtime_common.read_text_tail(mimics_log_path, 12000)
            process_tail = runtime_common.read_text_tail(process_log_path, 12000)
            if not handshake_exists:
                reason = (
                    "Background Mimics exited before executing the import runner "
                    "(exit code {0})."
                ).format(exit_code)
            else:
                reason = (
                    "Background Mimics executed the import runner but stopped before "
                    "reporting completion (exit code {0})."
                ).format(exit_code)
            if forced_reason:
                reason += " Forced shutdown reason: {0}.".format(forced_reason)
            details = []
            if mimics_tail:
                details.append("Mimics log tail:\n" + mimics_tail)
            if process_tail:
                details.append("Process log tail:\n" + process_tail)
            message = reason
            if details:
                message += "\n\n" + "\n\n".join(details)
            message += (
                "\n\nPrepared import data was kept for retry. "
                "Mimics log: {0}\nProcess log: {1}"
            ).format(mimics_log_path, process_log_path)
            _append_import_log(output_dir, message)
            try:
                runtime_common.write_json_atomic(
                    status_path,
                    {
                        "status": "failed",
                        "pid": getattr(process, "pid", 0),
                        "exit_code": exit_code,
                        "error": reason,
                        "mimics_log": mimics_log_path,
                        "process_log": process_log_path,
                        "runner_started": bool(handshake_exists),
                        "updated_at_epoch": time.time(),
                    },
                )
            except Exception:
                pass
    finally:
        _bg_clear_process(getattr(process, "pid", None))
        try:
            runtime_common.release_resource_lock(lock_path, lock_token)
        except Exception:
            pass


def _launch_background_mimics(output_dir, total_count=0, schedule_retry=True):
    """Launch Mimics in background mode to create .mcs files from prepared data.

    Mimics runs without GUI (-b flag), executing create_mcs_batch.py which
    reads prepare manifests and creates .mcs files one by one.
    Returns the Popen object, or None on failure.
    """
    global _BG_MIMICS_BUSY_NOTICE_AT
    output_key = os.path.normcase(os.path.abspath(output_dir))
    if _bg_failed_contains(output_key):
        return None
    if _background_stop_requested(output_dir):
        _append_import_log(output_dir, "Background .mcs creation was not started because stop was requested.")
        return None
    mimics_exe = _find_mimics_exe()
    if not mimics_exe:
        message = (
            "A separate background Mimics executable was not found. The open MimicsMedical.exe "
            "is not reused automatically because single-instance redirection can close or reconfigure "
            "the annotation window. Configure the background Mimics executable in the settings if needed."
        )
        _append_import_log(output_dir, message)
        _bg_mark_failed(output_key)
        runtime_common.write_json_atomic(
            _rt(output_dir, "_mcs_batch_status.json"),
            {
                "status": "failed",
                "error": message,
                "runner_started": False,
                "updated_at_epoch": time.time(),
            },
        )
        return None

    # Find the create_mcs_batch.py script
    here = os.path.dirname(os.path.abspath(__file__))
    script_candidates = [
        os.path.join(here, "create_mcs_batch.py"),
        os.path.join(here, "..", "runtime_py35", "create_mcs_batch.py"),
    ]
    script_path = None
    for c in script_candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            script_path = c
            break
    if not script_path:
        script_path = os.path.abspath(script_candidates[0])

    # Write a runner script that Mimics can execute
    runner_path = _rt(output_dir, "_run_create_mcs.py")
    handshake_path = _rt(output_dir, "_background_import_runner_started.json")
    runner_parent = os.path.dirname(runner_path)
    if not os.path.isdir(runner_parent):
        os.makedirs(runner_parent)
    script_dir = os.path.dirname(script_path)
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated runner for background Mimics .mcs creation\n")
        f.write("import sys, os, json, time\n")
        f.write(
            "open({0}, 'w').write(json.dumps({{'pid': os.getpid(), "
            "'started_at_epoch': time.time()}}))\n".format(json.dumps(handshake_path))
        )
        f.write("sys.path.insert(0, {0})\n".format(json.dumps(script_dir)))
        f.write("os.environ['MIMICS_BRIDGE_PYTHON'] = {0}\n".format(json.dumps(_python_exe())))
        f.write("os.environ['MIMICS_BRIDGE_SCRIPT'] = {0}\n".format(json.dumps(_bridge_script())))
        f.write("import create_mcs_batch\n")
        f.write(
            "create_mcs_batch.main({0}, runtime_dir={1})\n".format(
                json.dumps(output_dir), json.dumps(_rt(output_dir))
            )
        )

    # Launch Mimics in background mode
    log_path = _rt(output_dir, "_background_mimics.log")
    mimics_log_path = _rt(output_dir, "_background_mimics_application.log")
    for stale_path in (handshake_path, mimics_log_path):
        try:
            if os.path.isfile(stale_path):
                os.remove(stale_path)
        except OSError:
            pass
    supervisor_path = os.path.join(
        _project_root(), "tools", "mcs_creation_supervisor.py"
    )
    if not os.path.isfile(supervisor_path):
        message = "MCS creation supervisor was not found: {0}".format(supervisor_path)
        _append_import_log(output_dir, message)
        _bg_mark_failed(output_key)
        runtime_common.write_json_atomic(
            _rt(output_dir, "_mcs_batch_status.json"),
            {
                "status": "failed",
                "error": message,
                "runner_started": False,
                "updated_at_epoch": time.time(),
            },
        )
        return None
    cmd = runtime_common.mcs_creation_supervisor_command(
        _python_exe(),
        supervisor_path,
        mimics_exe,
        runner_path,
        _rt(output_dir),
        output_dir,
        handshake_path=handshake_path,
        mimics_log_path=mimics_log_path,
    )
    _rotate_log_file(log_path)
    lock_path = runtime_common.background_mimics_lock_path(
        _project_root(), output_dir
    )
    lock_token = runtime_common.acquire_resource_lock(
        lock_path,
        "background_mimics",
        "import .mcs creation",
        wait_seconds=0.0,
    )
    if not lock_token:
        now = time.time()
        with _BG_MIMICS_STATE_LOCK:
            should_log_busy = now - _BG_MIMICS_BUSY_NOTICE_AT >= 60.0
            if should_log_busy:
                _BG_MIMICS_BUSY_NOTICE_AT = now
        if should_log_busy:
            holder = runtime_common.active_resource_lock(
                _project_root(), runtime_common.background_mimics_lock_name(output_dir)
            )
            _append_import_log(
                output_dir,
                "Background Mimics is already running for another Mimics-Script task; .mcs creation will continue when that process exits.",
            )
            _append_import_log(
                output_dir,
                "Waiting task: {0}. Use Task Status"
                " (01 Data menu) to cancel.".format(
                    runtime_common.resource_lock_summary(holder)
                ),
            )
        if schedule_retry:
            _schedule_bg_mimics_retry(output_dir, total_count=total_count)
        return None
    log_handle = None
    process = None
    try:
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_background_process_kwargs()
        )
        _bg_set_process(process.pid, output_dir)
        runtime_common.register_process(
            runtime_common.project_root(),
            "background_mimics",
            process.pid,
            parent_pid=os.getpid(),
            state_path=os.path.join(_rt(output_dir), "_mcs_batch_status.json"),
        )
        if not runtime_common.update_resource_lock_pid(
            lock_path,
            lock_token,
            process.pid,
            {
                "kind": "create_mcs",
                "output_dir": os.path.abspath(output_dir),
                "runtime_dir": _rt(output_dir),
            },
        ):
            raise RuntimeError(
                "Background Mimics started, but lock ownership could not be transferred to PID {0}.".format(
                    process.pid
                )
            )
        _append_import_log(
            output_dir,
            "Background .mcs creation supervisor started (PID={0}). "
            "Waiting for runner handshake.".format(process.pid),
        )
        watcher = threading.Thread(
            target=_watch_background_mimics,
            args=(
                process,
                output_dir,
                handshake_path,
                mimics_log_path,
                log_path,
                lock_path,
                lock_token,
            ),
            name="MimicsBackgroundImportWatcher",
        )
        watcher.daemon = True
        watcher.start()
        return process
    except Exception as e:
        process_stopped = process is None
        if process is not None:
            _bg_clear_process(getattr(process, "pid", None))
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5.0)
                except Exception:
                    pass
            try:
                process_stopped = process.poll() is not None
            except Exception:
                process_stopped = not _is_pid_alive(getattr(process, "pid", 0))
        if process_stopped:
            runtime_common.release_resource_lock(lock_path, lock_token)
        else:
            try:
                runtime_common.update_resource_lock_pid(
                    lock_path,
                    lock_token,
                    process.pid,
                    {
                        "kind": "create_mcs",
                        "output_dir": os.path.abspath(output_dir),
                        "runtime_dir": _rt(output_dir),
                        "termination_pending": True,
                    },
                )
            except Exception:
                pass
            _append_import_log(
                output_dir,
                "Background Mimics PID {0} did not exit after launch failure; "
                "the shared lock was retained.".format(getattr(process, "pid", "?")),
            )
        _append_import_log(output_dir, "Could not start background Mimics: {0}".format(e))
        return None
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


# -- Auto-open first .mcs monitor --------------------------------------

def _first_mcs_monitor_tick(monitor):
    """Timer callback: check if first .mcs file exists and is stable."""
    if monitor.get("done"):
        return

    output_dir = monitor.get("output_dir")
    monitor_key = monitor.get("monitor_key")

    if monitor.get("first_notified"):
        status = runtime_common.read_json(_rt(output_dir, "_mcs_batch_status.json"), {}) or {}
        signature = (status.get("status"), status.get("case_id"), status.get("completed"), status.get("failed"))
        if signature != monitor.get("last_batch_signature"):
            monitor["last_batch_signature"] = signature
            state = str(status.get("status") or "waiting")
            _user_progress(
                logging.INFO,
                "Import .mcs creation: {0}; completed {1}, failed {2}{3}.".format(
                    state,
                    int(status.get("completed", 0) or 0),
                    int(status.get("failed", 0) or 0),
                    "; current " + str(status.get("case_id")) if status.get("case_id") else "",
                ),
            )
        if status.get("status") == "closed":
            monitor["done"] = True
            _stop_import_monitor(monitor_key)
            completed = int(status.get("completed", 0) or 0)
            failed = int(status.get("failed", 0) or 0)
            _safe_message_box(
                "导入完成（部分失败）" if failed else "导入完成",
                "后台 .mcs 创建已结束。\n\n成功：{0}\n失败：{1}\n输出目录：{2}{3}".format(
                    completed,
                    failed,
                    output_dir,
                    (
                        "\n\n每个失败病例均记录了其原始源路径。"
                        "请在任务状态窗口（01_Data > 04）中打开本次导入，"
                        "重试前先点击打开日志查看详情。"
                        if failed
                        else ""
                    ),
                ),
                ui_blocking=False,
            )
        elif status.get("status") == "failed":
            monitor["done"] = True
            _stop_import_monitor(monitor_key)
            worker_error = status.get("error", "Background Mimics could not complete .mcs creation.")
            _category, _guidance_message, guidance_action = _error_guidance(worker_error, "background_mimics")
            _safe_message_box(
                "导入后台任务失败",
                (
                    "{0}\n\n建议操作：{1}\n\n已准备的数据已保留，"
                    "当前打开的 Mimics 工程未被修改。\n\nMimics 日志：{2}\n进程日志：{3}"
                ).format(
                    worker_error,
                    guidance_action,
                    status.get("mimics_log", ""),
                    status.get("process_log", ""),
                ),
                ui_blocking=False,
            )
        elif status:
            try:
                stale_seconds = time.time() - float(status.get("updated_at_epoch", 0.0) or 0.0)
            except Exception:
                stale_seconds = 0.0
            holder = runtime_common.active_resource_lock(
                _project_root(), runtime_common.background_mimics_lock_name(output_dir)
            )
            if stale_seconds > 90.0 and not holder:
                monitor["done"] = True
                _stop_import_monitor(monitor_key)
                _safe_message_box(
                    "导入意外停止",
                    (
                        "后台 .mcs 创建在报告完成前停止了。\n\n"
                        "建议操作：已准备的文件已保留，可直接重试。请检查后台 Mimics "
                        "进程是否仍被允许运行（管理菜单 > 停止所有自有服务，然后重试），"
                        "或在任务状态窗口（01_Data > 04）中打开本次导入并查看其日志。"
                    ),
                    ui_blocking=False,
                )
            elif status.get("status") not in ("closed", "failed", "cancelled"):
                detail = "{0}|{1}|{2}".format(
                    status.get("status") or "waiting",
                    status.get("case_id") or "",
                    status.get("completed") or 0,
                )
                due, elapsed = runtime_common.progress_notice_due(
                    monitor,
                    "mcs_creation_progress",
                    detail=detail,
                    interval_seconds=60.0,
                    initial_delay_seconds=60.0,
                )
                if due:
                    _user_progress(
                        logging.INFO,
                        "Background .mcs creation is still running ({0}s in "
                        "the current stage); completed {1}, failed {2}. Use "
                        "Task Status"
                        " (01 Data menu) to cancel.".format(
                            int(elapsed),
                            int(status.get("completed", 0) or 0),
                            int(status.get("failed", 0) or 0),
                        ),
                    )
        return

    # Timeout check
    if time.time() > monitor.get("deadline", 0):
        monitor["done"] = True
        _stop_import_monitor(monitor_key)
        return

    target_mcs = monitor.get("target_mcs")
    after_epoch = monitor.get("after_epoch")
    candidates = []
    if target_mcs:
        candidates = [target_mcs]
    else:
        # Batch wait: listing the whole output dir costs one directory
        # stat per case on a network share and this timer can live for
        # days. The batch status file is a single cheap read, so use it
        # to decide when a scan is worth it: scan immediately once the
        # worker reports its first completed case, otherwise at most
        # every 30 seconds.
        now_epoch = time.time()
        status = runtime_common.read_json(_rt(output_dir, "_mcs_batch_status.json"), {}) or {}
        try:
            completed_count = int(status.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed_count = 0
        due_for_scan = (
            completed_count >= 1
            or now_epoch - float(monitor.get("last_scan_epoch", 0.0) or 0.0) >= 30.0
        )
        if not due_for_scan:
            return
        monitor["last_scan_epoch"] = now_epoch
        try:
            items = sorted(os.listdir(output_dir))
        except Exception:
            return
        candidates = [os.path.join(output_dir, item) for item in items if item.endswith(".mcs")]

    for mcs_path in candidates:
        if not os.path.isfile(mcs_path):
            continue
        if after_epoch:
            try:
                if os.path.getmtime(mcs_path) < after_epoch:
                    continue
            except Exception:
                continue
        try:
            size = os.path.getsize(mcs_path)
        except Exception:
            continue
        stable_key = os.path.abspath(mcs_path)
        previous = monitor.get("last_file")
        previous_size = monitor.get("last_size")
        if previous != stable_key or previous_size != size:
            monitor["last_file"] = stable_key
            monitor["last_size"] = size
            return
        keep_for_batch = not bool(target_mcs)
        if not keep_for_batch:
            monitor["done"] = True
            _stop_import_monitor(monitor_key)
        try:
            if monitor.get("notify_only"):
                _append_import_log(output_dir, ".mcs is ready: {0}".format(mcs_path))
                try:
                    mimics.dialogs.message_box(
                        title="MCS 已就绪",
                        message="已转换的 .mcs 文件已就绪：\n{0}".format(mcs_path),
                        ui_blocking=False,
                    )
                except TypeError:
                    mimics.dialogs.message_box(
                        title="MCS 已就绪",
                        message="已转换的 .mcs 文件已就绪：\n{0}".format(mcs_path),
                    )
                if keep_for_batch:
                    monitor["first_notified"] = True
                    monitor["deadline"] = time.time() + 7 * 86400
                return
            if monitor.get("skip_if_project_open"):
                try:
                    if len(mimics.data.images) > 0:
                        _append_import_log(output_dir, "Skipped automatic .mcs open because a project is already open: {0}".format(mcs_path))
                        return
                except Exception:
                    pass
            mimics.file.open_project(filename=mcs_path)
            _append_import_log(output_dir, "Automatically opened .mcs: {0}".format(mcs_path))
            if keep_for_batch:
                monitor["first_notified"] = True
                monitor["deadline"] = time.time() + 7 * 86400
        except Exception as exc:
            _append_import_log(output_dir, "Could not automatically open .mcs {0}: {1}".format(mcs_path, exc))
        return

    due, elapsed = runtime_common.progress_notice_due(
        monitor,
        "first_mcs_wait",
        detail=str(target_mcs or output_dir),
        interval_seconds=60.0,
        initial_delay_seconds=30.0,
    )
    if due:
        _user_progress(
            logging.INFO,
            "Waiting for the first .mcs file ({0}s). Preparation and "
            "background Mimics continue independently; use Task Status"
            " (01 Data menu) to cancel.".format(int(elapsed)),
        )


def _start_first_mcs_monitor(output_dir, target_mcs=None, timeout_seconds=900, poll_seconds=2.0,
                             after_epoch=None, skip_if_project_open=False, notify_only=False):
    """Start a timer that polls for the first stable .mcs file."""
    monitor = {
        "monitor_key": "first_mcs_" + (target_mcs or output_dir),
        "output_dir": output_dir,
        "target_mcs": target_mcs,
        "after_epoch": after_epoch,
        "skip_if_project_open": skip_if_project_open,
        "notify_only": notify_only,
        "done": False,
        "deadline": time.time() + (timeout_seconds if target_mcs else max(timeout_seconds, 7 * 86400)),
    }

    if os.name != "nt" and _start_mimics_event_monitor(
        monitor,
        lambda: _first_mcs_monitor_tick(monitor),
        poll_seconds,
        "_start_first_mcs_monitor Mimics timer callback failed",
    ):
        return True

    # Non-Windows fallback through Qt event loop.
    if os.name != "nt":
        try:
            from PyQt5.QtCore import QTimer
            timer = QTimer()
            _stop_import_monitor(monitor["monitor_key"])
            monitor["timer"] = timer
            _IMPORT_MONITORS[monitor["monitor_key"]] = monitor

            def _tick():
                try:
                    _first_mcs_monitor_tick(monitor)
                except Exception as exc:
                    _append_import_exception(monitor.get("output_dir", ""), "_start_first_mcs_monitor qtimer callback failed", exc)

            timer.timeout.connect(_tick)
            timer.start(int(poll_seconds * 1000))
            return True
        except Exception:
            pass

    # Fall back to Win32 SetTimer
    if os.name != "nt":
        return False
    try:
        import ctypes
        monitor_key = monitor["monitor_key"]
        _stop_import_monitor(monitor_key)
        user32 = ctypes.windll.user32
        timer_interval_ms = int(poll_seconds * 1000)
        TIMERPROC = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_uint,
        )

        def _timer_proc(hwnd, message, timer_id, tick_count):
            try:
                _first_mcs_monitor_tick(monitor)
            except Exception as exc:
                _append_import_exception(monitor.get("output_dir", ""), "_start_first_mcs_monitor win32 timer callback failed", exc)

        callback = TIMERPROC(_timer_proc)
        callback_ptr = ctypes.cast(callback, ctypes.c_void_p)
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_ptr)
        if not timer_id:
            if _allow_windows_event_monitor() and _start_mimics_event_monitor(
                monitor,
                lambda: _first_mcs_monitor_tick(monitor),
                poll_seconds,
                "_start_first_mcs_monitor Mimics timer callback failed",
            ):
                return True
            return False
        monitor["callback"] = callback
        monitor["win32_timer"] = (user32, timer_id)
        _IMPORT_MONITORS[monitor_key] = monitor
        return True
    except Exception:
        if _allow_windows_event_monitor() and _start_mimics_event_monitor(
            monitor,
            lambda: _first_mcs_monitor_tick(monitor),
            poll_seconds,
            "_start_first_mcs_monitor Mimics timer callback failed",
        ):
            return True
        return False


def _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for import result polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )

    def _timer_proc(hwnd, message, timer_id, tick_count):
        try:
            _import_monitor_tick(monitor)
        except Exception as exc:
            out = ""
            try:
                out = os.path.dirname(os.path.abspath(monitor.get("output_mcs") or ""))
            except Exception:
                out = ""
            _append_import_exception(out, "_start_win32_import_monitor timer callback failed", exc)

    callback = TIMERPROC(_timer_proc)
    callback_ptr = ctypes.cast(callback, ctypes.c_void_p)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_ptr)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _start_import_monitor(job_dir, output_mcs, work_dir, timeout_seconds=1800,
                          poll_seconds=0.5, batch_queue=None, batch_info=None):
    """Start a non-blocking timer that polls for bridge completion and auto-applies."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "output_mcs": output_mcs,
        "work_dir": work_dir,
        "case_id": os.path.splitext(os.path.basename(output_mcs or ""))[0],
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "batch_queue": batch_queue,
        "completed": 0,
        "failed": 0,
    }
    if batch_info:
        monitor.update(batch_info)

    if os.name == "nt":
        # Use Win32 timer first on Windows to avoid Mimics event subscription
        # cleanup issues in some Mimics builds.
        if _start_win32_import_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        if _allow_windows_event_monitor() and _start_mimics_event_monitor(
            monitor,
            lambda: _import_monitor_tick(monitor),
            poll_seconds,
            "_start_import_monitor Mimics timer callback failed",
        ):
            return True
    elif _start_mimics_event_monitor(
        monitor,
        lambda: _import_monitor_tick(monitor),
        poll_seconds,
        "_start_import_monitor Mimics timer callback failed",
    ):
        return True

    # Non-Windows fallback: use the existing Mimics Qt event loop.
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        mimics.dialogs.message_box(
            title="导入进行中",
            message=(
                "数据集转换已开始，但结果无法自动排队。"
                "请稍后重新运行导入以查看进度。"
            ),
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        mimics.dialogs.message_box(
            title="导入进行中",
            message=(
                "数据集转换已开始，但结果无法自动排队。"
                "请稍后重新运行导入以查看进度。"
            ),
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        try:
            _import_monitor_tick(monitor)
        except Exception as exc:
            out = ""
            try:
                out = os.path.dirname(os.path.abspath(monitor.get("output_mcs") or ""))
            except Exception:
                out = ""
            _append_import_exception(out, "_start_import_monitor qtimer callback failed", exc)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


# -- Disk space helpers -------------------------------------------------


def _check_disk_space(path, required_mb):
    """Check if the drive containing *path* has at least *required_mb* MB free.

    Returns (ok, free_mb). On platforms where statvfs is unavailable,
    returns (True, -1) so the caller can proceed.
    """
    try:
        if os.name == "nt":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(os.path.abspath(path)),
                ctypes.pointer(free_bytes),
                None,
                None,
            )
            free_mb = free_bytes.value / (1024 * 1024)
        else:
            stat = os.statvfs(path)
            free_mb = stat.f_bavail * stat.f_frsize / (1024 * 1024)
        return (free_mb >= required_mb, free_mb)
    except Exception:
        return (True, -1)


def _mask_name_from_path(path):
    """Derive a mask name from a file path, stripping common suffixes."""
    name = os.path.basename(path)
    lower = name.lower()
    # Strip compression suffix first
    if lower.endswith(".gz"):
        name = name[:-3]
        lower = name.lower()
    # Strip format suffix
    for suffix in (".nii", ".mha", ".mhd", ".nrrd", ".seg"):
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name or "mask"


def _filter_case_masks(case_info, selection):
    """Apply all/none/comma-separated mask selection to case metadata."""
    if not case_info:
        return case_info
    text = str(selection or "all").strip()
    if not text or text.lower() == "all":
        return case_info
    result = dict(case_info)
    if text.lower() in ("none", "no", "off"):
        result["masks"] = []
        return result
    wanted = set(item.strip().lower() for item in text.split(",") if item.strip())
    result["masks"] = [
        item for item in case_info.get("masks", [])
        if str(item.get("name", "")).strip().lower() in wanted
    ]
    return result


# -- Discover monitor (batch mode: discover -> import chain) ------------

def _discover_monitor_tick(monitor):
    """Timer callback for discover phase: check if bridge discover finished."""
    try:
        if monitor.get("done"):
            return

        job_dir = monitor.get("job_dir")
        monitor_key = monitor.get("monitor_key")

        if _import_task_stopped(monitor):
            _cancel_import_monitor(monitor)
            return

        # Timeout check
        if time.time() > monitor.get("deadline", 0):
            _fail_import_monitor_after_process(
                monitor,
                "scan_timeout",
                "Dataset scan timed out.",
                message_title="扫描超时",
                message=(
                    "数据集扫描已超时。外部扫描器正在被停止；"
                    "其退出后任务将变为失败。"
                ),
            )
            return

        status, result = _check_job_status(job_dir)
        previous_status = monitor.get("_last_discover_status", "")
        if status != previous_status:
            monitor["_last_discover_status"] = status
            _verbose_log(
                monitor.get("output_dir", ""),
                "Discover status | {0} -> {1}".format(previous_status or "<none>", status),
            )

        if status == "running":
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "dataset_discovery",
                detail="running",
                interval_seconds=60.0,
                initial_delay_seconds=30.0,
            )
            if due:
                _user_progress(
                    logging.INFO,
                    "Dataset discovery is still running in external Python "
                    "({0}s). Mimics remains available. Use Task Status"
                    " (01 Data menu) to cancel.".format(int(elapsed)),
                )
            return  # still discovering

        runtime_common.clear_progress_notice(monitor, "dataset_discovery")

        monitor["done"] = True
        _stop_import_monitor(monitor_key)

        if status == "error":
            _append_import_log(
                monitor.get("output_dir", ""),
                "Discover failed | {0}".format(result),
            )
            _safe_message_box("扫描错误", "数据集扫描失败：{0}".format(result))
            _write_import_task_status(
                monitor.get("task_status_path"),
                {"status": "failed", "phase": "scan_failed", "error": str(result)},
            )
            _cleanup_job_dir(job_dir)
            _release_import_producer_lease(monitor)
            return

        # Discover done; result contains cases list.
        cases = result.get("cases", [])
        count = result.get("count", len(cases))
        _verbose_log(
            monitor.get("output_dir", ""),
            "Discover done | case_count={0}".format(count),
        )
        _cleanup_job_dir(job_dir)

        if not cases:
            _safe_message_box("导入", "所选文件夹中未找到病例数据。")
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "failed",
                    "phase": "no_cases",
                    "error": "No case data was found in the selected folder.",
                    "completed": 0,
                    "failed": 0,
                    "total": 0,
                },
            )
            _release_import_producer_lease(monitor)
            return

        if result.get("mask_mode") == "named" and int(result.get("mask_count", 0) or 0) == 0:
            _safe_message_box(
                "未找到匹配的掩膜",
                "没有分割文件匹配所请求的掩膜名称。请检查拼写，或选择全部 Mask / 仅图像。",
                ui_blocking=False,
            )
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "failed",
                    "phase": "no_matching_masks",
                    "error": "No segmentation files matched the requested mask names.",
                    "completed": 0,
                    "failed": 0,
                    "total": count,
                },
            )
            _release_import_producer_lease(monitor)
            return

        output_dir = monitor.get("output_dir")
        _append_import_log(output_dir, "Discovered {0} case(s); starting preparation.".format(count))
        _write_import_task_status(
            monitor.get("task_status_path"),
            {
                "status": "running",
                "phase": "preparing",
                "completed": 0,
                "failed": 0,
                "total": count,
            },
        )
        if result.get("mask_mode") == "named":
            _append_import_log(
                output_dir,
                "Mask filter matched {0} file(s); {1} case(s) have no matching mask and will be imported as images only.".format(
                    int(result.get("mask_count", 0) or 0),
                    int(result.get("cases_without_selected_masks", 0) or 0),
                ),
            )

        # Derived DICOM and buffers now live in the local run directory, so
        # capacity must be checked there rather than on the .mcs output share.
        work_root = monitor.get("work_root")
        if not os.path.isdir(work_root):
            os.makedirs(work_root)
        estimated_mb = count * 500
        ok, free_mb = _check_disk_space(work_root, estimated_mb)
        _verbose_log(
            output_dir,
            "Disk space | mb_est={0} ok={1} free={2}".format(
                estimated_mb, ok,
                int(free_mb) if free_mb is not None else "?",
            ),
        )
        if not ok:
            _safe_message_box(
                "磁盘空间不足",
                "本地工作区磁盘空间不足。预计需要：{0} MB；"
                "可用：{1} MB。请释放磁盘空间后重试。".format(
                    estimated_mb, int(free_mb)),
            )
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "failed",
                    "phase": "insufficient_disk_space",
                    "error": "Insufficient local workspace disk space.",
                    "completed": 0,
                    "failed": 0,
                    "total": count,
                },
            )
            _release_import_producer_lease(monitor)
            return

        _mark_mcs_queue_active(output_dir, count)
        monitor["queue_marked_active"] = True

        # Start batch prepare-only flow (no Mimics API calls, GUI stays responsive)
        axes = monitor.get("axes")
        flips = monitor.get("flips")
        jobs_dir = monitor.get("jobs_dir")

        batch_info = {
            "total": count,
            "output_dir": output_dir,
            "work_root": monitor.get("work_root"),
            "axes": axes,
            "flips": flips,
            "jobs_dir": jobs_dir,
            "batch_started_epoch": time.time(),
            "task_status_path": monitor.get("task_status_path"),
            "task_stop_path": monitor.get("task_stop_path"),
            "producer_lock_path": monitor.get("producer_lock_path", ""),
            "producer_lock_token": monitor.get("producer_lock_token", ""),
            "producer_output_dir": monitor.get("producer_output_dir", output_dir),
            "queue_marked_active": True,
            # The first case uses the same guarded path as every later case.
            # No first-case exception can escape and stop the whole batch.
            "selecting_next": True,
        }
        initial_monitor_key = os.path.join(jobs_dir, "_batch_start")
        _verbose_log(output_dir, "Batch prepare | queued={0}".format(len(cases)))
        monitor_started = _start_batch_prepare_monitor(
            initial_monitor_key, "",
            batch_queue=cases,
            batch_info=batch_info,
        )
        if not monitor_started:
            _write_import_task_status(
                monitor.get("task_status_path"),
                {
                    "status": "failed",
                    "phase": "monitor_unavailable",
                    "error": "Mimics could not monitor the background conversion queue.",
                    "completed": 0,
                    "failed": 0,
                    "total": count,
                },
            )
            _finish_import_producer(monitor, completed=0, failed=0)
        else:
            # Ownership moved to the batch monitor. The discover monitor must
            # not release the same token during any later cleanup callback.
            monitor["producer_lock_token"] = ""
    except Exception as exc:
        output_dir = monitor.get("output_dir", "")
        _append_import_exception(output_dir, "_discover_monitor_tick fatal", exc)
        monitor["done"] = True
        try:
            _stop_import_monitor(monitor.get("monitor_key"))
        except Exception:
            pass
        _safe_message_box("导入错误", "导入失败，发生意外错误。技术详情见导入输出目录的 logs 子目录。")
        _write_import_task_status(
            monitor.get("task_status_path"),
            {"status": "failed", "phase": "monitor_failed", "error": str(exc)},
        )
        _finish_import_producer(
            monitor,
            completed=monitor.get("completed", 0),
            failed=max(1, int(monitor.get("failed", 0) or 0)),
        )


def _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
    """Start a Win32 timer for discover result polling."""
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    monitor_key = monitor.get("monitor_key")
    _stop_import_monitor(monitor_key)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )

    def _timer_proc(hwnd, message, timer_id, tick_count):
        try:
            _discover_monitor_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_win32_discover_monitor timer callback failed", exc)

    callback = TIMERPROC(_timer_proc)
    callback_ptr = ctypes.cast(callback, ctypes.c_void_p)
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_ptr)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _IMPORT_MONITORS[monitor_key] = monitor
    return True


def _start_import_discover_monitor(job_dir, ts_root, output_dir, axes, flips, jobs_dir, work_root,
                                    poll_seconds=0.5, timeout_seconds=900,
                                    task_status_path=None, task_stop_path=None,
                                    producer_lease=None):
    """Start a non-blocking timer that polls for discover completion, then auto-starts import."""
    monitor = {
        "monitor_key": job_dir,
        "job_dir": job_dir,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
        "ts_root": ts_root,
        "output_dir": output_dir,
        "axes": axes,
        "flips": flips,
        "jobs_dir": jobs_dir,
        "work_root": work_root,
        "task_status_path": task_status_path,
        "task_stop_path": task_stop_path,
    }
    if producer_lease:
        monitor.update(producer_lease)
    _verbose_log(
        output_dir,
        "Discover monitor | ts_root={0} | poll={1}s | timeout={2}s".format(
            ts_root, poll_seconds, timeout_seconds,
        ),
    )

    if os.name == "nt":
        # Use Win32 timer first on Windows to avoid Mimics event subscription
        # cleanup issues in some Mimics builds.
        if _start_win32_discover_monitor(monitor, poll_seconds, timeout_seconds):
            return True
        if _allow_windows_event_monitor() and _start_mimics_event_monitor(
            monitor,
            lambda: _discover_monitor_tick(monitor),
            poll_seconds,
            "_start_import_discover_monitor Mimics timer callback failed",
        ):
            return True
    elif _start_mimics_event_monitor(
        monitor,
        lambda: _discover_monitor_tick(monitor),
        poll_seconds,
        "_start_import_discover_monitor Mimics timer callback failed",
    ):
        return True

    # Non-Windows fallback: use the existing Qt event loop.
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        mimics.dialogs.message_box(
            title="扫描进行中",
            message="数据集扫描已开始，但进度无法自动监控。",
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        mimics.dialogs.message_box(
            title="扫描进行中",
            message="数据集扫描已开始，但进度无法自动监控。",
        )
        return False

    timer = QTimer()
    _stop_import_monitor(job_dir)
    monitor["timer"] = timer
    _IMPORT_MONITORS[job_dir] = monitor

    def _tick():
        try:
            _discover_monitor_tick(monitor)
        except Exception as exc:
            _append_import_exception(monitor.get("output_dir", ""), "_start_import_discover_monitor qtimer callback failed", exc)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


# -- Single case import -------------------------------------------------

def _build_bridge_params(case_info, axes, flips, work_dir):
    """Build bridge params for the prepare action."""
    buffers_dir = os.path.join(work_dir, "buffers")
    if not os.path.isdir(buffers_dir):
        os.makedirs(buffers_dir)
    dicom_out = os.path.join(work_dir, "derived_dicom")

    return {
        "action": "prepare",
        "image_path": case_info["image"],
        "masks": case_info.get("masks", []),
        "dicom_out": dicom_out,
        "buffers_out": buffers_dir,
        "axes": axes,
        "flips": flips,
        "case_id": case_info["case_id"],
        "case_dir": case_info.get("case_dir", ""),
    }


def _run_main_with_args(args, import_mode=None, case_info_override=None):
    previous = list(sys.argv)
    try:
        sys.argv = [previous[0]] + list(args)
        return main(import_mode=import_mode, case_info_override=case_info_override)
    finally:
        sys.argv = previous


def _single_case_worker_script():
    candidates = [
        os.path.join(_project_root(), "tools", "single_case_import_worker.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "single_case_import_worker.py"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    return os.path.abspath(candidates[0])


def _launch_single_case_worker(selection, axes=None, flips=None):
    """Start all single-case preparation outside the foreground Mimics process."""
    global _LAST_TASK_DESCRIPTOR

    source = str((selection or {}).get("source_path", "") or "")
    output_dir = os.path.abspath(str((selection or {}).get("output_path", "") or ""))
    case_info = (selection or {}).get("case_info") or {}
    mask_selection = str((selection or {}).get("mask_selection", "all") or "all")
    case_info = _filter_case_masks(case_info, mask_selection)
    if not source or not output_dir or not case_info:
        raise RuntimeError("Single-case import requires a valid source, output folder, and discovered image.")
    if (
        mask_selection.strip().lower() not in ("", "all", "none", "no", "off")
        and not case_info.get("masks")
    ):
        raise RuntimeError(
            "No segmentation file matched: {0}. Check the mask name or choose All masks / Images only.".format(
                mask_selection
            )
        )

    python_exe = _python_exe()
    worker_script = _single_case_worker_script()
    if not os.path.isfile(python_exe):
        raise RuntimeError(
            "The nninteractive_env Python was not found: {0}. Run setup_offline.bat first.".format(
                python_exe
            )
        )
    if not os.path.isfile(worker_script):
        raise RuntimeError("Single-case import worker was not found: {0}".format(worker_script))

    run_root = _new_import_run_root()
    status_path, stop_path = _set_last_import_task(
        run_root,
        output_dir,
        "Import single case",
    )
    selection_path = os.path.join(run_root, "selection.json")
    worker_log_path = os.path.join(run_root, "single_case_import.log")
    payload = dict(selection or {})
    payload.update({
        "source_path": source,
        "output_path": output_dir,
        "case_info": case_info,
        "mask_selection": mask_selection,
        "axes": list(axes or [0, 1, 2]),
        "flips": list(flips or [False, False, False]),
        "project_root": _project_root(),
    })
    _write_json_atomic(selection_path, payload)
    _write_import_task_status(
        status_path,
        {
            "status": "running",
            "phase": "starting_worker",
            "completed": 0,
            "failed": 0,
            "total": 1,
            "progress_percent": 1,
            "case_id": case_info.get("case_id", ""),
        },
    )

    log_handle = None
    try:
        log_handle = open(worker_log_path, "ab")
        process = subprocess.Popen(
            [
                python_exe,
                worker_script,
                "--selection-json", selection_path,
                "--status-path", status_path,
                "--stop-path", stop_path,
                "--log-path", worker_log_path,
            ],
            cwd=_project_root(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_background_env(),
            **_background_process_kwargs()
        )
    except Exception as exc:
        _write_import_task_status(
            status_path,
            {
                "status": "failed",
                "phase": "worker_launch_failed",
                "error": str(exc),
                "completed": 0,
                "failed": 1,
                "total": 1,
            },
        )
        raise
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass

    descriptor = dict(_LAST_TASK_DESCRIPTOR)
    descriptor["worker_pid"] = int(process.pid)
    descriptor["log_path"] = worker_log_path
    descriptor["output_mcs"] = os.path.join(output_dir, case_info.get("case_id", "case") + ".mcs")
    _LAST_TASK_DESCRIPTOR = descriptor
    runtime_common.register_process(
        runtime_common.project_root(),
        "single_case_import_worker",
        process.pid,
        parent_pid=os.getpid(),
        state_path=status_path,
    )
    _user_progress(
        logging.INFO,
        "Single-case import started in an external worker (PID={0}).".format(process.pid),
    )
    return dict(descriptor)


def _launch_external_import_setup(import_mode):
    import io_setup_mimics

    mode = "import_single" if import_mode == "single_case" else "import_batch"
    configured = str(_load_data_io_config().get("mimics_output_dir", "") or "")
    _checkpoint_record(
        "external_setup_launch",
        mode=mode,
        configured_output=configured,
    )

    def submitted(selection):
        global _LAST_TASK_DESCRIPTOR
        _checkpoint_record(
            "external_setup_submitted",
            mode=mode,
            has_case_info=bool((selection or {}).get("case_info")),
        )
        source = str(selection.get("source_path", "") or "")
        output_dir = str(selection.get("output_path", "") or "")
        if not source or not output_dir:
            _checkpoint_record("external_setup_missing_paths", source=source, output_dir=output_dir)
            raise RuntimeError("The source and output paths were not returned by the path window.")
        if mode == "import_single":
            descriptor = _launch_single_case_worker(
                selection,
                axes=[0, 1, 2],
                flips=[False, False, False],
            )
            _checkpoint_record(
                "external_setup_single_worker_started",
                descriptor_keys=sorted(descriptor.keys()),
                worker_pid=descriptor.get("worker_pid"),
            )
            return descriptor
        else:
            args = ["--ts-root", source, "--output-dir", output_dir]
        args.extend(["--masks", str(selection.get("mask_selection", "all") or "all")])
        _checkpoint_record("external_setup_args_ready", mode=mode, args=args)
        _LAST_TASK_DESCRIPTOR = {}
        result = _run_main_with_args(
            args,
            import_mode=import_mode,
            case_info_override=selection.get("case_info"),
        )
        descriptor = dict(_LAST_TASK_DESCRIPTOR)
        _checkpoint_record(
            "external_setup_after_main",
            mode=mode,
            result=result,
            descriptor_keys=sorted(descriptor.keys()),
        )
        if not descriptor:
            raise RuntimeError(
                "Import did not start (result {0}). Review the Mimics log for the reported validation or concurrency error.".format(
                    result
                )
            )
        return descriptor

    return io_setup_mimics.launch(
        mode,
        _python_exe(),
        {"configured_output": configured},
        submitted,
    )


# -- Main entry ---------------------------------------------------------

def main(import_mode=None, case_info_override=None):
    """Entry point. Reads config from argv or interactive dialog.

    Usage:
        mimics_import.py --ts-root <dir> [--cases s0000,s0001] [--masks all|none|name1,name2] [--output-dir <dir>] [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --case-dir <dir> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --image-file <volume.mhd> --output <file.mcs> [--axes 0,1,2] [--flips false,false,false]
        mimics_import.py --image-file <volume.mhd> --mask-files mask1.nii.gz,mask2.nii.gz --output <file.mcs>

    import_mode:
        None         — interactive (ask)
        "single_case" — skip dialog, single-case mode directly
    """
    _checkpoint_record(
        "main_enter",
        import_mode=import_mode,
        argv=list(sys.argv),
        has_case_info_override=bool(case_info_override),
    )
    with _BG_MIMICS_STATE_LOCK:
        _BG_MIMICS_FAILED_OUTPUTS.clear()
    live_bridge_jobs = []
    for job_path, process in list(_BRIDGE_PROCESSES.items()):
        try:
            if process.poll() is None:
                live_bridge_jobs.append(job_path)
            else:
                _BRIDGE_PROCESSES.pop(job_path, None)
        except Exception:
            _BRIDGE_PROCESSES.pop(job_path, None)
    if live_bridge_jobs:
        _checkpoint_record("main_blocked_live_bridge", count=len(live_bridge_jobs))
        _safe_message_box(
            "导入已在进行中",
            "另一项图像导入仍在准备数据。请等待其完成，或先在任务状态窗口（01_Data > 04）中停止它，再开始新的导入。",
            ui_blocking=False,
        )
        return 2

    # Safe cleanup runs in a daemon thread. By default it only removes stale
    # resource locks; process killing is explicit or aggressive opt-in.
    if _startup_cleanup_enabled():
        _checkpoint_record("startup_cleanup_thread_begin")
        cleanup_thread = threading.Thread(target=_cleanup_stale_processes)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    else:
        _checkpoint_record("startup_cleanup_thread_skipped")

    # Parse args (simple, Python 3.5 compatible)
    ts_root = None
    case_dir = None
    image_file = None
    output = None
    output_dir = None
    cases_filter = None
    mask_files = None
    mask_selection = "all"
    axes = [0, 1, 2]
    flips = [False, False, False]

    args = sys.argv[1:]
    _verbose_log("", "main() argv={0}".format(sys.argv))
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--ts-root" and i + 1 < len(args):
            ts_root = args[i + 1]
            i += 2
        elif arg == "--case-dir" and i + 1 < len(args):
            case_dir = args[i + 1]
            i += 2
        elif arg == "--image-file" and i + 1 < len(args):
            image_file = args[i + 1]
            case_dir = image_file
            i += 2
        elif arg == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif arg == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif arg == "--cases" and i + 1 < len(args):
            cases_filter = set(c.strip() for c in args[i + 1].split(","))
            i += 2
        elif arg == "--mask-files" and i + 1 < len(args):
            mask_files = [p.strip() for p in args[i + 1].split(",") if p.strip()]
            i += 2
        elif arg == "--masks" and i + 1 < len(args):
            mask_selection = args[i + 1].strip() or "all"
            i += 2
        elif arg == "--axes" and i + 1 < len(args):
            axes = [int(v.strip()) for v in args[i + 1].split(",")]
            i += 2
        elif arg == "--flips" and i + 1 < len(args):
            parts = [v.strip().lower() for v in args[i + 1].split(",")]
            flips = [p in ("true", "1", "yes") for p in parts]
            i += 2
        else:
            i += 1

    _verbose_log(
        "",
        "Args | ts_root={0} | case_dir={1} | output={2} | output_dir={3} | axes={4} | flips={5}".format(
            ts_root,
            case_dir,
            output,
            output_dir,
            axes,
            flips,
        ),
    )
    _checkpoint_record(
        "args_parsed",
        ts_root=ts_root,
        case_dir=case_dir,
        output=output,
        output_dir=output_dir,
        mask_selection=mask_selection,
    )

    # Interactive path selection is hosted in external PySide6. Mimics only
    # launches it and polls a tiny status JSON through a GUI timer.
    if not ts_root and not case_dir:
        _checkpoint_record("launch_external_setup", import_mode=import_mode)
        return _launch_external_import_setup(import_mode)

    # -- Single case mode ----------------------------------------------

    if case_dir:
        _checkpoint_record("single_case_begin", case_dir=case_dir, output=output, output_dir=output_dir)
        _verbose_log("", "Mode: single-case | source={0}".format(case_dir))
        selected_source = os.path.abspath(case_dir)
        source_is_file = _is_medical_image_file(selected_source)
        source_case_dir = os.path.dirname(selected_source) if source_is_file else selected_source
        source_case_id = _image_stem(selected_source) if source_is_file else os.path.basename(selected_source)
        case_info = case_info_override or _discover_single_case(selected_source)
        _checkpoint_record(
            "single_case_discovered",
            selected_source=selected_source,
            case_found=bool(case_info),
            source_is_file=source_is_file,
        )
        case_info = _filter_case_masks(case_info, mask_selection)
        if case_info is None:
            _checkpoint_record("single_case_no_supported_data", selected_source=selected_source)
            mimics.dialogs.message_box(title="错误", message="未找到受支持的图像数据：{0}".format(selected_source))
            return 1
        if not output:
            dataset_root = source_case_dir if source_is_file else os.path.dirname(source_case_dir)
            output_dir_default = output_dir or _resolve_import_output_dir(dataset_root)
            output = os.path.join(output_dir_default, case_info["case_id"] + ".mcs")

        # Merge --mask-files from command line into case_info.
        # For interactive mask import, use the dedicated "Import Masks" entry.
        if mask_files:
            extra_masks = []
            for p in mask_files:
                if os.path.isfile(p):
                    extra_masks.append({
                        "name": _mask_name_from_path(p),
                        "path": os.path.abspath(p),
                    })
            existing_names = {m["name"] for m in case_info.get("masks", [])}
            for em in extra_masks:
                if em["name"] in existing_names:
                    case_info["masks"] = [m for m in case_info["masks"] if m["name"] != em["name"]]
                case_info["masks"].append(em)

        if mask_selection.lower() not in ("all", "none", "no", "off") and not case_info.get("masks"):
            _safe_message_box(
                "未找到匹配的掩膜",
                "没有分割文件匹配：{0}。请检查掩膜名称，或选择全部 Mask / 仅图像。".format(mask_selection),
                ui_blocking=False,
            )
            return 2

        output_dir_abs = os.path.dirname(os.path.abspath(output))
        run_root = _new_import_run_root()
        _checkpoint_record("single_case_run_root_created", run_root=run_root, output_dir=output_dir_abs)
        _checkpoint_record("single_case_before_set_last_task", run_root=run_root, output_dir=output_dir_abs)
        task_status_path, task_stop_path = _set_last_import_task(
            run_root, output_dir_abs, "Import single case",
        )
        _checkpoint_record(
            "single_case_after_set_last_task",
            status_path=task_status_path,
            task_stop_path=task_stop_path,
        )
        _checkpoint_record("single_case_before_write_status", status_path=task_status_path)
        _write_import_task_status(
            task_status_path,
            {
                "status": "running",
                "phase": "preparing",
                "completed": 0,
                "failed": 0,
                "total": 1,
                "case_id": case_info["case_id"],
            },
        )
        _checkpoint_record("single_case_after_write_status", status_path=task_status_path)
        work_dir = os.path.join(run_root, "work", case_info["case_id"])
        jobs_dir = os.path.join(run_root, "jobs")
        job_dir = os.path.join(jobs_dir, case_info["case_id"])
        _checkpoint_record("single_case_paths_ready", work_dir=work_dir, jobs_dir=jobs_dir, job_dir=job_dir)

        # Launch bridge in background + start timer to queue .mcs creation.
        _checkpoint_record("single_case_before_build_bridge_params", case_id=case_info["case_id"])
        bridge_params = _build_bridge_params(case_info, axes, flips, work_dir)
        _checkpoint_record("single_case_after_build_bridge_params", case_id=case_info["case_id"])
        _checkpoint_record("single_case_bridge_launch", job_dir=job_dir, case_id=case_info["case_id"])
        producer_lease = _acquire_import_producer_lease(
            output_dir_abs,
            "Mimics single-case preparation",
        )
        if not producer_lease:
            _write_import_task_status(
                task_status_path,
                {
                    "status": "failed",
                    "phase": "output_queue_busy",
                    "error": "Another import is preparing data for this output folder.",
                    "total": 1,
                },
            )
            _safe_message_box(
                "导入输出目录忙",
                "另一项 Mimics-Script 导入正在为该输出目录准备数据。"
                "请使用其他输出目录，或等待该准备工作完成。",
                ui_blocking=False,
            )
            return 75
        try:
            _launch_bridge_job_thread(bridge_params, job_dir, output_dir_abs, "preparing", case_id=case_info["case_id"])
        except Exception:
            _release_import_producer_lease(producer_lease)
            raise

        # Bridge launched, timer will queue the result for background Mimics.
        single_batch_info = {
            "total": 1,
            "output_dir": output_dir_abs,
            "task_status_path": task_status_path,
            "task_stop_path": task_stop_path,
            # User-recognizable original input path for failure records (R61-5).
            "case_image": str(case_info.get("image") or case_info.get("case_dir") or ""),
        }
        single_batch_info.update(producer_lease)
        monitor_started = _start_import_monitor(
            job_dir,
            output,
            work_dir,
            batch_info=single_batch_info,
        )
        _checkpoint_record(
            "single_case_monitor_started",
            monitor_started=bool(monitor_started),
            job_dir=job_dir,
            status_path=task_status_path,
        )
        if not monitor_started:
            _terminate_job_process(job_dir)
            _release_import_producer_lease(single_batch_info)
            _write_import_task_status(
                task_status_path,
                {
                    "status": "failed",
                    "phase": "monitor_unavailable",
                    "error": "Mimics could not monitor the background preparation process.",
                    "total": 1,
                },
            )
        return 0

    # -- Batch mode: discover and prepare cases without blocking Mimics GUI.

    # No confirmation dialog; user already chose the folder, just start.
    if not output_dir:
        # Resolve only the path on the GUI thread. Directory creation and all
        # logging happen in the launch worker to avoid slow/network-drive I/O
        # immediately after the folder picker closes.
        output_dir = _resolve_import_output_dir(ts_root, create=False)
    _checkpoint_record("batch_begin", ts_root=ts_root, output_dir=output_dir)

    run_root = _new_import_run_root()
    task_status_path, task_stop_path = _set_last_import_task(
        run_root, output_dir, "Import dataset",
    )
    _write_import_task_status(
        task_status_path,
        {
            "status": "running",
            "phase": "discovering_cases",
            "completed": 0,
            "failed": 0,
            "total": 0,
        },
    )
    jobs_dir = os.path.join(run_root, "jobs")
    work_root = os.path.join(run_root, "work")

    discover_job_dir = os.path.join(jobs_dir, "_discover")
    bridge_params = {
        "action": "discover",
        "ts_root": ts_root,
        "cases_filter": list(cases_filter) if cases_filter else None,
        "mask_selection": mask_selection,
    }
    _update_gui()
    _checkpoint_record("batch_discover_launch", discover_job_dir=discover_job_dir)
    producer_lease = _acquire_import_producer_lease(
        output_dir,
        "Mimics dataset preparation",
    )
    if not producer_lease:
        _write_import_task_status(
            task_status_path,
            {
                "status": "failed",
                "phase": "output_queue_busy",
                "error": "Another import is preparing data for this output folder.",
            },
        )
        _safe_message_box(
            "导入输出目录忙",
            "另一项 Mimics-Script 导入正在为该输出目录准备数据。"
            "Mimics 导入与外部导入入口共用此队列；请等待、停止该准备工作，"
            "或选择其他输出目录。",
            ui_blocking=False,
        )
        return 75
    try:
        _launch_bridge_job_thread(bridge_params, discover_job_dir, output_dir, "discovering")
    except Exception:
        _release_import_producer_lease(producer_lease)
        raise
    monitor_started = _start_import_discover_monitor(
        discover_job_dir,
        ts_root,
        output_dir,
        axes,
        flips,
        jobs_dir,
        work_root,
        task_status_path=task_status_path,
        task_stop_path=task_stop_path,
        producer_lease=producer_lease,
    )
    _checkpoint_record(
        "batch_discover_monitor_started",
        monitor_started=bool(monitor_started),
        discover_job_dir=discover_job_dir,
        status_path=task_status_path,
    )
    if not monitor_started:
        _terminate_job_process(discover_job_dir)
        _release_import_producer_lease(producer_lease)
        _write_import_task_status(
            task_status_path,
            {
                "status": "failed",
                "phase": "monitor_unavailable",
                "error": "Mimics could not monitor the background dataset scan.",
            },
        )
    return 0


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        try:
            mimics.dialogs.message_box(title="致命错误", message="错误：{0}".format(error))
        except Exception:
            pass
        raise
