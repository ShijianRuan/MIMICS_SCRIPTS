# -*- coding: utf-8 -*-
"""Mimics-side controller for task-adapted nnInteractive models.

This module is intentionally Python 3.5 compatible. PySide6, dataset scans,
training, validation, and checkpoint inspection stay in nninteractive_env.
"""

from __future__ import print_function

import glob
import json
import logging
import os
import subprocess
import sys
import time
import uuid

import mimics

import nninteractive_mimics
import runtime_common


TITLE = "nnInteractive Custom Models"
ACTION_MANAGE = "manage"
ACTION_ANNOTATE = "annotate"
TASK_ID_METADATA = "nninteractive.task_id"
MODEL_ID_METADATA = "nninteractive.model_id"
_CHOOSER_MONITORS = {}
_GUI_PROCESSES = {}


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        (
            "nninteractive_finetune_config.json",
            "nninteractive_config.json",
            "runtime_py35",
            ".git",
        ),
    )


def _read_json(path, default=None):
    return runtime_common.read_json(path, default)


def _write_json(path, value):
    return runtime_common.write_json_atomic(path, value)


def _config():
    path = os.path.join(_project_root(), "nninteractive_finetune_config.json")
    value = _read_json(path, {}) or {}
    value["_config_path"] = path
    return value


def _resolve_path(value):
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        return ""
    if not os.path.isabs(text):
        text = os.path.join(_project_root(), text)
    return os.path.abspath(text)


def _workspace():
    configured = (
        os.environ.get("NNINTERACTIVE_TASK_MODELS_DIR", "").strip()
        or _config().get("workspace_dir")
        or "nninteractive_task_models"
    )
    path = _resolve_path(configured)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def _runtime_dir():
    path = os.path.join(_project_root(), ".mimics_runtime", "nninteractive_task_models")
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def _log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        print("[nnInteractive custom models] {0}".format(message))


def _environment_python():
    root = _project_root()
    candidates = (
        os.path.join(root, "nninteractive_env", "python.exe"),
        os.path.join(root, "nninteractive_env", "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "python", "python.exe"),
        os.path.join(root, "nninteractive_env", "bin", "python3"),
        os.path.join(root, "nninteractive_env", "bin", "python"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run Setup Environment first."
    )


def _gui_python():
    python_exe = _environment_python()
    if os.name == "nt":
        pythonw = os.path.join(os.path.dirname(python_exe), "pythonw.exe")
        if os.path.isfile(pythonw):
            return pythonw
    return python_exe


def _gui_environment():
    env = dict(os.environ)
    paths = [
        _project_root(),
        os.path.join(_project_root(), "tools"),
        os.path.join(
            _project_root(), "external", "nninteractive-finetune", "src"
        ),
    ]
    existing = [
        item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    for path in reversed(paths):
        if path not in existing:
            existing.insert(0, path)
    env["PYTHONPATH"] = os.pathsep.join(existing)
    return env


def _open_gui_process(script_name, context_path, key):
    existing = _GUI_PROCESSES.get(key)
    if existing is not None:
        try:
            if existing.poll() is None:
                _log(logging.INFO, "{0} is already open.".format(TITLE))
                return existing, False
        except Exception:
            pass
        _GUI_PROCESSES.pop(key, None)
    script = os.path.join(_project_root(), "tools", script_name)
    if not os.path.isfile(script):
        raise RuntimeError("External UI script is missing: {0}".format(script))
    stderr_path = os.path.join(
        _runtime_dir(), "{0}_stderr.log".format(runtime_common.safe_slug(key))
    )
    stderr_handle = open(stderr_path, "ab")
    try:
        process = subprocess.Popen(
            [_gui_python(), script, "--context", context_path],
            cwd=_project_root(),
            env=_gui_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stderr_handle,
            stderr=stderr_handle,
        )
    finally:
        stderr_handle.close()
    _GUI_PROCESSES[key] = process
    return process, True


def _current_project_path():
    try:
        info = mimics.file.get_project_information()
    except Exception:
        return ""
    for name in ("filename", "file_name", "path", "project_path", "project_file"):
        try:
            value = getattr(info, name, None)
        except Exception:
            value = None
        if value:
            return os.path.abspath(str(value))
    return ""


def _same_image(left, right):
    if left is right:
        return True
    return bool(
        left is not None
        and right is not None
        and getattr(left, "guid", None)
        and getattr(left, "guid", None) == getattr(right, "guid", None)
    )


def _selected_mask():
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    selected = []
    for mask in mimics.data.masks:
        if not bool(getattr(mask, "selected", False)):
            continue
        try:
            mask_image = getattr(mask, "image", None)
        except Exception:
            mask_image = None
        if image is not None and mask_image is not None and not _same_image(mask_image, image):
            continue
        selected.append(mask)
    if len(selected) > 1:
        raise RuntimeError("Select exactly one target Mask before annotation.")
    return selected[0] if selected else None


def _metadata_get(obj, name, default=""):
    if obj is None:
        return default
    try:
        item = obj.metadata.find(name)
        if item is not None:
            return item.value
    except Exception:
        pass
    try:
        return obj.metadata[name].value
    except Exception:
        return default


def _active_image_metadata(name, default=""):
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    return _metadata_get(image, name, default)


def _safe_slug(value):
    text = str(value or "").strip().lower()
    result = []
    previous_separator = False
    for char in text:
        if char.isalnum():
            result.append(char)
            previous_separator = False
        elif not previous_separator:
            result.append("_")
            previous_separator = True
    return "".join(result).strip("_") or "unknown"


def _registry():
    payload = _read_json(os.path.join(_workspace(), "registry.json"), {}) or {}
    tasks = payload.get("tasks") or []
    return [row for row in tasks if isinstance(row, dict)]


def _models(task):
    result = [row for row in task.get("models") or [] if isinstance(row, dict)]
    result.sort(
        key=lambda row: float(row.get("created_at_epoch") or 0),
        reverse=True,
    )
    return result


def _find_task(task_id):
    wanted = _safe_slug(task_id)
    for task in _registry():
        if _safe_slug(task.get("task_id")) == wanted:
            return task
    return None


def _recommended_model(task):
    wanted = str(task.get("recommended_model_id") or "")
    for model in _models(task):
        if str(model.get("model_id") or "") == wanted:
            return model
    return None


def _registered_model_dir(task, model):
    workspace = _workspace()
    candidates = []
    relative = str(model.get("model_relpath") or "").strip()
    if relative:
        candidates.append(os.path.abspath(os.path.join(workspace, relative)))
    legacy = str(model.get("model_dir") or "").strip()
    if legacy:
        candidates.append(_resolve_path(legacy))
    candidates.append(
        os.path.join(
            workspace,
            "tasks",
            _safe_slug(task.get("task_id")),
            "models",
            _safe_slug(model.get("model_id")),
        )
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def _model_is_complete(task, model):
    model_dir = _registered_model_dir(task, model)
    for name in (
        "dataset.json",
        "plans.json",
        "inference_info.json",
        "inference_session_class.json",
    ):
        if not os.path.isfile(os.path.join(model_dir, name)):
            return False
    return bool(glob.glob(os.path.join(model_dir, "fold_*", "checkpoint_final.pth")))


def _profile(task, model):
    if not _model_is_complete(task, model):
        raise RuntimeError(
            "The selected custom model is incomplete. Open Train and Manage "
            "Custom Models "
            "to inspect or replace it."
        )
    return {
        "source": "task_model",
        "profile_id": "{0}:{1}".format(
            task.get("task_id"), model.get("model_id")
        ),
        "task_id": str(task.get("task_id") or ""),
        "task_name": str(
            task.get("task_name") or task.get("task_id") or ""
        ),
        "model_id": str(model.get("model_id") or ""),
        "model_dir": _registered_model_dir(task, model),
        "checkpoint_sha256": str(model.get("checkpoint_sha256") or ""),
        "strategy": str(model.get("strategy") or ""),
    }


def _binding_key(project_path):
    return os.path.normcase(os.path.abspath(project_path or "")).lower()


def _project_binding(project_path):
    if not project_path:
        return ""
    payload = _read_json(
        os.path.join(_workspace(), "project_bindings.json"), {}
    ) or {}
    bindings = payload.get("bindings") or {}
    value = bindings.get(_binding_key(project_path)) or {}
    return str(value.get("task_id") or "")


def _save_project_binding(project_path, task_id):
    if not project_path or not task_id:
        return
    path = os.path.join(_workspace(), "project_bindings.json")
    payload = _read_json(path, {}) or {}
    bindings = payload.get("bindings") or {}
    bindings[_binding_key(project_path)] = {
        "project_path": os.path.abspath(project_path),
        "task_id": str(task_id),
        "updated_at_epoch": time.time(),
    }
    payload["schema_version"] = "nninteractive_project_bindings.v1"
    payload["bindings"] = bindings
    _write_json(path, payload)


def _task_for_selected_context():
    tasks = _registry()
    if not tasks:
        return None
    mask = _selected_mask()
    metadata_task = str(_metadata_get(mask, TASK_ID_METADATA, "") or "")
    task = _find_task(metadata_task) if metadata_task else None
    if task is not None:
        return task

    mask_name = _safe_slug(getattr(mask, "name", "")) if mask is not None else ""
    if mask_name:
        matching = []
        for row in tasks:
            aliases = list(row.get("mask_names") or [])
            aliases.extend(
                [row.get("task_name") or "", row.get("task_id") or ""]
            )
            if mask_name in [_safe_slug(value) for value in aliases if value]:
                matching.append(row)
        if len(matching) == 1:
            return matching[0]

    bound = _project_binding(_current_project_path())
    task = _find_task(bound) if bound else None
    if task is not None:
        return task

    usable = [row for row in tasks if _recommended_model(row) is not None]
    return usable[0] if len(usable) == 1 else None


def _initial_context():
    mask = _selected_mask()
    project_path = _current_project_path()
    source_case_dir = str(
        _active_image_metadata("mimics_script.source_case_dir", "") or ""
    )
    return {
        "schema_version": "nninteractive_task_model_center_context.v1",
        "workspace": _workspace(),
        "project_root": _project_root(),
        "project_path": project_path,
        "mcs_dir": os.path.dirname(project_path) if project_path else "",
        "image_root": (
            os.path.dirname(source_case_dir)
            if source_case_dir and os.path.isdir(source_case_dir)
            else source_case_dir
        ),
        "mask_name": str(getattr(mask, "name", "") or "") if mask else "",
        "mimics_exe": runtime_common.find_mimics_exe() or "",
        "created_at_epoch": time.time(),
    }


def open_model_center():
    token = uuid.uuid4().hex
    context_path = os.path.join(
        _runtime_dir(), "center_{0}.json".format(token)
    )
    status_path = os.path.join(
        _runtime_dir(), "center_{0}_status.json".format(token)
    )
    context = _initial_context()
    context["ui_status_path"] = status_path
    _write_json(context_path, context)
    _write_json(status_path, {"status": "opening"})
    process, started = _open_gui_process(
        "nninteractive_task_model_center.py",
        context_path,
        "model_center",
    )
    if not started:
        return 0
    monitor = {
        "key": "center_{0}".format(token),
        "mode": "startup",
        "process": process,
        "status_path": status_path,
        "stderr_path": os.path.join(
            _runtime_dir(), "model_center_stderr.log"
        ),
        "deadline": time.time() + 30.0,
    }
    if not _start_choice_monitor(monitor, poll_seconds=0.5):
        _log(
            logging.WARNING,
            "The model center opened, but Mimics could not monitor its startup.",
        )
    _log(
        logging.INFO,
        "nnInteractive custom model management opened outside Mimics (PID {0}).".format(
            process.pid
        ),
    )
    return 0


def _stop_chooser_monitor(key):
    monitor = _CHOOSER_MONITORS.pop(key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    if timer is not None:
        try:
            timer.stop()
            timer.deleteLater()
        except Exception:
            pass
    win32 = monitor.get("win32_timer")
    if win32:
        try:
            win32[0].KillTimer(None, win32[1])
        except Exception:
            pass


def _finish_choice(key, payload):
    monitor = _CHOOSER_MONITORS.get(key) or {}
    _stop_chooser_monitor(key)
    if str(payload.get("status") or "") != "selected":
        if str(payload.get("status") or "") == "failed":
            message = str(payload.get("error") or "The model chooser failed.")
            _log(logging.ERROR, message)
            mimics.dialogs.message_box(
                message=message,
                title=TITLE,
                ui_blocking=False,
            )
        return
    profile = payload.get("profile") or {}
    if payload.get("remember_project"):
        _save_project_binding(
            payload.get("project_path") or monitor.get("project_path") or "",
            profile.get("task_id") or "",
        )
    try:
        nninteractive_mimics.run_with_model_profile(profile)
    except Exception as exc:
        _log(logging.ERROR, "Task-model annotation could not start: {0}".format(exc))
        mimics.dialogs.message_box(
            message="Task-model annotation could not start.\n\n{0}".format(exc),
            title=TITLE,
            ui_blocking=False,
        )


def _choice_tick(key):
    monitor = _CHOOSER_MONITORS.get(key)
    if not monitor:
        return
    payload = _read_json(monitor["status_path"], {}) or {}
    status = str(payload.get("status") or "")
    if monitor.get("mode") == "startup":
        if status == "ready":
            _stop_chooser_monitor(key)
            return
        if status == "closed":
            _stop_chooser_monitor(key)
            return
        if status == "failed":
            _finish_choice(key, payload)
            return
    if status in ("selected", "cancelled", "failed"):
        _finish_choice(key, payload)
        return
    process = monitor.get("process")
    try:
        exited = process is not None and process.poll() is not None
    except Exception:
        exited = False
    if exited:
        _finish_choice(
            key,
            {
                "status": "failed",
                "error": (
                    "The task-model chooser exited before a model was selected. "
                    "See {0}."
                ).format(monitor.get("stderr_path") or "the runtime log"),
            },
        )
        return
    if time.time() >= monitor["deadline"]:
        _finish_choice(
            key,
            {
                "status": "failed",
                "error": "The task-model chooser timed out.",
            },
        )


def _start_choice_monitor(monitor, poll_seconds=0.5):
    key = monitor["key"]
    _stop_chooser_monitor(key)
    _CHOOSER_MONITORS[key] = monitor
    if os.name == "nt":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetTimer.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            user32.SetTimer.restype = ctypes.c_size_t
            user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            user32.KillTimer.restype = ctypes.c_int
            timer_proc = ctypes.WINFUNCTYPE(
                None,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.c_uint,
            )

            def callback(_hwnd, _message, _timer_id, _tick_count):
                _choice_tick(key)

            callback_ref = timer_proc(callback)
            timer_id = user32.SetTimer(
                None,
                0,
                max(250, int(poll_seconds * 1000)),
                ctypes.cast(callback_ref, ctypes.c_void_p),
            )
            if timer_id:
                monitor["callback"] = callback_ref
                monitor["win32_timer"] = (user32, timer_id)
                return True
        except Exception:
            pass
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is not None:
            timer = QTimer()
            timer.timeout.connect(lambda: _choice_tick(key))
            timer.start(max(250, int(poll_seconds * 1000)))
            monitor["timer"] = timer
            return True
    except Exception:
        pass
    _CHOOSER_MONITORS.pop(key, None)
    return False


def _open_model_chooser(task_id=""):
    token = uuid.uuid4().hex
    context_path = os.path.join(_runtime_dir(), "choice_{0}.json".format(token))
    status_path = os.path.join(
        _runtime_dir(), "choice_{0}_status.json".format(token)
    )
    project_path = _current_project_path()
    context = {
        "schema_version": "nninteractive_task_model_choice_context.v1",
        "workspace": _workspace(),
        "status_path": status_path,
        "task_id": str(task_id or ""),
        "project_path": project_path,
        "created_at_epoch": time.time(),
    }
    _write_json(context_path, context)
    _write_json(status_path, {"status": "opening"})
    process, _started = _open_gui_process(
        "nninteractive_task_model_chooser.py",
        context_path,
        "model_chooser_{0}".format(token),
    )
    key = "choice_{0}".format(token)
    stderr_path = os.path.join(
        _runtime_dir(),
        "model_chooser_{0}_stderr.log".format(token),
    )
    monitor = {
        "key": key,
        "process": process,
        "status_path": status_path,
        "project_path": project_path,
        "stderr_path": stderr_path,
        "deadline": time.time() + 1800.0,
    }
    if not _start_choice_monitor(monitor):
        try:
            runtime_common.terminate_process_async(
                process=process, graceful_seconds=2.0
            )
        except Exception:
            pass
        raise RuntimeError(
            "Mimics could not monitor the external model chooser without "
            "blocking its GUI."
        )
    _log(logging.INFO, "Select a custom model in the external window.")
    return 0


def annotate_with_task_model():
    task = _task_for_selected_context()
    if task is None:
        tasks = _registry()
        if not tasks:
            _log(
                logging.WARNING,
                "No custom nnInteractive model exists yet. Opening model management.",
            )
            return open_model_center()
        return _open_model_chooser()
    model = _recommended_model(task)
    if model is None:
        return _open_model_chooser(task.get("task_id") or "")
    try:
        profile = _profile(task, model)
    except Exception as exc:
        _log(logging.ERROR, str(exc))
        mimics.dialogs.message_box(
            message=str(exc),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    return nninteractive_mimics.run_with_model_profile(profile)


def _stop_all_monitors():
    for key in list(_CHOOSER_MONITORS.keys()):
        _stop_chooser_monitor(key)
    for process in list(_GUI_PROCESSES.values()):
        try:
            runtime_common.terminate_process_async(
                process=process, graceful_seconds=2.0
            )
        except Exception:
            pass
    _GUI_PROCESSES.clear()


def main(action=ACTION_MANAGE):
    try:
        if action == ACTION_ANNOTATE:
            return annotate_with_task_model()
        return open_model_center()
    except Exception as exc:
        _log(logging.ERROR, "{0}: {1}".format(TITLE, exc))
        mimics.dialogs.message_box(
            message="{0} could not continue.\n\n{1}".format(TITLE, exc),
            title=TITLE,
            ui_blocking=False,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
