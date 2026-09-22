# -*- coding: utf-8 -*-
"""Nonblocking Mimics entry points for FlexiCT few-shot tasks.

Phase 4 scope: training setup, status viewer, stop. Prediction (Phase 5) and
active learning (Phase 6) extend this module.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only.
"""

from __future__ import print_function

import logging
import os
import threading
import time
import traceback
import uuid

import mimics

import external_window_launcher
import mimics_mask_apply
import runtime_common


TITLE = "FlexiCT"
BUTTON_TRAIN = "Train Model..."
BUTTON_PREDICT = "Predict Current Case..."
BUTTON_STATUS = "Show Status and Models"
BUTTON_STOP = "Stop Running Task"
BUTTON_ACTIVE_LEARNING = "Active Learning Review"
BUTTON_CANCEL = "Cancel"

_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        __file__,
        ("flexict_config.json", "runtime_py35"),
    )


def _read_json(path, default=None):
    return runtime_common.read_json(path, default)


def _write_json(path, payload):
    return runtime_common.write_json_atomic(path, payload)


def _settings():
    path = os.path.join(
        os.path.expanduser("~"), ".mimics_script", "flexict_settings.json"
    )
    return _read_json(path, {}) or {}


def _workspace():
    workspace = str(_settings().get("workspace") or "")
    if workspace:
        return os.path.abspath(workspace)
    return os.path.abspath(os.path.join(_project_root(), "flexict_models"))


def _external_python():
    config = mimics_mask_apply._config()
    return mimics_mask_apply._integration_python(
        config, mimics_mask_apply._integration_root(config)
    )


def _log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return
    except Exception:
        pass
    print("[FlexiCT] {0}".format(message))


def _setup_root():
    path = os.path.join(_workspace(), "setup")
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def _launch_gui(script_name, context, monitor_kind):
    setup_id = "{0}_{1}_{2}".format(
        monitor_kind, time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )
    root = _setup_root()
    status_path = os.path.join(root, setup_id + "_status.json")
    context_path = os.path.join(root, setup_id + "_context.json")
    stderr_path = os.path.join(root, setup_id + "_stderr.log")
    context = dict(context)
    context["setup_status_path"] = status_path
    context["workspace"] = _workspace()
    _write_json(
        status_path,
        {
            "schema_version": "mimics_flexict_setup.v1",
            "job_id": setup_id,
            "status": "opening",
            "phase": "opening",
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    _write_json(context_path, context)
    script = os.path.join(_project_root(), "tools", script_name)
    if not os.path.isfile(script):
        raise RuntimeError("External FlexiCT window is missing: {0}".format(script))
    process = mimics_mask_apply._launch_gui_process(
        [_external_python(), script, "--context", context_path],
        cwd=_project_root(),
        stderr_log=stderr_path,
    )
    monitor = {
        "monitor_key": setup_id,
        "kind": monitor_kind,
        "status_path": status_path,
        "controller_pid": process.pid,
        "stderr_path": stderr_path,
        "deadline": time.time() + 3600,
        "last_line": "",
    }
    _start_monitor(monitor, 1.0)
    return process.pid


def _training_context():
    settings = _settings()
    selected = _selected_masks()
    dataset_root = str(settings.get("dataset_root") or "")
    mcs_dir = str(settings.get("mcs_dir") or "")
    try:
        inferred = mimics_mask_apply._infer_dataset_root_from_project()
        if inferred:
            dataset_root = inferred
    except Exception:
        pass
    try:
        project = mimics_mask_apply._current_project_path() or ""
        if project:
            mcs_dir = os.path.dirname(project)
    except Exception:
        pass
    return {
        "dataset_root": dataset_root,
        "mcs_dir": mcs_dir,
        "selected_mask_names": [
            str(getattr(mask, "name", "") or "") for mask in selected
        ],
        "mimics_exe": runtime_common.find_mimics_exe() or "",
    }


def _selected_masks():
    rows = []
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    for mask in mimics.data.masks:
        if not bool(getattr(mask, "selected", False)):
            continue
        try:
            bound = getattr(mask, "image", None)
            if active_image is not None and bound is not None and bound != active_image:
                continue
        except Exception:
            pass
        rows.append(mask)
    return rows


def start_training():
    try:
        pid = _launch_gui(
            "flexict_training_setup_ui.py", _training_context(), "train_setup"
        )
        _log(
            logging.INFO,
            "FlexiCT training setup opened outside Mimics. PID: {0}.".format(pid),
        )
        return 0
    except Exception as exc:
        _log(logging.ERROR, "Could not open FlexiCT training setup: {0}".format(exc))
        mimics.dialogs.message_box(
            "Could not open FlexiCT training setup.\n\n{0}".format(
                external_window_launcher.error_guidance(exc)
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1


def show_status():
    try:
        context = {"workspace": _workspace()}
        root = _setup_root()
        context_path = os.path.join(
            root, "status_context_{0}.json".format(uuid.uuid4().hex[:8])
        )
        stderr_path = context_path + ".stderr.log"
        _write_json(context_path, context)
        script = os.path.join(_project_root(), "tools", "flexict_status_viewer.py")
        process = mimics_mask_apply._launch_gui_process(
            [_external_python(), script, "--context", context_path],
            cwd=_project_root(),
            stderr_log=stderr_path,
        )
        _log(
            logging.INFO,
            "FlexiCT status viewer opened outside Mimics. PID: {0}.".format(
                process.pid
            ),
        )
        return 0
    except Exception as exc:
        _log(logging.ERROR, "Could not open FlexiCT status viewer: {0}".format(exc))
        mimics.dialogs.message_box(
            "Could not open FlexiCT status viewer.\n\n{0}".format(
                external_window_launcher.error_guidance(exc)
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1


def _job_status_paths():
    jobs = os.path.join(_workspace(), "jobs")
    rows = []
    if not os.path.isdir(jobs):
        return rows
    try:
        names = os.listdir(jobs)
    except OSError:
        return rows
    for name in names:
        path = os.path.join(jobs, name, "status.json")
        status = _read_json(path, {}) or {}
        if str(status.get("status") or "").lower() in (
            "completed", "failed", "cancelled", "abandoned"
        ):
            continue
        rows.append((float(status.get("created_at_epoch") or 0), path, status))
    rows.sort(reverse=True)
    return rows


def stop_running_task():
    rows = _job_status_paths()
    if not rows:
        mimics.dialogs.message_box(
            "No running FlexiCT task was found.", title=TITLE, ui_blocking=False
        )
        return 0
    _created, status_path, status = rows[0]
    answer = mimics.dialogs.question_box(
        message=(
            "Stop the latest FlexiCT task?\n\n{0}\n{1}\n\n"
            "The worker will release GPU and temporary resources before the task becomes Cancelled."
        ).format(
            status.get("task_name") or status.get("job_id"),
            status.get("message") or status.get("phase"),
        ),
        buttons=BUTTON_STOP + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer != BUTTON_STOP:
        return 0
    command = [
        _external_python(),
        os.path.join(_project_root(), "tools", "nnunet_jobs.py"),
        "stop",
        "--status",
        status_path,
    ]
    mimics_mask_apply._launch_process(command, cwd=_project_root())
    _log(
        logging.INFO,
        "Stop requested for FlexiCT task {0}.".format(status.get("job_id")),
    )
    return 0


def _process_finished_unexpectedly(monitor, status):
    state = str(status.get("status") or "")
    if state not in ("", "opening", "launching", "created"):
        return False
    pid = monitor.get("controller_pid")
    return bool(pid and not runtime_common.process_exists(pid))


def _monitor_tick_locked(monitor):
    key = monitor["monitor_key"]
    if time.time() > float(monitor.get("deadline") or 0):
        _stop_monitor(key)
        _log(
            logging.WARNING,
            "FlexiCT status monitor timed out; the external task was not stopped.",
        )
        return
    status = _read_json(monitor["status_path"], {}) or {}
    if _process_finished_unexpectedly(monitor, status):
        _stop_monitor(key)
        detail = ""
        try:
            detail = open(monitor.get("stderr_path"), "r").read()[-2000:]
        except Exception:
            pass
        mimics.dialogs.message_box(
            "The external FlexiCT window exited before starting a task.\n\n{0}".format(
                detail
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return
    kind = monitor.get("kind")
    state = str(status.get("status") or "")
    line = "{0} | {1}".format(
        state.replace("_", " ").title(),
        status.get("message") or status.get("phase") or "",
    )
    if line != monitor.get("last_line"):
        monitor["last_line"] = line
        _log(logging.INFO, "FlexiCT status: {0}".format(line))
    if state.startswith("waiting_for_"):
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "flexict_resource_wait",
            detail=line,
            interval_seconds=60.0,
            initial_delay_seconds=60.0,
        )
        if due:
            _log(
                logging.INFO,
                "FlexiCT is still waiting ({0}s): {1}. Use 04 Stop Running "
                "Task to cancel and release its resources.".format(
                    int(elapsed), line
                ),
            )
    else:
        runtime_common.clear_progress_notice(monitor, "flexict_resource_wait")
    if kind == "train_setup" and state == "training_started":
        monitor["kind"] = "train"
        monitor["status_path"] = status["training_status_path"]
        monitor["controller_pid"] = None
        monitor["deadline"] = time.time() + 14 * 24 * 60 * 60
        monitor["last_line"] = ""
        return
    if kind == "train_setup":
        if state in ("cancelled", "failed"):
            _stop_monitor(key)
            if state == "failed":
                mimics.dialogs.message_box(
                    "FlexiCT training setup failed.\n\n{0}".format(
                        status.get("error") or "Unknown error"
                    ),
                    title=TITLE,
                    ui_blocking=False,
                )
        return
    if kind == "train":
        if state in ("completed", "failed", "cancelled", "abandoned"):
            _stop_monitor(key)
            if state == "completed":
                message = (
                    "FlexiCT training completed. The model is available "
                    "for prediction."
                )
            elif state == "cancelled":
                message = "FlexiCT training was cancelled."
            else:
                message = "FlexiCT training failed.\n\n{0}".format(
                    status.get("error") or "Unknown error"
                )
            mimics.dialogs.message_box(message, title=TITLE, ui_blocking=False)
        return


def _monitor_tick(monitor):
    if monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        _monitor_tick_locked(monitor)
    except Exception as exc:
        _log(
            logging.ERROR,
            "FlexiCT monitor error: {0}\n{1}".format(exc, traceback.format_exc()),
        )
    finally:
        monitor["busy"] = False


def _stop_monitor(key):
    monitor = _MONITORS.pop(key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    win32 = monitor.get("win32_timer")
    if win32:
        try:
            win32[0].KillTimer(None, win32[1])
        except Exception:
            pass


def _start_win32_monitor(monitor, seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetTimer.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p
        ]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        user32.KillTimer.restype = ctypes.c_int
        callback_type = ctypes.WINFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint
        )

        def callback(hwnd, message, timer_id, tick_count):
            _monitor_tick(monitor)

        callback_ref = callback_type(callback)
        timer_id = user32.SetTimer(
            None,
            0,
            max(250, int(float(seconds) * 1000)),
            ctypes.cast(callback_ref, ctypes.c_void_p),
        )
        if not timer_id:
            return False
        monitor["callback"] = callback_ref
        monitor["win32_timer"] = (user32, timer_id)
        _MONITORS[monitor["monitor_key"]] = monitor
        return True
    except Exception:
        return False


def _start_monitor(monitor, seconds):
    key = monitor["monitor_key"]
    _stop_monitor(key)
    if _start_win32_monitor(monitor, seconds):
        return True
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is None:
            return False
        timer = QTimer()
        timer.timeout.connect(lambda: _monitor_tick(monitor))
        timer.start(max(250, int(float(seconds) * 1000)))
        monitor["timer"] = timer
        _MONITORS[key] = monitor
        return True
    except Exception:
        return False


def main(action=None):
    if action == BUTTON_TRAIN:
        return start_training()
    if action == BUTTON_PREDICT:
        # Phase 5: prediction entry. Placeholder until the prediction UI lands.
        mimics.dialogs.message_box(
            "FlexiCT prediction is not available yet in this build.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    if action == BUTTON_STATUS:
        return show_status()
    if action == BUTTON_STOP:
        return stop_running_task()
    if action == BUTTON_ACTIVE_LEARNING:
        # Phase 6: active-learning review UI. Placeholder until it lands.
        mimics.dialogs.message_box(
            "FlexiCT active learning is not available yet in this build.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    answer = mimics.dialogs.question_box(
        message="Choose a FlexiCT action.",
        buttons=";".join(
            [
                BUTTON_TRAIN,
                BUTTON_PREDICT,
                BUTTON_STATUS,
                BUTTON_STOP,
                BUTTON_ACTIVE_LEARNING,
                BUTTON_CANCEL,
            ]
        ),
        title=TITLE,
        ui_blocking=True,
    )
    return main(answer) if answer != BUTTON_CANCEL else 0
