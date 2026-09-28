# -*- coding: utf-8 -*-
"""Nonblocking Mimics entry points for FlexiCT few-shot tasks.

Training setup, prediction of the active case (result applied as a Mimics
Mask after grid verification), status viewer, and stop. Active learning
(Phase 6) extends this module.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only.
"""

from __future__ import print_function

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid

import mimics

import external_window_launcher
import fix_source_affine_metadata
import mimics_mask_apply
import runtime_common


def _offer_source_geometry_repair(error_text):
    """One-click stale-affine repair on the failure it actually fixes (D4)."""
    try:
        return fix_source_affine_metadata.offer_repair_for_prediction_failure(
            error_text, TITLE
        )
    except Exception:
        return False


TITLE = "FlexiCT"
BUTTON_TRAIN = "Train Model..."
BUTTON_PREDICT = "Predict Current Case..."
BUTTON_STATUS = "Show Status and Models"
BUTTON_STOP = "Stop Running Task"
BUTTON_ACTIVE_LEARNING = "Active Learning Review"
BUTTON_CANCEL = "Cancel"
BUTTON_UPDATE = "Update Matching Mask"
BUTTON_CREATE = "Create New Mask"

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


def _prediction_context():
    """Capture the launch-time grid contract (verified grid or no start).

    Mirrors nnunet_mimics._prediction_context: the active project must link
    to its original source image, and the live grid + source geometry must
    both be readable. They are frozen into the monitor and never
    reconstructed from mutable project state after inference finishes.
    """
    selected = _selected_masks()
    selected_mask = selected[0] if len(selected) == 1 else None
    ts_root, case_id, source_path = mimics_mask_apply._resolve_prediction_context()
    if not case_id or not source_path:
        raise RuntimeError(
            "The active project could not be linked to its original source "
            "image, so prediction was not started.\n\n"
            "Open the project that was created when the case was imported "
            "(01_Data > 01_Import_Dataset or 02_Import_Single_Case). "
            "If this project was moved or copied away from the dataset, "
            "re-import the case instead."
        )
    target_grid = mimics_mask_apply._active_live_grid_payload()
    source_geometry = mimics_mask_apply._active_source_geometry_payload()
    if not target_grid or not source_geometry:
        raise RuntimeError(
            "The active image physical grid could not be verified. "
            "Prediction was not started."
        )
    return {
        "selected_mask_name": (
            str(getattr(selected_mask, "name", "") or "") if selected_mask else ""
        ),
        "case_id": case_id,
        "ts_root": ts_root,
        "source_image_path": source_path,
        "source_geometry_expected": source_geometry,
        "target_grid": target_grid,
        "launch_project_path": mimics_mask_apply._current_project_path() or "",
        "matching_masks": [
            _mask_snapshot(mask) for mask in _active_image_masks()
        ],
    }


def _active_image_masks():
    rows = []
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    for mask in mimics.data.masks:
        try:
            bound = getattr(mask, "image", None)
            if active_image is not None and bound is not None and bound != active_image:
                continue
        except Exception:
            pass
        rows.append(mask)
    return rows


def _mask_snapshot(mask):
    return {
        "guid": mimics_mask_apply._mask_identity(mask),
        "name": str(getattr(mask, "name", "") or ""),
        "pixel_count": int(getattr(mask, "number_of_pixels", 0) or 0),
    }


def start_prediction():
    try:
        context = _prediction_context()
        return _start_prediction_with_context(context)
    except Exception as exc:
        _log(logging.ERROR, "FlexiCT prediction could not start: {0}".format(exc))
        mimics.dialogs.message_box(
            "FlexiCT prediction could not start.\n\n{0}".format(exc),
            title=TITLE,
            ui_blocking=False,
        )
        return 1


def _start_prediction_with_context(context):
    pid = _launch_gui(
        "flexict_prediction_setup_ui.py", context, "predict_setup"
    )
    # Preserve the launch-time grids on the in-memory monitor. They must not
    # be reconstructed from mutable project state after inference finishes.
    for row in _MONITORS.values():
        if int(row.get("controller_pid") or 0) == int(pid):
            row["prediction_context"] = context
            break
    _log(
        logging.INFO,
        "FlexiCT model selection opened outside Mimics for case {0}. PID: {1}.".format(
            context["case_id"], pid
        ),
    )
    return 0


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
        pending = [
            monitor
            for monitor in _MONITORS.values()
            if str(monitor.get("kind") or "") == "infer"
        ]
        if pending:
            pending.sort(
                key=lambda row: float(row.get("deadline") or 0), reverse=True
            )
            monitor = pending[0]
            answer = mimics.dialogs.question_box(
                message=(
                    "Cancel the pending FlexiCT result conversion/application?\n\n"
                    "The completed prediction file is kept, but it will not be "
                    "applied automatically."
                ),
                buttons=BUTTON_STOP + ";" + BUTTON_CANCEL,
                title=TITLE,
                ui_blocking=True,
            )
            if answer == BUTTON_STOP:
                status = _read_json(monitor.get("status_path"), {}) or {}
                status["application_cancelled"] = True
                status["application_cancelled_at_epoch"] = time.time()
                status["updated_at_epoch"] = time.time()
                try:
                    _write_json(monitor.get("status_path"), status)
                except Exception:
                    pass
                _stop_monitor(monitor["monitor_key"])
                _log(logging.INFO, "Pending FlexiCT result application was cancelled.")
            return 0
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
    kind = monitor.get("kind")
    if kind == "al_requests":
        # Overlay-request poller: no status.json, no deadline semantics beyond
        # the sweep in _monitor_tick.
        _al_tick()
        return
    key = monitor["monitor_key"]
    if time.time() > float(monitor.get("deadline") or 0):
        # Same rationale as nnunet_mimics: a setup form left open past the
        # 1h deadline is still a live window; a dead window is caught by the
        # process check below. Extend while the setup process is alive.
        setup_pid = monitor.get("controller_pid")
        if str(monitor.get("kind") or "").endswith("_setup") and setup_pid and runtime_common.process_exists(setup_pid):
            monitor["deadline"] = time.time() + 3600
        else:
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
                "FlexiCT is still waiting ({0}s): {1}. Open "
                "FlexiCT Show Status and Stop"
                " (02_AI menu) to cancel and release its resources.".format(
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
    if kind == "predict_setup" and state == "prediction_started":
        context = monitor.get("prediction_context") or {}
        monitor.update(context)
        monitor["kind"] = "infer"
        monitor["status_path"] = status["prediction_status_path"]
        monitor["job_dir"] = os.path.dirname(status["prediction_status_path"])
        monitor["controller_pid"] = None
        monitor["deadline"] = time.time() + 24 * 60 * 60
        monitor["last_line"] = ""
        monitor["bridge_started"] = False
        return
    if kind in ("train_setup", "predict_setup"):
        if state in ("cancelled", "failed"):
            _stop_monitor(key)
            if state == "failed":
                mimics.dialogs.message_box(
                    "FlexiCT setup failed.\n\n{0}".format(
                        status.get("error") or "Unknown error"
                    ),
                    title=TITLE,
                    ui_blocking=False,
                )
        return
    if kind in ("train", "infer") and _managed_job_process_stopped(status):
        status.update(
            {
                "status": "failed",
                "phase": "controller_stopped",
                "message": (
                    "The FlexiCT background process stopped before recording "
                    "completion."
                ),
                "error": "No FlexiCT controller or worker process is running.",
                "updated_at_epoch": time.time(),
            }
        )
        _write_json(monitor["status_path"], status)
        state = "failed"
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
    if kind != "infer":
        return
    if bool(monitor.get("application_cancelled")):
        _stop_monitor(key)
        return
    if state in ("failed", "cancelled", "abandoned"):
        _stop_monitor(key)
        error_text = str(status.get("error") or "")
        if state == "failed" and _offer_source_geometry_repair(error_text):
            return
        if state == "abandoned":
            # Same terminal wording as the training branch: the remote job
            # was abandoned, and the deadline message would mislead.
            message = "FlexiCT prediction failed.\n\n{0}".format(
                error_text or "The remote task was abandoned."
            )
        else:
            message = "FlexiCT prediction {0}.\n\n{1}".format(state, error_text)
        mimics.dialogs.message_box(
            message,
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state != "completed":
        return
    target_open, reason = _target_open(monitor)
    if not target_open:
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "flexict_apply_target",
            detail=reason,
            interval_seconds=60.0,
            initial_delay_seconds=0.0,
        )
        if due:
            monitor["waiting_logged"] = True
            _log(
                logging.INFO,
                "FlexiCT result is ready but waiting{0}: {1} The result is "
                "kept and will be applied as soon as the target project is "
                "reopened.".format(
                    " ({0}s)".format(int(elapsed)) if elapsed >= 1.0 else "",
                    reason,
                ),
            )
        return
    runtime_common.clear_progress_notice(monitor, "flexict_apply_target")
    if not monitor.get("bridge_started"):
        _launch_bridge(monitor, status)
        return
    bridge = _read_json(monitor.get("bridge_result_path"), None)
    if bridge is None:
        return
    if bridge.get("status") != "ok":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "FlexiCT result conversion failed.\n\n{0}".format(
                bridge.get("error") or "Unknown error"
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return
    try:
        if "apply_queue" not in monitor:
            _prepare_apply_queue(monitor, bridge)
        more = _apply_one(monitor)
        if more:
            return
    except Exception as exc:
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "Could not apply the FlexiCT prediction.\n\n{0}".format(exc),
            title=TITLE,
            ui_blocking=False,
        )
        return
    applied = list(monitor.get("applied_masks") or [])
    try:
        shutil.rmtree(monitor.get("bridge_root"), ignore_errors=True)
    except Exception:
        pass
    latest = _read_json(monitor.get("status_path"), {}) or {}
    latest["applied_to_mimics"] = True
    latest["applied_mask_names"] = applied
    latest["applied_at_epoch"] = time.time()
    application = latest.get("mimics_application") or {}
    application["state"] = "completed"
    application["completed_at_epoch"] = time.time()
    latest["mimics_application"] = application
    latest["application_phase"] = "completed"
    latest["updated_at_epoch"] = time.time()
    try:
        _write_json(monitor.get("status_path"), latest)
    except Exception:
        pass
    _stop_monitor(key)
    _log(
        logging.INFO,
        "FlexiCT result applied to {0} Mask(s): {1}.".format(
            len(applied), ", ".join(applied)
        ),
    )


def _managed_job_process_stopped(status):
    state = str(status.get("status") or "").lower()
    if state in (
        "completed", "failed", "cancelled", "abandoned"
    ):
        return False
    try:
        age = time.time() - float(
            status.get("updated_at_epoch") or status.get("created_at_epoch") or 0
        )
    except Exception:
        age = 0.0
    if age < 10.0:
        return False
    pids = []
    for name in ("worker_pid", "controller_pid", "launcher_pid"):
        try:
            pid = int(status.get(name) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return bool(pids and not any(runtime_common.process_exists(pid) for pid in pids))


def _target_open(monitor):
    return mimics_mask_apply._monitor_target_is_open(monitor)


def _launch_bridge(monitor, status):
    """Convert the prediction NIfTI to a grid-matched .u8 buffer.

    Zero-change reuse of mimics_bridge.py's prepare_masks_for_grid action:
    resample onto the frozen launch-time target grid, then map to the Mimics
    voxel buffer with the configured axes/flips.
    """
    bridge_root = os.path.join(
        monitor["job_dir"], "mimics_apply_" + uuid.uuid4().hex[:8]
    )
    buffers = os.path.join(bridge_root, "buffers")
    if not os.path.isdir(bridge_root):
        os.makedirs(bridge_root)
    axes, flips = mimics_mask_apply._buffer_mapping_from_config(
        mimics_mask_apply._config()
    )
    mask_name = "FlexiCT {0}".format(status.get("label_name") or "Prediction")
    params = {
        "action": "prepare_masks_for_grid",
        "masks": [{"name": "flexict_prediction", "mask_path": status["output_path"]}],
        "buffers_out": buffers,
        "target_shape": monitor["target_grid"]["target_shape"],
        "target_voxel_to_ras_matrix": monitor["target_grid"]["target_voxel_to_ras_matrix"],
        "source_voxel_to_ras_matrix": monitor.get(
            "source_geometry_expected", {}
        ).get("source_voxel_to_ras_matrix"),
        "axes": axes,
        "flips": flips,
    }
    input_path = os.path.join(bridge_root, "input.json")
    result_path = os.path.join(bridge_root, "result.json")
    _write_json(input_path, params)
    with open(input_path, "rb") as stdin_handle:
        process = subprocess.Popen(
            [_external_python(), os.path.join(_project_root(), "mimics_bridge.py")],
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **runtime_common.background_process_kwargs()
        )
    monitor["bridge_started"] = True
    monitor["bridge_root"] = bridge_root
    monitor["bridge_result_path"] = result_path
    monitor["bridge_pid"] = process.pid
    monitor["bridge_process"] = process
    monitor["bridge_mask_name"] = mask_name

    def wait_bridge():
        try:
            stdout, stderr = process.communicate(timeout=600)
            if process.returncode == 0:
                result = json.loads(stdout.decode("utf-8"))
            else:
                result = {
                    "status": "error",
                    "error": stderr.decode("utf-8", "replace")[-2000:],
                }
        except subprocess.TimeoutExpired:
            try:
                runtime_common.terminate_process_async(
                    process=process, graceful_seconds=0.0
                )
            except Exception:
                pass
            result = {
                "status": "error",
                "error": "Prediction conversion timed out after 600 seconds.",
            }
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        try:
            _write_json(result_path, result)
        except Exception:
            pass

    thread = threading.Thread(target=wait_bridge)
    thread.daemon = True
    thread.start()


def _matching_update_mask(monitor):
    """Find the single unchanged launch-time Mask matching the label name.

    FlexiCT predicts one binary target; the update destination is the Mask
    whose name matches the model's label_name (or the selected Mask at
    launch). Changed or ambiguous Masks fall back to a fresh copy.
    """
    label_name = str(monitor.get("label_name") or "")
    selected = str(monitor.get("selected_mask_name") or "")
    wanted = set(
        value.strip().lower() for value in (label_name, selected)
        if str(value).strip()
    )
    if not wanted:
        return None
    candidates = [
        mask
        for mask in _active_image_masks()
        if str(getattr(mask, "name", "") or "").strip().lower() in wanted
    ]
    if len(candidates) != 1:
        if len(candidates) > 1:
            _log(
                logging.WARNING,
                "More than one active-image Mask matches '{0}'; creating a copy instead.".format(
                    label_name or selected
                ),
            )
        return None
    mask = candidates[0]
    launch = None
    for row in monitor.get("matching_masks") or []:
        if str(row.get("guid") or "") == mimics_mask_apply._mask_identity(mask):
            launch = row
            break
    if launch is None:
        _log(
            logging.WARNING,
            "Mask '{0}' was not present when prediction started; creating a copy instead.".format(
                getattr(mask, "name", label_name)
            ),
        )
        return None
    current = int(getattr(mask, "number_of_pixels", 0) or 0)
    if current != int(launch.get("pixel_count") or 0):
        _log(
            logging.WARNING,
            "Mask '{0}' changed while prediction was running; creating a copy instead.".format(
                getattr(mask, "name", label_name)
            ),
        )
        return None
    return mask


def _unique_application_mask_name(base_name):
    names = set(
        str(getattr(mask, "name", "") or "") for mask in mimics.data.masks
    )
    if base_name not in names:
        return base_name
    index = 2
    while "{0} {1}".format(base_name, index) in names:
        index += 1
    return "{0} {1}".format(base_name, index)


def _application_state(monitor):
    status_path = monitor.get("status_path")
    latest = _read_json(status_path, {}) if status_path else {}
    application = (latest or {}).get("mimics_application") or {}
    if not isinstance(application, dict):
        application = {}
    application.setdefault("schema_version", "mimics_flexict_application.v1")
    application.setdefault("records", {})
    return application


def _persist_application_state(monitor, application):
    monitor["mimics_application"] = application
    status_path = monitor.get("status_path")
    if not status_path:
        return
    latest = _read_json(status_path, {}) or {}
    latest["mimics_application"] = application
    latest["application_phase"] = str(application.get("state") or "applying")
    latest["updated_at_epoch"] = time.time()
    _write_json(status_path, latest)


def _prepare_apply_queue(monitor, bridge_result):
    """One buffer, one apply step; ask update-vs-create only once."""
    application = _application_state(monitor)
    mode = str(application.get("mode") or "")
    if mode not in ("update", "create"):
        answer = mimics.dialogs.question_box(
            message=(
                "FlexiCT prediction is complete and ready to apply.\n\n"
                "Update Matching Mask replaces the unchanged Mask matching this "
                "target; Create New Mask keeps all existing Masks."
            ),
            buttons=BUTTON_UPDATE + ";" + BUTTON_CREATE,
            title="FlexiCT Prediction Ready",
            ui_blocking=True,
        )
        mode = "update" if answer == BUTTON_UPDATE else "create"
        application["mode"] = mode
        application["created_at_epoch"] = time.time()
    application["state"] = "applying"
    records = application.get("records") or {}
    application["records"] = records
    label_name = str(monitor.get("label_name") or "Prediction")
    record = records.get("1") or {}
    already_applied = []
    buffer_row = None
    for row in bridge_result.get("masks") or []:
        if int(row.get("foreground_voxels") or 0) > 0:
            buffer_row = row
            break
    if buffer_row is None:
        raise RuntimeError("The prediction is empty (no foreground voxels).")
    if str(record.get("state") or "") == "applied":
        already_applied.append(str(record.get("target_name") or ""))
    else:
        if not record:
            record = {
                "label_name": label_name,
                "state": "pending",
            }
            mask = (
                _matching_update_mask(monitor)
                if mode == "update" else None
            )
            if mask is not None:
                record["target_kind"] = "update"
                record["target_name"] = str(getattr(mask, "name", "") or label_name)
                record["target_guid"] = mimics_mask_apply._mask_identity(mask)
                record["fallback_name"] = _unique_application_mask_name(
                    "FlexiCT " + label_name
                )
            else:
                record["target_kind"] = "create"
                record["target_name"] = _unique_application_mask_name(
                    "FlexiCT " + label_name
                )
                record["target_guid"] = ""
                record["fallback_name"] = ""
            records["1"] = record
        monitor["apply_queue"] = [{"buffer": buffer_row, "mode": mode}]
    monitor["applied_masks"] = [name for name in already_applied if name]
    _persist_application_state(monitor, application)


def _mask_from_application_record(record):
    target_kind = str(record.get("target_kind") or "create")
    target_guid = str(record.get("target_guid") or "")
    target_name = str(record.get("target_name") or "")
    for mask in _active_image_masks():
        if target_kind == "update" and target_guid:
            if mimics_mask_apply._mask_identity(mask) == target_guid:
                return mask
        elif target_kind == "create" and str(
            getattr(mask, "name", "") or ""
        ) == target_name:
            return mask
    return None


def _apply_one(monitor):
    queue = monitor.get("apply_queue") or []
    if not queue:
        return False
    item = queue[0]
    application = monitor.get("mimics_application") or _application_state(monitor)
    records = application.get("records") or {}
    record = records.get("1") or {}
    record["state"] = "applying"
    record["attempted_at_epoch"] = time.time()
    records["1"] = record
    application["records"] = records
    _persist_application_state(monitor, application)
    mask = _mask_from_application_record(record)
    if mask is None and str(record.get("target_kind") or "") == "update":
        record["target_kind"] = "create"
        record["target_name"] = str(
            record.get("fallback_name")
            or _unique_application_mask_name(
                "FlexiCT " + str(record.get("label_name") or "Prediction")
            )
        )
        record["target_guid"] = ""
        _persist_application_state(monitor, application)
    if mask is None:
        mask = mimics_mask_apply._new_prediction_mask(record["target_name"])
        record["target_name"] = str(
            getattr(mask, "name", "") or record["target_name"]
        )
        record["target_guid"] = mimics_mask_apply._mask_identity(mask)
        _persist_application_state(monitor, application)
    row = item["buffer"]
    mimics_mask_apply._set_mask_from_u8(
        mask,
        row["output_path"],
        row["mimics_shape"],
        "Apply FlexiCT Prediction",
    )
    mask_name = str(getattr(mask, "name", "") or "")
    record["state"] = "applied"
    record["target_name"] = mask_name
    record["target_guid"] = mimics_mask_apply._mask_identity(mask)
    record["applied_at_epoch"] = time.time()
    _persist_application_state(monitor, application)
    monitor["applied_masks"].append(mask_name)
    queue.pop(0)
    return False  # single label: one buffer, done


def _monitor_tick(monitor):
    if monitor.get("busy"):
        return
    token = None
    if str(monitor.get("kind") or "") == "infer":
        token = runtime_common.try_acquire_local_operation(
            "mask_buffer_access", "FlexiCT result monitor"
        )
        if not token:
            owner = runtime_common.active_local_operation("mask_buffer_access") or {}
            owner_text = str(owner.get("owner") or "another Mask operation")
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "flexict_buffer_wait",
                detail=owner_text,
                interval_seconds=60.0,
                initial_delay_seconds=5.0,
            )
            if due:
                _log(
                    logging.INFO,
                    "FlexiCT result handling is waiting for {0} ({1}s). The "
                    "result is kept and will be applied when the other Mask "
                    "operation finishes.".format(owner_text, int(elapsed)),
                )
            return
        runtime_common.clear_progress_notice(monitor, "flexict_buffer_wait")
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
        if token is not None:
            runtime_common.release_local_operation("mask_buffer_access", token)


def _stop_monitor(key):
    monitor = _MONITORS.pop(key, None)
    if not monitor:
        return
    bridge_process = monitor.get("bridge_process")
    bridge_pid = monitor.get("bridge_pid")
    if bridge_process is not None and bridge_process.poll() is None:
        try:
            runtime_common.terminate_process_async(
                process=bridge_process,
                pid=bridge_pid,
                graceful_seconds=0.5,
            )
        except Exception:
            pass
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


def _resume_pending_inference_monitors():
    """Restart monitors for completed-but-unapplied inference jobs.

    If Mimics was closed while a prediction was waiting to be applied (for
    example the target project was closed), this revives the apply monitor
    on the next entry so the result is offered instead of silently lost.
    """
    jobs = os.path.join(_workspace(), "jobs")
    if not os.path.isdir(jobs):
        return
    try:
        names = os.listdir(jobs)
    except OSError:
        return
    for name in names:
        status_path = os.path.join(jobs, name, "status.json")
        status = _read_json(status_path, {}) or {}
        if str(status.get("kind") or "") != "infer":
            continue
        if bool(status.get("applied_to_mimics")):
            continue
        if bool(status.get("application_cancelled")):
            continue
        state = str(status.get("status") or "")
        if state in ("failed", "cancelled"):
            continue
        key = str(status.get("job_id") or name)
        if key in _MONITORS:
            continue
        request = _read_json(os.path.join(jobs, name, "request.json"), {}) or {}
        target_grid = request.get("target_grid") or {}
        source_geometry = request.get("source_geometry_expected") or {}
        if not target_grid or not source_geometry:
            continue
        monitor = {
            "monitor_key": key,
            "kind": "infer",
            "status_path": status_path,
            "job_dir": os.path.join(jobs, name),
            "deadline": time.time() + 24 * 60 * 60,
            "last_line": "",
            "bridge_started": False,
            "target_grid": target_grid,
            "source_geometry_expected": source_geometry,
            "label_name": request.get("label_name") or status.get("label_name") or "",
            "selected_mask_name": request.get("selected_mask_name") or "",
            "ts_root": request.get("ts_root") or "",
            "case_id": request.get("case_id") or "",
            "launch_project_path": request.get("launch_project_path") or "",
            "matching_masks": request.get("matching_masks") or [],
        }
        _start_monitor(monitor, 1.0)


def _start_active_learning():
    """Open the external review UI and start the overlay-request monitor."""
    context = {"workspace": _workspace()}
    _launch_gui("flexict_active_learning_ui.py", context, "al_setup")
    _start_al_request_monitor()
    return 0


_AL_MONITOR_KEY = "flexict_al_requests"
_AL_APPLY_TRANSACTIONS = {}


def _al_request_dir(job_dir):
    return os.path.join(job_dir, "apply_requests")


def _al_pending_requests():
    """Newest-first overlay requests across every active-learning job."""
    jobs = os.path.join(_workspace(), "jobs")
    requests = []
    if not os.path.isdir(jobs):
        return requests
    try:
        names = os.listdir(jobs)
    except OSError:
        return requests
    for name in names:
        status = _read_json(os.path.join(jobs, name, "status.json"), {}) or {}
        if str(status.get("kind") or "") != "active_learning":
            continue
        if str(status.get("status") or "") != "completed":
            continue
        request_dir = _al_request_dir(os.path.join(jobs, name))
        if not os.path.isdir(request_dir):
            continue
        try:
            files = os.listdir(request_dir)
        except OSError:
            continue
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(request_dir, file_name)
            payload = _read_json(path, {}) or {}
            if not payload:
                continue
            if str(payload.get("state") or "pending") != "pending":
                continue
            payload["_request_path"] = path
            payload["_job_dir"] = os.path.join(jobs, name)
            requests.append(payload)
    requests.sort(
        key=lambda row: float(row.get("requested_at_epoch") or 0),
        reverse=True,
    )
    return requests


def _al_tick():
    """Poll for review-UI overlay requests and apply the newest one."""
    if _AL_APPLY_TRANSACTIONS.get("busy"):
        return
    _AL_APPLY_TRANSACTIONS["busy"] = True
    try:
        requests = _al_pending_requests()
        if not requests:
            return
        request = requests[0]
        _al_apply_request(request)
    except Exception as exc:
        _log(
            logging.ERROR,
            "FlexiCT active-learning overlay error: {0}\n{1}".format(
                exc, traceback.format_exc()
            ),
        )
    finally:
        _AL_APPLY_TRANSACTIONS["busy"] = False


def _al_mark_request(request_path, state, detail=""):
    payload = _read_json(request_path, {}) or {}
    payload["state"] = state
    payload["detail"] = str(detail)
    payload["updated_at_epoch"] = time.time()
    _write_json(request_path, payload)


def _al_case_source_geometry(job_dir, case_id):
    geometries = _read_json(
        os.path.join(job_dir, "input_geometries.json"), {}
    ) or {}
    cases = geometries.get("cases") or {}
    return cases.get(case_id) or {}


def _al_mask_paths(job_dir, case_id, what):
    """The NIfTI(s) an overlay request maps to."""
    uncertainty_dir = os.path.join(job_dir, "uncertainty")
    if what == "consensus":
        path = os.path.join(uncertainty_dir, "{0}_consensus.nii.gz".format(case_id))
        if os.path.isfile(path):
            return [("FlexiCT Consensus", path)]
        return []
    bands_dir = os.path.join(uncertainty_dir, "bands")
    rows = []
    for level, title in (("high", "FlexiCT Uncertainty (High)"),
                         ("moderate", "FlexiCT Uncertainty (Moderate)")):
        path = os.path.join(bands_dir, "{0}_{1}.nii.gz".format(case_id, level))
        if os.path.isfile(path):
            rows.append((title, path))
    return rows


def _al_open_case(job_dir, case_id):
    """Open the case's .mcs project in this Mimics session.

    Returns (ok, detail). The path comes from input_geometries.json, where
    the AL materializer recorded it when the pool was built. Never stomps
    on an open unsaved session: if another project is active, the user is
    told to save and close it first (same protection import_undo_mimics
    applies). Only projects created through the 01_Import flow exist for
    a dataset, so a missing .mcs is a guidance case, not an error state.
    """
    geometry = _al_case_source_geometry(job_dir, case_id)
    mcs_path = str(geometry.get("mcs_path") or "")
    source_path = str(geometry.get("source_image_path") or "")
    if not mcs_path or not os.path.isfile(mcs_path):
        return False, (
            "No .mcs project was found for case {0}. Generate one first "
            "(01 Import, case folder {1}), then retry.".format(
                case_id, source_path or "(source path not recorded)")
        )
    # If the requested project is already active, opening it again would
    # close/reload it needlessly — skip straight to success.
    try:
        active = mimics.file.get_active_project()
        if active and os.path.abspath(str(active)) == os.path.abspath(mcs_path):
            return True, "project already open"
    except Exception:
        pass
    try:
        active = mimics.file.get_active_project()
        if active:
            mimics.dialogs.message_box(
                "Another project is open in Mimics.\n\n"
                "Opening case {0} requires its own project "
                "({1}).\n\nStep 1: close the currently open project (save "
                "it if needed).\nStep 2: open the case from the FlexiCT "
                "Active Learning window again.".format(case_id, mcs_path),
                title=TITLE,
                ui_blocking=False,
            )
            return False, "another project is open"
    except Exception:
        pass
    try:
        mimics.file.open_project(filename=mcs_path)
    except Exception as exc:
        return False, "could not open {0}: {1}".format(mcs_path, exc)
    return True, "opened {0}".format(mcs_path)


def _al_mark_annotated(job_dir, case_id):
    """Mark the case annotated in the review window's state file.

    Same schema the external UI writes (flexict_annotation_state.v1), so
    both writers converge on the same value regardless of who writes
    first. Never downgrades an explicit annotated/skipped entry."""
    path = os.path.join(job_dir, "annotation_state.json")
    payload = _read_json(path, {}) or {}
    if str(payload.get("schema_version") or "") != "flexict_annotation_state.v1":
        payload = {"schema_version": "flexict_annotation_state.v1", "cases": {}}
    cases = payload.setdefault("cases", {})
    entry = cases.get(case_id) or {}
    if str(entry.get("state") or "new") != "new":
        return
    cases[case_id] = {
        "state": "annotated",
        "updated_at_epoch": time.time(),
    }
    payload["updated_at_epoch"] = time.time()
    _write_json(path, payload)


def _al_apply_request(request):
    """Verify the active image matches the request's case, then apply the
    overlay masks through the same bridge path the prediction flow uses.

    `what` values: "bands"/"consensus" apply an overlay to the already
    active case; "open" opens the case's .mcs project; "open_bands" opens
    the project and then applies the uncertainty bands in one action."""
    request_path = request["_request_path"]
    job_dir = request["_job_dir"]
    case_id = str(request.get("case_id") or "")
    what = str(request.get("what") or "bands")
    if what == "open":
        opened, detail = _al_open_case(job_dir, case_id)
        _al_mark_request(
            request_path, "applied" if opened else "failed", detail)
        if opened:
            _log(
                logging.INFO,
                "FlexiCT active-learning opened case {0}: {1}.".format(
                    case_id, detail),
            )
        return
    if what == "open_bands":
        opened, detail = _al_open_case(job_dir, case_id)
        if not opened:
            _al_mark_request(request_path, "failed", detail)
            return
        # The just-opened project's active image is this case in the
        # normal flow; the verified-grid check below confirms it and
        # fails with guidance if the project's image is not the case.
        what = "bands"
    masks = _al_mask_paths(job_dir, case_id, what)
    if not masks:
        _al_mark_request(request_path, "failed", "overlay files not found")
        return
    # Verified-grid contract: the active image must still be this case on
    # its original source grid (the one the pool was materialized from).
    source_geometry = mimics_mask_apply._active_source_geometry_payload()
    if not source_geometry:
        _al_mark_request(
            request_path, "failed",
            "The active image source geometry could not be verified.",
        )
        return
    expected = _al_case_source_geometry(job_dir, case_id)
    if not expected:
        _al_mark_request(
            request_path, "failed",
            "No recorded source geometry for case {0}.".format(case_id),
        )
        return
    if list(source_geometry.get("source_shape") or []) != list(
            expected.get("source_shape") or []) or not _matrix_close_payload(
                source_geometry.get("source_voxel_to_ras_matrix"),
                expected.get("source_voxel_to_ras_matrix"),
            ):
        _al_mark_request(
            request_path, "failed",
            "The active image does not match case {0}. Open that case "
            "(its original source image) and retry.".format(case_id),
        )
        return
    target_grid = mimics_mask_apply._active_live_grid_payload()
    if not target_grid:
        _al_mark_request(
            request_path, "failed",
            "The active Mimics image grid could not be measured.",
        )
        return
    bridge_root = os.path.join(
        job_dir, "al_apply_" + uuid.uuid4().hex[:8]
    )
    buffers = os.path.join(bridge_root, "buffers")
    os.makedirs(buffers)
    axes, flips = mimics_mask_apply._buffer_mapping_from_config(
        mimics_mask_apply._config()
    )
    params = {
        "action": "prepare_masks_for_grid",
        "masks": [
            {"name": "al_{0}".format(index), "mask_path": path}
            for index, (_title, path) in enumerate(masks)
        ],
        "buffers_out": buffers,
        "target_shape": target_grid["target_shape"],
        "target_voxel_to_ras_matrix": target_grid[
            "target_voxel_to_ras_matrix"],
        "source_voxel_to_ras_matrix": expected.get(
            "source_voxel_to_ras_matrix"),
        "axes": axes,
        "flips": flips,
    }
    input_path = os.path.join(bridge_root, "input.json")
    result_path = os.path.join(bridge_root, "result.json")
    _write_json(input_path, params)
    with open(input_path, "rb") as stdin_handle:
        process = subprocess.Popen(
            [_external_python(), os.path.join(_project_root(), "mimics_bridge.py")],
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **runtime_common.background_process_kwargs()
        )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except Exception:
        try:
            runtime_common.terminate_process_async(
                process=process, graceful_seconds=0.0
            )
        except Exception:
            pass
        _al_mark_request(request_path, "failed", "conversion timed out")
        shutil.rmtree(bridge_root, ignore_errors=True)
        return
    shutil.rmtree(bridge_root, ignore_errors=True)
    if process.returncode != 0:
        _al_mark_request(
            request_path, "failed",
            stderr.decode("utf-8", "replace")[-2000:],
        )
        return
    result = json.loads(stdout.decode("utf-8"))
    applied = []
    for index, (title, _path) in enumerate(masks):
        row = None
        for candidate in result.get("masks") or []:
            if str(candidate.get("name") or "") == "al_{0}".format(index):
                row = candidate
                break
        if row is None:
            continue
        mask = mimics_mask_apply._new_prediction_mask(title)
        mimics_mask_apply._set_mask_from_u8(
            mask,
            row["output_path"],
            row["mimics_shape"],
            "Apply FlexiCT Active-Learning Overlay",
        )
        applied.append(str(getattr(mask, "name", "") or title))
    if not applied:
        _al_mark_request(request_path, "failed", "no foreground voxels")
        return
    _al_mark_request(
        request_path, "applied", "applied: {0}".format(", ".join(applied)))
    # The overlay is on the case's image now, so the annotator can start
    # (or has started) work on it — reflect that in the review state so
    # the external window's Status column updates without a manual click.
    _al_mark_annotated(job_dir, case_id)
    _log(
        logging.INFO,
        "FlexiCT active-learning overlay applied for {0}: {1}.".format(
            case_id, ", ".join(applied)),
    )
    try:
        mimics.dialogs.message_box(
            "Applied {0} for case {1}.".format(
                ", ".join(applied), case_id),
            title=TITLE,
            ui_blocking=False,
        )
    except TypeError:
        mimics.dialogs.message_box(
            title=TITLE,
            message="Applied {0} for case {1}.".format(
                ", ".join(applied), case_id),
        )


def _matrix_close_payload(a, b, tolerance=1e-4):
    if not a or not b:
        return False
    try:
        for row_index in range(4):
            for column_index in range(4):
                if abs(
                    float(a[row_index][column_index])
                    - float(b[row_index][column_index])
                ) > tolerance:
                    return False
    except Exception:
        return False
    return True


def _start_al_request_monitor():
    if _AL_MONITOR_KEY in _MONITORS:
        return
    monitor = {
        "monitor_key": _AL_MONITOR_KEY,
        "kind": "al_requests",
        "deadline": time.time() + 30 * 24 * 60 * 60,
        "last_line": "",
    }
    if _start_monitor(monitor, 2.0):
        _MONITORS[_AL_MONITOR_KEY] = monitor


def main(action=None):
    _resume_pending_inference_monitors()
    _start_al_request_monitor()
    if action == BUTTON_TRAIN:
        return start_training()
    if action == BUTTON_PREDICT:
        try:
            return _start_prediction_with_context(_prediction_context())
        except Exception as exc:
            _log(logging.ERROR, "FlexiCT prediction could not start: {0}".format(exc))
            mimics.dialogs.message_box(
                "FlexiCT prediction could not start.\n\n{0}".format(exc),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    if action == BUTTON_STATUS:
        return show_status()
    if action == BUTTON_STOP:
        return stop_running_task()
    if action == BUTTON_ACTIVE_LEARNING:
        return _start_active_learning()
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
