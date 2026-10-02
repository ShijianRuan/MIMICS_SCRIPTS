# -*- coding: utf-8 -*-
"""Nonblocking Mimics entry points for managed nnU-Net tasks."""

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


TITLE = "nnU-Net"
BUTTON_TRAIN = "Train Model..."
BUTTON_PREDICT = "Predict Current Case..."
BUTTON_STATUS = "Show Status and Models"
BUTTON_UPDATE = "Update Matching Masks"
BUTTON_CREATE = "Create Editable Copies"

# Stable marker carried by the pipeline's validate_materialized_source_geometry
# failure; the shared repair offer lives in fix_source_affine_metadata.
SOURCE_GEOMETRY_MISMATCH_MARKER = (
    fix_source_affine_metadata.SOURCE_GEOMETRY_MISMATCH_MARKER
)

_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        __file__,
        ("nnunet_config.json", "nninteractive_config.json", "runtime_py35"),
    )


def _read_json(path, default=None):
    return runtime_common.read_json(path, default)


def _write_json(path, payload):
    return runtime_common.write_json_atomic(path, payload)


def _settings():
    path = os.path.join(os.path.expanduser("~"), ".mimics_script", "nnunet_settings.json")
    return _read_json(path, {}) or {}


def _workspace():
    return os.path.abspath(
        str(
            _settings().get("workspace")
            or os.path.join(os.path.expanduser("~"), ".mimics_script", "nnunet")
        )
    )


def _external_python():
    config = mimics_mask_apply._config()
    return mimics_mask_apply._integration_python(config, mimics_mask_apply._integration_root(config))


def _log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
        return
    except Exception:
        pass
    print("[nnU-Net] {0}".format(message))


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


def _active_image_masks():
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    rows = []
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
    return mimics_mask_apply._mask_snapshot(mask)


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
            "schema_version": "mimics_nnunet_setup.v1",
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
        raise RuntimeError("External nnU-Net window is missing: {0}".format(script))
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
        "label_root": str(settings.get("label_root") or ""),
        "selected_mask_names": [
            str(getattr(mask, "name", "") or "") for mask in selected
        ],
        "mimics_exe": runtime_common.find_mimics_exe() or "",
    }


def start_training():
    try:
        pid = _launch_gui(
            "nnunet_training_setup_ui.py", _training_context(), "train_setup"
        )
        _log(
            logging.INFO,
            "nnU-Net training setup opened outside Mimics. PID: {0}.".format(pid),
        )
        return 0
    except Exception as exc:
        _log(logging.ERROR, "Could not open nnU-Net training setup: {0}".format(exc))
        mimics.dialogs.message_box(
            "Could not open nnU-Net training setup.\n\n{0}".format(
                external_window_launcher.error_guidance(exc)
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1


def _prediction_context():
    selected = _selected_masks()
    selected_mask = selected[0] if len(selected) == 1 else None
    ts_root, case_id, source_path = mimics_mask_apply._resolve_prediction_context()
    if not case_id or not source_path:
        raise RuntimeError(
            "The active project could not be linked to its original source "
            "image, so prediction was not started.\n\n"
            "Open the project that was created when the case was imported "
            "(01_Data > 01_Import_Data). "
            "If this project was moved or copied away from the dataset, "
            "re-import the case instead."
        )
    target_grid = mimics_mask_apply._active_live_grid_payload()
    source_geometry = mimics_mask_apply._active_source_geometry_payload()
    if not target_grid or not source_geometry:
        raise RuntimeError(
            "The active image physical grid could not be verified. Prediction was not started."
        )
    return {
        "selected_mask_name": str(getattr(selected_mask, "name", "") or "") if selected_mask else "",
        "case_id": case_id,
        "ts_root": ts_root,
        "source_image_path": source_path,
        "source_geometry_expected": source_geometry,
        "target_grid": target_grid,
        "launch_project_path": mimics_mask_apply._current_project_path() or "",
        "prediction_target": _mask_snapshot(selected_mask) if selected_mask else {},
        "matching_masks": [_mask_snapshot(mask) for mask in _active_image_masks()],
    }


def start_prediction():
    try:
        context = _prediction_context()
        return _start_prediction_with_context(context)
    except Exception as exc:
        _log(logging.ERROR, "nnU-Net prediction could not start: {0}".format(exc))
        mimics.dialogs.message_box(
            "nnU-Net prediction could not start.\n\n{0}".format(exc),
            title=TITLE,
            ui_blocking=False,
        )
        return 1


def _start_prediction_with_context(context):
    pid = _launch_gui(
        "nnunet_prediction_setup_ui.py", context, "predict_setup"
    )
    # Preserve the launch-time grids on the in-memory monitor. They must not
    # be reconstructed from mutable project state after inference finishes.
    for row in _MONITORS.values():
        if int(row.get("controller_pid") or 0) == int(pid):
            row["prediction_context"] = context
            break
    _log(
        logging.INFO,
        "nnU-Net model selection opened outside Mimics for case {0}. PID: {1}.".format(
            context["case_id"], pid
        ),
    )
    return 0


def show_status():
    try:
        settings = _settings()
        context = {
            "workspace": _workspace(),
            "task_id": str(settings.get("last_task_id") or ""),
        }
        root = _setup_root()
        context_path = os.path.join(
            root, "status_context_{0}.json".format(uuid.uuid4().hex[:8])
        )
        stderr_path = context_path + ".stderr.log"
        _write_json(context_path, context)
        script = os.path.join(_project_root(), "tools", "nnunet_status_viewer.py")
        process = mimics_mask_apply._launch_gui_process(
            [_external_python(), script, "--context", context_path],
            cwd=_project_root(),
            stderr_log=stderr_path,
        )
        _log(logging.INFO, "nnU-Net status viewer opened outside Mimics. PID: {0}.".format(process.pid))
        return 0
    except Exception as exc:
        _log(logging.ERROR, "Could not open nnU-Net status viewer: {0}".format(exc))
        mimics.dialogs.message_box(
            "Could not open nnU-Net status viewer.\n\n{0}".format(
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


def _resume_pending_inference_monitors():
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
            "ts_root": request.get("ts_root") or "",
            "case_id": request.get("case_id") or "",
            "launch_project_path": request.get("launch_project_path") or "",
            "matching_masks": request.get("matching_masks") or [],
            "prediction_target": request.get("prediction_target") or {},
        }
        _start_monitor(monitor, 1.0)


def _process_finished_unexpectedly(monitor, status):
    state = str(status.get("status") or "")
    if state not in ("", "opening", "launching", "created"):
        return False
    pid = monitor.get("controller_pid")
    return bool(pid and not runtime_common.process_exists(pid))


def _managed_job_process_stopped(status):
    state = str(status.get("status") or "").lower()
    if state in (
        "completed", "failed", "cancelled", "abandoned", "orphaned_remote",
        "remote_unreachable", "attention_required",
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


def _launch_bridge(monitor, status):
    bridge_root = os.path.join(
        monitor["job_dir"], "mimics_apply_" + uuid.uuid4().hex[:8]
    )
    buffers = os.path.join(bridge_root, "buffers")
    if not os.path.isdir(bridge_root):
        os.makedirs(bridge_root)
    axes, flips = mimics_mask_apply._buffer_mapping_from_config(
        mimics_mask_apply._config()
    )
    params = {
        "action": "prepare_masks_for_grid",
        "masks": [{"name": "nnunet_prediction", "mask_path": status["output_path"]}],
        "buffers_out": buffers,
        "target_shape": monitor["target_grid"]["target_shape"],
        "target_voxel_to_ras_matrix": monitor["target_grid"]["target_voxel_to_ras_matrix"],
        "source_voxel_to_ras_matrix": monitor.get("source_geometry_expected", {}).get(
            "source_voxel_to_ras_matrix"
        ),
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


def _matching_update_mask(monitor, label):
    label_name = str(label.get("name") or "Label")
    # A union label's aliases are constituent source structures, not safe
    # destinations for the combined prediction.
    aliases = [label_name]
    if str(label.get("source_mode") or "alternatives") != "union":
        aliases.extend(label.get("aliases") or [])
    wanted = set(value.strip().lower() for value in aliases if str(value).strip())
    candidates = [
        mask
        for mask in _active_image_masks()
        if str(getattr(mask, "name", "") or "").strip().lower() in wanted
    ]
    if len(candidates) > 1:
        _log(
            logging.WARNING,
            "More than one active-image Mask matches label '{0}'; creating a copy instead.".format(
                label_name
            ),
        )
    elif len(candidates) == 1:
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
        else:
            current = int(getattr(mask, "number_of_pixels", 0) or 0)
            if current != int(launch.get("pixel_count") or 0):
                _log(
                    logging.WARNING,
                    "Mask '{0}' changed while prediction was running; creating a copy instead.".format(
                        getattr(mask, "name", label_name)
                    ),
                )
            elif _mask_content_changed(mask, launch):
                _log(
                    logging.WARNING,
                    "Mask '{0}' was edited while prediction was running "
                    "(content changed at the same volume); creating a copy "
                    "instead.".format(
                        getattr(mask, "name", label_name)
                    ),
                )
            else:
                return mask
    return None


def _mask_content_changed(mask, launch):
    """F21: equal-volume edits leave number_of_pixels unchanged.

    Compares the launch-time content digest; when either digest is
    unavailable (buffer unreadable, legacy snapshot) the count check
    stands on its own, same as before.
    """
    expected_sha = str(launch.get("sha256") or "")
    if not expected_sha:
        return False
    current_sha = mimics_mask_apply._mask_content_digest(mask)
    return bool(current_sha) and current_sha != expected_sha


def _mask_for_label(monitor, label, mode):
    label_name = str(label.get("name") or "Label")
    if mode == "update":
        mask = _matching_update_mask(monitor, label)
        if mask is not None:
            return mask
    return mimics_mask_apply._new_prediction_mask("AI_" + label_name)


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
    application.setdefault("schema_version", "mimics_nnunet_application.v1")
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


def _planned_destination(monitor, label, mode):
    label_name = str(label.get("name") or "Label")
    if mode == "update":
        mask = _matching_update_mask(monitor, label)
        if mask is not None:
            return {
                "target_kind": "update",
                "target_name": str(getattr(mask, "name", "") or label_name),
                "target_guid": mimics_mask_apply._mask_identity(mask),
                "fallback_name": _unique_application_mask_name("AI_" + label_name),
            }
    return {
        "target_kind": "create",
        "target_name": _unique_application_mask_name("AI_" + label_name),
        "target_guid": "",
        "fallback_name": "",
    }


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


def _prepare_apply_queue(monitor, bridge_result):
    application = _application_state(monitor)
    mode = str(application.get("mode") or "")
    if mode not in ("update", "create"):
        answer = mimics.dialogs.question_box(
            message=(
                "nnU-Net prediction is complete and ready to apply.\n\n"
                "Update Matching Masks replaces unchanged Masks with matching label names.\n"
                "Create Editable Copies preserves all existing Masks."
            ),
            buttons=BUTTON_UPDATE + ";" + BUTTON_CREATE,
            title="nnU-Net Prediction Ready",
            ui_blocking=True,
        )
        mode = "update" if answer == BUTTON_UPDATE else "create"
        application["mode"] = mode
        application["created_at_epoch"] = time.time()
    application["state"] = "applying"
    records = application.get("records") or {}
    application["records"] = records
    labels = monitor.get("labels") or []
    by_id = dict((int(row.get("id") or 0), row) for row in labels)
    queue = []
    eligible = 0
    already_applied = []
    for row in bridge_result.get("masks") or []:
        if int(row.get("foreground_voxels") or 0) <= 0:
            continue
        eligible += 1
        label_value = int(row.get("label_value") or 0)
        label = by_id.get(label_value)
        if label is None and label_value == 0 and len(labels) == 1:
            label = labels[0]
        if label is None:
            label = {
                "id": label_value,
                "name": "Label {0}".format(label_value or 1),
                "aliases": [],
            }
        application_key = str(int(label.get("id") or label_value or 1))
        record = records.get(application_key) or {}
        if str(record.get("state") or "") == "applied":
            already_applied.append(str(record.get("target_name") or ""))
            continue
        if not record:
            record = {
                "label_id": int(label.get("id") or label_value or 1),
                "label_name": str(label.get("name") or "Label"),
                "state": "pending",
            }
            record.update(_planned_destination(monitor, label, mode))
            records[application_key] = record
        queue.append(
            {
                "buffer": row,
                "label": label,
                "mode": mode,
                "application_key": application_key,
            }
        )
    if eligible == 0:
        raise RuntimeError("The prediction contains no non-background labels.")
    monitor["apply_queue"] = queue
    monitor["applied_masks"] = [name for name in already_applied if name]
    _persist_application_state(monitor, application)


def _apply_one(monitor):
    queue = monitor.get("apply_queue") or []
    if not queue:
        return False
    item = queue[0]
    application = monitor.get("mimics_application") or _application_state(monitor)
    records = application.get("records") or {}
    record = records.get(item.get("application_key")) or {}
    record["state"] = "applying"
    record["attempted_at_epoch"] = time.time()
    records[item.get("application_key")] = record
    application["records"] = records
    _persist_application_state(monitor, application)
    mask = _mask_from_application_record(record)
    if mask is None and str(record.get("target_kind") or "") == "update":
        record["target_kind"] = "create"
        record["target_name"] = str(
            record.get("fallback_name")
            or _unique_application_mask_name(
                "AI_" + str(item["label"].get("name") or "Label")
            )
        )
        record["target_guid"] = ""
        _persist_application_state(monitor, application)
    if mask is None:
        mask = mimics_mask_apply._new_prediction_mask(record["target_name"])
        record["target_name"] = str(getattr(mask, "name", "") or record["target_name"])
        record["target_guid"] = mimics_mask_apply._mask_identity(mask)
        _persist_application_state(monitor, application)
    row = item["buffer"]
    mimics_mask_apply._set_mask_from_u8(
        mask,
        row["output_path"],
        row["mimics_shape"],
        "Apply nnU-Net Prediction",
    )
    mask_name = str(getattr(mask, "name", "") or "")
    record["state"] = "applied"
    record["target_name"] = mask_name
    record["target_guid"] = mimics_mask_apply._mask_identity(mask)
    record["applied_at_epoch"] = time.time()
    _persist_application_state(monitor, application)
    monitor["applied_masks"].append(mask_name)
    queue.pop(0)
    return bool(queue)


def _target_open(monitor):
    return mimics_mask_apply._monitor_target_is_open(monitor)


def _monitor_tick_locked(monitor):
    key = monitor["monitor_key"]
    if time.time() > float(monitor.get("deadline") or 0):
        # A setup form left open past the 1h deadline is still a live window
        # the annotator may return to; a dead window is caught by the
        # process check below. Extend while the setup process is alive so a
        # late submit still gets its completion dialog.
        setup_pid = monitor.get("controller_pid")
        if str(monitor.get("kind") or "").endswith("_setup") and setup_pid and runtime_common.process_exists(setup_pid):
            monitor["deadline"] = time.time() + 3600
        else:
            _stop_monitor(key)
            _log(logging.WARNING, "nnU-Net status monitor timed out; the external task was not stopped.")
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
            "The external nnU-Net window exited before starting a task.\n\n{0}".format(detail),
            title=TITLE,
            ui_blocking=False,
        )
        return
    kind = monitor.get("kind")
    state = str(status.get("status") or "")
    if kind in ("train", "infer") and _managed_job_process_stopped(status):
        remote = str(status.get("execution_backend") or "local") == "remote"
        remote_launch_possible = bool(
            str(status.get("remote_job_dir") or "").strip()
        )
        state = "orphaned_remote" if remote and remote_launch_possible else "failed"
        status.update(
            {
                "status": state,
                "phase": "controller_stopped",
                "message": (
                    "The local remote-task controller stopped. Use Show Status and Models, then Stop, to clean up the remote container."
                    if state == "orphaned_remote"
                    else "The nnU-Net background process stopped before recording completion."
                ),
                "error": "No nnU-Net controller or worker process is running.",
                "updated_at_epoch": time.time(),
            }
        )
        _write_json(monitor["status_path"], status)
    line = "{0} | {1}".format(
        state.replace("_", " ").title(), status.get("message") or status.get("phase") or ""
    )
    if line != monitor.get("last_line"):
        monitor["last_line"] = line
        _log(logging.INFO, "nnU-Net status: {0}".format(line))
    if state.startswith("waiting_for_"):
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "nnunet_resource_wait",
            detail=line,
            interval_seconds=60.0,
            initial_delay_seconds=60.0,
        )
        if due:
            _log(
                logging.INFO,
                "nnU-Net is still waiting ({0}s): {1}. Use Show Status "
                "and Models, then Stop, to cancel and release its "
                "resources.".format(
                    int(elapsed), line
                ),
            )
    else:
        runtime_common.clear_progress_notice(monitor, "nnunet_resource_wait")
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
                    "nnU-Net setup failed.\n\n{0}".format(status.get("error") or "Unknown error"),
                    title=TITLE,
                    ui_blocking=False,
                )
        return
    if kind == "train":
        if state == "orphaned_remote":
            _stop_monitor(key)
            mimics.dialogs.message_box(
                status.get("message"), title=TITLE, ui_blocking=False
            )
            return
        if state in ("completed", "failed", "cancelled", "abandoned"):
            _stop_monitor(key)
            if state == "completed":
                message = "nnU-Net training completed. The model is available for prediction."
            elif state == "cancelled":
                message = "nnU-Net training was cancelled."
            else:
                message = "nnU-Net training failed.\n\n{0}".format(status.get("error") or "Unknown error")
            mimics.dialogs.message_box(message, title=TITLE, ui_blocking=False)
        return
    if kind != "infer":
        return
    if state == "orphaned_remote":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            status.get("message"), title=TITLE, ui_blocking=False
        )
        return
    if state in ("failed", "cancelled"):
        _stop_monitor(key)
        error_text = str(status.get("error") or "")
        if state == "failed" and _offer_source_geometry_repair(error_text):
            return
        mimics.dialogs.message_box(
            "nnU-Net prediction {0}.\n\n{1}".format(state, error_text),
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
            "nnunet_apply_target",
            detail=reason,
            interval_seconds=60.0,
            initial_delay_seconds=0.0,
        )
        if due:
            monitor["waiting_logged"] = True
            _log(
                logging.INFO,
                "nnU-Net result is ready but waiting{0}: {1} Use Show Status "
                "and Models, then Stop, to discard the pending result.".format(
                    " ({0}s)".format(int(elapsed)) if elapsed >= 1.0 else "",
                    reason,
                ),
            )
        return
    runtime_common.clear_progress_notice(monitor, "nnunet_apply_target")
    if not monitor.get("bridge_started"):
        monitor["labels"] = status.get("labels") or []
        _launch_bridge(monitor, status)
        return
    bridge = _read_json(monitor.get("bridge_result_path"), None)
    if bridge is None:
        return
    if bridge.get("status") != "ok":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "nnU-Net result conversion failed.\n\n{0}".format(bridge.get("error") or "Unknown error"),
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
            "Could not apply nnU-Net prediction.\n\n{0}".format(exc),
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
        "nnU-Net result applied to {0} Mask(s): {1}.".format(
            len(applied), ", ".join(applied)
        ),
    )


def _monitor_tick(monitor):
    if monitor.get("busy"):
        return
    token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "nnU-Net result monitor"
    )
    if not token:
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        owner_text = str(owner.get("owner") or "another Mask operation")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "nnunet_buffer_wait",
            detail=owner_text,
            interval_seconds=60.0,
            initial_delay_seconds=5.0,
        )
        if due:
            _log(
                logging.INFO,
                "nnU-Net result handling is waiting for {0} ({1}s). Use Show "
                "Status and Models, then Stop, if the pending result should "
                "be discarded.".format(owner_text, int(elapsed)),
            )
        return
    runtime_common.clear_progress_notice(monitor, "nnunet_buffer_wait")
    monitor["busy"] = True
    try:
        _monitor_tick_locked(monitor)
    except Exception as exc:
        _log(logging.ERROR, "nnU-Net monitor error: {0}\n{1}".format(exc, traceback.format_exc()))
    finally:
        monitor["busy"] = False
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
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
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
    _resume_pending_inference_monitors()
    if action == BUTTON_TRAIN:
        return start_training()
    if action == BUTTON_PREDICT:
        try:
            return _start_prediction_with_context(_prediction_context())
        except Exception as exc:
            _log(logging.ERROR, "nnU-Net prediction could not start: {0}".format(exc))
            mimics.dialogs.message_box(
                "nnU-Net prediction could not start.\n\n{0}".format(exc),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    if action == BUTTON_STATUS:
        return show_status()
    return 0
