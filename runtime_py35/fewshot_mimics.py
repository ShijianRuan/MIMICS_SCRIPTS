# -*- coding: utf-8 -*-
"""Mimics-side entry for DINOv3 few-shot training and inference.

The foreground Mimics process only launches jobs, polls status files, and
applies completed prediction buffers. Training and inference run in external
Python processes.
"""

from __future__ import print_function

import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid

import mimics

import runtime_common


TITLE = "DINOv3 Few-Shot"
BUTTON_TRAIN = "Train/Update Model"
BUTTON_PREDICT = "Predict Current Case"
BUTTON_STATUS = "Show Status"
BUTTON_STOP = "Stop Latest Job"
BUTTON_CANCEL = "Cancel"

_MONITORS = {}


_write_json_atomic = runtime_common.write_json_atomic
_read_json = runtime_common.read_json
_safe_slug = runtime_common.safe_slug
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_env = runtime_common.background_env


def _settings_path():
    return os.path.join(_project_root(), ".fewshot_mimics_state.json")


def _load_settings():
    return _read_json(_settings_path(), {}) or {}


def _save_settings(value):
    try:
        _write_json_atomic(_settings_path(), value)
    except Exception:
        pass


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("fewshot_config.json", "nninteractive_config.json", ".git"),
    )


def _config():
    path = os.path.join(_project_root(), "fewshot_config.json")
    cfg = _read_json(path, {}) or {}
    return cfg


def _resolve_path(value, base):
    if not value:
        return None
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(base, value))


def _dinov3_root(config):
    env = os.environ.get("MIMICS_FEWSHOT_DINOV3_ROOT", "")
    value = env or config.get("dinov3_project") or "external/dinov3-medical-seg"
    return _resolve_path(value, _project_root())


def _fewshot_python(config, dinov3_root):
    candidates = []
    for value in (
        os.environ.get("MIMICS_FEWSHOT_PYTHON", ""),
        config.get("python", ""),
    ):
        if value:
            candidates.append(value)
    candidates.extend([
        os.path.join(dinov3_root, ".venv", "Scripts", "python.exe"),
        os.path.join(dinov3_root, ".venv", "bin", "python"),
        os.path.join(dinov3_root, "venv", "Scripts", "python.exe"),
        os.path.join(dinov3_root, "venv", "bin", "python"),
        "python",
        "python3",
    ])
    if sys.version_info[:2] >= (3, 10):
        candidates.append(sys.executable)
    for candidate in candidates:
        if candidate in ("python", "python3"):
            return candidate
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return candidates[-1]


def _pipeline_script():
    return os.path.join(_project_root(), "tools", "fewshot_pipeline.py")


def _bridge_script():
    return os.path.join(_project_root(), "mimics_bridge.py")


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _pick_directory(title, initial_dir=""):
    try:
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(None, title, initial_dir or "")
        return str(path) if path else None
    except Exception:
        pass
    try:
        import Tkinter as tk
        import tkFileDialog
    except ImportError:
        try:
            import tkinter as tk
            from tkinter import filedialog as tkFileDialog
        except ImportError:
            return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = tkFileDialog.askdirectory(parent=root, title=title)
    root.destroy()
    return path if path else None


def _selected_mask():
    selected = []
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    for mask in mimics.data.masks:
        if not bool(getattr(mask, "selected", False)):
            continue
        if active_image is not None:
            try:
                if getattr(mask, "image", None) is not None and getattr(mask, "image", None) != active_image:
                    continue
            except Exception:
                pass
        selected.append(mask)
    if len(selected) == 1:
        return selected[0]
    if len(selected) > 1:
        raise RuntimeError("Select exactly one Mask that names the organ.")
    return None


def _selected_organ():
    mask = _selected_mask()
    if mask is None:
        return None
    return str(getattr(mask, "name", "") or "").strip()


def _find_mimics_exe():
    env = os.environ.get("MIMICS_EXE", "")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend([
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Materialise", "Mimics Research 21.0", "MimicsResearch.exe"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Mimics Research 21.0", "MimicsResearch.exe"),
        "D:\\Mimics Research 21.0\\MimicsResearch.exe",
        "C:\\Mimics Research 21.0\\MimicsResearch.exe",
    ])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _workspace(ts_root):
    return os.path.join(os.path.abspath(ts_root), "fewshot_models")


def _status_path(ts_root, job_id):
    return os.path.join(_workspace(ts_root), "jobs", job_id + ".json")


def _current_project_path():
    try:
        info = mimics.file.get_project_information()
    except Exception:
        return None
    for attr in ("filename", "file_name", "path", "project_path", "project_file"):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if value:
            return os.path.abspath(str(value))
    try:
        for attr in dir(info):
            if attr.startswith("_"):
                continue
            value = getattr(info, attr, None)
            if value and str(value).lower().endswith(".mcs"):
                return os.path.abspath(str(value))
    except Exception:
        pass
    return None


def _infer_dataset_root_from_project():
    project_path = _current_project_path()
    if not project_path:
        return None
    project_dir = os.path.dirname(project_path)
    if os.path.basename(project_dir).lower() != "mcs_output":
        return None
    dataset_root = os.path.dirname(project_dir)
    if os.path.isdir(dataset_root):
        return dataset_root
    return None


def _choose_dataset_root(title):
    inferred = _infer_dataset_root_from_project()
    if inferred and os.path.isdir(inferred):
        settings = _load_settings()
        settings["last_dataset_root"] = inferred
        _save_settings(settings)
        return inferred
    settings = _load_settings()
    initial = settings.get("last_dataset_root", "")
    path = _pick_directory(title, initial)
    if path and os.path.isdir(path):
        settings["last_dataset_root"] = os.path.abspath(path)
        _save_settings(settings)
        return os.path.abspath(path)
    return None


def _infer_case_id(ts_root):
    project_path = _current_project_path()
    if not project_path:
        return None
    output_dir = os.path.abspath(os.path.join(ts_root, "mcs_output"))
    project_dir = os.path.abspath(os.path.dirname(project_path))
    if os.path.normcase(project_dir) != os.path.normcase(output_dir):
        return None
    name = os.path.basename(project_path)
    if name.lower().endswith(".mcs"):
        return name[:-4]
    return None


def _process_exists(pid):
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


def _terminate_process_tree(pid):
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
                **_hidden_process_kwargs()
            )
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def _latest_active_job(ts_root):
    jobs_dir = os.path.join(_workspace(ts_root), "jobs")
    if not os.path.isdir(jobs_dir):
        return None, None
    active = set(["launching", "preparing", "exporting_labels", "training", "running", "cancelling"])
    rows = []
    for name in os.listdir(jobs_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(jobs_dir, name)
        payload = _read_json(path, {}) or {}
        if payload.get("status") not in active:
            continue
        pid = payload.get("pid") or payload.get("launcher_pid") or payload.get("controller_pid")
        if pid and not _process_exists(pid):
            continue
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0
        rows.append((mtime, path, payload))
    if not rows:
        return None, None
    rows.sort(reverse=True)
    return rows[0][1], rows[0][2]


def _guard_no_active_job(ts_root):
    path, job = _latest_active_job(ts_root)
    if not job:
        return True
    mimics.dialogs.message_box(
        "A few-shot job is already running for this dataset.\n\n"
        "Job: {0}\nType: {1}\nOrgan: {2}\nStatus: {3}\n\n"
        "Use Show Status or Stop Latest Job before starting another GPU job.".format(
            job.get("job_id", os.path.basename(path or "")),
            job.get("kind", "?"),
            job.get("organ", "?"),
            job.get("status", "?"),
        ),
        title=TITLE,
        ui_blocking=False,
    )
    return False


def _launch_process(cmd, cwd=None):
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_background_env(),
        **_hidden_process_kwargs()
    )


def _train_model():
    organ = _selected_organ()
    if not organ:
        mimics.dialogs.message_box(
            "Select one Mask whose name is the organ to train, then run this entry again.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    ts_root = _choose_dataset_root("Select dataset folder")
    if not ts_root or not os.path.isdir(ts_root):
        mimics.dialogs.message_box("No valid dataset folder was selected.", title=TITLE, ui_blocking=False)
        return 1
    if not _guard_no_active_job(ts_root):
        return 1
    answer = mimics.dialogs.question_box(
        message=(
            "Training uses saved .mcs files.\n\n"
            "Save the current project before starting so the latest manual edits "
            "are included in the exported labels."
        ),
        buttons="Start Training;" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer != "Start Training":
        return 0

    config = _config()
    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
    base_config = config.get("base_config", "config/synthstrip_lora_segformer3d.yaml")
    run_id = "train_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    cmd = [
        python_exe,
        _pipeline_script(),
        "train",
        "--ts-root",
        ts_root,
        "--organ",
        organ,
        "--dinov3-root",
        dinov3_root,
        "--python",
        python_exe,
        "--base-config",
        base_config,
        "--epochs",
        str(config.get("default_epochs", 10)),
        "--batch-size",
        str(config.get("default_batch_size", 1)),
        "--grad-accumulation",
        str(config.get("default_grad_accumulation", 1)),
        "--lr",
        str(config.get("default_lr", 0.001)),
        "--img-size",
        str(config.get("default_img_size", "224,224")),
        "--modality",
        str(config.get("default_modality", "ct")),
        "--min-samples",
        str(config.get("default_min_samples", 1)),
        "--max-samples",
        str(config.get("default_max_samples", 0)),
        "--export-labels",
        "--run-id",
        run_id,
    ]
    mimics_exe = _find_mimics_exe()
    if mimics_exe:
        cmd.extend(["--mimics-exe", mimics_exe])

    process = _launch_process(cmd, cwd=_project_root())
    organ_slug = _safe_slug(organ)
    cancel_path = os.path.join(_workspace(ts_root), "runs", organ_slug, run_id, "cancel.request")
    _write_json_atomic(
        _status_path(ts_root, run_id),
        {
            "schema_version": "mimics_fewshot_job.v1",
            "job_id": run_id,
            "kind": "train",
            "status": "launching",
            "organ": organ,
            "ts_root": os.path.abspath(ts_root),
            "workspace": _workspace(ts_root),
            "launcher_pid": process.pid,
            "cancel_path": cancel_path,
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    _mimics_log(
        logging.INFO,
        "DINOv3 few-shot training started. Organ: {0}, PID: {1}, job: {2}.".format(organ, process.pid, run_id),
    )
    mimics.dialogs.message_box(
        "Few-shot training started in the background.\n\nOrgan: {0}\nJob: {1}\nStatus folder:\n{2}".format(
            organ,
            run_id,
            os.path.join(_workspace(ts_root), "jobs"),
        ),
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def _start_inference():
    organ = _selected_organ()
    if not organ:
        mimics.dialogs.message_box(
            "Select one Mask whose name is the organ/model to use, then run this entry again.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    ts_root = _choose_dataset_root("Select dataset folder")
    if not ts_root or not os.path.isdir(ts_root):
        mimics.dialogs.message_box("No valid dataset folder was selected.", title=TITLE, ui_blocking=False)
        return 1
    if not _guard_no_active_job(ts_root):
        return 1
    case_id = _infer_case_id(ts_root)
    if not case_id:
        mimics.dialogs.message_box(
            "Could not infer the current case.\n\nOpen a saved project from:\n{0}".format(
                os.path.join(ts_root, "mcs_output", "<case>.mcs")
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1

    config = _config()
    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
    job_id = "infer_{0}_{1}_{2}".format(_safe_slug(case_id), _safe_slug(organ), uuid.uuid4().hex[:8])
    cmd = [
        python_exe,
        _pipeline_script(),
        "infer",
        "--ts-root",
        ts_root,
        "--case-id",
        case_id,
        "--organ",
        organ,
        "--dinov3-root",
        dinov3_root,
        "--python",
        python_exe,
        "--job-id",
        job_id,
    ]
    process = _launch_process(cmd, cwd=_project_root())
    status_path = _status_path(ts_root, job_id)
    cancel_path = os.path.join(
        _workspace(ts_root),
        "predictions",
        _safe_slug(case_id),
        _safe_slug(organ),
        job_id + ".cancel",
    )
    _write_json_atomic(
        status_path,
        {
            "schema_version": "mimics_fewshot_job.v1",
            "job_id": job_id,
            "kind": "infer",
            "status": "launching",
            "organ": organ,
            "case_id": case_id,
            "ts_root": os.path.abspath(ts_root),
            "workspace": _workspace(ts_root),
            "launcher_pid": process.pid,
            "cancel_path": cancel_path,
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    monitor = {
        "monitor_key": job_id,
        "status_path": status_path,
        "ts_root": ts_root,
        "case_id": case_id,
        "organ": organ,
        "job_id": job_id,
        "pid": process.pid,
        "deadline": time.time() + 12 * 60 * 60,
        "bridge_started": False,
        "bridge_job_dir": os.path.join(_workspace(ts_root), "jobs", job_id + "_apply"),
        "mask_name": "AI_" + organ,
    }
    _start_monitor(monitor)
    _mimics_log(
        logging.INFO,
        "DINOv3 few-shot inference started. Organ: {0}, case: {1}, PID: {2}.".format(
            organ,
            case_id,
            process.pid,
        ),
    )
    mimics.dialogs.message_box(
        "Few-shot inference started in the background.\n\nOrgan: {0}\nCase: {1}\nThe result will be applied automatically when ready.".format(
            organ,
            case_id,
        ),
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def _launch_bridge_mask_to_buffer(monitor, status):
    job_dir = monitor["bridge_job_dir"]
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)
    output_path = os.path.join(job_dir, "prediction.u8")
    params = {
        "action": "mask_to_buffer",
        "image_path": status["image_path"],
        "mask_path": status["output_path"],
        "output_path": output_path,
        "axes": [1, 0, 2],
        "flips": [False, False, False],
    }
    input_path = os.path.join(job_dir, "bridge_input.json")
    result_path = os.path.join(job_dir, "bridge_result.json")
    _write_json_atomic(input_path, params)
    python_exe = _fewshot_python(_config(), _dinov3_root(_config()))
    with open(input_path, "rb") as stdin_handle:
        process = subprocess.Popen(
            [python_exe, _bridge_script()],
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_hidden_process_kwargs()
        )
    monitor["bridge_started"] = True
    monitor["bridge_pid"] = process.pid
    monitor["bridge_result_path"] = result_path
    monitor["bridge_output_path"] = output_path

    def _wait_bridge():
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            result = {
                "status": "error",
                "error": stderr.decode("utf-8", "replace")[:1000],
            }
        else:
            try:
                result = json.loads(stdout.decode("utf-8"))
            except Exception as exc:
                result = {"status": "error", "error": "invalid bridge JSON: {0}".format(exc)}
        _write_json_atomic(result_path, result)

    try:
        import threading
        thread = threading.Thread(target=_wait_bridge)
        thread.daemon = True
        thread.start()
    except Exception:
        pass


def _find_or_create_mask(name):
    active_image = mimics.data.images.get_active()
    for mask in mimics.data.masks:
        if str(getattr(mask, "name", "")) != name:
            continue
        try:
            if active_image is not None and getattr(mask, "image", None) not in (None, active_image):
                continue
        except Exception:
            pass
        return mask
    mask = mimics.segment.create_mask()
    mask.name = name
    try:
        mask.image = active_image
    except Exception:
        pass
    return mask


def _set_mask_from_u8(mask, path, shape):
    raw = open(path, "rb").read()
    expected = int(shape[0]) * int(shape[1]) * int(shape[2])
    if len(raw) != expected:
        raise RuntimeError("Prediction byte count mismatch: {0} != {1}".format(len(raw), expected))
    try:
        import numpy as np
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).astype(np.bool_)
        mask.set_voxel_buffer(pixels)
    except ImportError:
        pixels = memoryview(bytearray(raw)).cast("?", shape=list(shape))
        mask.set_voxel_buffer(pixels)
    try:
        mask.visible = True
        mask.selected = True
    except Exception:
        pass


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
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        try:
            user32, timer_id = win32_timer
            user32.KillTimer(None, timer_id)
        except Exception:
            pass


def _monitor_tick(monitor):
    key = monitor.get("monitor_key")
    if time.time() > monitor.get("deadline", 0):
        _stop_monitor(key)
        mimics.dialogs.message_box("Few-shot inference timed out.", title=TITLE, ui_blocking=False)
        return
    status = _read_json(monitor["status_path"], {}) or {}
    state = status.get("status")
    if state in ("", None, "launching", "running"):
        return
    if state in ("cancelled", "cancelling"):
        _stop_monitor(key)
        mimics.dialogs.message_box("Few-shot inference was cancelled.", title=TITLE, ui_blocking=False)
        return
    if state == "failed":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "Few-shot inference failed.\n\n{0}".format(status.get("error", "Unknown error")),
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state != "completed":
        return
    if not monitor.get("bridge_started"):
        _launch_bridge_mask_to_buffer(monitor, status)
        return
    bridge_result = _read_json(monitor.get("bridge_result_path"), None)
    if bridge_result is None:
        return
    if bridge_result.get("status") != "ok":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "Prediction conversion failed.\n\n{0}".format(bridge_result.get("error", "Unknown error")),
            title=TITLE,
            ui_blocking=False,
        )
        return
    try:
        mask = _find_or_create_mask(monitor["mask_name"])
        _set_mask_from_u8(mask, bridge_result["output_path"], bridge_result["mimics_shape"])
    except Exception as exc:
        _stop_monitor(key)
        mimics.dialogs.message_box("Could not apply prediction:\n\n{0}".format(exc), title=TITLE, ui_blocking=False)
        return
    _stop_monitor(key)
    _mimics_log(
        logging.INFO,
        "DINOv3 few-shot result applied. Mask: {0}, foreground voxels: {1}.".format(
            monitor["mask_name"],
            bridge_result.get("foreground_voxels", "?"),
        ),
    )
    mimics.dialogs.message_box(
        "Few-shot prediction applied to Mask:\n{0}".format(monitor["mask_name"]),
        title=TITLE,
        ui_blocking=False,
    )


def _start_win32_monitor(monitor, poll_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False
    key = monitor["monitor_key"]
    _stop_monitor(key)
    user32 = ctypes.windll.user32
    interval = max(250, int(max(0.25, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint)

    def _timer_proc(hwnd, message, timer_id, tick_count):
        _monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    timer_id = user32.SetTimer(None, 0, interval, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _MONITORS[key] = monitor
    return True


def _start_monitor(monitor, poll_seconds=1.0):
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_monitor(monitor, poll_seconds):
            return True
        return False
    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_monitor(monitor, poll_seconds):
            return True
        return False
    key = monitor["monitor_key"]
    _stop_monitor(key)
    timer = QTimer()
    monitor["timer"] = timer
    _MONITORS[key] = monitor

    def _tick():
        _monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(250, int(max(0.25, poll_seconds) * 1000)))
    return True


def _format_job_line(job):
    progress = job.get("training_progress") or {}
    extra = ""
    if progress:
        epoch = progress.get("epoch")
        epochs = progress.get("epochs")
        phase = progress.get("phase")
        best = progress.get("best_dsc")
        pieces = []
        if epoch is not None and epochs is not None:
            pieces.append("epoch {0}/{1}".format(epoch, epochs))
        if phase:
            pieces.append(str(phase))
        if best is not None:
            try:
                pieces.append("best_dsc {0:.4f}".format(float(best)))
            except Exception:
                pieces.append("best_dsc {0}".format(best))
        if pieces:
            extra = " | " + ", ".join(pieces)
    if job.get("sample_count") is not None:
        extra += " | samples {0}".format(job.get("sample_count"))
    return "{0} | {1} | {2} | {3}{4}".format(
        job.get("job_id", "?"),
        job.get("kind", "?"),
        job.get("organ", "?"),
        job.get("status", "?"),
        extra,
    )


def _latest_model_lines(ts_root, selected_organ=None):
    models_dir = os.path.join(_workspace(ts_root), "models")
    if not os.path.isdir(models_dir):
        return []
    rows = []
    for organ_slug in sorted(os.listdir(models_dir)):
        latest = os.path.join(models_dir, organ_slug, "latest.json")
        if not os.path.isfile(latest):
            continue
        manifest = _read_json(latest, {}) or {}
        organ = manifest.get("organ", organ_slug)
        if selected_organ and organ != selected_organ:
            continue
        rows.append(
            "Model {0}: {1}, samples {2}".format(
                organ,
                manifest.get("model_id", "?"),
                manifest.get("sample_count", "?"),
            )
        )
    return rows


def _show_status():
    ts_root = _choose_dataset_root("Select dataset folder")
    if not ts_root or not os.path.isdir(ts_root):
        return 1
    jobs_dir = os.path.join(_workspace(ts_root), "jobs")
    if not os.path.isdir(jobs_dir):
        mimics.dialogs.message_box("No few-shot jobs were found.", title=TITLE, ui_blocking=False)
        return 0
    jobs = []
    for name in os.listdir(jobs_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(jobs_dir, name)
        payload = _read_json(path, {}) or {}
        jobs.append((os.path.getmtime(path), payload))
    jobs.sort(reverse=True)
    lines = []
    for index, item in enumerate(jobs[:8]):
        _, job = item
        lines.append(_format_job_line(job))
        if index == 0:
            log_path = job.get("train_log") or job.get("log")
            if log_path:
                lines.append("  Log: {0}".format(log_path))
            model = job.get("model") or {}
            if model.get("checkpoint"):
                lines.append("  Model: {0}".format(model.get("checkpoint")))
            if job.get("output_path"):
                lines.append("  Output: {0}".format(job.get("output_path")))
    try:
        selected_organ = _selected_organ()
    except Exception:
        selected_organ = None
    model_lines = _latest_model_lines(ts_root, selected_organ)
    if model_lines:
        lines.append("")
        lines.extend(model_lines[:5])
    active_path, active_job = _latest_active_job(ts_root)
    if active_job:
        lines.append("")
        lines.append("Active job can be stopped with Stop Latest Job.")
    mimics.dialogs.message_box(
        "\n".join(lines) if lines else "No few-shot jobs were found.",
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def _stop_latest_job():
    ts_root = _choose_dataset_root("Select dataset folder")
    if not ts_root or not os.path.isdir(ts_root):
        return 1
    status_path, job = _latest_active_job(ts_root)
    if not job:
        mimics.dialogs.message_box("No running few-shot job was found.", title=TITLE, ui_blocking=False)
        return 0
    answer = mimics.dialogs.question_box(
        message=(
            "Stop the latest few-shot job?\n\n"
            "Job: {0}\nType: {1}\nOrgan: {2}\nStatus: {3}"
        ).format(
            job.get("job_id", "?"),
            job.get("kind", "?"),
            job.get("organ", "?"),
            job.get("status", "?"),
        ),
        buttons=BUTTON_STOP + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer != BUTTON_STOP:
        return 0
    cancel_path = job.get("cancel_path")
    if cancel_path:
        try:
            parent = os.path.dirname(cancel_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            with open(cancel_path, "w") as handle:
                handle.write("cancel requested at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        except Exception:
            pass
    killed = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        pid = job.get(key)
        if pid and _terminate_process_tree(pid):
            killed.append(str(pid))
    job["status"] = "cancelled"
    job["cancel_requested_at_epoch"] = time.time()
    job["cancelled_pids"] = killed
    job["updated_at_epoch"] = time.time()
    if status_path:
        _write_json_atomic(status_path, job)
    mimics.dialogs.message_box(
        "Stop request submitted for job:\n{0}".format(job.get("job_id", "?")),
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def main():
    action = mimics.dialogs.question_box(
        message=(
            "Select one organ Mask before training or prediction.\n\n"
            "Training uses saved .mcs files and runs fully in the background."
        ),
        buttons=";".join([BUTTON_TRAIN, BUTTON_PREDICT, BUTTON_STATUS, BUTTON_STOP, BUTTON_CANCEL]),
        title=TITLE,
        ui_blocking=True,
    )
    if action == BUTTON_TRAIN:
        return _train_model()
    if action == BUTTON_PREDICT:
        return _start_inference()
    if action == BUTTON_STATUS:
        return _show_status()
    if action == BUTTON_STOP:
        return _stop_latest_job()
    return 0


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        try:
            mimics.dialogs.message_box("Error: {0}".format(error), title=TITLE, ui_blocking=True)
        except Exception:
            pass
        raise
