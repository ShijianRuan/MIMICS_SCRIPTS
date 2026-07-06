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
BUTTON_TRAIN_ADVANCED = "Train Advanced..."
BUTTON_PREDICT = "Predict Current Case"
BUTTON_PREDICT_MODEL = "Predict With Model..."
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


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _with_gui_updates_disabled(fn, *args, **kwargs):
    gui_was_enabled = True
    try:
        try:
            gui_was_enabled = bool(mimics.is_update_gui_enabled())
        except Exception:
            gui_was_enabled = True
        try:
            mimics.disable_update_gui()
        except Exception:
            gui_was_enabled = False
        return fn(*args, **kwargs)
    finally:
        if gui_was_enabled:
            try:
                mimics.enable_update_gui()
            except Exception:
                pass
        _update_gui()


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
    logged = False
    try:
        mimics.logging.log_user_message(level=level, message=message)
        logged = True
    except Exception:
        pass
    if not logged:
        print("[fewshot {0}] {1}".format(
            {logging.INFO: "INFO", logging.WARNING: "WARN", logging.ERROR: "ERROR"}.get(level, "?"),
            message,
        ))


def _ensure_pyqt5():
    """Try to make PyQt5 importable by searching Mimics installation paths.

    Mimics GUI is built on Qt.  The Python bindings (PyQt5 or PySide2/6)
    may live in a non-standard directory that is not on sys.path.

    Returns True if Qt Python bindings are now importable.
    """
    try:
        import PyQt5  # noqa: F401
        return True
    except ImportError:
        pass

    # Known locations for Qt Python bindings inside Mimics installations
    search_roots = []
    for env_var in ("MIMICS_HOME", "MIMICS_EXE"):
        val = os.environ.get(env_var, "")
        if val and os.path.exists(val):
            search_roots.append(os.path.dirname(val) if os.path.isfile(val) else val)

    pf = os.environ.get("ProgramFiles", "C:\\Program Files")
    for base_dir in (
        os.path.join(pf, "Materialise"),
        os.path.join(pf, "Common Files", "Materialise"),
        "C:\\Program Files\\Materialise",
        "C:\\Program Files\\Common Files\\Materialise",
        "D:\\Program Files\\Materialise",
    ):
        if os.path.isdir(base_dir) and base_dir not in search_roots:
            search_roots.append(base_dir)

    for root in search_roots:
        for dirpath, _dirnames, _filenames in os.walk(root):
            # Stop walking too deep
            depth = dirpath.replace(root, "").count(os.sep)
            if depth > 5:
                continue
            lower = os.path.basename(dirpath).lower()
            if lower in ("site-packages", "pyqt5", "pyside2", "pyside6", "pyside"):
                if dirpath not in sys.path:
                    sys.path.insert(0, dirpath)
                try:
                    import PyQt5  # noqa: F401
                    return True
                except ImportError:
                    pass

    # Try alternatives that ship with some Mimics builds
    for mod_name in ("PySide2", "PySide6", "PySide"):
        try:
            mod = __import__(mod_name)
            sys.modules["PyQt5"] = mod
            try:
                sys.modules["PyQt5.QtWidgets"] = getattr(mod, "QtWidgets")
            except AttributeError:
                pass
            try:
                sys.modules["PyQt5.QtCore"] = getattr(mod, "QtCore")
            except AttributeError:
                pass
            return True
        except ImportError:
            pass

    return False


def _pick_directory(title, initial_dir=""):
    if _ensure_pyqt5():
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


def _global_model_registry_path():
    return os.path.join(os.path.expanduser("~"), ".mimics_script", "fewshot_model_index.json")


def _status_path(ts_root, job_id):
    return os.path.join(_workspace(ts_root), "jobs", job_id + ".json")


def _case_ids_from_dataset(ts_root):
    rows = set()
    mcs_dir = os.path.join(ts_root, "mcs_output")
    if os.path.isdir(mcs_dir):
        try:
            for name in os.listdir(mcs_dir):
                if name.lower().endswith(".mcs"):
                    rows.add(name[:-4])
        except Exception:
            pass
    try:
        for name in os.listdir(ts_root):
            path = os.path.join(ts_root, name)
            if os.path.isdir(path) and name not in ("mcs_output", "segmentations", "fewshot_models"):
                rows.add(name)
    except Exception:
        pass
    return sorted(rows)


def _default_training_options(config, profile_name=None):
    profiles = config.get("training_profiles") or {}
    default_profile = profile_name or config.get("default_training_profile") or config.get("default_profile")
    values = {}
    if default_profile and isinstance(profiles, dict):
        values.update(profiles.get(default_profile, {}) or {})
    values.setdefault("base_config", config.get("base_config", "config/synthstrip_lora_segformer3d.yaml"))
    values.setdefault("epochs", config.get("default_epochs", 10))
    values.setdefault("batch_size", config.get("default_batch_size", 1))
    values.setdefault("grad_accumulation", config.get("default_grad_accumulation", 1))
    values.setdefault("lr", config.get("default_lr", 0.001))
    values.setdefault("weight_decay", config.get("default_weight_decay", 0.01))
    values.setdefault("img_size", config.get("default_img_size", "224,224"))
    values.setdefault("modality", config.get("default_modality", "ct"))
    values.setdefault("min_samples", config.get("default_min_samples", 1))
    values.setdefault("max_samples", config.get("default_max_samples", 0))
    values.setdefault("sample_mode", config.get("default_sample_mode", "all"))
    values.setdefault("val_fraction", config.get("default_val_fraction", 0.2))
    values.setdefault("min_val_samples", config.get("default_min_val_samples", 1))
    values.setdefault("finetune_method", config.get("default_finetune_method", "lora"))
    values.setdefault("decoder", config.get("default_decoder", "segformer3d"))
    values.setdefault("model_scale", config.get("default_model_scale", "vitb16"))
    values.setdefault("model_path", config.get("default_model_path", ""))
    values.setdefault("lora_rank", config.get("default_lora_rank", 8))
    values.setdefault("lora_alpha", config.get("default_lora_alpha", 16))
    values.setdefault("adapter_bottleneck", config.get("default_adapter_bottleneck", 64))
    values.setdefault("mixed_precision", config.get("default_mixed_precision", False))
    values.setdefault("sub_volume", config.get("default_sub_volume", False))
    values.setdefault("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))
    values.setdefault("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2))
    values.setdefault("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault(
        "background_mimics_lock_timeout_seconds",
        config.get("background_mimics_lock_timeout_seconds", 21600),
    )
    return values


def _split_csv(values):
    if not values:
        return []
    if isinstance(values, (list, tuple)):
        return [str(item).strip() for item in values if str(item).strip()]
    return [item.strip() for item in str(values).replace(";", ",").split(",") if item.strip()]


def _append_training_args(cmd, config, options):
    cmd.extend([
        "--base-config",
        str(options.get("base_config", config.get("base_config", "config/synthstrip_lora_segformer3d.yaml"))),
        "--epochs",
        str(int(options.get("epochs", config.get("default_epochs", 10)))),
        "--batch-size",
        str(int(options.get("batch_size", config.get("default_batch_size", 1)))),
        "--grad-accumulation",
        str(int(options.get("grad_accumulation", config.get("default_grad_accumulation", 1)))),
        "--lr",
        str(float(options.get("lr", config.get("default_lr", 0.001)))),
        "--weight-decay",
        str(float(options.get("weight_decay", config.get("default_weight_decay", 0.01)))),
        "--img-size",
        str(options.get("img_size", config.get("default_img_size", "224,224"))),
        "--modality",
        str(options.get("modality", config.get("default_modality", "ct"))),
        "--min-samples",
        str(int(options.get("min_samples", config.get("default_min_samples", 1)))),
        "--max-samples",
        str(int(options.get("max_samples", config.get("default_max_samples", 0)))),
        "--sample-mode",
        str(options.get("sample_mode", config.get("default_sample_mode", "all"))),
        "--val-fraction",
        str(float(options.get("val_fraction", config.get("default_val_fraction", 0.2)))),
        "--min-val-samples",
        str(int(options.get("min_val_samples", config.get("default_min_val_samples", 1)))),
        "--finetune-method",
        str(options.get("finetune_method", config.get("default_finetune_method", "lora"))),
        "--decoder",
        str(options.get("decoder", config.get("default_decoder", "segformer3d"))),
        "--model-scale",
        str(options.get("model_scale", config.get("default_model_scale", "vitb16"))),
        "--lora-rank",
        str(int(options.get("lora_rank", config.get("default_lora_rank", 8)))),
        "--lora-alpha",
        str(int(options.get("lora_alpha", config.get("default_lora_alpha", 16)))),
        "--adapter-bottleneck",
        str(int(options.get("adapter_bottleneck", config.get("default_adapter_bottleneck", 64)))),
        "--gpu-lock-timeout-seconds",
        str(float(options.get("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400)))),
        "--background-mimics-lock-timeout-seconds",
        str(float(options.get("background_mimics_lock_timeout_seconds", config.get("background_mimics_lock_timeout_seconds", 21600)))),
        "--keep-last-checkpoints",
        str(int(options.get("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2)))),
    ])
    model_path = str(options.get("model_path", config.get("default_model_path", "")) or "")
    if model_path:
        cmd.extend(["--model-path", model_path])
    val_cases = _split_csv(options.get("val_cases", ""))
    if val_cases:
        cmd.extend(["--val-cases", ",".join(val_cases)])
    cases = _split_csv(options.get("cases", ""))
    if cases:
        cmd.extend(["--cases", ",".join(cases)])
    if bool(options.get("mixed_precision", config.get("default_mixed_precision", False))):
        cmd.append("--mixed-precision")
    if bool(options.get("sub_volume", config.get("default_sub_volume", False))):
        cmd.append("--sub-volume")
    cmd.extend(["--sub-volume-size", str(options.get("sub_volume_size", config.get("default_sub_volume_size", "32,256,256")))])
    if bool(options.get("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))):
        cmd.append("--keep-materialized-dataset")


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
    active = set([
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


def _display_status(value):
    labels = {
        "launching": "Launching",
        "preparing": "Preparing data",
        "exporting_labels": "Exporting labels",
        "waiting_for_background_mimics": "Waiting for background Mimics",
        "waiting_for_gpu": "Waiting for GPU",
        "training": "Training",
        "running": "Running inference",
        "cancelling": "Cancelling",
        "cancelled": "Cancelled",
        "failed": "Failed",
        "completed": "Completed",
    }
    return labels.get(str(value or ""), str(value or "Unknown").replace("_", " "))


def _display_resource(value):
    labels = {
        "gpu": "GPU",
        "background_mimics": "background Mimics",
    }
    return labels.get(str(value or ""), str(value or "resource").replace("_", " "))


def _resource_wait_text(job):
    resource_wait = job.get("resource_wait") or {}
    if not resource_wait:
        return ""
    resource = _display_resource(resource_wait.get("resource", "resource"))
    holder = resource_wait.get("owner", "unknown")
    pid = resource_wait.get("pid", "")
    if pid:
        return "Waiting for {0}: {1} (PID {2})".format(resource, holder, pid)
    return "Waiting for {0}".format(resource)


def _guard_no_active_job(ts_root):
    path, job = _latest_active_job(ts_root)
    if not job:
        return True
    wait_text = _resource_wait_text(job)
    detail = "\n{0}".format(wait_text) if wait_text else ""
    mimics.dialogs.message_box(
        "A few-shot job is already running for this dataset.\n\n"
        "Job: {0}\nType: {1}\nOrgan: {2}\nStatus: {3}{4}\n\n"
        "Use Show Status or Stop Latest Job before starting another GPU job.".format(
            job.get("job_id", os.path.basename(path or "")),
            job.get("kind", "?"),
            job.get("organ", "?"),
            _display_status(job.get("status", "?")),
            detail,
        ),
        title=TITLE,
        ui_blocking=False,
    )
    return False


def _native_training_param_picker(config):
    """Collect key training parameters via Mimics native dialogs.

    No PyQt5 required.  Each parameter is chosen from a short list of
    sensible presets so the user still has meaningful control.
    """
    values = _default_training_options(config)

    # -- Epochs --
    epoch_choices = [("5 epochs (fast test)", 5), ("10 epochs (quick)", 10),
                     ("20 epochs (standard)", 20), ("50 epochs (deep)", 50)]
    epoch_labels = [label for label, _ in epoch_choices]
    epoch_labels.append("Keep default ({0})".format(values.get("epochs", "?")))
    answer = mimics.dialogs.question_box(
        message="Training epochs:\n(more = longer training, potentially better results)",
        buttons=";".join(epoch_labels),
        title=TITLE,
        ui_blocking=True,
    )
    for label, val in epoch_choices:
        if answer == label:
            values["epochs"] = val
            break

    # -- Fine-tuning method --
    method_choices = [("LoRA (fast, low VRAM)", "lora"),
                      ("Adapter (balanced)", "adapter")]
    method_labels = [label for label, _ in method_choices]
    method_labels.append("Keep default ({0})".format(values.get("finetune_method", "lora")))
    answer = mimics.dialogs.question_box(
        message="Fine-tuning method:",
        buttons=";".join(method_labels),
        title=TITLE,
        ui_blocking=True,
    )
    for label, val in method_choices:
        if answer == label:
            values["finetune_method"] = val
            break

    # -- Batch size --
    batch_choices = [("Batch size 1 (minimum VRAM)", 1), ("Batch size 2", 2)]
    batch_labels = [label for label, _ in batch_choices]
    batch_labels.append("Keep default ({0})".format(values.get("batch_size", "?")))
    answer = mimics.dialogs.question_box(
        message="Batch size:\n(bigger = faster training but needs more GPU memory)",
        buttons=";".join(batch_labels),
        title=TITLE,
        ui_blocking=True,
    )
    for label, val in batch_choices:
        if answer == label:
            values["batch_size"] = val
            break

    return values


def _profile_training_options(config):
    profiles = config.get("training_profiles") or {}
    profile_names = sorted(profiles.keys()) if isinstance(profiles, dict) else []
    default_name = config.get("default_training_profile") or config.get("default_profile") or ""

    if profile_names:
        lines = [
            "PyQt5 / PySide is not available in this session.",
            "",
            "Choose a training profile from fewshot_config.json,",
            "or pick 'Configure manually' to set parameters interactively.",
            "",
        ]
        for name in profile_names:
            vals = _default_training_options(config, name)
            lines.append("{0}: {1}, {2} epoch(s), batch {3}".format(
                name, vals.get("finetune_method", "?"), vals.get("epochs", "?"), vals.get("batch_size", "?")))
        buttons = list(profile_names)
        buttons.append("Configure manually")
        buttons.append(BUTTON_CANCEL)
        answer = mimics.dialogs.question_box(
            message="\n".join(lines), buttons=";".join(buttons), title=TITLE, ui_blocking=True)
        if answer == BUTTON_CANCEL or not answer:
            return None
        if answer == "Configure manually":
            return _native_training_param_picker(config)
        if answer in profile_names:
            return _default_training_options(config, answer)
        return _default_training_options(config, default_name)

    # No profiles at all — let user configure or use defaults
    answer = mimics.dialogs.question_box(
        message=(
            "PyQt5 / PySide is not available and no training_profiles\n"
            "were found in fewshot_config.json.\n\n"
            "You can configure key parameters interactively, or use\n"
            "the defaults from fewshot_config.json.\n\n"
            "Tip: add training_profiles to fewshot_config.json to\n"
            "save and reuse your settings."
        ),
        buttons="Configure interactively;Use defaults;" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer == BUTTON_CANCEL or not answer:
        return None
    if answer == "Use defaults":
        return _default_training_options(config)
    return _native_training_param_picker(config)


def _advanced_training_options(config, ts_root):
    if not _ensure_pyqt5():
        _mimics_log(
            logging.WARNING,
            "DINOv3 PyQt5 / PySide2 / PySide6 is not available. Using fewshot_config.json profiles instead.",
        )
        return _profile_training_options(config)
    try:
        from PyQt5.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        _mimics_log(
            logging.WARNING,
            "DINOv3 PyQt5 widgets not importable; using profile selector.",
        )
        return _profile_training_options(config)

    profiles = config.get("training_profiles") or {}
    profile_names = sorted(profiles.keys()) if isinstance(profiles, dict) else []
    default_name = config.get("default_training_profile") or (profile_names[0] if profile_names else "")
    values = _default_training_options(config, default_name)

    dialog = QDialog()
    dialog.setWindowTitle("DINOv3 Few-Shot Training")
    dialog.resize(720, 620)
    root = QVBoxLayout(dialog)
    root.addWidget(QLabel("Choose training samples and key parameters. Empty case selection means all eligible cases."))

    tabs = QTabWidget(dialog)
    root.addWidget(tabs)

    sample_tab = QWidget()
    sample_layout = QVBoxLayout(sample_tab)
    profile_combo = QComboBox()
    if profile_names:
        profile_combo.addItems(profile_names)
        if default_name in profile_names:
            profile_combo.setCurrentIndex(profile_names.index(default_name))
    case_list = QListWidget()
    case_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    for case_id in _case_ids_from_dataset(ts_root):
        item = QListWidgetItem(case_id)
        case_list.addItem(item)
    sample_mode = QComboBox()
    sample_mode.addItems(["all", "latest"])
    if values.get("sample_mode") in ("all", "latest"):
        sample_mode.setCurrentIndex(["all", "latest"].index(values.get("sample_mode")))
    max_samples = QSpinBox()
    max_samples.setRange(0, 100000)
    max_samples.setValue(int(values.get("max_samples", 0)))
    min_samples = QSpinBox()
    min_samples.setRange(1, 100000)
    min_samples.setValue(int(values.get("min_samples", 1)))
    val_fraction = QDoubleSpinBox()
    val_fraction.setRange(0.0, 0.9)
    val_fraction.setSingleStep(0.05)
    val_fraction.setDecimals(2)
    val_fraction.setValue(float(values.get("val_fraction", 0.2)))

    sample_form = QFormLayout()
    sample_form.addRow("Profile", profile_combo)
    sample_form.addRow("Sample order", sample_mode)
    sample_form.addRow("Max samples (0 = all)", max_samples)
    sample_form.addRow("Min train samples", min_samples)
    sample_form.addRow("Validation fraction", val_fraction)
    sample_layout.addLayout(sample_form)
    sample_layout.addWidget(QLabel("Cases"))
    sample_layout.addWidget(case_list)
    tabs.addTab(sample_tab, "Samples")

    train_tab = QWidget()
    train_form = QFormLayout(train_tab)
    finetune = QComboBox()
    finetune.addItems(["lora", "frozen", "adapter", "full"])
    decoder = QComboBox()
    decoder.addItems(["segformer3d", "mlp_probe", "linear3d", "dpt3d"])
    model_scale = QComboBox()
    model_scale.addItems(["vitb16", "vitl16", "vith16plus"])
    modality = QComboBox()
    modality.addItems(["ct", "mri", "other"])
    epochs = QSpinBox()
    epochs.setRange(1, 10000)
    batch_size = QSpinBox()
    batch_size.setRange(1, 128)
    grad_accum = QSpinBox()
    grad_accum.setRange(1, 1024)
    lr = QLineEdit()
    weight_decay = QLineEdit()
    img_size = QLineEdit()
    base_config = QLineEdit()
    model_path = QLineEdit()
    mixed_precision = QCheckBox()
    sub_volume = QCheckBox()
    sub_volume_size = QLineEdit()
    keep_last_checkpoints = QSpinBox()
    keep_last_checkpoints.setRange(0, 1000)
    keep_materialized_dataset = QCheckBox()

    widgets = {
        "finetune_method": finetune,
        "decoder": decoder,
        "model_scale": model_scale,
        "modality": modality,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accumulation": grad_accum,
        "lr": lr,
        "weight_decay": weight_decay,
        "img_size": img_size,
        "base_config": base_config,
        "model_path": model_path,
        "mixed_precision": mixed_precision,
        "sub_volume": sub_volume,
        "sub_volume_size": sub_volume_size,
        "keep_last_checkpoints": keep_last_checkpoints,
        "keep_materialized_dataset": keep_materialized_dataset,
    }

    def apply_values(new_values):
        def combo_set(combo, value):
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo_set(finetune, new_values.get("finetune_method", "lora"))
        combo_set(decoder, new_values.get("decoder", "segformer3d"))
        combo_set(model_scale, new_values.get("model_scale", "vitb16"))
        combo_set(modality, new_values.get("modality", "ct"))
        epochs.setValue(int(new_values.get("epochs", 10)))
        batch_size.setValue(int(new_values.get("batch_size", 1)))
        grad_accum.setValue(int(new_values.get("grad_accumulation", 1)))
        lr.setText(str(new_values.get("lr", 0.001)))
        weight_decay.setText(str(new_values.get("weight_decay", 0.01)))
        img_size.setText(str(new_values.get("img_size", "224,224")))
        base_config.setText(str(new_values.get("base_config", config.get("base_config", ""))))
        model_path.setText(str(new_values.get("model_path", "")))
        mixed_precision.setChecked(bool(new_values.get("mixed_precision", False)))
        sub_volume.setChecked(bool(new_values.get("sub_volume", False)))
        sub_volume_size.setText(str(new_values.get("sub_volume_size", "32,256,256")))
        keep_last_checkpoints.setValue(int(new_values.get("keep_last_checkpoints", 2)))
        keep_materialized_dataset.setChecked(bool(new_values.get("keep_materialized_dataset", False)))

    def profile_changed(index):
        if profile_names and 0 <= index < len(profile_names):
            apply_values(_default_training_options(config, profile_names[index]))

    profile_combo.currentIndexChanged.connect(profile_changed)
    apply_values(values)

    train_form.addRow("Fine-tuning", finetune)
    train_form.addRow("Decoder", decoder)
    train_form.addRow("Pretrained scale", model_scale)
    train_form.addRow("Modality", modality)
    train_form.addRow("Epochs", epochs)
    train_form.addRow("Batch size", batch_size)
    train_form.addRow("Grad accumulation", grad_accum)
    train_form.addRow("Learning rate", lr)
    train_form.addRow("Weight decay", weight_decay)
    train_form.addRow("Image size", img_size)
    train_form.addRow("Base config", base_config)
    train_form.addRow("Model path override", model_path)
    train_form.addRow("Mixed precision", mixed_precision)
    train_form.addRow("Sub-volume", sub_volume)
    train_form.addRow("Sub-volume size", sub_volume_size)
    train_form.addRow("Keep last checkpoints", keep_last_checkpoints)
    train_form.addRow("Keep materialized dataset", keep_materialized_dataset)
    tabs.addTab(train_tab, "Training")

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    root.addWidget(buttons)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec_() != QDialog.Accepted:
        return None

    selected_cases = [str(item.text()) for item in case_list.selectedItems()]
    result = {
        "profile": str(profile_combo.currentText()),
        "sample_mode": str(sample_mode.currentText()),
        "max_samples": int(max_samples.value()),
        "min_samples": int(min_samples.value()),
        "val_fraction": float(val_fraction.value()),
        "cases": selected_cases,
        "finetune_method": str(finetune.currentText()),
        "decoder": str(decoder.currentText()),
        "model_scale": str(model_scale.currentText()),
        "modality": str(modality.currentText()),
        "epochs": int(epochs.value()),
        "batch_size": int(batch_size.value()),
        "grad_accumulation": int(grad_accum.value()),
        "lr": str(lr.text()).strip(),
        "weight_decay": str(weight_decay.text()).strip(),
        "img_size": str(img_size.text()).strip(),
        "base_config": str(base_config.text()).strip(),
        "model_path": str(model_path.text()).strip(),
        "mixed_precision": bool(mixed_precision.isChecked()),
        "sub_volume": bool(sub_volume.isChecked()),
        "sub_volume_size": str(sub_volume_size.text()).strip(),
        "keep_last_checkpoints": int(keep_last_checkpoints.value()),
        "keep_materialized_dataset": bool(keep_materialized_dataset.isChecked()),
    }
    if selected_cases:
        result["sample_mode"] = "all"
    return result


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


def _train_model(advanced=False):
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
    config = _config()
    if advanced:
        options = _advanced_training_options(config, ts_root)
        if options is None:
            return 0
    else:
        options = _default_training_options(config)
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

    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
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
        "--export-labels",
        "--run-id",
        run_id,
    ]
    _append_training_args(cmd, config, options)
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
            "training_options": options,
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )
    status_path = _status_path(ts_root, run_id)
    _mimics_log(
        logging.INFO,
        "DINOv3 few-shot training started. Organ: {0}, PID: {1}, job: {2}. Status: {3}".format(
            organ,
            process.pid,
            run_id,
            status_path,
        ),
    )
    monitor_started = _start_monitor(
        {
            "monitor_key": "train_" + run_id,
            "kind": "train",
            "deadline": time.time() + float(config.get("training_monitor_timeout_seconds", 7 * 86400)),
            "status_path": status_path,
            "last_line": "",
        },
        poll_seconds=float(config.get("training_monitor_poll_seconds", 5.0)),
    )
    if monitor_started:
        _mimics_log(
            logging.INFO,
            "DINOv3 training monitor enabled; epoch loss and validation Dice will be reported in this log.",
        )
    else:
        _mimics_log(
            logging.WARNING,
            "DINOv3 training monitor could not start in this Mimics session. Training continues in the background; use Show Status to check progress, or open the pipeline log:\n{0}".format(
                os.path.join(_workspace(ts_root), "fewshot_pipeline.log"),
            ),
        )
    mimics.dialogs.message_box(
        "Few-shot training started in the background.\n\nOrgan: {0}\nJob: {1}\n\nPipeline log (full details):\n{2}\n\nUse Show Status for progress.".format(
            organ,
            run_id,
            os.path.join(_workspace(ts_root), "fewshot_pipeline.log"),
        ),
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def _start_inference(choose_model=False):
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
    selected_model = None
    if choose_model:
        selected_model = _choose_model_manifest(ts_root, organ)
        if selected_model is None:
            return 0
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
        "--gpu-lock-timeout-seconds",
        str(float(config.get("inference_gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 3600)))),
        "--job-id",
        job_id,
    ]
    if selected_model:
        cmd.extend([
            "--model-manifest",
            selected_model["manifest_path"],
            "--model-id",
            selected_model.get("model_id", "latest") or "latest",
        ])
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
            "selected_model": selected_model,
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
        "axes": [0, 1, 2],
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
    if active_image is None:
        raise RuntimeError("No active Mimics image is available for mask import.")
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
    except Exception as exc:
        try:
            bound_image = getattr(mask, "image", None)
        except Exception:
            bound_image = None
        try:
            is_bound_to_active = bound_image == active_image
        except Exception:
            is_bound_to_active = bound_image is active_image
        if bound_image is None or not is_bound_to_active:
            raise RuntimeError("Failed to bind prediction mask to the active Mimics image: {0}".format(exc))
    return mask


def _set_mask_from_u8(mask, path, shape):
    raw = open(path, "rb").read()
    expected = int(shape[0]) * int(shape[1]) * int(shape[2])
    if len(raw) != expected:
        raise RuntimeError("Prediction byte count mismatch: {0} != {1}".format(len(raw), expected))
    def _apply():
        try:
            import numpy as np
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).astype(np.bool_)
            mask.set_voxel_buffer(pixels)
        except ImportError:
            pixels = memoryview(bytearray(raw)).cast("?", shape=list(shape))
            mask.set_voxel_buffer(pixels)
    _with_gui_updates_disabled(_apply)
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
        if monitor.get("kind") == "train":
            mimics.dialogs.message_box("Few-shot training monitor timed out. The background job may still be running; use Show Status.", title=TITLE, ui_blocking=False)
        else:
            mimics.dialogs.message_box("Few-shot inference timed out.", title=TITLE, ui_blocking=False)
        return
    status = _read_json(monitor["status_path"], {}) or {}
    if monitor.get("kind") == "train":
        line = _format_job_line(status)
        if line and line != monitor.get("last_line"):
            monitor["last_line"] = line
            _mimics_log(logging.INFO, "DINOv3 training status: {0}".format(line))
        state = status.get("status")
        if state in ("completed", "failed", "cancelled", "cancelling"):
            _stop_monitor(key)
            if state == "completed":
                message = "Few-shot training completed.\n\n{0}".format(line)
            elif state in ("cancelled", "cancelling"):
                message = "Few-shot training was cancelled.\n\n{0}".format(line)
            else:
                message = "Few-shot training failed.\n\n{0}\n\n{1}".format(
                    status.get("error", "Unknown error"),
                    line,
                )
            mimics.dialogs.message_box(message, title=TITLE, ui_blocking=False)
        return
    state = status.get("status")
    if state in ("", None, "launching", "running", "waiting_for_gpu", "waiting_for_background_mimics"):
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
    if not _ensure_pyqt5():
        if _start_win32_monitor(monitor, poll_seconds):
            return True
        return False
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except ImportError:
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
                pieces.append("best_val_dice {0:.4f}".format(float(best)))
            except Exception:
                pieces.append("best_val_dice {0}".format(best))
        metrics = progress.get("metrics") or {}
        if metrics.get("loss") is not None:
            try:
                pieces.append("train_loss {0:.4f}".format(float(metrics.get("loss"))))
            except Exception:
                pass
        if metrics.get("mean_dsc") is not None:
            try:
                pieces.append("val_dice {0:.4f}".format(float(metrics.get("mean_dsc"))))
            except Exception:
                pass
        if pieces:
            extra = " | " + ", ".join(pieces)
    if job.get("sample_count") is not None:
        extra += " | samples {0}".format(job.get("sample_count"))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        extra += " | train {0}, val {1}".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        )
    resource_wait = job.get("resource_wait") or {}
    if resource_wait:
        extra += " | " + _resource_wait_text(job)
    return "{0} | {1} | {2} | {3}{4}".format(
        job.get("job_id", "?"),
        job.get("kind", "?"),
        job.get("organ", "?"),
        _display_status(job.get("status", "?")),
        extra,
    )


def _latest_model_lines(ts_root, selected_organ=None):
    models_dir = os.path.join(_workspace(ts_root), "models")
    rows = []
    if os.path.isdir(models_dir):
        for organ_slug in sorted(os.listdir(models_dir)):
            latest = os.path.join(models_dir, organ_slug, "latest.json")
            if not os.path.isfile(latest):
                continue
            manifest = _read_json(latest, {}) or {}
            organ = manifest.get("organ", organ_slug)
            if selected_organ and _safe_slug(organ) != _safe_slug(selected_organ):
                continue
            rows.append(
                "Local model {0}: {1}, samples {2}".format(
                    organ,
                    manifest.get("model_id", "?"),
                    manifest.get("sample_count", "?"),
                )
            )
    global_payload = _read_json(_global_model_registry_path(), {}) or {}
    for row in global_payload.get("models", []) or []:
        organ = row.get("organ", row.get("organ_slug", ""))
        if selected_organ and _safe_slug(organ) != _safe_slug(selected_organ):
            continue
        manifest_path = row.get("manifest_path", "")
        if not manifest_path or not os.path.isfile(manifest_path):
            continue
        rows.append(
            "Reusable model {0}: {1}, samples {2}".format(
                organ,
                row.get("model_id", "?"),
                row.get("sample_count", "?"),
            )
        )
        if len(rows) >= 10:
            break
    return rows


def _model_candidates(ts_root, organ):
    candidates = []
    organ_slug = _safe_slug(organ)
    models_dir = os.path.join(_workspace(ts_root), "models", organ_slug)
    seen_manifests = set()
    if os.path.isdir(models_dir):
        latest_path = os.path.join(models_dir, "latest.json")
        for manifest_path in [latest_path]:
            manifest = _read_json(manifest_path, {}) or {}
            if manifest:
                seen_manifests.add(os.path.abspath(manifest_path))
                candidates.append({
                    "scope": "dataset latest",
                    "manifest_path": os.path.abspath(manifest_path),
                    "model_id": manifest.get("model_id", ""),
                    "organ": manifest.get("organ", organ),
                    "sample_count": manifest.get("sample_count", 0),
                    "created_at_epoch": manifest.get("created_at_epoch", 0.0),
                    "manifest": manifest,
                })
        try:
            for run_id in sorted(os.listdir(models_dir), reverse=True):
                manifest_path = os.path.join(models_dir, run_id, "manifest.json")
                manifest = _read_json(manifest_path, {}) or {}
                if not manifest:
                    continue
                abs_manifest = os.path.abspath(manifest_path)
                if abs_manifest in seen_manifests:
                    continue
                seen_manifests.add(abs_manifest)
                candidates.append({
                    "scope": "dataset",
                    "manifest_path": abs_manifest,
                    "model_id": manifest.get("model_id", run_id),
                    "organ": manifest.get("organ", organ),
                    "sample_count": manifest.get("sample_count", 0),
                    "created_at_epoch": manifest.get("created_at_epoch", 0.0),
                    "manifest": manifest,
                })
        except Exception:
            pass
    global_payload = _read_json(_global_model_registry_path(), {}) or {}
    for row in global_payload.get("models", []) or []:
        if row.get("organ_slug") != organ_slug:
            continue
        manifest_path = row.get("manifest_path")
        if not manifest_path or not os.path.isfile(manifest_path):
            continue
        abs_manifest = os.path.abspath(manifest_path)
        if abs_manifest in seen_manifests:
            continue
        seen_manifests.add(abs_manifest)
        candidates.append({
            "scope": "global",
            "manifest_path": abs_manifest,
            "model_id": row.get("model_id", ""),
            "organ": row.get("organ", organ),
            "sample_count": row.get("sample_count", 0),
            "created_at_epoch": row.get("created_at_epoch", 0.0),
            "manifest": _read_json(abs_manifest, {}) or {},
        })
    candidates.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    return candidates


def _choose_model_manifest(ts_root, organ):
    candidates = _model_candidates(ts_root, organ)
    if not candidates:
        mimics.dialogs.message_box(
            "No trained model was found for organ: {0}".format(organ),
            title=TITLE,
            ui_blocking=False,
        )
        return None
    if not _ensure_pyqt5():
        return candidates[0]
    try:
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem, QVBoxLayout
    except ImportError:
        return candidates[0]

    dialog = QDialog()
    dialog.setWindowTitle("Select DINOv3 Model")
    dialog.resize(760, 420)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Select a model for organ: {0}".format(organ)))
    list_widget = QListWidget()
    for candidate in candidates:
        created = candidate.get("created_at_epoch", 0.0)
        try:
            created_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(created)))
        except Exception:
            created_text = "unknown time"
        label = "{0} | {1} | samples {2} | {3}".format(
            candidate.get("scope", "?"),
            candidate.get("model_id", "?"),
            candidate.get("sample_count", "?"),
            created_text,
        )
        item = QListWidgetItem(label)
        item.setData(32, candidate)
        list_widget.addItem(item)
    list_widget.setCurrentRow(0)
    layout.addWidget(list_widget)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    if dialog.exec_() != QDialog.Accepted:
        return None
    item = list_widget.currentItem()
    return item.data(32) if item is not None else None


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


def main(action=None):
    if action is None:
        action = mimics.dialogs.question_box(
            message=(
                "Select one organ Mask before training or prediction.\n\n"
                "Training uses saved .mcs files and runs fully in the background."
            ),
            buttons=";".join([BUTTON_TRAIN, BUTTON_TRAIN_ADVANCED, BUTTON_PREDICT, BUTTON_PREDICT_MODEL, BUTTON_STATUS, BUTTON_STOP, BUTTON_CANCEL]),
            title=TITLE,
            ui_blocking=True,
        )
    if action == BUTTON_TRAIN:
        return _train_model(False)
    if action == BUTTON_TRAIN_ADVANCED:
        return _train_model(True)
    if action == BUTTON_PREDICT:
        return _start_inference(False)
    if action == BUTTON_PREDICT_MODEL:
        return _start_inference(True)
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
