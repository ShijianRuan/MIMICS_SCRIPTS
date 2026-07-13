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
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import mimics

import runtime_common


TITLE = "DINOv3 Few-Shot"
BUTTON_TRAIN = "Train Quick Model"
BUTTON_TRAIN_ADVANCED = "Train Advanced Setup..."
BUTTON_TRAIN_MODEL = "Train Model..."
BUTTON_PREDICT = "Predict Current Case (Latest Model)"
BUTTON_PREDICT_MODEL = "Predict Current Case (Choose Model)..."
BUTTON_STATUS = "Show Status"
BUTTON_STOP = "Stop Running Job"
BUTTON_CANCEL = "Cancel"
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"

_MONITORS = {}
_QT_CHECK_DONE = False
_QT_CHECK_RESULT = False


_write_json_atomic = runtime_common.write_json_atomic
_write_text_atomic = runtime_common.write_text_atomic
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
    return os.path.join(_project_root(), ".mimics_runtime", "fewshot_mimics_state.json")


def _migrate_old_settings():
    """Move legacy state file from project root into .mimics_runtime/."""
    old_path = os.path.join(_project_root(), ".fewshot_mimics_state.json")
    new_path = _settings_path()
    if os.path.isfile(old_path) and not os.path.isfile(new_path):
        try:
            new_dir = os.path.dirname(new_path)
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)
            os.rename(old_path, new_path)
        except Exception:
            pass


def _load_settings():
    _migrate_old_settings()
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


def _load_mimics_io_config():
    merged = {}
    for name in ("mimics_io_config.json", "nninteractive_config.json"):
        path = os.path.join(_project_root(), name)
        loaded = _read_json(path, {}) or {}
        if isinstance(loaded, dict):
            merged.update(loaded)
    return merged


def _resolve_mimics_output_dir(ts_root):
    default_dir = os.path.abspath(os.path.join(ts_root, "mcs_output"))
    config = _load_mimics_io_config()
    configured = config.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if os.path.isabs(configured):
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(ts_root, configured))


def _parse_axes_value(value):
    if value is None or value == "":
        return [0, 1, 2]
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    axes = [int(part) for part in value]
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("mimics buffer axes must be a permutation of 0,1,2: {0}".format(value))
    return axes


def _parse_flips_value(value):
    if value is None or value == "":
        return [False, False, False]
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(value) != 3:
        raise ValueError("mimics buffer flips must contain three values: {0}".format(value))
    return [
        bool(part) if isinstance(part, bool) else str(part).strip().lower() in ("1", "true", "yes", "y", "on")
        for part in value
    ]


def _buffer_mapping_from_config(config):
    config = config or {}
    axes_value = config.get("mimics_buffer_axes", config.get("platform_to_mimics_axes", config.get("axes", [0, 1, 2])))
    flips_value = config.get("mimics_buffer_flips", config.get("platform_to_mimics_flips", config.get("flips", [False, False, False])))
    return _parse_axes_value(axes_value), _parse_flips_value(flips_value)


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


def _project_python_candidates():
    root = _project_root()
    return [
        os.path.join(root, "nninteractive_env", "python.exe"),
        os.path.join(root, "nninteractive_env", "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "python", "python.exe"),
        os.path.join(root, "nninteractive_env", "bin", "python3"),
        os.path.join(root, "nninteractive_env", "bin", "python"),
    ]


def _append_python_candidate(candidates, value, base=None):
    if not value:
        return
    if value in ("python", "python3"):
        return
    if os.path.isabs(value):
        candidates.append(value)
    else:
        candidates.append(os.path.abspath(os.path.join(base or _project_root(), value)))


def _fewshot_python(config, dinov3_root):
    candidates = list(_project_python_candidates())
    _append_python_candidate(candidates, os.environ.get("MIMICS_FEWSHOT_PYTHON", ""))
    _append_python_candidate(candidates, config.get("python", ""))
    candidates.extend([
        os.path.join(dinov3_root, ".venv", "Scripts", "python.exe"),
        os.path.join(dinov3_root, ".venv", "bin", "python"),
        os.path.join(dinov3_root, "venv", "Scripts", "python.exe"),
        os.path.join(dinov3_root, "venv", "bin", "python"),
    ])
    env_root = os.path.abspath(os.path.join(_project_root(), "nninteractive_env"))
    try:
        current = os.path.abspath(sys.executable)
    except Exception:
        current = ""
    if current and current.startswith(env_root + os.sep):
        candidates.append(sys.executable)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run Setup Environment or setup_offline.bat before using DINOv3."
    )


def _pipeline_script():
    return os.path.join(_project_root(), "tools", "fewshot_pipeline.py")


def _training_setup_ui_script():
    return os.path.join(_project_root(), "tools", "fewshot_training_setup_ui.py")


def _status_viewer_script():
    return os.path.join(_project_root(), "tools", "fewshot_status_viewer.py")


def _model_chooser_script():
    return os.path.join(_project_root(), "tools", "fewshot_model_chooser.py")


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
    """Try to make PyQt5/PySide importable without blocking the Mimics GUI."""
    global _QT_CHECK_DONE
    global _QT_CHECK_RESULT
    if _QT_CHECK_DONE:
        return _QT_CHECK_RESULT
    try:
        import PyQt5  # noqa: F401
        _QT_CHECK_DONE = True
        _QT_CHECK_RESULT = True
        return True
    except ImportError:
        pass

    for env_var in ("MIMICS_QT_PYTHONPATH", "MIMICS_PYQT_PATH"):
        value = os.environ.get(env_var, "")
        if not value:
            continue
        for path in value.split(os.pathsep):
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            import PyQt5  # noqa: F401
            _QT_CHECK_DONE = True
            _QT_CHECK_RESULT = True
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
            _QT_CHECK_DONE = True
            _QT_CHECK_RESULT = True
            return True
        except ImportError:
            pass

    _QT_CHECK_DONE = True
    _QT_CHECK_RESULT = False
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


def _metadata_get(obj, name, default=""):
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


def _active_image_shape(image):
    try:
        dims = getattr(image, "logical_dimensions", None)
        if dims is not None:
            shape = [int(dims[0]), int(dims[1]), int(dims[2])]
            if all(value > 0 for value in shape):
                return shape
    except Exception:
        pass
    try:
        view = image.get_voxel_buffer()
        shape = [int(value) for value in view.shape]
        if all(value > 0 for value in shape):
            return shape
    except Exception:
        pass
    return None


def _point_values(point):
    if point is None:
        raise RuntimeError("Mimics returned an empty voxel center.")
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [float(getattr(point, names[0])), float(getattr(point, names[1])), float(getattr(point, names[2]))]
        except Exception:
            pass
    return [float(point[0]), float(point[1]), float(point[2])]


def _voxel_center(image, index):
    getter = getattr(image, "get_voxel_center", None)
    if not callable(getter):
        raise RuntimeError("Mimics image does not expose get_voxel_center().")
    values = [int(value) for value in index]
    try:
        return _point_values(getter(values))
    except TypeError:
        pass
    try:
        return _point_values(getter(tuple(values)))
    except TypeError:
        return _point_values(getter(values[0], values[1], values[2]))


def _active_live_grid_payload():
    """Measure the current open Mimics grid instead of trusting stale metadata."""
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    if image is None:
        return None
    shape = _active_image_shape(image)
    if not shape:
        return None
    try:
        origin_lps = _voxel_center(image, [0, 0, 0])
        origin = [-origin_lps[0], -origin_lps[1], origin_lps[2]]
        metadata_matrix = _parse_matrix_metadata(
            _metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
        )
        matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
        for axis in range(3):
            if int(shape[axis]) <= 1:
                if metadata_matrix is None:
                    return None
                for row in range(3):
                    matrix[row][axis] = float(metadata_matrix[row][axis])
            else:
                index = [0, 0, 0]
                index[axis] = 1
                point_lps = _voxel_center(image, index)
                point = [-point_lps[0], -point_lps[1], point_lps[2]]
                for row in range(3):
                    matrix[row][axis] = point[row] - origin[row]
        for row in range(3):
            matrix[row][3] = origin[row]
        matrix[3] = [0.0, 0.0, 0.0, 1.0]
        return {"target_shape": shape, "target_voxel_to_ras_matrix": matrix, "source": "live_mimics"}
    except Exception:
        return None


def _matrix_close(left, right, tolerance=1e-4):
    try:
        if len(left) != 4 or len(right) != 4:
            return False
        for row in range(4):
            if len(left[row]) != 4 or len(right[row]) != 4:
                return False
            for column in range(4):
                if abs(float(left[row][column]) - float(right[row][column])) > float(tolerance):
                    return False
        return True
    except Exception:
        return False


def _same_path(left, right):
    try:
        return os.path.normcase(os.path.abspath(str(left or ""))) == os.path.normcase(os.path.abspath(str(right or "")))
    except Exception:
        return False


def _monitor_target_is_open(monitor):
    if _infer_case_id(monitor.get("ts_root")) != monitor.get("case_id"):
        return False, "Open the source case {0} to apply this result.".format(monitor.get("case_id"))
    launch_project = monitor.get("launch_project_path")
    if launch_project and not _same_path(_current_project_path(), launch_project):
        return False, "Open the original project to apply this result: {0}".format(launch_project)
    expected = monitor.get("target_grid") or {}
    live = _active_live_grid_payload()
    if not live:
        return False, "The live Mimics image grid could not be measured."
    if list(live.get("target_shape") or []) != list(expected.get("target_shape") or []):
        return False, "The active image shape differs from the inference target."
    if not _matrix_close(
        live.get("target_voxel_to_ras_matrix") or [],
        expected.get("target_voxel_to_ras_matrix") or [],
    ):
        return False, "The active image physical grid differs from the inference target."
    return True, ""


def _active_mimics_grid_payload():
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    if image is None:
        return None
    shape = _active_image_shape(image)
    matrix_text = _metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
    if not shape or not matrix_text:
        return None
    try:
        matrix = json.loads(matrix_text)
        if len(matrix) != 4:
            return None
        for row in matrix:
            if len(row) != 4:
                return None
        return {"target_shape": shape, "target_voxel_to_ras_matrix": matrix}
    except Exception:
        return None


def _parse_shape_metadata(value):
    try:
        shape = json.loads(value)
    except Exception:
        return None
    try:
        shape = [int(shape[0]), int(shape[1]), int(shape[2])]
    except Exception:
        return None
    if all(item > 0 for item in shape):
        return shape
    return None


def _parse_matrix_metadata(value):
    try:
        matrix = json.loads(value)
    except Exception:
        return None
    try:
        if len(matrix) != 4:
            return None
        rows = []
        for row in matrix:
            if len(row) != 4:
                return None
            rows.append([float(item) for item in row])
        return rows
    except Exception:
        return None


def _active_source_geometry_payload():
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    if image is None:
        return None
    shape = _parse_shape_metadata(_metadata_get(image, SOURCE_IMAGE_SHAPE_METADATA, ""))
    matrix = _parse_matrix_metadata(_metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, ""))
    if not shape or not matrix:
        return None
    return {
        "source_image_path": _metadata_get(image, SOURCE_IMAGE_PATH_METADATA, ""),
        "source_shape": shape,
        "source_voxel_to_ras_matrix": matrix,
    }


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
    mcs_dir = _resolve_mimics_output_dir(ts_root)
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
    values.setdefault("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))
    values.setdefault("strategy", config.get("default_strategy", "adaptive"))
    values.setdefault("epochs", config.get("default_epochs", 20))
    values["batch_size"] = 1
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
    values.setdefault("export_labels_before_training", config.get("default_export_labels_before_training", True))
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault(
        "background_mimics_lock_timeout_seconds",
        config.get("background_mimics_lock_timeout_seconds", 1800),
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
        "--strategy",
        str(options.get("strategy", config.get("default_strategy", "adaptive"))),
        "--strategy-options-json",
        str(options.get("strategy_options_json", "{}")),
        "--base-config",
        str(options.get("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))),
        "--epochs",
        str(int(options.get("epochs", config.get("default_epochs", 20)))),
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
        str(float(options.get("background_mimics_lock_timeout_seconds", config.get("background_mimics_lock_timeout_seconds", 1800)))),
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
    settings = _load_settings()
    candidates = []
    last_root = settings.get("last_dataset_root", "")
    if last_root:
        candidates.append(last_root)
    parent = os.path.dirname(project_dir)
    if parent:
        candidates.append(parent)
    for dataset_root in candidates:
        if not dataset_root or not os.path.isdir(dataset_root):
            continue
        output_dir = _resolve_mimics_output_dir(dataset_root)
        if os.path.normcase(os.path.abspath(project_dir)) == os.path.normcase(os.path.abspath(output_dir)):
            return os.path.abspath(dataset_root)
    if os.path.basename(project_dir).lower() == "mcs_output":
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


def _count_available_labels(ts_root, organ):
    """Count how many cases under ts_root have saved .mcs files with labels."""
    mcs_dir = _resolve_mimics_output_dir(ts_root)
    if not os.path.isdir(mcs_dir):
        return 0
    count = 0
    for fname in sorted(os.listdir(mcs_dir)):
        if not fname.lower().endswith(".mcs"):
            continue
        dir_name = fname[:-4]
        seg_dir = os.path.join(ts_root, dir_name, "segmentations")
        if not os.path.isdir(seg_dir):
            continue
        for seg_name in os.listdir(seg_dir):
            stem = seg_name.replace(".nii.gz", "").replace(".nii", "")
            if stem.lower() == organ.lower() or stem.lower() == _safe_slug(organ).lower():
                count += 1
                break
    return count


def _infer_case_id(ts_root):
    """Match the open .mcs project to a dataset case.

    First tries the canonical path <ts_root>/mcs_output/<case>.mcs.
    If that fails, also checks <ts_root>/<case>.mcs and case-directory
    pattern so the annotator is not forced to use a single layout.
    """
    project_path = _current_project_path()
    if not project_path:
        return None
    name = os.path.basename(project_path)
    if name.lower().endswith(".mcs"):
        case_id = name[:-4]
    else:
        case_id = name

    # Canonical: configured .mcs output directory.
    output_dir = os.path.abspath(_resolve_mimics_output_dir(ts_root))
    project_dir = os.path.abspath(os.path.dirname(project_path))
    if os.path.normcase(project_dir) == os.path.normcase(output_dir):
        return case_id

    # Fallback 1: <ts_root>/<case>.mcs
    alt_dir = os.path.abspath(ts_root)
    if os.path.normcase(project_dir) == os.path.normcase(alt_dir):
        return case_id

    # Fallback 2: <ts_root>/<case_dir>/<case>.mcs (case dir matches case id)
    for item in sorted(os.listdir(ts_root)):
        item_path = os.path.join(ts_root, item)
        if not os.path.isdir(item_path):
            continue
        if item in ("mcs_output", "segmentations", "fewshot_models", "labels"):
            continue
        candidate_dir = os.path.abspath(os.path.join(item_path, "mcs_output"))
        if os.path.normcase(project_dir) == os.path.normcase(candidate_dir):
            return case_id
        if os.path.normcase(project_dir) == os.path.normcase(item_path):
            return item

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
        "configuring",
        "selecting_model",
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
        "configuring": "Configuring training",
        "selecting_model": "Selecting model",
        "training_started": "Training started",
        "closed": "Closed",
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


def _display_kind(value):
    labels = {
        "train": "training",
        "infer": "prediction",
        "model_choice": "model selection",
        "train_setup": "training setup",
    }
    return labels.get(str(value or ""), str(value or "?").replace("_", " "))


def _resource_wait_text(job):
    resource_wait = job.get("resource_wait") or {}
    if not resource_wait:
        return ""
    resource = _display_resource(resource_wait.get("resource", "resource"))
    holder = resource_wait.get("owner", "unknown")
    pid = resource_wait.get("pid", "")
    if pid:
        text = "Waiting for {0}: {1} (PID {2})".format(resource, holder, pid)
    else:
        text = "Waiting for {0}".format(resource)
    action = str(resource_wait.get("user_action") or "").strip()
    return text + ("\n" + action if action else "")


def _guard_no_active_job(ts_root, requested_kind="train"):
    path, job = _latest_active_job(ts_root)
    if not job:
        return True
    if requested_kind == "infer":
        active_kind = str(job.get("kind", ""))
        if active_kind in ("train", "train_setup"):
            _mimics_log(
                logging.INFO,
                "DINOv3 inference will start while {0} job {1} is {2}; GPU access is still serialized by the global lock.".format(
                    _display_kind(active_kind),
                    job.get("job_id", os.path.basename(path or "")),
                    _display_status(job.get("status", "?")),
                ),
            )
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

    values["batch_size"] = 1

    return values


def _profile_training_options(config):
    profiles = config.get("training_profiles") or {}
    profile_names = sorted(profiles.keys()) if isinstance(profiles, dict) else []
    default_name = config.get("default_training_profile") or config.get("default_profile") or ""

    if profile_names:
        lines = [
            "Advanced Qt widgets are not available in this Mimics Python session.",
            "",
            "Choose a training profile from fewshot_config.json,",
            "or pick 'Configure manually' to set parameters interactively.",
            "For the full Qt dialog, set MIMICS_QT_PYTHONPATH or MIMICS_PYQT_PATH",
            "to a Python-3.5-compatible PyQt5/PySide site-packages directory.",
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
            "save and reuse your settings. Do not install a modern PyQt5\n"
            "wheel into Mimics Python 3.5; point MIMICS_QT_PYTHONPATH\n"
            "to a compatible binding if a Qt dialog is required."
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
            "DINOv3 Advanced Qt dialog is not available in this Mimics Python session. Using fewshot_config.json profiles/native parameter dialogs. For a Qt dialog, set MIMICS_QT_PYTHONPATH or MIMICS_PYQT_PATH to a Python-3.5-compatible PyQt5/PySide path.",
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
    batch_size.setRange(1, 1)
    batch_size.setToolTip(
        "Fixed at 1 because Mimics cases can have different z-depth. Use Grad accumulation for a larger effective batch."
    )
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
    export_labels_before_training = QCheckBox()

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
        "export_labels_before_training": export_labels_before_training,
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
        export_labels_before_training.setChecked(bool(new_values.get("export_labels_before_training", True)))

    def profile_changed(index):
        if profile_names and 0 <= index < len(profile_names):
            apply_values(_default_training_options(config, profile_names[index]))

    profile_combo.currentIndexChanged.connect(profile_changed)
    sub_volume.stateChanged.connect(lambda _state: sub_volume_size.setEnabled(bool(sub_volume.isChecked())))
    apply_values(values)
    sub_volume_size.setEnabled(bool(sub_volume.isChecked()))

    train_form.addRow("Fine-tuning", finetune)
    train_form.addRow("Decoder", decoder)
    train_form.addRow("Pretrained scale", model_scale)
    train_form.addRow("Epochs", epochs)
    train_form.addRow("Batch size (fixed)", batch_size)
    train_form.addRow("Grad accumulation", grad_accum)
    train_form.addRow("Learning rate", lr)
    train_form.addRow("Weight decay", weight_decay)
    train_form.addRow("Image size", img_size)
    train_form.addRow("Mixed precision", mixed_precision)
    train_form.addRow("Sub-volume", sub_volume)
    train_form.addRow("Sub-volume size", sub_volume_size)
    train_form.addRow("Keep last checkpoints", keep_last_checkpoints)
    train_form.addRow("Refresh labels from saved .mcs", export_labels_before_training)
    train_form.addRow(
        "Backend config",
        QLabel("Base config, modality, and custom weight path are controlled by fewshot_config.json."),
    )
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
        "export_labels_before_training": bool(export_labels_before_training.isChecked()),
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


def _launch_gui_process(cmd, cwd=None):
    # Launch GUI apps without CREATE_NO_WINDOW so Tk/PySide windows are visible.
    # On Windows, prefer pythonw.exe (same environment, no console flash).
    launch_cmd = list(cmd)
    if os.name == "nt" and launch_cmd:
        exe = os.path.abspath(str(launch_cmd[0]))
        if os.path.basename(exe).lower() == "python.exe":
            pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            if os.path.isfile(pythonw):
                launch_cmd[0] = pythonw
    return subprocess.Popen(
        launch_cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_background_env(),
    )


def _launch_external_advanced_training(config, organ, ts_root):
    """Open the advanced training setup outside the Mimics process.

    This function must stay lightweight: it writes small JSON files, starts the
    external UI, and returns.  Dataset export and training are launched later by
    the external UI.
    """
    script = _training_setup_ui_script()
    if not os.path.isfile(script):
        raise RuntimeError("External training setup UI was not found: {0}".format(script))
    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
    setup_id = "setup_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    workspace = _workspace(ts_root)
    jobs_dir = os.path.join(workspace, "jobs")
    if not os.path.isdir(jobs_dir):
        os.makedirs(jobs_dir)
    status_path = _status_path(ts_root, setup_id)
    context_path = os.path.join(jobs_dir, setup_id + "_context.json")
    context = {
        "schema_version": "mimics_fewshot_setup_context.v1",
        "setup_id": setup_id,
        "organ": organ,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "mcs_output_dir": _resolve_mimics_output_dir(ts_root),
        "project_root": _project_root(),
        "pipeline_script": _pipeline_script(),
        "dinov3_root": dinov3_root,
        "python_exe": python_exe,
        "mimics_exe": _find_mimics_exe() or "",
        "config": config,
        "case_ids": _case_ids_from_dataset(ts_root),
        "current_project_path": _current_project_path() or "",
        "current_case_id": _infer_case_id(ts_root) or "",
        "setup_status_path": status_path,
        "created_at_epoch": time.time(),
    }
    status = {
        "schema_version": "mimics_fewshot_setup.v1",
        "job_id": setup_id,
        "kind": "train_setup",
        "status": "configuring",
        "organ": organ,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "context_path": context_path,
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }
    _write_json_atomic(status_path, status)
    _write_json_atomic(context_path, context)
    command = [python_exe, script, "--context", context_path]
    try:
        process = _launch_gui_process(command, cwd=_project_root())
    except Exception as exc:
        failed = _read_json(status_path, status) or status
        failed["status"] = "failed"
        failed["error"] = (
            "Could not start the external Advanced setup process. "
            "Python: {0}. Script: {1}. Context: {2}. Error: {3}".format(
                python_exe,
                script,
                context_path,
                exc,
            )
        )
        failed["updated_at_epoch"] = time.time()
        try:
            _write_json_atomic(status_path, failed)
        except Exception:
            pass
        raise RuntimeError(failed["error"])
    current_status = _read_json(status_path, status) or status
    current_status["controller_pid"] = process.pid
    current_status["updated_at_epoch"] = time.time()
    _write_json_atomic(status_path, current_status)

    # Detect immediate launcher failures and surface them to the user instead of
    # silently returning while the external window never appears.
    for _ in range(8):
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is not None:
        failed = _read_json(status_path, current_status) or current_status
        if failed.get("status") != "failed":
            failed["status"] = "failed"
            failed["error"] = (
                "Could not open the external Advanced setup window. "
                "The process exited immediately."
            )
            failed["updated_at_epoch"] = time.time()
            try:
                _write_json_atomic(status_path, failed)
            except Exception:
                pass
        mimics.dialogs.message_box(
            "Could not open the external Advanced setup UI.\n\n{0}".format(
                failed.get("error", "Unknown startup error")
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1

    monitor_started = False
    try:
        monitor_started = _start_monitor(
            {
                "monitor_key": "setup_" + setup_id,
                "kind": "train_setup",
                "deadline": time.time() + float(config.get("training_setup_timeout_seconds", 12 * 60 * 60)),
                "status_path": status_path,
                "controller_pid": process.pid,
                "last_line": "",
            },
            poll_seconds=float(config.get("training_monitor_poll_seconds", 5.0)),
        )
    except Exception as exc:
        _mimics_log(
            logging.WARNING,
            "DINOv3 setup monitor could not start: {0}".format(exc),
        )
    _mimics_log(
        logging.INFO,
        "DINOv3 advanced training setup opened in an external process. Organ: {0}, PID: {1}, setup: {2}".format(
            organ,
            process.pid,
            setup_id,
        ),
    )
    if not monitor_started:
        _mimics_log(
            logging.WARNING,
            "DINOv3 setup monitor could not start in this Mimics session. Use Show Status after starting training from the external setup window.",
        )
    return 0


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
        _mimics_log(logging.INFO, "DINOv3 training cancelled: no dataset folder was selected.")
        return 1
    if not _guard_no_active_job(ts_root, requested_kind="train"):
        return 1
    config = _config()
    if advanced:
        mode = str(config.get("advanced_ui_mode", "external")).strip().lower()
        if mode != "internal":
            try:
                return _launch_external_advanced_training(config, organ, ts_root)
            except Exception as exc:
                _mimics_log(
                    logging.ERROR,
                    "DINOv3 external advanced setup could not start: {0}".format(exc),
                )
                if not bool(config.get("advanced_ui_fallback_to_internal", True)):
                    mimics.dialogs.message_box(
                        "Could not open the external Advanced setup UI.\n\n{0}".format(exc),
                        title=TITLE,
                        ui_blocking=False,
                    )
                    return 1
                _mimics_log(
                    logging.WARNING,
                    "Falling back to Mimics internal Advanced dialogs.",
                )
        options = _advanced_training_options(config, ts_root)
        if options is None:
            return 0
    else:
        options = _default_training_options(config)

    # Pre-flight: count how many saved .mcs files have labels for this organ
    label_count = _count_available_labels(ts_root, organ)
    min_samples = int(options.get("min_samples", config.get("default_min_samples", 1)))
    label_info = ""
    if label_count < min_samples:
        msg = (
            "Training needs at least {0} saved .mcs with annotated labels for \"{1}\", "
            "but only {2} were found.\n\n"
            "Open cases from {3}, annotate the \"{1}\" mask, save them, and retry."
        ).format(min_samples, organ, label_count,
                 _resolve_mimics_output_dir(ts_root))
        mimics.dialogs.message_box(msg, title=TITLE, ui_blocking=False)
        return 1
    if label_count < 5:
        label_info = "\n\n{0} labeled case(s) found for \"{1}\" (minimum {2}).".format(
            label_count, organ, min_samples)

    answer = mimics.dialogs.question_box(
        message=(
            "Training uses saved .mcs files.\n\n"
            "Save the current project before starting so the latest manual edits "
            "are included in the exported labels." + label_info
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
        "--run-id",
        run_id,
    ]
    if bool(options.get("export_labels_before_training", True)):
        cmd.append("--export-labels")
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
    return 0


def _launch_inference_job(config, ts_root, case_id, organ, selected_model=None):
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
        "--gpu-lock-timeout-seconds",
        str(float(config.get("inference_gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 3600)))),
        "--job-id",
        job_id,
    ]
    source_geometry = _active_source_geometry_payload()
    if source_geometry:
        cmd.extend([
            "--expected-source-shape",
            json.dumps(source_geometry["source_shape"]),
            "--expected-source-voxel-to-ras-matrix",
            json.dumps(source_geometry["source_voxel_to_ras_matrix"]),
        ])
        source_image_path = str(source_geometry.get("source_image_path", "") or "").strip()
        if source_image_path:
            cmd.extend([
                "--expected-source-image-path",
                source_image_path,
            ])
    if selected_model:
        cmd.extend([
            "--model-manifest",
            selected_model["manifest_path"],
            "--model-id",
            selected_model.get("model_id", "latest") or "latest",
        ])
    target_grid = _active_live_grid_payload()
    if not target_grid:
        raise RuntimeError(
            "Could not measure the active Mimics image grid. Inference was not started because the result could not be applied safely."
        )
    launch_project_path = _current_project_path() or ""
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
            "source_geometry_expected": source_geometry,
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
        "launch_project_path": launch_project_path,
        "target_grid": target_grid,
        "waiting_to_apply_logged": False,
    }
    _start_monitor(monitor)
    _mimics_log(
        logging.INFO,
        "DINOv3 few-shot inference started. Organ: {0}, case: {1}, model: {2}, PID: {3}.".format(
            organ,
            case_id,
            (selected_model or {}).get("model_id", "latest"),
            process.pid,
        ),
    )
    return 0


def _launch_external_model_chooser(config, ts_root, case_id, organ):
    candidates = _model_candidates(ts_root, organ)
    if not candidates:
        mimics.dialogs.message_box(
            "No trained model was found for organ: {0}".format(organ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    if len(candidates) == 1:
        return _launch_inference_job(config, ts_root, case_id, organ, candidates[0])

    script = _model_chooser_script()
    if not os.path.isfile(script):
        raise RuntimeError("External model chooser was not found: {0}".format(script))
    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
    choice_id = "choose_model_{0}_{1}_{2}".format(_safe_slug(case_id), _safe_slug(organ), uuid.uuid4().hex[:8])
    workspace = _workspace(ts_root)
    jobs_dir = os.path.join(workspace, "jobs")
    if not os.path.isdir(jobs_dir):
        os.makedirs(jobs_dir)
    status_path = _status_path(ts_root, choice_id)
    context_path = os.path.join(jobs_dir, choice_id + "_context.json")
    status = {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": choice_id,
        "kind": "model_choice",
        "status": "selecting_model",
        "organ": organ,
        "case_id": case_id,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }
    context = {
        "schema_version": "mimics_fewshot_model_choice_context.v1",
        "status_path": status_path,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "organ": organ,
        "case_id": case_id,
        "candidates": candidates,
        "created_at_epoch": time.time(),
    }
    _write_json_atomic(status_path, status)
    _write_json_atomic(context_path, context)
    try:
        process = _launch_gui_process([python_exe, script, "--context", context_path], cwd=_project_root())
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = "Could not start the external model chooser. Python: {0}. Script: {1}. Error: {2}".format(
            python_exe,
            script,
            exc,
        )
        status["updated_at_epoch"] = time.time()
        _write_json_atomic(status_path, status)
        raise RuntimeError(status["error"])
    current = _read_json(status_path, status) or status
    current["controller_pid"] = process.pid
    current["context_path"] = context_path
    current["updated_at_epoch"] = time.time()
    _write_json_atomic(status_path, current)
    for _ in range(8):
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is not None:
        failed = _read_json(status_path, current) or current
        if failed.get("status") not in ("selected", "cancelled"):
            failed["status"] = "failed"
            failed["error"] = "The external model chooser exited immediately."
            failed["updated_at_epoch"] = time.time()
            _write_json_atomic(status_path, failed)
            mimics.dialogs.message_box(
                "Could not open the external DINOv3 model chooser.\n\n{0}".format(failed.get("error", "")),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    _start_monitor(
        {
            "monitor_key": choice_id,
            "kind": "model_choice",
            "status_path": status_path,
            "ts_root": ts_root,
            "case_id": case_id,
            "organ": organ,
            "deadline": time.time() + float(config.get("model_choice_timeout_seconds", 30 * 60)),
            "last_line": "",
        },
        poll_seconds=float(config.get("training_monitor_poll_seconds", 5.0)),
    )
    _mimics_log(
        logging.INFO,
        "DINOv3 model chooser opened in an external PySide6 process. Organ: {0}, case: {1}, PID: {2}.".format(
            organ,
            case_id,
            process.pid,
        ),
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
        _mimics_log(logging.INFO, "DINOv3 prediction cancelled: no dataset folder was selected.")
        return 1
    if not _guard_no_active_job(ts_root, requested_kind="infer"):
        return 1
    case_id = _infer_case_id(ts_root)
    if not case_id:
        mimics.dialogs.message_box(
            "Could not infer the current case.\n\nOpen a saved project from:\n{0}".format(
                os.path.join(_resolve_mimics_output_dir(ts_root), "<case>.mcs")
            ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1

    config = _config()
    if choose_model:
        try:
            return _launch_external_model_chooser(config, ts_root, case_id, organ)
        except Exception as exc:
            _mimics_log(logging.ERROR, "DINOv3 external model chooser could not start: {0}".format(exc))
            mimics.dialogs.message_box(
                "Could not open the external DINOv3 model chooser.\n\n{0}".format(exc),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    selected_model, reason = _dataset_latest_model_candidate(ts_root, organ)
    if not selected_model:
        mimics.dialogs.message_box(
            "{0}\n\nUse Predict Current Case (Choose Model) to select another valid model, or train a new model for this organ.".format(reason),
            title=TITLE,
            ui_blocking=False,
        )
        _mimics_log(logging.WARNING, "DINOv3 latest-model prediction was not started: {0}".format(reason))
        return 1
    return _launch_inference_job(config, ts_root, case_id, organ, selected_model)


def _launch_bridge_mask_to_buffer(monitor, status):
    job_dir = monitor["bridge_job_dir"]
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)
    output_path = os.path.join(job_dir, "prediction.u8")
    axes, flips = _buffer_mapping_from_config(_config())
    params = {
        "action": "mask_to_buffer",
        "image_path": status["image_path"],
        "mask_path": status["output_path"],
        "output_path": output_path,
        "axes": axes,
        "flips": flips,
    }
    grid_payload = monitor.get("target_grid")
    if grid_payload:
        params.update(grid_payload)
        _mimics_log(
            logging.INFO,
            "DINOv3 prediction conversion will use the active Mimics image grid (shape={0}).".format(
                grid_payload.get("target_shape")
            ),
        )
    else:
        _mimics_log(
            logging.WARNING,
            "DINOv3 prediction conversion has no verified launch-time Mimics grid. The result will not be applied.",
        )
        raise RuntimeError("No verified launch-time Mimics grid was recorded for this inference job.")
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


def _monitor_setup_tick(monitor, status):
    key = monitor.get("monitor_key")
    state = status.get("status")
    line = _format_job_line(status)
    if line and line != monitor.get("last_line"):
        monitor["last_line"] = line
        _mimics_log(logging.INFO, "DINOv3 advanced setup status: {0}".format(line))
    if state == "training_started":
        training_status_path = status.get("training_status_path")
        training_job_id = status.get("training_job_id", "?")
        if training_status_path and os.path.isfile(training_status_path):
            monitor["kind"] = "train"
            monitor["status_path"] = training_status_path
            monitor["last_line"] = ""
            _mimics_log(
                logging.INFO,
                "DINOv3 training started from external Advanced setup. Job: {0}. Status: {1}".format(
                    training_job_id,
                    training_status_path,
                ),
            )
            return
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "Advanced setup reported training started, but the training status file was not found.\n\n"
            "{0}".format(training_status_path or "(missing path)"),
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state in ("closed", "cancelled"):
        _stop_monitor(key)
        _mimics_log(logging.INFO, "DINOv3 advanced setup closed before launching training.")
        return
    if state == "failed":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "DINOv3 Advanced setup failed.\n\n{0}".format(status.get("error", "Unknown error")),
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state == "configuring":
        pid = status.get("controller_pid") or monitor.get("controller_pid")
        if pid and not _process_exists(pid):
            status["status"] = "closed"
            status["updated_at_epoch"] = time.time()
            try:
                _write_json_atomic(monitor["status_path"], status)
            except Exception:
                pass
            _stop_monitor(key)
            _mimics_log(logging.INFO, "DINOv3 advanced setup process ended before launching training.")
        return


def _monitor_model_choice_tick(monitor, status):
    key = monitor.get("monitor_key")
    state = status.get("status")
    if state == "selected":
        selected_model = status.get("selected_model")
        _stop_monitor(key)
        if not selected_model:
            mimics.dialogs.message_box(
                "DINOv3 model chooser did not return a model.",
                title=TITLE,
                ui_blocking=False,
            )
            return
        config = _config()
        _launch_inference_job(
            config,
            monitor.get("ts_root"),
            monitor.get("case_id"),
            monitor.get("organ"),
            selected_model,
        )
        return
    if state in ("cancelled", "closed"):
        _stop_monitor(key)
        _mimics_log(logging.INFO, "DINOv3 model selection cancelled.")
        return
    if state == "failed":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "DINOv3 model chooser failed.\n\n{0}".format(status.get("error", "Unknown error")),
            title=TITLE,
            ui_blocking=False,
        )
        return
    pid = status.get("controller_pid") or monitor.get("controller_pid")
    if pid and state in ("selecting_model", "configuring") and not _process_exists(pid):
        status["status"] = "closed"
        status["updated_at_epoch"] = time.time()
        try:
            _write_json_atomic(monitor["status_path"], status)
        except Exception:
            pass
        _stop_monitor(key)
        _mimics_log(logging.INFO, "DINOv3 model chooser closed before selecting a model.")


def _monitor_tick(monitor):
    key = monitor.get("monitor_key")
    if time.time() > monitor.get("deadline", 0):
        _stop_monitor(key)
        if monitor.get("kind") == "train":
            mimics.dialogs.message_box("Few-shot training monitor timed out. The background job may still be running; use Show Status.", title=TITLE, ui_blocking=False)
        elif monitor.get("kind") == "train_setup":
            mimics.dialogs.message_box("Few-shot advanced setup monitor timed out. The setup window or background job may still be running; use Show Status.", title=TITLE, ui_blocking=False)
        elif monitor.get("kind") == "model_choice":
            mimics.dialogs.message_box("Few-shot model selection timed out. Open Predict Current Case (Choose Model) again if needed.", title=TITLE, ui_blocking=False)
        else:
            mimics.dialogs.message_box("Few-shot inference timed out.", title=TITLE, ui_blocking=False)
        return
    status = _read_json(monitor["status_path"], {}) or {}
    if monitor.get("kind") == "train_setup":
        _monitor_setup_tick(monitor, status)
        return
    if monitor.get("kind") == "model_choice":
        _monitor_model_choice_tick(monitor, status)
        return
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
        target_open, reason = _monitor_target_is_open(monitor)
        if not target_open:
            if not monitor.get("waiting_to_apply_logged"):
                monitor["waiting_to_apply_logged"] = True
                _mimics_log(logging.INFO, "DINOv3 prediction is ready but was not applied: {0}".format(reason))
            status["application_state"] = "waiting_for_source_case"
            status["application_message"] = reason
            status["updated_at_epoch"] = time.time()
            try:
                _write_json_atomic(monitor["status_path"], status)
            except Exception:
                pass
            return
        monitor["waiting_to_apply_logged"] = False
        status.pop("application_state", None)
        status.pop("application_message", None)
        try:
            _write_json_atomic(monitor["status_path"], status)
        except Exception:
            pass
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
    target_open, reason = _monitor_target_is_open(monitor)
    if not target_open:
        status["application_state"] = "waiting_for_source_case"
        status["application_message"] = reason
        try:
            _write_json_atomic(monitor["status_path"], status)
        except Exception:
            pass
        return
    try:
        mask = _find_or_create_mask(monitor["mask_name"])
        _set_mask_from_u8(mask, bridge_result["output_path"], bridge_result["mimics_shape"])
    except Exception as exc:
        _stop_monitor(key)
        mimics.dialogs.message_box("Could not apply prediction:\n\n{0}".format(exc), title=TITLE, ui_blocking=False)
        return
    cleaned = []
    for path in (status.get("output_path"), monitor.get("bridge_job_dir")):
        if not path:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            cleaned.append(path)
        except Exception as error:
            _mimics_log(logging.WARNING, "Could not clean DINOv3 inference artifact {0}: {1}".format(path, error))
    status["applied_to_mimics"] = True
    status["applied_mask_name"] = monitor["mask_name"]
    status["artifact_cleanup"] = {"removed": cleaned, "completed_at_epoch": time.time()}
    try:
        _write_json_atomic(monitor["status_path"], status)
    except Exception as error:
        _mimics_log(logging.WARNING, "Could not record DINOv3 inference cleanup: {0}".format(error))
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
    # Make SetTimer/KillTimer signatures explicit to avoid callback type
    # identity mismatches when other modules configure argtypes separately.
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    user32.KillTimer.restype = ctypes.c_int
    interval = max(250, int(max(0.25, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint)

    def _timer_proc(hwnd, message, timer_id, tick_count):
        _monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    timer_id = user32.SetTimer(None, 0, interval, ctypes.cast(callback, ctypes.c_void_p))
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
        if progress.get("latest_epoch_line"):
            pieces = [str(progress.get("latest_epoch_line"))]
        else:
            pieces = []
            epoch = progress.get("epoch")
            epochs = progress.get("epochs")
            phase = progress.get("phase")
            best = progress.get("best_dsc")
            if epoch is not None and epochs is not None:
                pieces.append("epoch {0}/{1}".format(epoch, epochs))
            if phase:
                pieces.append(str(phase))
            if progress.get("batch") is not None and progress.get("batches") is not None:
                pieces.append("batch {0}/{1}".format(progress.get("batch"), progress.get("batches")))
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
            if progress.get("lr") is not None:
                try:
                    pieces.append("lr {0:.2e}".format(float(progress.get("lr"))))
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
        _display_kind(job.get("kind", "?")),
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
    def checkpoint_header_looks_valid(path):
        try:
            with open(path, "rb") as handle:
                header = handle.read(4)
            return header.startswith(b"PK") or header.startswith(b"\x80")
        except Exception:
            return False
    def usable_manifest(manifest):
        if not manifest:
            return False
        checkpoint = manifest.get("checkpoint", "")
        config_path = manifest.get("config", "")
        if not (checkpoint and config_path and os.path.isfile(checkpoint) and os.path.isfile(config_path)):
            return False
        try:
            if os.path.getsize(checkpoint) <= 0 or os.path.getsize(config_path) <= 0:
                return False
        except Exception:
            return False
        if not checkpoint_header_looks_valid(checkpoint):
            return False
        return True
    if os.path.isdir(models_dir):
        latest_path = os.path.join(models_dir, "latest.json")
        for manifest_path in [latest_path]:
            manifest = _read_json(manifest_path, {}) or {}
            if usable_manifest(manifest):
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
                if not usable_manifest(manifest):
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
        manifest = _read_json(abs_manifest, {}) or {}
        if not usable_manifest(manifest):
            continue
        candidates.append({
            "scope": "global",
            "manifest_path": abs_manifest,
            "model_id": row.get("model_id", ""),
            "organ": row.get("organ", organ),
            "sample_count": row.get("sample_count", 0),
            "created_at_epoch": row.get("created_at_epoch", 0.0),
            "manifest": manifest,
        })
    candidates.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    return candidates


def _dataset_latest_model_candidate(ts_root, organ):
    organ_slug = _safe_slug(organ)
    manifest_path = os.path.join(_workspace(ts_root), "models", organ_slug, "latest.json")
    if not os.path.isfile(manifest_path):
        return None, "No latest model was found for organ: {0}".format(organ)
    manifest = _read_json(manifest_path, {}) or {}
    checkpoint = manifest.get("checkpoint", "")
    config_path = manifest.get("config", "")
    if not checkpoint or not os.path.isfile(checkpoint):
        return None, "The latest model points to a missing checkpoint: {0}".format(checkpoint or "(empty)")
    if not config_path or not os.path.isfile(config_path):
        return None, "The latest model points to a missing config: {0}".format(config_path or "(empty)")
    try:
        if os.path.getsize(checkpoint) <= 0:
            return None, "The latest model checkpoint is empty: {0}".format(checkpoint)
        if os.path.getsize(config_path) <= 0:
            return None, "The latest model config is empty: {0}".format(config_path)
    except Exception as exc:
        return None, "Could not inspect the latest model files: {0}".format(exc)
    try:
        with open(checkpoint, "rb") as handle:
            header = handle.read(4)
        if not (header.startswith(b"PK") or header.startswith(b"\x80")):
            return None, "The latest model checkpoint does not look like a torch checkpoint: {0}".format(checkpoint)
    except Exception as exc:
        return None, "Could not inspect the latest model checkpoint: {0}".format(exc)
    return {
        "scope": "dataset latest",
        "manifest_path": os.path.abspath(manifest_path),
        "model_id": manifest.get("model_id", "latest") or "latest",
        "organ": manifest.get("organ", organ),
        "sample_count": manifest.get("sample_count", 0),
        "created_at_epoch": manifest.get("created_at_epoch", 0.0),
        "manifest": manifest,
    }, ""


def _launch_external_status_viewer(config, ts_root):
    script = _status_viewer_script()
    if not os.path.isfile(script):
        raise RuntimeError("External status viewer was not found: {0}".format(script))
    dinov3_root = _dinov3_root(config)
    python_exe = _fewshot_python(config, dinov3_root)
    workspace = _workspace(ts_root)
    jobs_dir = os.path.join(workspace, "jobs")
    if not os.path.isdir(jobs_dir):
        os.makedirs(jobs_dir)
    try:
        selected_organ = _selected_organ() or ""
    except Exception:
        selected_organ = ""
    context_path = os.path.join(
        jobs_dir,
        "status_viewer_{0}_{1}_context.json".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]),
    )
    context = {
        "schema_version": "mimics_fewshot_status_viewer_context.v1",
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "selected_organ": selected_organ,
        "project_root": _project_root(),
        "created_at_epoch": time.time(),
    }
    try:
        _write_json_atomic(context_path, context)
    except Exception as exc:
        raise RuntimeError(
            "Could not write the DINOv3 status viewer launch context. "
            "Path: {0}. Error: {1}".format(context_path, exc)
        )
    command = [python_exe, script, "--context", context_path]
    try:
        process = _launch_gui_process(command, cwd=_project_root())
    except Exception as exc:
        raise RuntimeError(
            "Could not start the DINOv3 status viewer process. "
            "Python: {0}. Script: {1}. Context: {2}. Error: {3}".format(
                python_exe,
                script,
                context_path,
                exc,
            )
        )

    # Catch immediate startup failures that otherwise look like "opened" but
    # no window appears.
    for _ in range(8):
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is not None:
        raise RuntimeError(
            "DINOv3 status viewer process exited immediately. "
            "Python: {0}. Script: {1}. Context: {2}".format(
                python_exe,
                script,
                context_path,
            )
        )

    _mimics_log(
        logging.INFO,
        "DINOv3 status viewer opened in an external process. PID: {0}".format(process.pid),
    )
    return process.pid


def _show_status_text(ts_root):
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
    try:
        selected_organ = _selected_organ()
    except Exception:
        selected_organ = None
    if selected_organ:
        jobs = [item for item in jobs if _safe_slug(item[1].get("organ", "")) == _safe_slug(selected_organ)]
    active_states = set([
        "launching", "preparing", "exporting_labels", "waiting_for_background_mimics",
        "waiting_for_gpu", "training", "running", "cancelling", "configuring",
        "selecting_model", "training_started",
    ])
    active_jobs = [item for item in jobs if item[1].get("status") in active_states]
    jobs = (active_jobs or jobs)[:1]
    lines = []
    for index, item in enumerate(jobs):
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


def _show_status():
    if not _selected_organ():
        mimics.dialogs.message_box(
            "Select one organ Mask to view its current DINOv3 task.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    ts_root = _choose_dataset_root("Select dataset folder")
    if not ts_root or not os.path.isdir(ts_root):
        return 1
    config = _config()
    mode = str(config.get("status_ui_mode", "external")).strip().lower()
    if mode != "text":
        try:
            _launch_external_status_viewer(config, ts_root)
            return 0
        except Exception as exc:
            _mimics_log(
                logging.WARNING,
                "DINOv3 external status viewer could not start: {0}".format(exc),
            )
            if not bool(config.get("status_ui_fallback_to_text", True)):
                mimics.dialogs.message_box(
                    "Could not open the external DINOv3 status viewer.\n\n{0}".format(exc),
                    title=TITLE,
                    ui_blocking=False,
                )
                return 1
    return _show_status_text(ts_root)


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
            _write_text_atomic(
                cancel_path,
                "cancel requested at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")),
            )
        except Exception as exc:
            job["cancel_marker_error"] = str(exc)
            _mimics_log(logging.WARNING, "Could not write DINOv3 cancel marker: {0}".format(exc))
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
        try:
            _write_json_atomic(status_path, job)
        except Exception as exc:
            _mimics_log(logging.WARNING, "Could not update DINOv3 job status after stop: {0}".format(exc))
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
            buttons=";".join([BUTTON_TRAIN_MODEL, BUTTON_PREDICT, BUTTON_PREDICT_MODEL, BUTTON_STATUS, BUTTON_STOP, BUTTON_CANCEL]),
            title=TITLE,
            ui_blocking=True,
        )
    if action == BUTTON_TRAIN:
        return _train_model(False)
    if action == BUTTON_TRAIN_ADVANCED:
        return _train_model(True)
    if action == BUTTON_TRAIN_MODEL:
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
