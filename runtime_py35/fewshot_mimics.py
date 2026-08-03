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

import dataset_manifest
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
BUTTON_UPDATE_SELECTED = "Update Selected Mask"
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_IMAGE_MODALITY_METADATA = "mimics_script.source_image_modality"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"

_MONITORS = {}
_GUI_PROCESSES = {}
_QT_CHECK_DONE = False
_QT_CHECK_RESULT = False


_write_json_atomic = runtime_common.write_json_atomic
_write_text_atomic = runtime_common.write_text_atomic
_read_json = runtime_common.read_json
_safe_slug = runtime_common.safe_slug
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs
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


def _remember_dataset_root(ts_root):
    path = os.path.abspath(str(ts_root or "")) if ts_root else ""
    if not path or not os.path.isdir(path):
        return
    settings = _load_settings()
    settings["last_dataset_root"] = path
    rows = []
    for row in settings.get("recent_dataset_roots", []) or []:
        if isinstance(row, dict):
            candidate = str(row.get("path") or "")
            used_at = float(row.get("used_at_epoch") or 0.0)
        else:
            candidate = str(row or "")
            used_at = 0.0
        if (
            candidate
            and os.path.normcase(os.path.abspath(candidate))
            != os.path.normcase(path)
        ):
            rows.append({"path": os.path.abspath(candidate), "used_at_epoch": used_at})
    rows.insert(0, {"path": path, "used_at_epoch": time.time()})
    settings["recent_dataset_roots"] = rows[:20]
    _save_settings(settings)


def _known_dataset_roots():
    settings = _load_settings()
    candidates = []
    inferred = _infer_dataset_root_from_project()
    if inferred:
        candidates.append(inferred)
    last_root = str(settings.get("last_dataset_root") or "").strip()
    if last_root:
        candidates.append(last_root)
    for row in settings.get("recent_dataset_roots", []) or []:
        if isinstance(row, dict):
            candidates.append(row.get("path"))
        else:
            candidates.append(row)
    registry = _read_json(_global_model_registry_path(), {}) or {}
    for row in registry.get("models", []) or []:
        candidates.append(row.get("ts_root"))
    result = []
    seen = set()
    for value in candidates:
        if not value:
            continue
        path = os.path.abspath(str(value))
        key = os.path.normcase(path)
        if key in seen or not os.path.isdir(path):
            continue
        seen.add(key)
        result.append(path)
    return result


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


def _mask_identity(mask):
    value = getattr(mask, "guid", None)
    return str(value) if value else str(getattr(mask, "name", "") or "")


def _deferred_prediction_target(mask):
    name = str(getattr(mask, "name", "") or "").strip()
    return {
        "mode": "choose_on_completion",
        "target_name": name,
        "target_guid": _mask_identity(mask),
        "initial_pixel_count": int(getattr(mask, "number_of_pixels", 0) or 0),
    }


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
        "source_modality": _metadata_get(image, SOURCE_IMAGE_MODALITY_METADATA, ""),
        "source_voxel_to_ras_matrix": matrix,
    }


def _find_mimics_exe():
    """Find MimicsResearch.exe installation path."""
    return runtime_common.find_mimics_exe()


def _workspace(ts_root):
    return os.path.join(os.path.abspath(ts_root), "fewshot_models")


def _setup_staging_workspace():
    return os.path.join(
        os.path.expanduser("~"),
        ".mimics_script",
        "fewshot_setup",
    )


def _global_model_registry_path():
    return os.path.join(os.path.expanduser("~"), ".mimics_script", "fewshot_model_index.json")


def _resolve_model_artifact(value, manifest_path, manifest=None):
    text = str(value or "").strip()
    if not text:
        return ""
    if not os.path.isabs(text):
        return os.path.abspath(os.path.join(os.path.dirname(manifest_path), text))
    if os.path.exists(text):
        return os.path.abspath(text)
    relocated = os.path.join(os.path.dirname(manifest_path), os.path.basename(text))
    if os.path.exists(relocated):
        return os.path.abspath(relocated)
    model_id = _safe_slug((manifest or {}).get("model_id", ""))
    versioned = os.path.join(
        os.path.dirname(manifest_path), model_id, os.path.basename(text)
    )
    if os.path.exists(versioned):
        return os.path.abspath(versioned)
    return os.path.abspath(text)


def _resolved_model_manifest(manifest_path):
    manifest = _read_json(manifest_path, {}) or {}
    if not isinstance(manifest, dict):
        return {}
    resolved = dict(manifest)
    for key in ("checkpoint", "config"):
        if manifest.get(key):
            resolved[key] = _resolve_model_artifact(
                manifest.get(key), manifest_path, manifest
            )
    resolved["_manifest_path"] = os.path.abspath(manifest_path)
    return resolved


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
    if values.get("decoder") and "training_dimension" not in values:
        values["preserve_legacy_decoder"] = True
    values.setdefault("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))
    values.setdefault("strategy", config.get("default_strategy", "adaptive"))
    values.setdefault("epochs", config.get("default_epochs", 20))
    values.setdefault("batch_size", config.get("default_batch_size", 0))
    values.setdefault("grad_accumulation", config.get("default_grad_accumulation", 1))
    values.setdefault("lr", config.get("default_lr", 0.001))
    values.setdefault("weight_decay", config.get("default_weight_decay", 0.01))
    values.setdefault("lr_scheduler", config.get("default_lr_scheduler", "cosine"))
    values.setdefault("warmup_epochs", config.get("default_warmup_epochs", 3))
    values.setdefault(
        "validation_interval",
        config.get("default_validation_interval", 2),
    )
    values.setdefault("img_size", config.get("default_img_size", "256,256"))
    values.setdefault("modality", config.get("default_modality", "auto"))
    values.setdefault("min_samples", config.get("default_min_samples", 1))
    values.setdefault("max_samples", config.get("default_max_samples", 0))
    values.setdefault("sample_mode", config.get("default_sample_mode", "all"))
    values.setdefault("val_fraction", config.get("default_val_fraction", 0.2))
    values.setdefault("min_val_samples", config.get("default_min_val_samples", 1))
    values.setdefault("finetune_method", config.get("default_finetune_method", "lora"))
    values.setdefault(
        "training_dimension",
        config.get("default_training_dimension", "auto"),
    )
    values.setdefault("quality_mode", config.get("default_quality_mode", "standard"))
    values.setdefault("decoder", config.get("default_decoder", "auto"))
    values.setdefault("model_scale", config.get("default_model_scale", "vitb16"))
    values.setdefault("model_path", config.get("default_model_path", ""))
    values.setdefault("model_sha256", config.get("default_model_sha256", ""))
    values.setdefault(
        "encoder_backend",
        config.get("default_feature_encoder_backend", "onnx"),
    )
    values.setdefault(
        "safetensors_sha256",
        config.get("default_safetensors_sha256", ""),
    )
    if (
        str(values.get("decoder", "")).lower() == "feature_unet2d"
        and str(values.get("encoder_backend", "")).lower() == "pytorch"
        and not str(values.get("model_path", "") or "").strip()
    ):
        values["model_sha256"] = str(values.get("safetensors_sha256") or "")
        if str(values.get("img_size", "")).strip() == "256,256":
            values["img_size"] = "224,224"
    values.setdefault("lora_rank", config.get("default_lora_rank", 8))
    values.setdefault("lora_alpha", config.get("default_lora_alpha", 16))
    values.setdefault("adapter_bottleneck", config.get("default_adapter_bottleneck", 64))
    values.setdefault("mixed_precision", config.get("default_mixed_precision", False))
    values.setdefault("sub_volume", config.get("default_sub_volume", False))
    values.setdefault("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))
    values.setdefault("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2))
    values.setdefault("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))
    values.setdefault("export_labels_before_training", config.get("default_export_labels_before_training", True))
    values.setdefault(
        "label_source",
        config.get("default_label_source")
        or (
            "mcs_refresh"
            if values.get("export_labels_before_training", True)
            else "source_dataset"
        ),
    )
    values.setdefault("label_root", "")
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault("gpu_memory_gb", config.get("default_gpu_memory_gb", 0.0))
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


def _configured_mask_names(config, organ):
    names = [str(organ or "").strip()]
    aliases = config.get("organ_mask_aliases") or {}
    if isinstance(aliases, dict):
        organ_key = _safe_slug(organ)
        for key, values in aliases.items():
            if _safe_slug(key) == organ_key:
                names.extend(_split_csv(values))
    result = []
    seen = set()
    for name in names:
        key = _safe_slug(name)
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


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
        str(int(options.get("batch_size", config.get("default_batch_size", 0)))),
        "--grad-accumulation",
        str(int(options.get("grad_accumulation", config.get("default_grad_accumulation", 1)))),
        "--lr",
        str(float(options.get("lr", config.get("default_lr", 0.001)))),
        "--weight-decay",
        str(float(options.get("weight_decay", config.get("default_weight_decay", 0.01)))),
        "--lr-scheduler",
        str(options.get("lr_scheduler", config.get("default_lr_scheduler", "cosine"))),
        "--warmup-epochs",
        str(int(options.get("warmup_epochs", config.get("default_warmup_epochs", 3)))),
        "--validation-interval",
        str(int(options.get(
            "validation_interval",
            config.get("default_validation_interval", 2),
        ))),
        "--img-size",
        str(options.get("img_size", config.get("default_img_size", "256,256"))),
        "--modality",
        str(options.get("modality", config.get("default_modality", "auto"))),
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
        "--model-scale",
        str(options.get("model_scale", config.get("default_model_scale", "vitb16"))),
        "--encoder-backend",
        str(options.get("encoder_backend", "auto")),
        "--lora-rank",
        str(int(options.get("lora_rank", config.get("default_lora_rank", 8)))),
        "--lora-alpha",
        str(int(options.get("lora_alpha", config.get("default_lora_alpha", 16)))),
        "--adapter-bottleneck",
        str(int(options.get("adapter_bottleneck", config.get("default_adapter_bottleneck", 64)))),
        "--gpu-lock-timeout-seconds",
        str(float(options.get("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400)))),
        "--gpu-memory-gb",
        str(float(options.get("gpu_memory_gb", config.get("default_gpu_memory_gb", 0.0)))),
        "--background-mimics-lock-timeout-seconds",
        str(float(options.get("background_mimics_lock_timeout_seconds", config.get("background_mimics_lock_timeout_seconds", 1800)))),
        "--keep-last-checkpoints",
        str(int(options.get("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2)))),
    ])
    if bool(options.get("preserve_legacy_decoder", False)):
        cmd.extend([
            "--decoder",
            str(options.get("decoder") or "feature_unet2d"),
        ])
    else:
        cmd.extend([
            "--training-dimension",
            str(options.get("training_dimension") or "auto"),
            "--quality-mode",
            str(options.get("quality_mode") or "standard"),
        ])
    model_path = str(options.get("model_path", config.get("default_model_path", "")) or "")
    if model_path:
        cmd.extend(["--model-path", model_path])
    model_sha256 = str(
        options.get("model_sha256", config.get("default_model_sha256", "")) or ""
    ).strip()
    if model_sha256:
        cmd.extend(["--model-sha256", model_sha256])
    label_root = str(options.get("label_root", "") or "").strip()
    if label_root:
        cmd.extend(["--label-root", label_root])
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
    mask_names = _split_csv(options.get("mask_names", ""))
    if mask_names:
        cmd.extend(["--mask-names", ",".join(mask_names)])


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
        _remember_dataset_root(path)
        return os.path.abspath(path)
    return None


def _initial_dataset_root():
    inferred = _infer_dataset_root_from_project()
    if inferred and os.path.isdir(inferred):
        return os.path.abspath(inferred)
    settings = _load_settings()
    last_root = str(settings.get("last_dataset_root", "") or "").strip()
    if last_root and os.path.isdir(last_root):
        return os.path.abspath(last_root)
    return ""


def _current_case_id():
    project_path = _current_project_path()
    if not project_path:
        return ""
    name = os.path.basename(project_path)
    return name[:-4] if name.lower().endswith(".mcs") else name


def _manifest_source_image(project_path, case_id):
    project_dir = os.path.dirname(project_path)
    for root in (project_dir, os.path.dirname(project_dir)):
        manifest_path = os.path.join(
            root, dataset_manifest.MANIFEST_FILENAME
        )
        if not os.path.isfile(manifest_path):
            continue
        payload = dataset_manifest.load_manifest(manifest_path)
        row = dataset_manifest.find_case(payload, case_id) or {}
        resolved = dataset_manifest.resolve_case_path(
            manifest_path, row, "image"
        )
        if resolved:
            return os.path.abspath(resolved)
    return ""


def _source_image_for_current_project(ts_root, case_id):
    project_path = _current_project_path() or ""
    if project_path:
        resolved = _manifest_source_image(project_path, case_id)
        if resolved:
            return resolved
    geometry = _active_source_geometry_payload() or {}
    recorded = str(geometry.get("source_image_path") or "").strip()
    if recorded and os.path.exists(recorded):
        return os.path.abspath(recorded)
    case_dir = os.path.join(ts_root, case_id) if ts_root else ""
    if case_dir and os.path.isdir(case_dir):
        preferred = (
            "ct.nii.gz", "mri.nii.gz", "mr.nii.gz",
            "ct.nii", "mri.nii", "mr.nii",
            "ct.mhd", "mri.mhd", "mr.mhd",
            "ct.mha", "mri.mha", "mr.mha",
            "ct.nrrd.gz", "mri.nrrd.gz", "mr.nrrd.gz",
            "ct.nrrd", "mri.nrrd", "mr.nrrd",
        )
        try:
            names = dict(
                (name.lower(), name) for name in os.listdir(case_dir)
            )
        except OSError:
            names = {}
        for name in preferred:
            if name in names:
                return os.path.join(case_dir, names[name])
        dicom_dir = os.path.join(case_dir, "dicom")
        if os.path.isdir(dicom_dir):
            return dicom_dir
    return ""


def _dataset_root_from_source(source_path, case_id):
    if not source_path:
        return ""
    path = os.path.abspath(source_path)
    case_dir = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.basename(case_dir).lower() == "dicom":
        case_dir = os.path.dirname(case_dir)
    if (
        os.path.basename(case_dir).lower() == str(case_id or "").lower()
        and os.path.isdir(os.path.dirname(case_dir))
    ):
        return os.path.dirname(case_dir)
    return ""


def _resolve_prediction_context():
    case_id = _current_case_id()
    if not case_id:
        return "", "", ""
    project_path = _current_project_path() or ""
    if project_path:
        source = _manifest_source_image(project_path, case_id)
        if source:
            root = (
                _dataset_root_from_source(source, case_id)
                or _infer_dataset_root_from_project()
                or os.path.dirname(os.path.dirname(project_path))
            )
            if root and os.path.isdir(root):
                root = os.path.abspath(root)
                _remember_dataset_root(root)
                return root, case_id, source
    candidates = _known_dataset_roots()
    initial = _initial_dataset_root()
    if initial and initial not in candidates:
        candidates.insert(0, initial)
    for root in candidates:
        source = _source_image_for_current_project(root, case_id)
        if source:
            derived = _dataset_root_from_source(source, case_id)
            resolved_root = derived or root
            _remember_dataset_root(resolved_root)
            return resolved_root, case_id, source
    geometry = _active_source_geometry_payload() or {}
    recorded = str(geometry.get("source_image_path") or "").strip()
    if recorded and os.path.exists(recorded):
        root = _dataset_root_from_source(recorded, case_id)
        if root:
            _remember_dataset_root(root)
            return root, case_id, os.path.abspath(recorded)
    return "", case_id, ""


def _management_dataset_root():
    return os.path.join(
        os.path.expanduser("~"),
        ".mimics_script",
        "dinov3_workspace",
    )


def _resolve_status_root(allow_management_fallback=False):
    try:
        selected_organ = _selected_organ() or ""
    except Exception:
        selected_organ = ""
    candidates = []
    for root in _known_dataset_roots():
        jobs_dir = os.path.join(_workspace(root), "jobs")
        models_dir = os.path.join(
            _workspace(root), "models", _safe_slug(selected_organ)
        )
        timestamps = []
        for directory in (jobs_dir, models_dir):
            if os.path.isdir(directory):
                try:
                    timestamps.append(os.path.getmtime(directory))
                except OSError:
                    pass
        if timestamps:
            candidates.append((max(timestamps), root))
    if not candidates:
        if not allow_management_fallback:
            return ""
        root = _management_dataset_root()
        workspace = _workspace(root)
        try:
            if not os.path.isdir(workspace):
                os.makedirs(workspace)
        except OSError:
            if not os.path.isdir(workspace):
                return ""
        _remember_dataset_root(root)
        return root
    candidates.sort(reverse=True)
    root = candidates[0][1]
    _remember_dataset_root(root)
    return root


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
    return runtime_common.process_exists(pid)


def _job_process_ids(job):
    result = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        try:
            pid = int((job or {}).get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in result:
            result.append(pid)
    return result


def _job_has_live_process(job):
    return any(_process_exists(pid) for pid in _job_process_ids(job))


def _mark_reaped_stopping_job_failed(status_path, job):
    latest = dict(job or {})
    if str(latest.get("status") or "").lower() != "stopping":
        return False
    if _job_has_live_process(latest):
        return False
    latest.update({
        "status": "failed",
        "phase": "termination_completed_after_error",
        "termination_pending": False,
        "terminated_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "error": str(
            latest.get("error")
            or "The background process stopped after an earlier task error."
        ),
    })
    try:
        _write_json_atomic(status_path, latest)
        return True
    except Exception as exc:
        _mimics_log(
            logging.WARNING,
            "Could not record the DINOv3 terminal state after process exit: {0}".format(
                exc
            ),
        )
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
        "stopping",
        "finalizing",
        "configuring",
        "selecting_model",
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
    rows = []
    for name in os.listdir(jobs_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(jobs_dir, name)
        payload = _read_json(path, {}) or {}
        if payload.get("status") not in active:
            continue
        if (
            str(payload.get("status") or "").lower() == "stopping"
            and not _job_has_live_process(payload)
        ):
            _mark_reaped_stopping_job_failed(path, payload)
            continue
        pids = _job_process_ids(payload)
        if pids and not _job_has_live_process(payload):
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
        "stopping": "Stopping after an error",
        "finalizing": "Finalizing",
        "configuring": "Configuring training",
        "selecting_model": "Selecting model",
        "training_started": "Training started",
        "preparing_remote": "Preparing remote data",
        "connecting_remote": "Connecting to remote server",
        "uploading": "Uploading training data",
        "starting_remote": "Starting remote training",
        "reconnecting_remote": "Remote training continues; reconnecting",
        "waiting_for_remote_gpu": "Waiting for remote GPU",
        "remote_control_unavailable": "Remote Docker status unavailable",
        "finalizing_remote": "Preparing remote model for local use",
        "downloading": "Downloading trained model",
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
        "A DINOv3 task is already running for this dataset.\n\n"
        "Task: {0}\nType: {1}\nOrgan: {2}\nStatus: {3}{4}\n\n"
        "Use 04 Show Status Results or 05 Stop AI Task before starting another GPU task.".format(
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
    modality.addItems(["auto", "ct", "mri", "other"])
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
        combo_set(modality, new_values.get("modality", "auto"))
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
        **_background_process_kwargs()
    )


def _launch_gui_process(cmd, cwd=None, stderr_log=None):
    # Launch GUI apps without CREATE_NO_WINDOW so Tk/PySide windows are visible.
    # On Windows, prefer pythonw.exe (same environment, no console flash).
    launch_cmd = list(cmd)
    if os.name == "nt" and launch_cmd:
        exe = os.path.abspath(str(launch_cmd[0]))
        if os.path.basename(exe).lower() == "python.exe":
            pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            if os.path.isfile(pythonw):
                launch_cmd[0] = pythonw
    env = _background_env()
    # Ensure the project root is on PYTHONPATH so external scripts can import
    # project-local modules (e.g. fewshot_strategies, tools.*).
    project_root = os.path.abspath(cwd or _project_root())
    existing = env.get("PYTHONPATH", "")
    paths = [p for p in existing.split(os.pathsep) if p] if existing else []
    if project_root not in paths:
        paths.insert(0, project_root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # Optionally capture stderr to a file for diagnostics on immediate exit.
    stderr_dest = subprocess.DEVNULL
    stderr_file_handle = None
    if stderr_log:
        try:
            parent = os.path.dirname(os.path.abspath(stderr_log))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            stderr_file_handle = open(stderr_log, "w", encoding="utf-8")
            stderr_dest = stderr_file_handle
        except Exception:
            stderr_dest = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            launch_cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_dest,
            env=env,
        )
    finally:
        # Close our handle; the child process has inherited the fd.
        if stderr_file_handle is not None:
            try:
                stderr_file_handle.close()
            except Exception:
                pass
    return proc


def _startup_stderr(path):
    if not path:
        return ""
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read(4096).strip()
    except Exception:
        pass
    return ""


def _existing_gui_process(key):
    process = _GUI_PROCESSES.get(key)
    if process is None:
        return None
    try:
        if process.poll() is None:
            return process
    except Exception:
        pass
    _GUI_PROCESSES.pop(key, None)
    return None


def _ai_draft_name(value):
    name = str(value or "Result").strip() or "Result"
    if name.lower().endswith(" - ai draft"):
        return name
    return name + " - AI Draft"


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
    ts_root = os.path.abspath(ts_root) if ts_root else ""
    workspace = _workspace(ts_root) if ts_root else _setup_staging_workspace()
    jobs_dir = os.path.join(workspace, "jobs")
    if not os.path.isdir(jobs_dir):
        os.makedirs(jobs_dir)
    status_path = os.path.join(jobs_dir, setup_id + ".json")
    context_path = os.path.join(jobs_dir, setup_id + "_context.json")
    context = {
        "schema_version": "mimics_fewshot_setup_context.v1",
        "setup_id": setup_id,
        "organ": organ,
        "ts_root": ts_root,
        "workspace": workspace,
        "mcs_output_dir": _resolve_mimics_output_dir(ts_root) if ts_root else "",
        "project_root": _project_root(),
        "pipeline_script": _pipeline_script(),
        "dinov3_root": dinov3_root,
        "python_exe": python_exe,
        "mimics_exe": _find_mimics_exe() or "",
        "config": config,
        "case_ids": _case_ids_from_dataset(ts_root) if ts_root else [],
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
        "ts_root": ts_root,
        "workspace": workspace,
        "context_path": context_path,
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }
    _write_json_atomic(status_path, status)
    _write_json_atomic(context_path, context)
    command = [python_exe, script, "--context", context_path]
    # Capture stderr to a temp file so immediate-exit diagnostics are not lost.
    stderr_log = os.path.join(jobs_dir, setup_id + "_stderr.log")
    try:
        process = _launch_gui_process(command, cwd=_project_root(), stderr_log=stderr_log)
    except Exception as exc:
        failed = _read_json(status_path, status) or status
        failed["status"] = "failed"
        failed["error"] = (
            "Could not start the external training setup process. "
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

    monitor_started = False
    try:
        monitor_started = _start_monitor(
            {
                "monitor_key": "setup_" + setup_id,
                "kind": "train_setup",
                "deadline": time.time() + float(config.get("training_setup_timeout_seconds", 12 * 60 * 60)),
                "status_path": status_path,
                "controller_pid": process.pid,
                "startup_stderr_path": stderr_log,
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
    config = _config()
    if advanced:
        mode = str(config.get("advanced_ui_mode", "external")).strip().lower()
        if mode != "internal":
            ts_root = _initial_dataset_root()
            if ts_root and not _guard_no_active_job(ts_root, requested_kind="train"):
                return 1
            try:
                return _launch_external_advanced_training(config, organ, ts_root)
            except Exception as exc:
                _mimics_log(
                    logging.ERROR,
                    "DINOv3 external advanced setup could not start: {0}".format(exc),
                )
                if not bool(config.get("advanced_ui_fallback_to_internal", True)):
                    mimics.dialogs.message_box(
                        (
                            "Could not open the external training setup UI.\n\n"
                            "{0}\n\n"
                            "Run setup_offline.bat to install or repair PySide6 in "
                            "nninteractive_env, then retry. Training was not started."
                        ).format(exc),
                        title=TITLE,
                        ui_blocking=False,
                    )
                    return 1
                _mimics_log(
                    logging.WARNING,
                    "Falling back to Mimics internal Advanced dialogs.",
                )
        ts_root = _choose_dataset_root("Select dataset folder")
        if not ts_root or not os.path.isdir(ts_root):
            _mimics_log(logging.INFO, "DINOv3 training cancelled: no dataset folder was selected.")
            return 1
        if not _guard_no_active_job(ts_root, requested_kind="train"):
            return 1
        options = _advanced_training_options(config, ts_root)
        if options is None:
            return 0
    else:
        ts_root = _choose_dataset_root("Select dataset folder")
        if not ts_root or not os.path.isdir(ts_root):
            _mimics_log(logging.INFO, "DINOv3 training cancelled: no dataset folder was selected.")
            return 1
        if not _guard_no_active_job(ts_root, requested_kind="train"):
            return 1
        options = _default_training_options(config)
        options["mask_names"] = ",".join(_configured_mask_names(config, organ))

    answer = mimics.dialogs.question_box(
        message=(
            "Training uses saved .mcs files.\n\n"
            "Save the current project before starting so the latest manual edits "
            "are included in the exported labels. Saved mask names are checked "
            "in the background before any voxel data is exported."
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
    if str(options.get("label_source") or "mcs_refresh") == "mcs_refresh":
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
            "retry_context": {
                "project_root": _project_root(),
                "pipeline_script": _pipeline_script(),
                "dinov3_root": dinov3_root,
                "python_exe": python_exe,
                "mimics_exe": mimics_exe or "",
                "mcs_output_dir": _resolve_mimics_output_dir(ts_root),
                "ts_root": os.path.abspath(ts_root),
                "workspace": _workspace(ts_root),
                "organ": organ,
                "case_ids": [],
                "config": config,
            },
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


def _launch_inference_job(
    config,
    ts_root,
    case_id,
    organ,
    selected_model=None,
    target_spec=None,
    source_image_path="",
):
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
    if source_image_path:
        cmd.extend(["--image-path", os.path.abspath(source_image_path)])
    source_geometry = _active_source_geometry_payload()
    if source_geometry:
        cmd.extend([
            "--expected-source-shape",
            json.dumps(source_geometry["source_shape"]),
            "--expected-source-voxel-to-ras-matrix",
            json.dumps(source_geometry["source_voxel_to_ras_matrix"]),
        ])
        expected_source_path = str(
            source_geometry.get("source_image_path", "") or ""
        ).strip()
        if (
            expected_source_path
            and os.path.exists(expected_source_path)
            and (
                not source_image_path
                or os.path.normcase(os.path.abspath(expected_source_path))
                == os.path.normcase(os.path.abspath(source_image_path))
            )
        ):
            cmd.extend([
                "--expected-source-image-path",
                expected_source_path,
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
            "prediction_target": target_spec or {},
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
        "mask_name": (target_spec or {}).get("target_name", _ai_draft_name(organ)),
        "prediction_target": target_spec or {"mode": "create_new", "target_name": _ai_draft_name(organ)},
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


def _launch_external_model_chooser(
    config,
    ts_root,
    case_id,
    organ,
    target_spec=None,
    source_image_path="",
):
    candidates = _model_candidates(ts_root, organ)
    if not candidates:
        mimics.dialogs.message_box(
            "No trained model was found for organ: {0}".format(organ),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    if len(candidates) == 1:
        return _launch_inference_job(
            config,
            ts_root,
            case_id,
            organ,
            candidates[0],
            target_spec=target_spec,
            source_image_path=source_image_path,
        )

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
        "prediction_target": target_spec or {},
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
    stderr_log = os.path.join(jobs_dir, choice_id + "_stderr.log")
    try:
        process = _launch_gui_process(
            [python_exe, script, "--context", context_path],
            cwd=_project_root(),
            stderr_log=stderr_log,
        )
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
    _start_monitor(
        {
            "monitor_key": choice_id,
            "kind": "model_choice",
            "status_path": status_path,
            "ts_root": ts_root,
            "case_id": case_id,
            "organ": organ,
            "prediction_target": target_spec or {},
            "source_image_path": source_image_path,
            "controller_pid": process.pid,
            "startup_stderr_path": stderr_log,
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
    selected_mask = _selected_mask()
    organ = str(getattr(selected_mask, "name", "") or "").strip() if selected_mask is not None else ""
    if not organ:
        mimics.dialogs.message_box(
            "Select one Mask whose name is the organ/model to use, then run this entry again.",
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    target_spec = _deferred_prediction_target(selected_mask)
    ts_root, case_id, source_image_path = _resolve_prediction_context()
    if not ts_root or not source_image_path:
        source_geometry = _active_source_geometry_payload() or {}
        recorded = str(
            source_geometry.get("source_image_path") or ""
        ).strip()
        detail = (
            "\n\nRecorded source path:\n{0}".format(recorded)
            if recorded else
            ""
        )
        mimics.dialogs.message_box(
            (
                "The current project could not be linked to its source medical "
                "image. DINOv3 prediction was not started because this model "
                "expects source-grid physical intensities, not an unverified "
                "Mimics display buffer.{0}\n\nMove the dataset manifest with "
                "the .mcs folder, restore/relink the source image, or use a "
                "model trained with an explicit Mimics-buffer input contract."
            ).format(detail),
            title=TITLE,
            ui_blocking=False,
        )
        _mimics_log(
            logging.ERROR,
            "DINOv3 prediction could not resolve the current source image.{0}".format(
                detail.replace("\n", " ")
            ),
        )
        return 1
    if not _guard_no_active_job(ts_root, requested_kind="infer"):
        return 1

    config = _config()
    if choose_model:
        try:
            return _launch_external_model_chooser(
                config,
                ts_root,
                case_id,
                organ,
                target_spec=target_spec,
                source_image_path=source_image_path,
            )
        except Exception as exc:
            _mimics_log(logging.ERROR, "DINOv3 external model chooser could not start: {0}".format(exc))
            mimics.dialogs.message_box(
                "Could not open the external DINOv3 model chooser.\n\n{0}".format(exc),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    model_rows = _model_candidates(ts_root, organ)
    selected_model = model_rows[0] if model_rows else None
    reason = (
        ""
        if selected_model
        else "No usable local or registered model was found for organ: {0}".format(
            organ
        )
    )
    if not selected_model:
        mimics.dialogs.message_box(
            "{0}\n\nUse Predict Current Case (Choose Model) to select another valid model, or train a new model for this organ.".format(reason),
            title=TITLE,
            ui_blocking=False,
        )
        _mimics_log(logging.WARNING, "DINOv3 latest-model prediction was not started: {0}".format(reason))
        return 1
    return _launch_inference_job(
        config,
        ts_root,
        case_id,
        organ,
        selected_model,
        target_spec=target_spec,
        source_image_path=source_image_path,
    )


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
            **_background_process_kwargs()
        )
    monitor["bridge_started"] = True
    monitor["bridge_process"] = process
    monitor["bridge_pid"] = process.pid
    monitor["bridge_result_path"] = result_path
    monitor["bridge_output_path"] = output_path

    def _wait_bridge():
        try:
            stdout, stderr = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            _write_json_atomic(
                result_path,
                {
                    "status": "error",
                    "error": "Prediction conversion timed out after 600 seconds.",
                },
            )
            return
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
    except Exception as exc:
        try:
            runtime_common.terminate_process_async(
                process=process,
                graceful_seconds=0.0,
            )
        except Exception:
            pass
        _write_json_atomic(
            result_path,
            {
                "status": "error",
                "error": "Could not start prediction conversion monitor: {0}".format(exc),
            },
        )


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
        bound_image = getattr(mask, "image", None)
    except Exception:
        bound_image = None
    try:
        bound_to_active = bound_image == active_image
    except Exception:
        bound_to_active = bound_image is active_image
    if bound_image is not None and not bound_to_active:
        raise RuntimeError(
            "Mimics created the prediction Mask on a different image."
        )
    return mask


def _unique_mask_name(base_name):
    names = set(str(getattr(mask, "name", "") or "") for mask in mimics.data.masks)
    if base_name not in names:
        return base_name
    index = 2
    while "{0} {1}".format(base_name, index) in names:
        index += 1
    return "{0} {1}".format(base_name, index)


def _new_prediction_mask(name):
    active_image = mimics.data.images.get_active()
    if active_image is None:
        raise RuntimeError("No active Mimics image is available for prediction import.")
    mask = mimics.segment.create_mask()
    mask.name = _unique_mask_name(name)
    try:
        bound = getattr(mask, "image", None)
    except Exception:
        bound = None
    try:
        bound_to_active = bound == active_image
    except Exception:
        bound_to_active = bound is active_image
    if bound is not None and not bound_to_active:
        raise RuntimeError(
            "Mimics created the prediction copy on a different image."
        )
    return mask


def _prediction_target_mask(monitor):
    spec = monitor.get("prediction_target") or {}
    if spec.get("mode") == "choose_on_completion":
        decision = mimics.dialogs.question_box(
            message=(
                "Few-shot prediction is complete and ready to apply.\n\n"
                "Update Selected Mask replaces {0}.\n"
                "Create Editable Copy keeps it unchanged and creates a separate Mask.\n\n"
                "Either result remains editable and can be refined with nnInteractive."
            ).format(spec.get("target_name") or monitor.get("organ", "the selected Mask")),
            buttons=";".join([BUTTON_UPDATE_SELECTED, "Create Editable Copy"]),
            title="DINOv3 Prediction Ready",
            ui_blocking=True,
        )
        if decision == BUTTON_UPDATE_SELECTED:
            spec["mode"] = "update_selected"
        else:
            if decision != "Create Editable Copy":
                _mimics_log(logging.INFO, "DINOv3 result destination was closed; using a new editable Mask to preserve the selected Mask.")
            spec["mode"] = "create_new"
            spec["target_name"] = _ai_draft_name(spec.get("target_name") or monitor.get("organ", "Result"))
        monitor["prediction_target"] = spec
    if spec.get("mode") != "update_selected":
        return _new_prediction_mask(
            spec.get("target_name") or monitor.get("mask_name") or _ai_draft_name("Result")
        )

    target_guid = str(spec.get("target_guid", "") or "")
    target_name = str(spec.get("target_name", "") or "")
    target = None
    for mask in mimics.data.masks:
        identity = _mask_identity(mask)
        if (target_guid and identity == target_guid) or (
            not target_guid and str(getattr(mask, "name", "") or "") == target_name
        ):
            target = mask
            break
    if target is None:
        _mimics_log(
            logging.WARNING,
            "The selected prediction target no longer exists; creating a new editable Mask instead.",
        )
        return _new_prediction_mask(_ai_draft_name(target_name or monitor.get("organ", "Result")))

    initial_count = spec.get("initial_pixel_count")
    current_count = int(getattr(target, "number_of_pixels", 0) or 0)
    if initial_count is not None and current_count != int(initial_count):
        _mimics_log(
            logging.WARNING,
            "The selected Mask changed while prediction was running. Manual edits were preserved and the prediction will be applied to a new Mask.",
        )
        return _new_prediction_mask(_ai_draft_name(target_name or monitor.get("organ", "Result")))
    return target


def _set_mask_from_u8(mask, path, shape, transaction_name=None):
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
    _update_gui()
    runtime_common.execute_mimics_transaction(
        mimics, _apply, transaction_name or "Apply DINOv3 Prediction"
    )
    _update_gui()
    try:
        mask.visible = True
        mask.selected = True
    except Exception:
        pass


def _stop_monitor(key):
    monitor = _MONITORS.pop(key, None)
    if not monitor:
        return
    bridge_process = monitor.get("bridge_process")
    if bridge_process is not None and bridge_process.poll() is None:
        try:
            runtime_common.terminate_process_async(
                process=bridge_process,
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
        _mimics_log(logging.INFO, "DINOv3 training setup status: {0}".format(line))
    if state == "training_started":
        training_status_path = status.get("training_status_path")
        training_job_id = status.get("training_job_id", "?")
        if training_status_path and os.path.isfile(training_status_path):
            monitor["kind"] = "train"
            monitor["status_path"] = training_status_path
            monitor["last_line"] = ""
            _mimics_log(
                logging.INFO,
                "DINOv3 training started from the external setup. Task: {0}. Status: {1}".format(
                    training_job_id,
                    training_status_path,
                ),
            )
            return
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "Training setup reported that training started, but its status file was not found.\n\n"
            "{0}".format(training_status_path or "(missing path)"),
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state in ("closed", "cancelled"):
        _stop_monitor(key)
        _mimics_log(logging.INFO, "DINOv3 training setup closed before launching training.")
        return
    if state == "failed":
        _stop_monitor(key)
        mimics.dialogs.message_box(
            "DINOv3 training setup failed.\n\n{0}".format(status.get("error", "Unknown error")),
            title=TITLE,
            ui_blocking=False,
        )
        return
    if state == "configuring":
        pid = status.get("controller_pid") or monitor.get("controller_pid")
        if pid and not _process_exists(pid):
            stderr_detail = _startup_stderr(monitor.get("startup_stderr_path"))
            status["status"] = "failed" if stderr_detail else "closed"
            if stderr_detail:
                status["error"] = "The external training setup exited before opening.\n\n{0}".format(stderr_detail)
            status["updated_at_epoch"] = time.time()
            try:
                _write_json_atomic(monitor["status_path"], status)
            except Exception:
                pass
            _stop_monitor(key)
            if stderr_detail:
                mimics.dialogs.message_box(
                    "DINOv3 training setup could not open.\n\n{0}".format(stderr_detail),
                    title=TITLE,
                    ui_blocking=False,
                )
            else:
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
            target_spec=monitor.get("prediction_target") or {},
            source_image_path=monitor.get("source_image_path") or "",
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
        stderr_detail = _startup_stderr(monitor.get("startup_stderr_path"))
        status["status"] = "failed" if stderr_detail else "closed"
        if stderr_detail:
            status["error"] = "The external model chooser exited before opening.\n\n{0}".format(stderr_detail)
        status["updated_at_epoch"] = time.time()
        try:
            _write_json_atomic(monitor["status_path"], status)
        except Exception:
            pass
        _stop_monitor(key)
        if stderr_detail:
            mimics.dialogs.message_box(
                "DINOv3 model chooser could not open.\n\n{0}".format(stderr_detail),
                title=TITLE,
                ui_blocking=False,
            )
        else:
            _mimics_log(logging.INFO, "DINOv3 model chooser closed before selecting a model.")


def _monitor_stopping_job(monitor, status, line):
    """Report a stuck shutdown and publish failure only after every PID exits."""
    key = monitor.get("monitor_key")
    kind = _display_kind(status.get("kind") or monitor.get("kind") or "task")
    pids = _job_process_ids(status)
    if pids and _job_has_live_process(status):
        if not monitor.get("stopping_notice_shown"):
            monitor["stopping_notice_shown"] = True
            message = (
                "DINOv3 {0} encountered an error and is still stopping an "
                "external process.\n\n"
                "Mimics remains available. The GPU/resource lock is being "
                "retained until the process exits, so another AI task may "
                "wait. Use Show Status to inspect details. If it does not "
                "finish stopping, use Stop AI Task again to force cleanup."
            ).format(kind)
            _mimics_log(
                logging.WARNING,
                "DINOv3 {0} is still stopping; live process IDs: {1}. "
                "Resource locks remain held until exit.".format(
                    kind,
                    ", ".join([str(pid) for pid in pids]),
                ),
            )
            mimics.dialogs.message_box(
                message,
                title=TITLE,
                ui_blocking=False,
            )
        return

    # Re-read immediately before publishing a terminal state. The pipeline may
    # have completed cleanup between the liveness check and this callback.
    latest = _read_json(monitor.get("status_path"), {}) or {}
    if str(latest.get("status") or "").lower() != "stopping":
        return
    if _job_has_live_process(latest):
        return
    error = str(
        latest.get("error")
        or status.get("error")
        or "The background process stopped after an earlier task error."
    )
    if not _mark_reaped_stopping_job_failed(
        monitor.get("status_path"),
        latest,
    ):
        return
    _stop_monitor(key)
    _mimics_log(
        logging.ERROR,
        "DINOv3 {0} stopped after an error and is now Failed: {1}".format(
            kind,
            error,
        ),
    )
    mimics.dialogs.message_box(
        "DINOv3 {0} failed.\n\n{1}\n\n{2}".format(kind, error, line or ""),
        title=TITLE,
        ui_blocking=False,
    )


def _monitor_tick_locked(monitor):
    key = monitor.get("monitor_key")
    if time.time() > monitor.get("deadline", 0):
        _stop_monitor(key)
        if monitor.get("kind") == "train":
            mimics.dialogs.message_box("Training status monitoring timed out. The background task may still be running; use 04 Show Status Results.", title=TITLE, ui_blocking=False)
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
        if state in (
            "waiting_for_gpu",
            "waiting_for_background_mimics",
            "waiting_for_remote_gpu",
            "waiting_for_dataset",
        ):
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "dino_train_wait",
                detail="{0}|{1}".format(state, line),
                interval_seconds=60.0,
                initial_delay_seconds=60.0,
            )
            if due:
                _mimics_log(
                    logging.INFO,
                    "DINOv3 training is still waiting ({0}s): {1}. Use 05 "
                    "Stop AI Task to cancel and release its resources.".format(
                        int(elapsed), line
                    ),
                )
        else:
            runtime_common.clear_progress_notice(monitor, "dino_train_wait")
        if state in ("completed", "failed", "cancelled", "abandoned"):
            _stop_monitor(key)
            if state == "completed":
                message = "Few-shot training completed.\n\n{0}".format(line)
            elif state == "cancelled":
                message = "Few-shot training was cancelled.\n\n{0}".format(line)
            else:
                message = "Few-shot training failed.\n\n{0}\n\n{1}".format(
                    status.get("error", "Unknown error"),
                    line,
                )
            mimics.dialogs.message_box(message, title=TITLE, ui_blocking=False)
        elif state == "stopping":
            _monitor_stopping_job(monitor, status, line)
            return
        elif state in ("cancelling", "finalizing"):
            return
        return
    state = status.get("status")
    if state in (
        "",
        None,
        "launching",
        "running",
        "waiting_for_gpu",
        "waiting_for_background_mimics",
        "waiting_for_remote_gpu",
    ):
        line = _format_job_line(status)
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "dino_infer_progress",
            detail="{0}|{1}".format(state, line),
            interval_seconds=60.0,
            initial_delay_seconds=0.0,
        )
        if due:
            suffix = (
                " Use 05 Stop AI Task to cancel and release its resources."
                if state and str(state).startswith("waiting_for")
                else ""
            )
            _mimics_log(
                logging.INFO,
                "DINOv3 prediction status{0}: {1}.{2}".format(
                    " ({0}s)".format(int(elapsed)) if elapsed >= 1.0 else "",
                    line or str(state or "Starting"),
                    suffix,
                ),
            )
        return
    runtime_common.clear_progress_notice(monitor, "dino_infer_progress")
    if state == "stopping":
        _monitor_stopping_job(monitor, status, _format_job_line(status))
        return
    if state == "cancelling":
        return
    if state == "cancelled":
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
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "dino_apply_target",
                detail=reason,
                interval_seconds=60.0,
                initial_delay_seconds=0.0,
            )
            if due:
                monitor["waiting_to_apply_logged"] = True
                _mimics_log(
                    logging.INFO,
                    "DINOv3 prediction is ready but has not been applied{0}: "
                    "{1} Use 05 Stop AI Task to discard the pending result.".format(
                        " ({0}s)".format(int(elapsed)) if elapsed >= 1.0 else "",
                        reason,
                    ),
                )
            status["application_state"] = "waiting_for_source_case"
            status["application_message"] = reason
            status["updated_at_epoch"] = time.time()
            try:
                _write_json_atomic(monitor["status_path"], status)
            except Exception:
                pass
            return
        monitor["waiting_to_apply_logged"] = False
        runtime_common.clear_progress_notice(monitor, "dino_apply_target")
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
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "dino_apply_target",
            detail=reason,
            interval_seconds=60.0,
            initial_delay_seconds=0.0,
        )
        if due:
            _mimics_log(
                logging.INFO,
                "DINOv3 prediction conversion is ready but application is "
                "waiting{0}: {1} Use 05 Stop AI Task to discard it.".format(
                    " ({0}s)".format(int(elapsed)) if elapsed >= 1.0 else "",
                    reason,
                ),
            )
        status["application_state"] = "waiting_for_source_case"
        status["application_message"] = reason
        try:
            _write_json_atomic(monitor["status_path"], status)
        except Exception:
            pass
        return
    runtime_common.clear_progress_notice(monitor, "dino_apply_target")
    try:
        mask = _prediction_target_mask(monitor)
        _set_mask_from_u8(mask, bridge_result["output_path"], bridge_result["mimics_shape"])
        monitor["mask_name"] = str(
            getattr(mask, "name", monitor.get("mask_name", "")) or ""
        )
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


def _monitor_tick(monitor):
    if monitor.get("busy"):
        return
    operation_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "DINOv3 result monitor"
    )
    if not operation_token:
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        owner_text = str(owner.get("owner") or "another Mask operation")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "dino_buffer_wait",
            detail=owner_text,
            interval_seconds=60.0,
            initial_delay_seconds=5.0,
        )
        if due:
            _mimics_log(
                logging.INFO,
                "DINOv3 result handling is waiting for {0} ({1}s). Mimics "
                "remains available; stop the owning task if this wait is no "
                "longer wanted.".format(owner_text, int(elapsed)),
            )
        return
    runtime_common.clear_progress_notice(monitor, "dino_buffer_wait")
    monitor["busy"] = True
    try:
        _monitor_tick_locked(monitor)
    finally:
        monitor["busy"] = False
        runtime_common.release_local_operation("mask_buffer_access", operation_token)


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
            manifest = _resolved_model_manifest(latest)
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
    def usable_manifest(manifest_path):
        manifest = _resolved_model_manifest(manifest_path)
        if not manifest:
            return None
        checkpoint = manifest.get("checkpoint", "")
        config_path = manifest.get("config", "")
        if not (checkpoint and config_path and os.path.isfile(checkpoint) and os.path.isfile(config_path)):
            return None
        try:
            if os.path.getsize(checkpoint) <= 0 or os.path.getsize(config_path) <= 0:
                return None
        except Exception:
            return None
        if not checkpoint_header_looks_valid(checkpoint):
            return None
        return manifest
    if os.path.isdir(models_dir):
        latest_path = os.path.join(models_dir, "latest.json")
        for manifest_path in [latest_path]:
            manifest = usable_manifest(manifest_path)
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
                manifest = usable_manifest(manifest_path)
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
        manifest = usable_manifest(abs_manifest)
        if not manifest:
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
    manifest = _resolved_model_manifest(manifest_path)
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
    gui_key = "status:{0}:{1}".format(os.path.abspath(workspace), _safe_slug(selected_organ))
    existing = _existing_gui_process(gui_key)
    if existing is not None:
        _mimics_log(
            logging.INFO,
            "DINOv3 status viewer is already open for this task (PID {0}).".format(existing.pid),
        )
        return existing.pid
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

    _GUI_PROCESSES[gui_key] = process

    _mimics_log(
        logging.INFO,
        "DINOv3 status viewer opened in an external process. PID: {0}".format(process.pid),
    )
    return process.pid


def _show_status_text(ts_root):
    jobs_dir = os.path.join(_workspace(ts_root), "jobs")
    if not os.path.isdir(jobs_dir):
        mimics.dialogs.message_box("No DINOv3 task history was found.", title=TITLE, ui_blocking=False)
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
        "waiting_for_gpu", "training", "running", "cancelling", "stopping",
        "finalizing", "configuring", "selecting_model", "training_started",
        "preparing_remote", "connecting_remote", "uploading",
        "starting_remote", "reconnecting_remote", "waiting_for_remote_gpu",
        "remote_control_unavailable",
        "finalizing_remote",
        "downloading",
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
        lines.append("The active task can be stopped with 05 Stop AI Task.")
    mimics.dialogs.message_box(
        "\n".join(lines) if lines else "No DINOv3 task history was found.",
        title=TITLE,
        ui_blocking=False,
    )
    return 0


def _show_status():
    ts_root = _resolve_status_root(allow_management_fallback=True)
    if not ts_root or not os.path.isdir(ts_root):
        mimics.dialogs.message_box(
            "No recent DINOv3 task or model workspace was found. Start a "
            "training or prediction task first.",
            title=TITLE,
            ui_blocking=False,
        )
        return 0
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


def _request_fewshot_cancel_async(job, status_path, grace_seconds=30.0):
    """Finish cancellation only after every recorded task process has stopped."""
    remote = str(job.get("execution_backend") or "") == "remote"
    if remote and status_path:
        try:
            config = _config()
            python_exe = _fewshot_python(config, _dinov3_root(config))
            _launch_process(
                [
                    python_exe,
                    os.path.join(
                        _project_root(),
                        "tools",
                        "remote_training_controller.py",
                    ),
                    "cancel",
                    "--status",
                    status_path,
                ],
                cwd=_project_root(),
            )
            grace_seconds = max(float(grace_seconds), 45.0)
        except Exception as exc:
            _mimics_log(
                logging.WARNING,
                "Could not start remote DINOv3 stop helper: {0}".format(exc),
            )
    pids = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        try:
            pid = int(job.get(key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and pid not in pids:
            pids.append(pid)

    def finalize_cancelled(killed):
        if not status_path:
            return
        latest = _read_json(status_path, {}) or {}
        if str(latest.get("status") or "").lower() in (
            "completed",
            "failed",
            "cancelled",
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
        if any(_process_exists(pid) for pid in latest_pids):
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
            try:
                _write_json_atomic(status_path, latest)
            except Exception:
                pass
            return
        latest.update({
            "status": "cancelled",
            "cancel_requested_at_epoch": time.time(),
            "cancelled_pids": killed,
            "updated_at_epoch": time.time(),
        })
        try:
            _write_json_atomic(status_path, latest)
        except Exception as exc:
            _mimics_log(
                logging.WARNING,
                "Could not record DINOv3 cancellation completion: {0}".format(exc),
            )

    if not pids or not any(_process_exists(pid) for pid in pids):
        finalize_cancelled([])
        return

    def worker():
        deadline = time.time() + max(0.0, float(grace_seconds))
        while time.time() < deadline:
            if all(not _process_exists(pid) for pid in pids):
                break
            time.sleep(0.25)
        if remote and status_path:
            latest = _read_json(status_path, {}) or {}
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
                try:
                    _write_json_atomic(status_path, latest)
                except Exception:
                    pass
                return
        killed = []
        for pid in pids:
            if _process_exists(pid) and _terminate_process_tree(pid):
                killed.append(pid)
        reap_deadline = time.time() + 10.0
        while time.time() < reap_deadline:
            if all(not _process_exists(pid) for pid in pids):
                break
            time.sleep(0.25)
        finalize_cancelled(killed)

    thread = threading.Thread(target=worker, name="MimicsFewshotCancel")
    thread.daemon = True
    thread.start()


def _stop_pending_inference_application():
    pending = []
    for monitor in list(_MONITORS.values()):
        if str(monitor.get("kind") or "") != "infer":
            continue
        status = _read_json(monitor.get("status_path"), {}) or {}
        if str(status.get("status") or "").lower() != "completed":
            continue
        if bool(status.get("applied_to_mimics")):
            continue
        pending.append(
            (
                float(status.get("updated_at_epoch", 0.0) or 0.0),
                monitor,
                status,
            )
        )
    if not pending:
        return False
    pending.sort(key=lambda row: row[0], reverse=True)
    _updated, monitor, status = pending[0]
    answer = mimics.dialogs.question_box(
        message=(
            "Discard the pending DINOv3 result application?\n\n"
            "The completed prediction file will be kept, but it will not be "
            "applied automatically to a Mimics Mask."
        ),
        buttons=BUTTON_STOP + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer != BUTTON_STOP:
        return True
    status["application_cancelled"] = True
    status["application_cancelled_at_epoch"] = time.time()
    status["application_state"] = "cancelled"
    status["application_message"] = (
        "Pending Mimics application was cancelled by the user."
    )
    status["updated_at_epoch"] = time.time()
    try:
        _write_json_atomic(monitor.get("status_path"), status)
    except Exception as exc:
        _mimics_log(
            logging.WARNING,
            "Could not record pending DINOv3 application cancellation: {0}".format(
                exc
            ),
        )
    _stop_monitor(monitor.get("monitor_key"))
    _mimics_log(
        logging.INFO,
        "Pending DINOv3 result application was cancelled; the existing "
        "Mimics Mask was not changed.",
    )
    return True


def _stop_latest_job():
    ts_root = _resolve_status_root()
    if not ts_root or not os.path.isdir(ts_root):
        if _stop_pending_inference_application():
            return 0
        mimics.dialogs.message_box(
            "No recent DINOv3 workspace was found.",
            title=TITLE,
            ui_blocking=False,
        )
        return 0
    status_path, job = _latest_active_job(ts_root)
    if not job:
        if _stop_pending_inference_application():
            return 0
        mimics.dialogs.message_box("No running DINOv3 task was found.", title=TITLE, ui_blocking=False)
        return 0
    current_status = str(job.get("status") or "").lower()
    if current_status in ("cancelling", "finalizing"):
        mimics.dialogs.message_box(
            (
                "Task {0} is already {1}. Wait for cleanup to finish; "
                "Show Status will report the final result."
            ).format(job.get("job_id", "?"), current_status),
            title=TITLE,
            ui_blocking=False,
        )
        return 0
    force_stopping = current_status == "stopping"
    if force_stopping and not _job_has_live_process(job):
        recorded = _mark_reaped_stopping_job_failed(status_path, job)
        mimics.dialogs.message_box(
            (
                "Task {0} has stopped after an error and is now Failed."
                if recorded else
                "Task {0} has stopped, but its final status could not be saved. "
                "Check the Mimics log and retry Show Status."
            ).format(job.get("job_id", "?")),
            title=TITLE,
            ui_blocking=False,
        )
        return 0
    answer = mimics.dialogs.question_box(
        message=(
            "{0}\n\n"
            "Task: {1}\nType: {2}\nOrgan: {3}\nStatus: {4}"
        ).format(
            (
                "This task is still stopping after an error. Force-stop its "
                "remaining external processes?"
                if force_stopping else
                "Stop the active DINOv3 task?"
            ),
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
    # The controller gets a cooperative cleanup window. The asynchronous
    # finalizer only writes the terminal state after all recorded processes are
    # gone, so it cannot overwrite a last-second completed/failed status.
    _request_fewshot_cancel_async(job, status_path, grace_seconds=30.0)
    mimics.dialogs.message_box(
        (
            "Stop requested for {0}.\n\n"
            "The task is releasing its model, files, and GPU lock. "
            "Show Status will report Cancelled when cleanup is complete."
        ).format(job.get("job_id", "?")),
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
