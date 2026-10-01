# -*- coding: utf-8 -*-
"""Shared Mimics-side helpers for applying AI prediction masks.

The mask-apply path (create/find target Mask, write a .u8 voxel buffer, verify
the live grid matches the inference target, resolve the dataset root and
source image of the open project) is identical across AI integrations
(nnU-Net, FlexiCT). Extracted from the original AI Mimics entry so
nnunet_mimics and future entries share one implementation.

Runs inside the foreground Mimics process (Python 3.5): .format() only, no
f-strings, no pathlib.
"""

from __future__ import print_function

import json
import logging
import os
import subprocess
import sys
import time

import mimics

import dataset_manifest
import runtime_common


SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_IMAGE_MODALITY_METADATA = "mimics_script.source_image_modality"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"

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


def _settings_path():
    return os.path.join(_project_root(), ".mimics_runtime", "mimics_mask_apply_state.json")


def _existing_dataset_roots(settings):
    """Dataset roots in ``settings`` that still exist on this machine."""
    roots = []

    def _add(value):
        path = str(value or "").strip()
        if path and os.path.isdir(path) and path not in roots:
            roots.append(path)

    _add(settings.get("last_dataset_root"))
    for row in settings.get("recent_dataset_roots") or []:
        _add(row.get("path") if isinstance(row, dict) else row)
    return roots


def _migrate_old_settings():
    """Move legacy state file from project root into .mimics_runtime/.

    A leftover legacy file (from the retired few-shot workflow) must not
    become this user's settings when every dataset root it remembers no
    longer exists — that would silently change the default dataset root.
    Such state is skipped, not migrated."""
    old_paths = [
        os.path.join(_project_root(), ".fewshot_mimics_state.json"),
        os.path.join(_project_root(), ".mimics_runtime", "fewshot_mimics_state.json"),
    ]
    new_path = _settings_path()
    for old_path in old_paths:
        if not os.path.isfile(old_path) or os.path.isfile(new_path):
            continue
        settings = _read_json(old_path, {}) or {}
        if not _existing_dataset_roots(settings):
            continue
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
        ("nninteractive_config.json", ".git"),
    )


def _config():
    path = os.path.join(_project_root(), "nnunet_config.json")
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


def _project_python_candidates():
    root = _project_root()
    return [
        os.path.join(root, "python_env", "python.exe"),
        os.path.join(root, "python_env", "Scripts", "python.exe"),
        os.path.join(root, "python_env", "python", "python.exe"),
        os.path.join(root, "python_env", "bin", "python3"),
        os.path.join(root, "python_env", "bin", "python"),
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


def _integration_python(config, project_root):
    """Resolve the external Python used by AI training/inference jobs.

    Canonical discovery first (env overrides + python_env/nninteractive_env
    layouts); the config/project-venv extras are kept for compatibility with
    existing integrations that pass a project directory.
    """
    found = runtime_common.find_external_python(_project_root())
    if found:
        return os.path.abspath(found)
    candidates = list(_project_python_candidates())
    _append_python_candidate(candidates, os.environ.get("MIMICS_AI_PYTHON", ""))
    _append_python_candidate(candidates, (config or {}).get("python", ""))
    candidates.extend([
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, ".venv", "bin", "python"),
        os.path.join(project_root, "venv", "Scripts", "python.exe"),
        os.path.join(project_root, "venv", "bin", "python"),
    ])
    env_roots = [
        os.path.abspath(os.path.join(_project_root(), name))
        for name in ("python_env", "nninteractive_env")
    ]
    try:
        current = os.path.abspath(sys.executable)
    except Exception:
        current = ""
    if current and any(current.startswith(env_root + os.sep) for env_root in env_roots):
        candidates.append(sys.executable)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run Setup Environment or setup_offline.bat before using AI training."
    )


def _integration_root(config):
    """Resolve the AI project directory (kept for call compatibility)."""
    env = os.environ.get("MIMICS_AI_PROJECT_ROOT", "")
    value = env or (config or {}).get("ai_project") or "integrations/nnunet_segmentation_workflow"
    return _resolve_path(value, _project_root())


def _mask_identity(mask):
    value = getattr(mask, "guid", None)
    return str(value) if value else str(getattr(mask, "name", "") or "")


def _mask_content_digest(mask):
    """SHA-256 of a Mask's voxel content, chunked so Mimics stays responsive.

    F21: number_of_pixels cannot detect an equal-volume edit (move a
    boundary, delete and add a voxel). This digest can. Like the
    nnInteractive base-sha precedent, it never copies the whole volume
    (buffer_byte_view + stream_buffer) and pumps the GUI between chunks.
    Returns "" when the buffer is unreadable so callers can degrade to the
    pixel-count check instead of blocking the apply flow.
    """
    try:
        view = mask.get_voxel_buffer()
        raw = runtime_common.buffer_byte_view(view)
        return runtime_common.stream_buffer(
            raw, compute_sha=True, progress_callback=_update_gui
        )
    except Exception:
        return ""


def _mask_snapshot(mask):
    """Launch-time identity of a Mask: guid, name, volume and content digest.

    The digest is the F21 edit detector; pixel_count is kept as a cheap
    early signal (a different count always means a changed Mask).
    """
    return {
        "guid": _mask_identity(mask),
        "name": str(getattr(mask, "name", "") or ""),
        "pixel_count": int(getattr(mask, "number_of_pixels", 0) or 0),
        "sha256": _mask_content_digest(mask),
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


def _global_model_registry_path():
    return os.path.join(os.path.expanduser("~"), ".mimics_script", "ai_model_index.json")


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
        if item in ("mcs_output", "segmentations", "fewshot_models", "flexict_models", "nnunet_models", "labels"):
            continue
        candidate_dir = os.path.abspath(os.path.join(item_path, "mcs_output"))
        if os.path.normcase(project_dir) == os.path.normcase(candidate_dir):
            return case_id
        if os.path.normcase(project_dir) == os.path.normcase(item_path):
            return item

    return None


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
    # Thin wrapper over the shared launcher: the project root is prepended to
    # PYTHONPATH so external scripts can import project-local modules
    # (tools.*).
    project_root = os.path.abspath(cwd or _project_root())
    return runtime_common.launch_external_gui_process(
        cmd,
        cwd=cwd,
        stderr_log=stderr_log,
        extra_pythonpath=[project_root],
    )


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
        mimics, _apply, transaction_name or "Apply AI Prediction"
    )
    _update_gui()
    try:
        mask.visible = True
        mask.selected = True
    except Exception:
        pass
