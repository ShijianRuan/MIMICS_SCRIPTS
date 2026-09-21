# -*- coding: utf-8 -*-
"""Standalone nnInteractive tool for Mimics Research 21.

This module runs inside Mimics Python 3.5. It intentionally has no dependency
on SegmentationPlatform cases, reviews, or registries. It may use the shared
relocatable dataset manifest written by the Mimics import/export tools.

The user selects a target Mask in the Project Tree and launches the
``nnInteractive`` Scripting Library entry. Prompts are collected with Mimics
native APIs and inference runs in an external Python 3.10+ environment.
"""

from __future__ import print_function

import atexit
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import mimics

import dataset_manifest
import runtime_common


TITLE = "nnInteractive Segmentation"
BUTTON_POINT = "Add Points"
BUTTON_SCRIBBLE = "Paint Scribble"
BUTTON_BOX = "Draw Box"
BUTTON_LASSO = "Draw Lasso"
BUTTON_UNDO = "Undo Last Prompt"
BUTTON_RESET = "Reset To Start"
BUTTON_FINISH = "Finish"
BUTTON_FOREGROUND = "Foreground"
BUTTON_BACKGROUND = "Background"
BUTTON_INCLUDE_POINT = "Add Include Point"
BUTTON_EXCLUDE_POINT = "Add Exclude Point"
BUTTON_REMOVE_POINT = "Remove Last Point"
BUTTON_RUN_POINTS = "Run Points"
BUTTON_DISCARD_POINTS = "Discard Points"
BUTTON_ADD_FOREGROUND_SCRIBBLE = "Add Foreground Scribble"
BUTTON_ADD_BACKGROUND_SCRIBBLE = "Add Background Scribble"
BUTTON_RUN_SCRIBBLES = "Run Scribbles"
BUTTON_DISCARD_SCRIBBLES = "Discard Scribbles"
BUTTON_CANCEL = "Cancel"
BUTTON_DISCARD_SESSION = "Discard AI Session"
BUTTON_RETRY = "Retry Prediction"
BUTTON_START_CURRENT = "Start From Current Mask"
BUTTON_START_NEW_MODEL = "Start New Model Session"
BUTTON_KEEP_CURRENT_MODEL = "Keep Current Session"

PROMPT_MASK_PREFIX = "nnInteractive Prompt"
DEFAULT_RESULT_NAME = "nnInteractive Result"
ASYNC_JOB_METADATA = "nninteractive.async_job_path"
MODEL_SOURCE_METADATA = "nninteractive.model_source"
MODEL_PROFILE_METADATA = "nninteractive.model_profile_id"
MODEL_ID_METADATA = "nninteractive.model_id"
MODEL_SHA256_METADATA = "nninteractive.checkpoint_sha256"
TASK_ID_METADATA = "nninteractive.task_id"
DRAFT_ROLE_METADATA = "nninteractive.role"
DRAFT_SOURCE_GUID_METADATA = "nninteractive.source_mask_guid"
DRAFT_SOURCE_NAME_METADATA = "nninteractive.source_mask_name"
DRAFT_SOURCE_SHA256_METADATA = "nninteractive.source_mask_sha256"
DRAFT_WRITE_MODE_METADATA = "nninteractive.write_mode"
DRAFT_ROLE_VALUE = "ai_draft"
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_KIND_METADATA = "mimics_script.source_image_kind"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_IMAGE_INDEX_SPACE_METADATA = "mimics_script.source_image_index_space"
SOURCE_IMAGE_MODALITY_METADATA = "mimics_script.source_image_modality"
SOURCE_INTENSITY_ENCODING_METADATA = "mimics_script.source_intensity_encoding"
SOURCE_INTENSITY_RESCALE_SLOPE_METADATA = "mimics_script.source_intensity_rescale_slope"
SOURCE_INTENSITY_RESCALE_INTERCEPT_METADATA = "mimics_script.source_intensity_rescale_intercept"
SOURCE_INTENSITY_VALUE_MIN_METADATA = "mimics_script.source_intensity_value_min"
SOURCE_INTENSITY_VALUE_MAX_METADATA = "mimics_script.source_intensity_value_max"
DICOM_STORED_VALUE_MIN_METADATA = "mimics_script.dicom_stored_value_min"
DICOM_STORED_VALUE_MAX_METADATA = "mimics_script.dicom_stored_value_max"
SOURCE_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.source_world_coordinate_system"
MIMICS_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.mimics_world_coordinate_system"
SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA = "mimics_script.source_to_mimics_world_matrix"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA = "mimics_script.mimics_to_source_index_matrix"
SOURCE_CASE_DIR_METADATA = "mimics_script.source_case_dir"
_ASYNC_VISUAL_OBJECTS = {}  # job_dir -> list of Mimics objects to delete after inference
_RUNTIME_PROBE_CACHE = {}
_DICOM_TAG_CACHE = {}
_ASYNC_MONITORS = {}
_ASYNC_IMAGE_WORKERS = {}  # official: image guid; task models: image guid + model fingerprint
LOG_ROTATE_BYTES = 10 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3


_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs
_write_json_atomic = runtime_common.write_json_atomic
_read_json = runtime_common.read_json


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


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("worklist_manifest.json", "nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def _integration_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _environment_root():
    # The environment may be inside the worklist, beside it, under the project
    # root, or be the standalone bundle root itself.
    found = runtime_common.find_external_python(_project_root())
    if found:
        return os.path.dirname(os.path.abspath(found))
    root = _project_root()
    return os.path.join(root, "python_env")


def _environment_python_candidates(env_root):
    return [
        os.path.join(env_root, "python.exe"),
        os.path.join(env_root, "Scripts", "python.exe"),
        os.path.join(env_root, "python", "python.exe"),
    ]


def _existing_environment_python(env_root):
    for candidate in _environment_python_candidates(env_root):
        if os.path.isfile(candidate):
            return candidate
    return _environment_python_candidates(env_root)[0]


def _mimics_bridge_paths(config):
    integration_root = _integration_root()
    # Canonical discovery (env overrides, python_env/nninteractive_env layouts).
    python_exe = runtime_common.find_external_python(_project_root())
    if not python_exe:
        # Fall back to the legacy candidate list so the error message still
        # lists every location that was checked.
        env_root = _environment_root()
        python_exe = _first_existing_file(
            _environment_python_candidates(env_root) + [
                os.environ.get("MIMICS_BRIDGE_PYTHON", ""),
                os.environ.get("NNINTERACTIVE_PYTHON", ""),
                config.get("python", ""),
                _existing_environment_python(env_root),
            ],
            "Mimics bridge Python",
        )
    bridge_script = _first_existing_file(
        [
            os.environ.get("MIMICS_BRIDGE_SCRIPT", ""),
            os.path.join(integration_root, "mimics_bridge.py"),
            os.path.join(_project_root(), "mimics_bridge.py"),
        ],
        "mimics_bridge.py",
    )
    return python_exe, bridge_script


def _resource_lock_dir():
    return runtime_common.resource_lock_dir(_project_root())


def _aggressive_auto_cleanup_enabled():
    return runtime_common.aggressive_auto_cleanup_enabled()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _config():
    path = os.environ.get("NNINTERACTIVE_CONFIG", "")
    candidates = [
        path,
        os.path.join(_integration_root(), "nninteractive_config.json"),
        os.path.join(_project_root(), "nninteractive_config.json"),
    ]
    path = next((item for item in candidates if item and os.path.isfile(item)), candidates[-1])
    value = _load_json(path) if os.path.isfile(path) else {}
    value["_config_path"] = path
    return value


def _first_existing_file(candidates, description):
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    checked = [str(item) for item in candidates if item]
    raise RuntimeError("{0} not found. Checked:\n{1}".format(description, "\n".join(checked)))


def _first_existing_dir(candidates, description):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    checked = [str(item) for item in candidates if item]
    raise RuntimeError("{0} not found. Checked:\n{1}".format(description, "\n".join(checked)))


def _model_folds(model_dir):
    result = []
    for name in sorted(os.listdir(model_dir)):
        checkpoint = os.path.join(model_dir, name, "checkpoint_final.pth")
        if name.startswith("fold_") and os.path.isfile(checkpoint):
            result.append(name[5:])
    return result


def _runtime_work_dir(config=None, model_dir=None):
    config = config or {}
    configured = (
        os.environ.get("NNINTERACTIVE_RUNTIME_DIR", "")
        or config.get("runtime_work_dir", "")
        or config.get("work_dir", "")
    )
    if configured:
        root = os.path.abspath(os.path.expandvars(os.path.expanduser(str(configured))))
    else:
        root = os.path.join(_project_root(), ".mimics_runtime", "nninteractive")
    if not os.path.isdir(root):
        os.makedirs(root)
    return root


def _runtime_log_path(model_dir, config=None):
    log_dir = os.path.join(_runtime_work_dir(config, model_dir), "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, "nninteractive_mimics.log")


def _rotate_log_file(path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_BACKUPS):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < max_bytes:
            return
        backups = int(backups)
        if backups <= 0:
            os.remove(path)
            return
        oldest = "{0}.{1}".format(path, backups)
        if os.path.isfile(oldest):
            os.remove(oldest)
        for index in range(backups - 1, 0, -1):
            src = "{0}.{1}".format(path, index)
            dst = "{0}.{1}".format(path, index + 1)
            if os.path.isfile(src):
                os.rename(src, dst)
        os.rename(path, path + ".1")
    except Exception:
        pass


def _append_runtime_log(path, event, details=None):
    _rotate_log_file(path)
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
    }
    if details:
        payload.update(details)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _sha256_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _cleanup_cache_entries(root, retention_days, max_entries):
    if not os.path.isdir(root):
        return 0
    now = time.time()
    cutoff = now - max(1, int(retention_days)) * 86400
    entries = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        try:
            modified = os.path.getmtime(path)
        except OSError:
            modified = 0.0
        entries.append((modified, path))
    entries.sort(reverse=True)
    removed = 0
    for index, (modified, path) in enumerate(entries):
        if modified >= cutoff and index < max(1, int(max_entries)):
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except Exception:
            pass
    return removed


def _object_id(obj):
    return str(getattr(obj, "guid", "") or getattr(obj, "name", ""))


def _metadata_get(obj, name, default=None):
    try:
        item = obj.metadata.find(name)
        return item.value if item is not None else default
    except Exception:
        try:
            return obj.metadata[name].value
        except Exception:
            return default


def _finite_float(value):
    try:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    except Exception:
        return None


def _image_dicom_tags(image):
    cache_key = _object_id(image)
    if cache_key and cache_key in _DICOM_TAG_CACHE:
        return _DICOM_TAG_CACHE[cache_key]
    try:
        try:
            tags = image.get_dicom_tags(0)
        except TypeError:
            tags = image.get_dicom_tags()
        tags = tags or {}
    except Exception:
        tags = {}
    if not cache_key:
        return tags
    if len(_DICOM_TAG_CACHE) >= 16:
        try:
            del _DICOM_TAG_CACHE[next(iter(_DICOM_TAG_CACHE))]
        except Exception:
            _DICOM_TAG_CACHE.clear()
    _DICOM_TAG_CACHE[cache_key] = tags
    return tags


def _dicom_tag_value(image, group, element):
    try:
        item = _image_dicom_tags(image).get((int(group), int(element)))
        value = getattr(item, "value", item)
        return value.split("\\", 1)[0].strip() if isinstance(value, str) else value
    except Exception:
        return None


def _dicom_tag_float(image, group, element):
    return _finite_float(_dicom_tag_value(image, group, element))


def _image_modality(image):
    modality = str(
        _metadata_get(image, SOURCE_IMAGE_MODALITY_METADATA, "") or ""
    ).strip().upper()
    if modality:
        return modality
    return str(_dicom_tag_value(image, 0x0008, 0x0060) or "").strip().upper()


def _source_intensity_mapping(image, modality=None):
    """Return a portable MR GV-to-source mapping when one is recoverable."""
    modality = str(modality or _image_modality(image) or "").strip().upper()
    if modality != "MR":
        return None

    encoding = str(
        _metadata_get(image, SOURCE_INTENSITY_ENCODING_METADATA, "") or ""
    ).strip()
    slope = _finite_float(
        _metadata_get(image, SOURCE_INTENSITY_RESCALE_SLOPE_METADATA, None)
    )
    intercept = _finite_float(
        _metadata_get(image, SOURCE_INTENSITY_RESCALE_INTERCEPT_METADATA, None)
    )
    basis = "mcs_metadata"
    if slope is None or intercept is None:
        # Older projects normally retain the source DICOM tags even though the
        # Mimics-Script metadata predates the portable intensity contract.
        slope = _dicom_tag_float(image, 0x0028, 0x1053)
        intercept = _dicom_tag_float(image, 0x0028, 0x1052)
        basis = "mcs_dicom_tags"
        if not encoding and slope is not None and intercept is not None:
            encoding = "dicom_uniform_rescale_v1"
    if slope is None or intercept is None or abs(float(slope)) <= 1.0e-12:
        return None
    source_kind = str(
        _metadata_get(image, SOURCE_IMAGE_KIND_METADATA, "") or ""
    ).strip().lower()
    restores_quantized_source = (
        encoding == "dicom_uint16_linear_rescale_v1"
        or (basis == "mcs_dicom_tags" and source_kind in ("nifti", "medical_image"))
    )
    return {
        "encoding": encoding or "dicom_uniform_rescale_v1",
        "basis": basis,
        "slope": float(slope),
        "intercept": float(intercept),
        "zero_tolerance": (
            abs(float(slope)) * 0.500001 if restores_quantized_source else None
        ),
        "source_value_min": _finite_float(
            _metadata_get(image, SOURCE_INTENSITY_VALUE_MIN_METADATA, None)
        ),
        "source_value_max": _finite_float(
            _metadata_get(image, SOURCE_INTENSITY_VALUE_MAX_METADATA, None)
        ),
        "stored_value_min": _finite_float(
            _metadata_get(image, DICOM_STORED_VALUE_MIN_METADATA, None)
        ),
        "stored_value_max": _finite_float(
            _metadata_get(image, DICOM_STORED_VALUE_MAX_METADATA, None)
        ),
    }


def _metadata_set(obj, name, value):
    text = "" if value is None else str(value)
    item = None
    try:
        item = obj.metadata.find(name)
    except Exception:
        pass
    if item is None:
        obj.metadata.create(name=name, value=text)
    else:
        item.value = text


def _metadata_delete(obj, name):
    try:
        obj.metadata.delete(name)
        return
    except Exception:
        pass
    try:
        item = obj.metadata.find(name)
        if item is not None:
            item.value = ""
    except Exception:
        pass


def _process_exists(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False

    if pid <= 0:
        return False

    # Delegate to runtime_common.process_exists, which declares pointer-sized
    # HANDLE ctypes signatures so the 64-bit process handle is not truncated to
    # c_int (which intermittently reports a live process as dead). It uses the
    # WinAPI path on Windows, avoiding the Mimics-embedded-CPython os.kill
    # SystemError noted historically.
    return runtime_common.process_exists(pid)


def _server_state_candidates():
    candidates = []
    try:
        config = _config()
    except Exception:
        config = {}
    root = _project_root()
    environment_root = _environment_root()
    runtime_dir = _runtime_work_dir(config)
    runtime_state = os.path.join(runtime_dir, ".nninteractive_server.json")
    candidates.append(runtime_state)
    model_dirs = [
        os.environ.get("NNINTERACTIVE_MODEL_DIR", ""),
        config.get("model_dir", ""),
        os.path.join(environment_root, "models", "nnInteractive_v1.0"),
        os.path.join(root, "python_env", "models", "nnInteractive_v1.0"),
        os.path.join(root, "nninteractive_env", "models", "nnInteractive_v1.0"),
    ]
    seen = set()
    for model_dir in model_dirs:
        if not model_dir:
            continue
        state_path = os.path.join(os.path.dirname(os.path.abspath(model_dir)), ".nninteractive_server.json")
        if state_path not in seen:
            seen.add(state_path)
            candidates.append(state_path)
    return candidates


def _release_state_lock(state):
    try:
        lock_path = state.get("gpu_lock_path")
        token = state.get("gpu_lock_token")
        if lock_path and token:
            runtime_common.release_resource_lock(lock_path, token)
    except Exception:
        pass


def _remove_state_file(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return not os.path.exists(path)


def _remove_owned_state_file(path, state):
    """Remove only the state record that belongs to *state*."""
    current = runtime_common.read_json(path, {}) or {}
    expected = str(state.get("ownership_token") or "")
    if current and expected and str(current.get("ownership_token") or "") != expected:
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return not os.path.exists(path)


def _cleanup_stale_owned_servers(records):
    if os.name != "nt":
        return []
    by_pid = {}
    for record in records or []:
        try:
            pid = int(record.get("ProcessId", 0))
        except (TypeError, ValueError):
            continue
        by_pid[pid] = str(record.get("CommandLine") or "")

    killed = []
    now = time.time()
    for state_path in _server_state_candidates():
        state = runtime_common.read_json(state_path, {}) or {}
        if state.get("schema_version") != "nninteractive_owned_server.v2":
            continue
        try:
            pid = int(state.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        if not pid or not _process_exists(pid):
            if _remove_owned_state_file(state_path, state):
                _release_state_lock(state)
            continue

        command_line = by_pid.get(pid, "").lower()
        model_dir = str(state.get("model_dir", "")).lower()
        ownership = str(state.get("ownership_token", "")).lower()
        if (
            "nninteractive.inference.server.main" not in command_line
            or (model_dir and model_dir not in command_line)
            or (ownership and ownership not in command_line)
        ):
            continue

        try:
            watchdog_pid = int(state.get("watchdog_pid", 0) or 0)
        except (TypeError, ValueError):
            watchdog_pid = 0
        watchdog_command = by_pid.get(watchdog_pid, "").lower() if watchdog_pid else ""
        watchdog_alive = watchdog_pid and _process_exists(watchdog_pid) and "--watchdog" in watchdog_command
        if watchdog_alive:
            continue

        try:
            idle_timeout = float(state.get("service_idle_timeout_seconds", 1800))
            last_activity = float(state.get("last_activity_epoch", now))
        except Exception:
            idle_timeout = 1800.0
            last_activity = now
        if now - last_activity < idle_timeout + 60.0:
            continue

        try:
            killer = subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_hidden_process_kwargs()
            )
            try:
                killer.wait(timeout=15.0)
            except Exception:
                pass
        except Exception:
            continue
        deadline = time.time() + 10.0
        while _process_exists(pid) and time.time() < deadline:
            time.sleep(0.25)
        if _process_exists(pid):
            continue
        killed.append(pid)
        if _remove_owned_state_file(state_path, state):
            _release_state_lock(state)
    return killed


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _probe_python(python_exe, timeout):
    """Check that the external Python has the required packages.

    The probe must not block the Mimics GUI.  Start it on a daemon thread and
    return an "unknown/pending" result immediately; the bridge process itself
    will surface real environment problems when it starts.
    """
    cache_key = (os.path.abspath(python_exe), int(timeout))
    cached = _RUNTIME_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = {"python": python_exe, "version": [], "missing": [], "pending": True}
    _RUNTIME_PROBE_CACHE[cache_key] = result
    probe = (
        "import importlib.util,json,sys;"
        "mods=['numpy','nibabel','torch','nnInteractive'];"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "print(json.dumps({"
        "'python':sys.executable,"
        "'version':list(sys.version_info[:3]),"
        "'missing':missing"
        "}))"
    )

    def _run_probe():
        try:
            process = subprocess.Popen(
                [python_exe, "-c", probe],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_hidden_process_kwargs()
            )
            try:
                stdout, stderr = process.communicate(timeout=min(timeout, 30))
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                _RUNTIME_PROBE_CACHE[cache_key] = {
                    "python": python_exe,
                    "version": [],
                    "missing": [],
                    "timed_out": True,
                }
                return
            if process.returncode != 0:
                _RUNTIME_PROBE_CACHE[cache_key] = {
                    "python": python_exe,
                    "version": [],
                    "missing": [],
                    "failed": True,
                    "stderr": stderr.decode("utf-8", "replace").strip()[:500],
                }
                return
            try:
                _RUNTIME_PROBE_CACHE[cache_key] = json.loads(stdout.decode("utf-8"))
            except ValueError:
                _RUNTIME_PROBE_CACHE[cache_key] = {"python": python_exe, "version": [], "missing": []}
        except Exception as exc:
            _RUNTIME_PROBE_CACHE[cache_key] = {
                "python": python_exe,
                "version": [],
                "missing": [],
                "failed": True,
                "error": str(exc),
            }

    thread = threading.Thread(target=_run_probe)
    thread.daemon = True
    thread.start()
    return result


def _runtime_paths(config):
    root = _project_root()
    integration_root = _integration_root()
    environment_root = _environment_root()
    python_candidates = _environment_python_candidates(environment_root) + [
        os.environ.get("NNINTERACTIVE_PYTHON", ""),
        config.get("python", ""),
    ]
    python_exe = (
        runtime_common.find_external_python(root)
        or _first_existing_file(python_candidates, "nnInteractive Python")
    )
    bridge_script = _first_existing_file(
        [
            os.environ.get("NNINTERACTIVE_BRIDGE", ""),
            config.get("bridge_script", ""),
            os.path.join(integration_root, "nninteractive_bridge.py"),
            os.path.join(integration_root, "bridge", "nninteractive_bridge.py"),
        ],
        "nnInteractive bridge script",
    )
    profile = _model_profile(config)
    if profile.get("source") == "task_model":
        model_dir = _first_existing_dir(
            [profile.get("model_dir", "")],
            "nnInteractive task model directory",
        )
    else:
        model_dir = _first_existing_dir(
            [
                os.environ.get("NNINTERACTIVE_MODEL_DIR", ""),
                config.get("model_dir", ""),
                os.path.join(environment_root, "models", "nnInteractive_v1.0"),
                os.path.join(root, "python_env", "models", "nnInteractive_v1.0"),
                os.path.join(root, "nninteractive_env", "models", "nnInteractive_v1.0"),
            ],
            "nnInteractive model directory",
        )
    folds = _model_folds(model_dir)
    if not folds:
        raise RuntimeError(
            "The nnInteractive model directory has no usable checkpoint.\n"
            "Expected fold_*/checkpoint_final.pth under:\n{0}".format(model_dir)
        )
    probe_timeout = int(config.get("environment_probe_timeout_seconds", 180))
    probe = _probe_python(python_exe, probe_timeout)
    return python_exe, bridge_script, model_dir, probe, folds


def _model_profile(config):
    """Return a stable, explicit model identity for this Mimics session."""
    raw = (config or {}).get("_model_profile") or {}
    if not isinstance(raw, dict) or str(raw.get("source") or "").lower() != "task_model":
        return {
            "source": "official",
            "profile_id": "official",
            "model_id": "official",
            "checkpoint_sha256": "",
            "task_id": "",
            "task_name": "",
            "model_dir": "",
            "validated_prompt_types": [
                "point",
                "scribble",
                "box",
                "lasso",
            ],
            "effective_model_fingerprint": "",
        }
    raw_model_dir = str(raw.get("model_dir") or "").strip()
    profile = {
        "source": "task_model",
        "profile_id": str(raw.get("profile_id") or raw.get("model_id") or "").strip(),
        "model_id": str(raw.get("model_id") or "").strip(),
        "checkpoint_sha256": str(raw.get("checkpoint_sha256") or "").strip().lower(),
        "task_id": str(raw.get("task_id") or "").strip(),
        "task_name": str(raw.get("task_name") or "").strip(),
        "model_dir": os.path.abspath(raw_model_dir) if raw_model_dir else "",
        "validated_prompt_types": list(
            raw.get("validated_prompt_types") or ["point"]
        ),
        "effective_model_fingerprint": str(
            raw.get("effective_model_fingerprint") or ""
        ),
        "input_contract": raw.get("input_contract") or {
            "schema_version": "nninteractive_input_contract.v1",
            "spatial_orientation": "canonical_ras",
            "source_grid_policy": "mimics_dicom_compatible_minimal_resample",
            "intensity_space": "source_physical_values",
            "dicom_rescale": "apply_rescale_slope_and_intercept",
            "normalization": "nonzero_spatial_bbox_zscore",
            "normalization_channel": 0,
            "standard_deviation_correction": 1,
        },
    }
    if not profile["profile_id"] or not profile["model_id"] or not profile["model_dir"]:
        raise RuntimeError(
            "The selected nnInteractive task model is incomplete. "
            "Model ID, profile ID, and model directory are required."
        )
    contract = profile.get("input_contract") or {}
    if (
        contract.get("schema_version") != "nninteractive_input_contract.v1"
        or contract.get("source_grid_policy")
        != "mimics_dicom_compatible_minimal_resample"
        or contract.get("intensity_space") != "source_physical_values"
        or contract.get("normalization") != "nonzero_spatial_bbox_zscore"
    ):
        raise RuntimeError(
            "The selected nnInteractive task model uses an unsupported input "
            "preprocessing contract."
        )
    return profile


def _model_identity(profile):
    profile = profile or {}
    if str(profile.get("source") or "") != "task_model":
        return "official"
    contract = profile.get("input_contract") or {}
    return "{0}|{1}|{2}|{3}".format(
        profile.get("profile_id") or profile.get("model_id") or "",
        profile.get("model_id") or "",
        profile.get("checkpoint_sha256") or "",
        contract.get("schema_version") or "",
    )


def _task_model_image_input_mode(config):
    """Return the task-model input policy without changing official inference."""
    value = str(
        (config or {}).get("task_model_image_input_mode", "auto") or "auto"
    ).strip().lower()
    aliases = {
        "original": "source",
        "original_image": "source",
        "source_image": "source",
        "buffer": "mimics",
        "mimics_buffer": "mimics",
    }
    value = aliases.get(value, value)
    if value not in ("auto", "source", "mimics"):
        raise RuntimeError(
            "Unsupported task_model_image_input_mode: {0}. Use auto, source, or mimics.".format(
                value
            )
        )
    return value


def _effective_image_input_mode(config):
    if _model_profile(config).get("source") == "task_model":
        return _task_model_image_input_mode(config)
    return str((config or {}).get("image_input_mode", "") or "").strip().lower()


def _is_network_source_path(path):
    text = str(path or "").strip().replace("\\", "/")
    return text.startswith("//")


def _worker_cache_key(config, image):
    image_key = _object_id(image)
    identity = _model_identity(_model_profile(config))
    # Preserve the historical official-model cache key for compatibility with
    # existing sessions and fake-Mimics tests.
    if identity == "official":
        return image_key
    return image_key + "::" + identity


def _state_model_matches(state, profile):
    expected = _model_identity(profile)
    actual = str(state.get("model_identity") or "official")
    return actual == expected


def _model_state_values(profile):
    return {
        "model_source": profile.get("source") or "official",
        "model_profile_id": profile.get("profile_id") or "official",
        "model_id": profile.get("model_id") or "official",
        "checkpoint_sha256": profile.get("checkpoint_sha256") or "",
        "effective_model_fingerprint": profile.get(
            "effective_model_fingerprint"
        ) or "",
        "task_id": profile.get("task_id") or "",
        "task_name": profile.get("task_name") or "",
        "input_contract": profile.get("input_contract") or {},
        "validated_prompt_types": list(
            profile.get("validated_prompt_types") or []
        ),
        "effective_model_fingerprint": profile.get(
            "effective_model_fingerprint"
        ) or "",
        "model_identity": _model_identity(profile),
    }


def _same_object(left, right):
    if left is right:
        return True
    return bool(
        left is not None
        and right is not None
        and getattr(left, "guid", None)
        and getattr(left, "guid", None) == getattr(right, "guid", None)
    )


def _mask_is_prompt(mask):
    return str(getattr(mask, "name", "")).startswith(PROMPT_MASK_PREFIX)


def _masks_for_image(image):
    result = []
    for mask in mimics.data.masks:
        if _mask_is_prompt(mask):
            continue
        mask_image = getattr(mask, "image", None)
        if mask_image is None or _same_object(mask_image, image):
            result.append(mask)
    return result


def _unique_mask_name(base_name):
    existing = set(str(getattr(mask, "name", "")) for mask in mimics.data.masks)
    if base_name not in existing:
        return base_name
    index = 2
    while "{0} {1}".format(base_name, index) in existing:
        index += 1
    return "{0} {1}".format(base_name, index)


def _unique_result_name():
    return _unique_mask_name(DEFAULT_RESULT_NAME)


def _create_result_mask(image, name=None):
    mask = mimics.segment.create_mask()
    mask.name = _unique_mask_name(name) if name else _unique_result_name()
    try:
        mask.image = image
    except Exception as error:
        # Some Mimics versions expose Mask.image as read-only and bind the new
        # mask to the active image automatically inside create_mask(). If the
        # mask is already bound to the requested image, the assignment is
        # unnecessary and we can continue. Only fail when it is genuinely
        # unbound or bound to a different image.
        bound_image = None
        try:
            bound_image = getattr(mask, "image", None)
        except Exception:
            bound_image = None
        already_bound = bound_image is not None and _same_object(bound_image, image)
        if not already_bound:
            try:
                mimics.data.masks.delete(mask)
            except Exception:
                pass
            raise RuntimeError("Could not bind the new result Mask to the active image: {0}".format(error))
    mask.visible = True
    mask.selected = True
    return mask


def _mask_is_empty(mask):
    return int(getattr(mask, "number_of_pixels", 0) or 0) <= 0


def _is_ai_draft(mask):
    return str(_metadata_get(mask, DRAFT_ROLE_METADATA, "") or "") == DRAFT_ROLE_VALUE


def _mark_ai_draft(target, source=None):
    _metadata_set(target, DRAFT_ROLE_METADATA, DRAFT_ROLE_VALUE)
    _metadata_set(target, DRAFT_WRITE_MODE_METADATA, "derived_copy" if source is not None else "new_empty")
    if source is not None:
        _metadata_set(target, DRAFT_SOURCE_GUID_METADATA, _object_id(source))
        _metadata_set(target, DRAFT_SOURCE_NAME_METADATA, str(getattr(source, "name", "")))


def _delete_unused_auto_draft(target, auto_created, source=None):
    if not auto_created or target is None or not _mask_is_empty(target):
        return
    if _metadata_get(target, ASYNC_JOB_METADATA, ""):
        return
    try:
        mimics.data.masks.delete(target)
    except Exception as error:
        _mimics_log(
            logging.WARNING,
            "Could not remove unused nnInteractive Draft {0}: {1}".format(
                getattr(target, "name", ""), error
            ),
        )
        return
    if source is not None and source is not target:
        try:
            source.selected = True
        except Exception:
            pass


def _select_session_masks(image, config, source_override=None):
    """Resolve immutable session source and mutable result target.

    Non-empty manual masks create a Draft by default. Empty masks, existing
    Drafts, and masks with an active legacy session continue in place.
    """
    if source_override is None:
        selected = [
            mask
            for mask in _masks_for_image(image)
            if bool(getattr(mask, "selected", False))
        ]
    else:
        selected = [source_override]
    if len(selected) > 1:
        raise RuntimeError("Select exactly one source or AI Draft Mask, then run nnInteractive again.")
    if not selected:
        target = _create_result_mask(image)
        _mark_ai_draft(target)
        return {
            "source": target,
            "target": target,
            "auto_created": True,
            "write_mode": "new_empty",
        }

    source = selected[0]
    try:
        source_image = getattr(source, "image", None)
    except Exception:
        source_image = None
    if source_image is None:
        try:
            source.image = image
        except Exception as error:
            # If the assignment is rejected because .image is read-only but the
            # mask is in fact already bound to the requested image, treat it as
            # success instead of failing the whole run.
            try:
                bound_image = getattr(source, "image", None)
            except Exception:
                bound_image = None
            if bound_image is None or not _same_object(bound_image, image):
                raise RuntimeError("The selected Mask is not bound to the active image: {0}".format(error))
    elif not _same_object(source_image, image):
        raise RuntimeError(
            "The selected Mask belongs to a different image set. "
            "Activate its image set or select another Mask."
        )

    has_active_session = bool(_metadata_get(source, ASYNC_JOB_METADATA, ""))
    if _mask_is_empty(source) or _is_ai_draft(source) or has_active_session:
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "in_place",
        }

    mode = str(config.get("existing_mask_result_mode", "ask") or "ask").strip().lower()
    if mode in ("ask", "choose", "prompt"):
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "choose_on_first_result",
        }

    if mode in ("in_place", "inplace", "overwrite"):
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "in_place",
        }

    draft_name = "{0} - AI Draft".format(str(getattr(source, "name", "") or "nnInteractive"))
    target = _create_result_mask(image, draft_name)
    _mark_ai_draft(target, source)
    try:
        source.selected = False
    except Exception:
        pass
    return {
        "source": source,
        "target": target,
        "auto_created": True,
        "write_mode": "derived_copy",
    }


def _buffer_dtype(view):
    fmt = str(getattr(view, "format", ""))
    if fmt in ("b", "<b", "=b"):
        return "int8"
    if fmt in ("B", "<B", "=B"):
        return "uint8"
    if fmt in ("h", "<h", "=h"):
        return "int16"
    if fmt in ("H", "<H", "=H"):
        return "uint16"
    if fmt in ("i", "<i", "=i"):
        return "int32"
    if fmt in ("I", "<I", "=I"):
        return "uint32"
    if fmt in ("f", "<f", "=f"):
        return "float32"
    raise RuntimeError(
        "Unsupported Mimics image buffer format {0!r}. "
        "Do not guess the voxel dtype; record this format during workstation validation.".format(fmt)
    )


def _image_shape(image):
    dims = getattr(image, "logical_dimensions", None)
    if dims is not None:
        try:
            shape = [int(dims[0]), int(dims[1]), int(dims[2])]
            if shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
                return shape
        except Exception:
            pass
    view = image.get_voxel_buffer()
    return [int(value) for value in view.shape]


def _parse_shape_metadata(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        shape = [int(loaded[0]), int(loaded[1]), int(loaded[2])]
        if shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            return shape
    except Exception:
        pass
    for sep in ("x", ",", ";", " "):
        if sep in text:
            parts = [part for part in text.replace("[", "").replace("]", "").split(sep) if part]
            if len(parts) == 3:
                try:
                    shape = [int(parts[0]), int(parts[1]), int(parts[2])]
                    if shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
                        return shape
                except Exception:
                    pass
    return None


def _parse_matrix_metadata(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        matrix = json.loads(text)
        if len(matrix) != 4:
            return None
        parsed = []
        for row in matrix:
            if len(row) != 4:
                return None
            parsed.append([float(item) for item in row])
        return parsed
    except Exception:
        return None


def _image_point_values(point):
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [
                float(getattr(point, names[0])),
                float(getattr(point, names[1])),
                float(getattr(point, names[2])),
            ]
        except Exception:
            pass
    return [float(point[0]), float(point[1]), float(point[2])]


def _image_voxel_center(image, index):
    getter = getattr(image, "get_voxel_center", None)
    if not callable(getter):
        raise RuntimeError("Mimics image does not expose get_voxel_center().")
    values = [int(value) for value in index]
    try:
        return _image_point_values(getter(values))
    except TypeError:
        pass
    try:
        return _image_point_values(getter(tuple(values)))
    except TypeError:
        pass
    return _image_point_values(getter(values[0], values[1], values[2]))


def _derive_image_voxel_to_ras_matrix(image, image_shape):
    """Derive the active image voxel grid from Mimics LPS voxel centers."""
    if not image_shape:
        return None
    try:
        origin_lps = _image_voxel_center(image, [0, 0, 0])
        origin = [
            -float(origin_lps[0]),
            -float(origin_lps[1]),
            float(origin_lps[2]),
        ]
        matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
        for axis in range(3):
            if int(image_shape[axis]) > 1:
                index = [0, 0, 0]
                index[axis] = 1
                point_lps = _image_voxel_center(image, index)
                point = [
                    -float(point_lps[0]),
                    -float(point_lps[1]),
                    float(point_lps[2]),
                ]
                vector = [point[row] - origin[row] for row in range(3)]
            else:
                vector = [0.0, 0.0, 0.0]
                vector[axis] = 1.0
            for row in range(3):
                matrix[row][axis] = vector[row]
        for row in range(3):
            matrix[row][3] = origin[row]
        matrix[3] = [0.0, 0.0, 0.0, 1.0]
        return matrix
    except Exception:
        return None


def _matrix_close(value, expected, tolerance=1.0e-6):
    matrix = _parse_matrix_metadata(value)
    if matrix is None:
        return False
    for row_index in range(4):
        for col_index in range(4):
            if abs(float(matrix[row_index][col_index]) - float(expected[row_index][col_index])) > tolerance:
                return False
    return True


def _hu_to_mimics_gv_transform():
    try:
        gv0 = float(mimics.segment.HU2GV(0))
        gv1 = float(mimics.segment.HU2GV(1))
        slope = gv1 - gv0
        if abs(slope) <= 1.0e-12:
            return None
        return slope, gv0
    except Exception:
        return None


def _current_project_path():
    try:
        info = mimics.file.get_project_information()
    except Exception:
        return ""
    for attr in (
        "filename",
        "file_name",
        "path",
        "project_path",
        "project_file",
    ):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if value:
            return os.path.abspath(str(value))
    return ""


def _same_project_path(left, right):
    if not left or not right:
        return not left and not right
    try:
        return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
            os.path.abspath(str(right))
        )
    except Exception:
        return False


def _relocated_source_image_path():
    project_path = _current_project_path()
    if not project_path:
        return ""
    case_name = os.path.basename(project_path)
    case_id = (
        case_name[:-4]
        if case_name.lower().endswith(".mcs")
        else case_name
    )
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


def _source_uses_hu_to_gv(kind, modality):
    modality_text = str(modality or "").strip().upper()
    if modality_text:
        return modality_text == "CT"
    # Backward compatibility for projects created before source modality was stored.
    return True




def _call_mimics_bridge(config, payload, timeout_seconds=1800):
    python_exe, bridge_script = _mimics_bridge_paths(config)
    process = subprocess.Popen(
        [python_exe, bridge_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_hidden_process_kwargs()
    )
    try:
        stdout, stderr = process.communicate(
            input=json.dumps(payload).encode("utf-8"),
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("mimics_bridge.py timed out")
    if process.returncode != 0:
        err = stderr.decode("utf-8", "replace") if stderr else ""
        raise RuntimeError("mimics_bridge.py failed (exit {}): {}".format(process.returncode, err))
    try:
        result = json.loads((stdout or b"{}").decode("utf-8", "replace"))
    except Exception:
        raise RuntimeError("mimics_bridge.py returned invalid JSON")
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "mimics_bridge.py returned non-ok status"))
    return result


def _source_image_export(image, config):
    image_input_mode = str(config.get("image_input_mode", "") or "").strip().lower()
    force_source = image_input_mode in ("source", "source_image", "original", "original_image")
    force_mimics_buffer = image_input_mode in ("mimics", "mimics_buffer", "buffer")
    allow_source_fallback = bool(config.get("fallback_to_mimics_buffer_when_source_unavailable", False))
    if force_mimics_buffer or (not force_source and not bool(config.get("prefer_source_image_for_nninteractive", False))):
        _mimics_log(
            logging.INFO,
            "nnInteractive image input mode: Mimics buffer. Source-image fast path is disabled by configuration.",
        )
        return None
    path = _metadata_get(image, SOURCE_IMAGE_PATH_METADATA, "")
    recorded_path = str(path or "").strip()
    relocated_path = _relocated_source_image_path()
    if relocated_path:
        path = relocated_path
        if (
            not recorded_path
            or os.path.normcase(os.path.abspath(recorded_path))
            != os.path.normcase(os.path.abspath(relocated_path))
        ):
            _mimics_log(
                logging.INFO,
                "nnInteractive resolved the moved source image from "
                "dataset_manifest.json: {0}".format(relocated_path),
            )
    kind = str(_metadata_get(image, SOURCE_IMAGE_KIND_METADATA, "") or "").lower()
    modality = str(_metadata_get(image, SOURCE_IMAGE_MODALITY_METADATA, "") or "").upper()
    index_space = str(_metadata_get(image, SOURCE_IMAGE_INDEX_SPACE_METADATA, "") or "")
    source_world = str(_metadata_get(image, SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, "") or "").lower()
    mimics_world = str(_metadata_get(image, MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, "") or "").lower()
    source_to_mimics_world = str(_metadata_get(image, SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA, "") or "")
    source_voxel_to_ras = str(_metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, "") or "")
    mimics_voxel_to_ras = str(_metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "") or "")
    mimics_to_source_index = str(_metadata_get(image, MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA, "") or "")
    source_case_dir = str(_metadata_get(image, SOURCE_CASE_DIR_METADATA, "") or "")
    ras_to_lps = [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    has_ras_affines = (
        _parse_matrix_metadata(source_voxel_to_ras) is not None
        and _parse_matrix_metadata(mimics_voxel_to_ras) is not None
    )
    if not path:
        return None
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
    if (
        _is_network_source_path(path)
        and allow_source_fallback
        and not force_source
        and bool(config.get("avoid_sync_network_source_access", True))
    ):
        _mimics_log(
            logging.INFO,
            "nnInteractive skipped synchronous access to the recorded network source path. "
            "The portable Mimics image buffer will be used instead: {0}".format(path),
        )
        return None
    lower_path = path.lower()
    derived_resampled_index_spaces = (
        "derived_dicom_axial_lps_resampled_from_nifti_v1",
        "derived_dicom_lps_resampled_from_source_image_v2",
    )
    supported_nifti_index_spaces = (
        "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1",
        "derived_dicom_axial_lps_resampled_from_nifti_v1",
        "derived_dicom_lps_resampled_from_source_image_v2",
    )
    supported_medical_index_spaces = (
        "medical_image_ijk_matches_derived_dicom_columns_rows_slices_v1",
        "derived_dicom_lps_resampled_from_source_image_v2",
    )
    is_nifti = kind == "nifti" and (
        index_space in supported_nifti_index_spaces
        and source_world == "ras"
        and mimics_world == "lps"
        and _matrix_close(source_to_mimics_world, ras_to_lps)
        and has_ras_affines
    )
    is_dicom = kind == "dicom_folder" and (
        index_space == "dicom_columns_rows_slices_sorted_by_position_v1"
        and source_world == "lps"
        and mimics_world == "lps"
        and _matrix_close(source_to_mimics_world, identity)
        and has_ras_affines
    )
    is_medical = kind == "medical_image" and (
        index_space in supported_medical_index_spaces
        and source_world == "ras"
        and mimics_world == "lps"
        and _matrix_close(source_to_mimics_world, ras_to_lps)
        and has_ras_affines
    )
    if not is_nifti and not is_dicom and not is_medical:
        _mimics_log(
            logging.INFO,
            "nnInteractive source-image fast path skipped because the stored source geometry is not supported: kind={0}, index_space={1}, source_world={2}, mimics_world={3}.".format(
                kind or "?",
                index_space or "?",
                source_world or "?",
                mimics_world or "?",
            ),
        )
        return None

    if (is_nifti or is_medical) and not os.path.isfile(path):
        message = "nnInteractive source image metadata points to a missing file: {0}".format(path)
        if allow_source_fallback and not force_source:
            _mimics_log(
                logging.WARNING,
                message + ". Falling back to the portable Mimics image buffer.",
            )
            return None
        raise RuntimeError(message + ". Fix the stored source path or use task_model_image_input_mode=\"auto\" for a portable buffer fallback.")
    if is_dicom and not os.path.isdir(path):
        message = "nnInteractive source DICOM metadata points to a missing folder: {0}".format(path)
        if allow_source_fallback and not force_source:
            _mimics_log(
                logging.WARNING,
                message + ". Falling back to the portable Mimics image buffer.",
            )
            return None
        raise RuntimeError(message + ". Fix the stored source path or use task_model_image_input_mode=\"auto\" for a portable buffer fallback.")

    # On-demand mode for derived oblique-NIfTI imports:
    # build a cached axial NIfTI only when nnInteractive is actually used.
    if (
        (is_nifti or is_medical)
        and index_space in derived_resampled_index_spaces
        and not bool(config.get("defer_source_alignment_to_worker", False))
    ):
        case_id = "case"
        if source_case_dir:
            case_id = os.path.basename(os.path.normpath(source_case_dir)) or case_id
        cache_root = os.path.join(_runtime_work_dir(config), "source_fastpath_cache")
        if not os.path.isdir(cache_root):
            os.makedirs(cache_root)
        _cleanup_cache_entries(
            cache_root,
            config.get("source_cache_retention_days", 7),
            config.get("source_cache_max_entries", 12),
        )
        try:
            source_stat = os.stat(path)
            source_signature = "{0}|{1}|{2}|{3}".format(
                path,
                int(source_stat.st_size),
                int(source_stat.st_mtime),
                json.dumps(mimics_voxel_to_ras, sort_keys=True),
            )
        except OSError:
            source_signature = "{0}|{1}".format(path, json.dumps(mimics_voxel_to_ras, sort_keys=True))
        cache_key = hashlib.sha256(source_signature.encode("utf-8")).hexdigest()[:16]
        safe_case_id = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in case_id
        )[:48] or "case"
        cache_path = os.path.join(
            cache_root,
            "{0}_{1}_aligned_source.nii.gz".format(safe_case_id, cache_key),
        )
        if not os.path.isfile(cache_path):
            try:
                _mimics_log(
                    logging.INFO,
                    "nnInteractive source-image fast path (on-demand) is preparing aligned source cache: {0}".format(cache_path),
                )
                _call_mimics_bridge(
                    config,
                    {
                        "action": "prepare_source_fastpath",
                        "image_path": path,
                        "source_nifti_out": cache_path,
                    },
                    timeout_seconds=float(config.get("bridge_timeout_seconds", 1800)),
                )
            except Exception as exc:
                message = "nnInteractive on-demand source cache generation failed: {}".format(exc)
                if allow_source_fallback and not force_source:
                    _mimics_log(
                        logging.WARNING,
                        message + ". Falling back to Mimics image buffer export.",
                    )
                    return None
                raise RuntimeError(message)
        path = os.path.abspath(cache_path)
        # The on-demand cache is generated on the Mimics target grid.
        source_voxel_to_ras = mimics_voxel_to_ras
        index_space = "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1"
    task_model_source_values = (
        _model_profile(config).get("source") == "task_model"
    )
    if _source_uses_hu_to_gv(kind, modality) and task_model_source_values:
        intensity_slope = 1.0
        intensity_intercept = 0.0
        intensity_space = "source_values"
    elif _source_uses_hu_to_gv(kind, modality):
        intensity_transform = _hu_to_mimics_gv_transform()
        if intensity_transform is None and not force_source:
            message = "nnInteractive source-image fast path cannot match Mimics CT intensity because Mimics HU2GV conversion is unavailable."
            if allow_source_fallback:
                _mimics_log(
                    logging.WARNING,
                    message + " Falling back to Mimics image buffer export because fallback_to_mimics_buffer_when_source_unavailable is enabled.",
                )
                return None
            raise RuntimeError(message + " Set image_input_mode to \"mimics\" for an explicit buffer-based run or run from a Mimics environment that exposes HU2GV.")
        if intensity_transform is None and force_source:
            intensity_slope = 1.0
            intensity_intercept = 0.0
            intensity_space = "source_values"
            _mimics_log(
                logging.WARNING,
                "nnInteractive source image was forced, but Mimics HU2GV conversion is unavailable. Source intensity values will be used unchanged.",
            )
        else:
            intensity_slope, intensity_intercept = intensity_transform
            intensity_space = "hu_to_mimics_gv"
    else:
        intensity_slope = 1.0
        intensity_intercept = 0.0
        intensity_space = "source_values"
    shape = _image_shape(image)
    recorded_shape = _parse_shape_metadata(_metadata_get(image, SOURCE_IMAGE_SHAPE_METADATA, ""))
    if recorded_shape is not None and recorded_shape != shape:
        _mimics_log(
            logging.INFO,
            "nnInteractive source image shape differs from the open Mimics image. The bridge will resample source data into the Mimics voxel grid. Source: {0}, Mimics: {1}.".format(
                recorded_shape,
                shape,
            ),
        )
    if is_nifti:
        source_name = "source_nifti_metadata"
        source_kind = "nifti"
        source_label = "NIfTI"
    elif is_medical:
        source_name = "source_medical_image_metadata"
        source_kind = "medical_image"
        source_label = "medical image"
    else:
        source_name = "source_dicom_metadata"
        source_kind = "dicom_folder"
        source_label = "DICOM"
    _mimics_log(
        logging.INFO,
        "nnInteractive image input mode: source image. Mimics image buffer export is skipped. Source: {0}; modality: {1}; intensity: {2}; path: {3}.".format(
            source_label,
            modality or "?",
            intensity_space,
            path,
        ),
    )
    return {
        "kind": "source_image",
        "path": "",
        "shape": shape,
        "dtype": "",
        "sha256": "",
        "image_path": path,
        "source": source_name,
        "source_kind": source_kind,
        "source_index_space": index_space,
        "source_modality": modality,
        "source_world_coordinate_system": source_world,
        "mimics_world_coordinate_system": mimics_world,
        "source_to_mimics_world_matrix": source_to_mimics_world,
        "source_voxel_to_ras_matrix": source_voxel_to_ras,
        "mimics_voxel_to_ras_matrix": mimics_voxel_to_ras,
        "mimics_to_source_index_matrix": mimics_to_source_index,
        "source_intensity_space": intensity_space,
        "source_to_mimics_gv_slope": float(intensity_slope),
        "source_to_mimics_gv_intercept": float(intensity_intercept),
    }


def _export_image_for_nninteractive(config, image, path, allow_buffer_export=True):
    profile = _model_profile(config)
    if profile.get("source") == "task_model":
        task_mode = _task_model_image_input_mode(config)
        strict_source = task_mode == "source"
        allow_fallback = bool(
            config.get("fallback_to_mimics_buffer_when_source_unavailable", True)
        ) and not strict_source
        legacy_fallback = config.get("allow_task_model_mimics_buffer_fallback")
        if legacy_fallback is not None:
            allow_fallback = bool(legacy_fallback) and not strict_source

        if task_mode != "mimics":
            # Training reads source physical values. A readable local source is
            # still preferred, but alignment is deferred to the external image
            # worker so no bridge conversion blocks the Mimics GUI.
            source_config = dict(config)
            source_config["image_input_mode"] = "source" if strict_source else "auto"
            source_config["prefer_source_image_for_nninteractive"] = True
            source_config[
                "fallback_to_mimics_buffer_when_source_unavailable"
            ] = allow_fallback
            source_config["defer_source_alignment_to_worker"] = True
            source_export = _source_image_export(image, source_config)
            if source_export is not None:
                source_export["image_input_provenance"] = "source_image"
                source_export["image_intensity_compatibility"] = "exact_source_values"
                return source_export

        if strict_source or (task_mode == "auto" and not allow_fallback):
            raise RuntimeError(
                "The selected nnInteractive task model was trained from source "
                "physical intensities, but this Mimics image has no usable source "
                "image geometry/path metadata. Restore the source dataset path, "
                "or set task_model_image_input_mode to \"auto\" to allow the "
                "portable Mimics-buffer compatibility path."
            )
        if not allow_buffer_export:
            return None

        buffer_export = _export_image(image, path)
        buffer_export["image_input_provenance"] = "mimics_project_buffer"
        buffer_export["task_model_buffer_fallback"] = True
        if (
            buffer_export.get("buffer_to_source_slope") is not None
            and buffer_export.get("buffer_to_source_intercept") is not None
        ):
            compatibility = "linear_source_values_restored"
        else:
            # Keep old projects usable even when neither Mimics-Script metadata
            # nor retained DICOM tags expose the original MR rescale mapping.
            # This is intentionally labelled best-effort: z-score cancels a
            # positive affine transform only when its nonzero crop is unchanged.
            compatibility = "raw_gv_zscore_best_effort"
        buffer_export["image_intensity_compatibility"] = compatibility
        _mimics_log(
            logging.WARNING,
            "nnInteractive custom model is using the image stored in this .mcs "
            "because the original source is unavailable. Geometry remains the "
            "open Mimics grid; intensity compatibility: {0}; recovery basis: {1}. "
            "Missing MR rescale metadata never blocks inference, but raw-GV "
            "best-effort input can differ when the source zero-background crop "
            "was not preserved. Model loading, "
            "upload, and preprocessing continue in the external worker.".format(
                compatibility,
                buffer_export.get("source_intensity_recovery_basis", "unavailable"),
            ),
        )
        return buffer_export

    image_input_mode = str(config.get("image_input_mode", "") or "").strip().lower()
    force_source = image_input_mode in ("source", "source_image", "original", "original_image")
    force_mimics_buffer = image_input_mode in ("mimics", "mimics_buffer", "buffer")
    buffer_error = None

    if allow_buffer_export and not force_source:
        try:
            return _export_image(image, path)
        except Exception as exc:
            buffer_error = exc
            if force_mimics_buffer and not bool(config.get("fallback_to_source_when_mimics_export_fails", False)):
                raise RuntimeError(
                    "nnInteractive Mimics image buffer export failed while image_input_mode is explicitly set to \"mimics\": {0}. "
                    "Source-image fallback is disabled. "
                    "Fix the buffer export issue, or set fallback_to_source_when_mimics_export_fails=true for temporary fallback."
                    .format(exc)
                )
            _mimics_log(
                logging.WARNING,
                "nnInteractive Mimics image buffer export failed: {0}. Falling back to source image metadata mode.".format(exc),
            )

    source_config = config
    if not force_source:
        source_config = dict(config)
        source_config["image_input_mode"] = "source"
        source_config["prefer_source_image_for_nninteractive"] = True
    else:
        source_config = dict(config)
    # Derived oblique imports would otherwise build an aligned source cache
    # right here on the Mimics GUI thread (a bridge call of up to
    # bridge_timeout_seconds, default 1800). The external image worker
    # resamples the raw source onto the Mimics grid from the recorded
    # affines instead, so alignment never blocks the GUI. An explicit
    # false in the config keeps the old synchronous behavior.
    source_config.setdefault("defer_source_alignment_to_worker", True)

    source_export = _source_image_export(image, source_config)
    if source_export is not None:
        return source_export

    if allow_buffer_export and force_source:
        try:
            _mimics_log(
                logging.INFO,
                "nnInteractive source image mode is unavailable. Falling back to Mimics image buffer export.",
            )
            return _export_image(image, path)
        except Exception as exc:
            buffer_error = exc

    if buffer_error is not None:
        raise RuntimeError(
            "nnInteractive could not prepare image input. Mimics buffer export failed and source image metadata mode was unavailable: {0}".format(
                buffer_error
            )
        )
    return None


def _log_effective_image_input_config(config):
    try:
        _mimics_log(
            logging.INFO,
            "nnInteractive effective config: path={0}; image_input_mode={1}; task_model_image_input_mode={2}; prefer_source_image_for_nninteractive={3}; fallback_to_mimics_buffer_when_source_unavailable={4}; fallback_to_source_when_mimics_export_fails={5}.".format(
                config.get("_config_path", "?"),
                config.get("image_input_mode", ""),
                config.get("task_model_image_input_mode", "auto"),
                bool(config.get("prefer_source_image_for_nninteractive", False)),
                bool(config.get("fallback_to_mimics_buffer_when_source_unavailable", False)),
                bool(config.get("fallback_to_source_when_mimics_export_fails", False)),
            ),
        )
    except Exception:
        pass


def _export_image(image, path):
    started = time.time()
    view = image.get_voxel_buffer()
    with open(path, "wb") as handle:
        handle.write(view.tobytes())
    result = {
        "kind": "mimics_buffer",
        "path": path,
        "shape": [int(value) for value in view.shape],
        "dtype": _buffer_dtype(view),
        "sha256": _sha256_file(path),
    }
    matrix = _parse_matrix_metadata(
        _metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
    )
    if matrix is None:
        matrix = _derive_image_voxel_to_ras_matrix(image, result["shape"])
    if matrix is not None:
        result["mimics_voxel_to_ras_matrix"] = matrix
    modality = _image_modality(image)
    result["source_modality"] = modality
    if _source_uses_hu_to_gv(
        _metadata_get(image, SOURCE_IMAGE_KIND_METADATA, ""),
        modality,
    ):
        transform = _hu_to_mimics_gv_transform()
        if transform is not None:
            slope, intercept = transform
            result["buffer_to_source_slope"] = 1.0 / float(slope)
            result["buffer_to_source_intercept"] = (
                -float(intercept) / float(slope)
            )
            result["source_intensity_recovery_basis"] = "mimics_hu_gv"
    elif modality == "MR":
        mapping = _source_intensity_mapping(image, modality=modality)
        if mapping is not None:
            result["buffer_to_source_slope"] = mapping["slope"]
            result["buffer_to_source_intercept"] = mapping["intercept"]
            result["source_intensity_encoding"] = mapping["encoding"]
            result["source_intensity_recovery_basis"] = mapping["basis"]
            result["buffer_to_source_zero_tolerance"] = mapping["zero_tolerance"]
            result["source_intensity_value_min"] = mapping["source_value_min"]
            result["source_intensity_value_max"] = mapping["source_value_max"]
            result["dicom_stored_value_min"] = mapping["stored_value_min"]
            result["dicom_stored_value_max"] = mapping["stored_value_max"]
        else:
            result["source_intensity_recovery_basis"] = "unavailable"
    _mimics_log(
        logging.INFO,
        "nnInteractive image buffer exported in {0}s. Shape: {1}.".format(
            round(time.time() - started, 2),
            result["shape"],
        ),
    )
    return result


def _empty_mask_sha256(shape):
    return "empty-mask:" + "x".join(str(int(value)) for value in shape)


def _empty_mask_export(shape):
    return {
        "path": "",
        "shape": [int(value) for value in shape],
        "pixel_count": 0,
        "byte_count": 0,
        "sha256": _empty_mask_sha256(shape),
    }


def _export_mask(mask, path, shape_hint=None):
    pixel_count = int(getattr(mask, "number_of_pixels", 0))
    if shape_hint is not None and pixel_count <= 0:
        result = _empty_mask_export(shape_hint)
        _mimics_log(
            logging.INFO,
            "nnInteractive target Mask is empty; full empty Mask buffer export is skipped.",
        )
        return result
    started = time.time()
    view = mask.get_voxel_buffer()
    raw = view.tobytes()
    with open(path, "wb") as handle:
        handle.write(raw)
    result = {
        "path": path,
        "shape": [int(value) for value in view.shape],
        "pixel_count": pixel_count,
        "byte_count": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    _mimics_log(
        logging.INFO,
        "nnInteractive target Mask exported in {0}s. Foreground voxels: {1}.".format(
            round(time.time() - started, 2),
            result["pixel_count"],
        ),
    )
    return result


def _mask_sha256(mask, shape_hint=None):
    if shape_hint is not None and int(getattr(mask, "number_of_pixels", 0)) <= 0:
        return _empty_mask_sha256(shape_hint)
    return _sha256_bytes(mask.get_voxel_buffer().tobytes())


def _set_mask_from_u8(mask, path, shape, transaction_name=None):
    with open(path, "rb") as handle:
        raw = handle.read()
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
    _with_gui_updates_disabled(
        lambda: runtime_common.execute_mimics_transaction(
            mimics,
            _apply,
            transaction_name or "Apply nnInteractive Prediction",
        )
    )


def _choose_completed_result_target(image, target, state, result):
    if state.get("write_mode") != "choose_on_first_result":
        return target
    decision = mimics.dialogs.question_box(
        title="nnInteractive Prediction Ready",
        message=(
            "nnInteractive prediction completed in {0}s.\n\n"
            "Update Selected Mask applies the result to {1}.\n"
            "Create Editable Copy keeps it unchanged and starts an editable AI Draft."
        ).format(result.get("elapsed_seconds", "?"), str(getattr(target, "name", "") or "the selected Mask")),
        buttons="Update Selected Mask;Create Editable Copy",
        ui_blocking=True,
    )
    if decision == "Update Selected Mask":
        state["write_mode"] = "in_place"
        return target
    if decision != "Create Editable Copy":
        _mimics_log(logging.INFO, "nnInteractive result destination was closed; using a new editable copy to preserve the selected Mask.")
    source = target
    target = _create_result_mask(image, "{0} - AI Draft".format(str(getattr(source, "name", "") or "nnInteractive")))
    _mark_ai_draft(target, source)
    try:
        source.selected = False
    except Exception:
        pass
    _metadata_delete(source, ASYNC_JOB_METADATA)
    _metadata_set(target, ASYNC_JOB_METADATA, state.get("_job_dir", ""))
    state["target_guid"] = _object_id(target)
    state["target_name"] = str(getattr(target, "name", ""))
    state["write_mode"] = "derived_copy"
    return target


def _restore_base(mask, base_path, base_shape):
    if not base_path:
        _with_gui_updates_disabled(mask.clear)
        return
    _set_mask_from_u8(mask, base_path, base_shape)


def _make_mask_visible(mask):
    try:
        mask.visible = True
    except Exception:
        pass
    try:
        mask.selected = True
    except Exception:
        pass


def _choose_sign():
    answer = mimics.dialogs.question_box(
        message=(
            "Foreground means the structure should be included.\n"
            "Background means the structure should be excluded."
        ),
        buttons=BUTTON_FOREGROUND + ";" + BUTTON_BACKGROUND + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer == BUTTON_FOREGROUND:
        return True
    if answer == BUTTON_BACKGROUND:
        return False
    return None


def _capture_point(image, include):
    try:
        coordinates = mimics.indicate_coordinate(
            message=(
                "Click inside the structure."
                if include
                else "Click an area that must be excluded from the structure."
            ),
            show_message_box=True,
            confirm=False,
            title=TITLE,
        )
    except mimics.UserInterrupted:
        return None
    indexes = image.get_voxel_indexes(coordinates)
    marker = None
    try:
        marker = mimics.analyze.create_point(
            point=_point_coordinates(coordinates),
            name="{0} {1} Point".format(
                PROMPT_MASK_PREFIX,
                "Include" if include else "Exclude",
            ),
            color=(0.1, 1.0, 0.2) if include else (1.0, 0.2, 0.1),
        )
    except Exception:
        pass
    return {
        "point": [int(value) for value in indexes],
        "include_interaction": bool(include),
        "_marker": marker,
    }


def _delete_point_marker(point, visual_objects=None):
    marker = point.pop("_marker", None)
    if marker is None:
        return
    if visual_objects is not None:
        visual_objects[:] = [
            candidate
            for candidate in visual_objects
            if candidate is not marker
        ]
    try:
        mimics.data.points.delete(marker)
    except Exception:
        pass


def _delete_mimics_object(obj):
    """Best-effort delete of a Mimics visual object (point, mask, measurement, spline)."""
    if obj is None:
        return
    # Try the most specific delete methods first, then fall back.
    for deleter in (
        lambda: mimics.data.points.delete(obj),
        lambda: mimics.data.masks.delete(obj),
        lambda: mimics.data.distance_measurements.delete(obj),
        lambda: mimics.data.measurements.delete(obj),
        lambda: mimics.data.splines.delete(obj),
    ):
        try:
            deleter()
            return
        except Exception:
            pass


def _capture_point_set(image, _visual_objects=None):
    points = []
    try:
        while True:
            include_count = len([point for point in points if point["include_interaction"]])
            exclude_count = len(points) - include_count
            run_button = "{0} ({1})".format(BUTTON_RUN_POINTS, len(points))
            buttons = [BUTTON_INCLUDE_POINT, BUTTON_EXCLUDE_POINT]
            if points:
                buttons.append(BUTTON_REMOVE_POINT)
                buttons.append(run_button)
            buttons.append(BUTTON_DISCARD_POINTS)
            answer = mimics.dialogs.question_box(
                message=(
                    "Add all points for the next prediction.\n\n"
                    "Include points: {0}\n"
                    "Exclude points: {1}\n\n"
                    "Temporary green/red markers show the current point set."
                ).format(include_count, exclude_count),
                buttons=";".join(buttons),
                title="Point Set",
                ui_blocking=True,
            )
            if answer == BUTTON_INCLUDE_POINT:
                point = _capture_point(image, True)
                if point is not None:
                    points.append(point)
                    if (
                        _visual_objects is not None
                        and point.get("_marker") is not None
                    ):
                        _visual_objects.append(point["_marker"])
            elif answer == BUTTON_EXCLUDE_POINT:
                point = _capture_point(image, False)
                if point is not None:
                    points.append(point)
                    if (
                        _visual_objects is not None
                        and point.get("_marker") is not None
                    ):
                        _visual_objects.append(point["_marker"])
            elif answer == BUTTON_REMOVE_POINT and points:
                removed = points.pop()
                _delete_point_marker(removed, _visual_objects)
            elif answer == run_button and points:
                if _visual_objects is None:
                    for point in points:
                        _delete_point_marker(point)
                return {
                    "interaction_type": "point_set",
                    "points": [
                        {
                            "point": point["point"],
                            "include_interaction": point["include_interaction"],
                        }
                        for point in points
                    ],
                    "coordinates": "mimics",
                }
            else:
                # Discard: delete markers immediately since no prediction will run.
                for point in points:
                    _delete_point_marker(point, _visual_objects)
                return None
    except Exception:
        # On exception, clean up markers to avoid orphaned points.
        for point in points:
            _delete_point_marker(point, _visual_objects)
        raise


def _voxel_points(image, geometry):
    points = []
    for point in geometry:
        indexes = image.get_voxel_indexes(_point_coordinates(point))
        current = [int(value) for value in indexes]
        if not points or current != points[-1]:
            points.append(current)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _single_slice_axis(points):
    if not points:
        return None
    ranges = [
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    ]
    constant_axes = [axis for axis, value in enumerate(ranges) if value == 0]
    return constant_axes[0] if len(constant_axes) == 1 else None


def _capture_box(image, _visual_objects=None):
    measurement = None
    try:
        measurement = mimics.measure.indicate_distance_measurement(
            message=(
                "Place the two endpoints on opposite corners of the structure.\n"
                "The line is used only as the diagonal of a 2D foreground box."
            ),
            show_message_box=True,
            confirm=True,
            title="Foreground Box",
        )
        if measurement is None:
            return None
        points = _voxel_points(image, [measurement.point1, measurement.point2])
        if len(points) != 2 or _single_slice_axis(points) is None:
            mimics.dialogs.message_box(
                "The two endpoints must define a non-degenerate box on one image slice.\n\n"
                "Use a 2D view and place the points on opposite corners.",
                title="Box Not Accepted",
                ui_blocking=True,
            )
            # Box not accepted: delete measurement immediately.
            try:
                mimics.data.distance_measurements.delete(measurement)
            except Exception:
                try:
                    mimics.data.measurements.delete(measurement)
                except Exception:
                    pass
            return None
        bbox = [
            [min(points[0][axis], points[1][axis]), max(points[0][axis], points[1][axis]) + 1]
            for axis in range(3)
        ]
        # Success: keep measurement visible during inference.
        if _visual_objects is not None:
            _visual_objects.append(measurement)
        else:
            try:
                mimics.data.distance_measurements.delete(measurement)
            except Exception:
                try:
                    mimics.data.measurements.delete(measurement)
                except Exception:
                    pass
        return {
            "interaction_type": "box",
            "include_interaction": True,
            "bbox": bbox,
            "coordinates": "mimics",
        }
    except mimics.UserInterrupted:
        # User cancelled: clean up immediately.
        if measurement is not None:
            try:
                mimics.data.distance_measurements.delete(measurement)
            except Exception:
                try:
                    mimics.data.measurements.delete(measurement)
                except Exception:
                    pass
        return None


def _capture_lasso(image, _visual_objects=None):
    spline = None
    try:
        spline = mimics.analyze.indicate_spline(
            message=(
                "Place control points around the structure and close the Spline before confirming.\n"
                "The closed curve becomes a 2D foreground Lasso prompt."
            ),
            show_message_box=True,
            confirm=True,
            title="Foreground Lasso",
        )
        if spline is None:
            return None
        if not bool(getattr(spline, "closed", False)):
            mimics.dialogs.message_box(
                "The Spline is open. A Lasso prompt must be closed.\n\n"
                "Run Draw Lasso again and close the curve before confirming.",
                title="Lasso Not Accepted",
                ui_blocking=True,
            )
            # Not accepted: delete spline immediately.
            try:
                mimics.data.splines.delete(spline)
            except Exception:
                pass
            return None
        points = _voxel_points(image, _spline_geometry(spline))
        if len(set(tuple(point) for point in points)) < 3:
            mimics.dialogs.message_box(
                "The closed Spline contains fewer than three distinct voxel points.",
                title="Lasso Not Accepted",
                ui_blocking=True,
            )
            try:
                mimics.data.splines.delete(spline)
            except Exception:
                pass
            return None
        if _single_slice_axis(points) is None:
            mimics.dialogs.message_box(
                "The Lasso must lie on one axis-aligned image slice.\n\n"
                "Draw it in an axial, coronal, or sagittal 2D view.",
                title="Lasso Not Accepted",
                ui_blocking=True,
            )
            try:
                mimics.data.splines.delete(spline)
            except Exception:
                pass
            return None
        # Success: keep spline visible during inference.
        if _visual_objects is not None:
            _visual_objects.append(spline)
        else:
            try:
                mimics.data.splines.delete(spline)
            except Exception:
                pass
        return {
            "interaction_type": "lasso",
            "include_interaction": True,
            "polyline_points": points,
            "polyline_closed": True,
            "coordinates": "mimics",
        }
    except mimics.UserInterrupted:
        # User cancelled: clean up immediately.
        if spline is not None:
            try:
                mimics.data.splines.delete(spline)
            except Exception:
                pass
        return None


def _capture_scribble(image, include, temp_dir, _visual_objects=None):
    if _visual_objects is None:
        return _capture_mask_prompt(image, include, "scribble", "Ellipse", temp_dir)
    return _capture_mask_prompt(image, include, "scribble", "Ellipse", temp_dir, _visual_objects)


def _capture_scribble_set(image, temp_dir, _visual_objects=None):
    scribbles = []
    while True:
        foreground_count = len([item for item in scribbles if item["include_interaction"]])
        background_count = len(scribbles) - foreground_count
        buttons = [BUTTON_ADD_FOREGROUND_SCRIBBLE, BUTTON_ADD_BACKGROUND_SCRIBBLE]
        if scribbles:
            buttons.append("{0} ({1})".format(BUTTON_RUN_SCRIBBLES, len(scribbles)))
        buttons.append(BUTTON_DISCARD_SCRIBBLES)
        answer = mimics.dialogs.question_box(
            message=(
                "Add foreground and background scribbles for one prediction.\n\n"
                "Foreground scribbles: {0}\n"
                "Background scribbles: {1}\n\n"
                "Run only after all scribbles for this round are added."
            ).format(foreground_count, background_count),
            buttons=";".join(buttons),
            title="Scribble Set",
            ui_blocking=True,
        )
        if answer == BUTTON_ADD_FOREGROUND_SCRIBBLE:
            prompt = _capture_scribble(image, True, temp_dir, _visual_objects)
            if prompt is not None:
                scribbles.append(prompt)
        elif answer == BUTTON_ADD_BACKGROUND_SCRIBBLE:
            prompt = _capture_scribble(image, False, temp_dir, _visual_objects)
            if prompt is not None:
                scribbles.append(prompt)
        elif answer.startswith(BUTTON_RUN_SCRIBBLES) and scribbles:
            return {
                "interaction_type": "scribble_set",
                "scribbles": scribbles,
                "coordinates": "mimics",
            }
        else:
            return None


def _point_coordinates(value):
    if hasattr(value, "coordinates"):
        value = value.coordinates
    return tuple(float(item) for item in value)


def _spline_geometry(spline):
    geometry = getattr(spline, "geometry_points", None)
    if not geometry:
        geometry = getattr(spline, "points", None)
    if not geometry:
        raise RuntimeError(
            "Mimics returned a Spline without geometry_points or points. "
            "The Lasso prompt cannot be converted to voxel coordinates."
        )
    return list(geometry)


def _create_prompt_mask(image, include, interaction_type):
    mask = mimics.segment.create_mask()
    try:
        mask.name = "{0} {1} {2}".format(
            PROMPT_MASK_PREFIX,
            "Foreground" if include else "Background",
            interaction_type,
        )
        try:
            mask.image = image
        except Exception:
            mask_image = getattr(mask, "image", None)
            if mask_image is not None and not _same_object(mask_image, image):
                raise
        try:
            mask.visible = True
        except Exception:
            pass
        try:
            mask.color = (0.1, 1.0, 0.2) if include else (1.0, 0.2, 0.1)
        except Exception:
            pass
        return mask
    except Exception:
        try:
            mimics.data.masks.delete(mask)
        except Exception:
            pass
        raise


def _capture_mask_prompt(image, include, interaction_type, edit_type, temp_dir, _visual_objects=None):
    prompt_mask = _create_prompt_mask(image, include, interaction_type)
    result = None
    try:
        mimics.segment.activate_edit_mask(prompt_mask, edit_type, "Draw")
        if int(getattr(prompt_mask, "number_of_pixels", 0)) <= 0:
            mimics.dialogs.message_box(
                "No {0} pixels were captured.\n\n"
                "Draw on the temporary prompt Mask before confirming the edit.".format(
                    interaction_type
                ),
                title="nnInteractive Prompt Empty",
                ui_blocking=True,
            )
            return None
        view = prompt_mask.get_voxel_buffer()
        full_shape = [int(value) for value in view.shape]
        try:
            import numpy as np

            prompt = np.asarray(view, dtype=np.uint8)
            nonzero = np.argwhere(prompt)
            if len(nonzero) == 0:
                return None
            minimum = nonzero.min(axis=0)
            maximum = nonzero.max(axis=0) + 1
            bbox = [
                [int(minimum[axis]), int(maximum[axis])]
                for axis in range(3)
            ]
            crop = prompt[
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
            ]
            path = os.path.join(
                temp_dir,
                "prompt_{0}_{1}.u8".format(interaction_type, uuid.uuid4().hex),
            )
            with open(path, "wb") as handle:
                handle.write(crop.tobytes(order="C"))
            result = {
                "interaction_type": interaction_type,
                "include_interaction": bool(include),
                "mask_path": path,
                "mask_shape": [int(value) for value in crop.shape],
                "interaction_bbox": bbox,
                "full_shape": full_shape,
                "coordinates": "mimics",
            }
        except ImportError:
            pass

        if result is None:
            path = os.path.join(
                temp_dir,
                "prompt_{0}_{1}.u8".format(interaction_type, uuid.uuid4().hex),
            )
            exported = _export_mask(prompt_mask, path)
            result = {
                "interaction_type": interaction_type,
                "include_interaction": bool(include),
                "mask_path": path,
                "mask_shape": exported["shape"],
                "full_shape": exported["shape"],
                "coordinates": "mimics",
            }
    except mimics.UserInterrupted:
        # User cancelled: clean up the mask immediately.
        try:
            mimics.data.masks.delete(prompt_mask)
        except Exception:
            pass
        return None
    except Exception:
        # Other error: clean up the mask immediately.
        try:
            mimics.data.masks.delete(prompt_mask)
        except Exception:
            pass
        raise

    # Success: keep the mask visible during inference, schedule deferred deletion.
    if result is not None:
        if _visual_objects is not None:
            _visual_objects.append(prompt_mask)
        else:
            try:
                mimics.data.masks.delete(prompt_mask)
            except Exception:
                pass
    return result


def _capture_prompt(kind, image, include, temp_dir, _visual_objects=None):
    if kind == BUTTON_POINT:
        return _capture_point_set(image, _visual_objects)
    if kind == BUTTON_SCRIBBLE:
        return _capture_scribble_set(image, temp_dir, _visual_objects)
    if kind == BUTTON_BOX:
        return _capture_box(image, _visual_objects)
    if kind == BUTTON_LASSO:
        return _capture_lasso(image, _visual_objects)
    raise RuntimeError("Unsupported prompt kind: {0}".format(kind))


def _bridge_parameters(config, image_export, base_export):
    python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)
    profile = _model_profile(config)
    runtime_dir = _runtime_work_dir(config, model_dir)
    runtime_log = _runtime_log_path(model_dir, config)
    requested_device = os.environ.get(
        "NNINTERACTIVE_DEVICE",
        config.get("device", "auto"),
    )
    startup_timeout = int(
        os.environ.get(
            "NNINTERACTIVE_SERVER_STARTUP_TIMEOUT",
            config.get("server_startup_timeout_seconds", 600),
        )
    )
    prediction_timeout = int(
        os.environ.get(
            "NNINTERACTIVE_PREDICTION_TIMEOUT",
            config.get("prediction_timeout_seconds", 1800),
        )
    )
    set_image_timeout = int(
        os.environ.get(
            "NNINTERACTIVE_SET_IMAGE_TIMEOUT",
            config.get("set_image_timeout_seconds", 1800),
        )
    )
    request = {
        "buffer_mapping": {
            "platform_to_mimics_axes": [0, 1, 2],
            "platform_to_mimics_flips": [False, False, False],
        },
        "initial_seg_path": base_export["path"] if base_export["pixel_count"] > 0 else None,
        "initial_seg_shape": base_export["shape"],
        "model_dir": model_dir,
        "model_source": profile.get("source") or "official",
        "model_profile_id": profile.get("profile_id") or "official",
        "model_id": profile.get("model_id") or "official",
        "checkpoint_sha256": profile.get("checkpoint_sha256") or "",
        "task_id": profile.get("task_id") or "",
        "device": requested_device,
        "allow_cpu_fallback": bool(config.get("allow_cpu_fallback", True)),
        "fold": config.get("fold", "auto"),
        "server_url": os.environ.get(
            "NNINTERACTIVE_SERVER_URL",
            config.get("server_url", "http://127.0.0.1:1527"),
        ),
        "auto_start_server": bool(config.get("auto_start_server", True)),
        "server_startup_timeout_seconds": startup_timeout,
        "prediction_timeout_seconds": prediction_timeout,
        "set_image_timeout_seconds": set_image_timeout,
        "log_dir": os.path.dirname(runtime_log),
        "runtime_work_dir": runtime_dir,
        "server_idle_timeout_seconds": int(
            os.environ.get(
                "NNINTERACTIVE_SERVER_IDLE_TIMEOUT",
                config.get("server_idle_timeout_seconds", 1800),
            )
        ),
        "gpu_lock_timeout_seconds": float(
            os.environ.get(
                "NNINTERACTIVE_GPU_LOCK_TIMEOUT",
                config.get("gpu_lock_timeout_seconds", 30),
            )
        ),
        "minimum_free_gpu_memory_gb": float(
            os.environ.get(
                "NNINTERACTIVE_MINIMUM_FREE_GPU_MEMORY_GB",
                config.get("minimum_free_gpu_memory_gb", 4),
            )
        ),
        "incremental_interaction_replay": bool(
            config.get("incremental_interaction_replay", True)
        ),
        "keep_server_warm_after_session": bool(
            config.get("keep_server_warm_after_session", True)
        ),
    }
    if profile.get("source") == "task_model":
        # Fine-tuning data is canonicalized to RAS. Reorient only task-model
        # inputs and prompts in the external bridge; official behavior remains
        # unchanged, and predictions are mapped back before Mimics applies them.
        request["model_input_space"] = "canonical_ras"
        request["model_input_intensity_space"] = "source_values"
    if image_export.get("image_path"):
        request["image_path"] = image_export["image_path"]
        request["interaction_shape"] = image_export["shape"]
        request["image_expected_shape"] = image_export["shape"]
        request["image_source"] = image_export.get("source", "source_image")
        request["image_source_kind"] = image_export.get("source_kind", "")
        request["image_source_index_space"] = image_export.get("source_index_space", "")
        request["image_source_modality"] = image_export.get("source_modality", "")
        request["image_source_world_coordinate_system"] = image_export.get("source_world_coordinate_system", "")
        request["image_mimics_world_coordinate_system"] = image_export.get("mimics_world_coordinate_system", "")
        request["image_source_to_mimics_world_matrix"] = image_export.get("source_to_mimics_world_matrix", "")
        request["image_source_voxel_to_ras_matrix"] = image_export.get("source_voxel_to_ras_matrix", "")
        request["image_mimics_voxel_to_ras_matrix"] = image_export.get("mimics_voxel_to_ras_matrix", "")
        request["image_mimics_to_source_index_matrix"] = image_export.get("mimics_to_source_index_matrix", "")
        request["image_source_intensity_space"] = image_export.get("source_intensity_space", "")
        request["image_source_to_mimics_gv_slope"] = image_export.get("source_to_mimics_gv_slope")
        request["image_source_to_mimics_gv_intercept"] = image_export.get("source_to_mimics_gv_intercept")
        request["image_input_provenance"] = image_export.get(
            "image_input_provenance", "source_image"
        )
        request["image_intensity_compatibility"] = image_export.get(
            "image_intensity_compatibility", "exact_source_values"
        )
    else:
        request["image_buffer_path"] = image_export["path"]
        request["image_buffer_shape"] = image_export["shape"]
        request["image_buffer_dtype"] = image_export["dtype"]
        request["image_buffer_coordinates"] = "mimics"
        request["image_mimics_voxel_to_ras_matrix"] = image_export.get(
            "mimics_voxel_to_ras_matrix", ""
        )
        request["image_input_provenance"] = image_export.get(
            "image_input_provenance", "mimics_project_buffer"
        )
        request["image_intensity_compatibility"] = image_export.get(
            "image_intensity_compatibility", ""
        )
        request["image_source_intensity_encoding"] = image_export.get(
            "source_intensity_encoding", ""
        )
        request["image_source_intensity_recovery_basis"] = image_export.get(
            "source_intensity_recovery_basis", ""
        )
        if profile.get("source") == "task_model":
            request["image_buffer_to_model_slope"] = image_export.get(
                "buffer_to_source_slope", 1.0
            )
            request["image_buffer_to_model_intercept"] = image_export.get(
                "buffer_to_source_intercept", 0.0
            )
            request["image_buffer_model_zero_tolerance"] = image_export.get(
                "buffer_to_source_zero_tolerance"
            )
    configured_timeout = config.get("bridge_timeout_seconds")
    if configured_timeout is None:
        configured_timeout = config.get("timeout_seconds")
        if configured_timeout is not None:
            _mimics_log(
                logging.WARNING,
                "nnInteractive config key 'timeout_seconds' is deprecated; use 'bridge_timeout_seconds' for the end-to-end bridge subprocess timeout.",
            )
    timeout = int(
        configured_timeout
        if configured_timeout is not None
        else startup_timeout + set_image_timeout + prediction_timeout + 120
    )
    return {
        "python_exe": python_exe,
        "bridge_script": bridge_script,
        "model_dir": model_dir,
        "probe": probe,
        "folds": folds,
        "runtime_log": runtime_log,
        "requested_device": requested_device,
        "startup_timeout": startup_timeout,
        "prediction_timeout": prediction_timeout,
        "set_image_timeout": set_image_timeout,
        "timeout": timeout,
        "request": request,
    }


def _async_prompt_menu(target, state, source=None, profile=None):
    count = len(state.get("interactions", [])) if state else 0
    menu_source = target if source is None else source
    source_name = state.get("source_name") if state else str(getattr(menu_source, "name", ""))
    target_name = state.get("target_name") if state else str(getattr(target, "name", ""))
    buttons = _prompt_buttons_for_profile(profile or {})
    if state and state.get("interactions"):
        buttons.extend([BUTTON_UNDO, BUTTON_RESET])
    buttons.append(BUTTON_FINISH)
    return mimics.dialogs.question_box(
        message=(
            "Source snapshot: {0}\n"
            "AI result Mask: {1}\n"
            "Prompts in this AI session: {2}\n\n"
            "Submitting a prompt starts background inference and immediately "
            "returns control to Mimics. The result is applied automatically when ready."
        ).format(source_name, target_name, count),
        buttons=";".join(buttons),
        title=TITLE,
        ui_blocking=True,
    )


def _run_async(
    image,
    target,
    config,
    source=None,
    auto_created=False,
    write_mode="in_place",
):
    source = target if source is None else source
    profile = _model_profile(config)
    _log_effective_image_input_config(config)
    state = _load_async_job(target)
    validated_target_hash = None
    if state is not None:
        if not _state_model_matches(state, profile):
            answer = mimics.dialogs.question_box(
                message=(
                    "The selected Mask already has an AI session created with another "
                    "nnInteractive model.\n\n"
                    "Start a new session from the current Mask with {0}?"
                ).format(profile.get("task_name") or profile.get("model_id") or "the selected model"),
                buttons=BUTTON_START_NEW_MODEL + ";" + BUTTON_KEEP_CURRENT_MODEL,
                title=TITLE,
                ui_blocking=True,
            )
            if answer != BUTTON_START_NEW_MODEL:
                _mimics_log(
                    logging.INFO,
                    "nnInteractive model change cancelled; the existing AI session was preserved.",
                )
                return 0
            _close_async_job(target, state, "model_profile_changed")
            state = None
            _mimics_log(
                logging.INFO,
                "nnInteractive started a new AI session with model {0}.".format(
                    profile.get("model_id") or "official"
                ),
            )
    if state is not None:
        if _object_id(image) != state.get("image_guid") or _object_id(target) != state.get(
            "target_guid"
        ):
            _close_async_job(target, state, "object_identity_changed")
            state = None
        else:
            outcome = _handle_async_result(image, target, state)
            if outcome == "waiting":
                return 0
            if outcome == "restart":
                state = None
            elif outcome == "applied":
                validated_target_hash = state.get("expected_target_sha256")

    if state is not None:
        if state.get("pending_sequence") is None and not _state_worker_alive(state):
            _mimics_log(
                logging.INFO,
                "nnInteractive previous AI session is no longer running; a new session will start from the current Mask.",
            )
            _close_async_job(target, state, "ready_worker_not_alive")
            state = None

    if state is not None:
        if validated_target_hash is not None and validated_target_hash == state.get("expected_target_sha256"):
            current_hash = validated_target_hash
        else:
            current_hash = _mask_sha256(target, state.get("shape"))
        if current_hash != state.get("expected_target_sha256"):
            _close_async_job(target, state, "manual_mask_change")
            state = None
            validated_target_hash = None
        else:
            validated_target_hash = current_hash

    _continue_session_prompt(
        image,
        target,
        state,
        config,
        source=source,
        auto_created=auto_created,
        write_mode=write_mode,
        validated_target_hash=validated_target_hash,
    )
    return 0


def _continue_session_prompt(
    image,
    target,
    state,
    config,
    source=None,
    auto_created=False,
    write_mode="in_place",
    validated_target_hash=None,
):
    """Show the prompt menu for a session and dispatch the chosen action.

    Shared by the interactive tool entry (_run_async) and by the async
    result monitor, which re-invokes it after applying a result so one
    menu entry supports continuous prompting instead of re-entering the
    tool for every prompt.

    Returns True when a new background prediction was enqueued and its
    result monitor started; False when the session ended, was reset, or
    no prompt was submitted.
    """
    if source is None:
        source = target
    profile = _model_profile(config)
    temp_dir = tempfile.mkdtemp(prefix="mimics_nninteractive_prompt_")
    pending_visual_objects = []
    visual_objects_registered = False
    prediction_enqueued = False
    try:
        action = _async_prompt_menu(target, state, source, profile)
        if action == BUTTON_FINISH or not action:
            if state is not None:
                _close_async_job(target, state, "user_finished")
            _mimics_log(logging.INFO, "nnInteractive session finished.")
            return False
        if action == BUTTON_UNDO and state is not None:
            interactions = state.get("interactions", [])
            if interactions:
                interactions.pop()
            state["interactions"] = interactions
            _mimics_log(
                logging.INFO,
                "nnInteractive undo. Remaining prompts: {0}.".format(len(interactions)),
            )
            # Clean up visual objects from previous prompts on undo.
            job_dir = state.get("_job_dir")
            if job_dir and job_dir in _ASYNC_VISUAL_OBJECTS:
                for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
                    _delete_mimics_object(obj)
            if interactions:
                _retire_different_model_workers(config)
                _enqueue_async_prediction(
                    state,
                    target,
                    expected_hash=validated_target_hash,
                    replay_all=True,
                )
                _start_async_result_monitor(image, target, state, config)
                return True
            _restore_base(target, state["base_path"], state["shape"])
            # Re-anchor on the restored mask buffer (see apply path): the
            # next prompt compares _mask_sha256(target) to this value.
            state["expected_target_sha256"] = _mask_sha256(target, state.get("shape"))
            state["pending_sequence"] = None
            state["status"] = "ready"
            _save_async_job(state)
            return False
        if action == BUTTON_RESET and state is not None:
            state["interactions"] = []
            state["pending_sequence"] = None
            state["status"] = "ready"
            _mimics_log(logging.INFO, "nnInteractive session reset to initial mask.")
            _restore_base(target, state["base_path"], state["shape"])
            # Re-anchor on the restored mask buffer (see apply path).
            state["expected_target_sha256"] = _mask_sha256(target, state.get("shape"))
            # Clean up visual objects from previous prompts.
            job_dir = state.get("_job_dir")
            if job_dir and job_dir in _ASYNC_VISUAL_OBJECTS:
                for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
                    _delete_mimics_object(obj)
            _save_async_job(state)
            return False

        _retire_different_model_workers(config)
        if state is None and bool(config.get("async_start_worker_before_prompt", True)):
            _update_gui()
            prewarmed = _prewarm_async_image_worker(config, image)
            if prewarmed is not None:
                _mimics_log(
                    logging.INFO,
                    "nnInteractive is preparing the image AI session before prompt capture. Model loading and image preprocessing will continue while you draw.",
                )
            _update_gui()

        include = None
        visual_objects = []
        prompt = _capture_prompt(
            action, image, include, temp_dir, visual_objects
        )
        pending_visual_objects = visual_objects
        if prompt is None:
            mimics.dialogs.message_box(
                "No prompt was submitted for {0}.\n\n"
                "The AI prediction was not started.".format(action),
                title="nnInteractive Prompt Empty",
                ui_blocking=False,
            )
            return False
        if state is None:
            state = _start_async_job(
                config,
                image,
                target,
                source=source,
                write_mode=write_mode,
            )
            validated_target_hash = state.get("expected_target_sha256")
        if str(prompt.get("interaction_type") or "") == "point_set":
            first_prompt = not bool(state.get("interactions"))
            empty_base = int(state.get("base_pixel_count") or 0) == 0
            prompt["prediction_policy"] = (
                "initial_empty_batch"
                if first_prompt and empty_base
                else "sequential"
            )
        prompt = _persist_interaction(state["_job_dir"], prompt)
        # Store visual objects for deferred deletion after async result is applied.
        if visual_objects:
            _ASYNC_VISUAL_OBJECTS.setdefault(state["_job_dir"], []).extend(visual_objects)
            visual_objects_registered = True
        state.setdefault("interactions", []).append(prompt)
        sequence = _enqueue_async_prediction(state, target, expected_hash=validated_target_hash)
        prediction_enqueued = True
        _mimics_log(
            logging.INFO,
            "nnInteractive background inference started. Prompt: {0}, sequence: {1}. "
            "Result will be applied automatically when ready.".format(action, sequence),
        )
        _start_async_result_monitor(image, target, state, config)
        return True
    finally:
        if visual_objects_registered and not prediction_enqueued and state is not None:
            job_dir = state.get("_job_dir")
            registered = _ASYNC_VISUAL_OBJECTS.get(job_dir) or []
            for obj in pending_visual_objects:
                for index in range(len(registered) - 1, -1, -1):
                    if registered[index] is obj:
                        registered.pop(index)
                        break
            if registered:
                _ASYNC_VISUAL_OBJECTS[job_dir] = registered
            else:
                _ASYNC_VISUAL_OBJECTS.pop(job_dir, None)
            visual_objects_registered = False
        if (
            pending_visual_objects
            and not visual_objects_registered
        ):
            for obj in pending_visual_objects:
                _delete_mimics_object(obj)
        if state is None:
            _delete_unused_auto_draft(target, auto_created, source)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_with_config(config):
    image = mimics.data.images.get_active()
    if image is None:
        raise RuntimeError("Open a project and activate an image set before running nnInteractive.")
    buffer_owner = runtime_common.active_local_operation("mask_buffer_access")
    if buffer_owner:
        mimics.dialogs.message_box(
            message=(
                "nnInteractive cannot take a consistent image and Mask snapshot while {0} is using Mimics buffers.\n\n"
                "Wait for that operation to finish or stop it, then retry."
            ).format(buffer_owner.get("owner") or "another Mimics-Script task"),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    gpu_holder = runtime_common.active_resource_lock(_project_root(), "gpu.lock")
    if gpu_holder:
        owner = str(gpu_holder.get("owner") or "").lower()
        holder_is_nninteractive_server = bool(
            "nninteractive" in owner
            and (
                "server" in owner
                or str(gpu_holder.get("state_path") or "").strip()
            )
        )
        if not holder_is_nninteractive_server:
            mimics.dialogs.message_box(
                message=(
                    "The GPU is currently used by {0}.\n\n"
                    "nnInteractive was not started, so no prompt or Mask state was changed. "
                    "Interactive prompting should not sit behind a long training or inference queue. "
                    "Wait for that task to finish, stop it from its task window, or use "
                    "Admin > Stop All Owned Background Services."
                ).format(runtime_common.resource_lock_summary(gpu_holder)),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
    profile = _model_profile(config)
    if profile.get("source") == "task_model":
        _mimics_log(
            logging.INFO,
            "Using nnInteractive task model {0} for task {1}.".format(
                profile.get("model_id"),
                profile.get("task_name") or profile.get("task_id") or "unnamed",
            ),
        )
    busy_workers = _different_model_busy_workers(config)
    if busy_workers:
        current = busy_workers[0]
        mimics.dialogs.message_box(
            message=(
                "Another nnInteractive model is still producing or applying a result.\n\n"
                "Model: {0}\n"
                "No Mask or prompt state was changed. Wait for that result to be applied, "
                "or stop its session before switching models."
            ).format(current.get("task_name") or current.get("model_id") or "official"),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    session = _select_session_masks(image, config)
    if session.get("write_mode") == "cancelled":
        _mimics_log(logging.INFO, "nnInteractive cancelled before creating an AI session.")
        return 0
    source = session["source"]
    target = session["target"]
    if source is not target:
        _mimics_log(
            logging.INFO,
            "nnInteractive created AI Draft {0} from source Mask {1}; the source will not be modified.".format(
                getattr(target, "name", ""),
                getattr(source, "name", ""),
            ),
        )
    # Async mode returns 0 after submitting a prompt; the background
    # worker and QTimer monitor are still running.
    return _run_async(
        image,
        target,
        config,
        source=source,
        auto_created=session.get("auto_created", False),
        write_mode=session.get("write_mode", "in_place"),
    )


def run_with_model_profile(profile):
    """Run the existing interaction workflow with an explicit task model."""
    config = _config()
    config["_model_profile"] = dict(profile or {})
    # Validate before selecting/creating a Mask so an invalid model cannot
    # mutate the Mimics project.
    _runtime_paths(config)
    selected = _model_profile(config)
    _mimics_log(
        logging.INFO,
        "nnInteractive custom model selected: {0} ({1}, checkpoint {2}).".format(
            selected.get("task_name") or selected.get("task_id") or "custom task",
            selected.get("model_id") or "unknown",
            (selected.get("checkpoint_sha256") or "unverified")[:12],
        ),
    )
    return _run_with_config(config)


def run():
    return _run_with_config(_config())


def _timing_summary(result):
    pieces = []
    for key, label in (
        ("image_load_seconds", "image_load"),
        ("server_ready_seconds", "server_ready"),
        ("set_image_seconds", "set_image"),
        ("set_target_seconds", "set_target"),
        ("prompt_apply_seconds", "prompt_apply"),
        ("elapsed_seconds", "total"),
    ):
        value = result.get(key)
        if value is not None:
            pieces.append("{0}={1}s".format(label, value))
    return ", ".join(pieces)




def _async_job_state_path(job_dir):
    return os.path.join(job_dir, "job.json")


def _async_worker_status(job_dir):
    return _read_json(os.path.join(job_dir, "worker_status.json"), {}) or {}


def _async_state_worker_dir(state):
    return state.get("_worker_dir") or state.get("worker_dir") or state["_job_dir"]


def _next_async_sequence(worker_dir):
    commands_dir = os.path.join(worker_dir, "commands")
    highest = 0
    if os.path.isdir(commands_dir):
        for name in os.listdir(commands_dir):
            if not name.startswith("command_") or not name.endswith(".json"):
                continue
            try:
                highest = max(highest, int(name[len("command_"):-len(".json")]))
            except ValueError:
                pass
    return highest + 1


def _load_async_job(target):
    job_dir = _metadata_get(target, ASYNC_JOB_METADATA, "")
    if not job_dir:
        return None
    state = _read_json(_async_job_state_path(job_dir))
    if not state:
        _metadata_delete(target, ASYNC_JOB_METADATA)
        return None
    if state.get("status") in ("closing", "closed", "discarded", "failed", "expired") or os.path.isfile(
        os.path.join(job_dir, "close.json")
    ):
        _metadata_delete(target, ASYNC_JOB_METADATA)
        return None
    state["_job_dir"] = job_dir
    state["_worker_dir"] = state.get("worker_dir", job_dir)
    return state


def _save_async_job(state):
    _write_json_atomic(_async_job_state_path(state["_job_dir"]), dict(
        (key, value) for key, value in state.items() if not key.startswith("_")
    ))


def _close_async_job(target, state, reason):
    if state:
        job_dir = state["_job_dir"]
        worker_dir = _async_state_worker_dir(state)
        # Clean up any remaining visual objects for this job.
        if job_dir in _ASYNC_VISUAL_OBJECTS:
            for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
                _delete_mimics_object(obj)
        if worker_dir == job_dir:
            _write_json_atomic(
                os.path.join(worker_dir, "close.json"),
                {
                    "reason": reason,
                    "requested_at_epoch": time.time(),
                },
            )
        state["status"] = "closing"
        state["closed_reason"] = reason
        state["updated_at_epoch"] = time.time()
        _save_async_job(state)
    _metadata_delete(target, ASYNC_JOB_METADATA)


def _cleanup_async_jobs(root, retention_days, max_terminal_jobs=20):
    if not os.path.isdir(root):
        return
    cutoff = time.time() - max(1, int(retention_days)) * 86400
    terminal_jobs = []
    for name in os.listdir(root):
        job_dir = os.path.join(root, name)
        if not os.path.isdir(job_dir):
            continue
        state = _read_json(_async_job_state_path(job_dir), {}) or {}
        status = state.get("status")
        updated = float(state.get("updated_at_epoch", 0))
        worker_status = _async_worker_status(job_dir).get("status")
        terminal = status in (
            "closed",
            "closing",
            "discarded",
            "failed",
            "expired",
        ) or worker_status in ("closed", "failed", "expired")
        if terminal:
            try:
                modified = max(updated, os.path.getmtime(job_dir))
            except OSError:
                modified = updated
            terminal_jobs.append((modified, job_dir))
    terminal_jobs.sort(reverse=True)
    for index, (modified, job_dir) in enumerate(terminal_jobs):
        if modified >= cutoff and index < max(1, int(max_terminal_jobs)):
            continue
        shutil.rmtree(job_dir, ignore_errors=True)


def _start_async_worker(python_exe, bridge_script, worker_dir):
    worker_log_path = os.path.join(worker_dir, "async_worker.log")
    _rotate_log_file(worker_log_path)
    worker_log = open(worker_log_path, "ab")
    try:
        process = subprocess.Popen(
            [python_exe, bridge_script, "--async-worker", worker_dir],
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            **_background_process_kwargs()
        )
    finally:
        worker_log.close()
    # Register with the process registry so the startup sweep and the
    # health panel can see this worker (best-effort; never blocks spawn).
    runtime_common.register_process(
        runtime_common.project_root(),
        "async_worker",
        process.pid,
        parent_pid=os.getpid(),
        state_path=os.path.join(worker_dir, "worker_status.json"),
    )
    return process, worker_log_path


def _async_worker_idle_timeout(config):
    return int(config.get("async_worker_idle_timeout_seconds", 3600))


def _shared_image_worker_alive(worker):
    if not worker:
        return False
    worker_dir = worker.get("worker_dir")
    pid = worker.get("pid")
    if not worker_dir or not os.path.isdir(worker_dir) or not _process_exists(pid):
        return False
    status = _async_worker_status(worker_dir).get("status")
    return status not in ("closing", "closed", "failed", "expired")


def _state_worker_alive(state):
    if not state:
        return False
    pid = state.get("pid")
    worker_dir = _async_state_worker_dir(state)
    if not pid or not _process_exists(pid):
        return False
    worker_status = _async_worker_status(worker_dir).get("status")
    return worker_status not in ("closing", "closed", "failed", "expired")


def _request_async_worker_close(worker, reason):
    worker_dir = worker.get("worker_dir") if worker else None
    if not worker_dir or not os.path.isdir(worker_dir):
        return
    try:
        _write_json_atomic(
            os.path.join(worker_dir, "close.json"),
            {
                "reason": reason,
                "requested_at_epoch": time.time(),
            },
        )
    except Exception:
        pass


def _worker_has_unapplied_prediction(worker):
    """Return whether a worker owns a result that Mimics has not consumed."""
    if not worker:
        return False
    worker_dir_value = str(worker.get("worker_dir") or "")
    if not worker_dir_value:
        return False
    worker_dir = os.path.abspath(worker_dir_value)
    worker_status = str(_async_worker_status(worker_dir).get("status") or "")
    if worker_status == "running":
        return True

    # ``result_ready`` remains in worker_status.json after Mimics applies the
    # result. The job's pending_sequence is the authoritative indication that
    # the result is still waiting to be consumed.
    jobs_root = os.path.dirname(worker_dir)
    if not os.path.isdir(jobs_root):
        return False
    try:
        names = os.listdir(jobs_root)
    except OSError:
        return False
    for name in names:
        if not name.startswith("job_"):
            continue
        state = _read_json(_async_job_state_path(os.path.join(jobs_root, name)), {}) or {}
        if not state.get("pending_sequence"):
            continue
        state_worker_dir = os.path.abspath(
            str(state.get("worker_dir") or os.path.join(jobs_root, name))
        )
        if state_worker_dir == worker_dir:
            return True
    return False


def _different_model_busy_workers(config):
    """Find other-model workers whose result must be preserved."""
    expected = _model_identity(_model_profile(config))
    busy = []
    seen = set()
    for monitor in list(_ASYNC_MONITORS.values()):
        if monitor.get("done"):
            continue
        state = monitor.get("state") or {}
        if str(state.get("model_identity") or "official") == expected:
            continue
        worker_dir_value = str(_async_state_worker_dir(state) or "")
        worker_dir = (
            os.path.abspath(worker_dir_value) if worker_dir_value else ""
        )
        if worker_dir and worker_dir not in seen:
            seen.add(worker_dir)
            busy.append(
                {
                    "worker_dir": worker_dir,
                    "model_id": state.get("model_id") or "official",
                    "task_name": state.get("task_name") or "",
                }
            )
    for worker in list(_ASYNC_IMAGE_WORKERS.values()):
        if str(worker.get("model_identity") or "official") == expected:
            continue
        worker_dir = os.path.abspath(str(worker.get("worker_dir") or ""))
        if worker_dir in seen or not _worker_has_unapplied_prediction(worker):
            continue
        seen.add(worker_dir)
        busy.append(worker)
    return busy


def _retire_different_model_workers(config):
    """Retire idle prewarm workers before a model-profile switch.

    This runs only after the user has selected an inference action. Workers
    with pending results are rejected earlier by
    ``_different_model_busy_workers`` and are never terminated here.
    """
    expected = _model_identity(_model_profile(config))
    retired = 0
    for cache_key, worker in list(_ASYNC_IMAGE_WORKERS.items()):
        if str(worker.get("model_identity") or "official") == expected:
            continue
        if _worker_has_unapplied_prediction(worker):
            continue
        _request_async_worker_close(worker, "gpu_contention")
        runtime_common.terminate_process_async(
            pid=worker.get("pid"),
            graceful_seconds=1.0,
        )
        _ASYNC_IMAGE_WORKERS.pop(cache_key, None)
        retired += 1
    if retired:
        # The replacement worker waits outside Mimics while an old server
        # releases its lock. Allow enough time for Windows process teardown
        # and watchdog cleanup without blocking the Mimics GUI.
        config["gpu_lock_timeout_seconds"] = max(
            120.0,
            float(config.get("gpu_lock_timeout_seconds", 30) or 30),
        )
        _mimics_log(
            logging.INFO,
            "nnInteractive is switching model profiles in the background. "
            "{0} idle image worker(s) were retired.".format(retired),
        )
    return retired


def _close_all_async_image_workers():
    for worker in list(_ASYNC_IMAGE_WORKERS.values()):
        _request_async_worker_close(worker, "mimics_python_exit")
    _ASYNC_IMAGE_WORKERS.clear()


atexit.register(_close_all_async_image_workers)


def _get_or_start_image_worker(config, image, jobs_root, allow_buffer_export=True):
    image_guid = _object_id(image)
    image_key = _worker_cache_key(config, image)
    profile = _model_profile(config)
    cached = _ASYNC_IMAGE_WORKERS.get(image_key)
    if _shared_image_worker_alive(cached):
        image_input_mode = _effective_image_input_mode(config)
        force_source = image_input_mode in ("source", "source_image", "original", "original_image")
        force_mimics_buffer = image_input_mode in ("mimics", "mimics_buffer", "buffer")
        cached_source = str(cached.get("image_source", "") or "").strip().lower()
        if force_mimics_buffer and cached_source != "mimics_buffer":
            _mimics_log(
                logging.INFO,
                "nnInteractive explicit mimics image mode requires a Mimics-buffer worker; replacing prewarmed source-image worker.",
            )
            _request_async_worker_close(cached, "image_mode_changed_to_mimics")
            _ASYNC_IMAGE_WORKERS.pop(image_key, None)
        elif force_source and cached_source == "mimics_buffer":
            _mimics_log(
                logging.INFO,
                "nnInteractive explicit source image mode requires a source-image worker; replacing prewarmed Mimics-buffer worker.",
            )
            _request_async_worker_close(cached, "image_mode_changed_to_source")
            _ASYNC_IMAGE_WORKERS.pop(image_key, None)
        else:
            return cached
    python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)

    worker_id = "image_worker_" + time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    worker_dir = os.path.join(jobs_root, worker_id)
    inputs_dir = os.path.join(worker_dir, "inputs")
    for path in (
        inputs_dir,
        os.path.join(worker_dir, "commands"),
        os.path.join(worker_dir, "results"),
        os.path.join(worker_dir, "prompts"),
    ):
        os.makedirs(path)

    snapshot_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "nnInteractive image preparation"
    )
    if not snapshot_token:
        shutil.rmtree(worker_dir, ignore_errors=True)
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        raise RuntimeError(
            "nnInteractive cannot prepare its image while {0} is using Mimics buffers. "
            "Wait for that short operation to finish or stop it, then retry.".format(
                owner.get("owner") or "another Mimics-Script task"
            )
        )
    try:
        image_export = _export_image_for_nninteractive(
            config,
            image,
            os.path.join(inputs_dir, "image.raw"),
            allow_buffer_export=allow_buffer_export,
        )
        try:
            current_image = mimics.data.images.get_active()
        except Exception:
            current_image = None
        if current_image is None or _object_id(current_image) != image_guid:
            raise RuntimeError(
                "The active image changed while nnInteractive was preparing its input. "
                "The partial worker was discarded; reactivate the intended image and retry."
            )
    except Exception:
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise
    finally:
        runtime_common.release_local_operation(
            "mask_buffer_access", snapshot_token
        )
    if image_export is None:
        shutil.rmtree(worker_dir, ignore_errors=True)
        _mimics_log(
            logging.INFO,
            "nnInteractive source-image prewarm was skipped because no valid source image metadata is available. No Mimics image buffer export was performed before prompt capture.",
        )
        return None
    empty_base_export = {
        "path": "",
        "shape": image_export["shape"],
        "pixel_count": 0,
        "byte_count": 0,
        "sha256": _empty_mask_sha256(image_export["shape"]),
    }
    parameters = _bridge_parameters(config, image_export, empty_base_export)
    initialize = dict(parameters["request"])
    initialize["async_worker_idle_timeout_seconds"] = _async_worker_idle_timeout(config)
    initialize["async_poll_seconds"] = float(config.get("async_poll_seconds", 0.25))
    initialize["async_worker_control_dir"] = worker_dir
    initialize["initial_seg_path"] = None
    initialize["parent_pid"] = os.getpid()
    _write_json_atomic(os.path.join(worker_dir, "initialize.json"), initialize)
    process, worker_log_path = _start_async_worker(python_exe, bridge_script, worker_dir)
    worker = {
        "worker_dir": worker_dir,
        "pid": process.pid,
        "worker_log": worker_log_path,
        "image_guid": image_guid,
        "cache_key": image_key,
        "image_name": str(getattr(image, "name", "")),
        "shape": image_export["shape"],
        "image_source": image_export.get("source", "mimics_buffer"),
        "image_source_kind": image_export.get("source_kind", ""),
        "image_source_index_space": image_export.get("source_index_space", ""),
        "image_source_modality": image_export.get("source_modality", ""),
        "image_source_world_coordinate_system": image_export.get("source_world_coordinate_system", ""),
        "image_mimics_world_coordinate_system": image_export.get("mimics_world_coordinate_system", ""),
        "image_mimics_to_source_index_matrix": image_export.get("mimics_to_source_index_matrix", ""),
        "image_source_intensity_space": image_export.get("source_intensity_space", ""),
        "image_source_to_mimics_gv_slope": image_export.get("source_to_mimics_gv_slope"),
        "image_source_to_mimics_gv_intercept": image_export.get("source_to_mimics_gv_intercept"),
        "image_input_provenance": image_export.get("image_input_provenance", ""),
        "image_intensity_compatibility": image_export.get("image_intensity_compatibility", ""),
        "image_source_intensity_encoding": image_export.get("source_intensity_encoding", ""),
        "image_source_intensity_recovery_basis": image_export.get("source_intensity_recovery_basis", ""),
        "python": python_exe,
        "python_version": probe.get("version"),
        "folds": folds,
        "runtime_log": parameters["runtime_log"],
        "created_at_epoch": time.time(),
    }
    worker.update(_model_state_values(profile))
    _write_json_atomic(os.path.join(worker_dir, "image_worker.json"), worker)
    _ASYNC_IMAGE_WORKERS[image_key] = worker
    _append_runtime_log(
        parameters["runtime_log"],
        "async_image_worker_started",
        {
            "worker_dir": worker_dir,
            "pid": process.pid,
            "image_guid": image_guid,
            "cache_key": image_key,
            "model_identity": worker.get("model_identity", "official"),
            "image_name": worker["image_name"],
            "image_source": worker.get("image_source", "mimics_buffer"),
            "image_source_kind": worker.get("image_source_kind", ""),
            "image_source_index_space": worker.get("image_source_index_space", ""),
            "image_source_modality": worker.get("image_source_modality", ""),
            "image_source_world_coordinate_system": worker.get("image_source_world_coordinate_system", ""),
            "image_mimics_world_coordinate_system": worker.get("image_mimics_world_coordinate_system", ""),
            "image_mimics_to_source_index_matrix": worker.get("image_mimics_to_source_index_matrix", ""),
            "image_source_intensity_space": worker.get("image_source_intensity_space", ""),
            "image_source_to_mimics_gv_slope": worker.get("image_source_to_mimics_gv_slope"),
            "image_source_to_mimics_gv_intercept": worker.get("image_source_to_mimics_gv_intercept"),
            "image_input_provenance": worker.get("image_input_provenance", ""),
            "image_intensity_compatibility": worker.get("image_intensity_compatibility", ""),
            "image_source_intensity_encoding": worker.get("image_source_intensity_encoding", ""),
            "image_source_intensity_recovery_basis": worker.get("image_source_intensity_recovery_basis", ""),
            "idle_timeout_seconds": initialize["async_worker_idle_timeout_seconds"],
        },
    )
    _mimics_log(
        logging.INFO,
        "nnInteractive AI session is preparing in the background. PID: {0}, idle timeout: {1}s.".format(
            process.pid,
            initialize["async_worker_idle_timeout_seconds"],
        ),
    )
    return worker


def _prewarm_async_image_worker(config, image):
    if not bool(config.get("async_reuse_image_worker", True)):
        return None
    python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)
    jobs_root = os.path.join(_runtime_work_dir(config, model_dir), "async_jobs")
    if not os.path.isdir(jobs_root):
        os.makedirs(jobs_root)
    _cleanup_async_jobs(
        jobs_root,
        config.get("async_job_retention_days", 3),
        config.get("async_job_max_terminal", 20),
    )
    image_input_mode = _effective_image_input_mode(config)
    force_mimics_buffer = image_input_mode in ("mimics", "mimics_buffer", "buffer")
    profile = _model_profile(config)
    task_buffer_fallback = (
        profile.get("source") == "task_model"
        and _task_model_image_input_mode(config) != "source"
        and bool(config.get("fallback_to_mimics_buffer_when_source_unavailable", True))
    )
    # In explicit Mimics mode, prewarm must use buffer export; otherwise the
    # prewarmed source-image worker can be reused by later prompts.
    return _get_or_start_image_worker(
        config,
        image,
        jobs_root,
        allow_buffer_export=bool(force_mimics_buffer or task_buffer_fallback),
    )


def _start_async_job(config, image, target, source=None, write_mode="in_place"):
    source = target if source is None else source
    profile = _model_profile(config)
    cache_key = _worker_cache_key(config, image)
    shared_worker = None
    if bool(config.get("async_reuse_image_worker", True)):
        cached = _ASYNC_IMAGE_WORKERS.get(cache_key)
        if _shared_image_worker_alive(cached):
            shared_worker = cached

    if shared_worker is not None:
        jobs_root = os.path.dirname(shared_worker["worker_dir"])
        python_exe = shared_worker.get("python", "")
        bridge_script = ""
        probe = {"version": shared_worker.get("python_version")}
        folds = shared_worker.get("folds")
    else:
        python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)
        jobs_root = os.path.join(_runtime_work_dir(config, model_dir), "async_jobs")
    if not os.path.isdir(jobs_root):
        os.makedirs(jobs_root)
    _cleanup_async_jobs(
        jobs_root,
        config.get("async_job_retention_days", 3),
        config.get("async_job_max_terminal", 20),
    )

    if shared_worker is None and bool(config.get("async_reuse_image_worker", True)):
        shared_worker = _get_or_start_image_worker(config, image, jobs_root)

    job_id = "job_" + time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    job_dir = os.path.join(jobs_root, job_id)
    inputs_dir = os.path.join(job_dir, "inputs")
    for path in (
        inputs_dir,
        os.path.join(job_dir, "commands"),
        os.path.join(job_dir, "results"),
        os.path.join(job_dir, "prompts"),
    ):
        os.makedirs(path)

    snapshot_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "nnInteractive input snapshot"
    )
    if not snapshot_token:
        shutil.rmtree(job_dir, ignore_errors=True)
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        raise RuntimeError(
            "nnInteractive cannot snapshot the image and Mask while {0} is using Mimics buffers. "
            "Wait for that short operation to finish or stop it, then retry.".format(
                owner.get("owner") or "another Mimics-Script task"
            )
        )
    try:
        if shared_worker is None:
            image_export = _export_image_for_nninteractive(config, image, os.path.join(inputs_dir, "image.raw"))
        else:
            image_export = {
                "kind": "shared_image_worker",
                "path": "",
                "shape": shared_worker["shape"],
                "dtype": "",
                "sha256": "",
                "source": shared_worker.get("image_source", "shared_image_worker"),
                "source_kind": shared_worker.get("image_source_kind", ""),
                "source_index_space": shared_worker.get("image_source_index_space", ""),
                "source_world_coordinate_system": shared_worker.get("image_source_world_coordinate_system", ""),
                "mimics_world_coordinate_system": shared_worker.get("image_mimics_world_coordinate_system", ""),
                "mimics_to_source_index_matrix": shared_worker.get("image_mimics_to_source_index_matrix", ""),
                "image_input_provenance": shared_worker.get("image_input_provenance", ""),
                "image_intensity_compatibility": shared_worker.get("image_intensity_compatibility", ""),
            }
        base_export = _export_mask(source, os.path.join(inputs_dir, "target_at_start.u8"), image_export["shape"])
        if image_export["shape"] != base_export["shape"]:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise RuntimeError(
                "Image and target Mask buffer shapes differ: {0} vs {1}".format(
                    image_export["shape"],
                    base_export["shape"],
                )
            )
        # Use the same Mimics-side lease for the base export and stale-check
        # anchor so another timer cannot modify the target between them.
        target_sha256 = _mask_sha256(target, image_export["shape"])
    finally:
        runtime_common.release_local_operation(
            "mask_buffer_access", snapshot_token
        )
    if shared_worker is None:
        parameters = _bridge_parameters(config, image_export, base_export)
        initialize = dict(parameters["request"])
        initialize["async_worker_idle_timeout_seconds"] = _async_worker_idle_timeout(config)
        initialize["async_poll_seconds"] = float(config.get("async_poll_seconds", 0.25))
        initialize["parent_pid"] = os.getpid()
        _write_json_atomic(os.path.join(job_dir, "initialize.json"), initialize)
        process, worker_log_path = _start_async_worker(python_exe, bridge_script, job_dir)
        worker_dir = job_dir
        worker_pid = process.pid
        runtime_log = parameters["runtime_log"]
    else:
        worker_dir = shared_worker["worker_dir"]
        worker_pid = shared_worker["pid"]
        worker_log_path = shared_worker["worker_log"]
        runtime_log = shared_worker["runtime_log"]

    state = {
        "_job_dir": job_dir,
        "_worker_dir": worker_dir,
        "schema_version": "nninteractive_async_job.v1",
        "job_id": job_id,
        "status": "initializing",
        "worker_dir": worker_dir,
        "image_guid": _object_id(image),
        "image_name": str(getattr(image, "name", "")),
        "launch_project_path": _current_project_path() or "",
        "image_source": image_export.get("source", "mimics_buffer"),
        "image_source_kind": image_export.get("source_kind", ""),
        "image_source_index_space": image_export.get("source_index_space", ""),
        "image_source_world_coordinate_system": image_export.get("source_world_coordinate_system", ""),
        "image_mimics_world_coordinate_system": image_export.get("mimics_world_coordinate_system", ""),
        "image_mimics_to_source_index_matrix": image_export.get("mimics_to_source_index_matrix", ""),
        "image_input_provenance": image_export.get("image_input_provenance", ""),
        "image_intensity_compatibility": image_export.get("image_intensity_compatibility", ""),
        "target_guid": _object_id(target),
        "target_name": str(getattr(target, "name", "")),
        "source_guid": _object_id(source),
        "source_name": str(getattr(source, "name", "")),
        "write_mode": str(write_mode or "in_place"),
        "shape": base_export["shape"],
        "base_path": base_export["path"],
        "base_sha256": base_export["sha256"],
        "base_pixel_count": int(base_export.get("pixel_count") or 0),
        "expected_target_sha256": target_sha256,
        "interactions": [],
        "next_sequence": 1,
        "pending_sequence": None,
        "applied_sequence": 0,
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "python": python_exe,
        "python_version": probe.get("version"),
        "folds": folds,
        "runtime_log": runtime_log,
    }
    state.update(_model_state_values(profile))
    _save_async_job(state)
    if source is not target and _is_ai_draft(target):
        _metadata_set(target, DRAFT_SOURCE_SHA256_METADATA, base_export["sha256"])

    state["pid"] = worker_pid
    state["worker_log"] = worker_log_path
    state["updated_at_epoch"] = time.time()
    _save_async_job(state)
    _metadata_set(target, ASYNC_JOB_METADATA, job_dir)
    _metadata_set(target, MODEL_SOURCE_METADATA, state.get("model_source", "official"))
    _metadata_set(target, MODEL_PROFILE_METADATA, state.get("model_profile_id", "official"))
    _metadata_set(target, MODEL_ID_METADATA, state.get("model_id", "official"))
    _metadata_set(target, MODEL_SHA256_METADATA, state.get("checkpoint_sha256", ""))
    _metadata_set(target, TASK_ID_METADATA, state.get("task_id", ""))
    _append_runtime_log(
        runtime_log,
        "async_job_started",
        {
            "job_id": job_id,
            "job_dir": job_dir,
            "worker_dir": worker_dir,
            "pid": worker_pid,
            "target_guid": state["target_guid"],
            "shared_image_worker": shared_worker is not None,
            "image_source": image_export.get("source", "mimics_buffer"),
            "image_source_kind": image_export.get("source_kind", ""),
            "image_source_index_space": image_export.get("source_index_space", ""),
            "image_source_world_coordinate_system": image_export.get("source_world_coordinate_system", ""),
            "image_mimics_world_coordinate_system": image_export.get("mimics_world_coordinate_system", ""),
            "image_mimics_to_source_index_matrix": image_export.get("mimics_to_source_index_matrix", ""),
        },
    )
    return state


def _persist_interaction(job_dir, interaction):
    result = dict(interaction)
    mask_path = result.get("mask_path")
    if mask_path:
        suffix = os.path.splitext(mask_path)[1] or ".u8"
        target_path = os.path.join(
            job_dir,
            "prompts",
            "prompt_" + uuid.uuid4().hex + suffix,
        )
        shutil.copy2(mask_path, target_path)
        result["mask_path"] = target_path
    if result.get("interaction_type") == "scribble_set":
        persisted = []
        for item in result.get("scribbles", []):
            persisted.append(_persist_interaction(job_dir, item))
        result["scribbles"] = persisted
    return result


def _interaction_prediction_step_count(interaction):
    interaction_type = str(interaction.get("interaction_type") or "")
    if interaction_type == "point_set":
        return max(1, len(interaction.get("points") or []))
    if interaction_type == "scribble_set":
        return max(1, len(interaction.get("scribbles") or []))
    return 1


def _enqueue_async_prediction(
    state, target, expected_hash=None, replay_all=False
):
    worker_dir = _async_state_worker_dir(state)
    sequence = _next_async_sequence(worker_dir)
    if expected_hash is None:
        expected_hash = _mask_sha256(target, state.get("shape"))
    interactions = state.get("interactions", [])
    if replay_all:
        prediction_steps = sum(
            _interaction_prediction_step_count(value)
            for value in interactions
        )
    else:
        prediction_steps = (
            _interaction_prediction_step_count(interactions[-1])
            if interactions
            else 1
        )
    command = {
        "schema_version": "nninteractive_async_command.v1",
        "command_id": uuid.uuid4().hex,
        "sequence": sequence,
        "created_at_epoch": time.time(),
        "expected_target_sha256": expected_hash,
        "interactions": interactions,
        "prediction_steps_hint": max(1, int(prediction_steps)),
        "initial_seg_path": state["base_path"] if state.get("base_path") else None,
        "initial_seg_shape": state.get("shape"),
        "target_guid": state.get("target_guid"),
        "target_name": state.get("target_name"),
        "job_dir": state.get("_job_dir"),
        "model_identity": state.get("model_identity", "official"),
        "model_profile_id": state.get("model_profile_id", "official"),
        "model_id": state.get("model_id", "official"),
        "checkpoint_sha256": state.get("checkpoint_sha256", ""),
        "effective_model_fingerprint": state.get(
            "effective_model_fingerprint", ""
        ),
        "task_id": state.get("task_id", ""),
    }
    state["pending_sequence"] = sequence
    state["pending_prediction_steps"] = max(1, int(prediction_steps))
    state["next_sequence"] = sequence + 1
    state["expected_target_sha256"] = expected_hash
    state["status"] = "queued"
    state["updated_at_epoch"] = time.time()
    _save_async_job(state)
    command_path = os.path.join(
        worker_dir,
        "commands",
        "command_{0:06d}.json".format(sequence),
    )
    _write_json_atomic(command_path, command)
    _mimics_log(
        logging.INFO,
        "nnInteractive prompt queued for background inference. Sequence: {0}; "
        "sequential prediction steps: {1}. Mimics will apply only the final "
        "result.".format(sequence, max(1, int(prediction_steps))),
    )
    return sequence


def _async_result_path(state, sequence):
    primary = os.path.join(
        _async_state_worker_dir(state),
        "results",
        "result_{0:06d}.json".format(int(sequence)),
    )
    legacy = os.path.join(
        state["_job_dir"],
        "results",
        "result_{0:06d}.json".format(int(sequence)),
    )
    if primary != legacy and os.path.isfile(legacy) and not os.path.isfile(primary):
        return legacy
    return primary


def _error_guidance(error_text, stage=""):
    """Map a raw nnInteractive error to a plain-English category and action.

    Returns (category, message, suggested_action) - all strings. category is
    one of: out_of_memory, server_unavailable, environment_broken,
    stale_target, unknown. Used by the failure dialogs so every error the
    annotator can see comes with a concrete next step.
    """
    text = str(error_text or "").lower()
    stage_text = str(stage or "").lower()
    combined = text + " " + stage_text
    if (
        "out of memory" in combined
        or "cuda error" in combined
        or "cudnn error" in combined
        or "oom" in combined
    ):
        return (
            "out_of_memory",
            "The AI model ran out of GPU memory.",
            "Close other GPU programs or stop running AI training jobs "
            "(Admin > Stop All Owned Background Services), then retry. "
            "If it keeps failing, try a smaller image or restart Mimics.",
        )
    if (
        "connection refused" in combined
        or "server is not running" in combined
        or "server died" in combined
        or "status code 500" in combined
        or "http 500" in combined
        or "internal server error" in combined
        or "not enough free gpu memory" in combined
    ):
        return (
            "server_unavailable",
            "The nnInteractive AI server stopped or could not start.",
            "Retry from the menu. If it fails again, run "
            "Admin > Setup / Repair Environment, or stop background services "
            "and retry.",
        )
    if (
        "no module named" in combined
        or "importerror" in combined
        or "modulenotfounderror" in combined
        or ("could not find" in combined and ("environment" in combined or "python" in combined))
        or "not verified" in combined
        or (
            "environment" in combined
            and ("broken" in combined or "missing" in combined or "not found" in combined)
        )
    ):
        return (
            "environment_broken",
            "The nnInteractive Python environment is incomplete or damaged.",
            "Run Admin > Setup / Repair Environment, then retry.",
        )
    if "changed" in combined and (
        "mask" in combined or "target" in combined or "project" in combined
    ):
        return (
            "stale_target",
            "The Mask or project changed while the AI was running, so the "
            "result can no longer be applied safely.",
            "Start a new AI session from the current Mask.",
        )
    return (
        "unknown",
        str(error_text or "Unknown error"),
        "Retry from the menu. If it keeps failing, check the log files in "
        "the AI session folder or contact support.",
    )


def _describe_async_worker_failure(worker, worker_status):
    stage = worker.get("stage", worker_status or "unknown")
    error = worker.get("error")
    if not error and stage == "idle_timeout":
        timeout_seconds = worker.get("idle_timeout_seconds")
        if timeout_seconds is None:
            error = (
                "The AI session was prepared, but no prompt result was produced before "
                "the worker idle timeout. This usually means the worker expired while "
                "waiting for a prompt, or the prompt command was submitted after the "
                "worker had already exited."
            )
        else:
            error = (
                "The AI session was prepared, but no prompt result was produced before "
                "the worker idle timeout ({0}s). This usually means the worker expired "
                "while waiting for a prompt, or the prompt command was submitted after "
                "the worker had already exited."
            ).format(int(timeout_seconds))
    if not error:
        error = "No result file was produced."
    return stage, error


def _show_async_running(target, state):
    worker = _async_worker_status(_async_state_worker_dir(state))
    stage = worker.get("stage", worker.get("status", state.get("status", "running")))
    answer = mimics.dialogs.question_box(
        message=(
            "nnInteractive is still running in the background.\n\n"
            "Stage: {0}\n"
            "You can continue using Mimics.\n\n"
            "Run nnInteractive again later to apply the result."
        ).format(stage),
        buttons="Keep Running;" + BUTTON_DISCARD_SESSION,
        title="nnInteractive Running",
        ui_blocking=True,
    )
    if answer == BUTTON_DISCARD_SESSION:
        _close_async_job(target, state, "user_cancelled_running_job")
        return "restart"
    return "waiting"


def _check_async_result_nonblocking(image, target, state):
    """Check async progress without prompting the user to relaunch the tool."""
    sequence = state.get("pending_sequence")
    if sequence is None:
        return "ready"

    result_path = _async_result_path(state, sequence)
    if not os.path.isfile(result_path):
        worker = _async_worker_status(_async_state_worker_dir(state))
        worker_status = worker.get("status")
        if worker_status in ("failed", "expired", "closed") or (
            state.get("pid") and not _process_exists(state.get("pid"))
        ):
            stage, error = _describe_async_worker_failure(worker, worker_status)
            raise RuntimeError(
                "The nnInteractive background worker stopped before producing a result.\n\n"
                "Stage: {0}\nError: {1}".format(
                    stage,
                    error,
                )
            )
        return "waiting"

    return _handle_async_result(image, target, state)


def _stop_async_monitor(job_dir):
    monitor = _ASYNC_MONITORS.pop(job_dir, None)
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


def _async_monitor_tick(monitor):
    if monitor.get("done"):
        return
    # _check_async_result_nonblocking may show a ui_blocking dialog (stale
    # target, worker error, ...). That dialog pumps the Mimics message loop,
    # so this Win32 timer fires again while the dialog is still open and
    # re-enters _handle_async_result, producing the "result dialog pops twice"
    # symptom. Guard re-entry for the whole tick.
    if monitor.get("busy"):
        return
    operation_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "nnInteractive result monitor"
    )
    if not operation_token:
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        owner_text = str(owner.get("owner") or "another Mask operation")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "nninteractive_buffer_wait",
            detail=owner_text,
            interval_seconds=60.0,
            initial_delay_seconds=5.0,
        )
        if due:
            _mimics_log(
                logging.INFO,
                "nnInteractive result handling is waiting for {0} ({1}s). "
                "Run nnInteractive again to inspect or discard the pending "
                "session.".format(owner_text, int(elapsed)),
            )
        return
    runtime_common.clear_progress_notice(monitor, "nninteractive_buffer_wait")
    monitor["busy"] = True
    job_dir = monitor["state"]["_job_dir"]
    try:
        if time.time() > monitor["deadline"]:
            raise RuntimeError(
                "nnInteractive background prediction timed out after {0} seconds.\n"
                "The result was not produced in time.".format(int(monitor["timeout_seconds"]))
            )
        outcome = _check_async_result_nonblocking(
            monitor["image"],
            monitor["target"],
            monitor["state"],
        )
        if outcome == "waiting":
            worker = _async_worker_status(
                _async_state_worker_dir(monitor["state"])
            )
            stage = str(
                worker.get("stage")
                or worker.get("status")
                or monitor["state"].get("status")
                or "running"
            )
            sequence = monitor["state"].get("pending_sequence")
            due, elapsed = runtime_common.progress_notice_due(
                monitor,
                "nninteractive_inference",
                detail="{0}|{1}".format(stage, sequence),
                interval_seconds=60.0,
                initial_delay_seconds=30.0,
            )
            if due:
                _mimics_log(
                    logging.INFO,
                    "nnInteractive prediction is still running ({0}s). "
                    "Stage: {1}; sequence: {2}. Mimics remains available. "
                    "Run nnInteractive again to inspect or discard it.".format(
                        int(elapsed), stage, sequence
                    ),
                )
        if outcome != "waiting":
            runtime_common.clear_progress_notice(
                monitor, "nninteractive_inference"
            )
            if outcome == "applied":
                # Continuous prompting: keep the monitor alive and offer the
                # next prompt immediately instead of forcing the user back
                # into the menu entry for every prompt. The session ends
                # through Finish / UNDO-to-empty / RESET inside the prompt
                # menu, or through the waiting/restart/error paths below.
                monitor["deadline"] = time.time() + monitor["timeout_seconds"]
                continued = _continue_session_prompt(
                    monitor["image"],
                    monitor["target"],
                    monitor["state"],
                    monitor.get("config") or {},
                )
                if continued:
                    return
            monitor["done"] = True
            _stop_async_monitor(job_dir)
    except Exception as error:
        monitor["done"] = True
        _stop_async_monitor(job_dir)
        _category, guidance_message, suggested_action = _error_guidance(
            error, "monitor"
        )
        mimics.dialogs.message_box(
            "nnInteractive background prediction failed.\n\n{0}\n\n{1}\n\n"
            "Suggested action: {2}".format(error, guidance_message, suggested_action),
            title="nnInteractive Failed",
            ui_blocking=True,
        )
    finally:
        monitor["busy"] = False
        runtime_common.release_local_operation("mask_buffer_access", operation_token)


def _start_win32_async_result_monitor(image, target, state, config, poll_seconds, timeout_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    job_dir = state["_job_dir"]
    _stop_async_monitor(job_dir)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    monitor = {
        "timer": None,
        "image": image,
        "target": target,
        "state": state,
        "config": config,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
    }

    def _timer_proc(hwnd, message, timer_id, tick_count):
        _async_monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    # Another Mimics module may have configured SetTimer with a distinct
    # WINFUNCTYPE class. Passing the callback as a raw pointer avoids that
    # ctypes type-identity mismatch.
    user32.SetTimer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    user32.KillTimer.restype = ctypes.c_int
    callback_void = ctypes.cast(callback, ctypes.c_void_p)
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_void)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["callback_void"] = callback_void
    monitor["win32_timer"] = (user32, timer_id)
    _ASYNC_MONITORS[job_dir] = monitor
    return True


def _start_async_result_monitor(image, target, state, config):
    """Start a non-blocking Mimics-side timer that applies async results."""
    poll_seconds = float(config.get("async_result_poll_seconds", config.get("async_poll_seconds", 0.25)))
    configured_timeout = config.get("async_result_wait_timeout_seconds")
    if configured_timeout is not None:
        timeout_seconds = float(configured_timeout)
    else:
        prediction_steps = max(
            1, int(state.get("pending_prediction_steps") or 1)
        )
        timeout_seconds = (
            float(config.get("prediction_timeout_seconds", 1800))
            * prediction_steps
            + 120
        )

    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_async_result_monitor(image, target, state, config, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            "Background inference started.\n\n"
            "Mimics did not expose a usable timer API in this session, so the result "
            "cannot be applied automatically. Run nnInteractive again later to "
            "check the result.",
            title="nnInteractive Running",
            ui_blocking=False,
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_async_result_monitor(image, target, state, config, poll_seconds, timeout_seconds):
            return True
        mimics.dialogs.message_box(
            "Background inference started.\n\n"
            "No active application timer was found, so the result cannot be applied "
            "automatically. Run nnInteractive again later to check the result.",
            title="nnInteractive Running",
            ui_blocking=False,
        )
        return False

    timer = QTimer()
    job_dir = state["_job_dir"]
    _stop_async_monitor(job_dir)
    monitor = {
        "timer": timer,
        "image": image,
        "target": target,
        "state": state,
        "config": config,
        "done": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
    }
    _ASYNC_MONITORS[job_dir] = monitor

    def _tick():
        _async_monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(max(0.1, poll_seconds) * 1000)))
    return True


def _handle_async_result(image, target, state):
    sequence = state.get("pending_sequence")
    if sequence is None:
        return "ready"
    result_path = _async_result_path(state, sequence)
    if not os.path.isfile(result_path):
        worker = _async_worker_status(_async_state_worker_dir(state))
        worker_status = worker.get("status")
        if worker_status in ("failed", "expired", "closed") or (
            state.get("pid") and not _process_exists(state.get("pid"))
        ):
            stage, error = _describe_async_worker_failure(worker, worker_status)
            _category, guidance_message, suggested_action = _error_guidance(
                error, stage
            )
            answer = mimics.dialogs.question_box(
                message=(
                    "The nnInteractive background worker stopped before producing a result.\n\n"
                    "Stage: {0}\nError: {1}\n\n"
                    "{2}\n\n"
                    "Suggested action: {3}\n\n"
                    "Start a new AI session from the current Mask?"
                ).format(
                    stage,
                    error,
                    guidance_message,
                    suggested_action,
                ),
                buttons=BUTTON_START_CURRENT + ";" + BUTTON_CANCEL,
                title="nnInteractive Worker Stopped",
                ui_blocking=True,
            )
            if answer == BUTTON_START_CURRENT:
                _close_async_job(target, state, "worker_stopped")
                return "restart"
            return "waiting"
        return _show_async_running(target, state)

    result = _read_json(result_path, {}) or {}
    if not isinstance(result, dict):
        result = {}
    expected_model_identity = str(state.get("model_identity") or "official")
    result_model_identity = str(result.get("model_identity") or "official")
    if result_model_identity != expected_model_identity:
        raise RuntimeError(
            "The nnInteractive result was produced by a different model. "
            "The result was not applied. Expected {0}, received {1}.".format(
                expected_model_identity,
                result_model_identity,
            )
        )
    status = result.get("status")

    if not status:
        worker = _async_worker_status(_async_state_worker_dir(state))
        worker_status = worker.get("status")
        if worker_status in ("failed", "expired", "closed"):
            raise RuntimeError(
                "The nnInteractive worker did not return a valid result payload.\n\n"
                "Worker status: {0}\n"
                "Result file: {1}\n"
                "Error: {2}".format(
                    worker_status,
                    result_path,
                    worker.get("error", "missing status field in result json"),
                )
            )
        # Result file may still be in-flight from external I/O / antivirus interference.
        return "waiting"

    if status == "error":
        error_stage = result.get("stage", "prediction")
        error_text = result.get("error", "Unknown error")
        _category, guidance_message, suggested_action = _error_guidance(
            error_text, error_stage
        )
        answer = mimics.dialogs.question_box(
            message=(
                "The background prediction failed.\n\n"
                "Stage: {0}\n"
                "Error: {1}\n\n"
                "{2}\n\n"
                "Suggested action: {3}\n\n"
                "Retry keeps the same prompts. Discard starts a new AI session "
                "from the current Mask."
            ).format(
                error_stage,
                error_text,
                guidance_message,
                suggested_action,
            ),
            buttons=BUTTON_RETRY + ";" + BUTTON_DISCARD_SESSION + ";" + BUTTON_CANCEL,
            title="nnInteractive Failed",
            ui_blocking=True,
        )
        if answer == BUTTON_RETRY:
            state["pending_sequence"] = None
            _enqueue_async_prediction(state, target, replay_all=True)
            _show_async_running(target, state)
            return "waiting"
        if answer == BUTTON_DISCARD_SESSION:
            _close_async_job(target, state, "prediction_failed_discarded")
            return "restart"
        state["pending_sequence"] = None
        state["status"] = "failed"
        state["updated_at_epoch"] = time.time()
        _save_async_job(state)
        return "ready"

    if status == "skipped":
        state["pending_sequence"] = None
        state["status"] = "ready"
        state["updated_at_epoch"] = time.time()
        _save_async_job(state)
        return "ready"

    if status != "refined":
        raise RuntimeError(
            "nnInteractive returned an unexpected result status: {0}\n"
            "Result file: {1}".format(
                status,
                result_path,
            )
        )
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    launch_project = str(state.get("launch_project_path") or "")
    project_changed = bool(
        launch_project
        and not _same_project_path(_current_project_path(), launch_project)
    )
    if (
        project_changed
        or active_image is None
        or _object_id(active_image) != state.get("image_guid")
    ):
        _mimics_log(
            logging.WARNING,
            "nnInteractive result was not applied because the active project or image changed while prediction was running. "
            "The newly opened project was not modified; start a new AI session there if needed.",
        )
        _close_async_job(target, state, "active_project_changed")
        return "ready"
    if _object_id(image) != state.get("image_guid") or _object_id(target) != state.get(
        "target_guid"
    ):
        raise RuntimeError(
            "The active image or target Mask no longer matches the background job. "
            "The result was not applied."
        )
    current_hash = _mask_sha256(target, state.get("shape"))
    expected_hash = result.get("expected_target_sha256")
    if current_hash != expected_hash:
        _mimics_log(
            logging.WARNING,
            "nnInteractive result treated as stale: target hash changed during inference. "
            "expected={0} current={1} target_pixels={2} state_shape={3} result_shape={4} "
            "state_expected={5} target_guid={6} state_target_guid={7}.".format(
                expected_hash,
                current_hash,
                int(getattr(target, "number_of_pixels", -1)),
                state.get("shape"),
                result.get("shape"),
                state.get("expected_target_sha256"),
                _object_id(target),
                state.get("target_guid"),
            ),
        )
        answer = mimics.dialogs.question_box(
            message=(
                "The target Mask changed while nnInteractive was running.\n\n"
                "The background result is now stale and will not be applied. "
                "Start a new AI session from the current Mask?"
            ),
            buttons=BUTTON_START_CURRENT + ";" + BUTTON_CANCEL,
            title="nnInteractive Result Is Stale",
            ui_blocking=True,
        )
        if answer == BUTTON_START_CURRENT:
            _close_async_job(target, state, "target_changed")
            return "restart"
        # Cancel: the stale result is unusable, so close the session instead of
        # leaving it active. A lingering session would make _select_session_masks
        # force in_place next time, hiding the "overwrite / new mask" choice, and
        # "waiting" would keep re-reading the same result file and re-prompting.
        _close_async_job(target, state, "target_changed_cancelled")
        return "ready"

    output_path = result.get("output_path")
    if not output_path or not os.path.isfile(output_path):
        raise RuntimeError(
            "nnInteractive completed but the prediction buffer is missing:\n{0}".format(
                output_path
            )
        )
    target = _choose_completed_result_target(image, target, state, result)
    _set_mask_from_u8(target, output_path, state["shape"])
    _make_mask_visible(target)
    # Clean up visual objects that were kept visible during inference.
    job_dir = state.get("_job_dir")
    if job_dir and job_dir in _ASYNC_VISUAL_OBJECTS:
        for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
            _delete_mimics_object(obj)
    state["pending_sequence"] = None
    state["pending_prediction_steps"] = 0
    state["applied_sequence"] = int(sequence)
    # Re-anchor the stale check on the mask buffer we just wrote, not on the
    # output file: the next prompt compares _mask_sha256(target, ...) against
    # this value, so they must use the same source. _sha256_file(output_path)
    # here made every subsequent prompt flag a "manual mask change" and restart.
    state["expected_target_sha256"] = _mask_sha256(target, state.get("shape"))
    state["status"] = "ready"
    state["updated_at_epoch"] = time.time()
    _save_async_job(state)
    _mimics_log(
        logging.INFO,
        "nnInteractive result applied to Mask {0}. Foreground voxels: {1}, "
        "sequential prediction steps: {2}, elapsed: {3}s, device: {4}. "
        "Timing: {5}.".format(
            getattr(target, "name", ""),
            result.get("foreground_voxels", "?"),
            result.get("prediction_steps", 1),
            result.get("elapsed_seconds", "?"),
            result.get("device", "?"),
            _timing_summary(result) or "not available",
        ),
    )
    if int(result.get("foreground_voxels", -1)) == 0:
        _mimics_log(
            logging.WARNING,
            "nnInteractive returned an empty mask (0 foreground voxels). "
            "Try moving the foreground point closer to the structure center and "
            "use background points farther away.",
        )
    return "applied"


def _cleanup_stale_processes():
    """Clean safe stale runtime state from a previous crashed session.

    The process registry sweep runs first: it clears dead records,
    terminates registered orphans (parent gone, kill-on-sweep policy) and
    releases locks they still hold - proving ownership via PID + start
    marker, so no aggressive flag is needed for those.

    The legacy cmdline path below still handles unregistered processes
    (started before the registry existed) and owned nnInteractive servers
    whose watchdog is gone and whose idle timeout has elapsed. Killing
    other live bridge/worker processes stays opt-in via
    MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START=1 or the explicit Stop
    Background Services entry.

    Uses a single hidden batch PowerShell call instead of per-process calls
    to avoid popping up visible console windows that freeze Mimics.
    """
    if os.name != "nt":
        return
    killed = []
    locks_removed = runtime_common.cleanup_stale_resource_locks(_resource_lock_dir())
    registry_summary = runtime_common.sweep_processes(
        runtime_common.project_root()
    )
    registry_killed = len(registry_summary.get("terminated_orphans") or [])
    locks_removed += len(registry_summary.get("released_locks") or [])
    _SERVER_PROTECT_MARKERS = (
        "nninteractive.inference.server.main",
        "--watchdog",
    )
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_TERMINATE = 0x0001
        kernel32.OpenProcess.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        # Single hidden PowerShell call to get all python.exe PIDs and
        # command lines at once.  This replaces the previous per-process
        # approach that spawned a visible PowerShell window for every
        # python.exe, causing Mimics to black-screen.
        proc = subprocess.Popen(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' } | "
                "Select-Object ProcessId,CommandLine | "
                "ConvertTo-Json -Compress",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_hidden_process_kwargs()
        )
        stdout, _ = proc.communicate(timeout=15)
        if proc.returncode != 0 or not stdout or not stdout.strip():
            return
        try:
            records = json.loads(stdout.decode("utf-8", errors="replace"))
        except ValueError:
            return
        if isinstance(records, dict):
            records = [records]

        killed.extend(_cleanup_stale_owned_servers(records))

        if _aggressive_auto_cleanup_enabled():
            env_root = _environment_root().lower()
            for record in records or []:
                try:
                    pid = int(record.get("ProcessId", 0))
                except (TypeError, ValueError):
                    continue
                if not pid:
                    continue
                cmdline = str(record.get("CommandLine") or "").lower()
                is_protected = any(
                    marker in cmdline
                    for marker in _SERVER_PROTECT_MARKERS
                )
                if not is_protected and env_root in cmdline:
                    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        killed.append(pid)
    except Exception:
        pass
    if killed or locks_removed or registry_killed:
        pieces = []
        if killed:
            pieces.append("terminated {0} stale integration process(es)".format(len(killed)))
        if registry_killed:
            pieces.append(
                "terminated {0} orphaned registered process(es)".format(registry_killed)
            )
        if locks_removed:
            pieces.append("removed {0} stale resource lock file(s)".format(locks_removed))
        _mimics_log(
            logging.INFO,
            "Startup cleanup: {0}.".format(", ".join(pieces)),
        )


def main():
    try:
        # Run cleanup in a background thread so Mimics UI does not freeze
        # while PowerShell queries process list.
        if runtime_common.auto_cleanup_enabled():
            cleanup_thread = threading.Thread(target=_cleanup_stale_processes)
            cleanup_thread.daemon = True
            cleanup_thread.start()
        result = run()
        # Async mode returns 0 after submitting a prompt while the background
        # worker keeps running; only log "ended" when no monitor is active.
        if result == 0 and not _ASYNC_MONITORS:
            _mimics_log(logging.INFO, "nnInteractive session ended.")
        return result
    except Exception as error:
        _mimics_log(logging.ERROR, "nnInteractive error: {0}".format(error))
        mimics.dialogs.message_box(
            "nnInteractive could not continue.\n\n{0}".format(str(error)),
            title=TITLE,
            ui_blocking=True,
        )
        return 2


if __name__ == "__main__":
    main()
