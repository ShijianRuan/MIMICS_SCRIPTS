# -*- coding: utf-8 -*-
"""Standalone nnInteractive tool for Mimics Research 21.

This module runs inside Mimics Python 3.5. It intentionally has no dependency
on SegmentationPlatform cases, reviews, registries, or runtime manifests.

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
BUTTON_CREATE = "Create New Result Mask"
BUTTON_CANCEL = "Cancel"
BUTTON_DISCARD_SESSION = "Discard AI Session"
BUTTON_RETRY = "Retry Prediction"
BUTTON_START_CURRENT = "Start From Current Mask"

PROMPT_MASK_PREFIX = "nnInteractive Prompt"
DEFAULT_RESULT_NAME = "nnInteractive Result"
ASYNC_JOB_METADATA = "nninteractive.async_job_path"
_ASYNC_VISUAL_OBJECTS = {}  # job_dir -> list of Mimics objects to delete after inference
_RUNTIME_PROBE_CACHE = {}
_ASYNC_MONITORS = {}
_ASYNC_IMAGE_WORKERS = {}  # image guid -> shared worker state


def _find_root(start_dir, sentinel_files, max_depth=6):
    """Walk upward from *start_dir* until a directory containing one of the
    *sentinel_files* is found.

    In the repo layout the root contains ``nninteractive_config.json``.  In an
    exported worklist the root contains ``worklist_manifest.json``.  A bundle
    is detected by the presence of ``nninteractive_env``.
    """
    current = os.path.abspath(start_dir)
    for _ in range(max_depth):
        for sentinel in sentinel_files:
            if os.path.isfile(os.path.join(current, sentinel)):
                return current
            if os.path.isdir(os.path.join(current, sentinel)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: nninteractive_mimics.py is at <root>/runtime_py35/ so two dirname
    # hops reach the parent of runtime_py35, which is the root in both the worklist
    # and bundle layouts.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("worklist_manifest.json", "nninteractive_config.json", "nninteractive_env", ".git"),
    )


def _integration_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _environment_root():
    # The environment may be inside the worklist, beside it, under the project
    # root, or be the standalone bundle root itself.
    root = _project_root()
    candidates = [
        os.path.join(root, "nninteractive_env"),
        os.path.join(os.path.dirname(root), "nninteractive_env"),
        root,
    ]
    for candidate in candidates:
        if (
            os.path.isfile(os.path.join(candidate, "python", "python.exe"))
            or os.path.isfile(os.path.join(candidate, "Scripts", "python.exe"))
        ):
            return candidate
    return candidates[0]


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


def _hidden_process_kwargs():
    """Return subprocess kwargs that suppress console windows on Windows.

    Mimics embeds Python 3.5.2 where ``subprocess.CREATE_NO_WINDOW`` may not
    exist.  We use ``ctypes`` to build the ``STARTUPINFO`` manually so the
    flag is always applied regardless of the Python version.
    """
    if os.name != "nt":
        return {}
    import ctypes

    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE

    # CREATE_NO_WINDOW (0x08000000) | CREATE_NEW_PROCESS_GROUP (0x00000200)
    creationflags = 0x08000000 | 0x00000200

    return {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
    }


def _model_folds(model_dir):
    result = []
    for name in sorted(os.listdir(model_dir)):
        checkpoint = os.path.join(model_dir, name, "checkpoint_final.pth")
        if name.startswith("fold_") and os.path.isfile(checkpoint):
            result.append(name[5:])
    return result


def _runtime_log_path(model_dir):
    log_dir = os.path.join(os.path.dirname(model_dir), "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, "nninteractive_mimics.log")


def _append_runtime_log(path, event, details=None):
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
    }
    if details:
        payload.update(details)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json_atomic(path, value):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _read_json(path, default=None):
    try:
        return _load_json(path)
    except (IOError, OSError, ValueError):
        return default


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

    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    # Windows (Mimics embedded Python): os.kill(pid, 0) may raise
    # "<built-in function kill> returned a result with an error set".
    # Use WinAPI instead to avoid that CPython-level SystemError.
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


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _probe_python(python_exe, timeout):
    """Check that the external Python has the required packages.

    Runs synchronously but with a short timeout.  The result is cached so
    subsequent calls are instant.  If the probe fails, the error is logged
    but **not raised** — the bridge process itself will surface real
    environment problems when it starts, and we don't want to block the
    Mimics UI for a lengthy subprocess call.
    """
    cache_key = (os.path.abspath(python_exe), int(timeout))
    cached = _RUNTIME_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached
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
    try:
        process = subprocess.Popen(
            [python_exe, "-c", probe],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_hidden_process_kwargs()
        )
    except (OSError, ValueError):
        _mimics_log(
            logging.WARNING,
            "Could not launch environment check for {0}. "
            "Proceeding — the bridge will report any real issues.".format(python_exe),
        )
        result = {"python": python_exe, "version": [], "missing": [], "failed": True}
        _RUNTIME_PROBE_CACHE[cache_key] = result
        return result
    try:
        stdout, stderr = process.communicate(timeout=min(timeout, 30))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        _mimics_log(
            logging.WARNING,
            "Environment check timed out after 30s for {0}. "
            "Proceeding — the bridge will report any real issues.".format(python_exe),
        )
        result = {"python": python_exe, "version": [], "missing": [], "timed_out": True}
        _RUNTIME_PROBE_CACHE[cache_key] = result
        return result
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        _mimics_log(
            logging.WARNING,
            "Environment check failed for {0}: {1}. "
            "Proceeding — the bridge will report any real issues.".format(python_exe, detail),
        )
        result = {"python": python_exe, "version": [], "missing": [], "failed": True}
        _RUNTIME_PROBE_CACHE[cache_key] = result
        return result
    try:
        result = json.loads(stdout.decode("utf-8"))
    except ValueError:
        result = {"python": python_exe, "version": [], "missing": []}
    _RUNTIME_PROBE_CACHE[cache_key] = result
    return result


def _runtime_paths(config):
    root = _project_root()
    integration_root = _integration_root()
    environment_root = _environment_root()
    python_candidates = [
        os.environ.get("NNINTERACTIVE_PYTHON", ""),
        config.get("python", ""),
        os.path.join(environment_root, "python", "python.exe"),
        os.path.join(environment_root, "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "python", "python.exe"),
    ]
    python_exe = _first_existing_file(python_candidates, "nnInteractive Python")
    bridge_script = _first_existing_file(
        [
            os.environ.get("NNINTERACTIVE_BRIDGE", ""),
            config.get("bridge_script", ""),
            os.path.join(integration_root, "nninteractive_bridge.py"),
            os.path.join(integration_root, "bridge", "nninteractive_bridge.py"),
        ],
        "nnInteractive bridge script",
    )
    model_dir = _first_existing_dir(
        [
            os.environ.get("NNINTERACTIVE_MODEL_DIR", ""),
            config.get("model_dir", ""),
            os.path.join(environment_root, "models", "nnInteractive_v1.0"),
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
    probe_timeout = int(
        os.environ.get(
            "NNINTERACTIVE_PROBE_TIMEOUT",
            config.get("environment_probe_timeout_seconds", 180),
        )
    )
    probe = _probe_python(python_exe, probe_timeout)
    return python_exe, bridge_script, model_dir, probe, folds


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


def _unique_result_name():
    existing = set(str(getattr(mask, "name", "")) for mask in mimics.data.masks)
    if DEFAULT_RESULT_NAME not in existing:
        return DEFAULT_RESULT_NAME
    index = 2
    while "{0} {1}".format(DEFAULT_RESULT_NAME, index) in existing:
        index += 1
    return "{0} {1}".format(DEFAULT_RESULT_NAME, index)


def _create_result_mask(image):
    mask = mimics.segment.create_mask()
    mask.name = _unique_result_name()
    try:
        mask.image = image
    except Exception as error:
        try:
            mimics.data.masks.delete(mask)
        except Exception:
            pass
        raise RuntimeError("Could not bind the new result Mask to the active image: {0}".format(error))
    mask.visible = True
    mask.selected = True
    return mask


def _select_target_mask(image):
    selected = [mask for mask in _masks_for_image(image) if bool(getattr(mask, "selected", False))]
    if len(selected) == 1:
        target = selected[0]
    elif len(selected) > 1:
        raise RuntimeError(
            "Select exactly one target Mask in the Project Tree, then run nnInteractive again."
        )
    else:
        answer = mimics.dialogs.question_box(
            message=(
                "No target Mask is selected.\n\n"
                "Select an existing Mask in the Project Tree, or create a new empty result Mask."
            ),
            buttons=BUTTON_CREATE + ";" + BUTTON_CANCEL,
            title=TITLE,
            ui_blocking=True,
        )
        if answer != BUTTON_CREATE:
            return None
        target = _create_result_mask(image)

    target_image = getattr(target, "image", None)
    if target_image is None:
        try:
            target.image = image
        except Exception as error:
            raise RuntimeError("The target Mask is not bound to the active image: {0}".format(error))
    elif not _same_object(target_image, image):
        raise RuntimeError(
            "The selected Mask belongs to a different image set. "
            "Activate its image set or select another Mask."
        )
    return target


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


def _export_image(image, path):
    view = image.get_voxel_buffer()
    with open(path, "wb") as handle:
        handle.write(view.tobytes())
    return {
        "path": path,
        "shape": [int(value) for value in view.shape],
        "dtype": _buffer_dtype(view),
        "sha256": _sha256_file(path),
    }


def _export_mask(mask, path):
    view = mask.get_voxel_buffer()
    raw = view.tobytes()
    with open(path, "wb") as handle:
        handle.write(raw)
    return {
        "path": path,
        "shape": [int(value) for value in view.shape],
        "pixel_count": int(getattr(mask, "number_of_pixels", 0)),
        "byte_count": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _mask_sha256(mask):
    return _sha256_bytes(mask.get_voxel_buffer().tobytes())


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


def _restore_base(mask, base_path, base_shape):
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


def _delete_point_marker(point):
    marker = point.get("_marker")
    if marker is None:
        return
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
                    if _visual_objects is not None and point.get("_marker"):
                        _visual_objects.append(point["_marker"])
            elif answer == BUTTON_EXCLUDE_POINT:
                point = _capture_point(image, False)
                if point is not None:
                    points.append(point)
                    if _visual_objects is not None and point.get("_marker"):
                        _visual_objects.append(point["_marker"])
            elif answer == BUTTON_REMOVE_POINT and points:
                removed = points.pop()
                _delete_point_marker(removed)
                if _visual_objects is not None and removed.get("_marker"):
                    try:
                        _visual_objects.remove(removed["_marker"])
                    except ValueError:
                        pass
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
                    _delete_point_marker(point)
                if _visual_objects is not None:
                    for point in points:
                        marker = point.get("_marker")
                        if marker and marker in _visual_objects:
                            try:
                                _visual_objects.remove(marker)
                            except ValueError:
                                pass
                return None
    except Exception:
        # On exception, clean up markers to avoid orphaned points.
        for point in points:
            _delete_point_marker(point)
        if _visual_objects is not None:
            for point in points:
                marker = point.get("_marker")
                if marker and marker in _visual_objects:
                    try:
                        _visual_objects.remove(marker)
                    except ValueError:
                        pass
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
    runtime_log = _runtime_log_path(model_dir)
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
        "image_buffer_path": image_export["path"],
        "image_buffer_shape": image_export["shape"],
        "image_buffer_dtype": image_export["dtype"],
        "image_buffer_coordinates": "mimics",
        "buffer_mapping": {
            "platform_to_mimics_axes": [0, 1, 2],
            "platform_to_mimics_flips": [False, False, False],
        },
        "initial_seg_path": base_export["path"] if base_export["pixel_count"] > 0 else None,
        "initial_seg_shape": base_export["shape"],
        "model_dir": model_dir,
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
        "server_idle_timeout_seconds": int(
            os.environ.get(
                "NNINTERACTIVE_SERVER_IDLE_TIMEOUT",
                config.get("server_idle_timeout_seconds", 1800),
            )
        ),
    }
    configured_timeout = config.get("bridge_timeout_seconds", config.get("timeout_seconds"))
    timeout = int(
        os.environ.get(
            "NNINTERACTIVE_TIMEOUT",
            configured_timeout
            if configured_timeout is not None
            else startup_timeout + set_image_timeout + prediction_timeout + 120,
        )
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


def _bridge_call(config, image_export, base_export, interactions, output_path):
    parameters = _bridge_parameters(config, image_export, base_export)
    python_exe = parameters["python_exe"]
    bridge_script = parameters["bridge_script"]
    runtime_log = parameters["runtime_log"]
    requested_device = parameters["requested_device"]
    probe = parameters["probe"]
    folds = parameters["folds"]
    timeout = parameters["timeout"]
    request = dict(parameters["request"])
    request["interactions"] = interactions
    request["output_path"] = output_path
    _append_runtime_log(
        runtime_log,
        "prediction_started",
        {
            "python": python_exe,
            "python_version": probe.get("version"),
            "cuda_available": probe.get("cuda_available"),
            "requested_device": requested_device,
            "folds": folds,
            "bridge_timeout_seconds": timeout,
        },
    )
    _mimics_log(
        logging.INFO,
        "nnInteractive inference started on {0}. Interactions: {1}, timeout: {2}s.".format(
            requested_device, len(interactions), timeout
        ),
    )
    try:
        mimics.view.show_log_panel()
    except Exception:
        pass
    process = subprocess.Popen(
        [python_exe, bridge_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_hidden_process_kwargs()
    )
    try:
        stdout, stderr = process.communicate(
            input=json.dumps(request).encode("utf-8"),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        _append_runtime_log(
            runtime_log,
            "prediction_timed_out",
            {
                "timeout_seconds": timeout,
                "stderr": stderr.decode("utf-8", "replace") if stderr else "",
            },
        )
        raise RuntimeError(
            "nnInteractive did not finish within {0} seconds.\n"
            "The process was stopped. Check:\n{1}".format(timeout, runtime_log)
        )
    if process.returncode != 0:
        stderr_text = stderr.decode("utf-8", "replace") if stderr else ""
        stdout_text = stdout.decode("utf-8", "replace") if stdout else ""
        _append_runtime_log(
            runtime_log,
            "bridge_process_failed",
            {
                "returncode": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        )
        try:
            result = json.loads(stdout_text)
        except ValueError:
            result = {}
        raise RuntimeError(
            "nnInteractive failed during {0}.\n\n{1}\n\nLogs:\n{2}\n{3}".format(
                result.get("stage", "bridge startup or inference"),
                result.get("error") or stderr_text or "No diagnostic text was returned.",
                result.get("bridge_log", runtime_log),
                result.get("server_log", ""),
            )
        )
    try:
        result = json.loads(stdout.decode("utf-8"))
    except ValueError:
        _append_runtime_log(
            runtime_log,
            "invalid_bridge_response",
            {"stdout": stdout.decode("utf-8", "replace")},
        )
        raise RuntimeError(
            "nnInteractive bridge returned invalid JSON.\nCheck:\n{0}".format(runtime_log)
        )
    if result.get("status") == "error":
        raise RuntimeError(
            "nnInteractive failed during {0}.\n\n{1}\n\nLogs:\n{2}\n{3}".format(
                result.get("stage", "inference"),
                result.get("error", "Unknown nnInteractive error"),
                result.get("bridge_log", runtime_log),
                result.get("server_log", ""),
            )
        )
    _append_runtime_log(
        runtime_log,
        "prediction_completed",
        {
            "elapsed_seconds": result.get("elapsed_seconds"),
            "device": result.get("device"),
            "device_warning": result.get("device_warning"),
            "server_url": result.get("server_url"),
            "first_call": result.get("first_call"),
        },
    )
    if result.get("device_warning"):
        _mimics_log(logging.WARNING, str(result["device_warning"]))
    _mimics_log(
        logging.INFO,
        "nnInteractive inference completed in {0}s on {1}.".format(
            result.get("elapsed_seconds", "?"),
            result.get("device", requested_device),
        ),
    )
    return result


class _BridgeWorker(object):
    """Persistent external bridge process for one Mimics interaction session."""

    def __init__(self, config, image_export, base_export):
        self.parameters = _bridge_parameters(config, image_export, base_export)
        self.runtime_log = self.parameters["runtime_log"]
        self.stderr_path = os.path.join(
            os.path.dirname(self.runtime_log),
            "nninteractive_worker.stderr.log",
        )
        self.stderr_handle = open(self.stderr_path, "ab")
        self.process = subprocess.Popen(
            [
                self.parameters["python_exe"],
                self.parameters["bridge_script"],
                "--worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            **_hidden_process_kwargs()
        )
        self.ready = False
        initialize = dict(self.parameters["request"])
        initialize["action"] = "initialize"
        self._write(initialize)
        _append_runtime_log(
            self.runtime_log,
            "worker_started",
            {
                "pid": self.process.pid,
                "python": self.parameters["python_exe"],
                "stderr_log": self.stderr_path,
            },
        )

    def _write(self, request):
        if self.process.poll() is not None:
            raise RuntimeError(
                "nnInteractive worker exited before accepting a request.\n"
                "Check:\n{0}\n{1}".format(self.runtime_log, self.stderr_path)
            )
        self.process.stdin.write(
            json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        self.process.stdin.flush()

    def _readline(self, timeout):
        result_queue = queue.Queue()

        def read_response():
            try:
                result_queue.put((self.process.stdout.readline(), None))
            except Exception as error:
                result_queue.put((b"", error))

        thread = threading.Thread(target=read_response)
        thread.daemon = True
        thread.start()
        try:
            line, error = result_queue.get(timeout=timeout)
        except queue.Empty:
            self.process.kill()
            raise RuntimeError(
                "nnInteractive worker did not respond within {0} seconds.\n"
                "Check:\n{1}\n{2}".format(timeout, self.runtime_log, self.stderr_path)
            )
        if error is not None:
            raise RuntimeError(
                "Could not read the nnInteractive worker response: {0}".format(error)
            )
        if not line:
            raise RuntimeError(
                "nnInteractive worker stopped without returning a result.\n"
                "Check:\n{0}\n{1}".format(self.runtime_log, self.stderr_path)
            )
        try:
            return json.loads(line.decode("utf-8"))
        except ValueError:
            raise RuntimeError(
                "nnInteractive worker returned invalid JSON.\n"
                "Response: {0}\nCheck:\n{1}".format(
                    line.decode("utf-8", "replace"),
                    self.stderr_path,
                )
            )

    def _raise_result_error(self, result):
        raise RuntimeError(
            "nnInteractive failed during {0}.\n\n{1}\n\nLogs:\n{2}\n{3}\n{4}".format(
                result.get("stage", "worker request"),
                result.get("error", "Unknown nnInteractive error"),
                result.get("bridge_log", self.runtime_log),
                result.get("server_log", ""),
                self.stderr_path,
            )
        )

    def ensure_ready(self):
        if self.ready:
            return
        timeout = (
            self.parameters["startup_timeout"]
            + self.parameters["set_image_timeout"]
            + 300
        )
        result = self._readline(timeout)
        if result.get("status") == "error":
            self._raise_result_error(result)
        if result.get("status") != "ready":
            raise RuntimeError(
                "nnInteractive worker did not initialize correctly: {0}".format(result)
            )
        self.ready = True
        _append_runtime_log(
            self.runtime_log,
            "worker_ready",
            {
                "device": result.get("device"),
                "device_warning": result.get("device_warning"),
                "server_url": result.get("server_url"),
                "first_call": result.get("first_call"),
            },
        )
        if result.get("device_warning"):
            _mimics_log(logging.WARNING, str(result["device_warning"]))
        _mimics_log(
            logging.INFO,
            "nnInteractive session ready on {0} (server: {1}).{2}".format(
                result.get("device", "?"),
                result.get("server_url", "?"),
                " First call — model was loaded." if result.get("first_call") else "",
            ),
        )

    def predict(self, interactions, output_path):
        self.ensure_ready()
        _mimics_log(
            logging.INFO,
            "nnInteractive prediction running ({0} interaction(s))...".format(
                len(interactions)
            ),
        )
        self._write(
            {
                "action": "predict",
                "interactions": interactions,
                "output_path": output_path,
            }
        )
        result = self._readline(self.parameters["prediction_timeout"] + 120)
        if result.get("status") == "error":
            self._raise_result_error(result)
        _append_runtime_log(
            self.runtime_log,
            "prediction_completed",
            {
                "elapsed_seconds": result.get("elapsed_seconds"),
                "device": result.get("device"),
                "server_url": result.get("server_url"),
            },
        )
        _mimics_log(
            logging.INFO,
            "nnInteractive prediction completed in {0}s on {1}.".format(
                result.get("elapsed_seconds", "?"),
                result.get("device", "?"),
            ),
        )
        return result

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                if self.ready:
                    try:
                        self._write({"action": "close"})
                        self._readline(15)
                    except Exception:
                        process.kill()
                else:
                    process.kill()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
        finally:
            try:
                self.stderr_handle.close()
            except Exception:
                pass
            self.process = None


def _run_prediction(
    config,
    image_export,
    base_export,
    target,
    interactions,
    temp_dir,
    worker=None,
):
    output_path = os.path.join(temp_dir, "prediction.u8")
    if worker is not None:
        result = worker.predict(interactions, output_path)
    else:
        result = _bridge_call(config, image_export, base_export, interactions, output_path)
    status = result.get("status")
    if status == "skipped":
        # No actionable interaction (e.g. all-empty prompts on undo replay): roll
        # the mask back to the base instead of treating the non-prediction as error.
        _restore_base(target, base_export["path"], base_export["shape"])
        return result
    if status != "refined":
        raise RuntimeError(
            "nnInteractive did not produce a prediction: {0}".format(result.get("reason", status))
        )
    _set_mask_from_u8(target, output_path, base_export["shape"])
    _mimics_log(
        logging.INFO,
        "nnInteractive result applied to Mask {0}. Foreground voxels: {1}, elapsed: {2}s.".format(
            getattr(target, "name", ""),
            result.get("foreground_voxels", "?"),
            result.get("elapsed_seconds", "?"),
        ),
    )
    return result


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


def _cleanup_async_jobs(root, retention_days):
    if not os.path.isdir(root):
        return
    cutoff = time.time() - max(1, int(retention_days)) * 86400
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
        if terminal and updated < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)


def _start_async_worker(python_exe, bridge_script, worker_dir):
    worker_log_path = os.path.join(worker_dir, "async_worker.log")
    worker_log = open(worker_log_path, "ab")
    try:
        process = subprocess.Popen(
            [python_exe, bridge_script, "--async-worker", worker_dir],
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            **_hidden_process_kwargs()
        )
    finally:
        worker_log.close()
    return process, worker_log_path


def _shared_image_worker_alive(worker):
    if not worker:
        return False
    worker_dir = worker.get("worker_dir")
    pid = worker.get("pid")
    if not worker_dir or not os.path.isdir(worker_dir) or not _process_exists(pid):
        return False
    status = _async_worker_status(worker_dir).get("status")
    return status not in ("closing", "closed", "failed", "expired")


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


def _close_all_async_image_workers():
    for worker in list(_ASYNC_IMAGE_WORKERS.values()):
        _request_async_worker_close(worker, "mimics_python_exit")
    _ASYNC_IMAGE_WORKERS.clear()


atexit.register(_close_all_async_image_workers)


def _get_or_start_image_worker(config, image, jobs_root):
    python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)
    image_key = _object_id(image)
    cached = _ASYNC_IMAGE_WORKERS.get(image_key)
    if _shared_image_worker_alive(cached):
        return cached

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

    image_export = _export_image(image, os.path.join(inputs_dir, "image.raw"))
    empty_base_export = {
        "path": "",
        "shape": image_export["shape"],
        "pixel_count": 0,
        "byte_count": 0,
        "sha256": "",
    }
    parameters = _bridge_parameters(config, image_export, empty_base_export)
    initialize = dict(parameters["request"])
    initialize["async_worker_idle_timeout_seconds"] = int(
        config.get("async_worker_idle_timeout_seconds", 600)
    )
    initialize["async_poll_seconds"] = float(config.get("async_poll_seconds", 0.25))
    initialize["initial_seg_path"] = None
    initialize["parent_pid"] = os.getpid()
    _write_json_atomic(os.path.join(worker_dir, "initialize.json"), initialize)
    process, worker_log_path = _start_async_worker(python_exe, bridge_script, worker_dir)
    worker = {
        "worker_dir": worker_dir,
        "pid": process.pid,
        "worker_log": worker_log_path,
        "image_guid": image_key,
        "image_name": str(getattr(image, "name", "")),
        "shape": image_export["shape"],
        "python": python_exe,
        "python_version": probe.get("version"),
        "folds": folds,
        "runtime_log": parameters["runtime_log"],
        "created_at_epoch": time.time(),
    }
    _write_json_atomic(os.path.join(worker_dir, "image_worker.json"), worker)
    _ASYNC_IMAGE_WORKERS[image_key] = worker
    _append_runtime_log(
        parameters["runtime_log"],
        "async_image_worker_started",
        {
            "worker_dir": worker_dir,
            "pid": process.pid,
            "image_guid": image_key,
            "image_name": worker["image_name"],
        },
    )
    return worker


def _start_async_job(config, image, target):
    python_exe, bridge_script, model_dir, probe, folds = _runtime_paths(config)
    jobs_root = os.path.join(os.path.dirname(model_dir), "async_jobs")
    if not os.path.isdir(jobs_root):
        os.makedirs(jobs_root)
    _cleanup_async_jobs(jobs_root, config.get("async_job_retention_days", 7))

    shared_worker = None
    if bool(config.get("async_reuse_image_worker", True)):
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

    if shared_worker is None:
        image_export = _export_image(image, os.path.join(inputs_dir, "image.raw"))
    else:
        image_export = {
            "path": "",
            "shape": shared_worker["shape"],
            "dtype": "",
            "sha256": "",
        }
    base_export = _export_mask(target, os.path.join(inputs_dir, "target_at_start.u8"))
    if image_export["shape"] != base_export["shape"]:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError(
            "Image and target Mask buffer shapes differ: {0} vs {1}".format(
                image_export["shape"],
                base_export["shape"],
            )
        )

    if shared_worker is None:
        parameters = _bridge_parameters(config, image_export, base_export)
        initialize = dict(parameters["request"])
        initialize["async_worker_idle_timeout_seconds"] = int(
            config.get("async_worker_idle_timeout_seconds", 600)
        )
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
        "target_guid": _object_id(target),
        "target_name": str(getattr(target, "name", "")),
        "shape": base_export["shape"],
        "base_path": base_export["path"],
        "base_sha256": base_export["sha256"],
        "expected_target_sha256": base_export["sha256"],
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
    _save_async_job(state)

    state["pid"] = worker_pid
    state["worker_log"] = worker_log_path
    state["updated_at_epoch"] = time.time()
    _save_async_job(state)
    _metadata_set(target, ASYNC_JOB_METADATA, job_dir)
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


def _enqueue_async_prediction(state, target):
    worker_dir = _async_state_worker_dir(state)
    sequence = _next_async_sequence(worker_dir)
    expected_hash = _mask_sha256(target)
    command = {
        "schema_version": "nninteractive_async_command.v1",
        "command_id": uuid.uuid4().hex,
        "sequence": sequence,
        "created_at_epoch": time.time(),
        "expected_target_sha256": expected_hash,
        "interactions": state.get("interactions", []),
        "initial_seg_path": state["base_path"] if state.get("base_path") else None,
        "initial_seg_shape": state.get("shape"),
        "target_guid": state.get("target_guid"),
        "target_name": state.get("target_name"),
        "job_dir": state.get("_job_dir"),
    }
    state["pending_sequence"] = sequence
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


def _describe_async_worker_failure(worker, worker_status):
    stage = worker.get("stage", worker_status or "unknown")
    error = worker.get("error")
    if not error and stage == "idle_timeout":
        timeout_seconds = worker.get("idle_timeout_seconds")
        if timeout_seconds is None:
            error = (
                "No result file was produced before the async worker idle timeout. "
                "Increase async_worker_idle_timeout_seconds in nninteractive_config.json "
                "if long-running sessions are expected."
            )
        else:
            error = (
                "No result file was produced before the async worker idle timeout "
                "({0}s). Increase async_worker_idle_timeout_seconds in "
                "nninteractive_config.json if long-running sessions are expected."
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
        if outcome != "waiting":
            monitor["done"] = True
            _stop_async_monitor(job_dir)
            if outcome == "applied":
                mimics.dialogs.message_box(
                    "nnInteractive result has been applied automatically.",
                    title=TITLE,
                    ui_blocking=False,
                )
    except Exception as error:
        monitor["done"] = True
        _stop_async_monitor(job_dir)
        mimics.dialogs.message_box(
            "nnInteractive background prediction failed.\n\n{0}".format(error),
            title="nnInteractive Failed",
            ui_blocking=True,
        )


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
    user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["win32_timer"] = (user32, timer_id)
    _ASYNC_MONITORS[job_dir] = monitor
    return True


def _start_async_result_monitor(image, target, state, config):
    """Start a non-blocking Mimics-side timer that applies async results."""
    poll_seconds = float(config.get("async_result_poll_seconds", config.get("async_poll_seconds", 0.5)))
    timeout_seconds = float(
        config.get(
            "async_result_wait_timeout_seconds",
            config.get("prediction_timeout_seconds", 1800) + 120,
        )
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
            answer = mimics.dialogs.question_box(
                message=(
                    "The nnInteractive background worker stopped before producing a result.\n\n"
                    "Stage: {0}\nError: {1}\n\n"
                    "Start a new AI session from the current Mask?"
                ).format(
                    stage,
                    error,
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
        answer = mimics.dialogs.question_box(
            message=(
                "The background prediction failed.\n\n"
                "Stage: {0}\n"
                "Error: {1}\n\n"
                "Retry keeps the same prompts. Discard starts a new AI session "
                "from the current Mask."
            ).format(
                result.get("stage", "prediction"),
                result.get("error", "Unknown error"),
            ),
            buttons=BUTTON_RETRY + ";" + BUTTON_DISCARD_SESSION + ";" + BUTTON_CANCEL,
            title="nnInteractive Failed",
            ui_blocking=True,
        )
        if answer == BUTTON_RETRY:
            state["pending_sequence"] = None
            _enqueue_async_prediction(state, target)
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
    if _object_id(image) != state.get("image_guid") or _object_id(target) != state.get(
        "target_guid"
    ):
        raise RuntimeError(
            "The active image or target Mask no longer matches the background job. "
            "The result was not applied."
        )
    current_hash = _mask_sha256(target)
    expected_hash = result.get("expected_target_sha256")
    if current_hash != expected_hash:
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
        return "waiting"

    output_path = result.get("output_path")
    if not output_path or not os.path.isfile(output_path):
        raise RuntimeError(
            "nnInteractive completed but the prediction buffer is missing:\n{0}".format(
                output_path
            )
        )
    _set_mask_from_u8(target, output_path, state["shape"])
    _make_mask_visible(target)
    # Clean up visual objects that were kept visible during inference.
    job_dir = state.get("_job_dir")
    if job_dir and job_dir in _ASYNC_VISUAL_OBJECTS:
        for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
            _delete_mimics_object(obj)
    state["pending_sequence"] = None
    state["applied_sequence"] = int(sequence)
    state["expected_target_sha256"] = _sha256_file(output_path)
    state["status"] = "ready"
    state["updated_at_epoch"] = time.time()
    _save_async_job(state)
    _mimics_log(
        logging.INFO,
        "nnInteractive result applied to Mask {0}. Foreground voxels: {1}, elapsed: {2}s, device: {3}.".format(
            getattr(target, "name", ""),
            result.get("foreground_voxels", "?"),
            result.get("elapsed_seconds", "?"),
            result.get("device", "?"),
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


def _async_prompt_menu(target, state):
    count = len(state.get("interactions", [])) if state else 0
    buttons = [
        BUTTON_POINT,
        BUTTON_SCRIBBLE,
        BUTTON_BOX,
        BUTTON_LASSO,
    ]
    if state and state.get("interactions"):
        buttons.extend([BUTTON_UNDO, BUTTON_RESET])
    buttons.append(BUTTON_FINISH)
    return mimics.dialogs.question_box(
        message=(
            "Target Mask: {0}\n"
            "Prompts in this AI session: {1}\n\n"
            "Submitting a prompt starts background inference and immediately "
            "returns control to Mimics. The result is applied automatically when ready."
        ).format(getattr(target, "name", ""), count),
        buttons=";".join(buttons),
        title=TITLE,
        ui_blocking=True,
    )


def _run_async(image, target, config):
    state = _load_async_job(target)
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

    if state is not None:
        current_hash = _mask_sha256(target)
        if current_hash != state.get("expected_target_sha256"):
            _close_async_job(target, state, "manual_mask_change")
            state = None

    temp_dir = tempfile.mkdtemp(prefix="mimics_nninteractive_prompt_")
    pending_visual_objects = []
    visual_objects_registered = False
    try:
        action = _async_prompt_menu(target, state)
        if action == BUTTON_FINISH or not action:
            if state is not None:
                _close_async_job(target, state, "user_finished")
            _mimics_log(logging.INFO, "nnInteractive session finished.")
            return 0
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
                _enqueue_async_prediction(state, target)
                _start_async_result_monitor(image, target, state, config)
            else:
                _restore_base(target, state["base_path"], state["shape"])
                state["expected_target_sha256"] = state["base_sha256"]
                state["pending_sequence"] = None
                state["status"] = "ready"
                _save_async_job(state)
            return 0
        if action == BUTTON_RESET and state is not None:
            state["interactions"] = []
            state["pending_sequence"] = None
            state["status"] = "ready"
            _mimics_log(logging.INFO, "nnInteractive session reset to initial mask.")
            _restore_base(target, state["base_path"], state["shape"])
            state["expected_target_sha256"] = state["base_sha256"]
            # Clean up visual objects from previous prompts.
            job_dir = state.get("_job_dir")
            if job_dir and job_dir in _ASYNC_VISUAL_OBJECTS:
                for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir):
                    _delete_mimics_object(obj)
            _save_async_job(state)
            return 0

        if state is None and bool(config.get("async_start_worker_before_prompt", True)):
            _mimics_log(
                logging.INFO,
                "nnInteractive is preparing the AI session before prompt capture.",
            )
            state = _start_async_job(config, image, target)

        include = None
        visual_objects = []
        prompt = _capture_prompt(action, image, include, temp_dir, visual_objects)
        pending_visual_objects = visual_objects
        if prompt is None:
            mimics.dialogs.message_box(
                "No prompt was submitted for {0}.\n\n"
                "The AI prediction was not started.".format(action),
                title="nnInteractive Prompt Empty",
                ui_blocking=False,
            )
            return 0
        if state is None:
            state = _start_async_job(config, image, target)
        prompt = _persist_interaction(state["_job_dir"], prompt)
        # Store visual objects for deferred deletion after async result is applied.
        if visual_objects:
            _ASYNC_VISUAL_OBJECTS.setdefault(state["_job_dir"], []).extend(visual_objects)
            visual_objects_registered = True
        state.setdefault("interactions", []).append(prompt)
        sequence = _enqueue_async_prediction(state, target)
        _mimics_log(
            logging.INFO,
            "nnInteractive background inference started. Prompt: {0}, sequence: {1}. "
            "Result will be applied automatically when ready.".format(action, sequence),
        )
        _start_async_result_monitor(image, target, state, config)
        return 0
    finally:
        if pending_visual_objects and not visual_objects_registered:
            for obj in pending_visual_objects:
                _delete_mimics_object(obj)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _prompt_menu(target, interaction_count):
    return mimics.dialogs.question_box(
        message=(
            "Target Mask: {0}\n"
            "Prompts in this session: {1}\n\n"
            "Each new prompt immediately updates the selected Mask."
        ).format(getattr(target, "name", ""), interaction_count),
        buttons=";".join(
            [
                BUTTON_POINT,
                BUTTON_SCRIBBLE,
                BUTTON_BOX,
                BUTTON_LASSO,
                BUTTON_UNDO,
                BUTTON_RESET,
                BUTTON_FINISH,
            ]
        ),
        title=TITLE,
        ui_blocking=True,
    )


def _run_sync(image, target, config):
    _mimics_log(logging.INFO, "Preparing nnInteractive session...")
    temp_dir = tempfile.mkdtemp(prefix="mimics_nninteractive_")
    worker = None
    visual_objects = []
    try:
        image_export = _export_image(image, os.path.join(temp_dir, "image.raw"))
        base_export = _export_mask(target, os.path.join(temp_dir, "target_at_start.u8"))
        if image_export["shape"] != base_export["shape"]:
            raise RuntimeError(
                "Image and target Mask buffer shapes differ: {0} vs {1}".format(
                    image_export["shape"], base_export["shape"]
                )
            )

        interactions = []
        visual_objects[:] = []  # Mimics objects kept visible during inference
        _mimics_log(
            logging.INFO,
            "nnInteractive session started. Target Mask: {0}, image shape: {1}.".format(
                getattr(target, "name", ""), image_export["shape"]
            ),
        )
        if bool(config.get("reuse_session", True)):
            _mimics_log(
                logging.INFO,
                "nnInteractive is starting the AI engine in the background. "
                "You can collect prompts while it loads.",
            )
            worker = _BridgeWorker(config, image_export, base_export)
            try:
                mimics.view.show_log_panel()
            except Exception:
                pass
        while True:
            action = _prompt_menu(target, len(interactions))
            if action == BUTTON_FINISH or not action:
                _mimics_log(
                    logging.INFO,
                    "nnInteractive session finished. {0} prompt(s) applied.".format(
                        len(interactions)
                    ),
                )
                return 0
            if action == BUTTON_RESET:
                interactions = []
                _mimics_log(logging.INFO, "nnInteractive session reset to initial mask.")
                _restore_base(target, base_export["path"], base_export["shape"])
                continue
            if action == BUTTON_UNDO:
                if not interactions:
                    continue
                interactions.pop()
                _mimics_log(
                    logging.INFO,
                    "nnInteractive undo last prompt. Remaining: {0}. Re-running prediction...".format(
                        len(interactions)
                    ),
                )
                if interactions:
                    try:
                        _run_prediction(
                            config,
                            image_export,
                            base_export,
                            target,
                            interactions,
                            temp_dir,
                            worker,
                        )
                    except Exception:
                        _restore_base(target, base_export["path"], base_export["shape"])
                        raise
                else:
                    _restore_base(target, base_export["path"], base_export["shape"])
                continue

            include = None
            prompt = _capture_prompt(action, image, include, temp_dir, visual_objects)
            if prompt is None:
                continue
            interactions.append(prompt)
            _mimics_log(
                logging.INFO,
                "nnInteractive prompt #{0}: {1}. Running prediction...".format(
                    len(interactions), action
                ),
            )
            prediction_error = None
            try:
                _run_prediction(
                    config,
                    image_export,
                    base_export,
                    target,
                    interactions,
                    temp_dir,
                    worker,
                )
            except Exception as exc:
                prediction_error = exc
                interactions.pop()
                # Restore the mask to its state before this failed prediction.
                _restore_base(target, base_export["path"], base_export["shape"])
            finally:
                # Delete visual objects after prediction completes (or fails).
                for obj in visual_objects:
                    _delete_mimics_object(obj)
                visual_objects[:] = []
            if prediction_error is not None:
                # Show the error but keep the session alive so the user can
                # adjust prompts and retry, instead of aborting entirely.
                _mimics_log(
                    logging.ERROR,
                    "nnInteractive prediction failed: {0}".format(prediction_error),
                )
                mimics.dialogs.message_box(
                    "Prediction failed for this prompt:\n\n{0}\n\n"
                    "The mask has been restored. You can try again with "
                    "different prompts.".format(str(prediction_error)),
                    title=TITLE,
                    ui_blocking=True,
                )
                continue
    finally:
        # Clean up any remaining visual objects on session exit.
        for obj in visual_objects:
            _delete_mimics_object(obj)
        if worker is not None:
            worker.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run():
    image = mimics.data.images.get_active()
    if image is None:
        raise RuntimeError("Open a project and activate an image set before running nnInteractive.")
    target = _select_target_mask(image)
    if target is None:
        _mimics_log(logging.INFO, "nnInteractive cancelled: no target Mask selected.")
        return 0
    config = _config()
    if str(config.get("execution_mode", "async")).lower() == "async":
        result = _run_async(image, target, config)
        # Async mode returns 0 after submitting a prompt — the background
        # worker and QTimer monitor are still running.  Don't log "ended".
        return result
    return _run_sync(image, target, config)


def _cleanup_stale_processes():
    """Kill leftover bridge python processes from a previous crashed session.

    nnInteractive server and watchdog processes are **excluded** so that a
    running inference server is not disrupted.

    Uses a single hidden batch PowerShell call instead of per-process calls
    to avoid popping up visible console windows that freeze Mimics.
    """
    if os.name != "nt":
        return
    killed = []
    _SERVER_PROTECT_MARKERS = (
        "nninteractive.inference.server.main",
        "--watchdog",
    )
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_TERMINATE = 0x0001

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
    if killed:
        _mimics_log(
            logging.INFO,
            "Startup cleanup: terminated {0} stale bridge process(es).".format(len(killed)),
        )


def main():
    try:
        # Run cleanup in a background thread so Mimics UI does not freeze
        # while PowerShell queries process list.
        cleanup_thread = threading.Thread(target=_cleanup_stale_processes)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        result = run()
        # Only log "ended" for sync mode (result 0 from _run_sync means the
        # user finished).  Async mode returns 0 after submitting a prompt
        # while the background worker keeps running.
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
