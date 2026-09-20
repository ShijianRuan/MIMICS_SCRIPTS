# -*- coding: utf-8 -*-
"""Non-blocking Mimics entry for ScribblePrompt."""

from __future__ import print_function

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid

import mimics

import nninteractive_mimics as nnm
import runtime_common


ACTION_SCRIBBLEPROMPT = "scribbleprompt"

DISPLAY_NAMES = {
    ACTION_SCRIBBLEPROMPT: "ScribblePrompt",
}

SESSION_PATH_METADATA = "scribbleprompt.session_logits_path"
SESSION_SHAPE_METADATA = "scribbleprompt.session_logits_shape"
SESSION_AXIS_METADATA = "scribbleprompt.session_plane_axis"
SESSION_INDEX_METADATA = "scribbleprompt.session_plane_index"
SESSION_MASK_SHA_METADATA = "scribbleprompt.session_mask_sha256"

_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("interactive_algorithms_config.json", "nninteractive_config.json", ".git"),
    )


def _config():
    path = os.environ.get("MIMICS_INTERACTIVE_ALGORITHMS_CONFIG", "").strip()
    if not path:
        path = os.path.join(_project_root(), "interactive_algorithms_config.json")
    value = runtime_common.read_json(path, {}) if os.path.isfile(path) else {}
    value["_config_path"] = path
    return value


def _runtime_root():
    return runtime_common.import_runtime_base(_project_root())


def _jobs_root(action):
    return os.path.join(_runtime_root(), "interactive_algorithms", action)


def _python_exe():
    root = _project_root()
    candidates = [
        os.environ.get("MIMICS_BRIDGE_PYTHON", ""),
        os.path.join(root, "python_env", "python.exe"),
        os.path.join(root, "python_env", "Scripts", "python.exe"),
        os.path.join(root, "python_env", "bin", "python3"),
        os.path.join(root, "python_env", "bin", "python"),
        os.path.join(root, "nninteractive_env", "python.exe"),
        os.path.join(root, "nninteractive_env", "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "bin", "python3"),
        os.path.join(root, "nninteractive_env", "bin", "python"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run Setup / Repair Environment first."
    )


def _worker_script():
    path = os.path.join(_project_root(), "tools", "interactive_algorithms_worker.py")
    if not os.path.isfile(path):
        raise RuntimeError("interactive_algorithms_worker.py was not found: {0}".format(path))
    return path


def _checkpoint_path(config):
    section = config.get("scribbleprompt") or {}
    value = os.environ.get("SCRIBBLEPROMPT_CHECKPOINT", "").strip()
    if not value:
        value = str(section.get("checkpoint") or "").strip()
    if value and not os.path.isabs(value):
        value = os.path.join(_project_root(), value)
    return os.path.abspath(value) if value else ""


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _active_image():
    image = None
    try:
        image = mimics.data.images.get_active()
    except Exception:
        pass
    if image is None:
        try:
            if len(mimics.data.images) == 1:
                image = mimics.data.images[0]
                try:
                    mimics.data.images.set_active(image)
                except Exception:
                    pass
        except Exception:
            pass
    if image is None:
        raise RuntimeError("Open and activate one image before running this tool.")
    return image


def _selected_mask(image):
    selected = []
    for mask in nnm._masks_for_image(image):
        if bool(getattr(mask, "selected", False)):
            selected.append(mask)
    if len(selected) != 1:
        raise RuntimeError("Select exactly one Mask attached to the active image.")
    return selected[0]


def _sha256_bytes(value):
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _buffer_byte_view(view):
    """Return a one-dimensional byte view without copying when possible."""
    try:
        raw = memoryview(view)
        if raw.ndim != 1 or raw.format not in ("B", "b", "c"):
            raw = raw.cast("B")
        elif raw.format != "B":
            raw = raw.cast("B")
        return raw
    except Exception:
        return memoryview(view.tobytes())


def _stream_buffer(raw, handle=None, compute_sha=True, progress_callback=None):
    digest = hashlib.sha256() if compute_sha else None
    chunk_bytes = 16 * 1024 * 1024
    byte_count = len(raw)
    for offset in range(0, byte_count, chunk_bytes):
        chunk = raw[offset:min(byte_count, offset + chunk_bytes)]
        if handle is not None:
            handle.write(chunk)
        if digest is not None:
            digest.update(chunk)
        if progress_callback is not None and offset + len(chunk) < byte_count:
            try:
                progress_callback()
            except Exception:
                pass
    return digest.hexdigest() if digest is not None else ""


def _write_raw_buffer(
    obj,
    path,
    expected_shape=None,
    dtype=None,
    compute_sha=True,
    progress_callback=None,
):
    view = obj.get_voxel_buffer()
    shape = [int(value) for value in view.shape]
    if expected_shape is not None and list(expected_shape) != shape:
        raise RuntimeError(
            "Mimics image and Mask buffer shapes differ: {0} vs {1}.".format(
                expected_shape, shape
            )
        )
    raw = _buffer_byte_view(view)
    with open(path, "wb") as handle:
        sha256 = _stream_buffer(
            raw,
            handle=handle,
            compute_sha=compute_sha,
            progress_callback=progress_callback,
        )
    return {
        "path": path,
        "shape": shape,
        "dtype": dtype or nnm._buffer_dtype(view),
        "sha256": sha256,
        "byte_count": len(raw),
    }


def _spacing_mm(image, shape):
    matrix = nnm._derive_image_voxel_to_ras_matrix(image, shape)
    if matrix is None:
        raw = nnm._metadata_get(image, nnm.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
        matrix = nnm._parse_matrix_metadata(raw)
    if matrix is None:
        _mimics_log(
            logging.WARNING,
            "The active image spacing could not be derived; 1 mm isotropic spacing will be used.",
        )
        return [1.0, 1.0, 1.0]
    result = []
    for axis in range(3):
        value = math.sqrt(sum(float(matrix[row][axis]) ** 2 for row in range(3)))
        result.append(value if value > 0 and math.isfinite(value) else 1.0)
    return result


def _unique_job_dir(action):
    root = _jobs_root(action)
    if not os.path.isdir(root):
        os.makedirs(root)
    name = "job_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:10])
    path = os.path.join(root, name)
    os.makedirs(path)
    os.makedirs(os.path.join(path, "inputs"))
    os.makedirs(os.path.join(path, "outputs"))
    return path


def _cleanup_old_jobs(action, config):
    root = _jobs_root(action)
    if not os.path.isdir(root):
        return
    days = max(1.0, float(config.get("job_retention_days", 3)))
    maximum = max(1, int(config.get("job_max_terminal", 20)))
    terminal = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        state = runtime_common.read_json(os.path.join(path, "status.json"), {})
        if state.get("status") not in ("completed", "failed", "cancelled"):
            continue
        terminal.append((float(state.get("updated_at_epoch", 0) or 0), path))
    terminal.sort(reverse=True)
    cutoff = time.time() - days * 86400.0
    for index, item in enumerate(terminal):
        updated, path = item
        if index >= maximum or (updated and updated < cutoff):
            shutil.rmtree(path, ignore_errors=True)


def _delete_visual_objects(values):
    for obj in values or []:
        deleter = getattr(nnm, "_delete_mimics_object", None)
        if callable(deleter):
            try:
                deleter(obj)
                continue
            except Exception:
                pass
        try:
            mimics.data.masks.delete(obj)
        except Exception:
            pass


def _pure_python_crop(raw, shape):
    minimum = [shape[0], shape[1], shape[2]]
    maximum = [-1, -1, -1]
    yz = int(shape[1]) * int(shape[2])
    zdim = int(shape[2])
    for offset, value in enumerate(bytearray(raw)):
        if not value:
            continue
        x = offset // yz
        remainder = offset - x * yz
        y = remainder // zdim
        z = remainder - y * zdim
        coordinates = (x, y, z)
        for axis in range(3):
            minimum[axis] = min(minimum[axis], coordinates[axis])
            maximum[axis] = max(maximum[axis], coordinates[axis])
    if maximum[0] < 0:
        return None
    crop_shape = [maximum[axis] - minimum[axis] + 1 for axis in range(3)]
    output = bytearray(crop_shape[0] * crop_shape[1] * crop_shape[2])
    out_yz = crop_shape[1] * crop_shape[2]
    for x in range(minimum[0], maximum[0] + 1):
        for y in range(minimum[1], maximum[1] + 1):
            source_start = x * yz + y * zdim + minimum[2]
            source_stop = source_start + crop_shape[2]
            target_start = (x - minimum[0]) * out_yz + (y - minimum[1]) * crop_shape[2]
            output[target_start:target_start + crop_shape[2]] = raw[source_start:source_stop]
    bbox = [[minimum[axis], maximum[axis] + 1] for axis in range(3)]
    return bytes(output), crop_shape, bbox


def _capture_scribble(image, include, job_dir, visual_objects):
    prompt = mimics.segment.create_mask()
    prompt.name = "ScribblePrompt {0} Scribble".format(
        "Foreground" if include else "Background"
    )
    try:
        try:
            prompt.image = image
        except Exception:
            if not nnm._same_object(getattr(prompt, "image", None), image):
                raise
        prompt.visible = True
        prompt.color = (0.1, 1.0, 0.2) if include else (1.0, 0.2, 0.1)
        mimics.segment.activate_edit_mask(prompt, "Ellipse", "Draw")
        if int(getattr(prompt, "number_of_pixels", 0) or 0) <= 0:
            mimics.data.masks.delete(prompt)
            return None
        view = prompt.get_voxel_buffer()
        shape = [int(value) for value in view.shape]
        raw = view.tobytes()
        try:
            import numpy as np

            array = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))
            nonzero = np.argwhere(array)
            if len(nonzero) == 0:
                mimics.data.masks.delete(prompt)
                return None
            minimum = nonzero.min(axis=0)
            maximum = nonzero.max(axis=0) + 1
            bbox = [[int(minimum[axis]), int(maximum[axis])] for axis in range(3)]
            crop = array[
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
            ]
            crop_raw = crop.tobytes(order="C")
            crop_shape = [int(value) for value in crop.shape]
        except ImportError:
            cropped = _pure_python_crop(raw, shape)
            if cropped is None:
                mimics.data.masks.delete(prompt)
                return None
            crop_raw, crop_shape, bbox = cropped
        path = os.path.join(
            job_dir,
            "inputs",
            "{0}_{1}.u8".format("foreground" if include else "background", uuid.uuid4().hex),
        )
        with open(path, "wb") as handle:
            handle.write(crop_raw)
        try:
            prompt.selected = False
        except Exception:
            pass
        visual_objects.append(prompt)
        return {
            "path": path,
            "shape": crop_shape,
            "bbox": bbox,
            "include": bool(include),
        }
    except mimics.UserInterrupted:
        try:
            mimics.data.masks.delete(prompt)
        except Exception:
            pass
        return None
    except Exception:
        try:
            mimics.data.masks.delete(prompt)
        except Exception:
            pass
        raise


def _prompt_plane_from_bbox(bbox, preferred_axis=None):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 3:
        return None
    candidates = []
    for axis in range(3):
        try:
            if int(bbox[axis][1]) - int(bbox[axis][0]) == 1:
                candidates.append(axis)
        except Exception:
            return None
    if preferred_axis in candidates:
        axis = int(preferred_axis)
    elif len(candidates) == 1:
        axis = candidates[0]
    else:
        return None
    return axis, int(bbox[axis][0])


def _click_plane_axes(image):
    matrix = nnm._parse_matrix_metadata(
        nnm._metadata_get(image, nnm.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
    )
    if matrix is None:
        dimensions = getattr(image, "logical_dimensions", None)
        try:
            shape = [int(dimensions[0]), int(dimensions[1]), int(dimensions[2])]
        except Exception:
            shape = None
        if shape:
            matrix = nnm._derive_image_voxel_to_ras_matrix(image, shape)
    if matrix is None:
        return None
    result = {}
    for label, world_axis in (
        ("Axial View", 2),
        ("Coronal View", 1),
        ("Sagittal View", 0),
    ):
        scores = []
        for voxel_axis in range(3):
            length = math.sqrt(
                sum(float(matrix[row][voxel_axis]) ** 2 for row in range(3))
            )
            score = (
                abs(float(matrix[world_axis][voxel_axis])) / length
                if length > 0.0
                else 0.0
            )
            scores.append(score)
        axis = max(range(3), key=lambda value: scores[value])
        if scores[axis] < 0.5:
            return None
        result[label] = axis
    if len(set(result.values())) != 3:
        return None
    return result


def _choose_click_plane(image):
    axes = _click_plane_axes(image)
    if not axes:
        mimics.dialogs.message_box(
            title="ScribblePrompt",
            message=(
                "The active image orientation could not be mapped safely to the three "
                "standard 2D views. Add a Scribble or Box first so the slice is explicit."
            ),
            ui_blocking=True,
        )
        return None
    labels = ["Axial View", "Coronal View", "Sagittal View"]
    answer = mimics.dialogs.question_box(
        title="ScribblePrompt Plane",
        message=(
            "Choose the 2D view in which you are placing prompts. All prompts in this "
            "prediction must stay on that same slice."
        ),
        buttons=";".join(labels + ["Cancel"]),
        ui_blocking=True,
    )
    if answer not in axes:
        return None
    return int(axes[answer])


def _choose_prompt_sign():
    answer = mimics.dialogs.question_box(
        title="ScribblePrompt",
        message=(
            "Foreground marks pixels that belong to the structure.\n"
            "Background marks pixels that must be excluded."
        ),
        buttons="Foreground;Background;Cancel",
        ui_blocking=True,
    )
    if answer == "Foreground":
        return True
    if answer == "Background":
        return False
    return None


def _capture_prompt_point(image, include):
    try:
        coordinates = mimics.indicate_coordinate(
            message=(
                "Click inside the structure."
                if include
                else "Click an area that must be excluded."
            ),
            show_message_box=True,
            confirm=False,
            title="ScribblePrompt",
        )
    except mimics.UserInterrupted:
        return None
    indexes = [int(value) for value in image.get_voxel_indexes(coordinates)]
    marker = None
    try:
        marker = mimics.analyze.create_point(
            point=nnm._point_coordinates(coordinates),
            name="ScribblePrompt {0} Click".format(
                "Foreground" if include else "Background"
            ),
            color=(0.1, 1.0, 0.2) if include else (1.0, 0.2, 0.1),
        )
    except Exception:
        pass
    return {
        "point": indexes,
        "include": bool(include),
        "_visual_object": marker,
    }


def _remove_visual_object(value, visual_objects):
    if value is None:
        return
    visual_objects[:] = [item for item in visual_objects if item is not value]
    _delete_visual_objects([value])


def _collect_scribbleprompt_prompts(image, target, job_dir, visual_objects):
    records = []
    plane_axis = None
    plane_index = None

    def accept_plane(candidate, visual):
        nonlocal plane_axis, plane_index
        if candidate is None:
            _remove_visual_object(visual, visual_objects)
            mimics.dialogs.message_box(
                title="ScribblePrompt",
                message=(
                    "This prompt does not identify one 2D image slice. Draw it in one "
                    "axial, coronal, or sagittal view and try again."
                ),
                ui_blocking=True,
            )
            return False
        axis, index = candidate
        if plane_axis is None:
            plane_axis, plane_index = int(axis), int(index)
            return True
        if int(axis) == plane_axis and int(index) == plane_index:
            return True
        _remove_visual_object(visual, visual_objects)
        mimics.dialogs.message_box(
            title="ScribblePrompt",
            message=(
                "All clicks, scribbles, and the box for one prediction must be placed "
                "on the same 2D slice. The last prompt was not added."
            ),
            ui_blocking=True,
        )
        return False

    while True:
        clicks = [item for item in records if item["kind"] == "point"]
        scribbles = [item for item in records if item["kind"] == "scribble"]
        boxes = [item for item in records if item["kind"] == "box"]
        positive = any(
            item["kind"] == "box" or bool(item["value"].get("include"))
            for item in records
        )
        can_run = bool(records) and (
            positive or int(getattr(target, "number_of_pixels", 0) or 0) > 0
        )
        buttons = ["Add Click", "Add Scribble"]
        if not boxes:
            buttons.append("Add Box")
        if records:
            buttons.append("Undo Last")
        if can_run:
            buttons.append("Run ScribblePrompt")
        buttons.append("Cancel")
        answer = mimics.dialogs.question_box(
            title="ScribblePrompt",
            message=(
                "Add clicks, scribbles, or one foreground box on a single 2D slice. "
                "Existing Mask content and the previous prediction are used for refinement.\n\n"
                "Clicks: {0}    Scribbles: {1}    Box: {2}\n"
                "Foreground prompts: {3}    Background prompts: {4}"
            ).format(
                len(clicks),
                len(scribbles),
                "added" if boxes else "none",
                len([item for item in records if item["kind"] == "box" or item["value"].get("include")]),
                len([item for item in records if item["kind"] != "box" and not item["value"].get("include")]),
            ),
            buttons=";".join(buttons),
            ui_blocking=True,
        )
        if answer == "Add Click":
            include = _choose_prompt_sign()
            if include is None:
                continue
            chosen_axis = plane_axis
            if chosen_axis is None:
                chosen_axis = _choose_click_plane(image)
                if chosen_axis is None:
                    continue
            value = _capture_prompt_point(image, include)
            if value is not None:
                visual = value.pop("_visual_object", None)
                if visual is not None:
                    visual_objects.append(visual)
                candidate = (chosen_axis, int(value["point"][chosen_axis]))
                if accept_plane(candidate, visual):
                    records.append({"kind": "point", "value": value, "visual": visual})
        elif answer == "Add Scribble":
            include = _choose_prompt_sign()
            if include is None:
                continue
            before = len(visual_objects)
            value = _capture_scribble(image, include, job_dir, visual_objects)
            if value is not None:
                visual = visual_objects[-1] if len(visual_objects) > before else None
                candidate = _prompt_plane_from_bbox(value.get("bbox"), plane_axis)
                if candidate is None and plane_axis is None:
                    chosen_axis = _choose_click_plane(image)
                    candidate = _prompt_plane_from_bbox(
                        value.get("bbox"), chosen_axis
                    )
                if accept_plane(candidate, visual):
                    records.append({"kind": "scribble", "value": value, "visual": visual})
        elif answer == "Add Box":
            before = len(visual_objects)
            value = nnm._capture_box(image, visual_objects)
            if value is not None:
                visual = visual_objects[-1] if len(visual_objects) > before else None
                candidate = _prompt_plane_from_bbox(value.get("bbox"), plane_axis)
                if accept_plane(candidate, visual):
                    records.append({"kind": "box", "value": value, "visual": visual})
        elif answer == "Undo Last" and records:
            removed = records.pop()
            _remove_visual_object(removed.get("visual"), visual_objects)
            if records:
                first = records[0]
                if first["kind"] == "point":
                    plane_axis = int(plane_axis)
                    plane_index = int(first["value"]["point"][plane_axis])
                else:
                    plane_axis, plane_index = _prompt_plane_from_bbox(
                        first["value"].get("bbox"), plane_axis
                    )
            else:
                plane_axis = None
                plane_index = None
        elif answer == "Run ScribblePrompt" and can_run:
            return {
                "scribbles": [item["value"] for item in records if item["kind"] == "scribble"],
                "points": [item["value"] for item in records if item["kind"] == "point"],
                "boxes": [item["value"] for item in records if item["kind"] == "box"],
                "plane_axis": plane_axis,
                "plane_index": plane_index,
            }
        else:
            return None


def _session_values(mask, base_sha):
    path = str(nnm._metadata_get(mask, SESSION_PATH_METADATA, "") or "")
    expected_sha = str(nnm._metadata_get(mask, SESSION_MASK_SHA_METADATA, "") or "")
    if not path or expected_sha != base_sha or not os.path.isfile(path):
        return {}
    try:
        shape = json.loads(str(nnm._metadata_get(mask, SESSION_SHAPE_METADATA, "") or "[]"))
        return {
            "previous_logits_path": path,
            "previous_logits_shape": [int(value) for value in shape],
            "previous_plane_axis": int(nnm._metadata_get(mask, SESSION_AXIS_METADATA, -1)),
            "previous_plane_index": int(nnm._metadata_get(mask, SESSION_INDEX_METADATA, -1)),
        }
    except Exception:
        return {}


def _write_initial_status(path, action):
    runtime_common.write_json_atomic(
        path,
        {
            "schema_version": "mimics_interactive_algorithm_job.v1",
            "status": "starting",
            "tool": action,
            "phase": "starting",
            "progress_percent": 0,
            "message": "Starting {0} in the external Python environment.".format(DISPLAY_NAMES[action]),
            "updated_at_epoch": time.time(),
        },
    )


def _prepare_request(action, image, target, config, job_dir, prompts=None):
    inputs = os.path.join(job_dir, "inputs")
    outputs = os.path.join(job_dir, "outputs")
    snapshot_started = time.time()
    _mimics_log(logging.INFO, "{0}: reading the active image and selected Mask.".format(DISPLAY_NAMES[action]))
    _update_gui()
    operation_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "{0} input export".format(DISPLAY_NAMES[action])
    )
    if not operation_token:
        raise RuntimeError("Another Mimics operation is currently reading or updating Mask buffers. Try again shortly.")
    try:
        image_export = _write_raw_buffer(
            image,
            os.path.join(inputs, "image.raw"),
            compute_sha=False,
            progress_callback=_update_gui,
        )
        mask_export = _write_raw_buffer(
            target,
            os.path.join(inputs, "selected_mask.u8"),
            expected_shape=image_export["shape"],
            dtype="uint8",
            progress_callback=_update_gui,
        )
    finally:
        runtime_common.release_local_operation("mask_buffer_access", operation_token)
        _update_gui()
    _mimics_log(
        logging.INFO,
        (
            "{0}: Mimics-grid snapshot completed in {1}s "
            "(image {2:.1f} MB, Mask {3:.1f} MB)."
        ).format(
            DISPLAY_NAMES[action],
            round(time.time() - snapshot_started, 2),
            float(image_export["byte_count"]) / (1024.0 * 1024.0),
            float(mask_export["byte_count"]) / (1024.0 * 1024.0),
        ),
    )
    request = {
        "schema_version": "mimics_interactive_algorithm_request.v1",
        "tool": action,
        "display_name": DISPLAY_NAMES[action],
        "parent_pid": os.getpid(),
        "shape": image_export["shape"],
        "spacing_mm": _spacing_mm(image, image_export["shape"]),
        "image_path": image_export["path"],
        "image_dtype": image_export["dtype"],
        "mask_path": mask_export["path"],
        "base_mask_sha256": mask_export["sha256"],
        "status_path": os.path.join(job_dir, "status.json"),
        "cancel_path": os.path.join(job_dir, "cancel.requested"),
        "log_path": os.path.join(job_dir, "worker.log"),
        "result_path": os.path.join(outputs, "result.u8"),
        "logits_result_path": os.path.join(outputs, "slice_logits.f32"),
    }
    section = config.get("scribbleprompt") or {}
    prompt_values = dict(prompts or {})
    request.update(
        {
            "checkpoint_path": _checkpoint_path(config),
            "device": section.get("device", "cpu"),
            "input_size": section.get("input_size", 128),
            "prior_logit_magnitude": section.get("prior_logit_magnitude", 6.0),
            "gpu_lock_timeout_seconds": section.get("gpu_lock_timeout_seconds", 120),
            "scribbles": list(prompt_values.get("scribbles") or []),
            "points": list(prompt_values.get("points") or []),
            "boxes": list(prompt_values.get("boxes") or []),
            "prompt_plane_axis": prompt_values.get("plane_axis"),
            "prompt_plane_index": prompt_values.get("plane_index"),
        }
    )
    request.update(_session_values(target, mask_export["sha256"]))
    request_path = os.path.join(job_dir, "request.json")
    runtime_common.write_json_atomic(request_path, request)
    _write_initial_status(request["status_path"], action)
    return request_path, request


def _launch(action, image, target, config, job_dir, prompts=None, visual_objects=None):
    request_path, request = _prepare_request(
        action, image, target, config, job_dir, prompts=prompts
    )
    process = subprocess.Popen(
        [_python_exe(), _worker_script(), "--request", request_path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=runtime_common.background_env(),
        creationflags=(
            # BELOW_NORMAL_PRIORITY_CLASS: keep the compute worker from
            # competing with the foreground Mimics GUI for CPU.
            getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
            if os.name == "nt" else 0
        ),
    )
    timeout = float((config.get(action) or {}).get("timeout_seconds", 1800))
    monitor = {
        "key": job_dir,
        "action": action,
        "display_name": DISPLAY_NAMES[action],
        "image": image,
        "image_guid": nnm._object_id(image),
        "launch_project_path": nnm._current_project_path() or "",
        "target": target,
        "target_guid": nnm._object_id(target),
        "target_name": str(getattr(target, "name", "") or DISPLAY_NAMES[action]),
        "base_sha256": request["base_mask_sha256"],
        "request": request,
        "process": process,
        "deadline": time.time() + timeout,
        "timeout_seconds": timeout,
        "last_message": "",
        "busy": False,
        "visual_objects": list(visual_objects or []),
    }
    if not _start_monitor(monitor, float(config.get("poll_seconds", 0.25))):
        runtime_common.terminate_process_async(process=process, graceful_seconds=1.0)
        raise RuntimeError("Mimics could not start a safe result monitor; the background task was stopped.")
    _mimics_log(
        logging.INFO,
        "{0} started in the background. PID: {1}. Continue using Mimics; completion will be shown automatically.".format(
            DISPLAY_NAMES[action], process.pid
        ),
    )
    return 0


def _stop_monitor(key):
    monitor = _MONITORS.pop(key, None)
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
        try:
            win32_timer[0].KillTimer(None, win32_timer[1])
        except Exception:
            pass


def _cancel_monitor(key):
    """Stop one owned worker before detaching its Mimics-side monitor."""
    monitor = _MONITORS.get(key)
    if not monitor:
        return
    try:
        _request_stop(monitor, "Stop Background Services")
    finally:
        _delete_visual_objects(monitor.get("visual_objects"))
        _stop_monitor(key)


def _find_mask(guid):
    for mask in mimics.data.masks:
        if nnm._object_id(mask) == guid:
            return mask
    return None


def _same_path(left, right):
    if not left or not right:
        return not left and not right
    try:
        return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
            os.path.abspath(str(right))
        )
    except Exception:
        return False


def _monitor_target_is_open(monitor):
    launch_project = str(monitor.get("launch_project_path") or "")
    current_project = str(nnm._current_project_path() or "")
    if launch_project and not _same_path(launch_project, current_project):
        return False, "Reopen the project used when {0} started.".format(
            monitor.get("display_name") or "the task"
        )
    image = _active_image()
    if image is None or nnm._object_id(image) != str(monitor.get("image_guid") or ""):
        return False, "Reactivate the image used when {0} started.".format(
            monitor.get("display_name") or "the task"
        )
    return True, ""


def _current_mask_sha(mask):
    if mask is None:
        return ""
    return _stream_buffer(
        _buffer_byte_view(mask.get_voxel_buffer()),
        compute_sha=True,
        progress_callback=_update_gui,
    )


def _create_copy(image, name):
    active = _active_image()
    if active is not None and nnm._object_id(active) == nnm._object_id(image):
        image = active
    return nnm._create_result_mask(image, name)


def _choose_result_target(monitor, result):
    target = _find_mask(monitor.get("target_guid"))
    changed = target is None
    if target is not None:
        try:
            changed = _current_mask_sha(target) != monitor.get("base_sha256")
        except Exception:
            changed = True
    display = monitor["display_name"]
    if changed:
        answer = mimics.dialogs.question_box(
            title="{0} Ready".format(display),
            message=(
                "{0} completed, but the selected Mask changed while it was running.\n\n"
                "Create Editable Copy preserves the newer Mask and applies the completed result to a new Mask."
            ).format(display),
            buttons="Create Editable Copy;Discard",
            ui_blocking=True,
        )
        if answer != "Create Editable Copy":
            return None
        return _create_copy(monitor["image"], "{0} - {1}".format(monitor["target_name"], display))
    answer = mimics.dialogs.question_box(
        title="{0} Ready".format(display),
        message=(
            "{0} completed in {1} seconds.\n\n"
            "Update Selected Mask applies the result to {2}.\n"
            "Create Editable Copy preserves it and creates a new editable Mask."
        ).format(
            display,
            result.get("elapsed_seconds", "?"),
            monitor["target_name"],
        ),
        buttons="Update Selected Mask;Create Editable Copy;Discard",
        ui_blocking=True,
    )
    if answer == "Update Selected Mask":
        return target
    if answer == "Create Editable Copy":
        return _create_copy(monitor["image"], "{0} - {1}".format(monitor["target_name"], display))
    return None


def _store_scribble_session(mask, result):
    if not result.get("logits_path"):
        return
    nnm._metadata_set(mask, SESSION_PATH_METADATA, result.get("logits_path"))
    nnm._metadata_set(mask, SESSION_SHAPE_METADATA, json.dumps(result.get("logits_shape") or []))
    nnm._metadata_set(mask, SESSION_AXIS_METADATA, str(result.get("plane_axis")))
    nnm._metadata_set(mask, SESSION_INDEX_METADATA, str(result.get("plane_index")))
    nnm._metadata_set(mask, SESSION_MASK_SHA_METADATA, result.get("result_sha256", ""))


def _apply_completed(monitor, state):
    result = state.get("result") or {}
    path = str(result.get("result_path") or "")
    shape = result.get("shape") or monitor["request"].get("shape")
    if not path or not os.path.isfile(path):
        raise RuntimeError("The background worker completed without a result Mask buffer.")
    target = _choose_result_target(monitor, dict(result, elapsed_seconds=state.get("elapsed_seconds", "?")))
    if target is None:
        _mimics_log(logging.INFO, "{0} result was discarded; the project was not changed.".format(monitor["display_name"]))
        return
    _mimics_log(
        logging.INFO,
        "{0}: applying the completed 3D Mask to Mimics.".format(monitor["display_name"]),
    )
    _update_gui()
    apply_started = time.time()
    nnm._set_mask_from_u8(
        target,
        path,
        shape,
        "Apply {0} Result".format(monitor.get("display_name") or "Interactive Algorithm"),
    )
    try:
        target.visible = True
        target.selected = True
    except Exception:
        pass
    if monitor["action"] == ACTION_SCRIBBLEPROMPT:
        _store_scribble_session(target, result)
    _mimics_log(
        logging.INFO,
        "{0} result applied to Mask '{1}'. Foreground voxels: {2}.".format(
            monitor["display_name"],
            getattr(target, "name", ""),
            result.get("foreground_voxels", "?"),
        ),
    )
    _mimics_log(
        logging.INFO,
        "{0}: Mimics Mask handoff completed in {1}s.".format(
            monitor["display_name"], round(time.time() - apply_started, 2)
        ),
    )
    _update_gui()


def _finish_monitor(monitor, state):
    key = monitor["key"]
    status = state.get("status")
    try:
        if status == "completed":
            _apply_completed(monitor, state)
        elif status == "failed":
            error = state.get("error") or state.get("message") or "Unknown background error."
            _mimics_log(logging.ERROR, "{0} failed: {1}".format(monitor["display_name"], error))
            mimics.dialogs.message_box(
                "{0} failed.\n\n{1}\n\nLog: {2}".format(
                    monitor["display_name"], error, state.get("log_path", monitor["request"].get("log_path"))
                ),
                title="{0} Failed".format(monitor["display_name"]),
                ui_blocking=True,
            )
        else:
            _mimics_log(logging.INFO, "{0} stopped; no Mask was changed.".format(monitor["display_name"]))
    finally:
        _delete_visual_objects(monitor.get("visual_objects"))
        _stop_monitor(key)
        _update_gui()


def _monitor_tick(monitor):
    if monitor.get("busy"):
        return
    token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "{0} result monitor".format(monitor["display_name"])
    )
    if not token:
        owner = runtime_common.active_local_operation("mask_buffer_access") or {}
        owner_text = str(owner.get("owner") or "another Mask operation")
        due, elapsed = runtime_common.progress_notice_due(
            monitor,
            "interactive_buffer_wait",
            detail=owner_text,
            interval_seconds=60.0,
            initial_delay_seconds=5.0,
        )
        if due:
            _mimics_log(
                logging.INFO,
                "{0} result handling is waiting for {1} ({2}s). Run {0} "
                "again to inspect or stop the pending task.".format(
                    monitor["display_name"], owner_text, int(elapsed)
                ),
            )
        return
    runtime_common.clear_progress_notice(monitor, "interactive_buffer_wait")
    monitor["busy"] = True
    try:
        state = runtime_common.read_json(monitor["request"]["status_path"], {})
        status = state.get("status")
        message = str(state.get("message") or "")
        if message and message != monitor.get("last_message"):
            monitor["last_message"] = message
            if state.get("progress_indeterminate"):
                log_message = "{0}: {1}".format(monitor["display_name"], message)
            else:
                log_message = "{0}: {1}% - {2}".format(
                    monitor["display_name"], state.get("progress_percent", 0), message
                )
            _mimics_log(logging.INFO, log_message)
        if status in ("completed", "failed", "cancelled"):
            if status == "completed":
                target_open, reason = _monitor_target_is_open(monitor)
                if not target_open:
                    due, elapsed = runtime_common.progress_notice_due(
                        monitor,
                        "interactive_apply_target",
                        detail=reason,
                        interval_seconds=60.0,
                        initial_delay_seconds=0.0,
                    )
                    if due:
                        monitor["waiting_for_target_reason"] = reason
                        _mimics_log(
                            logging.WARNING,
                            "{0} result is ready but was not applied{1}. {2} "
                            "Run {0} again to inspect or stop it.".format(
                                monitor["display_name"],
                                " ({0}s)".format(int(elapsed))
                                if elapsed >= 1.0 else "",
                                reason,
                            ),
                        )
                    return
                runtime_common.clear_progress_notice(
                    monitor, "interactive_apply_target"
                )
            _finish_monitor(monitor, state)
            return
        process = monitor.get("process")
        if process is not None and process.poll() is not None:
            if os.path.isfile(monitor["request"]["cancel_path"]):
                _finish_monitor(monitor, {"status": "cancelled"})
            else:
                _finish_monitor(
                    monitor,
                    {
                        "status": "failed",
                        "error": "The external worker exited before writing a terminal status (code {}).".format(
                            process.returncode
                        ),
                        "log_path": monitor["request"].get("log_path"),
                    },
                )
            return
        if time.time() > monitor["deadline"]:
            _request_stop(monitor, "timeout")
            _finish_monitor(
                monitor,
                {
                    "status": "failed",
                    "error": "The operation exceeded its {} second timeout and was stopped.".format(
                        int(monitor["timeout_seconds"])
                    ),
                    "log_path": monitor["request"].get("log_path"),
                },
            )
    except Exception as exc:
        _finish_monitor(
            monitor,
            {
                "status": "failed",
                "error": "Mimics could not process the background result: {0}".format(exc),
                "log_path": monitor["request"].get("log_path"),
            },
        )
    finally:
        monitor["busy"] = False
        runtime_common.release_local_operation("mask_buffer_access", token)


def _start_win32_monitor(monitor, poll_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        callback_type = ctypes.WINFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint
        )

        def callback(hwnd, message, timer_id, tick_count):
            _monitor_tick(monitor)

        callback_ptr = callback_type(callback)
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, callback_type]
        user32.SetTimer.restype = ctypes.c_size_t
        timer_id = user32.SetTimer(None, 0, max(100, int(poll_seconds * 1000)), callback_ptr)
        if not timer_id:
            return False
        monitor["callback"] = callback_ptr
        monitor["win32_timer"] = (user32, timer_id)
        _MONITORS[monitor["key"]] = monitor
        return True
    except Exception:
        return False


def _start_monitor(monitor, poll_seconds):
    if _start_win32_monitor(monitor, poll_seconds):
        return True
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is None:
            return False
        timer = QTimer()
        timer.timeout.connect(lambda: _monitor_tick(monitor))
        timer.start(max(100, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        _MONITORS[monitor["key"]] = monitor
        return True
    except Exception:
        return False


def _request_stop(monitor, reason):
    path = monitor["request"]["cancel_path"]
    try:
        with open(path, "w") as handle:
            handle.write(str(reason or "user") + "\n")
    except Exception:
        pass
    runtime_common.terminate_process_async(
        process=monitor.get("process"), graceful_seconds=3.0
    )
    _mimics_log(logging.INFO, "Stop requested for {0}.".format(monitor["display_name"]))


def _active_monitors(action):
    return [value for value in list(_MONITORS.values()) if value.get("action") == action]


def _show_running(action):
    active = _active_monitors(action)
    if not active:
        return False
    monitor = active[-1]
    state = runtime_common.read_json(monitor["request"]["status_path"], {})
    progress = (
        "Interactive session"
        if state.get("progress_indeterminate")
        else "{0}%".format(state.get("progress_percent", 0))
    )
    answer = mimics.dialogs.question_box(
        title="{0} Running".format(DISPLAY_NAMES[action]),
        message=(
            "{0} is running in the background.\n\nProgress: {1}\nStage: {2}\n\n"
            "Mimics remains available while it runs."
        ).format(
            DISPLAY_NAMES[action],
            progress,
            state.get("message", state.get("phase", "starting")),
        ),
        buttons="Keep Running;Stop",
        ui_blocking=True,
    )
    if answer == "Stop":
        _request_stop(monitor, "user")
    return True
def _start_scribbleprompt(config):
    checkpoint = _checkpoint_path(config)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise RuntimeError(
            "The official ScribblePrompt UNet checkpoint is missing.\n\nExpected: {0}\n\n"
            "Place ScribblePrompt_unet_v1_nf192_res128.pt at this path before drawing prompts.".format(
                checkpoint or "external/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt"
            )
        )
    image = _active_image()
    target = _selected_mask(image)
    job_dir = _unique_job_dir(ACTION_SCRIBBLEPROMPT)
    visual_objects = []
    try:
        prompts = _collect_scribbleprompt_prompts(
            image, target, job_dir, visual_objects
        )
        if not prompts:
            _delete_visual_objects(visual_objects)
            shutil.rmtree(job_dir, ignore_errors=True)
            return 0
        try:
            target.selected = True
        except Exception:
            pass
        return _launch(
            ACTION_SCRIBBLEPROMPT,
            image,
            target,
            config,
            job_dir,
            prompts=prompts,
            visual_objects=visual_objects,
        )
    except Exception:
        _delete_visual_objects(visual_objects)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
def main(action):
    display = DISPLAY_NAMES.get(action, str(action))
    try:
        if action not in DISPLAY_NAMES:
            raise RuntimeError("Unknown interactive algorithm action: {0}".format(action))
        if _show_running(action):
            return 0
        buffer_owner = runtime_common.active_local_operation("mask_buffer_access")
        if buffer_owner:
            mimics.dialogs.message_box(
                title=display,
                message=(
                    "{0} cannot take a consistent image and Mask snapshot while {1} is using Mimics buffers.\n\n"
                    "Wait for that operation to finish or stop it, then retry."
                ).format(display, buffer_owner.get("owner") or "another task"),
                ui_blocking=False,
            )
            return 1
        config = _config()
        _cleanup_old_jobs(action, config)
        return _start_scribbleprompt(config)
    except Exception as exc:
        _mimics_log(logging.ERROR, "{0} error: {1}".format(display, exc))
        mimics.dialogs.message_box(
            "{0} could not continue.\n\n{1}".format(display, exc),
            title=display,
            ui_blocking=True,
        )
        return 2
