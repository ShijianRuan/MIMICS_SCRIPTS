# -*- coding: utf-8 -*-
"""Background Mimics script to create .mcs files.

Runs inside Mimics Python 3.5.2 in background mode (MimicsResearch.exe -b).
Reads prepare manifests from work directories and creates .mcs files one by one.

Usage (from Mimics command line):
    MimicsResearch.exe -b -run_script create_mcs_batch.py <output_dir>

Or called programmatically:
    create_mcs_batch.main(output_dir)
"""

from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import mimics

import dataset_manifest
import runtime_common


QUEUE_ACTIVE_FILE = "_mcs_queue_active.json"
QUEUE_DONE_FILE = "_mcs_queue_done.json"
QUEUE_STOP_FILE = "_mcs_queue_stop.json"
STATUS_FILE = "_mcs_batch_status.json"
CURRENT_CASE_FILE = "_mcs_current_case.json"
LOCK_FILE = "_mcs_batch.lock"
LOG_FILE = os.path.join("logs", "_create_mcs_batch.log")
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
RUNTIME_SUBDIR = ".mimics_runtime"
_ACTIVE_RUNTIME_DIR = None
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_KIND_METADATA = "mimics_script.source_image_kind"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_IMAGE_INDEX_SPACE_METADATA = "mimics_script.source_image_index_space"
SOURCE_IMAGE_MODALITY_METADATA = "mimics_script.source_image_modality"
SOURCE_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.source_world_coordinate_system"
MIMICS_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.mimics_world_coordinate_system"
SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA = "mimics_script.source_to_mimics_world_matrix"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA = "mimics_script.mimics_to_source_index_matrix"
SOURCE_CASE_DIR_METADATA = "mimics_script.source_case_dir"
write_json_atomic = runtime_common.write_json_atomic
safe_case_filename = runtime_common.safe_filename


def _record_created_project(output_dir, case_id, mcs_path, manifest_data):
    """Publish the source/Mimics grid relationship after the .mcs is durable."""
    source_image = manifest_data.get("source_image_path") or ""
    source_case_dir = manifest_data.get("source_case_dir") or ""
    source_geometry = {
        "shape": manifest_data.get("source_image_shape") or [],
        "voxel_to_ras_matrix": manifest_data.get("source_voxel_to_ras_matrix") or [],
        "world_coordinate_system": "RAS",
        "source_world_coordinate_system": manifest_data.get(
            "source_world_coordinate_system", ""
        ),
    }
    mimics_geometry = {
        "shape": manifest_data.get("actual_mimics_shape") or (
            (manifest_data.get("masks") or [{}])[0].get("image_shape")
            if manifest_data.get("masks")
            else []
        ),
        "voxel_to_ras_matrix": manifest_data.get(
            "mimics_voxel_to_ras_matrix"
        ) or [],
        "world_coordinate_system": "RAS",
        "mimics_world_coordinate_system": manifest_data.get(
            "mimics_world_coordinate_system", ""
        ),
    }
    dataset_manifest.update_case(
        output_dir,
        case_id,
        image_path=source_image,
        mcs_path=mcs_path,
        source_case_dir=source_case_dir,
        source_geometry=source_geometry,
        mimics_geometry=mimics_geometry,
        provenance={
            "last_operation": "mimics_import",
            "source_modality": str(
                manifest_data.get("source_image_modality") or ""
            ).strip().upper(),
            "source_fingerprint": manifest_data.get("source_fingerprint") or "",
            "resampled_source_grid": bool(
                manifest_data.get("resampled_source_grid")
            ),
            "resample_reason": manifest_data.get("resample_reason") or "",
        },
    )


def runtime_path(output_dir, *parts):
    base = _ACTIVE_RUNTIME_DIR or os.path.join(output_dir, RUNTIME_SUBDIR)
    return os.path.join(base, *parts)


def prune_empty_work_parents(work_dir):
    current = os.path.dirname(work_dir or "")
    for _ in range(2):
        if not current or not os.path.isdir(current):
            return
        try:
            os.rmdir(current)
        except OSError:
            return
        current = os.path.dirname(current)


def rotate_log(path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_BACKUPS):
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


def log_message(output_dir, message):
    text = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(text)
    try:
        log_path = runtime_path(output_dir, LOG_FILE)
        log_dir = os.path.dirname(log_path)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        rotate_log(log_path)
        with open(log_path, "a") as handle:
            handle.write(text + "\n")
    except Exception:
        pass


def record_failed_case(output_dir, case_id, phase, error, traceback_text=None):
    try:
        failed_dir = runtime_path(output_dir or os.getcwd(), "_failed_cases")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir)
        payload = {
            "case_id": str(case_id or "unknown"),
            "phase": str(phase or "unknown"),
            "error": str(error or ""),
            "failed_at_epoch": time.time(),
        }
        if traceback_text:
            payload["traceback"] = traceback_text
        filename = "{0}_{1}.json".format(
            safe_case_filename(case_id),
            safe_case_filename(phase),
        )
        write_json_atomic(os.path.join(failed_dir, filename), payload)
    except Exception:
        pass


def update_status(output_dir, status, **details):
    creation_completed = int(details.get("completed", 0) or 0)
    creation_failed = int(details.get("failed", 0) or 0)
    preparation = _preparation_progress(output_dir)
    preparation_failed = int(preparation.get("failed", 0) or 0)
    details["completed"] = creation_completed
    details["failed"] = creation_failed + preparation_failed
    details["creation_completed"] = creation_completed
    details["creation_failed"] = creation_failed
    details["preparation_failed"] = preparation_failed
    if int(preparation.get("total", 0) or 0) > 0:
        details["total"] = int(preparation.get("total", 0) or 0)
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
    }
    payload.update(details)
    try:
        write_json_atomic(runtime_path(output_dir, STATUS_FILE), payload)
    except Exception:
        pass


def _preparation_progress(output_dir):
    """Read producer totals so creation status includes preparation failures."""
    active = runtime_common.read_json(
        runtime_path(output_dir, QUEUE_ACTIVE_FILE), {}
    ) or {}
    done = runtime_common.read_json(
        runtime_path(output_dir, QUEUE_DONE_FILE), {}
    ) or {}
    source = active if str(active.get("status") or "").lower() == "active" else done
    completed = int(source.get("completed", 0) or 0)
    failed = int(source.get("failed", 0) or 0)
    total = int(
        source.get("total", 0)
        or source.get("total_count", 0)
        or (completed + failed)
        or 0
    )
    return {"completed": completed, "failed": failed, "total": total}


def _staging_mcs_path(output_mcs):
    """Return a same-directory temporary project path for atomic publication."""
    path = os.path.abspath(output_mcs)
    stem = path[:-4] if path.lower().endswith(".mcs") else path
    return "{0}.creating.{1}.mcs".format(stem, os.getpid())


def _remove_file(path):
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _publish_mcs(staging_path, output_mcs, retries=20):
    """Publish a completed project without exposing a partially saved .mcs."""
    if not os.path.isfile(staging_path):
        raise RuntimeError(
            "Mimics reported a successful save but the staged project was not found: {0}".format(
                staging_path
            )
        )
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            os.replace(staging_path, output_mcs)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
    raise RuntimeError(
        "The completed staged project could not replace the destination {0}: {1}".format(
            output_mcs, last_error
        )
    )


def _write_current_case(output_dir, case_id, work_dir, output_mcs,
                        staging_mcs, descriptor_path, completed, failed):
    """Persist enough state for the external supervisor to recover a native crash."""
    marker_path = runtime_path(output_dir, CURRENT_CASE_FILE)
    write_json_atomic(
        marker_path,
        {
            "case_id": str(case_id),
            "work_dir": os.path.abspath(work_dir),
            "output_mcs": os.path.abspath(output_mcs),
            "staging_mcs": os.path.abspath(staging_mcs),
            "descriptor_path": os.path.abspath(descriptor_path) if descriptor_path else "",
            "worker_pid": os.getpid(),
            "completed_before_case": int(completed or 0),
            "failed_before_case": int(failed or 0),
            "started_at_epoch": time.time(),
        },
    )
    return marker_path


def _clear_current_case(marker_path, case_id):
    """Remove only the marker still owned by this worker and case."""
    try:
        payload = runtime_common.read_json(marker_path, {}) or {}
        if (
            int(payload.get("worker_pid") or 0) == os.getpid()
            and str(payload.get("case_id") or "") == str(case_id)
        ):
            os.remove(marker_path)
    except OSError:
        pass
    except Exception:
        pass


def acquire_lock(output_dir):
    lock_path = runtime_path(output_dir, LOCK_FILE)
    lock_parent = os.path.dirname(lock_path)
    if not os.path.isdir(lock_parent):
        try:
            os.makedirs(lock_parent)
        except OSError:
            if not os.path.isdir(lock_parent):
                raise
    guard = runtime_common._open_resource_guard(lock_path, wait_seconds=2.0)
    if guard is None:
        return None
    try:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            try:
                with open(lock_path, "r") as handle:
                    pid = int((handle.read() or "0").strip() or "0")
                if pid and process_exists(pid):
                    return None
            except Exception:
                pass
            try:
                os.remove(lock_path)
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError:
                return None
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return lock_path
    finally:
        runtime_common._close_resource_guard(guard)


def process_exists(pid):
    return runtime_common.process_exists(pid)


def metadata_set(obj, name, value):
    text = "" if value is None else str(value)
    try:
        item = obj.metadata.find(name)
    except Exception:
        item = None
    try:
        if item is None:
            obj.metadata.create(name=name, value=text)
        else:
            item.value = text
    except Exception as exc:
        raise RuntimeError(
            "Could not persist required image metadata '{0}': {1}".format(
                name,
                exc,
            )
        )


def _matrix_close(left, right, tol=1e-4):
    try:
        if not left or not right:
            return False
        if len(left) != 4 or len(right) != 4:
            return False
        for row_index in range(4):
            if len(left[row_index]) != 4 or len(right[row_index]) != 4:
                return False
            for col_index in range(4):
                if abs(float(left[row_index][col_index]) - float(right[row_index][col_index])) > tol:
                    return False
        return True
    except Exception:
        return False


def _lps_point_to_ras(point):
    return [-float(point[0]), -float(point[1]), float(point[2])]


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
        pass
    return _point_values(getter(values[0], values[1], values[2]))


def _derive_mimics_voxel_to_ras_matrix(image, image_shape, fallback):
    """Derive the actual open Mimics image voxel grid in RAS coordinates."""
    if not image_shape:
        return None
    try:
        origin = _lps_point_to_ras(_voxel_center(image, [0, 0, 0]))
        matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
        for axis in range(3):
            if int(image_shape[axis]) > 1:
                index = [0, 0, 0]
                index[axis] = 1
                point = _lps_point_to_ras(_voxel_center(image, index))
                vector = [point[i] - origin[i] for i in range(3)]
            elif fallback and len(fallback) == 4 and len(fallback[axis]) == 4:
                vector = [float(fallback[row][axis]) for row in range(3)]
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


def _bridge_python():
    return os.environ.get("MIMICS_BRIDGE_PYTHON", "")


def _bridge_script():
    return os.environ.get("MIMICS_BRIDGE_SCRIPT", "")


def _resample_masks_to_actual_grid(result, active_image_shape, actual_mimics_voxel_to_ras, work_dir):
    expected_matrix = result.get("mimics_voxel_to_ras_matrix") or []
    masks = result.get("masks") or []
    if not active_image_shape or not actual_mimics_voxel_to_ras or not masks:
        return result
    expected_shapes = []
    for row in masks:
        if row.get("mimics_shape"):
            expected_shapes.append([int(value) for value in row.get("mimics_shape")])
    shape_matches = all(shape == active_image_shape for shape in expected_shapes) if expected_shapes else False
    matrix_matches = _matrix_close(expected_matrix, actual_mimics_voxel_to_ras)
    # Shape equality does not imply physical-grid equality. A same-shaped image
    # may still be translated, mirrored, oblique, or use a different spacing.
    # Always resample source masks when the live Mimics matrix differs.
    need_resample = (not shape_matches) or (not matrix_matches)
    if not need_resample:
        return result

    python_exe = _bridge_python()
    bridge_script = _bridge_script()
    if not python_exe or not bridge_script or not os.path.isfile(bridge_script):
        raise RuntimeError(
            "Imported Mimics image grid differs from prepared mask grid, but the external bridge "
            "is not configured. Refusing to inject reshaped masks because that would corrupt orientation."
        )

    actual_buffers_dir = os.path.join(work_dir, "buffers_mimics_actual")
    params = {
        "action": "prepare_masks_for_grid",
        "masks": [{"name": row.get("name"), "mask_path": row.get("mask_path")} for row in masks],
        "target_shape": active_image_shape,
        "target_voxel_to_ras_matrix": actual_mimics_voxel_to_ras,
        "source_voxel_to_ras_matrix": result.get("source_voxel_to_ras_matrix") or [],
        "buffers_out": actual_buffers_dir,
        "axes": [0, 1, 2],
        "flips": [False, False, False],
    }
    process = subprocess.Popen(
        [python_exe, bridge_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **runtime_common.background_process_kwargs()
    )
    try:
        stdout, stderr = process.communicate(
            input=json.dumps(params).encode("utf-8"),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("mimics_bridge.py timed out after 600 seconds")
    if process.returncode != 0:
        raise RuntimeError(
            "External mask resampling failed with exit code {0}: {1}".format(
                process.returncode,
                stderr.decode("utf-8", "replace")[:1000],
            )
        )
    try:
        bridge_result = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("External mask resampling returned invalid JSON: {0}".format(exc))
    if bridge_result.get("status") != "ok":
        raise RuntimeError("External mask resampling failed: {0}".format(bridge_result.get("error", "unknown error")))

    updated = dict(result)
    updated["masks"] = bridge_result.get("masks") or []
    updated["mimics_voxel_to_ras_matrix"] = actual_mimics_voxel_to_ras
    if bridge_result.get("mimics_to_source_index_matrix"):
        updated["mimics_to_source_index_matrix"] = bridge_result.get("mimics_to_source_index_matrix")
    updated["actual_mimics_grid_resampled"] = True
    return updated


def inject_buffer(mask, buffer_path, mimics_shape):
    """Inject a .u8 buffer into a Mimics mask."""
    with open(buffer_path, "rb") as f:
        raw = f.read()
    expected = 1
    for dim in mimics_shape:
        expected *= int(dim)
    if len(raw) != expected:
        raise RuntimeError(
            "buffer byte count mismatch: {} != {}  path={}".format(len(raw), expected, buffer_path)
        )
    try:
        import numpy as np
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(mimics_shape)).astype(np.bool_)
        mask.set_voxel_buffer(pixels)
        return "numpy"
    except ImportError:
        view = memoryview(bytearray(raw)).cast("?", shape=list(mimics_shape))
        mask.set_voxel_buffer(view)
        return "memoryview"


def _shape_product(shape):
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


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


def _shape_voxels(shape):
    result = 1
    for value in shape or []:
        result *= int(value)
    return int(result)


def _normalize_shape(shape):
    if not shape or len(shape) < 3:
        return None
    try:
        return [int(shape[0]), int(shape[1]), int(shape[2])]
    except Exception:
        return None


def _expected_image_shape_from_manifest(result):
    masks = result.get("masks") or []
    for row in masks:
        shape = _normalize_shape(row.get("image_shape"))
        if shape:
            return shape
    shape = _normalize_shape(result.get("source_image_shape"))
    if shape:
        return shape
    return None


def _select_best_imported_image(expected_shape=None):
    """Pick the most likely volume image after DICOM import.

    Mimics can occasionally expose multiple imported image objects when a
    DICOM folder is interpreted as several series. Choosing images[0]
    silently risks saving a single-slice project. This selector prefers the
    manifest-expected shape, then largest depth and voxel count.
    """
    candidates = []
    total = len(mimics.data.images)
    for index in range(total):
        try:
            image = mimics.data.images[index]
        except Exception:
            continue
        shape = _active_image_shape(image)
        if not shape:
            continue
        shape = [int(shape[0]), int(shape[1]), int(shape[2])]
        voxels = _shape_voxels(shape)
        if expected_shape:
            diff = (
                abs(int(shape[0]) - int(expected_shape[0]))
                + abs(int(shape[1]) - int(expected_shape[1]))
                + abs(int(shape[2]) - int(expected_shape[2]))
            )
            exact = 1 if shape == expected_shape else 0
        else:
            diff = 10 ** 9
            exact = 0
        candidates.append({
            "index": index,
            "image": image,
            "shape": shape,
            "voxels": voxels,
            "diff": diff,
            "exact": exact,
        })

    if not candidates:
        return None, []

    def _score(row):
        # Higher is better.
        return (
            int(row.get("exact", 0)),
            -int(row.get("diff", 10 ** 9)),
            int(row.get("shape", [0, 0, 0])[2]),
            int(row.get("voxels", 0)),
            -int(row.get("index", 0)),
        )

    best = max(candidates, key=_score)
    return best, candidates


def _format_image_candidates(candidates):
    rows = []
    for item in candidates:
        rows.append(
            "#{0} shape={1} voxels={2} diff={3} exact={4}".format(
                int(item.get("index", -1)),
                item.get("shape"),
                int(item.get("voxels", 0)),
                int(item.get("diff", -1)),
                bool(item.get("exact", 0)),
            )
        )
    return "; ".join(rows)


def _shape_looks_single_slice(shape):
    shape = _normalize_shape(shape)
    if not shape:
        return False
    return min(int(shape[0]), int(shape[1]), int(shape[2])) <= 1


def _close_project_safely():
    try:
        mimics.file.close_project()
    except Exception:
        pass


def _dicom_import_profiles():
    # Try canonical grouping first, then progressively relax grouping keys
    # when background import splits one volume into many single-slice images.
    return [
        {
            "name": "default",
            "image_center_grouping": False,
            "patient_name_grouping": True,
            "series_description_grouping": True,
            "study_description_grouping": True,
        },
        {
            "name": "no_description_grouping",
            "image_center_grouping": False,
            "patient_name_grouping": True,
            "series_description_grouping": False,
            "study_description_grouping": False,
        },
        {
            "name": "no_grouping_keys",
            "image_center_grouping": False,
            "patient_name_grouping": False,
            "series_description_grouping": False,
            "study_description_grouping": False,
        },
    ]


def create_mcs_from_manifest(work_dir, output_mcs):
    """Create a .mcs file from a prepare manifest.

    Steps:
        1. Import DICOM into Mimics
        2. Create masks and inject .u8 buffers
        3. Save .mcs
        4. Close project
    """
    manifest_path = os.path.join(work_dir, "prepare_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("manifest not found: {}".format(manifest_path))

    with open(manifest_path, "r") as f:
        result = json.load(f)

    dicom_folder = result["dicom_folder"]
    expected_shape = _expected_image_shape_from_manifest(result)

    selected = None
    candidates = []
    used_profile = None
    import_profiles = _dicom_import_profiles()
    expected_is_volume = bool(expected_shape and min(expected_shape) > 1)

    for profile_index, profile in enumerate(import_profiles):
        _close_project_safely()
        mimics.file.import_dicom_images(
            source_folder=dicom_folder,
            image_center_grouping=bool(profile.get("image_center_grouping", False)),
            patient_name_grouping=bool(profile.get("patient_name_grouping", True)),
            series_description_grouping=bool(profile.get("series_description_grouping", True)),
            study_description_grouping=bool(profile.get("study_description_grouping", True)),
        )
        if len(mimics.data.images) == 0:
            continue

        selected, candidates = _select_best_imported_image(expected_shape)
        if not selected:
            continue

        all_single_slice = bool(candidates) and all(
            _shape_looks_single_slice(item.get("shape")) for item in candidates
        )

        if expected_is_volume and all_single_slice and profile_index < (len(import_profiles) - 1):
            print(
                "  import profile '{0}' produced {1} single-slice image objects; retrying with relaxed grouping.".format(
                    profile.get("name", "unknown"),
                    len(candidates),
                )
            )
            continue

        used_profile = profile
        break

    if not selected:
        raise RuntimeError("DICOM import produced no usable image volume")

    if len(candidates) > 1:
        print(
            "  imported {0} image objects (profile={1}); selecting candidate #{2}. candidates: {3}".format(
                len(candidates),
                (used_profile or {}).get("name", "unknown"),
                int(selected.get("index", -1)),
                _format_image_candidates(candidates),
            )
        )

    image = selected["image"]
    mimics.data.images.set_active(image)
    active_image_shape = selected.get("shape") or _active_image_shape(image)

    if expected_is_volume and _shape_looks_single_slice(active_image_shape):
        raise RuntimeError(
            "DICOM import selected a single-slice image ({0}) while manifest expects a volume ({1}). "
            "This indicates series splitting during background import; refusing to save a broken .mcs. "
            "Candidates: {2}".format(
                active_image_shape,
                expected_shape,
                _format_image_candidates(candidates),
            )
        )

    actual_mimics_voxel_to_ras = _derive_mimics_voxel_to_ras_matrix(
        image,
        active_image_shape,
        result.get("mimics_voxel_to_ras_matrix") or [],
    )
    if not active_image_shape or not actual_mimics_voxel_to_ras:
        raise RuntimeError(
            "Could not derive the live Mimics image grid. Refusing to import masks without a verified voxel-to-world transform."
        )
    result = _resample_masks_to_actual_grid(
        result,
        active_image_shape,
        actual_mimics_voxel_to_ras,
        work_dir,
    )
    result["actual_mimics_shape"] = list(active_image_shape)
    result["mimics_voxel_to_ras_matrix"] = actual_mimics_voxel_to_ras
    try:
        write_json_atomic(manifest_path, result)
    except Exception as exc:
        print(
            "  warning: could not persist verified grid metadata to the work "
            "manifest; the in-memory grid will still be recorded after save: {0}".format(
                exc
            )
        )
    if result.get("actual_mimics_grid_resampled"):
        print("  actual Mimics image grid differs from prepared grid; masks were resampled to the open Mimics grid.")
    else:
        print("  prepared mask grid accepted without external resampling.")
    mask_results = result["masks"]
    metadata_set(image, SOURCE_IMAGE_PATH_METADATA, result.get("source_image_path", ""))
    metadata_set(image, SOURCE_IMAGE_KIND_METADATA, result.get("source_image_kind", ""))
    metadata_set(image, SOURCE_IMAGE_SHAPE_METADATA, json.dumps(result.get("source_image_shape", [])))
    metadata_set(image, SOURCE_IMAGE_INDEX_SPACE_METADATA, result.get("source_image_index_space", ""))
    metadata_set(image, SOURCE_IMAGE_MODALITY_METADATA, result.get("source_image_modality", ""))
    metadata_set(image, SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, result.get("source_world_coordinate_system", ""))
    metadata_set(image, MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, result.get("mimics_world_coordinate_system", ""))
    metadata_set(
        image,
        SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA,
        json.dumps(result.get("source_to_mimics_world_matrix", [])),
    )
    metadata_set(
        image,
        SOURCE_VOXEL_TO_RAS_MATRIX_METADATA,
        json.dumps(result.get("source_voxel_to_ras_matrix", [])),
    )
    metadata_set(
        image,
        MIMICS_VOXEL_TO_RAS_MATRIX_METADATA,
        json.dumps(result.get("mimics_voxel_to_ras_matrix", [])),
    )
    metadata_set(
        image,
        MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA,
        json.dumps(result.get("mimics_to_source_index_matrix", [])),
    )
    metadata_set(image, SOURCE_CASE_DIR_METADATA, result.get("source_case_dir", ""))

    # Disable GUI updates during mask creation to prevent progressive
    # display and crashes from rapid UI refreshes.
    gui_was_enabled = True
    try:
        mimics.disable_update_gui()
    except Exception:
        gui_was_enabled = False

    try:
        # Create masks and inject buffers.
        # Note: mask.image is read-only; masks are automatically linked
        # to the active image at creation time.  Do NOT set mask.image.
        # Set visible=False BEFORE injecting buffer data so Mimics never
        # shows the mask even briefly during set_voxel_buffer.
        for mr in mask_results:
            name = mr["name"]
            u8_path = mr["u8_path"]
            buf_shape = mr["mimics_shape"]
            if active_image_shape and [int(value) for value in buf_shape] != active_image_shape:
                raise RuntimeError(
                    "mask buffer shape does not match imported Mimics image for {0}: buffer={1}, image={2}. "
                    "The buffer was not reshaped because blind reshape corrupts orientation.".format(
                        name,
                        buf_shape,
                        active_image_shape,
                    )
                )

            mask = mimics.segment.create_mask()
            mask.name = name
            # Hide immediately, before any data is injected
            try:
                mask.visible = False
            except Exception:
                pass

            method = inject_buffer(mask, u8_path, buf_shape)
            print("    -> {}".format(name))
    finally:
        if gui_was_enabled:
            try:
                mimics.enable_update_gui()
            except Exception:
                pass

    # Save .mcs
    mcs_path = os.path.abspath(output_mcs)
    mcs_dir = os.path.dirname(mcs_path)
    if mcs_dir and not os.path.isdir(mcs_dir):
        os.makedirs(mcs_dir)
    mimics.file.save_project(filename=mcs_path, save_as_type="Mimics Project Files")
    print("  saved: {}".format(mcs_path))

    # Close project to free memory
    try:
        mimics.file.close_project()
    except Exception:
        pass

    result["saved_mcs_path"] = mcs_path
    return result


def main(output_dir=None, runtime_dir=None):
    """Entry point. Find all prepare manifests and create .mcs files.

    Stays alive in a loop, periodically scanning for new manifests.
    This avoids frequent Mimics restarts when manifests are produced
    one-by-one by the conversion process.  Exits after two consecutive
    scans find no new work (meaning conversion is likely done).
    """
    global _ACTIVE_RUNTIME_DIR
    if output_dir is None:
        # Read from sys.argv
        if len(sys.argv) > 1:
            output_dir = sys.argv[1]
        else:
            print("Usage: create_mcs_batch.py <output_dir>")
            return 1

    if not os.path.isdir(output_dir):
        print("Output dir not found: {}".format(output_dir))
        return 1

    _ACTIVE_RUNTIME_DIR = os.path.abspath(runtime_dir) if runtime_dir else os.path.join(output_dir, RUNTIME_SUBDIR)
    if not os.path.isdir(_ACTIVE_RUNTIME_DIR):
        os.makedirs(_ACTIVE_RUNTIME_DIR)

    lock_path = acquire_lock(output_dir)
    if not lock_path:
        log_message(output_dir, "Another background Mimics batch process is already running; exiting.")
        return 0

    active_path = runtime_path(output_dir, QUEUE_ACTIVE_FILE)
    done_path = runtime_path(output_dir, QUEUE_DONE_FILE)
    stop_path = runtime_path(output_dir, QUEUE_STOP_FILE)
    # A stale producer marker must not hold a Mimics license forever after the
    # foreground process crashes. If preparation is merely slow, it will
    # relaunch this worker when the next manifest is committed.
    idle_timeout = 120
    producer_heartbeat_timeout = 180
    poll_seconds = 10
    last_activity = time.time()
    consecutive_empty = 0
    previous_status = runtime_common.read_json(
        runtime_path(output_dir, STATUS_FILE), {}
    ) or {}
    resume_counts = str(previous_status.get("status") or "").lower() in (
        "recovering",
        "restarting",
    )
    total_completed = int(
        previous_status.get("creation_completed", previous_status.get("completed", 0)) or 0
    ) if resume_counts else 0
    total_failed = int(
        previous_status.get("creation_failed", previous_status.get("failed", 0)) or 0
    ) if resume_counts else 0
    cancelled = False

    try:
        log_message(output_dir, "Background Mimics batch process started.")
        update_status(
            output_dir,
            "running",
            completed=total_completed,
            failed=total_failed,
        )

        while True:
            # Find local work referenced by lightweight queue descriptors.
            # The large derived DICOM/buffer data intentionally stays on the
            # workstation, not on a potentially unreliable SMB output share.
            work_dirs = []
            queue_dir = runtime_path(output_dir, "prepared_queue")
            if os.path.isdir(queue_dir):
                for descriptor_path in sorted(
                    os.path.join(queue_dir, name) for name in os.listdir(queue_dir) if name.lower().endswith(".json")
                ):
                    descriptor = runtime_common.read_json(descriptor_path, {}) or {}
                    work_dir = descriptor.get("work_dir", "")
                    manifest = os.path.join(work_dir, "prepare_manifest.json")
                    if not work_dir or not os.path.isfile(manifest):
                        try:
                            if time.time() - os.path.getmtime(descriptor_path) > 5.0:
                                log_message(output_dir, "Discarding stale prepared-work descriptor: {0}".format(descriptor_path))
                                os.remove(descriptor_path)
                        except OSError:
                            pass
                        continue
                    case_id = str(descriptor.get("case_id") or os.path.basename(work_dir).replace("_work", ""))
                    mcs_path = descriptor.get("output_mcs") or os.path.join(output_dir, case_id + ".mcs")
                    work_dirs.append((case_id, work_dir, mcs_path, descriptor_path))

            # Backward compatibility for manifests prepared by older tools.
            legacy_roots = [output_dir, runtime_path(output_dir)]
            queued_work = set(item[1] for item in work_dirs)
            for scan_root in legacy_roots:
                if not os.path.isdir(scan_root):
                    continue
                for item in sorted(os.listdir(scan_root)):
                    if not item.endswith("_work"):
                        continue
                    work_dir = os.path.join(scan_root, item)
                    if work_dir in queued_work:
                        continue
                    manifest = os.path.join(work_dir, "prepare_manifest.json")
                    if not os.path.isfile(manifest):
                        continue
                    case_id = item.replace("_work", "")
                    manifest_data = runtime_common.read_json(manifest, {}) or {}
                    mcs_path = manifest_data.get("output_mcs") or os.path.join(output_dir, case_id + ".mcs")
                    work_dirs.append((case_id, work_dir, mcs_path, ""))

            filtered_work_dirs = []
            for case_id, work_dir, mcs_path, descriptor_path in work_dirs:
                manifest = os.path.join(work_dir, "prepare_manifest.json")
                failed_marker = os.path.join(work_dir, "create_failed.json")
                if os.path.isfile(manifest):
                    try:
                        with open(manifest, "r") as handle:
                            manifest_data = json.load(handle)
                    except Exception:
                        manifest_data = {}
                    mcs_path = manifest_data.get("output_mcs") or os.path.join(output_dir, case_id + ".mcs")
                    fingerprint_path = runtime_path(output_dir, "fingerprints", safe_case_filename(case_id) + ".fingerprint")
                    current_fp = manifest_data.get("source_fingerprint", "")
                    if os.path.isfile(mcs_path):
                        stored_fp = ""
                        if os.path.isfile(fingerprint_path):
                            try:
                                with open(fingerprint_path, "r") as fp_handle:
                                    stored_fp = fp_handle.read().strip()
                            except Exception:
                                pass
                        if current_fp and stored_fp == current_fp:
                            # Source unchanged — skip
                            try:
                                shutil.rmtree(work_dir, ignore_errors=True)
                            except Exception:
                                pass
                            prune_empty_work_parents(work_dir)
                            if descriptor_path:
                                try:
                                    os.remove(descriptor_path)
                                except OSError:
                                    pass
                            continue
                        # Fingerprint changed or missing — reprocess
                        if current_fp:
                            log_message(output_dir, "Source fingerprint changed for {0}; reprocessing.".format(case_id))
                    if os.path.isfile(failed_marker) and not current_fp:
                        if descriptor_path:
                            try:
                                os.remove(descriptor_path)
                            except OSError:
                                pass
                        continue
                    filtered_work_dirs.append((case_id, work_dir, mcs_path, descriptor_path))

            work_dirs = filtered_work_dirs

            total = len(work_dirs)
            if total == 0:
                consecutive_empty += 1
                active = os.path.isfile(active_path)
                if active:
                    active_state = runtime_common.read_json(active_path, {}) or {}
                    try:
                        heartbeat = float(active_state.get("updated_at_epoch", os.path.getmtime(active_path)))
                    except Exception:
                        heartbeat = time.time()
                    if time.time() - heartbeat > producer_heartbeat_timeout:
                        active = False
                        log_message(output_dir, "Producer heartbeat is stale; releasing background Mimics.")
                        try:
                            os.remove(active_path)
                        except OSError:
                            pass
                done = os.path.isfile(done_path)
                stop = os.path.isfile(stop_path)
                if stop:
                    cancelled = True
                    log_message(output_dir, "Stop marker found; exiting.")
                    break
                if done and not active:
                    log_message(output_dir, "No pending manifests and producer marked the queue done; exiting.")
                    break
                if not active and time.time() - last_activity >= idle_timeout:
                    log_message(output_dir, "No active producer and idle timeout reached; exiting.")
                    break
                log_message(output_dir, "No pending manifests; waiting for new work.")
                update_status(
                    output_dir,
                    "idle",
                    completed=total_completed,
                    failed=total_failed,
                    consecutive_empty=consecutive_empty,
                )
                time.sleep(poll_seconds)
                continue

            consecutive_empty = 0
            log_message(output_dir, "Creating {} .mcs file(s).".format(total))

            # Re-check stop marker before starting work
            if os.path.isfile(stop_path):
                cancelled = True
                log_message(output_dir, "Stop marker found; exiting.")
                break

            for i, (case_id, work_dir, mcs_path, descriptor_path) in enumerate(work_dirs):
                if os.path.isfile(stop_path):
                    cancelled = True
                    log_message(
                        output_dir,
                        "Stop marker found before case {}; no additional projects will be created.".format(
                            case_id
                        ),
                    )
                    break
                log_message(output_dir, "[{}/{}] Creating: {}".format(i + 1, total, case_id))
                update_status(
                    output_dir,
                    "creating",
                    case_id=case_id,
                    completed=total_completed,
                    failed=total_failed,
                )
                staging_mcs = _staging_mcs_path(mcs_path)
                marker_path = runtime_path(output_dir, CURRENT_CASE_FILE)
                try:
                    _remove_file(staging_mcs)
                    marker_path = _write_current_case(
                        output_dir,
                        case_id,
                        work_dir,
                        mcs_path,
                        staging_mcs,
                        descriptor_path,
                        total_completed,
                        total_failed,
                    )
                    created_manifest = create_mcs_from_manifest(
                        work_dir, staging_mcs
                    )
                    _publish_mcs(staging_mcs, mcs_path)
                    total_completed += 1
                    last_activity = time.time()
                    log_message(output_dir, "Created: {}".format(mcs_path))
                    # Persist source fingerprint for incremental rebuild detection
                    manifest_data = (
                        created_manifest
                        if isinstance(created_manifest, dict)
                        else {}
                    )
                    if not manifest_data:
                        try:
                            with open(
                                os.path.join(work_dir, "prepare_manifest.json"),
                                "r",
                                encoding="utf-8",
                            ) as mf:
                                manifest_data = json.load(mf)
                        except Exception:
                            pass
                    fingerprint = manifest_data.get("source_fingerprint", "")
                    if fingerprint:
                        fingerprint_path = runtime_path(output_dir, "fingerprints", safe_case_filename(case_id) + ".fingerprint")
                        try:
                            fingerprint_dir = os.path.dirname(fingerprint_path)
                            if not os.path.isdir(fingerprint_dir):
                                os.makedirs(fingerprint_dir)
                            with open(fingerprint_path, "w", encoding="utf-8") as fp:
                                fp.write(fingerprint)
                        except Exception:
                            pass
                    try:
                        _record_created_project(
                            output_dir, case_id, mcs_path, manifest_data
                        )
                    except Exception as manifest_exc:
                        # The .mcs is already valid. Keep it, but make the
                        # missing provenance visible for later repair.
                        log_message(
                            output_dir,
                            "Created {0}, but dataset manifest update failed: {1}".format(
                                case_id, manifest_exc
                            ),
                        )
                    # Keep the manifest until its fingerprint is persisted.
                    # Deleting the work directory inside create_mcs_from_manifest
                    # made every later import look changed and reconvert the case.
                    try:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    except Exception:
                        pass
                    prune_empty_work_parents(work_dir)
                    if descriptor_path:
                        try:
                            os.remove(descriptor_path)
                        except OSError:
                            pass
                except Exception as e:
                    total_failed += 1
                    last_activity = time.time()
                    log_message(output_dir, "Create failed for {}: {}".format(case_id, e))
                    traceback_text = traceback.format_exc()
                    record_failed_case(output_dir, case_id, "create_mcs", e, traceback_text)
                    try:
                        write_json_atomic(
                            os.path.join(work_dir, "create_failed.json"),
                            {
                                "case_id": case_id,
                                "error": str(e),
                                "traceback": traceback_text,
                                "failed_at_epoch": time.time(),
                            },
                        )
                    except Exception:
                        pass
                    print(traceback_text)
                    try:
                        mimics.file.close_project()
                    except Exception:
                        pass
                    _remove_file(staging_mcs)
                    try:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    except Exception:
                        pass
                    prune_empty_work_parents(work_dir)
                    if descriptor_path:
                        try:
                            os.remove(descriptor_path)
                        except OSError:
                            pass
                finally:
                    _clear_current_case(marker_path, case_id)

        log_message(
            output_dir,
            "Background creation finished: {} succeeded, {} failed.".format(
                total_completed,
                total_failed,
            ),
        )
        update_status(
            output_dir,
            "cancelled" if cancelled else "closed",
            completed=total_completed,
            failed=total_failed,
        )
        return 0
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        raise
