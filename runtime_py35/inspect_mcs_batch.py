# -*- coding: utf-8 -*-
"""Read-only inspection of saved MCS projects (runs inside background Mimics).

For every case listed in the job config this script opens the .mcs, reads the
active image geometry (logical shape + voxel-to-RAS matrix), lists every Mask
name with its foreground voxel count, and writes one JSON report.  Nothing is
modified: the project is opened read-only in spirit (opened then closed
without saving) and no output .mcs is written.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only.
"""

from __future__ import print_function

import json
import os
import sys
import time
import traceback

import mimics

import mask_import
import runtime_common

MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = (
    "mimics_script.mimics_voxel_to_ras_matrix"
)
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = (
    "mimics_script.source_voxel_to_ras_matrix"
)


def _read_json(path, default=None):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json_atomic(path, payload):
    runtime_common.write_json_atomic(path, payload)


def _matrix_from_metadata(value):
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list) and len(parsed) == 4:
            return parsed
    except Exception:
        pass
    return None


def _image_info(image):
    """Return (shape, mimics_matrix, source_shape, source_matrix) for an image."""
    shape = mask_import._active_image_shape(image)
    matrix = None
    matrix_source = ""
    meta_matrix = _matrix_from_metadata(
        mask_import._metadata_get(image, MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, "")
    )
    if meta_matrix is not None:
        matrix = meta_matrix
        matrix_source = "image_metadata"
    else:
        derived = mask_import._derive_mimics_voxel_to_ras_matrix(image, shape)
        if derived is not None:
            matrix = derived
            matrix_source = "voxel_centers"
    source_shape = mask_import._metadata_get(image, SOURCE_IMAGE_SHAPE_METADATA, "")
    try:
        source_shape = [int(v) for v in json.loads(str(source_shape))]
    except Exception:
        source_shape = None
    source_matrix = _matrix_from_metadata(
        mask_import._metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, "")
    )
    return shape, matrix, matrix_source, source_shape, source_matrix


def _mask_foreground(mask):
    """Count foreground voxels via get_voxel_buffer, with sum() fallback."""
    try:
        buf = mask.get_voxel_buffer()
    except Exception:
        return None
    try:
        return int(buf.sum())
    except Exception:
        pass
    try:
        import numpy
        return int(numpy.asarray(buf).sum())
    except Exception:
        pass
    try:
        data = bytes(buf)
        return sum(1 for b in data if b)
    except Exception:
        return None


def _safe_close_project():
    try:
        mimics.file.close_project()
    except Exception:
        pass


def inspect_one(mcs_path):
    _safe_close_project()
    mimics.file.open_project(mcs_path)
    try:
        image = mimics.data.images.get_active()
    except Exception:
        image = None
    if image is None:
        try:
            image = mimics.data.masks[0].image if len(mimics.data.masks) else None
        except Exception:
            image = None
    shape, matrix, matrix_source, source_shape, source_matrix = (
        _image_info(image) if image is not None
        else (None, None, "", None, None)
    )
    source_image_path = (
        mask_import._metadata_get(image, SOURCE_IMAGE_PATH_METADATA, "")
        if image is not None else ""
    )
    masks = []
    for a_mask in mimics.data.masks:
        masks.append({
            "name": str(getattr(a_mask, "name", "") or ""),
            "foreground_voxels": _mask_foreground(a_mask),
        })
    return {
        "shape": shape,
        "voxel_to_ras_matrix": matrix,
        "voxel_to_ras_matrix_source": matrix_source,
        "source_image_path": str(source_image_path or ""),
        "source_shape": source_shape,
        "source_voxel_to_ras_matrix": source_matrix,
        "masks": masks,
    }


def run_job(config_path):
    config = _read_json(config_path, {}) or {}
    job_dir = os.path.abspath(config.get("job_dir") or os.path.dirname(config_path))
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)
    report_path = os.path.abspath(
        config.get("report_path") or os.path.join(job_dir, "inspect_report.json")
    )
    cases = config.get("cases") or []
    total = len(cases)
    completed = 0
    failed = 0
    results = {}
    errors = {}
    _write_json_atomic(report_path, {
        "status": "running", "completed": 0, "failed": 0, "total": total,
        "results": results, "errors": errors,
    })
    print("Inspect job started: {0} case(s).".format(total))
    try:
        for index, case in enumerate(cases):
            case_id = str(case.get("case_id", ""))
            mcs_path = str(case.get("mcs_path", ""))
            print("[{0}/{1}] {2}".format(index + 1, total, case_id))
            try:
                results[case_id] = inspect_one(mcs_path)
                completed += 1
            except Exception as exc:
                failed += 1
                errors[case_id] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print("{0}: FAILED: {1}".format(case_id, exc))
                _safe_close_project()
            if index % 5 == 0 or index + 1 == total:
                _write_json_atomic(report_path, {
                    "status": "running",
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "results": results,
                    "errors": errors,
                })
    finally:
        _safe_close_project()
    _write_json_atomic(report_path, {
        "status": "completed" if failed == 0 else "completed_with_errors",
        "completed": completed,
        "failed": failed,
        "total": total,
        "results": results,
        "errors": errors,
    })
    print(
        "Inspect finished: completed={0}, failed={1}, report={2}".format(
            completed, failed, report_path
        )
    )
    return 0 if failed == 0 else 1


def main(config_path=None):
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: inspect_mcs_batch.py <job_config.json>")
            return 2
        config_path = sys.argv[1]
    try:
        return run_job(config_path)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
