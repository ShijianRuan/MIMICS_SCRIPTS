# -*- coding: utf-8 -*-
"""Append arbitrary named NIfTI Masks to existing Mimics projects.

Runs inside a background Mimics Python process.  Each job case contains an
existing MCS path, an output MCS path, and one or more named mask files.  The
source project is opened, all named Masks are mapped to the active image grid,
injected into the project, and saved to the separate output path.

The job JSON is produced by ``tools/mimics_batch_cli.py append-masks``.
"""

from __future__ import print_function

import json
import os
import shutil
import sys
import time
import traceback

import mimics

import mask_import
import runtime_common


STATUS_FILE = "status.json"
LOG_FILE = "append.log"


def _read_json(path, default=None):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path, payload):
    runtime_common.write_json_atomic(path, payload)


def _log(job_dir, message):
    text = "[{0}] {1}\n".format(
        time.strftime("%Y-%m-%d %H:%M:%S"), str(message)
    )
    try:
        with open(os.path.join(job_dir, LOG_FILE), "a") as handle:
            handle.write(text)
    except Exception:
        pass
    print(text.rstrip())


def _status(job_dir, status, phase, completed, failed, total, **details):
    payload = {
        "status": str(status),
        "phase": str(phase),
        "pid": os.getpid(),
        "completed": int(completed),
        "failed": int(failed),
        "total": int(total),
        "updated_at_epoch": time.time(),
    }
    payload.update(details)
    try:
        _write_json(os.path.join(job_dir, STATUS_FILE), payload)
    except Exception:
        pass


def _safe_close_project():
    try:
        mimics.file.close_project()
    except Exception:
        pass


def _remove_file(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _staging_path(output_path):
    stem = output_path[:-4] if output_path.lower().endswith(".mcs") else output_path
    return "{0}.creating.{1}.mcs".format(stem, os.getpid())


def _publish(staging_path, output_path):
    if not os.path.isfile(staging_path):
        raise RuntimeError(
            "Mimics saved successfully but the temporary MCS was not found: {0}".format(
                staging_path
            )
        )
    last_error = None
    for attempt in range(20):
        try:
            os.replace(staging_path, output_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.5, 0.05 * (attempt + 1)))
    raise RuntimeError(
        "Could not publish the completed MCS to {0}: {1}".format(
            output_path, last_error
        )
    )


def _mask_names():
    names = set()
    try:
        for item in mimics.data.masks:
            names.add(str(getattr(item, "name", "") or ""))
    except Exception as exc:
        raise RuntimeError("Could not inspect existing Masks: {0}".format(exc))
    return names


def _create_named_mask(name, active_image):
    if name in _mask_names():
        raise RuntimeError(
            "Mask name already exists in the source project: {0}. "
            "Refusing to create an automatic '- Imported' name.".format(name)
        )
    mask = mimics.segment.create_mask()
    mask.name = name
    try:
        bound_image = getattr(mask, "image", None)
    except Exception:
        bound_image = None
    if bound_image is not None:
        try:
            same_image = bound_image == active_image
        except Exception:
            same_image = bound_image is active_image
        if not same_image:
            raise RuntimeError(
                "Mimics created Mask '{}' on a different image.".format(name)
            )
    try:
        mask.visible = False
    except Exception:
        pass
    return mask


def _prepare_masks(case, active_shape, active_affine, work_dir):
    mask_specs = case.get("masks") or []
    if not mask_specs:
        raise RuntimeError("The case does not contain any mask specifications.")
    masks = []
    for item in mask_specs:
        name = str(item.get("name") or "").strip()
        path = os.path.abspath(str(item.get("mask_path") or ""))
        if not name:
            raise RuntimeError("A mask specification has an empty name.")
        if not os.path.isfile(path):
            raise RuntimeError("Mask file not found for {}: {}".format(name, path))
        masks.append({"name": name, "mask_path": path})

    buffers_dir = os.path.join(work_dir, "buffers")
    if not os.path.isdir(buffers_dir):
        os.makedirs(buffers_dir)
    result = mask_import._call_bridge(
        {
            "action": "prepare_masks_for_grid",
            "case_id": case.get("case_id", ""),
            "masks": masks,
            "target_shape": list(active_shape),
            "target_voxel_to_ras_matrix": active_affine,
            "buffers_out": buffers_dir,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        }
    )
    prepared = result.get("masks") or []
    expected = [item["name"] for item in masks]
    if len(prepared) != len(expected):
        raise RuntimeError(
            "Expected {} prepared masks, got {}. Multi-label inputs are not "
            "accepted by this named-mask append operation.".format(
                len(expected), len(prepared)
            )
        )
    for index, item in enumerate(prepared):
        if str(item.get("name") or "") != expected[index]:
            raise RuntimeError(
                "Mask '{}' produced unexpected prepared name '{}'. "
                "Only binary masks are accepted.".format(
                    expected[index], item.get("name")
                )
            )
        if not os.path.isfile(item.get("u8_path", "")):
            raise RuntimeError(
                "Prepared buffer is missing for {}.".format(expected[index])
            )
    return prepared


def _append_one(case, job_dir, force, in_place=False):
    source_mcs = os.path.abspath(case["mcs_path"])
    output_mcs = os.path.abspath(case["output_mcs_path"])
    if not os.path.isfile(source_mcs):
        raise RuntimeError("Source MCS not found: {}".format(source_mcs))
    same_mcs = os.path.normcase(source_mcs) == os.path.normcase(output_mcs)
    if same_mcs and not in_place:
        raise RuntimeError("Output MCS must differ from source MCS: {}".format(source_mcs))
    if not in_place and os.path.isfile(output_mcs) and not force:
        return {"status": "skipped", "reason": "output_exists", "output_mcs": output_mcs}

    output_dir = os.path.dirname(output_mcs)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    staging_mcs = _staging_path(output_mcs)
    _remove_file(staging_mcs)
    work_dir = os.path.join(job_dir, "work", str(case.get("case_id", "case")))
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)

    opened = False
    try:
        _safe_close_project()
        mimics.file.open_project(source_mcs)
        opened = True
        active_image, active_shape, active_affine = mask_import._active_image_info()
        if active_image is None or not active_shape or not active_affine:
            raise RuntimeError("Could not determine the active Mimics image grid.")

        prepared = _prepare_masks(case, active_shape, active_affine, work_dir)
        existing = _mask_names()
        collisions = [
            str(item.get("name")) for item in prepared
            if str(item.get("name")) in existing
        ]
        if collisions:
            raise RuntimeError(
                "Target Mask name(s) already exist: {}".format(
                    ", ".join(collisions)
                )
            )

        gui_was_enabled = True
        try:
            mimics.disable_update_gui()
        except Exception:
            gui_was_enabled = False
        try:
            for item in prepared:
                name = str(item["name"])
                mask = _create_named_mask(name, active_image)
                mask_import._inject_buffer(
                    mask,
                    item["u8_path"],
                    item["mimics_shape"],
                    use_transaction=False,
                )
                try:
                    mask.visible = True
                except Exception:
                    pass
                _log(
                    job_dir,
                    "{}: created {} (foreground_voxels={})".format(
                        case.get("case_id", ""),
                        name,
                        item.get("foreground_voxels", 0),
                    ),
                )
        finally:
            if gui_was_enabled:
                try:
                    mimics.enable_update_gui()
                except Exception:
                    pass

        mimics.file.save_project(
            filename=staging_mcs,
            save_as_type="Mimics Project Files",
        )
        if not os.path.isfile(staging_mcs):
            raise RuntimeError("Mimics did not create the temporary MCS.")
        _safe_close_project()
        opened = False
        _publish(staging_mcs, output_mcs)
        return {
            "status": "completed",
            "output_mcs": output_mcs,
            "active_shape": list(active_shape),
            "foreground_voxels": {
                str(item["name"]): int(item.get("foreground_voxels", 0) or 0)
                for item in prepared
            },
        }
    finally:
        if opened:
            _safe_close_project()
        _remove_file(staging_mcs)
        shutil.rmtree(work_dir, ignore_errors=True)


def run_job(config_path):
    config = _read_json(config_path, {}) or {}
    job_dir = os.path.abspath(config.get("job_dir") or os.path.dirname(config_path))
    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)
    cases = config.get("cases") or []
    total = len(cases)
    completed = 0
    failed = 0
    skipped = 0
    failed_rows = []
    _status(job_dir, "running", "starting", 0, 0, total, skipped=0)
    _log(job_dir, "Append job started: {} case(s).".format(total))
    try:
        for index, case in enumerate(cases):
            stop_path = config.get("stop_path") or ""
            if stop_path and os.path.isfile(stop_path):
                _status(
                    job_dir, "cancelled", "stopped", completed, failed, total,
                    skipped=skipped, next_case=case.get("case_id", ""),
                )
                _log(job_dir, "Stop requested; remaining cases were not opened.")
                return 0
            case_id = str(case.get("case_id", ""))
            _status(
                job_dir, "running", "appending", completed, failed, total,
                skipped=skipped, case_id=case_id, index=index + 1,
            )
            try:
                result = _append_one(
                    case,
                    job_dir,
                    bool(config.get("force", False)),
                    in_place=bool(config.get("in_place", False)),
                )
                if result.get("status") == "skipped":
                    skipped += 1
                    _log(job_dir, "{}: skipped ({})".format(case_id, result.get("reason")))
                else:
                    completed += 1
                    _log(job_dir, "{}: published {}".format(case_id, result.get("output_mcs")))
            except Exception as exc:
                failed += 1
                detail = {
                    "case_id": case_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_rows.append(detail)
                _log(job_dir, "{}: FAILED: {}".format(case_id, exc))
                _safe_close_project()
            _status(
                job_dir, "running", "appending", completed, failed, total,
                skipped=skipped, case_id=case_id, index=index + 1,
            )
    finally:
        _safe_close_project()

    final_status = "completed" if failed == 0 else "completed_with_errors"
    _write_json(os.path.join(job_dir, "failed_cases.json"), failed_rows)
    _status(
        job_dir, final_status, "finished", completed, failed, total,
        skipped=skipped, failed_cases=failed_rows[:100],
    )
    _log(
        job_dir,
        "Append job finished: completed={}, skipped={}, failed={}.".format(
            completed, skipped, failed
        ),
    )
    return 0 if failed == 0 else 1


def main(config_path=None):
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: append_masks_batch.py <job_config.json>")
            return 2
        config_path = sys.argv[1]
    try:
        return run_job(config_path)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
