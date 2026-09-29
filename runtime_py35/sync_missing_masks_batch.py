# -*- coding: utf-8 -*-
"""Per-case check-and-append of missing named Masks (inside background Mimics).

For each case in the job config the script opens the .mcs, lists the existing
Mask names, and - immediately, before moving to the next case - appends every
missing source Mask that has foreground voxels in the source NIfTI.  Each case
is checked and completed in one open/append/save cycle, so a case is fully
done when the loop reaches it (no scan-everything-then-append pass).

Cases already containing all source masks are skipped without re-saving.
Source masks that are entirely empty in the source NIfTI are appended too
(consistent with the original import, which kept empty masks such as a
missing brain); their zero foreground is visible in the job results.

If the saved project already holds a mask whose name collides with a missing
source mask name, the case fails loudly instead of silently re-importing.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only.
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
LOG_FILE = "sync_missing_masks.log"


def _read_json(path, default=None):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path, payload):
    runtime_common.write_json_atomic(path, payload)


def _log(job_dir, message):
    text = "[{0}] {1}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), str(message))
    try:
        with open(os.path.join(job_dir, LOG_FILE), "a") as handle:
            handle.write(text)
    except Exception:
        pass
    print(text.rstrip())


def _status(job_dir, status, phase, completed, failed, skipped, total, **details):
    payload = {
        "status": str(status),
        "phase": str(phase),
        "pid": os.getpid(),
        "completed": int(completed),
        "failed": int(failed),
        "skipped": int(skipped),
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
    for item in mimics.data.masks:
        names.add(str(getattr(item, "name", "") or ""))
    return names


def _mask_foreground(mask):
    try:
        buf = mask.get_voxel_buffer()
        return int(buf.sum())
    except Exception:
        pass
    try:
        import numpy
        return int(numpy.asarray(buf).sum())
    except Exception:
        return None


def _prepare_masks(mask_specs, case_id, active_shape, active_affine, work_dir):
    """Resample source NIfTI masks onto the active Mimics grid (bridge call)."""
    buffers_dir = os.path.join(work_dir, "buffers")
    if not os.path.isdir(buffers_dir):
        os.makedirs(buffers_dir)
    result = mask_import._call_bridge(
        {
            "action": "prepare_masks_for_grid",
            "case_id": case_id,
            "masks": mask_specs,
            "target_shape": list(active_shape),
            "target_voxel_to_ras_matrix": active_affine,
            "buffers_out": buffers_dir,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        }
    )
    prepared = result.get("masks") or []
    if len(prepared) != len(mask_specs):
        raise RuntimeError(
            "Expected {0} prepared masks, got {1}. Multi-label inputs are not "
            "accepted by this sync operation.".format(len(mask_specs), len(prepared))
        )
    for index, item in enumerate(prepared):
        if str(item.get("name") or "") != mask_specs[index]["name"]:
            raise RuntimeError(
                "Mask '{0}' produced unexpected prepared name '{1}'.".format(
                    mask_specs[index]["name"], item.get("name")
                )
            )
        if not os.path.isfile(item.get("u8_path", "")):
            raise RuntimeError(
                "Prepared buffer is missing for {0}.".format(mask_specs[index]["name"])
            )
    return prepared


def _scratch_root(job_dir):
    """Per-case scratch location.

    Batches of 5 buffers peak at ~1.5 GB; the launcher sets
    MIMICS_SYNC_SCRATCH_ROOT when a drive with that much headroom is
    available, otherwise buffers land beside the job dir.  Wiped per case
    either way.
    """
    output_root = os.environ.get("MIMICS_SYNC_SCRATCH_ROOT", "").strip()
    if output_root:
        return output_root
    return os.path.join(job_dir, "work")


def _sync_one(case, job_dir):
    """Check one case and append its missing masks immediately."""
    case_id = str(case.get("case_id", ""))
    source_masks = list(case.get("source_masks") or [])
    if not source_masks:
        raise RuntimeError("The case has no source masks configured.")
    source_mcs = os.path.abspath(case["mcs_path"])
    output_mcs = os.path.abspath(case["output_mcs_path"])
    if not os.path.isfile(source_mcs):
        raise RuntimeError("Source MCS not found: {0}".format(source_mcs))
    # In output-dir mode the previous output (if any) holds a superset of the
    # source masks; open it so already-synced cases skip without re-appending.
    open_path = (
        output_mcs
        if output_mcs != source_mcs and os.path.isfile(output_mcs)
        else source_mcs
    )

    _safe_close_project()
    mimics.file.open_project(open_path)
    opened = True
    staging_mcs = _staging_path(output_mcs)
    _remove_file(staging_mcs)
    work_dir = os.path.join(_scratch_root(job_dir), case_id or "case")
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)

    try:
        return _sync_opened_case(
            case, job_dir, case_id, source_masks, output_mcs, staging_mcs, work_dir
        )
    finally:
        if opened:
            _safe_close_project()
        _remove_file(staging_mcs)
        shutil.rmtree(work_dir, ignore_errors=True)


def _sync_opened_case(case, job_dir, case_id, source_masks, output_mcs, staging_mcs, work_dir):
    existing = _mask_names()
    missing = []
    for spec in source_masks:
        name = str(spec["name"])
        if name in existing:
            continue
        missing.append({"name": name, "mask_path": os.path.abspath(spec["mask_path"])})
    summary = {
        "case_id": case_id,
        "existing_masks": sorted(existing),
        "missing_masks": [item["name"] for item in missing],
    }
    if not missing:
        _log(
            job_dir,
            "{0}: complete already ({1} mask(s))".format(case_id, len(existing)),
        )
        return dict(summary, status="skipped")

    active_image, active_shape, active_affine = mask_import._active_image_info()
    if active_image is None or not active_shape or not active_affine:
        raise RuntimeError("Could not determine the active Mimics image grid.")

    appended = {}
    # Prepare masks in small batches: prepare -> inject -> delete buffers.
    # Peak scratch stays at a few buffers instead of one per missing mask,
    # which keeps large-grid cases (~250 MB per buffer, 47+ missing masks)
    # within the free space of the scratch drive.
    batch_size = 5
    gui_was_enabled = True
    try:
        mimics.disable_update_gui()
    except Exception:
        gui_was_enabled = False
    try:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            prepared = _prepare_masks(
                batch, case_id, active_shape, active_affine, work_dir
            )
            for item in prepared:
                name = str(item["name"])
                if name in _mask_names():
                    raise RuntimeError(
                        "Target Mask name already exists mid-append: {0}".format(name)
                    )
                mask = mimics.segment.create_mask()
                mask.name = name
                mask_import._inject_buffer(
                    mask,
                    item["u8_path"],
                    item["mimics_shape"],
                    use_transaction=False,
                )
                try:
                    mask.visible = False
                except Exception:
                    pass
                _log(
                    job_dir,
                    "{0}: appended {1} (foreground_voxels={2})".format(
                        case_id, name, item.get("foreground_voxels", 0)
                    ),
                )
                appended[name] = int(item.get("foreground_voxels", 0) or 0)
                _remove_file(item["u8_path"])
    finally:
        if gui_was_enabled:
            try:
                mimics.enable_update_gui()
            except Exception:
                pass

    output_dir = os.path.dirname(output_mcs)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    mimics.file.save_project(
        filename=staging_mcs,
        save_as_type="Mimics Project Files",
    )
    if not os.path.isfile(staging_mcs):
        raise RuntimeError("Mimics did not create the temporary MCS.")
    _safe_close_project()
    _publish(staging_mcs, output_mcs)
    _log(job_dir, "{0}: published {1}".format(case_id, output_mcs))
    return dict(
        summary,
        status="completed",
        appended=appended,
        output_mcs=output_mcs,
    )


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
    results = []
    failed_rows = []
    _status(job_dir, "running", "starting", 0, 0, 0, total)
    _log(job_dir, "Sync job started: {0} case(s).".format(total))
    try:
        for index, case in enumerate(cases):
            stop_path = config.get("stop_path") or ""
            if stop_path and os.path.isfile(stop_path):
                _status(
                    job_dir, "cancelled", "stopped", completed, failed, skipped, total,
                    next_case=case.get("case_id", ""),
                )
                _log(job_dir, "Stop requested; remaining cases were not opened.")
                return 0
            case_id = str(case.get("case_id", ""))
            _status(
                job_dir, "running", "syncing", completed, failed, skipped, total,
                case_id=case_id, index=index + 1,
            )
            try:
                result = _sync_one(case, job_dir)
                results.append(result)
                if result.get("status") == "skipped":
                    skipped += 1
                else:
                    completed += 1
            except Exception as exc:
                failed += 1
                detail = {
                    "case_id": case_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_rows.append(detail)
                _log(job_dir, "{0}: FAILED: {1}".format(case_id, exc))
                _safe_close_project()
            _status(
                job_dir, "running", "syncing", completed, failed, skipped, total,
                case_id=case_id, index=index + 1,
            )
    finally:
        _safe_close_project()

    final_status = "completed" if failed == 0 else "completed_with_errors"
    _write_json(os.path.join(job_dir, "failed_cases.json"), failed_rows)
    _write_json(os.path.join(job_dir, "results.json"), results)
    _status(
        job_dir, final_status, "finished", completed, failed, skipped, total,
        failed_cases=failed_rows[:100],
    )
    _log(
        job_dir,
        "Sync job finished: completed={0}, skipped={1}, failed={2}.".format(
            completed, skipped, failed
        ),
    )
    return 0 if failed == 0 else 1


def main(config_path=None):
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: sync_missing_masks_batch.py <job_config.json>")
            return 2
        config_path = sys.argv[1]
    try:
        return run_job(config_path)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
