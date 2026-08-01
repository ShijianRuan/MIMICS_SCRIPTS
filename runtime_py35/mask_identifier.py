# -*- coding: utf-8 -*-
"""Identify which mask(s) contain the voxel under the cursor.

Allows the user to click on any 2D/3D view and see which masks contain
that voxel, along with the mask name, visibility status, and color.
"""

from __future__ import print_function

import os
import logging
import time

import mimics

import runtime_common


MAX_MASKS_PER_CLICK = int(os.environ.get("MIMICS_MASK_IDENTIFIER_MAX_MASKS_PER_CLICK", "0"))
MAX_SECONDS_PER_CLICK = float(os.environ.get("MIMICS_MASK_IDENTIFIER_MAX_SECONDS_PER_CLICK", "0"))
MAX_CACHED_BUFFERS = int(os.environ.get("MIMICS_MASK_IDENTIFIER_MAX_CACHED_BUFFERS", "4"))
USE_BOUNDING_BOX_FILTER = os.environ.get("MIMICS_MASK_IDENTIFIER_USE_BBOX", "1").strip().lower() not in ("0", "false", "no")
ALLOW_PARTIAL_SCAN = os.environ.get("MIMICS_MASK_IDENTIFIER_ALLOW_PARTIAL", "").strip().lower() in ("1", "true", "yes")
VISIBLE_ONLY = os.environ.get("MIMICS_MASK_IDENTIFIER_VISIBLE_ONLY", "").strip().lower() in ("1", "true", "yes")


def _mimics_log(message):
    try:
        mimics.logging.log_user_message(level=logging.INFO, message=message)
    except Exception:
        try:
            print(message)
        except Exception:
            pass


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _mask_key(mask):
    guid = str(getattr(mask, "guid", "") or "")
    if guid:
        return guid
    return str(id(mask))


def _mask_name(mask):
    return str(getattr(mask, "name", "") or "Mask")


def _mask_color(mask):
    try:
        return tuple(getattr(mask, "color"))
    except Exception:
        return (0, 0, 0)


def _mask_visible(mask):
    try:
        return bool(getattr(mask, "visible", False))
    except Exception:
        return False


def _mask_selected(mask):
    try:
        return bool(getattr(mask, "selected", False))
    except Exception:
        return False


def _mask_pixel_count(mask):
    try:
        return int(getattr(mask, "number_of_pixels", -1))
    except Exception:
        return -1


def _same_image(mask, active_image):
    try:
        image = getattr(mask, "image", None)
        return image is None or image is active_image
    except Exception:
        return True


def _collect_candidates(active_image):
    try:
        masks = list(mimics.data.masks)
    except Exception:
        return []
    rows = []
    for mask in masks:
        if not _same_image(mask, active_image):
            continue
        pixel_count = _mask_pixel_count(mask)
        if pixel_count == 0:
            continue
        visible = _mask_visible(mask)
        selected = _mask_selected(mask)
        if VISIBLE_ONLY and not (visible or selected):
            continue
        rows.append({
            "mask": mask,
            "name": _mask_name(mask),
            "visible": visible,
            "selected": selected,
            "color": _mask_color(mask),
            "pixel_count": pixel_count,
        })
    rows.sort(key=lambda row: (
        not row["selected"],
        not row["visible"],
        row["name"].lower(),
    ))
    return rows


def _mask_count():
    try:
        return len(list(mimics.data.masks))
    except Exception:
        return 0


def _vec3(value):
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [float(getattr(value, name)) for name in names]
        except Exception:
            pass
    for attr in ("coordinates", "coordinate", "point", "position"):
        try:
            candidate = getattr(value, attr)
            return [float(candidate[0]), float(candidate[1]), float(candidate[2])]
        except Exception:
            pass
    return [float(value[0]), float(value[1]), float(value[2])]


def _sub(a, b):
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def _det(a, b, c):
    return (
        float(a[0]) * (float(b[1]) * float(c[2]) - float(b[2]) * float(c[1]))
        - float(a[1]) * (float(b[0]) * float(c[2]) - float(b[2]) * float(c[0]))
        + float(a[2]) * (float(b[0]) * float(c[1]) - float(b[1]) * float(c[0]))
    )


def _bbox_contains_point(bbox, point, margin=1e-4):
    origin = _vec3(getattr(bbox, "origin"))
    first = _vec3(getattr(bbox, "first_vector"))
    second = _vec3(getattr(bbox, "second_vector"))
    third = _vec3(getattr(bbox, "third_vector"))
    rel = _sub(_vec3(point), origin)
    denom = _det(first, second, third)
    if abs(denom) < 1e-12:
        return True
    a = _det(rel, second, third) / denom
    b = _det(first, rel, third) / denom
    c = _det(first, second, rel) / denom
    return (
        -margin <= a <= 1.0 + margin
        and -margin <= b <= 1.0 + margin
        and -margin <= c <= 1.0 + margin
    )


def _get_cached_bbox(row, bbox_cache):
    key = _mask_key(row["mask"])
    if key in bbox_cache:
        return bbox_cache[key]
    try:
        bbox = mimics.measure.get_bounding_box(row["mask"])
        info = {"available": True, "bbox": bbox}
    except Exception as exc:
        info = {"available": False, "error": str(exc)}
    bbox_cache[key] = info
    _update_gui()
    return info


def _buffer_value(buffer, ix, iy, iz):
    try:
        return bool(buffer[ix, iy, iz])
    except Exception:
        pass
    try:
        shape = tuple(int(value) for value in buffer.shape)
        offset = (ix * shape[1] * shape[2]) + (iy * shape[2]) + iz
        if hasattr(buffer, "tobytes"):
            raw = buffer.tobytes()
        else:
            raw = buffer.tostring()
        return bool(raw[offset])
    except Exception:
        return False


def _remember_cache(cache, cache_order, key, value):
    cache[key] = value
    if key in cache_order:
        cache_order.remove(key)
    cache_order.append(key)
    while len(cache_order) > MAX_CACHED_BUFFERS:
        old_key = cache_order.pop(0)
        cache.pop(old_key, None)


def _get_cached_buffer(row, cache, cache_order):
    key = _mask_key(row["mask"])
    if key in cache:
        return cache[key]
    _update_gui()
    vb = row["mask"].get_voxel_buffer()
    info = {
        "buffer": vb,
        "shape": tuple(int(value) for value in getattr(vb, "shape", ())),
    }
    _remember_cache(cache, cache_order, key, info)
    _update_gui()
    return info


def _scan_point(candidates, point, ix, iy, iz, bbox_cache, cache, cache_order):
    found = []
    skipped = []
    unread = 0
    checked = 0
    bbox_checked = 0
    bbox_skipped = 0
    bbox_unavailable = 0
    started = time.time()
    for row in candidates:
        if ALLOW_PARTIAL_SCAN and MAX_MASKS_PER_CLICK > 0 and checked >= MAX_MASKS_PER_CLICK:
            unread += 1
            continue
        if ALLOW_PARTIAL_SCAN and MAX_SECONDS_PER_CLICK > 0 and checked > 0 and (time.time() - started) >= MAX_SECONDS_PER_CLICK:
            unread += 1
            continue
        if USE_BOUNDING_BOX_FILTER:
            bbox_info = _get_cached_bbox(row, bbox_cache)
            if bbox_info.get("available"):
                bbox_checked += 1
                try:
                    if not _bbox_contains_point(bbox_info["bbox"], point):
                        bbox_skipped += 1
                        continue
                except Exception:
                    bbox_unavailable += 1
            else:
                bbox_unavailable += 1
        try:
            info = _get_cached_buffer(row, cache, cache_order)
            checked += 1
            shape = info["shape"]
            if len(shape) != 3:
                skipped.append(row["name"])
                continue
            sx, sy, sz = shape
            if ix < 0 or ix >= sx or iy < 0 or iy >= sy or iz < 0 or iz >= sz:
                continue
            if _buffer_value(info["buffer"], ix, iy, iz):
                found.append(row)
        except Exception:
            skipped.append(row["name"])
    return {
        "found": found,
        "checked": checked,
        "bbox_checked": bbox_checked,
        "bbox_skipped": bbox_skipped,
        "bbox_unavailable": bbox_unavailable,
        "unread": unread,
        "skipped": skipped,
        "elapsed": time.time() - started,
    }


def main():
    # ── 1. Check prerequisites ──
    try:
        active_image = mimics.data.images.get_active()
    except Exception:
        active_image = None
    if active_image is None:
        mimics.dialogs.message_box(
            "No active image found.\nPlease open a project first."
        )
        return

    # Build only lightweight mask metadata here. Voxel buffers are read lazily
    # after a point is clicked, otherwise this entry can freeze Mimics on large
    # projects with many masks.
    candidates = _collect_candidates(active_image)
    if not candidates:
        if _mask_count() > 0:
            mimics.dialogs.message_box(
                "No non-empty masks are available to scan.\n"
                "If visible-only mode is enabled, disable MIMICS_MASK_IDENTIFIER_VISIBLE_ONLY."
            )
        else:
            mimics.dialogs.message_box("No masks in the current project.")
        return

    bbox_cache = {}
    cache = {}
    cache_order = []
    mode_note = ""
    if VISIBLE_ONLY:
        mode_note = " Visible-only mode is enabled."
    elif USE_BOUNDING_BOX_FILTER:
        mode_note = " Hidden masks are included; bounding-box filtering is enabled."
    else:
        mode_note = " Hidden masks are included; bounding-box filtering is disabled."
    _mimics_log(
        "Mask Identifier ready: scanning {} mask(s) on demand.{}".format(
            len(candidates),
            mode_note,
        )
    )

    # ── 4. Interactive loop ──
    # IMPORTANT: mimics.indicate_coordinate() is a modal, blocking call. While it
    # is waiting for a click, switching to another Mimics tool (zoom, pan,
    # measure, ...) corrupts Mimics' tool state machine and can crash the
    # application. To shrink that dangerous waiting window, we never stay inside
    # indicate_coordinate between clicks: each click result is shown via a
    # NON-blocking question_box, during which the user is free to switch tools
    # or adjust views. Only when the user explicitly chooses "Click Again" do we
    # re-enter indicate_coordinate for the brief click window.
    BUTTON_CLICK = "Click Again"
    BUTTON_FINISH = "Finish"

    # Pre-flight confirmation: let the user prepare the view (and switch tools)
    # before entering the first modal click.
    preflight = mimics.dialogs.question_box(
        title="Mask Identifier",
        message=(
            "Ready to identify masks.\n\n"
            "Click '{0}' then click a point in any view.\n\n"
            "Switch tools / adjust views only while this dialog is shown, "
            "not while the cursor is waiting for a click.".format(BUTTON_CLICK)
        ),
        buttons="{0};{1}".format(BUTTON_CLICK, BUTTON_FINISH),
        ui_blocking=False,
    )
    if preflight != BUTTON_CLICK:
        return

    while True:
        try:
            point = mimics.indicate_coordinate(
                message="Click to identify mask(s) at this point (Esc to finish)",
                show_message_box=False,
                confirm=False,
            )
        except mimics.UserInterrupted:
            break  # Esc pressed
        except Exception:
            break

        # Convert world coordinates → voxel index (x, y, z)
        try:
            idx = active_image.get_voxel_indexes(point)
        except ValueError:
            answer = mimics.dialogs.question_box(
                title="Mask Identifier",
                message=(
                    "Point ({:.1f}, {:.1f}, {:.1f}) mm is outside the image bounds.\n\n"
                    "Try another point?".format(*point)
                ),
                buttons="{0};{1}".format(BUTTON_CLICK, BUTTON_FINISH),
                ui_blocking=False,
            )
            if answer == BUTTON_CLICK:
                continue
            break
        except Exception:
            answer = mimics.dialogs.question_box(
                title="Mask Identifier",
                message="Could not compute voxel index for this point.\n\nTry another point?",
                buttons="{0};{1}".format(BUTTON_CLICK, BUTTON_FINISH),
                ui_blocking=False,
            )
            if answer == BUTTON_CLICK:
                continue
            break

        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])

        operation_token = runtime_common.try_acquire_local_operation(
            "mask_buffer_access", "Mask Identifier scan"
        )
        if not operation_token:
            owner = runtime_common.active_local_operation("mask_buffer_access") or {}
            answer = mimics.dialogs.question_box(
                title="Mask Identifier",
                message=(
                    "Mask buffers are currently used by {0}.\n\n"
                    "No Mask was scanned. Wait for that operation to finish, then try again."
                ).format(owner.get("owner") or "another Mimics-Script task"),
                buttons="{0};{1}".format(BUTTON_CLICK, BUTTON_FINISH),
                ui_blocking=False,
            )
            if answer == BUTTON_CLICK:
                continue
            break
        try:
            result = _scan_point(
                candidates, point, ix, iy, iz, bbox_cache, cache, cache_order
            )
        finally:
            runtime_common.release_local_operation(
                "mask_buffer_access", operation_token
            )
        found = result["found"]

        # ── 5. Build a concise result message ──
        if found:
            lines = []
            for info in found:
                state = "hidden" if not info["visible"] else "visible"
                lines.append("{0}  ({1})".format(info["name"], state))
            message_body = "\n".join(lines)
        else:
            message_body = "No mask at this point."

        processed = int(result.get("checked", 0)) + int(result.get("bbox_skipped", 0))
        summary = "Checked {0}/{1} masks in {2:.2f}s; read {3} voxel buffer(s).".format(
            processed,
            len(candidates),
            float(result.get("elapsed", 0.0)),
            int(result.get("checked", 0)),
        )
        notes = []
        if result.get("unread"):
            notes.append("{0} not scanned due to response limits".format(result.get("unread")))
        if result.get("skipped"):
            notes.append("{0} could not be read".format(len(result.get("skipped"))))
        if notes:
            summary += " " + "; ".join(notes) + "."
        message_body += "\n\n" + summary

        answer = mimics.dialogs.question_box(
            title="Mask Identifier",
            message=message_body,
            buttons="{0};{1}".format(BUTTON_CLICK, BUTTON_FINISH),
            ui_blocking=False,
        )
        if answer != BUTTON_CLICK:
            break
