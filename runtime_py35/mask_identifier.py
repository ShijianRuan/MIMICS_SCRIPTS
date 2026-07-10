# -*- coding: utf-8 -*-
"""Identify which mask(s) contain the voxel under the cursor.

Allows the user to click on any 2D/3D view and see which masks contain
that voxel, along with the mask name, visibility status, and color.
"""

from __future__ import print_function

import sys
import traceback


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

    # ── 2. Collect masks ──
    try:
        all_masks = list(mimics.data.masks)
    except Exception:
        all_masks = []
    if not all_masks:
        mimics.dialogs.message_box("No masks in the current project.")
        return

    # ── 3. Pre-cache voxel buffers & basic info ──
    mask_info = []
    for m in all_masks:
        try:
            vb = m.get_voxel_buffer()
            mask_info.append({
                "name": m.name,
                "visible": m.visible,
                "color": tuple(m.color),
                "vb": vb,
                "shape": tuple(vb.shape),
            })
        except Exception:
            pass  # skip masks that can't be read

    if not mask_info:
        mimics.dialogs.message_box("Could not read any mask voxel data.")
        return

    # Sort: visible masks first, then alphabetically
    mask_info.sort(key=lambda x: (not x["visible"], x["name"].lower()))

    # ── 4. Interactive loop ──
    while True:
        try:
            point = mimics.indicate_coordinate(
                message="Click to identify mask(s) at this point (Esc to exit)",
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
            mimics.dialogs.message_box(
                "Point ({:.1f}, {:.1f}, {:.1f}) mm\n"
                "is outside the image bounds.\nTry again.".format(*point)
            )
            continue
        except Exception:
            mimics.dialogs.message_box("Could not compute voxel index for this point.")
            continue

        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])

        # Check each cached mask
        found = []
        for info in mask_info:
            sx, sy, sz = info["shape"]
            if ix < 0 or ix >= sx or iy < 0 or iy >= sy or iz < 0 or iz >= sz:
                continue
            try:
                if info["vb"][ix, iy, iz]:
                    found.append(info)
            except Exception:
                pass

        # ── 5. Build result message ──
        parts = [
            "📍 Position: ({:.1f}, {:.1f}, {:.1f}) mm".format(*point),
            "🔢 Voxel index: ({}, {}, {})".format(ix, iy, iz),
            "",
        ]

        if found:
            parts.append("✅ Masks containing this voxel ({} found):\n".format(len(found)))
            for info in found:
                vis = "👁" if info["visible"] else "⊘"
                r, g, b = info["color"]
                parts.append("  {}  {}  (R={:.0f}  G={:.0f}  B={:.0f})".format(
                    vis, info["name"], r, g, b))
        else:
            parts.append("❌ This voxel is not inside any mask.")

        mimics.dialogs.message_box("\n".join(parts))
