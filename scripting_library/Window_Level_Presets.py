# -*- coding: utf-8 -*-
"""Apply CT window/level presets for annotation."""

from __future__ import print_function

import sys

import mimics


PRESETS = [
    ("Lung", 1500, -600, ("lung", "airway", "trachea", "bronch")),
    ("Soft Tissue", 400, 40, ("soft", "muscle", "organ", "kidney", "spleen", "pancreas")),
    ("Liver", 150, 60, ("liver", "hepatic")),
    ("Brain", 80, 40, ("brain", "tumor", "lesion")),
    ("Vessel Contrast", 700, 250, ("vessel", "artery", "vein", "aorta", "coronary")),
    ("Bone", 1800, 400, ("bone", "skull", "vertebra", "rib", "femur", "pelvis")),
]


def _selected_mask_name():
    try:
        for mask in mimics.data.masks:
            if bool(getattr(mask, "selected", False)):
                return str(getattr(mask, "name", ""))
    except Exception:
        pass
    return ""


def _preset_for_name(name):
    lowered = str(name or "").lower()
    for preset in PRESETS:
        for keyword in preset[3]:
            if keyword in lowered:
                return preset
    return None


def _hu_to_gv(value):
    try:
        return mimics.segment.HU2GV(float(value))
    except Exception:
        return float(value)


def _apply_window(width, level):
    low_hu = float(level) - float(width) / 2.0
    high_hu = float(level) + float(width) / 2.0
    low_gv = _hu_to_gv(low_hu)
    high_gv = _hu_to_gv(high_hu)
    mimics.view.set_contrast((int(round(low_gv)), 0.0), (int(round(high_gv)), 1.0))


def main():
    selected_name = _selected_mask_name()
    suggested = _preset_for_name(selected_name)
    buttons = [preset[0] for preset in PRESETS]
    buttons.append("Cancel")
    default_name = suggested[0] if suggested else PRESETS[1][0]
    message = "Select a CT window preset."
    if selected_name:
        message += "\nSelected Mask: {0}".format(selected_name)
    if suggested:
        message += "\nSuggested: {0}".format(default_name)
    answer = mimics.dialogs.question_box(
        title="Window Presets",
        message=message,
        buttons=";".join(buttons),
        ui_blocking=True,
    )
    if not answer or answer == "Cancel":
        return 0
    preset = None
    for item in PRESETS:
        if item[0] == answer:
            preset = item
            break
    if preset is None:
        return 0
    _apply_window(preset[1], preset[2])
    try:
        mimics.dialogs.message_box(
            title="Window Presets",
            message="Applied {0}: WW={1}, WL={2}".format(preset[0], preset[1], preset[2]),
            ui_blocking=False,
        )
    except TypeError:
        pass
    return 0


if __name__ == "__main__":
    main()
elif "mimics" in sys.modules:
    main()
