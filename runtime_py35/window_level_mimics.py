# -*- coding: utf-8 -*-
"""CT window/level helpers for Mimics annotation workflows."""

from __future__ import print_function

import json
import logging
import os
import sys
import time

import mimics

import runtime_common


TITLE = "Window/Level Presets"
STATE_FILE = ".mimics_window_level_state.json"

DEFAULT_PRESETS = [
    {"name": "Lung", "width": 1500, "level": -600, "keywords": [
        "lung", "lung_upper", "lung_lower", "lung_middle", "lobe", "trachea", "bronch", "airway", "pulmonary",
    ]},
    {"name": "Abdomen / Soft Tissue", "width": 400, "level": 50, "keywords": [
        "spleen", "kidney", "gallbladder", "stomach", "pancreas", "adrenal", "suprarenal",
        "duodenum", "small_bowel", "small_intestine", "colon", "urinary_bladder", "bladder",
        "prostate", "esophagus", "ureter", "uterus", "ovary", "seminal", "rectum",
        "soft", "muscle", "organ", "peritoneum", "mesentery", "omentum", "lymph",
        "gluteus", "iliopsoas", "autochthon", "psoas",
        "liver", "hepatic", "bile", "biliary",
        "thyroid", "thyroid_gland", "parathyroid", "goiter",
        "spinal_cord", "spinal", "cord", "thecal", "neuroforam", "nerve_root", "cauda_equina", "conus", "epidural",
        "brain", "cerebr", "cerebell", "thalamus", "ventricle", "caudate", "putamen",
        "lentiform", "insula", "cortex", "white_matter", "gray_matter",
        "cyst", "fluid", "effusion", "abscess", "collection", "hematoma",
        "seroma", "oedema", "edema", "ascites", "pleural",
    ]},
    {"name": "Bone", "width": 1800, "level": 400, "keywords": [
        "bone", "vertebra", "sacrum", "coccyx", "rib", "sternum", "clavicle", "scapula",
        "humerus", "radius", "ulna", "carpal", "metacarpal", "phalanx",
        "femur", "tibia", "fibula", "patella", "tarsal", "metatarsal", "calcaneus",
        "pelvis", "ilium", "ischium", "pubic", "hip",
        "clavicula",
        "costal_cartilage", "cartilage", "intervertebral_disc", "disc",
        "spine", "cervical", "thoracic", "lumbar", "odontoid", "dens",
    ]},
    {"name": "Skull / Cranium", "width": 2800, "level": 600, "keywords": [
        "skull", "mandible", "maxilla", "zygomatic", "cranium", "calvarium",
        "orbital_wall", "temporal_bone", "occipital", "frontal_bone", "parietal",
        "sphenoid", "ethmoid", "cranial_vault",
    ]},
    {"name": "Vessel / Heart", "width": 600, "level": 200, "keywords": [
        "aorta", "artery", "arterial", "vein", "venous", "vena_cava", "cava",
        "coronary", "carotid", "subclavian", "iliac", "femoral", "brachiocephalic",
        "pulmonary_vein", "atrial_appendage", "vessel", "vascular", "angiography",
        "portal", "splenic",
        "heart", "myocardium", "ventricle", "atrium", "atrial", "pericardium",
        "mediastinum", "cardiac", "epicardial",
    ]},
]


def _script_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mimics_log(level, message):
    try:
        mimics.logging.log_user_message(level=level, message=message)
    except Exception:
        pass


def _state_path():
    return os.path.join(_script_root(), STATE_FILE)


def _load_state():
    return runtime_common.read_json(_state_path(), {}) or {}


def _save_state(value):
    try:
        runtime_common.write_json_atomic(_state_path(), value)
    except Exception:
        pass


def _preset_config_path():
    return os.path.join(_script_root(), "window_level_presets.json")


def _load_presets():
    path = _preset_config_path()
    data = runtime_common.read_json(path, None)
    if isinstance(data, list) and data:
        return data
    return DEFAULT_PRESETS


def _selected_mask_name():
    try:
        for mask in mimics.data.masks:
            if bool(getattr(mask, "selected", False)):
                return str(getattr(mask, "name", ""))
    except Exception:
        pass
    return ""


def _preset_for_name(name, presets):
    lowered = str(name or "").lower()
    for preset in presets:
        for keyword in preset.get("keywords", []):
            if str(keyword).lower() in lowered:
                return preset
    return None


def _active_image():
    try:
        return mimics.data.images.get_active()
    except Exception:
        return None


def _image_min_max():
    image = _active_image()
    minimum = getattr(image, "minimum_value", None)
    maximum = getattr(image, "maximum_value", None)
    if minimum is None or maximum is None:
        minimum, maximum = 0, 4095
    minimum = int(round(float(minimum)))
    maximum = int(round(float(maximum)))
    if maximum <= minimum:
        maximum = minimum + 1
    return minimum, maximum


def _hu_to_gv(value):
    try:
        return float(mimics.segment.HU2GV(float(value)))
    except Exception:
        return float(value)


def _current_contrast():
    getter = getattr(mimics.view, "get_contrast", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return value
        except Exception:
            pass
    return None


def _json_contrast(value):
    if not value:
        return None
    try:
        return [
            [float(value[0][0]), float(value[0][1])],
            [float(value[1][0]), float(value[1][1])],
        ]
    except Exception:
        return None


def _set_contrast_points(low_gv, high_gv):
    minimum, maximum = _image_min_max()
    low = max(minimum, min(maximum, int(round(float(low_gv)))))
    high = max(minimum, min(maximum, int(round(float(high_gv)))))
    if high <= low:
        if low < maximum:
            high = min(maximum, low + 1)
        else:
            low = max(minimum, high - 1)
    mimics.view.set_contrast((low, 0.0), (high, 1.0))
    try:
        mimics.update_gui()
    except Exception:
        pass
    return low, high


def _apply_preset(preset, source):
    width = float(preset["width"])
    level = float(preset["level"])
    low_hu = level - width / 2.0
    high_hu = level + width / 2.0
    previous = _json_contrast(_current_contrast())
    low, high = _set_contrast_points(_hu_to_gv(low_hu), _hu_to_gv(high_hu))
    _save_state(
        {
            "previous_contrast": previous,
            "last_preset": preset.get("name"),
            "last_width": width,
            "last_level": level,
            "last_low_gv": low,
            "last_high_gv": high,
            "source": source,
            "updated_at_epoch": time.time(),
        }
    )
    _mimics_log(
        logging.INFO,
        "Applied window/level preset: {0} (WW={1}, WL={2}, GV={3}-{4}).".format(
            preset.get("name"),
            int(width),
            int(level),
            low,
            high,
        ),
    )
    return preset


def _choose_preset(presets, selected_name=""):
    buttons = [str(preset.get("name", "")) for preset in presets if preset.get("name")]
    buttons.extend(["Reset Full Range", "Cancel"])
    message = "Select a CT window/level preset."
    if selected_name:
        message += "\nSelected Mask: {0}".format(selected_name)
    answer = mimics.dialogs.question_box(
        title=TITLE,
        message=message,
        buttons=";".join(buttons),
        ui_blocking=True,
    )
    if not answer or answer == "Cancel":
        return None
    if answer == "Reset Full Range":
        reset_full_range()
        return None
    for preset in presets:
        if preset.get("name") == answer:
            return preset
    return None


def apply_from_selected_mask():
    presets = _load_presets()
    selected_name = _selected_mask_name()
    preset = _preset_for_name(selected_name, presets)
    if preset is None:
        preset = _choose_preset(presets, selected_name)
    if preset is None:
        return 0
    _apply_preset(preset, "selected_mask:{0}".format(selected_name or "none"))
    return 0


def choose_preset():
    presets = _load_presets()
    selected_name = _selected_mask_name()
    preset = _choose_preset(presets, selected_name)
    if preset is not None:
        _apply_preset(preset, "manual_choice")
    return 0


def reset_full_range():
    previous = _json_contrast(_current_contrast())
    low, high = _image_min_max()
    _set_contrast_points(low, high)
    _save_state(
        {
            "previous_contrast": previous,
            "last_preset": "Full Range",
            "last_low_gv": low,
            "last_high_gv": high,
            "source": "reset_full_range",
            "updated_at_epoch": time.time(),
        }
    )
    _mimics_log(logging.INFO, "Reset window/level to full image range (GV={0}-{1}).".format(low, high))
    return 0


def undo_last():
    state = _load_state()
    previous = state.get("previous_contrast")
    if previous:
        try:
            mimics.view.set_contrast(tuple(previous[0]), tuple(previous[1]))
            try:
                mimics.update_gui()
            except Exception:
                pass
            _mimics_log(logging.INFO, "Restored previous window/level preset.")
            return 0
        except Exception as exc:
            _mimics_log(logging.WARNING, "Could not restore previous contrast exactly: {0}".format(exc))
    _mimics_log(
        logging.WARNING,
        "Exact previous window/level is not available from this Mimics API session; resetting to full image range.",
    )
    return reset_full_range()


def main(action="auto"):
    if action == "choose":
        return choose_preset()
    if action == "undo":
        return undo_last()
    if action == "reset":
        return reset_full_range()
    return apply_from_selected_mask()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "auto"
    main(action)
