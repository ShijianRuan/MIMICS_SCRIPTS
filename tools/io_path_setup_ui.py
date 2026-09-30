#!/usr/bin/env python3
"""External PySide6 path setup for Mimics import and mask export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

# Embeddable Python can omit the script directory from sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _candidate in (_HERE, _ROOT, os.path.join(_ROOT, "runtime_py35")):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import (
    choose_existing_directory,
    choose_existing_directory_async,
    choose_open_file,
    choose_open_file_async,
    choose_open_files,
    configure_application,
    stylesheet as shared_stylesheet,
)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


LABEL_FILE_SUFFIXES = (
    ".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".mha", ".mha.gz", ".mhd",
)


def count_existing_label_files(output_root, case_id):
    """Count label files already present in the export target folder.

    Used by the pre-export skip-existing reminder: with "Skip existing"
    selected and files already on disk, the user is asked before the export
    runs rather than learning from the end-of-run summary that nothing was
    updated.
    """
    final_folder = os.path.join(str(output_root or ""), str(case_id or "case"), "segmentations")
    if not os.path.isdir(final_folder):
        return 0
    try:
        names = os.listdir(final_folder)
    except OSError:
        return 0
    count = 0
    for name in names:
        if str(name).lower().endswith(LABEL_FILE_SUFFIXES):
            count += 1
    return count


def process_exists(pid):
    """Return whether the owning Mimics process is still alive."""
    # Delegate to the shared implementation in resource_locks (it also
    # treats WAIT_OBJECT_0 and ERROR_INVALID_PARAMETER as dead, which the
    # old local copy missed).
    from resource_locks import process_exists as _process_exists

    return _process_exists(pid)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False)
    last_error = None
    for attempt in range(12):
        temp = path.with_name(path.name + ".{0}.{1}.tmp".format(os.getpid(), uuid.uuid4().hex))
        try:
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(temp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if temp.is_file():
                    temp.unlink()
            except Exception:
                pass
            time.sleep(min(0.2, 0.03 * (attempt + 1)))
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError:
        if last_error is not None:
            raise last_error
        raise


def source_default_output(mode, source):
    if not source:
        return ""
    source_path = Path(source).expanduser()
    if mode == "import_batch":
        return str(source_path / "mcs_output")
    if mode == "import_single":
        return str(source_path.parent / "mcs_output")
    case_parent = source_path.parent if source_path.name else source_path
    return str(case_parent / "mask_exports")


def resolve_configured_output(configured, source, mode):
    text = os.path.expandvars(os.path.expanduser(str(configured or "").strip()))
    if not text:
        return source_default_output(mode, source)
    if os.path.isabs(text):
        return os.path.abspath(text)
    source_path = Path(source) if source else Path.cwd()
    base = source_path if mode == "import_batch" else source_path.parent
    return os.path.abspath(str(base / text))


MASK_SUFFIXES = (".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz", ".nii", ".mha", ".mhd", ".nrrd")


def _medical_file(path):
    text = str(path).lower()
    return os.path.isfile(path) and any(text.endswith(suffix) for suffix in MASK_SUFFIXES)


def _has_medical_suffix(name):
    """Cheap suffix check without touching the filesystem (no isfile call).

    Used when scanning large directories where per-entry os.path.isfile would
    make the path picker hang on directories with tens of thousands of files.
    """
    lower = str(name).lower()
    return any(lower.endswith(suffix) for suffix in MASK_SUFFIXES)


def _image_stem(path):
    name = os.path.basename(str(path))
    for suffix in MASK_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)] or "case"
    return os.path.splitext(name)[0] or "case"


def _looks_like_dataset_root(case_dir, profile):
    """True when a directory holds several case folders but no top-level image.

    A dataset root dropped/pasted into the single-case flow used to fall
    through to ``image = case_dir`` and be accepted as one giant DICOM
    series (R61-9). Sampling a capped number of entries is enough: a real
    case folder's images appear early, and excluded dirs (mcs_output,
    segmentations, ...) are not cases.
    """
    child_case_dirs = 0
    try:
        with os.scandir(case_dir) as entries:
            for index, entry in enumerate(entries):
                if index >= 64:
                    break
                if _has_medical_suffix(entry.name) and entry.is_file():
                    return False
                if entry.name.lower().endswith(".dcm") and entry.is_file():
                    return False
                if (
                    entry.is_dir(follow_symlinks=False)
                    and entry.name not in profile["exclude_dirs"]
                ):
                    child_case_dirs += 1
                    if child_case_dirs >= 2:
                        return True
    except OSError:
        return False
    return False


def discover_single_source(source, profile_id=None):
    """Discover one case outside Mimics so large DICOM folders never block it."""
    from dataset_profiles import load_profile

    profile = load_profile(profile_id)
    preferred = list(profile["image_candidates"])
    dicom_dirs = list(profile["dicom_dirs"]) or ["dicom"]
    mask_dirs = list(profile["mask_dirs"]) or ["segmentations"]
    selected = os.path.abspath(source)
    if _medical_file(selected):
        image = selected
        case_dir = os.path.dirname(selected)
        case_id = _image_stem(selected)
        # Count sibling medical images, but cap the scan: a case directory can
        # hold tens of thousands of DICOM slices, and calling os.path.isfile on
        # every entry makes the path picker hang for a long time. Sampling the
        # first few hundred entries is enough to tell whether >1 volume lives
        # here, which is all this flag is used for.
        sibling_images = 0
        try:
            with os.scandir(case_dir) as entries:
                for entry in entries:
                    if _has_medical_suffix(entry.name):
                        sibling_images += 1
                        if sibling_images > 1:
                            break
        except OSError:
            pass
        allow_masks = sibling_images <= 1
    elif os.path.isfile(selected) and selected.lower().endswith(".dcm"):
        case_dir = os.path.dirname(selected)
        case_id = os.path.basename(case_dir.rstrip("\\/")) or _image_stem(selected)
        image = case_dir
        allow_masks = True
    elif os.path.isdir(selected):
        case_dir = selected
        case_id = os.path.basename(case_dir.rstrip("\\/")) or "case"
        image = ""
        for name in preferred:
            candidate = os.path.join(case_dir, name)
            if _medical_file(candidate):
                image = candidate
                break
        if not image:
            # Walk without sorting and stop early. A flat DICOM folder can
            # contain tens of thousands of slices; proving that no arbitrary
            # NIfTI exists by enumerating every slice adds no value here.
            try:
                with os.scandir(case_dir) as entries:
                    for index, entry in enumerate(entries):
                        if _has_medical_suffix(entry.name) and entry.is_file():
                            image = entry.path
                            break
                        if entry.name.lower().endswith(".dcm") or index >= 511:
                            break
            except OSError:
                pass
        if not image:
            for dicom_name in dicom_dirs:
                dicom_dir = os.path.join(case_dir, dicom_name)
                if os.path.isdir(dicom_dir):
                    image = dicom_dir
                    break
            else:
                # A dataset root (several case folders, no top-level image)
                # must not be accepted as one giant DICOM series (R61-9).
                # Return None so the caller tells the user to pick a case.
                if _looks_like_dataset_root(case_dir, profile):
                    return None
                image = case_dir
        allow_masks = True
    else:
        return None
    masks = []
    for mask_dir_name in mask_dirs:
        seg_dir = os.path.join(case_dir, mask_dir_name)
        if allow_masks and os.path.isdir(seg_dir):
            for name in sorted(os.listdir(seg_dir)):
                path = os.path.join(seg_dir, name)
                if not _medical_file(path):
                    continue
                masks.append({"name": _image_stem(name), "path": path})
    return {
        "case_id": case_id,
        "image": image,
        "image_type": "medical_image" if _medical_file(image) else "dicom_candidate",
        "masks": masks,
        "case_dir": case_dir,
    }


def summarize_dataset(source, profile_id=None, entry_cap=511):
    """Pre-scan a dataset root and build the recognition summary.

    Mirrors the import discovery order (preferred names, capped fallback scan,
    DICOM subfolders) so the one-line summary matches what import will do.
    Returns None when source is not an existing directory.
    """
    if not source or not os.path.isdir(source):
        return None
    from dataset_profiles import load_profile

    profile = load_profile(profile_id)
    mask_dirs = list(profile["mask_dirs"]) or ["segmentations"]
    image_kinds = {}
    mask_count = 0
    warnings = []
    skipped = []
    case_count = 0
    try:
        names = sorted(os.listdir(source))
    except OSError:
        return None
    for name in names:
        case_dir = os.path.join(source, name)
        if not os.path.isdir(case_dir):
            continue
        if name in profile["exclude_dirs"]:
            continue
        image = None
        for candidate in profile["image_candidates"]:
            path = os.path.join(case_dir, candidate)
            if _medical_file(path):
                image = candidate
                break
        volume_files = 0
        volume_names = []
        dicom_heuristic = False
        # One capped scan answers both "is there a fallback image?" and "does
        # more than one volume live here?" — the anomaly the recognition list
        # must interrupt on (e.g. ct.nii.gz + old_ct.nii.gz in one case dir).
        try:
            with os.scandir(case_dir) as entries:
                for index, entry in enumerate(entries):
                    if _has_medical_suffix(entry.name):
                        try:
                            if entry.is_file():
                                volume_files += 1
                                if len(volume_names) < 2:
                                    volume_names.append(entry.name)
                                if image is None and len(volume_names) == 1:
                                    image = entry.name
                                if volume_files > 1 and image is not None:
                                    break
                        except OSError:
                            pass
                    if entry.name.lower().endswith(".dcm") or index >= entry_cap:
                        dicom_heuristic = True
                        break
        except OSError:
            pass
        image_kind = "fallback"
        if image is not None and "." in image and not dicom_heuristic:
            image_kind = "image file"
        if image is None:
            for dicom_name in profile["dicom_dirs"]:
                if os.path.isdir(os.path.join(case_dir, dicom_name)):
                    image = dicom_name
                    image_kind = "dicom folder"
                    break
        if image is None:
            skipped.append(name)
            continue
        case_count += 1
        image_kinds[image_kind] = image_kinds.get(image_kind, 0) + 1
        if volume_files > 1:
            warnings.append(
                "{0}: {1} volume files found ({2}); '{3}' will be used".format(
                    name, volume_files, ", ".join(volume_names), image
                )
            )
        elif dicom_heuristic and image_kind == "fallback":
            warnings.append(
                "{0}: many files detected; the series will be validated during import".format(name)
            )
        for mask_dir_name in mask_dirs:
            seg_dir = os.path.join(case_dir, mask_dir_name)
            if not os.path.isdir(seg_dir):
                continue
            try:
                for fname in os.listdir(seg_dir):
                    if _has_medical_suffix(fname):
                        mask_count += 1
            except OSError:
                pass
    parts = ["Recognized {0} case(s)".format(case_count)]
    if image_kinds:
        detail = ", ".join(
            "{0} ({1})".format(kind, count)
            for kind, count in sorted(image_kinds.items(), key=lambda kv: -kv[1])
        )
        parts.append("images: " + detail)
    if mask_count:
        parts.append("{0} mask(s)".format(mask_count))
    if skipped:
        parts.append("{0} skipped (no image)".format(len(skipped)))
    return {
        "summary": "; ".join(parts),
        "warnings": warnings,
        "skipped": skipped,
        "case_count": case_count,
        "mask_count": mask_count,
    }


def run_ui(context, preview_path=""):
    from PySide6 import QtCore, QtGui, QtWidgets

    mode = str(context.get("mode") or "import_batch")
    status_path = context.get("status_path")
    state_path = context.get("state_path")
    owner_pid = context.get("owner_pid")
    bootstrap_stop_path = str(context.get("bootstrap_stop_path") or "")
    remembered = read_json(state_path, {}) or {}
    remembered_mode = remembered.get(mode) or {}

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "Mimics Data Paths")
    window = QtWidgets.QDialog()
    window.setObjectName("ioWindow")
    window.setModal(False)
    window.setMinimumWidth(680)
    window.resize(820, 610 if mode == "export_masks" else 590)
    window.setWindowTitle({
        "import_batch": "Import Dataset",
        "import_single": "Import Single Case",
        "export_masks": "Export Masks",
    }.get(mode, "Data Paths"))
    window.setStyleSheet(shared_stylesheet())

    root = QtWidgets.QVBoxLayout(window)
    root.setContentsMargins(30, 26, 30, 24)
    root.setSpacing(16)
    title = QtWidgets.QLabel({
        "import_batch": "Import dataset",
        "import_single": "Import one case",
        "export_masks": "Export project masks",
    }.get(mode, "Choose data paths"))
    title.setObjectName("title")
    subtitle = QtWidgets.QLabel({
        "import_batch": "Choose the dataset once. Prepared Mimics projects are written to the output folder.",
        "import_single": "Choose one image file, case folder, or DICOM folder and where its Mimics project should be saved.",
        "export_masks": "Confirm the source case and choose where editable label files should be written.",
    }.get(mode, "Choose the paths needed for this task."))
    subtitle.setObjectName("subtitle")
    subtitle.setWordWrap(True)
    root.addWidget(title)
    root.addWidget(subtitle)

    panel = QtWidgets.QFrame()
    panel.setObjectName("panel")
    form = QtWidgets.QVBoxLayout(panel)
    form.setContentsMargins(20, 18, 20, 18)
    form.setSpacing(15)
    root.addWidget(panel, 1)

    progress_panel = QtWidgets.QFrame()
    progress_panel.setObjectName("panel")
    progress_layout = QtWidgets.QVBoxLayout(progress_panel)
    progress_layout.setContentsMargins(20, 20, 20, 18)
    progress_layout.setSpacing(12)
    progress_heading = QtWidgets.QLabel("Starting task...")
    progress_heading.setObjectName("section")
    progress_layout.addWidget(progress_heading)
    progress_bar = QtWidgets.QProgressBar()
    progress_bar.setRange(0, 0)
    progress_layout.addWidget(progress_bar)
    progress_detail = QtWidgets.QLabel("Waiting for Mimics to start the workflow.")
    progress_detail.setWordWrap(True)
    progress_layout.addWidget(progress_detail)
    progress_activity = QtWidgets.QTextEdit()
    progress_activity.setReadOnly(True)
    progress_activity.setMaximumHeight(180)
    progress_layout.addWidget(progress_activity, 1)
    progress_actions = QtWidgets.QHBoxLayout()
    open_output = QtWidgets.QPushButton("Open Output")
    open_log = QtWidgets.QPushButton("Open Log")
    stop_task = QtWidgets.QPushButton("Stop")
    stop_task.setObjectName("dangerButton")
    hide_task = QtWidgets.QPushButton("Minimize")
    close_task = QtWidgets.QPushButton("Close")
    close_task.setEnabled(False)
    progress_actions.addWidget(open_output)
    progress_actions.addWidget(open_log)
    progress_actions.addStretch(1)
    progress_actions.addWidget(stop_task)
    progress_actions.addWidget(hide_task)
    progress_actions.addWidget(close_task)
    progress_layout.addLayout(progress_actions)
    progress_panel.hide()
    root.addWidget(progress_panel, 1)

    def path_row(label, initial, browse_folder=True, file_or_folder=False, hint=""):
        form.addWidget(_label(QtWidgets, label, "section"))
        row = QtWidgets.QHBoxLayout()
        edit = QtWidgets.QLineEdit(str(initial or ""))
        edit.setClearButtonEnabled(True)
        paste_button = QtWidgets.QPushButton("Paste")
        row.addWidget(edit, 1)
        row.addWidget(paste_button)
        file_button = None
        if file_or_folder:
            file_button = QtWidgets.QPushButton("Choose File...")
            row.addWidget(file_button)
        button = QtWidgets.QPushButton("Choose Folder...")
        row.addWidget(button)
        form.addLayout(row)
        if hint:
            form.addWidget(_label(QtWidgets, hint, "hint"))

        def browse_folder_path():
            def selected(value):
                if value:
                    edit.setProperty("chosenByBrowse", True)
                    edit.setText(str(value))

            choose_existing_directory_async(
                QtCore,
                QtWidgets,
                window,
                label,
                edit.text().strip() or str(Path.home()),
                selected,
                button=button,
            )

        def browse_file_path():
            def selected(value):
                if value:
                    edit.setProperty("chosenByBrowse", True)
                    edit.setText(str(value))

            choose_open_file_async(
                QtCore,
                QtWidgets,
                window,
                label,
                edit.text().strip() or str(Path.home()),
                "Medical volumes (*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz);;All files (*)",
                selected,
                button=file_button,
            )

        def paste_value():
            value = QtWidgets.QApplication.clipboard().text().strip()
            if value:
                edit.setProperty("chosenByBrowse", True)
                edit.setText(value)

        paste_button.clicked.connect(paste_value)
        button.clicked.connect(browse_folder_path)
        if file_button is not None:
            file_button.clicked.connect(browse_file_path)
        return edit

    source_initial = context.get("source_initial") or remembered_mode.get("source_path", "")
    if mode in ("import_single", "export_masks"):
        source_edit = path_row("SOURCE IMAGE OR CASE", source_initial, file_or_folder=True, hint="Supported: NIfTI, MHA/MHD, NRRD, a DICOM file, case folder, or DICOM series folder.")
    else:
        source_edit = path_row("SOURCE DATASET", source_initial, hint="Required. This path is never modified by import.")

    # R61-7: only a folder the user actually chose themselves sticks across
    # sessions. A remembered value that was just the computed default for
    # the *previous* source would pin every new dataset to that old folder.
    output_initial = context.get("output_initial") or (
        remembered_mode.get("output_path", "")
        if remembered_mode.get("output_custom")
        else ""
    )
    output_edit = path_row("MCS OUTPUT FOLDER" if mode.startswith("import") else "EXPORT ROOT", output_initial, hint="Optional to change. A safe default is filled automatically from the source path.")
    output_preview = _label(QtWidgets, "", "preview")
    output_preview.setWordWrap(True)
    form.addWidget(output_preview)

    # Recognition summary (P2): one line describing what import found in the
    # dataset, refreshed in a background thread so large roots never block the
    # path picker. Anomalies do not block here; they are surfaced at submit.
    recognition_state = {"summary": None, "scanning": False}
    recognition_label = None
    if mode == "import_batch":
        recognition_label = _label(QtWidgets, "", "preview")
        recognition_label.setWordWrap(True)
        form.addWidget(recognition_label)

    policy_row = None
    skip_radio = overwrite_radio = None
    mask_all = mask_none = mask_named = mask_names = None
    if mode.startswith("import"):
        form.addWidget(_label(QtWidgets, "MASKS TO IMPORT", "section"))
        mask_row = QtWidgets.QHBoxLayout()
        mask_all = QtWidgets.QRadioButton("All masks")
        mask_none = QtWidgets.QRadioButton("Images only")
        mask_named = QtWidgets.QRadioButton("Named masks")
        remembered_selection = str(remembered_mode.get("mask_selection", "all") or "all")
        mask_none.setChecked(remembered_selection.lower() == "none")
        mask_named.setChecked(remembered_selection.lower() not in ("all", "none"))
        mask_all.setChecked(not mask_none.isChecked() and not mask_named.isChecked())
        mask_row.addWidget(mask_all)
        mask_row.addWidget(mask_none)
        mask_row.addWidget(mask_named)
        mask_row.addStretch(1)
        form.addLayout(mask_row)
        mask_names = QtWidgets.QLineEdit(remembered_selection if mask_named.isChecked() else "")
        mask_names.setPlaceholderText("Example: liver, spleen, aorta")
        mask_names.setEnabled(mask_named.isChecked())
        form.addWidget(mask_names)
        mask_named.toggled.connect(mask_names.setEnabled)
    if mode == "export_masks":
        form.addWidget(_label(QtWidgets, "MASKS TO EXPORT", "section"))
        export_mask_row = QtWidgets.QHBoxLayout()
        mask_all = QtWidgets.QRadioButton("All masks")
        mask_named = QtWidgets.QRadioButton("Selected names")
        mask_all.setChecked(True)
        export_mask_row.addWidget(mask_all)
        export_mask_row.addWidget(mask_named)
        export_mask_row.addStretch(1)
        form.addLayout(export_mask_row)
        # Editable name field: the user can either type names directly
        # (comma-separated) or pick from the candidate list below.  Clicking a
        # candidate toggles it into the field, so a project with hundreds of
        # masks can be narrowed by typing without scrolling.
        mask_names = QtWidgets.QLineEdit()
        mask_names.setPlaceholderText("Type names (comma-separated) or pick from the list below")
        mask_names.setEnabled(False)
        form.addWidget(mask_names)
        available_masks = [str(name) for name in (context.get("mask_names") or []) if str(name).strip()]
        mask_candidate_list = QtWidgets.QListWidget()
        mask_candidate_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        mask_candidate_list.setMaximumHeight(120)
        mask_candidate_list.addItems(available_masks)
        mask_candidate_list.setEnabled(False)
        mask_candidate_list.itemClicked.connect(
            lambda item: _toggle_name_in_field(mask_names, item.text())
        )
        # Filter the candidate list as the user types in the name field.
        mask_names.textChanged.connect(
            lambda text: _filter_list_widget(mask_candidate_list, _last_token(text))
        )
        form.addWidget(mask_candidate_list)
        mask_named.toggled.connect(mask_names.setEnabled)
        mask_named.toggled.connect(mask_candidate_list.setEnabled)
        policy_row = QtWidgets.QHBoxLayout()
        policy_row.addWidget(_label(QtWidgets, "IF FILES ALREADY EXIST", "section"))
        policy_row.addStretch(1)
        skip_radio = QtWidgets.QRadioButton("Skip existing")
        overwrite_radio = QtWidgets.QRadioButton("Overwrite existing")
        overwrite_radio.setChecked(remembered_mode.get("conflict_policy") == "overwrite")
        skip_radio.setChecked(not overwrite_radio.isChecked())
        policy_row.addWidget(skip_radio)
        policy_row.addWidget(overwrite_radio)
        form.addLayout(policy_row)
        # P1 multi-format export: one or more formats per export, all sharing
        # identical geometry. NIfTI stays the default so historical behaviour
        # is unchanged unless the user opts in.
        form.addWidget(_label(QtWidgets, "EXPORT FORMATS", "section"))
        format_row = QtWidgets.QHBoxLayout()
        format_boxes = {}
        remembered_formats = set(remembered_mode.get("export_formats", []) or [])
        for fmt_key, fmt_label in (("nii.gz", "NIfTI (.nii.gz)"), ("nrrd", "NRRD (.nrrd)"), ("mha", "MHA (.mha)")):
            box = QtWidgets.QCheckBox(fmt_label)
            box.setChecked(
                fmt_key in remembered_formats if remembered_formats else fmt_key == "nii.gz"
            )
            format_boxes[fmt_key] = box
            format_row.addWidget(box)
        format_row.addStretch(1)
        form.addLayout(format_row)

    configured_default = context.get("configured_output", "")
    output_user_edited = {"value": bool(output_initial)}

    def refresh_default():
        source = source_edit.text().strip()
        if not output_user_edited["value"]:
            new_output = resolve_configured_output(configured_default, source, mode)
            # Avoid re-setting (and re-emitting textChanged) when nothing changed;
            # this keeps typing in the source field from churning the signal chain.
            if new_output != output_edit.text():
                output_edit.setText(new_output)
        output = output_edit.text().strip()
        if mode == "export_masks":
            case_id = str(context.get("case_id") or "case")
            preview_text = "Final folder: {0}".format(os.path.join(output, case_id, "segmentations")) if output else ""
        else:
            preview_text = "Mimics projects: {0}".format(output) if output else ""
        if output_preview.text() != preview_text:
            output_preview.setText(preview_text)
        submit.setEnabled(bool(source and output))

    def output_edited(_text):
        if output_edit.hasFocus() or bool(output_edit.property("chosenByBrowse")):
            output_user_edited["value"] = True
            output_edit.setProperty("chosenByBrowse", False)
        refresh_default()

    def apply_recognition(result):
        recognition_state["scanning"] = False
        recognition_state["summary"] = result
        if recognition_label is None:
            return
        if result is None:
            recognition_label.setText("")
        else:
            text = result["summary"]
            if result["warnings"]:
                text += "  |  {0} item(s) need attention at start".format(len(result["warnings"]))
            recognition_label.setText(text)

    def refresh_recognition():
        if recognition_label is None or recognition_state["scanning"]:
            return
        source = source_edit.text().strip()
        if not source or not os.path.isdir(source):
            recognition_state["summary"] = None
            recognition_label.setText("")
            return
        recognition_state["scanning"] = True
        recognition_label.setText("Scanning dataset...")

        def scan():
            try:
                result = summarize_dataset(source)
            except Exception:
                result = None
            recognition_queue.put(result)

        thread = threading.Thread(target=scan, name="io-recognition-scan")
        thread.daemon = True
        thread.start()

    recognition_queue = Queue()
    recognition_timer = QtCore.QTimer(window)
    recognition_timer.setInterval(150)

    def poll_recognition():
        try:
            apply_recognition(recognition_queue.get_nowait())
        except Empty:
            return

    recognition_timer.timeout.connect(poll_recognition)
    recognition_timer.start(150)

    # textChanged fires per character; on a network dataset each call
    # spawns a fresh directory scan thread. Debounce so a burst of
    # keystrokes (or a paste) triggers at most one scan, 400ms after
    # the last change.
    recognition_debounce = QtCore.QTimer(window)
    recognition_debounce.setSingleShot(True)
    recognition_debounce.setInterval(400)
    recognition_debounce.timeout.connect(refresh_recognition)
    source_edit.textChanged.connect(lambda _text: recognition_debounce.start())
    refresh_recognition()

    source_edit.textChanged.connect(lambda _text: refresh_default())
    output_edit.textChanged.connect(output_edited)
    footer_widget = QtWidgets.QWidget()
    footer = QtWidgets.QHBoxLayout(footer_widget)
    footer.setContentsMargins(0, 0, 0, 0)
    remember = QtWidgets.QCheckBox("Remember these folders on this workstation")
    remember.setChecked(True)
    footer.addWidget(remember)
    footer.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    submit = QtWidgets.QPushButton("Start Import" if mode.startswith("import") else "Start Export")
    submit.setObjectName("primary")
    footer.addWidget(cancel)
    footer.addWidget(submit)
    root.addWidget(footer_widget)

    def cancel_window():
        if submission_state.get("submitted"):
            window.accept()
            return
        if status_path:
            write_json(status_path, {"status": "cancelled", "updated_at_epoch": time.time()})
        window.accept()

    submission_results = Queue()
    submission_state = {"running": False, "submitted": False}
    bootstrap_descriptor = {
        "kind": "starting",
        "title": {
            "import_single": "Import single case",
            "import_batch": "Import dataset",
            "export_masks": "Export masks",
        }.get(mode, "Starting task"),
        "stop_path": bootstrap_stop_path,
        "stop_paths": [bootstrap_stop_path] if bootstrap_stop_path else [],
    }
    task_state = {
        "descriptor": bootstrap_descriptor,
        "last_signature": None,
        "stop_requested": False,
        "owner_lost": False,
    }
    submission_timer = QtCore.QTimer(window)
    task_timer = QtCore.QTimer(window)

    def open_path(path):
        target = str(path or "").strip()
        if not target:
            return
        if os.path.isfile(target):
            target = os.path.dirname(target)
        if not os.path.exists(target):
            return
        try:
            if os.name == "nt":
                os.startfile(target)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", target])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", target])
        except Exception:
            pass

    def append_activity(text):
        text = str(text or "").strip()
        if not text:
            return
        progress_activity.append("[{0}] {1}".format(time.strftime("%H:%M:%S"), text))

    def show_progress():
        submission_state["submitted"] = True
        panel.hide()
        footer_widget.hide()
        title.setText("Task in progress")
        subtitle.setText("Mimics remains available while this task runs.")
        progress_panel.show()
        stop_task.setEnabled(bool(bootstrap_stop_path))
        append_activity("Request submitted to Mimics.")

    def request_stop():
        descriptor = task_state.get("descriptor") or {}
        stop_paths = descriptor.get("stop_paths") or []
        if isinstance(stop_paths, str):
            stop_paths = [stop_paths]
        if descriptor.get("stop_path") and descriptor.get("stop_path") not in stop_paths:
            stop_paths.insert(0, descriptor.get("stop_path"))
        stop_paths = [str(path) for path in stop_paths if str(path or "").strip()]
        if bootstrap_stop_path and bootstrap_stop_path not in stop_paths:
            stop_paths.insert(0, bootstrap_stop_path)
        if not stop_paths:
            progress_detail.setText("This task cannot be stopped from this window. Close it and use the matching Stop entry in Mimics.")
            return
        try:
            errors = []
            for stop_path in stop_paths:
                try:
                    write_json(stop_path, {
                        "status": "cancel_requested",
                        "requested_at_epoch": time.time(),
                    })
                except Exception as exc:
                    errors.append("{0}: {1}".format(stop_path, exc))
            if errors:
                raise RuntimeError("; ".join(errors))
            task_state["stop_requested"] = True
            stop_task.setEnabled(False)
            progress_detail.setText("Stop requested. Waiting for the active process to exit safely.")
            append_activity("Stop requested. The current safe unit of work will finish first.")
        except Exception as exc:
            progress_detail.setText("Could not request stop: {0}".format(exc))

    def terminal_state(state):
        return str(state or "").lower() in (
            "closed", "completed", "done", "failed", "cancelled", "canceled",
        )

    def update_progress_from_payload(payload, descriptor):
        payload = payload or {}
        state = str(payload.get("status") or payload.get("phase") or "running")
        phase = str(payload.get("phase") or state).replace("_", " ")
        completed = int(payload.get("completed", 0) or 0)
        failed = int(payload.get("failed", 0) or 0)
        total = int(payload.get("total", 0) or payload.get("total_count", 0) or 0)
        index = int(payload.get("index", 0) or 0)
        progress_percent = payload.get("progress_percent")
        current = payload.get("case_id") or payload.get("mask_name") or ""
        progress_heading.setText(str(descriptor.get("title") or "Task progress"))
        if progress_percent is not None and not terminal_state(state):
            try:
                percent = min(100, max(0, int(float(progress_percent))))
            except (TypeError, ValueError):
                percent = 0
            progress_bar.setRange(0, 100)
            progress_bar.setValue(percent)
            progress_bar.setFormat("%p%")
        elif total > 0:
            progress_bar.setRange(0, total)
            progress_bar.setValue(min(total, max(completed + failed, index)))
            progress_bar.setFormat("%v / %m")
        else:
            progress_bar.setRange(0, 0)
        detail = phase.capitalize()
        if total:
            remaining = max(0, total - completed - failed)
            detail += " - {0} created, {1} failed, {2} total".format(completed, failed, total)
            if remaining and not terminal_state(state):
                detail += " ({0} remaining)".format(remaining)
        if current:
            detail += "\nCurrent: {0}".format(current)
        if payload.get("error"):
            detail += "\n{0}".format(payload.get("error"))
        progress_detail.setText(detail)
        signature = (state, phase, completed, failed, total, progress_percent, current, payload.get("error"))
        if signature != task_state.get("last_signature"):
            task_state["last_signature"] = signature
            append_activity(detail.replace("\n", " | "))
        if terminal_state(state):
            progress_bar.setRange(0, max(1, total))
            progress_bar.setValue(max(1, min(total or 1, completed + failed or 1)))
            stop_task.setEnabled(False)
            close_task.setEnabled(True)
            hide_task.setEnabled(False)

    def poll_task():
        setup_status = read_json(status_path, {}) or {}
        setup_state = str(setup_status.get("status") or "")
        owner_alive = process_exists(owner_pid) if owner_pid else True
        if not owner_alive and not terminal_state(setup_state):
            if not task_state.get("owner_lost"):
                task_state["owner_lost"] = True
                request_stop()
                error = (
                    "The Mimics process closed before it could continue monitoring this task. "
                    "A stop request was written for any owned background work. Reopen Mimics and retry."
                )
                try:
                    write_json(status_path, {
                        "status": "failed",
                        "error": error,
                        "updated_at_epoch": time.time(),
                    })
                except Exception:
                    pass
                update_progress_from_payload(
                    {"status": "failed", "error": error},
                    {"title": "Mimics closed unexpectedly"},
                )
            return
        if setup_state in ("failed", "cancelled", "canceled", "closed"):
            default_error = (
                "Task start was cancelled."
                if setup_state in ("cancelled", "canceled", "closed")
                else "The task could not start."
            )
            update_progress_from_payload(
                {
                    "status": "cancelled" if setup_state in ("cancelled", "canceled", "closed") else "failed",
                    "error": setup_status.get("error", default_error),
                },
                {"title": "Task cancelled" if setup_state in ("cancelled", "canceled", "closed") else "Could not start task"},
            )
            return
        if setup_state in ("opening", "configuring", "submitted", "launching", ""):
            descriptor = setup_status.get("task") or bootstrap_descriptor
            task_state["descriptor"] = descriptor
            stop_task.setEnabled(
                bool(descriptor.get("stop_path") or descriptor.get("stop_paths") or bootstrap_stop_path)
                and not task_state.get("stop_requested")
            )
            update_progress_from_payload(
                {
                    "status": "running",
                    "phase": "starting_in_mimics" if setup_state == "launching" else "waiting_for_mimics",
                },
                descriptor,
            )
            return
        if setup_state != "launched":
            return
        descriptor = setup_status.get("task") or {}
        task_state["descriptor"] = descriptor
        open_output.setEnabled(bool(descriptor.get("output_path")))
        open_log.setEnabled(bool(descriptor.get("log_path")))
        stop_task.setEnabled(
            bool(descriptor.get("stop_path") or descriptor.get("stop_paths"))
            and not task_state.get("stop_requested")
        )
        primary = read_json(descriptor.get("status_path"), {}) if descriptor.get("status_path") else {}
        secondary = read_json(descriptor.get("secondary_status_path"), {}) if descriptor.get("secondary_status_path") else {}
        payload = primary or {}
        if secondary:
            primary_phase = str((primary or {}).get("phase") or (primary or {}).get("status") or "")
            secondary_state = str(secondary.get("status") or secondary.get("phase") or "")
            if (
                primary_phase in ("prepared", "creating_mcs", "waiting_for_mcs", "queued")
                or not terminal_state(secondary_state)
            ):
                payload = secondary
        if not payload:
            payload = {"status": "launching", "phase": "starting"}
        update_progress_from_payload(payload, descriptor)

    task_timer.timeout.connect(poll_task)
    task_timer.start(400)
    open_output.clicked.connect(lambda: open_path((task_state.get("descriptor") or {}).get("output_path")))
    open_log.clicked.connect(lambda: open_path((task_state.get("descriptor") or {}).get("log_path")))
    stop_task.clicked.connect(request_stop)
    hide_task.clicked.connect(window.showMinimized)
    close_task.clicked.connect(window.accept)

    def poll_submission():
        try:
            result = submission_results.get_nowait()
        except Empty:
            return
        submission_state["running"] = False
        submit.setEnabled(True)
        if result.get("error"):
            output_preview.setText("")
            QtWidgets.QMessageBox.warning(window, result.get("title", "Path Error"), result["error"])
            refresh_default()
            return
        selection = result["selection"]
        if selection.get("remember") and state_path:
            state = read_json(state_path, {}) or {}
            state[mode] = selection
            write_json(state_path, state)
        write_json(status_path, {"status": "submitted", "selection": selection, "updated_at_epoch": time.time()})
        show_progress()

    submission_timer.timeout.connect(poll_submission)
    submission_timer.start(80)

    def submit_window():
        if submission_state["running"]:
            return
        source = os.path.abspath(os.path.expanduser(source_edit.text().strip()))
        output = os.path.abspath(os.path.expanduser(output_edit.text().strip()))
        selection = {
            "source_path": source,
            "output_path": output,
            "remember": bool(remember.isChecked()),
            # True only when the user deviated from the computed default this
            # round; see the output_initial comment above.
            "output_custom": bool(
                os.path.normcase(output)
                != os.path.normcase(
                    resolve_configured_output(configured_default, source, mode)
                )
            ),
        }
        if mode.startswith("import"):
            if mask_none.isChecked():
                selection["mask_selection"] = "none"
            elif mask_named.isChecked():
                names = ",".join(item.strip() for item in mask_names.text().split(",") if item.strip())
                if not names:
                    QtWidgets.QMessageBox.warning(window, "Mask Names Required", "Enter one or more mask names separated by commas.")
                    return
                selection["mask_selection"] = names
            else:
                selection["mask_selection"] = "all"
        if mode == "export_masks":
            selection["conflict_policy"] = "overwrite" if overwrite_radio.isChecked() else "skip"
            selected_formats = [fmt for fmt, box in format_boxes.items() if box.isChecked()]
            if not selected_formats:
                QtWidgets.QMessageBox.warning(
                    window,
                    "Export Format Required",
                    "Choose at least one export format.",
                )
                return
            selection["export_formats"] = selected_formats
            if mask_named.isChecked():
                names = ",".join(item.strip() for item in mask_names.text().split(",") if item.strip())
                if not names:
                    QtWidgets.QMessageBox.warning(
                        window,
                        "Mask Names Required",
                        "Enter one or more mask names (comma-separated) or pick from the list.",
                    )
                    return
                selection["mask_selection"] = names
            else:
                selection["mask_selection"] = "all"

        # Skip-existing proactive reminder (P6c): when the target folder
        # already holds label files and the user chose "Skip existing", say
        # so before the export runs - the end-of-run summary otherwise tells
        # them only after the fact that nothing was updated.
        if mode == "export_masks" and selection.get("conflict_policy") == "skip":
            case_id = str(context.get("case_id") or "case")
            existing_count = count_existing_label_files(output, case_id)
            final_folder = os.path.join(output, case_id, "segmentations")
            if existing_count:
                answer = QtWidgets.QMessageBox.question(
                    window,
                    "Existing Label Files",
                    (
                        "The target folder already contains {0} label file(s):\n\n{1}\n\n"
                        "With \"Skip existing\" selected they will be kept unchanged "
                        "(the export will report them as not updated).\n\n"
                        "Yes = continue with Skip existing\n"
                        "No = switch to Overwrite existing and continue\n"
                        "Cancel = go back and change settings"
                    ).format(existing_count, final_folder),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.No,
                )
                if answer == QtWidgets.QMessageBox.Cancel:
                    return
                if answer == QtWidgets.QMessageBox.No:
                    selection["conflict_policy"] = "overwrite"
                    overwrite_radio.setChecked(True)

        # Recognition anomalies interrupt once, with a plain list of what will
        # happen; the user can still proceed (the summary line already told
        # them the normal path is fine).
        if mode == "import_batch" and recognition_state.get("summary"):
            summary = recognition_state["summary"]
            if summary.get("warnings") or summary.get("skipped"):
                lines = []
                for warning in summary["warnings"][:20]:
                    lines.append("• " + warning)
                if summary.get("skipped"):
                    shown = ", ".join(summary["skipped"][:10])
                    more = "" if len(summary["skipped"]) <= 10 else " (+{0} more)".format(len(summary["skipped"]) - 10)
                    lines.append("• No usable image found in {0} case(s): {1}{2} — they will be skipped.".format(len(summary["skipped"]), shown, more))
                answer = QtWidgets.QMessageBox.question(
                    window,
                    "Review Recognition Results",
                    "The dataset scan found:\n\n{0}\n\nProceed with import?".format("\n".join(lines)),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return

        submission_state["running"] = True
        submit.setEnabled(False)
        output_preview.setText("Checking paths in the background...")

        def validate_paths():
            try:
                if not os.path.exists(source):
                    submission_results.put({"title": "Source Not Found", "error": "Choose an existing source file or folder."})
                    return
                if not os.path.isdir(output):
                    os.makedirs(output)
                fd, probe_path = tempfile.mkstemp(prefix=".mimics_write_test_", dir=output)
                os.close(fd)
                os.remove(probe_path)
                if mode in ("import_single", "export_masks"):
                    case_info = discover_single_source(source)
                    if not case_info:
                        submission_results.put({
                            "title": "Unsupported Source",
                            "error": "No supported 3D image or DICOM folder was found.",
                        })
                        return
                    selection["case_info"] = case_info
                submission_results.put({"selection": selection})
            except Exception as exc:
                submission_results.put({
                    "title": "Path Validation Failed",
                    "error": "The selected paths could not be validated.\n\n{0}".format(exc),
                })

        thread = threading.Thread(target=validate_paths, name="io-path-validation")
        thread.daemon = True
        thread.start()

    cancel.clicked.connect(cancel_window)
    submit.clicked.connect(submit_window)
    window.rejected.connect(cancel_window)
    refresh_default()

    if preview_path:
        window.show()
        app.processEvents()
        window.grab().save(preview_path)
        return 0
    window.show()
    return app.exec()


def _label(QtWidgets, text, object_name):
    label = QtWidgets.QLabel(text)
    label.setObjectName(object_name)
    return label


def _filter_list_widget(widget, text):
    """Hide list items whose text does not contain the filter (case-insensitive).

    Selection is preserved across filtering so a multi-select built up over
    several searches is not lost when the filter text changes.
    """
    needle = str(text or "").strip().lower()
    for index in range(widget.count()):
        item = widget.item(index)
        item.setHidden(bool(needle) and needle not in item.text().lower())


def _last_token(text):
    """Return the token being typed after the last comma.

    The name field holds comma-separated names; only the token currently being
    typed should drive the candidate-list filter, so an already-typed
    "liver, kidney" filters on "kidney", not the whole string.
    """
    return str(text or "").rsplit(",", 1)[-1].strip()


def _toggle_name_in_field(field, name):
    """Add or remove a mask name in the comma-separated name field.

    Clicking a candidate toggles its presence so the field always reads as a
    clean, de-duplicated, comma-separated list.  Names that are absent from
    the field (typed freehand) are likewise preserved.
    """
    current = [item.strip() for item in field.text().split(",") if item.strip()]
    target = str(name).strip()
    if not target:
        return
    if target in current:
        current = [item for item in current if item != target]
    else:
        current.append(target)
    field.setText(", ".join(current))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--preview", default="")
    args = parser.parse_args(argv)
    context = read_json(args.context, None)
    if not context:
        raise RuntimeError("I/O setup context is missing or invalid: {0}".format(args.context))
    try:
        return run_ui(context, args.preview)
    except Exception as exc:
        if context.get("status_path"):
            write_json(context["status_path"], {"status": "failed", "error": str(exc), "updated_at_epoch": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
