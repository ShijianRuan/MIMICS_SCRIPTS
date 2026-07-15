#!/usr/bin/env python3
"""External PySide6 path setup for Mimics import and mask export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".{0}.{1}.tmp".format(os.getpid(), uuid.uuid4().hex))
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(temp), str(path))


def source_default_output(mode, source):
    if not source:
        return ""
    source_path = Path(source).expanduser()
    if mode == "import_batch":
        return str(source_path / "mcs_output")
    if mode == "import_single":
        base = source_path.parent if source_path.is_file() else source_path.parent
        return str(base / "mcs_output")
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


MASK_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


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


def discover_single_source(source):
    """Discover one case outside Mimics so large DICOM folders never block it."""
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
            entries = os.listdir(case_dir)
        except OSError:
            entries = []
        for name in entries:
            if _has_medical_suffix(name):
                sibling_images += 1
                if sibling_images > 1:
                    break
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
        preferred = ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii", "ct.mhd", "mri.mhd", "ct.mha", "mri.mha")
        for name in preferred:
            candidate = os.path.join(case_dir, name)
            if _medical_file(candidate):
                image = candidate
                break
        if not image:
            # Walk without sorting and stop at the first medical file: sorting
            # a huge directory listing is what makes this hang.
            try:
                entries = os.listdir(case_dir)
            except OSError:
                entries = []
            for name in entries:
                if _has_medical_suffix(name) and _medical_file(os.path.join(case_dir, name)):
                    image = os.path.join(case_dir, name)
                    break
        if not image:
            dicom_dir = os.path.join(case_dir, "dicom")
            image = dicom_dir if os.path.isdir(dicom_dir) else case_dir
        allow_masks = True
    else:
        return None
    masks = []
    seg_dir = os.path.join(case_dir, "segmentations")
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


def run_ui(context, preview_path=""):
    from PySide6 import QtCore, QtGui, QtWidgets

    mode = str(context.get("mode") or "import_batch")
    status_path = context.get("status_path")
    state_path = context.get("state_path")
    remembered = read_json(state_path, {}) or {}
    remembered_mode = remembered.get(mode) or {}

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Mimics Data Paths")
    app.setStyle("Fusion")
    window = QtWidgets.QDialog()
    window.setObjectName("ioWindow")
    window.setModal(False)
    window.setMinimumWidth(680)
    window.resize(720, 455 if mode == "export_masks" else 440)
    window.setWindowTitle({
        "import_batch": "Import Dataset",
        "import_single": "Import Single Case",
        "export_masks": "Export Masks",
    }.get(mode, "Data Paths"))
    window.setStyleSheet("""
        QDialog#ioWindow { background: #f4f6f8; color: #17212b; }
        QLabel { color: #273442; font-size: 13px; }
        QLabel#title { color: #132231; font-size: 23px; font-weight: 650; }
        QLabel#subtitle { color: #667482; font-size: 13px; }
        QLabel#section { color: #344453; font-size: 12px; font-weight: 650; }
        QFrame#panel { background: #ffffff; border: 1px solid #dce2e7; border-radius: 6px; }
        QLineEdit { min-height: 34px; padding: 0 10px; border: 1px solid #c8d1d9; border-radius: 4px; background: #ffffff; selection-background-color: #1d6f8a; }
        QLineEdit:focus { border: 1px solid #1d6f8a; }
        QPushButton { min-height: 34px; padding: 0 14px; border: 1px solid #bcc7cf; border-radius: 4px; background: #ffffff; color: #263745; }
        QPushButton:hover { background: #edf3f5; border-color: #8da1ad; }
        QPushButton#primary { background: #176b75; border-color: #176b75; color: #ffffff; font-weight: 650; min-width: 126px; }
        QPushButton#primary:hover { background: #115963; }
        QPushButton#primary:disabled { background: #a9b7bc; border-color: #a9b7bc; }
        QRadioButton, QCheckBox { spacing: 8px; color: #344453; }
        QRadioButton::indicator, QCheckBox::indicator { width: 16px; height: 16px; }
        QLabel#hint { color: #75838f; font-size: 12px; }
        QLabel#preview { color: #176b75; font-size: 12px; font-weight: 600; }
    """)

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

    def path_row(label, initial, browse_folder=True, file_or_folder=False, hint=""):
        form.addWidget(_label(QtWidgets, label, "section"))
        row = QtWidgets.QHBoxLayout()
        edit = QtWidgets.QLineEdit(str(initial or ""))
        edit.setClearButtonEnabled(True)
        button = QtWidgets.QPushButton("Browse...")
        row.addWidget(edit, 1)
        row.addWidget(button)
        form.addLayout(row)
        if hint:
            form.addWidget(_label(QtWidgets, hint, "hint"))

        def browse():
            current = edit.text().strip()
            # Fall back to home if the field is empty or points somewhere that
            # no longer exists, so the native dialog never opens on a huge or
            # stale root that makes it hang.
            if not current or not os.path.exists(current):
                current = str(Path.home())
            # Avoid opening the dialog on a drive root (e.g. E:\) which can
            # cause the native Shell folder dialog to enumerate the entire
            # drive and freeze.  Fall back to the user home directory instead.
            try:
                if os.path.isdir(current) and os.path.dirname(current.rstrip("\\/")) == current.rstrip("\\/"):
                    current = str(Path.home())
            except Exception:
                pass
            # Use Qt's own dialog instead of the Windows native Shell dialog.
            # The native IFileDialog can hang on large/root directories because
            # the Shell tries to enumerate every child for icons/thumbnails.
            dont_use_native = QtWidgets.QFileDialog.DontUseNativeDialog
            if file_or_folder:
                menu = QtWidgets.QMenu(button)
                choose_file = menu.addAction("Choose image file")
                choose_folder = menu.addAction("Choose case or DICOM folder")
                action = menu.exec(button.mapToGlobal(QtCore.QPoint(0, button.height())))
                if action == choose_file:
                    value, _ = QtWidgets.QFileDialog.getOpenFileName(window, "Choose 3D image", current, "Medical volumes (*.nii *.nii.gz *.mha *.mhd *.nrrd *.dcm);;All files (*)", "", dont_use_native)
                elif action == choose_folder:
                    value = QtWidgets.QFileDialog.getExistingDirectory(window, "Choose case or DICOM folder", current, dont_use_native)
                else:
                    value = ""
            elif browse_folder:
                value = QtWidgets.QFileDialog.getExistingDirectory(window, label, current, dont_use_native)
            else:
                value, _ = QtWidgets.QFileDialog.getOpenFileName(window, label, current, "", "", dont_use_native)
            if value:
                edit.setProperty("chosenByBrowse", True)
                edit.setText(str(value))
        button.clicked.connect(browse)
        return edit

    source_initial = context.get("source_initial") or remembered_mode.get("source_path", "")
    if mode in ("import_single", "export_masks"):
        source_edit = path_row("SOURCE IMAGE OR CASE", source_initial, file_or_folder=True, hint="Supported: NIfTI, MHA/MHD, NRRD, a DICOM file, case folder, or DICOM series folder.")
    else:
        source_edit = path_row("SOURCE DATASET", source_initial, hint="Required. This path is never modified by import.")

    output_initial = context.get("output_initial") or remembered_mode.get("output_path", "")
    output_edit = path_row("MCS OUTPUT FOLDER" if mode.startswith("import") else "EXPORT ROOT", output_initial, hint="Optional to change. A safe default is filled automatically from the source path.")
    output_preview = _label(QtWidgets, "", "preview")
    output_preview.setWordWrap(True)
    form.addWidget(output_preview)

    policy_row = None
    skip_radio = overwrite_radio = None
    if mode == "export_masks":
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

    source_edit.textChanged.connect(lambda _text: refresh_default())
    output_edit.textChanged.connect(output_edited)
    footer = QtWidgets.QHBoxLayout()
    remember = QtWidgets.QCheckBox("Remember these folders on this workstation")
    remember.setChecked(bool(remembered_mode))
    footer.addWidget(remember)
    footer.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    submit = QtWidgets.QPushButton("Start Import" if mode.startswith("import") else "Start Export")
    submit.setObjectName("primary")
    footer.addWidget(cancel)
    footer.addWidget(submit)
    root.addLayout(footer)

    def cancel_window():
        if status_path:
            write_json(status_path, {"status": "cancelled", "updated_at_epoch": time.time()})
        window.accept()

    def submit_window():
        source = os.path.abspath(os.path.expanduser(source_edit.text().strip()))
        output = os.path.abspath(os.path.expanduser(output_edit.text().strip()))
        if not os.path.exists(source):
            QtWidgets.QMessageBox.warning(window, "Source Not Found", "Choose an existing source file or folder.")
            return
        try:
            if not os.path.isdir(output):
                os.makedirs(output)
            fd, probe_path = tempfile.mkstemp(prefix=".mimics_write_test_", dir=output)
            os.close(fd)
            os.remove(probe_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                window,
                "Output Not Writable",
                "The output folder could not be created or written.\n\n{0}\n\n{1}".format(output, exc),
            )
            return
        selection = {"source_path": source, "output_path": output, "remember": bool(remember.isChecked())}
        if mode in ("import_single", "export_masks"):
            case_info = discover_single_source(source)
            if not case_info:
                QtWidgets.QMessageBox.warning(window, "Unsupported Source", "No supported 3D image or DICOM folder was found.")
                return
            selection["case_info"] = case_info
        if mode == "export_masks":
            selection["conflict_policy"] = "overwrite" if overwrite_radio.isChecked() else "skip"
        if remember.isChecked() and state_path:
            state = read_json(state_path, {}) or {}
            state[mode] = selection
            write_json(state_path, state)
        write_json(status_path, {"status": "submitted", "selection": selection, "updated_at_epoch": time.time()})
        window.accept()

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
