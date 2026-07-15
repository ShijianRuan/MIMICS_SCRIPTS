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
BROWSER_FILE_SUFFIXES = MASK_SUFFIXES + (".dcm",)


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


def choose_path_without_shell(QtCore, QtWidgets, parent, title, initial, allow_file=False):
    """Browse paths without Windows Shell enumeration on the GUI thread."""
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(760, 520)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 18, 18, 16)
    layout.setSpacing(10)

    path_row = QtWidgets.QHBoxLayout()
    path_edit = QtWidgets.QLineEdit(str(initial or Path.home()))
    path_edit.setClearButtonEnabled(True)
    paste = QtWidgets.QPushButton("Paste")
    go = QtWidgets.QPushButton("Go")
    path_row.addWidget(path_edit, 1)
    path_row.addWidget(paste)
    path_row.addWidget(go)
    layout.addLayout(path_row)

    nav = QtWidgets.QHBoxLayout()
    locations = QtWidgets.QComboBox()
    if os.name == "nt":
        # Listing all letters is instantaneous; unavailable drives fail only
        # when explicitly opened instead of delaying the whole dialog.
        locations.addItems(["{0}:\\".format(chr(code)) for code in range(ord("A"), ord("Z") + 1)])
    else:
        locations.addItems([str(Path.home()), os.path.abspath(os.sep)])
    up = QtWidgets.QPushButton("Up")
    refresh = QtWidgets.QPushButton("Refresh")
    nav.addWidget(locations, 1)
    nav.addWidget(up)
    nav.addWidget(refresh)
    layout.addLayout(nav)

    entries = QtWidgets.QListWidget()
    entries.setAlternatingRowColors(True)
    entries.setUniformItemSizes(True)
    layout.addWidget(entries, 1)
    status = QtWidgets.QLabel("")
    status.setObjectName("hint")
    layout.addWidget(status)

    footer = QtWidgets.QHBoxLayout()
    footer.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    choose = QtWidgets.QPushButton("Use Selected File" if allow_file else "Use This Folder")
    choose.setObjectName("primary")
    footer.addWidget(cancel)
    footer.addWidget(choose)
    layout.addLayout(footer)

    results = Queue()
    state = {"generation": 0, "path": "", "selected": ""}
    try:
        user_role = QtCore.Qt.ItemDataRole.UserRole
    except AttributeError:
        user_role = QtCore.Qt.UserRole

    def normalize(value):
        return os.path.abspath(os.path.expandvars(os.path.expanduser(str(value or "").strip())))

    def scan(value):
        target = normalize(value)
        state["generation"] += 1
        generation = state["generation"]
        state["path"] = target
        path_edit.setText(target)
        entries.clear()
        entries.setEnabled(False)
        status.setText("Loading folder in the background...")

        def worker():
            rows = []
            error = ""
            truncated = False
            try:
                with os.scandir(target) as iterator:
                    for index, entry in enumerate(iterator):
                        if index >= 2000:
                            truncated = True
                            break
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            is_dir = False
                        if is_dir or (allow_file and entry.name.lower().endswith(BROWSER_FILE_SUFFIXES)):
                            rows.append((not is_dir, entry.name, entry.path))
                rows.sort(key=lambda item: (item[0], item[1].lower()))
            except Exception as exc:
                error = str(exc)
            results.put((generation, target, rows, truncated, error))

        thread = threading.Thread(target=worker, name="path-browser-scan")
        thread.daemon = True
        thread.start()

    poller = QtCore.QTimer(dialog)

    def poll_results():
        while True:
            try:
                generation, target, rows, truncated, error = results.get_nowait()
            except Empty:
                return
            if generation != state["generation"]:
                continue
            entries.setEnabled(True)
            if error:
                status.setText("Cannot read this location. Paste an exact path or choose another drive. {0}".format(error))
                continue
            for is_file, name, full_path in rows:
                item = QtWidgets.QListWidgetItem(("[File] " if is_file else "[Folder] ") + name)
                item.setData(user_role, (is_file, full_path))
                entries.addItem(item)
            suffix = " Showing the first 2,000 entries; paste an exact path for items not shown." if truncated else ""
            status.setText("{0} item(s).{1}".format(len(rows), suffix))

    poller.timeout.connect(poll_results)
    poller.start(80)

    def paste_path():
        text = QtWidgets.QApplication.clipboard().text().strip()
        if text:
            path_edit.setText(text)
            normalized = normalize(text)
            if allow_file and normalized.lower().endswith(BROWSER_FILE_SUFFIXES):
                scan(os.path.dirname(normalized))
            else:
                scan(normalized)

    def selected_data():
        item = entries.currentItem()
        return item.data(user_role) if item else None

    def activate_item(item):
        is_file, full_path = item.data(user_role)
        if is_file:
            if allow_file:
                state["selected"] = full_path
                dialog.accept()
        else:
            scan(full_path)

    def accept_value():
        data = selected_data()
        if allow_file and data and data[0]:
            state["selected"] = data[1]
            dialog.accept()
            return
        typed = normalize(path_edit.text())
        if allow_file and os.path.isfile(typed):
            state["selected"] = typed
            dialog.accept()
            return
        if os.path.isdir(typed):
            state["selected"] = typed
            dialog.accept()
            return
        QtWidgets.QMessageBox.warning(dialog, "Path Not Found", "Paste or select an existing file or folder.")

    paste.clicked.connect(paste_path)
    go.clicked.connect(lambda: scan(path_edit.text()))
    path_edit.returnPressed.connect(lambda: scan(path_edit.text()))
    locations.activated.connect(lambda _index: scan(locations.currentText()))
    up.clicked.connect(lambda: scan(os.path.dirname(state["path"].rstrip("\\/")) or state["path"]))
    refresh.clicked.connect(lambda: scan(state["path"]))
    entries.itemDoubleClicked.connect(activate_item)
    cancel.clicked.connect(dialog.reject)
    choose.clicked.connect(accept_value)
    if allow_file and initial and str(initial).lower().endswith(BROWSER_FILE_SUFFIXES):
        initial_folder = os.path.dirname(os.path.abspath(str(initial)))
    elif initial:
        initial_folder = str(initial)
    else:
        initial_folder = str(Path.home())
    scan(initial_folder)
    return state["selected"] if dialog.exec() == QtWidgets.QDialog.Accepted else ""


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
    window.resize(760, 455 if mode == "export_masks" else 560)
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
        paste_button = QtWidgets.QPushButton("Paste")
        button = QtWidgets.QPushButton("Browse...")
        row.addWidget(edit, 1)
        row.addWidget(paste_button)
        row.addWidget(button)
        form.addLayout(row)
        if hint:
            form.addWidget(_label(QtWidgets, hint, "hint"))

        def browse():
            current = edit.text().strip()
            if not current:
                current = str(Path.home())
            value = choose_path_without_shell(
                QtCore,
                QtWidgets,
                window,
                label,
                current,
                allow_file=bool(file_or_folder or not browse_folder),
            )
            if value:
                edit.setProperty("chosenByBrowse", True)
                edit.setText(str(value))

        def paste_value():
            value = QtWidgets.QApplication.clipboard().text().strip()
            if value:
                edit.setProperty("chosenByBrowse", True)
                edit.setText(value)

        paste_button.clicked.connect(paste_value)
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

    submission_results = Queue()
    submission_state = {"running": False}
    submission_timer = QtCore.QTimer(window)

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
        window.accept()

    submission_timer.timeout.connect(poll_submission)
    submission_timer.start(80)

    def submit_window():
        if submission_state["running"]:
            return
        source = os.path.abspath(os.path.expanduser(source_edit.text().strip()))
        output = os.path.abspath(os.path.expanduser(output_edit.text().strip()))
        selection = {"source_path": source, "output_path": output, "remember": bool(remember.isChecked())}
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
