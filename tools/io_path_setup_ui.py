#!/usr/bin/env python3
"""External PySide6 path setup for Mimics import and mask export."""

from __future__ import annotations

import argparse
import errno
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
for _candidate in (_HERE, _ROOT):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import configure_application, stylesheet as shared_stylesheet


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def process_exists(pid):
    """Return whether the owning Mimics process is still alive."""
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            SYNCHRONIZE = 0x00100000
            WAIT_TIMEOUT = 0x00000102
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(SYNCHRONIZE, 0, value)
            if not handle:
                return ctypes.get_last_error() == 5
            try:
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(value, 0)
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    except Exception:
        return True


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


def choose_path_without_shell(
    QtCore, QtWidgets, parent, title, initial, allow_file=False, multi_file=False
):
    """Browse paths without Windows Shell enumeration on the GUI thread."""
    allow_file = bool(allow_file or multi_file)
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
    if multi_file:
        entries.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    layout.addWidget(entries, 1)
    status = QtWidgets.QLabel("")
    status.setObjectName("hint")
    layout.addWidget(status)

    footer = QtWidgets.QHBoxLayout()
    footer.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    if multi_file:
        choose_text = "Use Selected Files"
    else:
        choose_text = "Use Selected File" if allow_file else "Use This Folder"
    choose = QtWidgets.QPushButton(choose_text)
    choose.setObjectName("primary")
    footer.addWidget(cancel)
    footer.addWidget(choose)
    layout.addLayout(footer)

    results = Queue()
    state = {"generation": 0, "path": "", "selected": [] if multi_file else ""}
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

    def selected_files():
        rows = []
        for item in entries.selectedItems():
            data = item.data(user_role)
            if data and data[0]:
                rows.append(data[1])
        return rows

    def activate_item(item):
        is_file, full_path = item.data(user_role)
        if is_file:
            if allow_file and not multi_file:
                state["selected"] = full_path
                dialog.accept()
        else:
            scan(full_path)

    def accept_value():
        if multi_file:
            selected = selected_files()
            if selected:
                state["selected"] = selected
                dialog.accept()
                return
            typed = normalize(path_edit.text())
            if typed:
                state["selected"] = [typed]
                dialog.accept()
                return
            QtWidgets.QMessageBox.warning(dialog, "Files Required", "Select one or more Mask files.")
            return
        data = selected_data()
        if allow_file and data and data[0]:
            state["selected"] = data[1]
            dialog.accept()
            return
        typed = normalize(path_edit.text())
        if typed:
            state["selected"] = typed
            dialog.accept()
            return
        QtWidgets.QMessageBox.warning(dialog, "Path Required", "Paste or select a file or folder.")

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
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        return state["selected"]
    return [] if multi_file else ""


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
            current = edit.text().strip()
            if not current:
                current = str(Path.home())
            if os.path.isfile(current):
                current = os.path.dirname(current)
            value = QtWidgets.QFileDialog.getExistingDirectory(
                window,
                label,
                current,
                QtWidgets.QFileDialog.ShowDirsOnly,
            )
            if value:
                edit.setProperty("chosenByBrowse", True)
                edit.setText(str(value))

        def browse_file_path():
            current = edit.text().strip()
            if not current:
                current = str(Path.home())
            value, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
                window,
                label,
                current,
                "Medical volumes (*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz);;All files (*)",
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
        button.clicked.connect(browse_folder_path)
        if file_button is not None:
            file_button.clicked.connect(browse_file_path)
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
        form.addWidget(_label(QtWidgets, "MASKS TO EXPORT", "section"))
        export_mask_row = QtWidgets.QHBoxLayout()
        mask_all = QtWidgets.QRadioButton("All masks")
        mask_named = QtWidgets.QRadioButton("Selected names")
        mask_all.setChecked(True)
        export_mask_row.addWidget(mask_all)
        export_mask_row.addWidget(mask_named)
        export_mask_row.addStretch(1)
        form.addLayout(export_mask_row)
        mask_names = QtWidgets.QListWidget()
        mask_names.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        mask_names.setMaximumHeight(110)
        available_masks = [str(name) for name in (context.get("mask_names") or []) if str(name).strip()]
        mask_names.addItems(available_masks)
        mask_names.setEnabled(False)
        form.addWidget(mask_names)
        mask_named.toggled.connect(mask_names.setEnabled)
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
    footer_widget = QtWidgets.QWidget()
    footer = QtWidgets.QHBoxLayout(footer_widget)
    footer.setContentsMargins(0, 0, 0, 0)
    remember = QtWidgets.QCheckBox("Remember these folders on this workstation")
    remember.setChecked(bool(remembered_mode))
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
            detail += " - {0} completed, {1} failed, {2} total".format(completed, failed, total)
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
            if mask_named.isChecked():
                names = ",".join(item.text().strip() for item in mask_names.selectedItems())
                if not names:
                    QtWidgets.QMessageBox.warning(
                        window,
                        "Mask Names Required",
                        "Select one or more masks from the list.",
                    )
                    return
                selection["mask_selection"] = names
            else:
                selection["mask_selection"] = "all"

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
