#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Floating drag-and-drop import window (P3).

Mimics' scripting API cannot receive native drag events, so this small
always-on-top external window receives them instead. Drop files, folders,
or a multi-selection (or paste one path per line) and the window:

1. classifies the payload: one volume file, one case folder, a DICOM
   series folder, or a dataset root with several case folders;
2. resolves it against the dataset profile (dataset_profiles.json, shared
   with every other discovery site);
3. shows a one-line recognition summary plus any anomalies;
4. on confirm, submits to the existing import workers - single-case cases
   go through single_case_import_worker.py, multi-case roots go through
   mimics_batch_cli.py prepare-import. No new pipeline code.

It exits by itself after IDLE_TIMEOUT_SECONDS without user interaction, so
it never becomes another resident service to manage. Last-used settings are
remembered in ui_state/io_paths.json (same file the path-setup UI uses).

Run inside Mimics via 01_Data/07_Quick_Drop_Import.py, or standalone:
    python tools/import_drop_window.py [--context <ctx.json>]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Embeddable Python can omit the script directory from sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _candidate in (_HERE, _ROOT, os.path.join(_ROOT, "runtime_py35")):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    configure_application,
    stylesheet as shared_stylesheet,
)

import io_path_setup_ui as io_ui  # noqa: E402

IDLE_TIMEOUT_SECONDS = 30 * 60  # half an hour without input -> quiet exit
POLL_MILLISECONDS = 250


def read_json(path, default=None):
    try:
        import json
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def write_json(path, value):
    import json
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(tmp, str(path))


def state_file_path():
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        candidate = os.path.join(local_appdata, "Mimics-Script", "ui_state", "io_paths.json")
        return candidate
    return os.path.join(_ROOT, ".mimics_runtime", "ui_state", "io_paths.json")


def external_python():
    """Locate the external tools Python that runs the import workers."""
    candidates = [
        os.path.join(_ROOT, "python_env", "python.exe"),
        os.path.join(_ROOT, "nninteractive_env", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def classify_payload(paths):
    """Turn dropped paths into (kind, source) where kind is single or batch.

    One dropped path is inspected directly (file -> single, folder -> could be
    a case folder or a dataset root). Multiple dropped paths: if they share a
    parent and look like case folders, treat the parent as a dataset root;
    otherwise each path is handled as its own single-case import.
    """
    paths = [p for p in paths if p]
    if not paths:
        return None, None, []
    if len(paths) == 1:
        path = paths[0]
        if os.path.isdir(path):
            # A dataset root: several child case folders each holding images.
            child_dirs = _child_case_dirs(path)
            if len(child_dirs) >= 2:
                return "batch", path, child_dirs
            return "single", path, child_dirs
        return "single", path, []
    parents = {os.path.dirname(p) for p in paths}
    if len(parents) == 1:
        parent = parents.pop()
        return "batch", parent, [p for p in paths if os.path.isdir(p)] or paths
    return "multi_single", None, paths


def _child_case_dirs(root, limit=3):
    """Return up to `limit` child directories that look like case folders."""
    from dataset_profiles import load_profile

    profile = load_profile()
    found = []
    try:
        for name in sorted(os.listdir(root)):
            if name in profile["exclude_dirs"]:
                continue
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if io_ui.discover_single_source(full):
                found.append(full)
            if len(found) >= limit:
                break
    except OSError:
        pass
    return found


def build_selection(kind, source, child_dirs, remembered):
    """Build the selection payload for the import worker(s)."""
    output_dir = remembered.get("output_path") or io_ui.source_default_output(
        "import_batch" if kind == "batch" else "import_single", source
    )
    mask_selection = remembered.get("mask_selection") or "all"
    return {
        "kind": kind,
        "source_path": source,
        "output_path": output_dir,
        "mask_selection": mask_selection,
    }


DROP_IMPORT_RETENTION_DAYS = 30


def prune_drop_import_state(status_dir, retention_days=DROP_IMPORT_RETENTION_DAYS):
    """Delete drop-import status/log/selection files past the retention window.

    Every drop writes 3-4 files here (status, selection with source paths,
    stop marker, log). Selection JSONs contain patient file paths, so they
    must not accumulate indefinitely.
    """
    try:
        cutoff = time.time() - retention_days * 86400
        for name in os.listdir(status_dir):
            path = os.path.join(status_dir, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                continue
    except OSError:
        pass


RECENT_DROP_LIMIT = 8


def collect_recent_drops(status_dir, limit=RECENT_DROP_LIMIT):
    """Summarize the newest drop-import status files for the activity list.

    Returns a list of dicts: case_id, status/phase, error line, and the log
    path so a click can open the full log. Dead or unreadable entries are
    skipped rather than shown as noise.
    """
    entries = []
    try:
        names = os.listdir(status_dir)
    except OSError:
        return entries
    candidates = []
    for name in names:
        if not name.endswith("_status.json"):
            continue
        path = os.path.join(status_dir, name)
        try:
            candidates.append((os.path.getmtime(path), name, path))
        except OSError:
            continue
    candidates.sort(reverse=True)
    for _mtime, name, path in candidates:
        if len(entries) >= limit:
            break
        payload = read_json(path, {}) or {}
        if not payload:
            continue
        # The sibling selection file holds the source path for the case id.
        stem = name[: -len("_status.json")]
        selection = read_json(
            os.path.join(status_dir, stem + "_selection.json"), {}
        ) or {}
        source = str(selection.get("source_path") or "")
        case_id = str(payload.get("case_id") or "") or (
            os.path.basename(source.rstrip("\\/")) or stem
        )
        status = str(payload.get("status") or "").lower()
        if not status:
            status = "unknown"
        error = ""
        # Cancelled rows also carry a human-readable reason in `error`.
        message = str(payload.get("error") or payload.get("message") or "")
        if status in ("failed", "error", "cancelled") and message:
            error = message.splitlines()[0][:160]
        entries.append({
            "case_id": case_id,
            "status": status,
            "phase": str(payload.get("phase") or ""),
            "error": error,
            "log_path": (
                payload.get("log_path")
                or os.path.join(status_dir, stem + ".log")
            ),
            "updated_at_epoch": payload.get("updated_at_epoch") or 0,
        })
    return entries


def submit_import(selection, context):
    """Launch the import worker(s). Returns a human-readable status line."""
    kind = selection.get("kind")
    python = external_python()
    status_dir = os.path.join(_ROOT, ".mimics_runtime", "drop_import")
    os.makedirs(status_dir, exist_ok=True)
    prune_drop_import_state(status_dir)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    if kind in ("single", "multi_single"):
        sources = (
            [selection["source_path"]]
            if kind == "single"
            else list(selection.get("paths") or [])
        )
        launched = []
        for index, source in enumerate(sources):
            case_info = io_ui.discover_single_source(source)
            if not case_info:
                launched.append((os.path.basename(source.rstrip("\\/")) or source, "no supported image"))
                continue
            case_id = case_info["case_id"]
            selection_path = os.path.join(status_dir, "{0}_{1}_selection.json".format(run_id, index))
            status_path = os.path.join(status_dir, "{0}_{1}_status.json".format(run_id, index))
            stop_path = os.path.join(status_dir, "{0}_{1}_stop.json".format(run_id, index))
            log_path = os.path.join(status_dir, "{0}_{1}.log".format(run_id, index))
            payload = {
                "source_path": source,
                "output_path": os.path.join(selection["output_path"]),
                "case_info": case_info,
                "mask_selection": selection.get("mask_selection") or "all",
                "project_root": _ROOT,
                "timeout_seconds": 7200.0,
            }
            write_json(selection_path, payload)
            write_json(status_path, {
                "status": "running", "phase": "starting_worker",
                "total": 1, "completed": 0, "failed": 0,
                "updated_at_epoch": time.time(),
            })
            try:
                subprocess.Popen(
                    [python, os.path.join(_HERE, "single_case_import_worker.py"),
                     "--selection-json", selection_path,
                     "--status-path", status_path,
                     "--stop-path", stop_path,
                     "--log-path", log_path],
                    cwd=_ROOT, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                launched.append((case_id, "submitted"))
            except Exception as exc:
                launched.append((case_id, "failed: {0}".format(exc)))
        return launched
    if kind == "batch":
        command = [
            python, os.path.join(_HERE, "mimics_batch_cli.py"),
            "prepare-import",
            "--ts-root", selection["source_path"],
            "--output-dir", selection["output_path"],
            "--masks", selection.get("mask_selection") or "all",
        ]
        log_path = os.path.join(status_dir, "{0}_batch.log".format(run_id))
        write_json(os.path.join(status_dir, "{0}_batch_status.json".format(run_id)), {
            "status": "running",
            "phase": "queued_to_prepare",
            "total": 0, "completed": 0, "failed": 0,
            "updated_at_epoch": time.time(),
        })
        log_handle = open(log_path, "w", encoding="utf-8")
        try:
            subprocess.Popen(
                command, cwd=_ROOT, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log_handle.close()
        return [("dataset", "submitted")]
    raise ValueError("unknown selection kind: {0}".format(kind))


def register_window_process():
    """Register in the process registry so the health panel can see us."""
    try:
        import resource_locks
        record = resource_locks.register_process(
            _ROOT, "external_ui", os.getpid(),
            parent_pid=os.getppid(),
            cleanup_policy="idle_timeout_s:{0}".format(IDLE_TIMEOUT_SECONDS),
        )
        return record.get("ownership_token") if record else ""
    except Exception:
        return ""


def unregister_window_process(ownership_token=""):
    try:
        import resource_locks
        resource_locks.unregister_process(
            _ROOT, "external_ui", os.getpid(),
            ownership_token=ownership_token or "",
        )
    except Exception:
        pass


class DropWindowStyle:
    STYLESHEET = shared_stylesheet() + """
    QLabel#dropzone {
        border: 2px dashed #94a3b8;
        border-radius: 8px;
        padding: 18px;
        color: #64748b;
        font-size: 13px;
    }
    QLabel#dropzone[active="true"] {
        border-color: #0f766e;
        color: #0f766e;
    }
    """


def run(context=None, preview_path=""):
    from PySide6 import QtCore, QtGui, QtWidgets

    context = context or {}
    remembered = read_json(state_file_path(), {}) or {}
    remembered_mode = remembered.get("drop_import", {}) or {}

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    app.setStyleSheet(DropWindowStyle.STYLESHEET)

    window = QtWidgets.QWidget()
    window.setWindowTitle("Drop to Import")
    window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
    window.setMinimumSize(420, 320)
    root = QtWidgets.QVBoxLayout(window)

    title = QtWidgets.QLabel("Drop to Import")
    title.setObjectName("title")
    root.addWidget(title)
    hint = QtWidgets.QLabel(
        "Drag files or folders here, or paste paths (Ctrl+V, one per line). "
        "The window exits automatically when idle."
    )
    hint.setObjectName("subtitle")
    hint.setWordWrap(True)
    root.addWidget(hint)

    dropzone = QtWidgets.QLabel("Drop images, case folders, or a dataset folder here")
    dropzone.setObjectName("dropzone")
    dropzone.setAlignment(QtCore.Qt.AlignCenter)
    dropzone.setWordWrap(True)
    root.addWidget(dropzone, 1)

    recognition = QtWidgets.QLabel("")
    recognition.setObjectName("preview")
    recognition.setWordWrap(True)
    root.addWidget(recognition)

    output_row = QtWidgets.QHBoxLayout()
    output_label = QtWidgets.QLabel("Output folder")
    output_edit = QtWidgets.QLineEdit(remembered_mode.get("output_path") or "")
    output_edit.setPlaceholderText("Optional - filled automatically from the source")
    browse = QtWidgets.QPushButton("Browse...")
    output_row.addWidget(output_label)
    output_row.addWidget(output_edit, 1)
    output_row.addWidget(browse)
    root.addLayout(output_row)

    mask_row = QtWidgets.QHBoxLayout()
    mask_all = QtWidgets.QRadioButton("All masks")
    mask_none = QtWidgets.QRadioButton("Images only")
    remembered_masks = str(remembered_mode.get("mask_selection", "all") or "all")
    mask_none.setChecked(remembered_masks.lower() == "none")
    mask_all.setChecked(not mask_none.isChecked())
    mask_row.addWidget(mask_all)
    mask_row.addWidget(mask_none)
    mask_row.addStretch(1)
    root.addLayout(mask_row)

    # -- Recent drops activity list ----------------------------------------
    recent_header = QtWidgets.QLabel("Recent drops")
    recent_header.setObjectName("subtitle")
    root.addWidget(recent_header)
    recent_list = QtWidgets.QListWidget()
    recent_list.setMaximumHeight(96)
    recent_list.setToolTip("Double-click a row to open its import log")
    root.addWidget(recent_list)

    actions = QtWidgets.QHBoxLayout()
    actions.addStretch(1)
    clear = QtWidgets.QPushButton("Clear")
    submit = QtWidgets.QPushButton("Import")
    submit.setObjectName("primary")
    submit.setEnabled(False)
    actions.addWidget(clear)
    actions.addWidget(submit)
    root.addLayout(actions)

    state = {
        "paths": [],
        "kind": None,
        "source": None,
        "last_activity": time.time(),
        "submitted": False,
    }

    def touch():
        state["last_activity"] = time.time()

    def set_recognition(text, warning=False):
        recognition.setText(text)
        recognition.setProperty("warning", warning)
        recognition.style().unpolish(recognition)
        recognition.style().polish(recognition)

    def analyze(paths):
        paths = [p for p in paths if p]
        if not paths:
            state["paths"] = []
            state["kind"] = None
            state["source"] = None
            submit.setEnabled(False)
            dropzone.setText("Drop images, case folders, or a dataset folder here")
            set_recognition("")
            return
        state["paths"] = paths
        kind, source, child_dirs = classify_payload(paths)
        state["kind"] = kind
        state["source"] = source
        if kind == "batch":
            result = io_ui.summarize_dataset(source)
            summary = result["summary"] if result else "Dataset folder recognized."
            warnings = result.get("warnings") or [] if result else []
            if warnings:
                summary += "\n" + "\n".join("⚠ " + w for w in warnings[:5])
            dropzone.setText("Dataset: {0}".format(source))
            set_recognition(summary, bool(warnings))
        else:
            display = paths[0] if kind == "single" else "{0} path(s)".format(len(paths))
            case_info = io_ui.discover_single_source(paths[0]) if kind == "single" else None
            if case_info:
                mask_names = ", ".join(m["name"] for m in case_info["masks"][:5])
                summary = "Case '{0}' - image found{1}".format(
                    case_info["case_id"],
                    "; masks: {0}".format(mask_names) if mask_names else "",
                )
                set_recognition(summary, False)
            else:
                set_recognition("No supported image was recognized in the dropped item.", True)
                submit.setEnabled(False)
                dropzone.setText(display)
                return
            dropzone.setText(display)
        if not output_edit.text().strip():
            mode = "import_batch" if kind == "batch" else "import_single"
            output_edit.setText(io_ui.source_default_output(mode, state["source"] or paths[0]))
        submit.setEnabled(True)

    def add_paths(values):
        cleaned = []
        for value in values or []:
            text = str(value).strip().strip('"')
            if not text:
                continue
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(text)))
            if os.path.exists(expanded) and expanded not in cleaned:
                cleaned.append(expanded)
        if not cleaned:
            return
        touch()
        analyze(cleaned)

    # -- Drag & drop ------------------------------------------------------
    window.setAcceptDrops(True)

    def drag_enter(event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            dropzone.setProperty("active", True)
            dropzone.style().unpolish(dropzone)
            dropzone.style().polish(dropzone)

    def drag_leave(event):
        dropzone.setProperty("active", False)
        dropzone.style().unpolish(dropzone)
        dropzone.style().polish(dropzone)
        event.accept()

    def drop(event):
        dropzone.setProperty("active", False)
        dropzone.style().unpolish(dropzone)
        dropzone.style().polish(dropzone)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        add_paths(paths)
        event.acceptProposedAction()

    window.dragEnterEvent = drag_enter
    window.dragLeaveEvent = drag_leave
    window.dropEvent = drop

    # -- Paste ------------------------------------------------------------
    def paste_paths():
        raw = QtWidgets.QApplication.clipboard().text()
        values = [line.strip() for line in raw.replace("\r", "\n").split("\n") if line.strip()]
        if not values:
            set_recognition("The clipboard does not contain any paths.", True)
            return
        add_paths(values)

    paste_button = QtWidgets.QPushButton("Paste Paths")
    paste_button.clicked.connect(paste_paths)
    output_row.insertWidget(3, paste_button)

    def browse_output():
        # A native dialog can hang on disconnected drives / SMB folders; the
        # async helper isolates it in its own process so this window and
        # Mimics stay responsive.
        def _chosen(value):
            if value:
                output_edit.setText(str(value))

        choose_existing_directory_async(
            QtCore, QtWidgets, window, "Choose Output Folder",
            output_edit.text().strip() or str(Path.home()),
            _chosen,
            button=browse,
        )

    browse.clicked.connect(browse_output)

    def clear_all():
        touch()
        analyze([])

    clear.clicked.connect(clear_all)

    def do_submit():
        if not state["kind"]:
            return
        touch()
        kind = state["kind"]
        source = state["source"] or (state["paths"][0] if state["paths"] else "")
        output = os.path.abspath(os.path.expanduser(output_edit.text().strip()))
        if not output:
            output = io_ui.source_default_output(
                "import_batch" if kind == "batch" else "import_single", source
            )
            output_edit.setText(output)
        mask_selection = "none" if mask_none.isChecked() else "all"
        selection = {
            "kind": kind,
            "source_path": source,
            "output_path": output,
            "mask_selection": mask_selection,
            "paths": list(state["paths"]) if kind == "multi_single" else [],
        }
        try:
            launched = submit_import(selection, context)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                window, "Import Failed",
                "The import could not be started:\n\n{0}".format(exc),
            )
            return
        # Remember for next time (same state file as the path-setup UI).
        remembered["drop_import"] = {
            "output_path": output,
            "mask_selection": mask_selection,
        }
        try:
            os.makedirs(os.path.dirname(state_file_path()), exist_ok=True)
            write_json(state_file_path(), remembered)
        except Exception:
            pass
        lines = ["{0}: {1}".format(name, status) for name, status in launched[:10]]
        if len(launched) > 10:
            lines.append("... and {0} more".format(len(launched) - 10))
        QtWidgets.QMessageBox.information(
            window, "Import Started",
            "Import started:\n\n{0}\n\nProgress continues in the background; "
            "Mimics projects will appear in:\n{1}".format("\n".join(lines), output),
        )
        state["submitted"] = True
        analyze([])

    submit.clicked.connect(do_submit)

    # -- Recent drops polling ---------------------------------------------
    _STATUS_LABELS = {
        "running": "Importing",
        "queued": "Queued",
        "completed": "Done",
        "failed": "Failed",
        "cancelled": "Cancelled",
        "unknown": "Unknown",
    }

    def refresh_recent():
        status_dir = os.path.join(_ROOT, ".mimics_runtime", "drop_import")
        entries = collect_recent_drops(status_dir)
        recent_list.clear()
        for entry in entries:
            label = _STATUS_LABELS.get(entry["status"], entry["status"].title())
            line = "{0} - {1}".format(entry["case_id"], label)
            if entry["error"]:
                line += ": {0}".format(entry["error"])
            item = QtWidgets.QListWidgetItem(line)
            item.setData(QtCore.Qt.UserRole, entry["log_path"])
            if entry["status"] == "failed":
                item.setForeground(QtGui.QColor("#b91c1c"))
            elif entry["status"] == "completed":
                item.setForeground(QtGui.QColor("#0f766e"))
            recent_list.addItem(item)

    def open_recent_log(row_item):
        log_path = str(row_item.data(QtCore.Qt.UserRole) or "")
        if log_path and os.path.isfile(log_path):
            os.startfile(log_path)  # noqa: P102 - Windows shell open

    recent_list.itemDoubleClicked.connect(open_recent_log)

    recent_timer = QtCore.QTimer(window)
    recent_timer.setInterval(2000)
    recent_timer.timeout.connect(refresh_recent)
    recent_timer.start(2000)
    refresh_recent()

    # -- Idle timeout -----------------------------------------------------
    def check_idle():
        if time.time() - state["last_activity"] > IDLE_TIMEOUT_SECONDS and not state["paths"]:
            window.close()

    idle_timer = QtCore.QTimer(window)
    idle_timer.setInterval(5000)
    idle_timer.timeout.connect(check_idle)
    idle_timer.start(5000)

    ownership_token = register_window_process()
    app.aboutToQuit.connect(
        lambda: unregister_window_process(ownership_token) if ownership_token else None
    )

    if preview_path:
        window.show()
        app.processEvents()
        window.grab().save(preview_path)
        return 0
    window.show()
    return int(app.exec())


def main(argv):
    context = {}
    preview_path = ""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--context" and index + 1 < len(argv):
            context = read_json(argv[index + 1], {}) or {}
            index += 2
            continue
        if arg == "--preview" and index + 1 < len(argv):
            preview_path = argv[index + 1]
            index += 2
            continue
        index += 1
    return run(context, preview_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
