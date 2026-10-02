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
    PALETTE,
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
    a case folder or a dataset root). Multiple dropped paths are ALWAYS the
    user's exact case list (F10): each path becomes its own single-case
    import, never the shared parent — dropping 2 of 10 cases in a folder must
    not escalate to importing all 10. The batch flow stays reachable by
    dropping the dataset root itself. Duplicate paths are dropped.
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
    return "multi_single", None, list(dict.fromkeys(paths))


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


def _batch_runtime_state(source_path, output_path):
    """Authoritative progress for a batch drop, from the queue runtime dir.

    Batch imports write their per-case status into the import queue's local
    runtime directory (_mcs_queue_done.json / _mcs_batch_status.json); the
    drop status file only records "running" at submit time. Reading those
    markers here lets the recent-drops row show the real final state
    (done/failed/cancelled) without the batch CLI writing into the drop
    status dir (R61-8). Returns {} when the queue markers say nothing.
    """
    if not output_path:
        return {}
    try:
        import runtime_common
        runtime_dir = runtime_common.import_queue_runtime_dir(_ROOT, output_path)
    except Exception:
        return {}
    done = read_json(os.path.join(runtime_dir, "_mcs_queue_done.json"), {}) or {}
    if done.get("status"):
        return done
    batch = read_json(os.path.join(runtime_dir, "_mcs_batch_status.json"), {}) or {}
    if batch.get("status"):
        return batch
    return {}


def _apply_batch_runtime_state(status, error, payload, source_path):
    """Fold authoritative queue state into a batch drop's display fields.

    A batch status file that still says "running" is not evidence the import
    is still running - the batch CLI never rewrites it (R61-8). Trust the
    queue runtime markers instead when they exist.
    """
    if status != "running":
        return status, error
    runtime_state = _batch_runtime_state(
        source_path, payload.get("output_path")
    )
    if not runtime_state:
        return status, error
    runtime_status = str(runtime_state.get("status") or "").lower()
    if not runtime_status:
        return status, error
    label = {
        "done": "completed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "closed": "completed",
        "failed": "failed",
        "error": "failed",
        "creating": "running",
        "running": "running",
        "idle": "running",
        "recovering": "running",
        "restarting": "running",
    }.get(runtime_status, "")
    if not label:
        return status, error
    if not error and label in ("failed", "cancelled"):
        message = str(runtime_state.get("error") or runtime_state.get("reason") or "")
        if message:
            error = message.splitlines()[0][:160]
    return label, error


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
        # Batch drops freeze at "running" (the batch CLI never rewrites the
        # drop status file); derive their real state from the queue runtime
        # markers instead (R61-8).
        if name.endswith("_batch_status.json"):
            status, error = _apply_batch_runtime_state(status, error, payload, source)
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
            # The batch CLI never rewrites this file; collect_recent_drops
            # uses output_path to find the queue runtime markers that hold
            # the real final state (R61-8).
            "output_path": selection["output_path"],
            "source_path": selection["source_path"],
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
    window.setWindowTitle("拖入导入")
    window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
    window.setMinimumSize(420, 320)
    root = QtWidgets.QVBoxLayout(window)

    title = QtWidgets.QLabel("拖入导入")
    title.setObjectName("title")
    root.addWidget(title)
    hint = QtWidgets.QLabel(
        "把文件或文件夹拖到这里，或粘贴路径（Ctrl+V，每行一条）。窗口空闲时自动退出。"
    )
    hint.setObjectName("subtitle")
    hint.setWordWrap(True)
    root.addWidget(hint)

    dropzone = QtWidgets.QLabel("将图像、病例文件夹或数据集文件夹拖到这里")
    dropzone.setObjectName("dropzone")
    dropzone.setAlignment(QtCore.Qt.AlignCenter)
    dropzone.setWordWrap(True)
    root.addWidget(dropzone, 1)

    recognition = QtWidgets.QLabel("")
    recognition.setObjectName("preview")
    recognition.setWordWrap(True)
    root.addWidget(recognition)

    output_row = QtWidgets.QHBoxLayout()
    output_label = QtWidgets.QLabel("输出文件夹")
    # Only a folder the user chose themselves sticks across sessions (same
    # rule as io_path_setup_ui, R61-7): a remembered computed default would
    # pin every new dataset to the previous dataset's output folder.
    output_edit = QtWidgets.QLineEdit(
        remembered_mode.get("output_path") or ""
        if remembered_mode.get("output_custom")
        else ""
    )
    output_edit.setPlaceholderText("可选——默认从来源自动填写")
    browse = QtWidgets.QPushButton("浏览...")
    output_row.addWidget(output_label)
    output_row.addWidget(output_edit, 1)
    output_row.addWidget(browse)
    root.addLayout(output_row)

    mask_row = QtWidgets.QHBoxLayout()
    mask_all = QtWidgets.QRadioButton("全部 Mask")
    mask_none = QtWidgets.QRadioButton("仅图像")
    remembered_masks = str(remembered_mode.get("mask_selection", "all") or "all")
    mask_none.setChecked(remembered_masks.lower() == "none")
    mask_all.setChecked(not mask_none.isChecked())
    mask_row.addWidget(mask_all)
    mask_row.addWidget(mask_none)
    mask_row.addStretch(1)
    root.addLayout(mask_row)

    # -- Recent drops activity list ----------------------------------------
    recent_header = QtWidgets.QLabel("最近拖入")
    recent_header.setObjectName("subtitle")
    root.addWidget(recent_header)
    recent_list = QtWidgets.QListWidget()
    recent_list.setMaximumHeight(96)
    recent_list.setToolTip("双击一行可打开对应的导入日志")
    root.addWidget(recent_list)

    actions = QtWidgets.QHBoxLayout()
    actions.addStretch(1)
    clear = QtWidgets.QPushButton("清空")
    submit = QtWidgets.QPushButton("导入")
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
            dropzone.setText("将图像、病例文件夹或数据集文件夹拖到这里")
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
        elif kind == "multi_single":
            # F10: a multi-selection is the user's exact case list — probe
            # each dropped path so a mixed selection is reportable per item
            # and stays submittable when at least one path is valid.
            results = []
            valid = 0
            for path in paths:
                case_info = io_ui.discover_single_source(path)
                if case_info:
                    valid += 1
                    results.append("{0} ✓".format(case_info["case_id"]))
                else:
                    results.append("{0} ✗ (无可用图像)".format(
                        os.path.basename(path.rstrip("\\/")) or path))
            summary = "多选 {0} 例：{1}".format(len(paths), "、".join(results[:6]))
            if len(results) > 6:
                summary += " …"
            set_recognition(summary, valid < len(paths))
            dropzone.setText("Multi: {0} path(s)".format(len(paths)))
            if not valid:
                submit.setEnabled(False)
                return
        else:
            display = paths[0]
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
        # R61-7: a new drop means a new dataset, so the computed default
        # follows it — unless the user typed/picked a folder themselves this
        # session (that choice stays). The pre-submit autofill from a
        # previous drop is not a user choice and must not stick.
        mode = "import_batch" if kind == "batch" else "import_single"
        default_output = io_ui.source_default_output(mode, state["source"] or paths[0])
        if not output_custom["value"]:
            output_edit.setText(default_output)
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
    # True once the user typed or browsed an output folder this session.
    output_custom = {"value": False}

    def output_edited():
        # hasFocus filters out programmatic setText from the autofill; a
        # programmatic default is not a user choice and must not stick.
        if output_edit.hasFocus():
            output_custom["value"] = True

    output_edit.textChanged.connect(output_edited)
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

    paste_button = QtWidgets.QPushButton("粘贴路径")
    paste_button.clicked.connect(paste_paths)
    output_row.insertWidget(3, paste_button)

    def browse_output():
        # A native dialog can hang on disconnected drives / SMB folders; the
        # async helper isolates it in its own process so this window and
        # Mimics stay responsive.
        def _chosen(value):
            if value:
                output_custom["value"] = True
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
                window, "导入失败",
                "导入无法启动：\n\n{0}".format(exc),
            )
            return
        # Remember for next time (same state file as the path-setup UI).
        remembered["drop_import"] = {
            "output_path": output,
            "mask_selection": mask_selection,
            # True only when the user typed/picked a folder different from
            # the computed default; a default must not stick (R61-7).
            "output_custom": bool(
                os.path.normcase(output)
                != os.path.normcase(
                    io_ui.source_default_output(
                        "import_batch" if kind == "batch" else "import_single",
                        source,
                    )
                )
            ),
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
            window, "导入已开始",
            "导入已开始：\n\n{0}\n\n任务在后台继续进行；Mimics 工程将生成于：\n{1}".format("\n".join(lines), output),
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
                item.setForeground(QtGui.QColor(PALETTE["danger"]))
            elif entry["status"] == "completed":
                item.setForeground(QtGui.QColor(PALETTE["teal"]))
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
