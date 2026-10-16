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

Run inside Mimics via 01_Data/01_Import_Data.py, or standalone:
    python tools/import_drop_window.py [--context <ctx.json>]
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

# Embeddable Python can omit the script directory from sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _candidate in (_HERE, _ROOT, os.path.join(_ROOT, "runtime_py35")):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import (  # noqa: E402
    PALETTE,
    choose_existing_directory_async,
    choose_open_files_async,
    configure_application,
    clamp_window_to_screen,
    scroll_layout_body,
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


def expand_mask_paths(values):
    """Turn user-supplied mask paths (files or folders) into {name, path}.

    Used when automatic mask discovery found nothing (mask files live outside
    segmentations/, or sibling volumes suppressed discovery) and the user
    points at them explicitly. The explicit list REPLACES discovery — the
    user's list is the final list, no merging (F24: explicit over inferred).

    Files keep name=stem (D16: stem = Mimics Mask name = export label name).
    Folders expand one level, mirroring the segmentations/ scan. Anything
    unusable (nonexistent path, non-medical file, empty folder) is returned
    in ``invalid`` so the caller can fail that case closed instead of
    importing it silently maskless. Returns (masks, invalid).
    """
    masks = []
    invalid = []
    seen = set()
    for value in values or []:
        raw = str(value).strip().strip('"').strip("'")
        if not raw:
            continue
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(raw)))
        key = os.path.normcase(path)
        if key in seen:
            continue
        if os.path.isdir(path):
            found = 0
            try:
                for name in sorted(os.listdir(path)):
                    candidate = os.path.join(path, name)
                    if not io_ui._medical_file(candidate):
                        continue
                    candidate_key = os.path.normcase(candidate)
                    if candidate_key in seen:
                        found += 1
                        continue
                    seen.add(candidate_key)
                    masks.append({"name": io_ui._image_stem(name), "path": candidate})
                    found += 1
            except OSError:
                pass
            if not found:
                invalid.append(path)
        elif os.path.isfile(path) and io_ui._medical_file(path):
            seen.add(key)
            masks.append({"name": io_ui._image_stem(os.path.basename(path)), "path": path})
        else:
            invalid.append(path)
    return masks, invalid


def _normalized_mask_names(names):
    """Match rule the import worker applies (single_case_import_worker
    _filter_masks): case-insensitive, surrounding whitespace ignored."""
    return set(str(n).strip().casefold() for n in (names or []) if str(n).strip())


def cases_missing_selected_masks(case_mask_names, picked):
    """UI24: which cases would import as images only under a named selection.

    `case_mask_names` is the {case_id: [mask names]} map from
    summarize_dataset; `picked` the checked names. Returns the sorted list
    of case ids whose folder contains none of them. Empty list when there
    is nothing to warn about (or nothing known — no map, no warning).

    R02: matching is case-insensitive and whitespace-trimmed, identical to
    the worker's _filter_masks, so the pre-check and the actual import
    agree on what "has the selected mask" means.
    """
    if not case_mask_names or not picked:
        return []
    picked_set = _normalized_mask_names(picked)
    if not picked_set:
        return []
    return sorted(
        case for case, names in case_mask_names.items()
        if not (picked_set & _normalized_mask_names(names))
    )


def cases_partial_mask_report(case_mask_names, picked):
    """R02: per-case gaps for a named selection, for the pre-submit dialog.

    Returns (missing_cases, partial) where missing_cases is the sorted
    list of case ids that hold none of the picked masks (image-only
    import) and partial maps case id -> sorted picked names the case is
    missing some (but not all) of — the multi-organ partial loss the old
    one-line warning could not show. Matching follows the worker's rule
    (case-insensitive, trimmed).
    """
    if not case_mask_names or not picked:
        return [], {}
    picked_set = _normalized_mask_names(picked)
    if not picked_set:
        return [], {}
    missing = []
    partial = {}
    for case, names in case_mask_names.items():
        held = _normalized_mask_names(names)
        gap = sorted(picked_set - held)
        if not held & picked_set:
            missing.append(case)
        elif gap:
            partial[case] = gap
    return sorted(missing), partial


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


# Annotator-facing labels for the worker phases that matter during a drop
# import. Anything unknown keeps the raw phase so logs stay greppable.
_PHASE_LABELS = {
    "preparing": "准备数据",
    "waiting_for_background_mimics": "等待后台 Mimics",
    "creating_mcs": "生成工程文件",
}

_STATUS_LABELS = {
    "running": "导入中",
    "queued": "排队中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "unknown": "未知",
}


def format_recent_drop_line(entry, now_epoch=None):
    """Render one recent-drop entry as the annotator-facing activity line.

    Format: "14:32  病例名 — 导入中（准备数据）". The time prefix lets the
    annotator tell this run from an older one; the phase in brackets says
    WHERE the import is, not just that it is alive. Errors are moved into
    the tooltip (double-click opens the full log anyway) so a long failure
    reason no longer stretches the row into unreadable noise.
    """
    label = _STATUS_LABELS.get(entry["status"], entry["status"].title())
    line = "{0}  {1} — {2}".format(
        time.strftime("%H:%M", time.localtime(entry["updated_at_epoch"] or 0)),
        entry["case_id"],
        label,
    )
    phase = str(entry.get("phase") or "")
    phase_label = _PHASE_LABELS.get(phase, phase)
    if entry["status"] == "running" and phase_label:
        line += "（{0}）".format(phase_label)
    return line


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
        # P3-4: the last resort used to be the raw run id stem
        # (20261003T120000_abcd) — meaningless to the annotator. Strip the
        # timestamp prefix so at least the random suffix stays short noise.
        readable_stem = re.sub(r"^\d{8}T\d{6}_", "", stem)
        case_id = str(payload.get("case_id") or "") or (
            os.path.basename(source.rstrip("\\/")) or readable_stem
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
    # F35: a timestamp alone collides when two windows (or two quick
    # submissions) land in the same second — both workers would then read
    # and write the SAME selection/status/stop files (the later submit's
    # selection overwrites the earlier one before it is read). A
    # per-submission token makes every control file unique; the timestamp
    # stays as a sortable prefix.
    run_id = "{0}_{1}".format(
        time.strftime("%Y%m%dT%H%M%S"), os.urandom(4).hex())
    if kind in ("single", "multi_single"):
        sources = (
            [selection["source_path"]]
            if kind == "single"
            else list(selection.get("paths") or [])
        )
        launched = []
        # One explicit mask list cannot serve several cases — it would be
        # handed to EVERY case (wrong data into every project). Multi-case
        # imports take masks from each case's segmentations/ folder; the UI
        # only offers the explicit row for single-case drops.
        explicit_mask_paths = (
            [p for p in (selection.get("mask_paths") or []) if str(p).strip()]
            if len(sources) == 1
            else []
        )
        for index, source in enumerate(sources):
            case_info = io_ui.discover_single_source(source)
            if not case_info:
                launched.append((os.path.basename(source.rstrip("\\/")) or source, "no supported image"))
                continue
            if explicit_mask_paths:
                # The user's explicit list replaces automatic discovery. An
                # unusable entry fails this case closed rather than importing
                # it silently maskless (F24: no silent drops).
                masks, invalid = expand_mask_paths(explicit_mask_paths)
                if invalid:
                    shown = "; ".join(invalid[:3])
                    launched.append((case_info["case_id"], "无效 Mask 路径：{0}".format(shown)))
                    continue
                if masks:
                    case_info["masks"] = masks
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
                # UI28: undo's receipt search scans the drop-import status
                # files for output folders — without this, a custom output
                # dir's import receipt is undiscoverable.
                "output_path": selection["output_path"],
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
                # F35: the status file was already pre-written as running;
                # a worker that never started must not sit at "running"
                # forever in the activity list. Rewrite it as failed.
                write_json(status_path, {
                    "status": "failed",
                    "phase": "worker_launch_failed",
                    "error": "无法启动导入进程：{0}".format(exc),
                    "total": 1, "completed": 0, "failed": 1,
                    "updated_at_epoch": time.time(),
                })
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
        except Exception as exc:
            # F35: same as the per-case path — never leave a pre-written
            # "running" status behind for a batch that never launched.
            write_json(os.path.join(status_dir, "{0}_batch_status.json".format(run_id)), {
                "status": "failed",
                "phase": "worker_launch_failed",
                "error": "无法启动批量导入进程：{0}".format(exc),
                "total": 0, "completed": 0, "failed": 1,
                "output_path": selection["output_path"],
                "source_path": selection["source_path"],
                "updated_at_epoch": time.time(),
            })
            return [("dataset", "failed: {0}".format(exc))]
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
    window.setWindowTitle("导入数据")
    window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
    window.resize(640, 660)
    window.setMinimumSize(560, 480)
    root = QtWidgets.QVBoxLayout(window)

    title = QtWidgets.QLabel("导入数据")
    title.setObjectName("title")
    root.addWidget(title)
    hint = QtWidgets.QLabel(
        "把文件或文件夹拖到这里，粘贴路径（Ctrl+V，每行一条），或用上方按钮选择。窗口空闲时自动退出。"
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

    # -- Source browse: same pipeline as a drop, minus the drag ------------
    source_row = QtWidgets.QHBoxLayout()
    source_row.addWidget(QtWidgets.QLabel("或选择"))
    pick_files = QtWidgets.QPushButton("选择文件...")
    pick_folder = QtWidgets.QPushButton("选择文件夹...")
    paste_button = QtWidgets.QPushButton("粘贴路径")
    source_row.addWidget(pick_files)
    source_row.addWidget(pick_folder)
    source_row.addWidget(paste_button)
    source_row.addStretch(1)
    root.addLayout(source_row)

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

    # -- On-demand explicit Mask paths --------------------------------------
    # Progressive disclosure: hidden while automatic mask discovery works.
    # It appears only in the "masks were lost" moment (none found, or sibling
    # volumes suppressed discovery) and IS the remedy for that moment.
    # Never persisted: a previous dataset's mask path pinned onto the next
    # dataset is exactly the silent half-data-loss this row prevents.
    mask_path_widget = QtWidgets.QWidget()
    mask_path_layout = QtWidgets.QHBoxLayout(mask_path_widget)
    mask_path_layout.setContentsMargins(0, 0, 0, 0)
    mask_path_edit = QtWidgets.QLineEdit()
    mask_path_edit.setPlaceholderText("输入或粘贴分割文件路径（每行一条或用 ; 分隔）")
    mask_path_browse = QtWidgets.QPushButton("浏览...")
    mask_path_layout.addWidget(QtWidgets.QLabel("Mask 在别处？"))
    mask_path_layout.addWidget(mask_path_edit, 1)
    mask_path_layout.addWidget(mask_path_browse)
    mask_path_widget.hide()
    root.addWidget(mask_path_widget)

    mask_row = QtWidgets.QHBoxLayout()
    mask_all = QtWidgets.QRadioButton("全部 Mask")
    mask_named = QtWidgets.QRadioButton("选定的名称")
    mask_none = QtWidgets.QRadioButton("仅图像")
    remembered_masks = str(remembered_mode.get("mask_selection", "all") or "all")
    mask_none.setChecked(remembered_masks.lower() == "none")
    mask_all.setChecked(not mask_none.isChecked())
    mask_row.addWidget(mask_all)
    mask_row.addWidget(mask_named)
    mask_row.addWidget(mask_none)
    mask_row.addStretch(1)
    root.addLayout(mask_row)
    # "选定的名称": pick which of the discovered masks to import. The list is
    # populated by analyze() from automatic discovery; checkable rows build a
    # comma-separated name list the worker's _filter_masks already accepts.
    # Hidden until a drop discovers masks — never adds noise to a fresh window.
    mask_pick_widget = QtWidgets.QWidget()
    mask_pick_layout = QtWidgets.QVBoxLayout(mask_pick_widget)
    mask_pick_layout.setContentsMargins(0, 0, 0, 0)
    mask_pick_list = QtWidgets.QListWidget()
    mask_pick_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    mask_pick_list.setMaximumHeight(110)
    mask_pick_list.setToolTip("选择按名称导入后，请至少勾选一个 Mask。仅导入图像请使用“仅图像”。")
    mask_pick_layout.addWidget(mask_pick_list)
    mask_pick_widget.hide()
    root.addWidget(mask_pick_widget)

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
    scroll_layout_body(root, first=2, footer_items=1)
    clamp_window_to_screen(window, (560, 480))
    mask_named.setEnabled(False)

    def show_named_masks(checked):
        mask_pick_widget.setVisible(bool(checked and mask_pick_list.count()))

    mask_named.toggled.connect(show_named_masks)

    state = {
        "paths": [],
        "kind": None,
        "source": None,
        "auto_masks_found": False,
        "discovered_mask_names": [],
        # UI24: {case_id: [mask names]} from the batch scan, so a named
        # selection can be checked per case BEFORE the import starts.
        "case_mask_names": {},
        # R02: the mask-gap summary the user confirmed for THIS submission —
        # replayed on the result page so the fact stays visible.
        "confirmed_mask_gaps": "",
        "last_activity": time.time(),
        "submitted": False,
    }

    def touch():
        state["last_activity"] = time.time()

    def _fill_mask_pick_list(masks):
        # A named selection is explicit. Empty does not mean all.
        state["discovered_mask_names"] = [str(m.get("name") or "") for m in masks]
        mask_pick_list.clear()
        for name in state["discovered_mask_names"]:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            mask_pick_list.addItem(item)
        mask_named.setEnabled(bool(mask_pick_list.count()))
        show_named_masks(mask_named.isChecked())

    def _picked_mask_names():
        # None = all/none radio; an empty named selection is rejected.
        if not mask_named.isChecked():
            return None
        picked = []
        for index in range(mask_pick_list.count()):
            item = mask_pick_list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                picked.append(item.text())
        return picked

    def set_recognition(text, warning=False):
        recognition.setText(text)
        recognition.setProperty("warning", warning)
        recognition.style().unpolish(recognition)
        recognition.style().polish(recognition)

    # Dataset scans (summarize_dataset / discover_single_source over a
    # network root) can take tens of seconds — running them on the GUI
    # thread froze the window for the whole scan. Scans run in a daemon
    # thread; results come back through a queue drained by a timer. A
    # generation counter keeps a stale scan from overwriting a newer one.
    scan_results = Queue()
    scan_generation = {"value": 0}

    def _scan_worker(paths, generation):
        """Heavy filesystem probing, off the GUI thread (铁律 1)."""
        try:
            kind, source, child_dirs = classify_payload(paths)
            payload = {"kind": kind, "source": source}
            if kind == "batch":
                result = io_ui.summarize_dataset(source)
                payload["batch_summary"] = result
            elif kind == "multi_single":
                # F10: probe each dropped path so a mixed selection is
                # reportable per item (same data the sync loop collected).
                cases = [
                    (path, io_ui.discover_single_source(path)) for path in paths
                ]
                payload["multi_cases"] = cases
            elif kind == "single":
                payload["single_case"] = io_ui.discover_single_source(paths[0])
            scan_results.put((generation, payload, ""))
        except Exception as exc:  # surfaced by the poll timer
            scan_results.put((generation, None, "{0}: {1}".format(
                type(exc).__name__, exc)))

    def _apply_scan_result(generation, payload, error):
        if generation != scan_generation["value"]:
            return  # a newer drop superseded this scan
        if error:
            set_recognition("识别拖入内容时出错：{0}".format(error[:300]), True)
            submit.setEnabled(False)
            dropzone.setText("识别失败——请重试或直接选择文件")
            return
        kind = payload["kind"]
        source = payload["source"]
        state["kind"] = kind
        state["source"] = source
        paths = list(state["paths"])
        if kind == "batch":
            result = payload.get("batch_summary")
            # UI24: per-case mask names drive the pre-submit missing-mask
            # warning for named selections.
            state["case_mask_names"] = dict(
                (result or {}).get("case_mask_names") or {}
            )
            if result and result.get("mask_names"):
                _fill_mask_pick_list([{"name": name} for name in result["mask_names"]])
            summary = result["summary"] if result else "已识别数据集文件夹。"
            warnings = result.get("warnings") or [] if result else []
            if warnings:
                summary += "\n" + "\n".join("⚠ " + w for w in warnings[:5])
            dropzone.setText("数据集：{0}".format(source))
            set_recognition(summary, bool(warnings))
        elif kind == "multi_single":
            # F10: a multi-selection is the user's exact case list — probe
            # each dropped path so a mixed selection is reportable per item
            # and stays submittable when at least one path is valid.
            results = []
            valid = 0
            all_masks_empty = True
            union_masks = {}
            for path, case_info in payload.get("multi_cases") or []:
                if case_info:
                    valid += 1
                    if case_info.get("masks"):
                        all_masks_empty = False
                        state["auto_masks_found"] = True
                        for m in case_info["masks"]:
                            union_masks.setdefault(m["name"], m)
                    results.append("{0} ✓".format(case_info["case_id"]))
                else:
                    results.append("{0} ✗ (无可用图像)".format(
                        os.path.basename(path.rstrip("\\/")) or path))
            summary = "多选 {0} 例：{1}".format(len(paths), "、".join(results[:6]))
            if len(results) > 6:
                summary += " …（共 {0} 例）".format(len(results))
            if valid and all_masks_empty:
                summary += (
                    "\n⚠ 未发现 Mask（自动发现只查找每个病例文件夹下的 segmentations/ 子目录）。"
                )
            # R02: the named-selection pre-check needs the per-case map here
            # too — the pick list is the UNION over all dropped cases, so a
            # case missing a checked name would otherwise fail at the worker
            # (no_matching_masks) with no warning up front.
            state["case_mask_names"] = dict(
                (case_info["case_id"], [m["name"] for m in case_info.get("masks") or []])
                for _path, case_info in (payload.get("multi_cases") or [])
                if case_info
            )
            set_recognition(summary, valid < len(paths) or (valid > 0 and all_masks_empty))
            dropzone.setText("多选：{0} 个路径".format(len(paths)))
            if union_masks:
                _fill_mask_pick_list(list(union_masks.values()))
            if not valid:
                submit.setEnabled(False)
                return
        else:
            display = paths[0]
            case_info = payload.get("single_case") if kind == "single" else None
            if case_info:
                state["auto_masks_found"] = bool(case_info.get("masks"))
                # R02: a single case feeds the same pre-check map the batch
                # path uses — although the pick list is built from this case
                # (so a named selection normally cannot miss), an explicit
                # mask-path override or a stale pick list must still be
                # caught before submit.
                state["case_mask_names"] = {
                    case_info["case_id"]: [
                        m["name"] for m in case_info.get("masks") or []
                    ]
                }
                if case_info.get("masks"):
                    _fill_mask_pick_list(case_info["masks"])
                mask_names = "、".join(m["name"] for m in case_info["masks"][:5])
                if case_info.get("masks_suppressed"):
                    # Sibling volumes made attribution ambiguous, so automatic
                    # mask discovery was suppressed — say so and offer the
                    # explicit path row as the way out.
                    summary = (
                        "病例“{0}”：已识别图像\n"
                        "⚠ 同一文件夹中有多个图像文件，无法自动确定 Mask；"
                        "如需导入，请在下方“Mask 在别处”指定路径"
                        "（已有工程补 Mask 请用 01_Data → 02_Import_Masks）。".format(
                            case_info["case_id"]
                        )
                    )
                    set_recognition(summary, True)
                elif mask_names:
                    summary = "病例“{0}”：已识别图像；Mask：{1}".format(
                        case_info["case_id"], mask_names
                    )
                    set_recognition(summary, False)
                else:
                    summary = (
                        "病例“{0}”：已识别图像\n"
                        "⚠ 未发现 Mask（自动发现只查找病例文件夹下的 segmentations/ 子目录）；"
                        "如需导入，请在下方“Mask 在别处”指定路径。".format(
                            case_info["case_id"]
                        )
                    )
                    set_recognition(summary, True)
            else:
                set_recognition("拖入项中未识别到支持的图像。", True)
                submit.setEnabled(False)
                dropzone.setText(display)
                return
            dropzone.setText(display)
        # Progressive disclosure: the explicit mask path row exists only for a
        # single-case drop whose automatic mask discovery came up empty
        # (nothing found, or suppression). Multi-case drops cannot use it —
        # one pasted list would be handed to EVERY case (wrong data into
        # every project); masks for multi-case imports belong in each case's
        # segmentations/ folder. Batch drops likewise read per-case folders.
        if kind == "single" and not state.get("auto_masks_found"):
            mask_path_widget.show()
        # R61-7: a new drop means a new dataset, so the computed default
        # follows it — unless the user typed/picked a folder themselves this
        # session (that choice stays). The pre-submit autofill from a
        # previous drop is not a user choice and must not stick.
        mode = "import_batch" if kind == "batch" else "import_single"
        default_output = io_ui.source_default_output(mode, state["source"] or paths[0])
        if not output_custom["value"]:
            output_edit.setText(default_output)
        submit.setEnabled(True)

    def poll_scan_results():
        try:
            generation, payload, error = scan_results.get_nowait()
        except Empty:
            return
        _apply_scan_result(generation, payload, error)

    scan_timer = QtCore.QTimer(window)
    scan_timer.setInterval(80)
    scan_timer.timeout.connect(poll_scan_results)
    scan_timer.start()

    def analyze(paths):
        paths = [p for p in paths if p]
        mask_path_edit.clear()
        mask_path_widget.hide()
        mask_pick_list.clear()
        mask_pick_widget.hide()
        mask_named.setEnabled(False)
        state["auto_masks_found"] = False
        state["discovered_mask_names"] = []
        state["case_mask_names"] = {}
        # R02: the confirmed-gap note belongs to the submission it was
        # confirmed for; a new drop (or the post-submit reset) must not
        # carry it into an unrelated result dialog.
        state["confirmed_mask_gaps"] = ""
        if not paths:
            scan_generation["value"] += 1  # cancel any in-flight scan
            state["paths"] = []
            state["kind"] = None
            state["source"] = None
            submit.setEnabled(False)
            dropzone.setText("将图像、病例文件夹或数据集文件夹拖到这里")
            set_recognition("")
            return
        state["paths"] = paths
        # kind/source are set by _apply_scan_result when the scan lands
        # (classify_payload does real filesystem probing and must not run
        # on the GUI thread). submit stays disabled until then.
        scan_generation["value"] += 1
        generation = scan_generation["value"]
        submit.setEnabled(False)  # re-enabled when the scan lands
        dropzone.setText("正在识别拖入内容…")
        set_recognition("正在识别拖入内容…")
        worker = threading.Thread(
            target=_scan_worker, args=(paths, generation),
            name="drop-scan", daemon=True)
        worker.start()

    def add_paths(values):
        cleaned = []
        invalid = []
        for value in values or []:
            text = str(value).strip().strip('"')
            if not text:
                continue
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(text)))
            if os.path.exists(expanded):
                if expanded not in cleaned:
                    cleaned.append(expanded)
            else:
                invalid.append(text)
        if invalid and not cleaned:
            # Every pasted/dropped path failed to resolve: silently doing
            # nothing reads as "the window is broken" (the user pasted
            # something, nothing happened). Name the offenders.
            set_recognition(
                "以下路径不存在或不可访问：\n{0}".format("\n".join(invalid[:5])),
                True,
            )
            return
        if not cleaned:
            return
        touch()
        if invalid:
            # Mixed input: import the valid paths, but never swallow the
            # fact that some were dropped (half a dataset is data loss).
            set_recognition(
                "已跳过不存在的路径：{0}".format("、".join(invalid[:5])), True
            )
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
            set_recognition("剪贴板中没有路径。", True)
            return
        add_paths(values)

    paste_button.clicked.connect(paste_paths)

    def browse_output():
        # A native dialog can hang on disconnected drives / SMB folders; the
        # async helper isolates it in its own process so this window and
        # Mimics stay responsive.
        def _chosen(value):
            if value:
                output_custom["value"] = True
                output_edit.setText(str(value))

        choose_existing_directory_async(
            QtCore, QtWidgets, window, "选择输出文件夹",
            output_edit.text().strip() or str(Path.home()),
            _chosen,
            button=browse,
        )

    browse.clicked.connect(browse_output)

    def browse_mask_paths():
        # Same async isolation as the other pickers: a native dialog on a
        # slow network share must never freeze this window or Mimics.
        def _chosen(values):
            incoming = [str(v) for v in (values if isinstance(values, (list, tuple)) else [values]) if str(v).strip()]
            if incoming:
                merged = [
                    part for part in
                    mask_path_edit.text().replace("\r", "\n").replace("\n", ";").split(";")
                    if part.strip()
                ]
                mask_path_edit.setText(";".join(merged + incoming))

        choose_open_files_async(
            QtCore, QtWidgets, window, "选择分割文件",
            str(Path.home()),
            "Segmentation files (*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz *.seg.nii *.seg.nii.gz);;"
            "All files (*)",
            _chosen,
            button=mask_path_browse,
        )

    mask_path_browse.clicked.connect(browse_mask_paths)

    def browse_source_files():
        # Multi-select feeds the same add_paths() pipeline as a multi-drop,
        # so a hand-picked case list behaves identically to a dragged one.
        def _chosen(values):
            add_paths(values if isinstance(values, (list, tuple)) else [values])

        choose_open_files_async(
            QtCore, QtWidgets, window, "选择图像文件",
            str(Path.home()),
            "Medical volumes (*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz);;"
            "All files (*)",
            _chosen,
            button=pick_files,
        )

    def browse_source_folder():
        def _chosen(value):
            if value:
                add_paths([value])

        choose_existing_directory_async(
            QtCore, QtWidgets, window, "选择病例或数据集文件夹",
            str(Path.home()),
            _chosen,
            button=pick_folder,
        )

    pick_files.clicked.connect(browse_source_files)
    pick_folder.clicked.connect(browse_source_folder)

    def clear_all():
        touch()
        analyze([])

    clear.clicked.connect(clear_all)

    # submit_import probes each multi-case path again (discover_single_source
    # per case) — same network-share freeze as the analyze scan, so it also
    # runs in a daemon thread with the result drained by the same pattern.
    submit_results_queue = Queue()

    def poll_submit_results():
        try:
            launched, output, exc = submit_results_queue.get_nowait()
        except Empty:
            return
        submit.setEnabled(True)
        if exc is not None:
            QtWidgets.QMessageBox.critical(
                window, "导入失败",
                "导入无法启动：\n\n{0}".format(exc),
            )
            return
        lines = ["{0}: {1}".format(name, status) for name, status in launched[:10]]
        if len(launched) > 10:
            lines.append("……其余 {0} 例".format(len(launched) - 10))
        # R02: when the user accepted a mask-gap warning, the fact must
        # survive onto the result page too — the annotator confirms the
        # image-only/partial import and later needs it stated again, not
        # only in a dialog that has already closed.
        gap_note = state.get("confirmed_mask_gaps") or ""
        QtWidgets.QMessageBox.information(
            window, "导入已开始",
            "导入已开始：\n\n{0}\n\n任务在后台继续进行；Mimics 工程将生成于：\n{1}\n{2}"
            "\n进度可在 01_Data > 04_Task_Status 查看。".format(
                "\n".join(lines), output, gap_note),
        )
        state["submitted"] = True
        analyze([])

    submit_timer = QtCore.QTimer(window)
    submit_timer.setInterval(80)
    submit_timer.timeout.connect(poll_submit_results)
    submit_timer.start()

    def do_submit():
        if not state["kind"] or not submit.isEnabled():
            return
        touch()
        kind = state["kind"]
        source = state["source"] or (state["paths"][0] if state["paths"] else "")
        raw_output = output_edit.text().strip()
        if not raw_output:
            # abspath("") is the process cwd (the project root) — without
            # this gate a cleared output field silently dumps .mcs files
            # and runtime state into the project directory.
            QtWidgets.QMessageBox.warning(
                window, "缺少输出文件夹",
                "请先填写输出根目录（或清空后重新拖入病例以恢复默认值）。",
            )
            return
        output = os.path.abspath(os.path.expanduser(raw_output))
        mask_selection = "none" if mask_none.isChecked() else "all"
        picked = _picked_mask_names()
        if mask_named.isChecked() and not picked:
            QtWidgets.QMessageBox.warning(
                window, "未选择 Mask", "请至少勾选一个 Mask；仅导入图像请切换到“仅图像”。")
            return
        if picked and any(
                "," in name or name.strip().lower() in ("all", "none", "no", "off")
                for name in picked):
            QtWidgets.QMessageBox.warning(
                window, "名称无法用于筛选",
                "包含逗号或 all/none/no/off 的名称暂不支持按名称筛选。请选择“全部 Mask”，或重命名分割文件后重试。")
            return
        if picked:
            # UI24/R02: a named selection means the user wants those masks.
            # A case whose folder lacks every picked name imports as images
            # only on the batch path — but FAILS at the single-case worker
            # (no_matching_masks) — and a case holding only some picks
            # silently drops the rest. Surface all of it, per case, before
            # anything starts. Matching is case-insensitive, same as the
            # worker's _filter_masks.
            if kind in ("batch", "multi_single", "single"):
                missing, partial = cases_partial_mask_report(
                    state.get("case_mask_names"), picked)
                if missing or partial:
                    # What "no matching mask" means downstream depends on
                    # the path: batch prepares the case with an empty mask
                    # list (image-only project), while the single-case
                    # worker refuses the case outright. The dialog must not
                    # promise image-only import for a case that will fail.
                    if kind == "batch":
                        missing_line = (
                            "以下 {0} 例不含任何所选 Mask，将仅导入图像：\n{1}")
                    else:
                        missing_line = (
                            "以下 {0} 例不含任何所选 Mask，这些病例将导入失败"
                            "（无匹配分割文件）：\n{1}")
                    lines = []
                    if missing:
                        shown = "、".join(missing[:5])
                        if len(missing) > 5:
                            shown += " 等共 {0} 例".format(len(missing))
                        lines.append(missing_line.format(len(missing), shown))
                    if partial:
                        detail = "；".join(
                            "{0}（缺 {1}）".format(
                                case, "、".join(names))
                            for case, names in sorted(partial.items())[:5])
                        if len(partial) > 5:
                            detail += "；等共 {0} 例".format(len(partial))
                        lines.append(
                            "以下病例缺少部分所选 Mask，缺失的标签不会导入：\n{0}".format(
                                detail))
                    answer = QtWidgets.QMessageBox.question(
                        window, "部分病例缺少所选 Mask",
                        "{0}\n\n如非预期，请取消并检查这些病例的 segmentations/ 文件夹，"
                        "或改选其他 Mask。".format("\n\n".join(lines)),
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if answer != QtWidgets.QMessageBox.Yes:
                        return
                    # R02: keep the accepted gap visible on the result page
                    # ("仅图像后结果页也必须保留该事实") — the confirm dialog
                    # closes, the fact must not vanish with it.
                    state["confirmed_mask_gaps"] = "\n\n{0}\n".format(
                        "；".join(lines))
            # The worker receives the exact checked names.
            mask_selection = ",".join(picked)
        mask_paths = []
        if kind == "single" and mask_path_widget.isVisible():
            # ; or newlines separate entries (paste keeps multi-line clipboard
            # text as-is); neither is valid inside a path in practice.
            mask_paths = [
                part.strip().strip('"')
                for part in mask_path_edit.text().replace("\r", "\n").replace("\n", ";").split(";")
                if part.strip().strip('"')
            ]
        selection = {
            "kind": kind,
            "source_path": source,
            "output_path": output,
            "mask_selection": mask_selection,
            "mask_paths": mask_paths,
            "paths": list(state["paths"]) if kind == "multi_single" else [],
        }
        # Persist the remembered state BEFORE the background submit (the
        # GUI thread owns it), then hand off only the launch itself.
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
        submit.setEnabled(False)
        set_recognition("正在启动导入…")

        def _launch():
            try:
                launched = submit_import(selection, context)
            except Exception as exc:  # surfaced by the poll timer
                submit_results_queue.put((None, output, exc))
                return
            submit_results_queue.put((launched, output, None))

        worker = threading.Thread(
            target=_launch, name="drop-submit", daemon=True)
        worker.start()

    submit.clicked.connect(do_submit)

    # -- Recent drops polling ---------------------------------------------
    def refresh_recent():
        status_dir = os.path.join(_ROOT, ".mimics_runtime", "drop_import")
        entries = collect_recent_drops(status_dir)
        recent_list.clear()
        for entry in entries:
            item = QtWidgets.QListWidgetItem(format_recent_drop_line(entry))
            item.setData(QtCore.Qt.UserRole, entry["log_path"])
            # Long failure reasons live in the tooltip, not the row itself.
            tip = "双击打开导入日志"
            if entry["error"]:
                tip = "{0}\n{1}".format(entry["error"], tip)
            item.setToolTip(tip)
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
