#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External read-only DINOv3 few-shot status viewer.

The foreground Mimics process starts this tool with Popen and returns
immediately.  This UI only reads JSON/log files and can request cancellation
through the existing cancel markers recorded in job status files.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path


TITLE = "DINOv3 Few-Shot Status"
ACTIVE_STATUSES = set([
    "launching",
    "preparing",
    "exporting_labels",
    "waiting_for_background_mimics",
    "waiting_for_gpu",
    "training",
    "running",
    "cancelling",
    "configuring",
    "training_started",
])


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text, retries=retries, max_sleep=max_sleep)


def write_text_atomic(path, text, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = str(text)
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def write_json_best_effort(path, payload):
    try:
        write_json_atomic(path, payload, retries=8, max_sleep=0.15)
        return True
    except Exception:
        return False


def write_cancel_marker(cancel_path):
    if not cancel_path:
        return None
    try:
        write_text_atomic(
            cancel_path,
            "cancel requested at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")),
            retries=8,
            max_sleep=0.15,
        )
        return None
    except Exception as exc:
        return str(exc)


def format_time(epoch):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except Exception:
        return "unknown"


def display_status(value):
    labels = {
        "launching": "Launching",
        "preparing": "Preparing data",
        "exporting_labels": "Exporting labels",
        "waiting_for_background_mimics": "Waiting for background Mimics",
        "waiting_for_gpu": "Waiting for GPU",
        "training": "Training",
        "running": "Running inference",
        "cancelling": "Cancelling",
        "configuring": "Configuring training",
        "training_started": "Training started",
        "closed": "Closed",
        "cancelled": "Cancelled",
        "failed": "Failed",
        "completed": "Completed",
    }
    return labels.get(str(value or ""), str(value or "Unknown").replace("_", " "))


def display_resource(value):
    labels = {
        "gpu": "GPU",
        "background_mimics": "background Mimics",
    }
    return labels.get(str(value or ""), str(value or "resource").replace("_", " "))


def resource_wait_text(job):
    resource_wait = job.get("resource_wait") or {}
    if not resource_wait:
        return ""
    resource = display_resource(resource_wait.get("resource", "resource"))
    holder = resource_wait.get("owner", "unknown")
    pid = resource_wait.get("pid", "")
    if pid:
        return "Waiting for {0}: {1} (PID {2})".format(resource, holder, pid)
    return "Waiting for {0}".format(resource)


def progress_line(progress):
    if not isinstance(progress, dict):
        return ""
    if progress.get("latest_epoch_line"):
        return str(progress.get("latest_epoch_line"))
    parts = []
    if progress.get("epoch") is not None and progress.get("epochs") is not None:
        parts.append("epoch {0}/{1}".format(progress.get("epoch"), progress.get("epochs")))
    if progress.get("phase"):
        parts.append(str(progress.get("phase")))
    if progress.get("batch") is not None and progress.get("batches") is not None:
        parts.append("batch {0}/{1}".format(progress.get("batch"), progress.get("batches")))
    metrics = progress.get("metrics") or {}
    if metrics.get("loss") is not None:
        try:
            parts.append("loss {0:.4f}".format(float(metrics.get("loss"))))
        except Exception:
            parts.append("loss {0}".format(metrics.get("loss")))
    if metrics.get("mean_dsc") is not None:
        try:
            parts.append("val_dice {0:.4f}".format(float(metrics.get("mean_dsc"))))
        except Exception:
            parts.append("val_dice {0}".format(metrics.get("mean_dsc")))
    if progress.get("best_dsc") is not None:
        try:
            parts.append("best {0:.4f}".format(float(progress.get("best_dsc"))))
        except Exception:
            parts.append("best {0}".format(progress.get("best_dsc")))
    if progress.get("lr") is not None:
        try:
            parts.append("lr {0:.2e}".format(float(progress.get("lr"))))
        except Exception:
            pass
    return ", ".join(parts)


def format_job_line(job):
    pieces = [
        job.get("job_id", "?"),
        job.get("kind", "?"),
        job.get("organ", "?"),
        display_status(job.get("status", "?")),
    ]
    if job.get("case_id"):
        pieces.append("case {0}".format(job.get("case_id")))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        pieces.append("train {0}, val {1}".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        ))
    wait = resource_wait_text(job)
    if wait:
        pieces.append(wait)
    progress = progress_line(job.get("training_progress") or {})
    if progress:
        pieces.append(progress)
    if job.get("error"):
        pieces.append("error: {0}".format(job.get("error")))
    if job.get("train_log_warning") or job.get("log_warning"):
        pieces.append("log warning")
    if job.get("train_log_unavailable") or job.get("log_unavailable"):
        pieces.append("log unavailable")
    if job.get("cancel_marker_error"):
        pieces.append("cancel marker warning: {0}".format(job.get("cancel_marker_error")))
    return " | ".join([str(part) for part in pieces if str(part)])


def process_exists(pid):
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def terminate_process_tree(pid):
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def open_path(path):
    if not path:
        return
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def read_log_text(path, max_bytes=2 * 1024 * 1024, retries=3):
    if not path or not os.path.isfile(path):
        return ""
    for attempt in range(max(1, int(retries))):
        try:
            with open(path, "rb") as handle:
                try:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if size > int(max_bytes):
                        handle.seek(-int(max_bytes), os.SEEK_END)
                    else:
                        handle.seek(0)
                except Exception:
                    pass
                data = handle.read(int(max_bytes))
            return data.decode("utf-8", "replace")
        except Exception:
            time.sleep(min(0.10, 0.03 * (attempt + 1)))
    return ""


def tail_text_from_text(text, max_lines=80):
    if not text:
        return ""
    lines = str(text).splitlines(True)
    return "".join(lines[-max_lines:])


def tail_text(path, max_lines=80):
    return tail_text_from_text(read_log_text(path), max_lines=max_lines)


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s*train_loss=([0-9eE+\-.]+)(?:.*?val_dice=([0-9eE+\-.]+))?"
)


def parse_epoch_metrics_from_texts(*texts):
    rows = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for match in EPOCH_RE.finditer(str(text)):
            epoch = int(match.group(1))
            key = (epoch, match.group(2), match.group(3), match.group(4))
            if key in seen:
                continue
            seen.add(key)
            row = {
                "epoch": epoch,
                "epochs": int(match.group(2)),
                "train_loss": float(match.group(3)),
                "val_dice": None,
            }
            if match.group(4) is not None:
                row["val_dice"] = float(match.group(4))
            rows.append(row)
    rows.sort(key=lambda item: item["epoch"])
    return rows


def _number_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_epoch_metrics_from_history(path):
    payload = read_json(path, None)
    if isinstance(payload, list):
        history = payload
        epoch_count = None
    elif isinstance(payload, dict):
        history = payload.get("history") or payload.get("rows") or []
        epoch_count = payload.get("epoch_count")
    else:
        return []

    rows = []
    seen = set()
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            epoch = int(item.get("epoch"))
        except Exception:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        train_loss = _number_or_none(item.get("train_loss"))
        if train_loss is None:
            train_loss = _number_or_none(metrics.get("train_loss"))
        if train_loss is None:
            train_loss = _number_or_none(metrics.get("loss"))
        if train_loss is None:
            continue
        val_dice = _number_or_none(item.get("val_dice"))
        if val_dice is None:
            val_dice = _number_or_none(metrics.get("val_dice"))
        if val_dice is None:
            val_dice = _number_or_none(metrics.get("mean_dsc"))
        try:
            epochs = int(item.get("epochs") or epoch_count or epoch)
        except Exception:
            epochs = epoch
        key = (epoch, train_loss, val_dice)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "val_dice": val_dice,
        })
    rows.sort(key=lambda item: item["epoch"])
    return rows


def training_curve_rows(job, log_text="", pipeline_text=""):
    job = job or {}
    progress = job.get("training_progress") if isinstance(job.get("training_progress"), dict) else {}
    history_path = (
        job.get("metrics_history")
        or job.get("metrics_history_path")
        or progress.get("metrics_history")
        or progress.get("metrics_history_path")
    )
    rows = parse_epoch_metrics_from_history(history_path)
    if rows:
        return rows
    return parse_epoch_metrics_from_texts(log_text, pipeline_text)


def parse_epoch_metrics(*paths):
    texts = []
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        texts.append(read_log_text(path, max_bytes=4 * 1024 * 1024))
    return parse_epoch_metrics_from_texts(*texts)


class StatusViewerApp(object):
    def __init__(self, root, context):
        self.root = root
        self.context = context
        self.ts_root = os.path.abspath(context.get("ts_root", ""))
        self.workspace = os.path.abspath(context.get("workspace") or os.path.join(self.ts_root, "fewshot_models"))
        self.jobs_dir = os.path.join(self.workspace, "jobs")
        self.selected_job_id = ""
        self.jobs = []
        self.job_paths = {}
        self.listbox = None
        self.summary_var = None
        self.detail_text = None
        self.log_text = None
        self.chart = None
        self.stop_button = None
        self.open_log_button = None
        self._refresh_after_id = None
        self._build()
        self.refresh()

    def _build(self):
        import tkinter as tk
        from tkinter import ttk

        self.root.title(TITLE)
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)
        try:
            self.root.attributes("-topmost", False)
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="DINOv3 Few-Shot Status", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        organ = self.context.get("selected_organ") or "all organs"
        ttk.Label(header, text="Dataset: {0}    Organ filter: {1}".format(self.ts_root, organ)).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(toolbar, text="Open Workspace", command=lambda: open_path(self.workspace)).pack(side="left", padx=(8, 0))
        self.open_log_button = ttk.Button(toolbar, text="Open Log Folder", command=self.open_log_folder)
        self.open_log_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(toolbar, text="Request Stop", command=self.request_stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Close", command=self.root.destroy).pack(side="right")

        self.summary_var = tk.StringVar(value="Loading jobs...")
        ttk.Label(outer, textvariable=self.summary_var).pack(fill="x")

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        body.add(left, weight=1)
        ttk.Label(left, text="Jobs").pack(anchor="w")
        self.listbox = tk.Listbox(left, exportselection=False, height=28)
        job_scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=job_scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        job_scroll.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(body)
        body.add(right, weight=3)

        detail = ttk.LabelFrame(right, text="Selected job", padding=8)
        detail.pack(fill="x")
        self.detail_text = tk.Text(detail, height=9, wrap="word", state="disabled")
        self.detail_text.pack(fill="x", expand=False)

        chart_box = ttk.LabelFrame(right, text="Training curve", padding=8)
        chart_box.pack(fill="x", pady=(8, 0))
        self.chart = tk.Canvas(chart_box, height=240, background="#f8fafc", highlightthickness=1, highlightbackground="#d1d5db")
        self.chart.pack(fill="x", expand=False)

        log_box = ttk.LabelFrame(right, text="Log tail", padding=8)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_box, height=14, wrap="none", state="disabled")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="left", fill="y")

    def refresh(self):
        old_selected = self.selected_job_id
        self.jobs, self.job_paths = self._load_jobs()
        self.listbox.delete(0, "end")
        selected_index = 0
        for index, job in enumerate(self.jobs):
            label = "{0}   {1}   {2}   {3}".format(
                display_status(job.get("status")),
                job.get("kind", "?"),
                job.get("organ", "?"),
                job.get("job_id", "?"),
            )
            self.listbox.insert("end", label)
            if job.get("job_id") == old_selected:
                selected_index = index
        if self.jobs:
            self.listbox.selection_set(selected_index)
            self.listbox.activate(selected_index)
            self.selected_job_id = self.jobs[selected_index].get("job_id", "")
            self.show_job(self.jobs[selected_index])
            active = sum(1 for job in self.jobs if job.get("status") in ACTIVE_STATUSES)
            self.summary_var.set("{0} job(s), {1} active. Auto-refresh every 2 seconds.".format(len(self.jobs), active))
        else:
            self.selected_job_id = ""
            self.summary_var.set("No DINOv3 few-shot jobs were found.")
            self._set_text(self.detail_text, "No jobs were found under:\n{0}".format(self.jobs_dir))
            self._set_text(self.log_text, "")
            self.draw_chart([])
        if self._refresh_after_id is not None:
            try:
                self.root.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self._refresh_after_id = self.root.after(2000, self.refresh)

    def _load_jobs(self):
        rows = []
        paths = {}
        root = Path(self.jobs_dir)
        if not root.is_dir():
            return [], {}
        for path in root.glob("*.json"):
            if path.name.endswith("_context.json"):
                continue
            payload = read_json(path, {}) or {}
            if not payload:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            rows.append((mtime, payload))
            if payload.get("job_id"):
                paths[payload.get("job_id")] = str(path)
        rows.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rows[:80]], paths

    def on_select(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index < 0 or index >= len(self.jobs):
            return
        job = self.jobs[index]
        self.selected_job_id = job.get("job_id", "")
        self.show_job(job)

    def show_job(self, job):
        lines = [
            "Job: {0}".format(job.get("job_id", "?")),
            "Type: {0}".format(job.get("kind", "?")),
            "Organ: {0}".format(job.get("organ", "?")),
            "Status: {0}".format(display_status(job.get("status"))),
            "Created: {0}".format(format_time(job.get("created_at_epoch"))),
            "Updated: {0}".format(format_time(job.get("updated_at_epoch"))),
        ]
        if job.get("case_id"):
            lines.append("Case: {0}".format(job.get("case_id")))
        if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
            lines.append("Samples: train {0}, validation {1}".format(
                job.get("train_sample_count", "?"),
                job.get("validation_sample_count", "?"),
            ))
        wait = resource_wait_text(job)
        if wait:
            lines.append(wait)
        progress = progress_line(job.get("training_progress") or {})
        if progress:
            lines.append("Progress: {0}".format(progress))
        if job.get("error"):
            lines.append("Error: {0}".format(job.get("error")))
        if job.get("train_log_warning"):
            lines.append("Training log warning: {0}".format(job.get("train_log_warning")))
        if job.get("log_warning"):
            lines.append("Log warning: {0}".format(job.get("log_warning")))
        if job.get("train_log_unavailable"):
            lines.append("Training log unavailable.")
        if job.get("log_unavailable"):
            lines.append("Inference log unavailable.")
        if job.get("cancel_marker_error"):
            lines.append("Cancel marker warning: {0}".format(job.get("cancel_marker_error")))
        if job.get("train_log"):
            lines.append("Train log: {0}".format(job.get("train_log")))
        if job.get("metrics_history"):
            lines.append("Metrics history: {0}".format(job.get("metrics_history")))
        if job.get("log"):
            lines.append("Inference log: {0}".format(job.get("log")))
        model = job.get("model") or {}
        if model.get("checkpoint"):
            lines.append("Model: {0}".format(model.get("checkpoint")))
        if job.get("output_path"):
            lines.append("Prediction: {0}".format(job.get("output_path")))
        self._set_text(self.detail_text, "\n".join(lines))

        log_path = job.get("train_log") or job.get("log")
        pipeline_log = os.path.join(self.workspace, "fewshot_pipeline.log")
        log_text = read_log_text(log_path, max_bytes=2 * 1024 * 1024)
        pipeline_text = ""
        log_tail = tail_text_from_text(log_text, 80)
        if pipeline_log and pipeline_log != log_path:
            pipeline_text = read_log_text(pipeline_log, max_bytes=1024 * 1024)
            pipeline_tail = tail_text_from_text(pipeline_text, 60)
            if pipeline_tail:
                log_tail = (log_tail + "\n" if log_tail else "") + "---- pipeline log ----\n" + pipeline_tail
        self._set_text(self.log_text, log_tail)
        rows = training_curve_rows(job, log_text, pipeline_text)
        self.draw_chart(rows)
        try:
            state = ["!disabled"] if job.get("status") in ACTIVE_STATUSES else ["disabled"]
            self.stop_button.state(state)
        except Exception:
            pass

    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text or "")
        widget.configure(state="disabled")

    def draw_chart(self, rows):
        canvas = self.chart
        canvas.delete("all")
        width = max(560, int(canvas.winfo_width() or 900))
        height = 240
        canvas.create_rectangle(0, 0, width, height, fill="#f8fafc", outline="")

        margin_l, margin_r, margin_t, margin_b = 76, 68, 54, 44
        x0, y0 = margin_l, height - margin_b
        x1, y1 = width - margin_r, margin_t
        plot_w = max(1, x1 - x0)
        plot_h = max(1, y0 - y1)

        canvas.create_rectangle(x0, y1, x1, y0, fill="#ffffff", outline="#d1d5db")
        canvas.create_text(16, 18, anchor="w", text="Training Progress", fill="#111827", font=("Segoe UI", 10, "bold"))
        if not rows:
            canvas.create_text(width / 2, height / 2, text="No epoch metrics yet", fill="#6b7280", font=("Segoe UI", 10))
            return

        epochs = [row["epoch"] for row in rows]
        min_epoch, max_epoch = min(epochs), max(epochs)
        if min_epoch == max_epoch:
            max_epoch = min_epoch + 1
        losses = [row["train_loss"] for row in rows if row.get("train_loss") is not None]
        min_loss = min(losses) if losses else 0.0
        max_loss = max(losses) if losses else 1.0
        if min_loss == max_loss:
            pad = max(0.01, abs(min_loss) * 0.1)
            min_loss -= pad
            max_loss += pad
        else:
            pad = (max_loss - min_loss) * 0.12
            min_loss = max(0.0, min_loss - pad)
            max_loss += pad

        tick_count = 4
        for i in range(tick_count + 1):
            frac = i / float(tick_count)
            y = y0 - frac * plot_h
            loss_value = min_loss + frac * (max_loss - min_loss)
            dice_value = frac
            canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            canvas.create_text(x0 - 10, y, anchor="e", text="{0:.3g}".format(loss_value), fill="#991b1b", font=("Segoe UI", 8))
            canvas.create_text(x1 + 10, y, anchor="w", text="{0:.2f}".format(dice_value), fill="#1d4ed8", font=("Segoe UI", 8))

        epoch_ticks = sorted(set([min(epochs), max(epochs)] + [row["epoch"] for row in rows]))
        if len(epoch_ticks) > 6:
            step = max(1, int(round(len(epoch_ticks) / 5.0)))
            epoch_ticks = epoch_ticks[::step]
            if max(epochs) not in epoch_ticks:
                epoch_ticks.append(max(epochs))

        def x_at(epoch):
            return x0 + (float(epoch) - min_epoch) / float(max_epoch - min_epoch) * plot_w

        def y_loss(value):
            return y0 - (float(value) - min_loss) / float(max_loss - min_loss) * plot_h

        def y_dice(value):
            value = max(0.0, min(1.0, float(value)))
            return y0 - value * plot_h

        for epoch in epoch_ticks:
            x = x_at(epoch)
            canvas.create_line(x, y0, x, y0 + 4, fill="#9ca3af")
            canvas.create_text(x, y0 + 17, text=str(epoch), fill="#4b5563", font=("Segoe UI", 8))

        loss_points = []
        dice_points = []
        for row in rows:
            if row.get("train_loss") is not None:
                loss_points.extend([x_at(row["epoch"]), y_loss(row["train_loss"])])
            if row.get("val_dice") is not None:
                dice_points.extend([x_at(row["epoch"]), y_dice(row["val_dice"])])
        if len(loss_points) >= 4:
            canvas.create_line(*loss_points, fill="#dc2626", width=2, smooth=True)
        elif len(loss_points) == 2:
            canvas.create_oval(loss_points[0] - 3, loss_points[1] - 3, loss_points[0] + 3, loss_points[1] + 3, fill="#dc2626")
        if len(dice_points) >= 4:
            canvas.create_line(*dice_points, fill="#2563eb", width=2, smooth=True)
        elif len(dice_points) == 2:
            canvas.create_oval(dice_points[0] - 3, dice_points[1] - 3, dice_points[0] + 3, dice_points[1] + 3, fill="#2563eb")

        if len(loss_points) >= 2:
            lx, ly = loss_points[-2], loss_points[-1]
            canvas.create_oval(lx - 4, ly - 4, lx + 4, ly + 4, fill="#dc2626", outline="#ffffff", width=1)
        if len(dice_points) >= 2:
            dx, dy = dice_points[-2], dice_points[-1]
            canvas.create_oval(dx - 4, dy - 4, dx + 4, dy + 4, fill="#2563eb", outline="#ffffff", width=1)

        canvas.create_text(x0, height - 12, anchor="w", text="Epoch", fill="#4b5563", font=("Segoe UI", 8))
        canvas.create_text(x0 - 42, y1 - 18, anchor="w", text="Loss", fill="#991b1b", font=("Segoe UI", 8, "bold"))
        canvas.create_text(x1 + 18, y1 - 18, anchor="w", text="Dice", fill="#1d4ed8", font=("Segoe UI", 8, "bold"))

        latest_loss = None
        latest_dice = None
        latest_epoch = epochs[-1]
        for row in reversed(rows):
            if latest_loss is None and row.get("train_loss") is not None:
                latest_loss = row.get("train_loss")
            if latest_dice is None and row.get("val_dice") is not None:
                latest_dice = row.get("val_dice")
        badges = [
            ("Epoch {0}/{1}".format(latest_epoch, max(epochs)), "#374151", "#f3f4f6"),
        ]
        if latest_loss is not None:
            badges.append(("loss {0:.4f}".format(float(latest_loss)), "#991b1b", "#fee2e2"))
        if latest_dice is not None:
            badges.append(("val dice {0:.4f}".format(float(latest_dice)), "#1d4ed8", "#dbeafe"))

        bx = 150
        for text, color, fill in badges:
            tw = max(76, len(text) * 7 + 20)
            canvas.create_rectangle(bx, 12, bx + tw, 36, fill=fill, outline="#e5e7eb")
            canvas.create_text(bx + 10, 24, anchor="w", text=text, fill=color, font=("Segoe UI", 9))
            bx += tw + 8

        legend_x = x1 - 178
        canvas.create_line(legend_x, 24, legend_x + 24, 24, fill="#dc2626", width=2)
        canvas.create_text(legend_x + 32, 24, anchor="w", text="train loss", fill="#374151", font=("Segoe UI", 8))
        canvas.create_line(legend_x + 104, 24, legend_x + 128, 24, fill="#2563eb", width=2)
        canvas.create_text(legend_x + 136, 24, anchor="w", text="val dice", fill="#374151", font=("Segoe UI", 8))

    def selected_job(self):
        if not self.selected_job_id:
            return None
        for job in self.jobs:
            if job.get("job_id") == self.selected_job_id:
                return job
        return None

    def open_log_folder(self):
        job = self.selected_job()
        path = ""
        if job:
            path = job.get("train_log") or job.get("log") or ""
        if path and os.path.isfile(path):
            open_path(os.path.dirname(path))
            return
        open_path(self.workspace)

    def request_stop(self):
        job = self.selected_job()
        if not job:
            return
        cancel_path = job.get("cancel_path")
        cancel_error = write_cancel_marker(cancel_path)
        killed = []
        for key in ("pid", "controller_pid", "launcher_pid"):
            pid = job.get(key)
            if pid and process_exists(pid) and terminate_process_tree(pid):
                killed.append(int(pid))
        job["status"] = "cancelled"
        job["cancel_requested_at_epoch"] = time.time()
        job["cancelled_pids"] = killed
        if cancel_error:
            job["cancel_marker_error"] = cancel_error
        job["updated_at_epoch"] = time.time()
        status_path = self.job_paths.get(job.get("job_id"))
        if status_path:
            write_json_best_effort(status_path, job)
        self.show_job(job)


def _load_pyside6():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class QtStatusViewerApp(object):
    """PySide6 implementation of the external read-only status viewer."""

    def __init__(self, window, context, qt_modules, curve_widget_cls):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        self.curve_widget_cls = curve_widget_cls
        self.ts_root = os.path.abspath(context.get("ts_root", ""))
        self.workspace = os.path.abspath(context.get("workspace") or os.path.join(self.ts_root, "fewshot_models"))
        self.jobs_dir = os.path.join(self.workspace, "jobs")
        self.selected_job_id = ""
        self.jobs = []
        self.job_paths = {}
        self.jobs_list = None
        self.summary_label = None
        self.detail_text = None
        self.log_text = None
        self.chart = None
        self.stop_button = None
        self.open_log_button = None
        self.timer = self.QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.refresh)
        self._build()
        self.refresh()
        self.timer.start(2000)

    def _build(self):
        QtWidgets = self.QtWidgets
        self.window.setWindowTitle(TITLE)
        self.window.resize(1120, 780)
        self.window.setMinimumSize(980, 680)
        self.window.setStyleSheet(self._stylesheet())
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QtWidgets.QLabel("DINOv3 Few-Shot Status")
        title.setObjectName("titleLabel")
        organ = self.context.get("selected_organ") or "all organs"
        subtitle = QtWidgets.QLabel("Dataset: {0}    Organ filter: {1}".format(self.ts_root, organ))
        subtitle.setObjectName("subtitleLabel")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        toolbar = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        toolbar.addWidget(refresh)
        open_workspace = QtWidgets.QPushButton("Open Workspace")
        open_workspace.clicked.connect(lambda: open_path(self.workspace))
        toolbar.addWidget(open_workspace)
        self.open_log_button = QtWidgets.QPushButton("Open Log Folder")
        self.open_log_button.clicked.connect(self.open_log_folder)
        toolbar.addWidget(self.open_log_button)
        self.stop_button = QtWidgets.QPushButton("Request Stop")
        self.stop_button.clicked.connect(self.request_stop)
        toolbar.addWidget(self.stop_button)
        toolbar.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.window.close)
        toolbar.addWidget(close)
        outer.addLayout(toolbar)

        self.summary_label = QtWidgets.QLabel("Loading jobs...")
        outer.addWidget(self.summary_label)

        splitter = QtWidgets.QSplitter(self.QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.addWidget(QtWidgets.QLabel("Jobs"))
        self.jobs_list = QtWidgets.QListWidget()
        self.jobs_list.currentRowChanged.connect(self.on_select)
        left_layout.addWidget(self.jobs_list)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        detail_group = QtWidgets.QGroupBox("Selected job")
        detail_layout = QtWidgets.QVBoxLayout(detail_group)
        self.detail_text = QtWidgets.QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFixedHeight(150)
        detail_layout.addWidget(self.detail_text)
        right_layout.addWidget(detail_group)

        chart_group = QtWidgets.QGroupBox("Training curve")
        chart_layout = QtWidgets.QVBoxLayout(chart_group)
        self.chart = self.curve_widget_cls()
        self.chart.setMinimumHeight(250)
        chart_layout.addWidget(self.chart)
        right_layout.addWidget(chart_group)

        log_group = QtWidgets.QGroupBox("Log tail")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        try:
            self.log_text.setFont(self.QtGui.QFont("Consolas", 9))
        except Exception:
            pass
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)
        self.window.setCentralWidget(central)

    def _stylesheet(self):
        return """
        QMainWindow, QWidget { background: #f4f5f7; color: #111827; font-family: Segoe UI, Arial; font-size: 10pt; }
        QLabel#titleLabel { font-size: 17pt; font-weight: 700; color: #111827; }
        QLabel#subtitleLabel { color: #374151; }
        QGroupBox { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; margin-top: 10px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #111827; font-weight: 600; }
        QListWidget, QTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px; }
        QPushButton { background: #ffffff; border: 1px solid #9ca3af; border-radius: 4px; padding: 7px 13px; }
        QPushButton:hover { background: #f3f4f6; }
        QPushButton:disabled { color: #9ca3af; background: #f3f4f6; }
        QListWidget::item { padding: 6px; }
        QListWidget::item:selected { background: #dbeafe; color: #111827; }
        """

    def refresh(self):
        old_selected = self.selected_job_id
        self.jobs, self.job_paths = self._load_jobs()
        self.jobs_list.blockSignals(True)
        self.jobs_list.clear()
        selected_index = 0
        for index, job in enumerate(self.jobs):
            label = "{0}   {1}   {2}   {3}".format(
                display_status(job.get("status")),
                job.get("kind", "?"),
                job.get("organ", "?"),
                job.get("job_id", "?"),
            )
            self.jobs_list.addItem(label)
            if job.get("job_id") == old_selected:
                selected_index = index
        self.jobs_list.blockSignals(False)
        if self.jobs:
            self.jobs_list.setCurrentRow(selected_index)
            self.selected_job_id = self.jobs[selected_index].get("job_id", "")
            self.show_job(self.jobs[selected_index])
            active = sum(1 for job in self.jobs if job.get("status") in ACTIVE_STATUSES)
            self.summary_label.setText("{0} job(s), {1} active. Auto-refresh every 2 seconds.".format(len(self.jobs), active))
        else:
            self.selected_job_id = ""
            self.summary_label.setText("No DINOv3 few-shot jobs were found.")
            self.detail_text.setPlainText("No jobs were found under:\n{0}".format(self.jobs_dir))
            self.log_text.setPlainText("")
            self.chart.set_rows([])

    def _load_jobs(self):
        rows = []
        paths = {}
        root = Path(self.jobs_dir)
        if not root.is_dir():
            return [], {}
        for path in root.glob("*.json"):
            if path.name.endswith("_context.json"):
                continue
            payload = read_json(path, {}) or {}
            if not payload:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            rows.append((mtime, payload))
            if payload.get("job_id"):
                paths[payload.get("job_id")] = str(path)
        rows.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rows[:80]], paths

    def on_select(self, row):
        if row < 0 or row >= len(self.jobs):
            return
        job = self.jobs[row]
        self.selected_job_id = job.get("job_id", "")
        self.show_job(job)

    def selected_job(self):
        if not self.selected_job_id:
            return None
        for job in self.jobs:
            if job.get("job_id") == self.selected_job_id:
                return job
        return None

    def show_job(self, job):
        lines = [
            "Job: {0}".format(job.get("job_id", "?")),
            "Type: {0}".format(job.get("kind", "?")),
            "Organ: {0}".format(job.get("organ", "?")),
            "Status: {0}".format(display_status(job.get("status"))),
            "Created: {0}".format(format_time(job.get("created_at_epoch"))),
            "Updated: {0}".format(format_time(job.get("updated_at_epoch"))),
        ]
        if job.get("case_id"):
            lines.append("Case: {0}".format(job.get("case_id")))
        if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
            lines.append("Samples: train {0}, validation {1}".format(
                job.get("train_sample_count", "?"),
                job.get("validation_sample_count", "?"),
            ))
        wait = resource_wait_text(job)
        if wait:
            lines.append(wait)
        progress = progress_line(job.get("training_progress") or {})
        if progress:
            lines.append("Progress: {0}".format(progress))
        if job.get("error"):
            lines.append("Error: {0}".format(job.get("error")))
        if job.get("train_log_warning"):
            lines.append("Training log warning: {0}".format(job.get("train_log_warning")))
        if job.get("log_warning"):
            lines.append("Log warning: {0}".format(job.get("log_warning")))
        if job.get("train_log_unavailable"):
            lines.append("Training log unavailable.")
        if job.get("log_unavailable"):
            lines.append("Inference log unavailable.")
        if job.get("cancel_marker_error"):
            lines.append("Cancel marker warning: {0}".format(job.get("cancel_marker_error")))
        if job.get("train_log"):
            lines.append("Train log: {0}".format(job.get("train_log")))
        if job.get("metrics_history"):
            lines.append("Metrics history: {0}".format(job.get("metrics_history")))
        if job.get("log"):
            lines.append("Inference log: {0}".format(job.get("log")))
        model = job.get("model") or {}
        if model.get("checkpoint"):
            lines.append("Model: {0}".format(model.get("checkpoint")))
        if job.get("output_path"):
            lines.append("Prediction: {0}".format(job.get("output_path")))
        self.detail_text.setPlainText("\n".join(lines))

        log_path = job.get("train_log") or job.get("log")
        pipeline_log = os.path.join(self.workspace, "fewshot_pipeline.log")
        log_text = read_log_text(log_path, max_bytes=2 * 1024 * 1024)
        pipeline_text = ""
        log_tail = tail_text_from_text(log_text, 80)
        if pipeline_log and pipeline_log != log_path:
            pipeline_text = read_log_text(pipeline_log, max_bytes=1024 * 1024)
            pipeline_tail = tail_text_from_text(pipeline_text, 60)
            if pipeline_tail:
                log_tail = (log_tail + "\n" if log_tail else "") + "---- pipeline log ----\n" + pipeline_tail
        self.log_text.setPlainText(log_tail)
        rows = training_curve_rows(job, log_text, pipeline_text)
        self.chart.set_rows(rows)
        self.stop_button.setEnabled(job.get("status") in ACTIVE_STATUSES)

    def open_log_folder(self):
        job = self.selected_job()
        path = ""
        if job:
            path = job.get("train_log") or job.get("log") or ""
        if path and os.path.isfile(path):
            open_path(os.path.dirname(path))
            return
        open_path(self.workspace)

    def request_stop(self):
        job = self.selected_job()
        if not job:
            return
        cancel_path = job.get("cancel_path")
        cancel_error = write_cancel_marker(cancel_path)
        killed = []
        for key in ("pid", "controller_pid", "launcher_pid"):
            pid = job.get(key)
            if pid and process_exists(pid) and terminate_process_tree(pid):
                killed.append(int(pid))
        job["status"] = "cancelled"
        job["cancel_requested_at_epoch"] = time.time()
        job["cancelled_pids"] = killed
        if cancel_error:
            job["cancel_marker_error"] = cancel_error
        job["updated_at_epoch"] = time.time()
        status_path = self.job_paths.get(job.get("job_id"))
        if status_path:
            write_json_best_effort(status_path, job)
        self.show_job(job)


def run_pyside6_ui(context):
    qt_modules = _load_pyside6()
    QtCore, QtGui, QtWidgets = qt_modules

    class CurveWidget(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.rows = []

        def set_rows(self, rows):
            self.rows = list(rows or [])
            self.update()

        def paintEvent(self, _event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            rect = self.rect()
            width = max(560, rect.width())
            height = max(240, rect.height())
            painter.fillRect(rect, QtGui.QColor("#f8fafc"))
            margin_l, margin_r, margin_t, margin_b = 76, 70, 54, 44
            x0, y0 = margin_l, height - margin_b
            x1, y1 = width - margin_r, margin_t
            plot_w = max(1, x1 - x0)
            plot_h = max(1, y0 - y1)
            painter.setPen(QtGui.QPen(QtGui.QColor("#d1d5db"), 1))
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawRect(QtCore.QRectF(x0, y1, plot_w, plot_h))
            painter.setPen(QtGui.QColor("#111827"))
            title_font = painter.font()
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(16, 24, "Training Progress")
            font = painter.font()
            font.setBold(False)
            painter.setFont(font)
            if not self.rows:
                painter.setPen(QtGui.QColor("#6b7280"))
                painter.drawText(rect, QtCore.Qt.AlignCenter, "No epoch metrics yet")
                return
            epochs = [row["epoch"] for row in self.rows]
            min_epoch, max_epoch = min(epochs), max(epochs)
            if min_epoch == max_epoch:
                max_epoch = min_epoch + 1
            losses = [row["train_loss"] for row in self.rows if row.get("train_loss") is not None]
            min_loss = min(losses) if losses else 0.0
            max_loss = max(losses) if losses else 1.0
            if min_loss == max_loss:
                pad = max(0.01, abs(min_loss) * 0.1)
                min_loss -= pad
                max_loss += pad
            else:
                pad = (max_loss - min_loss) * 0.12
                min_loss = max(0.0, min_loss - pad)
                max_loss += pad

            def x_at(epoch):
                return x0 + (float(epoch) - min_epoch) / float(max_epoch - min_epoch) * plot_w

            def y_loss(value):
                return y0 - (float(value) - min_loss) / float(max_loss - min_loss) * plot_h

            def y_dice(value):
                return y0 - max(0.0, min(1.0, float(value))) * plot_h

            for i in range(5):
                frac = i / 4.0
                y = y0 - frac * plot_h
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 1))
                painter.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
                painter.setPen(QtGui.QColor("#991b1b"))
                painter.drawText(8, int(y + 4), "{0:.3g}".format(min_loss + frac * (max_loss - min_loss)))
                painter.setPen(QtGui.QColor("#1d4ed8"))
                painter.drawText(int(x1 + 10), int(y + 4), "{0:.2f}".format(frac))

            epoch_ticks = sorted(set([min(epochs), max(epochs)] + [row["epoch"] for row in self.rows]))
            if len(epoch_ticks) > 6:
                step = max(1, int(round(len(epoch_ticks) / 5.0)))
                epoch_ticks = epoch_ticks[::step]
                if max(epochs) not in epoch_ticks:
                    epoch_ticks.append(max(epochs))
            painter.setPen(QtGui.QColor("#4b5563"))
            for epoch in epoch_ticks:
                x = x_at(epoch)
                painter.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, y0 + 4))
                painter.drawText(int(x - 8), y0 + 22, str(epoch))
            painter.drawText(x0, height - 12, "Epoch")
            painter.setPen(QtGui.QColor("#991b1b"))
            painter.drawText(x0 - 48, y1 - 18, "Loss")
            painter.setPen(QtGui.QColor("#1d4ed8"))
            painter.drawText(int(x1 + 18), y1 - 18, "Dice")

            loss_points = []
            dice_points = []
            for row in self.rows:
                if row.get("train_loss") is not None:
                    loss_points.append(QtCore.QPointF(x_at(row["epoch"]), y_loss(row["train_loss"])))
                if row.get("val_dice") is not None:
                    dice_points.append(QtCore.QPointF(x_at(row["epoch"]), y_dice(row["val_dice"])))
            if len(loss_points) >= 2:
                painter.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 2))
                painter.drawPolyline(QtGui.QPolygonF(loss_points))
            if len(dice_points) >= 2:
                painter.setPen(QtGui.QPen(QtGui.QColor("#2563eb"), 2))
                painter.drawPolyline(QtGui.QPolygonF(dice_points))
            for points, color in ((loss_points, "#dc2626"), (dice_points, "#2563eb")):
                if points:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
                    painter.setBrush(QtGui.QColor(color))
                    p = points[-1]
                    painter.drawEllipse(p, 4, 4)
            latest_loss = next((row.get("train_loss") for row in reversed(self.rows) if row.get("train_loss") is not None), None)
            latest_dice = next((row.get("val_dice") for row in reversed(self.rows) if row.get("val_dice") is not None), None)
            badge_x = 150
            badges = [("Epoch {0}/{1}".format(epochs[-1], max(epochs)), "#374151", "#f3f4f6")]
            if latest_loss is not None:
                badges.append(("loss {0:.4f}".format(float(latest_loss)), "#991b1b", "#fee2e2"))
            if latest_dice is not None:
                badges.append(("val dice {0:.4f}".format(float(latest_dice)), "#1d4ed8", "#dbeafe"))
            for text, color, fill in badges:
                badge_w = max(78, len(text) * 7 + 22)
                painter.setPen(QtGui.QColor("#e5e7eb"))
                painter.setBrush(QtGui.QColor(fill))
                painter.drawRect(int(badge_x), 12, int(badge_w), 25)
                painter.setPen(QtGui.QColor(color))
                painter.drawText(int(badge_x + 10), 29, text)
                badge_x += badge_w + 8
            legend_x = int(x1 - 178)
            painter.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 2))
            painter.drawLine(legend_x, 24, legend_x + 24, 24)
            painter.setPen(QtGui.QColor("#374151"))
            painter.drawText(legend_x + 32, 29, "train loss")
            painter.setPen(QtGui.QPen(QtGui.QColor("#2563eb"), 2))
            painter.drawLine(legend_x + 104, 24, legend_x + 128, 24)
            painter.setPen(QtGui.QColor("#374151"))
            painter.drawText(legend_x + 136, 29, "val dice")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(TITLE)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    window = QtWidgets.QMainWindow()
    QtStatusViewerApp(window, context, qt_modules, CurveWidget)
    window.show()
    return app.exec()


def run_ui(context):
    backend = os.environ.get("MIMICS_DINOV3_GUI_BACKEND", "auto").strip().lower()
    if backend in ("", "auto", "pyside6", "qt"):
        try:
            return run_pyside6_ui(context)
        except Exception:
            if backend in ("pyside6", "qt"):
                raise
    try:
        import tkinter as tk
    except Exception as exc:
        raise RuntimeError(
            "The external DINOv3 status window could not open because neither PySide6 nor Tkinter is "
            "available in the configured external Python environment: {0}".format(exc)
        )
    root = tk.Tk()
    StatusViewerApp(root, context)
    root.mainloop()


def generate_preview(path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required only for --preview PNG generation. Runtime status viewing does not "
            "depend on Pillow. Import error: {0}".format(exc)
        )

    width, height = 1280, 900
    image = Image.new("RGB", (width, height), "#f4f5f7")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 30)
        head_font = ImageFont.truetype("Arial.ttf", 19)
        font = ImageFont.truetype("Arial.ttf", 16)
        small = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        title_font = head_font = font = small = ImageFont.load_default()

    draw.rectangle((0, 0, width, 70), fill="#1f2937")
    draw.text((32, 20), "DINOv3 Few-Shot Status", fill="white", font=title_font)
    draw.text((32, 92), "Dataset: D:\\Dataset\\TotalSegmentator    Organ filter: liver", fill="#1f2937", font=head_font)
    for idx, label in enumerate(["Refresh", "Open Workspace", "Open Log Folder", "Request Stop"]):
        x = 32 + idx * 155
        draw.rectangle((x, 132, x + 140, 170), outline="#9ca3af", fill="#ffffff")
        draw.text((x + 16, 143), label, fill="#111827", font=font)

    draw.text((32, 194), "12 job(s), 1 active. Auto-refresh every 2 seconds.", fill="#374151", font=font)
    draw.rectangle((32, 225, 420, height - 35), outline="#d1d5db", fill="#ffffff")
    draw.text((50, 244), "Jobs", fill="#111827", font=head_font)
    rows = [
        ("Training", "train", "liver", "train_20260708_01", True),
        ("Completed", "train", "spleen", "train_20260707_03", False),
        ("Completed", "infer", "liver", "infer_s0401_liver", False),
        ("Failed", "train", "kidney_left", "train_20260706_02", False),
    ]
    y = 282
    for status, kind, organ, job_id, selected in rows:
        if selected:
            draw.rectangle((44, y - 6, 408, y + 31), fill="#dbeafe")
        draw.text((58, y), "{0}   {1}   {2}".format(status, kind, organ), fill="#111827", font=font)
        draw.text((58, y + 18), job_id, fill="#6b7280", font=small)
        y += 52

    right_x = 450
    draw.rectangle((right_x, 225, width - 32, 392), outline="#d1d5db", fill="#ffffff")
    draw.text((right_x + 18, 244), "Selected job", fill="#111827", font=head_font)
    details = [
        "Job: train_20260708_01",
        "Type: train",
        "Organ: liver",
        "Status: Training",
        "Samples: train 8, validation 2",
        "Progress: Epoch 6/10: train_loss=0.3182, val_dice=0.7421",
        "Train log: ...\\fewshot_models\\runs\\liver\\train_20260708_01\\train.log",
    ]
    y = 276
    for line in details:
        draw.text((right_x + 18, y), line, fill="#374151", font=font)
        y += 18

    chart_y = 420
    draw.rectangle((right_x, chart_y, width - 32, chart_y + 240), outline="#d1d5db", fill="#f8fafc")
    draw.text((right_x + 18, chart_y + 16), "Training curve", fill="#111827", font=head_font)
    badge_x = right_x + 185
    for label, color, fill, w in [
        ("Epoch 6/10", "#374151", "#f3f4f6", 100),
        ("loss 0.3182", "#991b1b", "#fee2e2", 110),
        ("val dice 0.7421", "#1d4ed8", "#dbeafe", 135),
    ]:
        draw.rectangle((badge_x, chart_y + 11, badge_x + w, chart_y + 37), outline="#e5e7eb", fill=fill)
        draw.text((badge_x + 10, chart_y + 17), label, fill=color, font=small)
        badge_x += w + 8

    gx0, gy0, gx1, gy1 = right_x + 78, chart_y + 190, width - 92, chart_y + 70
    draw.rectangle((gx0, gy1, gx1, gy0), outline="#d1d5db", fill="#ffffff")
    for i in range(5):
        y_tick = gy0 - i * (gy0 - gy1) / 4.0
        draw.line((gx0, y_tick, gx1, y_tick), fill="#e5e7eb")
        draw.text((gx0 - 48, y_tick - 7), ["0.26", "0.31", "0.36", "0.41", "0.46"][i], fill="#991b1b", font=small)
        draw.text((gx1 + 10, y_tick - 7), ["0.00", "0.25", "0.50", "0.75", "1.00"][i], fill="#1d4ed8", font=small)
    draw.text((gx0 - 42, gy1 - 24), "Loss", fill="#991b1b", font=small)
    draw.text((gx1 + 18, gy1 - 24), "Dice", fill="#1d4ed8", font=small)
    loss_points = [(gx0, gy1 + 22), (gx0 + 120, gy1 + 42), (gx0 + 240, gy1 + 60), (gx0 + 360, gy1 + 78), (gx0 + 480, gy1 + 96)]
    dice_points = [(gx0, gy0 - 30), (gx0 + 120, gy0 - 55), (gx0 + 240, gy0 - 74), (gx0 + 360, gy0 - 88), (gx0 + 480, gy0 - 104)]
    draw.line(loss_points, fill="#dc2626", width=3)
    draw.line(dice_points, fill="#2563eb", width=3)
    for point, color in [(loss_points[-1], "#dc2626"), (dice_points[-1], "#2563eb")]:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#ffffff")
    draw.text((gx0, gy0 + 14), "1", fill="#4b5563", font=small)
    draw.text((gx0 + 240, gy0 + 14), "3", fill="#4b5563", font=small)
    draw.text((gx0 + 480, gy0 + 14), "6", fill="#4b5563", font=small)
    draw.line((gx1 - 170, chart_y + 28, gx1 - 146, chart_y + 28), fill="#dc2626", width=3)
    draw.text((gx1 - 138, chart_y + 20), "train loss", fill="#374151", font=small)
    draw.line((gx1 - 58, chart_y + 28, gx1 - 34, chart_y + 28), fill="#2563eb", width=3)
    draw.text((gx1 - 26, chart_y + 20), "val dice", fill="#374151", font=small)

    log_y = 690
    draw.rectangle((right_x, log_y, width - 32, height - 35), outline="#d1d5db", fill="#ffffff")
    draw.text((right_x + 18, log_y + 16), "Log tail", fill="#111827", font=head_font)
    logs = [
        "[2026-07-08 14:20:03] Training job train_20260708_01 started for organ liver.",
        "[2026-07-08 14:22:11] Training progress: Epoch 4/10: train_loss=0.3921, val_dice=0.7015",
        "[2026-07-08 14:24:38] Training progress: Epoch 5/10: train_loss=0.3480, val_dice=0.7288",
        "[2026-07-08 14:27:02] Training progress: Epoch 6/10: train_loss=0.3182, val_dice=0.7421",
    ]
    y = log_y + 52
    for line in logs:
        draw.text((right_x + 18, y), line, fill="#374151", font=small)
        y += 24

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", help="Path to the status viewer context JSON written by Mimics.")
    parser.add_argument("--ts-root")
    parser.add_argument("--workspace")
    parser.add_argument("--organ")
    parser.add_argument("--preview", help="Write a static PNG preview of the status UI and exit.")
    args = parser.parse_args(argv)
    if args.preview:
        generate_preview(args.preview)
        print(args.preview)
        return 0
    if args.context:
        context = read_json(args.context, None)
        if not context:
            raise RuntimeError("Could not read status viewer context: {0}".format(args.context))
    elif args.ts_root:
        context = {
            "ts_root": os.path.abspath(args.ts_root),
            "workspace": os.path.abspath(args.workspace) if args.workspace else os.path.join(os.path.abspath(args.ts_root), "fewshot_models"),
            "selected_organ": args.organ or "",
        }
    else:
        parser.error("--context or --ts-root is required unless --preview is used")
    run_ui(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
