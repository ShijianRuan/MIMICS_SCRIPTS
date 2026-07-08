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


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(path))


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


def tail_text(path, max_lines=80):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-max_lines:])
    except TypeError:
        with open(path, "r") as handle:
            lines = handle.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s*train_loss=([0-9eE+\-.]+)(?:.*?val_dice=([0-9eE+\-.]+))?"
)


def parse_epoch_metrics(*paths):
    rows = []
    seen = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except TypeError:
            try:
                text = Path(path).read_text()
            except Exception:
                continue
        except Exception:
            continue
        for match in EPOCH_RE.finditer(text):
            epoch = int(match.group(1))
            total = int(match.group(2))
            key = (epoch, total, match.group(3), match.group(4))
            if key in seen:
                continue
            seen.add(key)
            row = {
                "epoch": epoch,
                "epochs": total,
                "train_loss": float(match.group(3)),
                "val_dice": None,
            }
            if match.group(4) is not None:
                row["val_dice"] = float(match.group(4))
            rows.append(row)
    rows.sort(key=lambda item: item["epoch"])
    return rows


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
        if job.get("train_log"):
            lines.append("Train log: {0}".format(job.get("train_log")))
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
        log_tail = tail_text(log_path, 80)
        if pipeline_log and pipeline_log != log_path:
            pipeline_tail = tail_text(pipeline_log, 60)
            if pipeline_tail:
                log_tail = (log_tail + "\n" if log_tail else "") + "---- pipeline log ----\n" + pipeline_tail
        self._set_text(self.log_text, log_tail)
        rows = parse_epoch_metrics(log_path, pipeline_log)
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
        if cancel_path:
            Path(cancel_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cancel_path).write_text(
                "cancel requested at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")),
                encoding="utf-8",
            )
        killed = []
        for key in ("pid", "controller_pid", "launcher_pid"):
            pid = job.get(key)
            if pid and process_exists(pid) and terminate_process_tree(pid):
                killed.append(int(pid))
        job["status"] = "cancelled"
        job["cancel_requested_at_epoch"] = time.time()
        job["cancelled_pids"] = killed
        job["updated_at_epoch"] = time.time()
        status_path = self.job_paths.get(job.get("job_id"))
        if status_path:
            write_json_atomic(status_path, job)
        self.show_job(job)


def run_ui(context):
    try:
        import tkinter as tk
    except Exception as exc:
        raise RuntimeError(
            "The external DINOv3 status window could not open because Tkinter is not available "
            "in the configured external Python environment: {0}".format(exc)
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
