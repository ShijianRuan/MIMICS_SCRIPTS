#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External DINOv3 few-shot training setup UI.

This tool is intentionally outside the foreground Mimics Python process.  The
Mimics entry starts it with Popen and returns immediately.  The UI writes a
setup/job status JSON file so Mimics can keep monitoring without blocking.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import uuid


TITLE = "DINOv3 Few-Shot Training"


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def safe_slug(value):
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", ".", " "):
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unnamed"


def split_csv(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def hidden_process_kwargs():
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def default_training_options(config, profile_name=None):
    profiles = config.get("training_profiles") or {}
    default_profile = profile_name or config.get("default_training_profile") or config.get("default_profile")
    values = {}
    if default_profile and isinstance(profiles, dict):
        values.update(profiles.get(default_profile, {}) or {})
    values.setdefault("base_config", config.get("base_config", "config/synthstrip_lora_segformer3d.yaml"))
    values.setdefault("epochs", config.get("default_epochs", 10))
    values.setdefault("batch_size", config.get("default_batch_size", 1))
    values.setdefault("grad_accumulation", config.get("default_grad_accumulation", 1))
    values.setdefault("lr", config.get("default_lr", 0.001))
    values.setdefault("weight_decay", config.get("default_weight_decay", 0.01))
    values.setdefault("img_size", config.get("default_img_size", "224,224"))
    values.setdefault("modality", config.get("default_modality", "ct"))
    values.setdefault("min_samples", config.get("default_min_samples", 1))
    values.setdefault("max_samples", config.get("default_max_samples", 0))
    values.setdefault("sample_mode", config.get("default_sample_mode", "all"))
    values.setdefault("val_fraction", config.get("default_val_fraction", 0.2))
    values.setdefault("min_val_samples", config.get("default_min_val_samples", 1))
    values.setdefault("finetune_method", config.get("default_finetune_method", "lora"))
    values.setdefault("decoder", config.get("default_decoder", "segformer3d"))
    values.setdefault("model_scale", config.get("default_model_scale", "vitb16"))
    values.setdefault("model_path", config.get("default_model_path", ""))
    values.setdefault("lora_rank", config.get("default_lora_rank", 8))
    values.setdefault("lora_alpha", config.get("default_lora_alpha", 16))
    values.setdefault("adapter_bottleneck", config.get("default_adapter_bottleneck", 64))
    values.setdefault("mixed_precision", config.get("default_mixed_precision", False))
    values.setdefault("sub_volume", config.get("default_sub_volume", False))
    values.setdefault("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))
    values.setdefault("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2))
    values.setdefault("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault(
        "background_mimics_lock_timeout_seconds",
        config.get("background_mimics_lock_timeout_seconds", 21600),
    )
    return values


def _float(value, name):
    try:
        return float(value)
    except Exception:
        raise ValueError("{0} must be a number.".format(name))


def _int(value, name, minimum=None):
    try:
        result = int(value)
    except Exception:
        raise ValueError("{0} must be an integer.".format(name))
    if minimum is not None and result < minimum:
        raise ValueError("{0} must be at least {1}.".format(name, minimum))
    return result


def validate_options(options):
    normalized = dict(options)
    normalized["epochs"] = _int(normalized.get("epochs", 10), "Epochs", 1)
    normalized["batch_size"] = _int(normalized.get("batch_size", 1), "Batch size", 1)
    normalized["grad_accumulation"] = _int(normalized.get("grad_accumulation", 1), "Grad accumulation", 1)
    normalized["min_samples"] = _int(normalized.get("min_samples", 1), "Min train samples", 1)
    normalized["max_samples"] = _int(normalized.get("max_samples", 0), "Max samples", 0)
    normalized["min_val_samples"] = _int(normalized.get("min_val_samples", 1), "Min validation samples", 0)
    normalized["lora_rank"] = _int(normalized.get("lora_rank", 8), "LoRA rank", 1)
    normalized["lora_alpha"] = _int(normalized.get("lora_alpha", 16), "LoRA alpha", 1)
    normalized["adapter_bottleneck"] = _int(normalized.get("adapter_bottleneck", 64), "Adapter bottleneck", 1)
    normalized["keep_last_checkpoints"] = _int(normalized.get("keep_last_checkpoints", 2), "Keep last checkpoints", 0)
    normalized["lr"] = _float(normalized.get("lr", 0.001), "Learning rate")
    normalized["weight_decay"] = _float(normalized.get("weight_decay", 0.01), "Weight decay")
    normalized["val_fraction"] = _float(normalized.get("val_fraction", 0.2), "Validation fraction")
    normalized["mixed_precision"] = _bool(normalized.get("mixed_precision", False))
    normalized["sub_volume"] = _bool(normalized.get("sub_volume", False))
    normalized["keep_materialized_dataset"] = _bool(normalized.get("keep_materialized_dataset", False))
    if normalized["val_fraction"] < 0.0 or normalized["val_fraction"] > 0.9:
        raise ValueError("Validation fraction must be between 0.0 and 0.9.")
    parts = [part.strip() for part in str(normalized.get("img_size", "224,224")).split(",")]
    if len(parts) != 2:
        raise ValueError("Image size must be formatted as width,height.")
    [_int(part, "Image size", 1) for part in parts]
    sv_parts = [part.strip() for part in str(normalized.get("sub_volume_size", "32,256,256")).split(",")]
    if len(sv_parts) != 3:
        raise ValueError("Sub-volume size must be formatted as z,y,x.")
    [_int(part, "Sub-volume size", 1) for part in sv_parts]
    return normalized


def append_training_args(cmd, config, options):
    cmd.extend([
        "--base-config",
        str(options.get("base_config", config.get("base_config", "config/synthstrip_lora_segformer3d.yaml"))),
        "--epochs",
        str(int(options.get("epochs", config.get("default_epochs", 10)))),
        "--batch-size",
        str(int(options.get("batch_size", config.get("default_batch_size", 1)))),
        "--grad-accumulation",
        str(int(options.get("grad_accumulation", config.get("default_grad_accumulation", 1)))),
        "--lr",
        str(float(options.get("lr", config.get("default_lr", 0.001)))),
        "--weight-decay",
        str(float(options.get("weight_decay", config.get("default_weight_decay", 0.01)))),
        "--img-size",
        str(options.get("img_size", config.get("default_img_size", "224,224"))),
        "--modality",
        str(options.get("modality", config.get("default_modality", "ct"))),
        "--min-samples",
        str(int(options.get("min_samples", config.get("default_min_samples", 1)))),
        "--max-samples",
        str(int(options.get("max_samples", config.get("default_max_samples", 0)))),
        "--sample-mode",
        str(options.get("sample_mode", config.get("default_sample_mode", "all"))),
        "--val-fraction",
        str(float(options.get("val_fraction", config.get("default_val_fraction", 0.2)))),
        "--min-val-samples",
        str(int(options.get("min_val_samples", config.get("default_min_val_samples", 1)))),
        "--finetune-method",
        str(options.get("finetune_method", config.get("default_finetune_method", "lora"))),
        "--decoder",
        str(options.get("decoder", config.get("default_decoder", "segformer3d"))),
        "--model-scale",
        str(options.get("model_scale", config.get("default_model_scale", "vitb16"))),
        "--lora-rank",
        str(int(options.get("lora_rank", config.get("default_lora_rank", 8)))),
        "--lora-alpha",
        str(int(options.get("lora_alpha", config.get("default_lora_alpha", 16)))),
        "--adapter-bottleneck",
        str(int(options.get("adapter_bottleneck", config.get("default_adapter_bottleneck", 64)))),
        "--gpu-lock-timeout-seconds",
        str(float(options.get("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400)))),
        "--background-mimics-lock-timeout-seconds",
        str(float(options.get(
            "background_mimics_lock_timeout_seconds",
            config.get("background_mimics_lock_timeout_seconds", 21600),
        ))),
        "--keep-last-checkpoints",
        str(int(options.get("keep_last_checkpoints", config.get("default_keep_last_checkpoints", 2)))),
    ])
    model_path = str(options.get("model_path", config.get("default_model_path", "")) or "")
    if model_path:
        cmd.extend(["--model-path", model_path])
    val_cases = split_csv(options.get("val_cases", ""))
    if val_cases:
        cmd.extend(["--val-cases", ",".join(val_cases)])
    cases = split_csv(options.get("cases", ""))
    if cases:
        cmd.extend(["--cases", ",".join(cases)])
    if bool(options.get("mixed_precision", config.get("default_mixed_precision", False))):
        cmd.append("--mixed-precision")
    if bool(options.get("sub_volume", config.get("default_sub_volume", False))):
        cmd.append("--sub-volume")
    cmd.extend([
        "--sub-volume-size",
        str(options.get("sub_volume_size", config.get("default_sub_volume_size", "32,256,256"))),
    ])
    if bool(options.get("keep_materialized_dataset", config.get("default_keep_materialized_dataset", False))):
        cmd.append("--keep-materialized-dataset")


def prepare_training_launch(context, options, run_id=None):
    config = context.get("config") or {}
    options = validate_options(options)
    organ = context["organ"]
    ts_root = context["ts_root"]
    workspace = context["workspace"]
    python_exe = context.get("python_exe") or sys.executable
    run_id = run_id or "train_{0}_{1}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    organ_slug = safe_slug(organ)
    cancel_path = os.path.join(workspace, "runs", organ_slug, run_id, "cancel.request")
    status_path = os.path.join(workspace, "jobs", run_id + ".json")
    cmd = [
        python_exe,
        context["pipeline_script"],
        "train",
        "--ts-root",
        ts_root,
        "--organ",
        organ,
        "--dinov3-root",
        context["dinov3_root"],
        "--python",
        python_exe,
        "--export-labels",
        "--run-id",
        run_id,
    ]
    append_training_args(cmd, config, options)
    mimics_exe = context.get("mimics_exe")
    if mimics_exe:
        cmd.extend(["--mimics-exe", mimics_exe])
    job_payload = {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": run_id,
        "kind": "train",
        "status": "launching",
        "organ": organ,
        "ts_root": os.path.abspath(ts_root),
        "workspace": workspace,
        "cancel_path": cancel_path,
        "training_options": options,
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "launched_by": "external_advanced_ui",
    }
    return {
        "cmd": cmd,
        "run_id": run_id,
        "status_path": status_path,
        "cancel_path": cancel_path,
        "job_payload": job_payload,
        "options": options,
    }


def launch_training(context, options):
    launch = prepare_training_launch(context, options)
    write_json_atomic(launch["status_path"], launch["job_payload"])
    try:
        process = subprocess.Popen(
            launch["cmd"],
            cwd=context.get("project_root") or None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_process_kwargs()
        )
    except Exception:
        payload = read_json(launch["status_path"], launch["job_payload"]) or launch["job_payload"]
        payload["status"] = "failed"
        payload["error"] = "Could not start DINOv3 training process."
        payload["traceback"] = traceback.format_exc()
        payload["updated_at_epoch"] = time.time()
        write_json_atomic(launch["status_path"], payload)
        raise
    payload = read_json(launch["status_path"], launch["job_payload"]) or launch["job_payload"]
    payload["launcher_pid"] = process.pid
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(launch["status_path"], payload)
    setup_status_path = context.get("setup_status_path")
    if setup_status_path:
        write_json_atomic(setup_status_path, {
            "schema_version": "mimics_fewshot_setup.v1",
            "job_id": context.get("setup_id"),
            "kind": "train_setup",
            "status": "training_started",
            "organ": context.get("organ"),
            "ts_root": os.path.abspath(context.get("ts_root", "")),
            "workspace": context.get("workspace"),
            "training_job_id": launch["run_id"],
            "training_status_path": launch["status_path"],
            "launcher_pid": process.pid,
            "updated_at_epoch": time.time(),
        })
    return launch["run_id"], launch["status_path"], process.pid


def _bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def format_status_line(job):
    if not isinstance(job, dict):
        return ""
    parts = []
    job_id = job.get("job_id")
    status = job.get("status")
    if job_id or status:
        parts.append("{0} | {1}".format(job_id or "job", status or "unknown"))
    if job.get("train_sample_count") is not None or job.get("validation_sample_count") is not None:
        parts.append("train {0}, val {1}".format(
            job.get("train_sample_count", "?"),
            job.get("validation_sample_count", "?"),
        ))
    progress = job.get("training_progress") or {}
    if isinstance(progress, dict):
        if progress.get("latest_epoch_line"):
            parts.append(str(progress.get("latest_epoch_line")))
        else:
            progress_parts = []
            if progress.get("epoch") is not None and progress.get("epochs") is not None:
                progress_parts.append("epoch {0}/{1}".format(progress.get("epoch"), progress.get("epochs")))
            if progress.get("phase"):
                progress_parts.append(str(progress.get("phase")))
            if progress.get("batch") is not None and progress.get("batches") is not None:
                progress_parts.append("batch {0}/{1}".format(progress.get("batch"), progress.get("batches")))
            metrics = progress.get("metrics") or {}
            if metrics.get("loss") is not None:
                try:
                    progress_parts.append("loss {0:.4f}".format(float(metrics.get("loss"))))
                except Exception:
                    progress_parts.append("loss {0}".format(metrics.get("loss")))
            if metrics.get("mean_dsc") is not None:
                try:
                    progress_parts.append("val_dice {0:.4f}".format(float(metrics.get("mean_dsc"))))
                except Exception:
                    progress_parts.append("val_dice {0}".format(metrics.get("mean_dsc")))
            if progress.get("best_dsc") is not None:
                try:
                    progress_parts.append("best {0:.4f}".format(float(progress.get("best_dsc"))))
                except Exception:
                    progress_parts.append("best {0}".format(progress.get("best_dsc")))
            if progress_parts:
                parts.append(", ".join(progress_parts))
    if job.get("error"):
        parts.append("error: {0}".format(job.get("error")))
    return " | ".join(parts)


def _option_label(value, labels):
    for label, item in labels:
        if str(item) == str(value):
            return label
    return "Custom ({0})".format(value)


def _labels_with_current(labels, current_label):
    values = [label for label, _item in labels]
    if current_label not in values:
        values.append(current_label)
    return values


def _choice_value(label, labels):
    for item_label, value in labels:
        if item_label == label:
            return value
    return None


def window_layout_for_screen(screen_width, screen_height):
    """Return geometry/minsize values that keep the action footer visible."""
    try:
        screen_width = int(screen_width)
        screen_height = int(screen_height)
    except Exception:
        screen_width, screen_height = 1280, 900
    width = min(1040, max(860, screen_width - 80))
    height = min(760, max(560, screen_height - 120))
    min_width = min(860, width)
    min_height = min(560, height)
    return width, height, min_width, min_height


class TrainingSetupApp(object):
    EPOCH_CHOICES = [("Fast check (3)", 3), ("Quick (5)", 5), ("Standard (10)", 10), ("More training (20)", 20)]
    VAL_CHOICES = [("No validation", 0.0), ("Small validation (10%)", 0.1), ("Standard validation (20%)", 0.2), ("Larger validation (30%)", 0.3)]
    MEMORY_CHOICES = [
        ("Balanced", "balanced"),
        ("Low GPU memory", "low_memory"),
        ("Higher quality", "quality"),
        ("Custom", "custom"),
    ]
    LR_CHOICES = ["0.001", "0.0005", "0.0002", "0.0001"]
    WEIGHT_DECAY_CHOICES = ["0.01", "0.001", "0.0001", "0.0"]
    IMG_SIZE_CHOICES = [
        ("Fast (192 x 192)", "192,192"),
        ("Balanced (224 x 224)", "224,224"),
        ("Detailed (256 x 256)", "256,256"),
        ("High detail (320 x 320)", "320,320"),
    ]
    IMG_SIZE_CUSTOM_LABEL = "Custom"
    SUB_VOLUME_DEPTH_CHOICES = ["16", "24", "32", "48", "64"]

    def __init__(self, root, context):
        self.root = root
        self.context = context
        self.config = context.get("config") or {}
        self.profiles = self.config.get("training_profiles") or {}
        self.profile_names = sorted(self.profiles.keys()) if isinstance(self.profiles, dict) else []
        default_profile = self.config.get("default_training_profile") or (self.profile_names[0] if self.profile_names else "")
        self.values = default_training_options(self.config, default_profile)
        self.started = False
        self.status_var = None
        self.vars = {}
        self.case_list = None
        self.manual_cases_var = None
        self.profile_var = None
        self.status_text = None
        self.start_button = None
        self.open_log_button = None
        self.close_button = None
        self.footer_frame = None
        self.training_status_path = None
        self.last_status_line = ""
        self.training_log_dir = ""
        self.epochs_choice_widget = None
        self.val_choice_widget = None
        self.memory_mode_widget = None
        self.img_size_choice_widget = None
        self.img_size_custom_widget = None
        self._syncing_quick = False
        self._build()

    def _build(self):
        import tkinter as tk
        from tkinter import ttk

        self.root.title(TITLE)
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
        except Exception:
            screen_width, screen_height = 1280, 900
        width, height, min_width, min_height = window_layout_for_screen(screen_width, screen_height)
        self.root.geometry("{0}x{1}".format(width, height))
        self.root.minsize(min_width, min_height)
        try:
            self.root.attributes("-topmost", False)
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.root, padding=(14, 14, 14, 8))
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        heading = ttk.Frame(outer)
        heading.grid(row=0, column=0, sticky="ew")
        title = ttk.Label(heading, text="DINOv3 Few-Shot Training", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")
        sub = ttk.Label(
            heading,
            text="Organ: {0}    Dataset: {1}".format(self.context.get("organ", "?"), self.context.get("ts_root", "?")),
        )
        sub.pack(anchor="w", pady=(4, 0))
        remind = ttk.Label(
            outer,
            text="Save edited .mcs projects before starting. Training exports labels from saved projects in the background.",
            foreground="#9a5b00",
        )
        remind.grid(row=1, column=0, sticky="w", pady=(8, 8))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=2, column=0, sticky="nsew")
        setup_tab = ttk.Frame(notebook, padding=10)
        expert_tab = ttk.Frame(notebook, padding=10)
        notebook.add(setup_tab, text="Setup")
        notebook.add(expert_tab, text="Expert")
        self._build_setup(setup_tab)
        self._build_expert(expert_tab)

        status_box = ttk.LabelFrame(outer, text="Status", padding=8)
        status_box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status_box.columnconfigure(0, weight=1)
        self.status_text = tk.Text(status_box, height=4, wrap="word", state="disabled")
        status_scroll = ttk.Scrollbar(status_box, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        self.status_text.grid(row=0, column=0, sticky="ew")
        status_scroll.grid(row=0, column=1, sticky="ns")
        self._append_log("Ready. Choose a profile and samples, then start background training.")

        footer = ttk.Frame(self.root, padding=(14, 8, 14, 12))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.footer_frame = footer
        self.status_var = tk.StringVar(value="Configure samples and parameters, then start background training.")
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="ew")
        self.start_button = ttk.Button(footer, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self.open_log_button = ttk.Button(footer, text="Open Log Folder", command=self.open_log_folder)
        self.open_log_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        try:
            self.open_log_button.state(["disabled"])
        except Exception:
            pass
        self.close_button = ttk.Button(footer, text="Cancel", command=self.close)
        self.close_button.grid(row=0, column=1, sticky="e")
        try:
            self.root.bind("<Return>", lambda _event: self.start_training())
            self.root.bind("<Escape>", lambda _event: self.close())
        except Exception:
            pass

    def _build_setup(self, parent):
        import tkinter as tk
        from tkinter import ttk

        for key, value in self.values.items():
            if key not in self.vars:
                if isinstance(value, bool):
                    self.vars[key] = tk.BooleanVar(value=value)
                else:
                    self.vars[key] = tk.StringVar(value=str(value))

        top = ttk.LabelFrame(parent, text="Training profile", padding=10)
        top.pack(fill="x")
        self.profile_var = tk.StringVar(value=self.config.get("default_training_profile") or (self.profile_names[0] if self.profile_names else ""))
        ttk.Label(top, text="Profile").pack(side="left")
        profile = ttk.Combobox(top, textvariable=self.profile_var, values=self.profile_names, state="readonly", width=24)
        profile.pack(side="left", padx=(8, 16))
        profile.bind("<<ComboboxSelected>>", lambda _event: self.apply_profile())
        ttk.Label(top, text="Use profiles for routine work; Expert is optional.").pack(side="left")

        quick = ttk.LabelFrame(parent, text="Key settings", padding=10)
        quick.pack(fill="x", pady=(10, 0))
        epoch_label = _option_label(self.values.get("epochs", 10), self.EPOCH_CHOICES)
        val_label = _option_label(self.values.get("val_fraction", 0.2), self.VAL_CHOICES)
        self.vars["epochs_choice"] = tk.StringVar(value=epoch_label)
        self.vars["val_fraction_choice"] = tk.StringVar(value=val_label)
        self.vars["memory_mode"] = tk.StringVar(value=self._memory_mode_from_values(self.values))

        quick_row = ttk.Frame(quick)
        quick_row.pack(fill="x")
        col1 = ttk.Frame(quick_row)
        col1.pack(side="left", fill="x", expand=True, padx=(0, 8))
        col2 = ttk.Frame(quick_row)
        col2.pack(side="left", fill="x", expand=True, padx=(0, 8))
        col3 = ttk.Frame(quick_row)
        col3.pack(side="left", fill="x", expand=True)

        self.epochs_choice_widget = self._labeled_combo_inline(
            col1, "Training length", "epochs_choice",
            _labels_with_current(self.EPOCH_CHOICES, epoch_label),
        )
        self.val_choice_widget = self._labeled_combo_inline(
            col2, "Validation", "val_fraction_choice",
            _labels_with_current(self.VAL_CHOICES, val_label),
        )
        self.memory_mode_widget = self._labeled_combo_inline(
            col3, "Resource preset", "memory_mode",
            [label for label, _ in self.MEMORY_CHOICES],
        )
        self.epochs_choice_widget.bind("<<ComboboxSelected>>", self._sync_quick_settings)
        self.val_choice_widget.bind("<<ComboboxSelected>>", self._sync_quick_settings)
        self.memory_mode_widget.bind("<<ComboboxSelected>>", self._sync_quick_settings)

        samples = ttk.LabelFrame(parent, text="Samples", padding=10)
        samples.pack(fill="both", expand=True, pady=(10, 0))
        sample_controls = ttk.Frame(samples)
        sample_controls.pack(fill="x")
        self.vars["sample_mode"] = tk.StringVar(value=str(self.values.get("sample_mode", "all")))
        ttk.Label(sample_controls, text="Sample order").pack(side="left")
        ttk.Combobox(sample_controls, textvariable=self.vars["sample_mode"], values=["all", "latest"], state="readonly", width=10).pack(side="left", padx=(8, 16))
        self.vars["max_samples"] = tk.StringVar(value=str(self.values.get("max_samples", 0)))
        ttk.Label(sample_controls, text="Max samples").pack(side="left")
        ttk.Spinbox(sample_controls, textvariable=self.vars["max_samples"], from_=0, to=100000, increment=1, width=8).pack(side="left", padx=(8, 0))

        mid = ttk.Frame(samples)
        mid.pack(fill="both", expand=True, pady=(12, 8))
        self.case_list = tk.Listbox(mid, selectmode="extended", exportselection=False, height=12)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=self.case_list.yview)
        self.case_list.configure(yscrollcommand=scrollbar.set)
        self.case_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")
        for case_id in self.context.get("case_ids", []) or []:
            self.case_list.insert("end", str(case_id))
        side = ttk.Frame(mid)
        side.pack(side="left", fill="y", padx=(12, 0))
        ttk.Button(side, text="Select All", command=lambda: self.case_list.select_set(0, "end")).pack(fill="x")
        ttk.Button(side, text="Clear", command=lambda: self.case_list.selection_clear(0, "end")).pack(fill="x", pady=(6, 0))
        ttk.Label(side, text="{0} case(s) found".format(len(self.context.get("case_ids", []) or []))).pack(anchor="w", pady=(18, 0))
        ttk.Label(side, text="Selected cases override sample order.").pack(anchor="w", pady=(8, 0))

        bottom = ttk.Frame(samples)
        bottom.pack(fill="x")
        self.manual_cases_var = tk.StringVar(value="")
        ttk.Label(bottom, text="Manual cases").pack(side="left")
        ttk.Entry(bottom, textvariable=self.manual_cases_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        vals = ttk.Frame(samples)
        vals.pack(fill="x", pady=(10, 0))
        self.vars["min_samples"] = tk.StringVar(value=str(self.values.get("min_samples", 1)))
        self.vars["min_val_samples"] = tk.StringVar(value=str(self.values.get("min_val_samples", 1)))
        self._labeled_entry(vals, "Min train", "min_samples", 8)
        self._labeled_entry(vals, "Min val", "min_val_samples", 8)

    def _build_expert(self, parent):
        import tkinter as tk
        from tkinter import ttk

        for key, value in self.values.items():
            if key not in self.vars:
                if isinstance(value, bool):
                    self.vars[key] = tk.BooleanVar(value=value)
                else:
                    self.vars[key] = tk.StringVar(value=str(value))
        self.vars["val_fraction"] = tk.StringVar(value=str(self.values.get("val_fraction", 0.2)))
        self.vars["sub_volume_depth"] = tk.StringVar(value=self._sub_volume_depth_from_values(self.values))
        self.vars["img_size_choice"] = tk.StringVar(
            value=self._img_size_choice_from_value(self.values.get("img_size", "224,224"))
        )
        self.vars["img_size_custom"] = tk.StringVar(value=str(self.values.get("img_size", "224,224")))
        grid = ttk.Frame(parent)
        grid.pack(fill="both", expand=True)
        left = ttk.LabelFrame(grid, text="Model and task", padding=12)
        right = ttk.LabelFrame(grid, text="Training and resources", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        ttk.Label(left, text="These change the model family used for this organ. Keep defaults unless comparing strategies.").pack(anchor="w", pady=(0, 8))
        self._labeled_combo(left, "Fine-tuning", "finetune_method", ["lora", "frozen", "adapter", "full"], width=24)
        self._labeled_combo(left, "Decoder", "decoder", ["segformer3d", "mlp_probe", "linear3d", "dpt3d"], width=24)
        self._labeled_combo(left, "Pretrained scale", "model_scale", self._available_model_scales(), width=24)
        self._labeled_spinbox(left, "LoRA rank", "lora_rank", 1, 128, 1, 10)
        self._labeled_spinbox(left, "LoRA alpha", "lora_alpha", 1, 512, 1, 10)
        self._labeled_spinbox(left, "Adapter bottleneck", "adapter_bottleneck", 1, 512, 1, 10)
        ttk.Label(
            left,
            text="Backend template, modality, and custom weight path are controlled by fewshot_config.json.",
            foreground="#6b7280",
            wraplength=420,
        ).pack(anchor="w", pady=(12, 0))

        ttk.Label(right, text="These affect runtime, memory use, validation, and disk retention.").pack(anchor="w", pady=(0, 8))
        self._labeled_spinbox(right, "Epochs", "epochs", 1, 10000, 1, 10)
        self._labeled_spinbox(right, "Batch size", "batch_size", 1, 128, 1, 10)
        self._labeled_spinbox(right, "Grad accumulation", "grad_accumulation", 1, 1024, 1, 10)
        self._labeled_combo(right, "Learning rate", "lr", self.LR_CHOICES, width=14)
        self._labeled_combo(right, "Weight decay", "weight_decay", self.WEIGHT_DECAY_CHOICES, width=14)
        self.img_size_choice_widget = self._labeled_combo(
            right,
            "Image detail",
            "img_size_choice",
            [label for label, _value in self.IMG_SIZE_CHOICES] + [self.IMG_SIZE_CUSTOM_LABEL],
            width=22,
        )
        self.img_size_choice_widget.bind("<<ComboboxSelected>>", self._sync_image_size_choice)
        self.img_size_custom_widget = self._labeled_entry(right, "Custom size", "img_size_custom", 14)
        try:
            self.img_size_custom_widget.bind("<KeyRelease>", self._sync_custom_image_size)
            self.img_size_custom_widget.bind("<FocusOut>", self._sync_custom_image_size)
        except Exception:
            pass
        self._labeled_combo(right, "Sub-volume depth", "sub_volume_depth", self.SUB_VOLUME_DEPTH_CHOICES, width=14)
        self._labeled_spinbox(right, "Keep checkpoints", "keep_last_checkpoints", 0, 1000, 1, 10)
        self._labeled_spinbox(right, "Validation fraction", "val_fraction", 0.0, 0.9, 0.05, 10)
        check_row = ttk.Frame(right)
        check_row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(check_row, text="Mixed precision", variable=self.vars["mixed_precision"]).pack(side="left")
        ttk.Checkbutton(check_row, text="Sub-volume training", variable=self.vars["sub_volume"]).pack(side="left", padx=(18, 0))
        self._install_quick_refresh_traces()

    def _labeled_entry(self, parent, label, key, width):
        from tkinter import ttk
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        entry = ttk.Entry(row, textvariable=self.vars[key], width=width)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _labeled_spinbox(self, parent, label, key, from_, to, increment, width):
        from tkinter import ttk
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        ttk.Spinbox(row, textvariable=self.vars[key], from_=from_, to=to, increment=increment, width=width).pack(side="left")

    def _labeled_combo(self, parent, label, key, values, width=20):
        from tkinter import ttk
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        if values and self.vars[key].get() not in values:
            self.vars[key].set(values[0])
        widget = ttk.Combobox(row, textvariable=self.vars[key], values=values, state="readonly", width=width)
        widget.pack(side="left", fill="x", expand=True)
        return widget

    def _labeled_combo_inline(self, parent, label, key, values):
        """Label-above-combo layout for horizontal multi-column sections."""
        from tkinter import ttk
        ttk.Label(parent, text=label, anchor="center").pack(anchor="center")
        if values and self.vars[key].get() not in values:
            self.vars[key].set(values[0])
        widget = ttk.Combobox(parent, textvariable=self.vars[key], values=values,
                               state="readonly", width=24)
        widget.pack(anchor="center", pady=(4, 0))
        return widget

    def _available_model_scales(self):
        choices = []
        root = self.context.get("dinov3_root") or ""
        model_dirs = [
            ("vitb16", "dinov3-vitb16"),
            ("vitl16", "dinov3-vitl16"),
            ("vith16plus", "dinov3-vith16plus"),
        ]
        for label, dirname in model_dirs:
            if root and os.path.isdir(os.path.join(root, "models", dirname)):
                choices.append(label)
        current = str(self.values.get("model_scale", "vitb16") or "vitb16")
        if current not in choices:
            choices.insert(0, current)
        return choices or ["vitb16"]

    def _base_config_choices(self):
        root = self.context.get("dinov3_root") or ""
        config_dir = os.path.join(root, "config")
        choices = []
        if os.path.isdir(config_dir):
            for name in sorted(os.listdir(config_dir)):
                if name.endswith((".yaml", ".yml")):
                    choices.append("config/" + name)
        current = str(self.values.get("base_config", "config/synthstrip_lora_segformer3d.yaml"))
        if current not in choices:
            choices.insert(0, current)
        return choices or [current]

    def _sub_volume_depth_from_values(self, values):
        parts = str(values.get("sub_volume_size", "32,256,256")).split(",")
        depth = parts[0].strip() if parts else "32"
        return depth if depth in self.SUB_VOLUME_DEPTH_CHOICES else "32"

    def _sync_sub_volume_size(self):
        depth = self.vars.get("sub_volume_depth").get() if self.vars.get("sub_volume_depth") else "32"
        img_size = self.vars.get("img_size").get() if self.vars.get("img_size") else "224,224"
        parts = [part.strip() for part in str(img_size).split(",")]
        if len(parts) == 2:
            self.vars["sub_volume_size"].set("{0},{1},{2}".format(depth, parts[0], parts[1]))

    def _img_size_choice_from_value(self, value):
        for label, item in self.IMG_SIZE_CHOICES:
            if str(item) == str(value):
                return label
        return self.IMG_SIZE_CUSTOM_LABEL

    def _set_custom_size_entry_state(self):
        if getattr(self, "img_size_custom_widget", None) is None or "img_size_choice" not in self.vars:
            return
        state = "normal" if self.vars["img_size_choice"].get() == self.IMG_SIZE_CUSTOM_LABEL else "disabled"
        try:
            self.img_size_custom_widget.configure(state=state)
        except Exception:
            pass

    def _sync_image_size_choice(self, _event=None):
        choice = self.vars.get("img_size_choice")
        target = self.vars.get("img_size")
        if choice is None or target is None:
            return
        if choice.get() == self.IMG_SIZE_CUSTOM_LABEL:
            custom = self.vars.get("img_size_custom")
            value = custom.get().strip() if custom is not None else target.get()
        else:
            value = _choice_value(choice.get(), self.IMG_SIZE_CHOICES)
        if value is not None:
            target.set(str(value))
            custom = self.vars.get("img_size_custom")
            if custom is not None and choice.get() != self.IMG_SIZE_CUSTOM_LABEL:
                custom.set(str(value))
        self._set_custom_size_entry_state()

    def _sync_custom_image_size(self, _event=None):
        if "img_size_choice" not in self.vars or "img_size_custom" not in self.vars:
            return
        if self.vars["img_size_choice"].get() != self.IMG_SIZE_CUSTOM_LABEL:
            return
        self.vars["img_size"].set(self.vars["img_size_custom"].get().strip())
        self._sync_sub_volume_size()

    def _memory_mode_from_values(self, values):
        sub_volume = _bool(values.get("sub_volume", False))
        try:
            grad_accumulation = int(values.get("grad_accumulation", 1) or 1)
            batch_size = int(values.get("batch_size", 1) or 1)
        except Exception:
            return "Custom"
        img_size = str(values.get("img_size", "") or "")
        depth = self._sub_volume_depth_from_values(values)
        if sub_volume and batch_size == 1 and grad_accumulation == 2 and img_size == "192,192" and depth == "24":
            return "Low GPU memory"
        if (not sub_volume) and batch_size == 1 and grad_accumulation == 2 and img_size == "256,256":
            return "Higher quality"
        if (not sub_volume) and batch_size == 1 and grad_accumulation == 1 and img_size == "224,224":
            return "Balanced"
        return "Custom"

    def _trace_var(self, key):
        var = self.vars.get(key)
        if var is None:
            return
        try:
            var.trace_add("write", lambda *_args: self._refresh_quick_labels())
        except Exception:
            try:
                var.trace("w", lambda *_args: self._refresh_quick_labels())
            except Exception:
                pass

    def _install_quick_refresh_traces(self):
        for key in (
            "epochs",
            "val_fraction",
            "batch_size",
            "grad_accumulation",
            "img_size",
            "sub_volume",
            "sub_volume_depth",
        ):
            self._trace_var(key)
        self._refresh_quick_labels()

    def _set_combo_value(self, widget, key, value):
        if widget is not None:
            try:
                values = list(widget.cget("values"))
                if value not in values:
                    values.append(value)
                    widget.configure(values=values)
            except Exception:
                pass
        var = self.vars.get(key)
        if var is not None:
            var.set(value)

    def _refresh_quick_labels(self):
        if self._syncing_quick:
            return
        if not self.vars:
            return
        self._sync_sub_volume_size()
        if "epochs" in self.vars and "epochs_choice" in self.vars:
            self._set_combo_value(
                self.epochs_choice_widget,
                "epochs_choice",
                _option_label(self.vars["epochs"].get(), self.EPOCH_CHOICES),
            )
        if "val_fraction" in self.vars and "val_fraction_choice" in self.vars:
            self._set_combo_value(
                self.val_choice_widget,
                "val_fraction_choice",
                _option_label(self.vars["val_fraction"].get(), self.VAL_CHOICES),
            )
        if "memory_mode" in self.vars:
            values = {}
            for key, var in self.vars.items():
                try:
                    values[key] = var.get()
                except Exception:
                    pass
            self._set_combo_value(self.memory_mode_widget, "memory_mode", self._memory_mode_from_values(values))
        if "img_size" in self.vars and "img_size_choice" in self.vars:
            label = self._img_size_choice_from_value(self.vars["img_size"].get())
            self._set_combo_value(
                self.img_size_choice_widget,
                "img_size_choice",
                label,
            )
            if "img_size_custom" in self.vars:
                self.vars["img_size_custom"].set(self.vars["img_size"].get())
            self._set_custom_size_entry_state()

    def _sync_quick_settings(self, _event=None):
        self._syncing_quick = True
        try:
            epoch_label = self.vars.get("epochs_choice").get()
            epoch_value = _choice_value(epoch_label, self.EPOCH_CHOICES)
            if epoch_value is not None:
                self.vars["epochs"].set(str(epoch_value))
            val_label = self.vars.get("val_fraction_choice").get()
            val_value = _choice_value(val_label, self.VAL_CHOICES)
            if val_value is not None:
                self.vars["val_fraction"].set(str(val_value))
            memory_label = self.vars.get("memory_mode").get()
            mode = _choice_value(memory_label, self.MEMORY_CHOICES)
            if mode == "low_memory":
                self.vars["batch_size"].set("1")
                self.vars["grad_accumulation"].set("2")
                self.vars["img_size"].set("192,192")
                self.vars["sub_volume"].set(True)
                if "sub_volume_depth" in self.vars:
                    self.vars["sub_volume_depth"].set("24")
                self.vars["mixed_precision"].set(False)
            elif mode == "quality":
                self.vars["batch_size"].set("1")
                self.vars["grad_accumulation"].set("2")
                self.vars["img_size"].set("256,256")
                self.vars["sub_volume"].set(False)
                if "sub_volume_depth" in self.vars:
                    self.vars["sub_volume_depth"].set("32")
            elif mode == "balanced":
                self.vars["batch_size"].set("1")
                self.vars["grad_accumulation"].set("1")
                self.vars["img_size"].set("224,224")
                self.vars["sub_volume"].set(False)
                if "sub_volume_depth" in self.vars:
                    self.vars["sub_volume_depth"].set("32")
            self._sync_sub_volume_size()
        finally:
            self._syncing_quick = False
        self._refresh_quick_labels()

    def apply_profile(self):
        profile_name = self.profile_var.get()
        values = default_training_options(self.config, profile_name)
        for key, value in values.items():
            var = self.vars.get(key)
            if var is None:
                continue
            try:
                if hasattr(var, "set"):
                    var.set(_bool(value) if isinstance(var.get(), bool) else str(value))
            except Exception:
                pass
        if "epochs_choice" in self.vars:
            self.vars["epochs_choice"].set(_option_label(values.get("epochs", 10), self.EPOCH_CHOICES))
        if "val_fraction_choice" in self.vars:
            self.vars["val_fraction_choice"].set(_option_label(values.get("val_fraction", 0.2), self.VAL_CHOICES))
        if "memory_mode" in self.vars:
            self.vars["memory_mode"].set(self._memory_mode_from_values(values))
        if "sub_volume_depth" in self.vars:
            self.vars["sub_volume_depth"].set(self._sub_volume_depth_from_values(values))
        if "img_size_choice" in self.vars:
            self.vars["img_size_choice"].set(self._img_size_choice_from_value(values.get("img_size", "224,224")))
        if "img_size_custom" in self.vars:
            self.vars["img_size_custom"].set(str(values.get("img_size", "224,224")))
        self._refresh_quick_labels()
        self.status_var.set("Applied profile: {0}".format(profile_name or "default"))
        self._append_log("Applied profile: {0}".format(profile_name or "default"))

    def collect_options(self):
        self._sync_image_size_choice()
        self._sync_sub_volume_size()
        options = {}
        for key, var in self.vars.items():
            if key in ("epochs_choice", "val_fraction_choice", "memory_mode", "sub_volume_depth", "img_size_choice", "img_size_custom"):
                continue
            value = var.get()
            options[key] = value
        selected = []
        if self.case_list is not None:
            for idx in self.case_list.curselection():
                selected.append(str(self.case_list.get(idx)))
        manual = split_csv(self.manual_cases_var.get() if self.manual_cases_var is not None else "")
        cases = selected + [case for case in manual if case not in selected]
        if cases:
            options["cases"] = cases
            options["sample_mode"] = "all"
        return validate_options(options)

    def start_training(self):
        if self.started:
            return
        try:
            options = self.collect_options()
        except Exception as exc:
            self.status_var.set("Cannot start training: {0}".format(exc))
            self._append_log("Cannot start training: {0}".format(exc))
            return
        try:
            run_id, status_path, pid = launch_training(self.context, options)
        except Exception as exc:
            self._write_setup_failure(exc)
            self.status_var.set("Could not start training: {0}".format(exc))
            self._append_log("Could not start training: {0}".format(exc))
            return
        self.started = True
        self.training_status_path = status_path
        self.training_log_dir = os.path.dirname(os.path.dirname(status_path))
        self.status_var.set("Training started: {0} (PID {1})".format(run_id, pid))
        self._append_log("Training started in the background: {0} (PID {1})".format(run_id, pid))
        self._append_log("Status file: {0}".format(status_path))
        try:
            self.start_button.state(["disabled"])
            self.open_log_button.state(["!disabled"])
            if self.close_button is not None:
                self.close_button.configure(text="Close")
        except Exception:
            pass
        self.root.after(500, self.poll_training_status)

    def poll_training_status(self):
        if not self.training_status_path:
            return
        status = read_json(self.training_status_path, {}) or {}
        line = format_status_line(status)
        if line and line != self.last_status_line:
            self.last_status_line = line
            self.status_var.set(line)
            self._append_log(line)
        if status.get("train_log"):
            self.training_log_dir = os.path.dirname(status.get("train_log"))
        if status.get("status") in ("completed", "failed", "cancelled", "cancelling"):
            return
        self.root.after(1500, self.poll_training_status)

    def _append_log(self, text):
        if self.status_text is None:
            return
        stamp = time.strftime("%H:%M:%S")
        self.status_text.configure(state="normal")
        self.status_text.insert("end", "[{0}] {1}\n".format(stamp, text))
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def open_log_folder(self):
        folder = self.training_log_dir or os.path.join(self.context.get("workspace", ""), "jobs")
        if not folder or not os.path.isdir(folder):
            self._append_log("Log folder is not available yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            self._append_log("Could not open log folder: {0}".format(exc))

    def _write_setup_failure(self, exc):
        path = self.context.get("setup_status_path")
        if not path:
            return
        write_json_atomic(path, {
            "schema_version": "mimics_fewshot_setup.v1",
            "job_id": self.context.get("setup_id"),
            "kind": "train_setup",
            "status": "failed",
            "organ": self.context.get("organ"),
            "ts_root": os.path.abspath(self.context.get("ts_root", "")),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "updated_at_epoch": time.time(),
        })

    def close(self):
        if not self.started:
            path = self.context.get("setup_status_path")
            if path:
                payload = read_json(path, {}) or {}
                payload.update({
                    "status": "closed",
                    "updated_at_epoch": time.time(),
                })
                write_json_atomic(path, payload)
        self.root.destroy()


def run_ui(context):
    try:
        import tkinter as tk
        from tkinter import messagebox  # noqa: F401
    except Exception as exc:
        message = (
            "The external DINOv3 training setup window could not open because Tkinter is not available "
            "in the configured external Python environment. Training was not started. Install Tkinter "
            "for that Python environment or use the default Train/Update Model entry."
        )
        setup_path = context.get("setup_status_path")
        if setup_path:
            write_json_atomic(setup_path, {
                "schema_version": "mimics_fewshot_setup.v1",
                "job_id": context.get("setup_id"),
                "kind": "train_setup",
                "status": "failed",
                "organ": context.get("organ"),
                "ts_root": os.path.abspath(context.get("ts_root", "")),
                "error": message,
                "details": str(exc),
                "updated_at_epoch": time.time(),
            })
        raise RuntimeError(message)
    root = tk.Tk()
    TrainingSetupApp(root, context)
    root.mainloop()


def generate_preview(path, tab="setup"):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required only for --preview PNG generation. Runtime DINOv3 setup UI does not "
            "depend on Pillow. Install pillow in the preview-generation Python environment or skip --preview. "
            "Import error: {0}".format(exc)
        )

    width, height = 1280, 960
    image = Image.new("RGB", (width, height), "#f4f5f7")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 30)
        head_font = ImageFont.truetype("Arial.ttf", 19)
        font = ImageFont.truetype("Arial.ttf", 17)
        small = ImageFont.truetype("Arial.ttf", 15)
    except Exception:
        title_font = head_font = font = small = ImageFont.load_default()

    draw.rectangle((0, 0, width, 72), fill="#1f2937")
    draw.text((32, 20), "DINOv3 Few-Shot Training", fill="white", font=title_font)
    draw.text((32, 92), "Organ: liver    Dataset: D:\\Dataset\\TotalSegmentator", fill="#1f2937", font=head_font)
    draw.text((32, 126), "Save edited .mcs projects before starting. Training exports labels from saved projects in the background.", fill="#9a5b00", font=font)

    draw.rectangle((30, 166, width - 30, height - 180), outline="#d1d5db", width=2, fill="#ffffff")
    draw.rectangle((30, 166, 250, 212), fill="#e8eef7" if tab == "setup" else "#ffffff", outline="#d1d5db")
    draw.text((60, 180), "Setup", fill="#111827", font=head_font)
    draw.rectangle((250, 166, 470, 212), fill="#e8eef7" if tab == "expert" else "#ffffff", outline="#d1d5db")
    draw.text((280, 180), "Expert", fill="#111827", font=head_font)

    def draw_field(x, y, label, value, kind="select", w=230):
        draw.text((x, y), label, fill="#374151", font=small)
        draw.rectangle((x, y + 22, x + w, y + 54), outline="#9ca3af", fill="#ffffff")
        draw.text((x + 10, y + 29), value, fill="#111827", font=small)
        if kind == "select":
            draw.polygon([(x + w - 20, y + 34), (x + w - 10, y + 34), (x + w - 15, y + 42)], fill="#4b5563")
        elif kind == "spin":
            draw.line((x + w - 30, y + 22, x + w - 30, y + 54), fill="#d1d5db")
            draw.polygon([(x + w - 20, y + 31), (x + w - 10, y + 31), (x + w - 15, y + 25)], fill="#4b5563")
            draw.polygon([(x + w - 20, y + 44), (x + w - 10, y + 44), (x + w - 15, y + 50)], fill="#4b5563")

    def draw_row(x, y, label, value, kind="select", w=300, label_w=168):
        draw.text((x, y + 8), label, fill="#374151", font=small)
        field_x = x + label_w
        draw.rectangle((field_x, y, field_x + w, y + 30), outline="#9ca3af", fill="#ffffff")
        draw.text((field_x + 10, y + 7), value, fill="#111827", font=small)
        if kind == "select":
            draw.polygon([(field_x + w - 20, y + 11), (field_x + w - 10, y + 11), (field_x + w - 15, y + 19)], fill="#4b5563")
        elif kind == "spin":
            draw.line((field_x + w - 30, y, field_x + w - 30, y + 30), fill="#d1d5db")
            draw.polygon([(field_x + w - 20, y + 9), (field_x + w - 10, y + 9), (field_x + w - 15, y + 3)], fill="#4b5563")
            draw.polygon([(field_x + w - 20, y + 21), (field_x + w - 10, y + 21), (field_x + w - 15, y + 27)], fill="#4b5563")

    if tab == "expert":
        draw.text((60, 240), "Expert parameters", fill="#111827", font=head_font)
        draw.text((235, 244), "Profile defaults are recommended for routine annotation.", fill="#6b7280", font=small)
        left_x, right_x = 60, 660
        card_y, card_bottom = 280, 780
        card_w = 560
        draw.rectangle((left_x, card_y, left_x + card_w, card_bottom), outline="#d1d5db", fill="#f9fafb")
        draw.text((left_x + 22, card_y + 22), "Model and task", fill="#111827", font=head_font)
        draw.text((left_x + 22, card_y + 54), "Model family used for this organ.", fill="#6b7280", font=small)
        model_rows = [
            ("Fine-tuning", "lora", "select"),
            ("Decoder", "segformer3d", "select"),
            ("Pretrained scale", "vitb16", "select"),
            ("LoRA rank", "8", "spin"),
            ("LoRA alpha", "16", "spin"),
            ("Adapter bottleneck", "64", "spin"),
        ]
        y = card_y + 88
        for label, value, kind in model_rows:
            draw_row(left_x + 22, y, label, value, kind, w=330, label_w=170)
            y += 36
        draw.text((left_x + 22, y + 12), "Backend template, modality, and custom weights are", fill="#6b7280", font=small)
        draw.text((left_x + 22, y + 34), "controlled centrally in fewshot_config.json.", fill="#6b7280", font=small)

        draw.rectangle((right_x, card_y, right_x + card_w, card_bottom), outline="#d1d5db", fill="#f9fafb")
        draw.text((right_x + 22, card_y + 22), "Training and resources", fill="#111827", font=head_font)
        draw.text((right_x + 22, card_y + 54), "Runtime, memory, validation, and retention.", fill="#6b7280", font=small)
        opt_rows = [
            ("Epochs", "10", "spin"),
            ("Batch size", "1", "spin"),
            ("Grad accumulation", "1", "spin"),
            ("Learning rate", "0.001", "select"),
            ("Weight decay", "0.01", "select"),
            ("Image detail", "Balanced (224 x 224)", "select"),
            ("Custom size", "224,224", "text"),
            ("Sub-volume depth", "32", "select"),
            ("Keep checkpoints", "2", "spin"),
            ("Validation fraction", "0.2", "spin"),
        ]
        y = card_y + 88
        for label, value, kind in opt_rows:
            draw_row(right_x + 22, y, label, value, kind, w=250, label_w=172)
            y += 36
        y += 10
        check_x = right_x + 22
        for label, checked in (("Mixed precision", False), ("Sub-volume training", False)):
            draw.rectangle((check_x, y, check_x + 16, y + 16), outline="#6b7280", fill="#ffffff")
            if checked:
                draw.line((check_x + 3, y + 8, check_x + 8, y + 13), fill="#2563eb", width=2)
                draw.line((check_x + 8, y + 13, check_x + 16, y + 3), fill="#2563eb", width=2)
            draw.text((check_x + 28, y - 2), label, fill="#374151", font=small)
            check_x += 190
    else:
        draw.rectangle((60, 238, width - 60, 300), outline="#d1d5db", fill="#f9fafb")
        draw.text((80, 256), "Profile", fill="#374151", font=font)
        draw.rectangle((155, 248, 360, 282), outline="#9ca3af", fill="#ffffff")
        draw.text((168, 256), "balanced", fill="#111827", font=font)
        draw.text((390, 256), "Use profiles for routine work; Expert is optional.", fill="#374151", font=small)

        draw.rectangle((60, 318, width - 60, 392), outline="#d1d5db", fill="#f9fafb")
        quick = [
            ("Training length", "Standard (10)"),
            ("Validation", "Standard validation (20%)"),
            ("Resource preset", "Balanced"),
        ]
        x = 80
        for label, value in quick:
            draw_field(x, 336, label, value, "select", w=245)
            x += 300

        draw.text((60, 420), "Samples", fill="#111827", font=head_font)
        draw.text((170, 424), "Sample order", fill="#374151", font=small)
        draw.rectangle((270, 416, 380, 446), outline="#9ca3af", fill="#ffffff")
        draw.text((282, 422), "all", fill="#111827", font=small)
        draw.text((410, 424), "Max samples", fill="#374151", font=small)
        draw.rectangle((510, 416, 610, 446), outline="#9ca3af", fill="#ffffff")
        draw.text((522, 422), "0", fill="#111827", font=small)

        draw.rectangle((60, 462, 835, 625), outline="#9ca3af", fill="#fbfdff")
        cases = ["s0001", "s0002", "s0003", "s0004", "s0005", "s0006"]
        y = 478
        for idx, case in enumerate(cases):
            if idx in (0, 1, 3, 6, 7):
                draw.rectangle((68, y - 4, 820, y + 22), fill="#dbeafe")
            draw.text((82, y), case, fill="#111827", font=font)
            y += 24
        draw.rectangle((870, 462, 990, 498), outline="#9ca3af", fill="#ffffff")
        draw.text((892, 471), "Select All", fill="#111827", font=font)
        draw.rectangle((870, 510, 990, 546), outline="#9ca3af", fill="#ffffff")
        draw.text((910, 519), "Clear", fill="#111827", font=font)
        draw.text((870, 575), "616 case(s) found", fill="#374151", font=small)
        draw.text((870, 607), "Selected cases override sample order.", fill="#374151", font=small)
        draw.text((60, 642), "Manual cases", fill="#374151", font=font)
        draw.rectangle((180, 634, 920, 670), outline="#9ca3af", fill="#ffffff")
        draw.text((194, 642), "s0012,s0016", fill="#111827", font=font)

    status_top = height - 155
    draw.rectangle((30, status_top, width - 30, height - 75), outline="#d1d5db", fill="#ffffff")
    draw.text((50, status_top + 8), "Status", fill="#111827", font=head_font)
    log_lines = [
        "[14:20:03] Ready. Choose a profile and samples, then start background training.",
        "[14:20:29] Training progress will appear here after Start Training.",
    ]
    y = status_top + 35
    for line in log_lines:
        draw.text((50, y), line, fill="#374151", font=small)
        y += 22

    draw.text((32, height - 40), "Training progress stays visible here; Mimics remains usable.", fill="#374151", font=font)
    draw.rectangle((width - 450, height - 55, width - 300, height - 14), fill="#ffffff", outline="#9ca3af")
    draw.text((width - 424, height - 44), "Open Log Folder", fill="#111827", font=font)
    draw.rectangle((width - 285, height - 55, width - 175, height - 14), fill="#ffffff", outline="#9ca3af")
    draw.text((width - 248, height - 44), "Cancel", fill="#111827", font=font)
    draw.rectangle((width - 160, height - 55, width - 32, height - 14), fill="#2563eb", outline="#1d4ed8")
    draw.text((width - 145, height - 44), "Start Training", fill="#ffffff", font=font)

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", help="Path to the setup context JSON written by Mimics.")
    parser.add_argument("--preview", help="Write a static PNG preview of the UI and exit.")
    parser.add_argument("--preview-tab", choices=("setup", "expert"), default="setup")
    args = parser.parse_args(argv)
    if args.preview:
        generate_preview(args.preview, tab=args.preview_tab)
        print(args.preview)
        return 0
    if not args.context:
        parser.error("--context is required unless --preview is used")
    context = read_json(args.context, None)
    if not context:
        raise RuntimeError("Could not read setup context: {0}".format(args.context))
    run_ui(context)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
