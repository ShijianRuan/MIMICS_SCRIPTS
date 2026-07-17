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
import threading
import time
import traceback
import uuid
from pathlib import Path
try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue

# Ensure the project root and tools/ directory are importable.
# The embeddable Python (nninteractive_env) uses a ._pth file that
# ignores PYTHONPATH, so we must inject paths directly into sys.path.
_here = os.path.dirname(os.path.abspath(__file__))
_tools_dir = _here
_project_root = os.path.dirname(_tools_dir)
for _candidate in (_tools_dir, _project_root):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    from fewshot_strategies import DEFAULT_OPTIONS, STRATEGIES, normalize_strategy_options, strategy_defaults, strategy_ids, strategy_label, strategy_summary, suggested_strategy
except ImportError:
    from tools.fewshot_strategies import DEFAULT_OPTIONS, STRATEGIES, normalize_strategy_options, strategy_defaults, strategy_ids, strategy_label, strategy_summary, suggested_strategy
try:
    from ui_theme import configure_application, stylesheet as shared_stylesheet
except ImportError:
    from tools.ui_theme import configure_application, stylesheet as shared_stylesheet
TITLE = "DINOv3 Few-Shot Training"
STRATEGY_DATA_KEYS = set(DEFAULT_OPTIONS.keys())
DECODER_CHOICES = (
    "segformer3d", "token_pyramid3d", "dpt3d", "linear3d", "mlp_probe",
    "conv2d",
)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
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


def configured_mask_names(config, organ):
    names = [str(organ or "").strip()]
    aliases = config.get("organ_mask_aliases") or {}
    if isinstance(aliases, dict):
        organ_key = safe_slug(organ)
        for key, values in aliases.items():
            if safe_slug(key) != organ_key:
                continue
            names.extend(split_csv(values))
    result = []
    seen = set()
    for name in names:
        key = safe_slug(name)
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


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
    values.setdefault("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))
    values.setdefault("strategy", config.get("default_strategy", "adaptive"))
    values.setdefault("epochs", config.get("default_epochs", 20))
    values["batch_size"] = 1
    values.setdefault("grad_accumulation", config.get("default_grad_accumulation", 1))
    values.setdefault("lr", config.get("default_lr", 0.001))
    values.setdefault("weight_decay", config.get("default_weight_decay", 0.01))
    values.setdefault("lr_scheduler", config.get("default_lr_scheduler", "cosine"))
    values.setdefault("warmup_epochs", config.get("default_warmup_epochs", 3))
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
    values.setdefault("export_labels_before_training", config.get("default_export_labels_before_training", True))
    values.setdefault("gpu_lock_timeout_seconds", config.get("gpu_lock_timeout_seconds", 86400))
    values.setdefault(
        "background_mimics_lock_timeout_seconds",
        config.get("background_mimics_lock_timeout_seconds", 1800),
    )
    return values


def training_options_for_organ(config, organ, profile_name=None):
    """Resolve one coherent initial strategy instead of mixing it with a legacy profile."""
    values = default_training_options(config, profile_name)
    if not config.get("default_strategy"):
        values["strategy"] = suggested_strategy(organ)
    values.update(strategy_defaults(values.get("strategy")))
    values["mask_names"] = ",".join(configured_mask_names(config, organ))
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
    if str(normalized.get("strategy", "adaptive")) not in strategy_ids():
        raise ValueError("Unknown DINOv3 training strategy.")
    normalized["epochs"] = _int(normalized.get("epochs", 20), "Epochs", 1)
    normalized["batch_size"] = _int(normalized.get("batch_size", 1), "Batch size", 1)
    if normalized["batch_size"] != 1:
        raise ValueError(
            "Batch size must stay 1 for variable-depth 3D Mimics cases. "
            "Use Grad accumulation to increase the effective batch size."
        )
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
    normalized["lr_scheduler"] = str(normalized.get("lr_scheduler", "cosine")).strip().lower()
    if normalized["lr_scheduler"] not in ("constant", "constant_warmup", "cosine"):
        raise ValueError("Learning-rate schedule must be constant, constant with warmup, or cosine.")
    normalized["warmup_epochs"] = _int(normalized.get("warmup_epochs", 3), "Warmup epochs", 0)
    if str(normalized.get("decoder", "segformer3d")) not in DECODER_CHOICES:
        raise ValueError("Unsupported decoder: {0}".format(normalized.get("decoder")))
    normalized["val_fraction"] = _float(normalized.get("val_fraction", 0.2), "Validation fraction")
    normalized["mixed_precision"] = _bool(normalized.get("mixed_precision", False))
    normalized["sub_volume"] = _bool(normalized.get("sub_volume", False))
    normalized["keep_materialized_dataset"] = _bool(normalized.get("keep_materialized_dataset", False))
    normalized["export_labels_before_training"] = _bool(normalized.get("export_labels_before_training", True))
    normalized["mask_names"] = ",".join(split_csv(normalized.get("mask_names", "")))
    normalized["mcs_output_dir"] = os.path.abspath(os.path.expanduser(
        str(normalized.get("mcs_output_dir", "") or "")
    )) if str(normalized.get("mcs_output_dir", "") or "").strip() else ""
    strategy_values = {key: normalized.get(key, value) for key, value in DEFAULT_OPTIONS.items()}
    normalized.update(normalize_strategy_options(strategy_values, preset=str(normalized.get("strategy", "adaptive"))))
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
    strategy_options = {key: options.get(key, value) for key, value in DEFAULT_OPTIONS.items()}
    cmd.extend([
        "--strategy",
        str(options.get("strategy", config.get("default_strategy", "adaptive"))),
        "--strategy-options-json",
        json.dumps(strategy_options, sort_keys=True),
        "--base-config",
        str(options.get("base_config", config.get("base_config", "config/research/ct_fewshot_fast.yaml"))),
        "--epochs",
        str(int(options.get("epochs", config.get("default_epochs", 20)))),
        "--batch-size",
        str(int(options.get("batch_size", config.get("default_batch_size", 1)))),
        "--grad-accumulation",
        str(int(options.get("grad_accumulation", config.get("default_grad_accumulation", 1)))),
        "--lr",
        str(float(options.get("lr", config.get("default_lr", 0.001)))),
        "--weight-decay",
        str(float(options.get("weight_decay", config.get("default_weight_decay", 0.01)))),
        "--lr-scheduler",
        str(options.get("lr_scheduler", config.get("default_lr_scheduler", "cosine"))),
        "--warmup-epochs",
        str(int(options.get("warmup_epochs", config.get("default_warmup_epochs", 3)))),
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
            config.get("background_mimics_lock_timeout_seconds", 1800),
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
    mask_names = split_csv(options.get("mask_names", ""))
    if mask_names:
        cmd.extend(["--mask-names", ",".join(mask_names)])


def prepare_training_launch(context, options, run_id=None):
    config = context.get("config") or {}
    organ = context["organ"]
    options = dict(options)
    if bool(options.get("export_labels_before_training", True)) and not split_csv(options.get("mask_names", "")):
        options["mask_names"] = ",".join(configured_mask_names(config, organ))
    options = validate_options(options)
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
        "--run-id",
        run_id,
    ]
    if bool(options.get("export_labels_before_training", True)):
        if "mimics_exe" in context and not str(context.get("mimics_exe") or "").strip():
            raise RuntimeError(
                "Refreshing labels requires a separate background Mimics executable. "
                "Configure MIMICS_BACKGROUND_EXE, or turn off label refresh and use already exported NIfTI labels."
            )
        cmd.append("--export-labels")
        mcs_output_dir = str(options.get("mcs_output_dir") or context.get("mcs_output_dir") or "").strip()
        if mcs_output_dir:
            cmd.extend(["--mcs-output-dir", os.path.abspath(mcs_output_dir)])
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
        "retry_context": {
            key: context.get(key)
            for key in (
                "project_root",
                "pipeline_script",
                "dinov3_root",
                "python_exe",
                "mimics_exe",
                "mcs_output_dir",
                "ts_root",
                "workspace",
                "organ",
                "case_ids",
                "config",
            )
        },
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
        write_json_best_effort(launch["status_path"], payload)
        raise
    payload = read_json(launch["status_path"], launch["job_payload"]) or launch["job_payload"]
    payload["launcher_pid"] = process.pid
    payload["updated_at_epoch"] = time.time()
    write_json_best_effort(launch["status_path"], payload)
    setup_status_path = context.get("setup_status_path")
    if setup_status_path:
        write_json_best_effort(setup_status_path, {
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
    export_progress = job.get("label_export_progress") or {}
    if isinstance(export_progress, dict) and export_progress:
        index = int(export_progress.get("index", 0) or 0)
        total = int(export_progress.get("total", 0) or 0)
        phase = str(export_progress.get("phase") or export_progress.get("status") or "running")
        parts.append("labels {0}/{1} {2}".format(index, total, phase) if total else "labels " + phase)
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
    width = min(1220, max(980, screen_width - 80))
    # Keep a real margin for the OS title bar/taskbar on shorter screens.  The
    # setup body is scrollable/compressible; a too-tall window hides the action
    # footer and feels broken.
    height = min(900, max(560, screen_height - 120))
    min_width = min(960, width)
    min_height = min(560, height)
    return width, height, min_width, min_height


def resolve_mimics_output_dir_for_ui(ts_root, project_root):
    default_dir = os.path.abspath(os.path.join(ts_root, "mcs_output"))
    merged = {}
    for name in ("mimics_io_config.json", "nninteractive_config.json"):
        path = os.path.join(project_root or "", name)
        loaded = read_json(path, {}) or {}
        if isinstance(loaded, dict):
            merged.update(loaded)
    configured = merged.get("mimics_output_dir", "")
    configured = str(configured or "").strip()
    if not configured:
        return default_dir
    configured = os.path.expandvars(os.path.expanduser(configured))
    if os.path.isabs(configured):
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(ts_root, configured))


def case_ids_from_dataset_for_ui(ts_root, project_root=""):
    rows = set()
    output_dir = resolve_mimics_output_dir_for_ui(ts_root, project_root)
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.lower().endswith(".mcs"):
                rows.add(name[:-4])
    if os.path.isdir(ts_root):
        for name in os.listdir(ts_root):
            path = os.path.join(ts_root, name)
            if os.path.isdir(path) and name not in ("mcs_output", "segmentations", "fewshot_models"):
                rows.add(name)
    return sorted(rows)


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
        self.values = training_options_for_organ(
            self.config, self.context.get("organ"), default_profile,
        )
        if isinstance(self.context.get("initial_options"), dict):
            self.values.update(self.context.get("initial_options") or {})
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
        self._applying_strategy = False
        self._closing = False
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
        self.vars["export_labels_before_training"] = tk.BooleanVar(
            value=_bool(self.values.get("export_labels_before_training", True))
        )
        ttk.Checkbutton(
            samples,
            text="Refresh labels from saved .mcs before training",
            variable=self.vars["export_labels_before_training"],
        ).pack(anchor="w", pady=(8, 0))
        mask_row = ttk.Frame(samples)
        mask_row.pack(fill="x", pady=(8, 0))
        ttk.Label(mask_row, text="Saved mask names").pack(side="left")
        ttk.Entry(mask_row, textvariable=self.vars["mask_names"]).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

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
        self._labeled_combo(left, "Decoder", "decoder", list(DECODER_CHOICES), width=24)
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
        self._labeled_spinbox(right, "Batch size (fixed)", "batch_size", 1, 1, 1, 10)
        self._labeled_spinbox(right, "Grad accumulation", "grad_accumulation", 1, 1024, 1, 10)
        self._labeled_combo(right, "Learning rate", "lr", self.LR_CHOICES, width=14)
        self._labeled_combo(right, "Weight decay", "weight_decay", self.WEIGHT_DECAY_CHOICES, width=14)
        self._labeled_combo(right, "LR schedule", "lr_scheduler", ["cosine", "constant_warmup", "constant"], width=18)
        self._labeled_spinbox(right, "Warmup epochs", "warmup_epochs", 0, 1000, 1, 10)
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
        current = str(self.values.get("base_config", "config/research/ct_fewshot_fast.yaml"))
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
        if self._closing or not self.training_status_path:
            return
        status = read_json(self.training_status_path, {}) or {}
        line = format_status_line(status)
        if line and line != self.last_status_line:
            self.last_status_line = line
            self.status_var.set(line)
            self._append_log(line)
        if status.get("train_log"):
            self.training_log_dir = os.path.dirname(status.get("train_log"))
        if status.get("status") in ("completed", "failed", "cancelled"):
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
        write_json_best_effort(path, {
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
        if self._closing:
            return
        self._closing = True
        if not self.started:
            path = self.context.get("setup_status_path")
            if path:
                payload = read_json(path, {}) or {}
                payload.update({
                    "status": "closed",
                    "updated_at_epoch": time.time(),
                })
                write_json_best_effort(path, payload)
        self.root.destroy()


def _load_pyside6():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class QtTrainingSetupApp(object):
    """PySide6 implementation of the external training setup window."""

    EPOCH_CHOICES = TrainingSetupApp.EPOCH_CHOICES
    VAL_CHOICES = TrainingSetupApp.VAL_CHOICES
    MEMORY_CHOICES = TrainingSetupApp.MEMORY_CHOICES
    LR_CHOICES = TrainingSetupApp.LR_CHOICES
    WEIGHT_DECAY_CHOICES = TrainingSetupApp.WEIGHT_DECAY_CHOICES
    IMG_SIZE_CHOICES = TrainingSetupApp.IMG_SIZE_CHOICES
    IMG_SIZE_CUSTOM_LABEL = TrainingSetupApp.IMG_SIZE_CUSTOM_LABEL
    SUB_VOLUME_DEPTH_CHOICES = TrainingSetupApp.SUB_VOLUME_DEPTH_CHOICES

    def __init__(self, window, context, qt_modules):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        self.config = context.get("config") or {}
        self.profiles = self.config.get("training_profiles") or {}
        self.profile_names = sorted(self.profiles.keys()) if isinstance(self.profiles, dict) else []
        default_profile = self.config.get("default_training_profile") or (self.profile_names[0] if self.profile_names else "")
        self.values = training_options_for_organ(
            self.config, self.context.get("organ"), default_profile,
        )
        if isinstance(self.context.get("initial_options"), dict):
            self.values.update(self.context.get("initial_options") or {})
        self.widgets = {}
        self.quick_widgets = {}
        self.case_list = None
        self.manual_cases = None
        self.dataset_edit = None
        self.mcs_folder_edit = None
        self.mask_names_edit = None
        self.status_label = None
        self.status_text = None
        self.start_button = None
        self.open_log_button = None
        self.close_button = None
        self.training_status_path = None
        self.training_log_dir = ""
        self.last_status_line = ""
        self.started = False
        self._closing = False
        self._syncing_quick = False
        self._dataset_scan_generation = 0
        self._dataset_scan_results = Queue()
        self._dataset_scan_timer = self.QtCore.QTimer(self.window)
        self._dataset_scan_timer.timeout.connect(self._poll_dataset_scan)
        self._dataset_scan_timer.start(80)
        self._build()

    def _build(self):
        QtCore, QtGui, QtWidgets = self.QtCore, self.QtGui, self.QtWidgets
        self.window.setWindowTitle(TITLE)
        width, height, min_width, min_height = window_layout_for_screen(1280, 900)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            size = screen.availableGeometry().size()
            width, height, min_width, min_height = window_layout_for_screen(size.width(), size.height())
        self.window.resize(width, height)
        self.window.setMinimumSize(min_width, min_height)
        self.window.setStyleSheet(self._stylesheet())

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        title = QtWidgets.QLabel("DINOv3 Few-Shot Training")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("Organ: {0}".format(self.context.get("organ", "?")))
        subtitle.setObjectName("subtitleLabel")
        warning = QtWidgets.QLabel(
            "Save edited .mcs projects before starting. Training exports labels from saved projects in the background."
        )
        warning.setObjectName("warningLabel")
        warning.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addWidget(warning)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_setup_tab(), "Data and samples")
        tabs.addTab(self._build_expert_tab(), "Model and policy")
        outer.addWidget(tabs, 1)

        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(78)
        status_layout.addWidget(self.status_text)
        outer.addWidget(status_group)
        self._append_log("Ready. Choose a profile and samples, then start background training.")

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Configure samples and parameters, then start background training.")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.close_button = QtWidgets.QPushButton("Cancel")
        self.close_button.clicked.connect(self.close)
        footer.addWidget(self.close_button)
        self.open_log_button = QtWidgets.QPushButton("Open Log Folder")
        self.open_log_button.setEnabled(False)
        self.open_log_button.clicked.connect(self.open_log_folder)
        footer.addWidget(self.open_log_button)
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_training)
        footer.addWidget(self.start_button)
        outer.addLayout(footer)
        self.window.setCentralWidget(central)

        shortcut_start = QtGuiShortcut(QtGui, "Return", self.window)
        shortcut_start.activated.connect(self.start_training)
        shortcut_close = QtGuiShortcut(QtGui, "Escape", self.window)
        shortcut_close.activated.connect(self.close)
        self.shortcuts = [shortcut_start, shortcut_close]
        self._refresh_quick_labels()
        if not (self.context.get("case_ids") or []):
            self.QtCore.QTimer.singleShot(
                0,
                lambda: self.apply_dataset_root(self.context.get("ts_root", "")),
            )

    def _stylesheet(self):
        return shared_stylesheet()

    def _build_setup_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(body)
        tab_layout.addWidget(scroll)

        profile_group = QtWidgets.QGroupBox("Dataset")
        profile_layout = QtWidgets.QGridLayout(profile_group)
        profile_layout.setColumnStretch(1, 1)
        profile_layout.addWidget(QtWidgets.QLabel("Source image root"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(os.path.abspath(self.context.get("ts_root", "")))
        self.dataset_edit.editingFinished.connect(self.apply_dataset_from_field)
        profile_layout.addWidget(self.dataset_edit, 0, 1)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_dataset)
        profile_layout.addWidget(browse, 0, 2)
        profile_layout.addWidget(QtWidgets.QLabel("Saved .mcs folder"), 1, 0)
        self.mcs_folder_edit = QtWidgets.QLineEdit(str(self.context.get("mcs_output_dir", "")))
        self.mcs_folder_edit.setToolTip("Defaults to mimics_io_config.json. You may override it for this training run.")
        profile_layout.addWidget(self.mcs_folder_edit, 1, 1)
        browse_mcs = QtWidgets.QPushButton("Browse")
        browse_mcs.clicked.connect(self.browse_mcs_folder)
        profile_layout.addWidget(browse_mcs, 1, 2)
        profile_layout.addWidget(QtWidgets.QLabel("Saved mask names"), 2, 0)
        self.mask_names_edit = QtWidgets.QLineEdit(str(self.values.get("mask_names", "")))
        self.mask_names_edit.setPlaceholderText("Example: liver, liver_seg")
        self.mask_names_edit.setToolTip(
            "Comma-separated names accepted in saved .mcs projects. Exactly one must match in each selected case."
        )
        profile_layout.addWidget(self.mask_names_edit, 2, 1, 1, 2)
        layout.addWidget(profile_group)

        sample_group = QtWidgets.QGroupBox("Samples")
        sample_layout = QtWidgets.QVBoxLayout(sample_group)
        sample_controls = QtWidgets.QHBoxLayout()
        sample_controls.addWidget(QtWidgets.QLabel("Sample order"))
        self.widgets["sample_mode"] = self._combo(["all", "latest"], self.values.get("sample_mode", "all"))
        sample_controls.addWidget(self.widgets["sample_mode"])
        sample_controls.addSpacing(16)
        sample_controls.addWidget(QtWidgets.QLabel("Max samples"))
        self.widgets["max_samples"] = self._spin(0, 100000, self.values.get("max_samples", 0))
        sample_controls.addWidget(self.widgets["max_samples"])
        sample_controls.addStretch(1)
        sample_layout.addLayout(sample_controls)

        middle = QtWidgets.QHBoxLayout()
        self.case_list = QtWidgets.QListWidget()
        self.case_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.case_list.setMinimumHeight(160)
        for case_id in self.context.get("case_ids", []) or []:
            self.case_list.addItem(str(case_id))
        middle.addWidget(self.case_list, 1)
        side = QtWidgets.QVBoxLayout()
        select_all = QtWidgets.QPushButton("Select All")
        select_all.clicked.connect(self._select_all_cases)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.case_list.clearSelection)
        side.addWidget(select_all)
        side.addWidget(clear)
        side.addSpacing(12)
        side.addWidget(QtWidgets.QLabel("{0} case(s) found".format(len(self.context.get("case_ids", []) or []))))
        hint = QtWidgets.QLabel("Selected cases override sample order.")
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addStretch(1)
        middle.addLayout(side)
        sample_layout.addLayout(middle, 1)

        manual = QtWidgets.QHBoxLayout()
        manual.addWidget(QtWidgets.QLabel("Manual cases"))
        self.manual_cases = QtWidgets.QLineEdit()
        manual.addWidget(self.manual_cases, 1)
        sample_layout.addLayout(manual)

        mins = QtWidgets.QHBoxLayout()
        mins.addWidget(QtWidgets.QLabel("Min train"))
        self.widgets["min_samples"] = self._spin(1, 100000, self.values.get("min_samples", 1))
        mins.addWidget(self.widgets["min_samples"])
        mins.addWidget(QtWidgets.QLabel("Min val"))
        self.widgets["min_val_samples"] = self._spin(0, 100000, self.values.get("min_val_samples", 1))
        mins.addWidget(self.widgets["min_val_samples"])
        mins.addStretch(1)
        sample_layout.addLayout(mins)
        self.widgets["export_labels_before_training"] = QtWidgets.QCheckBox(
            "Refresh labels from saved .mcs before training"
        )
        self.widgets["export_labels_before_training"].setChecked(
            _bool(self.values.get("export_labels_before_training", True))
        )
        self.widgets["export_labels_before_training"].setToolTip(
            "Turn this off only when labels have already been exported and a background import/export Mimics process is still busy."
        )
        sample_layout.addWidget(self.widgets["export_labels_before_training"])
        layout.addWidget(sample_group, 1)
        layout.addStretch(1)
        return tab

    def _build_expert_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        scroll.setWidget(body)
        tab_layout.addWidget(scroll)

        strategy_group = QtWidgets.QGroupBox("Training strategy")
        strategy_layout = QtWidgets.QGridLayout(strategy_group)
        strategy_layout.setColumnStretch(1, 1)
        self.widgets["strategy"] = QtWidgets.QComboBox()
        for strategy_id in strategy_ids():
            self.widgets["strategy"].addItem(strategy_label(strategy_id), strategy_id)
        initial_strategy = str(self.values.get("strategy", "adaptive"))
        initial_index = self.widgets["strategy"].findData(initial_strategy)
        self.widgets["strategy"].setCurrentIndex(max(0, initial_index))
        strategy_layout.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
        strategy_layout.addWidget(self.widgets["strategy"], 0, 1)
        self.strategy_summary = QtWidgets.QLabel()
        self.strategy_summary.setWordWrap(True)
        self.strategy_summary.setObjectName("strategySummary")
        strategy_layout.addWidget(self.strategy_summary, 1, 0, 1, 2)
        self.widgets["strategy"].currentTextChanged.connect(self._apply_strategy_defaults)
        layout.addWidget(strategy_group)

        data_group = QtWidgets.QGroupBox("Data policy")
        prediction_group = QtWidgets.QGroupBox("Objective and prediction")
        data_form = QtWidgets.QFormLayout(data_group)
        prediction_form = QtWidgets.QFormLayout(prediction_group)
        data_form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        prediction_form.setLabelAlignment(self.QtCore.Qt.AlignRight)

        self.widgets["sampling_mode"] = self._data_combo([
            ("Adaptive", "adaptive"), ("Full volume", "full"), ("Patch", "patch"),
        ], self.values.get("sampling_mode"))
        self.widgets["patch_size_mode"] = self._data_combo([
            ("From data fingerprint", "fingerprint"), ("Custom Z,Y,X", "custom"),
        ], self.values.get("patch_size_mode"))
        self.widgets["patch_size_zyx"] = QtWidgets.QLineEdit(str(self.values.get("patch_size_zyx")))
        self.widgets["patch_focus"] = self._data_combo([
            ("Foreground", "foreground"), ("Boundary", "boundary"),
            ("Foreground + nearby negatives", "negative_balanced"),
        ], self.values.get("patch_focus"))
        self.widgets["patches_per_case"] = self._spin(1, 32, self.values.get("patches_per_case", 2))
        self.widgets["channel_policy"] = self._data_combo([
            ("Single slice (repeat grayscale)", "repeat"), ("2.5D neighboring slices", "2_5d"),
        ], self.values.get("channel_policy"))
        self.widgets["slice_axis"] = self._data_combo([
            ("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal"),
        ], self.values.get("slice_axis"))
        self.widgets["neighbor_distance_mm"] = self._double_spin(
            0.1, 20.0, self.values.get("neighbor_distance_mm", 3.0), 0.5,
        )
        for label, key in (
            ("Sampling", "sampling_mode"), ("Patch size", "patch_size_mode"),
            ("Custom patch", "patch_size_zyx"), ("Patch focus", "patch_focus"),
            ("Patches / case", "patches_per_case"),
            ("Slice input", "channel_policy"), ("View plane", "slice_axis"),
            ("Neighbor distance (mm)", "neighbor_distance_mm"),
        ):
            data_form.addRow(label, self.widgets[key])

        self.widgets["loss_type"] = self._data_combo([
            ("Automatic (Dice + Focal)", "auto"), ("Dice + Focal", "dice_focal"),
            ("Dice + Cross Entropy", "dice_ce"),
        ], self.values.get("loss_type"))
        self.widgets["keep_largest_component"] = QtWidgets.QCheckBox("Keep largest connected component")
        self.widgets["keep_largest_component"].setChecked(_bool(self.values.get("keep_largest_component", False)))
        for label, key in (("Loss", "loss_type"),):
            prediction_form.addRow(label, self.widgets[key])
        inference_note = QtWidgets.QLabel("Inference is automatic: full-volume training uses whole-volume inference; patch training uses sliding windows.")
        inference_note.setWordWrap(True)
        prediction_form.addRow(inference_note)
        prediction_form.addRow(self.widgets["keep_largest_component"])
        policy_row = QtWidgets.QWidget()
        policy_layout = QtWidgets.QHBoxLayout(policy_row)
        policy_layout.setContentsMargins(0, 0, 0, 0)
        policy_layout.setSpacing(12)
        policy_layout.addWidget(data_group, 1)
        policy_layout.addWidget(prediction_group, 1)
        layout.addWidget(policy_row)

        left = QtWidgets.QGroupBox("Model and task")
        right = QtWidgets.QGroupBox("Training and resources")
        left_form = QtWidgets.QFormLayout(left)
        right_form = QtWidgets.QFormLayout(right)
        left_form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        right_form.setLabelAlignment(self.QtCore.Qt.AlignRight)

        model_hint = QtWidgets.QLabel("Model controls remain available, but the selected strategy supplies the validated data, loss, and inference policy.")
        model_hint.setWordWrap(True)
        left_form.addRow(model_hint)
        self.widgets["finetune_method"] = self._combo(["frozen", "lora", "adapter"], self.values.get("finetune_method", "frozen"))
        self.widgets["finetune_method"].currentTextChanged.connect(self._refresh_method_enabled)
        self.widgets["decoder"] = self._combo(DECODER_CHOICES, self.values.get("decoder", "segformer3d"))
        self.widgets["decoder"].setToolTip("3D heads fuse slices volumetrically. conv2d heads decode slices independently and stack them back into a 3D mask.")
        self.widgets["model_scale"] = self._combo(self._available_model_scales(), self.values.get("model_scale", "vitb16"))
        self.widgets["lora_rank"] = self._spin(1, 128, self.values.get("lora_rank", 8))
        self.widgets["lora_alpha"] = self._spin(1, 512, self.values.get("lora_alpha", 16))
        self.widgets["adapter_bottleneck"] = self._spin(1, 512, self.values.get("adapter_bottleneck", 64))
        for label, key in [
            ("Fine-tuning", "finetune_method"),
            ("Decoder", "decoder"),
            ("Pretrained scale", "model_scale"),
            ("LoRA rank", "lora_rank"),
            ("LoRA alpha", "lora_alpha"),
            ("Adapter bottleneck", "adapter_bottleneck"),
        ]:
            left_form.addRow(label, self.widgets[key])
        self.dimensionality_summary = QtWidgets.QLabel()
        self.dimensionality_summary.setWordWrap(True)
        self.dimensionality_summary.setObjectName("strategySummary")
        left_form.addRow(self.dimensionality_summary)
        backend_hint = QtWidgets.QLabel("Backend template, modality, and custom weight path are controlled by fewshot_config.json.")
        backend_hint.setWordWrap(True)
        left_form.addRow(backend_hint)

        resource_hint = QtWidgets.QLabel("Batch size is fixed at 1 for variable-depth volumes. Grad accumulation controls the effective batch.")
        resource_hint.setWordWrap(True)
        right_form.addRow(resource_hint)
        self.widgets["epochs"] = self._spin(1, 10000, self.values.get("epochs", 20))
        self.widgets["batch_size"] = self._spin(1, 1, 1)
        self.widgets["batch_size"].setVisible(False)
        self.widgets["batch_size"].setToolTip(
            "Fixed at 1 because Mimics cases can have different z-depth. Use Grad accumulation for a larger effective batch."
        )
        self.widgets["grad_accumulation"] = self._spin(1, 1024, self.values.get("grad_accumulation", 1))
        self.widgets["lr"] = self._combo(self.LR_CHOICES, self.values.get("lr", "0.001"), editable=True)
        self.widgets["weight_decay"] = self._combo(self.WEIGHT_DECAY_CHOICES, self.values.get("weight_decay", "0.01"), editable=True)
        self.widgets["lr_scheduler"] = self._combo(["cosine", "constant_warmup", "constant"], self.values.get("lr_scheduler", "cosine"))
        self.widgets["warmup_epochs"] = self._spin(0, 1000, self.values.get("warmup_epochs", 3))
        img_label = self._img_size_choice_from_value(self.values.get("img_size", "224,224"))
        self.quick_widgets["img_size_choice"] = self._combo([label for label, _ in self.IMG_SIZE_CHOICES] + [self.IMG_SIZE_CUSTOM_LABEL], img_label)
        self.widgets["img_size"] = QtWidgets.QLineEdit(str(self.values.get("img_size", "224,224")))
        self.widgets["img_size"].setPlaceholderText("width,height")
        self.widgets["sub_volume_depth"] = self._combo(self.SUB_VOLUME_DEPTH_CHOICES, self._sub_volume_depth_from_values(self.values))
        self.widgets["sub_volume_depth"].setToolTip(
            "Advanced memory option: split each 3D case into z-depth chunks during training."
        )
        self.widgets["keep_last_checkpoints"] = self._spin(0, 1000, self.values.get("keep_last_checkpoints", 2))
        self.widgets["val_fraction"] = self._double_spin(0.0, 0.9, self.values.get("val_fraction", 0.2), 0.05)
        self.widgets["mixed_precision"] = QtWidgets.QCheckBox("Mixed precision")
        self.widgets["mixed_precision"].setChecked(_bool(self.values.get("mixed_precision", False)))
        self.widgets["sub_volume"] = QtWidgets.QCheckBox("Sub-volume training")
        self.widgets["sub_volume"].setChecked(_bool(self.values.get("sub_volume", False)))

        for label, key in [
            ("Epochs", "epochs"),
            ("Grad accumulation", "grad_accumulation"),
            ("Learning rate", "lr"),
            ("Weight decay", "weight_decay"),
            ("LR schedule", "lr_scheduler"),
            ("Warmup epochs", "warmup_epochs"),
        ]:
            right_form.addRow(label, self.widgets[key])
        right_form.addRow("Image detail", self.quick_widgets["img_size_choice"])
        right_form.addRow("Custom size", self.widgets["img_size"])
        right_form.addRow("Validation fraction", self.widgets["val_fraction"])
        check_row = QtWidgets.QHBoxLayout()
        check_row.addWidget(self.widgets["mixed_precision"])
        check_row.addStretch(1)
        right_form.addRow(check_row)

        self.quick_widgets["img_size_choice"].currentTextChanged.connect(self._sync_image_size_choice)
        self.widgets["img_size"].textChanged.connect(self._sync_custom_image_size)
        for key in ("epochs", "batch_size", "grad_accumulation", "val_fraction", "sub_volume_depth"):
            self._connect_change(key, self._refresh_quick_labels)
        self.widgets["sub_volume"].stateChanged.connect(self._refresh_quick_labels)
        self.widgets["sub_volume"].stateChanged.connect(self._refresh_method_enabled)
        self.widgets["lr_scheduler"].currentTextChanged.connect(self._refresh_method_enabled)
        self.widgets["decoder"].currentTextChanged.connect(self._update_dimensionality_summary)
        self.widgets["sampling_mode"].currentTextChanged.connect(self._update_dimensionality_summary)
        self.widgets["channel_policy"].currentTextChanged.connect(self._update_dimensionality_summary)
        self._set_custom_size_entry_state()
        self._refresh_method_enabled()
        self._update_dimensionality_summary()
        epoch_label = _option_label(self.values.get("epochs", 10), self.EPOCH_CHOICES)
        val_label = _option_label(self.values.get("val_fraction", 0.2), self.VAL_CHOICES)
        memory_label = self._memory_mode_from_values(self.values)
        self.quick_widgets["epochs_choice"] = self._combo(_labels_with_current(self.EPOCH_CHOICES, epoch_label), epoch_label)
        self.quick_widgets["val_fraction_choice"] = self._combo(_labels_with_current(self.VAL_CHOICES, val_label), val_label)
        self.quick_widgets["memory_mode"] = self._combo([label for label, _ in self.MEMORY_CHOICES], memory_label)
        self.quick_widgets["img_size_choice"].setToolTip("Choose a preset or enter any positive width,height. Values should be multiples of 16.")
        parameter_row = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QHBoxLayout(parameter_row)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(12)
        parameter_layout.addWidget(left, 1)
        parameter_layout.addWidget(right, 1)
        layout.addWidget(parameter_row)
        layout.addStretch(1)
        self._update_strategy_summary()
        for key in STRATEGY_DATA_KEYS:
            self._connect_change(key, self._strategy_option_changed)
        self._refresh_strategy_enabled()
        return tab

    def _combo(self, values, current="", editable=False):
        combo = self.QtWidgets.QComboBox()
        combo.setEditable(bool(editable))
        text_values = [str(value) for value in values if str(value)]
        combo.addItems(text_values)
        current = str(current or "")
        if current and combo.findText(current) < 0:
            combo.addItem(current)
        if current:
            combo.setCurrentText(current)
        return combo

    def _data_combo(self, items, current=None):
        combo = self.QtWidgets.QComboBox()
        for label, value in items:
            combo.addItem(str(label), str(value))
        index = combo.findData(str(current))
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _spin(self, minimum, maximum, value, step=1):
        widget = self.QtWidgets.QSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setSingleStep(int(step))
        try:
            widget.setValue(int(value))
        except Exception:
            widget.setValue(int(minimum))
        return widget

    def _double_spin(self, minimum, maximum, value, step):
        widget = self.QtWidgets.QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(float(step))
        widget.setDecimals(4)
        try:
            widget.setValue(float(value))
        except Exception:
            widget.setValue(float(minimum))
        return widget

    def _connect_change(self, key, callback):
        widget = self.widgets.get(key)
        if widget is None:
            return
        if hasattr(widget, "valueChanged"):
            widget.valueChanged.connect(callback)
        elif hasattr(widget, "currentTextChanged"):
            widget.currentTextChanged.connect(callback)
        elif hasattr(widget, "textChanged"):
            widget.textChanged.connect(callback)

    def _select_all_cases(self):
        for idx in range(self.case_list.count()):
            self.case_list.item(idx).setSelected(True)

    def browse_dataset(self):
        path = self.QtWidgets.QFileDialog.getExistingDirectory(
            self.window,
            "Select dataset folder",
            self.context.get("ts_root", "") or str(Path.home()),
            self.QtWidgets.QFileDialog.ShowDirsOnly,
        )
        if path:
            self.apply_dataset_root(path)

    def browse_mcs_folder(self):
        current = str(self.mcs_folder_edit.text()).strip() if self.mcs_folder_edit is not None else ""
        path = self.QtWidgets.QFileDialog.getExistingDirectory(
            self.window,
            "Select folder containing saved .mcs projects",
            current or str(Path.home()),
            self.QtWidgets.QFileDialog.ShowDirsOnly,
        )
        if path and self.mcs_folder_edit is not None:
            self.mcs_folder_edit.setText(os.path.abspath(path))
            self.context["mcs_output_dir"] = os.path.abspath(path)

    def apply_dataset_from_field(self):
        if self.dataset_edit is None:
            return
        path = str(self.dataset_edit.text()).strip()
        if path:
            self.apply_dataset_root(path)

    def apply_dataset_root(self, path):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
        if self.dataset_edit is not None and self.dataset_edit.text() != path:
            self.dataset_edit.setText(path)
        self._dataset_scan_generation += 1
        generation = self._dataset_scan_generation
        project_root = self.context.get("project_root", "")
        self._set_status("Checking dataset in the background...")
        if self.case_list is not None:
            self.case_list.setEnabled(False)

        def scan_dataset():
            try:
                if not os.path.isdir(path):
                    raise RuntimeError("Dataset folder does not exist: {0}".format(path))
                output_dir = resolve_mimics_output_dir_for_ui(path, project_root)
                cases = case_ids_from_dataset_for_ui(path, project_root)
                self._dataset_scan_results.put((generation, path, output_dir, cases, ""))
            except Exception as exc:
                self._dataset_scan_results.put((generation, path, "", [], str(exc)))

        worker = threading.Thread(target=scan_dataset, name="fewshot-dataset-scan")
        worker.daemon = True
        worker.start()

    def _poll_dataset_scan(self):
        if self._closing:
            return
        while True:
            try:
                generation, path, output_dir, cases, error = self._dataset_scan_results.get_nowait()
            except Empty:
                return
            if generation != self._dataset_scan_generation:
                continue
            if error:
                self._set_status(error)
                self._append_log(error)
                if self.case_list is not None:
                    self.case_list.setEnabled(True)
                continue
            self.context["ts_root"] = path
            self.context["workspace"] = os.path.join(path, "fewshot_models")
            self.context["mcs_output_dir"] = output_dir
            if self.mcs_folder_edit is not None:
                self.mcs_folder_edit.setText(output_dir)
            self.context["case_ids"] = cases
            if self.case_list is not None:
                self.case_list.clear()
                self.case_list.addItems([str(case_id) for case_id in cases])
                self.case_list.setEnabled(True)
            message = "Dataset changed: {0} ({1} cases found)".format(path, len(cases))
            self._set_status(message)
            self._append_log(message)

    def _available_model_scales(self):
        app = object.__new__(TrainingSetupApp)
        app.context = self.context
        app.values = self.values
        return TrainingSetupApp._available_model_scales(app)

    def _sub_volume_depth_from_values(self, values):
        parts = str(values.get("sub_volume_size", "32,256,256")).split(",")
        depth = parts[0].strip() if parts else "32"
        return depth if depth in self.SUB_VOLUME_DEPTH_CHOICES else "32"

    def _img_size_choice_from_value(self, value):
        for label, item in self.IMG_SIZE_CHOICES:
            if str(item) == str(value):
                return label
        return self.IMG_SIZE_CUSTOM_LABEL

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

    def _widget_value(self, key):
        widget = self.widgets.get(key)
        if widget is None:
            return self.values.get(key)
        if isinstance(widget, self.QtWidgets.QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, self.QtWidgets.QSpinBox):
            return int(widget.value())
        if isinstance(widget, self.QtWidgets.QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, self.QtWidgets.QComboBox):
            if key == "strategy" or key in STRATEGY_DATA_KEYS:
                return str(widget.currentData() or "adaptive")
            return str(widget.currentText())
        if isinstance(widget, self.QtWidgets.QLineEdit):
            return str(widget.text()).strip()
        return self.values.get(key)

    def _set_widget_value(self, key, value):
        widget = self.widgets.get(key)
        if widget is None:
            self.values[key] = value
            return
        if isinstance(widget, self.QtWidgets.QCheckBox):
            widget.setChecked(_bool(value))
        elif isinstance(widget, self.QtWidgets.QSpinBox):
            try:
                widget.setValue(int(value))
            except Exception:
                pass
        elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
            try:
                widget.setValue(float(value))
            except Exception:
                pass
        elif isinstance(widget, self.QtWidgets.QComboBox):
            if key == "strategy" or key in STRATEGY_DATA_KEYS:
                index = widget.findData(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                self.values[key] = value
                return
            text = str(value)
            if widget.findText(text) < 0:
                widget.addItem(text)
            widget.setCurrentText(text)
        elif isinstance(widget, self.QtWidgets.QLineEdit):
            widget.setText(str(value))
        self.values[key] = value

    def _set_combo_value(self, widget, value):
        text = str(value)
        if widget.findText(text) < 0:
            widget.addItem(text)
        widget.setCurrentText(text)

    def _current_values(self):
        values = dict(self.values)
        if self.mcs_folder_edit is not None:
            values["mcs_output_dir"] = str(self.mcs_folder_edit.text()).strip()
        if self.mask_names_edit is not None:
            values["mask_names"] = str(self.mask_names_edit.text()).strip()
        for key in list(self.widgets.keys()):
            if key == "sub_volume_depth":
                continue
            values[key] = self._widget_value(key)
        depth = self._widget_value("sub_volume_depth") or self._sub_volume_depth_from_values(values)
        img_size = str(values.get("img_size", "224,224"))
        parts = [part.strip() for part in img_size.split(",")]
        if len(parts) == 2:
            values["sub_volume_size"] = "{0},{1},{2}".format(depth, parts[0], parts[1])
        return values

    def _sync_image_size_choice(self):
        choice = self.quick_widgets["img_size_choice"].currentText()
        if choice == self.IMG_SIZE_CUSTOM_LABEL:
            self._set_custom_size_entry_state()
            self.widgets["img_size"].setFocus()
            self._refresh_quick_labels()
            return
        else:
            value = _choice_value(choice, self.IMG_SIZE_CHOICES)
        if value:
            self.widgets["img_size"].setText(str(value))
        self._set_custom_size_entry_state()
        self._refresh_quick_labels()

    def _sync_custom_image_size(self):
        choice = self.quick_widgets.get("img_size_choice")
        if choice is not None:
            label = self._img_size_choice_from_value(self.widgets["img_size"].text().strip())
            if choice.currentText() != label:
                self._set_combo_value(choice, label)
        self._refresh_quick_labels()

    def _set_custom_size_entry_state(self):
        widget = self.widgets.get("img_size")
        choice = self.quick_widgets.get("img_size_choice")
        if widget is not None and choice is not None:
            widget.setEnabled(True)

    def _refresh_method_enabled(self):
        method_widget = self.widgets.get("finetune_method")
        method = str(method_widget.currentText() if method_widget is not None else "").lower()
        lora_enabled = method == "lora"
        adapter_enabled = method == "adapter"
        for key in ("lora_rank", "lora_alpha"):
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(lora_enabled)
        adapter = self.widgets.get("adapter_bottleneck")
        if adapter is not None:
            adapter.setEnabled(adapter_enabled)
        sub_volume = self.widgets.get("sub_volume")
        depth = self.widgets.get("sub_volume_depth")
        if sub_volume is not None and depth is not None:
            depth.setEnabled(bool(sub_volume.isChecked()))
        scheduler = self.widgets.get("lr_scheduler")
        warmup = self.widgets.get("warmup_epochs")
        if scheduler is not None and warmup is not None:
            warmup.setEnabled(str(scheduler.currentText()) in ("cosine", "constant_warmup"))

    def _update_strategy_summary(self):
        widget = self.widgets.get("strategy")
        if widget is None or not hasattr(self, "strategy_summary"):
            return
        strategy_id = str(widget.currentData() or "adaptive")
        self.strategy_summary.setText(strategy_summary(strategy_id))

    def _apply_strategy_defaults(self, _text=None):
        widget = self.widgets.get("strategy")
        if widget is None:
            return
        strategy_id = str(widget.currentData() or "adaptive")
        self._applying_strategy = True
        try:
            self.values["strategy"] = strategy_id
            for key, value in strategy_defaults(strategy_id).items():
                self._set_widget_value(key, value)
            self._update_strategy_summary()
            self._refresh_method_enabled()
            self._refresh_strategy_enabled()
            self._set_status("Preset applied: {0}".format(strategy_label(strategy_id)))
        finally:
            self._applying_strategy = False

    def _strategy_option_changed(self, *_args):
        if self._applying_strategy:
            return
        self.strategy_summary.setText(strategy_summary(str(self.widgets["strategy"].currentData())) + " Manual overrides are applied below.")
        self._refresh_strategy_enabled()

    def _refresh_strategy_enabled(self):
        sampling = str(self._widget_value("sampling_mode") or "full")
        patch_enabled = sampling in ("adaptive", "patch")
        for key in ("patch_size_mode", "patch_focus", "patches_per_case"):
            if self.widgets.get(key) is not None:
                self.widgets[key].setEnabled(patch_enabled)
        custom_patch = patch_enabled and self._widget_value("patch_size_mode") == "custom"
        self.widgets["patch_size_zyx"].setEnabled(custom_patch)
        self.widgets["neighbor_distance_mm"].setEnabled(self._widget_value("channel_policy") == "2_5d")
        self._update_dimensionality_summary()

    def _update_dimensionality_summary(self, *_args):
        if not hasattr(self, "dimensionality_summary"):
            return
        decoder = str(self._widget_value("decoder") or "segformer3d")
        sampling = str(self._widget_value("sampling_mode") or "adaptive")
        channels = str(self._widget_value("channel_policy") or "repeat")
        region = "a 3D sub-volume" if sampling == "patch" else ("a fingerprint-selected 3D region" if sampling == "adaptive" else "the full 3D volume")
        context = "neighboring-slice (2.5D) encoder context" if channels == "2_5d" else "single-slice encoder context"
        if decoder.startswith("conv2d"):
            decoding = "slice-wise 2D decoding followed by ordered 3D stacking"
        else:
            decoding = "volumetric 3D feature decoding"
        self.dimensionality_summary.setText("Uses {0}, {1}, and {2}.".format(region, context, decoding))

    def _sync_quick_settings(self):
        if self._syncing_quick:
            return
        self._syncing_quick = True
        try:
            epoch_value = _choice_value(self.quick_widgets["epochs_choice"].currentText(), self.EPOCH_CHOICES)
            if epoch_value is not None:
                self._set_widget_value("epochs", epoch_value)
            val_value = _choice_value(self.quick_widgets["val_fraction_choice"].currentText(), self.VAL_CHOICES)
            if val_value is not None:
                self._set_widget_value("val_fraction", val_value)
            mode = _choice_value(self.quick_widgets["memory_mode"].currentText(), self.MEMORY_CHOICES)
            if mode == "low_memory":
                self._set_widget_value("batch_size", 1)
                self._set_widget_value("grad_accumulation", 2)
                self.widgets["img_size"].setText("192,192")
                self.widgets["sub_volume"].setChecked(True)
                self._set_combo_value(self.widgets["sub_volume_depth"], "24")
                self.widgets["mixed_precision"].setChecked(False)
            elif mode == "quality":
                self._set_widget_value("batch_size", 1)
                self._set_widget_value("grad_accumulation", 2)
                self.widgets["img_size"].setText("256,256")
                self.widgets["sub_volume"].setChecked(False)
                self._set_combo_value(self.widgets["sub_volume_depth"], "32")
            elif mode == "balanced":
                self._set_widget_value("batch_size", 1)
                self._set_widget_value("grad_accumulation", 1)
                self.widgets["img_size"].setText("224,224")
                self.widgets["sub_volume"].setChecked(False)
                self._set_combo_value(self.widgets["sub_volume_depth"], "32")
        finally:
            self._syncing_quick = False
        self._refresh_quick_labels()

    def _refresh_quick_labels(self):
        if self._syncing_quick:
            return
        self._syncing_quick = True
        try:
            values = self._current_values()
            self._set_combo_value(self.quick_widgets["epochs_choice"], _option_label(values.get("epochs", 10), self.EPOCH_CHOICES))
            self._set_combo_value(self.quick_widgets["val_fraction_choice"], _option_label(values.get("val_fraction", 0.2), self.VAL_CHOICES))
            self._set_combo_value(self.quick_widgets["memory_mode"], self._memory_mode_from_values(values))
            img_label = self._img_size_choice_from_value(values.get("img_size", "224,224"))
            if self.quick_widgets["img_size_choice"].currentText() != self.IMG_SIZE_CUSTOM_LABEL:
                self._set_combo_value(self.quick_widgets["img_size_choice"], img_label)
            self._set_custom_size_entry_state()
            self._refresh_method_enabled()
        finally:
            self._syncing_quick = False

    def apply_profile(self):
        profile_widget = self.quick_widgets.get("profile")
        profile_name = profile_widget.currentText() if profile_widget is not None else ""
        values = default_training_options(self.config, profile_name)
        for key, value in values.items():
            self._set_widget_value(key, value)
        self._set_combo_value(self.widgets["sub_volume_depth"], self._sub_volume_depth_from_values(values))
        self._refresh_quick_labels()
        self._set_status("Applied profile: {0}".format(profile_name or "default"))
        self._append_log("Applied profile: {0}".format(profile_name or "default"))

    def collect_options(self):
        self._sync_image_size_choice()
        options = self._current_values()
        selected = []
        if self.case_list is not None:
            for item in self.case_list.selectedItems():
                selected.append(str(item.text()))
        manual = split_csv(self.manual_cases.text() if self.manual_cases is not None else "")
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
            self._set_status("Cannot start training: {0}".format(exc))
            self._append_log("Cannot start training: {0}".format(exc))
            return
        try:
            run_id, status_path, pid = launch_training(self.context, options)
        except Exception as exc:
            self._write_setup_failure(exc)
            self._set_status("Could not start training: {0}".format(exc))
            self._append_log("Could not start training: {0}".format(exc))
            return
        self.started = True
        self.training_status_path = status_path
        self.training_log_dir = os.path.dirname(os.path.dirname(status_path))
        self.start_button.setEnabled(False)
        self.open_log_button.setEnabled(True)
        self.close_button.setText("Close")
        self._set_status("Training started: {0} (PID {1})".format(run_id, pid))
        self._append_log("Training started in the background: {0} (PID {1})".format(run_id, pid))
        self._append_log("Status file: {0}".format(status_path))
        self.QtCore.QTimer.singleShot(500, self.poll_training_status)

    def poll_training_status(self):
        if self._closing or not self.training_status_path:
            return
        status = read_json(self.training_status_path, {}) or {}
        line = format_status_line(status)
        if line and line != self.last_status_line:
            self.last_status_line = line
            self._set_status(line)
            self._append_log(line)
        if status.get("train_log"):
            self.training_log_dir = os.path.dirname(status.get("train_log"))
        if status.get("status") in ("completed", "failed", "cancelled"):
            return
        self.QtCore.QTimer.singleShot(1500, self.poll_training_status)

    def _append_log(self, text):
        if self.status_text is None:
            return
        stamp = time.strftime("%H:%M:%S")
        self.status_text.append("[{0}] {1}".format(stamp, text))

    def _set_status(self, text):
        if self.status_label is not None:
            self.status_label.setText(str(text))

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
        write_json_best_effort(path, {
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

    def mark_closed_if_needed(self):
        if self._closing:
            return
        self._closing = True
        try:
            self._dataset_scan_timer.stop()
        except Exception:
            pass
        if self.started:
            return
        path = self.context.get("setup_status_path")
        if path:
            payload = read_json(path, {}) or {}
            payload.update({
                "status": "closed",
                "updated_at_epoch": time.time(),
            })
            write_json_best_effort(path, payload)

    def close(self):
        self.mark_closed_if_needed()
        self.window.close()


def QtGuiShortcut(QtGui, key, parent):
    return QtGui.QShortcut(QtGui.QKeySequence(key), parent)


def run_pyside6_ui(context):
    qt_modules = _load_pyside6()
    QtCore, _QtGui, QtWidgets = qt_modules

    class CloseAwareMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            QtWidgets.QMainWindow.__init__(self)
            self.controller = None

        def closeEvent(self, event):
            if self.controller is not None:
                self.controller.mark_closed_if_needed()
            QtWidgets.QMainWindow.closeEvent(self, event)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, TITLE)
    window = CloseAwareMainWindow()
    controller = QtTrainingSetupApp(window, context, qt_modules)
    window.controller = controller
    window.show()
    return app.exec()


def run_ui(context):
    errors = []
    backend = os.environ.get("MIMICS_DINOV3_GUI_BACKEND", "auto").strip().lower()
    if backend in ("", "auto", "pyside6", "qt"):
        try:
            return run_pyside6_ui(context)
        except Exception as exc:
            errors.append("PySide6: {0}".format(exc))
            if backend in ("pyside6", "qt"):
                raise
    try:
        import tkinter as tk
        from tkinter import messagebox  # noqa: F401
    except Exception as exc:
        message = (
            "The external DINOv3 training setup window could not open because neither PySide6 nor Tkinter "
            "is available in the configured external Python environment. Training was not started. "
            "Install the PySide6 Windows wheels in nninteractive_env, or use the default Train/Update Model entry."
        )
        setup_path = context.get("setup_status_path")
        if setup_path:
            write_json_best_effort(setup_path, {
                "schema_version": "mimics_fewshot_setup.v1",
                "job_id": context.get("setup_id"),
                "kind": "train_setup",
                "status": "failed",
                "organ": context.get("organ"),
                "ts_root": os.path.abspath(context.get("ts_root", "")),
                "error": message,
                "details": "; ".join(errors + ["Tkinter: {0}".format(exc)]),
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
            ("Batch size (fixed)", "1", "spin"),
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
        draw.rectangle((60, 238, width - 60, 320), outline="#d1d5db", fill="#f9fafb")
        draw.text((80, 256), "Profile", fill="#374151", font=font)
        draw.rectangle((155, 248, 360, 282), outline="#9ca3af", fill="#ffffff")
        draw.text((168, 256), "balanced", fill="#111827", font=font)
        draw.text((390, 256), "Use profiles for routine work; Expert is optional.", fill="#374151", font=small)
        draw.text((80, 292), "Dataset", fill="#374151", font=small)
        draw.rectangle((155, 286, 1030, 314), outline="#9ca3af", fill="#ffffff")
        draw.text((168, 292), "D:\\Dataset\\TotalSegmentator", fill="#111827", font=small)
        draw.rectangle((1045, 286, 1145, 314), outline="#9ca3af", fill="#ffffff")
        draw.text((1066, 292), "Browse", fill="#111827", font=small)

        draw.rectangle((60, 338, width - 60, 412), outline="#d1d5db", fill="#f9fafb")
        quick = [
            ("Training length", "Standard (10)"),
            ("Validation", "Standard validation (20%)"),
            ("Resource preset", "Balanced"),
        ]
        x = 80
        for label, value in quick:
            draw_field(x, 356, label, value, "select", w=245)
            x += 300

        draw.text((60, 440), "Samples", fill="#111827", font=head_font)
        draw.text((170, 444), "Sample order", fill="#374151", font=small)
        draw.rectangle((270, 436, 380, 466), outline="#9ca3af", fill="#ffffff")
        draw.text((282, 442), "all", fill="#111827", font=small)
        draw.text((410, 444), "Max samples", fill="#374151", font=small)
        draw.rectangle((510, 436, 610, 466), outline="#9ca3af", fill="#ffffff")
        draw.text((522, 442), "0", fill="#111827", font=small)

        draw.rectangle((60, 482, 835, 625), outline="#9ca3af", fill="#fbfdff")
        cases = ["s0001", "s0002", "s0003", "s0004", "s0005", "s0006"]
        y = 498
        for idx, case in enumerate(cases):
            if idx in (0, 1, 3, 6, 7):
                draw.rectangle((68, y - 4, 820, y + 22), fill="#dbeafe")
            draw.text((82, y), case, fill="#111827", font=font)
            y += 24
        draw.rectangle((870, 482, 990, 518), outline="#9ca3af", fill="#ffffff")
        draw.text((892, 491), "Select All", fill="#111827", font=font)
        draw.rectangle((870, 530, 990, 566), outline="#9ca3af", fill="#ffffff")
        draw.text((910, 539), "Clear", fill="#111827", font=font)
        draw.text((870, 575), "616 case(s) found", fill="#374151", font=small)
        draw.text((870, 607), "Selected cases override sample order.", fill="#374151", font=small)
        draw.text((60, 642), "Manual cases", fill="#374151", font=font)
        draw.rectangle((180, 634, 920, 670), outline="#9ca3af", fill="#ffffff")
        draw.text((194, 642), "s0012,s0016", fill="#111827", font=font)
        draw.rectangle((60, 682, 76, 698), outline="#6b7280", fill="#ffffff")
        draw.line((63, 690, 68, 695), fill="#2563eb", width=2)
        draw.line((68, 695, 76, 684), fill="#2563eb", width=2)
        draw.text((88, 679), "Refresh labels from saved .mcs before training", fill="#374151", font=small)

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
