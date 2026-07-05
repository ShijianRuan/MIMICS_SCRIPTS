#!/usr/bin/env python3
"""External DINOv3 few-shot pipeline for Mimics-Script.

This script never imports Mimics. It is safe to run from the foreground Mimics
process through subprocess.Popen because all long-running work happens here or
in child Python/Mimics background processes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import FileResourceLock, ResourceLockCancelled, ResourceLockTimeout, release_lock

DEFAULT_WORKSPACE = "fewshot_models"
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3
RESOURCE_LOCK_DIR = ROOT / ".mimics_runtime" / "locks"
GPU_LOCK_PATH = RESOURCE_LOCK_DIR / "gpu.lock"
BACKGROUND_MIMICS_LOCK_PATH = RESOURCE_LOCK_DIR / "background_mimics.lock"


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(path))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def rotate_log(path):
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size < LOG_ROTATE_BYTES:
            return
        backups = int(LOG_ROTATE_BACKUPS)
        if backups <= 0:
            path.unlink()
            return
        oldest = path.with_name(path.name + "." + str(backups))
        if oldest.is_file():
            oldest.unlink()
        for index in range(backups - 1, 0, -1):
            src = path.with_name(path.name + "." + str(index))
            dst = path.with_name(path.name + "." + str(index + 1))
            if src.is_file():
                src.rename(dst)
        path.rename(path.with_name(path.name + ".1"))
    except Exception:
        pass


def append_log(workspace, message):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "fewshot_pipeline.log"
    rotate_log(path)
    text = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text, flush=True)


def safe_slug(value):
    text = str(value or "unknown").strip()
    out = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("._") or "unknown"


def global_registry_path():
    return Path.home() / ".mimics_script" / "fewshot_model_index.json"


def parse_case_list(value):
    if not value:
        return None
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def load_repo_config():
    path = ROOT / "fewshot_config.json"
    if path.is_file():
        return read_json(path, {})
    return {}


def resolve_path(value, base):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def workspace_for(ts_root, workspace=None):
    if workspace:
        return Path(workspace).resolve()
    return Path(ts_root).resolve() / DEFAULT_WORKSPACE


def dinov3_root_from_args(args):
    repo_cfg = load_repo_config()
    value = args.dinov3_root or os.environ.get("MIMICS_FEWSHOT_DINOV3_ROOT") or repo_cfg.get("dinov3_project")
    if value:
        path = resolve_path(value, ROOT)
    else:
        path = (ROOT / "external" / "dinov3-medical-seg").resolve()
    if not (path / "scripts" / "train.py").is_file():
        raise RuntimeError("DINOv3 project was not found: {}".format(path))
    return path


def python_from_args(args, dinov3_root):
    repo_cfg = load_repo_config()
    explicit = args.python or os.environ.get("MIMICS_FEWSHOT_PYTHON") or repo_cfg.get("python")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        dinov3_root / ".venv" / "Scripts" / "python.exe",
        dinov3_root / ".venv" / "bin" / "python",
        dinov3_root / "venv" / "Scripts" / "python.exe",
        dinov3_root / "venv" / "bin" / "python",
        Path(sys.executable),
    ])
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    return str(candidates[-1])


def base_config_from_args(args, dinov3_root):
    repo_cfg = load_repo_config()
    value = args.base_config or repo_cfg.get("base_config") or "config/synthstrip_lora_segformer3d.yaml"
    path = resolve_path(value, dinov3_root)
    if not path.is_file():
        raise RuntimeError("base config was not found: {}".format(path))
    return path


def find_mimics_exe(explicit=None):
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))
    env = os.environ.get("MIMICS_EXE")
    if env and Path(env).is_file():
        return env
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Materialise" / "Mimics Research 21.0" / "MimicsResearch.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Mimics Research 21.0" / "MimicsResearch.exe",
        Path(r"D:\Mimics Research 21.0\MimicsResearch.exe"),
        Path(r"C:\Mimics Research 21.0\MimicsResearch.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def hidden_process_kwargs():
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"startupinfo": startupinfo, "creationflags": flags}


def background_env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


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


def _env_flag_disabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def gpu_lock_enabled():
    if _env_flag_disabled("MIMICS_DISABLE_GPU_LOCK"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        # If torch cannot be imported here, the child training process will
        # fail soon anyway. Keep the lock conservative so a broken environment
        # does not start competing with nnInteractive.
        return True


def _lock_wait_payload(resource, current):
    if not isinstance(current, dict):
        current = {}
    return {
        "resource": resource,
        "owner": current.get("owner", "unknown"),
        "pid": current.get("pid", ""),
        "created_at_epoch": current.get("created_at_epoch"),
    }


def cleanup_idle_nninteractive_server_lock(current):
    if not isinstance(current, dict):
        return False
    if current.get("resource") != "gpu":
        return False
    state_path = current.get("state_path")
    if not state_path:
        return False
    state = read_json(state_path, {}) or {}
    if state.get("schema_version") != "nninteractive_owned_server.v2":
        return False
    if state.get("gpu_lock_token") != current.get("token"):
        return False
    try:
        pid = int(state.get("pid", 0))
    except Exception:
        pid = 0
    if not pid:
        return False
    try:
        watchdog_pid = int(state.get("watchdog_pid", 0) or 0)
    except Exception:
        watchdog_pid = 0
    if watchdog_pid and process_exists(watchdog_pid):
        return False
    try:
        idle_timeout = float(state.get("service_idle_timeout_seconds", 1800))
        last_activity = float(state.get("last_activity_epoch", time.time()))
    except Exception:
        idle_timeout = 1800.0
        last_activity = time.time()
    if time.time() - last_activity < idle_timeout + 60.0:
        return False
    if process_exists(pid):
        terminate_process_tree(pid)
    release_lock(current.get("path") or GPU_LOCK_PATH, current.get("token"))
    try:
        Path(state_path).unlink()
    except Exception:
        pass
    return True


def acquire_gpu_lock_for_job(workspace, status_path, cancel_path, owner, timeout_seconds):
    if not gpu_lock_enabled():
        return None
    lock = FileResourceLock(GPU_LOCK_PATH, "gpu", owner)
    cleanup_idle_nninteractive_server_lock(lock.read())
    last_log = {"epoch": 0.0}

    def on_wait(current):
        if cleanup_idle_nninteractive_server_lock(current):
            append_log(workspace, "Cleaned an idle nnInteractive server whose watchdog was no longer running.")
            return
        update_status(status_path, {
            "status": "waiting_for_gpu",
            "resource_wait": _lock_wait_payload("gpu", current),
        })
        now = time.time()
        if now - last_log["epoch"] >= 60:
            holder = current.get("owner", "unknown") if isinstance(current, dict) else "unknown"
            pid = current.get("pid", "?") if isinstance(current, dict) else "?"
            append_log(workspace, "Waiting for GPU resource held by {} (pid {}).".format(holder, pid))
            last_log["epoch"] = now

    update_status(status_path, {
        "status": "waiting_for_gpu",
        "resource_wait": {"resource": "gpu"},
    })
    lock.acquire(
        wait_seconds=float(timeout_seconds),
        poll_seconds=2.0,
        on_wait=on_wait,
        should_cancel=lambda: Path(cancel_path).is_file(),
    )
    update_status(status_path, {"resource_wait": None})
    append_log(workspace, "GPU resource acquired for {}.".format(owner))
    return lock


def acquire_background_mimics_lock(workspace, status_path, cancel_path, owner, timeout_seconds):
    lock = FileResourceLock(BACKGROUND_MIMICS_LOCK_PATH, "background_mimics", owner)
    last_log = {"epoch": 0.0}

    def on_wait(current):
        update_status(status_path, {
            "status": "waiting_for_background_mimics",
            "resource_wait": _lock_wait_payload("background_mimics", current),
        })
        now = time.time()
        if now - last_log["epoch"] >= 60:
            holder = current.get("owner", "unknown") if isinstance(current, dict) else "unknown"
            pid = current.get("pid", "?") if isinstance(current, dict) else "?"
            append_log(workspace, "Waiting for background Mimics held by {} (pid {}).".format(holder, pid))
            last_log["epoch"] = now

    update_status(status_path, {
        "status": "waiting_for_background_mimics",
        "resource_wait": {"resource": "background_mimics"},
    })
    lock.acquire(
        wait_seconds=float(timeout_seconds),
        poll_seconds=5.0,
        on_wait=on_wait,
        should_cancel=lambda: Path(cancel_path).is_file(),
    )
    update_status(status_path, {"resource_wait": None})
    return lock


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
                **hidden_process_kwargs()
            )
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def latest_running_job(workspace):
    jobs_dir = Path(workspace) / "jobs"
    if not jobs_dir.is_dir():
        return None, None
    active_statuses = set([
        "launching",
        "preparing",
        "exporting_labels",
        "waiting_for_background_mimics",
        "waiting_for_gpu",
        "training",
        "running",
        "cancelling",
    ])
    rows = []
    for path in jobs_dir.glob("*.json"):
        payload = read_json(path, {}) or {}
        if payload.get("status") not in active_statuses:
            continue
        pid = payload.get("pid") or payload.get("launcher_pid") or payload.get("controller_pid")
        if pid and not process_exists(pid):
            continue
        rows.append((path.stat().st_mtime, path, payload))
    if not rows:
        return None, None
    rows.sort(reverse=True)
    return rows[0][1], rows[0][2]


def case_dirs(ts_root, cases=None):
    root = Path(ts_root)
    wanted = set(cases or [])
    result = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ("mcs_output", "segmentations", DEFAULT_WORKSPACE):
            continue
        if wanted and child.name not in wanted:
            continue
        result.append(child)
    return result


def find_image(case_dir):
    for name in ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii"):
        path = Path(case_dir) / name
        if path.is_file():
            return path
    for path in sorted(Path(case_dir).glob("*.nii.gz")) + sorted(Path(case_dir).glob("*.nii")):
        if "segmentations" not in [part.lower() for part in path.parts]:
            return path
    return None


def find_label(case_dir, organ):
    seg_dir = Path(case_dir) / "segmentations"
    if not seg_dir.is_dir():
        return None
    names = [
        organ + ".nii.gz",
        organ + ".nii",
        safe_slug(organ) + ".nii.gz",
        safe_slug(organ) + ".nii",
    ]
    for name in names:
        path = seg_dir / name
        if path.is_file():
            return path
    organ_lower = organ.lower()
    for path in sorted(seg_dir.glob("*.nii.gz")) + sorted(seg_dir.glob("*.nii")):
        stem = path.name.replace(".nii.gz", "").replace(".nii", "")
        if stem.lower() == organ_lower:
            return path
    return None


def discover_samples(ts_root, organ, cases=None):
    samples = []
    skipped = []
    for case_dir in case_dirs(ts_root, cases):
        image = find_image(case_dir)
        label = find_label(case_dir, organ)
        if image and label:
            samples.append({
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "image": str(image),
                "label": str(label),
                "label_mtime": float(label.stat().st_mtime),
            })
        else:
            skipped.append({
                "case_id": case_dir.name,
                "has_image": bool(image),
                "has_label": bool(label),
            })
    return samples, skipped


def select_samples(samples, mode, max_samples):
    selected = list(samples)
    if mode == "latest":
        selected.sort(key=lambda item: item.get("label_mtime", 0.0), reverse=True)
    else:
        selected.sort(key=lambda item: item["case_id"])
    if max_samples and max_samples > 0:
        selected = selected[:max_samples]
    selected.sort(key=lambda item: item["case_id"])
    return selected


def split_train_validation(samples, val_fraction=0.0, val_cases=None, min_train_samples=1, min_val_samples=1):
    selected = list(samples)
    selected.sort(key=lambda item: item["case_id"])
    val_case_set = set(val_cases or [])
    if val_case_set:
        train_samples = [item for item in selected if item["case_id"] not in val_case_set]
        val_samples = [item for item in selected if item["case_id"] in val_case_set]
        if len(train_samples) < int(min_train_samples):
            raise RuntimeError(
                "validation case selection leaves only {} training samples; at least {} required".format(
                    len(train_samples),
                    int(min_train_samples),
                )
            )
        return train_samples, val_samples

    fraction = float(val_fraction or 0.0)
    if fraction <= 0.0:
        return selected, []
    if len(selected) <= int(min_train_samples):
        return selected, []

    val_count = int(round(len(selected) * fraction))
    if val_count <= 0 and len(selected) >= int(min_train_samples) + int(min_val_samples):
        val_count = int(min_val_samples)
    val_count = max(0, min(val_count, len(selected) - int(min_train_samples)))
    if val_count <= 0:
        return selected, []
    return selected[:-val_count], selected[-val_count:]


def copy_or_link(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copy"


def _materialize_split(samples, image_dir, label_dir, split_name):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for sample in samples:
        case_id = safe_slug(sample["case_id"])
        image_src = Path(sample["image"])
        label_src = Path(sample["label"])
        image_ext = ".nii.gz" if image_src.name.endswith(".nii.gz") else ".nii"
        label_ext = ".nii.gz" if label_src.name.endswith(".nii.gz") else ".nii"
        image_dst = image_dir / (case_id + image_ext)
        label_dst = label_dir / (case_id + label_ext)
        image_method = copy_or_link(image_src, image_dst)
        label_method = copy_or_link(label_src, label_dst)
        row = dict(sample)
        row.update({
            "split": split_name,
            "dataset_image": str(image_dst),
            "dataset_label": str(label_dst),
            "image_materialization": image_method,
            "label_materialization": label_method,
        })
        materialized.append(row)
    return materialized


def materialize_dataset(train_samples, dataset_dir, val_samples=None):
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(str(dataset_dir))
    train_rows = _materialize_split(
        train_samples,
        dataset_dir / "imagesTr",
        dataset_dir / "labelsTr",
        "train",
    )
    val_rows = _materialize_split(
        val_samples or [],
        dataset_dir / "imagesVal",
        dataset_dir / "labelsVal",
        "validation",
    )
    return train_rows, val_rows


def yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(item) for item in value) + "]"
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return '"' + text + '"'


def write_training_config(path, base_config, dataset_dir, exp_name, args, status_path=None, cancel_path=None):
    path = Path(path)
    img_size = [int(part.strip()) for part in str(args.img_size).split(",")]
    model_path = args.model_path
    if not model_path:
        model_scale = str(args.model_scale or "vitb16").lower()
        if model_scale in ("vitl16", "vit_l", "large"):
            model_path = "./models/dinov3-vitl16"
        elif model_scale in ("vith16plus", "vit_h", "huge"):
            model_path = "./models/dinov3-vith16plus"
        else:
            model_path = "./models/dinov3-vitb16"
    finetune_method = str(args.finetune_method or "lora").lower()
    if finetune_method in ("decoder_only", "decode_only", "decoder-only", "decode-only"):
        finetune_method = "frozen"
    decoder_type = str(args.decoder or "segformer3d")
    lines = [
        "_base_:",
        "  - " + yaml_scalar(str(base_config)),
        "",
        "exp_name: " + yaml_scalar(exp_name),
        "",
        "model:",
        "  model_path: " + yaml_scalar(model_path),
        "  num_classes: 2",
        "",
        "finetune:",
        "  method: " + yaml_scalar(finetune_method),
        "  lora_rank: " + yaml_scalar(int(args.lora_rank)),
        "  lora_alpha: " + yaml_scalar(int(args.lora_alpha)),
        "  adapter_bottleneck: " + yaml_scalar(int(args.adapter_bottleneck)),
        "",
        "decoder:",
        "  type: " + yaml_scalar(decoder_type),
        "",
        "data:",
        "  name: " + yaml_scalar("mimics_fewshot_" + safe_slug(args.organ)),
        "  data_root: " + yaml_scalar(str(dataset_dir)),
        "  img_size: " + yaml_scalar(img_size),
        "  k_shot: -1",
        "  fold: 0",
        "  modality: " + yaml_scalar(args.modality),
        "",
        "training:",
        "  epochs: " + yaml_scalar(int(args.epochs)),
        "  batch_size: " + yaml_scalar(int(args.batch_size)),
        "  grad_accumulation: " + yaml_scalar(int(args.grad_accumulation)),
        "  mixed_precision: " + yaml_scalar(bool(args.mixed_precision)),
        "  lr: " + yaml_scalar(float(args.lr)),
        "  weight_decay: " + yaml_scalar(float(args.weight_decay)),
        "  keep_last_checkpoints: " + yaml_scalar(int(args.keep_last_checkpoints)),
        "  sub_volume:",
        "    enabled: " + yaml_scalar(bool(args.sub_volume)),
        "    size: " + yaml_scalar([int(part.strip()) for part in str(args.sub_volume_size).split(",")]),
        "",
    ]
    if status_path or cancel_path:
        lines.extend([
            "runtime:",
            "  status_path: " + yaml_scalar(str(status_path or "")),
            "  cancel_path: " + yaml_scalar(str(cancel_path or "")),
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def launch_mimics_export(
    ts_root,
    cases,
    mimics_exe,
    workspace,
    timeout_seconds,
    status_path=None,
    cancel_path=None,
    lock_timeout_seconds=None,
):
    mimics_exe = find_mimics_exe(mimics_exe)
    if not mimics_exe:
        append_log(workspace, "MimicsResearch.exe was not found; using existing exported labels only.")
        return {"launched": False, "reason": "mimics_not_found"}
    output_dir = Path(ts_root) / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "_export_batch_config.json"
    runner = output_dir / "_run_export_batch.py"
    write_json_atomic(config, {
        "ts_root": str(Path(ts_root).resolve()),
        "cases": sorted(cases) if cases else None,
        "axes": [0, 1, 2],
        "flips": [False, False, False],
    })
    runner.write_text(
        "\n".join([
            "# Auto-generated runner for Mimics few-shot label export",
            "import sys, os",
            "sys.path.insert(0, r'{}')".format(str(ROOT / "runtime_py35")),
            "import mimics_export",
            "mimics_export.run_background_batch_export(r'{}')".format(str(config)),
            "",
        ]),
        encoding="utf-8",
    )
    lock = None
    if status_path and cancel_path:
        lock = acquire_background_mimics_lock(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 label export",
            lock_timeout_seconds if lock_timeout_seconds is not None else timeout_seconds,
        )
    log_path = output_dir / "_fewshot_export_mimics.log"
    try:
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                [mimics_exe, "-b", "-run_script", str(runner)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **hidden_process_kwargs()
            )
        if lock is not None:
            lock.update_pid(proc.pid, kind="fewshot_label_export", ts_root=str(Path(ts_root).resolve()))
    except Exception:
        if lock is not None:
            lock.release()
        raise
    append_log(workspace, "Background Mimics export started, pid={}.".format(proc.pid))
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if proc.poll() is not None:
            if lock is not None:
                lock.release()
            return {"launched": True, "returncode": proc.returncode, "log": str(log_path)}
        time.sleep(2.0)
    return {"launched": True, "timed_out": True, "pid": proc.pid, "log": str(log_path)}


def latest_epoch_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    best = ckpt_dir / "best_model.pth"
    if best.is_file():
        return best
    candidates = sorted(ckpt_dir.glob("epoch_*.pth"))
    return candidates[-1] if candidates else None


def update_status(path, payload):
    existing = read_json(path, {}) or {}
    existing.update(payload)
    existing["updated_at_epoch"] = time.time()
    write_json_atomic(path, existing)


def register_global_model(manifest):
    path = global_registry_path()
    payload = read_json(path, {}) or {}
    models = payload.get("models") or []
    manifest_path = str(Path(manifest["checkpoint"]).parent / "manifest.json")
    row = {
        "organ": manifest.get("organ", ""),
        "organ_slug": manifest.get("organ_slug", safe_slug(manifest.get("organ", ""))),
        "model_id": manifest.get("model_id", ""),
        "manifest_path": manifest_path,
        "checkpoint": manifest.get("checkpoint", ""),
        "config": manifest.get("config", ""),
        "sample_count": manifest.get("sample_count", 0),
        "train_sample_count": manifest.get("train_sample_count", 0),
        "validation_sample_count": manifest.get("validation_sample_count", 0),
        "created_at_epoch": manifest.get("created_at_epoch", time.time()),
        "ts_root": manifest.get("ts_root", ""),
        "workspace": manifest.get("workspace", ""),
    }
    models = [
        item for item in models
        if not (
            item.get("organ_slug") == row["organ_slug"]
            and item.get("model_id") == row["model_id"]
            and item.get("manifest_path") == row["manifest_path"]
        )
    ]
    models.append(row)
    models.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    write_json_atomic(path, {"schema_version": "mimics_fewshot_global_models.v1", "models": models[:200]})


def global_model_rows(organ=None):
    payload = read_json(global_registry_path(), {}) or {}
    rows = []
    wanted = safe_slug(organ) if organ else None
    for row in payload.get("models") or []:
        if wanted and row.get("organ_slug") != wanted:
            continue
        manifest_path = Path(row.get("manifest_path", ""))
        if not manifest_path.is_file():
            continue
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    return rows


def cmd_train(args):
    ts_root = Path(args.ts_root).resolve()
    dinov3_root = dinov3_root_from_args(args)
    python_exe = python_from_args(args, dinov3_root)
    base_config = base_config_from_args(args, dinov3_root)
    workspace = workspace_for(ts_root, args.workspace)
    organ_slug = safe_slug(args.organ)
    run_id = args.run_id or "train_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_dir = workspace / "jobs"
    status_path = job_dir / (run_id + ".json")
    run_dir = workspace / "runs" / organ_slug / run_id
    dataset_dir = workspace / "datasets" / organ_slug / run_id
    config_path = run_dir / "config.yaml"
    train_log = run_dir / "train.log"
    train_status = run_dir / "train_status.json"
    cancel_path = run_dir / "cancel.request"
    exp_name = "mimics_fewshot_{}_{}".format(organ_slug, run_id)

    workspace.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(status_path, {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": run_id,
        "kind": "train",
        "status": "preparing",
        "organ": args.organ,
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "controller_pid": os.getpid(),
        "cancel_path": str(cancel_path),
        "created_at_epoch": time.time(),
    })
    append_log(workspace, "Training job {} started for organ {}.".format(run_id, args.organ))

    cases = set(parse_case_list(args.cases) or [])
    if not cases:
        cases = None
    if args.export_labels:
        update_status(status_path, {"status": "exporting_labels"})
        try:
            export_result = launch_mimics_export(
                ts_root,
                cases,
                args.mimics_exe,
                workspace,
                args.export_timeout_seconds,
                status_path=status_path,
                cancel_path=cancel_path,
                lock_timeout_seconds=args.background_mimics_lock_timeout_seconds,
            )
        except ResourceLockCancelled:
            update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for background Mimics"})
            return 130
        except ResourceLockTimeout as exc:
            update_status(status_path, {"status": "failed", "error": str(exc)})
            return 75
        update_status(status_path, {"label_export": export_result})

    samples, skipped = discover_samples(ts_root, args.organ, cases)
    selected = select_samples(samples, args.sample_mode, int(args.max_samples or 0))
    if len(selected) < int(args.min_samples):
        update_status(status_path, {
            "status": "failed",
            "error": "not enough labeled samples for organ {}; found {}, required {}".format(
                args.organ,
                len(selected),
                args.min_samples,
            ),
            "samples_found": len(samples),
            "skipped": skipped[:100],
        })
        return 2

    train_samples, val_samples = split_train_validation(
        selected,
        val_fraction=args.val_fraction,
        val_cases=parse_case_list(args.val_cases),
        min_train_samples=args.min_samples,
        min_val_samples=args.min_val_samples,
    )
    materialized_train, materialized_val = materialize_dataset(train_samples, dataset_dir, val_samples)
    write_training_config(
        config_path,
        base_config,
        dataset_dir,
        exp_name,
        args,
        status_path=train_status,
        cancel_path=cancel_path,
    )
    write_json_atomic(
        run_dir / "samples.json",
        {
            "samples": materialized_train + materialized_val,
            "train_samples": materialized_train,
            "validation_samples": materialized_val,
            "skipped": skipped,
            "selection": {
                "cases": sorted(cases) if cases else None,
                "sample_mode": args.sample_mode,
                "max_samples": int(args.max_samples or 0),
                "val_fraction": float(args.val_fraction or 0.0),
                "val_cases": parse_case_list(args.val_cases),
            },
        },
    )
    update_status(status_path, {
        "status": "waiting_for_gpu" if gpu_lock_enabled() else "training",
        "sample_count": len(materialized_train) + len(materialized_val),
        "train_sample_count": len(materialized_train),
        "validation_sample_count": len(materialized_val),
        "train_cases": [item["case_id"] for item in materialized_train],
        "validation_cases": [item["case_id"] for item in materialized_val],
        "dataset_dir": str(dataset_dir),
        "config_path": str(config_path),
        "train_log": str(train_log),
        "training_status": str(train_status),
        "cancel_path": str(cancel_path),
    })

    cmd = [python_exe, str(dinov3_root / "scripts" / "train.py"), "--config", str(config_path)]
    append_log(workspace, "Launching DINOv3 training: {}".format(" ".join(cmd)))
    proc = None
    gpu_lock = None
    try:
        gpu_lock = acquire_gpu_lock_for_job(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 training {}".format(run_id),
            args.gpu_lock_timeout_seconds,
        )
        update_status(status_path, {"status": "training"})
        with train_log.open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(dinov3_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=background_env(),
            )
            if gpu_lock is not None:
                gpu_lock.update_pid(proc.pid, kind="fewshot_train", job_id=run_id)
            update_status(status_path, {"pid": proc.pid, "command": cmd})
            cancel_started = None
            while proc.poll() is None:
                progress = read_json(train_status, {}) or {}
                payload = {"status": "training", "pid": proc.pid}
                if progress:
                    payload["training_progress"] = progress
                if cancel_path.is_file():
                    payload["status"] = "cancelling"
                    if cancel_started is None:
                        cancel_started = time.time()
                    elif time.time() - cancel_started > 30:
                        terminate_process_tree(proc.pid)
                update_status(status_path, payload)
                time.sleep(1.0)
    except ResourceLockCancelled:
        update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for GPU"})
        append_log(workspace, "Training job {} cancelled while waiting for GPU.".format(run_id))
        return 130
    except ResourceLockTimeout as exc:
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Training job {} could not acquire GPU: {}.".format(run_id, exc))
        return 75
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            terminate_process_tree(proc.pid)
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Training job {} failed before completion: {}.".format(run_id, exc))
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.release()

    final_progress = read_json(train_status, {}) or {}
    if cancel_path.is_file() or final_progress.get("status") == "cancelled":
        update_status(status_path, {
            "status": "cancelled",
            "returncode": proc.returncode,
            "training_progress": final_progress,
        })
        append_log(workspace, "Training job {} cancelled.".format(run_id))
        return 130

    if proc.returncode != 0:
        update_status(status_path, {
            "status": "failed",
            "returncode": proc.returncode,
            "error": "training process failed; see train_log",
        })
        return proc.returncode or 1

    exp_dir = dinov3_root / "experiments" / exp_name
    ckpt = latest_epoch_checkpoint(exp_dir / "checkpoints")
    if not ckpt:
        update_status(status_path, {"status": "failed", "error": "no checkpoint was produced"})
        return 1

    model_dir = workspace / "models" / organ_slug / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_ckpt = model_dir / "model.pth"
    shutil.copy2(str(ckpt), str(registry_ckpt))
    shutil.copy2(str(config_path), str(model_dir / "config.yaml"))
    manifest = {
        "schema_version": "mimics_fewshot_model.v1",
        "model_id": run_id,
        "organ": args.organ,
        "organ_slug": organ_slug,
        "checkpoint": str(registry_ckpt),
        "config": str(model_dir / "config.yaml"),
        "source_checkpoint": str(ckpt),
        "experiment_dir": str(exp_dir),
        "sample_count": len(materialized_train) + len(materialized_val),
        "train_sample_count": len(materialized_train),
        "validation_sample_count": len(materialized_val),
        "samples": materialized_train + materialized_val,
        "train_samples": materialized_train,
        "validation_samples": materialized_val,
        "dataset_dir": str(dataset_dir),
        "dataset_retained": bool(args.keep_materialized_dataset),
        "training_progress": final_progress,
        "training_parameters": {
            "base_config": str(base_config),
            "finetune_method": str(args.finetune_method),
            "decoder": str(args.decoder),
            "model_scale": str(args.model_scale),
            "model_path": str(args.model_path or ""),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accumulation": int(args.grad_accumulation),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "img_size": str(args.img_size),
            "modality": str(args.modality),
            "mixed_precision": bool(args.mixed_precision),
            "sub_volume": bool(args.sub_volume),
            "sub_volume_size": str(args.sub_volume_size),
            "keep_last_checkpoints": int(args.keep_last_checkpoints),
            "keep_materialized_dataset": bool(args.keep_materialized_dataset),
        },
        "created_at_epoch": time.time(),
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "dinov3_root": str(dinov3_root),
        "base_config": str(base_config),
    }
    write_json_atomic(model_dir / "manifest.json", manifest)
    write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
    register_global_model(manifest)
    if not args.keep_materialized_dataset:
        try:
            shutil.rmtree(str(dataset_dir))
            manifest["dataset_retained"] = False
            manifest["dataset_cleanup_at_epoch"] = time.time()
            write_json_atomic(model_dir / "manifest.json", manifest)
            write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
        except Exception as exc:
            manifest["dataset_retained"] = True
            manifest["dataset_cleanup_error"] = str(exc)
            write_json_atomic(model_dir / "manifest.json", manifest)
            write_json_atomic(workspace / "models" / organ_slug / "latest.json", manifest)
            append_log(workspace, "Could not clean materialized dataset {}: {}".format(dataset_dir, exc))
    update_status(status_path, {
        "status": "completed",
        "model": manifest,
        "returncode": 0,
    })
    append_log(workspace, "Training job {} completed. Model: {}".format(run_id, registry_ckpt))
    return 0


def load_model_manifest(workspace, organ, model_id=None, model_manifest=None):
    if model_manifest:
        manifest = read_json(model_manifest)
        if not manifest:
            raise RuntimeError("model manifest could not be read: {}".format(model_manifest))
        return manifest
    organ_slug = safe_slug(organ)
    if model_id and model_id != "latest":
        path = Path(workspace) / "models" / organ_slug / model_id / "manifest.json"
    else:
        path = Path(workspace) / "models" / organ_slug / "latest.json"
    manifest = read_json(path)
    if not manifest:
        raise RuntimeError("no model manifest found for organ {} at {}".format(organ, path))
    return manifest


def cmd_infer(args):
    ts_root = Path(args.ts_root).resolve()
    case_dir = ts_root / args.case_id
    if not case_dir.is_dir():
        raise RuntimeError("case directory was not found: {}".format(case_dir))
    image = find_image(case_dir)
    if not image:
        raise RuntimeError("no NIfTI image found for case: {}".format(args.case_id))

    dinov3_root = dinov3_root_from_args(args)
    python_exe = python_from_args(args, dinov3_root)
    workspace = workspace_for(ts_root, args.workspace)
    organ_slug = safe_slug(args.organ)
    model = load_model_manifest(workspace, args.organ, args.model_id, args.model_manifest)
    job_id = args.job_id or "infer_{}_{}_{}".format(
        safe_slug(args.case_id),
        organ_slug,
        uuid.uuid4().hex[:8],
    )
    status_path = workspace / "jobs" / (job_id + ".json")
    output_dir = workspace / "predictions" / safe_slug(args.case_id) / organ_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_model_id = args.model_id or model.get("model_id", "latest")
    if args.model_manifest and output_model_id == "latest":
        output_model_id = model.get("model_id", "external_model")
    output_path = output_dir / (safe_slug(output_model_id) + ".nii.gz")
    log_path = output_dir / (job_id + ".log")
    cancel_path = output_dir / (job_id + ".cancel")

    write_json_atomic(status_path, {
        "schema_version": "mimics_fewshot_job.v1",
        "job_id": job_id,
        "kind": "infer",
        "status": "waiting_for_gpu" if gpu_lock_enabled() else "running",
        "organ": args.organ,
        "case_id": args.case_id,
        "ts_root": str(ts_root),
        "workspace": str(workspace),
        "image_path": str(image),
        "output_path": str(output_path),
        "model_id": model.get("model_id", args.model_id or "latest"),
        "model_manifest": args.model_manifest or "",
        "model": model,
        "controller_pid": os.getpid(),
        "cancel_path": str(cancel_path),
        "created_at_epoch": time.time(),
    })
    cmd = [
        python_exe,
        str(dinov3_root / "scripts" / "infer.py"),
        "--config",
        str(model["config"]),
        "--checkpoint",
        str(model["checkpoint"]),
        "--input",
        str(image),
        "--output",
        str(output_path),
    ]
    append_log(workspace, "Launching DINOv3 inference: {}".format(" ".join(cmd)))
    proc = None
    gpu_lock = None
    try:
        gpu_lock = acquire_gpu_lock_for_job(
            workspace,
            status_path,
            cancel_path,
            "DINOv3 inference {}".format(job_id),
            args.gpu_lock_timeout_seconds,
        )
        update_status(status_path, {"status": "running"})
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(dinov3_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=background_env(),
            )
            if gpu_lock is not None:
                gpu_lock.update_pid(proc.pid, kind="fewshot_infer", job_id=job_id)
            update_status(status_path, {"pid": proc.pid, "command": cmd, "log": str(log_path)})
            while proc.poll() is None:
                if cancel_path.is_file():
                    update_status(status_path, {"status": "cancelling", "pid": proc.pid})
                    terminate_process_tree(proc.pid)
                else:
                    update_status(status_path, {"status": "running", "pid": proc.pid})
                time.sleep(2.0)
    except ResourceLockCancelled:
        update_status(status_path, {"status": "cancelled", "error": "cancelled while waiting for GPU"})
        append_log(workspace, "Inference job {} cancelled while waiting for GPU.".format(job_id))
        return 130
    except ResourceLockTimeout as exc:
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Inference job {} could not acquire GPU: {}.".format(job_id, exc))
        return 75
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            terminate_process_tree(proc.pid)
        update_status(status_path, {"status": "failed", "error": str(exc)})
        append_log(workspace, "Inference job {} failed before completion: {}.".format(job_id, exc))
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.release()
    if cancel_path.is_file():
        update_status(status_path, {
            "status": "cancelled",
            "returncode": proc.returncode,
        })
        append_log(workspace, "Inference job {} cancelled.".format(job_id))
        return 130
    if proc.returncode != 0:
        update_status(status_path, {
            "status": "failed",
            "returncode": proc.returncode,
            "error": "inference process failed; see log",
        })
        return proc.returncode or 1
    update_status(status_path, {
        "status": "completed",
        "returncode": 0,
        "output_path": str(output_path),
        "image_path": str(image),
    })
    append_log(workspace, "Inference job {} completed. Output: {}".format(job_id, output_path))
    return 0


def cmd_list_models(args):
    workspace = workspace_for(Path(args.ts_root).resolve(), args.workspace)
    root = workspace / "models"
    rows = []
    if root.is_dir():
        if args.all:
            manifests = sorted(root.glob("*/*/manifest.json"))
        else:
            manifests = sorted(root.glob("*/latest.json"))
        for manifest_path in manifests:
            manifest = read_json(manifest_path, {})
            organ = manifest.get("organ", manifest_path.parent.name)
            if args.organ and safe_slug(organ) != safe_slug(args.organ):
                continue
            rows.append({
                "scope": "dataset",
                "organ": organ,
                "model_id": manifest.get("model_id", ""),
                "sample_count": manifest.get("sample_count", 0),
                "train_sample_count": manifest.get("train_sample_count", 0),
                "validation_sample_count": manifest.get("validation_sample_count", 0),
                "checkpoint": manifest.get("checkpoint", ""),
                "manifest_path": str(manifest_path),
                "created_at_epoch": manifest.get("created_at_epoch", 0.0),
            })
    if args.include_global:
        for row in global_model_rows(args.organ):
            global_row = dict(row)
            global_row["scope"] = "global"
            rows.append(global_row)
    rows.sort(key=lambda item: float(item.get("created_at_epoch", 0.0) or 0.0), reverse=True)
    print(json.dumps({"models": rows}, indent=2, sort_keys=True))
    return 0


def cmd_discover(args):
    cases = set(parse_case_list(args.cases) or [])
    samples, skipped = discover_samples(Path(args.ts_root).resolve(), args.organ, cases or None)
    selected = select_samples(samples, args.sample_mode, int(args.max_samples or 0))
    print(json.dumps({
        "organ": args.organ,
        "sample_count": len(samples),
        "selected_count": len(selected),
        "samples": selected,
        "skipped": skipped,
    }, indent=2, sort_keys=True))
    return 0


def cmd_cancel(args):
    workspace = workspace_for(Path(args.ts_root).resolve(), args.workspace)
    if args.job_id:
        status_path = workspace / "jobs" / (args.job_id + ".json")
        status = read_json(status_path, {}) or {}
        if not status:
            print("No job was found: {}".format(args.job_id), file=sys.stderr)
            return 2
    else:
        status_path, status = latest_running_job(workspace)
        if not status:
            print("No running few-shot job was found.")
            return 0
    cancel_path = status.get("cancel_path")
    if cancel_path:
        Path(cancel_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cancel_path).write_text("cancel requested at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")), encoding="utf-8")
    killed = []
    for key in ("pid", "controller_pid", "launcher_pid"):
        pid = status.get(key)
        if pid and terminate_process_tree(pid):
            killed.append(int(pid))
    update_status(status_path, {
        "status": "cancelled",
        "cancel_requested_at_epoch": time.time(),
        "cancelled_pids": killed,
    })
    print("Cancel request submitted for job {}.".format(status.get("job_id", status_path.stem)))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Export labels, build a few-shot dataset, and train a DINOv3 model")
    train.add_argument("--ts-root", required=True)
    train.add_argument("--organ", required=True)
    train.add_argument("--workspace")
    train.add_argument("--dinov3-root")
    train.add_argument("--python")
    train.add_argument("--base-config")
    train.add_argument("--cases")
    train.add_argument("--sample-mode", choices=("all", "latest"), default="all")
    train.add_argument("--min-samples", type=int, default=1)
    train.add_argument("--max-samples", type=int, default=0)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--img-size", default="224,224")
    train.add_argument("--modality", default="ct")
    train.add_argument("--val-fraction", type=float, default=0.0)
    train.add_argument("--val-cases")
    train.add_argument("--min-val-samples", type=int, default=1)
    train.add_argument("--finetune-method", choices=("frozen", "decoder_only", "decode_only", "lora", "adapter", "full"), default="lora")
    train.add_argument("--decoder", choices=("linear3d", "mlp_probe", "segformer3d", "dpt3d"), default="segformer3d")
    train.add_argument("--model-scale", choices=("vitb16", "vitl16", "vith16plus"), default="vitb16")
    train.add_argument("--model-path")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--adapter-bottleneck", type=int, default=64)
    train.add_argument("--mixed-precision", action="store_true")
    train.add_argument("--sub-volume", action="store_true")
    train.add_argument("--sub-volume-size", default="32,256,256")
    train.add_argument("--export-labels", action="store_true")
    train.add_argument("--mimics-exe")
    train.add_argument("--export-timeout-seconds", type=float, default=21600)
    train.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=21600)
    train.add_argument("--gpu-lock-timeout-seconds", type=float, default=86400)
    train.add_argument("--keep-last-checkpoints", type=int, default=2)
    train.add_argument("--keep-materialized-dataset", action="store_true")
    train.add_argument("--run-id")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run inference for one case with the latest or selected organ model")
    infer.add_argument("--ts-root", required=True)
    infer.add_argument("--case-id", required=True)
    infer.add_argument("--organ", required=True)
    infer.add_argument("--workspace")
    infer.add_argument("--dinov3-root")
    infer.add_argument("--python")
    infer.add_argument("--model-id", default="latest")
    infer.add_argument("--model-manifest")
    infer.add_argument("--gpu-lock-timeout-seconds", type=float, default=3600)
    infer.add_argument("--job-id")
    infer.set_defaults(func=cmd_infer)

    models = sub.add_parser("list-models", help="List latest registered models")
    models.add_argument("--ts-root", required=True)
    models.add_argument("--workspace")
    models.add_argument("--organ")
    models.add_argument("--all", action="store_true")
    models.add_argument("--include-global", action="store_true")
    models.set_defaults(func=cmd_list_models)

    discover = sub.add_parser("discover", help="Discover exported image/label pairs for one organ")
    discover.add_argument("--ts-root", required=True)
    discover.add_argument("--organ", required=True)
    discover.add_argument("--cases")
    discover.add_argument("--sample-mode", choices=("all", "latest"), default="all")
    discover.add_argument("--max-samples", type=int, default=0)
    discover.set_defaults(func=cmd_discover)

    cancel = sub.add_parser("cancel", help="Cancel the latest running job or a selected job")
    cancel.add_argument("--ts-root", required=True)
    cancel.add_argument("--workspace")
    cancel.add_argument("--job-id")
    cancel.set_defaults(func=cmd_cancel)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except Exception as exc:
        ts_root = getattr(args, "ts_root", None)
        if ts_root:
            try:
                append_log(workspace_for(ts_root, getattr(args, "workspace", None)), "Failed: {}".format(exc))
            except Exception:
                pass
        print("Error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
