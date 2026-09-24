#!/usr/bin/env python3
"""FlexiCT few-shot training pipeline for Mimics-Script (job-dir mode).

Job lifecycle mirrors the nnU-Net pipeline: a UI creates jobs/<job_id>/
(request.json + status.json + control.json) and spawns this module as
`flexict_pipeline.py run --job-dir <dir>`. The controller prepares the
nnU-Net dataset on the source-image grid (reusing the nnU-Net pipeline's
validated case materialization), then delegates the heavyweight stages to
tools/nnunet_stage_worker.py child processes with a FlexiCT worker
environment (nnUNet_extTrainer pointing at the flexict-finetune trainers,
FLEXICT2D/3D_CKPT, NUM_EPOCHS, MIRROR_DISABLE_AXES, nnUNet_compile=0).

The validated recipe is locked: lr, optimizer, gradclip, fp32, deep
supervision, oversampling, and batch size live in the standalone
integrations/flexict-finetune repo; this host-side adapter never overrides
them. See docs/validation.md there for the numbers this recipe produced.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from flexict_common import (  # noqa: E402
    CONFIGURATIONS,
    DEFAULT_CONFIG,
    MODEL_SCHEMA_VERSION,
    TERMINAL_STATES,
    load_config,
    load_pair,
    flexict_repo_dir,
    flexict_suggest_dataset_id,
    register_model as register_flexict_model,
    resolve_pretrained_dir,
    workspace_paths,
)
from nnunet_common import (  # noqa: E402
    append_log,
    compact_completed_log,
    safe_identifier,
    stable_digest,
    update_status,
    write_json_atomic,
)
from nnunet_pipeline import (  # noqa: E402
    _acquire_dataset_lock,
    _acquire_local_gpu,
    _assert_dataset_id_available,
    _case_directories,
    _dataset_name,
    _find_case_image,
    _materialize_nnunet_raw,
    _preprocess_cache_valid,
    _runtime_roots,
    _spawn_worker,
    build_training_data_profile,
    prepare_source_grid_cases,
    validate_materialized_source_geometry,
)
from resource_locks import process_start_marker  # noqa: E402


SCHEMA_VERSION = "flexict_job.v1"
TRAINERS = {
    "2d": "flexict2d_Trainer",
    "3d_fullres": "flexict3d_Trainer",
}
EPOCH_MATCH = re.compile(r"(?i)\bepoch\s*[:#]?\s*(\d+)\b")


def _cancelled(control_path: Path) -> bool:
    from nnunet_common import read_json

    control = read_json(control_path, {}) or {}
    return str(control.get("action") or "").lower() in {"cancel", "stop"}


def _raise_if_cancelled(control_path: Path) -> None:
    if _cancelled(control_path):
        raise InterruptedError("cancel")


def normalize_flexict_request(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and default a FlexiCT training request.

    FlexiCT trains one binary target per model. Cases and label export work
    exactly like the nnU-Net pipeline (single label spec, 0/1 output); the
    differences are the trainer set, the dataset band (750-799), and the
    locked recipe env vars.
    """
    from nnunet_common import normalize_request

    out = dict(values or {})
    operation = str(out.get("operation") or "train").lower()
    if operation not in {"train", "infer", "active_learning"}:
        raise ValueError("Unsupported FlexiCT operation: {}".format(operation))
    out["operation"] = operation
    # Reuse the nnU-Net request normalizer for the shared fields (workspace,
    # dataset_id, epochs, label export, GPU lock timeouts, ...). It only accepts
    # concrete nnU-Net configurations, so the FlexiCT value (auto/pair) is
    # carried separately and normalized after.
    if operation == "train":
        flexict_configuration = str(out.get("configuration") or "auto").strip().lower()
        nnunet_values = dict(out)
        nnunet_values["configuration"] = "2d"  # concrete placeholder for the normalizer
        if not nnunet_values.get("dataset_id"):
            # The nnU-Net normalizer defaults a missing dataset_id to 701
            # (the nnU-Net band) — suggest a free id in the FlexiCT 750-799
            # band instead so programmatic callers without an explicit id
            # never squat on the nnU-Net namespace.
            nnunet_values["dataset_id"] = flexict_suggest_dataset_id(
                out.get("workspace"))
        nnunet_values.setdefault("labels", [
            {"name": str(out.get("label_name") or "target"),
             "aliases": [str(out.get("label_name") or "target")],
             "id": 1, "source_mode": "alternatives"},
        ])
        nnunet_values.setdefault("trainer", "flexict2d_Trainer")
        nnunet_values.setdefault("modality", "CT")
        out = normalize_request(nnunet_values)
        if flexict_configuration not in CONFIGURATIONS:
            raise ValueError(
                "configuration must be one of {}".format(", ".join(CONFIGURATIONS)))
        out["configuration"] = flexict_configuration
    else:
        # infer / active_learning share the inference-side normalization
        # (workspace, GPU fields); the FlexiCT operation must survive it.
        out = normalize_request(dict(out, operation="infer"))
        out["operation"] = operation
    return out


def _flexict_configurations(request: dict[str, Any]) -> list[str]:
    """Resolve the 'configuration' request value into concrete nnU-Net configs.

    'auto' picks 2d on <16GB GPUs (validated: 3D fullres needs ~32GB),
    otherwise 'pair' (2d + 3d sequential, the active-learning contract).
    """
    value = str(request.get("configuration") or "auto").strip().lower()
    if value == "2d":
        return ["2d"]
    if value == "3d_fullres":
        return ["3d_fullres"]
    if value == "pair":
        return ["2d", "3d_fullres"]
    # auto
    from mimics_label_export import detect_gpu_memory_gb

    vram = detect_gpu_memory_gb()
    if vram and vram >= 16.0:
        return ["2d", "3d_fullres"]
    return ["2d"]


def flexict_worker_environment(request: dict[str, Any],
                               roots: dict[str, Path],
                               configuration: str) -> dict[str, str]:
    """Environment for a FlexiCT stage worker (documented trainer contract).

    The standalone repo's trainers/flexict_trainer.py reads FLEXICT_EXT_DIR,
    FLEXICT2D/3D_CKPT, NUM_EPOCHS, and MIRROR_DISABLE_AXES; nnU-Net discovers
    the trainer via nnUNet_extTrainer. Everything recipe-related stays there.
    """
    config = load_config()
    repo = flexict_repo_dir(config)
    # Remote jobs resolve the backbone weights inside the container
    # (/models/flexict, admin-installed and fingerprint-verified by the
    # remote controller) instead of this workstation's weight location.
    weights_override = str(request.get("flexict_pretrained_dir") or "").strip()
    if weights_override:
        # Container-side mount path (/models/flexict): normalize to POSIX so
        # the env var is valid on Linux even when built on Windows.
        weights = PurePosixPath(weights_override)
    else:
        weights, _source = resolve_pretrained_dir(config)
    environment = {
        "nnUNet_raw": str(roots["raw"]),
        "nnUNet_preprocessed": str(roots["preprocessed"]),
        "nnUNet_results": str(roots["results"]),
        "nnUNet_extTrainer": str(repo / "trainers"),
        "FLEXICT_EXT_DIR": str(repo / "flexict"),
        "FLEXICT2D_CKPT": str(weights / "flexict_2d" / "model.safetensors"),
        "FLEXICT3D_CKPT": str(weights / "flexict_3d" / "model.safetensors"),
        "NUM_EPOCHS": str(int(request.get("epochs") or DEFAULT_CONFIG["default_epochs"])),
        "nnUNet_compile": "0",
    }
    if os.name == "nt":
        # Windows: nnU-Net's spawn'd data-augmentation workers each import
        # torch/OpenBLAS (~1GB commit each, 12 by default) and have a history
        # of allocation-failure crashes on RAM-tight workstations that then
        # poison the main process's CUDA context ("CUDA error: unknown
        # error"). Single-process DA is slower but stable — the same fix the
        # MedDINOv3 experiments needed on this machine. Linux (remote
        # containers) keeps nnU-Net's multiprocessing default.
        environment["nnUNet_n_proc_DA"] = "0"
    mirror = str(request.get("mirror_disable_axes")
                 or DEFAULT_CONFIG["default_mirror_disable_axes"] or "").strip()
    if mirror:
        environment["MIRROR_DISABLE_AXES"] = mirror
    gpu_id = str(request.get("gpu_id") or "").strip()
    if gpu_id:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def flexict_infer_environment(request: dict[str, Any],
                              roots: dict[str, Path],
                              configuration: str) -> dict[str, str]:
    """Environment for the FlexiCT inference worker.

    Inference resolves the custom trainer class through the same
    nnUNet_extTrainer seam as training (nnUNetPredictor needs the trainer to
    rebuild the network). The pretrained backbone checkpoints are not needed:
    the trained model checkpoint already contains the fine-tuned weights.
    """
    config = load_config()
    repo = flexict_repo_dir(config)
    environment = {
        "nnUNet_raw": str(roots["raw"]),
        "nnUNet_preprocessed": str(roots["preprocessed"]),
        "nnUNet_results": str(roots["results"]),
        "nnUNet_extTrainer": str(repo / "trainers"),
        "nnUNet_compile": "0",
    }
    gpu_id = str(request.get("gpu_id") or "").strip()
    if gpu_id:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def _materialize_flexict_raw(rows: list[dict[str, Any]],
                             request: dict[str, Any],
                             roots: dict[str, Path]) -> tuple[Path, str]:
    """Materialize the FlexiCT nnU-Net raw dataset (single 0/1 label).

    Reuses the nnU-Net raw materializer (hardlink/copy cases, dataset.json
    with NibabelIOWithReorient, splits_final.json) — the FlexiCT differences
    are only the dataset band and the binary single-label contract, which the
    request's single label spec already encodes.
    """
    return _materialize_nnunet_raw(
        rows,
        # _materialize_nnunet_raw reads task_id/task_name/dataset_id/labels/
        # modality/image_reader_writer from the request; the FlexiCT request
        # carries compatible values after normalize_flexict_request.
        request,
        roots["raw"],
        roots["preprocessed"],
    )


def _split_train_val(rows: list[dict[str, Any]],
                     request: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Train/val split for FlexiCT few-shot: user-chosen val case count.

    Stratified by the target's Z-start when >= 8 cases (few-shot train sets
    span the cranio-caudal extent, mirroring the validated recipe's
    build_dataset.py), otherwise the split is a deterministic sort split.
    """
    case_ids = sorted(str(row["case_id"]) for row in rows)
    val_cases = int(request.get("val_cases") or DEFAULT_CONFIG["default_val_cases"])
    val_cases = max(1, min(val_cases, len(case_ids) - 1))
    # Explicit per-case selection from the training UI wins over the heuristic.
    explicit = [str(v) for v in (request.get("val_case_ids") or [])
                if str(v) in case_ids]
    if explicit:
        validation = sorted(set(explicit))
        return sorted(c for c in case_ids if c not in validation), validation
    if len(case_ids) < 8 or not rows:
        validation = sorted(case_ids[-val_cases:])
        return sorted(case_ids[:-val_cases]), validation
    import nibabel as nib
    import numpy as np

    def zstart(case: str) -> float:
        for row in rows:
            if str(row["case_id"]) == case:
                try:
                    label = nib.load(str(row["label"]))
                    arr = np.asanyarray(label.dataobj) > 0
                    if not arr.any():
                        return 0.5
                    img = nib.load(str(row["image"]))
                    spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
                    zax = int(np.argmax(spacing))
                    zidx = np.where(
                        arr.any(axis=tuple(i for i in range(3) if i != zax)))[0]
                    if len(zidx):
                        return float(zidx.min()) / float(arr.shape[zax])
                except Exception:
                    pass
                return 0.5
        return 0.5

    ranked = sorted(case_ids, key=zstart)
    spread = max(1, len(ranked) // val_cases)
    validation = sorted(ranked[::spread][:val_cases])
    if len(validation) < val_cases:
        for case in ranked:
            if case not in validation:
                validation.append(case)
            if len(validation) >= val_cases:
                break
        validation.sort()
    return sorted(c for c in case_ids if c not in validation), validation


def _write_flexict_splits(rows: list[dict[str, Any]],
                          request: dict[str, Any],
                          roots: dict[str, Path],
                          dataset_name: str) -> list[dict[str, list[str]]]:
    """Write a single-fold splits_final.json with the FlexiCT split."""
    train, validation = _split_train_val(rows, request)
    case_key_by_id = {str(row["case_id"]): safe_identifier(row["case_id"])
                      for row in rows}
    folds = [{
        "train": [case_key_by_id[value] for value in train],
        "val": [case_key_by_id[value] for value in validation],
    }]
    split_dir = roots["preprocessed"] / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(split_dir / "splits_final.json", folds)
    return folds


def _flexict_preprocess_cache_valid(
    preprocessed_root: Path, dataset_name: str, dataset_fingerprint: str,
    request: dict[str, Any], configuration: str,
) -> tuple[bool, str]:
    """Per-configuration preprocess cache check (pair mode has two identities).

    The nnU-Net pipeline's manifest stores one identity per dataset; pair mode
    trains two configurations from the same fingerprint, so FlexiCT keeps a
    per-configuration identity map in its own manifest file.
    """
    manifest_path = (preprocessed_root / dataset_name
                     / "flexict_preprocess_manifest.json")
    manifest = read_json_file(manifest_path, {}) or {}
    planning = {
        "dataset_fingerprint": dataset_fingerprint,
        "configuration": configuration,
        "spacing": request.get("spacing"),
        "patch_size": request.get("patch_size"),
        "batch_size": request.get("batch_size"),
        "plans": str(request.get("plans") or "nnUNetPlans"),
    }
    identity = stable_digest(planning)
    expected = preprocessed_root / dataset_name / configuration
    identities = manifest.get("identities") or {}
    return (
        str(identities.get(configuration) or "") == identity and expected.is_dir(),
        identity,
    )


def _record_flexict_preprocess(preprocessed_root: Path, dataset_name: str,
                               dataset_fingerprint: str, configuration: str,
                               identity: str) -> None:
    manifest_path = (preprocessed_root / dataset_name
                     / "flexict_preprocess_manifest.json")
    manifest = read_json_file(manifest_path, {}) or {}
    identities = dict(manifest.get("identities") or {})
    identities[configuration] = identity
    write_json_atomic(
        manifest_path,
        {
            "schema_version": "flexict_preprocess.v1",
            "identities": identities,
            "dataset_fingerprint": dataset_fingerprint,
            "updated_at_epoch": time.time(),
        },
    )


def _flexict_model_dir(roots: dict[str, Path], request: dict[str, Any],
                       configuration: str) -> Path:
    return roots["results"] / _dataset_name(request) / "{}__nnUNetPlans__{}".format(
        TRAINERS[configuration], configuration)


def _register_flexict_model(request: dict[str, Any],
                            roots: dict[str, Path],
                            configuration: str,
                            dataset_fingerprint: str,
                            pair_id: str,
                            training_data_profile: dict[str, Any]) -> dict[str, Any]:
    """Copy the trainer output into flexict_models/models/ and register it.

    Registration mirrors flexict_common.register_model: schema
    flexict_model.v1, model_dir + configuration + pair_id + dataset_id +
    label_name. A pair shares one pair_id so active learning can find the
    2D+3D pair with one lookup.
    """
    source_model = _flexict_model_dir(roots, request, configuration)
    if not source_model.is_dir():
        raise RuntimeError(
            "FlexiCT training did not produce a model folder: {}".format(source_model))
    fold_dir = source_model / "fold_0"
    for required in ("checkpoint_best.pth", "checkpoint_final.pth"):
        if not (fold_dir / required).is_file():
            raise RuntimeError(
                "FlexiCT training completed without {}: {}".format(required, fold_dir))
    model_id = "flexict_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    workspace = Path(request["workspace"]).expanduser().resolve()
    destination = workspace / "models" / safe_identifier(request["task_id"]) / model_id
    staging = destination.with_name(destination.name + ".publishing_" + uuid.uuid4().hex)
    shutil.rmtree(str(staging), ignore_errors=True)
    shutil.copytree(str(source_model), str(staging))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(str(destination), ignore_errors=False)
    os.replace(str(staging), str(destination))
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "task_id": request["task_id"],
        "task_name": request["task_name"],
        "label_name": str(request["label_name"]),
        "configuration": configuration,
        "trainer": TRAINERS[configuration],
        "dataset_id": int(request["dataset_id"]),
        "dataset_name": _dataset_name(request),
        "dataset_fingerprint": dataset_fingerprint,
        "pair_id": pair_id,
        "epochs": int(request["epochs"]),
        "mirror_disable_axes": str(request.get("mirror_disable_axes") or ""),
        "training_data_profile": training_data_profile,
        "model_dir": str(destination),
        "execution_backend": str(request.get("execution_backend") or "local"),
        "created_at_epoch": time.time(),
    }
    write_json_atomic(destination / "flexict_model_manifest.json", manifest)
    register_flexict_model(manifest, workspace=str(workspace))
    return manifest


def _parse_training_progress(log_path: Path, request: dict[str, Any],
                             status_path: Path, last_epoch: int) -> int:
    """Parse the current epoch from the trainer log tail (job.log) 128KB window.

    Kept for tests and future worker variants; the live poll loop inside
    nnunet_pipeline._spawn_worker performs the same parse inline.
    """
    try:
        with log_path.open("rb") as reader:
            reader.seek(0, os.SEEK_END)
            reader.seek(max(0, reader.tell() - 131072), os.SEEK_SET)
            tail = reader.read().decode("utf-8", "replace")
        matches = EPOCH_MATCH.findall(tail)
        if not matches:
            return last_epoch
        epoch = max(int(value) for value in matches)
        if epoch == last_epoch:
            return last_epoch
        total = int(request.get("epochs") or DEFAULT_CONFIG["default_epochs"])
        completed = min(total, epoch + 1)
        update_status(
            status_path,
            status="training",
            phase="training",
            message="FlexiCT training epoch {} of {}.".format(completed, total),
            current_epoch=completed,
            total_epochs=total,
            progress_percent=min(95, 50 + int(45 * completed / max(1, total))),
        )
        return epoch
    except OSError:
        return last_epoch


def _sweep_expired_jobs(workspace: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prune terminal job dirs older than job_retention_days (keep status.json).

    Mirrors the nnInteractive pipeline's sweep: retention 0/absent disables;
    completed_at_epoch (fallback: status.json mtime) decides; status.json and
    the registry survive so history stays browsable.
    """
    cfg = config or load_config()
    try:
        retention_days = float(cfg.get("job_retention_days") or 0)
    except (TypeError, ValueError):
        retention_days = 0.0
    if retention_days <= 0:
        return {"removed_jobs": [], "kept_jobs": 0, "retention_days": 0.0}
    paths = workspace_paths(workspace)
    cutoff = time.time() - retention_days * 86400
    removed = []
    kept = 0
    jobs_dir = paths["jobs"]
    if not jobs_dir.is_dir():
        return {"removed_jobs": [], "kept_jobs": 0, "retention_days": retention_days}
    from nnunet_common import read_json

    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        if not status_path.is_file():
            continue
        status = read_json(status_path, {}) or {}
        if str(status.get("status") or "").lower() not in TERMINAL_STATES:
            continue
        try:
            finished = float(status.get("completed_at_epoch") or 0)
        except (TypeError, ValueError):
            finished = 0.0
        if finished <= 0:
            try:
                finished = status_path.stat().st_mtime
            except OSError:
                continue
        if finished >= cutoff:
            kept += 1
            continue
        for child in sorted(job_dir.iterdir()):
            if child.name == "status.json":
                continue
            if child.is_dir():
                shutil.rmtree(str(child), ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        removed.append(job_dir.name)
    return {"removed_jobs": removed, "kept_jobs": kept, "retention_days": retention_days}


def _spawn_flexict_worker(stage: str,
                          params: dict[str, Any],
                          request: dict[str, Any],
                          roots: dict[str, Path],
                          configuration: str,
                          job_dir: Path,
                          status_path: Path,
                          control_path: Path,
                          log_path: Path,
                          resource_lock=None) -> dict[str, Any]:
    """Spawn a stage worker with the FlexiCT environment.

    Delegates to nnunet_pipeline._spawn_worker, which writes <stage>_spec.json
    (stage/params/environment/start_gate/control_path), Popen's
    tools/nnunet_stage_worker.py, transfers the GPU lock to the worker, and
    parses training progress from the log tail. Two FlexiCT seams: the worker
    environment (env vars only — the integration boundary), and registering
    the FlexiCT trainers as epoch-honoring so progress parsing knows the
    total (NUM_EPOCHS drives them inside the trainer).
    """
    import nnunet_pipeline as np_mod

    original_env = np_mod._worker_environment
    original_known = set(np_mod.KNOWN_EPOCH_TRAINERS)
    try:
        np_mod._worker_environment = lambda _req, _roots: flexict_worker_environment(
            request, roots, configuration)
        np_mod.KNOWN_EPOCH_TRAINERS = original_known | set(TRAINERS.values())
        return _spawn_worker(
            stage, params, request, roots, job_dir, status_path,
            control_path, log_path, resource_lock=resource_lock)
    finally:
        np_mod._worker_environment = original_env
        np_mod.KNOWN_EPOCH_TRAINERS = original_known


def run_training(job_dir: Path) -> int:
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = normalize_flexict_request(read_json_file(request_path))
    write_json_atomic(request_path, request)
    roots = _runtime_roots(request)
    workspace = Path(request["workspace"]).expanduser().resolve()
    staging = job_dir / "prepared"
    shutil.rmtree(str(staging), ignore_errors=True)
    staging.mkdir(parents=True)
    update_status(
        status_path,
        schema_version=SCHEMA_VERSION,
        job_id=request["job_id"],
        kind="train",
        task_id=request["task_id"],
        task_name=request["task_name"],
        status="preparing_data",
        phase="discovering_cases",
        message="Discovering FlexiCT training cases.",
        log_path=str(log_path),
        control_path=str(control_path),
        controller_pid=os.getpid(),
        controller_start_marker=process_start_marker(os.getpid()),
        progress_percent=1,
    )
    try:
        configurations = _flexict_configurations(request)
        update_status(status_path, configurations=configurations)
        # prepare_source_grid_cases re-runs the nnU-Net request normalizer,
        # which only accepts concrete configurations; hand it a concrete copy.
        prepare_request = dict(request, configuration=configurations[0])
        rows = prepare_source_grid_cases(
            prepare_request, staging, status_path, control_path)
        training_data_profile = build_training_data_profile(
            rows, request.get("modality") or "")
        _raise_if_cancelled(control_path)
        dataset_lock = _acquire_dataset_lock(
            request, status_path, control_path, log_path)
        try:
            _assert_dataset_id_available(roots, request)
            dataset_dir, dataset_fingerprint = _materialize_flexict_raw(
                rows, request, roots)
            dataset_name = dataset_dir.name
            _write_flexict_splits(rows, request, roots, dataset_name)
            # Pair/auto preprocessing must cover every configuration that will
            # be trained; check the cache per configuration.
            pending: list[tuple[str, str]] = []
            for configuration in configurations:
                cache_valid, identity = _flexict_preprocess_cache_valid(
                    roots["preprocessed"], dataset_name, dataset_fingerprint,
                    request, configuration)
                if cache_valid and not bool(request.get("force_preprocess")):
                    continue
                pending.append((configuration, identity))
            if not pending:
                update_status(
                    status_path,
                    status="preprocessing",
                    phase="reusing_preprocessed_data",
                    message="Verified preprocessed FlexiCT data reused.",
                    progress_percent=45,
                    preprocess_cache_reused=True,
                )
            else:
                split_dir = roots["preprocessed"] / dataset_name
                split_dir.mkdir(parents=True, exist_ok=True)
                for configuration, identity in pending:
                    # Clear stale generated arrays for THIS configuration only —
                    # pair mode must keep the sibling configuration's results.
                    shutil.rmtree(str(split_dir / configuration), ignore_errors=True)
                    update_status(
                        status_path,
                        status="preprocessing",
                        phase="planning_and_preprocessing",
                        message="Fingerprinting, planning, and preprocessing the FlexiCT dataset ({}).".format(configuration),
                        progress_percent=30,
                    )
                    _spawn_flexict_worker(
                        "preprocess",
                        {
                            "dataset_id": int(request["dataset_id"]),
                            "configuration": configuration,
                            "num_processes": int(request["preprocess_workers"]),
                            "target_spacing": request.get("spacing"),
                            "target_patch_size": request.get("patch_size"),
                            "target_batch_size": request.get("batch_size"),
                            "verify_integrity": bool(request.get("verify_integrity", True)),
                        },
                        request, roots, configuration, job_dir, status_path,
                        control_path, log_path)
                    _record_flexict_preprocess(
                        roots["preprocessed"], dataset_name, dataset_fingerprint,
                        configuration, identity)
                # splits_final.json lives under the dataset dir; re-assert it
                # after preprocessing (nnU-Net never overwrites an existing one).
                write_json_atomic(
                    split_dir / "splits_final.json",
                    read_json_file(split_dir / "splits_final.json") or [])
        finally:
            dataset_lock.release()
        _raise_if_cancelled(control_path)
        pair_id = "flexpair_{}".format(uuid.uuid4().hex[:12])
        models = []
        for index, configuration in enumerate(configurations):
            label = "{}{}".format(
                request["task_name"],
                "" if len(configurations) == 1 else " ({})".format(
                    "2D" if configuration == "2d" else "3D"))
            if configuration not in ("2d", "3d_fullres"):
                raise RuntimeError(
                    "FlexiCT supports 2d and 3d_fullres, not '{}'.".format(configuration))
            update_status(
                status_path,
                status="waiting_for_gpu" if index else "training",
                phase="training_{}".format(configuration),
                message="FlexiCT {} training is running.".format(label),
                progress_percent=50,
                active_configuration=configuration,
                training_curve_path=str(
                    _flexict_model_dir(roots, request, configuration)
                    / "fold_0" / "progress.png"),
            )
            gpu_lock = _acquire_local_gpu(
                request, status_path, control_path, log_path,
                "flexict training {}".format(configuration))
            try:
                _spawn_flexict_worker(
                    "train",
                    {
                        "dataset_id": int(request["dataset_id"]),
                        "configuration": configuration,
                        "fold": 0,
                        "trainer": TRAINERS[configuration],
                        "plans": "nnUNetPlans",
                        "pretrained_weights": None,
                        "num_gpus": int(request["num_gpus"]),
                        "continue_training": False,
                        "only_run_validation": False,
                        "export_validation_probabilities": False,
                        "disable_checkpointing": False,
                        "val_with_best": False,
                        "gpu_id": None,
                        "epochs": None,
                    },
                    request, roots, configuration, job_dir, status_path,
                    control_path, log_path, resource_lock=gpu_lock)
            finally:
                if gpu_lock is not None:
                    gpu_lock.release()
            update_status(
                status_path,
                status="registering",
                phase="registering_model",
                message="Registering the trained FlexiCT model ({}).".format(label),
                progress_percent=96,
            )
            manifest = _register_flexict_model(
                request, roots, configuration, dataset_fingerprint,
                pair_id, training_data_profile)
            models.append(manifest)
        update_status(
            status_path,
            status="completed",
            phase="completed",
            message="FlexiCT training completed ({} model{}).".format(
                len(models), "s" if len(models) != 1 else ""),
            progress_percent=100,
            models=models,
            model=models[0] if len(models) == 1 else None,
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except InterruptedError:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="FlexiCT training was cancelled.",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except Exception as exc:
        append_log(log_path, traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 1
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)
        try:
            result = compact_completed_log(log_path)
            if result.get("compacted"):
                update_status(status_path, log_compaction=result)
        except OSError:
            pass
        _sweep_expired_jobs(workspace)


def read_json_file(path: Path, default: Any = None) -> Any:
    from nnunet_common import read_json

    return read_json(path, default)


def scan_flexict_cases(dataset_root: str | Path, label_name: str) -> list[dict[str, Any]]:
    """Scan a dataset root and report per-case image/label availability.

    Used by the training UI so the user sees exactly which cases will train,
    which will validate, and which are skipped (missing image or label).
    """
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise ValueError("Dataset folder does not exist: {}".format(root))
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image = None
        for candidate in case_dir.iterdir():
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in {".nii", ".gz", ".nrrd", ".mha"}:
                image = candidate
                break
        label = None
        if label_name:
            for candidate in case_dir.iterdir():
                if not candidate.is_file():
                    continue
                if candidate.stem.split(".")[0].lower() == safe_identifier(label_name).lower():
                    label = candidate
                    break
        if image is None and label is None:
            continue
        status = []
        if image is None:
            status.append("image missing")
        if label is None:
            status.append("label missing")
        rows.append(
            {
                "case_id": case_dir.name,
                "image": str(image) if image else "",
                "label": str(label) if label else "",
                "checked": image is not None and label is not None,
                "use_train": True,
                "use_val": False,
                "status": "OK" if not status else "; ".join(status),
            }
        )
    return rows


def run_inference(job_dir: Path) -> int:
    """FlexiCT inference on the active case (stages preparing_input ->
    waiting_for_gpu -> predicting -> completed).

    Mirrors nnunet_pipeline.run_inference: materialize the source image on
    its own grid, validate against the launch-time geometry, then spawn the
    infer worker with the FlexiCT trainer seam. The FlexiCT differences:
    the model comes from the flexict registry (model_manifest), inference
    uses checkpoint_best.pth (validation-Dice-selected) with TTA disabled
    per the locked recipe, and the trainer resolves through nnUNet_extTrainer.
    """
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = normalize_flexict_request(read_json_file(request_path))
    model_manifest_path = Path(
        str(request.get("model_manifest") or "")).expanduser().resolve()
    manifest = read_json_file(model_manifest_path, {}) or {}
    model_dir = Path(
        str(request.get("model_dir_override")
            or manifest.get("model_dir")
            or model_manifest_path.parent)
    ).expanduser().resolve()
    image_path = Path(str(request.get("image_path") or "")).expanduser().resolve()
    output_path = Path(
        str(request.get("output_path") or job_dir / "prediction.nii.gz")).resolve()
    roots = _runtime_roots(request)
    update_status(
        status_path,
        schema_version=SCHEMA_VERSION,
        job_id=request["job_id"],
        kind="infer",
        task_id=manifest.get("task_id") or request.get("task_id") or "",
        task_name=manifest.get("task_name") or request.get("task_name") or "",
        label_name=manifest.get("label_name") or request.get("label_name") or "",
        status="running",
        phase="preparing_input",
        message="Preparing the source image for FlexiCT prediction.",
        model_id=manifest.get("model_id"),
        model_manifest=str(model_manifest_path),
        image_path=str(image_path),
        output_path=str(output_path),
        log_path=str(log_path),
        control_path=str(control_path),
        controller_pid=os.getpid(),
        controller_start_marker=process_start_marker(os.getpid()),
        progress_percent=5,
    )
    try:
        if not manifest or not model_dir.is_dir():
            raise RuntimeError("Selected FlexiCT model is missing or invalid.")
        if not image_path.exists():
            raise RuntimeError(
                "Prediction image does not exist: {}".format(image_path))
        from tools.mimics_label_export import materialize_source_image as _materialize_source_image

        inference_input = job_dir / "input" / "source_image.nii.gz"
        inference_input.parent.mkdir(parents=True, exist_ok=True)
        _materialize_source_image(image_path, inference_input)
        validate_materialized_source_geometry(
            inference_input, request.get("source_geometry_expected")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        update_status(
            status_path,
            status="running",
            phase="waiting_for_gpu",
            message="Waiting for the shared GPU before FlexiCT prediction.",
            progress_percent=20,
        )
        gpu_lock = _acquire_local_gpu(
            request, status_path, control_path, log_path, "flexict inference")
        try:
            configuration = str(manifest.get("configuration") or "2d")
            import nnunet_pipeline as np_mod

            original_env = np_mod._worker_environment
            try:
                np_mod._worker_environment = lambda _req, _roots: flexict_infer_environment(
                    request, roots, configuration)
                _spawn_worker(
                    "infer",
                    {
                        "model_folder": str(model_dir),
                        "input_path": str(inference_input),
                        "output_path": str(output_path),
                        "disable_tta": True,  # locked recipe: no TTA
                        "use_cpu": bool(request.get("use_cpu", False)),
                        "enable_stats": False,
                        "gpu_device_id": 0,
                        "num_processes_preprocessing": int(
                            request.get("inference_workers") or 2),
                        "num_processes_segmentation_export": int(
                            request.get("inference_workers") or 2),
                        "checkpoint_name": "checkpoint_best.pth",
                    },
                    request, roots, job_dir, status_path,
                    control_path, log_path, resource_lock=gpu_lock)
            finally:
                np_mod._worker_environment = original_env
        finally:
            if gpu_lock is not None:
                gpu_lock.release()
        if not output_path.is_file():
            raise RuntimeError(
                "FlexiCT inference produced no output: {}".format(output_path))
        import nibabel as nib
        import numpy as np

        source = nib.load(str(inference_input))
        prediction = nib.load(str(output_path))
        if tuple(source.shape[:3]) != tuple(prediction.shape[:3]) or not np.allclose(
            source.affine, prediction.affine, atol=1e-4, rtol=0.0
        ):
            raise RuntimeError(
                "Prediction grid does not match the source image. The result was not offered to Mimics.")
        update_status(
            status_path,
            status="completed",
            phase="completed",
            message="FlexiCT inference completed and passed spatial validation.",
            progress_percent=100,
            label_name=manifest.get("label_name") or request.get("label_name") or "",
            inference_image_path=str(inference_input),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except InterruptedError:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="FlexiCT inference was cancelled.",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except Exception as exc:
        append_log(log_path, traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 1
    finally:
        try:
            result = compact_completed_log(log_path)
            if result.get("compacted"):
                update_status(status_path, log_compaction=result)
        except OSError:
            pass
        workspace = str(request.get("workspace") or "")
        if workspace:
            _sweep_expired_jobs(workspace)


def _resolve_flexict_pair(request: dict[str, Any],
                          workspace: Path) -> dict[str, dict[str, Any]]:
    """Find the 2D + 3D model pair an active-learning run needs.

    Order: explicit request manifests (model_manifest_2d/3d) > the newest
    usable pair in the registry (flexict_common.load_pair). Both ends must
    be usable; there is no silent single-model fallback — disagreement
    between the 2D and 3D predictions IS the uncertainty signal.
    """
    manifests = {}
    for key, configuration in (("model_manifest_2d", "2d"),
                               ("model_manifest_3d", "3d_fullres")):
        path = str(request.get(key) or "").strip()
        if path:
            manifest = read_json_file(Path(path).expanduser().resolve(), {}) or {}
            if str(manifest.get("configuration") or "") != configuration:
                raise ValueError(
                    "{} must reference a {} model.".format(key, configuration))
            manifests[configuration] = manifest
    if len(manifests) == 1:
        raise ValueError(
            "Active learning needs both a 2D and a 3D model; only one "
            "manifest was provided.")
    if manifests:
        return manifests
    model_2d, model_3d = load_pair(str(workspace))
    if model_2d is None or model_3d is None:
        raise RuntimeError(
            "No usable FlexiCT model pair was found. Train with "
            "configuration 'pair' (or 'auto' on a >=16GB GPU) first — "
            "active learning ranks cases by 2D-vs-3D disagreement.")
    return {"2d": model_2d, "3d_fullres": model_3d}


def _al_materialize_inputs(job_dir: Path, request: dict[str, Any],
                           status_path: Path,
                           control_path: Path) -> list[str]:
    """Materialize the unlabeled pool as flat <case>.nii.gz inputs."""
    from tools.mimics_label_export import materialize_source_image as _materialize

    dataset_root = Path(
        str(request.get("dataset_root") or "")).expanduser().resolve()
    requested = set(
        str(case).strip() for case in (request.get("cases") or [])
        if str(case).strip())
    case_dirs = _case_directories(dataset_root, requested or None)
    if not case_dirs:
        raise RuntimeError(
            "No image cases were found in {}.".format(dataset_root))
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    case_ids = []
    geometries: dict[str, dict[str, Any]] = {}
    total = len(case_dirs)
    for index, case_dir in enumerate(case_dirs, start=1):
        _raise_if_cancelled(control_path)
        image = _find_case_image(case_dir)
        if image is None:
            continue
        case_id = safe_identifier(case_dir.name)
        destination = input_dir / "{}.nii.gz".format(case_id)
        _materialize(image, destination)
        case_ids.append(case_id)
        try:
            import nibabel as nib

            header = nib.load(str(destination))
            geometries[case_id] = {
                "source_shape": [int(v) for v in header.shape[:3]],
                "source_voxel_to_ras_matrix": header.affine.astype(
                    float).tolist(),
            }
        except Exception:
            geometries[case_id] = {}
        update_status(
            status_path,
            phase="preparing_inputs",
            current_case=case_id,
            preparation_index=index,
            preparation_total=total,
            prepared_cases=len(case_ids),
            progress_percent=5 + int(10 * index / max(1, total)),
        )
    if not case_ids:
        raise RuntimeError(
            "No readable images were found in {}.".format(dataset_root))
    # Per-case source geometry: the Mimics-side overlay verification compares
    # the active image against this before applying any band mask.
    write_json_atomic(
        job_dir / "input_geometries.json",
        {"schema_version": "flexict_al_geometries.v1", "cases": geometries},
    )
    return sorted(case_ids)


def _al_predict_configuration(job_dir: Path,
                              request: dict[str, Any],
                              roots: dict[str, Path],
                              manifest: dict[str, Any],
                              configuration: str,
                              status_path: Path,
                              control_path: Path,
                              log_path: Path,
                              gpu_lock) -> Path:
    """Run one pair end over the materialized pool (folder-mode easy_predict)."""
    model_dir = Path(str(manifest.get("model_dir") or "")).expanduser().resolve()
    if not model_dir.is_dir():
        raise RuntimeError(
            "FlexiCT {} model folder is missing: {}".format(
                configuration, model_dir))
    predictions = job_dir / "predictions_{}".format(configuration)
    predictions.mkdir(parents=True, exist_ok=True)
    update_status(
        status_path,
        status="batch_predicting_2d" if configuration == "2d"
        else "batch_predicting_3d",
        phase="batch_predicting_{}".format(configuration),
        message="FlexiCT {} batch prediction is running.".format(
            "2D" if configuration == "2d" else "3D"),
        progress_percent=15 if configuration == "2d" else 45,
        active_configuration=configuration,
    )
    import nnunet_pipeline as np_mod

    original_env = np_mod._worker_environment
    try:
        np_mod._worker_environment = lambda _req, _roots: flexict_infer_environment(
            request, roots, configuration)
        _spawn_worker(
            "infer",
            {
                "model_folder": str(model_dir),
                "input_path": str(job_dir / "input"),
                "output_path": str(predictions),
                "disable_tta": True,  # locked recipe: no TTA
                "use_cpu": bool(request.get("use_cpu", False)),
                "enable_stats": False,
                "gpu_device_id": 0,
                "num_processes_preprocessing": int(
                    request.get("inference_workers") or 2),
                "num_processes_segmentation_export": int(
                    request.get("inference_workers") or 2),
                "checkpoint_name": "checkpoint_best.pth",
            },
            request, roots, job_dir, status_path,
            control_path, log_path, resource_lock=gpu_lock)
    finally:
        np_mod._worker_environment = original_env
    return predictions


def _al_write_uncertainty_bands(uncertainty_dir: Path,
                                case_ids: list[str],
                                method: str,
                                moderate: int,
                                high: int) -> dict[str, Any]:
    """Derive the two overlay band masks (moderate >= t1, high >= t2) from the
    uint8 x10 uncertainty maps. The bands stay on the source grid so the
    Mimics bridge can resample them like any prediction mask.

    The thresholds are data-aware: a two-model disagreement pair can only
    produce the level 5 (1 - 1/2 scaled by 10), so a high band at the
    default 6 would never fire. Both thresholds are clamped to the levels
    actually present in the maps."""
    import nibabel as nib
    import numpy as np

    bands_dir = uncertainty_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Any] = {}
    max_level = 0
    for case_id in case_ids:
        source = uncertainty_dir / "{}_uncertainty_{}.nii.gz".format(
            case_id, method)
        if not source.is_file():
            continue
        image = nib.load(str(source))
        values = np.asanyarray(image.dataobj)
        images[case_id] = (image, values)
        max_level = max(max_level, int(values.max()) if values.size else 0)
    if high > max_level:
        high = max_level
    if moderate >= high:
        moderate = max(1, high - 1)
    written = {"moderate": 0, "high": 0}
    for case_id, (image, values) in images.items():
        for level, threshold in (("moderate", moderate), ("high", high)):
            if threshold < 1:
                continue
            band = (values >= threshold).astype(np.uint8)
            if not band.any():
                continue
            nib.save(
                nib.Nifti1Image(band, image.affine, image.header),
                str(bands_dir / "{}_{}.nii.gz".format(case_id, level)))
            written[level] += 1
    return {"dir": str(bands_dir), "moderate_threshold": moderate,
            "high_threshold": high, "written": written}


def run_active_learning(job_dir: Path) -> int:
    """Active-learning cycle: batch-predict the unlabeled pool with the 2D and
    3D pair ends, rank cases by inter-model disagreement, and write overlay
    band masks for the review UI (stages preparing_inputs ->
    batch_predicting_2d -> batch_predicting_3d -> computing_uncertainty ->
    completed)."""
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    log_path = job_dir / "job.log"
    request = normalize_flexict_request(read_json_file(request_path))
    write_json_atomic(request_path, request)
    workspace = Path(request["workspace"]).expanduser().resolve()
    roots = _runtime_roots(request)
    method = str(request.get("uncertainty_method")
                 or DEFAULT_CONFIG["default_uncertainty_method"]).strip()
    moderate = int(request.get("uncertainty_moderate_threshold") or 3)
    high = int(request.get("uncertainty_high_threshold") or 6)
    update_status(
        status_path,
        schema_version=SCHEMA_VERSION,
        job_id=request["job_id"],
        kind="active_learning",
        task_id=request.get("task_id") or "",
        task_name=request.get("task_name") or "",
        label_name=request.get("label_name") or "",
        status="running",
        phase="preparing_inputs",
        message="Preparing the unlabeled pool for FlexiCT active learning.",
        log_path=str(log_path),
        control_path=str(control_path),
        controller_pid=os.getpid(),
        controller_start_marker=process_start_marker(os.getpid()),
        progress_percent=5,
    )
    gpu_lock = None
    try:
        pair = _resolve_flexict_pair(request, workspace)
        case_ids = _al_materialize_inputs(
            job_dir, request, status_path, control_path)
        update_status(status_path, case_count=len(case_ids), cases=case_ids)
        gpu_lock = _acquire_local_gpu(
            request, status_path, control_path, log_path, "flexict active learning")
        predictions = {}
        for configuration, manifest in pair.items():
            _raise_if_cancelled(control_path)
            predictions[configuration] = _al_predict_configuration(
                job_dir, request, roots, manifest, configuration,
                status_path, control_path, log_path, gpu_lock)
        if gpu_lock is not None:
            gpu_lock.release()
            gpu_lock = None
        update_status(
            status_path,
            status="computing_uncertainty",
            phase="computing_uncertainty",
            message="Ranking cases by 2D-vs-3D disagreement.",
            progress_percent=75,
        )
        _raise_if_cancelled(control_path)
        uncertainty_dir = job_dir / "uncertainty"
        # The one sanctioned sys.path seam into the standalone repo.
        repo = flexict_repo_dir(load_config())
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from uncertainty import analyze_uncertainty_dir, run_batch  # noqa: E402

        run_batch(
            mask_dirs=[str(predictions["2d"]), str(predictions["3d_fullres"])],
            out_dir=str(uncertainty_dir),
            case_names=case_ids,
            filename_template="{case}.nii.gz",
            method=method,
            min_masks=2,
            target_labels=[1],
            save_components=False,
            save_consensus=True,
            save_plots=False,
            log_path=str(job_dir / "uncertainty_log.txt"),
        )
        ranking = analyze_uncertainty_dir(
            str(uncertainty_dir),
            method=method,
            sort_key="integrated",
            report_csv=str(uncertainty_dir / "ranking.csv"),
            compute_boundary=False,
            compute_cc=False,
        )
        bands = _al_write_uncertainty_bands(
            uncertainty_dir, case_ids, method, moderate, high)
        summary = [
            {
                "case": str(row.get("case") or ""),
                "integrated": float(row.get("integrated") or 0.0),
                "uncertain_vol": float(row.get("uncertain_vol") or 0.0),
                "max": float(row.get("max") or 0.0),
            }
            for row in ranking
        ]
        update_status(
            status_path,
            status="completed",
            phase="completed",
            message="FlexiCT active learning ranked {} case(s).".format(
                len(summary)),
            progress_percent=100,
            label_name=request.get("label_name") or "",
            uncertainty_method=method,
            uncertainty_dir=str(uncertainty_dir),
            ranking_csv=str(uncertainty_dir / "ranking.csv"),
            uncertainty_bands=bands,
            ranking=summary,
            top_case=summary[0]["case"] if summary else "",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except InterruptedError:
        update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="FlexiCT active learning was cancelled.",
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 0
    except Exception as exc:
        append_log(log_path, traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
            completed_at_epoch=time.time(),
            worker_pid=None,
            worker_start_marker=None,
        )
        return 1
    finally:
        if gpu_lock is not None:
            try:
                gpu_lock.release()
            except Exception:
                pass
        try:
            result = compact_completed_log(log_path)
            if result.get("compacted"):
                update_status(status_path, log_compaction=result)
        except OSError:
            pass
        _sweep_expired_jobs(workspace)


def create_flexict_job(values: dict[str, Any]) -> dict[str, Any]:
    """Create a FlexiCT job folder and launch its controller.

    Mirrors nnunet_jobs.create_job but keeps the FlexiCT workspace separate
    (flexict_models/). Remote execution launches remote_training_controller
    through a remote spec, exactly like the nnU-Net job path.
    """
    request = normalize_flexict_request(values)
    job_id = "{}_{}_{}".format(
        str(request.get("operation") or "train"),
        time.strftime("%Y%m%dT%H%M%S"),
        uuid.uuid4().hex[:8],
    )
    request["job_id"] = job_id
    paths = workspace_paths(request.get("workspace"))
    job_dir = paths["jobs"] / safe_identifier(job_id)
    if job_dir.exists():
        raise RuntimeError("FlexiCT job already exists: {}".format(job_dir))
    job_dir.mkdir(parents=True)
    if request["operation"] == "infer" and not str(
        request.get("output_path") or ""
    ).strip():
        request["output_path"] = str(job_dir / "prediction.nii.gz")
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    control_path = job_dir / "control.json"
    write_json_atomic(request_path, request)
    write_json_atomic(control_path, {"action": "run", "updated_at_epoch": time.time()})
    status = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "kind": request["operation"],
        "task_id": request.get("task_id", ""),
        "task_name": request.get("task_name", ""),
        "status": "launching",
        "phase": "launching",
        "message": "Starting the FlexiCT background task.",
        "workspace": str(paths["root"]),
        "job_dir": str(job_dir),
        "request_path": str(request_path),
        "control_path": str(control_path),
        "log_path": str(job_dir / "job.log"),
        "execution_backend": str(request.get("execution_backend") or "local"),
        "remote_profile_id": str(request.get("remote_profile_id") or ""),
        "created_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "progress_percent": 0,
    }
    write_json_atomic(status_path, status)
    if status["execution_backend"] == "remote":
        if not status["remote_profile_id"]:
            raise RuntimeError("Remote training server was not selected.")
        spec_path = job_dir / "remote_spec.json"
        write_json_atomic(
            spec_path,
            {
                "schema_version": "mimics_remote_flexict_spec.v1",
                "kind": (
                    "flexict_infer"
                    if request["operation"] == "infer"
                    else "flexict"
                ),
                "job_id": job_id,
                "job_dir": str(job_dir),
                "status_path": str(status_path),
                "request_path": str(request_path),
                "remote_profile_id": status["remote_profile_id"],
            },
        )
        command = [
            sys.executable,
            str(ROOT / "tools" / "remote_training_controller.py"),
            "run",
            "--spec",
            str(spec_path),
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "tools" / "flexict_pipeline.py"),
            "run",
            "--job-dir",
            str(job_dir),
        ]
    from nnunet_jobs import hidden_process_kwargs

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **hidden_process_kwargs(),
    )
    update_status(
        status_path,
        launcher_pid=process.pid,
        launcher_start_marker=process_start_marker(process.pid),
        message="FlexiCT task started in an external process.",
    )
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "status_path": str(status_path),
        "control_path": str(control_path),
        "launcher_pid": process.pid,
    }


def list_flexict_jobs(workspace: str | Path | None = None) -> list[dict[str, Any]]:
    """List FlexiCT jobs with the same reconcile semantics as nnunet_jobs."""
    from nnunet_jobs import reconcile_job_status

    jobs_dir = workspace_paths(workspace)["jobs"]
    rows = []
    if not jobs_dir.is_dir():
        return rows
    for status_path in jobs_dir.glob("*/status.json"):
        status = reconcile_job_status(status_path)
        if status:
            status["status_path"] = str(status_path)
            rows.append(status)
    rows.sort(
        key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    job_dir = Path(args.job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    operation = str(
        (read_json_file(job_dir / "request.json") or {}).get("operation")
        or "train").lower()
    if operation == "infer":
        return run_inference(job_dir)
    if operation == "active_learning":
        return run_active_learning(job_dir)
    return run_training(job_dir)


if __name__ == "__main__":
    raise SystemExit(main())
