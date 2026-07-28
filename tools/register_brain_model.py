#!/usr/bin/env python3
"""Register the validated brain CLoPA-IN weights as an immutable task model.

The legacy v1 script edited checkpoint dictionary keys directly. nnInteractive
contains shared state-dict aliases, so later aliases could overwrite those
edits while loading and silently reconstruct the official network. This script
updates the real network parameters and uses the production exporter, which
reloads the result and verifies the effective parameter fingerprint.
"""

from __future__ import annotations

import json
import gc
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNTIME = ROOT / "runtime_py35"
FINETUNE_SRC = ROOT / "external" / "nninteractive-finetune" / "src"
for value in (ROOT, TOOLS, RUNTIME, FINETUNE_SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from nninteractive_finetune.checkpoint import export_model, verify_exported_model
from nninteractive_finetune.model import (
    configure_trainable_parameters,
    load_model,
    network_parameter_fingerprint,
)
from nninteractive_task_common import (
    NNINTERACTIVE_INPUT_CONTRACT,
    load_registry,
    official_model_dir,
    relative_model_path,
    safe_slug,
    save_registry,
    sha256_file,
    task_dir,
    workspace_root,
    write_json_atomic,
)

OFFICIAL_DIR = official_model_dir()
CLOPA_WEIGHTS = (
    ROOT
    / "external"
    / "nninteractive-finetune"
    / "validation"
    / "data"
    / "brain_clopa_in_weights.npz"
)
TRAIN_MANIFEST = (
    ROOT
    / "external"
    / "nninteractive-finetune"
    / "validation"
    / "data"
    / "manifest_ixi_t1_train_modal.json"
)
TASK_ID = "brain_extraction"
TASK_NAME = "Brain Extraction (MR T1)"
MASK_NAMES = ["Brain", "brain_mask", "brain"]
MODEL_ID = "brain_clopa_in_v2"
LEGACY_MODEL_ID = "brain_clopa_in_v1"


def _manifest_counts() -> tuple[int, int]:
    try:
        payload = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, 0
    cases = payload.get("cases") or []
    if cases:
        training = [row for row in cases if row.get("split") == "train"]
        validation = [row for row in cases if row.get("split") == "val"]
    else:
        training = payload.get("train") or payload.get("training") or []
        validation = payload.get("validation") or payload.get("val") or []
    return len(training), len(validation)


def _minimal_inference_info(path: Path) -> None:
    if path.is_file():
        return
    write_json_atomic(
        path,
        {
            "all_results_in_order": False,
            "perform_everything_on_gpu": True,
            "save_inference_info_for_faster_inference": False,
            "use_mirroring": False,
        },
    )


def _register_model(
    model_dir: Path,
    manifest: dict,
    train_count: int,
    validation_count: int,
) -> None:
    runtime = dict(manifest.get("runtime_verification") or {})
    expected = str(runtime.get("expected_parameter_fingerprint") or "")
    loaded = str(runtime.get("loaded_parameter_fingerprint") or "")
    if not runtime.get("verified") or not expected or expected != loaded:
        raise RuntimeError("The exported model failed effective runtime verification.")

    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    workspace = workspace_root()
    registry = load_registry(workspace)
    tasks = registry.get("tasks") or []
    task = next(
        (
            row
            for row in tasks
            if safe_slug(row.get("task_id")) == safe_slug(TASK_ID)
        ),
        None,
    )
    if task is None:
        task = {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "mask_names": list(MASK_NAMES),
            "models": [],
        }
        tasks.append(task)

    now = time.time()
    for row in task.get("models") or []:
        if str(row.get("model_id") or "") == LEGACY_MODEL_ID:
            row["state"] = "corrupt"
            row["compatible"] = False
            row["superseded_by"] = MODEL_ID
            row["error"] = (
                "Legacy direct state-dict patching was overwritten by shared "
                "parameter aliases during nnInteractive model loading."
            )

    model = {
        "model_id": MODEL_ID,
        "model_relpath": relative_model_path(workspace, model_dir),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fold": "0",
        "strategy": "clopa_in",
        "parent_model_id": "official",
        "created_at_epoch": now,
        "train_case_count": train_count,
        "validation_case_count": validation_count,
        "quality": {},
        "state": "unverified",
        "compatible": True,
        "source_mode": "prepared",
        "input_contract": NNINTERACTIVE_INPUT_CONTRACT,
        "validated_prompt_types": ["point"],
        "runtime_verified": True,
        "effective_model_fingerprint": loaded,
        "validation_note": (
            "Runtime loading is verified. Independent held-out segmentation "
            "quality has not yet been established."
        ),
    }
    task["models"] = [
        row
        for row in task.get("models") or []
        if str(row.get("model_id") or "") != MODEL_ID
    ] + [model]
    task["recommended_model_id"] = MODEL_ID
    task["updated_at_epoch"] = now
    task["mask_names"] = list(MASK_NAMES)
    registry["tasks"] = tasks
    save_registry(workspace, registry)
    write_json_atomic(task_dir(workspace, TASK_ID) / "task.json", task)


def main() -> int:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with nninteractive_env Python: {}".format(exc)
        )

    workspace = workspace_root()
    base_model_dir = OFFICIAL_DIR
    if not (base_model_dir / "fold_0" / "checkpoint_final.pth").is_file():
        base_model_dir = (
            task_dir(workspace, TASK_ID)
            / "models"
            / LEGACY_MODEL_ID
        )
    if not (base_model_dir / "fold_0" / "checkpoint_final.pth").is_file():
        raise FileNotFoundError(
            "Neither the official model nor the legacy model checkpoint is "
            "available. Expected: {}".format(OFFICIAL_DIR)
        )
    model_dir = task_dir(workspace, TASK_ID) / "models" / MODEL_ID
    if model_dir.exists():
        existing_manifest = model_dir / "finetune_manifest.json"
        if existing_manifest.is_file():
            raise RuntimeError(
                "The immutable model version already exists: {}. "
                "Use a new model ID for a different checkpoint.".format(model_dir)
            )
        # A terminated verification attempt may leave an unpublished directory.
        # It is safe to rebuild only while no manifest and no registry row exist.
        registry = load_registry(workspace)
        registered = any(
            str(model.get("model_id") or "") == MODEL_ID
            for task in registry.get("tasks") or []
            for model in task.get("models") or []
            if isinstance(model, dict)
        )
        if registered:
            raise RuntimeError(
                "The incomplete model directory is already registered and "
                "requires manual review: {}".format(model_dir)
            )
        shutil.rmtree(model_dir)

    print("Loading the effective nnInteractive base network from {}...".format(base_model_dir))
    bundle = load_model(base_model_dir, device=torch.device("cpu"), fold=0)
    official_fingerprint = network_parameter_fingerprint(bundle.network)
    parameters = dict(bundle.network.named_parameters())
    has_archive = CLOPA_WEIGHTS.is_file()
    if has_archive:
        archive = np.load(str(CLOPA_WEIGHTS))
        weight_names = sorted(archive.files)

        def adapted_value(name):
            return torch.from_numpy(archive[name])

        weight_source = "validated NPZ"
    else:
        legacy_checkpoint = torch.load(
            str(base_model_dir / "fold_0" / "checkpoint_final.pth"),
            map_location="cpu",
            weights_only=False,
        )
        legacy_state = legacy_checkpoint.get("network_weights") or {}
        policy = configure_trainable_parameters(bundle.network, "clopa_in")
        weight_names = list(policy["names"])

        def adapted_value(name):
            return legacy_state[name].detach().cpu()

        weight_source = "legacy checkpoint canonical CLoPA-IN keys"
    applied = 0
    missing = []
    for name in weight_names:
        parameter = parameters.get(name)
        if parameter is None or (
            not has_archive and name not in legacy_state
        ):
            missing.append(name)
            continue
        value = adapted_value(name)
        if tuple(parameter.shape) != tuple(value.shape):
            raise RuntimeError(
                "Shape mismatch for {}: {} != {}".format(
                    name, tuple(parameter.shape), tuple(value.shape)
                )
            )
        parameter.data.copy_(value.to(dtype=parameter.dtype))
        applied += 1
    if not has_archive:
        del legacy_state
        del legacy_checkpoint
        gc.collect()
    else:
        archive.close()
    if missing or applied != len(weight_names):
        raise RuntimeError(
            "CLoPA weights did not map to the real network parameters. "
            "Applied {}/{}; missing: {}".format(
                applied, len(weight_names), ", ".join(missing[:10])
            )
        )
    adapted_fingerprint = network_parameter_fingerprint(bundle.network)
    if adapted_fingerprint == official_fingerprint:
        raise RuntimeError("The supplied fine-tuned weights did not change the network.")

    train_count, validation_count = _manifest_counts()
    summary = {
        "strategy": "clopa_in",
        "prompt_mode": "click",
        "validated_prompt_types": ["point"],
        "epochs_completed": 10,
        "steps_per_epoch": 50,
        "train_case_count": train_count,
        "validation_case_count": validation_count,
        "source": (
            "validated_weight_import"
            if has_archive
            else "legacy_checkpoint_repair"
        ),
        "weight_source": weight_source,
    }
    print("Exporting and reloading the effective adapted network...")
    manifest = export_model(bundle, model_dir, summary, fold=0)
    del bundle
    gc.collect()
    manifest = verify_exported_model(model_dir, fold=0)
    _minimal_inference_info(model_dir / "inference_info.json")
    _register_model(model_dir, manifest, train_count, validation_count)

    verification = manifest["runtime_verification"]
    print("Registered {} for task {}.".format(MODEL_ID, TASK_ID))
    print("Applied {} real network parameters.".format(applied))
    print("Weight source: {}.".format(weight_source))
    print(
        "Effective fingerprint: {}".format(
            verification["loaded_parameter_fingerprint"]
        )
    )
    print(
        "Cases recorded from manifest: {} training, {} validation.".format(
            train_count, validation_count
        )
    )
    print(
        "Prompt capability: point only. Quality remains unverified until an "
        "independent held-out segmentation evaluation is completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
