"""Write inference-compatible adapted model folders."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from .data import nninteractive_input_contract
from .model import ModelBundle, audit_model_dir, checkpoint_sha256
from .runtime import write_json_atomic

METADATA_FILES = (
    "plans.json",
    "dataset.json",
    "inference_info.json",
    "inference_session_class.json",
    "LICENSE",
)


def _save_torch_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        "{}.{}.{}.tmp".format(path.name, os.getpid(), uuid.uuid4().hex)
    )
    try:
        torch.save(payload, str(temporary))
        last_error = None
        for attempt in range(40):
            try:
                os.replace(str(temporary), str(path))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.5, 0.025 * (attempt + 1)))
        raise OSError(
            "Could not publish checkpoint {} after Windows/SMB replace retries: {}".format(
                path, last_error
            )
        )
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def export_model(
    bundle: ModelBundle,
    output_dir: str | Path,
    training_summary: dict[str, Any],
    fold: int | str = 0,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if (
        destination == bundle.model_dir
        or destination.is_relative_to(bundle.model_dir)
        or bundle.model_dir.is_relative_to(destination)
    ):
        raise ValueError(
            "Fine-tuning output and base model directories must not overlap."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        source = bundle.model_dir / name
        if source.is_file():
            shutil.copy2(str(source), str(destination / name))

    # Only replace network_weights. Deep-copying the original checkpoint duplicates
    # hundreds of megabytes of tensor storage and can exhaust a 3060-class system.
    checkpoint = dict(bundle.checkpoint)
    checkpoint["network_weights"] = {
        name: tensor.detach().cpu()
        for name, tensor in bundle.network.state_dict().items()
    }
    checkpoint["current_epoch"] = int(training_summary.get("epochs_completed", 0))
    checkpoint["nninteractive_finetune"] = {
        "schema_version": "nninteractive_finetune_checkpoint.v1",
        "created_at_epoch": time.time(),
        "base_checkpoint_sha256": checkpoint_sha256(bundle.checkpoint_path),
        **training_summary,
    }
    fold_dir = destination / "fold_{}".format(fold)
    checkpoint_path = fold_dir / "checkpoint_final.pth"
    _save_torch_atomic(checkpoint_path, checkpoint)
    manifest = {
        "schema_version": "nninteractive_finetune_model.v1",
        "model_dir": str(destination),
        "checkpoint": str(checkpoint_path),
        "base_model_dir": str(bundle.model_dir),
        "base_checkpoint": str(bundle.checkpoint_path),
        "input_contract": nninteractive_input_contract(),
        "training": training_summary,
    }
    write_json_atomic(destination / "finetune_manifest.json", manifest)
    manifest["compatibility_audit"] = audit_model_dir(
        destination, fold=fold, checkpoint_name="checkpoint_final.pth"
    )
    write_json_atomic(destination / "finetune_manifest.json", manifest)
    return manifest


def save_training_state(path: str | Path, payload: dict[str, Any]) -> None:
    _save_torch_atomic(Path(path), payload)
