"""CLoPA-style task adaptation with true iterative prediction feedback."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import export_model, save_training_state
from .data import (
    INITIAL_MASK_QUALITY_THRESHOLDS,
    InteractivePatchDataset,
    load_manifest,
    prepare_cases,
    remove_prepared_cache,
    split_cases,
)
from .losses import binary_dice, dice_ce_loss
from .model import checkpoint_sha256, configure_trainable_parameters, load_model
from .prompts import InteractivePromptSampler
from .runtime import (
    CancelledError,
    StatusReporter,
    cancellation_requested,
    exclusive_job_lock,
)


def select_device(requested: str) -> torch.device:
    value = str(requested or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _parameter_delta(
    network: torch.nn.Module, trainable_names: list[str]
) -> dict[str, torch.Tensor]:
    parameters = dict(network.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in trainable_names}


@torch.no_grad()
def _load_parameter_delta(
    network: torch.nn.Module, delta: dict[str, torch.Tensor]
) -> None:
    parameters = dict(network.named_parameters())
    missing = sorted(set(delta) - set(parameters))
    if missing:
        raise RuntimeError(
            "Adaptation state has unknown parameters: {}".format(missing)
        )
    for name, tensor in delta.items():
        parameters[name].copy_(tensor.to(parameters[name].device))


def _training_state(
    network: torch.nn.Module,
    trainable_names: list[str],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_delta: dict[str, torch.Tensor] | None,
    history: list[dict[str, Any]],
    resume_signature: str,
    sampler_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "nninteractive_finetune_training_state.v2",
        "epoch": int(epoch),
        "best_score": float(best_score),
        "trainable_parameters": _parameter_delta(network, trainable_names),
        "best_parameters": best_delta,
        "history": history,
        "resume_signature": resume_signature,
        "sampler_state": sampler_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
    }


def _restore_training_state(
    path: Path,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    expected_signature: str,
    device: torch.device,
) -> tuple[
    int,
    float,
    dict[str, torch.Tensor] | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if state.get("schema_version") != "nninteractive_finetune_training_state.v2":
        raise ValueError("Unsupported training state: {}".format(path))
    if state.get("resume_signature") != expected_signature:
        raise ValueError(
            "The saved training state belongs to a different model, dataset, or "
            "training configuration. Use a new output directory or remove {}.".format(
                path
            )
        )
    _load_parameter_delta(network, state["trainable_parameters"])
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)
    scaler.load_state_dict(state.get("scaler") or {})
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_state"])
    torch.set_rng_state(state["torch_state"])
    return (
        int(state["epoch"]) + 1,
        float(state.get("best_score", -math.inf)),
        state.get("best_parameters"),
        list(state.get("history") or []),
        state.get("sampler_state"),
    )


def _resume_signature(config: dict[str, Any], base_checkpoint_hash: str) -> str:
    training = dict(config["training"])
    for key in ("epochs", "output_dir", "status_path", "cancel_path", "resume"):
        training.pop(key, None)
    payload = {
        "model": config["model"],
        "data": config["data"],
        "prompts": config["prompts"],
        "training": training,
        "base_checkpoint_sha256": base_checkpoint_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward(network: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    output = network(value)
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 5:
        raise RuntimeError("nnInteractive network returned an invalid output.")
    return output


def _batch_text_values(value: Any, size: int, default: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = []
    values.extend([default] * max(0, int(size) - len(values)))
    return [str(item or default) for item in values[: int(size)]]


def _record_initial_stratum_baseline(
    groups: dict[str, dict[str, Any]],
    key: str,
    case_id: str,
    baseline: float,
    horizon: int,
) -> None:
    group = groups.setdefault(
        str(key or "unknown"),
        {
            "case_ids": set(),
            "baseline": [],
            "trajectory": [[] for _ in range(int(horizon))],
        },
    )
    group["case_ids"].add(str(case_id))
    group["baseline"].append(float(baseline))


def _record_initial_stratum_dice(
    groups: dict[str, dict[str, Any]],
    key: str,
    interaction_index: int,
    value: float,
) -> None:
    group = groups.get(str(key or "unknown"))
    if group is not None:
        group["trajectory"][int(interaction_index)].append(float(value))


def _summarize_initial_strata(
    groups: dict[str, dict[str, Any]], validation_steps: list[int]
) -> dict[str, dict[str, Any]]:
    result = {}
    for key, group in sorted(groups.items()):
        trajectory = [
            float(np.mean(values)) if values else None
            for values in group["trajectory"]
        ]
        selected = [
            trajectory[step - 1]
            for step in validation_steps
            if trajectory[step - 1] is not None
        ]
        baseline = float(np.mean(group["baseline"]))
        auc = float(np.mean(selected)) if selected else None
        result[key] = {
            "case_count": len(group["case_ids"]),
            "patch_sample_count": len(group["baseline"]),
            "baseline_dice": baseline,
            "dice_by_interaction": trajectory,
            "trajectory_auc": auc,
            "auc_gain_vs_baseline": (
                float(auc) - baseline if auc is not None else None
            ),
        }
    return result


def _initial_mask_distribution(
    rows: list[dict[str, Any]], key: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        if not bool(metadata.get("has_initial_mask")):
            continue
        value = str(metadata.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


@torch.no_grad()
def validate(
    network: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prompt_config: dict[str, Any],
    batches: int,
    seed: int,
    cancel_path: str | Path | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    network.eval()
    sampler = InteractivePromptSampler(
        radius=int(prompt_config["point_radius"]),
        center_bias=float(prompt_config["center_bias"]),
        decay=float(prompt_config["interaction_decay"]),
        seed=seed,
        initial_mask_probability=float(
            prompt_config.get("initial_mask_probability", 0.0)
        ),
        correction_policy=str(
            prompt_config.get("correction_policy") or "clopa_paired"
        ),
    )
    validation_steps = [
        int(value)
        for value in (
            prompt_config.get("validation_interaction_steps")
            or [int(prompt_config["interaction_steps"])]
        )
    ]
    validation_horizon = max(validation_steps)
    trajectory: list[list[float]] = [
        [] for _ in range(validation_horizon)
    ]
    initial_mask_trajectory: list[list[float]] = [
        [] for _ in range(validation_horizon)
    ]
    empty_mask_baselines: list[float] = []
    initial_mask_baselines: list[float] = []
    initial_quality_groups: dict[str, dict[str, Any]] = {}
    initial_source_groups: dict[str, dict[str, Any]] = {}
    iterator = _infinite(loader)
    total_batches = max(1, int(batches))
    for batch_index in range(total_batches):
        if cancellation_requested(cancel_path):
            raise CancelledError(
                "Training cancellation was requested during validation."
            )
        batch = next(iterator)
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        empty_mask_baselines.extend(
            float(value)
            for value in binary_dice(torch.zeros_like(target), target).cpu()
        )
        interactions = sampler.new_interactions(
            target, allow_initial_mask=False
        )
        for interaction_index in range(len(trajectory)):
            logits = _forward(network, torch.cat([image, interactions], dim=1))
            prediction = logits.argmax(1)
            trajectory[interaction_index].extend(
                float(value) for value in binary_dice(prediction, target).cpu()
            )
            if interaction_index + 1 < len(trajectory):
                sampler.add_corrections(interactions, prediction, target)
        if bool(prompt_config.get("validate_initial_masks", True)):
            provided_initial = batch.get("initial_mask", None)
            provided_available = batch.get("has_initial_mask", None)
            if provided_initial is None or provided_available is None:
                available = torch.zeros(
                    target.shape[0], dtype=torch.bool, device=device
                )
                provided_initial = torch.zeros_like(target)
            else:
                provided_initial = provided_initial.to(
                    device, non_blocking=True
                )
                available = provided_available.to(
                    device, non_blocking=True
                ).bool()
            case_ids = _batch_text_values(
                batch.get("case_id"), target.shape[0], "unknown_case"
            )
            quality_bins = _batch_text_values(
                batch.get("initial_mask_quality_bin"),
                target.shape[0],
                "unknown",
            )
            source_types = _batch_text_values(
                batch.get("initial_mask_source_type"),
                target.shape[0],
                "unknown",
            )
            if bool(available.any()):
                baseline_values = binary_dice(provided_initial, target)
                initial_mask_baselines.extend(
                    float(value)
                    for value in baseline_values[available].cpu()
                )
                for sample_index in torch.nonzero(
                    available, as_tuple=False
                ).flatten().tolist():
                    baseline = float(baseline_values[sample_index].cpu())
                    _record_initial_stratum_baseline(
                        initial_quality_groups,
                        quality_bins[sample_index],
                        case_ids[sample_index],
                        baseline,
                        validation_horizon,
                    )
                    _record_initial_stratum_baseline(
                        initial_source_groups,
                        source_types[sample_index],
                        case_ids[sample_index],
                        baseline,
                        validation_horizon,
                    )
            interactions = sampler.new_interactions(
                target,
                allow_initial_mask=True,
                force_initial_mask=True,
                initial_prediction=provided_initial,
                initial_mask_available=available,
            )
            for interaction_index in range(len(initial_mask_trajectory)):
                logits = _forward(
                    network, torch.cat([image, interactions], dim=1)
                )
                prediction = logits.argmax(1)
                if bool(available.any()):
                    dice_values = binary_dice(prediction, target)
                    initial_mask_trajectory[interaction_index].extend(
                        float(value) for value in dice_values[available].cpu()
                    )
                    for sample_index in torch.nonzero(
                        available, as_tuple=False
                    ).flatten().tolist():
                        dice_value = float(dice_values[sample_index].cpu())
                        _record_initial_stratum_dice(
                            initial_quality_groups,
                            quality_bins[sample_index],
                            interaction_index,
                            dice_value,
                        )
                        _record_initial_stratum_dice(
                            initial_source_groups,
                            source_types[sample_index],
                            interaction_index,
                            dice_value,
                        )
                if interaction_index + 1 < len(initial_mask_trajectory):
                    sampler.add_corrections(
                        interactions, prediction, target, active=available
                    )
        if progress:
            progress(batch_index + 1, total_batches)
    means = [float(np.mean(values)) if values else 0.0 for values in trajectory]
    initial_mask_means = [
        float(np.mean(values)) if values else None
        for values in initial_mask_trajectory
    ]
    reported = {
        str(step): means[step - 1] for step in validation_steps
    }
    reported_initial = {
        str(step): initial_mask_means[step - 1]
        for step in validation_steps
        if initial_mask_means[step - 1] is not None
    }
    initial_probability = float(
        prompt_config.get("initial_mask_probability", 0.0)
    )
    empty_auc = float(np.mean([reported[str(step)] for step in validation_steps]))
    initial_auc = (
        float(np.mean(list(reported_initial.values())))
        if reported_initial
        else None
    )
    empty_baseline = float(np.mean(empty_mask_baselines))
    initial_baseline = (
        float(np.mean(initial_mask_baselines))
        if initial_mask_baselines
        else None
    )
    effective_initial_probability = (
        initial_probability if initial_auc is not None else 0.0
    )
    trajectory_auc = (
        (1.0 - effective_initial_probability) * empty_auc
        + effective_initial_probability * float(initial_auc or 0.0)
    )
    trajectory_baseline = (
        (1.0 - effective_initial_probability) * empty_baseline
        + effective_initial_probability * float(initial_baseline or 0.0)
    )
    return {
        "dice_by_interaction": means,
        "dice_at_interactions": reported,
        "initial_mask_dice_by_interaction": initial_mask_means,
        "initial_mask_dice_at_interactions": reported_initial,
        "empty_mask_baseline_dice": empty_baseline,
        "initial_mask_baseline_dice": initial_baseline,
        "initial_dice": trajectory_baseline,
        "final_dice": means[-1],
        "empty_mask_trajectory_auc": empty_auc,
        "initial_mask_trajectory_auc": initial_auc,
        "empty_mask_auc_gain_vs_baseline": empty_auc - empty_baseline,
        "initial_mask_auc_gain_vs_baseline": (
            float(initial_auc) - float(initial_baseline)
            if initial_auc is not None and initial_baseline is not None
            else None
        ),
        "trajectory_baseline_dice": trajectory_baseline,
        "trajectory_auc": trajectory_auc,
        "trajectory_auc_gain_vs_baseline": trajectory_auc - trajectory_baseline,
        "real_initial_mask_samples": len(initial_mask_baselines),
        "initial_mask_quality_strata": _summarize_initial_strata(
            initial_quality_groups, validation_steps
        ),
        "initial_mask_source_strata": _summarize_initial_strata(
            initial_source_groups, validation_steps
        ),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["training"]["output_dir"])
    base_model_dir = Path(config["model"]["base_model_dir"])
    resolved_output = output_dir.expanduser().resolve()
    resolved_base = base_model_dir.expanduser().resolve()
    if (
        resolved_output == resolved_base
        or resolved_output.is_relative_to(resolved_base)
        or resolved_base.is_relative_to(resolved_output)
    ):
        raise ValueError(
            "Fine-tuning output and base model directories must not overlap."
        )
    work_dir = output_dir.parent / "_nninteractive_finetune_work" / output_dir.name
    lock_job_id = "{}_{}".format(output_dir.name, int(time.time()))
    cancel_path = config["training"].get("cancel_path") or str(
        work_dir / "cancel.request"
    )
    with exclusive_job_lock(work_dir / "training.lock", lock_job_id):
        try:
            Path(cancel_path).unlink()
        except FileNotFoundError:
            pass
        return _train_unlocked(config)


def _train_unlocked(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    prompt_config = config["prompts"]
    output_dir = Path(training["output_dir"])
    work_dir = output_dir.parent / "_nninteractive_finetune_work" / output_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = training.get("status_path") or str(work_dir / "status.json")
    cancel_path = training.get("cancel_path") or str(work_dir / "cancel.request")
    job_id = "{}_{}".format(output_dir.name, int(time.time()))
    status = StatusReporter(status_path, job_id)
    status.update(
        status="initializing", phase="validating_configuration", pid=os.getpid()
    )

    seed = int(training["seed"])
    _seed_everything(seed)
    device = select_device(training["device"])
    mixed_precision = bool(training["mixed_precision"] and device.type == "cuda")
    cache_dir = (
        Path(data_config["prepared_cache_dir"])
        if data_config.get("prepared_cache_dir")
        else output_dir.parent / "_nninteractive_finetune_cache" / output_dir.name
    )
    training_state_path = work_dir / "training_state.pth"

    try:
        cases = load_manifest(data_config["manifest"])
        training_goal = str(
            prompt_config.get("training_goal") or "general"
        ).strip().lower()
        if training_goal == "refine_existing":
            cases = [
                row for row in cases if str(row.get("initial_mask") or "").strip()
            ]
            if not cases:
                raise ValueError(
                    "refine_existing requires cases with a real Initial Mask."
                )
        train_cases, val_cases = split_cases(
            cases, float(data_config["validation_fraction"]), seed
        )

        def report_preparation(values: dict[str, Any]) -> None:
            status.update(status="preparing", **values)

        prepared = prepare_cases(
            train_cases + val_cases,
            cache_dir,
            data_config["label_values"],
            progress=report_preparation,
            cancel_path=cancel_path,
        )
        by_id = {row["case_id"]: row for row in prepared}
        train_prepared = [by_id[row["case_id"]] for row in train_cases]
        val_prepared = [by_id[row["case_id"]] for row in val_cases]
        if training_goal == "refine_existing":
            train_prepared = [
                row
                for row in train_prepared
                if bool((row.get("metadata") or {}).get("has_initial_mask"))
            ]
            val_prepared = [
                row
                for row in val_prepared
                if bool((row.get("metadata") or {}).get("has_initial_mask"))
            ]
            if not train_prepared:
                raise ValueError(
                    "refine_existing requires at least one training case with "
                    "a real Initial Mask distinct from the final target."
                )

        status.update(
            status="initializing", phase="loading_base_model", device=str(device)
        )
        bundle = load_model(
            model_config["base_model_dir"],
            device=device,
            fold=model_config["fold"],
            checkpoint_name=model_config["checkpoint_name"],
        )
        policy = configure_trainable_parameters(
            bundle.network, model_config["strategy"]
        )
        resume_signature = _resume_signature(
            config, checkpoint_sha256(bundle.checkpoint_path)
        )
        optimizer = torch.optim.Adam(
            policy["parameters"],
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scaler = _make_scaler(mixed_precision)

        updates_per_epoch = int(training["steps_per_epoch"])
        accumulation = int(training["gradient_accumulation"])
        batch_size = int(training["batch_size"])
        virtual_length = updates_per_epoch * accumulation * batch_size
        train_dataset = InteractivePatchDataset(
            train_prepared,
            data_config["patch_size"],
            float(data_config["foreground_patch_probability"]),
            virtual_length,
            augmentation=data_config["augmentation"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(data_config["num_workers"]),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        train_iterator = _infinite(train_loader)
        val_loader = None
        if val_prepared:
            val_dataset = InteractivePatchDataset(
                val_prepared,
                data_config["patch_size"],
                1.0,
                max(batch_size, int(training["validation_batches"]) * batch_size),
                augmentation={"enabled": False},
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(data_config["num_workers"]),
                pin_memory=device.type == "cuda",
            )

        start_epoch = 0
        best_score = -math.inf
        best_delta = None
        history: list[dict[str, Any]] = []
        sampler = InteractivePromptSampler(
            radius=int(prompt_config["point_radius"]),
            center_bias=float(prompt_config["center_bias"]),
            decay=float(prompt_config["interaction_decay"]),
            seed=seed,
            initial_mask_probability=float(
                prompt_config.get("initial_mask_probability", 0.0)
            ),
            correction_policy=str(
                prompt_config.get("correction_policy") or "clopa_paired"
            ),
        )
        if bool(training.get("resume", True)) and training_state_path.is_file():
            (
                start_epoch,
                best_score,
                best_delta,
                history,
                sampler_state,
            ) = _restore_training_state(
                training_state_path,
                bundle.network,
                optimizer,
                scaler,
                resume_signature,
                device,
            )
            if sampler_state:
                sampler.rng.bit_generator.state = sampler_state
            status.update(
                status="initializing",
                phase="resumed",
                resumed_from_epoch=start_epoch,
            )

        started_at = time.time()
        for epoch in range(start_epoch, int(training["epochs"])):
            bundle.network.train()
            epoch_loss = 0.0
            epoch_dice = 0.0
            for update_index in range(updates_per_epoch):
                if cancellation_requested(cancel_path):
                    raise CancelledError("Training cancellation was requested.")
                optimizer.zero_grad(set_to_none=True)
                update_loss = 0.0
                update_dice = 0.0
                for _ in range(accumulation):
                    batch = next(train_iterator)
                    image = batch["image"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    interactions = sampler.new_interactions(
                        target,
                        allow_initial_mask=True,
                        initial_prediction=batch.get(
                            "initial_mask", None
                        ).to(target.device, non_blocking=True)
                        if batch.get("initial_mask", None) is not None
                        else None,
                        initial_mask_available=batch.get(
                            "has_initial_mask", None
                        ).to(target.device, non_blocking=True)
                        if batch.get("has_initial_mask", None) is not None
                        else None,
                    )
                    legacy_fixed_steps = (
                        "min_interaction_steps" not in prompt_config
                        and "max_interaction_steps" not in prompt_config
                    )
                    minimum_interactions = int(
                        prompt_config.get(
                            "min_interaction_steps",
                            (
                                prompt_config["interaction_steps"]
                                if legacy_fixed_steps
                                else 1
                            ),
                        )
                    )
                    maximum_interactions = int(
                        prompt_config.get(
                            "max_interaction_steps",
                            prompt_config["interaction_steps"],
                        )
                    )
                    budgets = sampler.sample_interaction_budgets(
                        target.shape[0],
                        minimum_interactions,
                        maximum_interactions,
                        float(
                            prompt_config.get(
                                "short_interaction_probability", 0.7
                            )
                        ),
                        weights=prompt_config.get(
                            "interaction_step_weights"
                        ),
                    ).to(device=target.device)
                    active = budgets > 0
                    final_dice = torch.zeros(
                        target.shape[0], dtype=torch.float32, device=target.device
                    )
                    for interaction_index in range(
                        int(budgets.max().item())
                    ):
                        if cancellation_requested(cancel_path):
                            raise CancelledError("Training cancellation was requested.")
                        if not bool(active.any()):
                            break
                        with torch.autocast(
                            device_type=device.type,
                            enabled=mixed_precision,
                        ):
                            logits = _forward(
                                bundle.network,
                                torch.cat([image, interactions], dim=1),
                            )
                            per_sample_loss = dice_ce_loss(
                                logits, target, reduction="none"
                            )
                            step_active = active & (budgets > interaction_index)
                            weighted = (
                                per_sample_loss[step_active]
                                / budgets[step_active].float()
                            )
                            loss = weighted.sum() / float(target.shape[0])
                            scaled_loss = loss / accumulation
                        scaler.scale(scaled_loss).backward()
                        prediction = logits.detach().argmax(1)
                        final_dice = binary_dice(prediction, target)
                        update_loss += float(loss.detach().cpu()) / accumulation
                        if interaction_index + 1 < int(budgets.max().item()):
                            active &= (final_dice < 1.0) & (
                                budgets > interaction_index + 1
                            )
                            sampler.add_corrections(
                                interactions, prediction, target, active=active
                            )
                        del logits, prediction, per_sample_loss, loss, scaled_loss
                    update_dice += float(final_dice.mean().cpu()) / accumulation
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += update_loss
                epoch_dice += update_dice
                status.update(
                    status="training",
                    phase="training",
                    epoch=epoch + 1,
                    epochs=int(training["epochs"]),
                    update=update_index + 1,
                    updates_per_epoch=updates_per_epoch,
                    loss=update_loss,
                    train_final_dice=update_dice,
                    elapsed_seconds=time.time() - started_at,
                    trainable_parameters=policy["trainable_count"],
                    trainable_fraction=policy["trainable_fraction"],
                )

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / updates_per_epoch,
                "train_final_dice": epoch_dice / updates_per_epoch,
            }
            if val_loader is not None:
                status.update(
                    status="validating",
                    phase="validating",
                    epoch=epoch + 1,
                    validation_batch=0,
                    validation_batches=int(training["validation_batches"]),
                )

                def report_validation(completed: int, total: int) -> None:
                    status.update(
                        status="validating",
                        phase="validating",
                        epoch=epoch + 1,
                        validation_batch=completed,
                        validation_batches=total,
                    )

                metrics = validate(
                    bundle.network,
                    val_loader,
                    device,
                    prompt_config,
                    int(training["validation_batches"]),
                    seed + epoch + 1,
                    cancel_path=cancel_path,
                    progress=report_validation,
                )
                epoch_record["validation"] = metrics
                score = float(metrics["trajectory_auc"])
            else:
                score = float(epoch_record["train_final_dice"])
            if score > best_score:
                best_score = score
                best_delta = _parameter_delta(bundle.network, policy["names"])
                epoch_record["best"] = True
            history.append(epoch_record)
            save_training_state(
                training_state_path,
                _training_state(
                    bundle.network,
                    policy["names"],
                    optimizer,
                    scaler,
                    epoch,
                    best_score,
                    best_delta,
                    history,
                    resume_signature,
                    sampler.rng.bit_generator.state,
                ),
            )
            status.update(
                status="training",
                phase="epoch_completed",
                latest_epoch=epoch_record,
                best_score=best_score,
            )

        if best_delta is not None:
            _load_parameter_delta(bundle.network, best_delta)
        summary = {
            "strategy": model_config["strategy"],
            "prompt_mode": str(prompt_config["mode"]),
            "training_goal": str(
                prompt_config.get("training_goal") or "legacy"
            ),
            "correction_policy": str(
                prompt_config.get("correction_policy") or "clopa_paired"
            ),
            "validated_prompt_types": ["point"],
            "epochs_completed": int(training["epochs"]),
            "steps_per_epoch": updates_per_epoch,
            "interaction_steps": int(prompt_config["interaction_steps"]),
            "min_interaction_steps": int(
                prompt_config.get(
                    "min_interaction_steps",
                    (
                        prompt_config["interaction_steps"]
                        if "max_interaction_steps" not in prompt_config
                        else 1
                    ),
                )
            ),
            "max_interaction_steps": int(
                prompt_config.get(
                    "max_interaction_steps",
                    prompt_config["interaction_steps"],
                )
            ),
            "validation_interaction_steps": list(
                prompt_config.get("validation_interaction_steps")
                or [int(prompt_config["interaction_steps"])]
            ),
            "interaction_step_weights": list(
                prompt_config.get("interaction_step_weights") or []
            ),
            "initial_mask_probability": float(
                prompt_config.get("initial_mask_probability", 0.0)
            ),
            "point_radius": int(prompt_config["point_radius"]),
            "center_bias": float(prompt_config["center_bias"]),
            "interaction_decay": float(prompt_config["interaction_decay"]),
            "patch_size": list(data_config["patch_size"]),
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "learning_rate": float(training["learning_rate"]),
            "trainable_count": policy["trainable_count"],
            "total_count": policy["total_count"],
            "trainable_fraction": policy["trainable_fraction"],
            "train_cases": [row["case_id"] for row in train_prepared],
            "validation_cases": [row["case_id"] for row in val_prepared],
            "real_initial_mask_train_cases": [
                row["case_id"]
                for row in train_prepared
                if bool((row.get("metadata") or {}).get("has_initial_mask"))
            ],
            "real_initial_mask_validation_cases": [
                row["case_id"]
                for row in val_prepared
                if bool((row.get("metadata") or {}).get("has_initial_mask"))
            ],
            "initial_mask_quality_thresholds": list(
                INITIAL_MASK_QUALITY_THRESHOLDS
            ),
            "initial_mask_train_quality_distribution": _initial_mask_distribution(
                train_prepared, "initial_mask_quality_bin"
            ),
            "initial_mask_validation_quality_distribution": _initial_mask_distribution(
                val_prepared, "initial_mask_quality_bin"
            ),
            "initial_mask_train_source_distribution": _initial_mask_distribution(
                train_prepared, "initial_mask_source_type"
            ),
            "initial_mask_validation_source_distribution": _initial_mask_distribution(
                val_prepared, "initial_mask_source_type"
            ),
            "augmentation": dict(data_config.get("augmentation") or {}),
            "best_score": best_score,
            "history": history,
            "config": config,
        }
        status.update(status="finalizing", phase="writing_inference_model")
        manifest = export_model(
            bundle,
            output_dir,
            summary,
            fold=model_config["fold"],
        )
        if not bool(data_config.get("keep_prepared_cache", True)):
            remove_prepared_cache(cache_dir)
        status.update(
            status="finalizing",
            phase="awaiting_runtime_verification",
            model_dir=str(output_dir),
            manifest=str(output_dir / "finetune_manifest.json"),
            best_score=best_score,
        )
        return manifest
    except CancelledError as exc:
        status.update(status="cancelled", phase="cancelled", error=str(exc))
        raise
    except Exception as exc:
        status.update(
            status="failed",
            phase="failed",
            error="{}: {}".format(type(exc).__name__, exc),
        )
        raise
