"""DINOv3 Medical Segmentation — Training Entry Point."""

import argparse
import sys
import os
import random
import json
import shutil
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.device import get_device
from src.models.segmentor import DINOv33DSegmentor
from src.data.dataset_3d import (
    CaseIdSubset,
    MedicalVolumeDataset,
    FewShotSubset,
    pad_volume_batch,
)
from src.data.input_contract import validate_input_contract
from src.data.augmentation import VolumeAugmentation
from src.training.trainer import Trainer3D
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset, get_worker_info


_LAST_CACHE_STATUS_WRITE_ERROR = {"key": "", "time": 0.0}


def _seed_process(seed: int) -> None:
    """Seed before dataset augmentation and model construction, not afterwards."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _base_dataset(dataset):
    """Unwrap torch Subset objects without depending on a concrete dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _seed_data_worker(_worker_id: int) -> None:
    """Seed Python, NumPy, torch, and the dataset-owned augmentation RNG."""
    info = get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    augmentation = getattr(_base_dataset(info.dataset), "augmentation", None)
    if augmentation is not None and hasattr(augmentation, "reseed"):
        augmentation.reseed(worker_seed + 1)


def _write_cache_status(config, payload):
    status_path = str(config.get("runtime", {}).get("status_path") or "")
    if not status_path:
        return
    data = {}
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        pass
    data.update({
        "status": "training",
        "updated_at_epoch": time.time(),
    })
    data.update(payload)
    parent = os.path.dirname(os.path.abspath(status_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    last_error = None
    for attempt in range(12):
        temporary = "{}.{}.{}.tmp".format(status_path, os.getpid(), uuid.uuid4().hex)
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, status_path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.isfile(temporary):
                    os.remove(temporary)
            except OSError:
                pass
            time.sleep(min(0.2, 0.03 * (attempt + 1)))
    key = "{}:{}".format(status_path, last_error)
    now = time.time()
    if (
        key != _LAST_CACHE_STATUS_WRITE_ERROR["key"]
        or now - _LAST_CACHE_STATUS_WRITE_ERROR["time"] >= 30.0
    ):
        print(
            "Warning: could not update frozen-feature preparation status {}: {}".format(
                status_path,
                last_error,
            ),
            flush=True,
        )
        _LAST_CACHE_STATUS_WRITE_ERROR.update({"key": key, "time": now})


def _cached_slice_cancelled(config) -> bool:
    cancel_path = str(config.get("runtime", {}).get("cancel_path") or "")
    return bool(cancel_path and os.path.isfile(cancel_path))


def _select_native_support(dataset, data_cfg, seed):
    support_case_ids = list(data_cfg.get("support_case_ids") or [])
    if support_case_ids:
        requested = {str(value) for value in support_case_ids}
        dataset.samples = [
            sample for sample in dataset.samples if str(sample["case_id"]) in requested
        ]
        missing = sorted(requested - {str(sample["case_id"]) for sample in dataset.samples})
        if missing:
            raise RuntimeError(
                "Requested support cases are missing from the native slice dataset: {}".format(
                    ", ".join(missing)
                )
            )
        return
    k_shot = int(data_cfg.get("k_shot", -1))
    if 0 < k_shot < len(dataset.samples):
        rng = random.Random(int(seed) + 42 + int(data_cfg.get("fold", 0)))
        selected = list(dataset.samples)
        rng.shuffle(selected)
        dataset.samples = sorted(selected[:k_shot], key=lambda row: row["case_id"])


def _train_cached_slices(config, seed):
    from src.data.frozen_feature_slices import (
        CachedFeatureSliceDataset,
        NativeSliceVolumeDataset,
        build_feature_cache,
        remove_feature_cache,
    )
    from src.training.cached_slice_trainer import CachedFeatureSliceTrainer

    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    decoder_type = str(config.get("decoder", {}).get("type") or "")
    if decoder_type != "feature_unet2d":
        raise ValueError("cached_slices pipeline requires decoder.type=feature_unet2d")
    if str(config.get("finetune", {}).get("method", "")) != "frozen":
        raise ValueError("cached_slices pipeline requires a frozen DINO encoder")
    slice_size = tuple(int(value) for value in (data_cfg.get("img_size") or []))
    if len(slice_size) != 2 or any(value <= 0 or value % 16 for value in slice_size):
        raise ValueError("cached_slices requires two image dimensions divisible by 16")
    if str(model_cfg.get("slice_axis", "axial")).lower() != "axial":
        raise ValueError("cached_slices pipeline currently preserves native axial slice order")
    if data_cfg.get("target_spacing") not in (None, [], ""):
        raise ValueError(
            "cached_slices preserves the source NIfTI grid and does not support "
            "data.target_spacing. Resample image and label together before training "
            "if a common physical spacing is required."
        )
    if bool((data_cfg.get("patch") or {}).get("enabled", False)):
        raise ValueError("cached_slices pipeline does not combine with 3D patch sampling")
    if bool(config.get("augmentation", {}).get("enabled", False)):
        raise ValueError("cached_slices uses its verified in-plane flip augmentation")
    if int(training_cfg.get("grad_accumulation", 1)) != 1:
        raise ValueError("cached_slices uses real slice batches; grad_accumulation must be 1")
    slice_normalization = str(
        data_cfg.get("slice_normalization") or "timeslice_casewise"
    )

    train_volumes = NativeSliceVolumeDataset(
        data_cfg["data_root"],
        "train",
        slice_size=slice_size,
        normalization=slice_normalization,
    )
    _select_native_support(train_volumes, data_cfg, seed)
    validation_enabled = bool(training_cfg.get("validation_enabled", True))
    val_volumes = (
        NativeSliceVolumeDataset(
            data_cfg["data_root"],
            "val",
            slice_size=slice_size,
            normalization=slice_normalization,
        )
        if validation_enabled
        else None
    )

    model = DINOv33DSegmentor(config)
    device = get_device()
    model = model.to(device)
    experiment_root = Path(training_cfg.get("experiment_root", "experiments"))
    experiment_dir = experiment_root / str(config.get("exp_name", "experiment"))
    cache_root = experiment_dir / "feature_cache"
    cache_paths = [cache_root / "train"]
    if val_volumes is not None:
        cache_paths.append(cache_root / "val")
    shutil.rmtree(cache_root, ignore_errors=True)

    def report(payload):
        _write_cache_status(
            config,
            dict(payload, latest_epoch_line=(
                "Preparing frozen image features: case {}/{}; slice {}/{}".format(
                    payload.get("case", 0),
                    payload.get("cases", 0),
                    payload.get("completed_slices", 0),
                    payload.get("total_slices", 0),
                )
            )),
        )

    train_dataset = None
    val_dataset = None
    try:
        train_manifest = build_feature_cache(
            model,
            train_volumes,
            cache_paths[0],
            device,
            slice_batch_size=int(model_cfg.get("slice_batch_size", 4)),
            progress=report,
            cancelled=lambda: _cached_slice_cancelled(config),
        )
        val_manifest = None
        if val_volumes is not None:
            val_manifest = build_feature_cache(
                model,
                val_volumes,
                cache_paths[1],
                device,
                slice_batch_size=int(model_cfg.get("slice_batch_size", 4)),
                progress=report,
                cancelled=lambda: _cached_slice_cancelled(config),
            )

        train_dataset = CachedFeatureSliceDataset(train_manifest)
        batch_size = int(training_cfg.get("batch_size", 4))
        if batch_size < 1:
            raise ValueError("training.batch_size must be positive")
        drop_last = len(train_dataset) > batch_size and len(train_dataset) % batch_size != 0
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=int(training_cfg.get("num_workers", 0)),
            generator=generator,
            worker_init_fn=_seed_data_worker,
        )
        val_dataset = (
            CachedFeatureSliceDataset(val_manifest)
            if val_manifest is not None
            else None
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(training_cfg.get("num_workers", 0)),
                worker_init_fn=_seed_data_worker,
            )
            if val_manifest is not None
            else None
        )
        trainer = CachedFeatureSliceTrainer(model, config, train_loader, val_loader)
        trainer.train()
    finally:
        if train_dataset is not None:
            train_dataset.close()
        if val_dataset is not None:
            val_dataset.close()
        if not bool(training_cfg.get("keep_feature_cache", False)):
            remaining = remove_feature_cache(cache_paths)
            if remaining:
                _write_cache_status(
                    config,
                    {
                        "feature_cache_cleanup_warning": (
                            "Feature cache cleanup will be retried by task maintenance: "
                            + ", ".join(remaining)
                        )
                    },
                )


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 3D medical segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root")
    parser.add_argument("--data.k_shot", type=int, default=None, dest="k_shot", help="Few-shot k")
    parser.add_argument("--training.epochs", type=int, default=None, dest="epochs")
    parser.add_argument("--training.lr", type=float, default=None, dest="lr")
    parser.add_argument("--training.batch_size", type=int, default=None, dest="batch_size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--decoder.type", type=str, default=None, dest="decoder_type")
    parser.add_argument("--finetune.method", type=str, default=None, dest="ft_method")
    args = parser.parse_args()

    # Build override dict from CLI args
    overrides = {}
    if args.data_root:
        overrides["data.data_root"] = args.data_root
    if args.k_shot is not None:
        overrides["data.k_shot"] = args.k_shot
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.lr is not None:
        overrides["training.lr"] = args.lr
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.decoder_type:
        overrides["decoder.type"] = args.decoder_type
    if args.ft_method:
        overrides["finetune.method"] = args.ft_method

    # Load config
    config = load_config(args.config, overrides)
    validate_input_contract(config)
    seed = int(config.get("training", {}).get("seed", 0))
    _seed_process(seed)

    if str(config.get("training", {}).get("pipeline", "volume")) == "cached_slices":
        _train_cached_slices(config, seed)
        return

    # Dataset
    data_cfg = config["data"]
    data_kwargs = dict(
        data_root=data_cfg["data_root"],
        img_size=tuple(data_cfg.get("img_size", [512, 512])),
        modality=data_cfg.get("modality", "other"),
        target_spacing=data_cfg.get("target_spacing", None),
        intensity=data_cfg.get("intensity", {}),
        channel_policy=config["model"].get("channel_policy", "repeat"),
        patch=data_cfg.get("patch", {}),
        roi=data_cfg.get("roi", {}),
        target=data_cfg.get("target", {}),
        slice_axis=config["model"].get("slice_axis", "axial"),
        resize_mode=data_cfg.get("resize_mode", "stretch"),
        normalization_scope=data_cfg.get(
            "normalization_scope", "sample"
        ),
    )

    # Augmentation (train only)
    aug_cfg = config.get("augmentation", {})
    aug = VolumeAugmentation(aug_cfg, seed=seed + 1) if aug_cfg.get("enabled", False) else None

    train_dataset = MedicalVolumeDataset(
        split="train", augmentation=aug, **data_kwargs
    )

    validation_enabled = bool(config.get("training", {}).get("validation_enabled", True))
    val_dataset = MedicalVolumeDataset(split="test", **data_kwargs) if validation_enabled else None

    # Few-shot sampling
    support_case_ids = data_cfg.get("support_case_ids")
    k_shot = data_cfg.get("k_shot", -1)
    if support_case_ids:
        train_dataset = CaseIdSubset(train_dataset, support_case_ids)
        print("Few-shot training: explicit support cases: {}".format(", ".join(support_case_ids)))
    elif k_shot > 0:
        train_dataset = FewShotSubset(train_dataset, k=k_shot, seed=42 + data_cfg.get("fold", 0))
        print("Few-shot training: {} random volumes selected (use support_case_ids for an auditable split)".format(k_shot))

    num_workers = config["training"].get("num_workers", 0)
    # Deterministic shuffle: a seeded generator makes the per-epoch sample order
    # reproducible across runs (Trainer3D also seeds torch/numpy/cuda).
    g = torch.Generator()
    g.manual_seed(seed)
    batch_size = int(config["training"].get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=_seed_data_worker,
        collate_fn=pad_volume_batch if batch_size > 1 else None,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_seed_data_worker,
        )

    # Model
    model = DINOv33DSegmentor(config)

    # Trainer
    trainer = Trainer3D(model, config, train_loader, val_loader)

    # Resume or train
    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
