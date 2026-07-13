"""DINOv3 Medical Segmentation — Training Entry Point."""

import argparse
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.device import get_device
from src.models.segmentor import DINOv33DSegmentor
from src.data.dataset_3d import CaseIdSubset, MedicalVolumeDataset, FewShotSubset
from src.data.augmentation import VolumeAugmentation
from src.training.trainer import Trainer3D
import torch
import numpy as np
from torch.utils.data import DataLoader


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
    seed = int(config.get("training", {}).get("seed", 0))
    _seed_process(seed)

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
    if batch_size != 1:
        raise ValueError(
            "Full-volume cases have variable Z length, so this pipeline requires "
            "training.batch_size=1. Increase training.grad_accumulation instead."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
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
