#!/usr/bin/env python3
"""
Batch experiment runner for DINOv3 few-shot fine-tuning experiments.

Runs a grid of experiments across:
- Organs: brain, liver, adrenal_gland, aorta, scapula
- Decoders: linear3d, mlp_probe, segformer3d, dpt3d, conv2d, conv2d_unet, conv2d_deeplab, conv2d_2_5d
- Fine-tuning methods: frozen, lora, adapter, full
- K-shot values: 1, 3, 5, 10
- Image sizes: 224, 512

Usage:
    # Run all Phase 1 experiments (baseline: frozen + segformer3d, vary k-shot and img_size)
    python scripts/batch_experiments.py --phase 1

    # Run Phase 2 (decoder comparison)
    python scripts/batch_experiments.py --phase 2

    # Run specific experiment
    python scripts/batch_experiments.py --organ liver --decoder segformer3d --finetune frozen --k-shot 5

    # Dry run (print experiments without running)
    python scripts/batch_experiments.py --phase 1 --dry-run

    # Resume from results file
    python scripts/batch_experiments.py --resume results.json
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_BASE = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"

# ──────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────

ORGANS = {
    "brain": {
        "total_seg_label": "brain",
        "description": "Brain (neurocranium)",
        "size_category": "large",
        "difficulty": "easy",
    },
    "liver": {
        "total_seg_label": "liver",
        "description": "Liver",
        "size_category": "large",
        "difficulty": "medium",
    },
    "aorta": {
        "total_seg_label": "aorta",
        "description": "Aorta",
        "size_category": "medium",
        "difficulty": "hard",
    },
    "scapula_left": {
        "total_seg_label": "scapula_left",
        "description": "Scapula (left)",
        "size_category": "medium",
        "difficulty": "hard",
    },
    "adrenal_gland_right": {
        "total_seg_label": "adrenal_gland_right",
        "description": "Adrenal Gland (right)",
        "size_category": "small",
        "difficulty": "very_hard",
    },
}

DECODERS_3D = ["linear3d", "mlp_probe", "segformer3d", "dpt3d"]
DECODERS_2D = ["conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d"]
ALL_DECODERS = DECODERS_3D + DECODERS_2D

FINETUNE_METHODS = ["frozen", "lora", "adapter", "full"]

K_SHOTS = [1, 3, 5, 10]
IMG_SIZES = [224, 512]

BACKBONES = {
    "vitb16": {
        "path": "./models/dinov3-vitb16",
        "embed_dim": 768,
    },
    "vitl16": {
        "path": "./models/dinov3-vitl16",
        "embed_dim": 1024,
    },
}


# ──────────────────────────────────────────────
# Phase definitions
# ──────────────────────────────────────────────

PHASE_1_BASELINE = {
    "name": "Phase 1: Baseline (Frozen + SegFormer3D)",
    "description": "Establish baseline performance for each organ",
    "grid": {
        "organ": list(ORGANS.keys()),
        "k_shot": K_SHOTS,
        "img_size": IMG_SIZES,
        "finetune": ["frozen"],
        "decoder": ["segformer3d"],
        "backbone": ["vitb16"],
    },
    "epochs": 50,
    "repeats": 2,
}

PHASE_2_DECODER = {
    "name": "Phase 2: Decoder Architecture Comparison",
    "description": "Compare all 8 decoder types (3D + 2D)",
    "grid": {
        "organ": list(ORGANS.keys()),
        "k_shot": [5],
        "img_size": [224],
        "finetune": ["frozen"],
        "decoder": ALL_DECODERS,
        "backbone": ["vitb16"],
    },
    "epochs": 50,
    "repeats": 2,
}

PHASE_3_FINETUNE = {
    "name": "Phase 3: Fine-Tuning Method Comparison",
    "description": "Compare frozen, LoRA, adapter, full fine-tuning",
    "grid": {
        "organ": list(ORGANS.keys()),
        "k_shot": [5],
        "img_size": [224],
        "finetune": FINETUNE_METHODS,
        "decoder": ["segformer3d"],
        "backbone": ["vitb16"],
    },
    "epochs": 50,
    "repeats": 2,
}

PHASE_4_ABLATION = {
    "name": "Phase 4: Ablation Studies",
    "description": "Measure contribution of each component",
    "grid": {
        "organ": list(ORGANS.keys()),
        "k_shot": [5],
        "img_size": IMG_SIZES,
        "finetune": ["frozen"],
        "decoder": ["segformer3d"],
        "backbone": ["vitb16"],
    },
    "epochs": 50,
    "repeats": 1,
    "ablations": [
        {"augmentation": {"enabled": True, "intensity": {"gamma_range": [0.7, 1.5], "brightness_range": [0.75, 1.25], "noise_sigma": 0.05}}},
        {"loss": {"class_weights": "auto"}},
        {"training": {"sub_volume": {"enabled": True, "size": [32, 256, 256]}}},
    ],
}

PHASES = {
    "1": PHASE_1_BASELINE,
    "2": PHASE_2_DECODER,
    "3": PHASE_3_FINETUNE,
    "4": PHASE_4_ABLATION,
}

# ──────────────────────────────────────────────
# Config generation
# ──────────────────────────────────────────────

BASE_CONFIG_TEMPLATE = {
    "model": {
        "model_path": "./models/dinov3-vitb16",
        "num_classes": 2,
        "out_indices": [2, 5, 8, 11],
        "slice_axis": 2,
        "input_normalization": "imagenet",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    },
    "finetune": {
        "method": "frozen",
        "lora_rank": 8,
        "lora_alpha": 16,
        "adapter_bottleneck": 64,
        "adapter_position": "after_attn",
    },
    "decoder": {
        "type": "segformer3d",
    },
    "data": {
        "name": "totalseg",
        "data_root": "./data",
        "img_size": [224, 224],
        "in_channels": 1,
        "modality": "ct",
        "k_shot": 5,
        "fold": 0,
    },
    "training": {
        "device": "auto",
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "epochs": 50,
        "batch_size": 1,
        "grad_accumulation": 2,
        "mixed_precision": False,
        "sub_volume": {"enabled": False},
        "keep_last_checkpoints": 2,
        "seed": 42,
        "num_workers": 0,
    },
    "loss": {
        "type": "dice_ce",
        "dice_weight": 0.5,
        "ce_weight": 0.5,
    },
    "augmentation": {
        "enabled": False,
    },
    "runtime": {
        "status_interval_seconds": 2.0,
    },
}


def generate_config(
    organ: str,
    k_shot: int,
    img_size: int,
    finetune: str,
    decoder: str,
    backbone: str = "vitb16",
    epochs: int = 50,
    seed: int = 42,
    overrides: Optional[Dict] = None,
    data_root: Optional[str] = None,
) -> Dict:
    """Generate a YAML config dict for one experiment."""
    cfg = json.loads(json.dumps(BASE_CONFIG_TEMPLATE))  # deep copy

    cfg["model"]["model_path"] = BACKBONES[backbone]["path"]
    cfg["finetune"]["method"] = finetune
    cfg["decoder"]["type"] = decoder
    cfg["data"]["img_size"] = [img_size, img_size]
    cfg["data"]["k_shot"] = k_shot
    cfg["data"]["data_root"] = data_root or str(DATA_BASE / "totalseg" / organ)
    cfg["data"]["name"] = f"totalseg_{organ}"
    cfg["training"]["epochs"] = epochs
    cfg["training"]["seed"] = seed

    if finetune == "full":
        cfg["model"]["freeze"] = False
    if finetune == "lora":
        cfg["finetune"]["lora_rank"] = 8
        cfg["finetune"]["lora_alpha"] = 16

    if overrides:
        _deep_update(cfg, overrides)

    return cfg


def _deep_update(base: Dict, update: Dict) -> Dict:
    """Deep merge update into base."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ──────────────────────────────────────────────
# Experiment grid expansion
# ──────────────────────────────────────────────

def expand_grid(phase_def: Dict) -> List[Dict]:
    """Expand a phase definition into a list of experiment specs."""
    grid = phase_def["grid"]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    experiments = []
    for combo in itertools.product(*values):
        exp = dict(zip(keys, combo))
        exp["epochs"] = phase_def["epochs"]
        exp["phase_name"] = phase_def["name"]
        exp["exp_id"] = _exp_id(exp)
        experiments.append(exp)

    # Handle repeats
    if phase_def.get("repeats", 1) > 1:
        repeated = []
        for i in range(phase_def["repeats"]):
            for exp in experiments:
                e = dict(exp)
                e["seed"] = 42 + i * 100
                e["exp_id"] = f"{_exp_id(exp)}_run{i}"
                repeated.append(e)
        experiments = repeated

    return experiments


def _exp_id(exp: Dict) -> str:
    """Generate a unique experiment ID."""
    parts = [
        exp.get("organ", "?"),
        exp.get("finetune", "?"),
        exp.get("decoder", "?"),
        f'k{exp.get("k_shot", "?")}',
        f'sz{exp.get("img_size", "?")}',
    ]
    return "_".join(str(p) for p in parts)


# ──────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────

class ExperimentRunner:
    """Run experiments with retry logic, logging, and results collection."""

    MAX_RETRIES = 3
    RETRY_DELAY = 30  # seconds

    def __init__(self, results_path: str, dry_run: bool = False):
        self.results_path = Path(results_path)
        self.dry_run = dry_run
        self.results = self._load_results()
        self.stats = {"total": 0, "completed": 0, "failed": 0, "skipped": 0}

    def _load_results(self) -> Dict:
        if self.results_path.exists():
            with open(self.results_path) as f:
                return json.load(f)
        return {"experiments": {}, "meta": {"created": datetime.now().isoformat()}}

    def _save_results(self):
        self.results["meta"]["updated"] = datetime.now().isoformat()
        self.results["meta"]["stats"] = self.stats
        with open(self.results_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def run_phase(self, phase_key: str, data_available: bool = True):
        """Run all experiments in a phase."""
        phase = PHASES[phase_key]
        experiments = expand_grid(phase)
        print(f"\n{'='*70}")
        print(f"  {phase['name']}")
        print(f"  {phase['description']}")
        print(f"  Total experiments: {len(experiments)}")
        print(f"{'='*70}\n")

        for i, exp in enumerate(experiments):
            self.stats["total"] += 1

            # Skip if already completed
            exp_key = exp["exp_id"]
            if exp_key in self.results.get("experiments", {}):
                prev = self.results["experiments"][exp_key]
                if prev.get("status") == "completed" and prev.get("exit_code") == 0:
                    print(f"[{i+1}/{len(experiments)}] {exp_key} — already completed, skipping")
                    self.stats["skipped"] += 1
                    continue

            print(f"[{i+1}/{len(experiments)}] {exp_key}")
            result = self.run_experiment(exp, data_available)
            self.results.setdefault("experiments", {})[exp_key] = result

            if result["status"] == "completed" and result["exit_code"] == 0:
                self.stats["completed"] += 1
                # Extract metrics
                metrics = self._collect_metrics(exp_key, exp)
                if metrics:
                    result["metrics"] = metrics
            else:
                self.stats["failed"] += 1

            self._save_results()

        self._print_summary()

    def run_experiment(self, exp: Dict, data_available: bool = True) -> Dict:
        """Run a single experiment with retries."""
        if not data_available:
            return {"status": "skipped", "reason": "data not available", "timestamp": datetime.now().isoformat()}

        # Generate config
        organ = exp["organ"]
        config = generate_config(
            organ=organ,
            k_shot=exp["k_shot"],
            img_size=exp["img_size"],
            finetune=exp["finetune"],
            decoder=exp["decoder"],
            backbone=exp.get("backbone", "vitb16"),
            epochs=exp.get("epochs", 50),
            seed=exp.get("seed", 42),
        )

        exp_dir = RESULTS_DIR / organ / exp["exp_id"]
        config_path = exp_dir / "config.yaml"
        checkpoint_dir = exp_dir / "checkpoints"

        if self.dry_run:
            print(f"  [DRY RUN] Would run: {exp['exp_id']}")
            print(f"    Config: {config_path}")
            print(f"    Decoder: {exp['decoder']}, Finetune: {exp['finetune']}, "
                  f"K-shot: {exp['k_shot']}, Img: {exp['img_size']}")
            return {"status": "dry_run", "timestamp": datetime.now().isoformat()}

        # Create directories
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Check if data exists
        data_root = config["data"]["data_root"]
        if not os.path.isdir(data_root):
            return {"status": "skipped", "reason": f"data root not found: {data_root}",
                    "timestamp": datetime.now().isoformat()}

        # Run training with retries
        train_script = SCRIPTS_DIR / "train.py"

        for attempt in range(self.MAX_RETRIES):
            print(f"  Attempt {attempt + 1}/{self.MAX_RETRIES}...")
            start_time = time.time()

            try:
                result = subprocess.run(
                    [sys.executable, str(train_script),
                     "--config", str(config_path)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout per experiment
                )

                elapsed = time.time() - start_time

                if result.returncode == 0:
                    return {
                        "status": "completed",
                        "exit_code": 0,
                        "elapsed_seconds": elapsed,
                        "attempts": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Check for retryable errors
                stderr = result.stderr.lower()
                if "cuda" in stderr or "out of memory" in stderr or "connection" in stderr:
                    print(f"  Retryable error (attempt {attempt + 1}): {result.stderr[:200]}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                        continue

                return {
                    "status": "failed",
                    "exit_code": result.returncode,
                    "elapsed_seconds": elapsed,
                    "attempts": attempt + 1,
                    "stderr": result.stderr[-1000:],
                    "stdout_tail": result.stdout[-500:],
                    "timestamp": datetime.now().isoformat(),
                }

            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "elapsed_seconds": 7200,
                    "attempts": attempt + 1,
                    "timestamp": datetime.now().isoformat(),
                }

        return {"status": "failed", "reason": "max retries exceeded",
                "timestamp": datetime.now().isoformat()}

    def _collect_metrics(self, exp_key: str, exp: Dict) -> Optional[Dict]:
        """Collect metrics from experiment output."""
        metrics_path = RESULTS_DIR / exp["organ"] / exp_key / "checkpoints"
        best_model = metrics_path / "best_model.pth"
        if best_model.exists():
            try:
                checkpoint = __import__("torch").load(str(best_model), map_location="cpu", weights_only=False)
                ckpt_metrics = checkpoint.get("metrics", {})
                return {
                    "best_dsc": ckpt_metrics.get("dsc", ckpt_metrics.get("dice", None)),
                    "best_epoch": checkpoint.get("epoch", None),
                }
            except Exception:
                pass
        return None

    def _print_summary(self):
        print(f"\n{'='*70}")
        print(f"  Summary: {self.stats['total']} total | "
              f"{self.stats['completed']} completed | "
              f"{self.stats['failed']} failed | "
              f"{self.stats['skipped']} skipped")
        print(f"  Results saved to: {self.results_path}")
        print(f"{'='*70}\n")


# ──────────────────────────────────────────────
# Results analysis
# ──────────────────────────────────────────────

def analyze_results(results_path: str):
    """Print a summary table of results."""
    with open(results_path) as f:
        data = json.load(f)

    experiments = data.get("experiments", {})
    if not experiments:
        print("No results found.")
        return

    # Build summary table
    rows = []
    for exp_id, result in sorted(experiments.items()):
        parts = exp_id.split("_")
        organ = parts[0] if parts else "?"
        metrics = result.get("metrics", {}) or {}
        dsc = metrics.get("best_dsc", None)
        rows.append({
            "exp_id": exp_id,
            "organ": organ,
            "status": result.get("status", "?"),
            "best_dsc": dsc,
            "elapsed": result.get("elapsed_seconds", 0),
        })

    # Print table
    print(f"\n{'Experiment':<50s} {'Organ':<15s} {'Status':<12s} {'Best DSC':<10s} {'Time'}")
    print("-" * 110)
    for row in rows:
        dsc_str = f"{row['best_dsc']:.4f}" if row['best_dsc'] else "N/A"
        time_str = f"{row['elapsed']/60:.0f}m" if row['elapsed'] else "N/A"
        print(f"{row['exp_id']:<50s} {row['organ']:<15s} {row['status']:<12s} {dsc_str:<10s} {time_str}")

    # Per-organ summary
    print(f"\n{'Organ':<20s} {'Completed':<12s} {'Failed':<8s} {'Best DSC':<10s}")
    print("-" * 55)
    for organ in sorted(set(r["organ"] for r in rows)):
        org_rows = [r for r in rows if r["organ"] == organ]
        completed = sum(1 for r in org_rows if r["status"] == "completed")
        failed = sum(1 for r in org_rows if r["status"] == "failed")
        dscs = [r["best_dsc"] for r in org_rows if r["best_dsc"] is not None]
        best = f"{max(dscs):.4f}" if dscs else "N/A"
        print(f"{organ:<20s} {completed:<12d} {failed:<8d} {best:<10s}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for DINOv3 few-shot fine-tuning"
    )
    parser.add_argument("--phase", choices=["1", "2", "3", "4"],
                        help="Experiment phase to run")
    parser.add_argument("--organ", choices=list(ORGANS.keys()),
                        help="Run single organ")
    parser.add_argument("--decoder", choices=ALL_DECODERS,
                        help="Decoder type")
    parser.add_argument("--finetune", choices=FINETUNE_METHODS,
                        help="Fine-tuning method")
    parser.add_argument("--k-shot", type=int, choices=K_SHOTS,
                        help="Number of training samples")
    parser.add_argument("--img-size", type=int, choices=IMG_SIZES,
                        help="Image size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--resume", type=str,
                        help="Resume from results file")
    parser.add_argument("--analyze", type=str,
                        help="Analyze results file")
    parser.add_argument("--results", type=str, default="results.json",
                        help="Results file path")
    parser.add_argument("--data-check", action="store_true",
                        help="Check data availability for all organs")
    args = parser.parse_args()

    # Analyze mode
    if args.analyze:
        analyze_results(args.analyze)
        return

    # Data check mode
    if args.data_check:
        check_data_availability()
        return

    # Runner setup
    results_path = args.resume or args.results
    runner = ExperimentRunner(results_path, dry_run=args.dry_run)

    # Single experiment
    if args.organ:
        exp = {
            "organ": args.organ,
            "k_shot": args.k_shot or 5,
            "img_size": args.img_size or 224,
            "finetune": args.finetune or "frozen",
            "decoder": args.decoder or "segformer3d",
            "backbone": "vitb16",
            "epochs": args.epochs,
            "seed": 42,
            "phase_name": "manual",
            "exp_id": _exp_id({
                "organ": args.organ,
                "finetune": args.finetune or "frozen",
                "decoder": args.decoder or "segformer3d",
                "k_shot": args.k_shot or 5,
                "img_size": args.img_size or 224,
            }),
        }
        result = runner.run_experiment(exp)
        runner.results.setdefault("experiments", {})[exp["exp_id"]] = result
        runner._save_results()
        print(json.dumps(result, indent=2))
        return

    # Phase run
    if args.phase:
        data_ok = all(
            os.path.isdir(str(DATA_BASE / "totalseg" / o))
            for o in ORGANS
        )
        if not data_ok:
            missing = [o for o in ORGANS if not os.path.isdir(str(DATA_BASE / "totalseg" / o))]
            print(f"WARNING: Data not available for: {missing}")
            print("Run with --data-check for details.")
        runner.run_phase(args.phase, data_available=data_ok)
        return

    # Default: show status
    parser.print_help()


def check_data_availability():
    """Check if data is available for all target organs."""
    ts_base = DATA_BASE / "totalseg"

    print(f"\nData availability check:")
    print(f"  Base: {ts_base}")
    print(f"  Exists: {ts_base.is_dir()}")
    print()

    if not ts_base.is_dir():
        print("  TotalSegmentator data not found. To prepare data:")
        print(f"    1. Download dataset to {DATA_BASE}/totalseg/source/")
        print(f"    2. Run: python scripts/materialize_totaltest.py --organ <organ>")
        return

    for organ, info in ORGANS.items():
        organ_dir = ts_base / organ
        images_tr = organ_dir / "imagesTr"
        labels_tr = organ_dir / "labelsTr"

        n_images = len(list(images_tr.glob("*.nii.gz"))) if images_tr.is_dir() else 0
        n_labels = len(list(labels_tr.glob("*.nii.gz"))) if labels_tr.is_dir() else 0

        status = "✅" if n_images > 0 and n_labels > 0 else "❌"
        print(f"  {status} {organ:<25s}: {n_images} images, {n_labels} labels  "
              f"({info['size_category']}, {info['difficulty']})")


if __name__ == "__main__":
    main()
