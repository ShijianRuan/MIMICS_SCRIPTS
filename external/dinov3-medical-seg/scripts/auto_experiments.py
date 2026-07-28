#!/usr/bin/env python3
"""
Fully automated DINOv3 few-shot experiment pipeline for Mac MPS.

Runs experiments sequentially, auto-optimizes for local hardware,
collects results, and generates reports — all without human intervention.

Strategy (optimized for Mac MPS speed):
  - 30 epochs (enough for convergence on few-shot)
  - fp32 only (MPS doesn't support bf16/fp16 autograd well)
  - img_size=224 (2x faster than 512)
  - No sub-volume (memory-constrained)
  - Sequential execution (one model at a time on MPS)

Usage:
  python scripts/auto_experiments.py          # Run all phases
  python scripts/auto_experiments.py --phase 1 # Phase 1 only
  python scripts/auto_experiments.py --dry-run # Print plan only
"""

import argparse, json, os, subprocess, sys, time, yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).resolve().parent.parent
DATA_BASE = PROJECT / "data" / "totalseg"
RESULTS_FILE = PROJECT / "experiments" / "auto_results.json"

# ── Organs ──
ORGANS = {
    "brain":              {"fg": 297957, "diff": "easy"},
    "liver":              {"fg": 279546, "diff": "medium"},
    "aorta":              {"fg": 33935,  "diff": "hard"},
    "scapula_left":       {"fg": 16456,  "diff": "hard"},
    "adrenal_gland_right":{"fg": 507,    "diff": "extreme"},
}

# ── Experiment Grid ──
PHASES = {
    "1_baseline": {
        "desc": "Baseline: Frozen + SegFormer3D, vary k-shot",
        "grid": {
            "organ": list(ORGANS),
            "finetune": ["frozen"],
            "decoder": ["segformer3d"],
            "k_shot": [1, 3, 5],
            "img_size": [224],
        },
        "epochs": 30,
    },
    "2_decoder": {
        "desc": "Decoder comparison: all 8 decoders, k=5",
        "grid": {
            "organ": list(ORGANS),
            "finetune": ["frozen"],
            "decoder": [
                "conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d",
                "linear3d", "mlp_probe", "segformer3d", "dpt3d",
            ],
            "k_shot": [5],
            "img_size": [224],
        },
        "epochs": 30,
    },
    "3_finetune": {
        "desc": "Finetune method: frozen vs lora vs adapter vs full, k=5",
        "grid": {
            "organ": list(ORGANS),
            "finetune": ["frozen", "lora", "adapter", "full"],
            "decoder": ["segformer3d"],
            "k_shot": [5],
            "img_size": [224],
        },
        "epochs": 30,
    },
    "4_loss": {
        "desc": "Loss function ablation on hardest organs",
        "grid": {
            "organ": ["aorta", "scapula_left", "adrenal_gland_right"],
            "finetune": ["frozen"],
            "decoder": ["segformer3d", "conv2d"],
            "k_shot": [5],
            "img_size": [224],
            "loss": ["dice_ce", "focal", "tversky", "dice_focal"],
        },
        "epochs": 30,
    },
}


def expand_grid(grid: Dict) -> List[Dict]:
    """Expand a param grid into experiment list."""
    import itertools
    keys, values = list(grid.keys()), list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def exp_key(exp: Dict) -> str:
    """Unique experiment ID."""
    return (f"{exp['organ']}_{exp.get('finetune','frozen')}_"
            f"{exp['decoder']}_k{exp['k_shot']}_sz{exp['img_size']}"
            + (f"_{exp['loss']}" if exp.get('loss') else ""))


def run_experiment(exp: Dict) -> Dict:
    """Run one experiment, return result dict."""
    key = exp_key(exp)
    organ = exp["organ"]
    data_root = str(DATA_BASE / organ)

    if not os.path.isdir(data_root):
        return {"key": key, "status": "skip", "reason": "no data"}

    # Build config
    cfg = {
        "_base_": ["train.yaml"],
        "model": {
            "model_path": "./models/dinov3-vitb16",
            "num_classes": 2,
            "input_normalization": "imagenet",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "out_indices": [2, 5, 8, 11],
            "slice_axis": 2,
        },
        "finetune": {
            "method": exp.get("finetune", "frozen"),
            "lora_rank": 8, "lora_alpha": 16,
            "adapter_bottleneck": 64, "adapter_position": "after_attn",
        },
        "decoder": {"type": exp["decoder"]},
        "data": {
            "name": f"totalseg_{organ}",
            "data_root": data_root,
            "img_size": [exp["img_size"], exp["img_size"]],
            "in_channels": 1,
            "modality": "ct",
            "k_shot": exp["k_shot"],
            "fold": 0,
        },
        "training": {
            "device": "auto",
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "epochs": exp.get("epochs", 30),
            "batch_size": 1,
            "grad_accumulation": 2,
            "mixed_precision": False,
            "sub_volume": {"enabled": False},
            "keep_last_checkpoints": 1,
            "seed": 42,
            "num_workers": 0,
        },
        "loss": {
            "type": exp.get("loss", "dice_ce"),
            "dice_weight": 0.5, "ce_weight": 0.5,
            "focal_alpha": 0.25, "focal_gamma": 2.0,
            "tversky_alpha": 0.3, "tversky_beta": 0.7,
        },
        "augmentation": {"enabled": False},
        "runtime": {"status_interval_seconds": 2.0},
    }

    exp_dir = PROJECT / "experiments" / organ / key
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    cfg_path = exp_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"  [{key}] training...", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT / "scripts" / "train.py"),
             "--config", str(cfg_path)],
            cwd=str(PROJECT),
            capture_output=True, text=True,
            timeout=3600,  # 1 hour max per experiment
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            # Extract best DSC from checkpoint
            best_path = exp_dir / "checkpoints" / "best_model.pth"
            best_dsc = None
            if best_path.exists():
                import torch
                ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
                m = ckpt.get("metrics", {}) or {}
                best_dsc = m.get("dsc", m.get("dice"))

            return {
                "key": key, "status": "ok", "elapsed": elapsed,
                "best_dsc": best_dsc, "epoch": ckpt.get("epoch") if best_path.exists() else None,
            }
        else:
            return {
                "key": key, "status": "failed",
                "elapsed": elapsed, "exit_code": result.returncode,
                "stderr_tail": result.stderr[-300:] if result.stderr else "",
            }
    except subprocess.TimeoutExpired:
        return {"key": key, "status": "timeout", "elapsed": 3600}
    except Exception as e:
        return {"key": key, "status": "error", "error": str(e)[:200]}


def print_summary(all_results: Dict):
    """Print experiment result summary."""
    exps = all_results.get("experiments", {})
    ok = sum(1 for v in exps.values() if v.get("status") == "ok")
    fail = sum(1 for v in exps.values() if v.get("status") == "failed")
    skip = sum(1 for v in exps.values() if v.get("status") == "skip")

    # Per-organ best
    print(f"\n{'='*80}")
    print(f"  {ok} ok | {fail} failed | {skip} skipped | {len(exps)} total")
    print(f"{'='*80}")
    print(f"{'Organ':<25s} {'Best DSC':<12s} {'Config'}")
    print("-" * 80)

    for organ in sorted(ORGANS):
        org_exps = {k: v for k, v in exps.items() if k.startswith(organ)}
        best = max(
            (v for v in org_exps.values() if v.get("best_dsc") is not None),
            key=lambda x: x.get("best_dsc", -1), default=None
        )
        if best:
            print(f"{organ:<25s} {best['best_dsc']:.4f}        {best['key']}")


def main():
    parser = argparse.ArgumentParser(description="Auto DINOv3 few-shot experiments")
    parser.add_argument("--phase", choices=list(PHASES), help="Phase to run (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    phases = [args.phase] if args.phase else list(PHASES)

    # Load existing results
    all_results = {"experiments": {}, "meta": {"started": datetime.now().isoformat()}}
    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

    total_exps = 0

    for phase_key in phases:
        phase = PHASES[phase_key]
        experiments = expand_grid(phase["grid"])

        # Add epoch count
        for e in experiments:
            e["epochs"] = phase["epochs"]

        print(f"\n{'='*80}")
        print(f"  Phase: {phase_key} — {phase['desc']}")
        print(f"  Experiments: {len(experiments)}")
        print(f"{'='*80}")

        if args.dry_run:
            for e in experiments:
                print(f"  {exp_key(e)}")
            continue

        for i, exp in enumerate(experiments):
            key = exp_key(exp)

            # Skip if already done
            if key in all_results["experiments"]:
                prev = all_results["experiments"][key]
                if prev.get("status") == "ok":
                    print(f"[{i+1}/{len(experiments)}] {key} — already done, skip")
                    continue

            print(f"[{i+1}/{len(experiments)}] {key}")
            result = run_experiment(exp)
            all_results["experiments"][key] = result

            # Save after each experiment
            all_results["meta"]["updated"] = datetime.now().isoformat()
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

            status = result.get("status", "?")
            dsc = result.get("best_dsc")
            dsc_str = f" DSC={dsc:.4f}" if dsc else ""
            elapsed = result.get("elapsed", 0)
            print(f"    → {status}{dsc_str} ({elapsed/60:.1f}m)", flush=True)

    print_summary(all_results)
    print(f"\nResults: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
