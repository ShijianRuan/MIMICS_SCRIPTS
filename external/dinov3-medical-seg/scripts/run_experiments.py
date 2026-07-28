#!/usr/bin/env python3
"""
Minimal high-impact experiments for DINOv3 few-shot fine-tuning.
12 experiments, ~3 hours on Mac MPS.

Key questions answered:
  Q1: 2D vs 3D decoder — liver (large), aorta (tubular), adrenal (tiny)
  Q2: K-shot scaling — brain (easy) & liver (medium) @ 1,3,5
  Q3: Focal loss — adrenal (extreme 0.004% foreground)
"""

import json, os, subprocess, sys, time, yaml, torch
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# ── Focused experiment list ──
EXPERIMENTS = [
    # Q1: 2D vs 3D on liver (large), aorta (tubular), adrenal (tiny)
    {"organ": "liver",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 5, "loss": "dice_ce"},
    {"organ": "liver",              "finetune": "frozen", "decoder": "conv2d",      "k_shot": 5, "loss": "dice_ce"},
    {"organ": "aorta",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 5, "loss": "dice_ce"},
    {"organ": "aorta",              "finetune": "frozen", "decoder": "conv2d",      "k_shot": 5, "loss": "dice_ce"},
    {"organ": "adrenal_gland_right","finetune": "frozen", "decoder": "segformer3d", "k_shot": 5, "loss": "dice_ce"},
    {"organ": "adrenal_gland_right","finetune": "frozen", "decoder": "conv2d",      "k_shot": 5, "loss": "dice_ce"},

    # Q2: K-shot scaling — brain (easy) & liver (medium)
    {"organ": "brain",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 1, "loss": "dice_ce"},
    {"organ": "brain",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 3, "loss": "dice_ce"},
    {"organ": "brain",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 5, "loss": "dice_ce"},
    {"organ": "liver",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 1, "loss": "dice_ce"},
    {"organ": "liver",              "finetune": "frozen", "decoder": "segformer3d", "k_shot": 3, "loss": "dice_ce"},

    # Q3: Focal loss on adrenal (extreme imbalance)
    {"organ": "adrenal_gland_right","finetune": "frozen", "decoder": "conv2d",      "k_shot": 5, "loss": "focal"},
]

RESULTS_FILE = PROJECT / "experiments" / "results.json"


def key(exp):
    return f"{exp['organ']}_{exp['decoder']}_k{exp['k_shot']}_{exp['loss']}"


def run(exp):
    k = key(exp)
    organ = exp["organ"]
    data_root = str(PROJECT / "data" / "totalseg" / organ)
    exp_dir = PROJECT / "experiments" / organ / k
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    cfg = {
        "model": {
            "model_path": "./models/dinov3-vitb16", "num_classes": 2,
            "input_normalization": "imagenet",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "out_indices": [2, 5, 8, 11], "slice_axis": 2,
        },
        "finetune": {"method": "frozen"},
        "decoder": {"type": exp["decoder"]},
        "data": {
            "data_root": data_root, "img_size": [224, 224],
            "modality": "ct", "k_shot": exp["k_shot"], "fold": 0,
        },
        "training": {
            "device": "auto", "optimizer": "adamw", "lr": 1.0e-3,
            "weight_decay": 0.01, "scheduler": "cosine", "warmup_epochs": 3,
            "epochs": 30, "batch_size": 1, "grad_accumulation": 2,
            "mixed_precision": False, "sub_volume": {"enabled": False},
            "seed": 42, "num_workers": 0,
        },
        "loss": {
            "type": exp["loss"], "dice_weight": 0.5, "ce_weight": 0.5,
            "focal_alpha": 0.25, "focal_gamma": 2.0,
        },
        "augmentation": {"enabled": False},
        "exp_name": k,  # Save to experiments/{key}/ instead of exp_{timestamp}
    }

    cfg_path = exp_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    t0 = time.time()
    print(f"  [{k}] training...", end=" ", flush=True)

    try:
        # Run training as background subprocess, stream output
        with open(exp_dir / "train.log", "w") as log:
            result = subprocess.run(
                [sys.executable, str(PROJECT / "scripts" / "train.py"), "--config", str(cfg_path)],
                cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT, timeout=3600,
            )
        elapsed = time.time() - t0

        if result.returncode != 0:
            return {"key": k, "status": "failed", "elapsed": elapsed,
                    "stderr": f"exit={result.returncode}, see train.log"}

        # Read metrics from best checkpoint (trainer saves to experiments/{exp_name}/)
        best_path = PROJECT / "experiments" / k / "checkpoints" / "best_model.pth"
        if not best_path.exists():
            best_path = exp_dir / "checkpoints" / "best_model.pth"

        best_dsc = None
        best_epoch = None
        if best_path.exists():
            ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            m = ckpt.get("metrics", {}) or {}
            best_dsc = m.get("mean_dsc", m.get("dsc", m.get("dice")))
            best_epoch = ckpt.get("epoch")

        return {"key": k, "status": "ok", "elapsed": elapsed,
                "best_dsc": best_dsc, "epoch": best_epoch}
    except subprocess.TimeoutExpired:
        return {"key": k, "status": "timeout"}
    except Exception as e:
        return {"key": k, "status": "error", "error": str(e)[:200]}


def print_table(results):
    """Print results grouped by question."""
    exps = results.get("experiments", {})

    print(f"\n{'='*90}")
    print("  Q1: 2D vs 3D Decoder (k=5, dice_ce)")
    print(f"{'='*90}")
    print(f"{'Organ':<25s} {'3D DSC':<12s} {'2D DSC':<12s} {'Winner':<10s}")
    print("-" * 60)
    for (o3, o2) in [("liver", "liver"), ("aorta", "aorta"), ("adrenal_gland_right", "adrenal_gland_right")]:
        k3 = f"{o3}_segformer3d_k5_dice_ce"
        k2 = f"{o2}_conv2d_k5_dice_ce"
        d3 = exps.get(k3, {}).get("best_dsc")
        d2 = exps.get(k2, {}).get("best_dsc")
        d3s = f"{d3:.4f}" if d3 else "N/A"
        d2s = f"{d2:.4f}" if d2 else "N/A"
        if d3 and d2:
            win = "2D ✅" if d2 > d3 else "3D ✅" if d3 > d2 else "Tie"
        else:
            win = "—"
        print(f"{o3:<25s} {d3s:<12s} {d2s:<12s} {win:<10s}")

    print(f"\n{'='*90}")
    print("  Q2: K-Shot Scaling (segformer3d, dice_ce)")
    print(f"{'='*90}")
    print(f"{'Organ':<25s} {'k=1':<12s} {'k=3':<12s} {'k=5':<12s}")
    print("-" * 60)
    for organ in ["brain", "liver"]:
        vals = []
        for k in [1, 3, 5]:
            d = exps.get(f"{organ}_segformer3d_k{k}_dice_ce", {}).get("best_dsc")
            vals.append(f"{d:.4f}" if d else "N/A")
        print(f"{organ:<25s} {vals[0]:<12s} {vals[1]:<12s} {vals[2]:<12s}")

    print(f"\n{'='*90}")
    print("  Q3: Focal Loss on Adrenal (k=5, conv2d)")
    print(f"{'='*90}")
    print(f"{'Method':<25s} {'DSC':<12s}")
    print("-" * 40)
    for loss in ["dice_ce", "focal"]:
        d = exps.get(f"adrenal_gland_right_conv2d_k5_{loss}", {}).get("best_dsc")
        print(f"{loss:<25s} {f'{d:.4f}' if d else 'N/A':<12s}")


def main():
    print(f"{'='*90}")
    print(f"  DINOv3 Few-Shot — 12 Core Experiments")
    print(f"  Estimated: ~3 hours on MPS")
    print(f"{'='*90}")

    # Load existing
    results = {"experiments": {}, "started": datetime.now().isoformat()}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done = sum(1 for v in results["experiments"].values() if v.get("status") == "ok")
        print(f"  Resuming: {done} already completed\n")

    for i, exp in enumerate(EXPERIMENTS):
        k = key(exp)

        if k in results["experiments"] and results["experiments"][k].get("status") == "ok":
            dsc = results["experiments"][k].get("best_dsc")
            print(f"[{i+1}/{len(EXPERIMENTS)}] {k} — done (DSC={dsc:.4f})" if dsc else f"[{i+1}/{len(EXPERIMENTS)}] {k} — done")
            continue

        print(f"[{i+1}/{len(EXPERIMENTS)}] {k}")
        result = run(exp)
        results["experiments"][k] = result

        results["updated"] = datetime.now().isoformat()
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        dsc = result.get("best_dsc")
        elapsed = result.get("elapsed", 0)
        dsc_s = f" DSC={dsc:.4f}" if dsc else ""
        print(f"    → {result['status']}{dsc_s} ({elapsed/60:.1f}m)", flush=True)

    print_table(results)
    print(f"\nResults saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
