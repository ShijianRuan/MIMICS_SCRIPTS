#!/usr/bin/env python3
"""Train and evaluate one coarse-to-fine candidate on a materialized fold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.fingerprint import build_training_fingerprint, derive_policy
from src.research.protocol import atomic_json
from src.utils.config import deep_merge, load_config


def _write_config(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run(command, log_path):
    print("$ {}".format(" ".join(str(value) for value in command)))
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def _localization_gate(metrics, minimum_detection=0.80, minimum_success=0.67, minimum_coverage=0.80):
    checks = {
        "coarse_detection_rate": (float(metrics.get("coarse_detection_rate", 0.0)), minimum_detection),
        "roi_localization_success_rate": (
            float(metrics.get("roi_localization_success_rate", 0.0)), minimum_success
        ),
        "mean_roi_target_coverage": (
            float(metrics.get("mean_roi_target_coverage", 0.0)), minimum_coverage
        ),
    }
    failures = [
        "{}={:.3f} < {:.3f}".format(name, observed, required)
        for name, (observed, required) in checks.items()
        if observed < required
    ]
    return {"passed": not failures, "checks": checks, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description="Run a trainable coarse-to-fine DINOv3 experiment")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--support-count", type=int, default=5)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fine-method", choices=["frozen", "lora", "adapter"], default="frozen")
    parser.add_argument("--training-seed", type=int, help="Override the reproducible coarse/fine seed")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output = Path(args.output_root).resolve()
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    support_ids = manifest["selection"]["support_case_ids_ordered"][:args.support_count]
    if len(support_ids) != args.support_count:
        raise SystemExit("Requested support count exceeds this fold's support pool")
    base = load_config(args.base_config, {})
    fingerprint = build_training_fingerprint(data_root, support_ids, args.task)
    policy = derive_policy(fingerprint)
    if not policy["two_stage_candidate"]:
        print("Warning: fingerprint does not classify this target as tiny; this is a controlled cascade ablation.")

    common = {
        "data": {"data_root": str(data_root), "support_case_ids": support_ids, "k_shot": args.support_count},
        "training": {"experiment_root": str(output / "artifacts")},
    }
    if args.training_seed is not None:
        common["training"]["seed"] = int(args.training_seed)
    coarse = deep_merge(base, common)
    coarse = deep_merge(coarse, {
        "exp_name": "two_stage/coarse",
        "model": {"channel_policy": "2_5d", "neighbor_distance_mm": 3.0, "slice_batch_size": 1},
        "decoder": {"type": "segformer3d"},
        "finetune": {"method": "frozen"},
        "data": {
            "img_size": [256, 256],
            "patch": {"enabled": False},
            "target": {"mode": "localization_ball", "radius_mm_zyx": [24.0, 40.0, 40.0]},
        },
        "loss": {"type": "dice_focal", "dice_weight": 0.7, "focal_weight": 0.3,
                 "focal_alpha": 0.75, "focal_gamma": 2.0},
        "inference": {"keep_largest_component": True},
        "training": {"epochs": 30, "validation_interval": 2,
                     "early_stopping": {"min_epochs": 8, "patience": 4, "min_delta": 0.001}},
    })
    fine = deep_merge(base, common)
    fine = deep_merge(fine, {
        "exp_name": "two_stage/fine",
        "model": {"channel_policy": "2_5d", "neighbor_distance_mm": 3.0,
                  "slice_batch_size": 1},
        "decoder": {"type": "segformer3d"},
        "finetune": {"method": args.fine_method},
        "data": {"img_size": [384, 384], "patch": dict(policy["patch"], **{
            "patches_per_case_per_epoch": 8,
            "sampling": {"interior": 0.30, "boundary": 0.30, "near_negative": 0.25,
                         "random": 0.15, "boundary_width": 2, "near_negative_width": 8},
        })},
        # Two extremes observed: dice_focal collapsed to all-background
        # (recall~0); tversky 0.2/0.8 swung to all-foreground (precision~0.003,
        # recall 0.98). The model CAN place foreground correctly, so the fix is
        # balance, not more recall pressure. Symmetric Tversky (0.5/0.5 = Dice)
        # penalises the massive false positives while keeping the escape from
        # the all-background solution.
        "loss": {"type": "tversky", "tversky_alpha": 0.5, "tversky_beta": 0.5},
        # Validation Dice is a whole-volume argmax metric that stays ~0 for a
        # needle target long after training-loss starts descending, so a
        # val-Dice patience would early-stop a model that is still learning
        # (observed: loss 0.705->0.598 while val_dice=0 until an epoch-10 stop).
        # Run the full budget and judge on the search-ROI two-stage evaluation.
        "training": {"epochs": 60, "validation_interval": 2,
                     "early_stopping": {"min_epochs": 60, "patience": 60, "min_delta": 0.0}},
    })
    output.mkdir(parents=True, exist_ok=True)
    _write_config(output / "coarse.yaml", coarse)
    _write_config(output / "fine.yaml", fine)
    atomic_json(output / "training_fingerprint.json", fingerprint)
    atomic_json(output / "derived_data_policy.json", policy)
    coarse_checkpoint = output / "artifacts" / "two_stage" / "coarse" / "checkpoints" / "best_model.pth"
    returncode = _run(
        [sys.executable, "scripts/train.py", "--config", str(output / "coarse.yaml")],
        output / "coarse.log",
    )
    if returncode:
        atomic_json(output / "failed.json", {"stage": "coarse", "returncode": int(returncode)})
        raise SystemExit("coarse stage failed; inspect {}".format(output / "coarse.log"))

    returncode = _run([
        sys.executable,
        "scripts/research/evaluate_localizer.py",
        "--coarse-config", str(output / "coarse.yaml"),
        "--coarse-checkpoint", str(coarse_checkpoint),
        "--fine-config", str(output / "fine.yaml"),
        "--data-root", str(data_root),
        "--output", str(output / "localization_evaluation.json"),
    ], output / "localization_evaluation.log")
    if returncode:
        atomic_json(output / "failed.json", {"stage": "localization_evaluation", "returncode": int(returncode)})
        raise SystemExit("Coarse-localizer evaluation failed; inspect {}".format(output / "localization_evaluation.log"))
    localization_metrics = json.loads((output / "localization_evaluation.json").read_text(encoding="utf-8"))
    gate = _localization_gate(localization_metrics)
    atomic_json(output / "localization_gate.json", gate)
    if not gate["passed"]:
        atomic_json(output / "failed.json", {"stage": "localization_gate", "failures": gate["failures"]})
        raise SystemExit("Coarse localizer did not pass the ROI gate: {}".format("; ".join(gate["failures"])))

    returncode = _run(
        [sys.executable, "scripts/train.py", "--config", str(output / "fine.yaml")],
        output / "fine.log",
    )
    if returncode:
        atomic_json(output / "failed.json", {"stage": "fine", "returncode": int(returncode)})
        raise SystemExit("fine stage failed; inspect {}".format(output / "fine.log"))
    fine_checkpoint = output / "artifacts" / "two_stage" / "fine" / "checkpoints" / "best_model.pth"
    returncode = _run([
        sys.executable,
        "scripts/research/evaluate_two_stage.py",
        "--coarse-config", str(output / "coarse.yaml"),
        "--coarse-checkpoint", str(coarse_checkpoint),
        "--fine-config", str(output / "fine.yaml"),
        "--fine-checkpoint", str(fine_checkpoint),
        "--data-root", str(data_root),
        "--output", str(output / "evaluation.json"),
    ], output / "evaluation.log")
    if returncode:
        atomic_json(output / "failed.json", {"stage": "evaluation", "returncode": int(returncode)})
        raise SystemExit("Two-stage evaluation failed; inspect {}".format(output / "evaluation.log"))


if __name__ == "__main__":
    main()
