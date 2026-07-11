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
        "model": {"channel_policy": "repeat", "slice_batch_size": 2},
        "decoder": {"type": "linear3d"},
        "finetune": {"method": "frozen"},
        "data": {"img_size": [224, 224], "patch": {"enabled": False}},
    })
    fine = deep_merge(base, common)
    fine = deep_merge(fine, {
        "exp_name": "two_stage/fine",
        "model": {"channel_policy": "repeat", "slice_batch_size": policy["slice_batch_size"]},
        "decoder": {"type": "segformer3d"},
        "finetune": {"method": args.fine_method},
        "data": {"img_size": policy["input_size"], "patch": policy["patch"]},
        "loss": {"type": "dice_focal", "dice_weight": 0.7, "focal_weight": 0.3, "focal_alpha": 0.75, "focal_gamma": 2.0},
    })
    output.mkdir(parents=True, exist_ok=True)
    _write_config(output / "coarse.yaml", coarse)
    _write_config(output / "fine.yaml", fine)
    atomic_json(output / "training_fingerprint.json", fingerprint)
    atomic_json(output / "derived_data_policy.json", policy)
    for name, config in (("coarse", coarse), ("fine", fine)):
        returncode = _run([sys.executable, "scripts/train.py", "--config", str(output / (name + ".yaml"))], output / (name + ".log"))
        if returncode:
            atomic_json(output / "failed.json", {"stage": name, "returncode": int(returncode)})
            raise SystemExit("{} stage failed; inspect {}".format(name, output / (name + ".log")))
    coarse_checkpoint = output / "artifacts" / "two_stage" / "coarse" / "checkpoints" / "best_model.pth"
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
