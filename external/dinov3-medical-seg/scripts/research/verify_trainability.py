#!/usr/bin/env python3
"""Verify that a configured DINOv3 adaptation path receives real gradients.

This is a preflight, not a performance experiment.  It catches the class of
bug where a pseudo-3D encoder detaches features and makes LoRA/adapter/full
rows look like legitimate fine-tuning despite never updating the backbone.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentor import DINOv33DSegmentor
from src.utils.config import deep_merge, load_config
from src.utils.device import get_device


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradient_norm(parameters) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm().cpu()) if values else 0.0


def verify(config: dict, method: str, device: torch.device, size: int, depth: int) -> dict:
    _seed(int(config.get("training", {}).get("seed", 20260711)))
    runtime = deep_merge(config, {
        "finetune": {"method": method},
        "data": {"img_size": [size, size]},
        "model": {"slice_batch_size": 1},
    })
    model = DINOv33DSegmentor(runtime).to(device)
    model.train()
    volume = torch.rand(1, 1, depth, size, size, device=device)
    logits = model(volume, spacing_zyx=torch.ones(1, 3, device=device))
    logits[:, 1].mean().backward()
    backbone_parameters = list(model.backbone.named_parameters())
    lora_parameters = [parameter for name, parameter in model.named_parameters() if "lora_A" in name or "lora_B" in name]
    adapter_parameters = [parameter for name, parameter in model.named_parameters() if "_adapter_" in name]
    report = {
        "method": method,
        "device": str(device),
        "input_shape": list(volume.shape),
        "backbone_gradient_norm": _gradient_norm([parameter for _, parameter in backbone_parameters]),
        "lora_gradient_norm": _gradient_norm(lora_parameters),
        "adapter_gradient_norm": _gradient_norm(adapter_parameters),
        "decoder_gradient_norm": _gradient_norm(model.decoder_3d.parameters()),
        "trainable_parameters": model.get_trainable_info()["trainable"],
    }
    if report["decoder_gradient_norm"] <= 0.0:
        raise RuntimeError("Decoder did not receive a gradient")
    if method == "frozen" and report["backbone_gradient_norm"] != 0.0:
        raise RuntimeError("Frozen backbone unexpectedly received a gradient")
    if method == "lora" and report["lora_gradient_norm"] <= 0.0:
        raise RuntimeError("LoRA parameters did not receive a gradient")
    if method == "adapter" and report["adapter_gradient_norm"] <= 0.0:
        raise RuntimeError("Adapter parameters did not receive a gradient")
    if method == "full" and report["backbone_gradient_norm"] <= 0.0:
        raise RuntimeError("Full backbone did not receive a gradient")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-step DINOv3 PEFT gradient preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", action="append", choices=["frozen", "lora", "adapter", "full"])
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--model-path", help="Override model.model_path for an isolated worker")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.size <= 0 or args.depth <= 0:
        raise SystemExit("--size and --depth must be positive")
    config = load_config(args.config, {})
    if args.model_path:
        config.setdefault("model", {})["model_path"] = str(Path(args.model_path).resolve())
    device = torch.device(args.device) if args.device else get_device()
    methods = args.method or [str(config.get("finetune", {}).get("method", "frozen"))]
    rows = [verify(config, method, device, args.size, args.depth) for method in methods]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
