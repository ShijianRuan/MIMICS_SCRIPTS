"""Checkpoint saving/loading utilities."""

import os
import torch
from typing import Dict


def _cleanup_old_epoch_checkpoints(save_dir: str, keep_last: int):
    """Keep best_model.pth plus the latest N epoch checkpoints."""
    try:
        keep_last = int(keep_last)
    except Exception:
        keep_last = 0
    if keep_last < 0:
        return
    try:
        if not os.path.isfile(os.path.join(save_dir, "best_model.pth")):
            keep_last = max(keep_last, 1)
        candidates = [
            os.path.join(save_dir, name)
            for name in os.listdir(save_dir)
            if name.startswith("epoch_") and name.endswith(".pth")
        ]
        candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for path in candidates[keep_last:]:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:
        pass


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    config: Dict,
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save training checkpoint.

    For LoRA fine-tuning, saves only LoRA + decoder weights (compact).
    For full fine-tuning, saves the complete model.
    """
    os.makedirs(save_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    # Save only trainable parameters (compact, avoids storing frozen backbone)
    state["model_state_dict"] = {
        n: p.data.clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    state["save_mode"] = "trainable_only"

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    path = os.path.join(save_dir, filename)
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)

    keep_last = config.get("training", {}).get("keep_last_checkpoints", 2)
    _cleanup_old_epoch_checkpoints(save_dir, keep_last)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> Dict:
    """Load training checkpoint.

    For LoRA checkpoints, loads only the saved parameters.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    save_mode = state.get("save_mode", "full")
    if save_mode in ("lora_only", "trainable_only"):
        # Load compact checkpoints that contain only trainable parameters
        # such as LoRA, adapters, and decoder weights.
        model_state = model.state_dict()
        for n, p in state["model_state_dict"].items():
            if n in model_state:
                if tuple(model_state[n].shape) == tuple(p.shape):
                    model_state[n].copy_(p)
    else:
        model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state
