"""Device detection: CUDA > MPS > CPU."""

import torch


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device = None) -> torch.dtype:
    """Auto-select best precision."""
    if device is None:
        device = get_device()
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif device.type == "cuda":
        return torch.float16
    return torch.float32
