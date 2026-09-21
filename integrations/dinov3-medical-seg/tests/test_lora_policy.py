"""Tests for the explicit DINOv3 attention LoRA policy."""

import pytest
import torch.nn as nn

from src.models.lora import LoRALinear, apply_lora_to_dinov3


class _Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = _Attention(width)


class _Model(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.layer = nn.ModuleList([_Block(width) for _ in range(depth)])


class _WrappedBackbone(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = _Model(depth, width)


@pytest.mark.parametrize("depth,width", [(12, 384), (12, 768), (24, 1024)])
def test_default_lora_policy_targets_q_and_v_at_every_depth(depth, width):
    backbone = _WrappedBackbone(depth, width)
    apply_lora_to_dinov3(backbone, r=4, alpha=8)

    for block in backbone.backbone.model.layer:
        assert isinstance(block.attention.q_proj, LoRALinear)
        assert isinstance(block.attention.v_proj, LoRALinear)
        assert isinstance(block.attention.k_proj, nn.Linear)
        assert isinstance(block.attention.o_proj, nn.Linear)


def test_lora_policy_rejects_unknown_or_duplicate_targets():
    backbone = _WrappedBackbone(4, 32)
    with pytest.raises(ValueError, match="target_modules"):
        apply_lora_to_dinov3(backbone, target_modules=["q_proj", "q_proj"])
    with pytest.raises(ValueError, match="target_modules"):
        apply_lora_to_dinov3(backbone, target_modules=["query"])
