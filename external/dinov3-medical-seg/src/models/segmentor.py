"""
Complete 3D segmentation model: backbone + encoder + decoder.
"""

import torch
import torch.nn as nn
from typing import Dict

from .backbone import DINOv3Backbone
from .encoder_3d import SliceWiseEncoder3D
from .decoder_3d import DecoderFactory
from .lora import apply_lora_to_dinov3, get_lora_params
from .adapter import apply_adapter_to_dinov3


class DINOv33DSegmentor(nn.Module):
    """DINOv3-based 3D medical image segmentation model.

    Architecture:
        Volume(3D) → SliceWiseEncoder3D → pseudo-3D features → 3D Decoder → 3D mask

    Supports four fine-tuning modes via config:
        - 'frozen': freeze backbone, train decoder only
        - 'lora': inject LoRA into backbone Q/V, train LoRA + decoder
        - 'full': unfreeze all, train everything
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: dict with keys:
                model.hf_name, model.out_indices, model.num_classes
                finetune.method, finetune.lora_rank, finetune.lora_alpha
                decoder.type, decoder.*
        """
        super().__init__()
        cfg = config["model"]
        ft_cfg = config.get("finetune", {})
        dec_cfg = config.get("decoder", {})

        # ── 1. Backbone ──
        freeze_backbone = ft_cfg.get("method", "frozen") != "full"
        self.backbone = DINOv3Backbone(
            model_path=cfg["model_path"],
            out_indices=cfg.get("out_indices", [2, 5, 8, 11]),
            freeze=freeze_backbone,
            input_normalization=cfg.get("input_normalization", "none"),
            image_mean=cfg.get("image_mean"),
            image_std=cfg.get("image_std"),
        )
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        # ── 2. Fine-tuning method ──
        self.ft_method = ft_cfg.get("method", "frozen")
        if self.ft_method == "lora":
            r = ft_cfg.get("lora_rank", 8)
            alpha = ft_cfg.get("lora_alpha", 16)
            apply_lora_to_dinov3(self.backbone, r=r, alpha=alpha)
        elif self.ft_method == "adapter":
            bn = ft_cfg.get("adapter_bottleneck", 64)
            pos = ft_cfg.get("adapter_position", "after_attn")
            apply_adapter_to_dinov3(self.backbone, bottleneck=bn, position=pos)
        elif self.ft_method == "full":
            self.backbone.unfreeze()

        # ── 3. 3D Encoder ──
        self.encoder_3d = SliceWiseEncoder3D(
            self.backbone,
            slice_axis=cfg.get("slice_axis", 2),
        )

        # ── 4. 3D Decoder ──
        feature_dims = [self.embed_dim] * len(cfg.get("out_indices", [2, 5, 8, 11]))
        self.decoder_type = dec_cfg.get("type", "segformer3d")
        self.decoder_3d = DecoderFactory.create(
            self.decoder_type,
            feature_dims,
            num_classes=cfg["num_classes"],
        )

        # Store original_shape for decoder upsampling
        self._original_shape = None

    def forward(self, volume_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume_3d: (B, 1, D, H, W) input volume

        Returns:
            (B, num_classes, D, H, W) segmentation logits
        """
        self._original_shape = volume_3d.shape

        # Slice-wise encoding → pseudo-3D features
        features_3d = self.encoder_3d(volume_3d)

        # 3D decoding → segmentation
        output = self.decoder_3d(features_3d, self._original_shape)

        return output

    def get_trainable_info(self) -> Dict[str, int]:
        """Return trainable parameter counts by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_trainable = sum(
            p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad
        )
        decoder_trainable = sum(
            p.numel() for n, p in self.decoder_3d.named_parameters() if p.requires_grad
        )

        lora_trainable = 0
        if self.ft_method == "lora":
            lora_trainable = sum(
                p.numel() for n, p in self.named_parameters()
                if "lora_A" in n or "lora_B" in n
            )

        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100 * trainable / total,
            "backbone_trainable": backbone_trainable,
            "decoder_trainable": decoder_trainable,
            "lora_trainable": lora_trainable,
            "ft_method": self.ft_method,
            "decoder_type": self.decoder_type,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
        }
