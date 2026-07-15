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
from .feature_augmentation import build_feature_augmentation


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
        self.slice_axis = cfg.get("slice_axis", 2)
        self.slice_batch_size = int(cfg.get("slice_batch_size", 4))
        if self.slice_batch_size < 1:
            raise ValueError("model.slice_batch_size must be at least one")
        self.encoder_3d = SliceWiseEncoder3D(
            self.backbone,
            channel_policy=cfg.get("channel_policy", "repeat"),
            neighbor_distance_mm=cfg.get("neighbor_distance_mm"),
        )

        # ── 4. 3D Decoder ──
        feature_dims = [self.embed_dim] * len(cfg.get("out_indices", [2, 5, 8, 11]))
        self.decoder_type = dec_cfg.get("type", "segformer3d")

        # z_smooth_sigma is configured in physical mm.  Convert to voxel
        # units using the dataset's Z spacing so the same 6 mm value
        # applies a comparable amount of smoothing regardless of whether
        # the scan is thin-slice (0.5 mm) or thick-slice (5.0 mm).
        z_smooth_sigma_mm = float(dec_cfg.get("z_smooth_sigma", 0.0))
        z_smooth_sigma = 0.0
        if z_smooth_sigma_mm > 0:
            data_cfg = config.get("data", {})
            target_spacing = data_cfg.get("target_spacing", [1.5, 1.0, 1.0])
            z_spacing_mm = float(
                target_spacing[0] if isinstance(target_spacing, (list, tuple)) else target_spacing
            )
            if z_spacing_mm <= 0:
                raise ValueError("data.target_spacing Z must be positive for z_smooth_sigma")
            sigma_voxels = z_smooth_sigma_mm / z_spacing_mm
            # Clamp to a sensible voxel range: at least 1 voxel (don't
            # undersmooth thin slices) and at most 8 voxels (don't
            # obliterate small targets on thick slices with a 48 mm kernel).
            sigma_voxels = max(1.0, min(8.0, sigma_voxels))
            z_smooth_sigma = float(sigma_voxels)

        self.decoder_3d = DecoderFactory.create(
            self.decoder_type,
            feature_dims,
            num_classes=cfg["num_classes"],
            z_smooth_sigma=z_smooth_sigma,
        )
        self.feature_augmentation = build_feature_augmentation(config.get("feature_augmentation"))

        # Store original_shape for decoder upsampling
        self._original_shape = None

    def _slice_axis_name(self) -> str:
        value = self.slice_axis
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {"z": "axial", "y": "coronal", "x": "sagittal"}
            normalized = aliases.get(normalized, normalized)
            if normalized in ("axial", "coronal", "sagittal"):
                return normalized
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = -1
        mapping = {2: "axial", 1: "coronal", 0: "sagittal"}
        if numeric in mapping:
            return mapping[numeric]
        raise ValueError("slice_axis must be axial/coronal/sagittal or 2/1/0")

    def _to_slice_space(self, volume_3d: torch.Tensor) -> torch.Tensor:
        # Input/output convention is (B, C, Z, Y, X).  The encoder always
        # processes dimension 2 as depth; reordering here keeps all three views
        # correct without spreading axis assumptions across the pipeline.
        axis = self._slice_axis_name()
        if axis == "axial":
            return volume_3d
        if axis == "coronal":
            return volume_3d.permute(0, 1, 3, 2, 4)
        return volume_3d.permute(0, 1, 4, 2, 3)

    def _from_slice_space(self, output: torch.Tensor) -> torch.Tensor:
        axis = self._slice_axis_name()
        if axis == "axial":
            return output
        if axis == "coronal":
            return output.permute(0, 1, 3, 2, 4)
        return output.permute(0, 1, 3, 4, 2)

    def _to_slice_space_spacing(self, spacing_zyx: torch.Tensor | None):
        if spacing_zyx is None:
            return None
        spacing = torch.as_tensor(spacing_zyx)
        if spacing.shape[-1] != 3:
            raise ValueError("spacing_zyx must end with three values")
        axis = self._slice_axis_name()
        if axis == "axial":
            return spacing
        if axis == "coronal":
            return spacing[..., [1, 0, 2]]
        return spacing[..., [2, 0, 1]]

    def forward(self, volume_3d: torch.Tensor, spacing_zyx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            volume_3d: (B, 1, D, H, W) input volume

        Returns:
            (B, num_classes, D, H, W) segmentation logits
        """
        slice_space_volume = self._to_slice_space(volume_3d)
        self._original_shape = slice_space_volume.shape

        # Slice-wise encoding → pseudo-3D features
        features_3d = self.encoder_3d(
            slice_space_volume,
            slice_batch_size=self.slice_batch_size,
            spacing_zyx=self._to_slice_space_spacing(spacing_zyx),
        )
        features_3d = self.feature_augmentation(features_3d)

        # 3D decoding → segmentation
        output = self.decoder_3d(features_3d, self._original_shape)
        return self._from_slice_space(output)

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
