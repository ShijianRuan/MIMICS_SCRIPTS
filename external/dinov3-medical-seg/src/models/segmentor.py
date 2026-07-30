"""
Complete 3D segmentation model: backbone + encoder + decoder.
"""

import os

import torch
import torch.nn as nn
from typing import Dict

from .backbone import DINOv3Backbone, ONNXDINOv3Backbone
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
        backend = str(cfg.get("encoder_backend", "auto") or "auto").lower()
        model_path = cfg["model_path"]
        onnx_path = (
            str(model_path).lower().endswith(".onnx")
            or (
                backend == "auto"
                and os.path.isfile(os.path.join(str(model_path), "model.onnx"))
            )
        )
        if backend == "onnx" or onnx_path:
            self.backbone = ONNXDINOv3Backbone(
                model_path=model_path,
                out_indices=cfg.get("out_indices", [11]),
                freeze=freeze_backbone,
                input_normalization=cfg.get("input_normalization", "imagenet"),
                image_mean=cfg.get("image_mean"),
                image_std=cfg.get("image_std"),
                providers=cfg.get("onnx_providers"),
                expected_sha256=cfg.get("expected_sha256"),
                input_size=config.get("data", {}).get("img_size"),
            )
            self.encoder_backend = "onnx"
        else:
            self.backbone = DINOv3Backbone(
                model_path=model_path,
                out_indices=cfg.get("out_indices"),
                freeze=freeze_backbone,
                input_normalization=cfg.get("input_normalization", "none"),
                image_mean=cfg.get("image_mean"),
                image_std=cfg.get("image_std"),
                expected_sha256=cfg.get("expected_sha256"),
            )
            self.encoder_backend = "pytorch"
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        # ── 2. Fine-tuning method ──
        self.ft_method = ft_cfg.get("method", "frozen")
        if self.ft_method == "lora":
            r = ft_cfg.get("lora_rank", 8)
            alpha = ft_cfg.get("lora_alpha", 16)
            apply_lora_to_dinov3(
                self.backbone,
                r=r,
                alpha=alpha,
                target_modules=ft_cfg.get("target_modules"),
            )
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
        feature_dims = [self.embed_dim] * len(self.backbone.out_indices)
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
            decoder_options=dec_cfg,
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

    def forward(
        self,
        volume_3d: torch.Tensor,
        spacing_zyx: torch.Tensor | None = None,
        valid_shape_zyx: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        output = self._decode_features(
            features_3d,
            slice_space_volume,
            valid_shape_zyx=valid_shape_zyx,
        )
        return self._from_slice_space(output)

    def _decode_one(self, features_3d, raw_volume):
        if getattr(self.decoder_3d, "requires_raw_input", False):
            return self.decoder_3d(
                features_3d,
                raw_volume.shape,
                raw_volume=raw_volume,
            )
        return self.decoder_3d(features_3d, raw_volume.shape)

    def _decode_features(
        self,
        features_3d,
        slice_space_volume,
        *,
        valid_shape_zyx=None,
    ):
        """Decode padded batches on each case's real slice extent.

        Slice-wise DINO encoding is independent across depth, so it can safely
        share one padded batch. Volumetric decoder interpolation is not: using
        the longest case as every case's depth changes the feature pyramid for
        shorter scans. Decode those scans independently, then pad only their
        final logits for the ignore-index loss.
        """
        if valid_shape_zyx is None:
            return self._decode_one(features_3d, slice_space_volume)
        shapes = torch.as_tensor(valid_shape_zyx).detach().cpu()
        if shapes.ndim != 2 or shapes.shape != (slice_space_volume.shape[0], 3):
            raise ValueError(
                "valid_shape_zyx must have shape (batch, 3), got {}".format(
                    tuple(shapes.shape)
                )
            )
        axis_index = {
            "axial": 0,
            "coronal": 1,
            "sagittal": 2,
        }[self._slice_axis_name()]
        valid_depths = [int(row[axis_index]) for row in shapes.tolist()]
        padded_depth = int(slice_space_volume.shape[2])
        if any(depth <= 0 or depth > padded_depth for depth in valid_depths):
            raise ValueError(
                "valid_shape_zyx contains an invalid slice extent: {}".format(
                    valid_depths
                )
            )
        if all(depth == padded_depth for depth in valid_depths):
            return self._decode_one(features_3d, slice_space_volume)
        decoded = []
        for sample_index, valid_depth in enumerate(valid_depths):
            raw = slice_space_volume[
                sample_index : sample_index + 1,
                :,
                :valid_depth,
            ]
            sample_features = [
                feature[
                    sample_index : sample_index + 1,
                    :,
                    :valid_depth,
                ]
                for feature in features_3d
            ]
            logits = self._decode_one(sample_features, raw)
            if valid_depth < padded_depth:
                logits = torch.nn.functional.pad(
                    logits,
                    (0, 0, 0, 0, 0, padded_depth - valid_depth),
                    mode="constant",
                    value=0.0,
                )
            decoded.append(logits)
        return torch.cat(decoded, dim=0)

    def decode_cached_slices(
        self,
        embeddings: torch.Tensor,
        raw_slices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode precomputed frozen features without rerunning the backbone."""
        decoder = self.decoder_3d
        if not hasattr(decoder, "forward_slices"):
            raise RuntimeError(
                "Cached slice decoding is only supported by feature_unet2d"
            )
        return decoder.forward_slices(embeddings, raw_slices)

    def get_trainable_info(self) -> Dict:
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

        external_backbone = self.encoder_backend == "onnx"
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": None if external_backbone else 100 * trainable / total,
            "backbone_trainable": backbone_trainable,
            "decoder_trainable": decoder_trainable,
            "lora_trainable": lora_trainable,
            "ft_method": self.ft_method,
            "decoder_type": self.decoder_type,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "encoder_backend": self.encoder_backend,
            "external_backbone": external_backbone,
        }
