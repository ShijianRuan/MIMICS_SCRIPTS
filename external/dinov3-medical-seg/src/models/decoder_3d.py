"""
3D decoder heads for pseudo-3D feature volumes.
Four options: linear3d, mlp_probe, segformer3d, dpt3d.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def norm3d(channels: int) -> nn.GroupNorm:
    """Batch-size-independent normalization for few-shot full volumes."""
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────

class LearnableZSmooth(nn.Module):
    """Learnable depth-wise Gaussian smoothing along the Z axis.

    Applied after the decoder head to improve inter-slice consistency in
    pseudo-3D segmentation. The kernel is initialised as a 1D Gaussian so
    the smoothing starts from a reasonable prior, then the per-class
    weights are free to adapt during training (each class learns its own
    optimal smoothing strength independently via grouped convolution).

    Reference: DINO-MVR (arXiv:2605.07221) uses a fixed z-axis Gaussian
    kernel as non-parametric post-processing. This module makes the same
    operation learnable so gradients flow through it during training.

    .. note::

       *sigma* is in **voxel units**.  The caller (:class:`DINOv33DSegmentor`)
       converts the user-facing ``decoder.z_smooth_sigma`` (physical mm) to
       voxel units using ``data.target_spacing[0]`` (Z spacing), clamped to
       [1.0, 8.0] voxels so the same mm value behaves consistently across
       thin-slice and thick-slice acquisitions.

    Parameters
    ----------
    num_channels : int
        Number of classes (one independent kernel per class).
    sigma : float
        Standard deviation of the initial Gaussian kernel in **voxel units**
        (already converted from physical mm by the segmentor).
    learnable : bool
        If False the kernel is frozen (non-parametric DINO-MVR behaviour).
    """

    def __init__(self, num_channels: int, sigma: float = 4.0, learnable: bool = True):
        super().__init__()
        kernel_size = int(sigma * 3.0) * 2 + 1
        # Build 1D Gaussian: exp(-0.5 * (t / sigma)^2)
        t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian = torch.exp(-0.5 * (t / sigma) ** 2)
        gaussian = gaussian / gaussian.sum()
        # Conv3d weight shape: (out_ch, in_ch/groups, kD, kH, kW)
        weight = gaussian.view(1, 1, kernel_size, 1, 1).repeat(num_channels, 1, 1, 1, 1)
        self.conv = nn.Conv3d(
            num_channels,
            num_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=num_channels,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad = bool(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-class 1D depth convolution.

        Args:
            x: (B, C, D, H, W) logits or probabilities.
        Returns:
            (B, C, D, H, W) smoothed output.
        """
        return self.conv(x)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class DecoderFactory:
    """Create decoder by name.

    Supports both 3D decoders (linear3d, mlp_probe, segformer3d,
    token_pyramid3d, dpt3d)
    and 2D decoders (conv2d, conv2d_unet, conv2d_deeplab, conv2d_2_5d).
    2D decoders are imported lazily to avoid hard dependency.
    """

    _2D_DECODERS = {"conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d"}

    @staticmethod
    def create(
        decoder_type: str,
        feature_dims: List[int],
        num_classes: int,
        z_smooth_sigma: float = 0.0,
    ) -> nn.Module:
        # ── 3D decoders ──
        if decoder_type == "linear3d":
            return LinearDecoder3D(feature_dims[0], num_classes, num_levels=len(feature_dims),
                                   z_smooth_sigma=z_smooth_sigma)
        elif decoder_type == "mlp_probe":
            return MLPProbeDecoder3D(feature_dims[-3:], num_classes,
                                     z_smooth_sigma=z_smooth_sigma)
        elif decoder_type == "segformer3d":
            return SegFormer3DDecoder(feature_dims, num_classes,
                                      z_smooth_sigma=z_smooth_sigma)
        elif decoder_type == "token_pyramid3d":
            return TokenPyramid3DDecoder(feature_dims, num_classes,
                                         z_smooth_sigma=z_smooth_sigma)
        elif decoder_type == "dpt3d":
            return DPT3DDecoder(feature_dims, num_classes,
                                z_smooth_sigma=z_smooth_sigma)

        # ── 2D decoders (lazy import) ──
        elif decoder_type in DecoderFactory._2D_DECODERS:
            from .decoder_2d import (
                Conv2DDecoder,
                Conv2DUNetDecoder,
                Conv2DDeepLabDecoder,
                Conv2D_2_5D_Decoder,
            )
            _map = {
                "conv2d": Conv2DDecoder,
                "conv2d_unet": Conv2DUNetDecoder,
                "conv2d_deeplab": Conv2DDeepLabDecoder,
                "conv2d_2_5d": Conv2D_2_5D_Decoder,
            }
            return _map[decoder_type](feature_dims, num_classes)

        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. "
                f"Choose from: linear3d, mlp_probe, segformer3d, token_pyramid3d, dpt3d, "
                f"conv2d, conv2d_unet, conv2d_deeplab, conv2d_2_5d"
            )


# ──────────────────────────────────────────────
# 1. LinearDecoder3D — simplest baseline
# ──────────────────────────────────────────────

class LinearDecoder3D(nn.Module):
    """Multi-scale linear probe with feature fusion → trilinear upsample → output.

    Uses all backbone feature levels (shallow + deep), fuses them via
    1x1x1 conv projections + 3D conv refinement, then upsamples to full
    resolution.  The original single-conv design could not learn organ
    boundaries from frozen DINOv3 features because a single linear
    projection of the deepest layer lacks the spatial detail needed for
    segmentation.

    ~0.3M parameters (4 feature levels × proj + fusion).
    """

    def __init__(self, feature_dim: int, num_classes: int, num_levels: int = 4,
                 z_smooth_sigma: float = 0.0):
        super().__init__()
        self.num_levels = num_levels

        # Per-level 1x1x1 projection to a common channel dimension
        proj_dim = max(64, feature_dim // 4)
        self.projections = nn.ModuleList([
            nn.Conv3d(feature_dim, proj_dim, kernel_size=1)
            for _ in range(num_levels)
        ])

        # Feature fusion: concatenate projected features → 3D conv refinement
        self.fuse = nn.Sequential(
            nn.Conv3d(proj_dim * num_levels, proj_dim * 2, kernel_size=3, padding=1),
            norm3d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim * 2, proj_dim, kernel_size=1),
            norm3d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Conv3d(proj_dim, num_classes, kernel_size=1)

        self.z_smooth = (
            LearnableZSmooth(num_classes, sigma=float(z_smooth_sigma))
            if z_smooth_sigma > 0
            else None
        )

    def forward(self, features_3d: List[torch.Tensor], original_shape: tuple) -> torch.Tensor:
        """
        Args:
            features_3d: list of (B, C, D, h, w) from encoder, one per level
            original_shape: (B, C_in, D, H, W) of input volume

        Returns:
            (B, num_classes, D, H, W)
        """
        # Use up to num_levels feature maps
        feats = features_3d[:self.num_levels]

        # Project each level to common dim, upsample to shallowest spatial size
        target_shape = feats[0].shape[-3:]
        projected = []
        for proj, feat in zip(self.projections, feats):
            p = proj(feat)
            if p.shape[-3:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode="trilinear", align_corners=False)
            projected.append(p)

        # Fuse and predict
        fused = self.fuse(torch.cat(projected, dim=1))
        out = self.head(fused)
        if self.z_smooth is not None:
            out = self.z_smooth(out)
        out = F.interpolate(out, size=original_shape[2:], mode="trilinear", align_corners=False)
        return out


# ──────────────────────────────────────────────
# 2. MLPProbeDecoder3D — DINO-MVR style
# ──────────────────────────────────────────────

class MLPProbe(nn.Module):
    """Single MLP probe on patch features."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, h, w) → (B, num_classes, D, h, w)
        B, C, D, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*h*w, C)
        x = self.mlp(x)
        return x.reshape(B, D, h, w, -1).permute(0, 4, 1, 2, 3)  # (B, num_classes, D, h, w)


class MLPProbeDecoder3D(nn.Module):
    """DINO-MVR style: MLP probes on last 3 blocks + z-axis Gaussian smoothing.

    ~0.3M parameters. Reference: DINO-MVR (arXiv:2605.07221), BraTS 0.908 DSC.

    When *z_smooth_sigma* > 0 a :class:`LearnableZSmooth` module replaces the
    fixed non-parametric z-axis kernel so the smoothing strength is optimised
    during training. Pass 0 to keep the original fixed-DINO-MVR behaviour.
    """

    def __init__(self, feature_dims: List[int], num_classes: int, probe_hidden: int = 256,
                 z_smooth_sigma: float = 0.0):
        super().__init__()
        # One probe per feature layer (last 3 blocks)
        self.probes = nn.ModuleList([
            MLPProbe(dim, probe_hidden, num_classes) for dim in feature_dims
        ])
        # Use learnable z-smooth when explicitly requested, otherwise keep the
        # original fixed-DINO-MVR kernel for backward compatibility.
        self._learnable_z_smooth = (
            LearnableZSmooth(num_classes, sigma=float(z_smooth_sigma))
            if z_smooth_sigma > 0
            else None
        )
        self.z_smooth_sigma = 4.0

    def forward(self, features_3d: List[torch.Tensor], original_shape: tuple) -> torch.Tensor:
        # Each probe predicts independently
        preds = []
        for probe, feat in zip(self.probes, features_3d):
            p = probe(feat)  # (B, C, D, h, w)
            p = F.interpolate(p, size=original_shape[2:], mode="trilinear", align_corners=False)
            preds.append(p)

        # Average fusion
        pred = torch.stack(preds).mean(dim=0)

        # Z-axis smoothing: learnable when configured, otherwise fixed DINO-MVR kernel.
        if self._learnable_z_smooth is not None:
            pred = self._learnable_z_smooth(pred)
        elif pred.shape[2] > 1:
            kernel_size = int(self.z_smooth_sigma * 3) * 2 + 1
            kernel = torch.exp(-0.5 * (torch.arange(kernel_size, device=pred.device).float()
                                        - kernel_size // 2) ** 2 / self.z_smooth_sigma ** 2)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, -1, 1, 1)
            # Apply 1D convolution along depth
            padding = kernel_size // 2
            pred = F.conv3d(
                F.pad(pred, (0, 0, 0, 0, padding, padding), mode="replicate"),
                kernel.expand(pred.shape[1], 1, -1, 1, 1),
                groups=pred.shape[1],
            )

        return pred


# ──────────────────────────────────────────────
# 3. SegFormer3DDecoder — MLP-style fusion
# ──────────────────────────────────────────────

class SegFormer3DDecoder(nn.Module):
    """SegFormer3D-inspired: project → upsample → concat → MLP fusion → output.

    ~4M parameters. Reference: an-mistral compact SegFormer3D design.

    Set *z_smooth_sigma* > 0 to append a learnable 1D depth Gaussian after the
    output head for improved inter-slice consistency.
    """

    def __init__(self, feature_dims: List[int], num_classes: int, proj_dim: int = 128,
                 z_smooth_sigma: float = 0.0):
        super().__init__()
        self.proj_dim = proj_dim

        # Project each feature level to uniform channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, proj_dim, 1),
                norm3d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])

        # MLP fusion
        total_dim = proj_dim * len(feature_dims)
        self.fuse = nn.Sequential(
            nn.Conv3d(total_dim, proj_dim * 2, 1),
            norm3d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim * 2, proj_dim, 1),
            norm3d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Conv3d(proj_dim, proj_dim // 2, 3, padding=1),
            norm3d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim // 2, num_classes, 1),
        )

        self.z_smooth = (
            LearnableZSmooth(num_classes, sigma=float(z_smooth_sigma))
            if z_smooth_sigma > 0
            else None
        )

    def forward(self, features_3d: List[torch.Tensor], original_shape: tuple) -> torch.Tensor:
        target_shape = features_3d[0].shape[-3:]  # spatial shape of shallowest layer

        projected = []
        for proj, feat in zip(self.projections, features_3d):
            p = proj(feat)
            if p.shape[-3:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode="trilinear", align_corners=False)
            projected.append(p)

        fused = self.fuse(torch.cat(projected, dim=1))
        out = self.head(fused)
        if self.z_smooth is not None:
            out = self.z_smooth(out)
        out = F.interpolate(out, size=original_shape[2:], mode="trilinear", align_corners=False)
        return out


# ──────────────────────────────────────────────
# 4. TokenPyramid3DDecoder — scale-aware ViT readout
# ──────────────────────────────────────────────

class TokenPyramid3DDecoder(nn.Module):
    """Create a learned in-plane pyramid from equal-resolution ViT tokens.

    DINOv3 blocks expose different semantic depths but retain one patch grid;
    treating those blocks as a CNN feature pyramid is therefore inaccurate.
    This decoder makes the scale operation explicit: shallow tokens remain at
    their native resolution while progressively deeper tokens are pooled in
    plane, projected, then reassembled on the shallow grid.  It is a compact
    3D testable analogue of token-pyramid readouts, not a claim to reproduce a
    particular paper's TPA implementation.
    """

    def __init__(self, feature_dims: List[int], num_classes: int, proj_dim: int = 128,
                 z_smooth_sigma: float = 0.0):
        super().__init__()
        if not feature_dims:
            raise ValueError("TokenPyramid3DDecoder requires at least one feature level")
        self.pool_factors = tuple(2 ** index for index in range(len(feature_dims)))
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, proj_dim, kernel_size=1),
                norm3d(proj_dim),
                nn.GELU(),
            )
            for dim in feature_dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(proj_dim * len(feature_dims), proj_dim * 2, kernel_size=3, padding=1),
            norm3d(proj_dim * 2),
            nn.GELU(),
            nn.Conv3d(proj_dim * 2, proj_dim, kernel_size=3, padding=1),
            norm3d(proj_dim),
            nn.GELU(),
        )
        self.head = nn.Conv3d(proj_dim, num_classes, kernel_size=1)

        self.z_smooth = (
            LearnableZSmooth(num_classes, sigma=float(z_smooth_sigma))
            if z_smooth_sigma > 0
            else None
        )

    @staticmethod
    def _pool_inplane(feature: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return feature
        _, _, _, height, width = feature.shape
        kernel_h = min(int(factor), int(height))
        kernel_w = min(int(factor), int(width))
        if kernel_h == 1 and kernel_w == 1:
            return feature
        return F.avg_pool3d(
            feature,
            kernel_size=(1, kernel_h, kernel_w),
            stride=(1, kernel_h, kernel_w),
            ceil_mode=True,
        )

    def forward(self, features_3d: List[torch.Tensor], original_shape: tuple) -> torch.Tensor:
        target_shape = features_3d[0].shape[-3:]
        assembled = []
        for projection, feature, factor in zip(self.projections, features_3d, self.pool_factors):
            current = projection(self._pool_inplane(feature, factor))
            if current.shape[-3:] != target_shape:
                current = F.interpolate(current, size=target_shape, mode="trilinear", align_corners=False)
            assembled.append(current)
        logits = self.head(self.fuse(torch.cat(assembled, dim=1)))
        if self.z_smooth is not None:
            logits = self.z_smooth(logits)
        return F.interpolate(logits, size=original_shape[2:], mode="trilinear", align_corners=False)


# ──────────────────────────────────────────────
# 5. DPT3DDecoder — Dense Prediction Transformer style
# ──────────────────────────────────────────────

class Reassemble3D(nn.Module):
    """DPT Reassemble block adapted to 3D."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_dim, out_dim, 1)
        self.refine = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, 3, padding=1),
            norm3d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-3:] != target_size:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.refine(x)


class DPT3DDecoder(nn.Module):
    """DPT-style 3D decoder: Reassemble → progressive Fusion.

    ~8M parameters. Reference: Neonatal Brain MR (arXiv:2602.23962).
    """

    def __init__(self, feature_dims: List[int], num_classes: int, out_dim: int = 256,
                 z_smooth_sigma: float = 0.0):
        super().__init__()
        assert len(feature_dims) == 4, "DPT3D expects exactly 4 feature levels"

        # Reassemble blocks (deep → shallow)
        self.reassembles = nn.ModuleList([
            Reassemble3D(dim, out_dim) for dim in feature_dims
        ])

        # Fusion blocks (progressive residual fusion)
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(out_dim, out_dim, 3, padding=1),
                norm3d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_dim, out_dim, 3, padding=1),
                norm3d(out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)  # 3 fusion stages
        ])

        # Output
        self.head = nn.Sequential(
            nn.Conv3d(out_dim, 128, 3, padding=1),
            norm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1),
        )

        self.z_smooth = (
            LearnableZSmooth(num_classes, sigma=float(z_smooth_sigma))
            if z_smooth_sigma > 0
            else None
        )

    def forward(self, features_3d: List[torch.Tensor], original_shape: tuple) -> torch.Tensor:
        # Reassemble from deepest to shallowest
        reassembled = []
        for reassemble, feat in zip(self.reassembles, features_3d[::-1]):
            r = reassemble(feat, features_3d[0].shape[-3:])
            reassembled.append(r)

        # Progressive fusion: start from shallowest, add deeper residuals
        x = reassembled[0]
        for fusion, residual in zip(self.fusions, reassembled[1:]):
            x = fusion(x + residual)

        out = self.head(x)
        if self.z_smooth is not None:
            out = self.z_smooth(out)
        out = F.interpolate(out, size=original_shape[2:], mode="trilinear", align_corners=False)
        return out
