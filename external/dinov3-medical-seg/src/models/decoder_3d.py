"""
3D decoder heads for pseudo-3D feature volumes.
Four options: linear3d, mlp_probe, segformer3d, dpt3d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class DecoderFactory:
    """Create decoder by name."""

    @staticmethod
    def create(decoder_type: str, feature_dims: List[int], num_classes: int) -> nn.Module:
        if decoder_type == "linear3d":
            return LinearDecoder3D(feature_dims[0], num_classes, num_levels=len(feature_dims))
        elif decoder_type == "mlp_probe":
            return MLPProbeDecoder3D(feature_dims[-3:], num_classes)
        elif decoder_type == "segformer3d":
            return SegFormer3DDecoder(feature_dims, num_classes)
        elif decoder_type == "dpt3d":
            return DPT3DDecoder(feature_dims, num_classes)
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. "
                f"Choose from: linear3d, mlp_probe, segformer3d, dpt3d"
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

    def __init__(self, feature_dim: int, num_classes: int, num_levels: int = 4):
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
            nn.BatchNorm3d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim * 2, proj_dim, kernel_size=1),
            nn.BatchNorm3d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Conv3d(proj_dim, num_classes, kernel_size=1)

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
    """

    def __init__(self, feature_dims: List[int], num_classes: int, probe_hidden: int = 256):
        super().__init__()
        # One probe per feature layer (last 3 blocks)
        self.probes = nn.ModuleList([
            MLPProbe(dim, probe_hidden, num_classes) for dim in feature_dims
        ])
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

        # Z-axis Gaussian smoothing (non-parametric)
        if pred.shape[2] > 1:
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
    """

    def __init__(self, feature_dims: List[int], num_classes: int, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = proj_dim

        # Project each feature level to uniform channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, proj_dim, 1),
                nn.BatchNorm3d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])

        # MLP fusion
        total_dim = proj_dim * len(feature_dims)
        self.fuse = nn.Sequential(
            nn.Conv3d(total_dim, proj_dim * 2, 1),
            nn.BatchNorm3d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim * 2, proj_dim, 1),
            nn.BatchNorm3d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Conv3d(proj_dim, proj_dim // 2, 3, padding=1),
            nn.BatchNorm3d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_dim // 2, num_classes, 1),
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
        out = F.interpolate(out, size=original_shape[2:], mode="trilinear", align_corners=False)
        return out


# ──────────────────────────────────────────────
# 4. DPT3DDecoder — Dense Prediction Transformer style
# ──────────────────────────────────────────────

class Reassemble3D(nn.Module):
    """DPT Reassemble block adapted to 3D."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_dim, out_dim, 1)
        self.refine = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm3d(out_dim),
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

    def __init__(self, feature_dims: List[int], num_classes: int, out_dim: int = 256):
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
                nn.BatchNorm3d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm3d(out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)  # 3 fusion stages
        ])

        # Output
        self.head = nn.Sequential(
            nn.Conv3d(out_dim, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1),
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
        out = F.interpolate(out, size=original_shape[2:], mode="trilinear", align_corners=False)
        return out
