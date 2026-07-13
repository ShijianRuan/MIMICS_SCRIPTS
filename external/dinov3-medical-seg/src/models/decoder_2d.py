"""
2D decoder heads for pseudo-3D feature volumes.

Unlike their 3D counterparts (decoder_3d.py), these decoders process each
axial slice independently with 2D convolutions — there are no cross-slice
operations.  This design has several advantages for *few-shot* settings:

- **Fewer parameters**: 2D convolutions have ~D× fewer parameters than
  equivalent 3D convolutions (where D is depth).
- **Better generalisation with tiny datasets**: fewer degrees of freedom
  → less overfitting when training on k=1…10 samples.
- **Natural fit for small / thin organs**: adrenals (a few slices), aorta
  (thin wall), scapula (flat bone) — 3D convolutions can over-smooth the
  signal across slices that contain mostly background.

Four variants are provided:

    conv2d          Simple 2D conv fusion (lightest, ~0.1M)
    conv2d_unet     Lightweight 2D U-Net per slice (~0.5M)
    conv2d_deeplab  ASPP-style multi-scale context (~0.3M)
    conv2d_2_5d     3 adjacent slices as pseudo-RGB input (~0.2M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ──────────────────────────────────────────────
# 1. Conv2DDecoder — simplest 2D baseline
# ──────────────────────────────────────────────

class Conv2DDecoder(nn.Module):
    """Multi-scale 2D conv fusion — one slice at a time.

    Projects each backbone feature level to a common channel dimension,
    upsamples to the shallowest spatial resolution, concatenates,
    fuses with 2D convolutions, and upsamples to the original in-plane
    resolution.  No cross-slice operations.

    ~0.1M parameters (4 levels × project + 2 conv layers).
    """

    def __init__(self, feature_dims: List[int], num_classes: int,
                 proj_dim: int = 64):
        super().__init__()
        self.num_levels = len(feature_dims)
        self.proj_dim = proj_dim

        # Per-level 1×1 projection
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])

        # 2D fusion (replaces 3×3×3 conv with 3×3 conv2d)
        in_fuse = proj_dim * len(feature_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_fuse, proj_dim * 2, 3, padding=1),
            nn.BatchNorm2d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim * 2, proj_dim, 3, padding=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Conv2d(proj_dim, num_classes, 1)

    def forward(self, features_3d: List[torch.Tensor],
                original_shape: tuple) -> torch.Tensor:
        """
        Args:
            features_3d: list of (B, C, D, h, w)
            original_shape: (B, 1, D, H, W)

        Returns:
            (B, num_classes, D, H, W)
        """
        B, _, D_in = features_3d[0].shape[:3]
        target_2d = features_3d[0].shape[-2:]  # (h, w)

        # Collapse batch + depth → "batch" for 2D ops
        feats_2d = [f.permute(0, 2, 1, 3, 4).reshape(B * D_in, C, h, w)
                    for f, (_, C, _, h, w) in
                    zip(features_3d, [f.shape for f in features_3d])]

        # Project and upsample to common spatial size
        projected = []
        for proj, f2d in zip(self.projections, feats_2d):
            p = proj(f2d)                     # (B*D, proj_dim, h, w)
            if p.shape[-2:] != target_2d:
                p = F.interpolate(p, size=target_2d, mode="bilinear",
                                  align_corners=False)
            projected.append(p)

        # Fuse and predict
        fused = self.fuse(torch.cat(projected, dim=1))  # (B*D, proj_dim, h, w)
        out_2d = self.head(fused)                        # (B*D, num_classes, h, w)

        # Restore depth dimension
        out_3d = out_2d.reshape(B, D_in, -1, *out_2d.shape[-2:])
        out_3d = out_3d.permute(0, 2, 1, 3, 4)          # (B, C, D, h, w)

        # Upsample to original spatial size
        out_3d = F.interpolate(out_3d, size=original_shape[2:],
                               mode="trilinear", align_corners=False)
        return out_3d


# ──────────────────────────────────────────────
# 2. Conv2DUNetDecoder — lightweight 2D U-Net per slice
# ──────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv → BN → ReLU × 2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Conv2DUNetDecoder(nn.Module):
    """Lightweight 2D U-Net applied independently to each slice.

    Uses the deepest backbone feature as the bottleneck input and the
    shallower features as skip connections.  All convolutions are 2D.

    ~0.5M parameters.
    """

    def __init__(self, feature_dims: List[int], num_classes: int,
                 base_ch: int = 32):
        super().__init__()
        n_levels = len(feature_dims)

        # Project backbone features → base_ch * 2^i
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(dim, base_ch * (2 ** i), 1)
            for i, dim in enumerate(feature_dims)
        ])

        # Decoder path (deep → shallow)
        dec_ch = base_ch * (2 ** (n_levels - 1))
        self.dec_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_levels - 1):
            skip_ch = base_ch * (2 ** (n_levels - 2 - i))
            self.up_convs.append(
                nn.ConvTranspose2d(dec_ch, skip_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(_ConvBlock(skip_ch * 2, skip_ch))
            dec_ch = skip_ch

        # Output
        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, features_3d: List[torch.Tensor],
                original_shape: tuple) -> torch.Tensor:
        B, _, D_in = features_3d[0].shape[:3]

        # Collapse B+D → "batch" for 2D ops
        feats_2d = []
        for f in features_3d:
            Bf, Cf, Df, hf, wf = f.shape
            feats_2d.append(f.permute(0, 2, 1, 3, 4).reshape(Bf * Df, Cf, hf, wf))

        # Project backbone features → skip features (deepest first)
        skips = [proj(f2d) for proj, f2d in zip(self.skip_projs, feats_2d)]

        # Decode deepest → shallowest
        x = skips[-1]  # deepest
        for up_conv, dec_block, skip in zip(
            self.up_convs, self.dec_blocks, skips[-2::-1]
        ):
            x = up_conv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = dec_block(torch.cat([x, skip], dim=1))

        # Output
        out_2d = self.head(x)  # (B*D, num_classes, h, w)

        # Restore depth
        out_3d = out_2d.reshape(B, D_in, -1, *out_2d.shape[-2:])
        out_3d = out_3d.permute(0, 2, 1, 3, 4)

        out_3d = F.interpolate(out_3d, size=original_shape[2:],
                               mode="trilinear", align_corners=False)
        return out_3d


# ──────────────────────────────────────────────
# 3. Conv2DDeepLabDecoder — ASPP-style 2D
# ──────────────────────────────────────────────

class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (2D)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch, 1)
        self.branch2 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.branch4 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        p = self.pool(x)
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(torch.cat([b1, b2, b3, b4, p], dim=1))


class Conv2DDeepLabDecoder(nn.Module):
    """ASPP-style 2D decoder — multi-scale dilated conv per slice.

    Projects the deepest backbone feature through ASPP for multi-scale
    context, then fuses with shallower features via 2D convs.

    ~0.3M parameters.
    """

    def __init__(self, feature_dims: List[int], num_classes: int,
                 proj_dim: int = 64):
        super().__init__()
        n_levels = len(feature_dims)

        # Projection for each feature level
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])

        # ASPP on deepest features
        self.aspp = _ASPP(proj_dim, proj_dim)

        # Progressive fusion from deep to shallow
        self.fuse_blocks = nn.ModuleList()
        for _ in range(n_levels - 1):
            self.fuse_blocks.append(
                nn.Sequential(
                    nn.Conv2d(proj_dim * 2, proj_dim, 3, padding=1),
                    nn.BatchNorm2d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(proj_dim, proj_dim, 3, padding=1),
                    nn.BatchNorm2d(proj_dim),
                    nn.ReLU(inplace=True),
                )
            )

        self.head = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim // 2, 3, padding=1),
            nn.BatchNorm2d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim // 2, num_classes, 1),
        )

    def forward(self, features_3d: List[torch.Tensor],
                original_shape: tuple) -> torch.Tensor:
        B, _, D_in = features_3d[0].shape[:3]

        # Collapse to 2D
        feats_2d = []
        for f in features_3d:
            Bf, Cf, Df, hf, wf = f.shape
            feats_2d.append(f.permute(0, 2, 1, 3, 4).reshape(Bf * Df, Cf, hf, wf))

        proj = [p(f2d) for p, f2d in zip(self.projections, feats_2d)]

        # ASPP on deepest
        x = self.aspp(proj[-1])

        # Fuse with shallower features
        for fuse_block, skip in zip(self.fuse_blocks, proj[-2::-1]):
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = fuse_block(torch.cat([x, skip], dim=1))

        out_2d = self.head(x)

        # Restore depth
        out_3d = out_2d.reshape(B, D_in, -1, *out_2d.shape[-2:])
        out_3d = out_3d.permute(0, 2, 1, 3, 4)
        out_3d = F.interpolate(out_3d, size=original_shape[2:],
                               mode="trilinear", align_corners=False)
        return out_3d


# ──────────────────────────────────────────────
# 4. Conv2D_2_5D_Decoder — 3 adjacent slices as input
# ──────────────────────────────────────────────

class Conv2D_2_5D_Decoder(nn.Module):
    """2.5D decoder: stack 3 adjacent slices as pseudo-RGB input.

    For each target slice, the features of slices [d-1, d, d+1] are
    concatenated along the channel dimension and processed by a 2D conv
    stack.  This captures local cross-slice context without 3D convolutions.

    ~0.2M parameters.
    """

    def __init__(self, feature_dims: List[int], num_classes: int,
                 proj_dim: int = 64):
        super().__init__()
        self.num_levels = len(feature_dims)
        self.proj_dim = proj_dim

        # Per-level projection (3× channels for 3 adjacent slices)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])

        in_fuse = proj_dim * len(feature_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_fuse, proj_dim * 2, 3, padding=1),
            nn.BatchNorm2d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim * 2, proj_dim, 3, padding=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(proj_dim, num_classes, 1)

    def _stack_neighbours(self, feat_3d: torch.Tensor) -> torch.Tensor:
        """Stack [d-1, d, d+1] along channel dim for each slice.

        feat_3d: (B, C, D, h, w) → (B*D, 3*C, h, w)
        Edge slices use replication padding.
        """
        B, C, D, h, w = feat_3d.shape
        # Work in 4D to avoid F.pad limitation on 5D tensors:
        # (B, C, D, h, w) → (B*C, D, h, w) → pad D dim → restore
        f = feat_3d.permute(0, 2, 1, 3, 4).reshape(B * D, C, h, w)
        # Pad along h/w only for the 3-slice stacking via a different route:
        # Use simple slice indexing with edge replication
        slices = feat_3d.permute(0, 2, 1, 3, 4)  # (B, D, C, h, w)
        # Build padded version: [first_slice, slice_0, slice_1, ..., slice_{D-1}, last_slice]
        first = slices[:, :1, :, :, :]   # (B, 1, C, h, w)
        last = slices[:, -1:, :, :, :]   # (B, 1, C, h, w)
        padded = torch.cat([first, slices, last], dim=1)  # (B, D+2, C, h, w)
        # Extract [d-1, d, d+1] triples
        prev_slice = padded[:, :D, :, :, :]
        curr_slice = padded[:, 1:D+1, :, :, :]
        next_slice = padded[:, 2:D+2, :, :, :]
        stacked = torch.cat([prev_slice, curr_slice, next_slice], dim=2)  # (B, D, 3*C, h, w)
        return stacked.reshape(B * D, 3 * C, h, w)

    def forward(self, features_3d: List[torch.Tensor],
                original_shape: tuple) -> torch.Tensor:
        B, _, D_in = features_3d[0].shape[:3]
        target_2d = features_3d[0].shape[-2:]

        projected = []
        for proj, f3d in zip(self.projections, features_3d):
            f_stacked = self._stack_neighbours(f3d)           # (B*D, 3*C, h, w)
            p = proj(f_stacked)                               # (B*D, proj_dim, h, w)
            if p.shape[-2:] != target_2d:
                p = F.interpolate(p, size=target_2d, mode="bilinear",
                                  align_corners=False)
            projected.append(p)

        fused = self.fuse(torch.cat(projected, dim=1))
        out_2d = self.head(fused)
        out_3d = out_2d.reshape(B, D_in, -1, *out_2d.shape[-2:])
        out_3d = out_3d.permute(0, 2, 1, 3, 4)
        out_3d = F.interpolate(out_3d, size=original_shape[2:],
                               mode="trilinear", align_corners=False)
        return out_3d
