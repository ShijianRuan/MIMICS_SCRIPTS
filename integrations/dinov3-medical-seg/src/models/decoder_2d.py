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

Five variants are provided:

    conv2d          Simple 2D conv fusion (lightest, ~0.1M)
    conv2d_unet     Lightweight 2D U-Net per slice (~0.5M)
    conv2d_deeplab  ASPP-style multi-scale context (~0.3M)
    conv2d_2_5d     3 adjacent slices as pseudo-RGB input (~0.2M)
    feature_unet2d  Frozen DINO feature decoder with raw-image skip channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _norm2d(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ScaleAwareRefine2D(nn.Module):
    """Depthwise spatial refinement followed by channel mixing."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            _norm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _norm2d(channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.body(values)


class ScaleAware2DDecoder(nn.Module):
    """DINO-aware 2D decoder with an explicit spatial token pyramid.

    DINO intermediate blocks have different semantic depth but the same patch
    grid. This decoder projects four levels, reassembles them at H/2, H/4,
    H/8, and H/16, and performs lightweight top-down fusion. Slices are
    decoded in bounded chunks so a long CT volume does not materialize every
    high-resolution feature map at once.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        proj_dim: int = 48,
        decode_slice_batch_size: int = 16,
    ):
        super().__init__()
        if len(feature_dims) < 4:
            raise ValueError(
                "scale_aware2d requires four DINO intermediate feature levels"
            )
        self.feature_dims = list(feature_dims[-4:])
        self.decode_slice_batch_size = max(1, int(decode_slice_batch_size))
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, proj_dim, kernel_size=1, bias=False),
                    _norm2d(proj_dim),
                    nn.GELU(),
                )
                for dim in self.feature_dims
            ]
        )
        self.refine = nn.ModuleList(
            [_ScaleAwareRefine2D(proj_dim) for _ in range(4)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            _norm2d(proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, num_classes, kernel_size=1),
        )

    @staticmethod
    def _target_sizes(height: int, width: int) -> list[tuple[int, int]]:
        return [
            (max(1, height // 2), max(1, width // 2)),
            (max(1, height // 4), max(1, width // 4)),
            (max(1, height // 8), max(1, width // 8)),
            (max(1, height // 16), max(1, width // 16)),
        ]

    def _decode_chunk(
        self,
        features: List[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        sizes = self._target_sizes(*output_size)
        levels = []
        for projection, feature, target in zip(
            self.projections, features[-4:], sizes
        ):
            current = projection(feature)
            if current.shape[-2:] != target:
                current = F.interpolate(
                    current,
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                )
            levels.append(current)
        current = self.refine[-1](levels[-1])
        for index in range(2, -1, -1):
            current = F.interpolate(
                current,
                size=levels[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            current = self.refine[index](current + levels[index])
        logits = self.head(current)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        features_3d: List[torch.Tensor],
        original_shape: tuple,
    ) -> torch.Tensor:
        batch, _, depth = features_3d[0].shape[:3]
        flattened = [
            feature.permute(0, 2, 1, 3, 4).reshape(
                batch * depth,
                feature.shape[1],
                feature.shape[3],
                feature.shape[4],
            )
            for feature in features_3d[-4:]
        ]
        output_size = (int(original_shape[-2]), int(original_shape[-1]))
        chunks = []
        for start in range(0, batch * depth, self.decode_slice_batch_size):
            end = min(batch * depth, start + self.decode_slice_batch_size)
            chunks.append(
                self._decode_chunk(
                    [feature[start:end] for feature in flattened],
                    output_size,
                )
            )
        logits = torch.cat(chunks, dim=0)
        return logits.reshape(
            batch,
            depth,
            logits.shape[1],
            output_size[0],
            output_size[1],
        ).permute(0, 2, 1, 3, 4)


def _tinygrad_uniform_(module: nn.Module, input_channels: int) -> None:
    """Match the recovered decoder's tinygrad convolution initialization."""
    kernel = module.kernel_size
    if not isinstance(kernel, tuple):
        kernel = (kernel, kernel)
    bound = 1.0 / float(input_channels * kernel[0] * kernel[1]) ** 0.5
    nn.init.uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        nn.init.uniform_(module.bias, -bound, bound)


class _FeatureDoubleConv(nn.Module):
    """Two instance-normalized convolutions with the recovered residual path."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)
        _tinygrad_uniform_(self.conv1, in_channels)
        _tinygrad_uniform_(self.conv2, out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.leaky_relu(self.norm1(self.conv1(values)), negative_slope=0.01)
        residual = self.norm2(self.conv2(values))
        return F.leaky_relu(residual + values, negative_slope=0.01)


class _FeatureUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.body = _FeatureDoubleConv(out_channels + skip_channels, out_channels)
        _tinygrad_uniform_(self.up, in_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                "Feature decoder skip shape mismatch: {} vs {}".format(
                    tuple(values.shape[-2:]), tuple(skip.shape[-2:])
                )
            )
        return self.body(torch.cat([values, skip], dim=1))


class FrozenFeatureUNet2D(nn.Module):
    """Decode the final DINO feature map with raw grayscale pixel skips.

    This is a direct PyTorch implementation of the verified network topology:
    the final 1/16 DINO feature map is decoded through four upsampling stages.
    The skip tensors are pixel-unshuffled views of the normalized source slice,
    not intermediate backbone features. Each slice is independent and the
    resulting masks are stacked back into their original volume order.
    """

    requires_raw_input = True
    requires_single_feature = True

    def __init__(self, feature_dims: List[int], num_classes: int):
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_unet2d requires one DINO feature dimension")
        feature_dim = int(feature_dims[-1])
        self.bottleneck = _FeatureDoubleConv(feature_dim, 256)
        self.up1 = _FeatureUp(256, 64, 128)
        self.up2 = _FeatureUp(128, 16, 64)
        self.up3 = _FeatureUp(64, 4, 32)
        self.up4 = _FeatureUp(32, 1, 16)
        self.outc = nn.Conv2d(16, num_classes, 1)
        _tinygrad_uniform_(self.outc, 16)

    @staticmethod
    def raw_skip(values: torch.Tensor, factor: int) -> torch.Tensor:
        if values.dim() != 4 or values.shape[1] != 1:
            raise RuntimeError(
                "Raw feature skips require (N,1,H,W), got {}".format(tuple(values.shape))
            )
        if values.shape[-2] % factor or values.shape[-1] % factor:
            raise RuntimeError(
                "Raw slice shape {} must be divisible by {}".format(
                    tuple(values.shape[-2:]), factor
                )
            )
        return F.pixel_unshuffle(values, factor)

    def forward_slices(self, embeddings: torch.Tensor, raw_slices: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 4:
            raise RuntimeError(
                "DINO embeddings must be (N,C,H,W), got {}".format(tuple(embeddings.shape))
            )
        if raw_slices.dim() != 4 or raw_slices.shape[1] != 1:
            raise RuntimeError(
                "Raw slices must be (N,1,H,W), got {}".format(tuple(raw_slices.shape))
            )
        if raw_slices.shape[-2] % 16 or raw_slices.shape[-1] % 16:
            raise RuntimeError("feature_unet2d requires image dimensions divisible by 16")
        expected_grid = (raw_slices.shape[-2] // 16, raw_slices.shape[-1] // 16)
        if tuple(embeddings.shape[-2:]) != expected_grid:
            raise RuntimeError(
                "DINO feature grid {} does not match raw slice grid {}".format(
                    tuple(embeddings.shape[-2:]), expected_grid
                )
            )

        values = self.bottleneck(embeddings)
        values = self.up1(values, self.raw_skip(raw_slices, 8))
        values = self.up2(values, self.raw_skip(raw_slices, 4))
        values = self.up3(values, self.raw_skip(raw_slices, 2))
        values = self.up4(values, raw_slices)
        return self.outc(values)

    def forward(
        self,
        features_3d: List[torch.Tensor],
        original_shape: tuple,
        raw_volume: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_volume is None:
            raise RuntimeError("feature_unet2d requires the normalized raw volume")
        feature = features_3d[-1]
        batch, channels, depth, height, width = feature.shape
        embeddings = feature.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, height, width
        )
        raw_slices = raw_volume.permute(0, 2, 1, 3, 4).reshape(
            batch * depth,
            raw_volume.shape[1],
            raw_volume.shape[3],
            raw_volume.shape[4],
        )
        logits = self.forward_slices(embeddings, raw_slices)
        logits = logits.reshape(batch, depth, logits.shape[1], *logits.shape[-2:])
        logits = logits.permute(0, 2, 1, 3, 4)
        if tuple(logits.shape[2:]) != tuple(original_shape[2:]):
            raise RuntimeError(
                "feature_unet2d preserves the 2D training grid; got {} vs {}".format(
                    tuple(logits.shape[2:]), tuple(original_shape[2:])
                )
            )
        return logits


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
