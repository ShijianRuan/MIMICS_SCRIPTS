"""FlexiCT-compatible Primus decoders (2D and 3D) for few-shot segmentation.

Same idea as MedDINOv3's Primus_Multiscale (multi-scale token aggregation +
patch-decode upsampling), but:
  * NO x.repeat(1,3,1,1) — FlexiCT in_chans=1, eats single-channel CT directly.
  * 2D variant uses ConvTranspose2d (for FlexiCT-2D weights).
  * 3D variant uses trilinear+Conv3d (for FlexiCT-3D weights, true volumetric;
    trilinear+conv instead of ConvTranspose3d to cut activation memory —
    ConvTranspose3d's backward needs a huge intermediate).

FlexiCT backbone's get_intermediate_layers(reshape=True) returns:
  * 2D input (B,1,H,W) -> list of (B, C, H/p, W/p)
  * 3D input (B,1,D,H,W) -> list of (B, C, D/p, H/p, W/p)
"""
import math
import torch
from torch import nn


class PatchDecode2D(nn.Module):
    """ConvTranspose2d ladder: patch_size -> 1 (must be 2^x)."""
    def __init__(self, patch_size, embed_dim, out_channels, activation=nn.GELU):
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))
        assert 2 ** n == patch_size and n >= 1
        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1] // 2)
        ch.append(out_channels)
        stages = []
        for i in range(n):
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                nn.GroupNorm(1, ch[i + 1]), activation()))
        stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        return self.decode(x)


class PatchDecode3D(nn.Module):
    """Trilinear-upsample + Conv3d ladder: patch_size -> 1 (must be 2^x).

    Uses interpolate+conv instead of ConvTranspose3d to cut activation memory
    (ConvTranspose3d's backward needs a huge intermediate). Each stage halves
    channels and doubles spatial size via trilinear interpolation.
    """
    def __init__(self, patch_size, embed_dim, out_channels, activation=nn.GELU):
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))
        assert 2 ** n == patch_size and n >= 1
        ch = [embed_dim]
        for _ in range(n):
            ch.append(max(ch[-1] // 2, out_channels))
        ch.append(out_channels)
        stages = []
        for i in range(n):
            stages.append(nn.Sequential(
                nn.Conv3d(ch[i], ch[i + 1], kernel_size=3, padding=1),
                nn.GroupNorm(1, ch[i + 1]), activation()))
        self.up_stages = nn.ModuleList(stages)
        self.up_factor = 2
        self.final = nn.Conv3d(ch[-2], ch[-1], kernel_size=1)

    def forward(self, x):
        for stage in self.up_stages:
            x = stage(x)
            x = nn.functional.interpolate(x, scale_factor=self.up_factor, mode='trilinear', align_corners=False)
        return self.final(x)


class FlexiCTPrimus2D(nn.Module):
    """FlexiCT-2D backbone + multi-scale patch decode. Input (B,1,H,W) -> (B,num_classes,H,W)."""
    def __init__(self, embed_dim, patch_size, num_classes, dino_encoder, interaction_indices):
        super().__init__()
        self.up_projection = PatchDecode2D(
            patch_size, embed_dim * len(interaction_indices), num_classes)
        self.dino_encoder = dino_encoder
        self.interaction_indices = interaction_indices
        self.decoder = nn.Module()  # placeholder for API compat

    def forward(self, x):
        # FlexiCT in_chans=1: NO repeat. x is (B,1,H,W).
        hier = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        hier = torch.cat(hier, dim=1)  # (B, C*n_levels, H/p, W/p)
        return self.up_projection(hier)


class FlexiCTPrimus3D(nn.Module):
    """FlexiCT-3D backbone + multi-scale patch decode. Input (B,1,D,H,W) -> (B,num_classes,D,H,W).

    Concat multi-scale tokens (C*n_levels wide, NO projection) -> PatchDecode3D.
    The wide (n_levels x) channel bandwidth retains multi-scale detail; this is
    the variant that won under few-shot (vs a 1x1x1 projection that bottlenecked
    detail). Do NOT change the structure — would break checkpoint loading.
    """
    def __init__(self, embed_dim, patch_size, num_classes, dino_encoder, interaction_indices):
        super().__init__()
        self.up_projection = PatchDecode3D(
            patch_size, embed_dim * len(interaction_indices), num_classes)
        self.dino_encoder = dino_encoder
        self.interaction_indices = interaction_indices
        self.decoder = nn.Module()  # placeholder for API compat

    def forward(self, x):
        # FlexiCT in_chans=1: NO repeat. x is (B,1,D,H,W).
        hier = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        hier = torch.cat(hier, dim=1)  # (B, C*n_levels, D/p, H/p, W/p)
        return self.up_projection(hier)
