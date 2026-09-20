"""FlexiCT-compatible Primus decoders (2D and 3D).

Same idea as MedDINOv3's Primus_Multiscale (multi-scale token aggregation +
patch-decode upsampling), but:
  * NO x.repeat(1,3,1,1) — FlexiCT in_chans=1, eats single-channel CT directly.
  * 2D variant uses ConvTranspose2d (for FlexiCT-2D weights).
  * 3D variant uses ConvTranspose3d (for FlexiCT-3D weights, true volumetric).

FlexiCT backbone's get_intermediate_layers(reshape=True) returns:
  * 2D input (B,1,H,W) -> list of (B, C, H/p, W/p)
  * 3D input (B,1,D,H,W) -> list of (B, C, D/p, H/p, W/p)
"""
import math
import torch
from torch import nn

from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


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

    Original (no projection): concat 3456 -> PatchDecode3D directly. Kept for exp4
    (checkpoint_final.pth was trained with this structure). Do NOT modify — would break
    checkpoint loading. The v2 variant with 1x1x1 projection is FlexiCTPrimus3D_v2.
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


class FlexiCTPrimus3D_v2(nn.Module):
    """FlexiCT-3D v2: adds 1x1x1 conv projection (concat 3456 -> 864) + GroupNorm before
    PatchDecode, matching the paper's 3D decoder. Upsampling stays trilinear+Conv3d to keep
    activation memory bounded. Used by flexict3d_v2_252_Trainer (new experiment)."""
    def __init__(self, embed_dim, patch_size, num_classes, dino_encoder, interaction_indices):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(embed_dim * len(interaction_indices), embed_dim, kernel_size=1),
            nn.GroupNorm(1, embed_dim),
        )
        self.up_projection = PatchDecode3D(
            patch_size, embed_dim, num_classes)
        self.dino_encoder = dino_encoder
        self.interaction_indices = interaction_indices
        self.decoder = nn.Module()  # placeholder for API compat

    def forward(self, x):
        hier = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        hier = torch.cat(hier, dim=1)  # (B, 3456, D/p, H/p, W/p)
        hier = self.proj(hier)          # (B, embed_dim, D/p, H/p, W/p)
        return self.up_projection(hier)


class PatchDecode3D_Official(nn.Module):
    """Official-aligned 3D patch decode: ConvTranspose3d ladder + LayerNormNd + GELU.

    Mirrors the official FlexiCT primus.py PatchDecode(dim==3) branch exactly:
    each stage is ConvTranspose3d(k=2,s=2) + LayerNormNd + GELU, channel halving;
    final 1x1x1 Conv3d to out_channels. InitWeights_He(1e-2) init.

    Replaces the trilinear+Conv3d PatchDecode3D (which was a memory-saving hack).
    ConvTranspose3d is a learnable transpose conv -> better boundary reconstruction
    (addresses 3D HD95 27-63mm vs 2D 2.66mm). patch_size must be 2^x.
    """
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
                nn.ConvTranspose3d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                LayerNormNd(ch[i + 1]), activation()))
        stages.append(nn.Conv3d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)
        self.decode.apply(InitWeights_He(1e-2))

    def forward(self, x):
        return self.decode(x)


class FlexiCTPrimus3D_Official(nn.Module):
    """Official-aligned 3D multiscale: concat 3456 -> directly into 3456-wide
    ConvTranspose3d decoder, NO projection.

    Mirrors official primus.py Primus_Multiscale(dim=3). The 4x channel bandwidth
    (3456 vs projection-v2's 864) retains multi-scale detail; under 5-shot this
    beats the projection variant (exp4 multiscale 0.77 > exp8 v2 0.59).

    Input (B,1,D,H,W) -> (B,num_classes,D,H,W). FlexiCT in_chans=1, NO repeat.
    """
    def __init__(self, embed_dim, patch_size, num_classes, dino_encoder, interaction_indices):
        super().__init__()
        self.up_projection = PatchDecode3D_Official(
            patch_size, embed_dim * len(interaction_indices), num_classes)
        self.dino_encoder = dino_encoder
        self.interaction_indices = interaction_indices
        self.decoder = self.up_projection  # API compat

    def forward(self, x):
        hier = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        hier = torch.cat(hier, dim=1)  # (B, 3456, D/p, H/p, W/p)
        return self.up_projection(hier)
