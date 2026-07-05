"""
Slice-wise 3D encoder: converts 3D volumes to pseudo-3D feature volumes
by encoding each axial slice through a 2D DINOv3 backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SliceWiseEncoder3D(nn.Module):
    """Encode 3D volumes slice-by-slice using a frozen/fine-tuned 2D backbone.

    Input:  (B, 1, D, H, W) volume
    Output: List of (B, D, C, h, w) feature volumes, one per backbone output layer

    The backbone is applied independently to each axial slice. Features are
    stacked along the depth dimension to form pseudo-3D feature volumes.
    """

    def __init__(self, backbone_2d: nn.Module, slice_axis: int = 2):
        """
        Args:
            backbone_2d: DINOv3Backbone instance
            slice_axis: axis to slice along (0=sagittal, 1=coronal, 2=axial)
        """
        super().__init__()
        self.backbone = backbone_2d
        self.slice_axis = slice_axis

    def forward(self, volume_3d: torch.Tensor, slice_batch_size: int = 8) -> List[torch.Tensor]:
        B, C_in, D, H, W = volume_3d.shape

        # Extract slices
        if self.slice_axis == 2:
            slices_2d = [volume_3d[:, :, d, :, :] for d in range(D)]
        elif self.slice_axis == 1:
            slices_2d = [volume_3d[:, :, :, c, :] for c in range(H)]
            D = H
        elif self.slice_axis == 0:
            slices_2d = [volume_3d[:, :, s, :, :] for s in range(W)]
            D = W

        # Preprocess all slices → list of (1, 3, H', W')
        processed = []
        for slc in slices_2d:
            if slc.shape[1] == 1:
                slc = slc.repeat(1, 3, 1, 1)
            slc = F.interpolate(slc, size=(self.backbone.img_size, self.backbone.img_size),
                                mode="bilinear", align_corners=False)
            processed.append(slc)

        # First slice to determine feature shapes
        with torch.no_grad():
            first_feats = self.backbone(processed[0])
        num_layers = len(first_feats)
        fshape = [(B, D) + tuple(f.shape[1:]) for f in first_feats]

        # Process in mini-batches, offload to CPU
        all_features = [torch.zeros(s, device='cpu') for s in fshape]
        for start in range(0, D, slice_batch_size):
            end = min(start + slice_batch_size, D)
            chunk = end - start
            batch = torch.cat(processed[start:end], dim=0)  # (B * chunk, 3, H, W)
            feats = self.backbone(batch)
            for l in range(num_layers):
                _, channels, height, width = feats[l].shape
                reshaped = feats[l].detach().cpu().reshape(chunk, B, channels, height, width)
                all_features[l][:, start:end] = reshaped.permute(1, 0, 2, 3, 4)
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        device = volume_3d.device
        # (B, D, C, h, w) -> (B, C, D, h, w)
        return [f.permute(0, 2, 1, 3, 4).to(device) for f in all_features]

    def forward_sub_volume(
        self, volume_3d: torch.Tensor, sub_volume_size: tuple
    ) -> List[torch.Tensor]:
        """Memory-efficient encoding: process sub-volumes sequentially.

        Args:
            volume_3d: (B, 1, D, H, W)
            sub_volume_size: (d_sub, h_sub, w_sub) — size of each sub-volume

        Returns:
            same as forward(), but with gradient checkpointing per sub-volume
        """
        B, C_in, D, H, W = volume_3d.shape
        d_sub, h_sub, w_sub = sub_volume_size

        # Calculate number of sub-volumes along each dimension
        n_d = max(1, (D + d_sub - 1) // d_sub)
        n_h = max(1, (H + h_sub - 1) // h_sub)
        n_w = max(1, (W + w_sub - 1) // w_sub)

        # Process each sub-volume independently
        all_stacked = None
        for id_ in range(n_d):
            d_start = id_ * d_sub
            d_end = min(d_start + d_sub, D)

            sub_vol = volume_3d[:, :, d_start:d_end, :, :]
            sub_features = self.forward(sub_vol)  # List[(B, d_sub, C, h, w)]

            if all_stacked is None:
                all_stacked = [[] for _ in range(len(sub_features))]
            for layer_idx, feat in enumerate(sub_features):
                # Pad if needed
                if feat.shape[1] < d_sub:
                    pad = torch.zeros(
                        B, d_sub - feat.shape[1], *feat.shape[2:],
                        device=feat.device, dtype=feat.dtype,
                    )
                    feat = torch.cat([feat, pad], dim=2)
                all_stacked[layer_idx].append(feat)

        # Concatenate sub-volumes along depth
        result = [torch.cat(layer_parts, dim=2) for layer_parts in all_stacked]
        return result
