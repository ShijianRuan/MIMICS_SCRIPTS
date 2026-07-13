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

    def __init__(
        self,
        backbone_2d: nn.Module,
        channel_policy: str = "repeat",
        neighbor_distance_mm: float | None = None,
    ):
        """
        Args:
            backbone_2d: DINOv3Backbone instance
            channel_policy: ``repeat``/``single`` or ``2_5d``. ``ct_windows``
                is already three-channel before it reaches this module.
            neighbor_distance_mm: physical offset for 2.5D neighbour slices.
                The caller supplies ``spacing_zyx`` at forward time.
        """
        super().__init__()
        self.backbone = backbone_2d
        self.channel_policy = str(channel_policy or "repeat").lower()
        self.neighbor_distance_mm = (
            None if neighbor_distance_mm is None else float(neighbor_distance_mm)
        )

    def forward(
        self,
        volume_3d: torch.Tensor,
        slice_batch_size: int = 8,
        spacing_zyx: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        B, C_in, D, H, W = volume_3d.shape
        if C_in not in (1, 3):
            raise RuntimeError("Expected one or three input channels, got {}".format(C_in))
        if D < 1:
            raise RuntimeError("Cannot encode an empty depth axis")

        neighbour_offset = 1
        if self.neighbor_distance_mm is not None:
            if spacing_zyx is None:
                raise RuntimeError("2.5D physical neighbour distance requires spacing_zyx")
            spacing = float(torch.as_tensor(spacing_zyx).reshape(-1, 3)[0, 0].item())
            if spacing <= 0:
                raise RuntimeError("Invalid Z spacing for 2.5D input: {}".format(spacing))
            neighbour_offset = max(1, int(round(self.neighbor_distance_mm / spacing)))

        # Prepare all axial model-space slices as (B, 3, H, W).
        processed = []
        for depth_index in range(D):
            if C_in == 3:
                slc = volume_3d[:, :, depth_index, :, :]
            elif self.channel_policy in ("2_5d", "2.5d"):
                before = max(0, depth_index - neighbour_offset)
                after = min(D - 1, depth_index + neighbour_offset)
                slc = torch.cat(
                    [
                        volume_3d[:, :, before, :, :],
                        volume_3d[:, :, depth_index, :, :],
                        volume_3d[:, :, after, :, :],
                    ],
                    dim=1,
                )
            else:
                slc = volume_3d[:, :, depth_index, :, :].repeat(1, 3, 1, 1)
            # DINOv3 uses dynamic RoPE coordinates and supports variable input
            # sizes. Resizing every candidate back to config.image_size here
            # would make a 224-vs-320 experiment a meaningless double-resize.
            patch_size = int(getattr(self.backbone, "patch_size", 1))
            if slc.shape[-2] % patch_size or slc.shape[-1] % patch_size:
                raise RuntimeError(
                    "DINO input size {} must be divisible by patch size {}".format(
                        tuple(slc.shape[-2:]), patch_size
                    )
                )
            processed.append(slc)

        num_layers = len(self.backbone.out_indices)

        # Process in mini-batches, preserving gradient flow through the backbone
        # (LoRA weights etc.).  We accumulate outputs in a list and torch.cat
        # at the end — in-place slice assignment would break the autograd graph.
        layer_buffers = [[] for _ in range(num_layers)]
        for start in range(0, D, slice_batch_size):
            end = min(start + slice_batch_size, D)
            chunk = end - start
            batch = torch.cat(processed[start:end], dim=0)  # (B * chunk, 3, H, W)
            feats = self.backbone(batch)
            for l in range(num_layers):
                _, channels, height, width = feats[l].shape
                reshaped = feats[l].reshape(chunk, B, channels, height, width)
                layer_buffers[l].append(reshaped.permute(1, 0, 2, 3, 4))

        # Concatenate along depth, then convert to (B, C, D, h, w)
        result = [torch.cat(buf, dim=1) for buf in layer_buffers]
        return [f.permute(0, 2, 1, 3, 4) for f in result]

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
            sub_features = self.forward(sub_vol)  # List[(B, C, d_sub, h, w)]

            if all_stacked is None:
                all_stacked = [[] for _ in range(len(sub_features))]
            for layer_idx, feat in enumerate(sub_features):
                # Pad if needed
                if feat.shape[2] < d_sub:
                    pad = torch.zeros(
                        B, feat.shape[1], d_sub - feat.shape[2], *feat.shape[3:],
                        device=feat.device, dtype=feat.dtype,
                    )
                    feat = torch.cat([feat, pad], dim=2)
                all_stacked[layer_idx].append(feat)

        # Concatenate sub-volumes along depth
        result = [torch.cat(layer_parts, dim=2) for layer_parts in all_stacked]
        return result
