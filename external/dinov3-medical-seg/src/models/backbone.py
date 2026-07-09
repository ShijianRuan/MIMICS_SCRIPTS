"""
DINOv3 backbone via HuggingFace transformers (local weights).
"""

import os
import torch
import torch.nn as nn
from transformers import DINOv3ViTBackbone, AutoImageProcessor


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT backbone loaded from local HuggingFace-format weights.

    Args:
        model_path: path to local model dir (config.json + model.safetensors)
        out_indices: which transformer layers to extract (0-indexed, 0-11 for ViT-B)
        freeze: freeze backbone params
    """

    def __init__(
        self,
        model_path: str,
        out_indices: list = None,
        freeze: bool = True,
    ):
        super().__init__()
        if out_indices is None:
            out_indices = [2, 5, 8, 11]

        # Resolve relative path to absolute so transformers doesn't
        # misinterpret it as a HuggingFace Hub repo ID.
        self.model_path = os.path.abspath(model_path)
        self.out_indices = out_indices

        # DINOv3 stage names: stem, stage1, stage2, ..., stage12 (1-indexed)
        # transformer layer i → stage{i+1}
        out_features = [f"stage{i+1}" for i in out_indices]

        self.backbone = DINOv3ViTBackbone.from_pretrained(
            self.model_path,
            out_features=out_features,
            reshape_hidden_states=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)

        cfg = self.backbone.config
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 4)
        self.img_size = getattr(cfg, "image_size", 518)

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images_2d: torch.Tensor) -> list:
        """Extract multi-scale feature maps from 2D images.

        Args:
            images_2d: (B, 3, H, W)

        Returns:
            list of (B, C, H/p, W/p) feature maps
        """
        outputs = self.backbone(images_2d)
        return list(outputs.feature_maps)

    def preprocess(self, images_2d):
        if images_2d.dim() == 3:
            images_2d = images_2d.unsqueeze(0)
        return self.processor(images=images_2d, return_tensors="pt")
