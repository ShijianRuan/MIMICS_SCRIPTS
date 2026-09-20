"""Transformers model wrappers for FlexiCT custom-code repos."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_flexict import FlexiCTConfig

try:  # Local Hub repo dynamic-module imports.
    from .models import Flexi_CT_Backbone
    from .text_tower import build_hf_text_model
    from .layers import LayerScale as _FlexiCTLayerScale
    from .attention import SelfAttention as _FlexiCTSelfAttention
    from .block import SelfAttentionBlock as _FlexiCTSelfAttentionBlock
    from .ffn_layers import Mlp as _FlexiCTMlp
    from .layer_scale import LayerScale as _FlexiCTLayerScaleDirect
    from .patch_embed import PatchEmbed as _FlexiCTPatchEmbed
    from .rms_norm import RMSNorm as _FlexiCTRMSNorm
    from .rope_position_encoding import RopePositionEmbedding as _FlexiCTRopePositionEmbedding
    from .utils import named_apply as _flexict_named_apply
except ImportError:  # Source-tree tests import from hf_package.
    from flexi_ct.models import Flexi_CT_Backbone
    from flexi_ct.text_tower import build_hf_text_model


@dataclass
class FlexiCTModelOutput(ModelOutput):
    cls_token: torch.FloatTensor | None = None
    patch_tokens: torch.FloatTensor | None = None


@dataclass
class FlexiCTVLMOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    logits_per_image: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    logit_scale: torch.FloatTensor | None = None


def _build_vision_backbone(config: FlexiCTConfig) -> nn.Module:
    backbone = Flexi_CT_Backbone(**config.backbone_kwargs())
    if hasattr(backbone, "init_weights"):
        backbone.init_weights()
    return backbone


class FlexiCTPreTrainedModel(PreTrainedModel):
    config_class = FlexiCTConfig
    base_model_prefix = "flexict"
    supports_gradient_checkpointing = False
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class FlexiCTModel(FlexiCTPreTrainedModel):
    """FlexiCT vision encoder for the 2D and 3D checkpoints."""

    def __init__(self, config: FlexiCTConfig):
        super().__init__(config)
        if config.variant not in {"2d", "3d"}:
            raise ValueError("FlexiCTModel only supports variant='2d' or variant='3d'")
        self.backbone = _build_vision_backbone(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool | None = None,
        **_: object,
    ) -> FlexiCTModelOutput | tuple[torch.Tensor, torch.Tensor]:
        if pixel_values is None:
            raise ValueError("pixel_values is required")
        if self.config.variant == "2d" and pixel_values.dim() != 4:
            raise ValueError(f"FlexiCT-2D expects [B, C, H, W], got {tuple(pixel_values.shape)}")
        if self.config.variant == "3d" and pixel_values.dim() != 5:
            raise ValueError(f"FlexiCT-3D expects [B, C, D, H, W], got {tuple(pixel_values.shape)}")

        features = self.backbone(pixel_values, is_training=True)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        if return_dict if return_dict is not None else self.config.use_return_dict:
            return FlexiCTModelOutput(cls_token=cls_token, patch_tokens=patch_tokens)
        return cls_token, patch_tokens


class FlexiCTVLMModel(FlexiCTPreTrainedModel):
    """FlexiCT 3D vision-language model with Qwen3 embedding text tower."""

    def __init__(self, config: FlexiCTConfig):
        super().__init__(config)
        if config.variant != "vlm":
            raise ValueError("FlexiCTVLMModel requires variant='vlm'")

        self.vision_model = _build_vision_backbone(config)
        self.text_model = build_hf_text_model(
            model_name_or_path=config.text_model_id,
            embed_dim=config.vlm_embed_dim,
            pooling_type=config.text_pooling_type,
            use_flash_attention=False,
            torch_dtype=config.text_torch_dtype,
            freeze_backbone=False,
            use_projection=True,
            max_length=config.max_text_length,
            padding_side="left",
        )
        self._build_empty_text_backbone()
        self.logit_scale = nn.Parameter(torch.ones(1))
        self.vlm_vision_projection = nn.Linear(2 * config.embed_dim, config.vlm_embed_dim, bias=False)

    def _build_empty_text_backbone(self) -> None:
        self.text_model.build_backbone_from_config()
        backbone = self.text_model.backbone
        if hasattr(backbone, "to_empty"):
            backbone.to_empty(device=torch.device("cpu"))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_model(pixel_values, is_training=True)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        pooled = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        return F.normalize(self.vlm_vision_projection(pooled), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **_: object,
    ) -> FlexiCTVLMOutput | tuple[torch.Tensor | None, ...]:
        image_embeds = self.encode_image(pixel_values) if pixel_values is not None else None
        text_embeds = self.encode_text(input_ids, attention_mask) if input_ids is not None else None

        logits_per_image = None
        logits_per_text = None
        logit_scale = self.logit_scale.exp()
        if image_embeds is not None and text_embeds is not None:
            logits_per_image = logit_scale * image_embeds @ text_embeds.T
            logits_per_text = logits_per_image.T

        if return_dict if return_dict is not None else self.config.use_return_dict:
            return FlexiCTVLMOutput(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
                logit_scale=logit_scale,
            )
        return image_embeds, text_embeds, logits_per_image, logits_per_text, logit_scale
