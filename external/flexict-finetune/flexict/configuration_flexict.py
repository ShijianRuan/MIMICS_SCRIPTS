"""Transformers configuration for FlexiCT custom-code repos."""

from __future__ import annotations

from transformers import PretrainedConfig


class FlexiCTConfig(PretrainedConfig):
    """Configuration shared by the 2D, 3D, and 3D-VLM FlexiCT variants."""

    model_type = "flexict"

    def __init__(
        self,
        variant: str = "3d",
        image_size: list[int] | None = None,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 864,
        depth: int = 16,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.2,
        layerscale_init: float | None = 1e-5,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        pos_embed_rope_dtype: str = "bf16",
        text_model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        max_text_length: int = 8192,
        text_pooling_type: str = "last_token",
        text_torch_dtype: str = "float32",
        vlm_embed_dim: int = 1024,
        **kwargs,
    ):
        variant = variant.lower()
        if variant not in {"2d", "3d", "vlm"}:
            raise ValueError("variant must be one of '2d', '3d', or 'vlm'")

        if image_size is None:
            image_size = [512, 512] if variant == "2d" else [160, 160, 160]

        self.variant = variant
        self.image_size = [int(v) for v in image_size]
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.qkv_bias = bool(qkv_bias)
        self.drop_path_rate = float(drop_path_rate)
        self.layerscale_init = layerscale_init
        self.norm_layer = norm_layer
        self.ffn_layer = ffn_layer
        self.ffn_bias = bool(ffn_bias)
        self.proj_bias = bool(proj_bias)
        self.n_storage_tokens = int(n_storage_tokens)
        self.mask_k_bias = bool(mask_k_bias)
        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        self.text_model_id = text_model_id
        self.max_text_length = int(max_text_length)
        self.text_pooling_type = text_pooling_type
        self.text_torch_dtype = text_torch_dtype
        self.vlm_embed_dim = int(vlm_embed_dim)
        super().__init__(**kwargs)

    def backbone_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for ``Flexi_CT_Backbone``."""

        return {
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "qkv_bias": self.qkv_bias,
            "drop_path_rate": self.drop_path_rate,
            "layerscale_init": self.layerscale_init,
            "norm_layer": self.norm_layer,
            "ffn_layer": self.ffn_layer,
            "ffn_bias": self.ffn_bias,
            "proj_bias": self.proj_bias,
            "n_storage_tokens": self.n_storage_tokens,
            "mask_k_bias": self.mask_k_bias,
            "pos_embed_rope_dtype": self.pos_embed_rope_dtype,
        }
