"""
LoRA (Low-Rank Adaptation) for DINOv3 ViT attention layers.
"""

import torch
import torch.nn as nn
import math

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")
SUPPORTED_LORA_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    Applies low-rank update: h = Wx + (alpha/r) * BAx
    where A ∈ R^{r×in}, B ∈ R^{out×r}
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.linear = linear  # frozen original
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        # A: Kaiming uniform init
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        # B: zero init (so training starts with original weights)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()
        self._frozen = False

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if not self._frozen:
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
            result = result + lora_out
        return result

    def set_frozen(self, frozen: bool):
        """Temporarily disable LoRA without removing parameters."""
        self._frozen = frozen

    def merge(self):
        """Merge LoRA weights into the original linear layer (for inference)."""
        merged_weight = self.linear.weight.data + (self.lora_B @ self.lora_A) * self.scale
        self.linear.weight.data = merged_weight
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def unmerge(self):
        """Reverse the merge operation."""
        self.linear.weight.data = self.linear.weight.data - (self.lora_B @ self.lora_A) * self.scale


def apply_lora_to_dinov3(
    backbone: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list = None,
) -> nn.Module:
    """Apply LoRA to DINOv3 ViT attention layers.

    Injects LoRA into the Q and V projections of each transformer block.

    Args:
        backbone: DINOv3ViTBackbone instance (HuggingFace)
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        target_modules: attention projections to target (default: ['q_proj', 'v_proj'])

    Returns:
        backbone with LoRA applied (modified in-place)
    """
    if target_modules is None:
        target_modules = list(DEFAULT_LORA_TARGET_MODULES)
    target_modules = [str(value).strip() for value in target_modules]
    if (
        not target_modules
        or len(target_modules) != len(set(target_modules))
        or any(value not in SUPPORTED_LORA_TARGET_MODULES for value in target_modules)
    ):
        raise ValueError(
            "LoRA target_modules must be a non-empty, unique subset of {}.".format(
                sorted(SUPPORTED_LORA_TARGET_MODULES)
            )
        )

    lora_modules = []

    # HF DINOv3ViTBackbone structure: backbone.model.layer[i].attention.{q_proj,v_proj,...}
    layers = backbone.backbone.model.layer
    for block in layers:
        attn = block.attention
        for module_name in target_modules:
            if hasattr(attn, module_name):
                original = getattr(attn, module_name)
                if isinstance(original, nn.Linear):
                    lora_linear = LoRALinear(original, r=r, alpha=alpha, dropout=dropout)
                    setattr(attn, module_name, lora_linear)
                    lora_modules.append(lora_linear)

    n_layers = len(backbone.backbone.model.layer)
    expected_modules = n_layers * len(target_modules)
    if len(lora_modules) != expected_modules:
        raise RuntimeError(
            "DINOv3 LoRA expected {} configured attention projections but found {}. "
            "The installed Transformers model structure is incompatible.".format(
                expected_modules, len(lora_modules)
            )
        )
    print(
        "Applied LoRA (r={}, alpha={}, targets={}) to {} modules across {} layers".format(
            r,
            alpha,
            ",".join(target_modules),
            len(lora_modules),
            n_layers,
        )
    )
    return backbone


def get_lora_params(backbone: nn.Module) -> list:
    """Get all LoRA parameters for optimizer."""
    lora_params = []
    for name, param in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
    return lora_params


def set_lora_frozen(backbone: nn.Module, frozen: bool):
    """Enable/disable all LoRA modules without removing them."""
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.set_frozen(frozen)


def merge_lora(backbone: nn.Module):
    """Merge all LoRA weights for inference."""
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def get_lora_state_dict(backbone: nn.Module) -> dict:
    """Extract only LoRA parameters for saving."""
    state = {}
    for name, param in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state[name] = param.data.clone()
    return state
