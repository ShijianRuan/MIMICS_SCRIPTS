"""
Adapter modules for DINOv3 ViT.
Bottleneck adapters registered as submodules so .to(device) works automatically.
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: down → GELU → up, with zero-init residual."""

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))


def apply_adapter_to_dinov3(backbone, bottleneck=64, dropout=0.0, position="after_attn"):
    """Inject bottleneck adapters into DINOv3 transformer blocks.

    Adapters are registered as submodules so they automatically move
    to the correct device when the model is moved.
    """
    layers = backbone.backbone.model.layer
    count = 0

    for i, block in enumerate(layers):
        adapter = BottleneckAdapter(backbone.embed_dim, bottleneck=bottleneck, dropout=dropout)

        if position == "after_attn":
            block.attention.add_module(f"_adapter_{i}", adapter)
            _orig = block.attention.forward

            def make_new_forward(orig, adp):
                def new_forward(hidden_states, *args, **kwargs):
                    out = orig(hidden_states, *args, **kwargs)
                    ctx = out[0] if isinstance(out, tuple) else out
                    adapted = ctx + adp(ctx)
                    return (adapted,) + out[1:] if isinstance(out, tuple) else adapted
                return new_forward

            block.attention.forward = make_new_forward(_orig, adapter)

        elif position == "after_mlp":
            block.mlp.add_module(f"_adapter_{i}", adapter)
            _orig = block.mlp.forward

            def make_new_forward(orig, adp):
                def new_forward(hidden_states, *args, **kwargs):
                    out = orig(hidden_states, *args, **kwargs)
                    return out + adp(out)
                return new_forward

            block.mlp.forward = make_new_forward(_orig, adapter)

        count += 1

    print(f"Applied {position} adapter (bn={bottleneck}) to {count} blocks "
          f"across {len(layers)} layers")
    return backbone
