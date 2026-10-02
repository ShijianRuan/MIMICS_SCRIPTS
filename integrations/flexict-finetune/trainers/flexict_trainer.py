"""nnU-Net trainers for FlexiCT few-shot finetuning (2D and 3D).

Validated recipe: FlexiCT ViT backbone (2D or 3D pretrained) + Primus
multi-scale decoder, full finetune, fp32. Two trainers:
  - flexict2d_Trainer  (config 2d,          FlexiCT-2D weights)
  - flexict3d_Trainer  (config 3d_fullres,  FlexiCT-3D weights)

Compatible with BOTH nnU-Net API versions (auto-detected at runtime):
  - nnunetv2 2.5.2: build_network_architecture(architecture_class_name,
        arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels,
        num_output_channels, enable_deep_supervision)        # OLD (6-pos)
  - nnunetv2 2.8.0: build_network_architecture(plans_manager,
        configuration_manager, num_input_channels, num_output_channels,
        enable_deep_supervision)                             # NEW (5-pos)
We accept *args/**kwargs and pull num_input_channels / num_output_channels by
parameter name, so the same class works on either version.

Install (one-time), either way works:
  A) nnunetv2 >= 2.6 (recommended): point the nnUNet_extTrainer env var at the
     repo's trainers/ dir — nnU-Net appends it to its trainer search path:
     export nnUNet_extTrainer=/path/to/flexict-finetune/trainers
  B) any version: copy this file into the nnU-Net env's trainer dir:
     <env>/lib/pythonX.Y/site-packages/nnunetv2/training/nnUNetTrainer/flexict_trainer.py

Env vars:
  FLEXICT_EXT_DIR   dir containing the flexict/ package (default: this file's
                    parent's sibling, i.e. the repo's flexict/ dir)
  FLEXICT2D_CKPT    path to flexict_2d/model.safetensors
  FLEXICT3D_CKPT    path to flexict_3d/model.safetensors
  NUM_EPOCHS        override training length (default 150)
  MIRROR_DISABLE_AXES  comma-separated axes to EXCLUDE from mirroring, e.g. "1"
                    for a single-side organ (never mirror the L-R axis). Default:
                    none excluded (standard nnU-Net mirroring).
"""
import os
import sys
import inspect
import torch
from torch import nn

# Put the flexict/ package on sys.path so `from flexict.models import ...` works
# regardless of where this trainer file lives (repo dir or site-packages).
_EXT_DIR = os.environ.get(
    "FLEXICT_EXT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "flexict")),
)
# If copied into site-packages, FLEXICT_EXT_DIR must be set to the repo's flexict/
# parent. Also accept the repo root (flexict/ as subdir).
for _p in (_EXT_DIR, os.path.dirname(_EXT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# nnU-Net (2.8.0) records init kwargs by introspecting the SUBCLASS __init__
# signature and indexing the PARENT frame's locals() — any parameter name that
# is not also a parameter of nnUNetTrainer.__init__ itself raises KeyError at
# construction (2.5.2 instead accepts unpack_dataset as a real parameter).
# Our trainers keep unpack_dataset in their real signature (so 2.5.2's
# get_trainer_from_args can pass it) but publish the parent's signature, which
# is the only parameter-name set that survives both versions' introspection.
_PARENT_INIT_PARAMS = frozenset(
    inspect.signature(nnUNetTrainer.__init__).parameters)


def _parent_accepts(name):
    return name in _PARENT_INIT_PARAMS


def _publish_parent_init_signature(cls):
    """Make inspect.signature(cls.__init__) report the parent signature."""
    cls.__init__.__signature__ = inspect.signature(nnUNetTrainer.__init__)
    return cls


# --- backbone architecture constants (match the pretrained weights) ---
_EMBED_DIM = 864
_PATCH_SIZE = 8
_INTERACTION_INDICES = [3, 7, 11, 15]


class FlexiCTBaseTrainer(nnUNetTrainer):
    """Base: optimizer (backbone/decoder split), epochs, configurable mirroring.

    Subclasses (flexict2d/3d_Trainer) only override build_network_architecture.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        # nnunetv2 2.5.2's nnUNetTrainer.__init__ takes unpack_dataset;
        # 2.8.0 removed it. Forward it only when the parent accepts it.
        if _parent_accepts("unpack_dataset"):
            super().__init__(plans, configuration, fold, dataset_json,
                             unpack_dataset=unpack_dataset, device=device)
        else:
            super().__init__(plans, configuration, fold, dataset_json,
                             device=device)
        self.initial_lr = 3e-4
        self.vit_lr = 3e-5
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.num_epochs = int(os.environ.get("NUM_EPOCHS", "150"))
        self.save_every = 25
        self.enable_deep_supervision = False
        self.oversample_foreground_percent = 0.33
        # NOTE: batch_size comes from the nnU-Net plans (set during preprocessing).
        # FlexiCT is memory-heavy (175M 2D / 355M 3D, fp32), so after
        # nnUNetv2_plan_and_preprocess you may need to lower the plans' batch_size
        # (scripts/set_plan_batch_size.py does this). We do NOT override
        # batch_size in initialize() — overriding _set_batch_size_and_oversample's
        # result desyncs the oversample schedule from the dataloader.

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Standard nnU-Net rotation + mirroring, with optional axis exclusion.

        Mirroring defaults: 2D -> (0,), 3D -> (0,1,2) (nnU-Net default for
        non-dummy-2d). Set MIRROR_DISABLE_AXES (e.g. "1") to drop axes — use
        this for single-side / asymmetric targets so the lateral axis is never
        mirrored (mirroring it would flip left<->right and cause contralateral
        false positives).
        """
        import numpy as np
        from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        if len(patch_size) == 2:
            do_dummy_2d_data_aug = False
            mirror_axes = (0,)  # nnU-Net 2D default
        else:
            do_dummy_2d_data_aug = False
            mirror_axes = (0, 1, 2)  # nnU-Net 3D default

        disable = os.environ.get("MIRROR_DISABLE_AXES", "")
        if disable.strip():
            disabled = {int(a) for a in disable.split(",") if a.strip().lstrip("-").isdigit()}
            mirror_axes = tuple(a for a in mirror_axes if a not in disabled)
            self.print_to_log_file(
                f"mirror_axes: {mirror_axes} (disabled axes {sorted(disabled) or 'none'})")
        else:
            self.print_to_log_file(f"mirror_axes: {mirror_axes} (nnU-Net default)")

        initial_patch_size = get_patch_size(
            patch_size, rotation_for_DA, rotation_for_DA, rotation_for_DA, (0.85, 1.25))
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def configure_optimizers(self):
        """Two param groups: pretrained backbone (vit_lr) vs decoder (initial_lr).

        Backbone params matched by name fragments: dino_encoder / backbone / vit.
        """
        vit_params, other_params = [], []
        for name, param in self.network.named_parameters():
            if 'dino_encoder' in name or 'backbone' in name or 'vit.' in name:
                vit_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.initial_lr, 'weight_decay': self.weight_decay},
            {'params': vit_params, 'lr': self.vit_lr, 'weight_decay': self.vit_weight_decay},
        ], betas=(0.9, 0.98))
        total_iters = max(self.num_epochs, 1)

        def lr_lambda(current_iter):
            progress = current_iter / max(total_iters, 1)
            return (1 - progress) ** 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler


class FlexiCTFP32Trainer(FlexiCTBaseTrainer):
    """Force fp32 training: FlexiCT RoPE overflows to NaN under fp16 autocast.

    Disables GradScaler and runs the forward without autocast. Validation keeps
    the default (fp16 forward-only is fine; NaN only appears in backward).
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        # nnunetv2 2.5.2's nnUNetTrainer.__init__ takes unpack_dataset;
        # 2.8.0 removed it. Forward it only when the parent accepts it.
        if _parent_accepts("unpack_dataset"):
            super().__init__(plans, configuration, fold, dataset_json,
                             unpack_dataset=unpack_dataset, device=device)
        else:
            super().__init__(plans, configuration, fold, dataset_json,
                             device=device)
        self.grad_scaler = None  # disable GradScaler -> pure fp32

    def train_step(self, batch):
        from nnunetv2.utilities.helpers import dummy_context
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with dummy_context():  # NO autocast -> fp32 (FlexiCT RoPE overflows in fp16)
            output = self.network(data)
            l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


# Both trainer bases declare unpack_dataset in their real signature (needed by
# nnunetv2 2.5.2's get_trainer_from_args), but nnU-Net 2.8.0 introspects the
# subclass signature and indexes the parent frame's locals() — publish the
# parent signature so the kwargs dump only sees names that exist on both sides.
FlexiCTBaseTrainer = _publish_parent_init_signature(FlexiCTBaseTrainer)
FlexiCTFP32Trainer = _publish_parent_init_signature(FlexiCTFP32Trainer)


# ---------------------------------------------------------------------------
# backbone + decoder construction
# ---------------------------------------------------------------------------
def _build_backbone():
    """FlexiCT backbone (2D/3D dual-mode, shared class). No weights loaded here."""
    from flexict.models import flexi_ct_backbone_base
    return flexi_ct_backbone_base(
        patch_size=_PATCH_SIZE, in_chans=1, n_storage_tokens=4,
        qkv_bias=False, mask_k_bias=True,
        drop_path_rate=0.2, layerscale_init=1.0e-05)


def _load_flexict2d(num_classes=2):
    """FlexiCT-2D pretrained backbone + FlexiCTPrimus2D decoder.

    FLEXICT_SKIP_BACKBONE=1 skips the pretrained-backbone init: inference
    rebuilds the network only to have nnU-Net's Predictor load the full
    trained checkpoint into it immediately after, so a migrated model works
    on machines that never had the pretrained backbone files.
    """
    from flexict.flexict_primus import FlexiCTPrimus2D
    from safetensors.torch import load_file
    model = _build_backbone()
    if os.environ.get("FLEXICT_SKIP_BACKBONE", "") != "1":
        ckpt_path = os.environ.get(
            "FLEXICT2D_CKPT",
            os.path.join(os.path.dirname(__file__), "..", "weights", "flexict_2d", "model.safetensors"),
        )
        sd = load_file(ckpt_path)
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, \
            f"flexict2d: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus2D(embed_dim=_EMBED_DIM, patch_size=_PATCH_SIZE, num_classes=num_classes,
                           dino_encoder=model, interaction_indices=_INTERACTION_INDICES)


def _load_flexict3d(num_classes=2):
    """FlexiCT-3D pretrained backbone + FlexiCTPrimus3D decoder.

    FLEXICT_SKIP_BACKBONE=1 skips the pretrained-backbone init (see
    _load_flexict2d).
    """
    from flexict.flexict_primus import FlexiCTPrimus3D
    from safetensors.torch import load_file
    model = _build_backbone()
    if os.environ.get("FLEXICT_SKIP_BACKBONE", "") != "1":
        ckpt_path = os.environ.get(
            "FLEXICT3D_CKPT",
            os.path.join(os.path.dirname(__file__), "..", "weights", "flexict_3d", "model.safetensors"),
        )
        sd = load_file(ckpt_path)
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, \
            f"flexict3d: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus3D(embed_dim=_EMBED_DIM, patch_size=_PATCH_SIZE, num_classes=num_classes,
                           dino_encoder=model, interaction_indices=_INTERACTION_INDICES)


def _parse_build_args(args, kwargs):
    """Extract num_output_channels from either nnU-Net API signature.

    nnU-Net 2.5.2 OLD: (architecture_class_name, arch_init_kwargs,
                        arch_init_kwargs_req_import, num_input_channels,
                        num_output_channels, enable_deep_supervision=True)
    nnU-Net 2.8.0 NEW: (plans_manager, configuration_manager, num_input_channels,
                        num_output_channels, enable_deep_supervision=True)

    Both put num_input_channels / num_output_channels as positional ints near
    the tail. Prefer keyword lookup, fall back to the trailing ints.
    enable_deep_supervision is ignored (FlexiCT decoder is single-scale).

    NOTE: bool is a subclass of int in Python, so we must exclude booleans
    (enable_deep_supervision) when scanning positional args, else it gets
    picked as num_output_channels.
    """
    n_in = kwargs.get("num_input_channels")
    n_out = kwargs.get("num_output_channels")
    if n_in is None or n_out is None:
        ints = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
        if n_in is None and len(ints) >= 1:
            n_in = ints[-2] if len(ints) >= 2 else ints[-1]
        if n_out is None and len(ints) >= 2:
            n_out = ints[-1]
    # single-label binary seg => num_output_channels==2 (bg+fg); fall back to 2
    return (int(n_out) if n_out is not None else 2)


class flexict2d_Trainer(FlexiCTFP32Trainer):
    """FlexiCT-2D few-shot finetune. Use with config `2d`."""

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        num_classes = _parse_build_args(args, kwargs)
        return _load_flexict2d(num_classes=num_classes)


class flexict3d_Trainer(FlexiCTFP32Trainer):
    """FlexiCT-3D few-shot finetune. Use with config `3d_fullres`."""

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        num_classes = _parse_build_args(args, kwargs)
        return _load_flexict3d(num_classes=num_classes)
