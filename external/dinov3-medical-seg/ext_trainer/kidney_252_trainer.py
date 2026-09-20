"""nnU-Net 2.5.2 compatible trainers for kidney_left (4 experiments).

Built for the remote nnUnet env (nnunetv2 2.5.2, python 3.9, torch 2.1.0).
2.5.2's build_network_architecture uses the OLD signature:
    (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
     num_input_channels, num_output_channels, enable_deep_supervision)

Install: copy this file to
  <env>/lib/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/kidney_252_trainer.py
Then train:
  nnUNetv2_train 901 2d_512 0 -tr kidneymeddinov3_252_Trainer

All 4 trainers inherit kidney_base_252_Trainer which provides:
  * L-R mirror fix (mirror_axes=(0,), +-30deg rotation) for single-side organ
  * configure_optimizers (2 param groups: backbone vit_lr, decoder initial_lr)
  * num_epochs from NUM_EPOCHS env (default 150)
"""
import os, sys
import torch
from torch import nn

_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
# When this file is in site-packages/nnunetv2/training/nnUNetTrainer/, we need
# the package dir (with dinov3/ and flexict/) on sys.path. Point at the portable
# package's ext_trainer dir.
_PKG_EXT = os.environ.get(
    "KIDNEY_EXT_DIR",
    r"/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer",
)
for _p in (_PKG_EXT, os.path.join(_PKG_EXT, "dinov3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Merlin-nnUNet repo (I3ResNet CLIP 3D CT foundation model). Only imported lazily
# inside _load_merlin(), so absence of the repo does not break other trainers.
_MERLIN_DIR = os.environ.get(
    "MERLIN_DIR", r"/home/wenwen_zhang/Merlin-nnUNet")

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class kidney_base_252_Trainer(nnUNetTrainer):
    """Base: L-R mirror fix + optimizer + epochs. Subclasses override build_network."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=unpack_dataset, device=device)
        self.initial_lr = 3e-4
        self.vit_lr = 3e-5
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.num_epochs = int(os.environ.get("NUM_EPOCHS", "150"))
        self.save_every = 25
        self.enable_deep_supervision = False
        self.oversample_foreground_percent = 0.33

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Disable L-R mirroring for single-side organ (kidney_left).
        2D: mirror only axis 0 (A-P), +-30deg rotation.
        3D: mirror axes 0,2 (A-P and S-I, NOT axis 1=L-R), +-30deg rotation on all axes.
        Never mirror the L-R axis (axis 1 for both 2D and 3D after nnU-Net transpose).
        """
        import numpy as np
        from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        if len(patch_size) == 2:
            do_dummy_2d_data_aug = False
            mirror_axes = (0,)  # axis 0 = A-P; axis 1 = L-R (disabled)
            initial_patch_size = get_patch_size(patch_size, rotation_for_DA, rotation_for_DA,
                                                rotation_for_DA, (0.85, 1.25))
        else:
            # 3D: axes are (z=0 A-P-ish, y=1 L-R, x=2 S-I). Mirror 0 and 2, NOT 1 (L-R).
            do_dummy_2d_data_aug = False
            mirror_axes = (0, 2)
            initial_patch_size = get_patch_size(patch_size, rotation_for_DA, rotation_for_DA,
                                                rotation_for_DA, (0.85, 1.25))
        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.print_to_log_file(f'mirror_axes: {mirror_axes} (L-R DISABLED for single-side organ)')
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def configure_optimizers(self):
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


class frozen_encoder_Trainer(kidney_base_252_Trainer):
    """Freeze backbone encoder, train decoder only. Same fp16 as exp1/2 (DINOv3 OK in fp16).

    For ablation B (exp5/exp6): isolate value of training encoder vs frozen.
    exp5 = CT-3M frozen, exp6 = original DINOv3 frozen. Compare to exp1/exp2 (full).
    """

    def initialize(self):
        super().initialize()
        # freeze backbone after network is built
        for name, param in self.network.named_parameters():
            if 'dino_encoder' in name or 'backbone' in name or 'vit.' in name:
                param.requires_grad = False
        # set backbone to eval mode (no dropout/BN update) — but it still forward in train_step
        self.print_to_log_file("[frozen] encoder params frozen, decoder only trainable")

    def configure_optimizers(self):
        # only decoder params (backbone frozen, not in optimizer)
        other_params = [p for n, p in self.network.named_parameters()
                        if not ('dino_encoder' in n or 'backbone' in n or 'vit.' in n)]
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.initial_lr, 'weight_decay': self.weight_decay},
        ], betas=(0.9, 0.98))
        total_iters = max(self.num_epochs, 1)
        def lr_lambda(current_iter):
            progress = current_iter / max(total_iters, 1)
            return (1 - progress) ** 1.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler

    def on_epoch_start(self):
        # keep backbone in eval mode every epoch (no BN/dropout updates)
        super().on_epoch_start()
        if hasattr(self.network, 'dino_encoder'):
            self.network.dino_encoder.eval()
        elif hasattr(self.network, 'backbone'):
            self.network.backbone.eval()
        elif hasattr(self.network, 'vit'):
            self.network.vit.eval()


class flexict_fp32_Trainer(kidney_base_252_Trainer):
    """FlexiCT backbone (RoPE + 864-dim attn) produces NaN under fp16 autocast.
    Force fp32 training/inference. DINOv3 trainers (exp1/2) keep default fp16.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=unpack_dataset, device=device)
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
        # NO autocast -> fp32 (FlexiCT RoPE overflows in fp16)
        with dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    # validation_step: keep default (fp16 forward-only is fine; NaN only in backward).
    # Default validation_step returns tp_hard/fp_hard/fn_hard needed by on_validation_epoch_end.



def _load_meddinov3_ct3m():
    """MedDINOv3 CT-3M ViT-B/16 + Primus_Multiscale."""
    from dinov3.models.vision_transformer import vit_base
    from dinov3.models.primus import Primus_Multiscale
    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05,
                     n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    ckpt_path = os.environ.get(
        "MEDDINOV3_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/ct3m/model.pth")
    chkpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = chkpt['teacher']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()
                  if 'ibot' not in k and 'dino_head' not in k}
    model.load_state_dict(state_dict, strict=True)
    return Primus_Multiscale(embed_dim=768, patch_embed_size=16, num_classes=2,
                             dino_encoder=model, interaction_indices=[2, 5, 8, 11])


def _load_original_dinov3():
    """Original DINOv3 ViT-B/16 (natural image LVD-1689M) + Primus_Multiscale."""
    from dinov3.models.vision_transformer import vit_base
    from dinov3.models.primus import Primus_Multiscale
    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05,
                     n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    ckpt_path = os.environ.get(
        "ORIGINALDINOV3_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/dinov3_vitb16_lvd1689m_native.pth")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert all('rope_embed' in k for k in missing), f"unexpected missing: {missing}"
    return Primus_Multiscale(embed_dim=768, patch_embed_size=16, num_classes=2,
                             dino_encoder=model, interaction_indices=[2, 5, 8, 11])


def _load_flexict2d():
    """FlexiCT-2D (864/16/patch8) + FlexiCTPrimus2D."""
    from flexict.models import flexi_ct_backbone_base
    from flexict.flexict_primus import FlexiCTPrimus2D
    from safetensors.torch import load_file
    model = flexi_ct_backbone_base(patch_size=8, in_chans=1, n_storage_tokens=4,
                                    qkv_bias=False, mask_k_bias=True,
                                    drop_path_rate=0.2, layerscale_init=1.0e-05)
    ckpt_path = os.environ.get(
        "FLEXICT2D_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/flexict_2d/model.safetensors")
    sd = load_file(ckpt_path)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"flexict2d: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus2D(embed_dim=864, patch_size=8, num_classes=2,
                           dino_encoder=model, interaction_indices=[3, 7, 11, 15])


def _load_flexict3d():
    """FlexiCT-3D (true 3D) + FlexiCTPrimus3D."""
    from flexict.models import flexi_ct_backbone_base
    from flexict.flexict_primus import FlexiCTPrimus3D
    from safetensors.torch import load_file
    model = flexi_ct_backbone_base(patch_size=8, in_chans=1, n_storage_tokens=4,
                                    qkv_bias=False, mask_k_bias=True,
                                    drop_path_rate=0.2, layerscale_init=1.0e-05)
    ckpt_path = os.environ.get(
        "FLEXICT3D_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/flexict_3d/model.safetensors")
    sd = load_file(ckpt_path)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"flexict3d: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus3D(embed_dim=864, patch_size=8, num_classes=2,
                           dino_encoder=model, interaction_indices=[3, 7, 11, 15])


def _load_flexict3d_v2():
    """FlexiCT-3D v2: backbone (same weights) + FlexiCTPrimus3D_v2 (1x1x1 projection decoder).
    Backbone loading identical to _load_flexict3d; only the decoder wrapper differs."""
    from flexict.models import flexi_ct_backbone_base
    from flexict.flexict_primus import FlexiCTPrimus3D_v2
    from safetensors.torch import load_file
    model = flexi_ct_backbone_base(patch_size=8, in_chans=1, n_storage_tokens=4,
                                    qkv_bias=False, mask_k_bias=True,
                                    drop_path_rate=0.2, layerscale_init=1.0e-05)
    ckpt_path = os.environ.get(
        "FLEXICT3D_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/flexict_3d/model.safetensors")
    sd = load_file(ckpt_path)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"flexict3d_v2: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus3D_v2(embed_dim=864, patch_size=8, num_classes=2,
                              dino_encoder=model, interaction_indices=[3, 7, 11, 15])


def _load_flexict3d_official():
    """FlexiCT-3D official-aligned: same backbone weights + FlexiCTPrimus3D_Official
    (multiscale, NO projection, ConvTranspose3d decoder matching official primus.py).
    Backbone loading identical to _load_flexict3d; only the decoder wrapper differs."""
    from flexict.models import flexi_ct_backbone_base
    from flexict.flexict_primus import FlexiCTPrimus3D_Official
    from safetensors.torch import load_file
    model = flexi_ct_backbone_base(patch_size=8, in_chans=1, n_storage_tokens=4,
                                    qkv_bias=False, mask_k_bias=True,
                                    drop_path_rate=0.2, layerscale_init=1.0e-05)
    ckpt_path = os.environ.get(
        "FLEXICT3D_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/flexict_3d/model.safetensors")
    sd = load_file(ckpt_path)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"flexict3d_off: missing={missing[:3]} unexp={unexpected[:3]}"
    return FlexiCTPrimus3D_Official(embed_dim=864, patch_size=8, num_classes=2,
                                    dino_encoder=model, interaction_indices=[3, 7, 11, 15])


class kidneymeddinov3_252_Trainer(kidney_base_252_Trainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_meddinov3_ct3m()


class originaldinov3_252_Trainer(kidney_base_252_Trainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_original_dinov3()


class kidneymeddinov3_frozen_252_Trainer(frozen_encoder_Trainer):
    """exp5: CT-3M weights, encoder FROZEN, decoder only."""
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_meddinov3_ct3m()


class originaldinov3_frozen_252_Trainer(frozen_encoder_Trainer):
    """exp6: original DINOv3 weights, encoder FROZEN, decoder only."""
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_original_dinov3()


class flexict2d_252_Trainer(flexict_fp32_Trainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_flexict2d()


class flexict3d_252_Trainer(flexict_fp32_Trainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_flexict3d()


class flexict3d_v2_252_Trainer(flexict_fp32_Trainer):
    """exp4 v2: FlexiCT-3D + decoder with 1x1x1 projection (3456->864) before PatchDecode.
    Same backbone/weights/mirror/lr/epochs as flexict3d, only the decoder projection differs.
    """
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_flexict3d_v2()


class flexict3d_off_252_Trainer(flexict_fp32_Trainer):
    """T1: official-aligned 3D baseline. FlexiCT-3D + FlexiCTPrimus3D_Official
    (multiscale, NO projection, ConvTranspose3d decoder matching official primus.py)
    + official-aligned hyperparams: vit_lr=1e-4 (was 3e-5), grad_clip=1.0 (was 12),
    default epochs 300 (was 150). Patch/spacing unchanged (96^3 @ 1.5mm) to isolate
    the effect of decoder-upsampling + hyperparam alignment vs exp4 (trilinear + 3e-5/12/150).
    Inherits fp32 train (RoPE fp16 NaN), L-R mirror fix, grouped optimizer from flexict_fp32_Trainer.
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=unpack_dataset, device=device)
        self.vit_lr = 1e-4                      # official-aligned (base uses 3e-5)
        self.num_epochs = int(os.environ.get("NUM_EPOCHS", "300"))  # default 300 (base 150)
        self._grad_clip = 1.0                   # official-aligned (base uses 12)
        self._clip_logged = False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_flexict3d_official()

    def train_step(self, batch):
        """fp32 train_step (same as flexict_fp32_Trainer) but grad_clip=1.0 (official)
        instead of 12. Tighter clipping stabilizes 3D training (old exp4 ±0.25 EMA swings
        traced partly to grad_clip=12 being too loose)."""
        from nnunetv2.utilities.helpers import dummy_context
        if not self._clip_logged:
            self.print_to_log_file(f"[T1] vit_lr={self.vit_lr} grad_clip={self._grad_clip} "
                                   f"num_epochs={self.num_epochs}")
            self._clip_logged = True
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
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._grad_clip)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


def _load_merlin(num_output_channels):
    """Merlin (Stanford 3D CT CLIP foundation model) visual encoder + UNetDecoder.

    Backbone: I3ResNet (ResNet-152 inflated to 3D), pretrained via 3D-CT contrastive
    vision-language learning. We use ONLY the visual encoder (encode_image), NOT the
    text encoder (Clinical-Longformer) — avoids downloading it and is irrelevant to
    segmentation. Decoder: UNetDecoder with ConvTranspose3d upsampling + skip conns.

    Architecture-wise this is the ONLY CNN-backbone experiment (vs ViT in
    flexict3d/dinounetr), and its pretraining is 3D-CT-specific contrastive (vs
    FlexiCT CT-specific / DINOv2 natural-image). Enables comparing backbone type +
    pretraining domain under 5-shot.
    """
    # Merlin's models/ (clip_model_3d, unet_decoder, src/i3res) are installed INTO
    # site-packages/nnunetv2 (copied there) to avoid the Merlin repo's own nnunetv2
    # shadowing the env's nnunetv2. Import from the env nnunetv2 directly — NO sys.path
    # manipulation (that would break other trainers by shadowing).
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models.clip_model_3d import \
        ImageEncoder
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import unet_decoder

    model_config = {
        "architecture": "i3_resnet_clinical_longformer",
        "text_encoder": "clinical_longformer",
        "use_ehr": True,
    }
    # ImageEncoder builds only the I3ResNet visual backbone (no text encoder download).
    encoder = ImageEncoder(model_config)

    # Load Merlin CLIP pretrained weights (visual encoder subset).
    ckpt_path = os.environ.get(
        "MERLIN_CKPT",
        "/home/wenwen_zhang/kidney_experiments_portable/weights/merlin/"
        "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt")
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state_dict = encoder.state_dict()
        # Merlin checkpoint keys are prefixed with encode_image.i3_resnet.* (full Clip3D).
        # Strip the "encode_image." prefix to match the standalone ImageEncoder.
        filtered = {}
        for k, v in checkpoint.items():
            kk = k.replace("encode_image.", "") if k.startswith("encode_image.") else k
            if kk in model_state_dict and model_state_dict[kk].size() == v.size():
                filtered[kk] = v
        missing, unexpected = encoder.load_state_dict(filtered, strict=False)
        print(f"[Merlin] loaded {len(filtered)}/{len(model_state_dict)} encoder keys; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[Merlin] WARNING: checkpoint not found at {ckpt_path}, training from "
              f"ImageNet-resnet152 inflated init (no CLIP pretrain)")

    decoder = unet_decoder.UNetDecoder(
        num_classes=num_output_channels, deep_supervision=False)

    # Wrap in a Module (NOT Sequential) so nnU-Net can access network.decoder
    # (nnUNetTrainer.set_deep_supervision_enabled does mod.decoder.deep_supervision).
    # encoder.forward returns (contrastive, ehr, skips); decoder.forward unpacks
    # skips = inputs[2] then upsamples. Matches official nnUNetTrainerMerlin wiring.
    class MerlinSegModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder  # exposes .deep_supervision attr for nnU-Net

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = MerlinSegModel(encoder, decoder)
    return model


class merlin_252_Trainer(flexict_fp32_Trainer):
    """Merlin 3D CT CLIP foundation model (I3ResNet CNN backbone) + UNetDecoder,
    full fine-tune on kidney_left CT. Aligned to flexict3d for ablation: same 3d_96
    patch, fp32, L-R mirror fix, Dice+CE, 150 epochs, deep supervision off.

    Only the backbone type (CNN vs ViT) and pretraining domain (3D-CT contrastive CLIP)
    differ from flexict3d/dinounetr — isolates the effect of backbone architecture +
    pretraining strategy under 5-shot.

    NOTE: I3ResNet params (i3_resnet.*) must go to the backbone (vit_lr) group; the
    base configure_optimizers keys on 'dino_encoder'/'backbone'/'vit.' which do NOT
    match I3ResNet. We override configure_optimizers to route i3_resnet.* -> backbone.
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=unpack_dataset, device=device)
        self.vit_lr = 3e-5   # same as flexict3d backbone lr for fair comparison
        self.num_epochs = int(os.environ.get("NUM_EPOCHS", "150"))
        # Merlin I3ResNet forward overflows in fp16 (inf->nan), and mirror TTA averages
        # flipped fp16 logits that collapse to background. Disable mirror TTA entirely
        # so the checkpoint stores no mirror axes -> predict/validation skip mirroring.
        self.inference_allowed_mirroring_axes = None

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return _load_merlin(num_output_channels)

    def configure_optimizers(self):
        """Route I3ResNet backbone (i3_resnet.*) to vit_lr group, decoder to initial_lr.
        Overrides base because I3ResNet param names don't match base's backbone filters."""
        vit_params, other_params = [], []
        for name, param in self.network.named_parameters():
            if 'i3_resnet' in name or 'encode_image' in name:
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

    def validation_step(self, batch: dict) -> dict:
        """fp32 validation (Merlin I3ResNet forward overflows in fp16 -> val_loss=nan
        since ~epoch 17). Override base validation_step which uses autocast(fp16).
        Mirrors nnUNetTrainer.validation_step logic but with dummy_context (no autocast)."""
        from nnunetv2.utilities.helpers import dummy_context
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with dummy_context():  # NO autocast -> fp32
            output = self.network(data)
            l = self.loss(output, target)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        mask = None
        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target[target == self.label_manager.ignore_label] = 0
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Force fp32 for the predict forward (Merlin I3ResNet overflows in fp16).
        nnU-Net 2.5.2's predictor hardcodes autocast(enabled=True); patch torch.autocast
        to no-op during validation predict. Mirror TTA already disabled via
        inference_allowed_mirroring_axes=None."""
        _orig_autocast = torch.autocast
        def _disabled_autocast(*a, **k):
            k['enabled'] = False
            return _orig_autocast(a[0] if a else 'cuda', **k)
        torch.autocast = _disabled_autocast
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            torch.autocast = _orig_autocast


class flexict3d_aug_252_Trainer(flexict3d_252_Trainer):
    """T2b: FlexiCT-3D (exp4 config) + STRONGER data augmentation to combat 5-shot
    overfitting (EMA pseudo-dice 0.90 on 3 val cases but only 0.77 mean dice on 100 test).

    Inherits everything from flexict3d_252_Trainer (fp32, L-R mirror fix, grouped
    optimizer vit_lr=3e-5, 150 epochs). Only overrides:
      * get_training_transforms — stronger elastic/brightness/contrast/gamma/sim_lowres/gaussian_noise
      * oversample_foreground_percent = 0.5 (up from 0.33) for tiny 0.1-0.7% fg kidney

    Augmentation changes vs nnU-Net 2.5.2 default get_training_transforms:
      - Elastic deformation: ENABLED (default p=0) -> p=0.35, scale=(0.05,0.1) sigma,
        magnitude=(0,10) px
      - Brightness: range (0.75,1.25)->(0.6,1.4), p 0.15->0.25
      - Contrast: range (0.75,1.25)->(0.5,1.75), p 0.15->0.25
      - Gamma (non-invert): range (0.7,1.5)->(0.5,1.75), p 0.3->0.4
      - Gamma (invert): range (0.7,1.5)->(0.5,1.75), p 0.1->0.15
      - SimulateLowRes: scale (0.5,1)->(0.2,1) (downsample factor 1-5x), p stays 0.25
      - GaussianNoise: variance (0,0.1)->(0,0.15), p 0.1->0.15
      - GaussianBlur: unchanged (p=0.2)
      - Rotation/scaling/mirror: unchanged (rotation p=0.2, scaling p=0.2, mirror from base)
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.oversample_foreground_percent = 0.5

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[T2b] strong augmentation: elastic p0.35, brightness/contrast/gamma widened, "
            "sim_lowres p0.25, gaussian_noise p0.15, oversample 0.5")

    def get_training_transforms(
            self,
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
    ):
        # Imports matching nnUNetTrainer.get_training_transforms (nnunetv2 2.5.2)
        from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
        from batchgeneratorsv2.transforms.intensity.brightness import \
            MultiplicativeBrightnessTransform
        from batchgeneratorsv2.transforms.intensity.contrast import \
            ContrastTransform, BGContrast
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
        from batchgeneratorsv2.transforms.intensity.gaussian_noise import \
            GaussianNoiseTransform
        from batchgeneratorsv2.transforms.noise.gaussian_blur import \
            GaussianBlurTransform
        from batchgeneratorsv2.transforms.spatial.low_resolution import \
            SimulateLowResolutionTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import \
            DownsampleSegForDSTransform
        from batchgeneratorsv2.transforms.utils.nnunet_masking import \
            MaskImageTransform
        from batchgeneratorsv2.transforms.utils.pseudo2d import \
            Convert3DTo2DTransform, Convert2DTo3DTransform
        from batchgeneratorsv2.transforms.utils.random import RandomTransform
        from batchgeneratorsv2.transforms.utils.remove_label import \
            RemoveLabelTansform
        from batchgeneratorsv2.transforms.utils.seg_to_regions import \
            ConvertSegmentationToRegionsTransform

        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # --- SpatialTransform: ENABLE elastic deformation (default p=0), keep rotation/scaling ---
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
                p_elastic_deform=0,  # elastic disabled: batchgeneratorsv2 elastic_deform_scale API misused (needs per-axis len-3 not range), causes IndexError swallowed by worker -> data-seg desync. Keep other aug.
                elastic_deform_scale=(0.05, 0.1),       # sigma = 5-10% of patch edge (~5-10 px for 96)
                elastic_deform_magnitude=(0, 10),        # max displacement 0-10 pixels
                p_synchronize_def_scale_across_axes=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # --- Gaussian noise: p 0.1 -> 0.15, variance widened ---
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.15),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.15
        ))
        # --- Gaussian blur: unchanged ---
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        # --- Brightness: range (0.75,1.25) -> (0.6,1.4), p 0.15 -> 0.25 ---
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.6, 1.4)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25
        ))
        # --- Contrast: range (0.75,1.25) -> (0.5,1.75), p 0.15 -> 0.25 ---
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.5, 1.75)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25
        ))
        # --- Simulated low resolution: scale (0.5,1) -> (0.2,1) (factor 1-5x downsample), p stays 0.25 ---
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.2, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        # --- Gamma (inverted): range (0.7,1.5) -> (0.5,1.75), p 0.1 -> 0.15 ---
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.5, 1.75)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.15
        ))
        # --- Gamma (non-inverted): range (0.7,1.5) -> (0.5,1.75), p 0.3 -> 0.4 ---
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.5, 1.75)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.4
        ))
        # --- Mirror: use mirror_axes as-is from base (L-R axis 1 already excluded) ---
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            from batchgeneratorsv2.transforms.nnunet.random_binary_operator import \
                ApplyRandomBinaryOperatorTransform
            from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
                RemoveRandomConnectedComponentFromOneHotEncodingTransform
            from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import \
                MoveSegAsOneHotToDataTransform
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class flexict3d_fix_252_Trainer(flexict3d_252_Trainer):
    """T2c: 修复幽灵器官过分割. 基于诊断(exp4正常,T2b强增强学到明亮结构=前景快捷方式致幽灵右肾).
    修复三方面(用户建议):
    1. intensity增强恢复nnU-Net默认(去快捷方式) - 继承exp4的get_training_transforms(默认),不覆盖
    2. 完全关闭训练mirror(mirror_axes=()) - 最干净,排除任何镜像干扰
    3. oversample 0.33(默认,增加负样本) - 让模型多见纯背景含明亮结构
    4. 推理use_mirroring=False(TTA关). elastic继承exp4默认p_elastic=0(关).
    """
    oversample_foreground_percent = 0.33

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """完全关闭mirror: mirror_axes=(). 保留rotation和initial_patch_size(复用base逻辑)."""
        import numpy as np
        from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        do_dummy_2d_data_aug = False
        mirror_axes = ()
        initial_patch_size = get_patch_size(patch_size, rotation_for_DA, rotation_for_DA,
                                            rotation_for_DA, (0.85, 1.25))
        self.print_to_log_file(f"mirror_axes: {mirror_axes} (ALL MIRROR DISABLED for T2c fix)")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[T2c] ghost-organ fix: default intensity aug (no shortcut), mirror OFF, oversample 0.33 (more neg samples)")


class flexict3d_repro_252_Trainer(flexict3d_252_Trainer):
    """T2d: 完全复刻exp4配置(默认增强+mirror(0,2)+oversample0.33),独立重训作对照.
    验证exp4配置是否可复现(排除随机性). 若成功(EMA0.9)则T2c失败是关mirror致."""
    pass


class flexict3d_aug2_252_Trainer(flexict3d_252_Trainer):
    """T2e: 强增强v2(吸取compile/幽灵器官教训). 重设计:
    1. compile=0由启动命令保证(nnUNet_compile=0)
    2. 完全关闭mirror(mirror_axes=()) - 单侧器官不需镜像,L-R镜像会让左肾变右肾
    3. oversample 0.33(默认) - 保证负样本,学好明亮非目标=背景
    4. intensity增强温和加强(contrast0.7-1.3/gamma0.7-1.5) - 不极端,避免明亮结构快捷方式
    5. elastic关(p_elastic_deform=0) - 避免API风险
    6. 推理use_mirroring=False(TTA关)
    公平测试强增强(compile=0)能否突破exp4 0.7726.
    """
    oversample_foreground_percent = 0.33

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        import numpy as np
        from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        do_dummy_2d_data_aug = False
        mirror_axes = ()
        initial_patch_size = get_patch_size(patch_size, rotation_for_DA, rotation_for_DA,
                                            rotation_for_DA, (0.85, 1.25))
        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.print_to_log_file(f"mirror_axes: {mirror_axes} (ALL MIRROR DISABLED for single-side organ)")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None,
                                ignore_label=None):
        from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
        from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
        from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
        from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
        from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
        from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
        from batchgeneratorsv2.transforms.utils.random import RandomTransform
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
        from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
            p_elastic_deform=0, elastic_deform_scale=(0.05, 0.1), elastic_deform_magnitude=(0, 10),
            p_synchronize_def_scale_across_axes=0,
            p_rotation=0.2, rotation=rotation_for_DA,
            p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False))
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        transforms.append(RandomTransform(GaussianNoiseTransform(noise_variance=(0, 0.15), p_per_channel=1, synchronize_channels=True), apply_probability=0.15))
        transforms.append(RandomTransform(GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False, synchronize_axes=False, p_per_channel=0.5, benchmark=True), apply_probability=0.2))
        transforms.append(RandomTransform(MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1), apply_probability=0.2))
        transforms.append(RandomTransform(ContrastTransform(contrast_range=BGContrast((0.7, 1.3)), preserve_range=True, synchronize_channels=False, p_per_channel=1), apply_probability=0.2))
        transforms.append(RandomTransform(SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True, ignore_axes=ignore_axes, allowed_channels=None, p_per_channel=0.5), apply_probability=0.2))
        transforms.append(RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1, synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.1))
        transforms.append(RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0, synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.3))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]], channel_idx_in_seg=0, set_outside_to=0))
        transforms.append(RemoveLabelTansform(-1, 0))
        if is_cascaded:
            assert foreground_labels is not None
            from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
            from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
            from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
            transforms.append(MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True))
            transforms.append(RandomTransform(ApplyRandomBinaryOperatorTransform(channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1), apply_probability=0.4))
            transforms.append(RandomTransform(RemoveRandomConnectedComponentFromOneHotEncodingTransform(channel_idx=list(range(-len(foreground_labels), 0)), fill_with_other_class_p=0, dont_do_if_covers_more_than_x_percent=0.15, p_per_label=1), apply_probability=0.2))
        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(regions=list(regions) + [ignore_label] if ignore_label is not None else regions, channel_in_seg=0))
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file("[T2e] aug2: compile=0(cmd), mirror OFF, oversample 0.33, mild intensity aug, elastic OFF")
