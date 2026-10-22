# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (Apache 2.0)

# Modifications Copyright (c) 2026 IRCAD France

import torch.nn.functional as F

from .modeling.cnn import generate_model
from .modeling.mask_decoder3D import MaskDecoder3D
from .modeling.patch_attention import MemoryAttention, MemoryAttentionLayer
from .modeling.patch_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from .modeling.prompt_encoder3D import PromptEncoder3D
from .modeling.SLIP import Samusa
from .modeling.transformer import Attention
from .modeling.volume_encoder import FpnNeck, ImageEncoder


def build_slip(image_size_x_y=192, image_size_z=192, checkpoint=None, num_point_embeddings=2):
    return _build_slip(image_size_x_y=image_size_x_y,
                       image_size_z=image_size_z,
                       checkpoint=checkpoint,
                       num_point_embeddings=num_point_embeddings)


slip_model_registry3D = {
    "slip": build_slip,
}


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_slip(image_size_x_y=192, image_size_z=32, checkpoint=None, num_point_embeddings=2):
    prompt_embed_dim = 256
    embed_size = 8

    encoder = generate_model()

    neck = FpnNeck(
        d_model=prompt_embed_dim,
        backbone_channel_list=[512, 256, 128, 64],
        fpn_top_down_levels=[2, 3],  # output level 0 and 1 directly use the backbone features
        fpn_interp_model="nearest",
    )

    model = ImageEncoder(
        scalp=1,
        trunk=encoder,
        neck=neck,
    )
    SA = Attention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )
    CA = Attention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
    )

    memory_layer = MemoryAttentionLayer(
        dim_feedforward=2048,
        activation=F.relu,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=SA,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=CA,
    )
    sam = Samusa(
        image_encoder=model,
        prompt_encoder=PromptEncoder3D(embed_dim=prompt_embed_dim,
                                       image_embedding_size=(image_size_z // embed_size, image_size_x_y // embed_size,
                                                             image_size_x_y // embed_size),
                                       input_image_size=(image_size_z, image_size_x_y, image_size_x_y),
                                       mask_in_chans=16,
                                       num_point_embeddings=num_point_embeddings),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=1,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=prompt_embed_dim,
            use_high_res_features=True,
            pred_obj_scores=True,
        ),
        memory_attention=MemoryAttention(
            d_model=256,
            pos_enc_at_input=False,
            layer=memory_layer,
            num_layers=4,
        ),
        memory_encoder=MemoryEncoder(out_dim=64,
                                     mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
                                     fuser=Fuser(num_layers=2,
                                                 layer=CXBlock(
                                                     dim=256,
                                                     kernel_size=7,
                                                     padding=3,
                                                     layer_scale_init_value=0.000001,
                                                     use_dwconv=True,
                                                 ))),
    )

    return sam
