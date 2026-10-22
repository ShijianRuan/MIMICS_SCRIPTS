# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (Apache 2.0)

# Modifications Copyright (c) 2026 IRCAD France

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .mask_decoder3D import MaskDecoder3D
from .patch_attention import MemoryAttention
from .patch_encoder import MemoryEncoder
from .prompt_encoder3D import PromptEncoder3D
from .volume_encoder import ImageEncoder


class Samusa(nn.Module):

    def __init__(
        self,
        image_encoder: ImageEncoder,
        prompt_encoder: PromptEncoder3D,
        mask_decoder: MaskDecoder3D,
        memory_encoder: MemoryEncoder,
        memory_attention: MemoryAttention,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.maskmem_enc = nn.Parameter(torch.zeros(6, 1, 64))
        trunc_normal_(self.maskmem_enc, std=0.02)

        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, 256))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, 64))
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, 64))
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self.memory_encoder = memory_encoder
        self.memory_attention = memory_attention
        self.mask_decoder = mask_decoder
