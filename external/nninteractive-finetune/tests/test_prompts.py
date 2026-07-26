import numpy as np
import torch

from nninteractive_finetune.prompts import (
    NEGATIVE_POINT_CHANNEL,
    POSITIVE_POINT_CHANNEL,
    PREVIOUS_SEGMENTATION_CHANNEL,
    InteractivePromptSampler,
    _point_kernel,
    add_soft_point_,
)


def test_point_kernel_is_normalized_discrete_edt():
    kernel = _point_kernel(4)
    assert tuple(kernel.shape) == (9, 9, 9)
    assert float(kernel[4, 4, 4]) == 1.0
    assert float(kernel[0, 0, 0]) == 0.0


def test_point_clips_cleanly_at_volume_boundary():
    channel = torch.zeros((5, 5, 5))
    add_soft_point_(channel, (0, 0, 0), 4)
    assert float(channel[0, 0, 0]) == 1.0
    assert float(channel.max()) == 1.0


def test_sequential_correction_updates_previous_mask_and_both_click_signs():
    target = torch.zeros((1, 12, 12, 12), dtype=torch.long)
    target[:, 2:5, 2:5, 2:5] = 1
    sampler = InteractivePromptSampler(radius=2, center_bias=8, decay=0.9, seed=3)
    interactions = sampler.new_interactions(target)
    assert interactions[:, POSITIVE_POINT_CHANNEL].max() == 1
    prediction = torch.zeros_like(target)
    prediction[:, 2:4, 2:4, 2:4] = 1
    prediction[:, 8:10, 8:10, 8:10] = 1
    old_positive = interactions[:, POSITIVE_POINT_CHANNEL].clone()
    sampler.add_corrections(interactions, prediction, target)
    assert torch.equal(
        interactions[:, PREVIOUS_SEGMENTATION_CHANNEL], prediction.float()
    )
    assert interactions[:, NEGATIVE_POINT_CHANNEL].max() == 1
    assert torch.all(interactions[:, POSITIVE_POINT_CHANNEL] >= old_positive * 0.9)
