import numpy as np
import torch

from nninteractive_finetune.prompts import (
    NEGATIVE_POINT_CHANNEL,
    POSITIVE_POINT_CHANNEL,
    PREVIOUS_SEGMENTATION_CHANNEL,
    InteractivePromptSampler,
    _point_kernel,
    add_soft_point_,
    sample_correction_event,
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


def test_sequential_correction_updates_previous_mask():
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
    assert torch.all(interactions[:, POSITIVE_POINT_CHANNEL] >= old_positive * 0.9)


def test_interaction_budgets_are_variable_and_bounded():
    sampler = InteractivePromptSampler(seed=11)
    budgets = sampler.sample_interaction_budgets(
        batch_size=64,
        minimum=1,
        maximum=10,
        short_probability=0.7,
    )
    assert int(budgets.min()) >= 1
    assert int(budgets.max()) <= 10
    assert len(torch.unique(budgets)) > 1


def test_weighted_interaction_budgets_follow_configured_support():
    sampler = InteractivePromptSampler(seed=13)
    budgets = sampler.sample_interaction_budgets(
        batch_size=32,
        minimum=1,
        maximum=5,
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert torch.equal(budgets, torch.ones_like(budgets))


def test_existing_mask_training_populates_previous_segmentation_channel():
    target = torch.zeros((1, 24, 24, 24), dtype=torch.long)
    target[:, 5:19, 5:19, 5:19] = 1
    sampler = InteractivePromptSampler(
        radius=2,
        seed=17,
        initial_mask_probability=1.0,
    )
    interactions = sampler.new_interactions(
        target,
        allow_initial_mask=True,
        force_initial_mask=True,
    )
    previous = interactions[:, PREVIOUS_SEGMENTATION_CHANNEL]
    assert float(previous.sum()) > 0
    assert not torch.equal(previous, target.float())
    assert (
        interactions[:, POSITIVE_POINT_CHANNEL].max() == 1
        or interactions[:, NEGATIVE_POINT_CHANNEL].max() == 1
    )


def test_corrections_do_not_force_both_click_signs():
    target = torch.zeros((1, 16, 16, 16), dtype=torch.long)
    target[:, 4:12, 4:12, 4:12] = 1
    sampler = InteractivePromptSampler(radius=2, seed=23)

    from_empty = sampler.new_interactions(
        target, allow_initial_mask=False
    )
    assert from_empty[:, POSITIVE_POINT_CHANNEL].max() == 1
    assert from_empty[:, NEGATIVE_POINT_CHANNEL].max() == 0

    oversegmented = torch.ones_like(target)
    corrections = torch.zeros_like(from_empty)
    sampler.add_corrections(
        corrections, oversegmented, target, initial=True
    )
    assert corrections[:, POSITIVE_POINT_CHANNEL].max() == 0
    assert corrections[:, NEGATIVE_POINT_CHANNEL].max() == 1


def test_mixed_error_chooses_exactly_one_signed_event():
    target = np.zeros((20, 20, 20), dtype=bool)
    target[2:10, 2:10, 2:10] = True
    prediction = np.zeros_like(target)
    prediction[2:7, 2:7, 2:7] = True
    prediction[14:18, 14:18, 14:18] = True
    event = sample_correction_event(
        target & ~prediction,
        prediction & ~target,
        np.random.default_rng(31),
    )
    assert event is not None
    point, include = event
    assert (target & ~prediction)[point] if include else (prediction & ~target)[point]


def test_supplied_initial_mask_is_used_before_synthetic_fallback():
    target = torch.zeros((1, 24, 24, 24), dtype=torch.long)
    target[:, 5:19, 5:19, 5:19] = 1
    supplied = torch.zeros_like(target)
    supplied[:, 6:18, 6:18, 6:18] = 1
    sampler = InteractivePromptSampler(
        radius=2,
        seed=37,
        initial_mask_probability=1.0,
        provided_initial_mask_probability=1.0,
    )
    interactions = sampler.new_interactions(
        target,
        force_initial_mask=True,
        initial_prediction=supplied,
        initial_mask_available=torch.tensor([True]),
    )
    assert torch.equal(
        interactions[:, PREVIOUS_SEGMENTATION_CHANNEL], supplied.float()
    )


def test_supplied_initial_masks_keep_a_synthetic_fraction():
    target = torch.zeros((1, 24, 24, 24), dtype=torch.long)
    target[:, 5:19, 5:19, 5:19] = 1
    supplied = torch.zeros_like(target)
    supplied[:, 6:18, 6:18, 6:18] = 1
    sampler = InteractivePromptSampler(
        seed=41,
        initial_mask_probability=1.0,
        provided_initial_mask_probability=0.0,
    )
    interactions = sampler.new_interactions(
        target,
        force_initial_mask=True,
        initial_prediction=supplied,
        initial_mask_available=torch.tensor([True]),
    )
    previous = interactions[:, PREVIOUS_SEGMENTATION_CHANNEL]
    assert not torch.equal(previous, supplied.float())
    assert not torch.equal(previous, target.float())
