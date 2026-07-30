import numpy as np

import pytest

from nninteractive_finetune.evaluate import (
    _blend_trajectories,
    _largest_correction_event,
    evaluate_model,
)


def test_evaluation_chooses_one_largest_signed_error_component():
    false_negative = np.zeros((20, 20, 20), dtype=bool)
    false_positive = np.zeros_like(false_negative)
    false_negative[2:10, 2:10, 2:10] = True
    false_positive[14:17, 14:17, 14:17] = True
    point, include = _largest_correction_event(
        false_negative, false_positive
    )
    assert include is True
    assert false_negative[point]


def test_evaluation_uses_background_for_pure_oversegmentation():
    false_negative = np.zeros((12, 12, 12), dtype=bool)
    false_positive = np.zeros_like(false_negative)
    false_positive[4:9, 4:9, 4:9] = True
    point, include = _largest_correction_event(
        false_negative, false_positive
    )
    assert include is False
    assert false_positive[point]


def test_evaluation_blends_real_and_synthetic_initial_mask_trajectories():
    combined = _blend_trajectories(
        [0.8, 0.9],
        [0.4, 0.6],
        0.7,
    )
    assert combined == pytest.approx([0.68, 0.81])


def test_evaluation_blend_extends_a_shorter_trajectory_with_last_value():
    combined = _blend_trajectories([0.8], [0.4, 0.6], 0.5)
    assert combined == pytest.approx([0.6, 0.7])


def test_evaluation_rejects_invalid_initial_mask_mix_before_model_load(
    tmp_path,
):
    with pytest.raises(
        ValueError, match="provided_initial_mask_probability"
    ):
        evaluate_model(
            tmp_path / "model",
            tmp_path / "manifest.json",
            tmp_path / "evaluation.json",
            [1],
            provided_initial_mask_probability=1.2,
        )
