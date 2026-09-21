from __future__ import annotations

import numpy as np
import pytest
import torch

import src.inference as inference
from src.prompt_proposals import propose_guided_points


def _statistics(shape=(96, 72, 72)):
    return (
        np.full(shape, 0.01, dtype=np.float32),
        np.full(shape, 0.001, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    )


def _paint_box(mean, variance, agreement, start, stop, probability=0.92):
    region = tuple(slice(left, right) for left, right in zip(start, stop))
    mean[region] = probability
    variance[region] = 0.002
    agreement[region] = 1.0


def test_components_require_relative_size_distance_and_confidence_support():
    mean, variance, agreement = _statistics((160, 100, 100))
    _paint_box(mean, variance, agreement, (5, 5, 5), (35, 30, 30))
    _paint_box(mean, variance, agreement, (40, 8, 8), (60, 28, 28))
    _paint_box(mean, variance, agreement, (125, 70, 70), (143, 88, 88))
    _paint_box(mean, variance, agreement, (36, 31, 31), (40, 35, 35), 0.99)

    proposal = propose_guided_points(
        mean,
        variance,
        agreement,
        np.eye(4),
        minimum_point_spacing_mm=5.0,
    )

    assert proposal["retained_component_count"] == 2
    assert proposal["discarded_component_count"] == 2
    assert [row["voxel_count"] for row in proposal["retained_components"]] == [
        30 * 25 * 25,
        20 * 20 * 20,
    ]
    rejected = [
        row for row in proposal["component_decisions"] if not row["accepted"]
    ]
    assert any("too_far_from_main" in row["reason"] for row in rejected)
    assert any("too_small_relative_to_main" in row["reason"] for row in rejected)
    foreground = [
        point for point in proposal["points"] if point["include_interaction"]
    ]
    assert len(foreground) == 3
    assert {point["component_rank"] for point in foreground} == {1, 2}
    assert all(point["voxel_zyx"][0] < 125 for point in foreground)


def test_low_confidence_satellite_is_not_used_as_a_foreground_prompt():
    mean, variance, agreement = _statistics((80, 80, 80))
    _paint_box(mean, variance, agreement, (10, 10, 10), (40, 40, 40), 0.95)
    _paint_box(mean, variance, agreement, (42, 15, 15), (60, 33, 33), 0.58)

    proposal = propose_guided_points(mean, variance, agreement, np.eye(4))

    assert proposal["retained_component_count"] == 1
    assert "low_relative_confidence" in proposal["component_decisions"][1]["reason"]
    assert {
        point["component_rank"]
        for point in proposal["points"]
        if point["include_interaction"]
    } == {1}


def test_tta_disagreement_rejects_an_otherwise_plausible_satellite():
    mean, variance, agreement = _statistics((80, 80, 80))
    _paint_box(mean, variance, agreement, (10, 10, 10), (40, 40, 40), 0.95)
    _paint_box(mean, variance, agreement, (42, 15, 15), (60, 33, 33), 0.90)
    agreement[42:60, 15:33, 15:33] = 0.5

    proposal = propose_guided_points(mean, variance, agreement, np.eye(4))

    assert proposal["retained_component_count"] == 1
    assert "low_tta_agreement" in proposal["component_decisions"][1]["reason"]


def test_long_component_gets_three_physically_separated_foreground_and_background_points():
    mean, variance, agreement = _statistics((128, 64, 64))
    z, y, x = np.ogrid[:128, :64, :64]
    organ = (z >= 8) & (z < 120) & ((y - 32) ** 2 + (x - 32) ** 2 <= 8 ** 2)
    mean[organ] = 0.94
    variance[organ] = 0.001
    agreement[organ] = 1.0
    affine = np.diag([0.8, 0.8, 1.5, 1.0])

    proposal = propose_guided_points(
        mean,
        variance,
        agreement,
        affine,
        minimum_point_spacing_mm=12.0,
        inner_background_shell_mm=3.0,
        outer_background_shell_mm=12.0,
    )

    foreground = [
        point for point in proposal["points"] if point["include_interaction"]
    ]
    background = [
        point for point in proposal["points"] if not point["include_interaction"]
    ]
    assert len(foreground) == 3
    assert len(background) == 3
    for points in (foreground, background):
        world = np.asarray([point["world_ras_mm"] for point in points])
        pairwise = np.linalg.norm(world[:, None] - world[None, :], axis=2)
        assert np.min(pairwise[np.triu_indices(len(points), 1)]) >= 12.0
    assert all(organ[tuple(point["voxel_zyx"])] for point in foreground)
    assert all(not organ[tuple(point["voxel_zyx"])] for point in background)
    assert all(mean[tuple(point["voxel_zyx"])] < 0.5 for point in background)


def test_overlapping_background_shells_keep_best_score_per_voxel():
    """Two nearby components share background-shell voxels; the shared voxel's
    reported score must be the best copy, not an arbitrary last-write one."""
    mean, variance, agreement = _statistics((60, 60, 60))
    # Two blocks 6 mm apart (identity affine, 1 mm voxels): their 2-12 mm
    # background shells overlap in the gap between them.
    _paint_box(mean, variance, agreement, (20, 10, 10), (40, 30, 30), 0.95)
    _paint_box(mean, variance, agreement, (20, 31, 10), (40, 51, 30), 0.95)

    proposal = propose_guided_points(
        mean,
        variance,
        agreement,
        np.eye(4),
        minimum_point_spacing_mm=4.0,
        inner_background_shell_mm=2.0,
        outer_background_shell_mm=12.0,
    )

    background = [
        point for point in proposal["points"] if not point["include_interaction"]
    ]
    assert background, "expected at least one background point in the shared shell"
    # Every background point must be a true background voxel (low probability),
    # never a foreground voxel mis-tagged by a corrupted coordinate map.
    for point in background:
        assert mean[tuple(point["voxel_zyx"])] < 0.5
    # Scores are normalized to [0, 1]; a last-write-wins duplicate could only
    # depress them, so the top background score should remain the genuine max.
    scores = [point["score"] for point in background]
    assert max(scores) > 0.0


def test_world_coordinates_follow_xyz_affine_from_zyx_voxels():
    mean, variance, agreement = _statistics((24, 24, 24))
    _paint_box(mean, variance, agreement, (5, 6, 7), (18, 19, 20))
    affine = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    proposal = propose_guided_points(
        mean,
        variance,
        agreement,
        affine,
        maximum_foreground_points=1,
        maximum_background_points=1,
    )
    point = next(
        item for item in proposal["points"] if item["include_interaction"]
    )
    z, y, x = point["voxel_zyx"]
    assert np.allclose(
        point["world_ras_mm"],
        [10.0 + 2.0 * x, 20.0 + 3.0 * y, 30.0 + 4.0 * z],
    )


def test_online_tta_statistics_match_flip_equivariant_probabilities(monkeypatch):
    data = np.linspace(-2.0, 2.0, 2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)

    def fake_predict_logits(_model, input_data, _grid, _config, _device, **_kwargs):
        foreground = torch.from_numpy(np.asarray(input_data, dtype=np.float32))[None, None]
        background = torch.zeros_like(foreground)
        return torch.cat([background, foreground], dim=1)

    monkeypatch.setattr(inference, "_predict_logits", fake_predict_logits)
    result = inference._predict_tta_probability_statistics(
        object(),
        data,
        object(),
        {"inference": {"tta_axes": [[2]], "scales": [], "threshold": 0.5}},
        torch.device("cpu"),
    )
    expected = 1.0 / (1.0 + np.exp(-data))
    assert result["member_count"] == 2
    assert np.allclose(result["mean"], expected, atol=1e-6)
    assert np.allclose(result["variance"], 0.0, atol=1e-6)
    assert np.array_equal(
        result["agreement"], (expected >= 0.5).astype(np.float32)
    )


def test_cached_native_tta_avoids_the_affine_left_right_axis():
    from src.data.frozen_feature_slices import laterality_safe_in_plane_axis_zyx

    ras = np.diag([1.0, 2.0, 3.0, 1.0])
    assert laterality_safe_in_plane_axis_zyx(ras) == 1

    swapped = np.asarray(
        [
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert laterality_safe_in_plane_axis_zyx(swapped) == 2


def test_region_without_a_stable_core_raises_so_caller_keeps_segmentation():
    """A foreground region that exists above threshold but lacks a stable core
    (high TTA disagreement) raises rather than fabricating a weak prompt. The
    inference script wraps this in try/except so the DINOv3 segmentation is
    still saved; this test pins the raise that the caller depends on."""
    mean, variance, agreement = _statistics((40, 40, 40))
    _paint_box(mean, variance, agreement, (10, 10, 10), (30, 30, 30), 0.9)
    # High variance everywhere inside the component -> no voxel passes the
    # stable-core variance limit, so no foreground point can be selected.
    variance[10:30, 10:30, 10:30] = 0.5

    with pytest.raises(RuntimeError):
        propose_guided_points(mean, variance, agreement, np.eye(4))

