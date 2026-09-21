"""Fast, conservative point proposals for DINO-guided nnInteractive.

The proposal stage is deliberately not another segmentation model. It keeps
the dominant DINO component and accepts at most two additional components only
when their physical size, distance, confidence, and TTA agreement support them.
It then selects spatially separated foreground anchors inside stable cores and
background anchors in a nearby physical shell. The annotator remains
responsible for accepting, moving, adding, or deleting the temporary Mimics
points before nnInteractive receives them.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)


def _as_volume(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or any(int(value) <= 0 for value in result.shape):
        raise ValueError("{} must be a non-empty 3D volume, got {}".format(name, result.shape))
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result


def _spacing_zyx_from_affine(affine) -> tuple[float, float, float]:
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("voxel_to_ras_affine must be a finite 4x4 matrix")
    spacing_xyz = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(spacing_xyz <= 0.0):
        raise ValueError("voxel_to_ras_affine contains a zero direction vector")
    return tuple(float(value) for value in spacing_xyz[::-1])


def _bbox_distance_mm(first, second, spacing_zyx) -> float:
    first_start = np.asarray([item.start for item in first], dtype=float)
    first_stop = np.asarray([item.stop for item in first], dtype=float)
    second_start = np.asarray([item.start for item in second], dtype=float)
    second_stop = np.asarray([item.stop for item in second], dtype=float)
    separation = np.maximum(
        np.maximum(first_start - second_stop, second_start - first_stop),
        0.0,
    )
    return float(np.linalg.norm(separation * np.asarray(spacing_zyx, dtype=float)))


def _select_components(
    mask: np.ndarray,
    mean: np.ndarray,
    agreement: np.ndarray,
    spacing_zyx,
    *,
    threshold: float,
    maximum: int,
    minimum_relative_size: float,
):
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=CONNECTIVITY_26)
    if count < 1:
        return np.zeros(mask.shape, dtype=np.int8), [], [], 0
    sizes = np.bincount(labels.ravel())
    ranked = sorted(
        range(1, count + 1),
        key=lambda label: (-int(sizes[label]), int(label)),
    )
    objects = ndimage.find_objects(labels)
    largest_label = ranked[0]
    largest_size = max(1, int(sizes[largest_label]))
    largest_object = objects[largest_label - 1]
    largest_extent = np.asarray(
        [item.stop - item.start for item in largest_object], dtype=float
    ) * np.asarray(spacing_zyx, dtype=float)
    largest_diagonal_mm = max(float(np.linalg.norm(largest_extent)), 1.0)

    decisions = []
    accepted = []
    largest_confidence = 0.0
    for source_rank, label in enumerate(ranked, start=1):
        object_slices = objects[label - 1]
        component = labels[object_slices] == label
        probabilities = mean[object_slices][component]
        agreements = agreement[object_slices][component]
        confidence_p75 = float(np.percentile(probabilities, 75.0))
        agreement_mean = float(np.mean(agreements))
        if source_rank == 1:
            largest_confidence = confidence_p75
        relative_size = float(sizes[label]) / float(largest_size)
        distance_mm = (
            0.0
            if source_rank == 1
            else _bbox_distance_mm(object_slices, largest_object, spacing_zyx)
        )
        # Similar-sized paired structures may legitimately be farther apart;
        # small satellite predictions must stay proportionally closer.
        distance_limit_mm = min(
            180.0,
            max(
                24.0,
                largest_diagonal_mm * (0.75 + 2.0 * np.sqrt(relative_size)),
            ),
        )
        confidence_floor = max(
            float(threshold) + 0.04,
            largest_confidence - 0.18,
        )
        reasons = []
        if source_rank > 1 and relative_size < float(minimum_relative_size):
            reasons.append("too_small_relative_to_main")
        if source_rank > 1 and distance_mm > distance_limit_mm:
            reasons.append("too_far_from_main")
        if source_rank > 1 and confidence_p75 < confidence_floor:
            reasons.append("low_relative_confidence")
        if source_rank > 1 and agreement_mean < 0.75:
            reasons.append("low_tta_agreement")
        accepted_component = source_rank == 1 or not reasons
        if accepted_component and len(accepted) >= max(1, int(maximum)):
            accepted_component = False
            reasons = ["component_limit"]
        row = {
            "source_rank": int(source_rank),
            "voxel_count": int(sizes[label]),
            "relative_size": float(relative_size),
            "bbox_distance_mm": float(distance_mm),
            "distance_limit_mm": float(distance_limit_mm),
            "confidence_p75": float(confidence_p75),
            "agreement_mean": float(agreement_mean),
            "accepted": bool(accepted_component),
            "reason": "accepted" if accepted_component else ",".join(reasons),
            "_label": int(label),
        }
        decisions.append(row)
        if accepted_component:
            accepted.append(row)

    remapped = np.zeros(mask.shape, dtype=np.int8)
    rows = []
    for rank, decision in enumerate(accepted, start=1):
        label = int(decision["_label"])
        remapped[labels == label] = rank
        row = dict(decision)
        row.pop("_label", None)
        row["rank"] = int(rank)
        rows.append(row)
    public_decisions = []
    for decision in decisions:
        row = dict(decision)
        row.pop("_label", None)
        public_decisions.append(row)
    return remapped, rows, public_decisions, int(count)


def _bounded_coordinates(mask: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(values))
    limit = max(1, int(limit))
    if count <= limit:
        return np.argwhere(values)

    # Select evenly spaced foreground ranks without first materializing every
    # 3D coordinate. Peak temporary storage is one 2D slice plus ``limit`` rows.
    ranks = np.unique(np.linspace(0, count - 1, limit, dtype=np.int64))
    rows = []
    consumed = 0
    rank_start = 0
    for z in range(values.shape[0]):
        flat_yx = np.flatnonzero(values[z])
        next_consumed = consumed + int(len(flat_yx))
        rank_stop = int(np.searchsorted(ranks, next_consumed, side="left"))
        if rank_stop > rank_start:
            local_ranks = ranks[rank_start:rank_stop] - consumed
            selected = flat_yx[local_ranks]
            y, x = np.unravel_index(selected, values.shape[1:])
            rows.append(
                np.column_stack(
                    [
                        np.full(len(selected), z, dtype=np.int64),
                        y.astype(np.int64, copy=False),
                        x.astype(np.int64, copy=False),
                    ]
                )
            )
            rank_start = rank_stop
        consumed = next_consumed
        if rank_start >= len(ranks):
            break
    if not rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.concatenate(rows, axis=0)


def _world_ras_from_zyx(coordinates_zyx: np.ndarray, affine) -> np.ndarray:
    coordinates = np.asarray(coordinates_zyx, dtype=float)
    xyz = coordinates[:, ::-1]
    homogeneous = np.concatenate(
        [xyz, np.ones((len(xyz), 1), dtype=float)], axis=1
    )
    return homogeneous.dot(np.asarray(affine, dtype=float).T)[:, :3]


def _candidate_rows_for_foreground(
    component_labels: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    agreement: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    threshold: float,
    maximum_candidates_per_component: int,
):
    rows = []
    component_anchors = []
    objects = ndimage.find_objects(component_labels)
    for component_rank, object_slices in enumerate(objects, start=1):
        if object_slices is None:
            continue
        component = component_labels[object_slices] == component_rank
        component_probability = mean[object_slices]
        component_agreement = agreement[object_slices]
        component_variance = variance[object_slices]
        values = component_probability[component]
        high_threshold = max(
            float(threshold) + 0.04,
            min(0.90, float(np.percentile(values, 70.0))),
        )
        variance_limit = min(
            0.02,
            max(0.0025, float(np.percentile(component_variance[component], 75.0))),
        )
        stable = (
            component
            & (component_probability >= high_threshold)
            & (component_agreement >= 0.75)
            & (component_variance <= variance_limit)
        )
        # Do not fabricate a foreground prompt from a weak component. The
        # review workflow can still let the user add a point manually.
        if not np.any(stable):
            continue

        padded = np.pad(component, 1, mode="constant", constant_values=False)
        distance = ndimage.distance_transform_edt(
            padded, sampling=spacing_zyx
        )[1:-1, 1:-1, 1:-1].astype(np.float32, copy=False)
        maximum_depth = max(float(distance[component].max()), 1e-6)
        reliability = (
            component_probability
            * (0.5 + 0.5 * component_agreement)
            * np.exp(-8.0 * component_variance)
        )
        quality = reliability * (
            0.35 + 0.65 * np.sqrt(np.clip(distance / maximum_depth, 0.0, 1.0))
        )
        local_coordinates = _bounded_coordinates(
            stable, maximum_candidates_per_component
        )
        if not len(local_coordinates):
            continue
        offsets = np.asarray([item.start for item in object_slices], dtype=np.int64)
        global_coordinates = local_coordinates + offsets
        local_scores = quality[tuple(local_coordinates.T)]
        component_rows = []
        for coordinate, score in zip(global_coordinates, local_scores):
            row = {
                "voxel_zyx": tuple(int(value) for value in coordinate),
                "score": float(score),
                "component_rank": int(component_rank),
            }
            rows.append(row)
            component_rows.append(row)
        component_anchors.append(max(component_rows, key=lambda item: item["score"]))
    return rows, component_anchors


def _candidate_rows_for_background(
    component_labels: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    agreement: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    threshold: float,
    inner_shell_mm: float,
    outer_shell_mm: float,
    maximum_candidates_per_component: int,
):
    rows = []
    component_anchors = []
    shape = np.asarray(mean.shape, dtype=int)
    padding_voxels = np.ceil(
        float(outer_shell_mm) / np.asarray(spacing_zyx, dtype=float)
    ).astype(int) + 1
    for component_rank in range(1, int(component_labels.max()) + 1):
        coordinates = np.argwhere(component_labels == component_rank)
        if not len(coordinates):
            continue
        start = np.maximum(coordinates.min(axis=0) - padding_voxels, 0)
        stop = np.minimum(coordinates.max(axis=0) + padding_voxels + 1, shape)
        slices = tuple(slice(int(left), int(right)) for left, right in zip(start, stop))
        component = component_labels[slices] == component_rank
        outside_distance = ndimage.distance_transform_edt(
            ~component, sampling=spacing_zyx
        ).astype(np.float32, copy=False)
        shell = (
            (outside_distance >= float(inner_shell_mm))
            & (outside_distance <= float(outer_shell_mm))
            & ~component
            & (component_labels[slices] == 0)
        )
        local_mean = mean[slices]
        local_variance = variance[slices]
        local_agreement = agreement[slices]
        background_probability_limit = max(
            0.05, min(0.35, float(threshold) - 0.12)
        )
        stable_background = (
            shell
            & (local_mean <= background_probability_limit)
            & (local_agreement <= 0.25)
            & (local_variance <= 0.02)
        )
        if not np.any(stable_background):
            continue
        shell_midpoint = 0.5 * (float(inner_shell_mm) + float(outer_shell_mm))
        shell_width = max(1.0, 0.5 * (float(outer_shell_mm) - float(inner_shell_mm)))
        proximity = np.exp(
            -0.5 * ((outside_distance - shell_midpoint) / shell_width) ** 2
        )
        quality = (
            (1.0 - local_mean)
            * (1.0 - 0.75 * local_agreement)
            * np.exp(-8.0 * local_variance)
            * proximity
        )
        local_coordinates = _bounded_coordinates(
            stable_background, maximum_candidates_per_component
        )
        if not len(local_coordinates):
            continue
        global_coordinates = local_coordinates + start
        local_scores = quality[tuple(local_coordinates.T)]
        component_rows = []
        for coordinate, score in zip(global_coordinates, local_scores):
            row = {
                "voxel_zyx": tuple(int(value) for value in coordinate),
                "score": float(score),
                "component_rank": int(component_rank),
            }
            rows.append(row)
            component_rows.append(row)
        component_anchors.append(max(component_rows, key=lambda item: item["score"]))
    return rows, component_anchors


def _select_spread_points(
    candidates,
    component_anchors,
    affine,
    *,
    maximum_points: int,
    minimum_spacing_mm: float,
):
    if not candidates:
        return []
    coordinates = np.asarray(
        [item["voxel_zyx"] for item in candidates], dtype=np.int64
    )
    world = _world_ras_from_zyx(coordinates, affine)
    scores = np.asarray([max(float(item["score"]), 1e-8) for item in candidates])
    maximum_score = max(float(scores.max()), 1e-8)
    scores = scores / maximum_score
    # Background shells of nearby components can overlap, so the same voxel may
    # appear as a candidate from more than one component with different scores.
    # Keep the highest-scoring copy so an anchor maps back to its best row
    # instead of an arbitrary last-write-wins duplicate.
    coordinate_to_index = {}
    for index, coordinate in enumerate(coordinates):
        key = tuple(int(value) for value in coordinate)
        existing = coordinate_to_index.get(key)
        if existing is None or scores[index] > scores[existing]:
            coordinate_to_index[key] = index
    selected = []
    selected_indexes = set()
    for anchor in sorted(component_anchors, key=lambda item: item["component_rank"]):
        index = coordinate_to_index.get(tuple(anchor["voxel_zyx"]))
        if index is None or index in selected_indexes:
            continue
        if selected and float(
            np.min(np.linalg.norm(world[index] - world[selected], axis=1))
        ) < float(minimum_spacing_mm):
            continue
        selected.append(index)
        selected_indexes.add(index)
        if len(selected) >= int(maximum_points):
            break

    if not selected:
        selected = [int(np.argmax(scores))]
        selected_indexes.add(selected[0])
    while len(selected) < int(maximum_points):
        distances = np.min(
            np.linalg.norm(world[:, None, :] - world[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[list(selected_indexes)] = -1.0
        available_distances = distances[distances >= 0.0]
        if not len(available_distances):
            break
        spread_scale = max(float(np.percentile(available_distances, 90.0)), 1e-6)
        objective = scores * (0.25 + 0.75 * np.clip(distances / spread_scale, 0.0, 1.0))
        objective[distances < float(minimum_spacing_mm)] = -1.0
        index = int(np.argmax(objective))
        if objective[index] < 0.0:
            break
        selected.append(index)
        selected_indexes.add(index)

    result = []
    for index in selected:
        item = dict(candidates[index])
        item["world_ras_mm"] = [float(value) for value in world[index]]
        item["score"] = float(scores[index])
        result.append(item)
    return result


def propose_guided_points(
    foreground_mean,
    foreground_variance,
    foreground_agreement,
    voxel_to_ras_affine,
    *,
    threshold: float = 0.5,
    maximum_components: int = 3,
    maximum_foreground_points: int = 3,
    maximum_background_points: int = 3,
    minimum_point_spacing_mm: float = 30.0,
    inner_background_shell_mm: float = 2.0,
    outer_background_shell_mm: float = 12.0,
    maximum_candidates_per_component: int = 20000,
    minimum_component_relative_size: float = 0.03,
):
    """Propose at most three dispersed foreground and background points."""
    mean = _as_volume(foreground_mean, "foreground_mean")
    variance = _as_volume(foreground_variance, "foreground_variance")
    agreement = _as_volume(foreground_agreement, "foreground_agreement")
    if variance.shape != mean.shape or agreement.shape != mean.shape:
        raise ValueError("TTA probability statistic shapes do not match")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between zero and one")
    if not 0.0 < float(minimum_component_relative_size) <= 1.0:
        raise ValueError("minimum_component_relative_size must be in (0, 1]")
    if float(inner_background_shell_mm) < 0.0 or (
        float(outer_background_shell_mm) <= float(inner_background_shell_mm)
    ):
        raise ValueError("background shell radii are invalid")
    affine = np.asarray(voxel_to_ras_affine, dtype=float)
    spacing_zyx = _spacing_zyx_from_affine(affine)
    component_labels, component_rows, component_decisions, original_component_count = _select_components(
        mean >= float(threshold),
        mean,
        agreement,
        spacing_zyx,
        threshold=float(threshold),
        maximum=maximum_components,
        minimum_relative_size=float(minimum_component_relative_size),
    )
    if not component_rows:
        raise RuntimeError(
            "DINOv3 produced no foreground region at probability threshold {:.3f}".format(
                float(threshold)
            )
        )

    foreground_candidates, foreground_anchors = _candidate_rows_for_foreground(
        component_labels,
        mean,
        variance,
        agreement,
        spacing_zyx,
        threshold=float(threshold),
        maximum_candidates_per_component=int(maximum_candidates_per_component),
    )
    foreground = _select_spread_points(
        foreground_candidates,
        foreground_anchors,
        affine,
        maximum_points=max(1, min(3, int(maximum_foreground_points))),
        minimum_spacing_mm=max(float(minimum_point_spacing_mm), 2.0 * min(spacing_zyx)),
    )
    if not foreground:
        raise RuntimeError("DINOv3 foreground regions contained no usable prompt point")

    background_candidates, background_anchors = _candidate_rows_for_background(
        component_labels,
        mean,
        variance,
        agreement,
        spacing_zyx,
        threshold=float(threshold),
        inner_shell_mm=float(inner_background_shell_mm),
        outer_shell_mm=float(outer_background_shell_mm),
        maximum_candidates_per_component=int(maximum_candidates_per_component),
    )
    background = _select_spread_points(
        background_candidates,
        background_anchors,
        affine,
        maximum_points=max(1, min(3, int(maximum_background_points))),
        minimum_spacing_mm=max(float(minimum_point_spacing_mm), 2.0 * min(spacing_zyx)),
    )

    points = []
    for include, selected in ((True, foreground), (False, background)):
        for index, row in enumerate(selected, start=1):
            points.append(
                {
                    "name": "FG {}".format(index) if include else "BG {}".format(index),
                    "include_interaction": include,
                    "voxel_zyx": list(row["voxel_zyx"]),
                    "world_ras_mm": list(row["world_ras_mm"]),
                    "score": float(row["score"]),
                    "component_rank": int(row["component_rank"]),
                }
            )
    return {
        "schema_version": "dinov3_guided_points.v1",
        "points": points,
        "foreground_point_count": len(foreground),
        "background_point_count": len(background),
        "retained_component_count": len(component_rows),
        "retained_components": component_rows,
        "component_decisions": component_decisions,
        "discarded_component_count": max(
            0, original_component_count - len(component_rows)
        ),
        "threshold": float(threshold),
        "spacing_zyx_mm": list(spacing_zyx),
        "selection": {
            "maximum_components": min(3, int(maximum_components)),
            "minimum_component_relative_size": float(
                minimum_component_relative_size
            ),
            "maximum_foreground_points": min(3, int(maximum_foreground_points)),
            "maximum_background_points": min(3, int(maximum_background_points)),
            "minimum_point_spacing_mm": float(minimum_point_spacing_mm),
            "background_shell_mm": [
                float(inner_background_shell_mm),
                float(outer_background_shell_mm),
            ],
        },
    }
