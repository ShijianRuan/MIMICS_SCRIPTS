"""Data protocol primitives for auditable multi-organ few-shot experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ..data.spatial import SPATIAL_CONVENTION, canonicalize


TASKS: dict[str, dict[str, object]] = {
    "brain": {"label": "brain", "family": "large_compact", "description": "Brain"},
    "liver": {"label": "liver", "family": "large_compact", "description": "Liver"},
    "adrenal_gland_right": {
        "label": "adrenal_gland_right",
        "family": "small_compact",
        "description": "Right adrenal gland",
    },
    "aorta": {"label": "aorta", "family": "elongated", "description": "Aorta"},
    "scapula_left": {"label": "scapula_left", "family": "thin_bone", "description": "Left scapula"},
}


def _sha256_payload(payload: Mapping) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@lru_cache(maxsize=None)
def _image_intensity_percentiles(image_path: str) -> tuple[float, float, float]:
    """Return image-only CT percentiles, decompressing each gzip image once.

    ``ArrayProxy`` stride indexing is a poor fit for ``.nii.gz``: nibabel must
    seek through gzip streams and can repeatedly decompress most of a volume.
    Materializing one native-dtype array is faster and predictable.  The cache
    stores only three scalar values, so the temporary image array is released
    after each distinct CT and reused across all organ tasks in one process.
    """
    image = nib.load(image_path)
    volume = np.asanyarray(image.dataobj)
    shape = volume.shape[:3]
    steps = tuple(max(1, int(np.ceil(float(length) / 128.0))) for length in shape)
    sample = volume[::steps[0], ::steps[1], ::steps[2]]
    finite = np.asarray(sample, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise RuntimeError("Image has no finite intensity samples")
    return tuple(float(value) for value in np.percentile(finite, [1, 50, 99]))


def _image_feature_vector(image: nib.spatialimages.SpatialImage, intensity_percentiles: Sequence[float]) -> list[float]:
    """Return image-only geometry and intensity features for representative support selection."""
    shape = [float(value) for value in image.shape[:3]]
    spacing = [float(value) for value in image.header.get_zooms()[:3]]
    return [*np.log1p(shape), *np.log(spacing), *[float(value) / 1000.0 for value in intensity_percentiles]]


def discover_records(source_root: Path, task_name: str) -> list[dict]:
    """Discover valid source cases for one task without changing the source data."""
    if task_name not in TASKS:
        raise KeyError("Unknown task {}. Choices: {}".format(task_name, ", ".join(sorted(TASKS))))
    label_name = str(TASKS[task_name]["label"])
    records = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("s")):
        image_path = case_dir / "ct.nii.gz"
        label_path = case_dir / "segmentations" / (label_name + ".nii.gz")
        if not image_path.is_file() or not label_path.is_file():
            continue
        try:
            original_image = nib.load(str(image_path))
            image = canonicalize(original_image)
            label = canonicalize(nib.load(str(label_path)))
            # Preserve the compact integer on-disk dtype during the scan. A
            # float32 ``get_fdata`` allocation can multiply the memory and I/O
            # cost for every compressed label in a large source directory.
            label_data = np.asanyarray(label.dataobj) > 0
            foreground = int(np.count_nonzero(label_data))
            if foreground == 0:
                continue
            spacing = tuple(float(value) for value in label.header.get_zooms()[:3])
            voxel_volume_mm3 = float(np.prod(spacing))
            intensity_percentiles = list(_image_intensity_percentiles(str(image_path.resolve())))
            record = {
                "case_id": case_dir.name,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "source_shape_xyz": [int(value) for value in image.shape[:3]],
                "source_spacing_xyz": [float(value) for value in image.header.get_zooms()[:3]],
                "source_orientation": [str(value) for value in nib.aff2axcodes(original_image.affine)],
                "canonical_orientation": [str(value) for value in nib.aff2axcodes(image.affine)],
                "image_intensity_percentiles": intensity_percentiles,
                "image_features": _image_feature_vector(image, intensity_percentiles),
                # The fields below are for post-selection reporting only. The
                # support selector intentionally never reads them.
                "label_foreground_voxels": foreground,
                "label_volume_ml": foreground * voxel_volume_mm3 / 1000.0,
            }
            records.append(record)
        except Exception as exc:
            raise RuntimeError("Could not read {} for {}: {}".format(case_dir.name, task_name, exc))
    if not records:
        raise RuntimeError("No non-empty {} labels under {}".format(task_name, source_root))
    return records


def select_diverse_support(records: Sequence[Mapping], count: int, seed: int) -> list[dict]:
    """Select representative supports from image-only metadata.

    The selection uses CT shape, spacing and sampled intensity quantiles, never the
    organ label. It is therefore valid when the rest of the pool is unlabelled.
    """
    if count < 1 or count > len(records):
        raise ValueError("support count must be between 1 and {}".format(len(records)))
    rows = [dict(record) for record in records]
    features = np.asarray([record["image_features"] for record in rows], dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    rng = random.Random(int(seed))
    order = list(range(len(rows)))
    rng.shuffle(order)
    # Start nearest the population centroid, resolving exact ties by seed.
    distances_to_center = np.linalg.norm(normalized, axis=1)
    first = min(order, key=lambda index: (float(distances_to_center[index]), order.index(index)))
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in order if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                float(min(np.linalg.norm(normalized[index] - normalized[picked]) for picked in selected)),
                -order.index(index),
            ),
        )
        selected.append(next_index)
    return [rows[index] for index in selected]


def materialize_case(record: Mapping, image_destination: Path, label_destination: Path) -> None:
    """Write a canonical image/label pair onto the label grid without mutation."""
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    image = canonicalize(nib.load(str(record["image_path"])))
    label = canonicalize(nib.load(str(record["label_path"])))
    # The source label is the reference grid. CT intensities must be resampled
    # from the CT image, never reconstructed from label voxels.
    image_on_label_grid = resample_from_to(image, (label.shape, label.affine), order=1)
    label_data = (label.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    output_label = nib.Nifti1Image(label_data, label.affine, header=label.header)
    nib.save(image_on_label_grid, str(image_destination))
    nib.save(output_label, str(label_destination))


def _link_or_copy(source: Path, destination: Path) -> None:
    """Reuse a materialized case while supporting filesystems without hard links."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def materialize_fold(
    records: Sequence[Mapping],
    task_name: str,
    destination: Path,
    *,
    support_count: int,
    seed: int,
    overwrite: bool = False,
    case_cache: Path | None = None,
) -> dict:
    """Build a reusable support/evaluation fold and its immutable manifest."""
    destination = destination.resolve()
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError("Refusing to overwrite non-empty dataset {}".format(destination))
    if destination.exists() and overwrite:
        for child in destination.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(str(child))
            else:
                child.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    supports = select_diverse_support(records, support_count, seed)
    support_ids = {row["case_id"] for row in supports}
    evaluation = [dict(row) for row in records if row["case_id"] not in support_ids]
    if not evaluation:
        raise RuntimeError("A fold needs at least one held-out evaluation case")
    for split, rows in (("Tr", supports), ("Val", evaluation)):
        for row in rows:
            image_destination = destination / "images{}".format(split) / (row["case_id"] + ".nii.gz")
            label_destination = destination / "labels{}".format(split) / (row["case_id"] + ".nii.gz")
            if case_cache is None:
                materialize_case(row, image_destination, label_destination)
                continue
            cache_image = case_cache / "images" / (row["case_id"] + ".nii.gz")
            cache_label = case_cache / "labels" / (row["case_id"] + ".nii.gz")
            if not cache_image.is_file() or not cache_label.is_file():
                materialize_case(row, cache_image, cache_label)
            _link_or_copy(cache_image, image_destination)
            _link_or_copy(cache_label, label_destination)
    manifest = {
        "schema_version": "dinov3_medical_fewshot_fold.v2",
        "spatial_convention": SPATIAL_CONVENTION,
        "task": task_name,
        "task_family": TASKS[task_name]["family"],
        "selection": {
            "method": "diverse_image_only_maxmin",
            "seed": int(seed),
            "support_pool_size": int(support_count),
            "support_case_ids_ordered": [row["case_id"] for row in supports],
            "evaluation_case_ids": [row["case_id"] for row in evaluation],
        },
        "records": {row["case_id"]: dict(row) for row in records},
    }
    manifest["fingerprint_sha256"] = _sha256_payload(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def training_fingerprint(manifest: Mapping, support_count: int) -> dict:
    """Return a train-only fingerprint for one nested K-shot training set."""
    ordered_ids = list(manifest["selection"]["support_case_ids_ordered"])
    if support_count < 1 or support_count > len(ordered_ids):
        raise ValueError("Requested K={} outside materialized support pool".format(support_count))
    records = manifest["records"]
    selected = [records[case_id] for case_id in ordered_ids[:support_count]]
    payload = {
        "spatial_convention": SPATIAL_CONVENTION,
        "task": manifest["task"],
        "support_case_ids": ordered_ids[:support_count],
        "spacing_xyz": [record["source_spacing_xyz"] for record in selected],
        "shape_xyz": [record["source_shape_xyz"] for record in selected],
        "image_geometry_features": [record["image_features"] for record in selected],
    }
    payload["fingerprint_sha256"] = _sha256_payload(payload)
    return payload
