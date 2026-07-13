"""Shared NIfTI geometry helpers for training, evaluation, and inference.

The model operates on canonical RAS volumes in ``(Z, Y, X)`` tensor order.
NIfTI itself stores the canonical array as ``(X, Y, Z)``.  Keeping the
conversion and restoration in one module prevents a training/inference axis
mismatch from silently producing an anatomically displaced mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nibabel.processing import resample_from_to, resample_to_output


SPATIAL_CONVENTION = "canonical_ras_array_xyz__model_tensor_zyx"


@dataclass(frozen=True)
class CanonicalVolume:
    """A canonical NIfTI image plus the model-facing tensor metadata."""

    image: nib.Nifti1Image
    data_zyx: np.ndarray
    spacing_zyx: Tuple[float, float, float]


def canonicalize(image: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Return an image in closest RAS orientation without changing its grid."""
    return nib.as_closest_canonical(image)


def xyz_to_zyx(array_xyz: np.ndarray) -> np.ndarray:
    """Convert a canonical NIfTI array from ``(X, Y, Z)`` to ``(Z, Y, X)``."""
    if array_xyz.ndim != 3:
        raise ValueError("Expected a 3D NIfTI array, got shape {}".format(array_xyz.shape))
    return np.ascontiguousarray(np.transpose(array_xyz, (2, 1, 0)))


def zyx_to_xyz(array_zyx: np.ndarray) -> np.ndarray:
    """Convert a model tensor array from ``(Z, Y, X)`` to NIfTI ``(X, Y, Z)``."""
    if array_zyx.ndim != 3:
        raise ValueError("Expected a 3D model array, got shape {}".format(array_zyx.shape))
    return np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)))


def spacing_zyx(image: nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    """Return canonical NIfTI voxel spacing in model ``(Z, Y, X)`` order."""
    zooms_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
    return zooms_xyz[2], zooms_xyz[1], zooms_xyz[0]


def load_canonical_volume(path: str) -> CanonicalVolume:
    """Load a NIfTI file and expose it in the project-wide tensor convention."""
    image = canonicalize(nib.load(path))
    data_xyz = image.get_fdata(dtype=np.float32)
    return CanonicalVolume(
        image=image,
        data_zyx=xyz_to_zyx(data_xyz),
        spacing_zyx=spacing_zyx(image),
    )


def load_canonical_pair(image_path: str, label_path: str, *, affine_atol: float = 1e-4):
    """Load a paired image/label after enforcing one canonical physical grid."""
    image = canonicalize(nib.load(image_path))
    label = canonicalize(nib.load(label_path))
    if image.shape[:3] != label.shape[:3] or not np.allclose(
        image.affine, label.affine, atol=affine_atol, rtol=0.0
    ):
        raise RuntimeError(
            "Image/label grid mismatch after canonical RAS orientation: {} vs {}".format(
                image_path, label_path
            )
        )
    image_data = xyz_to_zyx(image.get_fdata(dtype=np.float32))
    label_data = xyz_to_zyx(label.get_fdata(dtype=np.float32))
    return image, label, image_data, label_data, spacing_zyx(image)


def resample_pair_to_spacing(
    image: nib.spatialimages.SpatialImage,
    label: nib.spatialimages.SpatialImage,
    target_spacing_xyz: Iterable[float] | None,
):
    """Resample a pair to an image-defined grid, using nearest labels.

    This deliberately resamples the label *onto the generated image grid*
    instead of independently deriving two grids.  The resulting image and
    label therefore always have exactly matching affine and shape.
    """
    if target_spacing_xyz is None:
        return image, label
    spacing = tuple(float(value) for value in target_spacing_xyz)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("target_spacing must be three positive XYZ values, got {}".format(spacing))
    resampled_image = resample_to_output(image, voxel_sizes=spacing, order=1)
    resampled_label = resample_from_to(
        label,
        (resampled_image.shape, resampled_image.affine),
        order=0,
    )
    return resampled_image, resampled_label


def resample_image_to_spacing(
    image: nib.spatialimages.SpatialImage,
    target_spacing_xyz: Iterable[float] | None,
):
    """Resample one canonical image for inference with linear intensities."""
    if target_spacing_xyz is None:
        return image
    spacing = tuple(float(value) for value in target_spacing_xyz)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("target_spacing must be three positive XYZ values, got {}".format(spacing))
    return resample_to_output(image, voxel_sizes=spacing, order=1)


def restore_prediction_to_original(
    prediction_zyx: np.ndarray,
    *,
    model_grid: nib.spatialimages.SpatialImage,
    original_image: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    """Resample a model-grid prediction and restore the original orientation.

    ``model_grid`` must be canonical RAS. ``original_image`` is the input NIfTI
    before canonicalization.  The returned NIfTI has the original array shape,
    affine, and orientation, with nearest-neighbour label interpolation.
    """
    canonical_prediction = nib.Nifti1Image(
        zyx_to_xyz(np.asarray(prediction_zyx, dtype=np.int16)),
        model_grid.affine,
        header=model_grid.header,
    )
    canonical_original = canonicalize(original_image)
    if (
        canonical_prediction.shape[:3] != canonical_original.shape[:3]
        or not np.allclose(canonical_prediction.affine, canonical_original.affine, atol=1e-4, rtol=0.0)
    ):
        canonical_prediction = resample_from_to(
            canonical_prediction,
            (canonical_original.shape, canonical_original.affine),
            order=0,
        )

    canonical_data = canonical_prediction.get_fdata(dtype=np.float32).astype(np.int16)
    source_ornt = io_orientation(canonical_original.affine)
    destination_ornt = io_orientation(original_image.affine)
    to_original = ornt_transform(source_ornt, destination_ornt)
    original_data = apply_orientation(canonical_data, to_original).astype(np.int16, copy=False)
    if original_data.shape != original_image.shape[:3]:
        raise RuntimeError(
            "Orientation restoration changed prediction shape from {} to {}; expected {}".format(
                canonical_data.shape, original_data.shape, original_image.shape[:3]
            )
        )
    return nib.Nifti1Image(original_data, original_image.affine, header=original_image.header)
