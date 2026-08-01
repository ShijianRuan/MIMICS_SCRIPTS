#!/usr/bin/env python3
"""Verify that an exported NIfTI Mask uses the original image grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimics_bridge import get_source_image_geometry  # noqa: E402


def _as_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) < 3 or any(int(value) <= 0 for value in image.shape[:3]):
        raise RuntimeError("Mask is not a valid 3D NIfTI image: {}".format(path))
    return np.asarray(image.dataobj) != 0, np.asarray(image.affine, dtype=float)


def _orientation(affine: np.ndarray) -> str:
    return "".join(str(value) for value in nib.aff2axcodes(affine))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Original image file or DICOM folder")
    parser.add_argument("--mask", required=True, help="Exported .nii or .nii.gz Mask")
    parser.add_argument("--reference-mask", help="Optional reference NIfTI for Dice")
    parser.add_argument("--affine-atol", type=float, default=1.0e-4)
    args = parser.parse_args()

    image_geometry = get_source_image_geometry(str(Path(args.image).expanduser()))
    if not image_geometry:
        raise RuntimeError("Original image geometry could not be read: {}".format(args.image))
    image_shape = tuple(int(value) for value in image_geometry["shape"][:3])
    image_affine = np.asarray(image_geometry["affine"], dtype=float)
    mask, mask_affine = _as_mask(Path(args.mask).expanduser())
    mask_shape = tuple(int(value) for value in mask.shape[:3])

    shape_ok = image_shape == mask_shape
    affine_ok = bool(
        np.allclose(image_affine, mask_affine, rtol=0.0, atol=float(args.affine_atol))
    )
    print("Original image shape: {}".format(image_shape))
    print("Exported Mask shape: {}".format(mask_shape))
    print("Original orientation: {}".format(_orientation(image_affine)))
    print("Mask orientation: {}".format(_orientation(mask_affine)))
    print("Affine max abs delta: {:.8g}".format(float(np.max(np.abs(image_affine - mask_affine)))))
    print("Foreground voxels: {}".format(int(np.count_nonzero(mask))))
    print("Grid match: {}".format("PASS" if shape_ok and affine_ok else "FAIL"))

    if args.reference_mask:
        reference, reference_affine = _as_mask(Path(args.reference_mask).expanduser())
        if reference.shape != mask.shape or not np.allclose(
            reference_affine, mask_affine, rtol=0.0, atol=float(args.affine_atol)
        ):
            raise RuntimeError("Reference Mask is not on the exported Mask grid.")
        intersection = int(np.count_nonzero(reference & mask))
        denominator = int(np.count_nonzero(reference)) + int(np.count_nonzero(mask))
        dice = 1.0 if denominator == 0 else (2.0 * intersection / float(denominator))
        print("Reference Dice: {:.6f}".format(dice))

    return 0 if shape_ok and affine_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
