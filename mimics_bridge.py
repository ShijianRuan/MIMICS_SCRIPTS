# -*- coding: utf-8 -*-
"""External Python bridge for dataset <-> Mimics conversion.

Runs in the nninteractive_env Python (3.10+ with numpy, nibabel, pydicom).
Called by mimics_import.py / mimics_export.py inside Mimics via subprocess.

Protocol: JSON stdin -> JSON stdout.

Actions:
    "prepare" - NIfTI image -> derived DICOM, NIfTI masks -> .u8 buffers
    "convert" - .u8 buffers -> NIfTI files (inverse buffer mapping)
    "resample_image_to_grid" - resample an image to a supplied target grid
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
LPS_TO_RAS = RAS_TO_LPS
DEFAULT_MIMICS_BUFFER_AXES = [0, 1, 2]
DEFAULT_MIMICS_BUFFER_FLIPS = [False, False, False]


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return np.maximum(spacing.astype(float), 0.001)


def _unit_axis(affine: np.ndarray, axis: int, spacing: np.ndarray) -> np.ndarray:
    spacing_value = float(spacing[axis])
    if not np.isfinite(spacing_value) or spacing_value <= 0:
        raise ValueError(
            "invalid affine axis {}: non-positive spacing {}".format(axis, spacing_value)
        )
    vector = affine[:3, axis].astype(float) / spacing_value
    if not np.all(np.isfinite(vector)):
        raise ValueError("invalid affine axis {}: non-finite direction".format(axis))
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid affine axis {}: zero-length direction".format(axis))
    return vector / norm


# -- Image inspection ---------------------------------------------------

def is_dicom_folder(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    import pydicom
    count = 0
    for child in sorted(p.rglob("*")):
        if not child.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(child), stop_before_pixels=True, force=False)
            if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                count += 1
        except Exception:
            continue
    return count > 0


def infer_dicom_modality(path: str) -> str:
    p = Path(path)
    if not p.is_dir():
        return ""
    import pydicom
    for child in sorted(p.rglob("*")):
        if not child.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(child), stop_before_pixels=True, force=False)
        except Exception:
            continue
        modality = str(getattr(ds, "Modality", "") or "").strip()
        if modality:
            return modality
    return ""


def is_nifti_file(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


# -- NIfTI -> derived DICOM --------------------------------------------

def _sitk_is_axial(sitk_img, threshold: float = 0.001) -> bool:
    """Return True if the SimpleITK image has no Z-tilt in row/column cosines.

    A purely axial (or XY-plane-rotated) image has near-zero Z components in
    both the row direction (column 0 of the direction matrix) and the column
    direction (column 1).  Oblique acquisitions with gantry-tilt have non-zero
    Z components and must be resampled before DICOM export so that Mimics
    background import_dicom_images can group all slices into a single series.
    """
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    spacing = np.array(sitk_img.GetSpacing())
    row_cos = direction[:, 0] / max(np.linalg.norm(direction[:, 0]), 1e-12)
    col_cos = direction[:, 1] / max(np.linalg.norm(direction[:, 1]), 1e-12)
    return abs(float(row_cos[2])) < threshold and abs(float(col_cos[2])) < threshold


def _resample_to_axial(sitk_img):
    """Resample an oblique image to an axis-aligned LPS grid with the same spacing.

    The output grid covers the full bounding box of the original image so that
    no data is lost.  Linear interpolation is used; out-of-bounds voxels are
    filled with -1024 (air HU, safe for CT resampling).
    """
    import SimpleITK as sitk

    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    origin = np.array(sitk_img.GetOrigin())
    size = np.array(sitk_img.GetSize(), dtype=float)

    # 8 corners of the original image in world (LPS) coordinates
    corners = []
    for ix in [0.0, size[0] - 1]:
        for iy in [0.0, size[1] - 1]:
            for iz in [0.0, size[2] - 1]:
                idx = np.array([ix, iy, iz])
                corners.append(origin + direction @ (spacing * idx))
    corners = np.array(corners)

    new_origin = corners.min(axis=0)
    new_size = np.ceil((corners.max(axis=0) - new_origin) / spacing + 1).astype(int)

    ref = sitk.Image(int(new_size[0]), int(new_size[1]), int(new_size[2]),
                     sitk_img.GetPixelID())
    ref.SetOrigin(new_origin.tolist())
    ref.SetSpacing(spacing.tolist())
    ref.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    return resampler.Execute(sitk_img)


def nifti_to_derived_dicom(
    nifti_path: str,
    dicom_out: str,
    case_id: str = "case",
    source_nifti_out: str | None = None,
) -> dict:
    """Convert NIfTI image to derived DICOM series for Mimics import."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage

    sitk_img = _read_image_sitk_lps(nifti_path)

    # Resample oblique (3D-tilted) images to axis-aligned LPS so that Mimics
    # background import_dicom_images groups all slices into one series.
    resampled_to_axial = False
    if not _sitk_is_axial(sitk_img):
        sitk_img = _resample_to_axial(sitk_img)
        resampled_to_axial = True

    if source_nifti_out and resampled_to_axial:
        import SimpleITK as sitk

        out_path = Path(source_nifti_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk_img, str(out_path), useCompression=True)

    array = _sitk_to_xyz_array(sitk_img)
    if array.ndim != 3:
        raise ValueError("NIfTI image must be 3D: {} shape={}".format(nifti_path, array.shape))

    # SimpleITK image is already LPS-oriented; extract the LPS affine directly.
    affine_lps = _sitk_to_lps_affine(sitk_img)
    shape_3d = array.shape

    # Use affine column norms so spacing and direction cosines share exactly
    # the same source geometry, including oblique acquisitions.
    spacing = _voxel_spacing_from_affine(affine_lps)

    origin = affine_lps[:3, 3]
    direction_matrix = affine_lps[:3, :3] / spacing

    if array.dtype == np.uint8:
        pixel_array = array.astype(np.uint16)
        bits_allocated = 16
        bits_stored = 16
        pixel_representation = 0
    else:
        pixel_array = array.astype(np.int16)
        bits_allocated = 16
        bits_stored = 16
        pixel_representation = 1

    source_hash = "sha256:" + hashlib.sha256(str(Path(nifti_path).resolve()).encode("utf-8")).hexdigest()
    study_uid = generate_uid(entropy_srcs=[source_hash, "study"])
    series_uid = generate_uid(entropy_srcs=[source_hash, "series"])
    frame_of_reference_uid = generate_uid(entropy_srcs=[source_hash, "frame_of_reference"])

    out_dir = Path(dicom_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file():
            child.unlink()

    # -- DICOM dimension mapping --------------------------------------
    # NIfTI array shape: (i, j, k) where i=first axis, j=second axis, k=slices
    # DICOM pixel data layout: pixel[row, col] stored row-by-row
    #   Rows = number of rows (vertical) -> corresponds to NIfTI j axis
    #   Columns = number of columns (horizontal) -> corresponds to NIfTI i axis
    #
    # DICOM ImageOrientationPatient stores:
    #   first triplet  = direction of the first row, i.e. increasing column
    #                    index -> NIfTI i axis in this layout
    #   second triplet = direction of the first column, i.e. increasing row
    #                    index -> NIfTI j axis in this layout
    #
    # PixelSpacing: [row_spacing, col_spacing] = [spacing_j, spacing_i]
    #
    # Pixel data must be transposed from NIfTI (i,j) to DICOM (row=j, col=i)

    num_slices = shape_3d[2]
    dicom_rows = shape_3d[1]      # NIfTI j axis -> DICOM Rows
    dicom_columns = shape_3d[0]   # NIfTI i axis -> DICOM Columns

    row_cosine = _unit_axis(affine_lps, 0, spacing)
    column_cosine = _unit_axis(affine_lps, 1, spacing)

    for slice_idx in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid(
            entropy_srcs=[source_hash, "sop", str(slice_idx)]
        )

        dcm_path = str(out_dir / "slice_{:04d}.dcm".format(slice_idx + 1))
        ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "ANON"
        ds.PatientID = case_id
        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.StudyID = "1"
        ds.StudyDate = "19000101"
        ds.StudyTime = "000000"
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.SeriesDate = "19000101"
        ds.SeriesTime = "000000"
        ds.SeriesDescription = "Derived from NIfTI"
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.AcquisitionNumber = 1
        ds.Rows = dicom_rows
        ds.Columns = dicom_columns
        ds.BitsAllocated = bits_allocated
        ds.BitsStored = bits_stored
        ds.HighBit = bits_stored - 1
        ds.PixelRepresentation = pixel_representation
        # PixelSpacing = [row_spacing, col_spacing] = [spacing_j, spacing_i]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])
        ds.InstanceNumber = int(slice_idx + 1)
        # Compute slice position using the full LPS affine. This handles
        # oblique and non-standard orientations correctly.
        slice_origin = affine_lps @ np.array([0, 0, slice_idx, 1])
        ds.ImagePositionPatient = [
            float(slice_origin[0]),
            float(slice_origin[1]),
            float(slice_origin[2]),
        ]
        # SliceLocation: signed distance of the slice from the origin along
        # the normal direction (cross product of row/column cosines).
        normal = np.cross(row_cosine, column_cosine)
        ds.SliceLocation = float(np.dot(normal, slice_origin[:3]))
        ds.ImageOrientationPatient = [
            float(row_cosine[0]), float(row_cosine[1]), float(row_cosine[2]),
            float(column_cosine[0]), float(column_cosine[1]), float(column_cosine[2]),
        ]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0

        # Transpose slice data from NIfTI (i,j) to DICOM (row=j, col=i)
        slice_data_nifti = pixel_array[:, :, slice_idx]
        slice_data_dicom = slice_data_nifti.T  # (j, i) = (Rows, Columns)
        ds.PixelData = slice_data_dicom.tobytes()
        ds.save_as(dcm_path)

    _validate_derived_dicom_series(out_dir, num_slices, str(series_uid), str(study_uid))

    return {
        "shape": [int(dicom_columns), int(dicom_rows), int(num_slices)],
        "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "origin": [float(v) for v in origin],
        "direction": [float(v) for v in direction_matrix.flatten()],
        "affine_lps": affine_lps.tolist(),
        "resampled_to_axial": resampled_to_axial,
        "source_nifti_path": str(Path(source_nifti_out).resolve()) if (source_nifti_out and resampled_to_axial) else "",
        "series_uid": str(series_uid),
        "dicom_folder": str(out_dir),
    }


def prepare_source_fastpath_nifti(nifti_path: str, source_nifti_out: str) -> dict:
    """Prepare an on-demand source NIfTI aligned with Mimics-imported grid.

    This is a lightweight helper for nnInteractive fast path. It performs the
    same oblique->axial resampling rule as :func:`nifti_to_derived_dicom`, but
    does not emit DICOM slices.
    """
    import SimpleITK as sitk

    sitk_img = _read_image_sitk_lps(nifti_path)
    resampled_to_axial = False
    if not _sitk_is_axial(sitk_img):
        sitk_img = _resample_to_axial(sitk_img)
        resampled_to_axial = True

    out_path = Path(source_nifti_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(out_path), useCompression=True)

    array = _sitk_to_xyz_array(sitk_img)
    affine_lps = _sitk_to_lps_affine(sitk_img)
    shape_3d = array.shape
    spacing = _voxel_spacing_from_affine(affine_lps)
    return {
        "shape": [int(shape_3d[0]), int(shape_3d[1]), int(shape_3d[2])],
        "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "affine_lps": affine_lps.tolist(),
        "resampled_to_axial": resampled_to_axial,
        "source_nifti_path": str(out_path.resolve()),
    }


def _validate_derived_dicom_series(out_dir: Path, expected_slices: int, series_uid: str, study_uid: str) -> None:
    """Validate generated DICOM slices are complete and internally consistent.

    This catches truncated/empty or partially-written files that can make
    Mimics split one volume into many pseudo-series.
    """
    import pydicom

    files = sorted([p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"])
    if len(files) != int(expected_slices):
        raise RuntimeError(
            "derived DICOM slice count mismatch: expected={}, got={} in {}".format(
                int(expected_slices), len(files), str(out_dir)
            )
        )

    required_tags = (
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "InstanceNumber",
    )

    for path in files:
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        if size <= 0:
            raise RuntimeError("derived DICOM contains empty file: {}".format(str(path)))

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception as exc:
            raise RuntimeError("failed to read generated DICOM header {}: {}".format(str(path), exc))

        for tag in required_tags:
            if not hasattr(ds, tag):
                raise RuntimeError("generated DICOM missing required tag {}: {}".format(tag, str(path)))

        if str(ds.SeriesInstanceUID) != series_uid:
            raise RuntimeError(
                "generated DICOM series UID mismatch in {}: expected {}, got {}".format(
                    str(path), series_uid, str(ds.SeriesInstanceUID)
                )
            )
        if str(ds.StudyInstanceUID) != study_uid:
            raise RuntimeError(
                "generated DICOM study UID mismatch in {}: expected {}, got {}".format(
                    str(path), study_uid, str(ds.StudyInstanceUID)
                )
            )


# -- NIfTI header normalization ----------------------------------------

def _normalize_nifti_affine(nifti_img) -> np.ndarray:
    """Return a normalized affine, fixing ambiguous qform/sform codes.

    Some NIfTI files have qform_code=0 or sform_code=0, which causes
    tools like ITK-Snap, SimpleITK and 3D Slicer to interpret the
    spatial mapping differently.  Re-setting both codes forces a
    consistent sform_code=2 / qform_code=2 header so every downstream
    reader sees the same affine.

    Falls back to pixdim-derived affine when either qform or sform is
    corrupt (NaN, all-zero, or otherwise unable to be decomposed by
    nibabel).

    The file on disk is NOT modified; only the in-memory image object
    header is normalized.
    """
    import nibabel as nib

    def _safe_get_qform(img):
        try:
            return img.get_qform()
        except Exception:
            return None

    def _safe_get_sform(img):
        try:
            return img.get_sform()
        except Exception:
            return None

    def _usable_affine(mat):
        """True when affine has a non-trivial finite linear component."""
        if mat is None:
            return False
        return (
            np.all(np.isfinite(mat))
            and mat.shape == (4, 4)
            and bool(np.linalg.norm(mat[:3, :3]) > 1e-8)
        )

    def _try_set_qform(img, affine):
        try:
            img.set_qform(affine, code=2)
        except Exception:
            pass

    def _try_set_sform(img, affine):
        try:
            img.set_sform(affine, code=2)
        except Exception:
            pass

    qform = _safe_get_qform(nifti_img)
    sform = _safe_get_sform(nifti_img)

    if _usable_affine(qform):
        _try_set_qform(nifti_img, qform)
    if _usable_affine(sform):
        _try_set_sform(nifti_img, sform)
    # If neither was usable, force-set at least sform from pixdim so the
    # image has a valid spatial mapping.
    if not _usable_affine(qform) and not _usable_affine(sform):
        pixdim = nifti_img.header.get_zooms()[:3]
        fallback = np.eye(4)
        for i, p in enumerate(pixdim):
            pp = float(p)
            if np.isfinite(pp) and pp >= 0.001:
                fallback[i, i] = pp
            else:
                fallback[i, i] = 1.0
        _try_set_sform(nifti_img, fallback)
        _try_set_qform(nifti_img, fallback)
        return fallback

    result = nifti_img.affine.copy()
    if not np.all(np.isfinite(result)):
        # Defensive: pixdim fallback for any remaining NaN in the affine
        pixdim = nifti_img.header.get_zooms()[:3]
        result = np.eye(4)
        for i, p in enumerate(pixdim):
            pp = float(p)
            if np.isfinite(pp) and pp >= 0.001:
                result[i, i] = pp
            else:
                result[i, i] = 1.0

    return result


# -- SimpleITK-based image reading with header correction and LPS orient

def _correct_nifti_header(path: str) -> None:
    """Fix qform/sform codes in a NIfTI file in-place (disk modification).

    Re-sets both qform and sform with code=2 so that SimpleITK can read
    the file.  This addresses cases where the original header has
    ambiguous or corrupt qform/sform codes.
    """
    import nibabel as nib
    img = nib.load(path)
    qform = img.get_qform()
    if qform is not None:
        img.set_qform(qform, code=2)
    sform = img.get_sform()
    if sform is not None:
        img.set_sform(sform, code=2)
    nib.save(img, path)


def _read_image_sitk_lps(path: str):
    """Read an image via SimpleITK, correcting NIfTI header if needed, then orient to LPS.

    Tries ``sitk.ReadImage`` first.  If it fails and the file is NIfTI,
    calls :func:`_correct_nifti_header` to fix qform/sform in-place and
    retries.  Finally applies ``DICOMOrient("LPS")`` so the returned
    image is always in LPS orientation.

    Works with both NIfTI files and DICOM folders.
    """
    import SimpleITK as sitk

    is_nifti = is_nifti_file(path)

    try:
        sitk_img = sitk.ReadImage(path)
    except Exception:
        if not is_nifti:
            raise
        _correct_nifti_header(path)
        sitk_img = sitk.ReadImage(path)

    sitk_img = sitk.DICOMOrient(sitk_img, "LPS")
    return sitk_img


def _sitk_to_lps_affine(sitk_img) -> np.ndarray:
    """Extract a 4×4 LPS affine matrix from a SimpleITK image."""
    spacing = np.array(sitk_img.GetSpacing(), dtype=float)
    origin = np.array(sitk_img.GetOrigin(), dtype=float)
    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    return affine


def _sitk_to_xyz_array(sitk_img) -> np.ndarray:
    """Extract a numpy array in (x, y, z) axis order from a SimpleITK image.

    SimpleITK ``GetArrayFromImage`` returns data in (z, y, x) order;
    this helper transposes to (x, y, z) to match the existing NIfTI-based
    convention used throughout the bridge.
    """
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)
    return np.transpose(arr, (2, 1, 0))     # (x, y, z)


# -- NIfTI mask -> .u8 buffer -----------------------------------------

def read_nifti_mask(path: str) -> np.ndarray:
    sitk_img = _read_image_sitk_lps(path)
    array = _sitk_to_xyz_array(sitk_img)
    if array.ndim != 3:
        raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
    return (array != 0).astype(np.uint8)


def read_nifti_mask_with_affine(path: str) -> tuple[np.ndarray, np.ndarray]:
    sitk_img = _read_image_sitk_lps(path)
    array = _sitk_to_xyz_array(sitk_img)
    if array.ndim != 3:
        raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
    affine_lps = _sitk_to_lps_affine(sitk_img)
    affine_ras = LPS_TO_RAS @ affine_lps  # Convert to RAS for internal consistency
    return (array != 0).astype(np.uint8), affine_ras


def _affine_is_usable(affine: np.ndarray) -> bool:
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return False
    linear = affine[:3, :3]
    if abs(float(np.linalg.det(linear))) < 1e-8:
        return False
    spacing = np.linalg.norm(linear, axis=0)
    return bool(np.all(np.isfinite(spacing)) and np.all(spacing > 1e-8))


def _affine_close(left: np.ndarray, right: np.ndarray, atol: float = 1e-4) -> bool:
    return bool(np.allclose(left, right, atol=atol, rtol=0.0))


def resample_mask_to_image_grid(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    image_shape: tuple[int, int, int],
    image_affine: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor resample of a mask into image voxel index space."""
    if tuple(mask.shape) == tuple(image_shape) and _affine_close(mask_affine, image_affine):
        return np.ascontiguousarray(mask.astype(np.uint8))
    if tuple(mask.shape) == tuple(image_shape) and not _affine_is_usable(mask_affine):
        return np.ascontiguousarray(mask.astype(np.uint8))

    inv_mask_affine = np.linalg.inv(mask_affine)
    out = np.zeros(tuple(int(v) for v in image_shape), dtype=np.uint8)
    x_count, y_count, z_count = [int(v) for v in image_shape]
    yy_template = np.arange(y_count, dtype=float)
    xx_template = np.arange(x_count, dtype=float)

    # Work slice-by-slice to avoid allocating a full 4D coordinate grid for
    # large CT volumes.
    for z_index in range(z_count):
        xx, yy = np.meshgrid(xx_template, yy_template, indexing="ij")
        coords = np.vstack([
            xx.reshape(1, -1),
            yy.reshape(1, -1),
            np.full((1, x_count * y_count), float(z_index)),
            np.ones((1, x_count * y_count), dtype=float),
        ])
        src = inv_mask_affine @ (image_affine @ coords)
        src_idx = np.rint(src[:3]).astype(np.int64)
        valid = (
            (src_idx[0] >= 0) & (src_idx[0] < mask.shape[0]) &
            (src_idx[1] >= 0) & (src_idx[1] < mask.shape[1]) &
            (src_idx[2] >= 0) & (src_idx[2] < mask.shape[2])
        )
        if np.any(valid):
            values = np.zeros(x_count * y_count, dtype=np.uint8)
            values[valid] = mask[src_idx[0, valid], src_idx[1, valid], src_idx[2, valid]]
            out[:, :, z_index] = values.reshape((x_count, y_count))
    return np.ascontiguousarray(out)


def apply_buffer_mapping(array: np.ndarray, axes: list[int], flips: list[bool]) -> np.ndarray:
    transformed = np.transpose(array, axes)
    for axis, flip in enumerate(flips):
        if flip:
            transformed = np.flip(transformed, axis=axis)
    return np.ascontiguousarray(transformed)


# -- .u8 buffer -> NIfTI (inverse) -------------------------------------

def inverse_buffer_mapping(array: np.ndarray, axes: list[int], flips: list[bool]) -> np.ndarray:
    for axis, flip in enumerate(flips):
        if flip:
            array = np.flip(array, axis=axis)
    inv_axes = [0] * 3
    for i, a in enumerate(axes):
        inv_axes[a] = i
    array = np.transpose(array, inv_axes)
    return np.ascontiguousarray(array)


def write_mask_nifti(array: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    import nibabel as nib
    nii = nib.Nifti1Image(array.astype(np.uint8), affine)
    nib.save(nii, output_path)


def resample_image_to_grid(
    image_path: str,
    target_shape,
    target_voxel_to_ras_matrix,
    output_path: str,
    default_value: float = -1024.0,
) -> dict:
    """Linearly resample an image to a target RAS voxel grid.

    This is used by few-shot training materialization when labels come from a
    Mimics project that was imported on a resampled Mimics grid. In that case
    the exported labels are correct in Mimics space, so the image must be
    resampled to the label grid before training.
    """
    import SimpleITK as sitk

    target_shape = _shape_from_params(target_shape)
    target_affine_ras = _matrix_from_params(target_voxel_to_ras_matrix)
    source = _read_image_sitk_lps(image_path)

    target_affine_lps = RAS_TO_LPS @ target_affine_ras
    spacing = _voxel_spacing_from_affine(target_affine_lps)
    direction = target_affine_lps[:3, :3] / spacing
    if not np.all(np.isfinite(direction)):
        raise ValueError("target_voxel_to_ras_matrix has invalid direction columns")

    reference = sitk.Image(
        int(target_shape[0]),
        int(target_shape[1]),
        int(target_shape[2]),
        source.GetPixelID(),
    )
    reference.SetOrigin([float(value) for value in target_affine_lps[:3, 3]])
    reference.SetSpacing([float(value) for value in spacing])
    reference.SetDirection([float(value) for value in direction.reshape(-1)])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    resampled = resampler.Execute(source)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    sitk.WriteImage(resampled, str(out), useCompression=out.name.endswith(".gz"))
    return {
        "status": "ok",
        "image_path": str(Path(image_path).resolve()),
        "output_path": str(out.resolve()),
        "target_shape": [int(value) for value in target_shape],
        "target_voxel_to_ras_matrix": target_affine_ras.tolist(),
    }


# -- Affine from NIfTI or DICOM ----------------------------------------

def get_image_affine(image_path: str) -> np.ndarray:
    if not Path(image_path).is_file():
        raise ValueError("image file not found: {}".format(image_path))
    sitk_img = _read_image_sitk_lps(image_path)
    affine_lps = _sitk_to_lps_affine(sitk_img)
    return LPS_TO_RAS @ affine_lps  # Convert to RAS for internal consistency


def get_image_shape(image_path: str) -> tuple[int, int, int]:
    if not Path(image_path).is_file():
        raise ValueError("image file not found: {}".format(image_path))
    sitk_img = _read_image_sitk_lps(image_path)
    size = sitk_img.GetSize()  # (x, y, z)
    shape = tuple(int(value) for value in size[:3])
    if len(shape) != 3:
        raise ValueError("image must be 3D: {} shape={}".format(image_path, size))
    return shape


def get_nifti_disk_geometry(image_path: str) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return the on-disk NIfTI shape and RAS affine exactly as nibabel sees it."""
    if not Path(image_path).is_file():
        raise ValueError("image file not found: {}".format(image_path))
    import nibabel as nib
    img = nib.load(str(image_path))
    shape = tuple(int(value) for value in img.shape[:3])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("image must be 3D: {} shape={}".format(image_path, img.shape))
    return shape, np.asarray(img.affine, dtype=float)


def _shape_from_params(value) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in value)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("target_shape must contain three positive integers: {}".format(value))
    return shape


def _matrix_from_params(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("target_voxel_to_ras_matrix must be a finite 4x4 matrix")
    if not _affine_is_usable(matrix):
        raise ValueError("target_voxel_to_ras_matrix is not invertible")
    return matrix


def get_image_affine_from_dicom(dicom_folder: str) -> np.ndarray:
    import pydicom
    folder = Path(dicom_folder)
    headers = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except Exception:
            continue
        if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
            headers.append(ds)
    if not headers:
        raise ValueError("no readable DICOM slices in: {}".format(dicom_folder))

    def _image_position(ds):
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return np.array([float(v) for v in ds.ImagePositionPatient], dtype=float)
            except Exception:
                return None
        return None

    first = headers[0]
    pixel_spacing = [float(v) for v in first.PixelSpacing] if hasattr(first, "PixelSpacing") else [1.0, 1.0]
    slice_thickness = float(first.SliceThickness) if hasattr(first, "SliceThickness") else pixel_spacing[0]
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if hasattr(first, "ImageOrientationPatient"):
        iop = [float(v) for v in first.ImageOrientationPatient]

    # DICOM IOP first triplet is the direction of increasing column index;
    # the second triplet is the direction of increasing row index.
    row_cosine = np.array(iop[0:3], dtype=float)
    column_cosine = np.array(iop[3:6], dtype=float)
    row_cosine = row_cosine / np.linalg.norm(row_cosine)
    column_cosine = column_cosine / np.linalg.norm(column_cosine)
    slice_dir = np.cross(row_cosine, column_cosine)
    slice_dir = slice_dir / np.linalg.norm(slice_dir)

    positions = [(ds, _image_position(ds)) for ds in headers]
    positions = [(ds, pos) for ds, pos in positions if pos is not None]
    if positions:
        positions.sort(key=lambda item: float(np.dot(item[1], slice_dir)))
        origin_lps = positions[0][1]
        if len(positions) > 1:
            distances = [
                abs(float(np.dot(positions[i][1] - positions[i - 1][1], slice_dir)))
                for i in range(1, len(positions))
            ]
            distances = [value for value in distances if value > 1e-6]
            if distances:
                slice_thickness = float(np.median(distances))
    else:
        origin_lps = np.array([0.0, 0.0, 0.0], dtype=float)

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_cosine * float(pixel_spacing[1])
    affine_lps[:3, 1] = column_cosine * float(pixel_spacing[0])
    affine_lps[:3, 2] = slice_dir * float(slice_thickness)
    affine_lps[:3, 3] = origin_lps
    return LPS_TO_RAS @ affine_lps


def get_image_shape_from_dicom(dicom_folder: str) -> tuple[int, int, int]:
    import pydicom
    folder = Path(dicom_folder)
    headers = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except Exception:
            continue
        if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
            headers.append(ds)
    if not headers:
        raise ValueError("no readable DICOM slices in: {}".format(dicom_folder))
    first = headers[0]
    return (int(first.Columns), int(first.Rows), int(len(headers)))


def _nifti_candidates(case_dir: Path):
    preferred = ("ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii")
    files = []
    for child in case_dir.iterdir():
        if not child.is_file():
            continue
        name = child.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            files.append(child)
    preferred_rows = []
    other_rows = []
    for path in files:
        lower_name = path.name.lower()
        if lower_name in preferred:
            preferred_rows.append(path)
        else:
            other_rows.append(path)
    return sorted(preferred_rows, key=lambda p: preferred.index(p.name.lower())) + sorted(other_rows)


def find_image_geometry_in_case_dir(case_dir: str) -> dict | None:
    """Find source image geometry in a TS-like case directory.

    For NIfTI input this deliberately uses nibabel's on-disk RAS affine, because
    exported labels in case_dir/segmentations are expected to align with the
    source image file that external tools will open.
    """
    d = Path(case_dir)
    for candidate in _nifti_candidates(d):
        try:
            shape, affine = get_nifti_disk_geometry(str(candidate))
            return {
                "path": str(candidate),
                "kind": "nifti",
                "shape": shape,
                "affine": affine,
            }
        except Exception:
            continue
    dicom_dir = d / "dicom"
    if dicom_dir.is_dir():
        return {
            "path": str(dicom_dir),
            "kind": "dicom_folder",
            "shape": get_image_shape_from_dicom(str(dicom_dir)),
            "affine": get_image_affine_from_dicom(str(dicom_dir)),
        }
    return None


def find_affine_in_case_dir(case_dir: str) -> np.ndarray | None:
    """Find affine from image in a TS-like case directory."""
    geometry = find_image_geometry_in_case_dir(case_dir)
    if not geometry:
        return None
    return geometry["affine"]


# -- Actions ------------------------------------------------------------

def do_prepare(params: dict) -> dict:
    """Prepare a single case: NIfTI -> DICOM + masks -> .u8 buffers."""
    image_path = params["image_path"]
    masks = params.get("masks", [])
    dicom_out = params.get("dicom_out", "")
    buffers_out = params["buffers_out"]
    axes = params.get("axes", DEFAULT_MIMICS_BUFFER_AXES)
    flips = params.get("flips", DEFAULT_MIMICS_BUFFER_FLIPS)

    os.makedirs(buffers_out, exist_ok=True)

    # Determine image source
    if is_dicom_folder(image_path):
        dicom_folder = image_path
        # Internal resampling uses RAS affines. get_image_affine_from_dicom()
        # converts the DICOM LPS geometry to RAS, matching nibabel NIfTI masks.
        image_affine = get_image_affine_from_dicom(image_path)
        source_image_kind = "dicom_folder"
        source_image_index_space = "dicom_columns_rows_slices_sorted_by_position_v1"
        source_world_coordinate_system = "lps"
        mimics_world_coordinate_system = "lps"
        source_to_mimics_world_matrix = np.eye(4)
        source_image_modality = infer_dicom_modality(image_path)
        image_shape = None
    elif is_nifti_file(image_path):
        if not dicom_out:
            return {"status": "error", "error": "dicom_out required for NIfTI images"}
        case_id = params.get("case_id", "case")
        info = nifti_to_derived_dicom(
            image_path,
            dicom_out,
            case_id=case_id,
        )
        dicom_folder = info["dicom_folder"]
        # Use the actual DICOM grid affine for mask resampling and Mimics coordinate
        # metadata.  For oblique NIfTI files that were resampled to axis-aligned,
        # this differs from the original NIfTI affine.
        dicom_affine_lps = np.array(info["affine_lps"]).reshape(4, 4)
        image_affine = LPS_TO_RAS @ dicom_affine_lps
        source_nifti_affine = get_image_affine(image_path)  # original NIfTI affine (RAS)
        source_image_path_for_fastpath = str(Path(image_path).resolve())
        source_image_kind = "nifti"
        source_image_index_space = (
            "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1"
            if not info.get("resampled_to_axial")
            else "derived_dicom_axial_lps_resampled_from_nifti_v1"
        )
        source_world_coordinate_system = "ras"
        mimics_world_coordinate_system = "lps"
        source_to_mimics_world_matrix = RAS_TO_LPS
        source_image_modality = "CT"
        image_shape = tuple(int(v) for v in info["shape"])
    else:
        return {"status": "error", "error": "image path is neither DICOM folder nor NIfTI file: {}".format(image_path)}
    if image_shape is None:
        import pydicom
        headers = []
        for p in sorted(Path(dicom_folder).rglob("*")):
            if not p.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
            except Exception:
                continue
            if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                headers.append(ds)
        if not headers:
            return {"status": "error", "error": "no readable DICOM slices in: {}".format(dicom_folder)}
        image_shape = (int(headers[0].Columns), int(headers[0].Rows), int(len(headers)))

    # Both matrices are expressed in RAS world coordinates.
    # source_voxel_to_ras_matrix: source_image_path geometry.
    # mimics_voxel_to_ras_matrix: the grid that Mimics actually imports
    #   (may differ from source when oblique NIfTI is resampled to axial).
    if is_nifti_file(image_path) and info.get("resampled_to_axial"):
        source_voxel_to_ras_matrix = source_nifti_affine.astype(float)
        mimics_voxel_to_ras_matrix = image_affine.astype(float)  # DICOM axial affine
    else:
        source_voxel_to_ras_matrix = image_affine.astype(float)
        mimics_voxel_to_ras_matrix = image_affine.astype(float)
    mimics_to_source_index_matrix = (
        np.linalg.inv(source_voxel_to_ras_matrix) @ mimics_voxel_to_ras_matrix
    )

    # Convert masks -> .u8
    mask_results = []
    for m in masks:
        name = m["name"]
        mask_path = m["path"]
        if not Path(mask_path).is_file():
            return {"status": "error", "error": "mask file not found: {}".format(mask_path)}
        array, mask_affine = read_nifti_mask_with_affine(mask_path)
        image_grid_mask = resample_mask_to_image_grid(array, mask_affine, image_shape, image_affine)
        transformed = apply_buffer_mapping(image_grid_mask, axes, flips)
        u8_path = os.path.join(buffers_out, name + ".u8")
        with open(u8_path, "wb") as f:
            f.write(transformed.tobytes(order="C"))
        mask_results.append({
            "name": name,
            "mask_path": str(Path(mask_path).resolve()),
            "u8_path": u8_path,
            "mimics_shape": list(transformed.shape),
            "nifti_shape": list(array.shape),
            "image_shape": list(image_shape),
            "mask_affine_usable": bool(_affine_is_usable(mask_affine)),
            "mask_affine_matches_image": bool(_affine_close(mask_affine, image_affine)),
            "mask_voxel_to_ras_matrix": mask_affine.astype(float).tolist(),
            "buffer_axes": list(axes),
            "buffer_flips": list(flips),
        })

    # Build a content fingerprint for incremental rebuild detection
    image_ident = str(Path(image_path).resolve())
    mask_fingerprints = ["{0}:{1}".format(
        m["name"], m["u8_hash"] if "u8_hash" in m else hashlib.sha256(
            open(m["u8_path"], "rb").read()
        ).hexdigest()
    ) for m in mask_results]
    source_fingerprint = "sha256:" + hashlib.sha256(
        (image_ident + "|" + "|".join(sorted(mask_fingerprints))).encode("utf-8")
    ).hexdigest()

    return {
        "status": "ok",
        "case_id": params.get("case_id", ""),
        "dicom_folder": dicom_folder,
        "source_image_path": source_image_path_for_fastpath if is_nifti_file(image_path) else str(Path(image_path).resolve()),
        "source_image_kind": source_image_kind,
        "source_image_shape": list(image_shape),
        "source_image_index_space": source_image_index_space,
        "source_image_modality": source_image_modality,
        "source_world_coordinate_system": source_world_coordinate_system,
        "mimics_world_coordinate_system": mimics_world_coordinate_system,
        "source_to_mimics_world_matrix": source_to_mimics_world_matrix.astype(float).tolist(),
        "source_voxel_to_ras_matrix": source_voxel_to_ras_matrix.tolist(),
        "source_fingerprint": source_fingerprint,
        "mimics_voxel_to_ras_matrix": mimics_voxel_to_ras_matrix.tolist(),
        "mimics_to_source_index_matrix": mimics_to_source_index_matrix.astype(float).tolist(),
        "source_case_dir": params.get("case_dir", ""),
        "masks": mask_results,
    }


def _mask_buffer_for_target_grid(mask_path: str, target_shape, target_voxel_to_ras, axes, flips, output_path: str):
    mask, mask_affine = read_nifti_mask_with_affine(mask_path)
    target_shape = _shape_from_params(target_shape)
    target_voxel_to_ras = _matrix_from_params(target_voxel_to_ras)
    target_grid_mask = resample_mask_to_image_grid(
        mask,
        mask_affine,
        target_shape,
        target_voxel_to_ras,
    )
    transformed = apply_buffer_mapping(target_grid_mask, axes, flips)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "wb") as handle:
        handle.write(transformed.tobytes(order="C"))
    return {
        "output_path": str(output),
        "u8_path": str(output),
        "mimics_shape": [int(value) for value in transformed.shape],
        "image_shape": [int(value) for value in target_shape],
        "foreground_voxels": int(np.count_nonzero(target_grid_mask)),
        "mask_affine_usable": bool(_affine_is_usable(mask_affine)),
        "mask_affine_matches_target": bool(_affine_close(mask_affine, target_voxel_to_ras)),
        "mask_voxel_to_ras_matrix": mask_affine.astype(float).tolist(),
        "target_voxel_to_ras_matrix": target_voxel_to_ras.astype(float).tolist(),
        "buffer_axes": list(axes),
        "buffer_flips": list(flips),
    }


def do_prepare_masks_for_grid(params: dict) -> dict:
    """Resample source NIfTI masks into an actual Mimics voxel grid."""
    masks = params.get("masks") or []
    buffers_out = params["buffers_out"]
    target_shape = params["target_shape"]
    target_voxel_to_ras = params["target_voxel_to_ras_matrix"]
    axes = params.get("axes", DEFAULT_MIMICS_BUFFER_AXES)
    flips = params.get("flips", DEFAULT_MIMICS_BUFFER_FLIPS)
    target_matrix = _matrix_from_params(target_voxel_to_ras)
    os.makedirs(buffers_out, exist_ok=True)
    results = []
    for item in masks:
        name = item["name"]
        mask_path = item.get("mask_path") or item.get("path")
        if not mask_path or not Path(mask_path).is_file():
            return {"status": "error", "error": "mask file not found for {}: {}".format(name, mask_path)}
        output_path = os.path.join(buffers_out, name + ".u8")
        row = _mask_buffer_for_target_grid(
            mask_path,
            target_shape,
            target_voxel_to_ras,
            axes,
            flips,
            output_path,
        )
        row["name"] = name
        row["mask_path"] = str(Path(mask_path).resolve())
        results.append(row)
    source_matrix = None
    if params.get("source_voxel_to_ras_matrix"):
        try:
            source_matrix = _matrix_from_params(params.get("source_voxel_to_ras_matrix"))
        except Exception:
            source_matrix = None
    mimics_to_source = None
    if source_matrix is not None:
        mimics_to_source = (np.linalg.inv(source_matrix) @ target_matrix).astype(float).tolist()
    return {
        "status": "ok",
        "target_shape": [int(value) for value in _shape_from_params(target_shape)],
        "target_voxel_to_ras_matrix": target_matrix.tolist(),
        "mimics_to_source_index_matrix": mimics_to_source,
        "masks": results,
    }


def do_convert(params: dict) -> dict:
    """Convert .u8 buffers -> NIfTI files (inverse buffer mapping)."""
    buffers_dir = params["buffers_dir"]
    manifest_path = params["manifest_path"]
    case_dir = params["case_dir"]
    axes = params.get("axes", DEFAULT_MIMICS_BUFFER_AXES)
    flips = params.get("flips", DEFAULT_MIMICS_BUFFER_FLIPS)

    # Read manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mimics_shape = manifest.get("mimics_shape")
    if not mimics_shape:
        return {"status": "error", "error": "manifest has no mimics_shape"}

    # First recover the mask into the Mimics image grid. The mask buffer may use
    # a platform-specific buffer order, so inverse_buffer_mapping restores the
    # image voxel index space before any world-space resampling happens.
    mimics_affine_source = ""
    mimics_affine_value = (
        params.get("target_voxel_to_ras_matrix")
        or params.get("mimics_voxel_to_ras_matrix")
        or manifest.get("mimics_voxel_to_ras_matrix")
    )
    if mimics_affine_value:
        mimics_affine = _matrix_from_params(mimics_affine_value)
        if params.get("target_voxel_to_ras_matrix"):
            mimics_affine_source = "params.target_voxel_to_ras_matrix"
        elif params.get("mimics_voxel_to_ras_matrix"):
            mimics_affine_source = "params.mimics_voxel_to_ras_matrix"
        else:
            mimics_affine_source = "manifest.mimics_voxel_to_ras_matrix"
    else:
        mimics_affine = find_affine_in_case_dir(case_dir)
        mimics_affine_source = "case_image_fallback"
        if mimics_affine is None:
            return {"status": "error", "error": "no image found in case_dir for affine: {}".format(case_dir)}

    source_geometry = None
    try:
        source_geometry = find_image_geometry_in_case_dir(case_dir)
    except Exception:
        source_geometry = None
    export_space = str(params.get("export_space") or manifest.get("export_space") or "source_image").lower()
    if export_space in ("source", "source_image", "source_grid") and source_geometry:
        export_shape = tuple(int(value) for value in source_geometry["shape"])
        export_affine = np.asarray(source_geometry["affine"], dtype=float)
        export_affine_source = "source_image:{}".format(source_geometry.get("kind", "unknown"))
        export_space = "source_image"
    else:
        export_shape = None
        export_affine = mimics_affine
        export_affine_source = mimics_affine_source
        export_space = "mimics_grid"

    # Create segmentations dir
    seg_dir = os.path.join(case_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)

    # Convert each mask
    exported = []
    total_new = 0
    total_overwritten = 0
    total_unchanged = 0

    for mask_info in manifest["masks"]:
        name = mask_info["original_name"]
        u8_filename = mask_info.get("u8_filename") or (mask_info.get("safe_name", name) + ".u8")
        u8_path = os.path.join(buffers_dir, u8_filename)

        if not os.path.isfile(u8_path):
            exported.append({"name": name, "action": "skipped", "reason": "buffer not found"})
            continue

        nifti_path = os.path.join(seg_dir, name + ".nii.gz")

        # Read .u8, inverse map
        with open(u8_path, "rb") as f:
            raw = f.read()
        expected = 1
        for dim in mimics_shape:
            expected *= int(dim)
        if len(raw) != expected:
            exported.append({
                "name": name,
                "action": "error",
                "error": "buffer size mismatch: {} != {}".format(len(raw), expected),
            })
            continue

        array = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(mimics_shape))
        mimics_grid_array = inverse_buffer_mapping(array, axes, flips)
        if export_shape is not None:
            nifti_array = resample_mask_to_image_grid(
                mimics_grid_array,
                mimics_affine,
                export_shape,
                export_affine,
            )
        else:
            nifti_array = mimics_grid_array

        # If file already exists, compare content; skip write if unchanged.
        if os.path.isfile(nifti_path):
            import nibabel as nib
            existing_img = nib.load(nifti_path)
            existing = np.asanyarray(existing_img.dataobj)
            same_affine = _affine_close(existing_img.affine, export_affine)
            if existing.shape == nifti_array.shape and np.array_equal(existing, nifti_array) and same_affine:
                total_unchanged += 1
                exported.append({"name": name, "action": "unchanged", "path": nifti_path})
                continue
            action = "overwritten"
        else:
            action = "new"

        write_mask_nifti(nifti_array, export_affine, nifti_path)

        if action == "new":
            total_new += 1
        else:
            total_overwritten += 1
        exported.append({"name": name, "action": action, "path": nifti_path})

    return {
        "status": "ok",
        "exported": exported,
        "total_new": total_new,
        "total_overwritten": total_overwritten,
        "total_unchanged": total_unchanged,
        "export_space": export_space,
        "mimics_voxel_to_ras_matrix_source": mimics_affine_source,
        "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
        "export_voxel_to_ras_matrix_source": export_affine_source,
        "export_voxel_to_ras_matrix": export_affine.tolist(),
    }


def do_resample_image_to_grid(params: dict) -> dict:
    return resample_image_to_grid(
        params["image_path"],
        params["target_shape"],
        params["target_voxel_to_ras_matrix"],
        params["output_path"],
        float(params.get("default_value", -1024.0)),
    )


def do_mask_to_buffer(params: dict) -> dict:
    """Convert a NIfTI mask into a Mimics-shaped .u8 buffer for foreground apply."""
    image_path = params.get("image_path", "")
    mask_path = params["mask_path"]
    output_path = params["output_path"]
    axes = params.get("axes", DEFAULT_MIMICS_BUFFER_AXES)
    flips = params.get("flips", DEFAULT_MIMICS_BUFFER_FLIPS)

    if params.get("target_shape") and params.get("target_voxel_to_ras_matrix"):
        result = _mask_buffer_for_target_grid(
            mask_path,
            params["target_shape"],
            params["target_voxel_to_ras_matrix"],
            axes,
            flips,
            output_path,
        )
        result["status"] = "ok"
        return result

    if is_dicom_folder(image_path):
        # Internal resampling uses RAS affines. get_image_affine_from_dicom()
        # converts the DICOM LPS geometry to RAS, matching nibabel NIfTI masks.
        image_affine = get_image_affine_from_dicom(image_path)
        import pydicom
        headers = []
        for p in sorted(Path(image_path).rglob("*")):
            if not p.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
            except Exception:
                continue
            if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                headers.append(ds)
        if not headers:
            return {"status": "error", "error": "no readable DICOM slices in: {}".format(image_path)}
        image_shape = (int(headers[0].Columns), int(headers[0].Rows), int(len(headers)))
    elif is_nifti_file(image_path):
        image_affine = get_image_affine(image_path)
        image_shape = get_image_shape(image_path)
    else:
        return {"status": "error", "error": "image path is neither DICOM folder nor NIfTI file: {}".format(image_path)}

    mask, mask_affine = read_nifti_mask_with_affine(mask_path)
    image_grid_mask = resample_mask_to_image_grid(mask, mask_affine, image_shape, image_affine)
    transformed = apply_buffer_mapping(image_grid_mask, axes, flips)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "wb") as handle:
        handle.write(transformed.tobytes(order="C"))
    return {
        "status": "ok",
        "output_path": str(output),
        "mimics_shape": [int(value) for value in transformed.shape],
        "image_shape": [int(value) for value in image_shape],
        "foreground_voxels": int(np.count_nonzero(image_grid_mask)),
        "target_voxel_to_ras_matrix": image_affine.astype(float).tolist(),
        "buffer_axes": list(axes),
        "buffer_flips": list(flips),
    }


# -- Discover cases (for batch import) --------------------------------

def do_discover_case_dirs(params: dict) -> dict:
    """Discover candidate case directories without inspecting image files."""
    ts_root = params["ts_root"]
    cases_filter = params.get("cases_filter")

    if not os.path.isdir(ts_root):
        return {"status": "error", "error": "ts_root is not a directory: {}".format(ts_root)}

    cases = []
    for name in sorted(os.listdir(ts_root)):
        if name in ("mcs_output", "segmentations"):
            continue
        if cases_filter and name not in cases_filter:
            continue
        case_dir = os.path.join(ts_root, name)
        if os.path.isdir(case_dir):
            cases.append({"case_id": name, "case_dir": case_dir})
    return {"status": "ok", "cases": cases, "count": len(cases)}

def do_discover(params: dict) -> dict:
    """Discover TS-like cases in a dataset root directory.

    This runs in the bridge (nninteractive_env) so it doesn't block Mimics GUI.
    Returns list of case_info dicts suitable for do_prepare.
    """
    ts_root = params["ts_root"]
    cases_filter = params.get("cases_filter")

    if not os.path.isdir(ts_root):
        return {"status": "error", "error": "ts_root is not a directory: {}".format(ts_root)}

    cases = []
    for name in sorted(os.listdir(ts_root)):
        case_dir = os.path.join(ts_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name in ("mcs_output", "segmentations"):
            continue
        if cases_filter and name not in cases_filter:
            continue

        # Find image
        image_path = None
        image_type = None
        for img_name in ("ct.nii.gz", "mri.nii.gz"):
            candidate = os.path.join(case_dir, img_name)
            if os.path.isfile(candidate):
                image_path = candidate
                image_type = "nifti"
                break
        if image_path is None:
            dicom_dir = os.path.join(case_dir, "dicom")
            if os.path.isdir(dicom_dir):
                image_path = dicom_dir
                image_type = "dicom"
        if image_path is None:
            for fname in sorted(os.listdir(case_dir)):
                if fname.endswith(".nii.gz"):
                    image_path = os.path.join(case_dir, fname)
                    image_type = "nifti"
                    break

        if image_path is None:
            continue  # no image found, skip

        # Find masks
        masks = []
        seg_dir = os.path.join(case_dir, "segmentations")
        if os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                if fname.endswith(".nii.gz"):
                    organ = fname.replace(".nii.gz", "")
                    masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

        cases.append({
            "case_id": name,
            "image": image_path,
            "image_type": image_type,
            "masks": masks,
            "case_dir": case_dir,
        })

    return {"status": "ok", "cases": cases, "count": len(cases)}


# -- Main ---------------------------------------------------------------

def do_prepare_source_fastpath(params: dict) -> dict:
    """Build source NIfTI cache on-demand for nnInteractive fast path."""
    image_path = params.get("image_path", "")
    source_nifti_out = params.get("source_nifti_out", "")
    if not image_path or not is_nifti_file(image_path):
        return {"status": "error", "error": "image_path must be a NIfTI file"}
    if not source_nifti_out:
        return {"status": "error", "error": "source_nifti_out is required"}
    info = prepare_source_fastpath_nifti(image_path, source_nifti_out)
    return {
        "status": "ok",
        "image_path": str(Path(image_path).resolve()),
        **info,
    }


def main():
    try:
        params = json.load(sys.stdin)
    except Exception as e:
        json.dump({"status": "error", "error": "failed to read stdin JSON: {}".format(e)}, sys.stdout)
        return

    try:
        action = params.get("action", "")
        if action == "prepare":
            result = do_prepare(params)
        elif action == "convert":
            result = do_convert(params)
        elif action == "resample_image_to_grid":
            result = do_resample_image_to_grid(params)
        elif action == "mask_to_buffer":
            result = do_mask_to_buffer(params)
        elif action == "prepare_masks_for_grid":
            result = do_prepare_masks_for_grid(params)
        elif action == "prepare_source_fastpath":
            result = do_prepare_source_fastpath(params)
        elif action == "discover":
            result = do_discover(params)
        elif action == "discover_case_dirs":
            result = do_discover_case_dirs(params)
        else:
            result = {"status": "error", "error": "unknown action: {}".format(action)}
    except Exception as e:
        import traceback
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
