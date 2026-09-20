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
import tempfile
import time
from pathlib import Path

import numpy as np


RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
LPS_TO_RAS = RAS_TO_LPS
DEFAULT_MIMICS_BUFFER_AXES = [0, 1, 2]
DEFAULT_MIMICS_BUFFER_FLIPS = [False, False, False]
DEFAULT_DICOM_RESAMPLE_MODE = "auto"
DEFAULT_MASK_RESAMPLE_METHOD = "nearest"


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


def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if (not np.isfinite(norm)) or norm <= eps:
        raise ValueError("cannot normalize degenerate vector")
    return arr / norm


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


def is_medical_image_file(path: str) -> bool:
    """Return whether SimpleITK can represent this supported 3D image file."""
    p = Path(path)
    if not p.is_file():
        return False
    name = p.name.lower()
    return name.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nrrd.gz"))


def get_medical_image_disk_geometry(path: str) -> tuple[tuple[int, int, int], np.ndarray]:
    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    if image.GetDimension() != 3:
        raise ValueError("medical image must be 3D: {}".format(path))
    shape = tuple(int(value) for value in image.GetSize())
    affine_ras = LPS_TO_RAS @ _sitk_to_lps_affine(image)
    return shape, affine_ras


# -- NIfTI -> derived DICOM --------------------------------------------

def _sitk_is_axial(sitk_img, threshold: float = 0.001) -> bool:
    """Return True if the SimpleITK image has no Z-tilt in row/column cosines."""
    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    row_cos = _normalize_vector(direction[:, 0])
    col_cos = _normalize_vector(direction[:, 1])
    return abs(float(row_cos[2])) < threshold and abs(float(col_cos[2])) < threshold


def _sitk_is_classic_dicom_geometry_compatible(sitk_img, ortho_tol: float = 5e-4) -> bool:
    """Whether the grid can be represented by a parallel classic-DICOM stack.

    Classic single-frame DICOM uses row/column direction cosines + per-slice
    position. The in-plane row and column axes must be orthogonal. The slice
    step does not have to be normal to that plane: a constant in-plane offset
    between slices (for example gantry tilt) is represented exactly by each
    slice's ImagePositionPatient.
    """
    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    try:
        row = _normalize_vector(direction[:, 0])
        col = _normalize_vector(direction[:, 1])
        slc = _normalize_vector(direction[:, 2])
    except ValueError:
        return False
    if not np.all(np.isfinite([row, col, slc])):
        return False
    cross = np.cross(row, col)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-8:
        return False
    cross /= cross_norm
    return (
        abs(float(np.dot(row, col))) <= ortho_tol
        and abs(float(np.dot(cross, slc))) > ortho_tol
    )


def _orthonormalize_direction(direction: np.ndarray) -> np.ndarray:
    """Build a right-handed orthonormal basis close to the input direction."""
    row = _normalize_vector(direction[:, 0])
    col_raw = np.asarray(direction[:, 1], dtype=float)
    col_ortho = col_raw - row * float(np.dot(col_raw, row))
    if float(np.linalg.norm(col_ortho)) <= 1e-10:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(row, fallback))) > 0.95:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        col_ortho = fallback - row * float(np.dot(fallback, row))
    col = _normalize_vector(col_ortho)
    normal = _normalize_vector(np.cross(row, col))
    source_k = _normalize_vector(direction[:, 2])
    if float(np.dot(normal, source_k)) < 0.0:
        normal = -normal
    col = _normalize_vector(np.cross(normal, row))
    return np.column_stack([row, col, normal])


def _resample_to_oriented_grid(
    sitk_img, target_direction: np.ndarray, default_value: float | None = None
):
    """Resample image to a supplied orthonormal LPS direction, preserving FOV."""
    import SimpleITK as sitk

    spacing = np.array(sitk_img.GetSpacing(), dtype=float)
    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.array(sitk_img.GetOrigin(), dtype=float)
    size = np.array(sitk_img.GetSize(), dtype=float)
    target_direction = np.asarray(target_direction, dtype=float).reshape(3, 3)

    if not np.all(np.isfinite(target_direction)):
        raise ValueError("target_direction contains non-finite values")
    if default_value is None:
        values = sitk.GetArrayViewFromImage(sitk_img)
        finite = np.asarray(values)[np.isfinite(values)]
        default_value = float(np.min(finite)) if finite.size else 0.0

    corners = []
    for ix in [0.0, size[0] - 1]:
        for iy in [0.0, size[1] - 1]:
            for iz in [0.0, size[2] - 1]:
                idx = np.array([ix, iy, iz])
                corners.append(origin + direction @ (spacing * idx))
    corners = np.array(corners)

    projected = corners @ target_direction
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    new_origin = target_direction @ mins
    new_size = np.ceil((maxs - mins) / spacing + 1.0).astype(int)
    new_size = np.maximum(new_size, 1)

    ref = sitk.Image(int(new_size[0]), int(new_size[1]), int(new_size[2]),
                     sitk_img.GetPixelID())
    ref.SetOrigin([float(v) for v in new_origin.tolist()])
    ref.SetSpacing(spacing.tolist())
    ref.SetDirection([float(v) for v in target_direction.reshape(-1)])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(sitk_img)


def _dicom_resample_policy():
    mode = str(os.environ.get("MIMICS_DICOM_RESAMPLE_MODE", DEFAULT_DICOM_RESAMPLE_MODE) or "").strip().lower()
    if mode not in ("auto", "axial", "never"):
        mode = "auto"
    return mode


def _prepare_dicom_source_grid(sitk_img):
    """Return (image, info) after applying DICOM export grid policy.

    info fields:
      - resampled_source_grid: bool
      - resampled_to_axial: bool
      - resample_mode: str
      - resample_reason: str
      - target_grid: "original" | "axial" | "orthonormal_oblique"
    """
    mode = _dicom_resample_policy()
    info = {
        "resampled_source_grid": False,
        "resampled_to_axial": False,
        "resample_mode": mode,
        "resample_reason": "",
        "target_grid": "original",
    }

    if mode == "never":
        if not _sitk_is_classic_dicom_geometry_compatible(sitk_img):
            raise ValueError(
                "source image has a non-orthogonal in-plane voxel basis that "
                "classic DICOM cannot represent; resampling is disabled by "
                "MIMICS_DICOM_RESAMPLE_MODE=never"
            )
        return sitk_img, info

    if mode == "axial":
        if not _sitk_is_axial(sitk_img):
            sitk_img = _resample_to_oriented_grid(sitk_img, np.eye(3, dtype=float))
            info.update({
                "resampled_source_grid": True,
                "resampled_to_axial": True,
                "resample_reason": "forced_axial",
                "target_grid": "axial",
            })
        return sitk_img, info

    if _sitk_is_classic_dicom_geometry_compatible(sitk_img):
        return sitk_img, info

    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    target_direction = _orthonormalize_direction(direction)
    sitk_img = _resample_to_oriented_grid(sitk_img, target_direction)
    info.update({
        "resampled_source_grid": True,
        "resampled_to_axial": False,
        "resample_reason": "in_plane_shear_not_representable_by_classic_dicom",
        "target_grid": "orthonormal_oblique",
    })
    return sitk_img, info


def nifti_to_derived_dicom(
    nifti_path: str,
    dicom_out: str,
    case_id: str = "case",
    source_nifti_out: str | None = None,
    modality: str = "CT",
) -> dict:
    """Convert NIfTI image to derived DICOM series for Mimics import."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage, MRImageStorage

    modality = str(modality or "CT").upper()
    if modality not in ("CT", "MR"):
        modality = "CT"
    storage_class = MRImageStorage if modality == "MR" else CTImageStorage

    sitk_img = _read_image_sitk_lps(nifti_path)

    sitk_img, grid_info = _prepare_dicom_source_grid(sitk_img)
    resampled_to_axial = bool(grid_info.get("resampled_to_axial"))
    resampled_source_grid = bool(grid_info.get("resampled_source_grid"))

    if source_nifti_out and (resampled_source_grid or not is_nifti_file(nifti_path)):
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

    rescale_slope = 1.0
    rescale_intercept = 0.0
    intensity_encoding = "dicom_integer_identity_v1"
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array.reshape(-1)
    value_min = float(np.min(finite)) if finite.size else 0.0
    value_max = float(np.max(finite)) if finite.size else 0.0
    requires_scaling = (
        (np.issubdtype(array.dtype, np.floating) and modality == "MR")
        or value_min < -32768.0 or value_max > 32767.0
    )
    if requires_scaling and value_max > value_min:
        intensity_encoding = "dicom_uint16_linear_rescale_v1"
        rescale_intercept = value_min
        rescale_slope = (value_max - value_min) / 65535.0
        clean = np.nan_to_num(array.astype(np.float64), nan=value_min, posinf=value_max, neginf=value_min)
        pixel_array = np.rint((clean - rescale_intercept) / rescale_slope).clip(0, 65535).astype(np.uint16)
        bits_allocated = 16
        bits_stored = 16
        pixel_representation = 0
    elif array.dtype == np.uint8:
        pixel_array = array.astype(np.uint16)
        bits_allocated = 16
        bits_stored = 16
        pixel_representation = 0
    else:
        if np.issubdtype(array.dtype, np.floating):
            intensity_encoding = "dicom_int16_cast_v1"
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
    slice_normal = _normalize_vector(np.cross(row_cosine, column_cosine))
    slice_step = np.asarray(affine_lps[:3, 2], dtype=float)
    normal_slice_spacing = abs(float(np.dot(slice_normal, slice_step)))
    if normal_slice_spacing <= 1e-8:
        raise ValueError("slice positions do not advance through the image plane")

    for slice_idx in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = storage_class
        file_meta.MediaStorageSOPInstanceUID = generate_uid(
            entropy_srcs=[source_hash, "sop", str(slice_idx)]
        )

        dcm_path = str(out_dir / "slice_{:04d}.dcm".format(slice_idx + 1))
        ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        # Keep the encoded dataset consistent with the transfer syntax in
        # file_meta. Older pydicom releases otherwise default the body to
        # implicit VR while advertising Explicit VR Little Endian.
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = storage_class
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "ANON"
        ds.PatientID = case_id
        ds.Modality = modality
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
        ds.SliceThickness = float(normal_slice_spacing)
        ds.SpacingBetweenSlices = float(normal_slice_spacing)
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
        ds.SliceLocation = float(np.dot(slice_normal, slice_origin[:3]))
        ds.ImageOrientationPatient = [
            float(row_cosine[0]), float(row_cosine[1]), float(row_cosine[2]),
            float(column_cosine[0]), float(column_cosine[1]), float(column_cosine[2]),
        ]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = float(rescale_intercept)
        ds.RescaleSlope = float(rescale_slope)

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
        "resampled_source_grid": resampled_source_grid,
        "resampled_to_axial": resampled_to_axial,
        "resample_mode": grid_info.get("resample_mode", "auto"),
        "resample_reason": grid_info.get("resample_reason", ""),
        "resampled_grid": grid_info.get("target_grid", "original"),
        "source_intensity_encoding": intensity_encoding,
        "source_intensity_rescale_slope": float(rescale_slope),
        "source_intensity_rescale_intercept": float(rescale_intercept),
        "source_intensity_value_min": float(value_min),
        "source_intensity_value_max": float(value_max),
        "dicom_stored_value_min": int(np.min(pixel_array)) if pixel_array.size else 0,
        "dicom_stored_value_max": int(np.max(pixel_array)) if pixel_array.size else 0,
        "source_nifti_path": str(Path(source_nifti_out).resolve()) if (source_nifti_out and resampled_source_grid) else "",
        "series_uid": str(series_uid),
        "dicom_folder": str(out_dir),
    }


def prepare_source_fastpath_nifti(nifti_path: str, source_nifti_out: str) -> dict:
    """Prepare an on-demand source NIfTI aligned with Mimics-imported grid.

    This is a lightweight helper for nnInteractive fast path. It performs the
    same DICOM-grid preparation rule as :func:`nifti_to_derived_dicom`, but
    does not emit DICOM slices.
    """
    import SimpleITK as sitk

    sitk_img = _read_image_sitk_lps(nifti_path)
    sitk_img, grid_info = _prepare_dicom_source_grid(sitk_img)

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
        "resampled_source_grid": bool(grid_info.get("resampled_source_grid")),
        "resampled_to_axial": bool(grid_info.get("resampled_to_axial")),
        "resample_mode": grid_info.get("resample_mode", "auto"),
        "resample_reason": grid_info.get("resample_reason", ""),
        "resampled_grid": grid_info.get("target_grid", "original"),
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
    """Choose one authoritative affine and normalize ambiguous form codes.

    Some NIfTI files have qform_code=0 or sform_code=0, which causes
    tools like ITK-Snap, SimpleITK and 3D Slicer to interpret the
    spatial mapping differently.  Re-setting both codes forces a
    consistent sform_code=2 / qform_code=2 header so every downstream
    reader sees the same affine.

    A valid coded sform is preferred because it can represent the general
    voxel-to-world transform. A valid coded qform is the second choice.
    Uncoded but usable forms are accepted only as recovery inputs. If neither
    is usable, a pixdim-derived affine is used.

    The file on disk is NOT modified; only the in-memory image object
    header is normalized.
    """
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
        if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
            return False
        linear = np.asarray(mat[:3, :3], dtype=float)
        spacing = np.linalg.norm(linear, axis=0)
        return bool(
            np.all(spacing > 1e-8)
            and abs(float(np.linalg.det(linear))) > 1e-8
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
    try:
        qform_code = int(nifti_img.header["qform_code"])
    except Exception:
        qform_code = 0
    try:
        sform_code = int(nifti_img.header["sform_code"])
    except Exception:
        sform_code = 0

    if sform_code > 0 and _usable_affine(sform):
        result = np.asarray(sform, dtype=float)
    elif qform_code > 0 and _usable_affine(qform):
        result = np.asarray(qform, dtype=float)
    elif _usable_affine(sform):
        result = np.asarray(sform, dtype=float)
    elif _usable_affine(qform):
        result = np.asarray(qform, dtype=float)
    else:
        pixdim = nifti_img.header.get_zooms()[:3]
        result = np.eye(4)
        for i, p in enumerate(pixdim):
            pp = float(p)
            if np.isfinite(pp) and pp >= 0.001:
                result[i, i] = pp
            else:
                result[i, i] = 1.0

    # sform is authoritative. qform is synchronized when the transform is
    # quaternion-representable; otherwise it is disabled so readers cannot
    # prefer an approximation over the exact sform.
    _try_set_sform(nifti_img, result)
    _try_set_qform(nifti_img, result)
    normalized_qform = _safe_get_qform(nifti_img)
    if not _usable_affine(normalized_qform) or not np.allclose(
        normalized_qform, result, atol=1e-5, rtol=0.0
    ):
        try:
            nifti_img.header["qform_code"] = 0
        except Exception:
            pass

    return result


def _nifti_header_is_unambiguous(nifti_img) -> bool:
    """Whether qform and sform are both coded, finite, and equivalent."""
    try:
        qform_code = int(nifti_img.header["qform_code"])
        sform_code = int(nifti_img.header["sform_code"])
        qform = np.asarray(nifti_img.get_qform(), dtype=float)
        sform = np.asarray(nifti_img.get_sform(), dtype=float)
        if qform_code <= 0 or sform_code <= 0:
            return False
        if qform.shape != (4, 4) or sform.shape != (4, 4):
            return False
        if not np.all(np.isfinite(qform)) or not np.all(np.isfinite(sform)):
            return False
        if abs(float(np.linalg.det(qform[:3, :3]))) <= 1e-8:
            return False
        if abs(float(np.linalg.det(sform[:3, :3]))) <= 1e-8:
            return False
        return bool(np.allclose(qform, sform, atol=1e-5, rtol=0.0))
    except Exception:
        return False


# -- SimpleITK-based image reading with header correction and LPS orient

def _nibabel_image_to_sitk_lps(nifti_img, affine_ras):
    """Build a SimpleITK image without losing a general NIfTI sform."""
    import SimpleITK as sitk

    array_xyz = np.asanyarray(nifti_img.dataobj)
    if array_xyz.ndim != 3:
        raise ValueError(
            "NIfTI image must be 3D: shape={}".format(array_xyz.shape)
        )
    affine_lps = RAS_TO_LPS @ np.asarray(affine_ras, dtype=float)
    linear = affine_lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if (
        not np.all(np.isfinite(spacing))
        or np.any(spacing <= 1.0e-8)
    ):
        raise ValueError("NIfTI affine has invalid voxel spacing")
    direction = linear / spacing
    image = sitk.GetImageFromArray(np.transpose(array_xyz, (2, 1, 0)))
    image.SetSpacing([float(value) for value in spacing])
    image.SetOrigin([float(value) for value in affine_lps[:3, 3]])
    image.SetDirection([float(value) for value in direction.reshape(-1)])
    return image


def _read_image_sitk_lps(path: str):
    """Read an image via SimpleITK, correcting NIfTI header if needed, then orient to LPS.

    Tries ``sitk.ReadImage`` first. If a NIfTI header needs correction, a
    corrected temporary copy is read instead. The source image is never
    modified. Finally ``DICOMOrient("LPS")`` performs a lossless axis
    permutation/flip; it does not interpolate voxel values.

    Works with both NIfTI files and DICOM folders.
    """
    import SimpleITK as sitk

    is_nifti = is_nifti_file(path)

    source = None
    requires_header_correction = False
    if is_nifti:
        try:
            import nibabel as nib

            source = nib.load(path)
            requires_header_correction = not _nifti_header_is_unambiguous(source)
        except Exception:
            # Preserve compatibility with NIfTI variants that ITK can read
            # even when nibabel cannot inspect their header.
            source = None
            requires_header_correction = False

    try:
        if requires_header_correction:
            raise RuntimeError("NIfTI qform/sform normalization required")
        sitk_img = sitk.ReadImage(path)
    except Exception:
        if not is_nifti:
            raise
        if source is None:
            import nibabel as nib
            source = nib.load(path)
        affine = _normalize_nifti_affine(source)
        fd, corrected_path = tempfile.mkstemp(suffix=".nii.gz")
        os.close(fd)
        try:
            nib.save(source, corrected_path)
            try:
                sitk_img = sitk.ReadImage(corrected_path)
            except Exception:
                sitk_img = _nibabel_image_to_sitk_lps(source, affine)
        finally:
            try:
                os.remove(corrected_path)
            except OSError:
                pass
        if not _sitk_is_classic_dicom_geometry_compatible(sitk_img):
            # DICOMOrient cannot represent a sheared basis. The caller's
            # DICOM-grid policy performs the one required interpolation.
            return sitk_img

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

def _read_mask_array_and_affine(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read label data in its on-disk voxel order with an explicit RAS affine."""
    if is_nifti_file(path):
        import nibabel as nib

        image = nib.load(path)
        array = np.asanyarray(image.dataobj)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
        affine_ras = _normalize_nifti_affine(image)
        return np.ascontiguousarray(array), np.asarray(affine_ras, dtype=float)

    sitk_img = _read_image_sitk_lps(path)
    array = _sitk_to_xyz_array(sitk_img)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
    affine_ras = LPS_TO_RAS @ _sitk_to_lps_affine(sitk_img)
    return np.ascontiguousarray(array), affine_ras


def read_nifti_mask(path: str) -> np.ndarray:
    array, _affine = _read_mask_array_and_affine(path)
    return (array != 0).astype(np.uint8)


def read_nifti_mask_with_affine(path: str) -> tuple[np.ndarray, np.ndarray]:
    array, affine_ras = _read_mask_array_and_affine(path)
    return (array != 0).astype(np.uint8), affine_ras


def read_mask_labels_with_affine(path: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Read a medical mask while preserving integer label values and RAS geometry."""
    array, affine_ras = _read_mask_array_and_affine(path)
    if not np.all(np.isfinite(array)):
        raise ValueError("mask contains NaN or infinite label values: {}".format(path))

    if np.issubdtype(array.dtype, np.floating):
        rounded = np.rint(array)
        if np.allclose(array, rounded, atol=1e-5, rtol=0.0):
            array = rounded.astype(np.int64)
        else:
            # Probability-like masks are binary segmentations, not thousands
            # of separate floating-point labels.
            array = (array != 0).astype(np.uint8)
    labels = [value.item() if hasattr(value, "item") else value for value in np.unique(array)]
    return np.ascontiguousarray(array), affine_ras, labels


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


def _read_canonical_source_mask(
    path: str, expected_shape: tuple[int, int, int], expected_affine: np.ndarray
) -> tuple[np.ndarray, str]:
    """Read and strictly validate an explicit source-grid segmentation."""
    source_path = os.path.abspath(os.path.expanduser(str(path or "")))
    if not os.path.isfile(source_path):
        raise ValueError("canonical source Mask file was not found: {}".format(source_path))
    array, affine = _read_mask_array_and_affine(source_path)
    if tuple(int(value) for value in array.shape) != tuple(int(value) for value in expected_shape):
        raise ValueError(
            "canonical source Mask shape does not match the source image: {} != {} ({})".format(
                tuple(int(value) for value in array.shape),
                tuple(int(value) for value in expected_shape),
                source_path,
            )
        )
    if not _affine_close(np.asarray(affine, dtype=float), expected_affine):
        raise ValueError(
            "canonical source Mask affine does not match the source image: {}".format(
                source_path
            )
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("canonical source Mask contains NaN or infinite values: {}".format(source_path))
    unique = np.unique(array)
    if not np.all(np.logical_or(unique == 0, unique == 1)):
        raise ValueError(
            "canonical source Mask must contain only binary values 0 and 1: {} values={}".format(
                source_path, unique[:20].tolist()
            )
        )
    canonical = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    digest = hashlib.sha256()
    with open(source_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return canonical, "sha256:" + digest.hexdigest()


def _mask_file_declares_spatial_geometry(path: str) -> bool:
    """Return False only when a supported text header actually omits geometry.

    Origin zero and identity direction are valid physical geometry. They must
    never be guessed to mean "missing" from the numeric affine alone. MHD/MHA
    and NRRD expose whether the corresponding header fields were present, so
    voxel-aligned fallback is limited to those demonstrably incomplete files.
    """
    name = str(path or "").lower()
    if not name.endswith((".mhd", ".mha", ".nrrd")):
        return True
    try:
        with open(path, "rb") as handle:
            header = handle.read(262144).decode("latin-1", "ignore").lower()
    except Exception:
        return True
    if name.endswith((".mhd", ".mha")):
        origin_keys = ("offset", "position", "origin")
        direction_keys = ("transformmatrix", "orientation")
        keys = set()
        for line in header.splitlines():
            if "=" not in line:
                continue
            key = line.split("=", 1)[0].strip().replace(" ", "")
            keys.add(key)
            if key == "elementdatafile" and "local" in line:
                break
        return bool(keys.intersection(origin_keys) or keys.intersection(direction_keys))
    return "space origin:" in header or "space directions:" in header


def _mask_resample_method() -> str:
    method = str(os.environ.get("MIMICS_MASK_RESAMPLE_METHOD", DEFAULT_MASK_RESAMPLE_METHOD) or "").strip().lower()
    if method not in ("distance", "nearest"):
        method = DEFAULT_MASK_RESAMPLE_METHOD
    return method


def _sitk_reference_from_shape_affine(image_shape, image_affine):
    import SimpleITK as sitk

    affine_lps = RAS_TO_LPS @ np.asarray(image_affine, dtype=float)
    spacing = _voxel_spacing_from_affine(affine_lps)
    direction = affine_lps[:3, :3] / spacing
    if not np.all(np.isfinite(direction)):
        raise ValueError("target image affine has invalid direction columns")
    reference = sitk.Image(int(image_shape[0]), int(image_shape[1]), int(image_shape[2]), sitk.sitkFloat32)
    reference.SetOrigin([float(value) for value in affine_lps[:3, 3]])
    reference.SetSpacing([float(value) for value in spacing])
    reference.SetDirection([float(value) for value in direction.reshape(-1)])
    return reference


def _sitk_binary_from_mask(mask: np.ndarray, mask_affine: np.ndarray):
    import SimpleITK as sitk

    binary = np.ascontiguousarray((mask != 0).astype(np.uint8))
    affine_lps = RAS_TO_LPS @ np.asarray(mask_affine, dtype=float)
    spacing = _voxel_spacing_from_affine(affine_lps)
    direction = affine_lps[:3, :3] / spacing
    if not np.all(np.isfinite(direction)):
        raise ValueError("mask affine has invalid direction columns")
    image = sitk.GetImageFromArray(np.transpose(binary, (2, 1, 0)))
    image = sitk.Cast(image, sitk.sitkUInt8)
    image.SetOrigin([float(value) for value in affine_lps[:3, 3]])
    image.SetSpacing([float(value) for value in spacing])
    image.SetDirection([float(value) for value in direction.reshape(-1)])
    return image


def _resample_mask_distance_field(mask: np.ndarray, mask_affine: np.ndarray, image_shape, image_affine) -> np.ndarray:
    """Resample binary mask by distance field + linear interpolation.

    Compared with pure nearest-neighbor index mapping, distance-field
    interpolation typically reduces stair-step artifacts after rotation or
    oblique regridding while preserving a binary output via zero-threshold.
    """
    import SimpleITK as sitk

    source = _sitk_binary_from_mask(mask, mask_affine)
    reference = _sitk_reference_from_shape_affine(image_shape, image_affine)

    distance = sitk.SignedMaurerDistanceMap(
        source,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    max_spacing = max(float(value) for value in reference.GetSpacing())
    resampler.SetDefaultPixelValue(max(1.0, max_spacing * 4.0))
    distance_resampled = resampler.Execute(distance)
    distance_xyz = _sitk_to_xyz_array(distance_resampled)
    return np.ascontiguousarray((distance_xyz <= 0.0).astype(np.uint8))


def _resample_mask_nearest_numpy(mask: np.ndarray, mask_affine: np.ndarray, image_shape, image_affine) -> np.ndarray:
    """Nearest-neighbor resample of a mask into image voxel index space."""
    inv_mask_affine = np.linalg.inv(mask_affine)
    out = np.zeros(tuple(int(v) for v in image_shape), dtype=np.uint8)
    x_count, y_count, z_count = [int(v) for v in image_shape]
    yy_template = np.arange(y_count, dtype=float)
    xx_template = np.arange(x_count, dtype=float)

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


def _grid_mapping_is_discrete(source_affine: np.ndarray, target_affine: np.ndarray, atol: float = 1e-4) -> bool:
    """Return whether target indices map to source by integer flips/permutation.

    RAS/LPS reorientation and DICOM row/column conventions often change only
    index order and sign. Those cases must be handled as exact voxel reindexing,
    not as an interpolating resample.
    """
    try:
        mapping = np.linalg.inv(np.asarray(source_affine, dtype=float)) @ np.asarray(target_affine, dtype=float)
    except (ValueError, np.linalg.LinAlgError):
        return False
    linear = mapping[:3, :3]
    rounded = np.rint(linear)
    if not np.allclose(linear, rounded, atol=atol, rtol=0.0):
        return False
    absolute = np.abs(rounded).astype(int)
    if not (
        np.all(np.sum(absolute, axis=0) == 1)
        and np.all(np.sum(absolute, axis=1) == 1)
    ):
        return False
    translation = mapping[:3, 3]
    if not np.allclose(translation, np.rint(translation), atol=atol, rtol=0.0):
        return False
    return bool(np.allclose(mapping[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0))


def resample_mask_to_image_grid(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    image_shape: tuple[int, int, int],
    image_affine: np.ndarray,
    allow_voxel_aligned_fallback: bool = False,
) -> np.ndarray:
    """Resample mask into image voxel index space.

    Exact matching grids return unchanged. Axis permutations/flips use exact
    voxel reindexing. For a genuinely different physical grid, nearest-neighbor
    is the conservative default because labels must remain discrete. Set
    MIMICS_MASK_RESAMPLE_METHOD=distance to explicitly opt into distance-field
    interpolation for binary masks.
    """
    if tuple(mask.shape) == tuple(image_shape) and _affine_close(mask_affine, image_affine):
        return np.ascontiguousarray(mask.astype(np.uint8))
    if tuple(mask.shape) == tuple(image_shape) and not _affine_is_usable(mask_affine):
        return np.ascontiguousarray(mask.astype(np.uint8))
    if allow_voxel_aligned_fallback:
        if tuple(mask.shape) == tuple(image_shape):
            return np.ascontiguousarray(mask.astype(np.uint8))
        raise ValueError(
            "mask header has no origin/direction and its shape does not match the target grid: "
            "{} != {}".format(tuple(mask.shape), tuple(image_shape))
        )

    if not _affine_is_usable(mask_affine):
        raise ValueError(
            "mask affine is singular or invalid; spatial resampling is unsafe"
        )
    if not _affine_is_usable(image_affine):
        raise ValueError(
            "target image affine is singular or invalid; spatial resampling is unsafe"
        )

    # A coordinate-system or storage-orientation change is not a geometric
    # resample. Keep every label voxel exact in this common path.
    if _grid_mapping_is_discrete(mask_affine, image_affine):
        return _resample_mask_nearest_numpy(mask, mask_affine, image_shape, image_affine)

    method = _mask_resample_method()
    binary_like = bool(np.all((mask == 0) | (mask == 1)))

    if method == "distance" and binary_like:
        try:
            return _resample_mask_distance_field(mask, mask_affine, image_shape, image_affine)
        except Exception:
            # Keep legacy behavior if SimpleITK distance-map resample fails.
            pass

    try:
        return _resample_mask_nearest_numpy(mask, mask_affine, image_shape, image_affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "mask affine could not be inverted for spatial resampling: {}".format(exc)
        )


def _validate_resampled_mask_foreground(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    context: str,
) -> tuple[int, int]:
    """Reject a non-empty label that silently maps to an empty target grid."""
    source_foreground = int(np.count_nonzero(source_mask))
    target_foreground = int(np.count_nonzero(target_mask))
    if source_foreground > 0 and target_foreground == 0:
        raise RuntimeError(
            "{} produced an empty mask from {} foreground voxel(s). "
            "The source and target grids likely do not overlap, so the operation "
            "was stopped instead of accepting a silently misregistered label.".format(
                context, source_foreground
            )
        )
    return source_foreground, target_foreground


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


# Formats do_convert can write. SimpleITK covers .nrrd/.mha/.mhd without
# adding new dependencies, while .nii(.gz) keeps the nibabel path that also
# sets qform/sform explicitly.
SUPPORTED_EXPORT_FORMATS = ("nii.gz", "nrrd", "mha", "nii")


def _write_mask_sitk(array: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    """Write a mask via SimpleITK (.nrrd/.mha) with an RAS affine."""
    import SimpleITK as sitk

    # The export pipeline works in the nibabel (x, y, z) voxel convention;
    # SimpleITK's GetImageFromArray expects (z, y, x), so transpose.
    raster = sitk.GetImageFromArray(np.transpose(array.astype(np.uint8), (2, 1, 0)))
    matrix = np.asarray(affine, dtype=float)
    # SimpleITK stores direction as a unit-orthonormal matrix with the voxel
    # size in spacing, so split the affine's linear part into the two. Affines
    # with shear (non-orthogonal columns) cannot be represented; keep spacing
    # 1 and the identity direction there rather than emitting wrong geometry.
    direction_ras = matrix[:3, :3]
    spacing = np.linalg.norm(direction_ras, axis=0)
    if np.all(spacing > 1e-8):
        unit_direction = direction_ras / spacing
        residual = unit_direction.T.dot(unit_direction) - np.eye(3)
        if np.max(np.abs(residual)) < 1e-4:
            spacing_tuple = tuple(float(v) for v in spacing)
            direction = unit_direction
        else:
            spacing_tuple = (1.0, 1.0, 1.0)
            direction = np.eye(3)
    else:
        spacing_tuple = (1.0, 1.0, 1.0)
        direction = np.eye(3)
    # NIfTI world space is RAS; SimpleITK's LPS convention needs a flip of the
    # first two direction/origin components to keep exported masks
    # co-registered with the source images they came from.
    direction_lps = direction.copy()
    direction_lps[:2, :] *= -1.0
    origin_lps = matrix[:3, 3].copy()
    origin_lps[:2] *= -1.0
    raster.SetOrigin(tuple(float(v) for v in origin_lps))
    raster.SetDirection(tuple(float(v) for v in direction_lps.flatten()))
    raster.SetSpacing(spacing_tuple)
    sitk.WriteImage(raster, output_path, useCompression=True)


def write_mask_file(array: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    """Atomically write a mask in the format implied by the output extension."""
    destination = os.path.abspath(output_path)
    lowered = destination.lower()
    if lowered.endswith((".nrrd", ".mha", ".mhd")):
        _write_mask_sitk_atomic(array, affine, destination)
    elif lowered.endswith(".nii.gz") or lowered.endswith(".nii"):
        write_mask_nifti(array, affine, destination)
    else:
        raise ValueError(
            "Unsupported mask export format: {}".format(output_path)
        )


def _read_existing_mask(path: str, export_affine: np.ndarray):
    """Read an exported mask back for skip-existing comparison.

    Returns (array, affine_close) or (None, False) when the file cannot be
    read. NIfTI is loaded via nibabel; .nrrd/.mha/.mhd go through SimpleITK
    and are converted back to the RAS affine convention.
    """
    lowered = path.lower()
    try:
        if lowered.endswith(".nrrd") or lowered.endswith(".mha") or lowered.endswith(".mhd"):
            import SimpleITK as sitk

            raster = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(raster)
            # sitk arrays come back z/y/x; the export loop works in the
            # NIfTI x/y/z convention, so reverse the axes.
            array = np.transpose(array, (2, 1, 0))
            direction = np.asarray(raster.GetDirection(), dtype=float).reshape(3, 3)
            spacing = np.asarray(raster.GetSpacing(), dtype=float)
            origin = np.asarray(raster.GetOrigin(), dtype=float)
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing
            affine[:3, 3] = origin
            affine[:2, :] *= -1.0  # LPS -> RAS
            return array, _affine_close(affine, export_affine)
        import nibabel as nib

        existing_img = nib.load(path)
        array = np.asanyarray(existing_img.dataobj)
        return array, _affine_close(existing_img.affine, export_affine)
    except Exception:
        return None, False


def write_mask_nifti(array: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    import nibabel as nib

    destination = os.path.abspath(output_path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    suffix = ".nii.gz" if destination.lower().endswith(".nii.gz") else ".nii"
    fd, temporary = tempfile.mkstemp(
        prefix="._mimics_mask_", suffix=suffix, dir=parent
    )
    os.close(fd)
    try:
        nii = nib.Nifti1Image(array.astype(np.uint8), affine)
        nii.set_qform(np.asarray(affine, dtype=float), code=1)
        nii.set_sform(np.asarray(affine, dtype=float), code=1)
        nib.save(nii, temporary)
        try:
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        except Exception:
            pass

        last_error = None
        for attempt in range(12):
            try:
                os.replace(temporary, destination)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.2, 0.02 * (attempt + 1)))
        raise RuntimeError(
            "Could not publish exported Mask '{}'. The existing file, if "
            "present, was left unchanged. Error: {}".format(
                destination, last_error
            )
        )
    finally:
        try:
            if os.path.isfile(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _write_mask_sitk_atomic(array: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    """Atomically publish a SimpleITK-writable mask (.nrrd/.mha/.mhd).

    sitk.WriteImage has no temp+replace mode, so the file is written next to
    the destination and swapped in with the same replace-retry loop used by
    the NIfTI writer.
    """
    destination = os.path.abspath(output_path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(destination)[1] or ".nrrd"
    fd, temporary = tempfile.mkstemp(
        prefix="._mimics_mask_", suffix=suffix, dir=parent
    )
    os.close(fd)
    try:
        os.remove(temporary)  # sitk needs to create the file itself
        _write_mask_sitk(array, affine, temporary)
        try:
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        except Exception:
            pass
        last_error = None
        for attempt in range(12):
            try:
                os.replace(temporary, destination)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.2, 0.02 * (attempt + 1)))
        raise RuntimeError(
            "Could not publish exported Mask '{}'. The existing file, if "
            "present, was left unchanged. Error: {}".format(
                destination, last_error
            )
        )
    finally:
        try:
            if os.path.isfile(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _force_nifti_affine(path: Path, target_affine: np.ndarray) -> None:
    """Rewrite a NIfTI so qform/sform exactly match target_affine."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    out = nib.Nifti1Image(data, np.asarray(target_affine, dtype=float), header=img.header)
    out.set_qform(np.asarray(target_affine, dtype=float), code=1)
    out.set_sform(np.asarray(target_affine, dtype=float), code=1)
    nib.save(out, str(path))


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
    # SimpleITK may introduce tiny direction/origin float drift.  Downstream
    # training compares image/label affines strictly, so enforce exact target.
    _force_nifti_affine(out, target_affine_ras)
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
        slice_step = None
        if len(positions) > 1:
            position_steps = [
                positions[i][1] - positions[i - 1][1]
                for i in range(1, len(positions))
            ]
            valid_steps = [
                value for value in position_steps
                if float(np.dot(value, slice_dir)) > 1e-6
            ]
            if valid_steps:
                slice_step = np.median(np.asarray(valid_steps, dtype=float), axis=0)
                slice_thickness = abs(float(np.dot(slice_step, slice_dir)))
    else:
        origin_lps = np.array([0.0, 0.0, 0.0], dtype=float)
        slice_step = None

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_cosine * float(pixel_spacing[1])
    affine_lps[:3, 1] = column_cosine * float(pixel_spacing[0])
    affine_lps[:3, 2] = (
        np.asarray(slice_step, dtype=float)
        if slice_step is not None
        else slice_dir * float(slice_thickness)
    )
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


def _other_medical_image_candidates(case_dir: Path):
    preferred = (
        "ct.mhd", "mri.mhd", "ct.mha", "mri.mha",
        "ct.nrrd", "mri.nrrd", "ct.nrrd.gz", "mri.nrrd.gz",
    )
    files = [child for child in case_dir.iterdir() if child.is_file() and is_medical_image_file(str(child)) and not is_nifti_file(str(child))]
    return sorted(files, key=lambda path: (preferred.index(path.name.lower()) if path.name.lower() in preferred else len(preferred), path.name.lower()))


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
    for candidate in _other_medical_image_candidates(d):
        try:
            shape, affine = get_medical_image_disk_geometry(str(candidate))
            return {"path": str(candidate), "kind": "medical_image", "shape": shape, "affine": affine}
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
    if d.is_dir() and is_dicom_folder(str(d)):
        return {
            "path": str(d), "kind": "dicom_folder",
            "shape": get_image_shape_from_dicom(str(d)),
            "affine": get_image_affine_from_dicom(str(d)),
        }
    return None


def get_source_image_geometry(path: str) -> dict | None:
    if is_nifti_file(path):
        shape, affine = get_nifti_disk_geometry(path)
        return {"path": str(Path(path).resolve()), "kind": "nifti", "shape": shape, "affine": affine}
    if is_medical_image_file(path):
        shape, affine = get_medical_image_disk_geometry(path)
        return {"path": str(Path(path).resolve()), "kind": "medical_image", "shape": shape, "affine": affine}
    if is_dicom_folder(path):
        return {"path": str(Path(path).resolve()), "kind": "dicom_folder",
                "shape": get_image_shape_from_dicom(path), "affine": get_image_affine_from_dicom(path)}
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
    source_image_shape = None
    prepare_grid_info = {
        "resampled_source_grid": False,
        "resampled_to_axial": False,
        "resampled_grid": "original",
        "resample_mode": _dicom_resample_policy(),
        "resample_reason": "",
    }
    source_intensity_encoding = ""
    source_intensity_rescale_slope = None
    source_intensity_rescale_intercept = None
    source_intensity_value_min = None
    source_intensity_value_max = None
    dicom_stored_value_min = None
    dicom_stored_value_max = None
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
    elif is_medical_image_file(image_path):
        if not dicom_out:
            return {"status": "error", "error": "dicom_out required for medical image files"}
        case_id = params.get("case_id", "case")
        requested_modality = str(params.get("modality") or "").upper()
        if not requested_modality:
            lower_name = Path(image_path).name.lower()
            requested_modality = "MR" if ("mri" in lower_name or lower_name.startswith("mr")) else "CT"
        source_cache = ""
        info = nifti_to_derived_dicom(
            image_path,
            dicom_out,
            case_id=case_id,
            source_nifti_out=source_cache or None,
            modality=requested_modality,
        )
        prepare_grid_info = {
            "resampled_source_grid": bool(info.get("resampled_source_grid")),
            "resampled_to_axial": bool(info.get("resampled_to_axial")),
            "resampled_grid": str(info.get("resampled_grid", "original") or "original"),
            "resample_mode": str(info.get("resample_mode", _dicom_resample_policy()) or _dicom_resample_policy()),
            "resample_reason": str(info.get("resample_reason", "") or ""),
        }
        source_intensity_encoding = str(info.get("source_intensity_encoding", "") or "")
        source_intensity_rescale_slope = info.get("source_intensity_rescale_slope")
        source_intensity_rescale_intercept = info.get("source_intensity_rescale_intercept")
        source_intensity_value_min = info.get("source_intensity_value_min")
        source_intensity_value_max = info.get("source_intensity_value_max")
        dicom_stored_value_min = info.get("dicom_stored_value_min")
        dicom_stored_value_max = info.get("dicom_stored_value_max")
        dicom_folder = info["dicom_folder"]
        # Use the actual DICOM grid affine for mask resampling and Mimics coordinate
        # metadata.  For source images that were resampled for DICOM compatibility,
        # this differs from the original NIfTI affine.
        dicom_affine_lps = np.array(info["affine_lps"]).reshape(4, 4)
        image_affine = LPS_TO_RAS @ dicom_affine_lps
        source_path = image_path
        if is_nifti_file(source_path):
            source_image_shape, source_nifti_affine = get_nifti_disk_geometry(source_path)
        else:
            source_image_shape, source_nifti_affine = get_medical_image_disk_geometry(source_path)
        source_image_shape = tuple(int(value) for value in source_image_shape)
        source_image_path_for_fastpath = str(Path(source_path).resolve())
        source_image_kind = "nifti" if is_nifti_file(image_path) else "medical_image"
        source_grid_matches_mimics = (
            tuple(source_image_shape) == tuple(int(v) for v in info["shape"])
            and _affine_close(source_nifti_affine, image_affine)
        )
        source_image_index_space = (
            ("nifti_ijk_matches_derived_dicom_columns_rows_slices_v1" if is_nifti_file(image_path)
             else "medical_image_ijk_matches_derived_dicom_columns_rows_slices_v1")
            if source_grid_matches_mimics
            else "derived_dicom_lps_resampled_from_source_image_v2"
        )
        source_world_coordinate_system = "ras"
        mimics_world_coordinate_system = "lps"
        source_to_mimics_world_matrix = RAS_TO_LPS
        source_image_modality = requested_modality
        image_shape = tuple(int(v) for v in info["shape"])
    else:
        return {"status": "error", "error": "image path is neither a DICOM folder nor a supported 3D image file: {}".format(image_path)}
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
        try:
            slopes = set(float(getattr(ds, "RescaleSlope", 1.0) or 1.0) for ds in headers)
            intercepts = set(float(getattr(ds, "RescaleIntercept", 0.0) or 0.0) for ds in headers)
            if len(slopes) == 1 and len(intercepts) == 1:
                source_intensity_encoding = "dicom_uniform_rescale_v1"
                source_intensity_rescale_slope = next(iter(slopes))
                source_intensity_rescale_intercept = next(iter(intercepts))
        except (TypeError, ValueError):
            # Variable or malformed source tags must not break data import. The
            # project remains usable and inference will use its raw-GV fallback.
            pass
        image_shape = (int(headers[0].Columns), int(headers[0].Rows), int(len(headers)))
    if source_image_shape is None:
        source_image_shape = tuple(int(value) for value in image_shape)

    # Both matrices are expressed in RAS world coordinates.
    # source_voxel_to_ras_matrix: source_image_path geometry.
    # mimics_voxel_to_ras_matrix: the grid that Mimics actually imports
    #   (may differ from source when geometry is resampled for DICOM compatibility).
    if is_medical_image_file(image_path):
        source_voxel_to_ras_matrix = source_nifti_affine.astype(float)
        mimics_voxel_to_ras_matrix = image_affine.astype(float)  # DICOM import affine
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
        image_grid_mask = resample_mask_to_image_grid(
            array, mask_affine, image_shape, image_affine,
            allow_voxel_aligned_fallback=not _mask_file_declares_spatial_geometry(mask_path),
        )
        source_foreground, image_foreground = _validate_resampled_mask_foreground(
            array,
            image_grid_mask,
            "Importing mask '{}' onto the prepared Mimics grid".format(name),
        )
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
            "source_foreground_voxels": source_foreground,
            "foreground_voxels": image_foreground,
            "buffer_axes": list(axes),
            "buffer_flips": list(flips),
        })

    # Build a content fingerprint for incremental rebuild detection
    image_ident = str(Path(image_path).resolve())
    mask_fingerprints = []
    for mask_result in mask_results:
        digest = mask_result.get("u8_hash")
        if not digest:
            with open(mask_result["u8_path"], "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        mask_fingerprints.append("{0}:{1}".format(mask_result["name"], digest))
    source_fingerprint = "sha256:" + hashlib.sha256(
        (image_ident + "|" + "|".join(sorted(mask_fingerprints))).encode("utf-8")
    ).hexdigest()

    return {
        "status": "ok",
        "case_id": params.get("case_id", ""),
        "dicom_folder": dicom_folder,
        "source_image_path": source_image_path_for_fastpath if is_nifti_file(image_path) else str(Path(image_path).resolve()),
        "source_image_kind": source_image_kind,
        "source_image_shape": list(source_image_shape),
        "source_image_index_space": source_image_index_space,
        "source_image_modality": source_image_modality,
        "source_intensity_encoding": source_intensity_encoding,
        "source_intensity_rescale_slope": source_intensity_rescale_slope,
        "source_intensity_rescale_intercept": source_intensity_rescale_intercept,
        "source_intensity_value_min": source_intensity_value_min,
        "source_intensity_value_max": source_intensity_value_max,
        "dicom_stored_value_min": dicom_stored_value_min,
        "dicom_stored_value_max": dicom_stored_value_max,
        "source_world_coordinate_system": source_world_coordinate_system,
        "mimics_world_coordinate_system": mimics_world_coordinate_system,
        "source_to_mimics_world_matrix": source_to_mimics_world_matrix.astype(float).tolist(),
        "source_voxel_to_ras_matrix": source_voxel_to_ras_matrix.tolist(),
        "source_fingerprint": source_fingerprint,
        "mimics_voxel_to_ras_matrix": mimics_voxel_to_ras_matrix.tolist(),
        "mimics_to_source_index_matrix": mimics_to_source_index_matrix.astype(float).tolist(),
        "resampled_source_grid": bool(prepare_grid_info.get("resampled_source_grid")),
        "resampled_to_axial": bool(prepare_grid_info.get("resampled_to_axial")),
        "resampled_grid": prepare_grid_info.get("resampled_grid", "original"),
        "resample_mode": prepare_grid_info.get("resample_mode", _dicom_resample_policy()),
        "resample_reason": prepare_grid_info.get("resample_reason", ""),
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
        allow_voxel_aligned_fallback=not _mask_file_declares_spatial_geometry(mask_path),
    )
    source_foreground, target_foreground = _validate_resampled_mask_foreground(
        mask,
        target_grid_mask,
        "Importing mask '{}' onto the active Mimics grid".format(mask_path),
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
        "source_foreground_voxels": source_foreground,
        "foreground_voxels": target_foreground,
        "mask_affine_usable": bool(_affine_is_usable(mask_affine)),
        "mask_affine_matches_target": bool(_affine_close(mask_affine, target_voxel_to_ras)),
        "mask_voxel_to_ras_matrix": mask_affine.astype(float).tolist(),
        "target_voxel_to_ras_matrix": target_voxel_to_ras.astype(float).tolist(),
        "buffer_axes": list(axes),
        "buffer_flips": list(flips),
    }


def do_prepare_masks_for_grid(params: dict) -> dict:
    """Resample source NIfTI masks into an actual Mimics voxel grid.

    Multi-label masks are automatically split into one binary mask per
    non-zero label value, named {name}_label{value}.
    """
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

        # Read mask preserving label values to detect multi-label masks.
        label_array, mask_affine, labels = read_mask_labels_with_affine(mask_path)
        nonzero_labels = [value for value in labels if value != 0]
        is_multi_label = len(nonzero_labels) > 1

        if is_multi_label:
            for label_val in labels:
                if label_val == 0:
                    continue
                binary = (label_array == label_val).astype(np.uint8)
                label_name = "{0}_label{1}".format(name, label_val)
                output_path = os.path.join(buffers_out, label_name + ".u8")
                target_grid_mask = resample_mask_to_image_grid(
                    binary, mask_affine, _shape_from_params(target_shape), target_matrix,
                    allow_voxel_aligned_fallback=not _mask_file_declares_spatial_geometry(mask_path),
                )
                source_foreground, target_foreground = (
                    _validate_resampled_mask_foreground(
                        binary,
                        target_grid_mask,
                        "Importing mask '{}' label {} onto the active Mimics grid".format(
                            name, label_val
                        ),
                    )
                )
                transformed = apply_buffer_mapping(target_grid_mask, axes, flips)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(str(output_path), "wb") as handle:
                    handle.write(transformed.tobytes(order="C"))
                row = {
                    "name": label_name,
                    "original_name": name,
                    "label_value": label_val,
                    "output_path": str(output_path),
                    "u8_path": str(output_path),
                    "mimics_shape": [int(value) for value in transformed.shape],
                    "image_shape": [int(value) for value in _shape_from_params(target_shape)],
                    "source_foreground_voxels": source_foreground,
                    "foreground_voxels": target_foreground,
                    "mask_affine_usable": bool(_affine_is_usable(mask_affine)),
                    "mask_affine_matches_target": bool(_affine_close(mask_affine, target_voxel_to_ras)),
                    "mask_voxel_to_ras_matrix": mask_affine.astype(float).tolist(),
                    "target_voxel_to_ras_matrix": target_matrix.astype(float).tolist(),
                    "buffer_axes": list(axes),
                    "buffer_flips": list(flips),
                }
                row["mask_path"] = str(Path(mask_path).resolve())
                results.append(row)
        else:
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
            row["label_value"] = (
                int(nonzero_labels[0]) if len(nonzero_labels) == 1 else 0
            )
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
        source_geometry = get_source_image_geometry(str(params.get("source_image_path") or ""))
        if source_geometry is None:
            source_geometry = find_image_geometry_in_case_dir(case_dir)
    except Exception:
        source_geometry = None
    export_space = str(params.get("export_space") or manifest.get("export_space") or "source_image").lower()
    source_mask_paths = params.get("source_mask_paths") or {}
    if not isinstance(source_mask_paths, dict):
        return {
            "status": "error",
            "error": "source_mask_paths must be a mapping from saved Mask name to source Mask path",
        }
    if source_mask_paths and export_space not in ("source", "source_image", "source_grid"):
        return {
            "status": "error",
            "error": "canonical source Mask overrides require export_space=source_image",
        }
    if (
        export_space in ("source", "source_image", "source_grid")
        and bool(params.get("require_source_geometry", False))
        and source_geometry is None
    ):
        return {
            "status": "error",
            # Machine-readable marker so the Mimics side can offer the
            # degraded Mimics-grid export instead of showing a raw failure.
            "error_code": "source_geometry_unavailable",
            "error": (
                "Source-image export was requested, but the original image geometry could not be "
                "resolved. Select the original image or case folder; export was stopped instead of "
                "writing a mask with ambiguous shape or orientation."
            ),
        }
    if source_mask_paths and source_geometry is None:
        return {
            "status": "error",
            "error": (
                "Canonical source Mask export requires the original source image geometry; "
                "the source image could not be resolved."
            ),
        }
    # True only when source_image was requested but its geometry is missing and
    # the Mimics side let the job through (after the one-time degraded-export
    # confirmation). See the else branch below.
    degraded_export = False
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
        # P1 degraded path: source_image was requested but its geometry is
        # unavailable, so the Mimics grid (with its measured voxel-to-RAS
        # matrix) is used instead. The Mimics side asks the user once per
        # batch before allowing the job through with this flag set.
        degraded_export = True

    # Create segmentations dir. Normal user exports keep the historical
    # case_dir/segmentations destination; few-shot training may pass a
    # job-scoped destination so refreshed labels do not overwrite source data.
    seg_dir = params.get("output_seg_dir") or os.path.join(case_dir, "segmentations")
    seg_dir = os.path.abspath(seg_dir)
    os.makedirs(seg_dir, exist_ok=True)
    overwrite_existing = bool(params.get("overwrite_existing", params.get("output_seg_dir") is None))

    # Export formats. The default keeps the historical .nii.gz behaviour;
    # "nrrd" and "mha" are written via SimpleITK with the same geometry.
    export_formats = params.get("export_formats") or ["nii.gz"]
    if isinstance(export_formats, str):
        export_formats = [export_formats]
    normalized_formats = []
    for fmt in export_formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        if fmt_clean not in SUPPORTED_EXPORT_FORMATS:
            return {
                "status": "error",
                "error": "Unsupported export format '{}'. Supported: {}".format(
                    fmt, ", ".join(SUPPORTED_EXPORT_FORMATS)
                ),
            }
        if fmt_clean not in normalized_formats:
            normalized_formats.append(fmt_clean)
    if not normalized_formats:
        normalized_formats = ["nii.gz"]

    # Convert each mask
    exported = []
    total_new = 0
    total_overwritten = 0
    total_unchanged = 0
    total_skipped_existing = 0
    final_export_shape = list(export_shape) if export_shape is not None else None

    for mask_info in manifest["masks"]:
        name = mask_info["original_name"]
        safe_name = str(mask_info.get("safe_name") or name)
        u8_filename = mask_info.get("u8_filename") or (safe_name + ".u8")
        u8_path = os.path.join(buffers_dir, u8_filename)
        canonical_source_path = str(source_mask_paths.get(name) or "").strip()
        mask_source = "mcs_mask_buffer"
        mcs_mask_bypassed = False
        source_mask_sha256 = ""

        if not canonical_source_path and not os.path.isfile(u8_path):
            exported.append({"name": name, "action": "skipped", "reason": "buffer not found"})
            continue

        format_paths = {
            fmt: os.path.join(seg_dir, safe_name + "." + fmt)
            for fmt in normalized_formats
        }
        # Report path: the historical .nii.gz when present, else the first
        # requested format so the entry points at a file that was written.
        if "nii.gz" in format_paths:
            nifti_path = format_paths["nii.gz"]
        else:
            nifti_path = format_paths[normalized_formats[0]]

        if canonical_source_path:
            # The saved project remains the source of case/mask selection and
            # provenance, but its potentially damaged voxel buffer is bypassed.
            nifti_array, source_mask_sha256 = _read_canonical_source_mask(
                canonical_source_path,
                tuple(export_shape or ()),
                export_affine,
            )
            mask_source = "canonical_source_mask"
            mcs_mask_bypassed = True
            mimics_foreground = int(np.count_nonzero(nifti_array))
            export_foreground = mimics_foreground
        else:
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
            mimics_foreground, export_foreground = (
                _validate_resampled_mask_foreground(
                    mimics_grid_array,
                    nifti_array,
                    "Exporting Mimics mask '{}' onto the original image grid".format(
                        name
                    ),
                )
            )
        if final_export_shape is None:
            final_export_shape = [int(value) for value in nifti_array.shape]

        # Compare per-format existing files first: a mask counts as unchanged
        # (or skipped-existing) only when every requested format matches.
        per_format_action = {}
        needs_write = False
        for fmt, target_path in format_paths.items():
            if os.path.isfile(target_path):
                existing, same_affine = _read_existing_mask(target_path, export_affine)
                if (
                    existing is not None
                    and existing.shape == nifti_array.shape
                    and np.array_equal(existing, nifti_array)
                    and same_affine
                ):
                    per_format_action[fmt] = "unchanged"
                    continue
                if not overwrite_existing:
                    per_format_action[fmt] = "skipped_existing"
                    continue
                per_format_action[fmt] = "overwritten"
                needs_write = True
            else:
                per_format_action[fmt] = "new"
                needs_write = True
        if all(action == "unchanged" for action in per_format_action.values()):
            total_unchanged += 1
            exported.append({
                "name": name,
                "action": "unchanged",
                "path": nifti_path,
                "paths": dict(format_paths),
                "per_format_action": per_format_action,
                "source_foreground_voxels": mimics_foreground,
                "foreground_voxels": export_foreground,
                "mask_source": mask_source,
                "source_mask_path": canonical_source_path,
                "source_mask_sha256": source_mask_sha256,
                "mcs_mask_bypassed": mcs_mask_bypassed,
            })
            continue
        if not needs_write:
            # Every requested format exists but differs, and overwriting is off.
            total_skipped_existing += 1
            exported.append({
                "name": name,
                "action": "skipped_existing",
                "path": nifti_path,
                "paths": dict(format_paths),
                "per_format_action": per_format_action,
                "reason": "custom export destinations do not overwrite existing files",
            })
            continue

        # Overall action for reporting: the highest-priority write performed.
        if "new" in per_format_action.values():
            action = "new"
        elif "overwritten" in per_format_action.values():
            action = "overwritten"
        else:
            action = "unchanged"

        for fmt, target_path in format_paths.items():
            if per_format_action.get(fmt) == "unchanged":
                continue
            write_mask_file(nifti_array, export_affine, target_path)

        if action == "new":
            total_new += 1
        elif action == "overwritten":
            total_overwritten += 1
        exported.append({
            "name": name,
            "action": action,
            "path": nifti_path,
            "paths": dict(format_paths),
            "per_format_action": per_format_action,
            "mask_source": mask_source,
            "source_mask_path": canonical_source_path,
            "source_mask_sha256": source_mask_sha256,
            "mcs_mask_bypassed": mcs_mask_bypassed,
        })
        exported[-1]["source_foreground_voxels"] = mimics_foreground
        exported[-1]["foreground_voxels"] = export_foreground

    return {
        "status": "ok",
        "exported": exported,
        "total_new": total_new,
        "total_overwritten": total_overwritten,
        "total_unchanged": total_unchanged,
        "total_skipped_existing": total_skipped_existing,
        "export_space": export_space,
        "degraded_export": degraded_export,
        "export_formats": list(normalized_formats),
        "mimics_voxel_to_ras_matrix_source": mimics_affine_source,
        "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
        "export_voxel_to_ras_matrix_source": export_affine_source,
        "export_voxel_to_ras_matrix": export_affine.tolist(),
        "output_seg_dir": seg_dir,
        "export_shape": final_export_shape or [],
        "source_mask_paths": source_mask_paths,
        "mcs_mask_bypassed": bool(source_mask_paths),
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
    elif is_medical_image_file(image_path):
        image_affine = get_image_affine(image_path)
        image_shape = get_image_shape(image_path)
    else:
        return {"status": "error", "error": "image path is neither DICOM folder nor NIfTI file: {}".format(image_path)}

    mask, mask_affine = read_nifti_mask_with_affine(mask_path)
    image_grid_mask = resample_mask_to_image_grid(
        mask, mask_affine, image_shape, image_affine,
        allow_voxel_aligned_fallback=not _mask_file_declares_spatial_geometry(mask_path),
    )
    source_foreground, image_foreground = _validate_resampled_mask_foreground(
        mask,
        image_grid_mask,
        "Applying prediction '{}' to the active Mimics grid".format(mask_path),
    )
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
        "source_foreground_voxels": source_foreground,
        "foreground_voxels": image_foreground,
        "target_voxel_to_ras_matrix": image_affine.astype(float).tolist(),
        "buffer_axes": list(axes),
        "buffer_flips": list(flips),
    }


# -- Discover cases (for batch import) --------------------------------

def do_read_nifti_ras_affine(params: dict) -> dict:
    """Return the on-disk NIfTI shape and nibabel RAS affine for a file path.

    Used by the Mimics-side metadata repair entry so that the nibabel read
    happens in the external Python (which has nibabel/numpy), not in Mimics'
    embedded interpreter.
    """
    image_path = params.get("image_path", "")
    if not image_path:
        return {"status": "error", "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"status": "error", "error": "image file not found: {}".format(image_path)}
    try:
        shape, affine = get_nifti_disk_geometry(image_path)
    except Exception as exc:
        return {"status": "error", "error": "could not read NIfTI geometry: {}".format(exc)}
    return {
        "status": "ok",
        "image_path": str(Path(image_path).resolve()),
        "shape": [int(v) for v in shape],
        "voxel_to_ras_matrix": np.asarray(affine, dtype=float).tolist(),
    }


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


_DISCOVER_VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".nrrd.gz")
_DISCOVER_MASK_SUFFIXES = (".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz", ".nii", ".mha", ".mhd", ".nrrd")


def _mask_selection_names(value):
    """Return None for all, an empty set for none, or normalized mask names."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return None
        if text.lower() in ("none", "no", "off"):
            return set()
        values = text.split(",")
    else:
        values = list(value)
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def _volume_stem(filename: str) -> str:
    lower = filename.lower()
    for suffix in _DISCOVER_MASK_SUFFIXES:
        if lower.endswith(suffix):
            return filename[:-len(suffix)] or "mask"
    return Path(filename).stem or "mask"

def do_discover(params: dict) -> dict:
    """Discover TS-like cases in a dataset root directory.

    This runs in the bridge (nninteractive_env) so it doesn't block Mimics GUI.
    Returns list of case_info dicts suitable for do_prepare.
    """
    ts_root = params["ts_root"]
    cases_filter = params.get("cases_filter")
    raw_mask_selection = params.get("mask_selection", "all")
    selected_masks = _mask_selection_names(raw_mask_selection)
    if selected_masks is None:
        mask_mode = "all"
    elif selected_masks:
        mask_mode = "named"
    else:
        mask_mode = "none"

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
        for img_name in (
            "ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii",
            "ct.mhd", "mri.mhd", "ct.mha", "mri.mha",
            "ct.nrrd", "mri.nrrd", "ct.nrrd.gz", "mri.nrrd.gz",
        ):
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
                if fname.lower().endswith(_DISCOVER_MASK_SUFFIXES):
                    image_path = os.path.join(case_dir, fname)
                    image_type = "medical_image"
                    break

        if image_path is None:
            continue  # no image found, skip

        # Find masks
        masks = []
        seg_dir = os.path.join(case_dir, "segmentations")
        if selected_masks != set() and os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                if fname.lower().endswith(_DISCOVER_VOLUME_SUFFIXES):
                    organ = _volume_stem(fname)
                    if selected_masks is not None and organ.casefold() not in selected_masks:
                        continue
                    masks.append({"name": organ, "path": os.path.join(seg_dir, fname)})

        cases.append({
            "case_id": name,
            "image": image_path,
            "image_type": image_type,
            "masks": masks,
            "case_dir": case_dir,
        })

    mask_count = sum(len(case.get("masks", [])) for case in cases)
    return {
        "status": "ok",
        "cases": cases,
        "count": len(cases),
        "mask_selection": "all" if selected_masks is None else sorted(selected_masks),
        "mask_mode": mask_mode,
        "mask_count": mask_count,
        "cases_without_selected_masks": sum(1 for case in cases if not case.get("masks")),
    }


# -- Main ---------------------------------------------------------------

def do_prepare_source_fastpath(params: dict) -> dict:
    """Build source NIfTI cache on-demand for nnInteractive fast path."""
    image_path = params.get("image_path", "")
    source_nifti_out = params.get("source_nifti_out", "")
    if not image_path or not is_medical_image_file(image_path):
        return {
            "status": "error",
            "error": "image_path must be a supported 3D medical image file",
        }
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
        elif action == "read_nifti_ras_affine":
            result = do_read_nifti_ras_affine(params)
        else:
            result = {"status": "error", "error": "unknown action: {}".format(action)}
    except Exception as e:
        import traceback
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Keep the bridge protocol ASCII-safe. Mimics 21's Python 3.5 child
    # process may expose stdout using the active Windows code page; emitting
    # non-ASCII paths directly would make the UTF-8 decoder in call_bridge()
    # reject an otherwise valid JSON response.
    json.dump(result, sys.stdout, ensure_ascii=True)


if __name__ == "__main__":
    main()
