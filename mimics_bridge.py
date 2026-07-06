# -*- coding: utf-8 -*-
"""External Python bridge for dataset <-> Mimics conversion.

Runs in the nninteractive_env Python (3.10+ with numpy, nibabel, pydicom).
Called by mimics_import.py / mimics_export.py inside Mimics via subprocess.

Protocol: JSON stdin -> JSON stdout.

Actions:
    "prepare" - NIfTI image -> derived DICOM, NIfTI masks -> .u8 buffers
    "convert" - .u8 buffers -> NIfTI files (inverse buffer mapping)
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
    return p.name.endswith(".nii") or p.name.endswith(".nii.gz")


# -- NIfTI -> derived DICOM --------------------------------------------

def nifti_to_derived_dicom(nifti_path: str, dicom_out: str, case_id: str = "case") -> dict:
    """Convert NIfTI image to derived DICOM series for Mimics import."""
    import nibabel as nib
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage

    img = nib.load(nifti_path)
    array = np.asanyarray(img.dataobj)
    if array.ndim != 3:
        raise ValueError("NIfTI image must be 3D: {} shape={}".format(nifti_path, array.shape))

    # Normalize qform/sform codes before reading the affine so the DICOM
    # geometry is derived from a consistent header.  See _normalize_nifti_affine.
    affine_ras = _normalize_nifti_affine(img)
    affine_lps = RAS_TO_LPS @ affine_ras
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
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = "Derived from NIfTI"
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

    return {
        "shape": [int(dicom_columns), int(dicom_rows), int(num_slices)],
        "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "origin": [float(v) for v in origin],
        "direction": [float(v) for v in direction_matrix.flatten()],
        "series_uid": str(series_uid),
        "dicom_folder": str(out_dir),
    }


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


# -- NIfTI mask -> .u8 buffer -----------------------------------------

def read_nifti_mask(path: str) -> np.ndarray:
    import nibabel as nib
    img = nib.load(path)
    array = np.asanyarray(img.dataobj)
    if array.ndim != 3:
        raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
    return (array != 0).astype(np.uint8)


def read_nifti_mask_with_affine(path: str) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    img = nib.load(path)
    array = np.asanyarray(img.dataobj)
    if array.ndim != 3:
        raise ValueError("mask must be 3D: {} shape={}".format(path, array.shape))
    affine = _normalize_nifti_affine(img)
    return (array != 0).astype(np.uint8), affine


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


# -- Affine from NIfTI or DICOM ----------------------------------------

def get_image_affine(image_path: str) -> np.ndarray:
    import nibabel as nib
    if not Path(image_path).is_file():
        raise ValueError("image file not found: {}".format(image_path))
    img = nib.load(image_path)
    return _normalize_nifti_affine(img)


def get_image_shape(image_path: str) -> tuple[int, int, int]:
    import nibabel as nib
    if not Path(image_path).is_file():
        raise ValueError("image file not found: {}".format(image_path))
    img = nib.load(image_path)
    shape = tuple(int(value) for value in img.shape[:3])
    if len(shape) != 3:
        raise ValueError("image must be 3D: {} shape={}".format(image_path, img.shape))
    return shape


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


def find_affine_in_case_dir(case_dir: str) -> np.ndarray | None:
    """Find affine from image in a TS-like case directory."""
    d = Path(case_dir)
    for img_name in ["ct.nii.gz", "mri.nii.gz"]:
        candidate = d / img_name
        if candidate.is_file():
            return get_image_affine(str(candidate))
    dicom_dir = d / "dicom"
    if dicom_dir.is_dir():
        return get_image_affine_from_dicom(str(dicom_dir))
    for nii_file in sorted(d.glob("*.nii.gz")):
        if "segmentations" not in str(nii_file.parent):
            return get_image_affine(str(nii_file))
    return None


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
        info = nifti_to_derived_dicom(image_path, dicom_out, case_id=params.get("case_id", "case"))
        dicom_folder = info["dicom_folder"]
        image_affine = get_image_affine(image_path)
        source_image_kind = "nifti"
        source_image_index_space = "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1"
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

    # Both matrices are expressed in RAS world coordinates. The current import
    # path preserves voxel index order, but persisting both affines lets external
    # workers resample a source image into the exact Mimics voxel grid if a
    # future import path changes orientation, spacing or origin.
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
        "source_image_path": str(Path(image_path).resolve()),
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

    # Get affine
    affine = find_affine_in_case_dir(case_dir)
    if affine is None:
        return {"status": "error", "error": "no image found in case_dir for affine: {}".format(case_dir)}

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
        nifti_array = inverse_buffer_mapping(array, axes, flips)

        # If file already exists, compare content; skip write if unchanged.
        if os.path.isfile(nifti_path):
            import nibabel as nib
            existing = np.asanyarray(nib.load(nifti_path).dataobj)
            if existing.shape == nifti_array.shape and np.array_equal(existing, nifti_array):
                total_unchanged += 1
                exported.append({"name": name, "action": "unchanged", "path": nifti_path})
                continue
            action = "overwritten"
        else:
            action = "new"

        write_mask_nifti(nifti_array, affine, nifti_path)

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
    }


def do_mask_to_buffer(params: dict) -> dict:
    """Convert a NIfTI mask into a Mimics-shaped .u8 buffer for foreground apply."""
    image_path = params["image_path"]
    mask_path = params["mask_path"]
    output_path = params["output_path"]
    axes = params.get("axes", DEFAULT_MIMICS_BUFFER_AXES)
    flips = params.get("flips", DEFAULT_MIMICS_BUFFER_FLIPS)

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
        elif action == "mask_to_buffer":
            result = do_mask_to_buffer(params)
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
