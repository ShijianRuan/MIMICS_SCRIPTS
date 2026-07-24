"""Shared model-grid inference with optional sliding-window and mirror TTA."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .data.dataset_3d import prepare_model_input
from .data.spatial import spacing_zyx, xyz_to_zyx


def predict_cached_feature_slices(
    model,
    images_zyx: np.ndarray,
    device,
    *,
    slice_batch_size: int = 4,
) -> np.ndarray:
    """Run the frozen-feature 2D path with verified four-way mirror TTA."""
    images = np.asarray(images_zyx, dtype=np.float32)
    if images.ndim != 3 or any(value <= 0 or value % 16 for value in images.shape[-2:]):
        raise RuntimeError(
            "Cached feature inference expects (Z,H,W) with H/W divisible by 16, got {}".format(
                images.shape
            )
        )
    predictions = []
    model.backbone.eval()
    model.decoder_3d.eval()
    batch_size = max(1, int(slice_batch_size))
    encoder_device = (
        torch.device("cpu")
        if getattr(model, "encoder_backend", "pytorch") == "onnx"
        else device
    )
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            raw = torch.from_numpy(images[start:start + batch_size, None]).to(
                device=encoder_device,
                dtype=torch.float32,
            )
            embeddings = model.backbone(raw.repeat(1, 3, 1, 1))[-1]
            if encoder_device != device:
                raw = raw.to(device=device)
                embeddings = embeddings.to(device=device)
            probabilities = []
            for dimensions in ((), (2,), (3,), (2, 3)):
                current_embeddings = (
                    torch.flip(embeddings, dims=dimensions).contiguous()
                    if dimensions
                    else embeddings
                )
                current_raw = (
                    torch.flip(raw, dims=dimensions).contiguous()
                    if dimensions
                    else raw
                )
                logits = model.decode_cached_slices(current_embeddings, current_raw)
                current = torch.softmax(logits, dim=1)
                if dimensions:
                    current = torch.flip(current, dims=dimensions)
                probabilities.append(current)
            prediction = torch.stack(probabilities, dim=0).mean(dim=0).argmax(dim=1)
            predictions.append(prediction.cpu().numpy().astype(np.int16, copy=False))
    return np.concatenate(predictions, axis=0)


def _predict_logits(model, data_zyx: np.ndarray, model_grid, config, device, input_scale: float = 1.0) -> torch.Tensor:
    """Predict logits and restore them to the supplied raw ZYX crop grid."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    base_size = tuple(int(value) for value in data_cfg.get("img_size", [224, 224]))
    patch_size = int(getattr(model, "patch_size", 1))
    scaled_size = tuple(
        max(patch_size, int(round((size * float(input_scale)) / patch_size)) * patch_size)
        for size in base_size
    )
    tensor = prepare_model_input(
        data_zyx,
        scaled_size,
        modality=data_cfg.get("modality", "other"),
        intensity=data_cfg.get("intensity", {}),
        channel_policy=model_cfg.get("channel_policy", "repeat"),
        slice_axis=model_cfg.get("slice_axis", "axial"),
    ).unsqueeze(0).to(device=device, dtype=torch.float32)
    spacing = torch.tensor(spacing_zyx(model_grid), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(tensor, spacing_zyx=spacing)
            return F.interpolate(logits, size=data_zyx.shape, mode="trilinear", align_corners=False)


def _parse_tta_axes(raw_axes: Iterable) -> list[tuple[int, ...]]:
    parsed = [()]
    for item in raw_axes or []:
        if not isinstance(item, (list, tuple)):
            raise ValueError("inference.tta_axes entries must be axis lists such as [1] or [1, 2]")
        axes = tuple(sorted(set(int(axis) for axis in item)))
        if any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("TTA axes use model ZYX indices 0, 1, 2")
        if axes and axes not in parsed:
            parsed.append(axes)
    return parsed


def _parse_scales(raw_scales: Iterable) -> list[float]:
    values = [1.0]
    for raw in raw_scales or []:
        value = float(raw)
        if value <= 0.0:
            raise ValueError("inference.scales values must be positive")
        if all(abs(value - existing) > 1e-6 for existing in values):
            values.append(value)
    return values


def _predict_with_tta(model, data_zyx, model_grid, config, device) -> torch.Tensor:
    tta_axes = _parse_tta_axes(config.get("inference", {}).get("tta_axes", []))
    scales = _parse_scales(config.get("inference", {}).get("scales", []))
    logits = []
    for scale in scales:
        for axes in tta_axes:
            input_data = np.flip(data_zyx, axis=axes).copy() if axes else data_zyx
            current = _predict_logits(model, input_data, model_grid, config, device, input_scale=scale)
            if axes:
                current = torch.flip(current, dims=[axis + 2 for axis in axes])
            logits.append(current)
    return torch.stack(logits, dim=0).mean(dim=0)


def _starts(length: int, size: int, overlap: float) -> list[int]:
    if size < 1:
        raise ValueError("sliding-window size must be positive")
    if size >= length:
        return [0]
    if not 0.0 <= overlap < 1.0:
        raise ValueError("sliding-window overlap must be in [0, 1)")
    stride = max(1, int(round(size * (1.0 - overlap))))
    values = list(range(0, max(1, length - size + 1), stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def _gaussian_weight(shape_zyx: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for length in shape_zyx:
        coordinate = np.linspace(-1.0, 1.0, int(length), dtype=np.float32)
        axes.append(np.exp(-0.5 * (coordinate / 0.35) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weight, 1e-3).astype(np.float32, copy=False)


def _extract_padded(data: np.ndarray, start: tuple[int, int, int], size: tuple[int, int, int]) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    """Extract a fixed-size crop, padding edge intensity outside a small volume."""
    shape = np.asarray(data.shape, dtype=int)
    requested_end = np.asarray(start, dtype=int) + np.asarray(size, dtype=int)
    pad_after = np.maximum(0, requested_end - shape)
    if np.any(pad_after):
        data = np.pad(data, tuple((0, int(value)) for value in pad_after), mode="edge")
    slices = tuple(slice(int(origin), int(origin + length)) for origin, length in zip(start, size))
    return data[slices], slices


def prediction_center_zyx(mask: np.ndarray) -> tuple[int, int, int] | None:
    """Return the centroid of a non-empty coarse prediction in model ZYX order."""
    coordinates = np.where(np.asarray(mask) > 0)
    if not coordinates[0].size:
        return None
    return tuple(int(round(float(np.mean(axis)))) for axis in coordinates)


def prediction_bbox_zyx(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """Return the half-open bounding box (z0,z1,y0,y1,x0,x1) of a coarse prediction.

    Half-open means ``z0 <= z < z1``. Returns ``None`` for an empty mask so the
    caller records a cascade recall failure instead of fabricating an ROI.
    """
    coordinates = np.where(np.asarray(mask) > 0)
    if not coordinates[0].size:
        return None
    z, y, x = coordinates
    return (
        int(z.min()), int(z.max()) + 1,
        int(y.min()), int(y.max()) + 1,
        int(x.min()), int(x.max()) + 1,
    )


def expand_bbox(
    bbox: tuple[int, int, int, int, int, int],
    margin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Grow a half-open bbox by a per-axis margin, clamped to ``[0, shape)``."""
    z0, z1, y0, y1, x0, x1 = bbox
    mz, my, mx = (int(value) for value in margin_zyx)
    sz, sy, sx = (int(value) for value in shape_zyx)
    return (
        max(0, z0 - mz), min(sz, z1 + mz),
        max(0, y0 - my), min(sy, y1 + my),
        max(0, x0 - mx), min(sx, x1 + mx),
    )


def predict_refinement_array(model, model_grid, config, device, center_zyx: tuple[int, int, int] | None,
                             search_size_zyx: tuple[int, int, int] | None = None):
    """Refine one coarse-predicted ROI without ever using a ground-truth crop.

    The returned mask is on the complete model grid. A missing coarse object
    remains an explicit empty prediction so the evaluation records cascade
    recall failures instead of silently falling back to a label-derived ROI.

    When ``search_size_zyx`` is given, the fine model tiles its training window
    (``data.patch.size_zyx``) across a search ROI of that size centred on the
    coarse centre, then thresholds the aggregated foreground probability. This
    decouples the small training window (high foreground fraction) from the
    coverage needed to tolerate coarse centre error (16-30mm for adrenal): a
    single centred small window would drop an offset target, but a tiled search
    ROI does not. When ``search_size_zyx`` is ``None`` the legacy single-window
    crop is used (backwards compatible).
    """
    if center_zyx is None:
        data = xyz_to_zyx(model_grid.get_fdata(dtype=np.float32))
        return np.zeros(data.shape, dtype=np.int16)
    patch = dict(config.get("data", {}).get("patch", {}))
    size = tuple(int(value) for value in patch.get("size_zyx", []))
    if len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("Fine-stage refinement requires data.patch.size_zyx")
    data = xyz_to_zyx(model_grid.get_fdata(dtype=np.float32))
    shape = np.asarray(data.shape, dtype=int)
    center = np.asarray(center_zyx, dtype=int)

    if search_size_zyx is not None:
        # Search ROI centred on the coarse centre, clamped to the volume.
        roi = np.asarray(search_size_zyx, dtype=int)
        roi = np.maximum(roi, np.asarray(size, dtype=int))  # never smaller than one window
        start = np.clip(center - roi // 2, 0, np.maximum(shape - roi, 0))
        end = np.minimum(shape, start + roi)
        region = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        sub_data = data[region]
        # Tile the training window across the search ROI and threshold the
        # aggregated foreground probability with the same rule as predict_array.
        foreground_probability = _sliding_window_logits(model, sub_data, model_grid, config, device)
        local = postprocess_foreground(
            foreground_probability,
            threshold=float(config.get("inference", {}).get("threshold", 0.5)),
            keep_largest_component=bool(config.get("inference", {}).get("keep_largest_component", False)),
        )
        result = np.zeros(shape, dtype=np.int16)
        result[region] = local
        return result

    size_array = np.asarray(size, dtype=int)
    start = center - size_array // 2
    start = np.maximum(start, 0)
    start = np.minimum(start, np.maximum(shape - size_array, 0))
    crop, _ = _extract_padded(data, tuple(int(value) for value in start), size)
    logits = _predict_with_tta(model, crop, model_grid, config, device)
    local = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16, copy=False)
    result = np.zeros(shape, dtype=np.int16)
    valid = np.minimum(size_array, shape - start)
    region = tuple(slice(int(origin), int(origin + length)) for origin, length in zip(start, valid))
    result[region] = local[:valid[0], :valid[1], :valid[2]]
    return result


def _sliding_window_logits(model, data_zyx, model_grid, config, device) -> np.ndarray:
    patch = dict(config.get("data", {}).get("patch", {}))
    size = tuple(int(value) for value in patch.get("size_zyx", []))
    if len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("sliding-window inference requires data.patch.size_zyx")
    overlap = float(patch.get("inference_overlap", 0.5))
    shape = tuple(int(value) for value in data_zyx.shape)
    # Aggregate only foreground probability on CPU. This keeps all transient
    # DINO activations on the GPU and bounds GPU memory by one patch.
    probability_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    weight = _gaussian_weight(size)
    for z in _starts(shape[0], size[0], overlap):
        for y in _starts(shape[1], size[1], overlap):
            for x in _starts(shape[2], size[2], overlap):
                crop, _ = _extract_padded(data_zyx, (z, y, x), size)
                logits = _predict_with_tta(model, crop, model_grid, config, device)
                foreground = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                valid = (min(size[0], shape[0] - z), min(size[1], shape[1] - y), min(size[2], shape[2] - x))
                region = (slice(z, z + valid[0]), slice(y, y + valid[1]), slice(x, x + valid[2]))
                local_weight = weight[:valid[0], :valid[1], :valid[2]]
                probability_sum[region] += foreground[:valid[0], :valid[1], :valid[2]] * local_weight
                weight_sum[region] += local_weight
    return probability_sum / np.maximum(weight_sum, 1e-6)


def postprocess_foreground(
    foreground_probability: np.ndarray,
    *,
    threshold: float,
    keep_largest_component: bool,
) -> np.ndarray:
    """Threshold a foreground-probability volume and optionally keep the largest blob.

    Shared by ``predict_array`` and the inference-path diagnostics so both apply
    an identical decision rule to a probability map. Returns an ``int16`` mask on
    the same grid as ``foreground_probability``.
    """
    prediction = (np.asarray(foreground_probability) >= float(threshold)).astype(np.int16)
    if keep_largest_component and np.any(prediction):
        components, count = ndimage.label(prediction > 0)
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            prediction = (components == int(np.argmax(sizes))).astype(np.int16)
    return prediction


def predict_array(model, model_grid, config, device):
    """Return a native model-grid class-index prediction in ``(Z,Y,X)`` order."""
    data_zyx = xyz_to_zyx(model_grid.get_fdata(dtype=np.float32))
    full_shape = data_zyx.shape
    roi = dict(config.get("data", {}).get("roi", {}))
    roi_slices = None
    if roi.get("enabled", False):
        from .data.dataset_3d import MedicalVolumeDataset
        roi_slices = MedicalVolumeDataset._normalized_roi_slices(full_shape, roi.get("normalized_zyx"))
        data_zyx = data_zyx[roi_slices]
    patch = dict(config.get("data", {}).get("patch", {}))
    if bool(patch.get("inference_sliding_window", False)):
        foreground_probability = _sliding_window_logits(model, data_zyx, model_grid, config, device)
        prediction = postprocess_foreground(
            foreground_probability,
            threshold=float(config.get("inference", {}).get("threshold", 0.5)),
            keep_largest_component=bool(config.get("inference", {}).get("keep_largest_component", False)),
        )
    else:
        logits = _predict_with_tta(model, data_zyx, model_grid, config, device)
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16, copy=False)
        if bool(config.get("inference", {}).get("keep_largest_component", False)) and np.any(prediction):
            components, count = ndimage.label(prediction > 0)
            if count > 1:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                prediction = (components == int(np.argmax(sizes))).astype(np.int16)
    if roi_slices is not None:
        restored = np.zeros(full_shape, dtype=np.int16)
        restored[roi_slices] = prediction
        return restored
    return prediction
