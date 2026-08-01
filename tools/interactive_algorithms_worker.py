#!/usr/bin/env python3
"""External workers for ITK Snake and ScribblePrompt Mimics entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Cancelled(RuntimeError):
    pass


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(str(temporary), str(path))
            return
        except OSError:
            if attempt == 11:
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                try:
                    temporary.unlink()
                except OSError:
                    pass
                return
            time.sleep(0.03 * (attempt + 1))


class Job:
    def __init__(self, request_path: Path, request: dict[str, Any]) -> None:
        self.request_path = request_path
        self.request = request
        self.status_path = Path(request["status_path"])
        self.cancel_path = Path(request["cancel_path"])
        self.log_path = Path(request["log_path"])
        self.started = time.time()

    def cancelled(self) -> bool:
        if self.cancel_path.exists():
            return True
        parent_pid = int(self.request.get("parent_pid") or 0)
        if parent_pid > 0:
            try:
                from resource_locks import process_exists

                return not process_exists(parent_pid)
            except Exception:
                pass
        return False

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise Cancelled("The operation was stopped by the user.")

    def log(self, message: str) -> None:
        line = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def status(self, phase: str, progress: int, message: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": "mimics_interactive_algorithm_job.v1",
            "status": "running",
            "tool": self.request.get("tool"),
            "phase": phase,
            "progress_percent": max(0, min(100, int(progress))),
            "message": str(message),
            "pid": os.getpid(),
            "started_at_epoch": self.started,
            "updated_at_epoch": time.time(),
            "log_path": str(self.log_path),
        }
        payload.update(extra)
        _write_json_atomic(self.status_path, payload)
        self.log(message)

    def finish(self, status: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": "mimics_interactive_algorithm_job.v1",
            "status": status,
            "tool": self.request.get("tool"),
            "progress_percent": 100 if status == "completed" else 0,
            "pid": os.getpid(),
            "started_at_epoch": self.started,
            "updated_at_epoch": time.time(),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "log_path": str(self.log_path),
        }
        payload.update(extra)
        _write_json_atomic(self.status_path, payload)


def _shape(request: dict[str, Any]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in request["shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("A positive three-dimensional Mimics buffer shape is required.")
    return shape


def _raw_array(path: str, dtype: str, shape: tuple[int, ...], mode: str = "r") -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = Path(path).stat().st_size
    if actual != expected:
        raise RuntimeError("Raw buffer byte count mismatch for {}: {} != {}".format(path, actual, expected))
    return np.memmap(path, dtype=np.dtype(dtype), mode=mode, shape=shape, order="C")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spacing(request: dict[str, Any]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in request.get("spacing_mm") or (1.0, 1.0, 1.0))
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Mimics voxel spacing must contain three positive finite values.")
    return values


def _mask_bbox(mask: np.ndarray, margin_voxels: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        raise RuntimeError("ITK Snake requires a non-empty selected Mask.")
    minimum = foreground.min(axis=0)
    maximum = foreground.max(axis=0) + 1
    result = []
    for axis in range(3):
        start = max(0, int(minimum[axis]) - int(margin_voxels[axis]))
        stop = min(mask.shape[axis], int(maximum[axis]) + int(margin_voxels[axis]))
        result.append(slice(start, stop))
    return tuple(result)  # type: ignore[return-value]


def _normalise_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("The active image contains no finite voxel values.")
    low, high = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        raise RuntimeError("The active image has no usable intensity variation.")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def run_itk_snake(job: Job) -> dict[str, Any]:
    try:
        import SimpleITK as sitk
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("ITK Snake requires SimpleITK and scipy in nninteractive_env: {}".format(exc))

    request = job.request
    shape = _shape(request)
    spacing = _spacing(request)
    profile = dict(request.get("parameters") or {})
    max_distance = float(profile.get("maximum_displacement_mm", 5.0))
    sigma = float(profile.get("gradient_sigma_mm", 1.0))
    if max_distance <= 0 or sigma <= 0:
        raise ValueError("ITK Snake distance and gradient sigma must be positive.")

    job.status("reading_inputs", 5, "Reading the selected Mask and active image.")
    image = _raw_array(request["image_path"], request["image_dtype"], shape)
    original = np.asarray(_raw_array(request["mask_path"], "uint8", shape), dtype=bool)
    original_count = int(np.count_nonzero(original))
    if original_count <= 0:
        raise RuntimeError("ITK Snake requires a non-empty selected Mask.")
    job.check_cancelled()

    margin = tuple(max(2, int(math.ceil((max_distance + 3.0 * sigma) / value))) for value in spacing)
    crop_slices = _mask_bbox(original, margin)
    crop_image = _normalise_image(np.asarray(image[crop_slices]))
    crop_mask = np.asarray(original[crop_slices], dtype=np.uint8)

    # SimpleITK arrays are z,y,x. Mimics buffers are exported in x,y,z order.
    # The explicit transpose preserves the physical spacing assigned below.
    sitk_image = sitk.GetImageFromArray(np.transpose(crop_image, (2, 1, 0)))
    sitk_mask = sitk.GetImageFromArray(np.transpose(crop_mask, (2, 1, 0)))
    sitk_image.SetSpacing(spacing)
    sitk_mask.SetSpacing(spacing)

    job.status("edge_map", 18, "Building the 3D edge-potential image.")
    # This filter defines sigma in physical image-spacing units. SetSpacing above
    # therefore makes gradient_sigma_mm isotropic in millimetres even when the
    # voxel grid is anisotropic; dividing sigma by spacing here would be wrong.
    gradient = sitk.GradientMagnitudeRecursiveGaussian(sitk_image, sigma=sigma)
    gradient_values = sitk.GetArrayViewFromImage(gradient)
    positive = np.asarray(gradient_values)[np.asarray(gradient_values) > 0]
    scale = float(np.percentile(positive, 75.0)) if positive.size else 1.0
    scale = max(scale, 1.0e-6)
    edge_potential = sitk.Cast(1.0 / (1.0 + gradient / scale), sitk.sitkFloat32)
    initial_level_set = sitk.Cast(sitk.SignedMaurerDistanceMap(
        sitk_mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    ), sitk.sitkFloat32)

    level_set = sitk.GeodesicActiveContourLevelSetImageFilter()
    level_set.SetPropagationScaling(float(profile.get("propagation_scaling", 0.7)))
    level_set.SetCurvatureScaling(float(profile.get("curvature_scaling", 0.8)))
    level_set.SetAdvectionScaling(float(profile.get("advection_scaling", 1.0)))
    level_set.SetMaximumRMSError(0.01)
    maximum_iterations = max(1, int(profile.get("maximum_iterations", 80)))
    level_set.SetNumberOfIterations(maximum_iterations)

    def on_iteration() -> None:
        if job.cancelled():
            raise Cancelled("ITK Snake was stopped by the user.")
        iteration = int(level_set.GetElapsedIterations())
        if iteration == 1 or iteration % 5 == 0:
            progress = 25 + int(50.0 * min(iteration, maximum_iterations) / maximum_iterations)
            job.status(
                "evolving_contour",
                progress,
                "Evolving the 3D contour: iteration {} of {}.".format(iteration, maximum_iterations),
                iteration=iteration,
                maximum_iterations=maximum_iterations,
            )

    level_set.AddCommand(sitk.sitkIterationEvent, on_iteration)
    job.status("evolving_contour", 25, "Starting the 3D contour evolution.")
    try:
        evolved = level_set.Execute(initial_level_set, edge_potential)
    except RuntimeError:
        if job.cancelled():
            raise Cancelled("ITK Snake was stopped by the user.")
        raise
    job.check_cancelled()

    candidate_crop = np.transpose(sitk.GetArrayFromImage(evolved) <= 0.0, (2, 1, 0))
    original_crop = np.asarray(original[crop_slices], dtype=bool)
    inside_distance = distance_transform_edt(original_crop, sampling=spacing)
    outside_distance = distance_transform_edt(~original_crop, sampling=spacing)
    permitted_band = np.where(original_crop, inside_distance, outside_distance) <= max_distance
    constrained_crop = np.where(permitted_band, candidate_crop, original_crop)

    result = np.asarray(original, dtype=bool).copy()
    result[crop_slices] = constrained_crop
    result_count = int(np.count_nonzero(result))
    if result_count <= 0:
        raise RuntimeError("ITK Snake produced an empty Mask; the original Mask was left unchanged.")
    volume_change = abs(result_count - original_count) / float(original_count)
    maximum_change = float(profile.get("maximum_volume_change_ratio", 0.6))
    if volume_change > maximum_change:
        raise RuntimeError(
            "ITK Snake changed the Mask volume by {:.1%}, above the {:.1%} safety limit. "
            "Use a more conservative profile or improve the initial Mask.".format(volume_change, maximum_change)
        )

    job.status("writing_result", 88, "Validating and writing the ITK Snake result.")
    result_path = Path(request["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(result, dtype=np.uint8).tofile(str(result_path))
    return {
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path),
        "shape": list(shape),
        "foreground_voxels": result_count,
        "original_foreground_voxels": original_count,
        "volume_change_ratio": round(volume_change, 6),
        "elapsed_iterations": int(level_set.GetElapsedIterations()),
        "maximum_displacement_mm": max_distance,
    }


def _load_prompt_crop(spec: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in spec["shape"])
    return np.asarray(_raw_array(spec["path"], "uint8", shape), dtype=np.float32)


def _plane_from_volume(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.asarray(np.take(array, indices=index, axis=axis))


def _write_plane(volume: np.ndarray, plane: np.ndarray, axis: int, index: int) -> None:
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = int(index)
    volume[tuple(slices)] = plane


def _prompt_plane(specs: list[dict[str, Any]], shape: tuple[int, int, int]) -> tuple[int, int]:
    axis = None
    index = None
    for spec in specs:
        bbox = spec.get("bbox") or []
        if len(bbox) != 3:
            raise RuntimeError("A ScribblePrompt interaction is missing its 3D bounding box.")
        candidates = [value for value in range(3) if int(bbox[value][1]) - int(bbox[value][0]) == 1]
        if len(candidates) != 1:
            raise RuntimeError("Each ScribblePrompt scribble must lie on exactly one 2D image slice.")
        current_axis = candidates[0]
        current_index = int(bbox[current_axis][0])
        if current_index < 0 or current_index >= shape[current_axis]:
            raise RuntimeError("A ScribblePrompt interaction lies outside the active image.")
        if axis is None:
            axis, index = current_axis, current_index
        elif axis != current_axis or index != current_index:
            raise RuntimeError("All ScribblePrompt scribbles in one prediction must be drawn on the same slice.")
    if axis is None or index is None:
        raise RuntimeError("At least one ScribblePrompt scribble is required.")
    return int(axis), int(index)


def _scribble_planes(
    specs: list[dict[str, Any]], shape: tuple[int, int, int], axis: int, index: int
) -> np.ndarray:
    plane_shape = tuple(shape[value] for value in range(3) if value != axis)
    output = np.zeros((2,) + plane_shape, dtype=np.float32)
    for spec in specs:
        crop = _load_prompt_crop(spec)
        bbox = spec["bbox"]
        crop_plane = np.squeeze(crop, axis=axis)
        plane_slices = tuple(
            slice(int(bbox[value][0]), int(bbox[value][1]))
            for value in range(3)
            if value != axis
        )
        channel = 0 if bool(spec.get("include", True)) else 1
        output[(channel,) + plane_slices] = np.maximum(
            output[(channel,) + plane_slices], crop_plane
        )
    overlap = (output[0] > 0) & (output[1] > 0)
    if np.any(overlap):
        raise RuntimeError("Foreground and background ScribblePrompt strokes overlap.")
    return output


def _resolve_checkpoint(request: dict[str, Any]) -> Path:
    configured = str(request.get("checkpoint_path") or "").strip()
    if not configured:
        raise RuntimeError("The ScribblePrompt checkpoint path was not configured.")
    path = Path(configured)
    if not path.is_file():
        raise RuntimeError(
            "The official ScribblePrompt UNet checkpoint was not found: {}".format(path)
        )
    return path


def _load_scribbleprompt_model(checkpoint: Path, device: Any) -> Any:
    import torch

    source_root = ROOT / "external" / "ScribblePrompt"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from scribbleprompt.models.network import UNet

    model = UNet(in_channels=5, out_channels=1, features=[192, 192, 192, 192])
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("The ScribblePrompt checkpoint does not contain a model state dictionary.")
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def _gpu_lock(job: Job, device: Any) -> Any:
    if getattr(device, "type", str(device)) != "cuda":
        return None
    from resource_locks import FileResourceLock, default_resource_lock_dir

    lock = FileResourceLock(
        default_resource_lock_dir(ROOT) / "gpu.lock",
        resource="gpu",
        owner="ScribblePrompt inference",
    )
    last_notice = [0.0]

    def on_wait(holder: dict[str, Any]) -> None:
        try:
            from tools.fewshot_pipeline import (
                request_nninteractive_server_release_on_contention,
            )

            request_nninteractive_server_release_on_contention(holder)
        except Exception:
            pass
        now = time.time()
        if now - last_notice[0] >= 10.0:
            job.status(
                "waiting_for_gpu",
                8,
                "Waiting for the GPU held by {}. Mimics remains available.".format(
                    str((holder or {}).get("owner") or "another AI task")
                ),
            )
            last_notice[0] = now
    lock.acquire(
        wait_seconds=float(job.request.get("gpu_lock_timeout_seconds", 120.0)),
        poll_seconds=0.5,
        on_wait=on_wait,
        should_cancel=job.cancelled,
    )
    lock.update_pid(os.getpid(), command="interactive_algorithms_worker.py ScribblePrompt")
    return lock


def run_scribbleprompt(job: Job) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise RuntimeError("ScribblePrompt requires PyTorch in nninteractive_env: {}".format(exc))

    request = job.request
    shape = _shape(request)
    specs = list(request.get("scribbles") or [])
    axis, index = _prompt_plane(specs, shape)
    checkpoint = _resolve_checkpoint(request)
    device_name = str(request.get("device") or "cpu").lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ScribblePrompt was configured for CUDA, but CUDA is unavailable.")
    device = torch.device(device_name)
    lock = None
    try:
        job.status("reading_inputs", 8, "Reading the active image, selected Mask, and scribbles.")
        image = _raw_array(request["image_path"], request["image_dtype"], shape)
        base_mask = np.asarray(_raw_array(request["mask_path"], "uint8", shape), dtype=bool)
        image_plane = _normalise_image(_plane_from_volume(image, axis, index))
        base_plane = _plane_from_volume(base_mask, axis, index)
        scribbles = _scribble_planes(specs, shape, axis, index)
        if not np.any(scribbles[0]):
            raise RuntimeError("ScribblePrompt requires at least one foreground scribble.")
        job.check_cancelled()

        input_size = max(32, int(request.get("input_size", 128)))
        image_tensor = torch.from_numpy(image_plane.copy())[None, None].float()
        scribble_tensor = torch.from_numpy(scribbles.copy())[None].float()
        image_small = torch_functional.interpolate(
            image_tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        scribble_small = torch_functional.interpolate(
            scribble_tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)

        prior_source = "empty"
        prior_path = str(request.get("previous_logits_path") or "")
        previous_plane_matches = (
            int(request.get("previous_plane_axis", -1)) == axis
            and int(request.get("previous_plane_index", -1)) == index
        )
        if prior_path and Path(prior_path).is_file() and previous_plane_matches:
            previous_shape = tuple(int(value) for value in request.get("previous_logits_shape") or ())
            if previous_shape == image_plane.shape:
                previous = np.asarray(_raw_array(prior_path, "float32", previous_shape), dtype=np.float32)
                prior = torch.from_numpy(previous.copy())[None, None]
                prior_source = "previous_logits"
            else:
                prior = None
        else:
            prior = None
        if prior is None and np.any(base_plane):
            magnitude = float(request.get("prior_logit_magnitude", 6.0))
            values = np.where(base_plane, magnitude, -magnitude).astype(np.float32)
            prior = torch.from_numpy(values)[None, None]
            prior_source = "selected_mask"
        if prior is None:
            prior = torch.zeros_like(image_tensor)
        prior_small = torch_functional.interpolate(
            prior.float(), size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        box_channel = torch.zeros_like(image_small)
        model_input = torch.cat((image_small, box_channel, scribble_small, prior_small), dim=1)

        job.status("waiting_for_compute", 25, "Preparing the ScribblePrompt model on {}.".format(device))
        lock = _gpu_lock(job, device)
        job.check_cancelled()
        model = _load_scribbleprompt_model(checkpoint, device)
        job.status("predicting", 58, "Running ScribblePrompt on the selected 2D slice.")
        with torch.inference_mode():
            logits_small = model(model_input.to(device))
            logits = torch_functional.interpolate(
                logits_small,
                size=image_plane.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].float().cpu().numpy()
        job.check_cancelled()

        predicted_plane = logits > 0.0
        result = np.asarray(base_mask, dtype=bool).copy()
        _write_plane(result, predicted_plane, axis, index)
        result_path = Path(request["result_path"])
        logits_path = Path(request["logits_result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(result, dtype=np.uint8).tofile(str(result_path))
        np.asarray(logits, dtype=np.float32).tofile(str(logits_path))
        job.status("writing_result", 90, "Writing the ScribblePrompt result for the selected slice.")
        return {
            "result_path": str(result_path),
            "result_sha256": _sha256_file(result_path),
            "shape": list(shape),
            "foreground_voxels": int(np.count_nonzero(result)),
            "slice_foreground_pixels": int(np.count_nonzero(predicted_plane)),
            "plane_axis": axis,
            "plane_index": index,
            "logits_path": str(logits_path),
            "logits_shape": list(image_plane.shape),
            "prior_source": prior_source,
            "device": str(device),
        }
    finally:
        if lock is not None:
            lock.release()
        if device_name.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def run(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    job = Job(request_path, request)
    tool = str(request.get("tool") or "")
    try:
        if tool == "itk_snake":
            result = run_itk_snake(job)
        elif tool == "scribbleprompt":
            result = run_scribbleprompt(job)
        else:
            raise ValueError("Unknown interactive algorithm: {}".format(tool))
        job.finish("completed", result=result, message="{} completed.".format(request.get("display_name", tool)))
        return 0
    except Cancelled as exc:
        job.log(str(exc))
        job.finish("cancelled", message=str(exc))
        return 2
    except Exception as exc:
        job.log("ERROR: {}".format(exc))
        job.log(traceback.format_exc())
        job.finish("failed", message=str(exc), error=str(exc), traceback=traceback.format_exc())
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    return run(Path(args.request).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
