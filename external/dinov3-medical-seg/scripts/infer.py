"""Run one DINOv3 medical segmentation checkpoint on a NIfTI volume.

Each training pipeline selects its matching inference path. The frozen-feature
slice method preserves the native source grid; volumetric methods use their
canonicalized model grid. Both restore predictions to the input NIfTI grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial import (
    canonicalize,
    resample_image_to_spacing,
    restore_prediction_to_original,
    spacing_zyx,
)
from src.inference import (
    predict_array,
    predict_cached_feature_slices,
    predict_cached_feature_slice_statistics,
    predict_probability_statistics,
)
from src.data.frozen_feature_slices import (
    laterality_safe_in_plane_axis_zyx,
    prepare_native_case,
    restore_native_prediction,
    restore_native_scalar_array,
)
from src.data.input_contract import validate_input_contract
from src.models.segmentor import DINOv33DSegmentor
from src.prompt_proposals import propose_guided_points
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(uuid.uuid4().hex))
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Infer a binary mask from a NIfTI image")
    parser.add_argument("--config", required=True, help="Experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth or epoch checkpoint")
    parser.add_argument("--input", required=True, help="Input NIfTI image")
    parser.add_argument("--output", required=True, help="Output NIfTI label path")
    parser.add_argument(
        "--prompt-suggestions-output",
        help="Optional JSON path for DINO-guided nnInteractive point proposals",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu")
    args = parser.parse_args()

    config = load_config(args.config, {})
    validate_input_contract(config)
    device = torch.device(args.device) if args.device else get_device()
    original_image = nib.load(args.input)
    model = DINOv33DSegmentor(config).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    if str(config.get("training", {}).get("pipeline", "volume")) == "cached_slices":
        prepared = prepare_native_case(
            Path(args.input),
            slice_size=tuple(config.get("data", {}).get("img_size", [256, 256])),
            normalization=str(
                config.get("data", {}).get("slice_normalization")
                or "timeslice_casewise"
            ),
        )
        if args.prompt_suggestions_output:
            cached_inference = copy.deepcopy(config.get("inference", {}) or {})
            if not cached_inference.get("tta_axes"):
                # Native-grid cached slices are not canonicalized. Resolve the
                # source voxel axis most aligned with RAS left/right and mirror
                # the other in-plane axis instead.
                safe_axis = laterality_safe_in_plane_axis_zyx(
                    original_image.affine
                )
                cached_inference["tta_axes"] = [[safe_axis]]
            threshold = float(cached_inference.get("threshold", 0.5))
            statistics = predict_cached_feature_slice_statistics(
                model,
                prepared["images"],
                device,
                slice_batch_size=int(config.get("model", {}).get("slice_batch_size", 4)),
                threshold=threshold,
                tta_axes=cached_inference.get("tta_axes", []),
            )
            source_statistics = {
                name: restore_native_scalar_array(statistics[name], original_image)
                for name in ("mean", "variance", "agreement")
            }
            proposal = propose_guided_points(
                source_statistics["mean"],
                source_statistics["variance"],
                source_statistics["agreement"],
                original_image.affine,
                threshold=threshold,
            )
            proposal.update(
                {
                    "coordinate_system": "world_ras_mm",
                    "tta_member_count": int(statistics["member_count"]),
                    "tta_axes": cached_inference.get("tta_axes", []),
                    "proposal_grid": "source_image",
                    "source_shape_xyz": [int(value) for value in original_image.shape[:3]],
                    "source_voxel_to_ras_matrix": np.asarray(
                        original_image.affine, dtype=float
                    ).tolist(),
                }
            )
            _write_json_atomic(Path(args.prompt_suggestions_output), proposal)
            prediction_zyx = (
                source_statistics["mean"] >= threshold
            ).astype(np.int16, copy=False)
        else:
            prediction_zyx = predict_cached_feature_slices(
                model,
                prepared["images"],
                device,
                slice_batch_size=int(config.get("model", {}).get("slice_batch_size", 4)),
                tta_axes=(config.get("inference", {}) or {}).get("tta_axes", []),
            )
        if args.prompt_suggestions_output:
            restored_xyz = np.transpose(prediction_zyx, (2, 1, 0))
            header = original_image.header.copy()
            header.set_data_dtype(np.int16)
            restored = nib.Nifti1Image(
                restored_xyz.astype(np.int16, copy=False),
                original_image.affine,
                header=header,
            )
        else:
            restored = restore_native_prediction(prediction_zyx, original_image)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(restored, str(output_path))
        print("Input: {}".format(args.input))
        print("Model grid: native source voxel grid {}".format(original_image.shape))
        print(
            "Saved: {} (foreground voxels: {})".format(
                output_path,
                int(np.count_nonzero(np.asanyarray(restored.dataobj))),
            )
        )
        return

    canonical_image = canonicalize(original_image)
    model_grid = resample_image_to_spacing(
        canonical_image,
        config.get("data", {}).get("target_spacing"),
    )

    print("Input: {}".format(args.input))
    print("Model grid: shape={}, spacing_zyx={}".format(model_grid.shape, spacing_zyx(model_grid)))
    if args.prompt_suggestions_output:
        guided_config = copy.deepcopy(config)
        guided_inference = guided_config.setdefault("inference", {})
        if not guided_inference.get("tta_axes") and not guided_inference.get("scales"):
            # Canonical model arrays use ZYX; axis 2 is left/right in RAS and is
            # deliberately excluded. Original + Y mirror keeps the two-member
            # latency budget without swapping laterality-specific anatomy.
            guided_inference["tta_axes"] = [[1]]
        threshold = float(guided_inference.get("threshold", 0.5))
        statistics = predict_probability_statistics(
            model, model_grid, guided_config, device
        )
        proposal = propose_guided_points(
            statistics["mean"],
            statistics["variance"],
            statistics["agreement"],
            model_grid.affine,
            threshold=threshold,
        )
        proposal.update(
            {
                "coordinate_system": "world_ras_mm",
                "tta_member_count": int(statistics["member_count"]),
                "tta_axes": guided_inference.get("tta_axes", []),
                "tta_scales": guided_inference.get("scales", []),
                "proposal_grid": "dinov3_model_grid",
                "model_shape_xyz": [int(value) for value in model_grid.shape[:3]],
                "model_voxel_to_ras_matrix": np.asarray(
                    model_grid.affine, dtype=float
                ).tolist(),
            }
        )
        _write_json_atomic(Path(args.prompt_suggestions_output), proposal)
        prediction_zyx = (
            statistics["mean"] >= threshold
        ).astype(np.int16, copy=False)
    else:
        prediction_zyx = predict_array(model, model_grid, config, device)
    restored = restore_prediction_to_original(
        prediction_zyx,
        model_grid=model_grid,
        original_image=original_image,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(restored, str(output_path))
    print("Saved: {} (foreground voxels: {})".format(output_path, int(np.count_nonzero(prediction_zyx))))


if __name__ == "__main__":
    main()
