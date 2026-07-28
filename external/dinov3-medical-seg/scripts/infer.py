"""Run one DINOv3 medical segmentation checkpoint on a NIfTI volume.

Each training pipeline selects its matching inference path. The frozen-feature
slice method preserves the native source grid; volumetric methods use their
canonicalized model grid. Both restore predictions to the input NIfTI grid.
"""

from __future__ import annotations

import argparse
import sys
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
from src.inference import predict_array, predict_cached_feature_slices
from src.data.frozen_feature_slices import prepare_native_case, restore_native_prediction
from src.data.input_contract import validate_input_contract
from src.models.segmentor import DINOv33DSegmentor
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Infer a binary mask from a NIfTI image")
    parser.add_argument("--config", required=True, help="Experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth or epoch checkpoint")
    parser.add_argument("--input", required=True, help="Input NIfTI image")
    parser.add_argument("--output", required=True, help="Output NIfTI label path")
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
        prediction_zyx = predict_cached_feature_slices(
            model,
            prepared["images"],
            device,
            slice_batch_size=int(config.get("model", {}).get("slice_batch_size", 4)),
        )
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
