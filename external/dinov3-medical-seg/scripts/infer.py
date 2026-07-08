"""DINOv3 Medical Segmentation — Inference Entry Point."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import nibabel as nib
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

from src.utils.config import load_config
from src.utils.device import get_device
from src.models.segmentor import DINOv33DSegmentor
from src.utils.checkpoint import load_checkpoint
from src.data.dataset_3d import normalize_volume


def main():
    parser = argparse.ArgumentParser(description="Inference with DINOv3 3D segmentation")
    parser.add_argument("--config", type=str, required=True, help="Config from experiment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI volume")
    parser.add_argument("--output", type=str, required=True, help="Output NIfTI segmentation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    config = load_config(args.config)

    model = DINOv33DSegmentor(config)
    load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    model.eval()

    print(f"Loading: {args.input}")
    original_img = nib.load(args.input)
    img = nib.as_closest_canonical(original_img)
    data = normalize_volume(
        img.get_fdata(dtype=np.float32),
        config.get("data", {}).get("modality", "other"),
    )

    # Resize slices
    img_size = tuple(config["data"].get("img_size", [512, 512]))
    data_resized = np.zeros((data.shape[0], *img_size), dtype=np.float32)
    for d in range(data.shape[0]):
        slc = torch.from_numpy(data[d]).unsqueeze(0).unsqueeze(0)
        slc = torch.nn.functional.interpolate(slc, size=img_size, mode="bilinear")
        data_resized[d] = slc.numpy()

    volume = torch.from_numpy(data_resized).unsqueeze(0).unsqueeze(0).to(device)

    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu",
                            dtype=torch.float16, enabled=(device.type == "cuda")):
            pred = model(volume)

    pred = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # Resize back to original shape
    pred_resized = np.zeros(data.shape, dtype=np.int32)
    for d in range(pred_resized.shape[0]):
        slc = torch.from_numpy(pred[d]).unsqueeze(0).unsqueeze(0).float()
        slc = torch.nn.functional.interpolate(
            slc, size=data.shape[1:], mode="nearest"
        )
        pred_resized[d] = slc.numpy()

    # Map prediction from canonical model space back to the input image grid.
    original_ornt = io_orientation(original_img.affine)
    canonical_ornt = io_orientation(img.affine)
    back_to_original = ornt_transform(canonical_ornt, original_ornt)
    pred_original = apply_orientation(pred_resized, back_to_original).astype(np.int32, copy=False)
    if pred_original.shape != original_img.shape[:3]:
        raise RuntimeError(
            "prediction orientation restore failed: {} != {}".format(
                pred_original.shape,
                original_img.shape[:3],
            )
        )

    # Save
    print(f"Saving: {args.output}")
    out_img = nib.Nifti1Image(pred_original, original_img.affine, original_img.header)
    nib.save(out_img, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
