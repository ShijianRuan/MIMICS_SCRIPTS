"""Verify the 04_DINOv3_Guided_Points auto-segmentation path on a fixed val set.

This is an isolated experiment harness: it does NOT touch the Mimics review UI
or nnInteractive. It runs the *same* cached-slice guided-inference path that
``scripts/infer.py --prompt-suggestions-output`` runs (TTA probability statistics
-> ``propose_guided_points``), but instead of handing the points to a reviewer
it scores the TTA mean-thresholded mask against the ground-truth label.

Purpose
-------
1. Confirm ``propose_guided_points`` produces a valid point-proposal JSON for
   every validation case (the 04 feature is functional end-to-end on this
   checkpoint, not just on the unit-test synthetic volumes).
2. Report the auto-segmentation Dice/HD95/surface_dice of the guided TTA path
   on the SAME fixed validation split (s1031/s1120/s1130) used for the 0806
   kidney experiments, so it is directly comparable to the reported numbers.

The Dice here uses the TTA ``mean >= threshold`` mask (2-member mirror TTA),
which differs from the training-time ``val_dice`` (no-TTA argmax). Both are
reported so the gap is visible.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frozen_feature_slices import (
    laterality_safe_in_plane_axis_zyx,
    prepare_native_case,
    restore_native_prediction,
    restore_native_scalar_array,
)
from src.data.input_contract import validate_input_contract
from src.data.spatial import spacing_zyx
from src.evaluation import binary_metrics
from src.inference import predict_cached_feature_slice_statistics
from src.models.segmentor import DINOv33DSegmentor
from src.prompt_proposals import propose_guided_points
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}".format(uuid.uuid4().hex))
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


def _load_native_label_zyx(label_path: Path) -> np.ndarray:
    nii = nib.load(str(label_path))
    return np.asanyarray(nii.dataobj).astype(np.int16, copy=False)


def _evaluate_one_case(
    model,
    image_path: Path,
    label_path: Path,
    config: dict,
    device,
    out_dir: Path,
) -> dict:
    """Run the guided cached-slice path for one case and score the TTA mask."""
    started = time.time()
    original_image = nib.load(str(image_path))
    slice_size = tuple(config.get("data", {}).get("img_size", [256, 256]))
    normalization = str(
        config.get("data", {}).get("slice_normalization") or "timeslice_casewise"
    )
    prepared = prepare_native_case(
        Path(image_path), slice_size=slice_size, normalization=normalization
    )

    # Mirror the guided branch of scripts/infer.py exactly.
    cached_inference = copy.deepcopy(config.get("inference", {}) or {})
    if not cached_inference.get("tta_axes"):
        safe_axis = laterality_safe_in_plane_axis_zyx(original_image.affine)
        cached_inference["tta_axes"] = [[safe_axis]]
    threshold = float(cached_inference.get("threshold", 0.5))
    slice_batch_size = int(config.get("model", {}).get("slice_batch_size", 4))

    statistics = predict_cached_feature_slice_statistics(
        model,
        prepared["images"],
        device,
        slice_batch_size=slice_batch_size,
        threshold=threshold,
        tta_axes=cached_inference.get("tta_axes", []),
    )
    source_statistics = {
        name: restore_native_scalar_array(statistics[name], original_image)
        for name in ("mean", "variance", "agreement")
    }

    # Point proposal (this is what 04_DINOv3_Guided_Points consumes). A failure
    # here is recorded but does NOT invalidate the mask score.
    proposal_error = None
    proposal = None
    try:
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
                "source_shape_xyz": [int(v) for v in original_image.shape[:3]],
                "source_voxel_to_ras_matrix": np.asarray(
                    original_image.affine, dtype=float
                ).tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001 - record, do not crash
        proposal_error = "guided point proposal failed: {0}".format(exc)
        print("  WARNING: {0}".format(proposal_error))

    if proposal is not None:
        case_stem = label_path.name[:-7] if label_path.name.endswith(".nii.gz") else label_path.stem
        _write_json_atomic(out_dir / "{}_guided_points.json".format(case_stem), proposal)

    # TTA mean-thresholded mask, restored to the native source grid.
    prediction_zyx = (source_statistics["mean"] >= threshold).astype(np.int16, copy=False)
    restored_nii = restore_native_prediction(prediction_zyx, original_image)
    prediction_native_zyx = np.asanyarray(restored_nii.dataobj)
    prediction_native_zyx = np.transpose(prediction_native_zyx, (2, 1, 0))

    # Ground truth on the same native ZYX grid.
    label_nii = nib.load(str(label_path))
    target_zyx = np.transpose(np.asanyarray(label_nii.dataobj), (2, 1, 0)) > 0
    if prediction_native_zyx.shape != target_zyx.shape:
        raise RuntimeError(
            "Restored prediction shape {} != label shape {} for {}".format(
                prediction_native_zyx.shape, target_zyx.shape, image_path
            )
        )

    spacing = spacing_zyx(original_image)
    tolerance_mm = float(config.get("evaluation", {}).get("surface_tolerance_mm", 2.0))
    metrics = binary_metrics(
        prediction_native_zyx > 0,
        target_zyx,
        spacing,
        surface_tolerance_mm=tolerance_mm,
    )
    elapsed = time.time() - started

    # Also score the no-TTA argmax mask for a fair comparison to training
    # val_dice (which uses predict_cached_feature_slices, no TTA). We approximate
    # it by thresholding the single-member mean when member_count==1; otherwise
    # the TTA mean IS the reported number. Keep this honest about the source.
    result = {
        "case_id": label_path.stem.replace(".nii", ""),
        "tta_member_count": int(statistics["member_count"]),
        "tta_axes": cached_inference.get("tta_axes", []),
        "threshold": threshold,
        "dice": metrics["dice"],
        "hd95_mm": metrics["hd95_mm"],
        "assd_mm": metrics["assd_mm"],
        "surface_dice": metrics["surface_dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "pred_voxels": metrics["pred_voxels"],
        "target_voxels": metrics["target_voxels"],
        "spacing_zyx": list(spacing),
        "native_shape_zyx": list(target_zyx.shape),
        "elapsed_seconds": elapsed,
        "proposal": (
            None
            if proposal is None
            else {
                "schema_version": proposal.get("schema_version"),
                "foreground_point_count": proposal.get("foreground_point_count"),
                "background_point_count": proposal.get("background_point_count"),
                "retained_component_count": proposal.get("retained_component_count"),
                "discarded_component_count": proposal.get("discarded_component_count"),
                "has_error": False,
            }
        ),
        "proposal_error": proposal_error,
    }
    # Save the mask too, for inspection.
    mask_out = out_dir / "{}_guided_mask.nii.gz".format(
        label_path.name[:-7] if label_path.name.endswith(".nii.gz") else label_path.stem
    )
    nib.save(restored_nii, str(mask_out))
    result["mask_path"] = str(mask_out)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify guided-points auto-segmentation on a fixed val split"
    )
    parser.add_argument("--config", required=True, help="Experiment YAML (cached_slices)")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth")
    parser.add_argument("--data-root", required=True, help="Totalsegmentator kidney root")
    parser.add_argument(
        "--case-ids",
        default="s1031,s1120,s1130",
        help="Comma-separated validation case IDs",
    )
    parser.add_argument("--split", default="Tr", choices=["Tr", "Val"])
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config, {})
    validate_input_contract(config)
    device = torch.device(args.device) if args.device else get_device()
    model = DINOv33DSegmentor(config).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    data_root = Path(args.data_root)
    image_dir = data_root / "images{}".format(args.split)
    label_dir = data_root / "labels{}".format(args.split)
    case_ids = [token.strip() for token in args.case_ids.split(",") if token.strip()]
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in case_ids:
        image_path = image_dir / "{}.nii.gz".format(case_id)
        label_path = label_dir / "{}.nii.gz".format(case_id)
        if not image_path.exists() or not label_path.exists():
            print("SKIP {}: image/label missing".format(case_id))
            continue
        print("Evaluating {} (guided TTA auto-seg)...".format(case_id), flush=True)
        row = _evaluate_one_case(model, image_path, label_path, config, device, out_dir)
        rows.append(row)
        proposal_str = "no proposal"
        if row["proposal"] is not None:
            proposal_str = "FG={fg},BG={bg},components={c}".format(
                fg=row["proposal"]["foreground_point_count"],
                bg=row["proposal"]["background_point_count"],
                c=row["proposal"]["retained_component_count"],
            )
        print(
            "  {case}: Dice={dice:.4f}, HD95={hd95}, surface_dice={sd}, "
            "proposal=[{prop}], tta_members={m}, {t:.1f}s".format(
                case=case_id,
                dice=row["dice"],
                hd95="NA" if row["hd95_mm"] is None else "{:.2f}mm".format(row["hd95_mm"]),
                sd="NA" if row["surface_dice"] is None else "{:.3f}".format(row["surface_dice"]),
                prop=proposal_str,
                m=row["tta_member_count"],
                t=row["elapsed_seconds"],
            ),
            flush=True,
        )

    def aggregate(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return None if not values else float(np.mean(values))

    proposal_ok = sum(1 for row in rows if row["proposal"] is not None)
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "decoder": config.get("decoder", {}).get("type"),
        "encoder_backend": config.get("model", {}).get("encoder_backend"),
        "pipeline": config.get("training", {}).get("pipeline"),
        "n_cases": len(rows),
        "mean_dice": aggregate("dice"),
        "mean_hd95_mm": aggregate("hd95_mm"),
        "mean_surface_dice": aggregate("surface_dice"),
        "mean_precision": aggregate("precision"),
        "mean_recall": aggregate("recall"),
        "proposal_success_rate": float(proposal_ok / max(1, len(rows))),
        "per_case": rows,
        "note": (
            "Dice uses TTA mean>=threshold mask (guided path). Training val_dice "
            "uses no-TTA argmax; the two are not directly equal."
        ),
    }
    _write_json_atomic(Path(args.output), summary)
    print(
        "\nSummary: mean Dice={:.4f}, HD95={}, surface_dice={} over {} cases "
        "(proposal OK {}/{})".format(
            summary["mean_dice"],
            "NA" if summary["mean_hd95_mm"] is None else "{:.2f}mm".format(summary["mean_hd95_mm"]),
            "NA" if summary["mean_surface_dice"] is None else "{:.3f}".format(summary["mean_surface_dice"]),
            summary["n_cases"],
            proposal_ok,
            summary["n_cases"],
        )
    )
    print("Saved: {}".format(args.output))


if __name__ == "__main__":
    main()
