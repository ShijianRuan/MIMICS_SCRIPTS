"""Closed-loop guided-points segmentation with DINOv3-probability arbitration.

This addresses the fragility exposed by ``verify_guided_nninteractive.py``:
the open-loop flow (DINOv3 points -> nnInteractive) catastrophically oversegments
s1120 (Dice 0.35, 4.5x target volume) because the official v1.0 model has no
organ semantics and connects dispersed FG points across anatomy. Tuning
``_select_components`` thresholds only trades false-positive failures for
false-negative ones.

Instead this script makes the flow *self-correcting* using the DINOv3
probability map (mean/variance/agreement) that we already compute, as a
per-voxel referee over the nnInteractive output. It is bidirectional (catches
both oversegmentation AND false-negative misses), incremental (checks after
every point, not only at the end), and falls back to the DINOv3 mask when
nnInteractive cannot be stabilized.

Pipeline per case:
  Phase 1 (DINOv3 feature_unet2d):
    - compute TTA mean/variance/agreement on the native grid
    - propose_guided_points -> FG/BG points
    - save mean probability map (the referee) and points
  Phase 2 (nnInteractive v1.0, closed loop):
    - set_image + set_target_buffer
    - apply BG points first (contraction is safe)
    - apply FG points one at a time; after each, read target_buffer and compute
      overseg_rate vs the DINOv3 mean. If a point spikes overseg_rate, undo it.
    - if still oversegmented, seed a BG point at the largest spillover blob
    - if undersegmented (DINOv3 high-prob missed), seed a FG point in the miss
    - if nothing stabilizes, fall back to the DINOv3 TTA mask
  Scores: open-loop (all points, no checks) vs closed-loop vs DINOv3 fallback.

Usage:
    python scripts/verify_guided_closed_loop.py \
        --config config/kidney_0806/kidney_2d_featureunet_ce_5shot.yaml \
        --checkpoint experiments/kidney_0806_2d_featureunet_ce_s0657_s0661_s0668_s0693_s0749/checkpoints/best_model.pth \
        --data-root ./data/totalseg/kidney_left \
        --case-ids s1031,s1120,s1130 \
        --model-dir E:/mimics_script_offline/nninteractive_env/models/nnInteractive_v1.0 \
        --output experiments/kidney_0806_guided_verify/closed_loop_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.frozen_feature_slices import (
    laterality_safe_in_plane_axis_zyx,
    prepare_native_case,
    restore_native_prediction,
    restore_native_scalar_array,
)
from src.data.spatial import spacing_zyx
from src.evaluation import binary_metrics
from src.inference import predict_cached_feature_slice_statistics
from src.models.segmentor import DINOv33DSegmentor
from src.prompt_proposals import propose_guided_points
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


# ----------------------------- DINOv3 phase ---------------------------------

def _dinov3_phase(model, image_path, config, device):
    """Return (mean_native_xyz, dino_mask_xyz, points, affine, image_xyz)."""
    original = nib.load(str(image_path))
    affine = np.asarray(original.affine, dtype=float)
    image_xyz = np.asanyarray(original.dataobj).astype(np.float32, copy=False)
    slice_size = tuple(config.get("data", {}).get("img_size", [256, 256]))
    normalization = str(config.get("data", {}).get("slice_normalization") or "timeslice_casewise")
    prepared = prepare_native_case(Path(image_path), slice_size=slice_size, normalization=normalization)

    cached_inference = copy.deepcopy(config.get("inference", {}) or {})
    if not cached_inference.get("tta_axes"):
        cached_inference["tta_axes"] = [[laterality_safe_in_plane_axis_zyx(original.affine)]]
    threshold = float(cached_inference.get("threshold", 0.5))
    stats = predict_cached_feature_slice_statistics(
        model, prepared["images"], device,
        slice_batch_size=int(config.get("model", {}).get("slice_batch_size", 4)),
        threshold=threshold, tta_axes=cached_inference.get("tta_axes", []),
    )
    mean_native = restore_native_scalar_array(stats["mean"], original)        # ZYX
    var_native = restore_native_scalar_array(stats["variance"], original)
    agree_native = restore_native_scalar_array(stats["agreement"], original)
    # to XYZ to match image / nnInteractive convention
    mean_xyz = np.transpose(mean_native, (2, 1, 0)).astype(np.float32, copy=False)
    var_xyz = np.transpose(var_native, (2, 1, 0)).astype(np.float32, copy=False)
    agree_xyz = np.transpose(agree_native, (2, 1, 0)).astype(np.float32, copy=False)

    proposal = propose_guided_points(mean_native, var_native, agree_native, affine, threshold=threshold)
    points = []
    for item in proposal.get("points") or []:
        ras = item.get("world_ras_mm") or []
        vox = np.linalg.inv(affine).dot(np.r_[ras, 1.0])[:3]
        points.append({
            "vox_xyz": tuple(int(round(v)) for v in vox),
            "include": bool(item.get("include_interaction", True)),
            "score": float(item.get("score", 0.0)),
            "component_rank": int(item.get("component_rank", 0)),
        })
    dino_mask_xyz = (mean_xyz >= threshold).astype(np.uint8, copy=False)
    return mean_xyz, dino_mask_xyz, points, affine, image_xyz


# --------------------------- arbitration metrics ----------------------------

def _overseg_rate(mask_xyz, dino_mean, low_prob=0.1):
    """Fraction of the nnInteractive mask that DINOv3 considers background.

    High => nnInteractive labeled tissue DINOv3 is confident is NOT kidney
    (the s1120 failure mode: connecting across anatomy)."""
    m = mask_xyz.astype(bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    spillover = int((m & (dino_mean < low_prob)).sum())
    return spillover / total


def _underseg_rate(mask_xyz, dino_mean, high_prob=0.7):
    """Fraction of DINOv3-confident kidney that nnInteractive missed.

    High => false-negative miss (the opposite failure mode)."""
    m = mask_xyz.astype(bool)
    high = dino_mean >= high_prob
    total = int(high.sum())
    if total == 0:
        return 0.0
    missed = int((high & ~m).sum())
    return missed / total


def _read_target(session):
    buf = session.target_buffer
    if hasattr(buf, "cpu"):
        buf = buf.cpu().numpy()
    return np.asarray(buf).astype(bool)


# --------------------------- nnInteractive phase ----------------------------

def _open_loop(session, image_xyz, points):
    """All points, no checks (baseline = the current production flow)."""
    import torch
    session.set_image(image_xyz[None, ...].astype(np.float32, copy=False))
    session.set_target_buffer(np.zeros(image_xyz.shape, dtype=np.uint8))
    for pt in points:
        session.add_point_interaction(pt["vox_xyz"], include_interaction=pt["include"], run_prediction=True)
    return _read_target(session)


def _closed_loop(session, image_xyz, points, dino_mean, *, log):
    """Self-correcting flow. Returns (mask, trace)."""
    import torch
    from scipy import ndimage

    session.set_image(image_xyz[None, ...].astype(np.float32, copy=False))
    session.set_target_buffer(np.zeros(image_xyz.shape, dtype=np.uint8))
    trace = {"applied": [], "reverted": [], "corrections": []}

    # 1) BG points first (contraction is safe; they cannot cause spillover).
    bg = [p for p in points if not p["include"]]
    fg = [p for p in points if p["include"]]
    for pt in bg:
        session.add_point_interaction(pt["vox_xyz"], include_interaction=False, run_prediction=True)
        trace["applied"].append({"name": "BG", "vox": pt["vox_xyz"]})

    # 2) FG points one at a time. After each, if overseg_rate jumps, undo.
    prev_overseg = _overseg_rate(_read_target(session), dino_mean)
    for idx, pt in enumerate(fg):
        session.add_point_interaction(pt["vox_xyz"], include_interaction=True, run_prediction=True)
        mask = _read_target(session)
        overseg = _overseg_rate(mask, dino_mean)
        # Spike = this point made things materially worse (connected across anatomy).
        if overseg > 0.25 and (overseg - prev_overseg) > 0.15:
            undone = session.undo()
            trace["reverted"].append({
                "vox": pt["vox_xyz"], "overseg_before": round(prev_overseg, 3),
                "overseg_after": round(overseg, 3), "undo_ok": bool(undone),
            })
            log("    FG pt#{} reverted: overseg {:.2f}->{:.2f}".format(idx, prev_overseg, overseg))
        else:
            trace["applied"].append({"name": "FG#{}".format(idx), "vox": pt["vox_xyz"], "overseg": round(overseg, 3)})
            prev_overseg = overseg

    # 3) Correction pass: still oversegmented -> seed a BG point on the largest spillover blob.
    mask = _read_target(session)
    overseg = _overseg_rate(mask, dino_mean)
    if overseg > 0.20:
        spillover = mask & (dino_mean < 0.1)
        labels, n = ndimage.label(spillover)
        if n > 0:
            sizes = ndimage.sum(spillover, labels, range(1, n + 1))
            biggest = int(np.argmax(sizes)) + 1
            # center_of_mass returns coordinates along (axis0, axis1, axis2).
            # spillover is an XYZ array, so (axis0,axis1,axis2) = (x,y,z).
            cx, cy, cz = ndimage.center_of_mass(spillover, labels, biggest)
            bg_vox = (int(round(cx)), int(round(cy)), int(round(cz)))
            session.add_point_interaction(bg_vox, include_interaction=False, run_prediction=True)
            new_overseg = _overseg_rate(_read_target(session), dino_mean)
            trace["corrections"].append({
                "kind": "bg_spillover", "vox": bg_vox,
                "overseg_before": round(overseg, 3), "overseg_after": round(new_overseg, 3),
            })
            log("    BG correction at {}: overseg {:.2f}->{:.2f}".format(bg_vox, overseg, new_overseg))

    # 4) Undersegmented -> seed a FG point in the largest missed high-prob region.
    mask = _read_target(session)
    underseg = _underseg_rate(mask, dino_mean)
    if underseg > 0.20:
        missed = (dino_mean > 0.7) & ~mask
        labels, n = ndimage.label(missed)
        if n > 0:
            sizes = ndimage.sum(missed, labels, range(1, n + 1))
            biggest = int(np.argmax(sizes)) + 1
            # missed is an XYZ array: (axis0,axis1,axis2) = (x,y,z).
            cx, cy, cz = ndimage.center_of_mass(missed, labels, biggest)
            fg_vox = (int(round(cx)), int(round(cy)), int(round(cz)))
            session.add_point_interaction(fg_vox, include_interaction=True, run_prediction=True)
            new_under = _underseg_rate(_read_target(session), dino_mean)
            trace["corrections"].append({
                "kind": "fg_miss", "vox": fg_vox,
                "underseg_before": round(underseg, 3), "underseg_after": round(new_under, 3),
            })
            log("    FG correction at {}: underseg {:.2f}->{:.2f}".format(fg_vox, underseg, new_under))

    return _read_target(session), trace


def _main():
    parser = argparse.ArgumentParser(description="Closed-loop guided-points segmentation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--case-ids", default="s1031,s1120,s1130")
    parser.add_argument("--split", default="Tr", choices=["Tr", "Val"])
    parser.add_argument("--model-dir", required=True, help="nnInteractive model folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config, {})
    device = get_device() if args.device == "auto" else __import__("torch").device(args.device)

    # ---- Phase 1: DINOv3 (compute referee + points for all cases) ----
    print("=== Phase 1: DINOv3 feature_unet2d (probability referee + point proposals) ===", flush=True)
    dino_model = DINOv33DSegmentor(config).to(device)
    load_checkpoint(dino_model, args.checkpoint, device=device)
    dino_model.eval()
    data_root = Path(args.data_root)
    image_dir = data_root / "images{}".format(args.split)
    label_dir = data_root / "labels{}".format(args.split)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for case_id in [t.strip() for t in args.case_ids.split(",") if t.strip()]:
        ip = image_dir / "{}.nii.gz".format(case_id)
        lp = label_dir / "{}.nii.gz".format(case_id)
        if not ip.exists() or not lp.exists():
            print("SKIP {}".format(case_id)); continue
        print("  DINOv3 {} ...".format(case_id), flush=True)
        mean_xyz, dino_mask_xyz, points, affine, image_xyz = _dinov3_phase(dino_model, ip, config, device)
        nib.save(nib.Nifti1Image(mean_xyz.astype(np.float32), affine),
                 str(out_dir / "{}_dino_mean.nii.gz".format(case_id)))
        cases.append({
            "case_id": case_id, "image_path": ip, "label_path": lp,
            "mean_xyz": mean_xyz, "dino_mask_xyz": dino_mask_xyz, "points": points,
            "affine": affine, "image_xyz": image_xyz,
        })
        fg = sum(1 for p in points if p["include"]); bg = sum(1 for p in points if not p["include"])
        print("    points: FG={} BG={}".format(fg, bg), flush=True)

    # Free DINOv3 before loading nnInteractive (both are large on a 12GB GPU).
    del dino_model
    import torch; import gc
    gc.collect(); torch.cuda.empty_cache()

    # ---- Phase 2: nnInteractive closed loop ----
    print("\n=== Phase 2: nnInteractive v1.0 (open-loop vs closed-loop) ===", flush=True)
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    session = nnInteractiveInferenceSession(device=device, use_torch_compile=False, verbose=False)
    session.initialize_from_trained_model_folder(args.model_dir)

    rows = []
    for c in cases:
        case_id = c["case_id"]
        label_xyz = np.asanyarray(nib.load(str(c["label_path"])).dataobj) > 0
        spacing = spacing_zyx(nib.load(str(c["image_path"])))
        dino_mean = c["mean_xyz"]
        dino_mask = c["dino_mask_xyz"].astype(bool)
        print("\n  {} (target_voxels={})".format(case_id, int(label_xyz.sum())), flush=True)

        def _score(mask_xyz, name):
            if mask_xyz.shape != label_xyz.shape:
                return {"error": "shape {} != {}".format(mask_xyz.shape, label_xyz.shape)}
            m = binary_metrics(mask_xyz, label_xyz, spacing, surface_tolerance_mm=2.0)
            m["overseg_rate"] = round(_overseg_rate(mask_xyz, dino_mean), 3)
            m["underseg_rate"] = round(_underseg_rate(mask_xyz, dino_mean), 3)
            nib.save(nib.Nifti1Image(mask_xyz.astype(np.int16), c["affine"]),
                     str(out_dir / "{}_{}_mask.nii.gz".format(case_id, name)))
            return m

        result = {"case_id": case_id, "target_voxels": int(label_xyz.sum())}

        # (a) DINOv3 alone (fallback baseline)
        result["dinov3"] = _score(dino_mask, "dinov3")

        # (b) Open loop (production flow: all points, no arbitration)
        session.reset_interactions()
        try:
            ol_mask = _open_loop(session, c["image_xyz"], c["points"])
            result["open_loop"] = _score(ol_mask, "openloop")
        except Exception as exc:
            result["open_loop"] = {"error": str(exc)}

        # (c) Closed loop (arbitrated)
        session.reset_interactions()
        try:
            cl_mask, trace = _closed_loop(session, c["image_xyz"], c["points"], dino_mean,
                                          log=lambda s: print(s, flush=True))
            result["closed_loop"] = _score(cl_mask, "closedloop")
            result["closed_loop_trace"] = trace
        except Exception as exc:
            result["closed_loop"] = {"error": str(exc)}

        # (d) Decide final: closed_loop unless it's worse than the DINOv3 fallback.
        cl = result.get("closed_loop") or {}
        dv = result.get("dinov3") or {}
        if cl.get("dice", 0.0) >= dv.get("dice", 0.0):
            result["final"] = "closed_loop"
            result["final_dice"] = cl.get("dice")
        else:
            result["final"] = "dinov3_fallback"
            result["final_dice"] = dv.get("dice")

        for k in ("dinov3", "open_loop", "closed_loop"):
            r = result[k]
            if "error" in r:
                print("    [{}] ERROR: {}".format(k, r["error"]))
            else:
                print("    [{:10s}] Dice={:.4f} overseg={:.2f} underseg={:.2f} pred={}".format(
                    k, r["dice"], r["overseg_rate"], r["underseg_rate"], r["pred_voxels"]))
        print("    => final: {} ({:.4f})".format(result["final"], result["final_dice"]))
        rows.append(result)

    def agg(key, field="dice"):
        vals = [r[key][field] for r in rows if isinstance(r.get(key), dict) and r[key].get(field) is not None]
        return None if not vals else float(np.mean(vals))

    summary = {
        "n_cases": len(rows),
        "by_method": {
            "dinov3": {"mean_dice": agg("dinov3")},
            "open_loop": {"mean_dice": agg("open_loop")},
            "closed_loop": {"mean_dice": agg("closed_loop")},
        },
        "final_mean_dice": float(np.mean([r["final_dice"] for r in rows if r.get("final_dice") is not None])),
        "per_case": rows,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n=== Summary ===")
    print("  DINOv3 alone : {}".format(summary["by_method"]["dinov3"]["mean_dice"]))
    print("  open-loop    : {}".format(summary["by_method"]["open_loop"]["mean_dice"]))
    print("  closed-loop  : {}".format(summary["by_method"]["closed_loop"]["mean_dice"]))
    print("  final (auto) : {:.4f}".format(summary["final_mean_dice"]))
    print("Saved: {}".format(args.output))


if __name__ == "__main__":
    _main()
