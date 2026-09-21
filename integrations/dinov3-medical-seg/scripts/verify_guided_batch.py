"""Batch closed-loop guided-points verification on the Totalsegmentator v2.0.1 set.

Runs the same DINOv3-point -> nnInteractive pipeline as
``verify_guided_closed_loop.py``, but on many cases from the raw v201 dataset
layout (``sXXXX/ct.nii.gz`` + ``sXXXX/segmentations/kidney_left.nii.gz``) to test
generalization beyond the fixed 3-case validation split.

Design for scale:
  - Two phases stay separate (DINOv3 for ALL cases, then nnInteractive for ALL)
    so each large model is loaded once. DINOv3 phase writes per-case mean/points
    to disk so it is memory-bounded (one case at a time) and resumable.
  - Resumable: a case is skipped if its result JSON already exists. Kill and
    rerun to continue. Phase 1 and Phase 2 each checkpoint a manifest.
  - Never touches the v201 dataset: all outputs go under the output dir.
  - Excludes the 5 training support cases by default (they would inflate scores).

Usage:
    python scripts/verify_guided_batch.py \
        --config config/kidney_0806/kidney_2d_featureunet_ce_5shot.yaml \
        --checkpoint experiments/.../best_model.pth \
        --v201-root /z/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201 \
        --v201-split val \
        --organ kidney_left \
        --model-dir E:/mimics_script_offline/nninteractive_env/models/nnInteractive_v1.0 \
        --output experiments/kidney_0806_guided_verify/batch_val/summary.json \
        --max-cases 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
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

TRAIN_EXCLUDE = {"s0657", "s0661", "s0668", "s0693", "s0749"}  # few-shot support set


# --------------------------- case discovery ---------------------------------

def _discover_v201_cases(v201_root, v201_split, organ, max_cases, extra_case_ids, min_target_voxels=0):
    """Return list of (case_id, image_path, label_path) from v201 raw layout.

    When min_target_voxels > 0, cases whose label has fewer foreground voxels
    (or is empty) are skipped. This excludes cases where the organ is absent
    (e.g. nephrectomy) or tiny, which are a different problem from segmentation
    quality on a present, normal-sized organ.
    """
    import numpy as np
    root = Path(v201_root)
    meta = root / "meta.csv"
    chosen_ids = []
    if meta.exists():
        with open(meta, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            split_map = {row["image_id"]: row.get("split") for row in reader}
    else:
        split_map = {}
    all_ids = sorted(d.name for d in root.glob("s*") if d.is_dir())
    if extra_case_ids:
        for cid in extra_case_ids:
            if cid in all_ids:
                chosen_ids.append(cid)
    else:
        for cid in all_ids:
            if split_map.get(cid) == v201_split:
                chosen_ids.append(cid)
    cases = []
    for cid in chosen_ids:
        ip = root / cid / "ct.nii.gz"
        lp = root / cid / "segmentations" / "{}.nii.gz".format(organ)
        if not (ip.exists() and lp.exists()):
            continue
        if min_target_voxels > 0:
            try:
                n = int(np.asanyarray(nib.load(str(lp)).dataobj).sum())
            except Exception:
                continue
            if n < min_target_voxels:
                continue
        cases.append((cid, ip, lp))
        if max_cases and len(cases) >= max_cases:
            break
    return cases


# --------------------------- DINOv3 phase -----------------------------------

def _dinov3_one(model, image_path, config, device):
    """Return (mean_xyz, dino_mask_xyz, points, affine). Bounded memory."""
    original = nib.load(str(image_path))
    affine = np.asarray(original.affine, dtype=float)
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
    mean_native = restore_native_scalar_array(stats["mean"], original)
    var_native = restore_native_scalar_array(stats["variance"], original)
    agree_native = restore_native_scalar_array(stats["agreement"], original)
    mean_xyz = np.transpose(mean_native, (2, 1, 0)).astype(np.float32, copy=False)
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
    return mean_xyz, dino_mask_xyz, points, affine


# --------------------------- arbitration ------------------------------------

def _overseg_rate(mask, dino_mean, low_prob=0.1):
    m = mask.astype(bool); total = int(m.sum())
    if total == 0:
        return 0.0
    return int((m & (dino_mean < low_prob)).sum()) / total


def _underseg_rate(mask, dino_mean, high_prob=0.7):
    m = mask.astype(bool); high = dino_mean >= high_prob
    total = int(high.sum())
    if total == 0:
        return 0.0
    return int((high & ~m).sum()) / total


def _read_target(session):
    buf = session.target_buffer
    if hasattr(buf, "cpu"):
        buf = buf.cpu().numpy()
    return np.asarray(buf).astype(bool)


def _open_loop(session, image_xyz, points):
    session.set_image(image_xyz[None, ...].astype(np.float32, copy=False))
    session.set_target_buffer(np.zeros(image_xyz.shape, dtype=np.uint8))
    for pt in points:
        session.add_point_interaction(pt["vox_xyz"], include_interaction=pt["include"], run_prediction=True)
    return _read_target(session)


def _closed_loop(session, image_xyz, points, dino_mean, *, log):
    from scipy import ndimage
    session.set_image(image_xyz[None, ...].astype(np.float32, copy=False))
    session.set_target_buffer(np.zeros(image_xyz.shape, dtype=np.uint8))
    trace = {"applied": [], "reverted": [], "corrections": []}

    bg = [p for p in points if not p["include"]]
    fg = [p for p in points if p["include"]]
    for pt in bg:
        session.add_point_interaction(pt["vox_xyz"], include_interaction=False, run_prediction=True)
        trace["applied"].append({"name": "BG", "vox": list(pt["vox_xyz"])})

    prev_overseg = _overseg_rate(_read_target(session), dino_mean)
    for idx, pt in enumerate(fg):
        session.add_point_interaction(pt["vox_xyz"], include_interaction=True, run_prediction=True)
        mask = _read_target(session)
        overseg = _overseg_rate(mask, dino_mean)
        if overseg > 0.25 and (overseg - prev_overseg) > 0.15:
            undone = session.undo()
            trace["reverted"].append({
                "vox": list(pt["vox_xyz"]), "overseg_before": round(prev_overseg, 3),
                "overseg_after": round(overseg, 3), "undo_ok": bool(undone),
            })
            log("      FG#{} reverted: overseg {:.2f}->{:.2f}".format(idx, prev_overseg, overseg))
        else:
            trace["applied"].append({"name": "FG#{}".format(idx), "vox": list(pt["vox_xyz"]), "overseg": round(overseg, 3)})
            prev_overseg = overseg

    # correction: oversegmented -> BG on largest spillover blob
    mask = _read_target(session)
    overseg = _overseg_rate(mask, dino_mean)
    if overseg > 0.20:
        spillover = mask & (dino_mean < 0.1)
        labels, n = ndimage.label(spillover)
        if n > 0:
            sizes = ndimage.sum(spillover, labels, range(1, n + 1))
            biggest = int(np.argmax(sizes)) + 1
            cx, cy, cz = ndimage.center_of_mass(spillover, labels, biggest)
            bg_vox = (int(round(cx)), int(round(cy)), int(round(cz)))
            session.add_point_interaction(bg_vox, include_interaction=False, run_prediction=True)
            new_overseg = _overseg_rate(_read_target(session), dino_mean)
            trace["corrections"].append({"kind": "bg_spillover", "vox": list(bg_vox),
                                         "overseg_before": round(overseg, 3), "overseg_after": round(new_overseg, 3)})
            log("      BG correction at {}: overseg {:.2f}->{:.2f}".format(bg_vox, overseg, new_overseg))

    # correction: undersegmented -> FG on largest missed high-prob blob
    mask = _read_target(session)
    underseg = _underseg_rate(mask, dino_mean)
    if underseg > 0.20:
        missed = (dino_mean > 0.7) & ~mask
        labels, n = ndimage.label(missed)
        if n > 0:
            sizes = ndimage.sum(missed, labels, range(1, n + 1))
            biggest = int(np.argmax(sizes)) + 1
            cx, cy, cz = ndimage.center_of_mass(missed, labels, biggest)
            fg_vox = (int(round(cx)), int(round(cy)), int(round(cz)))
            session.add_point_interaction(fg_vox, include_interaction=True, run_prediction=True)
            new_under = _underseg_rate(_read_target(session), dino_mean)
            trace["corrections"].append({"kind": "fg_miss", "vox": list(fg_vox),
                                         "underseg_before": round(underseg, 3), "underseg_after": round(new_under, 3)})
            log("      FG correction at {}: underseg {:.2f}->{:.2f}".format(fg_vox, underseg, new_under))

    return _read_target(session), trace


# --------------------------- main ------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description="Batch closed-loop guided-points verification on v201")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v201-root", required=True, help="Totalsegmentator v201 dataset root")
    parser.add_argument("--v201-split", default="val", help="meta.csv split to use: train/val/test")
    parser.add_argument("--organ", default="kidney_left")
    parser.add_argument("--case-ids", default=None, help="Override: comma-separated case IDs (ignore split)")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all in split")
    parser.add_argument("--min-target-voxels", type=int, default=0,
                        help="Skip cases whose label has fewer foreground voxels "
                             "(0=all; 5000 excludes absent/tiny-organ cases)")
    parser.add_argument("--model-dir", required=True, help="nnInteractive model folder")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-phase", default="auto", choices=["auto", "1", "2"],
                        help="Force start at phase 1 (DINOv3) or 2 (nnInteractive). auto resumes.")
    args = parser.parse_args()

    config = load_config(args.config, {})
    import torch
    device = get_device() if args.device == "auto" else torch.device(args.device)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "dinov3_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover cases.
    extra = [t.strip() for t in args.case_ids.split(",")] if args.case_ids else None
    cases = _discover_v201_cases(args.v201_root, args.v201_split, args.organ, args.max_cases, extra, args.min_target_voxels)
    # Exclude training support cases (never test on them).
    cases = [(c, i, l) for c, i, l in cases if c not in TRAIN_EXCLUDE]
    print("=== {} cases from v201 split={} (organ={}, excl train) ===".format(len(cases), args.v201_split, args.organ), flush=True)
    if not cases:
        print("No cases found."); return

    # ---------------- Phase 1: DINOv3 (one case at a time, cache to disk) ----
    p1_manifest = cache_dir / "phase1_manifest.json"
    done_p1 = set()
    if p1_manifest.exists():
        done_p1 = set(json.loads(p1_manifest.read_text()).get("done", []))

    force_p1 = args.start_phase == "1"
    skip_p1 = args.start_phase == "2"
    if force_p1:
        done_p1 = set()

    if not skip_p1:
        remaining = [c for c in cases if c[0] not in done_p1]
        if remaining:
            print("\n=== Phase 1: DINOv3 feature_unet2d ({} remaining) ===".format(len(remaining)), flush=True)
            dino_model = DINOv33DSegmentor(config).to(device)
            load_checkpoint(dino_model, args.checkpoint, device=device)
            dino_model.eval()
            for idx, (cid, ip, lp) in enumerate(remaining):
                cache_file = cache_dir / "{}.npz".format(cid)
                try:
                    t0 = time.time()
                    mean_xyz, dino_mask, points, affine = _dinov3_one(dino_model, ip, config, device)
                    np.savez_compressed(
                        cache_file,
                        mean_xyz=mean_xyz.astype(np.float32),
                        dino_mask=dino_mask.astype(np.uint8),
                        affine=affine.astype(np.float64),
                        points=json.dumps(points),
                    )
                    done_p1.add(cid)
                    p1_manifest.write_text(json.dumps({"done": sorted(done_p1)}))
                    fg = sum(1 for p in points if p["include"]); bg = sum(1 for p in points if not p["include"])
                    print("  [{}/{}] {} FG={} BG={} ({:.1f}s)".format(
                        idx + 1, len(remaining), cid, fg, bg, time.time() - t0), flush=True)
                except Exception as exc:
                    print("  [{}] DINOv3 ERROR: {}".format(cid, exc), flush=True)
                    # Record error so phase 2 can skip it.
                    (results_dir / "{}_error.json".format(cid)).write_text(
                        json.dumps({"case_id": cid, "phase": "dinov3", "error": str(exc)}))
            del dino_model
            gc.collect(); torch.cuda.empty_cache()
        else:
            print("\n=== Phase 1: all cases cached, skipping ===", flush=True)
    else:
        print("\n=== Phase 1: skipped (--start-phase 2) ===", flush=True)

    # ---------------- Phase 2: nnInteractive closed loop ---------------------
    print("\n=== Phase 2: nnInteractive v1.0 (open vs closed) ===", flush=True)
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    session = nnInteractiveInferenceSession(device=device, use_torch_compile=False, verbose=False)
    session.initialize_from_trained_model_folder(args.model_dir)

    rows = []
    for idx, (cid, ip, lp) in enumerate(cases):
        result_path = results_dir / "{}.json".format(cid)
        if result_path.exists():
            # Resume: load existing result.
            try:
                rows.append(json.loads(result_path.read_text()))
                print("  [{}/{}] {} (cached)".format(idx + 1, len(cases), cid), flush=True)
                continue
            except Exception:
                pass
        cache_file = cache_dir / "{}.npz".format(cid)
        if not cache_file.exists():
            print("  [{}/{}] {} SKIP (no DINOv3 cache)".format(idx + 1, len(cases), cid), flush=True)
            continue
        try:
            t0 = time.time()
            data = np.load(cache_file, allow_pickle=True)
            mean_xyz = data["mean_xyz"]
            dino_mask = data["dino_mask"].astype(bool)
            affine = data["affine"]
            points = json.loads(str(data["points"]))
            image_xyz = np.asanyarray(nib.load(str(ip)).dataobj).astype(np.float32, copy=False)
            label_xyz = np.asanyarray(nib.load(str(lp)).dataobj) > 0
            spacing = spacing_zyx(nib.load(str(ip)))

            def _score(mask_xyz):
                if mask_xyz.shape != label_xyz.shape:
                    return {"error": "shape {} != {}".format(mask_xyz.shape, label_xyz.shape)}
                m = binary_metrics(mask_xyz, label_xyz, spacing, surface_tolerance_mm=2.0)
                m["overseg_rate"] = round(_overseg_rate(mask_xyz, mean_xyz), 3)
                m["underseg_rate"] = round(_underseg_rate(mask_xyz, mean_xyz), 3)
                return m

            result = {"case_id": cid, "target_voxels": int(label_xyz.sum())}
            result["dinov3"] = _score(dino_mask)
            session.reset_interactions()
            try:
                ol = _open_loop(session, image_xyz, points)
                result["open_loop"] = _score(ol)
            except Exception as exc:
                result["open_loop"] = {"error": str(exc)}
            session.reset_interactions()
            try:
                cl, trace = _closed_loop(session, image_xyz, points, mean_xyz,
                                         log=lambda s: print(s, flush=True))
                result["closed_loop"] = _score(cl)
                result["closed_loop_trace"] = trace
            except Exception as exc:
                result["closed_loop"] = {"error": str(exc)}
            cl = result.get("closed_loop") or {}
            dv = result.get("dinov3") or {}
            ol = result.get("open_loop") or {}
            # final = best of {closed_loop, open_loop, dinov3} by dice (honest, no preference).
            cands = {"closed_loop": cl, "open_loop": ol, "dinov3": dv}
            best = max(cands, key=lambda k: cands[k].get("dice", -1.0) if "error" not in cands[k] else -1.0)
            result["final"] = best
            result["final_dice"] = cands[best].get("dice")
            result["elapsed_seconds"] = round(time.time() - t0, 1)

            for k in ("dinov3", "open_loop", "closed_loop"):
                r = result[k]
                if "error" in r:
                    print("    [{}] ERROR".format(k))
                else:
                    print("    [{:10s}] Dice={:.4f}".format(k, r["dice"]))
            print("  [{}/{}] {} final={} d={:.4f} ({:.0f}s)".format(
                idx + 1, len(cases), cid, result["final"], result["final_dice"], time.time() - t0), flush=True)
            result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
            rows.append(result)
        except Exception as exc:
            print("  [{}] PHASE2 ERROR: {}".format(cid, exc), flush=True)
            (results_dir / "{}.json".format(cid)).write_text(
                json.dumps({"case_id": cid, "error": str(exc)}, default=str))

    # ---------------- summary ----------------
    def agg(method, field="dice"):
        vals = [r[method][field] for r in rows
                if isinstance(r.get(method), dict) and r[method].get(field) is not None and "error" not in r[method]]
        return None if not vals else float(np.mean(vals))

    def median(method):
        vals = [r[method]["dice"] for r in rows
                if isinstance(r.get(method), dict) and r[method].get("dice") is not None and "error" not in r[method]]
        return None if not vals else float(np.median(vals))

    # GT-free routing (the deployable strategy): pick open vs closed by a
    # DINOv3-arbitrated quality score, with DINOv3-volume anomaly forcing open.
    # This is what can run in production; "final" below is the GT-cheating upper bound.
    def _quality(r, m):
        d = r.get(m, {})
        return float(d.get("overseg_rate", 1.0)) + float(d.get("underseg_rate", 1.0))

    def _route_q(r):
        pv = int(r.get("dinov3", {}).get("pred_voxels", 0))
        if pv < 5000 or pv > 100000:
            return "open_loop"
        qo = _quality(r, "open_loop")
        qc = _quality(r, "closed_loop")
        if qo < qc - 0.1:
            return "open_loop"
        if qc < qo - 0.1:
            return "closed_loop"
        return "closed_loop"  # tie -> closed_loop (higher median in practice)

    routed_dice = []
    for r in rows:
        m = _route_q(r)
        d = r.get(m, {})
        routed_dice.append(d.get("dice") if d.get("dice") is not None and "error" not in d else 0.0)
        r["route_q"] = m
        r["route_q_dice"] = d.get("dice")
    routed_vals = [v for v in routed_dice if v is not None]

    summary = {
        "n_cases": len(rows),
        "organ": args.organ,
        "v201_split": args.v201_split if not args.case_ids else "explicit",
        "by_method": {
            "dinov3": {"mean_dice": agg("dinov3"), "median_dice": median("dinov3")},
            "open_loop": {"mean_dice": agg("open_loop"), "median_dice": median("open_loop")},
            "closed_loop": {"mean_dice": agg("closed_loop"), "median_dice": median("closed_loop")},
        },
        "route_q": {  # GT-free, deployable
            "mean_dice": float(np.mean(routed_vals)) if routed_vals else None,
            "median_dice": float(np.median(routed_vals)) if routed_vals else None,
            "description": "pv<5000 or >100000 -> open_loop; else lower (overseg+underseg) wins, tie->closed_loop",
        },
        "final_mean_dice": float(np.mean([r["final_dice"] for r in rows if r.get("final_dice") is not None])),
        "final_median_dice": float(np.median([r["final_dice"] for r in rows if r.get("final_dice") is not None])),
        "per_case": rows,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n=== Summary ({} cases) ===".format(summary["n_cases"]))
    for m in ("dinov3", "open_loop", "closed_loop"):
        s = summary["by_method"][m]
        print("  {:11s}: mean={:.4f} median={:.4f}".format(m, s["mean_dice"], s["median_dice"]))
    rq = summary["route_q"]
    print("  route_q     : mean={:.4f} median={:.4f}  (GT-free, deployable)".format(rq["mean_dice"], rq["median_dice"]))
    print("  final (best): mean={:.4f} median={:.4f}  (GT upper bound)".format(summary["final_mean_dice"], summary["final_median_dice"]))
    print("Saved: {}".format(args.output))


if __name__ == "__main__":
    _main()
