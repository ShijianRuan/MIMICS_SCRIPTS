"""Evaluate segmentation predictions vs ground truth: per-case Dice + HD95.

Self-contained (only nibabel + scipy + numpy). Matches each prediction in
--pred to the same-named file in --gt, binarizes both at >0, and reports
per-case Dice + 95th-percentile Hausdorff distance (in mm, from the affine),
plus mean/median.

Usage:
    python scripts/evaluate.py --pred <pred_dir> --gt <gt_dir> --out eval.json
    python scripts/evaluate.py --pred preds --gt gt --cases s0001,s0002 --out eval.json
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def _surface_distances(pred, gt, spacing):
    """Boundary-to-boundary distances (mm) between binary pred and gt."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return None
    # boundary = volume XOR eroded volume
    def boundary(mask):
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=1)
        return mask ^ eroded
    pb, gb = boundary(pred), boundary(gt)
    # distance from each pred boundary voxel to nearest gt boundary, and vice versa
    dt_gt = distance_transform_edt(~gb, sampling=spacing)
    dt_pred = distance_transform_edt(~pb, sampling=spacing)
    d_pred_to_gt = dt_gt[pb]
    d_gt_to_pred = dt_pred[gb]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def _dice(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = (p & g).sum()
    s = p.sum() + g.sum()
    return float(2 * inter / s) if s > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="directory of predicted .nii.gz")
    ap.add_argument("--gt", required=True, help="directory of ground-truth .nii.gz")
    ap.add_argument("--cases", default="", help="comma-separated case ids to evaluate (default: all in --pred)")
    ap.add_argument("--out", default="", help="write JSON summary here")
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    if args.cases.strip():
        names = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        names = sorted(p.name for p in pred_dir.glob("*.nii.gz"))

    rows = []
    for name in names:
        if not name.endswith(".nii.gz"):
            name = name + ".nii.gz"
        pf, gf = pred_dir / name, gt_dir / name
        if not pf.exists() or not gf.exists():
            print(f"  SKIP {name} (missing pred or gt)")
            continue
        pn = nib.load(str(pf)); gn = nib.load(str(gf))
        pred = np.asanyarray(pn.dataobj) > 0
        gt = np.asanyarray(gn.dataobj) > 0
        # resample pred onto gt shape if needed (nnU-Net output should already match)
        if pred.shape != gt.shape:
            print(f"  SKIP {name} (shape mismatch {pred.shape} vs {gt.shape})")
            continue
        spacing = np.linalg.norm(gn.affine[:3, :3], axis=0)
        d = _dice(pred, gt)
        sd = _surface_distances(pred, gt, spacing)
        hd95 = float(np.percentile(sd, 95)) if sd is not None else float("nan")
        cid = name.replace(".nii.gz", "")
        rows.append({"case": cid, "dice": d, "hd95_mm": hd95})
        print(f"  {cid:12s} dice={d:.4f}  hd95={hd95:7.2f}mm")

    dices = [r["dice"] for r in rows]
    hd95s = [r["hd95_mm"] for r in rows if not np.isnan(r["hd95_mm"])]
    summary = {
        "n": len(rows),
        "mean_dice": float(np.mean(dices)) if dices else None,
        "median_dice": float(np.median(dices)) if dices else None,
        "mean_hd95_mm": float(np.mean(hd95s)) if hd95s else None,
        "fails": int(sum(1 for d in dices if d < 0.1)),
        "cases": rows,
    }
    print(f"\n=== {len(rows)} cases: mean Dice={summary['mean_dice']:.4f}  "
          f"median={summary['median_dice']:.4f}  mean HD95={summary['mean_hd95_mm']:.2f}mm  "
          f"fails(<0.1)={summary['fails']} ===")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
