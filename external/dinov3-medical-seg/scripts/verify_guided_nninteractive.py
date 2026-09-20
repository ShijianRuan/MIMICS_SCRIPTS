"""Verify the nnInteractive half of 04_DINOv3_Guided_Points end-to-end (offline).

This completes the chain that ``verify_guided_points.py`` only half-covered:

    ① DINOv3 proposes points  ->  ② (skip Mimics review)  ->  ③ nnInteractive segments

Step ① is already done: ``experiments/kidney_0806_guided_verify/*_guided_points.json``
contain the RAS-mm point proposals from the feature_unet2d checkpoint. This
script drives step ③ directly with the nnInteractive Python session API — no
Mimics, no HTTP server, no review UI — feeding those points to the official
nnInteractive v1.0 model and scoring the resulting mask against the GT label.

Two prediction policies are run per case, matching what the bridge supports:
  - ``sequential``        : predict after every point (each point sees the prior
                            prediction through nnInteractive's prev-seg channel)
  - ``initial_empty_batch``: accumulate all points, predict once on the last

The official v1.0 model is prompt-based (dataset.json has only 'background' as a
fixed label; the target region is defined by include/exclude points), so it can
segment kidney_left from the DINOv3-proposed points without a kidney-specific
checkpoint.

Usage:
    python scripts/verify_guided_nninteractive.py \
        --points-dir experiments/kidney_0806_guided_verify \
        --data-root ./data/totalseg/kidney_left \
        --case-ids s1031,s1120,s1130 \
        --model-dir E:/mimics_script_offline/nninteractive_env/models/nnInteractive_v1.0 \
        --output experiments/kidney_0806_guided_verify/nninteractive_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial import spacing_zyx
from src.evaluation import binary_metrics


def _case_stem(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else name.rsplit(".", 1)[0]


def _ras_to_voxel_xyz(ras_mm, affine: np.ndarray) -> tuple[int, int, int]:
    """World RAS mm -> image voxel XYZ (the space nnInteractive uses)."""
    homogeneous = np.array([float(ras_mm[0]), float(ras_mm[1]), float(ras_mm[2]), 1.0])
    vox = np.linalg.inv(affine).dot(homogeneous)[:3]
    return tuple(int(round(v)) for v in vox)


def _load_points(points_json: Path) -> list[dict]:
    """Read a guided-points JSON; keep FG first then BG (bridge order)."""
    data = json.loads(points_json.read_text(encoding="utf-8"))
    if data.get("schema_version") != "dinov3_guided_points.v1":
        raise RuntimeError("Unexpected schema: {}".format(data.get("schema_version")))
    points = []
    for item in data.get("points") or []:
        points.append(
            {
                "include": bool(item.get("include_interaction", True)),
                "world_ras_mm": [float(v) for v in item.get("world_ras_mm") or []],
                "name": str(item.get("name") or ""),
            }
        )
    if not any(p["include"] for p in points):
        raise RuntimeError("No foreground point in {}".format(points_json))
    return points


def _run_policy(
    session,
    image_xyz: np.ndarray,
    points: list[dict],
    affine,
    *,
    policy: str,
    initial_seg_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Feed points (and optional DINOv3 initial_seg) under one policy; return mask (XYZ bool).

    Policies:
      - sequential            : predict after every point
      - initial_empty_batch   : accumulate all points, predict once on the last
      - initial_seg_then_points: add the DINOv3 mask as nnInteractive's
                                 prev-seg prior (no prediction), then apply
                                 points sequentially. This FAITHFULLY reproduces
                                 the real guided flow: the bridge passes the
                                 DINOv3 mask via add_initial_seg_interaction
                                 before the reviewed points are applied.
    """
    import torch

    session.set_image(image_xyz[None, ...].astype(np.float32, copy=False))
    target = np.zeros(image_xyz.shape, dtype=np.uint8)
    session.set_target_buffer(target)

    if policy == "initial_seg_then_points":
        if initial_seg_xyz is None:
            raise RuntimeError("initial_seg_then_points requires an initial_seg mask")
        # add_initial_seg_interaction resets interactions and seeds both the
        # target buffer and the prev-seg interaction channel with the DINOv3
        # mask. run_prediction=False: the points refine it, matching the bridge.
        session.add_initial_seg_interaction(
            initial_seg_xyz.astype(np.uint8, copy=False),
            run_prediction=False,
        )
        for point in points:
            vox = _ras_to_voxel_xyz(point["world_ras_mm"], affine)
            session.add_point_interaction(
                vox,
                include_interaction=point["include"],
                run_prediction=True,
            )
    else:
        n = len(points)
        for index, point in enumerate(points):
            vox = _ras_to_voxel_xyz(point["world_ras_mm"], affine)
            run_prediction = policy == "sequential" or index == n - 1
            session.add_point_interaction(
                vox,
                include_interaction=point["include"],
                run_prediction=run_prediction,
            )
    buf = session.target_buffer
    if isinstance(buf, torch.Tensor):
        buf = buf.cpu().numpy()
    return np.asarray(buf).astype(bool)


def _evaluate_one(session, image_path, label_path, points_json, out_dir, initial_seg_dir=None):
    started = time.time()
    original = nib.load(str(image_path))
    affine = np.asarray(original.affine, dtype=float)
    image_xyz = np.asanyarray(original.dataobj).astype(np.float32, copy=False)
    spacing = spacing_zyx(original)
    points = _load_points(points_json)

    label_xyz = np.asanyarray(nib.load(str(label_path)).dataobj)
    target_xyz = label_xyz > 0

    stem = _case_stem(image_path.name)
    initial_seg_xyz = None
    if initial_seg_dir is not None:
        seg_path = Path(initial_seg_dir) / "{}_guided_mask.nii.gz".format(stem)
        if seg_path.exists():
            initial_seg_xyz = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8, copy=False)
            if initial_seg_xyz.shape != image_xyz.shape:
                raise RuntimeError(
                    "initial_seg shape {} != image shape {} for {}".format(
                        initial_seg_xyz.shape, image_xyz.shape, stem
                    )
                )
        else:
            print("  WARNING: no initial_seg at {}".format(seg_path))

    policies = ["sequential", "initial_empty_batch"]
    if initial_seg_xyz is not None:
        policies.append("initial_seg_then_points")

    results = {}
    for policy in policies:
        pol_started = time.time()
        try:
            session.reset_interactions()
            mask_xyz = _run_policy(
                session, image_xyz, points, affine,
                policy=policy, initial_seg_xyz=initial_seg_xyz,
            )
            if mask_xyz.shape != target_xyz.shape:
                raise RuntimeError(
                    "mask shape {} != label shape {}".format(mask_xyz.shape, target_xyz.shape)
                )
            metrics = binary_metrics(mask_xyz, target_xyz, spacing, surface_tolerance_mm=2.0)
            results[policy] = {
                "dice": metrics["dice"],
                "hd95_mm": metrics["hd95_mm"],
                "surface_dice": metrics["surface_dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "pred_voxels": metrics["pred_voxels"],
                "elapsed_seconds": time.time() - pol_started,
            }
            mask_nii = nib.Nifti1Image(mask_xyz.astype(np.int16), affine, header=original.header.copy())
            mask_nii.header.set_data_dtype(np.int16)
            nib.save(mask_nii, str(out_dir / "{}_nninteractive_{}.nii.gz".format(stem, policy)))
        except Exception as exc:  # noqa: BLE001 - record per-policy, keep going
            results[policy] = {"error": "{0}: {1}".format(type(exc).__name__, exc), "elapsed_seconds": time.time() - pol_started}

    fg = sum(1 for p in points if p["include"])
    bg = sum(1 for p in points if not p["include"])
    return {
        "case_id": _case_stem(image_path.name),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "points_json": str(points_json),
        "foreground_points": fg,
        "background_points": bg,
        "target_voxels": int(target_xyz.sum()),
        "spacing_zyx": list(spacing),
        "native_shape_xyz": list(image_xyz.shape),
        "elapsed_seconds": time.time() - started,
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Drive nnInteractive with DINOv3-guided points and score Dice (offline)"
    )
    parser.add_argument("--points-dir", required=True, help="Dir with <case>_guided_points.json")
    parser.add_argument("--data-root", required=True, help="Totalsegmentator kidney root")
    parser.add_argument("--case-ids", default="s1031,s1120,s1130")
    parser.add_argument("--split", default="Tr", choices=["Tr", "Val"])
    parser.add_argument("--model-dir", required=True, help="nnInteractive model folder (has plans.json)")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fold", default=None, help="fold number or 'all' (default: infer)")
    parser.add_argument(
        "--initial-seg-dir",
        default=None,
        help="Dir with <case>_guided_mask.nii.gz (DINOv3 TTA masks). When set, "
        "adds the 'initial_seg_then_points' policy that faithfully reproduces "
        "the real guided flow (DINOv3 mask as nnInteractive prev-seg prior).",
    )
    args = parser.parse_args()

    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    import torch

    device = torch.device(args.device)
    session = nnInteractiveInferenceSession(device=device, use_torch_compile=False, verbose=False)
    print("Loading nnInteractive model from {} ...".format(args.model_dir), flush=True)
    session.initialize_from_trained_model_folder(
        args.model_dir,
        use_fold=(int(args.use_fold) if args.use_fold and args.use_fold.isdigit() else None),
    )

    data_root = Path(args.data_root)
    image_dir = data_root / "images{}".format(args.split)
    label_dir = data_root / "labels{}".format(args.split)
    points_dir = Path(args.points_dir)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    case_ids = [t.strip() for t in args.case_ids.split(",") if t.strip()]

    rows = []
    for case_id in case_ids:
        image_path = image_dir / "{}.nii.gz".format(case_id)
        label_path = label_dir / "{}.nii.gz".format(case_id)
        points_json = points_dir / "{}_guided_points.json".format(case_id)
        if not image_path.exists() or not label_path.exists() or not points_json.exists():
            print("SKIP {}: missing input".format(case_id))
            continue
        print("Evaluating {} (nnInteractive guided) ...".format(case_id), flush=True)
        row = _evaluate_one(session, image_path, label_path, points_json, out_dir, initial_seg_dir=args.initial_seg_dir)
        rows.append(row)
        for policy, res in row["policies"].items():
            if "error" in res:
                print("  {case} [{pol}] ERROR: {err}".format(case=case_id, pol=policy, err=res["error"]))
            else:
                print(
                    "  {case} [{pol}] Dice={dice:.4f}, HD95={hd95}, surf={sd}, "
                    "P={p:.3f}, R={r:.3f}, {t:.1f}s".format(
                        case=case_id,
                        pol=policy,
                        dice=res["dice"],
                        hd95="NA" if res["hd95_mm"] is None else "{:.1f}mm".format(res["hd95_mm"]),
                        sd="NA" if res["surface_dice"] is None else "{:.3f}".format(res["surface_dice"]),
                        p=res["precision"],
                        r=res["recall"],
                        t=res["elapsed_seconds"],
                    ),
                    flush=True,
                )

    def agg(policy, key):
        vals = [r["policies"][policy][key] for r in rows
                if policy in r["policies"] and key in r["policies"][policy] and r["policies"][policy].get(key) is not None]
        return None if not vals else float(np.mean(vals))

    all_policies = []
    for r in rows:
        for pol in r["policies"]:
            if pol not in all_policies:
                all_policies.append(pol)

    summary = {
        "model_dir": str(args.model_dir),
        "model": "nnInteractive_v1.0_official",
        "points_source": "feature_unet2d DINOv3-guided proposals (verify_guided_points.py)",
        "initial_seg_dir": str(args.initial_seg_dir) if args.initial_seg_dir else None,
        "n_cases": len(rows),
        "per_case": rows,
        "by_policy": {
            pol: {
                "mean_dice": agg(pol, "dice"),
                "mean_hd95_mm": agg(pol, "hd95_mm"),
                "mean_surface_dice": agg(pol, "surface_dice"),
                "mean_precision": agg(pol, "precision"),
                "mean_recall": agg(pol, "recall"),
            }
            for pol in all_policies
        },
        "note": (
            "Mask comes from the official nnInteractive v1.0 model driven by "
            "DINOv3-proposed points. 'initial_seg_then_points' faithfully "
            "reproduces the real guided flow (DINOv3 mask as prev-seg prior)."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n=== Summary ===")
    for pol, s in summary["by_policy"].items():
        print(
            "{pol}: mean Dice={d}, HD95={h}, surface_dice={sd}".format(
                pol=pol,
                d="NA" if s["mean_dice"] is None else "{:.4f}".format(s["mean_dice"]),
                h="NA" if s["mean_hd95_mm"] is None else "{:.1f}mm".format(s["mean_hd95_mm"]),
                sd="NA" if s["mean_surface_dice"] is None else "{:.3f}".format(s["mean_surface_dice"]),
            )
        )
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()
