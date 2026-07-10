"""Materialize E:/total_test cases into a train.py-readable dataset for one organ.

Builds:
    <out>/imagesTr/<case>.nii.gz   (symlink/hardlink to ct.nii.gz)
    <out>/labelsTr/<case>.nii.gz   (copy of segmentations/<organ>.nii.gz)
    <out>/imagesVal/... labelsVal/...   (held-out cases)

Skips cases whose organ label is empty (all zeros).

Usage:
    python scripts/materialize_totaltest.py \
        --src E:/total_test --organ liver \
        --out E:/total_test/datasets/liver_direct \
        --val-frac 0.2 --seed 42
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def resample_image_to_label_grid(ct_path, lb_path, out_path):
    """Resample CT onto the label's affine/shape so the two share geometry.

    Matches the fewshot pipeline's 'resampled_to_label_grid' strategy. Uses
    nearest-neighbour in voxel-index space after mapping through the two
    affines (CT intensity — no interpolation needed across the grid remap for
    a coarse materialize; trilinear would be nicer but scipy map_coordinates
    with order=1 is fine and fast enough).
    """
    ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
    lb_img = nib.as_closest_canonical(nib.load(str(lb_path)))
    ct = ct_img.get_fdata(dtype=np.float32)
    lb_shape = lb_img.shape
    lb_aff = lb_img.affine
    ct_aff = ct_img.affine

    # voxel grid of the label
    zz, yy, xx = np.meshgrid(
        np.arange(lb_shape[0]), np.arange(lb_shape[1]), np.arange(lb_shape[2]),
        indexing="ij")
    grid = np.stack([zz.ravel(), yy.ravel(), xx.ravel(), np.ones(zz.size)], axis=0)  # (4,N)
    world = lb_aff @ grid  # world coords (3,N)
    # map world -> ct voxel
    inv = np.linalg.inv(ct_aff)
    vox = inv @ np.vstack([world[:3], np.ones((1, world.shape[1]))])  # (4,N)
    coords = vox[:3].reshape(3, *lb_shape)
    resampled = map_coordinates(ct, coords, order=1, mode="nearest").astype(np.float32)

    out = nib.Nifti1Image(resampled, lb_aff, header=lb_img.header)
    nib.save(out, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="total_test root with sXXXX/ dirs")
    ap.add_argument("--organ", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    organ = args.organ

    cases = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("s")])
    usable = []
    for c in cases:
        ct = c / "ct.nii.gz"
        lb = c / "segmentations" / (organ + ".nii.gz")
        if not ct.is_file() or not lb.is_file():
            continue
        d = np.asanyarray(nib.load(str(lb)).dataobj)
        if (d > 0).sum() == 0:
            print(f"  skip {c.name}: empty {organ} label")
            continue
        usable.append((c.name, ct, lb))
    print(f"[materialize] {len(usable)}/{len(cases)} cases usable for {organ}")

    rng = random.Random(args.seed)
    rng.shuffle(usable)
    n_val = max(1, int(round(len(usable) * args.val_frac))) if len(usable) > 2 else 0
    val = usable[:n_val]
    train = usable[n_val:]
    print(f"[materialize] train={len(train)} val={len(val)}")

    for split, rows in (("Tr", train), ("Val", val)):
        idir = out / f"images{split}"
        ldir = out / f"labels{split}"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for name, ct, lb in rows:
            idst = idir / f"{name}.nii.gz"
            ldst = ldir / f"{name}.nii.gz"
            if not ldst.exists():
                shutil.copy2(str(lb), str(ldst))
            if not idst.exists():
                # always resample image onto label grid to guarantee affine match
                resample_image_to_label_grid(ct, lb, idst)
    print(f"[materialize] done -> {out}")


if __name__ == "__main__":
    main()
