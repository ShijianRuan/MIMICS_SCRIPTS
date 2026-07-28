"""Experiment data preparation for the DINOv3 few-shot ablation study.

Two-stage pipeline (separated so the slow prescreen is done once per organ):

Stage A -- prescreen: scan the Totalsegmentator tree, for one organ find all
cases whose mask is NON-empty, and write a deterministic usable-case list to
a JSON manifest.  Reading only the label (not the CT) makes this fast.

Stage B -- materialize: for a chosen subset of cases (train+val), resample each
CT onto its label grid and write imagesTr/labelsTr (+ Val) under <out>.  Uses a
process pool so many cases resample in parallel.  Never writes into the source
tree.

Manifest layout (per organ):
    <manifest_dir>/<organ>.json
    {
      "organ": "liver",
      "src": "Z:/.../Totalsegmentator_dataset_v201",
      "cases": ["s0001", "s0003", ...],   # all non-empty, sorted
      "n": 1234
    }

A single fixed train/val split is derived deterministically from the manifest
(seed-controlled shuffle) so every ablation for an organ uses the SAME val set
-- required for fair comparison across configs.
"""
import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def prescreen(src, organ, manifest_path):
    src = Path(src).resolve()
    cases = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("s")])
    usable = []
    for c in cases:
        lb = c / "segmentations" / f"{organ}.nii.gz"
        if not lb.is_file():
            continue
        try:
            d = np.asanyarray(nib.load(str(lb)).dataobj)
        except Exception:
            continue
        if (d > 0).sum() > 0:
            usable.append(c.name)
    manifest = {"organ": organ, "src": str(src),
                "cases": usable, "n": len(usable)}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prescreen] {organ}: {len(usable)}/{len(cases)} non-empty cases "
          f"-> {manifest_path}")
    return manifest


def split_cases(cases, n_train, n_val, seed=42):
    """Deterministic train/val split. val is held-out; train is the pool for
    k-shot sampling."""
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:n_val + n_train])
    val = [cases[i] for i in val_idx]
    train = [cases[i] for i in train_idx]
    return train, val


def resample_one(args):
    src, case, organ, idst, ldst = args
    src = Path(src)
    ct_path = src / case / "ct.nii.gz"
    lb_path = src / case / "segmentations" / f"{organ}.nii.gz"
    if not idst.exists():
        ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
        lb_img = nib.as_closest_canonical(nib.load(str(lb_path)))
        ct = ct_img.get_fdata(dtype=np.float32)
        lb_shape = lb_img.shape
        lb_aff = lb_img.affine
        ct_aff = ct_img.affine
        zz, yy, xx = np.meshgrid(
            np.arange(lb_shape[0]), np.arange(lb_shape[1]), np.arange(lb_shape[2]),
            indexing="ij")
        grid = np.stack([zz.ravel(), yy.ravel(), xx.ravel(),
                         np.ones(zz.size)], axis=0)
        world = lb_aff @ grid
        inv = np.linalg.inv(ct_aff)
        vox = inv @ np.vstack([world[:3], np.ones((1, world.shape[1]))])
        coords = vox[:3].reshape(3, *lb_shape)
        resampled = map_coordinates(ct, coords, order=1, mode="nearest").astype(np.float32)
        out = nib.Nifti1Image(resampled, lb_aff, header=lb_img.header)
        nib.save(out, str(idst))
    if not ldst.exists():
        shutil.copy2(str(lb_path), str(ldst))
    return case


def materialize(manifest_path, out, n_train, n_val, seed=42, workers=6):
    with open(manifest_path) as f:
        m = json.load(f)
    cases = m["cases"]
    organ = m["organ"]
    src = m["src"]
    n_val = min(n_val, len(cases))
    n_train = min(n_train, len(cases) - n_val)
    train, val = split_cases(cases, n_train, n_val, seed)
    out = Path(out).resolve()
    print(f"[materialize] {organ}: train_pool={len(train)} val={len(val)} -> {out}")

    jobs = []
    for split, rows in (("Tr", train), ("Val", val)):
        idir = out / f"images{split}"; ldir = out / f"labels{split}"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for name in rows:
            jobs.append((src, name, organ,
                         idir / f"{name}.nii.gz", ldir / f"{name}.nii.gz"))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(resample_one, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            if done % 20 == 0:
                print(f"  ...{done}/{len(jobs)}")
    # write a split manifest so training reads deterministic val
    with open(out / "split.json", "w") as f:
        json.dump({"organ": organ, "train": train, "val": val}, f, indent=2)
    print(f"[materialize] done -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prescreen")
    p1.add_argument("--src", required=True)
    p1.add_argument("--organ", required=True)
    p1.add_argument("--manifest", required=True)
    p2 = sub.add_parser("materialize")
    p2.add_argument("--manifest", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--n-train", type=int, required=True,
                    help="train POOL size (k-shot sampled from this)")
    p2.add_argument("--n-val", type=int, required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "prescreen":
        prescreen(args.src, args.organ, args.manifest)
    else:
        materialize(args.manifest, args.out, args.n_train, args.n_val,
                    args.seed, args.workers)


if __name__ == "__main__":
    main()
