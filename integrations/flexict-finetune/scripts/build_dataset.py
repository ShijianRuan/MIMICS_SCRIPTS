"""Build a standard nnU-Net raw dataset for few-shot segmentation, from a
Totalsegmentator-style source tree.

Source layout expected (per case dir under --source):
    <source>/<case_id>/ct.nii.gz
    <source>/<case_id>/segmentations/<label-name>.nii.gz

Output: a standard nnU-Net raw dataset at <out>/Dataset<id>_<name>/
    imagesTr/<case>_0000.nii.gz   labelsTr/<case>.nii.gz   (train)
    imagesTs/<case>_0000.nii.gz   labelsTs/<case>.nii.gz   (test)
    dataset.json   (channel 0 = CT, labels {background:0, <name>:1}, 0/1 single label)

Images and labels are reoriented to RAS (canonical) and checked for matching
shape/affine. The train/test split is stratified by the target's Z-start so a
few-shot train set spans the cranio-caudal extent of the organ.

Example (liver few-shot from Totalsegmentator v201):
    python scripts/build_dataset.py \
        --source  Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201 \
        --label-name liver --dataset-id 907 --dataset-name LiverFS \
        --n-train 8 --out ./data/nnUNet_raw
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def _zstart(ct_path, lbl_path):
    """Relative Z-start of the target in the volume, for stratified splitting."""
    la = np.asanyarray(nib.load(str(lbl_path)).dataobj) > 0
    img = nib.load(str(ct_path))
    sp = np.linalg.norm(img.affine[:3, :3], axis=0)
    zax = int(np.argmax(sp))
    zidx = np.where(la.any(axis=tuple(i for i in range(3) if i != zax)))[0]
    return (int(zidx.min()) / la.shape[zax]) if len(zidx) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="source root: <source>/<case>/ct.nii.gz + segmentations/<label>.nii.gz")
    ap.add_argument("--label-name", required=True,
                    help="target label file name under segmentations/ (e.g. liver)")
    ap.add_argument("--dataset-id", type=int, required=True, help="nnU-Net dataset id (e.g. 907)")
    ap.add_argument("--dataset-name", required=True, help="dataset suffix name (e.g. LiverFS)")
    ap.add_argument("--n-train", type=int, default=8, help="number of few-shot train cases")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="cap total cases scanned (0 = all in source)")
    ap.add_argument("--out", required=True, help="nnUNet_raw directory to write into")
    args = ap.parse_args()

    src = Path(args.source)
    ds_dir = Path(args.out) / f"Dataset{args.dataset_id}_{args.dataset_name}"

    # discover cases that have both ct and the requested label
    cases = []
    for cdir in sorted(p for p in src.iterdir() if p.is_dir()):
        ct = cdir / "ct.nii.gz"
        lbl = cdir / "segmentations" / f"{args.label_name}.nii.gz"
        if ct.exists() and lbl.exists():
            cases.append((cdir.name, ct, lbl))
        if args.max_cases and len(cases) >= args.max_cases:
            break
    if not cases:
        raise SystemExit(f"no cases with ct.nii.gz + segmentations/{args.label_name}.nii.gz "
                         f"under {src}")
    print(f"found {len(cases)} cases with label '{args.label_name}'")

    # stratified split by Z-start: train cases spread across cranio-caudal range
    n_train = min(args.n_train, len(cases) - 1)
    cases_sorted = sorted(cases, key=lambda c: _zstart(c[1], c[2]))
    step = len(cases_sorted) / n_train
    train_idx = set(int(i * step) for i in range(n_train))
    train = [cases_sorted[i] for i in sorted(train_idx)]
    test = [c for i, c in enumerate(cases_sorted) if i not in train_idx]
    print(f"split: {len(train)} train + {len(test)} test")

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = {"train": [], "test": []}
    subdir = {"train": ("imagesTr", "labelsTr"), "test": ("imagesTs", "labelsTs")}

    for split_name, group in [("train", train), ("test", test)]:
        img_dir, lbl_dir = subdir[split_name]
        for cid, ct, lbl in group:
            img_nii = nib.as_closest_canonical(nib.load(str(ct)))
            lbl_nii = nib.as_closest_canonical(nib.load(str(lbl)))
            la = (np.asanyarray(lbl_nii.dataobj) > 0).astype(np.uint8)
            if (img_nii.shape[:3] != la.shape
                    or not np.allclose(img_nii.affine, lbl_nii.affine, atol=1e-4)):
                print(f"  SKIP {cid} (shape/affine mismatch)")
                continue
            lbl_out = nib.Nifti1Image(la, img_nii.affine, img_nii.header)
            lbl_out.header.set_data_dtype(np.uint8)
            nib.save(img_nii, str(ds_dir / img_dir / f"{cid}_0000.nii.gz"))
            nib.save(lbl_out, str(ds_dir / lbl_dir / f"{cid}.nii.gz"))
            manifest[split_name].append(cid)

    n_tr, n_te = len(manifest["train"]), len(manifest["test"])
    dsjson = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, args.label_name: 1},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "name": f"Dataset{args.dataset_id}_{args.dataset_name}",
        "description": (f"Few-shot {args.label_name} segmentation. Single label 0/1. "
                        f"{n_tr} train (imagesTr) + {n_te} test (imagesTs), CT channel 0, "
                        f"reoriented RAS. Built from {src.name}."),
        "reference": "Totalsegmentator",
        "release": "0.0",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
        "split": manifest,
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dsjson, indent=2) + "\n", encoding="utf-8")
    print(f"done: {ds_dir}  ({n_tr} train + {n_te} test)")


if __name__ == "__main__":
    main()
