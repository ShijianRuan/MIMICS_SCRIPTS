"""Build Dataset904_KidneyLR (left+right whole-kidney, multi-label) from the new
Mimics batch annotations under E:/mimics_batch_export_kidney.

Differences from Dataset901/902:
  - Two foreground labels: 0=background, 1=kidney_left, 2=kidney_right.
  - Labels come from NEW Mimics annotations ("whole kidney", more complete than
    the v201 originals: +~5000 voxels/case, IoU ~0.88 with v201). Source images
    still come from v201 ct.nii.gz.
  - >50% train (per user request): 30 train (imagesTr) + 22 test (imagesTs).

The manifest at E:/mimics_batch_export_kidney/dataset_manifest.json maps each
case to its v201 image and the new label paths. All 56 cases are kept (a training
set should stay diverse: extreme Z-start and small-kidney cases are real and
useful, not to be filtered out).
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

MANIFEST = Path("E:/mimics_batch_export_kidney/dataset_manifest.json")
V201 = Path("Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201")
CANDS = Path("E:/mimics_script_offline/external/dinov3-medical-seg/experiments/kidney_0806_guided_verify/kidney_lr_candidates.json")
DST = Path("E:/kidney_experiments_portable/data/nnUNet_raw/Dataset904_KidneyLR")


def main():
    manifest = json.loads(MANIFEST.read_text())
    cands = sorted(manifest["cases"].keys())  # ALL 56 cases; no filtering (train set stays diverse)
    print("all cases: {}/{}".format(len(cands), len(manifest["cases"])))

    # Stratified split: sort by Z-start, pick 32 train (>50%) spread evenly, rest test.
    # Re-derive zstart per candidate (cheap, from labels).
    def zstart(cid):
        info = manifest["cases"][cid]
        lp = Path(info["labels"]["kidney_left"]["path"]["absolute"])
        la = np.asanyarray(nib.load(str(lp)).dataobj) > 0
        img = nib.load(str(V201 / cid / "ct.nii.gz"))
        sp = np.linalg.norm(img.affine[:3, :3], axis=0)
        zax = int(np.argmax(sp))
        zidx = np.where(la.any(axis=tuple(i for i in range(3) if i != zax)))[0]
        return (int(zidx.min()) / la.shape[zax]) if len(zidx) else 0.5

    cands_sorted = sorted(cands, key=zstart)
    n_train = 32
    step = len(cands_sorted) / n_train
    train_idx = set(int(i * step) for i in range(n_train))
    train = [cands_sorted[i] for i in sorted(train_idx)]
    test = [c for i, c in enumerate(cands_sorted) if i not in train_idx]
    print("split: {} train + {} test (train={:.0%})".format(len(train), len(test), len(train) / len(cands)))

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    manifest_out = {"train": [], "test": []}
    subdir = {"train": ("imagesTr", "labelsTr"), "test": ("imagesTs", "labelsTs")}
    for split_name, group in [("train", train), ("test", test)]:
        img_dir, lbl_dir = subdir[split_name]
        for cid in group:
            info = manifest["cases"][cid]
            src_img = V201 / cid / "ct.nii.gz"
            lp = Path(info["labels"]["kidney_left"]["path"]["absolute"])
            rp = Path(info["labels"]["kidney_right"]["path"]["absolute"])
            if not (src_img.exists() and lp.exists() and rp.exists()):
                print("  SKIP {} (missing)".format(cid))
                continue
            img_nii = nib.as_closest_canonical(nib.load(str(src_img)))
            left = nib.as_closest_canonical(nib.load(str(lp)))
            right = nib.as_closest_canonical(nib.load(str(rp)))
            la = (np.asanyarray(left.dataobj) > 0).astype(np.uint8)
            ra = (np.asanyarray(right.dataobj) > 0).astype(np.uint8)
            # Merge: 0=bg, 1=left, 2=right. L/R have zero overlap (verified).
            label = la * 1 + ra * 2
            # Sanity: shape + affine must align after reorient.
            if (img_nii.shape[:3] != la.shape or img_nii.shape[:3] != ra.shape
                    or not np.allclose(img_nii.affine, left.affine, atol=1e-4)
                    or not np.allclose(img_nii.affine, right.affine, atol=1e-4)):
                print("  SKIP {} (shape/affine mismatch)".format(cid))
                continue
            lbl_out = nib.Nifti1Image(label, img_nii.affine, img_nii.header)
            lbl_out.header.set_data_dtype(np.uint8)
            nib.save(img_nii, str(DST / img_dir / "{}_0000.nii.gz".format(cid)))
            nib.save(lbl_out, str(DST / lbl_dir / "{}.nii.gz".format(cid)))
            manifest_out[split_name].append(cid)

    n_train = len(manifest_out["train"])
    n_test = len(manifest_out["test"])
    dsjson = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "kidney_left": 1, "kidney_right": 2},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": "Dataset904_KidneyLR",
        "description": ("kidney left+right WHOLE-kidney (new Mimics annotations, more complete than v201: "
                        "+~5000 voxels/case, IoU~0.88). Multi-label 0/1/2. {} train (imagesTr) + {} test (imagesTs), "
                        "images from v201 ct, reoriented RAS. All 56 cases kept (diverse training set). Note: 13 case "
                        "ids overlap Dataset902 but labels differ (new whole-kidney vs v201 originals).".format(n_train, n_test)),
        "reference": "Totalsegmentator v2.0.1 + new Mimics annotations",
        "release": "0.0",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
        "split": manifest_out,
    }
    (DST / "dataset.json").write_text(json.dumps(dsjson, indent=2) + "\n", encoding="utf-8")
    print("Done: {} train + {} test -> {}".format(n_train, n_test, DST))


if __name__ == "__main__":
    main()
