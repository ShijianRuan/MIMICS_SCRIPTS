"""Build Dataset903_KidneyLeftMRI from TotalsegmentatorMRI v2.0.0.

Reads the 92 complete-kidney candidates already filtered, splits into
train/test (~75/17) stratified by Z-start, reorients each case to RAS
canonical (MRI cases have inconsistent axes), binarizes the label to 0/1,
and writes nnU-Net raw format identical to Dataset901/Dataset902.

Original MRI dataset is read-only; all outputs go to the new Dataset903 folder.
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path("Z:/ImageAnalysisData/1-CT/Segmentation/data/TotalsegmentatorMRI_dataset_v200")
CAND = Path("E:/mimics_script_offline/external/dinov3-medical-seg/experiments/kidney_0806_guided_verify/mri_kidney_candidates.json")
DST = Path("E:/kidney_experiments_portable/data/nnUNet_raw/Dataset903_KidneyLeftMRI")
# Dataset902 CT cases, to guard against any id overlap
CT902 = json.loads(Path("E:/mimics_script_offline/external/dinov3-medical-seg/experiments/kidney_0806_guided_verify/dataset902_sampled100.json").read_text())


def main():
    cands = json.loads(CAND.read_text())
    # Exclude any case that also appears in the CT Dataset902 (same patient scanned
    # with both modalities) to avoid cross-dataset leakage.
    ct_ids = set(CT902)
    cands = [c for c in cands if c["cid"] not in ct_ids]
    print("after excluding CT-overlap ({} ids): {} MRI cases remain".format(len(ct_ids), len(cands)))

    # 5-shot split mirroring Dataset901 (CT 5-shot): 5 train + rest test.
    # Hand-picked 5 train cases spread across Z-start (0.12-0.56), each near the
    # median volume and spanning different Z-spacings, so the few-shot support
    # set is representative (anterior/mid/posterior kidneys, thin/thick slices).
    train_ids = ["s0384", "s0020", "s0272", "s0291", "s0597"]
    train = [c for c in cands if c["cid"] in train_ids]
    if len(train) != len(train_ids):
        missing = set(train_ids) - {c["cid"] for c in train}
        raise RuntimeError("Train ids missing from candidates: {}".format(missing))
    test = [c for c in cands if c["cid"] not in train_ids]
    print("5-shot split: total={} train={} test={}".format(len(cands), len(train), len(test)))
    print("train cases:", [(c["cid"], "z={:.2f}".format(c["zs"])) for c in train])

    overlap = {c["cid"] for c in train + test} & set(ct_ids)
    if overlap:
        raise RuntimeError("Overlap with Dataset902 CT: {}".format(overlap))

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    (DST / "imagesTs").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTs").mkdir(parents=True, exist_ok=True)

    manifest = {"train": [], "test": []}
    # train -> imagesTr/labelsTr; test -> imagesTs/labelsTs (nnU-Net convention).
    subdir = {"train": ("imagesTr", "labelsTr"), "test": ("imagesTs", "labelsTs")}
    for split_name, group in [("train", train), ("test", test)]:
        img_dir, lbl_dir = subdir[split_name]
        for c in group:
            cid = c["cid"]
            src_img = SRC / cid / "mri.nii.gz"
            src_lbl = SRC / cid / "segmentations" / "kidney_left.nii.gz"
            if not src_img.exists() or not src_lbl.exists():
                print("  SKIP {} (missing)".format(cid))
                continue
            # Reorient to RAS canonical so axes are consistent across cases.
            img_nii = nib.as_closest_canonical(nib.load(str(src_img)))
            lbl_nii = nib.as_closest_canonical(nib.load(str(src_lbl)))
            # Binarize label to 0/1 (defensive).
            lbl_arr = (np.asanyarray(lbl_nii.dataobj) > 0).astype(np.uint8)
            lbl_out = nib.Nifti1Image(lbl_arr, lbl_nii.affine, lbl_nii.header)
            lbl_out.header.set_data_dtype(np.uint8)
            # Sanity: image and label must align after reorient.
            if img_nii.shape[:3] != lbl_nii.shape[:3] or not np.allclose(img_nii.affine, lbl_nii.affine, atol=1e-4):
                print("  SKIP {} (shape/affine mismatch after reorient: {} vs {})".format(cid, img_nii.shape, lbl_nii.shape))
                continue
            nib.save(img_nii, str(DST / img_dir / "{}_0000.nii.gz".format(cid)))
            nib.save(lbl_out, str(DST / lbl_dir / "{}.nii.gz".format(cid)))
            manifest[split_name].append(cid)

    n_train = len(manifest["train"])
    n_test = len(manifest["test"])
    dsjson = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "kidney_left": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": "Dataset903_KidneyLeftMRI",
        "description": ("kidney_left MRI 5-shot (mirrors Dataset901 CT 5-shot): {} train (imagesTr) + {} test (imagesTs). "
                        "From TotalsegmentatorMRI v2.0.0, complete-kidney cases (>=5000 voxels, Z-ratio>=8%, "
                        "zstart 0.10-0.90, Z-spacing<=8mm), reoriented RAS canonical, excluding CT-overlap. "
                        "Train cases spread evenly by Z-start.".format(n_train, n_test)),
        "reference": "TotalSegmentatorMRI v2.0.0",
        "release": "0.0",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
        "split": manifest,
    }
    (DST / "dataset.json").write_text(json.dumps(dsjson, indent=2) + "\n", encoding="utf-8")
    print("Done: {} train + {} test -> {}".format(n_train, n_test, DST))


if __name__ == "__main__":
    main()
