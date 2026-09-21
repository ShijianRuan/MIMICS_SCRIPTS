"""Build Dataset905_KidneyLeftWhole and Dataset906_KidneyRightWhole from the new
Mimics whole-kidney annotations under E:/mimics_batch_export_kidney.

Two SEPARATE single-label datasets (not one multi-label dataset):
  Dataset905_KidneyLeftWhole  — labels: 0=bg, 1=kidney_left  (left whole kidney)
  Dataset906_KidneyRightWhole — labels: 0=bg, 1=kidney_right (right whole kidney)

Each mirrors the Dataset901/902 layout: imagesTr/labelsTr (train) + imagesTs/
labelsTs (test), single 0/1 label, CT channel 0, reoriented RAS.

Both datasets share the same 56 cases and the same train/test split (32/24,
>50% train), so results are directly comparable between left and right. The
"Whole" in the name marks these as the NEW Mimics annotations (more complete
than the v201 originals: +~5000 voxels/case, IoU~0.88), distinct from
Dataset901/902 which use the v201 kidney_left label.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

MANIFEST = Path("E:/mimics_batch_export_kidney/dataset_manifest.json")
V201 = Path("Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201")
ROOT = Path("E:/kidney_experiments_portable/data/nnUNet_raw")

DATASETS = {
    905: ("Dataset905_KidneyLeftWhole", "kidney_left", "kidney_left"),
    906: ("Dataset906_KidneyRightWhole", "kidney_right", "kidney_right"),
}


def _zstart(manifest, cid):
    """Z-start of the left kidney (used only for stratified split ordering)."""
    info = manifest["cases"][cid]
    lp = Path(info["labels"]["kidney_left"]["path"]["absolute"])
    la = np.asanyarray(nib.load(str(lp)).dataobj) > 0
    img = nib.load(str(V201 / cid / "ct.nii.gz"))
    sp = np.linalg.norm(img.affine[:3, :3], axis=0)
    zax = int(np.argmax(sp))
    zidx = np.where(la.any(axis=tuple(i for i in range(3) if i != zax)))[0]
    return (int(zidx.min()) / la.shape[zax]) if len(zidx) else 0.5


def main():
    manifest = json.loads(MANIFEST.read_text())
    cands = sorted(manifest["cases"].keys())  # all 56, no filtering
    print("all cases: {}".format(len(cands)))

    # Shared stratified split: 32 train (>50%), rest test, spread by Z-start.
    cands_sorted = sorted(cands, key=lambda c: _zstart(manifest, c))
    n_train = 32
    step = len(cands_sorted) / n_train
    train_idx = set(int(i * step) for i in range(n_train))
    train = [cands_sorted[i] for i in sorted(train_idx)]
    test = [c for i, c in enumerate(cands_sorted) if i not in train_idx]
    print("shared split: {} train + {} test".format(len(train), len(test)))

    for ds_num, (ds_name, label_key, label_name) in DATASETS.items():
        dst = ROOT / ds_name
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
        manifest_out = {"train": [], "test": []}
        subdir = {"train": ("imagesTr", "labelsTr"), "test": ("imagesTs", "labelsTs")}
        for split_name, group in [("train", train), ("test", test)]:
            img_dir, lbl_dir = subdir[split_name]
            for cid in group:
                info = manifest["cases"][cid]
                src_img = V201 / cid / "ct.nii.gz"
                lp = Path(info["labels"][label_key]["path"]["absolute"])
                if not (src_img.exists() and lp.exists()):
                    print("  SKIP {} (missing)".format(cid))
                    continue
                img_nii = nib.as_closest_canonical(nib.load(str(src_img)))
                lbl_nii = nib.as_closest_canonical(nib.load(str(lp)))
                la = (np.asanyarray(lbl_nii.dataobj) > 0).astype(np.uint8)
                if (img_nii.shape[:3] != la.shape
                        or not np.allclose(img_nii.affine, lbl_nii.affine, atol=1e-4)):
                    print("  SKIP {} (shape/affine mismatch)".format(cid))
                    continue
                lbl_out = nib.Nifti1Image(la, img_nii.affine, img_nii.header)
                lbl_out.header.set_data_dtype(np.uint8)
                nib.save(img_nii, str(dst / img_dir / "{}_0000.nii.gz".format(cid)))
                nib.save(lbl_out, str(dst / lbl_dir / "{}.nii.gz".format(cid)))
                manifest_out[split_name].append(cid)

        n_train = len(manifest_out["train"])
        n_test = len(manifest_out["test"])
        dsjson = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, label_name: 1},
            "numTraining": n_train,
            "file_ending": ".nii.gz",
            "name": ds_name,
            "description": ("{} whole-kidney (new Mimics annotations, more complete than v201: "
                            "+~5000 voxels/case, IoU~0.88). Single label 0/1. {} train (imagesTr) + {} test "
                            "(imagesTs), images from v201 ct, reoriented RAS. All 56 cases kept. Shares the "
                            "same cases/split as Dataset905/906 counterpart. 'Whole' marks new annotations "
                            "vs Dataset901/902 v201 originals.".format(label_name, n_train, n_test)),
            "reference": "Totalsegmentator v2.0.1 + new Mimics whole-kidney annotations",
            "release": "0.0",
            "overwrite_image_reader_writer": "NibabelIOWithReorient",
            "split": manifest_out,
        }
        (dst / "dataset.json").write_text(json.dumps(dsjson, indent=2) + "\n", encoding="utf-8")
        print("{}: {} train + {} test".format(ds_name, n_train, n_test))


if __name__ == "__main__":
    main()
