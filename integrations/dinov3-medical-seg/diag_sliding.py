"""Diagnostic 4: run ACTUAL sliding-window inference on a validation case for
both exp4 and T2b, and locate WHERE the FP voxels are (which z-slices, distance
from true kidney). This isolates whether T2b's 127k FP come from a specific region.

Single-patch tests showed both models have ~1500 FP on a kidney-centered patch and
0 FP on pure-background patches. But T2b's whole-volume FP is 127082 vs exp4's 1449.
So the extra FP must come from the sliding-window assembly. This script runs the real
predictor on the preprocessed case and reports FP distribution.
"""
import os, sys
import numpy as np
import torch

os.environ.setdefault("nnUNet_def_n_proc", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("nnUNet_raw",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results",
                      "/home/wenwen_zhang/kidney_experiments_portable/results/nnUNet_results")
os.environ.setdefault("KIDNEY_EXT_DIR",
                      "/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer")


def load_raw_case(case):
    """Load raw image + GT seg in nnU-Net expected (C,Z,Y,X) ordering with spacing."""
    import nibabel as nib
    raw = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_raw/"
        "Dataset901_KidneyLeft5shot/imagesTr/" + case + "_0000.nii.gz")
    gtf = os.path.expanduser(
        "~/kidney_exhangs_programs_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/gt_segmentations/" + case + ".nii.gz") \
        if False else os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/gt_segmentations/" + case + ".nii.gz")
    im = nib.load(raw)
    seg = nib.load(gtf)
    img = im.get_fdata().astype(np.float32)   # (X,Y,Z) per nibabel/SimpleITK
    segarr = seg.get_fdata().astype(np.int16)
    spacing = list(im.header.get_zooms())[:3]  # (X,Y,Z)
    # nnU-Net image_reader_writer = NibabelIOWithReorient: expects (C,X,Y,Z)? We
    # pass as (C,*spatial) and let predictor preprocess. NibabelIO reads as (X,Y,Z)
    # so we provide (1,X,Y,Z).
    return img[None], segarr, {'spacing': spacing}


def load_predictor(trainer, config='3d_96', ckpt='checkpoint_best.pth'):
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import importlib
    mod = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.kidney_252_trainer")
    _ = getattr(mod, trainer)
    model_dir = join(os.environ["nnUNet_results"],
                     "Dataset901_KidneyLeft5shot",
                     f"{trainer}__nnUNetPlans__{config}")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=False, device=torch.device("cuda"),
                           verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(0,),
                                              checkpoint_name=ckpt)
    return pred


def predict_volume(pred, img, props):
    """img: (C,X,Y,Z) raw, props: {'spacing':(sx,sy,sz)}. Returns (X,Y,Z) seg."""
    seg = pred.predict_single_npy_array(
        img.astype(np.float32), props, None, None, False)
    return seg


def analyze(case, img, seg_gt, pred_seg, tag):
    seg_gt = seg_gt.astype(int)
    fp = np.logical_and(pred_seg > 0, seg_gt == 0)
    fn = np.logical_and(pred_seg == 0, seg_gt > 0)
    tp = np.logical_and(pred_seg > 0, seg_gt > 0)
    dice = 2 * tp.sum() / (pred_seg.sum() + seg_gt.sum()) if (pred_seg.sum() + seg_gt.sum()) else 1.0
    print(f"\n=== {tag} on {case} ===")
    print(f"  pred shape {pred_seg.shape} gt shape {seg_gt.shape} "
          f"TP={int(tp.sum())} FP={int(fp.sum())} FN={int(fn.sum())} dice={dice:.4f}")
    # FP distribution along axis 0 (the long z=419 axis after reorient)
    fg_ax0 = np.where(seg_gt.sum((1, 2)) > 0)[0]
    z_lo, z_hi = (int(fg_ax0.min()), int(fg_ax0.max())) if len(fg_ax0) else (-1, -1)
    fp_per_ax0 = fp.sum((1, 2))
    print(f"  true kidney axis0-range: [{z_lo},{z_hi}] (len {z_hi-z_lo+1})")
    fp_inside = int(fp[z_lo:z_hi + 1].sum()) if z_lo >= 0 else -1
    fp_outside = int(fp.sum()) - fp_inside
    print(f"  FP inside kidney axis0-range: {fp_inside}  | outside: {fp_outside}")
    top = np.argsort(fp_per_ax0)[::-1][:8]
    print(f"  top FP slices (axis0, fp_voxels): " +
          ", ".join(f"({z},{int(fp_per_ax0[z])})" for z in top))
    # distance from kidney bbox
    coords = np.argwhere(seg_gt > 0)
    if len(coords) and len(np.argwhere(fp)):
        bb_lo, bb_hi = coords.min(0), coords.max(0)
        fp_coords = np.argwhere(fp)
        d_lo = np.maximum(0, bb_lo - fp_coords).max(1)
        d_hi = np.maximum(0, fp_coords - bb_hi).max(1)
        dist = np.maximum(d_lo, d_hi)
        print(f"  FP INSIDE kidney bbox: {int((dist == 0).sum())}  "
              f"within 20vox: {int((dist <= 20).sum())}  "
              f">20vox away: {int((dist > 20).sum())}")


def main():
    case = 's1031'
    print(f"### Diagnostic 4: actual sliding-window inference FP localization ###")
    img, seg_gt, props = load_raw_case(case)
    print(f"  img {img.shape} spacing {props['spacing']} seg_gt {seg_gt.shape} "
          f"fg={int((seg_gt>0).sum())}")

    for trainer in ['flexict3d_252_Trainer', 'flexict3d_aug_252_Trainer']:
        tag = 'exp4' if 'aug' not in trainer else 'T2b'
        print(f"\n--- loading {trainer} ({tag}) ---")
        pred = load_predictor(trainer)
        pred_seg = predict_volume(pred, img, props)
        print(f"  pred_seg shape {pred_seg.shape} fg={int((pred_seg>0).sum())}")
        analyze(case, img, seg_gt, pred_seg, tag)
        del pred
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
