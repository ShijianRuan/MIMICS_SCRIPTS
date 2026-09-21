"""Diagnostic 5: characterize the GHOST-ORGAN FP that T2b produces at axis0~175
(far from true kidney at axis0 62-102).

Questions:
1) Is the ghost reproducible across checkpoints (best vs final) for T2b?
2) What does the raw CT look like at the ghost location? (is there real tissue
   like right kidney / spleen / liver there?)
3) Does the ghost exist for exp4 too (maybe just smaller)?
4) Is the ghost a single connected component? its size?
"""
import os, sys
import numpy as np
import torch
import nibabel as nib

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


def load_raw(case):
    raw = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_raw/"
        "Dataset901_KidneyLeft5shot/imagesTr/" + case + "_0000.nii.gz")
    gtf = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/gt_segmentations/" + case + ".nii.gz")
    im = nib.load(raw)
    seg = nib.load(gtf)
    img = im.get_fdata().astype(np.float32)
    segarr = seg.get_fdata().astype(np.int16)
    return img[None], segarr, {'spacing': list(im.header.get_zooms())[:3]}


def load_predictor(trainer, ckpt='checkpoint_best.pth', config='3d_96'):
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


def cc_stats(mask):
    """Return sizes of connected components in a 3D bool mask via scipy."""
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes = sizes[1:]  # drop background (0)
    return np.sort(sizes)[::-1]


def analyze_ghost(case, img, seg_gt, pred_seg, tag, ckpt):
    fp = np.logical_and(pred_seg > 0, seg_gt == 0)
    tp = np.logical_and(pred_seg > 0, seg_gt > 0)
    coords = np.argwhere(seg_gt > 0)
    bb_lo, bb_hi = coords.min(0), coords.max(0)
    fp_coords = np.argwhere(fp)
    if len(fp_coords):
        d_lo = np.maximum(0, bb_lo - fp_coords).max(1)
        d_hi = np.maximum(0, fp_coords - bb_hi).max(1)
        dist = np.maximum(d_lo, d_hi)
        ghost = fp & (dist > 20)[..., None] if False else None
        ghost_mask = np.zeros_like(fp)
        ghost_mask[tuple(fp_coords[dist > 20].T)] = True
    else:
        ghost_mask = np.zeros_like(fp)
    print(f"\n=== {tag} ({ckpt}) on {case} ===")
    print(f"  TP={int(tp.sum())} FP={int(fp.sum())} ghost(>20vox from kidney)={int(ghost_mask.sum())}")
    ccs = cc_stats(ghost_mask)
    print(f"  ghost connected components (sizes): {ccs[:8]}{'...' if len(ccs)>8 else ''}")
    if len(ccs):
        # centroid of largest ghost CC
        from scipy.ndimage import label, center_of_mass
        lab, n = label(ghost_mask)
        sizes = np.bincount(lab.ravel())[1:]
        biggest = sizes.argmax() + 1
        cz, cy, cx = center_of_mass(ghost_mask, lab, biggest)
        print(f"  largest ghost CC size={int(sizes.max())} centroid axis0={cz:.0f} (kidney bbox axis0 [{bb_lo[0]},{bb_hi[0]}])")
        # image intensity stats at ghost centroid slice
        sl = int(cz)
        img_sl = img[0, :, :, sl]
        print(f"  raw CT intensity at ghost slice z={sl}: min={img_sl.min():.0f} "
              f"max={img_sl.max():.0f} mean={img_sl.mean():.0f} "
              f"(kidney slice z={(bb_lo[0]+bb_hi[0])//2}: mean={img[0,:,:,(bb_lo[0]+bb_hi[0])//2].mean():.0f})")
        # what fraction of ghost is in positive-HU (soft tissue) vs lung/air?
        # CTNormalization used mean120 std57.8; raw HU = norm*57.8+120
        ghost_hu = img[0][ghost_mask]
        print(f"  ghost raw HU: mean={ghost_hu.mean():.0f} min={ghost_hu.min():.0f} "
              f"max={ghost_hu.max():.0f} (soft-tissue ~40-60 HU, liver ~60, spleen ~50)")
    return int(ghost_mask.sum()), (ccs[0] if len(ccs) else 0)


def main():
    print(f"### Diagnostic 5: GHOST-ORGAN FP characterization ###")
    # test validation case s1031 + a TRAINING case s0657 (both reported over-seg)
    cases = ['s1031', 's0657']
    rows = []
    for case in cases:
        print(f"\n############## CASE {case} ##############")
        try:
            img, seg_gt, props = load_raw(case)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        print(f"  img {img.shape} spacing {props['spacing']} seg fg={int((seg_gt>0).sum())}")
        for trainer, tag in [('flexict3d_252_Trainer', 'exp4'),
                             ('flexict3d_aug_252_Trainer', 'T2b')]:
            for ckpt in (['checkpoint_best.pth'] if tag == 'exp4'
                         else ['checkpoint_best.pth', 'checkpoint_final.pth']):
                print(f"\n--- {case} {trainer} {tag} {ckpt} ---")
                try:
                    pred = load_predictor(trainer, ckpt=ckpt)
                except FileNotFoundError as e:
                    print(f"  skip (missing): {e}")
                    continue
                pred_seg = pred.predict_single_npy_array(
                    img.astype(np.float32), props, None, None, False)
                gsz, biggest = analyze_ghost(case, img, seg_gt, pred_seg, tag, ckpt)
                rows.append((case, tag, ckpt, int((pred_seg>0).sum()), gsz, biggest))
                del pred
                torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    print(f"{'case':<8} {'tag':<6} {'ckpt':<22} {'pred_fg':>10} {'ghost':>8} {'biggest_CC':>11}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<6} {r[2]:<22} {r[3]:>10} {r[4]:>8} {r[5]:>11}")


if __name__ == "__main__":
    main()
