"""Diagnostic 6: what image structure does the T2b ghost fire on?

The ghost fires ~70-88 slices away from the true kidney, on slices that are
mostly air (mean HU -612) but with bright structures (ghost voxel HU mean 144-245,
max 1260-1717). This script localizes the ghost relative to bright image
structures (bone/contrast, HU>200) to determine if the model learned to fire on
"bright blob" instead of "kidney shape".

Also tests: does blurring a clean kidney-region patch (SimulateLowRes scale 0.2)
make T2b fire MORE false positives? If yes, the strong SimLowRes augmentation
trained the model to be blob-sensitive.
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
    return (im.get_fdata().astype(np.float32)[None],
            seg.get_fdata().astype(np.int16),
            {'spacing': list(im.header.get_zooms())[:3]})


def load_predictor(trainer, ckpt='checkpoint_best.pth'):
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import importlib
    mod = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.kidney_252_trainer")
    _ = getattr(mod, trainer)
    model_dir = join(os.environ["nnUNet_results"],
                     "Dataset901_KidneyLeft5shot",
                     f"{trainer}__nnUNetPlans__3d_96")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=False, device=torch.device("cuda"),
                           verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(0,),
                                              checkpoint_name=ckpt)
    return pred


def analyze_trigger(case, img, seg_gt, pred_seg, tag):
    fp = np.logical_and(pred_seg > 0, seg_gt == 0)
    coords = np.argwhere(seg_gt > 0)
    bb_lo, bb_hi = coords.min(0), coords.max(0)
    fp_coords = np.argwhere(fp)
    d_lo = np.maximum(0, bb_lo - fp_coords).max(1)
    d_hi = np.maximum(0, fp_coords - bb_hi).max(1)
    dist = np.maximum(d_lo, d_hi)
    ghost_mask = np.zeros_like(fp)
    ghost_mask[tuple(fp_coords[dist > 20].T)] = True
    print(f"\n=== {tag} on {case} ghost analysis ===")
    if ghost_mask.sum() == 0:
        print("  no ghost")
        return
    img3 = img[0]
    # 1) overlap of ghost with bright structures (HU>200 = bone/contrast)
    bright = img3 > 200
    print(f"  ghost voxels: {int(ghost_mask.sum())}")
    print(f"  ghost overlap with bright(HU>200): "
          f"{int(np.logical_and(ghost_mask, bright).sum())} "
          f"({100*np.logical_and(ghost_mask, bright).sum()/ghost_mask.sum():.1f}%)")
    # 2) ghost overlap with soft-tissue (0-100 HU)
    soft = np.logical_and(img3 > 0, img3 < 100)
    print(f"  ghost overlap with soft(0-100 HU): "
          f"{int(np.logical_and(ghost_mask, soft).sum())} "
          f"({100*np.logical_and(ghost_mask, soft).sum()/ghost_mask.sum():.1f}%)")
    # 3) distance from ghost centroid to nearest bright-blob centroid
    from scipy.ndimage import label, center_of_mass
    lab, n = label(bright)
    if n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        bright_centroids = [center_of_mass(bright, lab, i + 1) for i in range(min(n, 20))]
        glab, gn = label(ghost_mask)
        if gn > 0:
            gsz = np.bincount(glab.ravel())[1:]
            gbig = gsz.argmax() + 1
            gc = center_of_mass(ghost_mask, glab, gbig)
            dists = [np.sqrt(sum((a - b) ** 2 for a, b in zip(gc, c))) for c in bright_centroids]
            nearest = int(np.argmin(dists))
            print(f"  largest ghost CC centroid {tuple(round(x) for x in gc)}; "
                  f"nearest bright blob centroid {tuple(round(x) for x in bright_centroids[nearest])} "
                  f"(size {int(sizes[nearest])}, dist {dists[nearest]:.0f} vox)")
    # 4) does the ghost sit where a second kidney-like organ could be?
    #    check symmetry: kidney_left bbox; is there a same-z mirrored blob?
    print(f"  true kidney bbox: lo={tuple(bb_lo)} hi={tuple(bb_hi)}")


def blur_patch_test(case, img, seg_gt, trainer_tag):
    """Take a kidney-centered patch, apply SimulateLowRes scale=(0.2,1) (T2b's setting),
    run the model, see if blurring induces FP. Compare exp4 vs T2b model."""
    print(f"\n=== blur-patch test: {trainer_tag} on {case} kidney patch ===")
    coords = np.argwhere(seg_gt > 0)
    center = coords.mean(0).astype(int)
    ps = 96
    shp = np.array(img[0].shape)
    lbs = np.clip(center - ps // 2, 0, None)
    ubs = np.minimum(shp, lbs + ps)
    sl_img = tuple([slice(0, 1)] + [slice(l, u) for l, u in zip(lbs, ubs)])
    sl_seg = tuple([slice(l, u) for l, u in zip(lbs, ubs)])
    d = img[sl_img]
    s = seg_gt[sl_seg]
    pad_lo = np.clip(ps // 2 - center, 0, None)
    pad_hi = (ps - (ubs - lbs)) - pad_lo
    pad_hi = np.clip(pad_hi, 0, None)
    pad_img = ((0, 0), (pad_lo[0], pad_hi[0]), (pad_lo[1], pad_hi[1]), (pad_lo[2], pad_hi[2]))
    pad_seg = ((pad_lo[0], pad_hi[0]), (pad_lo[1], pad_hi[1]), (pad_lo[2], pad_hi[2]))
    d = np.pad(d, pad_img, 'constant', constant_values=0)[:, :ps, :ps, :ps]
    s = np.pad(s, pad_seg, 'constant', constant_values=0)[:ps, :ps, :ps]
    print(f"  kidney patch shape {d.shape} fg={int((s>0).sum())}")

    from batchgeneratorsv2.transforms.spatial.low_resolution import \
        SimulateLowResolutionTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    simlow = RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.2, 1), synchronize_channels=False, synchronize_axes=True,
            ignore_axes=None, allowed_channels=None, p_per_channel=1),
        apply_probability=1.0)

    pred = load_predictor('flexict3d_aug_252_Trainer' if trainer_tag == 'T2b'
                          else 'flexict3d_252_Trainer')
    pred.network = pred.network.to(torch.device("cuda")).eval()
    net = pred.network
    for variant, inp in [('clean', torch.from_numpy(d).float()),
                         ('simlow_0.2', None)]:
        if inp is None:
            out = simlow(image=torch.from_numpy(d).float(),
                         segmentation=torch.from_numpy(s[None]).to(torch.int16))
            inp = out['image']
        with torch.no_grad():
            o = net(inp[None].cuda())
            if isinstance(o, (list, tuple)):
                o = o[0]
        pseg = o.argmax(1)[0].cpu().numpy()
        tp = int(np.logical_and(pseg > 0, s > 0).sum())
        fp = int(np.logical_and(pseg > 0, s == 0).sum())
        print(f"  {variant}: pred_fg={int((pseg>0).sum())} TP={tp} FP={fp}")
    del pred
    torch.cuda.empty_cache()


def main():
    print("### Diagnostic 6: what triggers the T2b ghost? ###")
    for case in ['s1031', 's0657']:
        img, seg_gt, props = load_raw(case)
        print(f"\n########## {case} ##########")
        for trainer, tag in [('flexict3d_252_Trainer', 'exp4'),
                             ('flexict3d_aug_252_Trainer', 'T2b')]:
            pred = load_predictor(trainer)
            pred_seg = pred.predict_single_npy_array(
                img.astype(np.float32), props, None, None, False)
            analyze_trigger(case, img, seg_gt, pred_seg, tag)
            del pred
            torch.cuda.empty_cache()
        # blur-patch test for both models
        blur_patch_test(case, img, seg_gt, 'exp4')
        blur_patch_test(case, img, seg_gt, 'T2b')


if __name__ == "__main__":
    main()
