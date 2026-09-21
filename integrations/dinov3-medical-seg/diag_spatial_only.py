"""Diagnostic 2: isolate SPATIAL alignment from intensity-aug artifacts.

The cube-boundary data is binary (0/1), so intensity transforms (contrast/gamma/noise)
can push 1.0 below the 0.5 threshold and make it LOOK like data-seg misalign even when
spatial alignment is perfect. This script:

1) Runs the FULL T2b + exp4 chains but measures SPATIAL alignment using a
   threshold-free metric: count voxels where seg>0 but the ORIGINAL (pre-intensity-aug)
   cube region does NOT cover them (true spatial spill). We do this by also carrying the
   cube mask as a SEPARATE 'regression_target' which intensity transforms also ignore
   (ImageOnlyTransform). SpatialTransform DOES apply to regression_target, so it tracks
   the seg's spatial deformation exactly.

2) Runs ONLY SpatialTransform (strip all intensity aug) to measure pure spatial
   data-seg agreement for both configs (should be identical since params are identical).

3) Runs the FULL chain but bins on a ROBUST foreground test: data>mean(data) (Otsu-like)
   instead of data>0.5, to remove the intensity-threshold artifact.
"""
import numpy as np
import torch


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def build_transforms(which, patch=(96, 96, 96)):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    rot = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    mirror_axes = (0, 2)
    do_dummy_2d = False
    ds_scales = None
    use_mask_for_norm = [False]
    fg_labels = [1]
    if which == 'exp4':
        return nnUNetTrainer.get_training_transforms(
            patch, rot, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=False,
            foreground_labels=fg_labels, regions=None, ignore_label=None)
    else:
        from nnunetv2.training.nnUNetTrainer.kidney_252_trainer import \
            flexict3d_aug_252_Trainer as T

        class _Stub(T):
            def __init__(self):
                pass

        return _Stub().get_training_transforms(
            patch, rot, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=False,
            foreground_labels=fg_labels, regions=None, ignore_label=None)


def run_full_robust(which, n=30, seed=0):
    """Full chain, but measure spatial alignment via data>median(data) (intensity-robust)
    AND via a regression_target (the cube mask) that only SpatialTransform touches."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    trans = build_transforms(which)

    dices_thr = []      # data>0.5 (intensity-sensitive)
    dices_robust = []   # data > data.median() (intensity-robust)
    seg_spills = []     # seg voxels outside deformed cube region (true spatial spill)
    for i in range(n):
        data_np = np.zeros((1, 96, 96, 96), dtype=np.float32)
        seg_np = np.zeros((1, 96, 96, 96), dtype=np.int16)
        cube = np.zeros((1, 96, 96, 96), dtype=np.float32)  # track deformed region
        data_np[0, 30:66, 30:66, 30:66] = 1.0
        seg_np[0, 30:66, 30:66, 30:66] = 1
        cube[0, 30:66, 30:66, 30:66] = 1.0
        out = trans(image=torch.from_numpy(data_np).float(),
                    segmentation=torch.from_numpy(seg_np).to(torch.int16),
                    regression_target=torch.from_numpy(cube).float())
        im = out['image'][0]
        sg = out['segmentation'][0]
        rt = out['regression_target'][0]
        if isinstance(im, torch.Tensor):
            im = im.numpy()
        if isinstance(sg, torch.Tensor):
            sg = sg.numpy()
        if isinstance(rt, torch.Tensor):
            rt = rt.numpy()
        dd_thr = im > 0.5
        dd_rob = im > np.median(im)
        ss = sg > 0
        rt_mask = rt > 0.5
        dices_thr.append(dice(dd_thr, ss))
        dices_robust.append(dice(dd_rob, ss))
        # seg voxels outside the SPATIALLY-DEFORMED cube region = true spatial spill
        seg_spills.append(int(np.logical_and(ss, ~rt_mask).sum()))
    dices_thr = np.array(dices_thr)
    dices_robust = np.array(dices_robust)
    seg_spills = np.array(seg_spills)
    print(f"\n=== {which} full chain ({n} runs) ===")
    print(f"  dice(data>0.5, seg)  [intensity-sensitive]: mean={dices_thr.mean():.4f} "
          f"min={dices_thr.min():.4f} #<0.9={int((dices_thr<0.9).sum())}")
    print(f"  dice(data>median,seg)[intensity-robust]   : mean={dices_robust.mean():.4f} "
          f"min={dices_robust.min():.4f} #<0.9={int((dices_robust<0.9).sum())}")
    print(f"  TRUE spatial spill (seg outside deformed cube): mean={seg_spills.mean():.1f} "
          f"max={seg_spills.max()}  (>0 = real spatial data-seg misalign)")


def run_spatial_only(which, n=30, seed=0):
    """Only SpatialTransform + Mirror (strip intensity). Pure spatial alignment test.
    exp4 and t2b have IDENTICAL SpatialTransform params, so results MUST match."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    rot = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    patch = (96, 96, 96)
    sp = SpatialTransform(
        patch, patch_center_dist_from_border=0, random_crop=False,
        p_elastic_deform=0, p_rotation=0.2, rotation=rot,
        p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
        bg_style_seg_sampling=False)
    mirror = MirrorTransform(allowed_axes=(0, 2))
    dices = []
    seg_spills = []
    for i in range(n):
        data_np = np.zeros((1, 96, 96, 96), dtype=np.float32)
        seg_np = np.zeros((1, 96, 96, 96), dtype=np.int16)
        cube = np.zeros((1, 96, 96, 96), dtype=np.float32)
        data_np[0, 30:66, 30:66, 30:66] = 1.0
        seg_np[0, 30:66, 30:66, 30:66] = 1
        cube[0, 30:66, 30:66, 30:66] = 1.0
        out = sp(image=torch.from_numpy(data_np).float(),
                 segmentation=torch.from_numpy(seg_np).to(torch.int16),
                 regression_target=torch.from_numpy(cube).float())
        out = mirror(**out)
        im = out['image'][0]
        sg = out['segmentation'][0]
        rt = out['regression_target'][0]
        for a in (im, sg, rt):
            pass
        im = im.numpy() if isinstance(im, torch.Tensor) else im
        sg = sg.numpy() if isinstance(sg, torch.Tensor) else sg
        rt = rt.numpy() if isinstance(rt, torch.Tensor) else rt
        ss = sg > 0
        rt_mask = rt > 0.5
        dices.append(dice(im > 0.5, ss))
        seg_spills.append(int(np.logical_and(ss, ~rt_mask).sum()))
    dices = np.array(dices)
    seg_spills = np.array(seg_spills)
    print(f"\n=== {which} SPATIAL-ONLY (SpatialTransform+Mirror, {n} runs) ===")
    print(f"  dice(data>0.5, seg): mean={dices.mean():.4f} min={dices.min():.4f} "
          f"#<0.9={int((dices<0.9).sum())}")
    print(f"  spatial spill (seg outside deformed cube): mean={seg_spills.mean():.1f} "
          f"max={seg_spills.max()}")


def main():
    print("### Diagnostic 2: isolate spatial alignment from intensity artifacts ###")
    run_full_robust('exp4', n=30, seed=0)
    run_full_robust('t2b', n=30, seed=0)
    run_spatial_only('exp4', n=30, seed=0)
    run_spatial_only('t2b', n=30, seed=0)
    print("\n### Conclusion interpretation ###")
    print("If spatial-only dice ~1.0 and spill ~0 for BOTH -> transforms do NOT break")
    print("spatial alignment. The over-seg bug is then a TRAINING-DISTRIBUTION shift")
    print("(strong aug makes train inputs too different from clean inference inputs),")
    print("NOT a data-seg desync bug.")


if __name__ == "__main__":
    main()
