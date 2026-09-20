"""Diagnostic: compare exp4 (default nnU-Net transforms) vs T2b (flexict3d_aug)
training transforms for data-seg alignment + boundary behaviour.

Builds a sharp cube boundary (30:66 = 1, else 0) in data+seg, runs each transform
chain N times, reports dice(data>thr, seg>0). Also reports the fraction of seg
foreground voxels that land OUTSIDE the cube region (boundary spill) and the data
intensity statistics inside vs outside the seg mask (should match for identity-ish
aug, diverge for strong intensity aug -- but that's expected and NOT a bug).

Run on remote:
  source ~/miniconda/etc/profile.d/conda.sh; conda activate nnUnet
  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
  python ~/kidney_experiments_portable/code/diag_aug_compare.py
"""
import numpy as np
import torch


def build_transforms(which, patch=(96, 96, 96)):
    """which: 'exp4' (default nnUNet) or 't2b' (flexict3d_aug)."""
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    rot = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    mirror_axes = (0, 2)          # base trainer: L-R disabled
    do_dummy_2d = False
    ds_scales = None              # deep supervision off -> None
    use_mask_for_norm = [False]
    fg_labels = [1]

    if which == 'exp4':
        return nnUNetTrainer.get_training_transforms(
            patch, rot, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=False,
            foreground_labels=fg_labels, regions=None, ignore_label=None)
    elif which == 't2b':
        from nnunetv2.training.nnUNetTrainer.kidney_252_trainer import \
            flexict3d_aug_252_Trainer as T

        class _Stub(T):
            def __init__(self):
                pass

        return _Stub().get_training_transforms(
            patch, rot, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=False,
            foreground_labels=fg_labels, regions=None, ignore_label=None)
    else:
        raise ValueError(which)


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def run(which, n=30, seed=0, verbose=False):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trans = build_transforms(which)

    # sharp cube boundary: data=1 inside [30:66]^3, 0 outside; seg = same mask
    # transforms expect torch tensors (dataloader converts np->torch before calling)
    data_np = np.zeros((1, 96, 96, 96), dtype=np.float32)
    seg_np = np.zeros((1, 96, 96, 96), dtype=np.int16)
    data_np[0, 30:66, 30:66, 30:66] = 1.0
    seg_np[0, 30:66, 30:66, 30:66] = 1

    dices = []
    seg_sizes = []
    spills = []           # seg voxels outside the TRUE cube region after aug
    data_outside_mask = []  # mean data intensity where seg==0 (should be ~0 unless aug smears)

    for i in range(n):
        out = trans(image=torch.from_numpy(data_np).float(),
                    segmentation=torch.from_numpy(seg_np).to(torch.int16))
        im = out['image'][0]
        if isinstance(im, torch.Tensor):
            im = im.numpy()
        # seg may be a list (deep supervision) but ds_scales=None -> tensor
        sg = out['segmentation'][0]
        if isinstance(sg, torch.Tensor):
            sg = sg.numpy()
        dd = im > 0.5
        ss = sg > 0
        d = dice(dd, ss)
        dices.append(d)
        seg_sizes.append(int(ss.sum()))
        # spill = seg voxels outside the original cube
        true_cube = np.zeros((96, 96, 96), dtype=bool)
        true_cube[30:66, 30:66, 30:66] = True
        spills.append(int(np.logical_and(ss, ~true_cube).sum()))
        # data intensity in seg==0 region
        data_outside_mask.append(float(im[~ss].mean()) if (~ss).any() else 0.0)
        if verbose and d < 0.95:
            print(f"  [{which} #{i}] dice={d:.4f} seg_size={ss.sum()} "
                  f"spill={spills[-1]} data@outside={data_outside_mask[-1]:.3f}")

    dices = np.array(dices)
    spills = np.array(spills)
    data_outside_mask = np.array(data_outside_mask)
    print(f"\n=== {which} ({n} runs) ===")
    print(f"  dice(data>0.5, seg>0):  mean={dices.mean():.4f} min={dices.min():.4f} "
          f"max={dices.max():.4f} #<0.9={int((dices<0.9).sum())} #<0.95={int((dices<0.95).sum())}")
    print(f"  seg foreground voxels:  mean={seg_sizes and np.mean(seg_sizes):.0f} "
          f"(true cube={36**3}={36*36*36})")
    print(f"  seg spill outside cube: mean={spills.mean():.1f} max={spills.max()}  "
          f"(>0 means seg leaks past data boundary -> spatial misalign)")
    print(f"  data mean where seg==0: mean={data_outside_mask.mean():.4f} "
          f"max={data_outside_mask.max():.4f} (intensity aug only, not a bug)")
    return dices, spills


def main():
    # Reproduce real config: patch 96^3, fp32, deep supervision OFF (ds_scales=None)
    print("### Diagnostic: exp4 (default) vs T2b (flexict3d_aug) data-aug alignment ###")
    print("Config: patch=(96,96,96), mirror_axes=(0,2), do_dummy_2d=False, "
          "deep_supervision OFF, cube [30:66]^3")
    # exp4 first (baseline)
    run('exp4', n=30, seed=0, verbose=True)
    # T2b
    run('t2b', n=30, seed=0, verbose=True)

    # Also test SimulateLowRes alone (the most-suspected transform) at T2b scale
    print("\n### SimulateLowRes alone: exp4 scale=(0.5,1) vs T2b scale=(0.2,1) ###")
    from batchgeneratorsv2.transforms.spatial.low_resolution import \
        SimulateLowResolutionTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    for tag, scale in [('exp4', (0.5, 1)), ('t2b', (0.2, 1))]:
        t = RandomTransform(
            SimulateLowResolutionTransform(
                scale=scale, synchronize_channels=False, synchronize_axes=True,
                ignore_axes=None, allowed_channels=None, p_per_channel=0.5),
            apply_probability=1.0)
        np.random.seed(0)
        torch.manual_seed(0)
        dices = []
        for _ in range(20):
            data_np = np.zeros((1, 96, 96, 96), dtype=np.float32)
            seg_np = np.zeros((1, 96, 96, 96), dtype=np.int16)
            data_np[0, 30:66, 30:66, 30:66] = 1.0
            seg_np[0, 30:66, 30:66, 30:66] = 1
            out = t(image=torch.from_numpy(data_np).float(),
                    segmentation=torch.from_numpy(seg_np).to(torch.int16))
            im = out['image'][0]
            sg = out['segmentation'][0]
            if isinstance(im, torch.Tensor):
                im = im.numpy()
            if isinstance(sg, torch.Tensor):
                sg = sg.numpy()
            dices.append(dice(im > 0.5, sg > 0))
        dices = np.array(dices)
        print(f"  SimLowRes {tag} scale={scale}: dice mean={dices.mean():.4f} "
              f"min={dices.min():.4f} (seg is NOT touched by ImageOnlyTransform, "
              f"low dice here = data blurs past seg boundary, expected & same as exp4 default)")


if __name__ == "__main__":
    main()
