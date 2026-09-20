"""Diagnostic 7: measure the intensity-distribution shift that T2b's stronger
augmentation imposes, to confirm the root-cause mechanism.

T2b's aug (vs nnU-Net default):
  - Gamma invert p=0.15 (was 0.1), range (0.5,1.75) (was (0.7,1.5)), p_retain_stats=1
  - Gamma non-invert p=0.4 (was 0.3), range (0.5,1.75)
  - Contrast (0.5,1.75) (was (0.75,1.25)) p=0.25 (was 0.15)
  - Brightness (0.6,1.4) (was (0.75,1.25)) p=0.25
  - GaussianNoise var (0,0.15) (was 0.1) p=0.15
  - SimulateLowRes scale (0.2,1) (was (0.5,1)) p=0.25

Question: does T2b's aug produce training images whose intensity statistics
(esp. for bright HU structures like bone/contrast) diverge so far from the clean
inference distribution that the model learns to fire on bright blobs?

This script takes a real preprocessed patch (CT-normalized), applies each aug
chain 200x, and reports:
  - distribution of foreground-kidney-pixel mean intensity (should stay ~-1.4 norm)
  - distribution of bright-structure (bone) pixel intensity after aug
  - fraction of runs where kidney mean > 0 (i.e. aug flipped kidney bright)
  - fraction where bone and kidney means become indistinguishable
"""
import os
import numpy as np
import torch

os.environ.setdefault("nnUNet_raw",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results",
                      "/home/wenwen_zhang/kidney_experiments_portable/results/nnUNet_results")
os.environ.setdefault("KIDNEY_EXT_DIR",
                      "/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer")


def load_patch(case='s0657'):
    base = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/nnUNetPlans_3d_96")
    d = np.load(os.path.join(base, case + ".npz"))
    data = d['data']  # (1,Z,Y,X) CT-normalized
    seg = d['seg']
    # kidney-centered 96^3 patch
    coords = np.argwhere(seg[0] > 0)
    center = coords.mean(0).astype(int)
    ps = 96
    lbs = np.clip(center - ps // 2, 0, None)
    ubs = np.minimum(np.array(seg.shape[1:]), lbs + ps)
    sl = tuple([slice(0, 1)] + [slice(l, u) for l, u in zip(lbs, ubs)])
    sls = tuple([slice(l, u) for l, u in zip(lbs, ubs)])
    dd = data[sl]
    ss = seg[0][sls]
    pad_lo = np.clip(ps // 2 - center, 0, None)
    pad_hi = (ps - (ubs - lbs)) - pad_lo
    pad_hi = np.clip(pad_hi, 0, None)
    pad_i = ((0, 0), (pad_lo[0], pad_hi[0]), (pad_lo[1], pad_hi[1]), (pad_lo[2], pad_hi[2]))
    pad_s = ((pad_lo[0], pad_hi[0]), (pad_lo[1], pad_hi[1]), (pad_lo[2], pad_hi[2]))
    dd = np.pad(dd, pad_i, 'constant', constant_values=0)[:, :ps, :ps, :ps]
    ss = np.pad(ss, pad_s, 'constant', constant_values=0)[:ps, :ps, :ps]
    return dd, ss


def build(which):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    rot = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    if which == 'exp4':
        return nnUNetTrainer.get_training_transforms(
            (96, 96, 96), rot, None, (0, 2), False, use_mask_for_norm=[False],
            is_cascaded=False, foreground_labels=[1], regions=None, ignore_label=None)
    else:
        from nnunetv2.training.nnUNetTrainer.kidney_252_trainer import \
            flexict3d_aug_252_Trainer as T

        class _S(T):
            def __init__(self):
                pass

        return _S().get_training_transforms(
            (96, 96, 96), rot, None, (0, 2), False, use_mask_for_norm=[False],
            is_cascaded=False, foreground_labels=[1], regions=None, ignore_label=None)


def measure(which, dd, ss, n=200, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    trans = build(which)
    # identify "bright" voxels in the ORIGINAL patch: normalized HU > 3 (≈ +290 HU,
    # bone/contrast). kidney is ~-1.4 norm (≈ 40 HU).
    bright_mask = dd[0] > 3.0
    kidney_mask = ss > 0
    print(f"\n=== {which} ({n} aug runs) ===")
    print(f"  patch: kidney voxels={int(kidney_mask.sum())} bright(>3norm) voxels={int(bright_mask.sum())}")
    if bright_mask.sum() == 0:
        print("  (no bright voxels in this patch; using >2 norm)")
        bright_mask = dd[0] > 2.0
        print(f"  bright(>2norm) voxels={int(bright_mask.sum())}")
    kidney_means = []
    bright_means = []
    kidney_flipped = 0   # kidney mean > 0 after aug (sign-flipped bright)
    indistinguishable = 0
    for _ in range(n):
        out = trans(image=torch.from_numpy(dd).float(),
                    segmentation=torch.from_numpy(ss[None]).to(torch.int16))
        im = out['image'][0]
        if isinstance(im, torch.Tensor):
            im = im.numpy()
        # seg may have moved; use the transformed seg to find kidney
        sg = out['segmentation'][0]
        if isinstance(sg, torch.Tensor):
            sg = sg.numpy()
        km = sg > 0
        if km.sum() == 0:
            continue
        km_mean = float(im[km].mean())
        kidney_means.append(km_mean)
        # bright region: use original bright_mask coords (aug is spatial, so this is
        # approximate) — better: track via regression_target
        # We'll just measure the global bright-voxel mean using the original mask
        # (spatial aug moves them, so this is a rough proxy). Use the top 1% brightest
        # voxels in the augmented image as "bright structure" proxy.
        top_thr = np.percentile(im, 99)
        bright_now = im >= top_thr
        if bright_now.sum() > 0:
            bm = float(im[bright_now].mean())
            bright_means.append(bm)
        if km_mean > 0:
            kidney_flipped += 1
        # indistinguishable: kidney mean within 0.5 of bright mean
        if bright_now.sum() > 0 and abs(bm - km_mean) < 0.5:
            indistinguishable += 1
    kidney_means = np.array(kidney_means)
    bright_means = np.array(bright_means)
    print(f"  kidney mean after aug: mean={kidney_means.mean():.2f} std={kidney_means.std():.2f} "
          f"min={kidney_means.min():.2f} max={kidney_means.max():.2f} (clean kidney ~-1.4)")
    print(f"  bright(top1%) mean after aug: mean={bright_means.mean():.2f} "
          f"std={bright_means.std():.2f} (clean bright ~+4)")
    print(f"  runs where kidney mean > 0 (sign-flipped/brightened): "
          f"{kidney_flipped}/{len(kidney_means)} ({100*kidney_flipped/max(len(kidney_means),1):.1f}%)")
    print(f"  runs where kidney & bright means indistinguishable(<0.5 apart): "
          f"{indistinguishable}/{len(kidney_means)} ({100*indistinguishable/max(len(kidney_means),1):.1f}%)")


def main():
    print("### Diagnostic 7: intensity-distribution shift from T2b aug ###")
    dd, ss = load_patch('s0657')
    print(f"  patch shape {dd.shape} kidney fg={int((ss>0).sum())} "
          f"data range [{dd.min():.2f},{dd.max():.2f}] mean {dd.mean():.2f}")
    print(f"  clean kidney mean (norm HU) = {dd[0][ss>0].mean():.2f}  "
          f"(≈ {dd[0][ss>0].mean()*57.8+120:.0f} HU)")
    measure('exp4', dd, ss, n=200, seed=0)
    measure('t2b', dd, ss, n=200, seed=0)
    print("\n### Interpretation ###")
    print("If T2b shows many runs where kidney mean > 0 or kidney/bright are")
    print("indistinguishable, the aug is destroying the HU sign information -> model")
    print("cannot rely on 'kidney is soft-tissue-dark-ish' and instead learns")
    print("shape/brightness shortcuts that fire on bright blobs (bone/contrast) -> ghost FP.")


if __name__ == "__main__":
    main()
