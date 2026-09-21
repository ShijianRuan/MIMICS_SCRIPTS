"""Diagnostic 8: confirm the GammaTransform(p_invert_image=1, p_retain_stats=1)
mechanism. When the image is inverted, CT-normalized kidney (-1.4) becomes +1.4
(bright) and bone (+4) becomes -4 (dark). With p_retain_stats=1 the mean/std are
restored, but the RELATIVE ordering of structures flips: kidney becomes the
brightest soft structure. If this happens in 15% of training patches (T2b) the
network may learn 'bright soft blob = kidney'.

This script isolates GammaTransform(invert, retain_stats) and measures, over many
runs, how often the kidney becomes brighter than the median background after aug.
"""
import os
import numpy as np
import torch

os.environ.setdefault("KIDNEY_EXT_DIR",
                      "/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer")


def load_patch(case='s0657'):
    base = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/nnUNetPlans_3d_96")
    d = np.load(os.path.join(base, case + ".npz"))
    data = d['data']
    seg = d['seg']
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


def main():
    from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast

    dd, ss = load_patch('s0657')
    kidney_mask = ss > 0
    bg_mask = ss == 0
    print("### Diagnostic 8: GammaTransform invert effect on kidney brightness ###")
    print(f"  clean: kidney mean={dd[0][kidney_mask].mean():.2f} bg mean={dd[0][bg_mask].mean():.2f}")
    print(f"  (kidney is DARKER than bg in clean CT-normalized: {dd[0][kidney_mask].mean() < dd[0][bg_mask].mean()})")

    # T2b invert gamma: range (0.5,1.75), p_invert=1, p_retain_stats=1, p=0.15
    configs = [
        ('exp4 invert gamma (0.7,1.5) p=0.10', GammaTransform(
            gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
            synchronize_channels=False, p_per_channel=1, p_retain_stats=1), 0.10),
        ('T2b  invert gamma (0.5,1.75) p=0.15', GammaTransform(
            gamma=BGContrast((0.5, 1.75)), p_invert_image=1,
            synchronize_channels=False, p_per_channel=1, p_retain_stats=1), 0.15),
        ('T2b  noninv gamma (0.5,1.75) p=0.40', GammaTransform(
            gamma=BGContrast((0.5, 1.75)), p_invert_image=0,
            synchronize_channels=False, p_per_channel=1, p_retain_stats=1), 0.40),
    ]
    for name, gt, p in configs:
        rt = RandomTransform(gt, apply_probability=p)
        np.random.seed(0)
        torch.manual_seed(0)
        kidney_brighter = 0
        n_eff = 0
        k_means = []
        for _ in range(300):
            out = rt(image=torch.from_numpy(dd).float(),
                     segmentation=torch.from_numpy(ss[None]).to(torch.int16))
            im = out['image'][0]
            if isinstance(im, torch.Tensor):
                im = im.numpy()
            sg = out['segmentation'][0]
            if isinstance(sg, torch.Tensor):
                sg = sg.numpy()
            km = sg > 0
            bm = sg == 0
            if km.sum() == 0:
                continue
            n_eff += 1
            km_mean = float(im[km].mean())
            bm_mean = float(im[bm].mean())
            k_means.append(km_mean)
            if km_mean > bm_mean:
                kidney_brighter += 1
        k_means = np.array(k_means)
        print(f"\n  {name}:")
        print(f"    kidney mean after aug: {k_means.mean():.2f} std {k_means.std():.2f} "
              f"range [{k_means.min():.2f},{k_means.max():.2f}]")
        print(f"    runs where kidney BRIGHTER than bg: {kidney_brighter}/{n_eff} "
              f"({100*kidney_brighter/max(n_eff,1):.1f}%)")
        print(f"    (when invert fires, kidney flips from dark to bright vs bg)")

    print("\n### Interpretation ###")
    print("If invert-gamma makes kidney brighter than bg in a sizeable fraction of")
    print("runs, the network sees 'kidney=bright' in ~p_invert*p_retain of patches.")
    print("T2b widens gamma range AND raises p_invert 0.10->0.15 -> more bright-kidney")
    print("exposures. Under 5-shot (only 5 cases), this can dominate the learned")
    print("decision boundary -> at inference the model fires on bright blobs (bone/" )
    print("contrast) far from kidney -> ghost FP.")


if __name__ == "__main__":
    main()
