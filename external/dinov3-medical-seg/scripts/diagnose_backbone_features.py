"""Diagnose whether DINOv3 backbone features are actually useful for segmentation.

Question being answered:
    The frozen + linear3d probe on liver reached only DSC 0.025 after 20 epochs.
    Is that because (a) the backbone features carry no discriminative signal for
    foreground vs background, or (b) the features are fine but the probe/optimizer
    can't exploit them?

Method:
    Load a real CT slice + its organ mask, run it through the frozen DINOv3
    backbone under the SAME input pipeline the trainer uses (single-channel ->
    repeat to 3 -> ImageNet normalize), extract the 4 feature maps, and measure
    how separable foreground vs background voxels are in feature space:
        - per-channel mean feature value inside vs outside the mask
        - the "discrimination ratio" = |mean_fg - mean_bg| / (std_fg + std_bg)
          averaged over channels — a proxy for linear separability.
    A ratio near 0 means the features cannot distinguish the organ from
    background even with an optimal linear readout → confirms the backbone is
    not contributing usable features.

    We compare THREE input pipelines so the root cause is isolated:
        1. repeat3 + imagenet_norm   (the current trainer pipeline)
        2. repeat3 + ct_norm          (CT-dataset mean/std, 3 channels identical)
        3. single slice stats print   (sanity: raw input distribution)

Usage:
    python scripts/diagnose_backbone_features.py \
        --ct E:/total_test/s0001/ct.nii.gz \
        --label E:/total_test/s0001/segmentations/liver.nii.gz \
        --model ./models/dinov3-vitb16
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.backbone import DINOv3Backbone, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_slice(ct_path, label_path, slice_axis=2):
    """Load a CT volume + label, pick an axial slice that actually contains the organ."""
    ct = nib.load(ct_path).get_fdata(dtype=np.float32)
    lb = nib.load(label_path).get_fdata(dtype=np.float32)
    # normalize like the dataset: CT clip [-1024,1024] -> min-max [0,1]
    ct = np.clip(ct, -1024.0, 1024.0)
    lo, hi = float(ct.min()), float(ct.max())
    ct = (ct - lo) / (hi - lo + 1e-8)
    lb = (lb > 0).astype(np.float32)

    if slice_axis == 2:
        slices = [ct[:, :, d] for d in range(ct.shape[2])]
        masks = [lb[:, :, d] for d in range(lb.shape[2])]
    else:
        raise NotImplementedError("only axial for now")

    # pick the slice with the largest foreground area (most informative)
    areas = [m.sum() for m in masks]
    idx = int(np.argmax(areas))
    slc = slices[idx].astype(np.float32)   # (H, W) in [0,1]
    mask = masks[idx].astype(np.float32)   # (H, W) binary
    print(f"[data] volume shape={ct.shape}, picked slice idx={idx}, "
          f"fg voxels={int(mask.sum())} ({100*mask.mean():.1f}% of slice)")
    return slc, mask


def to_3ch_tensor(slc, device):
    """(H,W) -> (1,3,H,W) float on device."""
    t = torch.from_numpy(slc)[None, None]  # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1).to(device)
    return t


def normalize(t, mean, std):
    m = torch.tensor(mean, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    return (t - m) / s


def measure_separability(feats, mask_lowres, level_name):
    """feats: (1, C, h, w). mask_lowres: (h, w) float in {0,1}.

    Returns discrimination ratio = mean over channels of |mean_fg-mean_bg|/(std_fg+std_bg).
    Also returns the raw per-channel fg/bg means so we can see structure.
    """
    x = feats[0]  # (C, h, w)
    C, h, w = x.shape
    xflat = x.reshape(C, -1)              # (C, h*w)
    mflat = mask_lowres.reshape(-1)       # (h*w,)
    fg = mflat > 0.5
    bg = ~fg
    n_fg, n_bg = int(fg.sum()), int(bg.sum())
    if n_fg == 0 or n_bg == 0:
        print(f"  [{level_name}] empty fg/bg after downsample (n_fg={n_fg}, n_bg={n_bg})")
        return float("nan"), None

    mean_fg = xflat[:, fg].mean(dim=1)
    mean_bg = xflat[:, bg].mean(dim=1)
    std_fg = xflat[:, fg].std(dim=1) + 1e-6
    std_bg = xflat[:, bg].std(dim=1) + 1e-6

    diff = (mean_fg - mean_bg).abs()
    ratio = (diff / (std_fg + std_bg)).mean().item()

    # also: how many channels have a "strong" signal (ratio > 0.1)?
    per_ch_ratio = (diff / (std_fg + std_bg))
    n_strong = int((per_ch_ratio > 0.1).sum())

    print(f"  [{level_name}] C={C} h×w={h}×{w}  n_fg={n_fg} n_bg={n_bg}")
    print(f"         feat mean={xflat.mean():.4f} std={xflat.std():.4f} "
          f"min={xflat.min():.4f} max={xflat.max():.4f}")
    print(f"         mean_fg(top5)={mean_fg.topk(5).values.tolist()}")
    print(f"         mean_bg(top5)={mean_bg.topk(5).values.tolist()}")
    print(f"         >>> discrimination_ratio={ratio:.4f}  "
          f"(channels with ratio>0.1: {n_strong}/{C})")
    return ratio, (mean_fg, mean_bg, per_ch_ratio)


def run_pipeline(name, slc, mask, backbone, device, mean, std, do_norm):
    """Run one input pipeline and report per-level separability."""
    t = to_3ch_tensor(slc, device)
    if do_norm:
        t = normalize(t, mean, std)
    # interpolate to backbone.img_size (matches encoder_3d.py)
    t = F.interpolate(t, size=(backbone.img_size, backbone.img_size),
                      mode="bilinear", align_corners=False)
    print(f"\n=== pipeline: {name} ===")
    print(f"  input to backbone: shape={tuple(t.shape)} mean={t.mean():.4f} "
          f"std={t.std():.4f} min={t.min():.4f} max={t.max():.4f}")

    with torch.no_grad():
        feats = backbone(t)  # list of (1, C, h, w)

    ratios = []
    for i, f in enumerate(feats):
        # downsample the slice mask to this feature's spatial size
        mh, mw = f.shape[-2], f.shape[-1]
        mt = torch.from_numpy(mask)[None, None].to(device).float()
        mt = F.interpolate(mt, size=(mh, mw), mode="nearest")
        r, _ = measure_separability(f, mt[0, 0], f"level{i}")
        ratios.append(r)
    valid = [r for r in ratios if r == r]  # drop nan
    avg = float(np.mean(valid)) if valid else float("nan")
    print(f"  >>> {name}: mean discrimination_ratio across levels = {avg:.4f}")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="./models/dinov3-vitb16")
    ap.add_argument("--out_indices", type=int, nargs="+", default=[2, 5, 8, 11])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    backbone = DINOv3Backbone(
        model_path=args.model,
        out_indices=args.out_indices,
        freeze=True,
        input_normalization="none",  # we normalize manually so we can compare variants
    ).to(device).eval()
    print(f"[model] img_size={backbone.img_size} embed_dim={backbone.embed_dim} "
          f"layers={backbone.num_layers} patch={backbone.patch_size}")
    # sanity: confirm weights are loaded, not random — check a known param stat
    w = next(backbone.backbone.parameters())
    print(f"[model] first param shape={tuple(w.shape)} mean={w.mean():.5f} std={w.std():.5f}")

    slc, mask = load_slice(args.ct, args.label, slice_axis=2)
    print(f"[input] slice range=[{slc.min():.3f},{slc.max():.3f}] "
          f"mean={slc.mean():.3f} std={slc.std():.3f}")

    # --- Pipeline 1: the CURRENT trainer pipeline (repeat3 + imagenet) ---
    r1 = run_pipeline("repeat3+imagenet_norm (CURRENT)", slc, mask, backbone, device,
                      IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, do_norm=True)

    # --- Pipeline 2: repeat3 + CT-dataset stats (3 channels identical, natural) ---
    # estimate per-slice mean/std (a proxy for dataset stats; real pipeline would use
    # a running estimate, but per-slice is enough to show the distribution effect)
    ct_mean = float(slc.mean())
    ct_std = float(slc.std() + 1e-6)
    print(f"\n[ct-stats] per-slice mean={ct_mean:.4f} std={ct_std:.4f}")
    r2 = run_pipeline("repeat3+ct_norm (3ch identical)", slc, mask, backbone, device,
                      (ct_mean, ct_mean, ct_mean), (ct_std, ct_std, ct_std), do_norm=True)

    # --- Pipeline 3: NO normalization (raw [0,1] repeat3) — what frozen saw if norm off ---
    r3 = run_pipeline("repeat3+no_norm (raw [0,1])", slc, mask, backbone, device,
                      None, None, do_norm=False)

    print("\n" + "=" * 60)
    print("SUMMARY — mean discrimination_ratio (higher = more separable fg/bg)")
    print("=" * 60)
    print(f"  repeat3 + imagenet_norm (CURRENT trainer):  {r1:.4f}")
    print(f"  repeat3 + ct_norm   (3ch identical):        {r2:.4f}")
    print(f"  repeat3 + no_norm   (raw [0,1]):            {r3:.4f}")
    print()
    print("Interpretation:")
    print("  ratio < ~0.05  -> features CANNOT distinguish organ from background")
    print("                    (linear probe has nothing to work with — root cause")
    print("                    is the feature pipeline, not the decoder/optimizer)")
    print("  ratio > ~0.2   -> features ARE separable; poor probe DSC then points")
    print("                    to decoder capacity / optimization / loss instead.")
    print()
    best = max(r1, r2, r3)
    if r1 < 0.05 and best > r1 * 1.5:
        print(f"  >> CURRENT pipeline is near-zero ({r1:.4f}) but an alternative reaches")
        print(f"     {best:.4f} — the input normalization/channel handling is the bottleneck.")
    elif r1 < 0.05:
        print(f"  >> All pipelines near-zero (max {best:.4f}). Backbone features carry no")
        print("     usable signal for this organ on 2D slices — investigate the backbone")
        print("     weights or the 2D-slice-no-3D-context assumption.")
    else:
        print(f"  >> CURRENT pipeline shows separability ({r1:.4f}) — features are usable;")
        print("     poor probe DSC points to decoder/optimization, not the features.")


if __name__ == "__main__":
    main()
