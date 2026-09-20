"""Diagnostic 3: WHY does T2b get patch val dice 0.96 but whole-volume dice 0.43?

Hypothesis: validation patches are 50% force-fg (T2b oversample=0.5) so the
reported 'Pseudo dice 0.96' is dominated by kidney-centered patches. The OTHER
half (background patches, sampled randomly) may produce huge FP, but their
contribution to the global tp/fp/fn dice is small because they have few ref fg
voxels. In whole-volume sliding-window inference, MOST patches are
background-dominant -> the per-patch FP accumulates -> 100k+ FP.

This script:
1) Loads a T2b checkpoint + a validation case (preprocessed npz).
2) Extracts ONE kidney-centered 96^3 patch and SEVERAL background 96^3 patches
   (far from kidney center).
3) Runs the network (fp32, eval mode) on each patch. Reports predicted fg voxel
   count per patch type. If background patches predict many fg voxels -> that's
   the FP source in whole-volume inference.
4) Compare to exp4 checkpoint on the SAME patches.
"""
import os, sys, json
import numpy as np
import torch

os.environ.setdefault("nnUNet_def_n_proc", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# env paths
os.environ.setdefault("nnUNet_raw",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed",
                      "/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results",
                      "/home/wenwen_zhang/kidney_experiments_portable/results/nnUNet_results")
os.environ.setdefault("KIDNEY_EXT_DIR",
                      "/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer")


def load_case_npz(case):
    base = os.path.expanduser(
        "~/kidney_experiments_portable/data/nnUNet_preprocessed/"
        "Dataset901_KidneyLeft5shot/nnUNetPlans_3d_96")
    path = os.path.join(base, case + ".npz")
    d = np.load(path)
    return {'data': d['data'], 'seg': d['seg']}


def get_fg_center(seg):
    coords = np.argwhere(seg[0] > 0)
    if len(coords) == 0:
        return np.array(seg.shape[1:]) // 2
    return coords.mean(0).astype(int)


def extract_patch(data, seg, center, ps=96):
    """Extract ps^3 patch centered at `center`, zero-pad if out of bounds."""
    shp = np.array(data.shape[1:])
    lbs = np.clip(center - ps // 2, 0, None)
    ubs = np.minimum(shp, lbs + ps)
    pad_lo = np.clip(ps // 2 - center, 0, None)
    pad_hi = (ps - (ubs - lbs)) - pad_lo
    pad_hi = np.clip(pad_hi, 0, None)
    sl = tuple([slice(0, data.shape[0])] + [slice(l, u) for l, u in zip(lbs, ubs)])
    d = data[sl]
    s = seg[sl]
    pad = ((0, 0), (pad_lo[0], pad_hi[0]), (pad_lo[1], pad_hi[1]), (pad_lo[2], pad_hi[2]))
    d = np.pad(d, pad, 'constant', constant_values=0)
    s = np.pad(s, pad, 'constant', constant_values=0)
    # ensure exact size
    d = d[:, :ps, :ps, :ps]
    s = s[:, :ps, :ps, :ps]
    return d, s


def load_model(trainer, config='3d_96', ckpt='checkpoint_best.pth'):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    model_dir = join(os.environ["nnUNet_results"],
                     "Dataset901_KidneyLeft5shot",
                     f"{trainer}__nnUNetPlans__{config}")
    # ensure trainer module imported (registers build_network_architecture)
    import importlib
    mod = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.kidney_252_trainer")
    _ = getattr(mod, trainer)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=False, device=torch.device("cuda"),
                           verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(0,),
                                              checkpoint_name=ckpt)
    pred.network = pred.network.to(torch.device("cuda")).eval()
    return pred.network, pred


def run_patches(net, data, seg, ps=96):
    net.eval()
    fg_c = get_fg_center(seg)
    shp = np.array(data.shape[1:])
    # kidney-centered patch
    d_k, s_k = extract_patch(data, seg, fg_c, ps)
    # background patches: corners far from fg center
    bg_centers = [
        np.array([ps // 2, ps // 2, ps // 2]),               # top-front-left corner
        np.array([shp[0] - ps // 2, shp[1] - ps // 2, shp[2] - ps // 2]),  # opposite
        np.array([ps // 2, shp[1] - ps // 2, ps // 2]),
    ]
    results = []
    with torch.no_grad():
        # kidney patch
        x = torch.from_numpy(d_k[None]).float().cuda()
        out = net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = out.argmax(1)[0].cpu().numpy()
        results.append(('kidney', int(s_k[0].sum()), int(pred.sum()),
                        int(np.logical_and(pred > 0, s_k[0] > 0).sum())))
        # background patches
        for i, c in enumerate(bg_centers):
            if np.any(c < 0) or np.any(c > shp):
                continue
            d_b, s_b = extract_patch(data, seg, c, ps)
            x = torch.from_numpy(d_b[None]).float().cuda()
            out = net(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pred = out.argmax(1)[0].cpu().numpy()
            results.append((f'bg{i}', int(s_b[0].sum()), int(pred.sum()),
                            int(np.logical_and(pred > 0, s_b[0] > 0).sum())))
    return results, d_k, s_k


def main():
    case = 's1031'   # a validation case
    print(f"### Diagnostic 3: per-patch FP (kidney vs background) on {case} ###")
    case_data = load_case_npz(case)
    data = case_data['data']   # (C, D, H, W)
    seg = case_data['seg']     # (C, D, H, W)
    # normalize data like CTNormalization (z-score) - already done in preprocessed npz
    print(f"  data shape {data.shape} seg fg voxels={int((seg>0).sum())} "
          f"data range [{data.min():.1f},{data.max():.1f}] mean {data.mean():.1f}")

    for trainer in ['flexict3d_252_Trainer', 'flexict3d_aug_252_Trainer']:
        print(f"\n--- {trainer} (checkpoint_best) ---")
        try:
            net, _ = load_model(trainer)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        results, d_k, s_k = run_patches(net, data, seg)
        print(f"  {'patch':<10} {'ref_fg':>8} {'pred_fg':>10} {'TP':>8} {'FP':>10}")
        for name, ref, pred, tp in results:
            fp = pred - tp
            tag = ' <-- BG patch predicts fg!' if (name.startswith('bg') and pred > 1000) else ''
            print(f"  {name:<10} {ref:>8} {pred:>10} {tp:>8} {fp:>10}{tag}")
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
