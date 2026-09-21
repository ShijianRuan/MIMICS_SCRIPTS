"""本地预测Z盘v201所有ct.nii.gz: 905-3d(左肾)+906-3d(右肾),后处理区分左右连通域,存R盘."""
import os,sys,glob,shutil
import numpy as np
import nibabel as nib
import torch

# 环境路径
EXT_DIR = r"E:\mimics_script_offline\external\dinov3-medical-seg\ext_trainer"
for _p in (EXT_DIR, os.path.join(EXT_DIR,"dinov3")):
    if _p not in sys.path: sys.path.insert(0,_p)
os.environ.setdefault("nnUNet_compile","0")
os.environ.setdefault("KIDNEY_EXT_DIR",EXT_DIR)
os.environ.setdefault("FLEXICT3D_CKPT",r"E:\kidney_experiments_portable\weights\flexict_3d\model.safetensors")
os.environ.setdefault("nnUNet_raw",r"E:\kidney_experiments_portable\data\nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed",r"E:\kidney_experiments_portable\data\nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results",r"E:\kidney_experiments_portable\results\nnUNet_results")

Z_ROOT = r"Z:\ImageAnalysisData\1-CT\Segmentation\data\Totalsegmentator_dataset_v201"
R_LEFT = r"R:\kidney_pred_3d\left"
R_RIGHT = r"R:\kidney_pred_3d\right"
RES_BASE = r"E:\kidney_experiments_portable\results\nnUNet_results"

def load_predictor(dataset, trainer, config):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    mdir = os.path.join(RES_BASE, f"Dataset{dataset}", f"{trainer}__nnUNetPlans__{config}")
    if not os.path.exists(os.path.join(mdir,"dataset.json")):
        print(f"[warn] {dataset} 模型不存在,跳过"); return None
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        device=torch.device("cuda"), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(mdir, use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    return p

def keep_side_cc(mask, side):
    """保留指定侧最大连通域. side='left'保留图像左半, 'right'保留右半. 无法区分则保留最大."""
    from scipy.ndimage import label
    if mask.sum()==0: return mask
    lab,n = label(mask)
    if n==0: return mask
    # 图像左右轴: 通常nifti的axis0是L-R(需确认), 这里用axis0的中线分左右
    shp = mask.shape
    mid = shp[0]//2
    keep = []
    for i in range(1,n+1):
        comp = (lab==i)
        cx = np.argwhere(comp)[:,0].mean()
        if side=="left" and cx < mid: keep.append((i,comp.sum()))
        elif side=="right" and cx >= mid: keep.append((i,comp.sum()))
    if not keep:  # 无法区分(都在一侧),保留最大连通域
        sizes = [(i,(lab==i).sum()) for i in range(1,n+1)]
        keep = sizes
    best = max(keep, key=lambda x:x[1])[0]
    return (lab==best).astype(np.uint8)

def predict_one(predictor, inp_nii, out_nii, side, cid="case"):
    """预测单例+后处理+保存."""
    import tempfile
    td = tempfile.mkdtemp()
    try:
        std = os.path.join(td, cid + "_0000.nii.gz")
        shutil.copy(inp_nii, std)
        od = tempfile.mkdtemp()
        predictor.predict_from_files(td, od, save_probabilities=False, overwrite=True, num_processes_segmentation_export=1)
        # 读预测结果
        preds = glob.glob(os.path.join(od, "*.nii.gz"))
        if not preds: return False
        pred = nib.load(preds[0]).get_fdata()>0
        pred = keep_side_cc(pred, side)
        # 用原始nii的affine/header
        orig = nib.load(inp_nii)
        out = nib.Nifti1Image(pred.astype(np.uint8), orig.affine, orig.header)
        out.header.set_data_dtype(np.uint8)
        nib.save(out, out_nii)
        return True
    finally:
        shutil.rmtree(td, ignore_errors=True); shutil.rmtree(od, ignore_errors=True)

def main():
    import sys
    smoke = "--smoke" in sys.argv
    cases = sorted(glob.glob(os.path.join(Z_ROOT, "*", "ct.nii.gz")))
    print(f"找到 {len(cases)} 个 ct.nii.gz")
    # 左右模型
    p_left = load_predictor("905_KidneyLeftWhole","flexict3d_252_Trainer","3d_fullres")
    p_right = load_predictor("906_KidneyRightWhole","flexict3d_252_Trainer","3d_fullres")
    os.makedirs(R_LEFT, exist_ok=True); os.makedirs(R_RIGHT, exist_ok=True)
    if smoke: cases=cases[:2]
    for i,c in enumerate(cases):
        cid = os.path.basename(os.path.dirname(c))  # s0000 等
        ol = os.path.join(R_LEFT, cid+".nii.gz")
        orr = os.path.join(R_RIGHT, cid+".nii.gz")
        if os.path.exists(ol) and os.path.exists(orr):
            print(f"[{i+1}/{len(cases)}] {cid} skip"); continue
        try:
            if p_left is not None: predict_one(p_left, c, ol, "left", cid)
            if p_right is not None: predict_one(p_right, c, orr, "right", cid)
            print(f"[{i+1}/{len(cases)}] {cid} done")
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL: {e}")
    print("ALL DONE")

if __name__=="__main__":
    main()
