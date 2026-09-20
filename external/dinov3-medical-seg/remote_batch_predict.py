"""远程批量预测: 常驻加载模型,循环处理stdin传入的case(每行: cid ct_remote_path).
模型加载一次,避免每例重载. 结果存 REMOTE_TMP/<cid>_left.nii.gz / _right.nii.gz."""
import os,sys,glob,shutil,tempfile
import numpy as np
import nibabel as nib
import torch
EXT_DIR=os.environ.get("KIDNEY_EXT_DIR","/home/wenwen_zhang/kidney_experiments_portable/code/ext_trainer")
sys.path.insert(0,EXT_DIR); sys.path.insert(0,os.path.join(EXT_DIR,"dinov3"))
os.environ.setdefault("nnUNet_compile","0")
os.environ.setdefault("nnUNet_raw","/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed","/home/wenwen_zhang/kidney_experiments_portable/data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results","/home/wenwen_zhang/kidney_experiments_portable/results/nnUNet_results")
RES="/home/wenwen_zhang/kidney_experiments_portable/results/nnUNet_results"
TMP=os.environ.get("REMOTE_TMP","/home/wenwen_zhang/kidney_experiments_portable/tmp/stream")

def get_predictor(dataset):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    mdir=os.path.join(RES,f"Dataset{dataset}","flexict3d_252_Trainer__nnUNetPlans__3d_fullres")
    if not os.path.exists(os.path.join(mdir,"dataset.json")): return None
    p=nnUNetPredictor(tile_step_size=0.5,use_gaussian=True,use_mirroring=False,device=torch.device("cuda"),verbose=False,allow_tqdm=False)
    p.initialize_from_trained_model_folder(mdir,use_folds=(0,),checkpoint_name="checkpoint_best.pth")
    return p

def keep_side_cc(mask,side):
    from scipy.ndimage import label
    if mask.sum()==0: return mask
    lab,n=label(mask)
    if n==0: return mask
    mid=mask.shape[0]//2; keep=[]
    for i in range(1,n+1):
        comp=(lab==i); cx=np.argwhere(comp)[:,0].mean()
        if side=="left" and cx<mid: keep.append((i,comp.sum()))
        elif side=="right" and cx>=mid: keep.append((i,comp.sum()))
    if not keep: keep=[(i,(lab==i).sum()) for i in range(1,n+1)]
    best=max(keep,key=lambda x:x[1])[0]
    return (lab==best).astype(np.uint8)

def predict_one(p,side,ct_path,out_path):
    if p is None: return False
    td=tempfile.mkdtemp(); od=tempfile.mkdtemp()
    try:
        shutil.copy(ct_path,os.path.join(td,"case_0000.nii.gz"))
        p.predict_from_files(td,od,save_probabilities=False,overwrite=True,num_processes_segmentation_export=1)
        preds=glob.glob(os.path.join(od,"*.nii.gz"))
        if not preds: return False
        pred=nib.load(preds[0]).get_fdata()>0
        pred=keep_side_cc(pred,side)
        orig=nib.load(ct_path)
        out=nib.Nifti1Image(pred.astype(np.uint8),orig.affine,orig.header)
        out.header.set_data_dtype(np.uint8); nib.save(out,out_path)
        return True
    finally:
        shutil.rmtree(td,ignore_errors=True); shutil.rmtree(od,ignore_errors=True)

if __name__=="__main__":
    print("LOADING_MODELS",flush=True)
    p_left=get_predictor("905_KidneyLeftWhole")
    p_right=get_predictor("906_KidneyRightWhole")  # 906已收敛EMA0.955
    print("MODELS_READY",flush=True)
    for line in sys.stdin:
        line=line.strip()
        if not line: continue
        parts=line.split(" ",1)
        if len(parts)<2: continue
        cid,ct=parts
        ol=f"{TMP}/{cid}_left.nii.gz"; orr=f"{TMP}/{cid}_right.nii.gz"
        try:
            r1=predict_one(p_left,"left",ct,ol)
            r2=predict_one(p_right,"right",ct,orr)
            print(f"DONE {cid} left={r1} right={r2}",flush=True)
        except Exception as e:
            print(f"FAIL {cid} {e}",flush=True)
