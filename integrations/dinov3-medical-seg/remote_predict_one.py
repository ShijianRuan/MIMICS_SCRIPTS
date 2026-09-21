"""远程单例预测: 读stdin的ct路径(已上传到远程tmp), 预测905左+906右, 后处理, 输出mask到指定路径.
用法: python remote_predict_one.py <ct_remote_path> <out_left_remote> <out_right_remote>
模型不存在则对应输出跳过."""
import os,sys,glob,shutil
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

_predictors={}
def get_predictor(dataset):
    if dataset in _predictors: return _predictors[dataset]
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    mdir=os.path.join(RES,f"Dataset{dataset}","flexict3d_252_Trainer__nnUNetPlans__3d_fullres")
    if not os.path.exists(os.path.join(mdir,"dataset.json")): 
        _predictors[dataset]=None; return None
    p=nnUNetPredictor(tile_step_size=0.5,use_gaussian=True,use_mirroring=False,device=torch.device("cuda"),verbose=False,allow_tqdm=False)
    p.initialize_from_trained_model_folder(mdir,use_folds=(0,),checkpoint_name="checkpoint_best.pth")
    _predictors[dataset]=p; return p

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

def predict_one(dataset,side,ct_path,out_path):
    p=get_predictor(dataset)
    if p is None: return False
    import tempfile
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
    ct=sys.argv[1]; ol=sys.argv[2]; orr=sys.argv[3]
    r1=predict_one("905_KidneyLeftWhole","left",ct,ol)
    r2=predict_one("906_KidneyRightWhole","right",ct,orr)
    print(f"RESULT left={r1} right={r2}")
