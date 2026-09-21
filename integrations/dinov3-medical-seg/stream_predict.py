"""本地流式协调器: 逐例上传Z盘ct到远程→远程预测905左+906右→传回R盘→清理远程临时.
无需人为监控: 断点续跑(已存在skip), 失败跳过记录, 远程磁盘恒定(~1例)."""
import os,sys,glob,shutil,subprocess,time
Z_ROOT=r"Z:\ImageAnalysisData\1-CT\Segmentation\data\Totalsegmentator_dataset_v201"
R_LEFT=r"R:\kidney_pred_3d\left"; R_RIGHT=r"R:\kidney_pred_3d\right"
REMOTE="wenwen_zhang@10.9.87.50"; PORT="11208"
REMOTE_TMP="/home/wenwen_zhang/kidney_experiments_portable/tmp/stream"
REMOTE_SCRIPT="/home/wenwen_zhang/kidney_experiments_portable/code/remote_predict_one.py"
FAIL_LOG=r"R:\kidney_pred_3d\fail_log.txt"

def sh(cmd):
    return subprocess.run(cmd,shell=True,capture_output=True,text=True)

def main():
    cases=sorted(glob.glob(os.path.join(Z_ROOT,"*","ct.nii.gz")))
    print(f"共{len(cases)}例",flush=True)
    os.makedirs(R_LEFT,exist_ok=True); os.makedirs(R_RIGHT,exist_ok=True)
    # 预建远程tmp
    sh(f'ssh -p {PORT} {REMOTE} "mkdir -p {REMOTE_TMP}"')
    for i,c in enumerate(cases):
        cid=os.path.basename(os.path.dirname(c))
        ol=os.path.join(R_LEFT,cid+".nii.gz"); orr=os.path.join(R_RIGHT,cid+".nii.gz")
        if os.path.exists(ol) and os.path.exists(orr):
            if (i+1)%50==0: print(f"[{i+1}/{len(cases)}] {cid} skip",flush=True)
            continue
        rct=f"{REMOTE_TMP}/{cid}_ct.nii.gz"; rrl=f"{REMOTE_TMP}/{cid}_left.nii.gz"; rrr=f"{REMOTE_TMP}/{cid}_right.nii.gz"
        try:
            t0=time.time()
            # 1.上传ct
            r=sh(f'scp -P {PORT} "{c}" {REMOTE}:{rct}')
            if r.returncode!=0: raise Exception("upload ct fail")
            # 2.远程预测
            env='source ~/miniconda/etc/profile.d/conda.sh; conda activate nnUnet; export KIDNEY_EXT_DIR=$HOME/kidney_experiments_portable/code/ext_trainer FLEXICT3D_CKPT=$HOME/kidney_experiments_portable/weights/flexict_3d/model.safetensors CUDA_VISIBLE_DEVICES=5'
            r=sh(f'ssh -p {PORT} {REMOTE} \'{env}; cd ~/kidney_experiments_portable; python3 {REMOTE_SCRIPT} {rct} {rrl} {rrr}\'')
            if r.returncode!=0: raise Exception(f"predict fail: {r.stderr[-200:]}")
            # 3.传回
            if not os.path.exists(ol): sh(f'scp -P {PORT} {REMOTE}:{rrl} "{ol}"')
            if not os.path.exists(orr): sh(f'scp -P {PORT} {REMOTE}:{rrr} "{orr}"')
            # 4.清理远程临时
            sh(f'ssh -p {PORT} {REMOTE} "rm -f {rct} {rrl} {rrr}"')
            print(f"[{i+1}/{len(cases)}] {cid} done {time.time()-t0:.0f}s",flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL: {e}",flush=True)
            with open(FAIL_LOG,"a") as f: f.write(f"{cid}: {e}\n")
            sh(f'ssh -p {PORT} {REMOTE} "rm -f {rct} {rrl} {rrr}"')
    print("ALL DONE",flush=True)

if __name__=="__main__":
    main()
