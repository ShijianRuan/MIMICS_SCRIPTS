"""右肾专用流式协调器: 只预测右肾(906已收敛), skip看right是否存在.
独立REMOTE_TMP(stream_r)和GPU, 与左肾协调器隔离."""
import os,sys,glob,subprocess,time,atexit,signal
Z_ROOT=r"Z:\ImageAnalysisData\1-CT\Segmentation\data\Totalsegmentator_dataset_v201"
R_RIGHT=r"R:\kidney_pred_3d\right"
REMOTE="wenwen_zhang@10.9.87.50"; PORT="11208"
REMOTE_TMP="/home/wenwen_zhang/kidney_experiments_portable/tmp/stream_r"
REMOTE_SCRIPT="/home/wenwen_zhang/kidney_experiments_portable/code/remote_right_only.py"
FAIL_LOG=r"R:\kidney_pred_3d\fail_log_right.txt"
GPU=os.environ.get("STREAM_GPU","1")
KILL_REMOTE='for p in $(ps aux|grep remote_right_only|grep -v grep|awk "{print \\$2}"); do kill -9 $p 2>/dev/null; done'

def sh(cmd):
    return subprocess.run(cmd,shell=True,capture_output=True,text=True)

def kill_remote_orphans():
    try: sh(f'ssh -p {PORT} {REMOTE} "{KILL_REMOTE}"')
    except: pass

proc=None
def cleanup():
    global proc
    try:
        if proc and proc.poll() is None:
            proc.stdin.close(); proc.terminate()
    except: pass
    kill_remote_orphans()
atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(0)))

def main():
    kill_remote_orphans(); time.sleep(2)
    cases=sorted(glob.glob(os.path.join(Z_ROOT,"*","ct.nii.gz")))
    print(f"右肾 共{len(cases)}例 GPU={GPU}",flush=True)
    os.makedirs(R_RIGHT,exist_ok=True)
    sh(f'ssh -p {PORT} {REMOTE} "mkdir -p {REMOTE_TMP}; rm -f {REMOTE_TMP}/*"')
    env=f'source ~/miniconda/etc/profile.d/conda.sh; conda activate nnUnet; export KIDNEY_EXT_DIR=$HOME/kidney_experiments_portable/code/ext_trainer FLEXICT3D_CKPT=$HOME/kidney_experiments_portable/weights/flexict_3d/model.safetensors CUDA_VISIBLE_DEVICES={GPU} REMOTE_TMP={REMOTE_TMP}'
    cmd=f'ssh -p {PORT} {REMOTE} \'{env}; cd ~/kidney_experiments_portable; python3 -u {REMOTE_SCRIPT}\''
    global proc
    proc=subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    while True:
        ln=proc.stdout.readline()
        if not ln: print("远程进程意外退出",flush=True); return
        print("[remote]",ln.strip(),flush=True)
        if "READY_906" in ln: break
    for i,c in enumerate(cases):
        cid=os.path.basename(os.path.dirname(c))
        orr=os.path.join(R_RIGHT,cid+".nii.gz")
        if os.path.exists(orr):
            if (i+1)%100==0: print(f"[{i+1}/{len(cases)}] skip到{cid}",flush=True)
            continue
        rct=f"{REMOTE_TMP}/{cid}_ct.nii.gz"; rrr=f"{REMOTE_TMP}/{cid}_right.nii.gz"
        try:
            t0=time.time()
            r=sh(f'scp -P {PORT} "{c}" {REMOTE}:{rct}')
            if r.returncode!=0: raise Exception("upload ct fail")
            proc.stdin.write(f"{cid} {rct}\n"); proc.stdin.flush()
            got=False
            while True:
                ln=proc.stdout.readline()
                if not ln: raise Exception("远程进程退出")
                if f"DONE {cid}" in ln or f"FAIL {cid}" in ln:
                    print(f"[{i+1}/{len(cases)}] {ln.strip()} {time.time()-t0:.0f}s",flush=True); got=True; break
            if not got: raise Exception("no response")
            if not os.path.exists(orr): sh(f'scp -P {PORT} {REMOTE}:{rrr} "{orr}"')
            sh(f'ssh -p {PORT} {REMOTE} "rm -f {rct} {rrr}"')
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL: {e}",flush=True)
            with open(FAIL_LOG,"a") as f: f.write(f"{cid}: {e}\n")
            sh(f'ssh -p {PORT} {REMOTE} "rm -f {rct} {rrr}"')
            if "远程进程退出" in str(e):
                print("重新拉起远程进程...",flush=True)
                kill_remote_orphans(); time.sleep(2)
                proc=subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                while True:
                    ln=proc.stdout.readline()
                    if not ln: break
                    if "READY_906" in ln: break
    try: proc.stdin.close()
    except: pass
    print("ALL DONE",flush=True)

if __name__=="__main__":
    main()
