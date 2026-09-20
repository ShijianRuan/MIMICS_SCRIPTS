"""Stream cases to a remote GPU box for 2D+3D inference, rank by uncertainty.

Avoids staging all 1228 raw CTs on the remote at once (its disk is full): each
case is scp'd over, predicted with both models, scp'd back, and deleted on the
remote — so remote disk only ever holds a few cases. Predictions accumulate
locally; once all cases are done, the per-voxel uncertainty + ranking runs
locally (numpy/scipy/nibabel only, no GPU needed).

This is the streaming variant of ``run_uncertainty_rank.sh`` for when the data
lives where there is no GPU, and the GPU box has no room for the full dataset.

Flow per case:
    1. scp <source>/<case>/ct.nii.gz -> remote:<stage>/<case>_0000.nii.gz
    2. remote: nnUNetv2_predict 2D  -> <stage>/preds_2d/<case>_0000.nii.gz
    3. remote: nnUNetv2_predict 3D  -> <stage>/preds_3d/<case>_0000.nii.gz
    4. scp both preds back -> local <local_masks>/{preds_2d,preds_3d}/<case>.nii.gz
    5. remote: rm the case + its preds (free the stage dir)
After all cases: run_uncertainty logic locally -> uncertainty_ranking.csv

Usage:
    python scripts/stream_uncertainty_rank.py \
        --source Z:/.../Totalsegmentator_dataset_v201 \
        --remote wenwen_zhang@10.9.87.50 -p 11208 \
        --remote-root '~/flexict-finetune' --dataset-id 907 \
        --local-masks ./unc_masks --out ./unc_out \
        --gpu-2d 0 --gpu-3d 2

    # smoke (first 5 cases):
    ... --max-cases 5
    # resume from a case id (skip already-done):
    ... --start-case s0500
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run(cmd, check=True, capture=False):
    """Run a command (string or list). Returns CompletedProcess."""
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    return subprocess.run(cmd, shell=shell, check=check,
                          text=True, capture_output=capture)


def ssh(remote, port, remote_cmd, check=True, capture=False):
    """Run a command on the remote over ssh."""
    full = ["ssh", "-p", str(port), remote, remote_cmd]
    return subprocess.run(full, check=check, text=True, capture_output=capture)


def scp_to(remote, port, local, remote_dst):
    """scp local -> remote:dst ."""
    subprocess.run(["scp", "-P", str(port), local, f"{remote}:{remote_dst}"],
                   check=True)


def scp_from(remote, port, remote_src, local_dst):
    """scp remote:src -> local."""
    subprocess.run(["scp", "-P", str(port), f"{remote}:{remote_src}", local_dst],
                   check=True)


def scp_from_retry(remote, port, remote_src, local_dst, retries=3):
    """scp remote:src -> local, retrying on transient failures."""
    last = None
    for attempt in range(retries):
        r = subprocess.run(["scp", "-P", str(port), f"{remote}:{remote_src}",
                            local_dst], check=False, text=True, capture_output=True)
        if r.returncode == 0:
            return
        last = r.stderr.strip()
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"scp failed after {retries} tries: {remote_src}: {last}")


def iter_cases(source):
    for cdir in sorted(p for p in source.iterdir() if p.is_dir()):
        ct = cdir / "ct.nii.gz"
        if ct.exists():
            yield cdir.name, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source root: <source>/<case>/ct.nii.gz")
    ap.add_argument("--remote", required=True, help="user@host")
    ap.add_argument("-p", "--port", type=int, default=22, help="ssh port")
    ap.add_argument("--remote-root", default="~/flexict-finetune", help="remote repo root")
    ap.add_argument("--dataset-id", type=int, required=True, help="nnU-Net dataset id (e.g. 907)")
    ap.add_argument("--local-masks", required=True, help="local dir to accumulate 2D/3D preds")
    ap.add_argument("--out", required=True, help="local dir for uncertainty output + CSV")
    ap.add_argument("--gpu-2d", default="0")
    ap.add_argument("--gpu-3d", default="2")
    ap.add_argument("--max-cases", type=int, default=0, help="cap (0 = all)")
    ap.add_argument("--start-case", default="", help="skip until this case id (resume)")
    ap.add_argument("--method", default="disagreement")
    ap.add_argument("--sort-key", default="integrated")
    ap.add_argument("--target-labels", nargs="*", type=int, default=[1])
    ap.add_argument("--rank-only", action="store_true",
                    help="skip inference; just rank the masks already in --local-masks")
    args = ap.parse_args()

    source = Path(args.source)
    local_masks = Path(args.local_masks)
    out_dir = Path(args.out)
    (local_masks / "preds_2d").mkdir(parents=True, exist_ok=True)
    (local_masks / "preds_3d").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_file = out_dir / "case_list.txt"

    if not args.rank_only:
        stage = f"{args.remote_root}/stream_stage_{args.dataset_id}"
        # fresh stage dir on remote
        ssh(args.remote, args.port,
            f"rm -rf {stage} && mkdir -p {stage}/preds_2d {stage}/preds_3d")

        cases = list(iter_cases(source))
        if args.start_case:
            idx = next((i for i, (c, _) in enumerate(cases) if c == args.start_case), 0)
            cases = cases[idx:]
        if args.max_cases:
            cases = cases[:args.max_cases]

        print(f"streaming {len(cases)} cases: {args.remote} (2D=GPU{args.gpu_2d}, 3D=GPU{args.gpu_3d})")
        t0 = time.time()
        done_ids = []
        for i, (cid, ct) in enumerate(cases, 1):
            local2 = local_masks / "preds_2d" / f"{cid}.nii.gz"
            local3 = local_masks / "preds_3d" / f"{cid}.nii.gz"
            if local2.exists() and local3.exists():
                done_ids.append(cid)
                print(f"[{i}/{len(cases)}] {cid} skip (cached)")
                continue

            rct = f"{stage}/{cid}_0000.nii.gz"
            # nnU-Net predict names outputs after the case id, dropping the
            # channel suffix: input s0000_0000.nii.gz -> output s0000.nii.gz
            r2 = f"{stage}/preds_2d/{cid}.nii.gz"
            r3 = f"{stage}/preds_3d/{cid}.nii.gz"

            try:
                # 1. upload CT
                scp_to(args.remote, args.port, str(ct), rct)

                # 2+3. launch predict detached (nohup) — the remote script runs
                #     2D then 3D under a flock, writes a DONE/ERROR marker. We
                #     poll the marker with short ssh calls instead of holding one
                #     ssh open for the (slow) 3D prediction, which avoids ssh
                #     timeouts and orphaned processes piling up.
                ssh(args.remote, args.port,
                    f"cd {args.remote_root} && nohup bash scripts/remote_predict_one.sh "
                    f"{stage} {args.dataset_id} {args.gpu_2d} {args.gpu_3d} "
                    f"> {stage}/launch.log 2>&1 &", check=False)

                # poll for DONE or ERROR marker (3D is ~1-2 min; allow up to 8 min)
                outcome = None
                deadline = time.time() + 480
                while time.time() < deadline:
                    time.sleep(10)
                    chk = ssh(args.remote, args.port,
                              f"if [ -f {stage}/DONE ]; then echo DONE; "
                              f"elif [ -f {stage}/ERROR ]; then echo ERROR; "
                              f"else echo RUN; fi", check=False, capture=True)
                    st = chk.stdout.strip()
                    if st in ("DONE", "ERROR"):
                        outcome = st
                        break
                if outcome != "DONE":
                    err = ""
                    if outcome == "ERROR":
                        err = ssh(args.remote, args.port, f"cat {stage}/ERROR",
                                  check=False, capture=True).stdout.strip()
                    raise RuntimeError(f"predict {outcome or 'timeout'} {err}")

                # 4. download preds (retry on transient scp errors)
                scp_from_retry(args.remote, args.port, r2, str(local2))
                scp_from_retry(args.remote, args.port, r3, str(local3))
                done_ids.append(cid)
                # persist case list periodically for resume
                if len(done_ids) % 10 == 0:
                    cases_file.write_text("\n".join(done_ids) + "\n", encoding="utf-8")

                el = time.time() - t0
                avg = el / i
                print(f"[{i}/{len(cases)}] {cid} done ({avg:.1f}s/case, "
                      f"ETA {avg*(len(cases)-i)/60:.0f}min)", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(cases)}] {cid} FAILED: {exc}", flush=True)
            finally:
                # 5. always clean remote stage for this case (never leave CT/preds).
                #    The flock in remote_predict_one.sh serializes; rm touches
                #    only this case's files + markers.
                ssh(args.remote, args.port,
                    f"rm -f {rct} {r2} {r3} {stage}/DONE {stage}/ERROR "
                    f"{stage}/.predict.lock {stage}/predict.log "
                    f"{stage}/predict_2d.log {stage}/predict_3d.log {stage}/launch.log",
                    check=False)

        cases_file.write_text("\n".join(done_ids) + "\n", encoding="utf-8")
        print(f"inference done: {len(done_ids)} cases in {(time.time()-t0)/60:.1f} min")
        # final cleanup of stage dir
        ssh(args.remote, args.port, f"rm -rf {stage}", check=False)
    else:
        done_ids = sorted(p.name[:-len(".nii.gz")] for p in (local_masks / "preds_2d").glob("*.nii.gz"))
        cases_file.write_text("\n".join(done_ids) + "\n", encoding="utf-8")

    # ---- local uncertainty ranking ----
    if not done_ids:
        print("no cases succeeded — skipping ranking")
        return
    print("\n=== uncertainty ranking (local) ===")
    from uncertainty import run_batch, analyze_uncertainty_dir
    report_csv = str(out_dir / "uncertainty_ranking.csv")
    run_batch(
        [str(local_masks / "preds_2d"), str(local_masks / "preds_3d")],
        str(out_dir), case_list_txt=str(cases_file),
        filename_template="{case}.nii.gz", method=args.method,
        target_labels=args.target_labels, save_components=False,
        save_consensus=False, save_plots=False, n_workers=1,
        log_path=str(out_dir / "uncertainty_batch.log"),
    )
    analyze_uncertainty_dir(str(out_dir), method=args.method,
                            sort_key=args.sort_key, report_csv=report_csv,
                            compute_boundary=False, compute_cc=True)
    print(f"\nranking -> {report_csv}")


if __name__ == "__main__":
    main()
