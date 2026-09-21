#!/usr/bin/env bash
# Kidney_left 0806 ablation suite — runner.
#
# Runs the 5 experiments (Exp 0, A, B, C, D) sequentially, each for 30 epochs,
# then offline-evaluates each best checkpoint on the fixed 3-case validation set
# (s1031, s1120, s1130). All logs and eval JSONs land under experiments/.
#
# Each 3D run is ~20-40 min on the RTX 3060 (4.5 GB VRAM, per_chunk_backward).
# Total ~3 hours. Safe to Ctrl-C between runs; resume by re-running (checkpoints
# are saved per epoch, but this script does not auto-resume — pass --resume
# manually if you need to continue a partially-trained run).
#
# Exp D (2.5D) is conditional: run it only if Exp A or Exp C beat 0.5877.
# It is last, so you can skip it by Ctrl-C after Exp C if neither won.
#
# Usage:
#   cd E:/mimics_script_offline/external/dinov3-medical-seg
#   bash scripts/run_kidney_0806_suite.sh
#
# Or run experiments one at a time (see the per-experiment commands below).

set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="E:/mimics_script_offline/nninteractive_env/python.exe"
CFG="config/kidney_0806"
VAL_IDS="s1031,s1120,s1130"
DATA="data/totalseg/kidney_left"

run_exp () {
  local name=$1
  local cfg=$2
  echo "============================================================"
  echo "TRAIN: $name ($cfg)"
  echo "============================================================"
  $PY -u scripts/train.py --config "$cfg" 2>&1 | tee "experiments/kidney_0806_${name}_train.log"
}

eval_exp () {
  local name=$1
  local cfg=$2
  local expdir=$3
  echo "============================================================"
  echo "EVAL:  $name"
  echo "============================================================"
  $PY -u scripts/evaluate_model.py \
    --config "$cfg" \
    --checkpoint "experiments/${expdir}/checkpoints/best_model.pth" \
    --data-root "$DATA" \
    --split Tr \
    --case-ids "$VAL_IDS" \
    --output "experiments/${expdir}/eval_3val.json" \
    --save-predictions 2>&1 | tee "experiments/kidney_0806_${name}_eval.log"
}

# Exp 0 — 3D single-level control (segformer3d, out_indices [11])
run_exp  "segformer_1level" "$CFG/kidney_3d_segformer_1level_5shot.yaml"
eval_exp "segformer_1level" "$CFG/kidney_3d_segformer_1level_5shot.yaml" \
  "kidney_0806_3d_segformer_1level_s0657_s0661_s0668_s0693_s0749"

# Exp A — 3D multi-scale hybrid (LEADING)
run_exp  "hybrid" "$CFG/kidney_3d_hybrid_5shot.yaml"
eval_exp "hybrid" "$CFG/kidney_3d_hybrid_5shot.yaml" \
  "kidney_0806_3d_hybrid_s0657_s0661_s0668_s0693_s0749"

# Exp B — 2D multi-scale (scale_aware2d)
run_exp  "scaleaware" "$CFG/kidney_2d_scaleaware_5shot.yaml"
eval_exp "scaleaware" "$CFG/kidney_2d_scaleaware_5shot.yaml" \
  "kidney_0806_2d_scaleaware_s0657_s0661_s0668_s0693_s0749"

# Exp C — 3D multi-scale volumetric (context3d_multiscale)
run_exp  "multiscale" "$CFG/kidney_3d_multiscale_5shot.yaml"
eval_exp "multiscale" "$CFG/kidney_3d_multiscale_5shot.yaml" \
  "kidney_0806_3d_multiscale_s0657_s0661_s0668_s0693_s0749"

# Exp D — 2.5D channel policy on the hybrid (CONDITIONAL: only if A or C > 0.5877).
# Review the eval JSONs above first; skip this block if neither won.
echo "============================================================"
echo "Exp D is conditional. Check eval_3val.json for Exp A and C."
echo "If either beat 0.5877, run Exp D; otherwise Ctrl-C now."
echo "============================================================"
read -r -p "Run Exp D (2.5D)? [y/N] " ans
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
  run_exp  "hybrid_2_5d" "$CFG/kidney_3d_hybrid_2_5d_5shot.yaml"
  eval_exp "hybrid_2_5d" "$CFG/kidney_3d_hybrid_2_5d_5shot.yaml" \
    "kidney_0806_3d_hybrid_2_5d_s0657_s0661_s0668_s0693_s0749"
fi

echo "============================================================"
echo "Suite complete. Eval JSONs:"
echo "  experiments/kidney_0806_3d_segformer_1level_*/eval_3val.json"
echo "  experiments/kidney_0806_3d_hybrid_*/eval_3val.json"
echo "  experiments/kidney_0806_2d_scaleaware_*/eval_3val.json"
echo "  experiments/kidney_0806_3d_multiscale_*/eval_3val.json"
echo "  experiments/kidney_0806_3d_hybrid_2_5d_*/eval_3val.json (if run)"
echo "Send these JSONs + the *_train.log files back for analysis."
echo "============================================================"
