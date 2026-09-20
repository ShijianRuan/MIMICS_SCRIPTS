#!/usr/bin/env bash
# Uncertainty ranking: run trained 2D + 3D models on a Totalsegmentator-style
# source tree, then rank every case by per-voxel disagreement between the two
# models.
#
# Pipeline:
#   1. prepare_inputs.py  — link <source>/<case>/ct.nii.gz -> <case>_0000.nii.gz
#   2. nnUNetv2_predict 2D  -> preds_2d/<case>_0000.nii.gz      (GPU $GPU_2D)
#   3. nnUNetv2_predict 3D  -> preds_3d_fullres/<case>_0000.nii.gz (GPU $GPU_3D)
#   4. run_uncertainty.py   — 2D vs 3D disagreement -> uncertainty_ranking.csv
#
# 2D and 3D prediction run concurrently on separate GPUs (3D is the bottleneck).
# Usage (on remote):
#   bash scripts/run_uncertainty_rank.sh <source> <dataset_id> [max_cases] [gpu_2d] [gpu_3d]
# Example (smoke, 5 cases):
#   bash scripts/run_uncertainty_rank.sh /mnt/v201 907 5 0 2
# Example (full 1228):
#   bash scripts/run_uncertainty_rank.sh /mnt/v201 907 0 0 2
set -e

SOURCE="${1:?usage: $0 <source> <dataset_id> [max_cases] [gpu_2d] [gpu_3d]}"
DSID="${2:?dataset id required, e.g. 907}"
MAX_CASES="${3:-0}"
GPU_2D="${4:-0}"
GPU_3D="${5:-2}"

cd ~/flexict-finetune
source ~/miniconda/etc/profile.d/conda.sh
conda activate nnUnet
source scripts/setup_env.sh
export FLEXICT_EXT_DIR="$PWD/flexict"

WORK=~/flexict-finetune/unc_work_${DSID}
INPUT="$WORK/infer_input"
PREP_2D="$WORK/preds_2d"
PREP_3D="$WORK/preds_3d_fullres"
UNC_OUT="$WORK/uncertainty"
CASES_TXT="$INPUT/case_list.txt"
mkdir -p "$WORK" "$INPUT" "$PREP_2D" "$PREP_3D" "$UNC_OUT"

# ---- 1. prepare inputs ----
PREP_ARGS=(--source "$SOURCE" --out "$INPUT" --cases-file "$CASES_TXT")
if [ "$MAX_CASES" != "0" ]; then
    PREP_ARGS+=(--max-cases "$MAX_CASES")
fi
echo "=== [1/4] prepare inputs $(date) ==="
python3 scripts/prepare_inputs.py "${PREP_ARGS[@]}"

# ---- 2 & 3. predict 2D + 3D concurrently on separate GPUs ----
echo "=== [2-3/4] predict 2D (GPU $GPU_2D) + 3D (GPU $GPU_3D) concurrently $(date) ==="
CUDA_VISIBLE_DEVICES="$GPU_2D" nnUNetv2_predict -i "$INPUT" -o "$PREP_2D" \
    -d "$DSID" -c 2d -f 0 -tr flexict2d_Trainer -chk checkpoint_best.pth --disable_tta \
    > "$WORK/predict_2d.log" 2>&1 &
PID_2D=$!

CUDA_VISIBLE_DEVICES="$GPU_3D" nnUNetv2_predict -i "$INPUT" -o "$PREP_3D" \
    -d "$DSID" -c 3d_fullres -f 0 -tr flexict3d_Trainer -chk checkpoint_best.pth --disable_tta \
    > "$WORK/predict_3d.log" 2>&1 &
PID_3D=$!

echo "  2D predict PID=$PID_2D (log $WORK/predict_2d.log)"
echo "  3D predict PID=$PID_3D (log $WORK/predict_3d.log)"

wait $PID_2D && echo "  2D predict done $(date)" || { echo "  2D predict FAILED — see $WORK/predict_2d.log"; exit 1; }
wait $PID_3D && echo "  3D predict done $(date)" || { echo "  3D predict FAILED — see $WORK/predict_3d.log"; exit 1; }

# ---- 4. uncertainty ranking ----
echo "=== [4/4] uncertainty ranking $(date) ==="
python3 scripts/run_uncertainty.py \
    --mask-dirs "$PREP_2D" "$PREP_3D" \
    --cases-file "$CASES_TXT" \
    --out "$UNC_OUT" \
    --method disagreement \
    --filename-template "{case}_0000.nii.gz" \
    --sort-key integrated \
    --target-labels 1
echo "=== DONE $(date) ==="
echo "ranking CSV: $UNC_OUT/uncertainty_ranking.csv"
