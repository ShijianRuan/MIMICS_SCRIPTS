#!/usr/bin/env bash
# Predict ONE case with both 2D and 3D FlexiCT models. Designed to be launched
# by the streaming orchestrator (stream_uncertainty_rank.py) per case.
#
# Serializes on a per-stage flock so a crashed/lingering previous run cannot
# collide, writes a DONE marker on success, and is safe to nohup (detached from
# the ssh that launched it). The orchestrator polls the marker rather than
# holding the ssh open for the (slow) 3D prediction.
#
# Usage (on remote):
#   bash scripts/remote_predict_one.sh <stage_dir> <dataset_id> <gpu_2d> <gpu_3d>
# Expects <stage_dir>/<case>_0000.nii.gz already uploaded. Writes:
#   <stage_dir>/preds_2d/<case>.nii.gz   (nnU-Net drops the _0000 suffix)
#   <stage_dir>/preds_3d/<case>.nii.gz
#   <stage_dir>/DONE            (empty file, on success)
#   <stage_dir>/ERROR           (error text, on failure)
set -u

STAGE="${1:?stage dir required}"
DSID="${2:?dataset id required}"
GPU_2D="${3:-0}"
GPU_3D="${4:-2}"

cd ~/flexict-finetune
source ~/miniconda/etc/profile.d/conda.sh
conda activate nnUnet
source scripts/setup_env.sh
export FLEXICT_EXT_DIR="$PWD/flexict"

# serialize: only one prediction run in this stage dir at a time
exec 9>"$STAGE/.predict.lock"
flock -x 9 || { echo "flock failed" > "$STAGE/ERROR"; exit 1; }

# find the single uploaded case (xxx_0000.nii.gz)
CT=$(ls "$STAGE"/*_0000.nii.gz 2>/dev/null | head -1)
if [ -z "$CT" ]; then
    echo "no *_0000.nii.gz found in $STAGE" > "$STAGE/ERROR"
    exit 1
fi
CASE=$(basename "$CT" _0000.nii.gz)

rm -f "$STAGE/DONE" "$STAGE/ERROR"
log() { echo "[$CASE] $*"; }

# Run 2D then 3D serially (each ~170-210s). Parallel was tried but caused a
# race: the orchestrator's poll loop sees the first DONE and uploads the next
# case's CT while the other predict is still writing, corrupting the stage.
# Serial is correct; the flock guarantees one remote_predict_one.sh at a time.
log "2D predict (GPU $GPU_2D)"
CUDA_VISIBLE_DEVICES="$GPU_2D" nnUNetv2_predict \
    -i "$STAGE" -o "$STAGE/preds_2d" -d "$DSID" -c 2d -f 0 \
    -tr flexict2d_Trainer -chk checkpoint_best.pth --disable_tta --disable_progress_bar \
    >> "$STAGE/predict_2d.log" 2>&1 || { echo "2D predict failed" > "$STAGE/ERROR"; exit 1; }

log "3D predict (GPU $GPU_3D)"
CUDA_VISIBLE_DEVICES="$GPU_3D" nnUNetv2_predict \
    -i "$STAGE" -o "$STAGE/preds_3d" -d "$DSID" -c 3d_fullres -f 0 \
    -tr flexict3d_Trainer -chk checkpoint_best.pth --disable_tta --disable_progress_bar \
    >> "$STAGE/predict_3d.log" 2>&1 || { echo "3D predict failed" > "$STAGE/ERROR"; exit 1; }

# verify outputs exist (nnU-Net drops the _0000 suffix)
P2="$STAGE/preds_2d/${CASE}.nii.gz"
P3="$STAGE/preds_3d/${CASE}.nii.gz"
if [ ! -s "$P2" ] || [ ! -s "$P3" ]; then
    echo "missing output: p2=$([ -s "$P2" ] && echo ok || echo MISS) p3=$([ -s "$P3" ] && echo ok || echo MISS)" > "$STAGE/ERROR"
    exit 1
fi

touch "$STAGE/DONE"
log "DONE"
