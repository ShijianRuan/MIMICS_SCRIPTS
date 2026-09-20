#!/usr/bin/env bash
# Remote smoke test: train a few epochs to verify the pipeline runs end-to-end.
# Usage (on remote): bash remote_smoke.sh <2d|3d_fullres> [gpu] [num_epochs]
set -e
cd ~/flexict-finetune
source ~/miniconda/etc/profile.d/conda.sh
conda activate nnUnet
# set NUM_EPOCHS BEFORE setup_env.sh (which also defaults NUM_EPOCHS)
export NUM_EPOCHS="${3:-2}"
export CUDA_VISIBLE_DEVICES="${2:-2}"
source scripts/setup_env.sh
export FLEXICT_EXT_DIR="$PWD/flexict"
CFG="$1"
case "$CFG" in
    2d)         TRAINER=flexict2d_Trainer ;;
    3d_fullres) TRAINER=flexict3d_Trainer ;;
    *) echo "config must be 2d or 3d_fullres"; exit 1 ;;
esac
echo "=== SMOKE: $CFG $TRAINER ${NUM_EPOCHS}ep GPU$CUDA_VISIBLE_DEVICES ==="
nnUNetv2_train 907 "$CFG" 0 -tr "$TRAINER"
echo "=== SMOKE $CFG DONE ==="
