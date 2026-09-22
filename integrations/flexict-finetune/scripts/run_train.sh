#!/usr/bin/env bash
# Train a FlexiCT few-shot model with nnU-Net.
#
# Usage:  bash scripts/run_train.sh <dataset_id> <config> [fold]
#   config: 2d | 3d_fullres
#   fold:   0 (default)
# Example:
#   NUM_EPOCHS=150 bash scripts/run_train.sh 907 2d
#   NUM_EPOCHS=150 bash scripts/run_train.sh 907 3d_fullres
#
# Run `source scripts/setup_env.sh` first. The trainer is discovered via the
# nnUNet_extTrainer env var (nnunetv2 >= 2.6) or must be installed into
# nnU-Net's trainer dir (see README → "Install the trainer into nnU-Net").
set -e

DSID="$1"; CONFIG="$2"; FOLD="${3:-0}"
[ -z "$DSID" ] || [ -z "$CONFIG" ] && { echo "usage: $0 <dataset_id> <2d|3d_fullres> [fold]"; exit 1; }

case "$CONFIG" in
    2d)         TRAINER="flexict2d_Trainer" ;;
    3d_fullres) TRAINER="flexict3d_Trainer" ;;
    *) echo "config must be '2d' or '3d_fullres' (got '$CONFIG')"; exit 1 ;;
esac

echo "===== train: Dataset$DSID $CONFIG fold$FOLD  $TRAINER  ${NUM_EPOCHS}ep ====="
nnUNetv2_train "$DSID" "$CONFIG" "$FOLD" -tr "$TRAINER"
echo "===== done. checkpoints under $nnUNet_results/Dataset${DSID}_*/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}/ ====="
