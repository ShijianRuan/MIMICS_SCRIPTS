#!/usr/bin/env bash
# Predict with a trained FlexiCT model.
#
# Usage:  bash scripts/run_predict.sh <dataset_id> <config> <input_dir> <output_dir> [fold]
#   config: 2d | 3d_fullres
# Example:
#   bash scripts/run_predict.sh 907 2d $nnUNet_raw/Dataset907_LiverFS/imagesTs ./preds
#
# Run `source scripts/setup_env.sh` first. TTA is disabled (the recipe was
# validated with --disable_tta; for single-side targets mirroring TTA can flip
# the lateral axis and create contralateral false positives).
set -e

DSID="$1"; CONFIG="$2"; INDIR="$3"; OUTDIR="$4"; FOLD="${5:-0}"
[ -z "$DSID" ] || [ -z "$CONFIG" ] || [ -z "$INDIR" ] || [ -z "$OUTDIR" ] && {
    echo "usage: $0 <dataset_id> <2d|3d_fullres> <input_dir> <output_dir> [fold]"; exit 1; }

case "$CONFIG" in
    2d)         TRAINER="flexict2d_Trainer" ;;
    3d_fullres) TRAINER="flexict3d_Trainer" ;;
    *) echo "config must be '2d' or '3d_fullres' (got '$CONFIG')"; exit 1 ;;
esac

mkdir -p "$OUTDIR"
echo "===== predict: Dataset$DSID $CONFIG fold$FOLD  $TRAINER ====="
nnUNetv2_predict -i "$INDIR" -o "$OUTDIR" -d "$DSID" -c "$CONFIG" -f "$FOLD" \
    -tr "$TRAINER" -chk checkpoint_best.pth --disable_tta
echo "===== done. predictions in $OUTDIR ====="
