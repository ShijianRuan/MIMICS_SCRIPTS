#!/usr/bin/env bash
# Predict test set + evaluate. Usage (on remote): bash scripts/remote_eval.sh <2d|3d_fullres> [gpu]
set -e
cd ~/flexict-finetune
source ~/miniconda/etc/profile.d/conda.sh
conda activate nnUnet
export CUDA_VISIBLE_DEVICES="${2:-3}"
source scripts/setup_env.sh
export FLEXICT_EXT_DIR="$PWD/flexict"
CFG="$1"
case "$CFG" in
    2d)         TRAINER=flexict2d_Trainer ;;
    3d_fullres) TRAINER=flexict3d_Trainer ;;
    *) echo "config must be 2d or 3d_fullres"; exit 1 ;;
esac
RAW=~/flexict-finetune/data/nnUNet_raw/Dataset907_LiverFS
OUT=~/flexict-finetune/preds_${CFG}
mkdir -p "$OUT"
echo "=== PREDICT $CFG test ==="
nnUNetv2_predict -i "$RAW/imagesTs" -o "$OUT" -d 907 -c "$CFG" -f 0 \
    -tr "$TRAINER" -chk checkpoint_best.pth --disable_tta
echo "=== EVALUATE $CFG ==="
python3 scripts/evaluate.py --pred "$OUT" --gt "$RAW/labelsTs" \
    --out ~/flexict-finetune/eval_${CFG}.json
echo "=== EVAL $CFG DONE ==="
