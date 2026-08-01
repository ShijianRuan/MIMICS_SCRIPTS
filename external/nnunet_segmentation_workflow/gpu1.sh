#!/bin/bash
# gpu1.sh —— 自动生成的 GPU 1 训练和预测脚本
# 生成时间: 2026-07-17 08:10:22
# 配置来源: Config_CTWholeBodyBone.toml

set -e

# ── 训练 ──────────────────────────────────────────────────
echo "[GPU 1] Training Start: Dataset304_Extremities_Leg (ID=304)"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 304 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 1] Training End: Dataset304_Extremities_Leg"

# ── 预测 ──────────────────────────────────────────────────
echo "[GPU 1] Predicting Start: Dataset304_Extremities_Leg (ID=304)"
cd /data1/segmentationForTrain/traindata/CTWholeBodyBone_RAI/nnUNet_raw/Dataset304_Extremities_Leg
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 304 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 1] Predicting End: Dataset304_Extremities_Leg"

echo "[GPU 1] 所有任务完成！"
