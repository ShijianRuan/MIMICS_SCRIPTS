#!/bin/bash
# gpu3.sh —— 自动生成的 GPU 3 训练和预测脚本
# 生成时间: 2026-07-28 08:29:04
# 配置来源: Config_CTWholeBodyBone.toml

set -e

# ── 训练 ──────────────────────────────────────────────────
echo "[GPU 3] Training Start: Dataset305_Extremities_Leg_1.5 (ID=305)"
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 305 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 3] Training End: Dataset305_Extremities_Leg_1.5"

# ── 预测 ──────────────────────────────────────────────────
echo "[GPU 3] Predicting Start: Dataset305_Extremities_Leg_1.5 (ID=305)"
cd /data1/segmentationForTrain/traindata/CTWholeBodyBone_RAI/nnUNet_raw/Dataset305_Extremities_Leg_1.5
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 305 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 3] Predicting End: Dataset305_Extremities_Leg_1.5"

echo "[GPU 3] 所有任务完成！"
