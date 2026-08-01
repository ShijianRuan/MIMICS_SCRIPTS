#!/bin/bash
# gpu2.sh —— 自动生成的 GPU 2 训练和预测脚本
# 生成时间: 2026-05-12 08:36:07
# 配置来源: Config_CTWholeBodyBone.toml

set -e

# ── 训练 ──────────────────────────────────────────────────
echo "[GPU 0] Training Start: Dataset302_Rapid_Bone_Simple (ID=302)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 302 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 0] Training End: Dataset302_Rapid_Bone_Simple"

# ── 预测 ──────────────────────────────────────────────────
echo "[GPU 0] Predicting Start: Dataset302_Rapid_Bone_Simple (ID=302)"
cd /data1/segmentationForTrain/traindata/CTWholeBodyBone_RAI/nnUNet_raw/Dataset302_Rapid_Bone_Simple
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 302 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 0] Predicting End: Dataset302_Rapid_Bone_Simple"

echo "[GPU 0] 所有任务完成！"
