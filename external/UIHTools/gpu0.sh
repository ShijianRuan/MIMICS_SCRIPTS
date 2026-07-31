#!/bin/bash
# gpu0.sh —— 自动生成的 GPU 0 训练和预测脚本
# 生成时间: 2026-05-21 04:16:38
# 配置来源: Config_CT_v500.toml

set -e

# ── 训练 ──────────────────────────────────────────────────
echo "[GPU 0] Training Start: Dataset417_CT17_Bodybone (ID=417)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 417 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 0] Training End: Dataset417_CT17_Bodybone"

# ── 预测 ──────────────────────────────────────────────────
echo "[GPU 0] Predicting Start: Dataset417_CT17_Bodybone (ID=417)"
cd /data1/segmentationForTrain/traindata/MIv500_RAI_CT/nnUNet_raw/Dataset417_CT17_Bodybone
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 417 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 0] Predicting End: Dataset417_CT17_Bodybone"

echo "[GPU 0] 所有任务完成！"
