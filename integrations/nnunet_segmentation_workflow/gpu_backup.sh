#!/bin/bash
# gpu0.sh —— 自动生成的 GPU 0 训练和预测脚本
# 生成时间: 2026-05-13 05:13:58
# 配置来源: Config_CT_v500.toml

set -e

# ── 训练 ──────────────────────────────────────────────────
echo "[GPU 0] Training Start: Dataset412_CT12_Rib (ID=412)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 412 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 0] Training End: Dataset412_CT12_Rib"

echo "[GPU 0] Training Start: Dataset413_CT13_Sternum (ID=413)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 413 3d_fullres 0 -tr nnUNetTrainerNoMirroring -p nnUNetPlans
echo "[GPU 0] Training End: Dataset413_CT13_Sternum"

# ── 预测 ──────────────────────────────────────────────────
echo "[GPU 0] Predicting Start: Dataset412_CT12_Rib (ID=412)"
cd /data1/segmentationForTrain/traindata/MIv500_RAI_CT/nnUNet_raw/Dataset412_CT12_Rib
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 412 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 0] Predicting End: Dataset412_CT12_Rib"

echo "[GPU 0] Predicting Start: Dataset413_CT13_Sternum (ID=413)"
cd /data1/segmentationForTrain/traindata/MIv500_RAI_CT/nnUNet_raw/Dataset413_CT13_Sternum
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 413 -c 3d_fullres -tr nnUNetTrainerNoMirroring -p nnUNetPlans -f 0 --disable_tta
echo "[GPU 0] Predicting End: Dataset413_CT13_Sternum"

echo "[GPU 0] 所有任务完成！"
