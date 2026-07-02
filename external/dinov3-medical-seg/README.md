# DINOv3 3D Medical Image Segmentation

基于 DINOv3 的 3D 医学图像少样本分割微调框架。

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 下载预训练权重 (HuggingFace, 公开可用)
python scripts/download_weights.py --model vitb16

# 3. 准备数据
# data/
# ├── imagesTr/   (NIfTI .nii.gz)
# └── labelsTr/   (NIfTI .nii.gz, 整数标签)

# 4. 训练
python scripts/train.py --config config/vit_b_lora.yaml --data_root ./data --data.k_shot 5

# 5. 推理
python scripts/infer.py \
    --config experiments/exp_xxx/config.yaml \
    --checkpoint experiments/exp_xxx/checkpoints/best_model.pth \
    --input ct_volume.nii.gz \
    --output pred_seg.nii.gz
```

## 支持的微调方法

| 方法 | 配置 | 参数量 | 说明 |
|------|------|--------|------|
| Frozen | `vit_b_frozen.yaml` | ~4M (decoder only) | 最快 baseline |
| LoRA | `vit_b_lora.yaml` | ~4.6M | **推荐默认** |
| Full | `vit_b_full.yaml` | ~90M | 数据充足时使用 |

## 支持的解码头

| 解码器 | 配置 | 参数量 | 场景 |
|--------|------|--------|------|
| `linear3d` | decoder.type: linear3d | ~0.05M | 快速验证 |
| `mlp_probe` | decoder.type: mlp_probe | ~0.3M | 极少标注 |
| `segformer3d` | decoder.type: segformer3d | ~4M | **推荐默认** |
| `dpt3d` | decoder.type: dpt3d | ~8M | 追求精度 |

```bash
# 切换解码头
python scripts/train.py --config config/vit_b_lora.yaml --decoder.type dpt3d
```

## 支持的模型

- ViT-B/16 (86M params): `config/vit_b_*.yaml`
- ViT-L/16 (300M params): `config/vit_l_*.yaml`

## 文档

- [调研综述](docs/research/survey.md)
- [代码仓库调研](docs/research/code_repos_survey.md)
- [架构设计](docs/design/architecture.md)
