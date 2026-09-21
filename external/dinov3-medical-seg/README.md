# DINOv3 Medical Image Segmentation

基于 DINOv3 的医学图像少样本分割微调框架，支持原生网格 2D
冻结特征训练和 3D/2.5D 研究策略。

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 准备预训练权重
# 默认方法需要 models/dinov3-vits16/model.onnx。
# HuggingFace 权重需要先接受模型许可并完成登录。
python scripts/download_weights.py --model vits16

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
| Frozen Feature 2D | `decoder.type: feature_unet2d` | 2.13M (decoder only) | **Mimics 默认** |
| Frozen | `vit_b_frozen.yaml` | ~4M (decoder only) | 3D baseline |
| LoRA | `vit_b_lora.yaml` | ~4.6M | 3D 少样本研究 |
| Full | `vit_b_full.yaml` | ~90M | 数据充足时使用 |

## 支持的解码头

| 解码器 | 配置 | 参数量 | 场景 |
|--------|------|--------|------|
| `linear3d` | decoder.type: linear3d | ~0.05M | 快速验证 |
| `mlp_probe` | decoder.type: mlp_probe | ~0.3M | 极少标注 |
| `segformer3d` | decoder.type: segformer3d | ~4M | 3D 通用解码 |
| `dpt3d` | decoder.type: dpt3d | ~8M | 追求精度 |
| `feature_unet2d` | decoder.type: feature_unet2d | 2.13M | **Mimics 默认，冻结特征 + 原图跳接** |

```bash
# 切换解码头
python scripts/train.py --config config/vit_b_lora.yaml --decoder.type dpt3d
```

## 支持的模型

- ViT-S/16 (冻结 ONNX 或本地 HuggingFace 权重): 默认 2D 方法
- ViT-B/16 (86M params): `config/vit_b_*.yaml`
- ViT-L/16 (300M params): `config/vit_l_*.yaml`

## 文档

- [调研综述](docs/research/survey.md)
- [代码仓库调研](docs/research/code_repos_survey.md)
- [架构设计](docs/design/architecture.md)
- [冻结特征 2D 协议、验证结果与 Windows 要求](docs/frozen_feature_2d_protocol.md)
