# DINOv3 3D 医学图像少样本分割微调框架 — 设计文档 v4.0

> **版本**: v4.0 (2026-06-28) — 消融实验验证 + 参考来源标注
> **原则**: 每个设计决策都基于可追溯的权威来源或实验证据

---

## 0. 设计决策矩阵（附参考来源）

### 0.1 Backbone 加载

| 决策 | 选择 | 参考来源 | 验证状态 |
|------|------|---------|:------:|
| 加载方式 | HuggingFace `DINOv3ViTBackbone` (本地权重) | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (10.8k★) | ✅ 已验证 |
| 权重获取 | atomgit 镜像下载 (hf_mirrors) | 国内网络环境实测 | ✅ ViT-B/L 均已下载 |
| 特征提取 | `out_features=["stage3","stage6","stage9","stage12"]` + `reshape_hidden_states=True` | HF transformers v4.56.0+ [PR #41276](https://github.com/huggingface/transformers/issues/41276) | ✅ |
| 输入分辨率 | 224×224 (SynthStrip) / 518×518 (DINOv3 原生) | DINOv3 paper [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | ✅ |

### 0.2 3D 编码策略

| 决策 | 选择 | 参考来源 | 验证状态 |
|------|------|---------|:------:|
| 3D 路线 | Slice-wise axial 编码 + pseudo-3D 特征体 | [DINOv3 Benchmark](https://arxiv.org/abs/2509.06467) (14 datasets) + [DINO-MVR](https://arxiv.org/abs/2605.07221) | ✅ |
| 内存优化 | Mini-batch 切片处理 (batch=8) + CPU offload | [Neonatal Brain MR](https://arxiv.org/abs/2602.23962) sub-volume strategy | ✅ 实测解 MPS OOM |
| 不选用路线 | 真·3D ViT 预训练 (3DINO/VoCo) | 需要 100K+ 体积从头预训练，非「微调 DINOv3」 | ❌ 明确了不做 |

### 0.3 解码头

| 决策 | 选择 | 参考来源 | 消融 DSC | 状态 |
|------|------|---------|:-------:|:----:|
| 默认解码器 | **SegFormer3D** (~4M params) | [an-mistral/SegFormer3D](https://github.com/an-mistral/Cardiac_Right_Ventricle_MRI_Segmentation) compact design | **0.924** ★★ | ✅ |
| 轻量优选 | MLP Probe (~0.6M params) | [DINO-MVR](https://arxiv.org/abs/2605.07221) multi-probe + z-smooth | 0.889 ★ | ✅ |
| DPT3D | Dense Prediction Transformer 3D | [Neonatal Brain MR](https://arxiv.org/abs/2602.23962) + [DPT paper](https://arxiv.org/abs/2103.13413) | 0.797 | ⚠️ 低于预期 |
| 最简基线 | Linear3D (1x1x1 Conv) | DINOv3 Benchmark lightweight head | 0.727 | ✅ |
| 待加入 | UPerNet, SegFormer (全MLP) | [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (8.5k★) + 调研建议 | — | P1 |

### 0.4 微调方法

| 决策 | 选择 | 参考来源 | 消融 DSC (segformer3d) | 状态 |
|------|------|---------|:---------------------:|:----:|
| 默认方法 | **frozen backbone + 可训 decoder** | [DINOv3 Model Card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md): "fine-tuning is expected to increase biases" | **0.916** | ✅ |
| LoRA | r=8, Q_proj + V_proj | [RobvanGastel/dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) (499★) + [LoRA paper](https://arxiv.org/abs/2106.09685) | 0.920 | ✅ |
| Adapter | Bottleneck=64, after_attn | [Houlsby et al. ICML 2019] + [AdaptFormer NeurIPS 2022] | **0.924** | ✅ |
| DoRA | LoRA + weight decomposition | [DoRA ICML 2024 Oral] + HF PEFT `use_dora=True` | — | P1 |
| Full FT | 全量微调 | — | — | ❌ MPS 显存不足 |

### 0.5 训练超参

| 决策 | 选择 | 参考来源 | 验证状态 |
|------|------|---------|:------:|
| 优化器 | AdamW, lr=1e-3 (frozen/adapter), 1e-3 (LoRA) | [dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) + 全文献共识 | ✅ |
| 调度器 | Cosine annealing + 5 epoch warmup | DINOv3 训练配方简化版 | ✅ |
| 损失函数 | Dice + CE (0.5:0.5) | 医学分割事实标准 | ✅ |
| Epochs | 10 (快速消融) / 50+ (生产) | 消融实验收敛曲线 | ✅ |
| 数据增强 | 3D Flip + Rotation + Intensity | nnU-Net 风格医学增强 | 基础版 |

### 0.6 不引入的依赖

| 依赖 | 理由 | 参考 |
|------|------|------|
| nnUNet | 太重，学习曲线陡，定制困难 | [DebuggerCafe](https://debuggercafe.com) 等多开发者报告 |
| MONAI | 体积大，只需基本 3D 操作 | 项目规模不匹配 |
| MMSegmentation | 依赖复杂，DINOv3 无官方支持 | 多名开发者绕过 |
| PEFT 库 | LoRA 自实现 ~80 行，Adapter ~60 行 | 避免版本兼容 + 完全控制 |
| PyTorch Lightning | 纯 PyTorch 更透明 | 所有调研仓库均使用纯 PyTorch |

---

## 1. 消融实验结果 (SynthStrip v1.5, 5-shot, ViT-B)

### 完整矩阵 (SynthStrip v1.5, 5-shot, ViT-B/16)

```
Decoder       Frozen          LoRA (r=8)      Adapter (bn=64)
────────────────────────────────────────────────────────────────
linear3d      0.727 (90m)     0.758 (43m)     0.748 (29m)
mlp_probe     0.889 (30m)     0.879 (50m)     0.888 (40m)
DPT3D         0.797 (55m)     —                —
segformer3d   0.916 (30m) ★   0.920 (50m)     0.924 (33m) ★★
```

### ViT-B vs ViT-L

| Backbone | Params | Decoder | DSC | Time | 结论 |
|----------|:------:|---------|:---:|:----:|------|
| ViT-B/16 | 86M | frozen+segformer3d | **0.916** | 30m | ★ 推荐 |
| ViT-L/16 | 300M | frozen+segformer3d | **0.876** | ~3h+ | 更差+更慢，不推荐 ⚠️ |

> ViT-L 在 MPS 上收敛更慢、最终 DSC 更低、训练时间 6x+。验证了 DINOv3 Benchmark 论文的发现：医学领域放大模型不一定有效。

### 跨数据集泛化

| 数据集 | 任务 | 前景占比 | frozen+segformer3d DSC |
|--------|------|:------:|:---------------------:|
| SynthStrip v1.5 | 脑部剥离 | ~50% | **0.916** |
| MSD Task02 Heart | 左心房分割 | ~1-5% | **0.646** |

> 简单任务（大脑）效果优异，极小目标（心房）5-shot 极具挑战性。证明方法可泛化，但性能高度取决于任务难度。

### 核心结论

1. **Decoder 是最大杠杆**：segformer3d (0.92) ≫ mlp_probe (0.89) ≫ DPT3D (0.80) ≫ linear3d (0.73)
2. **微调方法影响微小**：在 segformer3d 上 frozen/LoRA/adapter 仅差 ±0.008
3. **Frozen+segformer3d 最佳性价比**：30min, 9MB ckpt, 0.916 DSC
4. **mlp_probe 是轻量级优选**：0.6M params 达 0.889，参数量仅为 segformer3d 的 1/8
5. **LoRA/Adapter 在小 decoder 上价值更高**：linear3d 上分别 +3.1%/+2.1%
6. **ViT-B 已足够**：在 5-shot 脑部剥离任务上 0.92 接近天花板
7. **跨数据集泛化成立**：但任务难度决定性能上限

---

## 2. 项目结构

```
dinov3-medical-seg/
├── README.md
├── pyproject.toml                    # 依赖: torch, transformers, nibabel, pyyaml, tqdm, tensorboard
├── .gitignore
│
├── config/
│   ├── train.yaml                    # 全局默认值
│   ├── synthstrip_frozen.yaml
│   ├── synthstrip_lora_{linear3d,segformer3d}.yaml
│   ├── synthstrip_adapter_{linear3d,segformer3d}.yaml
│   ├── synthstrip_frozen_dpt3d.yaml
│   ├── synthstrip_vitl_frozen_segformer3d.yaml
│   └── vit_b_{lora,frozen,full}.yaml / vit_l_*.yaml
│
├── src/
│   ├── models/
│   │   ├── backbone.py       # HF DINOv3ViTBackbone 本地加载
│   │   ├── encoder_3d.py     # Slice-wise 3D 编码 (batch=8 + CPU offload)
│   │   ├── decoder_3d.py     # Linear3D / MLPProbe3D / SegFormer3D / DPT3D
│   │   ├── lora.py           # LoRA (Q_proj, V_proj) — 参考 dinov3-finetune
│   │   ├── adapter.py        # BottleneckAdapter (after_attn, bn=64)
│   │   └── segmentor.py      # 模型组装 (frozen/lora/adapter/full + decoder)
│   ├── data/
│   │   └── dataset_3d.py     # NIfTI 加载 + FewShotSubset
│   ├── training/
│   │   ├── trainer.py        # 3D 训练循环 (AMP + 梯度累积)
│   │   ├── losses.py         # DiceLoss / CELoss / DiceCELoss
│   │   └── metrics.py        # DSC / HD95
│   └── utils/
│       ├── config.py         # YAML _base_ 继承 + deep merge
│       ├── device.py         # CUDA/MPS/CPU 自动检测
│       └── checkpoint.py     # 仅保存可训参数
│
├── scripts/
│   ├── train.py              # 训练入口
│   ├── infer.py              # 推理入口
│   └── download_weights.py
│
├── docs/
│   ├── research/
│   │   ├── survey.md                          # 论文综述 (17+8 refs)
│   │   ├── code_repos_survey.md               # 20+ 仓库分析
│   │   ├── decoder_lora_adapter_supplement.md # Decoder/LoRA/Adapter 专项
│   │   └── ablation_results.md                # 消融实验完整记录
│   └── design/
│       └── architecture.md                    # 本文档
│
├── models/                      # 本地权重 (gitignored)
│   ├── dinov3-vitb16/          # ViT-B/16 (327MB)
│   └── dinov3-vitl16/          # ViT-L/16 (1.1GB)
│
├── data/synthstrip/             # 示例数据集
└── experiments/                 # 实验输出 (gitignored)
```

---

## 3. 核心模块 API

### Backbone
```python
from src.models.backbone import DINOv3Backbone
bb = DINOv3Backbone("models/dinov3-vitb16", out_indices=[2,5,8,11], freeze=True)
feats = bb(image_2d)  # List[(B, 768, H/16, W/16)], len=4
```

### Segmentor
```python
from src.models.segmentor import DINOv33DSegmentor
model = DINOv33DSegmentor(config)        # config 指定 model_path, finetune.method, decoder.type
pred = model(volume_3d)                   # (1,1,D,H,W) → (1,C,D,H,W)
info = model.get_trainable_info()         # 参数量统计
```

### LoRA
```python
from src.models.lora import apply_lora_to_dinov3, LoRALinear
apply_lora_to_dinov3(backbone, r=8, alpha=16)  # 注入 Q_proj, V_proj
```

### Adapter
```python
from src.models.adapter import apply_adapter_to_dinov3
apply_adapter_to_dinov3(backbone, bottleneck=64, position="after_attn")
```

---

## 4. 使用方式

```bash
# 基础训练 (推荐方案)
python scripts/train.py --config config/synthstrip_frozen.yaml \
    --decoder.type segformer3d --data.k_shot 5

# LoRA 微调
python scripts/train.py --config config/synthstrip_lora_segformer3d.yaml

# Adapter 微调
python scripts/train.py --config config/synthstrip_adapter_segformer3d.yaml

# ViT-L 对比
python scripts/train.py --config config/synthstrip_vitl_frozen_segformer3d.yaml

# 推理
python scripts/infer.py \
    --config experiments/exp_xxx/config.yaml \
    --checkpoint experiments/exp_xxx/checkpoints/best_model.pth \
    --input ct.nii.gz --output pred.nii.gz
```

---

## 5. 完整参考来源索引

### 论文
| ID | 标题 | arXiv / Venue | 关联决策 |
|----|------|---------------|---------|
| [1] | DINOv3 | arXiv:2508.10104, Meta AI 2025 | Backbone, 特征提取 |
| [2] | DINOv3 Medical Benchmark | arXiv:2509.06467 | 3D slice-wise 路线 |
| [3] | DINO-MVR | arXiv:2605.07221 | MLP Probe decoder, z 轴平滑 |
| [4] | Neonatal Brain MR 3D | arXiv:2602.23962 | Sub-volume + DPT3D |
| [5] | MedDINOv3 | arXiv:2509.02379 | 域自适应预训练 |
| [6] | LoRA | arXiv:2106.09685, Hu et al. 2021 | LoRA 微调 |
| [7] | DoRA | ICML 2024 Oral, Liu et al. | LoRA 增强 (P1) |
| [8] | AdaptFormer | NeurIPS 2022, Chen et al. | Adapter 设计 |
| [9] | DPT | arXiv:2103.13413, Ranftl et al. 2021 | DPT3D 解码头 |
| [10] | MELBA 2025 Benchmark | MELBA Journal | 18 种微调组合 |

### 代码仓库
| ID | 仓库 | Stars | 关联决策 |
|----|------|:-----:|---------|
| [R1] | facebookresearch/dinov3 | 10.8k | 官方权重 + API |
| [R2] | RobvanGastel/dinov3-finetune | 499 | LoRA 实现参考 |
| [R3] | yifangao112/DinoUNet | 307 | 医学 DINOv3 分割 |
| [R4] | ricklisz/MedDINOv3 | 214 | nnUNet + Primus |
| [R5] | script-Yang/segdino | 256 | Frozen + MLP decoder |
| [R6] | sovit-123/dinov3_stack | 112 | YAML 配置系统 |
| [R7] | AICONSlab/3DINO | — | 3D ViT 预训练参考 |
| [R8] | AIM-Harvard/DINOv2-3D-Med | — | 3D SSL + MONAI |
| [R9] | an-mistral/SegFormer3D | — | SegFormer3D decoder 设计 |
| [R10] | apple1986/DINO-AugSeg | 14 | 少样本医学分割 |
| [R11] | jinlab-imvr/DINOv3-FD | — | PEFT (LoRA/IA3/VeRA) |
| [R12] | yeerwen/Awesome-Medical-Efficient-Fine-Tuning | 35 | EFT 论文索引 |

---

> **文档维护**: 随项目演进持续更新。最新版本见 `docs/design/architecture.md`。
