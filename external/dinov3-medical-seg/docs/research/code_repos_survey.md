# DINOv3/DINOv2 微调仓库与 ViT 分割框架调研报告

> **调研日期**: 2026-06-26
> **调研范围**: GitHub 开源仓库、arXiv 论文、HuggingFace 生态
> **聚焦领域**: DINOv3 医学图像分割微调、通用 ViT 微调框架、LoRA/Adapter 实现

---

## 1. DINOv3 微调仓库总览

### 1.1 官方/Meta 仓库

| 仓库 | Stars | 描述 | 状态 |
|------|-------|------|------|
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | ~10,000 | DINOv3 官方 PyTorch 实现，提供 ViT-S/B/L/H/7B 预训练权重，含 ADE20K 线性分割评估代码 | 活跃维护 |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | ~12,800 | DINOv2 官方实现，已被 DINOv3 取代但仍被大量引用，提供 MMSegmentation 集成 | 维护模式 (推荐迁移至 v3) |

**关键发现**: DINOv3 官方仓库已于 2025-10-13 发布 ADE20K 线性分割和 NYUv2 深度估计代码。官方推荐使用 frozen backbone + 轻量级 head 的方式，不鼓励全量微调。官方 Model Card 明确指出：「Fine-tuning is expected to increase the biases in the features... it is recommended to keep this option as a last resort」。

### 1.2 DINOv3 医学图像分割专用仓库

| 仓库 | Stars | 描述 | 框架集成 | 关键架构 |
|------|-------|------|----------|----------|
| [ricklisz/MedDINOv3](https://github.com/ricklisz/MedDINOv3) | ~214 | DINOv3 医学分割适配框架，CT-3M 域自适应预训练 | nnUNet v2 | Primus + Multi-scale Token Aggregation |
| [yifangao112/DinoUNet](https://github.com/yifangao112/DinoUNet) | ~302 | DINOv3 + U-Net 架构，FAPM (Fidelity-Aware Projection Module) | nnUNet v2 | Frozen DINOv3 + Dual-branch Adapter + FAPM |
| [script-Yang/segdino](https://github.com/script-Yang/segdino) | ~256 | 轻量级 MLP decoder + frozen DINOv3，SegDINO-V2 已发布 | 独立框架 | Frozen DINOv3 + Lightweight MLP Decoder |
| [WZH0120/DINOv3-UNet](https://github.com/WZH0120/DINOv3-UNet) | ~31 | DINOv3 + U-Net 分割适配 | SAM2-UNet 工具链 | DINOv3 encoder + U-Net decoder |
| [nramira/dino_finetuning](https://github.com/nramira/dino_finetuning) | ~2 | DINOv2 vs DINOv3 脑肿瘤分割对比基准 | 独立框架 (HuggingFace) | Frozen backbone + CNN segmentation head |

**关键对比**:
- **MedDINOv3** 采用域自适应预训练策略（三阶段：DINOv2-style → Gram Anchoring → 高分辨率适应），在 nnUNet 框架内实现，CT-3M（387 万张 CT 切片）预训练。但需要大量领域数据支撑。
- **DinoUNet** 提出 FAPM（低秩共享投影 + 动态特征调制），是当前最先进的 DINOv3 医学分割架构。支持 ViT-S 到 ViT-7B 多尺度。
- **SegDINO** 最轻量级，仅用一个 MLP decoder，证明 frozen DINOv3 特征已足够强大。

### 1.3 DINOv3 通用分割/下游任务仓库

| 仓库 | Stars | 描述 | 特点 |
|------|-------|------|------|
| [RobvanGastel/dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) | ~472 | DINOv2/v3 LoRA 微调 + 1x1 Conv decoder | 同时支持 v2/v3，LoRA 实现最完整 |
| [sovit-123/dinov3_stack](https://github.com/sovit-123/dinov3_stack) | ~112 | 分类/分割/检测三任务统一框架 | YAML 配置驱动，argparse CLI |
| [brisyramshere/downstream-dinov3](https://github.com/brisyramshere/downstream-dinov3) | ~10 | UNet/DPT/FAPM 三种分割架构 | 统一的 config 驱动，支持医学和自然图像 |
| [ub216/dinov3](https://github.com/ub216/dinov3) | ~5 | DPT head + DINOv3 手术器械分割 | 基于 SegDINO 代码改编 |
| [annayah/VolSegInfantHippocampi](https://github.com/annayah/VolSegInfantHippocampi) | - | 3D 婴儿海马体分割 (DINOv3 + MONAI) | 2D encoder 切片编码 + 3D decoder |

### 1.4 DINOv3 商用/平台级支持

| 平台 | 支持情况 |
|------|----------|
| **timm** (v1.0.15+) | 已添加 DINOv3 ConvNeXt 和 ViT 模型，ViT 基于 EVA 基类实现，新增 `RotaryEmbeddingDinoV3` |
| **HuggingFace Transformers** (v4.56.0+) | 完整支持 DINOv3 backbones（ViT-S/B/L/H/7B + ConvNeXt 全系列） |
| **LightlyTrain** | 支持 DINOv3 蒸馏预训练 + EoMT 语义分割微调 |
| **HuggingFace PEFT** | 可通过 custom model 方式支持 DINOv3 LoRA |

---

## 2. DINOv2 成熟微调仓库（可迁移至 DINOv3）

### 2.1 高星 DINOv2 分割仓库

| 仓库 | Stars | 核心贡献 | DINOv3 迁移可行性 |
|------|-------|----------|-------------------|
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | ~12,800 | 官方实现，MMSegmentation 集成 | 官方已推荐迁移至 v3 |
| [itsprakhar/Downstream-Dinov2](https://github.com/itsprakhar/Downstream-Dinov2) | ~272 | 分类+分割+深度估计统一实现 | 架构模式可迁移，需替换 backbone 加载逻辑 |
| [lorebianchi98/Talk2DINO](https://github.com/lorebianchi98/Talk2DINO) | ~182 | ICCV 2025，开放词汇分割，CLIP+DINOv2 | **已支持 DINOv3** (2025-10) |
| [JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) | ~435 | LoRA for ViT，支持 timm 和 DeepLab 分割 | LoRA 注入模式可直接用于 DINOv3 |
| [zshn25/DINOv2_mmseg](https://github.com/zshn25/DINOv2_mmseg) | ~44 | DINOv2 + MMSegmentation 训练测试流程 | 架构清晰，但依赖 MMSegmentation |
| [dhk1349/seg-dinov2](https://github.com/dhk1349/seg-dinov2) | ~32 | DINOv2 + Linear+Conv head MSCOCO 分割 | 简单的 head 设计可参考 |
| [2U1/DINOv2-Finetune](https://github.com/2U1/DINOv2-Finetune) | ~14 | DINOv2 下游微调框架 | 分类为主，分割待实现 |
| [AIM-Harvard/DINOv2-3D-Med](https://github.com/AIM-Harvard/DINOv2-3D-Med) | - | 3D DINOv2 自监督预训练 + nnUNet 导出 | 3D 医学预训练范式可直接参考 |

### 2.2 可迁移技术分析

| 技术 | 来源仓库 | DINOv3 迁移难度 | 说明 |
|------|----------|----------------|------|
| LoRA 注入 attention 层 | dinov3-finetune (RobvanGastel) | **低** | 该仓库已直接支持 DINOv3，代码可直接参考 |
| 1x1 Conv decoder | dinov3-finetune | **低** | 最简单的 decoder，适合 baseline |
| FPN decoder | dinov3-finetune | **低** | 多尺度特征融合 |
| MMSegmentation 集成 | DINOv2_mmseg, dinov2 官方 | **中** | 需要注册 DINOv3 backbone 到 MMSegmentation |
| `get_intermediate_layers` API | dinov2 官方 | **低** | DINOv3 有类似 API，接口一致 |
| PyTorch Hub 加载 | 所有仓库 | **低** | `torch.hub.load('facebookresearch/dinov3', ...)` |
| HuggingFace 加载 | dino_finetuning (nramira) | **低** | `AutoModel.from_pretrained('facebook/dinov3-...')` |
| 3D slice-wise encoding | VolSegInfantHippocampi | **中** | 2D 特征堆叠为伪 3D 卷 |

---

## 3. 通用 ViT 微调框架分析

### 3.1 框架对比总览

| 框架 | Stars | ViT Backbone | DINO 支持 | 医学分割友好度 | 推荐度 |
|------|-------|-------------|-----------|---------------|--------|
| **MMSegmentation** | ~8,500 | 原生 ViT + Swin + BEiT | DINOv2 (需手动配置) | 中 | 避免采用 |
| **timm** | ~33,000 | 1200+ 架构 | DINOv3 (v1.0.15+) | 低 (分类为主) | 作为 backbone 来源 |
| **segmentation_models.pytorch (smp)** | ~9,800 | 800+ timm encoder | 间接 (通过 timm) | 中 | 条件采用 |
| **detectron2** | ~30,000 | 基础 ViT | 无直接支持 | 低 (检测为主) | 避免采用 |
| **nnUNet** | ~6,000 | 间接 (Primus 桥接) | DINOv3 (MedDINOv3) | **最高** | **强烈推荐** |
| **MONAI** | ~8,000 | ViT/ViTAutoEnc/UNETR | 间接 | 高 | 推荐 3D 场景 |
| **PyTorch Lightning** | ~28,000 | 无内置 | 无 | 中 | 作为训练框架 |
| **HuggingFace Transformers + PEFT** | ~140,000 | 完整 ViT | DINOv3 (v4.56.0+) | 高 | 推荐 LoRA 场景 |

### 3.2 详细分析

#### MMSegmentation (OpenMMLab)
- **优势**: 模型库最丰富（UPerNet、Mask2Former、SegFormer、DPT 等），完善的配置系统和分布式训练支持
- **劣势**: 
  - 依赖复杂（mmcv + mmengine + mmsegmentation 版本锁定）
  - DINOv2 需要手动注册 backbone，DINOv3 无官方支持
  - 模型定制需要深入了解 mmseg 架构（Registry、Hook 系统）
  - DebuggerCafe 文章明确指出「discarded MMSegmentation requirement」因为「dependency issues」
- **结论**: **避免采用**。对于研究性项目，纯 PyTorch 实现更灵活；对于医学分割，nnUNet 更成熟。

#### timm (PyTorch Image Models)
- **DINOv3 支持**: v1.0.15+ 已添加 DINOv3 ViT 和 ConvNeXt。ViT 基于 EVA 基类实现，新增 `RotaryEmbeddingDinoV3` 匹配 DINOv3 特有的 RoPE 实现。
- **使用方式**: 
  ```python
  model = timm.create_model('vit_base_patch16_224.dinov3_lvd1689m', pretrained=True, features_only=True)
  ```
- **注意**: `features_only=True` 对 ViT 模型的支持在 2024-12 才完善（通过新的 `forward_intermediates()` API），需要确认 DINOv3 模型是否兼容。
- **结论**: 作为 backbone 加载来源之一，但不作为训练框架主体。

#### segmentation_models.pytorch (smp)
- **v0.5.0 重大更新**: 
  - 新增 **DPT 模型**，原生支持 ViT encoder（通过 timm）
  - 800+ timm encoders 支持（`tu-` 前缀）
  - ViT encoders 仅适用于 DPT 等 transformer-style 模型，不兼容 U-Net 等传统架构
- **使用示例**:
  ```python
  model = smp.DPT("tu-vit_base_patch16_224.augreg_in21k", classes=2)
  ```
- **DINOv3 潜在路径**: 将 DINOv3 ViT 作为 timm encoder 加载 → 使用 smp.DPT decoder
- **结论**: **条件采用**。DPT + ViT 路线对通用分割有价值，但医学分割场景下 nnUNet 更优。

#### nnUNet
- **DINOv3 集成现状**: 
  - MedDINOv3 和 DinoUNet 均在 nnUNet v2 框架内实现
  - 通过自定义 Trainer 类 (`dinov3Trainer.py`) 注入 DINOv3 backbone
  - 训练命令: `nnUNetv2_train dataset_id 2d 0 -tr dinov3_base_primus_multiscale_Trainer`
- **优势**: 
  - 医学分割的事实标准框架
  - 自动数据预处理（spacing、normalization、patch size）
  - 五折交叉验证、集成推理内置
  - 大量医学分割 baseline 可直接对比
- **劣势**: 
  - 框架较重，定制超出标准 pipeline 的功能较复杂
  - 学习曲线陡峭
- **结论**: **强烈推荐**作为医学分割实验框架。

#### detectron2
- ViT backbone 存在基础实现 (`detectron2/modeling/backbone/vit.py`)，但无 DINO 系列支持
- 主要面向检测和实例分割，语义分割不是核心场景
- **结论**: **避免采用**。

### 3.3 框架选型决策矩阵

| 决策维度 | 推荐框架 | 原因 |
|----------|----------|------|
| 医学 2D 分割实验 | nnUNet v2 | 标准化流程，MedDINOv3/DinoUNet 已集成 |
| 医学 3D 分割实验 | nnUNet v2 + MONAI | 3DINO, DINOv2-3D-Med 已验证 |
| 通用分割快速原型 | smp.DPT + timm | 最简洁的 API，一行代码创建模型 |
| LoRA 微调实验 | HuggingFace PEFT + Transformers | 最成熟的 PEFT 生态 |
| 自监督预训练 | dinov3 官方仓库 | 唯一完整实现 |
| 轻量级推理部署 | SegDINO 模式 (frozen + MLP) | 最小参数量 |

---

## 4. LoRA / Adapter 实现参考

### 4.1 LoRA 实现对比

| 实现来源 | 方式 | 目标层 | Stars | 优点 | 缺点 |
|----------|------|--------|-------|------|------|
| [RobvanGastel/dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) | 自定义 LoRA | Attention QKV + FFN | ~472 | **DINOv3 原生支持**，同时支持 v2/v3，FPN decoder | Jupyter Notebook 为主，工程化程度低 |
| [JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) | 自定义 LoRA | Attention QV | ~435 | 支持 timm、DeepLab 分割、多 LoRA | 代码较旧 (2023)，不支持 DINOv3 |
| [HuggingFace PEFT](https://github.com/huggingface/peft) | peft 库 | 可配置 target_modules | ~21,300 | 最成熟生态，支持 LoRA/DoRA/AdaLoRA 等，与 Trainer 集成 | DINOv3 非原生支持，需配置 custom model |
| [neelkanthrawat/Comp_vision_final_project](https://github.com/neelkanthrawat/Comp_vision_final_project_2025) | 自定义 + peft | Attention | - | 实现了 LoRA/Serial LoRA/Localized LoRA 三种变体对比 | 学术项目，代码质量一般 |
| [ghassenbaklouti/ARENA](https://github.com/ghassenbaklouti/ARENA) | 自定义 PEFT | SVD-based adaptive rank | - | MICCAI，自适应 rank 选择 | 针对 few-shot |

### 4.2 LoRA 在 DINOv3 上的注入点分析

根据 `RobvanGastel/dinov3-finetune` 的实现经验：

```python
# DINOv3 ViT 中适合 LoRA 注入的层
target_modules = [
    "qkv",       # Attention 的 QKV 投影 (效果最显著)
    "proj",      # Attention 输出投影
    "fc1",       # MLP 第一层
    "fc2",       # MLP 第二层
]
```

**实验结论** (来自 dinov3-finetune):
- LoRA + 1x1 Conv decoder: DINOv3 ViT-L/14 → Pascal VOC mIoU ~40.0%
- LoRA + FPN decoder: 效果更好
- LoRA (r=4, alpha=4) 仅增加约 0.1% 可训练参数
- DINOv3 ViT-H 比 ViT-B 对 corruption 更鲁棒

### 4.3 Adapter 实现对比

| 实现来源 | Adapter 类型 | 设计特点 | 可迁移性 |
|----------|-------------|----------|----------|
| [DinoUNet](https://github.com/yifangao112/DinoUNet) | **Dual-branch DINO Adapter** | 空间先验分支 + 特征适配分支，FAPM 低秩投影 | **高** - 最先进的 DINOv3 适配器设计 |
| [MedDINOv3](https://github.com/ricklisz/MedDINOv3) | **Primus Multi-scale Token Aggregation** | 从 ViT 不同层提取多尺度 token 并聚合 | **高** - 简单有效 |
| [ViT-Adapter](https://github.com/czczup/ViT-Adapter) | **Spatial Prior Module + Feature Interaction** | 注入 CNN 空间先验到 ViT，支持 MMSegmentation | **中** - 依赖 MMSegmentation |
| [SegDINO](https://github.com/script-Yang/segdino) | **无 Adapter (纯 frozen)** | 直接使用 frozen DINOv3 特征 + MLP decoder | **最高** - 最简洁，证明特征质量足够 |
| [Annayah/VolSeg](https://github.com/annayah/VolSegInfantHippocampi) | **Depth-aware Reboxing** | 2D slices → 3D 空间重组 | **低** - 3D 专用 |

### 4.4 LoRA/Adapter 推荐方案

**对于 DINOv3 医学分割微调，推荐三层渐进策略**:

| 层级 | 策略 | 参数量 | 适用场景 |
|------|------|--------|----------|
| Level 1 (最小化) | Frozen DINOv3 + 可训练 decoder head | ~1-5M | 数据充足、域差异小 |
| Level 2 (参数高效) | LoRA (r=8-16) + 可训练 decoder | ~0.5-2M (新增) | **推荐首选**，数据有限 |
| Level 3 (最大性能) | Full fine-tuning + decoder | ~86M-7B | 大量领域数据，极致性能 |

**LoRA 实现推荐**: 优先使用 **HuggingFace PEFT** 库，原因：
1. 21K+ stars，生态最成熟
2. 支持 LoRA/DoRA/AdaLoRA 等多种变体
3. 与 Transformers Trainer 无缝集成
4. 内置 `save_pretrained` / `merge_and_unload` 等工具
5. 支持 timm 模型的 custom model 方式集成

若 PEFT 与 DINOv3 自有模型代码不兼容，退而参考 **dinov3-finetune (RobvanGastel)** 的自定义 LoRA 实现，该实现已针对 DINOv3 验证。

---

## 5. 最佳实践提炼

### 5.1 项目结构模式

对比所有调研仓库，高星项目的目录结构模式分为三类：

**模式 A: nnUNet 集成型** (MedDINOv3, DinoUNet)
```
project/
├── nnUNet/                          # nnUNet 框架
│   └── nnunetv2/training/nnUNetTrainer/
│       ├── dinov3/                  # DINOv3 源码 (vendored)
│       └── dinov3Trainer.py         # 自定义 Trainer
├── environment.yaml
└── readme.md
```
- **优点**: 利用 nnUNet 的全部基础设施
- **缺点**: 与特定框架强绑定

**模式 B: 独立框架型** (SegDINO, sovit-123/dinov3_stack)
```
project/
├── src/
│   ├── models/           # 模型定义
│   ├── data/             # 数据加载
│   ├── engine/           # 训练/验证循环
│   └── pipeline/         # 训练/推理入口脚本
├── configs/              # YAML 配置
├── notebooks/            # Jupyter 分析
├── dinov3/               # DINOv3 源码 (submodule 或 clone)
├── checkpoints/          # 预训练权重
└── requirements.txt
```
- **优点**: 灵活、可定制、依赖最少
- **缺点**: 需要自己实现训练基础设施

**模式 C: HuggingFace 生态型** (nramira/dino_finetuning)
```
project/
├── src/
│   ├── components/       # model_builder, dataloaders, engine, loss, evaluation
│   ├── pipeline/         # train.py, predict.py
│   └── api/              # FastAPI 服务
├── notebooks/
├── models/               # 本地模型存储
└── requirements.txt
```
- **优点**: 利用 HuggingFace 生态，代码量最少
- **缺点**: 灵活性受限于 HuggingFace API

**推荐**: 对于我们的项目，建议采用 **模式 B (独立框架型)**，原因：
- nnUNet 太重，且我们可能需要定制不兼容 nnUNet 的架构
- HuggingFace 方式对 DINOv3 的支持仍在完善中
- 独立框架给予最大灵活性

### 5.2 配置系统对比

| 配置方式 | 代表仓库 | 优点 | 缺点 |
|----------|----------|------|------|
| **argparse CLI** | SegDINO, dinov3-finetune, DinoUNet | 简单直接，适合脚本化 | 超参数多时命令行过长 |
| **YAML 配置** | sovit-123/dinov3_stack, downstream-dinov3, AIM-Harvard/DINOv2-3D-Med | 结构化，可版本控制，可嵌套 | 需要解析逻辑 |
| **Hydra** | (未见 DINO fine-tuning 仓库使用) | 强大的配置组合能力 | 学习曲线，依赖较重 |
| **mmseg Config** | DINOv2_mmseg | 继承 mmseg 配置体系 | 仅限 MMSegmentation 生态 |

**推荐**: YAML 配置文件 + argparse 覆盖关键参数（如 `--epochs`, `--lr`, `--batch_size`）。这是最常见且最灵活的模式。参考 `sovit-123/dinov3_stack` 的 `segmentation_configs/` 目录设计。

### 5.3 训练循环设计模式

对比各仓库的训练循环实现：

| 模式 | 代表仓库 | 特点 |
|------|----------|------|
| **纯 PyTorch 循环** | SegDINO, dinov3-finetune | 完全手动控制，适合研究和调试 |
| **nnUNet Trainer 继承** | MedDINOv3, DinoUNet | 继承 nnUNet 训练基础设施 |
| **HuggingFace Trainer** | dino_finetuning (nramira) | 利用 Trainer 回调、日志等 |
| **PyTorch Lightning** | AIM-Harvard/DINOv2-3D-Med | 结构化训练，DDP 支持 |

**共同的最佳实践**:
1. **混合精度训练 (AMP)** - 几乎所有仓库都使用 `torch.cuda.amp`
2. **梯度累积** - 小 batch size 场景下的标配
3. **学习率 warmup + cosine decay** - 最常用的调度策略
4. **Combined Loss** - Dice Loss + Cross Entropy Loss 是医学分割的标配
5. **Early Stopping** - 基于验证 Dice/IoU

### 5.4 Checkpoint 管理

各仓库的 checkpoint 保存策略：

| 策略 | 说明 | 代表仓库 |
|------|------|----------|
| **Best metric checkpoint** | 保存验证集最佳 Dice/IoU 模型 | 几乎所有仓库 |
| **定期 checkpoint** | 每 N epochs 保存 | dinov3 官方 (每 12500 iter) |
| **LoRA 权重分离** | adapter 权重与 base model 分开保存 | dinov3-finetune, LoRA-ViT |
| **safetensors 格式** | 安全且加载快 | LoRA-ViT, HuggingFace 生态 |

**推荐**:
- 使用 `safetensors` 格式（安全、跨平台）
- 对 LoRA 微调，仅保存 adapter 权重（~几 MB）
- 对 full fine-tuning，保存完整 checkpoint + optimizer state（用于恢复训练）
- 始终保存 `config.yaml` 与 checkpoint 关联

### 5.5 多模型支持设计模式

| 仓库 | 多模型支持方式 | 评价 |
|------|---------------|------|
| sovit-123/dinov3_stack | YAML 配置中指定 `model_name`，`--model-name` CLI 参数 | 清晰灵活 |
| downstream-dinov3 | `backbones.py` 统一加载器 + 不同 model 文件 | 解耦良好 |
| DinoUNet | `--model` 参数选择 `dinounet_s/b/l/7b` | 简单直接 |
| RobvanGastel/dinov3-finetune | `--dino_type dinov2/dinov3` + `--size base/large` | 支持多版本对比 |

**推荐设计**:
```python
# 统一的 backbone 工厂函数
def create_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    name: dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov2_vitb14 | ...
    """
    if name.startswith("dinov3"):
        return torch.hub.load("facebookresearch/dinov3", model=name, ...)
    elif name.startswith("dinov2"):
        return torch.hub.load("facebookresearch/dinov2", model=name, ...)
```

---

## 6. 对我们的设计建议

### 6.1 直接参考的仓库 (按优先级排序)

| 优先级 | 仓库 | 参考内容 | 理由 |
|--------|------|----------|------|
| **P0** | [yifangao112/DinoUNet](https://github.com/yifangao112/DinoUNet) | FAPM decoder 架构、Dual-branch Adapter 设计 | 最先进的 DINOv3 医学分割架构，nnUNet 集成范例 |
| **P0** | [ricklisz/MedDINOv3](https://github.com/ricklisz/MedDINOv3) | nnUNet 集成方式、多尺度 token 聚合、域自适应预训练范式 | 最完整的 DINOv3 医学分割框架 |
| **P0** | [RobvanGastel/dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) | LoRA 实现、DINOv2/v3 权重加载、1x1 Conv / FPN decoder | 最成熟的 LoRA + DINOv3 实现 |
| **P1** | [script-Yang/segdino](https://github.com/script-Yang/segdino) | 轻量级 MLP decoder 设计、数据加载 pipeline | 证明 "frozen + 简单 decoder" 足够好的 baseline |
| **P1** | [sovit-123/dinov3_stack](https://github.com/sovit-123/dinov3_stack) | YAML 配置系统、多任务统一框架设计 | 结构清晰的工程化实现 |
| **P1** | [brisyramshere/downstream-dinov3](https://github.com/brisyramshere/downstream-dinov3) | UNet/DPT/FAPM 三种架构对比、统一 backbone 加载器 | 多架构对比的参考范例 |
| **P2** | [JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) | LoRA 保存/加载 API 设计、DeepLab 分割集成 | 经典的 LoRA for ViT 实现 |
| **P2** | [nramira/dino_finetuning](https://github.com/nramira/dino_finetuning) | HuggingFace 加载 DINOv3、DINOv2 vs DINOv3 对比 | 展示了最简单的 HuggingFace 集成方式 |

### 6.2 推荐采用的模式

1. **Backbone 加载**: 使用 `torch.hub.load('facebookresearch/dinov3', ...)` 作为主要方式，HuggingFace Transformers 作为备选。DINOv3 官方推荐 PyTorch Hub 方式。
2. **Decoder 设计**: 
   - **Baseline**: Frozen DINOv3 + 轻量级 MLP/卷积 decoder（参考 SegDINO）
   - **进阶**: FAPM (DinoUNet) 或 Primus Multi-scale (MedDINOv3)
   - **注意**: DPT head (smp/ub216) 在自然图像上效果好，但医学分割上 FAPM 更优
3. **微调策略**: 优先使用 **LoRA** (参考 dinov3-finetune)，仅在 LoRA 效果不足时才考虑 full fine-tuning
4. **配置系统**: YAML + argparse override（参考 sovit-123/dinov3_stack）
5. **训练框架**: 纯 PyTorch + AMP + TensorBoard/WandB logging
6. **数据 pipeline**: 参考 nnUNet 的数据预处理规范（spacing normalization, patch-based training），但不依赖 nnUNet 框架

### 6.3 应避免的模式

1. **避免 MMSegmentation 依赖** - 多名开发者报告依赖问题和灵活性受限（DebuggerCafe, Talk2DINO 等都选择了绕过）
2. **避免 detectron2** - 面向检测任务，语义分割支持不足
3. **避免过重的配置系统 (Hydra)** - DINO 社区无一使用
4. **避免在未验证 frozen baseline 的情况下进行 full fine-tuning** - DINOv3 官方反复强调 frozen features 已足够强大
5. **避免直接使用 smp 的 U-Net/FPN 等 CNN 风格 decoder** - 这些 decoder 与 ViT encoder 的扁平特征不兼容

### 6.4 更新的架构/Decoder 推荐

基于全面的代码调研，以下架构组合在当前 (2026-06) 最具竞争力：

| 策略 | Encoder | Decoder | 适用场景 | 预期性能 |
|------|---------|---------|----------|----------|
| **最小可行方案** | Frozen DINOv3 ViT-B/16 | 1x1 Conv + Bilinear Upsample | 快速 baseline | 中等 |
| **推荐方案** | DINOv3 ViT-B/16 + LoRA (r=8) | FAPM (DinoUNet) | **医学分割首选** | **最高** |
| **轻量方案** | Frozen DINOv3 ViT-S/16 | MLP Decoder (SegDINO) | 资源受限 | 良好 |
| **3D 方案** | DINOv3 2D slice-wise | 3D DPT/UNETR (MONAI) | 3D 体积分割 | 取决于数据量 |
| **极致性能** | DINOv3 ViT-L/16 + 域自适应预训练 | Primus Multi-scale (MedDINOv3) | 大规模项目 | 理论最高 |

**核心推荐**: 对于大多数医学分割场景，**DINOv3 ViT-B/16 + LoRA (r=8-16) + FAPM decoder** 是当前最优的精度/效率平衡点。这一组合已在 DinoUNet 论文中在 7 个医学分割数据集上验证，且代码完全开源。

---

## 附录 A: 仓库快速索引

| 仓库 | URL | Stars | 类别 |
|------|-----|-------|------|
| facebookresearch/dinov3 | https://github.com/facebookresearch/dinov3 | ~10,000 | 官方 |
| facebookresearch/dinov2 | https://github.com/facebookresearch/dinov2 | ~12,800 | 官方 |
| yifangao112/DinoUNet | https://github.com/yifangao112/DinoUNet | ~302 | 医学 DINOv3 |
| ricklisz/MedDINOv3 | https://github.com/ricklisz/MedDINOv3 | ~214 | 医学 DINOv3 |
| script-Yang/segdino | https://github.com/script-Yang/segdino | ~256 | 通用 DINOv3 |
| RobvanGastel/dinov3-finetune | https://github.com/RobvanGastel/dinov3-finetune | ~472 | LoRA DINOv3 |
| sovit-123/dinov3_stack | https://github.com/sovit-123/dinov3_stack | ~112 | 通用 DINOv3 |
| WZH0120/DINOv3-UNet | https://github.com/WZH0120/DINOv3-UNet | ~31 | 医学 DINOv3 |
| JamesQFreeman/LoRA-ViT | https://github.com/JamesQFreeman/LoRA-ViT | ~435 | LoRA ViT |
| HuggingFace PEFT | https://github.com/huggingface/peft | ~21,300 | PEFT 库 |
| timm | https://github.com/huggingface/pytorch-image-models | ~33,000 | Backbone 库 |
| smp | https://github.com/qubvel-org/segmentation_models.pytorch | ~9,800 | 分割库 |
| MMSegmentation | https://github.com/open-mmlab/mmsegmentation | ~8,500 | 分割框架 |
| nnUNet | https://github.com/MIC-DKFZ/nnUNet | ~6,000 | 医学分割框架 |
| MONAI | https://github.com/Project-MONAI/MONAI | ~8,000 | 医学 AI 框架 |
| lorebianchi98/Talk2DINO | https://github.com/lorebianchi98/Talk2DINO | ~182 | 开放词汇分割 |

## 附录 B: 关键论文映射

| 论文 | 仓库 | 核心贡献 |
|------|------|----------|
| DINOv3 (Siméoni et al., 2025) | facebookresearch/dinov3 | SSL 基础模型 |
| MedDINOv3 (2025) | ricklisz/MedDINOv3 | CT 域自适应 + 多尺度 token 聚合 |
| Dino U-Net (Gao et al., 2025) | yifangao112/DinoUNet | FAPM + Dual-branch DINO Adapter |
| SegDINO (Yang et al., 2025) | script-Yang/segdino | Frozen DINOv3 + MLP Decoder |
| LoRA (Hu et al., 2021) | microsoft/LoRA | Low-Rank Adaptation |
| ViT-Adapter (Chen et al., 2022) | czczup/ViT-Adapter | 空间先验注入 ViT |
| DPT (Ranftl et al., 2021) | isl-org/DPT | Dense Prediction Transformer |
| 3DINO (2025) | AICONSlab/3DINO | 3D DINOv2 医学预训练 |
| PEFT for Medical (Silva-Rodríguez et al., 2024) | jusiro/fewshot-finetuning | Few-shot PEFT 医学分割 |
