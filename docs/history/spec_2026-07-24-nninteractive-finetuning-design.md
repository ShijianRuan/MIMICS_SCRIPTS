# nnInteractive 微调集成设计文档

- **日期**：2026-07-24
- **状态**：设计阶段
- **依赖**：nnInteractive bridge（已集成）、DINOv3 few-shot pipeline（参考设计模式；**已于 2026-09-23 删除**，commit b632d70，改由 FlexiCT 微调路线取代——本文属历史设计文档，few-shot 相关引用仅存历史价值，现状以 `docs/MIMICS_PROJECT_ARCHITECTURE_CN.md` 为准）

---

## 1. 目标

在 Mimics 中集成 nnInteractive 的微调能力，使用户能够针对特定任务（器官/模态/扫描协议）用自己的标注数据微调 nnInteractive 模型，显著提升交互分割精度。

核心原则：
- **通用微调框架**：不限定器官或模态，用户可选择自己的数据和任务
- **覆盖 Few-shot 到 Full-scale**：3-20 例快速适配 → 100+ 例深度优化
- **数据来源灵活**：支持从 `.mcs` 标注导出和外部 NIfTI 导入
- **完全复用 Mimics 现有设计模式**：外部进程 + JSON 状态 + GPU 锁 + 异步监控

---

## 2. 调研结论

### 2.1 nnInteractive 技术血统

nnInteractive 不是凭空出现的模型。它的技术 genealogy：

```
nnU-Net (2018, Nature Methods)
  └─ STU-Net (2023, arXiv, 159 citations)
       └─ nnU-Net ResEnc L (2024)
            └─ nnInteractive (2025/03, arXiv, 57 citations)
                 └─ CLoPA (2026/03) ← 首个微调 nnInteractive 的工作
```

**nnInteractive = ResEnc L backbone + 交互式 Prompt 处理（points/scribbles/boxes/lasso）+ 120+ 数据集预训练**

训练代码未开源（官方 Issue #16 确认），但预训练权重公开可用，且官方明确允许非商业微调（Issue #11 确认）。

### 2.2 三条被验证的微调路径

通过穷举搜索（arXiv、Semantic Scholar、Google Scholar、PubMed、GitHub、HuggingFace），找到 3 篇直接微调 nnInteractive 的论文：

#### 路径 A：CLoPA 式极轻量 Continual Adaptation（推荐主要参考）

> Esmaeili et al., King's College London + Siemens Healthineers (2026/03)

- **微调参数**：仅 Instance Normalization 的 affine 参数（<0.01% 总参数量）或加上浅层 encoder/decoder 的 convolution kernels
- **数据需求**：5-25% 目标数据集即可触发首次训练
- **训练成本**：极低（patch 192³、batch 2、Adam lr 1e-3、10 epochs × 50 steps）
- **性能**：8 个 MSD 任务上一致提升，包括 nnInteractive 之前失败的任务（肝血管 Dice: 0.165 → 0.698；海马体: 0.805 → 0.911）
- **核心发现**：单次训练 episode 即能获得大部分收益
- **代码**：未公开，但方法描述极为详细（可直接复现）

#### 路径 B：Interactive-MEN-RT 式全量训练/迁移学习

> Lee et al., Seoul National University Hospital, MICCAI CLIP 2025 Workshop

- **微调方式**：完整 ResEnc L + Prompt 机制训练，支持从头训练和从 nnInteractive 权重迁移学习
- **数据需求**：400 例 CE-T1w MRI
- **训练参数**：patch 128³、batch 8、SGD Nesterov lr 1e-2、polynomial decay
- **代码**：✅ 开源 `github.com/snuh-rad-aicon/Interactive-MEN-RT`（`train.py` + `nnInteractiveTrainer`，基于 MONAI + nnUNet v2）
- **实验缺陷**：迁移学习在某些 prompt 类型下（如 lasso）反而不如从头训练

#### 路径 C：Dynamic Prompt Generation 式从权重初始化训练

> Ndir et al., Freiburg University, CVPR 2025 MedSegFM Workshop（Coreset Track 第 2 名）

- **微调方式**：动态体积 prompt 生成 + 内容感知自适应裁剪 + nnInteractive 权重初始化
- **训练成本**：单 GPU 可训练
- **验证**：CVPR 2025 MedSegFM 竞赛方案
- **代码**：✅ **完整开源** `github.com/tidiane-camaret/segfm3d_nora_team`（⭐4）
  - `experiments/training/train_nnint.py` — PyTorch Lightning 微调脚本
  - `experiments/training/load_weights_and_train.py` — 直接加载 nnInteractive 权重训练
  - `src/training/nnunetcustomtrainer.py` — CustomTrainer（nnUNet +7 通道）
  - `src/training/model_wrap.py` — **完整的 click 模拟训练**（error-region 采样 + 多步迭代优化）
  - `src/methods/nninteractivecore.py` — nnInteractive checkpoint 加载器
  - `src/training/` — dataloading.py / loss.py / transforms.py / utils.py

#### 额外发现：DIEAP Flap 临床微调

> Andrade et al., *"AI-based planning for DIEAP flap procedures"*, Frontiers in Medicine (2026)

- 在临床 CT 血管造影数据上微调 nnInteractive，用于穿孔器动脉分割
- 9 例测试集上 Dice 显著提升
- 代码未公开

### 2.3 关键支撑文献

- **nnUNet v2 官方预训练/微调文档**（MIC-DKFZ）：提供完整的 plans 迁移 + `-pretrained_weights` 机制
- **STU-Net**（Huang et al., 2023, 159 citations）：验证 ResEnc 架构预训练→微调范式的有效性（14 个下游数据集）
- **nnU-Net Revisited**（Isensee et al., 2024）：ResEnc 架构的权威基准，确认 ResEnc L 为当前最优配置

### 2.4 代码实现现状（穷举结论）

对 **54 个 nnInteractive fork**、**CVPR 竞赛全部获奖团队**、**论文作者独立仓库**、**HuggingFace**、以及 **所有 GitHub 代码搜索** 进行全面排查后，结论如下：

| 来源 | Trainer | train.py | Prompt 模拟 | 数据管线 | 通用性 |
|------|---------|----------|------------|---------|--------|
| **MIC-DKFZ/nnInteractive**（官方） | 25 行 stub | ❌ | ❌ | ❌ | — |
| **tidiane-camaret/segfm3d_nora_team** 🏆 | ✅ CustomTrainer | ✅ 完整 | ✅ click 迭代模拟 | ✅ nnUNet+自定义 | ⭐⭐⭐⭐ |
| **Interactive-MEN-RT**（SNU） | 24 行 stub | 290 行（脑膜瘤） | ❌ | MONAI | 单任务 |
| **CLoPA 作者** | ❌ 空仓库 | ❌ | ❌ | ❌ | — |
| **其余 50+ fork** | ❌ 纯镜像 | ❌ | ❌ | ❌ | — |

**核心发现**：

1. **`tidiane-camaret/segfm3d_nora_team` 是目前全球唯一完整的 nnInteractive 微调代码库。** 它包含：
   - `nnunetcustomtrainer.py`：继承 nnUNetTrainer，`num_input_channels + 7`、2 输出通道
   - `model_wrap.py`：训练时模拟 click 交互（error-region 中心采样、正/负 click mask、最多 5 步迭代优化）
   - `train_nnint.py`：PyTorch Lightning 微调脚本，加载 nnInteractive v1.0 权重
   - `load_weights_and_train.py`：直接用 nnUNet API 加载 nnInteractive checkpoint 并训练
   - `nninteractivecore.py`：从 nnInteractive checkpoint 构建网络的加载器
   - 完整的 transforms（bbox/seg/click 通道注入）、dataloading、loss 模块

2. **官方 Trainer** 本质上就是 `nnUNetTrainer` + `num_input_channels + 7` + `num_output_channels = 2`。7 个额外通道编码了交互 prompt（图像 + 前次分割 + positive/negative 点/scribble/box/lasso）。

3. **DIEAP Flap 论文**（Frontiers in Medicine, 2026）在临床数据上成功微调了 nnInteractive，证明了微调在真实临床场景中的价值。
   - 硬编码为脑膜瘤 MRI 单任务
   - 使用 MONAI transforms 而非 nnUNet 原生管线
   - 没有实现 prompt 训练模拟（trainer stub 中未见 interaction simulation 逻辑）
   - **不是通用微调框架，是领域特化的一次性实现**

3. **全球没有任何公开仓库提供了完整的、通用的、与 nnInteractive 一脉相承的微调代码。**

### 2.5 文献可信度总结

| 论文 | 机构 | 发表 | 代码 | 可信度 |
|------|------|------|------|--------|
| CLoPA | KCL + Siemens | arXiv only | 方法足够复现 | ⭐⭐⭐⭐⭐ |
| Interactive-MEN-RT | SNU Hospital | MICCAI CLIP 2025 | ✅ 开源但领域特化 | ⭐⭐⭐ |
| Dynamic Prompt Gen | Freiburg | arXiv only | 未公开 | ⭐⭐⭐⭐ |

**结论：Mimics 的 nnInteractive 微调模块将是该领域第一个完整的开源通用实现。** 三条论文路径提供了足够的方法论支撑，但代码层面需要从头构建（以 nnUNet v2 为训练引擎）。

---

## 3. 设计方案

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│               Mimics nnInteractive 微调框架                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  数据准备层   │ →  │   微调策略层       │ →  │   Mimics 集成层  │  │
│  │              │    │                  │    │                 │  │
│  │ .mcs → NIfTI │    │ Few-shot (3-20)  │    │ Scripting Lib   │  │
│  │ NIfTI import │    │  → IN 微调        │    │ PySide6 UI      │  │
│  │ nnUNet 格式  │    │                  │    │ JSON 状态文件    │  │
│  │              │    │ Medium (20-100)  │    │ GPU 锁          │  │
│  │              │    │  → IN + Conv     │    │ 异步监控         │  │
│  │              │    │                  │    │                 │  │
│  │              │    │ Large (100+)     │    │                 │  │
│  │              │    │  → Full FT       │    │                 │  │
│  └─────────────┘    └──────────────────┘    └─────────────────┘  │
│                                                                   │
│  训练基础设施：nnUNet v2 数据管线 + 自定义 nnInteractive Trainer       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 微调策略矩阵

根据数据量自动选择微调策略：

| 场景 | 数据量 | 可训练参数 | 参数量占比 | 训练时间（估） | 适用场景 |
|------|--------|-----------|-----------|--------------|---------|
| **Few-shot** | 3-20 例 | IN affine only | <0.01% | ~5 min | 快速适配新器官 |
| **Medium** | 20-100 例 | IN + 浅层 Conv | <5% | ~30 min | 特定任务优化 |
| **Full** | 100+ 例 | Full ResEnc L | 100% | ~数小时 | 大幅度域迁移 |

策略选择逻辑：
1. 默认从 Few-shot（IN only）开始
2. 如果用户标注数据持续增长，自动升级到 Medium
3. 如果用户选择全量微调并有足够 GPU 资源，直接 Full

### 3.3 数据管线

```
┌──────────────────────────────────────────────────────────────┐
│                     数据准备流程                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  方式 1：从 .mcs 导出（复用 DINOv3 pipeline 的导出逻辑）         │
│  ─────────────────────────────────────────────                │
│  用户选择 .mcs 文件 → 选择目标 Mask → 导出为 NIfTI             │
│  → 按 nnUNet 格式组织（imagesTr / labelsTr / dataset.json）    │
│                                                               │
│  方式 2：外部 NIfTI 导入                                       │
│  ─────────────────────────                                    │
│  用户指定 images/ 和 labels/ 文件夹 → 验证配对                  │
│  → 按 nnUNet 格式组织                                          │
│                                                               │
│  共同后处理：                                                  │
│  nnUNetv2_plan_and_preprocess → 自动标准化/重采样/patch 配置    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

参考 nnUNet v2 官方预训练/微调流程：

```bash
# 1. 为目标数据集做 planning
nnUNetv2_plan_and_preprocess -d TARGET_DATASET

# 2. 将 nnInteractive checkpoint 转换为 nnUNet 兼容格式
# (自定义脚本，处理 key mapping)

# 3. 微调训练
nnUNetv2_train TARGET_DATASET CONFIG FOLD \
  -pretrained_weights PATH_TO_CONVERTED_CHECKPOINT
```

### 3.4 训练基础设施选型

**选择 nnUNet v2 作为训练引擎**，理由：
- nnInteractive 基于 ResEnc L，与 nnUNet v2 同一架构家族
- nnUNet v2 的数据管线（标准化、重采样、patch 提取、数据增强）成熟可靠
- 官方提供完整的 `pretraining_and_finetuning.md` 文档
- 社区有 TotalSegmentator fine-tuning 的实战经验（Issue #298）
- 避免从零构建数据管线

**需要自研的部分**：
1. **Checkpoint 格式转换器**：nnInteractive checkpoint → nnUNet v2 兼容格式
2. **nnInteractiveTrainer**：继承 nnUNet v2 Trainer，加入：

   - Prompt 模拟训练（训练时模拟 click/scribble/box 交互）
   - IN-only 微调模式（冻结除 IN 外的所有参数）
   - 渐进式微调策略（IN → IN+Conv → Full）
3. **Prompt 训练数据合成器**：根据标注 mask 自动生成模拟的用户交互 prompt

### 3.5 Mimics 集成方式

完全复用 DINOv3 few-shot pipeline 的外部进程模式：

```
Scripting Library 新增入口：
  scripting_library/02_AI/nnInteractive/
    ├── 01_nnInteractive.py          ← 已有（交互推理）
    ├── 02_FineTune_Model.py         ← 新增（微调入口）
    ├── 03_FineTune_Predict.py       ← 新增（使用微调模型推理）
    └── 04_Show_FineTune_Status.py   ← 新增（微调状态查看）

工具层新增：
  tools/
    ├── nninteractive_finetune_setup_ui.py   ← 微调设置 UI（PySide6）
    ├── nninteractive_finetune_pipeline.py   ← 微调管线（外部进程）
    └── nninteractive_checkpoint_converter.py ← checkpoint 格式转换

配置文件扩展：
  nninteractive_config.json → 增加 finetune 相关配置
```

**外部进程通信模式**（与 DINOv3 pipeline 一致）：

```
Mimics (Python 3.5)
  │  Popen
  ├─→ tools/nninteractive_finetune_setup_ui.py (PySide6, 外部进程)
  │     │ 用户选择数据和策略 → 写入 setup JSON
  │     │ Popen
  │     └─→ tools/nninteractive_finetune_pipeline.py (外部进程)
  │           │ nnUNet v2 数据预处理
  │           │ 训练循环（写入 status JSON）
  │           │ 保存微调 checkpoint
  │
  │  Mimics 定期轮询 status JSON → 更新进度显示
```

**GPU 资源管理**：
- 复用现有 `resource_locks.py` 的 GPU 锁机制
- 微调训练与 nnInteractive 推理互斥（共用 GPU）
- 微调训练与 DINOv3 训练互斥

---

## 4. 实现路径

### Phase 1：可行性验证（Checkpoint 格式兼容性）

**目标**：确认 nnInteractive checkpoint 可以直接被 nnUNet v2 加载

**步骤**：
1. 下载 nnInteractive 预训练 checkpoint（`checkpoint_final.pth`）
2. 打印 state_dict 的所有 key
3. 与 nnUNet v2 ResEnc L 的 state_dict 结构对比
4. 如果 key 不兼容，编写映射脚本
5. 验证：加载转换后的权重 → 跑一次推理 → 与原始 nnInteractive server 对比结果

**成功标准**：转换后的 checkpoint 推理结果与原始 nnInteractive server 一致

**预计时间**：1-2 天

### Phase 2：核心微调实现

**目标**：基于 `segfm3d_nora_team` 的完整代码，实现支持 IN-only / IN+Conv / Full 三种模式的通用微调训练器

**参考代码（已确认可用）**：
- `github.com/tidiane-camaret/segfm3d_nora_team` — **最完整的参考实现**
  - `src/training/nnunetcustomtrainer.py` — CustomTrainer 骨架
  - `src/training/model_wrap.py` — Prompt 模拟训练（click 迭代）
  - `experiments/training/load_weights_and_train.py` — 加载 nnInteractive 权重训练
  - `experiments/training/train_nnint.py` — PyTorch Lightning 训练脚本
- `github.com/MIC-DKFZ/nnUNet/documentation/pretraining_and_finetuning.md` — 官方微调流程
- CLoPA 论文 — IN-only/IN+Conv 微调策略

**步骤**：
1. 基于 nnUNet v2 的 `nnUNetTrainer`，实现 `nnInteractiveFinetuneTrainer`：

   - 继承 `nnUNetTrainer`，覆写 `build_network_architecture`（+7 通道、2 输出）
   - 实现 IN-only 模式：`freeze_all_except_instance_norm_affine()`
   - 实现 IN+Conv 模式：`freeze_all_except_instance_norm_and_shallow_convs()`
   - 实现 Full FT 模式：标准 nnUNet 训练（低 LR）
2. 实现 Prompt 训练数据模拟器 `InteractivePromptSampler`：

   - Click 模拟：从 GT mask 的 false-negative 区域随机采样正/负点（参考 CLoPA 方法）
   - Scribble 模拟：在 GT mask 边界区域生成随机 scribble（参考 nnInteractive 论文）
   - Box 模拟：从 GT mask 生成带随机 margin 的 bounding box
   - 交互迭代：模拟 N=5 步用户交互（与 CLoPA 一致）
3. 实现 nnUNet v2 数据管线对接：

   - `.mcs` 导出 → NIfTI → nnUNet 格式（`imagesTr/labelsTr/dataset.json`）
   - 外部 NIfTI → nnUNet 格式
   - `nnUNetv2_plan_and_preprocess` 自动配置
4. 实现 Checkpoint 格式转换器：

   - nnInteractive ckpt → nnUNet v2 加载 → 微调 → nnInteractive 兼容输出

**预计时间**：2-3 周

### Phase 3：Mimics 集成

**目标**：将微调能力完全集成到 Mimics 工作流中

**步骤**：
1. 实现 `scripting_library/02_AI/nnInteractive/02_FineTune_Model.py`
2. 实现 `tools/nninteractive_finetune_setup_ui.py`（PySide6 UI，数据选择 + 策略配置）
3. 实现 `tools/nninteractive_finetune_pipeline.py`（外部训练进程）
4. 扩展 `nninteractive_config.json` 配置
5. 实现微调模型的推理切换（在 nnInteractive bridge 中支持指定微调后的 checkpoint）

**预计时间**：2-3 周

---

## 5. 核心挑战与风险

| 挑战 | 风险等级 | 缓解策略 |
|------|---------|---------|
| **Checkpoint 格式兼容性** | 🟡 中 | Phase 1 先验证，不兼容就写映射脚本 |
| **Prompt 训练数据构造** | 🟡 中 | 参考 nnInteractive 论文和 CLoPA 的 click 模拟方法；Interactive-MEN-RT 有完整的 prompt 模拟代码 |
| **nnUNet v2 强依赖** | 🟢 低 | nnUNet v2 是成熟框架，Mimics 已有 DINOv3 pipeline 的外部进程模式可以直接套用 |
| **微调后模型推理兼容性** | 🟡 中 | 需要确保微调后的 checkpoint 可以被 nnInteractive server 加载；可能需要修改 server 的模型加载逻辑 |
| **GPU 内存** | 🟢 低 | Few-shot IN-only < 1% 参数量几乎不增加显存；Full FT 需要与现有 DINOv3 类似的 GPU 资源 |

---

## 6. 不做的事情（范围边界）

- ❌ 不自己实现训练数据管线（直接用 nnUNet v2）
- ❌ 不修改 nnInteractive 推理 server 的核心逻辑（只替换 checkpoint）
- ❌ 不支持在线实时微调（CLoPA 的 continual 模式 — 未来可扩展）
- ❌ 不实现自动器官识别/检测（微调范围由用户指定）
- ❌ 不改变现有 nnInteractive bridge 的协议格式

---

## 7. 参考文献

1. **CLoPA** (Esmaeili et al., 2026) — 直接微调 nnInteractive 的核心参考，提供 IN-only 和 IN+Conv 微调策略
2. **Interactive-MEN-RT** (Lee et al., 2025) — 唯一开源的 nnInteractive 训练代码，提供 `train.py` 和 `trainer/` 参考实现
3. **Dynamic Prompt Generation** (Ndir et al., 2025) — 提供 prompt 模拟训练策略参考
4. **nnInteractive** (Isensee et al., 2025) — 原始论文，ResEnc L 架构 + Prompt 机制
5. **nnU-Net Revisited** (Isensee et al., 2024) — ResEnc 架构权威基准
6. **STU-Net** (Huang et al., 2023) — 验证 ResEnc 预训练→微调范式
7. **nnUNet v2 pretraining_and_finetuning.md** — MIC-DKFZ 官方微调文档
8. **MELBA 2025 SAM Fine-tuning Study** (Gu et al., 2025) — 18 种微调策略的系统对比（提供通用 fine-tuning 原则参考）
