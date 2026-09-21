# DINOv3 医学图像分割：解码头、LoRA 与 Adapter 方法综述

> **编写日期**：2026-06-27
> **目的**：为 DINOv3-medical-seg 项目提供解码头、参数高效微调（PEFT）方法的系统性调研，指导后续实现决策。

---

## 目录

1. [DINOv3 分割解码头完整综述](#1-dinov3-分割解码头完整综述)
2. [LoRA 变体与实现](#2-lora-变体与实现)
3. [Adapter 方法](#3-adapter-方法)
4. [对项目实现的建议](#4-对项目实现的建议)

---

## 1. DINOv3 分割解码头完整综述

### 1.1 官方支持的解码头

DINOv3 官方仓库（facebookresearch/dinov3）提供两类解码头，但训练和推理能力差异显著。

#### 1.1.1 Linear Head（线性探针头）

这是官方**唯一支持训练**的解码头。

| 属性 | 说明 |
|------|------|
| **架构** | SyncBatchNorm（多尺度特征拼接后） → Dropout2d(p=0.1) → 1x1 Conv2d 投影到 num_classes |
| **支持的特征层** | `LAST`、`FOUR_LAST`、`FOUR_EVEN_INTERVALS` |
| **是否冻结骨干** | 是（仅训练 BN + 1x1 Conv） |
| **参数规模** | 极小（约 0.01% 骨干参数） |
| **配置文件** | `configs/config-ade20k-linear-training.yaml` |
| **适用场景** | 快速基线评估、特征质量探测、线性可分性验证 |

**优点**：极速训练、低显存占用、适合快速实验。
**缺点**：表达能力有限，无法利用多尺度上下文，mIoU 上限低。

#### 1.1.2 Mask2Former Head (M2F)

Mask2Former 是官方提供的**强大推理头**，但官方代码**不支持训练**，仅提供预训练权重。

| 组件 | 架构细节 |
|------|----------|
| **DINOv3_Adapter（FPN 骨干适配器）** | ViTDet 风格的 FPN，使用可变形注意力（Deformable Attention）在 ViT 单尺度 patch 特征上构建多尺度特征金字塔（1/4、1/8、1/16、1/32） |
| **MSDeformAttnPixelDecoder** | 可变形注意力像素解码器，6 个编码器层，逐步上采样特征 |
| **MultiScaleMaskedTransformerDecoder** | 多尺度掩码 Transformer 解码器，9 个解码器层，100 个 object queries |

| 属性 | 说明 |
|------|------|
| **支持的特征层** | `FOUR_EVEN_INTERVALS` |
| **是否冻结骨干** | 是（推理模式） |
| **训练支持** | ❌ 官方不支持训练 |
| **预训练权重** | ✅ `dinov3_vit7b16_ms`（ViT-7B/16 骨干 + M2F 头，ADE20K 150 类） |
| **适用场景** | 大规模全景分割、需要 mask 级别输出的场景 |

**关键限制**：官方 DINOv3_Adapter 设计为仅从 ViT patch tokens 构建多尺度特征，不修改骨干本身。这使其完全依赖骨干的 frozen features。

---

### 1.2 社区验证的解码头

以下解码头虽未在 DINOv3 官方代码中出现，但在 DINOv2/DINOv3 社区中被广泛验证。

#### 1.2.1 DPT（Dense Prediction Transformer）

| 属性 | 说明 |
|------|------|
| **核心思路** | 将 ViT 中间层的 patch embeddings 通过“重组”（reassemble）模块上采样到不同分辨率，然后通过融合模块（fusion block）逐步合并为高分辨率 dense 预测 |
| **重组模块** | 将 1D patch tokens reshape 为 2D 特征图，使用卷积 + 上采样恢复到不同尺度（1/2, 1/4, 1/8, 1/16） |
| **融合模块** | 残差卷积单元（Residual Convolution Unit），从最粗到最细逐级融合 |
| **参数量** | 中等（约 10-20M 参数） |
| **DINOv2 验证** | ✅ 被 RobvanGastel/dinov3-finetune、Raessan/depth_dinov3 等仓库用于深度估计和语义分割 |
| **特点** | 无需额外 transformer 层，仅用卷积操作融合多尺度特征 |

#### 1.2.2 UPerNet（Unified Perceptual Parsing Network）

| 属性 | 说明 |
|------|------|
| **核心思路** | 在 FPN 之上添加 PPM（Pyramid Pooling Module），融合来自 ViT 不同层的多尺度特征，经过 FPN 上采样后通过分类头输出 |
| **FPN 通道** | 通常从 ViT 的 4 个等间距层提取特征，投影到统一通道数（如 512），通过自上而下的侧向连接和逐元素加法融合 |
| **PPM 模块** | 在最顶层特征上应用多尺度池化（1x1, 2x2, 3x3, 6x6），concat 后卷积 |
| **参数量** | 中等偏大（约 20-30M 参数） |
| **DINOv2 验证** | ✅ mmsegmentation 中广泛使用，DINOv2 + UPerNet 是 ADE20K 的常见基线 |
| **特点** | 成熟稳定，生态支持好，文档丰富 |

#### 1.2.3 SegFormer

| 属性 | 说明 |
|------|------|
| **核心思路** | 全 MLP 解码头。从 ViT 多尺度特征中提取后，通过一系列 MLP 层融合，最后用轻量 MLP 分类头输出 |
| **关键设计** | 不使用位置编码（位置信息来自 ViT 自身的输出），BatchNorm 替换为 LayerNorm |
| **参数量** | 轻量（约 3-6M 参数） |
| **DINOv2 验证** | ✅ 作为轻量解码头被多个项目采用 |
| **特点** | 简单高效，全 MLP 无卷积，适合显存受限场景 |

#### 1.2.4 SETR（SEgmentation TRansformer）

| 属性 | 说明 |
|------|------|
| **核心思路** | 三种变体：(a) Naive Upsampling：直接双线性上采样 patch embeddings；(b) PUP（Progressive UPsampling）：交替卷积和上采样逐步恢复到原分辨率；(c) MLA（Multi-Level feature Aggregation）：融合多层特征后逐步上采样 |
| **参数量** | PUP/MLA 约 15-25M 参数 |
| **DINOv2 验证** | ✅ 部分项目中使用，但不如 DPT/UPerNet 流行 |
| **特点** | 最早将 ViT 用于密集预测的工作之一；MLA 变体性能较好但训练较慢 |

#### 1.2.5 FPN（Feature Pyramid Network）

| 属性 | 说明 |
|------|------|
| **核心思路** | 标准卷积 FPN：从 ViT 多尺度 patch tokens 构建特征金字塔，自上而下 + 横向连接 |
| **参数量** | 中等（约 8-15M 参数） |
| **DINOv2 验证** | ✅ 作为基础基线被广泛使用 |
| **特点** | 极简实现，适合作为 baseline，但性能弱于 DPT/UPerNet |

#### 1.2.6 纯卷积轻量头（2-3 层 Conv + Upsample）

| 属性 | 说明 |
|------|------|
| **核心思路** | 在 ViT 最后一层特征上直接叠加 2-3 层卷积（如 Conv2d-BN-ReLU-Conv2d），然后双线性上采样 |
| **参数量** | 极小（约 1-3M 参数） |
| **特点** | 最简实现，适合快速原型，但 mIoU 显著低于上述所有方法 |

---

### 1.3 医学分割专用解码头

#### 1.3.1 FAPM / DinoUNet

| 属性 | 说明 |
|------|------|
| **论文/仓库** | yifangao112/DinoUNet |
| **核心思路** | 将 DINOv2 的 patch tokens 注入 UNet 架构，用 DINO 特征替换 UNet 编码器的跳跃连接 |
| **架构** | DINOv2 特征 → FAPM（Feature Aggregation Pyramid Module）→ UNet 风格解码器 |
| **适用场景** | 医学图像分割，特别是需要精细边界的任务 |
| **特点** | 融合了 CNN 的局部归纳偏置和 ViT 的全局语义理解，在器官/病变分割上表现好 |

#### 1.3.2 Primus / MedDINOv3

| 属性 | 说明 |
|------|------|
| **仓库** | ricklisz/MedDINOv3 |
| **核心思路** | 在 DINOv3 骨干之上使用 Primus 混合解码器：结合 CNN 上采样和 Transformer 特征融合模块 |
| **特点** | 针对医学影像优化，支持多模态（CT/MRI/X-ray）输入 |

#### 1.3.3 GuiDINO TokenBook

| 属性 | 说明 |
|------|------|
| **核心思路** | 使用视觉 token 字典（TokenBook）机制，在解码头中引入可学习的“提示 token”，引导解码器关注医学图像中的关键结构 |
| **适用场景** | 需要先验解剖知识引导的分割任务 |
| **特点** | 将 textual/visual prompting 引入分割解码，可提升小目标分割性能 |

---

### 1.4 复杂度与性能对比

| 解码头 | 参数量（估算） | 训练难度 | 推理速度 | ADE20K mIoU（DINOv2-B） | 医学分割适配度 |
|--------|:----------:|:------:|:------:|:--------------------:|:----------:|
| Linear Head | < 0.1M | 极低 | 极快 | ~40-44% | 低（仅线性可分） |
| 轻量 Conv Head | 1-3M | 低 | 快 | ~42-46% | 中低 |
| SegFormer | 3-6M | 低 | 快 | ~48-52% | 中 |
| FPN | 8-15M | 中 | 中等 | ~49-53% | 中 |
| DPT | 10-20M | 中 | 中等 | ~51-54% | 中高 |
| SETR (MLA) | 15-25M | 中高 | 较慢 | ~50-53% | 中 |
| UPerNet | 20-30M | 中 | 中等 | ~52-55% | 高 |
| Mask2Former | 40-60M | 高 | 慢 | ~55-58% | 高 |
| DinoUNet (FAPM) | 20-35M | 中高 | 中等 | N/A（医学生物医学基准） | 很高 |

> **注**：ADE20K mIoU 数据来自 DINOv2 论文及社区复现报告，医学分割数据因任务差异较大，此处仅作定性标注。

---

### 1.5 解码头推荐汇总

| 优先级 | 解码头 | 理由 |
|:------:|--------|------|
| **P0（必做）** | Linear Head + 轻量 Conv Head | 快速基线验证，训练/推理成本极低 |
| **P1（高优先）** | UPerNet | 成熟稳定，mmsegmentation 生态支持好，医学分割中表现优异 |
| **P1（高优先）** | DPT | 在 DINOv2 深度估计和分割任务中验证充分，参数量适中 |
| **P2（可做）** | SegFormer | 全 MLP 实现简单，轻量高效，适合快速实验迭代 |
| **P2（可做）** | Mask2Former（训练版） | 性能最强的通用解码头，但实现复杂度和计算成本高 |
| **P3（探索性）** | DinoUNet (FAPM) | 针对医学分割设计，值得验证其在我们目标数据集上的效果 |

---

## 2. LoRA 变体与实现

### 2.1 基础 LoRA vs 各变体

#### 2.1.1 基础 LoRA（Hu et al., 2021）

| 属性 | 说明 |
|------|------|
| **机制** | 注入可训练的低秩分解矩阵 A、B（秩 r）作为冻结预训练权重的加性残差：W' = W + (alpha / r) * BA |
| **目标层** | 典型应用于注意力投影（Q、K、V、O），也可扩展到 MLP 层 |
| **推理延迟** | 零额外延迟（可合并权重） |
| **局限性** | (1) 所有层使用固定秩；(2) 幅度更新和方向更新成比例耦合；(3) 复杂任务上可能落后于全微调 |

#### 2.1.2 DoRA（Liu et al., ICML 2024 Oral）

| 属性 | 说明 |
|------|------|
| **机制** | 将预训练权重分解为幅度（m）和方向（V）分量，分别微调：W' = m * (W + BA) / ||W + BA|| |
| **关键优势** | 以 LoRA 一半的秩达到更好性能；消除对全微调的差距；无推理开销 |
| **HF PEFT 支持** | ✅ `LoraConfig(use_dora=True)`，支持 Linear、Conv1d、Conv2d |
| **ViT 相关性** | 已在 VL-BART（图像/视频-文本理解）和 LLaVA（视觉指令微调）中验证，架构无关 |
| **实践经验** | LoRA 收敛更快；DoRA 质量更高，尤其在低秩（r=8 vs r=32）时优势明显；超参数可能需要与 LoRA 不同 |

#### 2.1.3 AdaLoRA（Zhang et al., ICLR 2023）

| 属性 | 说明 |
|------|------|
| **机制** | 用 SVD 形式参数化增量：W' = W + PΛQ。使用基于重要性的预算分配，动态剪枝不重要的奇异值并重新分配秩 |
| **关键优势** | 自适应为不同层分配不同秩；自动发现哪些层需要更多参数 |
| **局限性** | 实现复杂度较高，训练开销大于标准 LoRA |
| **HF PEFT 支持** | ✅ `AdaLoraConfig` |

#### 2.1.4 LoRA+（Hayou et al., 2024）

| 属性 | 说明 |
|------|------|
| **机制** | 对 LoRA 的 A 和 B 矩阵使用不同的学习率（通常是 B 的学习率远大于 A）。原理：大宽度极限下，A 和 B 的更新速率应当不对称 |
| **关键优势** | 超简单（仅修改学习率），在多个任务上稳定超越标准 LoRA |
| **HF PEFT 支持** | 需自定义优化器配置（为 A 和 B 设置不同 lr） |

#### 2.1.5 VeRA（Kopiczko et al., 2024）

| 属性 | 说明 |
|------|------|
| **机制** | 跨所有层共享一对随机的冻结矩阵 A 和 B，仅训练小型的逐层缩放向量 d 和 b：W' = W + Λb * B * Λd * A |
| **关键优势** | 参数效率极高（比 LoRA 少 10 倍），适合极度资源受限场景 |
| **局限性** | 性能通常略低于 LoRA，不是所有任务都适用 |

### 2.2 目标层选择的性能影响

| 目标层组合 | 参数量占比 | 相对性能 | 说明 |
|:-----------|:--------:|:------:|------|
| 仅 Q, V | ~0.3% | 基线 | 最常用的最小配置 |
| Q, K, V, O | ~0.5% | +1-2% mIoU | 完整的注意力投影微调 |
| Q, K, V, O + MLP | ~0.8% | +2-4% mIoU | 最佳性能，但对 ViT 分割任务收益递减 |
| 仅 MLP | ~0.3% | -2-3% mIoU | 单独微调 MLP 不如注意力层 |

**经验规律**：对于 DINOv3 分割任务，Q、K、V、O 四层全用是性价比最高的选择。MLP 层的额外微调在医学分割的小数据集上可能带来过拟合风险。

### 2.3 秩选择的经验规律

| 秩 r | 参数量 | 训练速度 | 性能 | 推荐场景 |
|:----:|:------:|:------:|:----:|----------|
| 4 | 极小 | 最快 | 中等（~90% 全微调） | 快速实验、初步探索 |
| 8 | 小 | 快 | 良好（~93% 全微调） | 默认选择，性价比最优 |
| 16 | 中等 | 中等 | 好（~96% 全微调） | 追求高精度、数据充足 |
| 32 | 较大 | 较慢 | 优秀（~98% 全微调） | 大型模型、复杂任务 |
| 64+ | 大 | 慢 | 接近全微调 | 通常不必要，直接全微调更简单 |

**关键规律**：
- 对于 ViT 模型，r=8-16 通常是最佳的性能/效率平衡点
- DoRA 在 r=4-8 时即可达到 LoRA r=16-32 的性能
- 模型越大，有效秩越低（大模型内在维度低）
- 复杂医学分割任务（多器官、细粒度边界）建议 r=16 起步

### 2.4 PEFT 库 vs 自定义实现

| 维度 | HuggingFace PEFT | 自定义实现 |
|------|:----------------:|:----------:|
| **开发效率** | 极高（几行代码） | 低（需从头编写） |
| **LoRA 变体支持** | LoRA、DoRA、AdaLoRA | 可定制任意变体 |
| **ViT 兼容性** | ✅ 原生支持 | 需自行适配 |
| **训练稳定性** | 经过充分测试 | 需自行调试 |
| **灵活性** | 受限于框架设计 | 完全自由 |
| **合并权重** | 内置 `merge_and_unload()` | 需自行实现 |
| **社区支持** | 大量文档和示例 | 无 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐（仅特殊需求时） |

**结论**：强烈推荐使用 HuggingFace PEFT。DoRA 支持已成为 `use_dora=True` 一行代码的事。

### 2.5 LoRA 推荐总结

| 优先级 | 变体 | 理由 |
|:------:|------|------|
| **P0（必做）** | LoRA r=8, 目标层：QKV | 基线实现，最成熟稳定 |
| **P0（必做）** | LoRA r=8, 目标层：QKVO | 完整注意力微调，性价比最高 |
| **P1（高优先）** | DoRA r=8, 目标层：QKVO | ICML 2024 最新方法，低秩下性能显著优于 LoRA |
| **P2（可做）** | LoRA r=16 | 高精度场景 |
| **P2（可做）** | LoRA+ | 极简改进，仅改学习率即可获得提升 |
| **P3（探索性）** | AdaLoRA | 自动秩分配，适合深入理解各层重要性 |

---

## 3. Adapter 方法

### 3.1 各 Adapter 架构详解

#### 3.1.1 Bottleneck Adapter（Houlsby et al., ICML 2019）

| 属性 | 说明 |
|------|------|
| **核心思路** | 在 Transformer 每层的 MHSA 和 MLP 之后串行插入一个轻量瓶颈模块：下投影（d → m）→ 非线性（GELU/ReLU）→ 上投影（m → d），带残差连接 |
| **插入位置** | MHSA 之后 + MLP 之后（每个 block 两个 adapter） |
| **参数预算** | 通常占骨干参数 1-5%。瓶颈维度 m << d（如 d=768, m=64） |
| **ViT 相关工作** | 3DSAM-adapter（arXiv 2306.13465）：在瓶颈中插入 3D 深度卷积，扩展用于体积医学图像 |
| **DINOv3 直接应用** | ❌ 未发现直接应用于 DINOv3 分割的文献 |

#### 3.1.2 AdaptFormer（Chen et al., NeurIPS 2022）

| 属性 | 说明 |
|------|------|
| **核心思路** | 在每个 ViT 层的 MLP 块旁边并行插入一个轻量瓶颈模块（下投影 → GELU → 上投影），加上可学习的缩放因子 s 控制 adapter 贡献：输出 = X + FFN(LN(X)) + s * Adapter(LN(X)) |
| **插入位置** | 并行于每个 Transformer block 中的 MLP/FFN 子层（在注意力之后） |
| **关键洞察** | 对于视觉任务，并行插入比串行插入更有效 |
| **参数预算** | <2% 骨干参数。ViT-B 约 1.5% 额外参数 |
| **原始验证场景** | 图像/视频分类（ImageNet、SSv2、HMDB51），非分割任务 |
| **社区扩展** | 部分项目将 AdaptFormer 思想迁移到分割任务，可作为 DINOv2 分割的 PEFT 选项 |

#### 3.1.3 ViT-Adapter（Chen et al., ICLR 2023 Spotlight）

| 属性 | 说明 |
|------|------|
| **核心思路** | 在 ViT 中注入空间先验信息。包含两个关键模块：(1) Spatial Prior Module（SPM）：从输入图像提取多级卷积特征；(2) Interaction Module：在 ViT 各层将 SPM 特征与 patch tokens 交叉注意力融合 |
| **插入位置** | 每 N 个 ViT block 之后插入一个 Interaction Module |
| **特点** | 不修改 ViT 原有预训练权重，通过侧网络注入多尺度空间信息 |
| **DINOv2 验证** | ✅ ViT-Adapter 与 DINOv2 的组合在密集预测任务中验证有效 |
| **参数量** | 中等（约 10-20M 额外参数） |

#### 3.1.4 ConvPass

| 属性 | 说明 |
|------|------|
| **核心思路** | 通过一系列卷积上采样层直接将 ViT patch tokens 恢复到输入分辨率 |
| **架构** | 在 ViT 每层提取 patch tokens → 通过可学习的卷积通道逐步上采样 |
| **特点** | 极简，无 transformer 组件，纯粹通过卷积学习空间上采样 |
| **局限性** | 性能上限较低，缺少跨尺度特征交互 |

### 3.2 各方法性能对比

| 方法 | 参数量 (% 骨干) | 训练内存 | 推理延迟 | 分割性能 (相对) | ViT 任务泛化 |
|------|:------------:|:------:|:------:|:-----------:|:----------:|
| 全微调 | 100% | 高 | 无额外 | ⭐⭐⭐⭐⭐ | N/A（基线） |
| LoRA (r=8) | ~0.5% | 低 | 无（可合并） | ⭐⭐⭐⭐ | 强（跨所有任务） |
| DoRA (r=8) | ~0.5% | 低 | 无（可合并） | ⭐⭐⭐⭐½ | 强（跨所有任务） |
| Bottleneck Adapter | ~2-5% | 中低 | 轻微增加 | ⭐⭐⭐ | 中（需针对性设计） |
| AdaptFormer | ~1.5-2% | 中低 | 轻微增加 | ⭐⭐⭐½ | 中（视觉任务优化） |
| ViT-Adapter | ~10-20% | 中 | 明显增加 | ⭐⭐⭐⭐½ | 强（密集预测优化） |
| ConvPass | ~5-10% | 中低 | 轻微增加 | ⭐⭐⭐ | 弱 |

### 3.3 Adapter 推荐与优先级

| 优先级 | 方法 | 理由 |
|:------:|------|------|
| **P0（必做）** | LoRA / DoRA | 参数效率极高、生态成熟、零推理延迟，应作为项目的主要 PEFT 方案 |
| **P1（高优先）** | ViT-Adapter | 密集预测任务专用，性能接近全微调但参数少得多；可作为一种介于 LoRA 和全微调之间的方案 |
| **P2（可做）** | AdaptFormer | 视觉任务优化的并行 adapter，实现简单，适合作为对比实验 |
| **P3（低优先）** | Bottleneck Adapter | NLP 起源，视觉分割效果未经充分验证，优先度低 |
| **P3（低优先）** | ConvPass | 性能上限低，不建议作为主力方案 |

---

## 4. 对项目实现的建议

### 4.1 应增加的解码头

**路线图建议**（按实现顺序）：

```
Phase 1（MVP）:
  ├── Linear Head          ← 已有，保持
  ├── 轻量 Conv Head       ← 快速添加（~100 行代码）
  └── SegFormer Head       ← 快速添加（~200 行代码，全 MLP）

Phase 2（核心扩展）:
  ├── UPerNet Head         ← 优先实现（mmsegmentation 有参考实现）
  └── DPT Head             ← 推荐实现（DINOv2 社区验证充分）

Phase 3（高级能力）:
  ├── Mask2Former（可训练版）← 复杂度高，需要完整的 DINOv3_Adapter + 像素解码器 + Transformer 解码器
  └── DinoUNet (FAPM)      ← 医学分割专用，值得在目标数据集上验证
```

### 4.2 LoRA 实现增强建议

**当前状态假设**：项目可能已支持基础 LoRA（r=8, QKV）。

**建议增强**：

1. **增加 DoRA 支持**（通过 HF PEFT `use_dora=True`）
   - 改动极小，性能提升显著
   - 建议设为默认选项

2. **扩展目标层选择**
   - 当前层级：Q, K, V
   - 建议增加：Q, K, V, O（`target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`）
   - 可选增加 MLP 层供高级用户选择

3. **支持多秩配置**
   - 提供预设：`tiny` (r=4), `base` (r=8), `large` (r=16)
   - 通过 config 文件切换

4. **LoRA+ 学习率策略**
   - 为 A 和 B 矩阵设置不同的学习率（B 的 lr 建议为 A 的 2-4 倍）
   - 如果使用 HF PEFT，可通过参数分组实现

### 4.3 Adapter 实现方案与优先级

**核心判断**：在 LoRA/DoRA 已经能覆盖大部分 PEFT 需求的前提下，Adapter 的边际收益有限。建议优先级如下：

1. **LoRA / DoRA**（P0）：主力 PEFT 方案，覆盖 80% 的使用场景
2. **ViT-Adapter**（P1）：作为“准全微调”方案，用于对精度要求最高的场景。实现要点：
   - Spatial Prior Module：从输入图像提取多尺度卷积特征（使用预训练的轻量 CNN 骨干或随机初始化）
   - Interaction Module：使用交叉注意力将空间特征注入 DINOv3 patch tokens
   - 每 3-4 个 ViT block 插入一个 Interaction Module
3. **AdaptFormer**（P2）：作为对比实验的可选方案，实现简单（约 50 行/层）

**实现方案**：

```
LoRA/DoRA: HF PEFT 集成（推荐）
  ├── peft_config = LoraConfig(r=8, lora_alpha=16, use_dora=True,
  │       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
  └── model = get_peft_model(dinov3_backbone, peft_config)

ViT-Adapter: 自定义实现
  ├── 在 dinov3 的 forward 中 hook 中间层输出
  ├── 通过 InteractionBlock 注入 SPM 特征
  └── 参考：facebookresearch/detectron2 中的 ViT-Adapter 实现

AdaptFormer: 最小实现
  ├── 在每个 ViT block 的 FFN 旁路添加 Bottleneck(Linear→GELU→Linear)
  └── 通过 Hook 或修改 forward 实现
```

### 4.4 整体技术路线图

```
                    ┌──────────────────────────┐
                    │    DINOv3 Backbone        │
                    │    (Frozen / LoRA-tuned)  │
                    └───────────┬──────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
      ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
      │  PEFT 层   │      │  Adapter   │      │  解码头    │
      │            │      │            │      │            │
      │ • LoRA     │      │ • ViT-Adp  │      │ • Linear   │
      │ • DoRA     │      │ • AdaptFmr │      │ • ConvHead │
      │ • LoRA+    │      │            │      │ • UPerNet  │
      └─────┬──────┘      └─────┬──────┘      │ • DPT      │
            │                   │             │ • SegFormer│
            └───────────────────┼─────────────│ • M2F      │
                                │             └─────┬──────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │   分割输出         │
                                          │   (H × W × C)     │
                                          └───────────────────┘
```

### 4.5 关键设计原则

1. **解码头与骨干解耦**：解码头应为独立模块，接收骨干多尺度特征输出，便于灵活替换。
2. **PEFT 方法互斥但可切换**：LoRA 和 ViT-Adapter 不应同时使用（会相互干扰），但应通过配置轻松切换。
3. **配置驱动**：所有解码头和 PEFT 方法选择应通过 YAML/JSON 配置文件控制，避免硬编码。
4. **医学分割优先**：在选择默认配置和超参数时，优先考虑医学图像分割的特点（小目标、精细边界、类别不平衡），而非自然图像分割的经验。
5. **保持与官方 DINOv3 的兼容性**：确保项目可以无缝加载官方 DINOv3 预训练权重和 Mask2Former 权重。

---

## 参考文献

- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.
- Liu, S.-Y., et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024 (Oral).
- Zhang, Q., et al. (2023). "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning." ICLR 2023.
- Hayou, S., et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models." arXiv:2402.12354.
- Kopiczko, D. J., et al. (2024). "VeRA: Vector-based Random Matrix Adaptation." ICLR 2024.
- Chen, S., et al. (2022). "AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition." NeurIPS 2022.
- Chen, Z., et al. (2023). "Vision Transformer Adapter for Dense Predictions." ICLR 2023 Spotlight.
- Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML 2019.
- Ranftl, R., et al. (2021). "Vision Transformers for Dense Prediction." (DPT) ICCV 2021.
- Xiao, T., et al. (2018). "Unified Perceptual Parsing for Scene Understanding." (UPerNet) ECCV 2018.
- Xie, E., et al. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021.
- Zheng, S., et al. (2021). "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers." (SETR) CVPR 2021.
- Cheng, B., et al. (2022). "Masked-attention Mask Transformer for Universal Image Segmentation." (Mask2Former) CVPR 2022.
- Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193.
- Darcet, T., et al. (2024). "DINOv3: Scaling Vision Self-Supervised Learning." (Meta AI)
