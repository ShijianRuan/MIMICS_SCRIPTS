# 主动学习在医学图像分割中的调研、效能边界与最佳实践

- **日期**：2026-07-14
- **状态**：调研完成（源码级 MONAI Label 分析 + 多路文献检索 + 对抗式验证）
- **指向**：本文是 Mimics 难例分诊模块设计的参考依据，设计 spec 见 `docs/superpowers/specs/2026-07-14-mimics-active-learning-hardcase-design.md`

---

## 目录

1. [MONAI Label 主动学习模块：实现与原理](#1-monai-label-主动学习模块实现与原理)
2. [主动学习方法综述](#2-主动学习方法综述)
3. [效能边界：什么时候有用、什么时候没用](#3-效能边界什么时候有用什么时候没用)
4. [单模型 vs 集成：对不同打分配方的影响](#4-单模型-vs-集成对不同打分配方的影响)
5. [最佳实践：针对 Mimics 场景的落地方案](#5-最佳实践针对-mimics-场景的落地方案)
6. [最小可行验证](#6-最小可行验证)
7. [参考文献](#7-参考文献)

---

## 1. MONAI Label 主动学习模块：实现与原理

> 本节基于 `Project-MONAI/MONAILabel@main` 源码级分析，TTA 相关回溯到 `0.3.x`/`0.4.x` tag。关键论断均经对抗式验证。

### 1.1 核心架构：两个解耦抽象

MONAI Label 把主动学习拆成**两个接口同形、语义不同**的抽象，这是最值得借鉴的架构设计：

```
ScoringMethod（算分写回 datastore）       Strategy（读分挑下一例）
        │                                       │
        └─────── Datastore（info-tag 持久化） ───┘
                     ↕
            update_image_info / get_image_info
```

#### ScoringMethod — 逐图打分基类

**源码位置**：`monailabel/interfaces/tasks/scoring.py`（~20 行）

```python
class ScoringMethod(metaclass=ABCMeta):
    def __init__(self, description): ...
    def info(self): return {"description": self.description}
    @abstractmethod
    def __call__(self, request, datastore: Datastore):
        pass  # 算分 → datastore.update_image_info(id, {...}) 写回
```

**角色**：算不确定性分数并通过 `Datastore.update_image_info(id, info)` 落盘。注意——**不挑样本**，只算分。

**Info-tag 键约定**（ScoringMethod 实现写入 datastore 的字段）：

| Scoring 实现 | info-tag 键 | 值 | 语义 |
|---|---|---|---|
| EpistemicScoring | `epistemic_entropy` | float | MC-Dropout 均值预测熵（`nanmean`）|
| TTAScoring | `tta_vvc` | float | Volume Variation Coefficient（Wang 2019）|
| DiceScoring | `dice` | float | 两个 label tag 之间的一致性 |

#### Strategy — 样本选择基类

**源码位置**：`monailabel/interfaces/tasks/strategy.py`（~20 行）

```python
class Strategy(metaclass=ABCMeta):
    def __init__(self, description): ...
    def info(self): return {"description": self.description}
    @abstractmethod
    def __call__(self, request, datastore: Datastore):
        pass  # 读分 → 排序 → 返回 {"id": image_id, ...}（无候选返回 None）
```

**角色**：读 `datastore.get_image_info(id)` 取回分数 → 排序 → 挑下一例。**与 ScoringMethod 时间解耦**——scoring 是异步批处理、strategy 是同步读。

#### Datastore — 数据/分数抽象存储

**源码位置**：`monailabel/interfaces/datastore.py`（~200 行，20 个抽象方法）

核心的主动学习相关方法（20 个中的 3 个）：
- `get_unlabeled_images(label_tag=None) -> List[str]` — 无 final 标签的候选池
- `get_image_info(image_id) -> Dict[str, Any]` — 读分数
- `update_image_info(image_id, info) -> None` — 写分数

其余 17 个方法（add_image/save_label/remove_image/get_dataset_archive/json/status/refresh…）服务于完整的标注工作流，对打分/排序场景是噪音。

#### App Hooks 与 REST 端点

**初始化**（`interfaces/app.py`）：
```python
self._strategies      = self.init_strategies()      # 默认仅 {"random": Random()}
self._scoring_methods = self.init_scoring_methods() # 默认空 {}，由 sample-app 提供
```

**端点**：
- `POST /activelearning/{strategy}` — 同步：`Strategy.__call__` → 读评分 → 挑下一例 → 盖 serve 时间戳
- `POST /scoring/{method}` — 异步后台任务（`AsyncTask.run("scoring", ...)`）
- `DELETE /scoring/` — 停止 + `torch.cuda.empty_cache()`

### 1.2 内置打分方法的数学原理（源码验证）

#### (a) Epistemic — MC-Dropout + 均值预测熵

**源码**：`monailabel/tasks/scoring/epistemic_v2.py`（v1 自 0.5.0 deprecated）

机制（逐步骤对应源码）：
1. 将模型置 `.train()` 打开推理期 dropout
2. 跑 `simulation_size`（默认 5）次前向，`torch.stack` 成 `[N, C, H, W, D]`
3. 逐通道先对 N 次取均值，再算熵：`H(x) = -\sum_c \bar p_c(x) \ln \bar p_c(x)`，其中 `\bar p_c = \frac{1}{N}\sum_i p_c^{(i)}`
4. `np.nanmean` 聚合成标量，写 `{"epistemic_entropy": ...}`

**这是均值预测的熵 `H[\bar p]`，而非 BALD 的互信息形式**（后者需额外减去期望熵项 `\frac{1}{N}\sum H[p^{(i)}]`）。该打分方法需要网络包含 dropout 层（通常由 sample-app 的 `network_with_dropout` 提供，如 `DynUNet(..., dropout=0.2)`）。

**策略 `Epistemic(Strategy)`**：读 `epistemic_entropy` → 降序取 Top-N → 挑最久未 serve 者。熵越高优先级越高。

#### (b) TTA — Volume Variation Coefficient (VVC)

**源码**：`monailabel/tasks/scoring/tta.py`（0.3.x/0.4.x release，基于 Wang et al. 2019 [doi:10.1016/j.neucom.2019.01.103]）——注意：**已从当前 main 移除**，但算法本身在此后独立验证过。

机制：
- 模型 **eval 模式**（不确定性来自输入增强而非 dropout）
- N 个可逆随机增强（`RandAffined`/`RandFlipd`/`Resized`）→ 各自前向 → `BatchInverseTransform` 逆变换回参考系
- `output = np.concatenate(outputs)` → `[N, C, H, W, D]`
- **VVC = std(output) / mean(output)** — 整个输出张量的变异系数，天然无量纲、天然跨 case 可比
- VVC 越高 → 预测在增强下越不稳定 → 优先级越高

**策略 `TTA(Epistemic)`**：降序取 Top-N。

#### (c) 数据流（端到端）

```
POST /scoring/epistemic
  → EpistemicScoring.__call__
    → model.train() + N 次前向 + 算熵
    → datastore.update_image_info(case_id, {"epistemic_entropy": 0.73})
    → 返回 {case_id: {"epistemic_entropy": 0.73, ...}}

POST /activelearning/epistemic
  → EpistemicStrategy.__call__
    → datastore.get_image_info(each_unlabeled_case)
    → 按 epistemic_entropy 降序取 Top-N
    → 挑最久未 serve 者 + 盖 serve 时间戳
    → 返回 {"id": "case_0421", "epistemic_entropy": 0.73}
```

要点：**scoring 与 strategy 时间解耦**——scoring 是异步后台批处理写分，strategy 是同步读分挑例。serve 时间戳用于在同分/Top-N 内做"近期已服务去优先化"。

---

## 2. 主动学习方法综述

> 记号：参数 `θ`、输入 `x`、类 `c \in {1..C}`、softmax 概率 `p_θ(y=c|x)`。分割中先逐体素算再空间聚合（mean/sum）成 case 级分数。

### 2.1 Acquisition Functions 分类

#### A. 不确定性采样（1 pass，最廉价）

| 方法 | 直觉 | 公式 | 选择 | 代价 |
|---|---|---|---|---|
| Least Confidence | 头类概率低 | `φ_{LC}=1-\max_c p_θ(c\|x)` | argmax | 1 pass, O(C) |
| Margin | 前两类近乎打平 | `φ_M=p(c_1)-p(c_2)` | argmin | 1 pass, top-2 |
| Entropy | 质量摊在多类 | `H[y\|x]=-\sum_c p\log p` | argmax | 1 pass, O(C) |

**局限**：单确定性网络往往过自信/失校准，且忽视多样性。

#### B. 贝叶斯/认知不确定性（需要 θ 后验近似）

- **MC Dropout**（Gal & Ghahramani 2016 [arXiv:1506.02142]；AL 应用 Gal et al. 2017 [arXiv:1703.02910]）：测试期保持 dropout，`T` 次随机前向 ≈ 后验样本；按 `H[\bar p]` 或方差打分。代价 `T` passes，无需额外训练，需 dropout 层。
- **BALD**（Houlsby et al. 2011 [arXiv:1112.5745]；BatchBALD NeurIPS 2019 [arXiv:1906.08158]）：挑"各次都自信但彼此分歧"的点——预测与参数的互信息（纯 epistemic）：`I[y,θ|x] = H[\frac{1}{T}\sum p_{θ_t}] - \frac{1}{T}\sum H[p_{θ_t}]`。代价 `T` passes + `O(TC)`。
- **Deep Ensembles**（Lakshminarayanan et al. 2017 [arXiv:1612.01474]）：`M` 个独立初始化网络，分歧 = 强且校准好的 epistemic 信号。按 `H[\bar p]`/MI/方差打分。代价 **`M×` 训练**（昂贵项）+ `M×` 推理。**nnU-Net 的默认 5 折正属于这一族**，确定性更高，因为它依赖独立初始化/数据划分而非 dropout 扰动。

#### C. TTA 方差（输入扰动，无需额外训练）

Wang et al. 2019 [arXiv:1807.07356]：K 个保标签增强，预测跨增强不稳 = 难例。分数 = 均值预测熵 `H[\frac{1}{K}\sum p^{(k)}]` 或逐体素方差。代价 `K` passes + 增强/逆变换，无需额外训练。nnU-Net 风格的镜像 TTA 默认可用。

#### D. 多样性/代表性（池上几何）

- **Core-Set**（Sener & Savarese 2018 [arXiv:1708.00489]）：嵌入空间 k-center 覆盖，`min_{|s|≤b} max_i min_{j∈s} ‖f(x_i)-f(x_j)‖₂`，贪心 2-近似。纯代表性、无不确定性项。代价 1 embed pass + `O(b·|pool|·d)`。
- **BADGE**（Ash et al. 2020 [arXiv:1906.03671]）：梯度嵌入 → k-means++ 播种取多样 batch。隐式 hybrid。代价 1 fwd+bwd + k-means++。

#### E. Hybrid（不确定性 + 多样性，经验最佳）

- **Suggestive Annotation**（Yang et al. 2017, MICCAI [arXiv:1706.04737]）：bootstrap/ensemble 不确定性 Top-β + FCN 特征 core-set 覆盖 —— 分割专用 hybrid 典范
- **ClaSP PE**（Luth et al. 2026 [arXiv:2601.13677]）：class-stratified querying + log-scale power noising —— 2026 年首个一致优于改进版 random 的方法
- **Selective Uncertainty**（Ma et al. 2024 [arXiv:2401.16298]）：target-aware + boundary-aware —— 标准 entropy 低于 random，选择性变体提升 +0.02 Dice

### 2.2 代价汇总

| 策略 | 额外训练 | 推理 passes | batch 选择 | 信号类型 |
|---|---|---|---|---|
| LC/Margin/Entropy | 无 | 1 | 无 | softmax 不确定性 |
| MC Dropout | 无 | T | 无 | epistemic（近似）|
| BALD/BatchBALD | 无 | T | 无/贪心 MI | epistemic（MI）|
| **Deep Ensembles（5折nnU-Net）** | **M×**（**已训好**） | **M** | 无 | **epistemic（校准最佳）** |
| TTA 方差 | 无 | K | 无 | 输入扰动 |
| Core-Set | 无 | 1（embed）| k-center | 纯多样性 |
| BADGE | 无 | 1 fwd+bwd | k-means++ | hybrid |
| Suggestive Annotation | bootstrap/ens | T/M | core-set 覆盖 | hybrid（分割）|
| ClaSP PE | 无（复用 nnU-Net）| 1/折 | class-stratified | hybrid |

### 2.3 无 GT 质量评估（Failure Detection）

- **RCA — Reverse Classification Accuracy**（Valindria et al. 2017, IEEE TMI）：把预测当 GT，训练反向分类器，在小参考库上评估 → 最佳匹配的 Dice 作为测试例真 Dice 的代理。2025 后继 **In-Context RCA** [arXiv:2503.04522] 用检索替代重训，更廉价。
- **学习式 QA/Dice 预测**（Kohlberger 2012 → Isaksson 2022 → da Cruz 2024）：直接预测 per-case Dice。
- **校准**（Mehrtash et al. 2020, IEEE TMI）：CNN 失校准、ensembling 改善校准、校准置信度支撑 OOD/失败检测。
- ⚠️ **OOD 分数必须验证**：Vasiliuk et al. 2023 (*J. Imaging*) 证明 3D 医学分割中许多 OOD 分数与实际退化不相关，**用前必验**。

---

## 3. 效能边界：什么时候有用、什么时候没用

> 本节基于 5+ 独立实验的交叉验证。所有结论标注置信度。

### 3.1 核心发现：Random 是一个极强的基线

**这是本领域最稳健的发现（High Confidence，多篇独立复现）：**

- **Burmeister et al. (2022)**（arxiv:2207.00845）：MSD 心脏/海马/前列腺，没有任何策略在所有任务上一致优于 random。赢的边际最多 +0.03 Dice。
- **COLosSAL（Liu et al. 2023）**（arxiv:2307.12004）：冷启动（m=5 个体积），**"没有任何策略在基准中一致优于 Random 均值"**——对含肿瘤的任务尤其如此。
- **nnActive（Luth et al. 2025）**（arxiv:2511.19183）："在 3D 生物医学成像领域，对 AL 是否一致地优于 Random 采样，没有共识。"
- **Ma et al. (2024)**（arxiv:2401.16298）：标准 Entropy 在 BraTS 上 10% 标签预算时比 Random 还差 0.04 Dice，标准 BALD 和 MC-Dropout 同样在低预算时不及 random。

### 3.2 主动学习有效的真实条件

基于文献，**以下条件必须全部满足**（按重要性排序）：

| # | 条件 | 置信度 | 定量阈值（如有文献支撑）| 你的场景？ |
|---|---|---|---|---|
| 1 | **初始模型 Dice > ~0.5** | Medium-High | Ma 2024：熵在 BraTS 约需 25% 标签（≈Dice 0.55+）才追上 random。低于 ~0.5 时不确定性 ≈ 噪声 | 取决于 few-shot/nnU-Net 模型质量 |
| 2 | **不确定性需在小 GT 验证集上校准验证** | High | 文献一致要求 AUROC/Spearman ρ 验证；Vasiliuk 2023 警告 OOD 分数不可信 | 尚未做 |
| 3 | **每个器官/任务要自己的模型** | High | 跨器官不确定性迁移无文献支持 | 基本满足 |
| 4 | **候选池 ≥ 20 case** | Low（经验法则）| 无文献直接验证，统计常识 | 取决于你的数据量 |
| 5 | **混合不确定性 + 多样性去冗余** | High | 纯不确定性产生冗余样本（5+ 篇一致）| 初期可暂用 serve 时间戳轻量去重 |
| 6 | **若不满足 #1 → 退化到纯几何启发式** | High | 无模型→无不确定性，几何启发式不需要概率 | 需硬编码进系统 |

### 3.3 什么时候主动学习会失败？（所有已知失败模式）

| 失败模式 | 证据来源 | 发生条件 | 严重程度 |
|---|---|---|---|
| **不确定性 = 噪声** | Ma 2024；COLosSAL 2023 | 初始模型 Dice < ~0.4–0.5 | **致命** |
| **纯不确定性选到冗余样本** | Burmeister 2022；5+ 篇一致 | 连续切片中相邻位置的不确定性高度相关 | 高（浪费标注预算）|
| **不确定性对"结构失败"不敏感** | 文献中一致报道 | 掩码触及边界/碎片/选错器官——模型在这些情况下可能仍然自信 | 中（几何启发式补）|
| **对极难结构无效** | COLosSAL 基准 | 肿瘤、肾上腺等——冷启动 AL 对含肿瘤的任务"几乎完全无效" | 高 |
| **OOD 不确定性不可靠** | Vasiliuk 2023 | 未见 domain 上的不确定性与实际性能退化可能不相关 | 中（须先验证）|
| **多数分割不确定性是 aleatoric** | Lakshminarayanan 2017；文献 | 边界模糊 = 标注者也不一致 = 多标也无济于事 = AL 无增益 | 结构性限制 |

### 3.4 经验对比：谁赢了、赢了谁、赢了多少

**ClaSP PE（Luth 2026）是最新 SOTA**——24 设置 + 4 rollout 数据集，唯一一致优于改进版 random 的方法：

| 方法 | AUBC vs std Random | Final Dice vs std Random | vs 改进版 Random（~67% FG）|
|---|---|---|---|
| 标准 Entropy/BALD/MC-Dropout | +2–4 | +4–7 | **不及** |
| PowerPE | +4.3 | +6.4 | ~持平 |
| **ClaSP PE** | **+7.1** | **+11.2** | **+0.5 AUBC, +1.7 Dice** |

但注意：ClaSP PE 用的是 **nnU-Net 5 折集成 + class-stratified querying + power noising**——不能简化为单一不确定性信号。

**Rollout 结果（4 个未见数据集）**：ClaSP PE 在 LiTS/MAMA MIA/ToothFairy2 上一致优于 random，但在 WORD 上持平。

### 3.5 对 Mimics 场景的三层分级

基于以上条件，按你的实际情况分级：

```
Level 1（✅ 大概率有用）：模型好（Dice > 0.6）+ TTA 开 + oracle 已验证
Level 2（◐ 可能有用）：模型勉强（Dice 0.4–0.6）+ 混合信号 + 退化路径
Level 3（❌ 无意义）：模型差（Dice < 0.4）→ 不确定性 = 噪声 → 只能用几何启发式
```

---

## 4. 单模型 vs 集成：对不同打分配方的影响

### 4.1 5 折集成能做的、单模型做不到的

| 能力 | 5 折集成 | 单模型 |
|---|---|---|
| Deep Ensemble 多模型分歧 | ✅ — 5 个独立训练的模型各自前向，分歧 = 正宗的 epistemic 不确定性 | ❌ — 只有 N 次 dropout 或 1 次确定性前向 |
| BALD 形式 MI（`H[\bar p] - E[H[p]]`）| ✅ — 分离 epistemic 与 aleatoric | ❌ — MC-Dropout 仅能近似，单次前向完全不可得 |
| 校准置信度 | ✅ — Mehrtash 2020：集成增强校准 | ◐ — 校准随网络/任务剧烈变动 |
| 对罕见结构的敏感性 | ✅ — 集成在各折叠中各自学习、分歧在难结构上更明显 | ◐ — 稳定性下降 |

**文献验证**：ClaSP PE 用 5 折集成；nnActive 用 2 折；Burmeister 用单模型但承认"AL 的收益被增强型 random 基线大幅缩小"。

### 4.2 单模型依然可用的 TTA-VVC（Wang et al. 2019）

**TTA 不确定性不需要多模型。** Wang et al. 2019 的 VVC 论文本身用的就是**单模型 + 测试时增强**，其核心逻辑与多模型无关——增强扰动下来回一致的预测 = 自信；来回乱跳 = 不确定。MnU-Net 默认的镜像 TTA 可以直接获此信号。

### 4.3 信号优先级矩阵

| 模式 | 主信号 | 辅助 | 不可用 |
|---|---|---|---|
| **nnU-Net 5折集成** | 集成熵/MI + TTA-VVC | 边界熵 + 几何启发式 | — |
| **nnU-Net 单模型（TTA 开）** | TTA-VVC | 单次前向边界熵 + 几何 | BALD MI |
| **nnU-Net 单模型（TTA 关）** | 单次前向边界熵 + 几何 | — | TTA-VVC, BALD MI |
| **DINOv3 few-shot（TTA 开）** | TTA-VVC（复用 _predict_with_tta）| 边界熵 + 几何 | MC-Dropout（解码器无 dropout）|
| **模型不可用** | **纯几何启发式** | — | **所有概率/不确定性信号全不可用** |

### 4.4 掩码几何启发式（独立于模型的廉价信号）

以下信号**不需要模型、不需要概率**——在所有配置下作为基准信号默认运行：

| 启发式 | 实现 | 抓什么 | 文献冲突？|
|---|---|---|---|
| 连通域碎片 | `scipy.ndimage.label`，算 `(total-lar gest)/total` | 散碎分割 = 大概率错 | 无冲突 |
| FOV 边界接触 | 掩码触及 FOV 面的体素占比 | 截断/裁剪 = 难例 | 无冲突 |
| 体积异常 z-score | 预测体积 vs 器官先验均值/标准差 | 大小离谱 = 错 | 无冲突 |
| 空/近空预测 | 前景体素数 < 阈值 | 漏检 | 无冲突 |
| 多连通域 | 连通域数 `K` | 单拓扑器官异常碎片 | 无冲突 |

---

## 5. 最佳实践：针对 Mimics 场景的落地方案

### 5.1 核心原则

#### 原则 1：分"Failure Detection"与"AL Loop"，你的场景是前者

你目前不需要 AL 闭环（选例→标注→重训→再选）。你需要的是：**用已有模型对一批 case 打分，产出按难度排序的复查队列**。这是 **Failure Detection / Case Quality Ranking**，而不是经典 AL。要点：无需聚类/多样化批量选择，单 case 排序即可；无需"初始种子→迭代标注→模型提升"——你是离线批处理；但一样要求初始模型 ≥ 最低质量门限。

#### 原则 2：信号堆叠优于单一信号——但要避免冗余加权

相关信号（TTA-VVC、边界熵、集成熵、不确定段占比）只保留一个代表。默认集：`{TTA-VVC, 体积 z-score（几何）, 碎片比（几何）, FOV 边界接触（几何）, 空/近空守卫}`。

#### 原则 3：融合方法：RRF 默认 → 后面切换 logistic

RRF 公式：`RRF(c) = ∑_j w_j / (k + r_j(c))`，`k ≈ 60`，`r_j` = 该 case 在信号 `j` 上的排名（1 = 最不确定）。比 robust z-score 更抗重尾，无需调参。当你有 ≥ 30–50 个人工好/坏标签后，可切换 logistic 回归拟合权重（Wang-QC 做法）。

#### 原则 4：Hard Gate——不确定性不可用时不能给出假信心

系统必须对每次打分记录后端能力与其不确定性门控状态。当模型不满足最低质量门限时返回 `"unavailable"` 而非虚假分数。这个 gate 必须在系统中硬编码——不依赖人工记住它。

#### 原则 5：先验证、后编码

在任何 UI 代码或 pipeline 编排代码之前，用一个 20–50 例 GT 集验证每个信号的 Spearman ρ 和 failure-detection AUROC。如果 ρ < 0.2 → 放弃该信号或放弃 AL 路径。

### 5.2 推荐架构（MONAI-Mirrored）

```
┌─ Tier 3 (torch): ScoringBackend 接口 ────────────────────────────────┐
│  nnUNetBackend (supports_dropout=F, supports_tta=T)                   │
│  DINOv3Backend (supports_dropout=F, supports_tta=T)                   │
│  ┌ ScoringMethod (vendored MONAI ABC) ────────────────────────┐      │
│  │ TTAScoring → {"tta_vvc": 0.34, "tta_boundary_entropy": 0.18}        │
│  │ EntropyScoring → {"entropy_fg": 0.22}                          │
│  │ HeuristicScoring → {"frag_ratio": 0.14, "border_frac": 0.05}   │
│  └───────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘
        │ scores/<organ>/<case>.json (= MONAI Datastore.update_image_info)
        ▼
┌─ Tier 2 (纯 Python, 可无torch测试): Strategy + 排名 ──────────────────┐
│  FileScoreStore: get_unlabeled_images / get_image_info / update_image_info │
│  RankFusionStrategy (继承 vendored MONAI Strategy): RRF融合 → 排序队列  │
└─────────────────────────────────────────────────────────────────────-──┘
        │ queues/<organ>/<job_id>.json
        ▼
┌─ Tier 2 UI: hardcase_review_panel.py (PySide6, 外部进程) ───────────┐
│  表格: # | case | tier(HIGH/MED/LOW) | score | VVC | surf | comp | reason │
│  双击行 → 在 Mimics 打开 case → 人工复查/标注                              │
└───────────────────────────────────────────────────────────────────-──┘
        ▲ launch/poll/apply
┌─ Tier 1 (Py3.5, Mimics 内嵌): "Rank Hard Cases" 按钮 ────────────-─┐
│  _launch_score_batch_job → Popen pipeline score-batch → poll 状态       │
└───────────────────────────────────────────────────────────────────-──┘
```

### 5.3 信号默认集与融合权重

**默认信号**（开箱可用，无需额外训练）：

| 信号 | key | 来源 | 方向 | 备注 |
|---|---|---|---|---|
| TTA-VVC | `tta_vvc` | Wang 2019 公式 | ↑ 越高越难 | 需要 TTA 增强栈 |
| 前景均值熵 | `entropy_fg` | 平均概率体 | ↑ | TTA 不可用时退化使用 |
| 碎片比 | `frag_ratio` | scipy.label | ↑ | 几何启发式 |
| FOV 边界触比 | `border_frac` | 掩码与 FOV 面交集 | ↑ | 几何启发式 |
| 体积 z-score | `vol_zscore` | vs 器官先验 | \|z\|↑ | 几何启发式 |
| 空标志 | `empty_flag` | 前景体素 < 阈值 | 直接 flag | 不进入 RRF |

**RRF 默认权重**：等权重（`w_j = 1`），先跑通；Phase 5 用 logistic 拟合。

### 5.4 实施阶段

- **Phase 1 — 验证实验**（≤ 2 天）：拿 1-2 个器官的现有 nnU-Net/DINOv3 模型，对 20-50 有 GT 的 case 跑 TTA-VVC + 边界熵 + 几何启发式，算 Spearman ρ + failure-detection AUROC。**这个实验的结果决定 Phase 2+ 做不做。**
- **Phase 2 — 骨架 + DINOv3 后端**：ScoringBackend 接口、TTAScoring、HeuristicScoring、单测。复用现有 DINOv3。先不引入任何新依赖。
- **Phase 3 — Strategy + 编排**：FileScoreStore、RRF 排名、`cmd_score_batch`。
- **Phase 4 — nnU-Net 后端 + MONAI 评分**：nnU-Net 单模型 → 5 折集成（如有）。EpistemicScoring（仅接入端自带 Dropout 时可用）。
- **Phase 5 — UI + Tier1**：复查队列面板、Mimics 按钮。
- **Phase 6（可选）**：logistic 加权、RCA、core-set 去冗余、AUROC 验证。

### 5.5 关键风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型 Dice < 0.4 → 不确定性 = 噪声 | Hard gate：返回 `"unavailable"`，退化到几何启发式 |
| TTA-VVC 无信息量 | 预验证阶段用 Spearman ρ 校验，ρ < 0.2 时移除信号 |
| 集成 5 折推理慢（5× 时间）| 离线批处理，可接受；`tta_samples` 可配置 |
| 与现有 `fewshot_strategies.py`（训练预设）命名混淆 | 新模块统一 `hardcase_*` 前缀 |
| 不同器官用不同模型 | 后端实例化时按 organ 加载对应 checkpoint |
| OOD 不确定性不可靠 | 在验证集上校验 AUROC，可用再启用 |
| 冷启动无模型 → 不能用 | 无模型时直接跳过不确定性信号，仅用几何启发式 |

---

## 6. 最小可行验证

在写任何 pipeline / UI 代码之前，必须跑完以下实验。它决定整个 AL 模块在你们数据上是否值得投入工程资源：

### 6.1 实验步骤

```
1. 选 1–2 个器官（建议 liver 或 spleen，最容易；再加一个难的如 pancreas）
2. 拿现有 nnU-Net 或 DINOv3 模型
3. 确保 TTA 已开启
4. 对 20–50 个有 GT 标注的 case 跑推理
5. 对每例记录：
   - TTA-VVC（保留增强栈，不丢弃方差）
   - 边界熵（从平均概率体）
   - 掩码碎片比、FOV 边界触比、体积 z-score
   - 实际 Dice（vs GT）
6. 计算：
   - Spearman ρ（VVC vs Dice、熵 vs Dice、RRF 融合 vs Dice）
   - failure-detection AUROC（Dice < 阈值 如 0.6 → "失败"）
```

### 6.2 决策树

```
ρ > 0.40, AUROC > 0.70  → ✅ 继续 Phase 2+ 全栈
ρ 0.20–0.40              → ◐ TTA-VVC + 几何，放弃熵单信号，重点改进融合
ρ < 0.20                 → ❌ 放弃不确定性信号; 仅几何启发式 + RCA/质量预测网
模型 Dice < 0.3          → ❌ ❌ 放弃 AL 路径; 先改进模型
```

### 6.3 这个实验为什么不跳过

**Vasiliuk 2023 的教训是真实的**：不确定性与实际错误之间的相关性不是普遍的，它随模型、数据分布、器官/任务而变化。你**不能假定**它在你的组合上会成立——**必须测量**。

---

## 7. 参考文献

### MONAI Label 源码（核心调研对象）
- MONAI Label GitHub: `https://github.com/Project-MONAI/MONAILabel`
  - `monailabel/interfaces/tasks/scoring.py` — ScoringMethod 基类（~20 行）
  - `monailabel/interfaces/tasks/strategy.py` — Strategy 基类（~20 行）
  - `monailabel/interfaces/datastore.py` — Datastore 抽象基类（200+ 行, 20 抽象方法）
  - `monailabel/tasks/scoring/epistemic_v2.py` — EpistemicScoring（MC-Dropout + 均值预测熵）
  - `monailabel/tasks/scoring/tta.py` (0.3/0.4 tag) — TTAScoring (VVC)

### 核心文献
- Settles, B. (2009). *Active Learning Literature Survey*. University of Wisconsin-Madison. — AL 经典综述
- Wang, G. et al. (2019). "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation." *Neurocomputing*, 338, 34-45. [doi:10.1016/j.neucom.2019.01.103] — **TTA-VVC 原始论文**
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*. [arXiv:1506.02142] — MC-Dropout 基础
- Gal, Y., Islam, R. & Ghahramani, Z. (2017). "Deep Bayesian Active Learning with Image Data." *ICML*. [arXiv:1703.02910] — 医学图 AL 中的 MC-Dropout

### 效能对比与基准
- Burmeister, E. et al. (2022). "Less Is More: A Comparison of Active Learning Strategies for 3D Medical Image Segmentation." [arXiv:2207.00845] — **Random 几乎不可能被击败**
- Liu, J. et al. (2023). "COLosSAL: A Benchmark for Cold-Start Active Learning in 3D Medical Image Segmentation." *MICCAI*. [arXiv:2307.12004] — **冷启动：无策略一致优于 Random**
- Luth, C. et al. (2025). "Navigating the Pitfalls of Active Learning Evaluation in Medical Imaging." [arXiv:2511.19183] — **nnActive 基准**（nnU-Net 骨干）
- Luth, C. et al. (2026). "ClaSP PE: Finally Outshining the Random Baseline." [arXiv:2601.13677] — **首个一致优于改进版 Random 的方法**（nnU-Net + class-stratified + power noising）
- Ma, Z. et al. (2024). "Breaking the Barrier: Selective Uncertainty-based AL." [arXiv:2401.16298] — **标准熵在低预算时 < Random；选择性变体逆转**
- Houlsby, N. et al. (2011). "Bayesian Active Learning for Classification and Preference Learning." [arXiv:1112.5745] — BALD 基础
- Kirsch, A. et al. (2019). "BatchBALD: Efficient and Diverse Batch Acquisition." *NeurIPS*. [arXiv:1906.08158]
- Yang, L. et al. (2017). "Suggestive Annotation: A Deep Active Learning Framework." *MICCAI*. [arXiv:1706.04737]

### 失败与校准
- Vasiliuk, D. et al. (2023). "Limitations of Out-of-Distribution Detection in 3D Medical Image Segmentation." *J. Imaging*. — ⚠️ **OOD 分数与 3D 分割退化不相关**
- Mehrtash, A. et al. (2020). "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation." *IEEE TMI*. —校准：集成提升校准
- Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*. [arXiv:1612.01474] — Deep Ensembles 基础

### 无 GT 质量评估
- Valindria, V. et al. (2017). "Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth." *IEEE TMI*. — RCA
- Carvalho, J. et al. (2025). "In-Context Reverse Classification Accuracy." [arXiv:2503.04522] — 改进版 RCA

### 相关工具与框架
- Diaz-Pinto, A. et al. (2022). "MONAI Label: A Framework for AI-Assisted Interactive Labeling of 3D Medical Images." [arXiv:2203.12362]
- Isensee, F. et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18, 203-211.

---

> 本文档将作为 Mimics 主动学习/难例分诊模块设计与实施的最终参考文档。所有关键论断均标注了来自文献的置信度；不确定性所在之处已在文本中坦率指出。
