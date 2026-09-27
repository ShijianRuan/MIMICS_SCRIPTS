# Mimics 难例分诊模块设计（MONAI-Mirrored Active Learning）

> **归档说明（2026-09-27）**：本文为历史设计文档，从未实施。文中引用的
> DINOv3 few-shot 模块（`fewshot_pipeline.py`、`fewshot_mimics.py` 等）
> 已于 2026-09-23 整体删除（commit b632d70，改由 FlexiCT 微调路线取代），
> 文中所有 few-shot 路径与模块引用仅存历史价值。`docs/active_learning_
> implementation_plan.md` 亦已不存在。阅读时以
> `docs/MIMICS_PROJECT_ARCHITECTURE_CN.md` 为现状唯一依据。


- **日期**: 2026-07-14（2026-08-01 第二版：按"三点主线"对齐 + 多器官/成本约束勘误 + 交互引擎决策）
- **状态**: 设计待评审（v2 待评审）
- **作者**: Claude (与 ShijianRuan 协作 brainstorming)
- **目标**: 在 Mimics 脚本库中集成"难例分诊"模块（标注提效三点主线中的 **C 线——减少数据量**）：对一批未标注/待复查 case 用模型算**不确定性/难度分数**，产出**按难度排序的复查队列**，人优先复查/标注最难的例。定位为**叠加组件**（决定"先标谁/必标谁"），不是"省 X% 标注"的独立卖点——能对外引用的节省数字必须来自 FG-Random 对照实测。
- **参考**: 调研与最佳实践见 `docs/active_learning_research_and_best_practices.md`；实施规划（三点主线：工具效率 A / 少样本冷启动 B / 减少数据量 C）见 `docs/active_learning_implementation_plan.md`；本 spec 对应 C1（难例分诊）。

---

## 1. 背景与研究基础

### 1.1 需求

用户在 Mimics（Materialise 医学影像分割工具）里已经有一套 DINOv3 few-shot 分割与 nnInteractive 交互分割工作流。标注提效按三点主线组织（详见实施规划）：**A 工具效率**（交互式 AI + 蒙版编辑，确定性最高、先行）、**B 少样本冷启动**（无初始标注器官：预训练交互模型零样本出种子 → DINOv3 few-shot 微调）、**C 减少数据量**（主动学习/难例分诊 + 半监督 + 标签质量检测，验证门控后叠加）。

**本 spec 只覆盖 C1（难例分诊）**：对一批未标注或待复查的 case，用模型跑推理并算**不确定性/难度分数**，产出一个**按难度排序的复查队列**，人优先处理最难/最可能出错的例。多器官场景（ModelMap: CT 16 + MR 8 + 粗分割模型）下**按器官算分、按病例聚合**——队列每行一个病例，显示"最差器官"让标注者知道重点修哪个器官。

### 1.2 MONAI Label 主动学习：实现与原理（调研结论）

MONAI Label 把主动学习拆成**两个接口同形、语义解耦**的抽象——这是本设计要忠实镜像的核心：

| 抽象 | 职责 | 源码 | 契约 |
|---|---|---|---|
| **`ScoringMethod`** | 逐图算不确定性分数并**写回 datastore** | `monailabel/interfaces/tasks/scoring.py` | `__call__(request, datastore)` → `datastore.update_image_info(id, {...})` 落盘 |
| **`Strategy`** | **读回分数**排序、挑下一例 | `monailabel/interfaces/tasks/strategy.py` | `__call__(request, datastore)` → 返回 `{"id": ...}`（无候选返回 `None`） |

**数据流**（两者时间解耦）：
```
POST /scoring/{method}      → 后台异步批处理算分 → update_image_info 写回 {epistemic_entropy / tta_vvc}
POST /activelearning/{strategy} → 同步读分 → 降序取 Top-N → 盖 serve 时间戳去优先化近期已服务例 → 返回下一例
```

**内置打分方法（数学原理）**：
- **Epistemic（MC-Dropout）** `tasks/scoring/epistemic_v2.py`：把模型 `.train()` 打开推理期 dropout，跑 N 次前向，算**均值预测的熵**
  $$H(x) = -\sum_c \bar p_c(x)\ln\bar p_c(x),\qquad \bar p_c(x)=\tfrac1N\sum_i p_c^{(i)}(x)$$
  `nanmean` 聚合成标量，写 `{"epistemic_entropy": ...}`。**熵越高越优先**。⚠️ 是 $H[\bar p]$ 而非 BALD 互信息。
- **TTA（VVC）** `tasks/scoring/tta.py`（0.3/0.4 tag，现 main 已移除）：eval 模式下 N 个可逆随机增强 → 逆变换回参考系 → **Volume Variation Coefficient**
  $$\text{VVC}=\frac{\operatorname{std}(\text{output})}{\operatorname{mean}(\text{output})}$$
  无量纲、天然跨 case 可比。基于 Wang et al. 2019。**VVC 越高越优先**。
- **Random/First**：池 = `get_unlabeled_images`；按"距上次 serve 时长"加权随机 / 取排序首例。

**App hooks**（`interfaces/app.py`）：`init_strategies()` / `init_scoring_methods()` 注册；基础 app 默认只有 `random` 策略、不含 scoring（scoring 由 sample-app 提供）。

### 1.3 类似方法综述（更广，验证过）

| 族 | 代表 | 额外训练 | 推理代价 | 本场景适用性 |
|---|---|---|---|---|
| 不确定性采样 | Least-Conf / Margin / **Entropy** | 无 | 1 pass | ✅ 退化信号 |
| 贝叶斯 | **MC-Dropout** / BALD / BatchBALD | 无 | N passes | ✅（需 dropout 层） |
| 深度集成 | Deep Ensembles | **M× 训练** | M passes | ❌ 违反单模型 |
| 输入扰动 | **TTA / VVC** | 无 | K passes | ✅ 主信号 |
| 多样性 | Core-Set / BADGE | 无 | 1 pass + 聚类 | ◐ 仅成批选样时 |
| 混合（最佳） | **Suggestive Annotation**（分割专用） | 可选 | — | ◐ 后续阶段 |

**无 GT 质量评估**（判断分得好不好、验证 rank 有效性）：Reverse Classification Accuracy (RCA，预测 per-case Dice 代理)、校准（Mehrtash 2020）、OOD（⚠️ Vasiliuk 2023：3D 医学分割中许多 OOD 分数与实际退化不相关，须先验证）。评测指标：**failure-detection AUROC** / **Spearman ρ vs 真 Dice**。

### 1.4 已验证的代码事实（决定设计取舍）

1. 本仓库对 `monai`/`monailabel` **零代码依赖**（仅 dinov3 文档提及）。
2. 项目是**离线可移植部署**（`setup_offline.bat`、`tools/package_portable.py`，`wheels/` 仅 4 个 PySide6 Windows wheel）——每引入一个包都要进离线 wheel 包。
3. `external/dinov3-medical-seg/src/models/decoder_3d.py` 的四个解码器（Linear/MLPProbe/SegFormer3D/TokenPyramid3D）**均无 `nn.Dropout` 层**（只有 GroupNorm/ReLU/GELU）；`dropout` 仅出现在 LoRA/adapter 且默认 0.0，backbone 在 frozen/lora 下冻结。**→ MC-Dropout 在 DINOv3 上开箱不可用**。
4. **训练成本受限（2026-08-01 新增事实）**：nnU-Net 工作流配置 `fold = 0`（单折训练）→ **不存在 5 折集成权重，集成分歧信号默认不可用**，不为打分训练集成；`disable_tta = true`（推理默认关镜像 TTA）→ TTA-VVC 的增强栈**不依赖 nnU-Net 镜像 TTA**，统一复用 DINOv3 的 `_predict_with_tta` 轻量增强作为通用增强组件。
5. **多器官 = 每器官独立模型**（ModelMap.toml）→ 打分按器官调用对应模型；几何启发式"体积 z 分数"需要**每器官体积先验表**。
6. **交互引擎决策（2026-08-01）**：交互修正引擎确定为 **nnInteractive（本地已有资产）**；经三方对比调研（nnInteractive vs MedSAM2 vs DeepEdit，详见 §2.1），**不集成 MedSAM2/DeepEdit**——3D CT/MR 上 nnInteractive 精度/速度占优、依赖更轻，后两者无实质增量。
4. `src/inference.py` 的 `_predict_with_tta`（line 69-80）**已经在跑 TTA**，但 `torch.stack(...).mean(dim=0)` 后把方差丢弃了 → TTA-VVC 几乎零成本可得。
5. 现有三层架构 + 文件通信 + 外部 UI + JSON 轮询范式（见 §2）。

> **⚠️ 效能边界与先决验证实验**：主动学习并非在所有条件下都有效。在编写任何 pipeline/UI 代码前，**必须**先完成 Phase 1 的最小可行验证实验（对 20–50 例有 GT 的 case 测量 Spearman ρ 和 failure-detection AUROC）。如果 ρ < 0.2 或模型 Dice < 0.4，不确定性信号就是噪声——此时应退化到纯几何启发式或放弃 AL 路径。详见详解见 §3.4–§3.5、完整文献依据见 `docs/active_learning_research_and_best_practices.md`。

---

### 1.5 交互引擎工具决策：nnInteractive 为唯一交互引擎（2026-08-01 调研结论）

对 nnInteractive / MedSAM2 / MONAI Label DeepEdit 三工具的对比调研（依据：论文 + 独立评测 + 工程适配评估）：

| 维度 | nnInteractive（选定） | MedSAM2 | DeepEdit |
|---|---|---|---|
| 原理 | **early prompting**：提示编码为额外输入通道，在进入特征提取前拼接（非 SAM 式 latent 融合）；基于 nnU-Net ResEnc-L；120+ 数据集（64,518 volume）训练 | SAM2 的 3D 扩展：Hiera 编码器 + **streaming memory bank**（把切片序列当视频帧传播）；微调 SAM2.1 权重；45.5 万 3D 对训练 | 自动分割 + 交互分割单模型两用：图像 + 正/负点击通道拼接；训练时一半迭代零点击（自动模式）、一半模拟点击；**需带数据训练，非开箱** |
| 擅长 | 原生 3D、2D 提示→3D 分割、点/框/涂鸦/lasso 多提示、open-set | 3D 体积 + **视频/超声时序**、box 为主 | CT/MR 交互标注（需每例训练） |
| 零样本 | ✅ 开箱即用 | ✅ 开箱（但需 GPU，有 CPU 版） | ❌ 必须训练 |
| GPU | 10GB 推荐，<6GB 小物体；**有 torch-free 远程客户端 + Docker server + 离线缓存** | 需 GPU；依赖重（CUDA 12.4 + SAM2 CUDA 扩展编译） | 依赖 MONAI 生态；离线打包成本高 |
| 独立评测 | CVPR 2025 交互 3D 分割挑战 **第 1 名**；BVM 2026 评测：**推理最快且精度较高** | BVM 2026：最"基础"（foundational）；超声/视频场景最强 | 未进入基础模型时代评测（需训练的老方法） |

**裁决：集成 nnInteractive（本地已有），不集成 MedSAM2/DeepEdit。** 理由：
1. 3D CT/MR 静态分割场景（Mimics 业务）上，nnInteractive 精度/速度占优（第三方评测），功能与 MedSAM2 高度重合且更轻；
2. MedSAM2 的唯一实质差异能力是**视频/时序**（超声、内镜）——对 Mimics 静态 CT/MR 不相关；且其依赖（CUDA 12.4 + SAM2 扩展编译）违反离线可移植约束；
3. DeepEdit 无开箱权重、需带数据训练，且精度不如 nnInteractive；唯一价值是 MONAI Label 生态（本模块已裁定不引整包，路线自洽）；
4. **落地动作（线 A 的内容，非本 spec）**：nnInteractive 正式组件化——Mimics 按钮"AI 修正"→ 交互进程（点/涂鸦/框）→ 回写蒙版；与难例分诊衔接：分诊队列的"修正"环节用它。

---

## 2. 现有架构约束（新模块必须遵守）

项目是**三层 + 双 Python 运行时**，只用**文件**（原子 JSON + 文件锁）和 **subprocess（JSON stdin/stdout）** 通信，无 socket/HTTP。

| 层 | 运行时 | 职责 | 代表文件 |
|---|---|---|---|
| **Tier 1** | Mimics 内嵌 **Python 3.5** | 只 launch 作业、poll 状态 JSON、apply 结果回 Mimics；**不做重计算** | `runtime_py35/fewshot_mimics.py`、`runtime_common.py` |
| **Tier 2** | 现代 Python 3.10+（`nninteractive_env`） | 编排、GPU 锁、数据转换、外部 UI | `mimics_bridge.py`、`tools/fewshot_pipeline.py`、`tools/fewshot_status_viewer.py` |
| **Tier 3** | 现代 Python（PyTorch） | 模型推理、算法 | `external/dinov3-medical-seg/src/` |

**必须遵守的模式**：
- Tier 1 新代码保持 **Py3.5-clean**（`.format()`、无 f-string / 类型注解），仿 `_launch_inference_job`（`fewshot_mimics.py:1789`）的 launch→写状态 JSON→`_start_monitor`→apply 生命周期。
- Tier 2 子命令模式：`fewshot_pipeline.py` 用 argparse subparsers（`train`/`infer`/`discover`/...，line 2584+），GPU 锁 `acquire_gpu_lock_for_job`（line 796），原子状态写 `write_json_atomic` / `update_status`。
- 外部 UI = PySide6 主、Tkinter 回退，独立进程 `Popen`，通过 JSON 状态文件与 Mimics 通信（`fewshot_status_viewer.py`、`fewshot_training_setup_ui.py`）。配置开关 `*_ui_mode="external"` + `*_fallback_*`。
- 测试：`sys.modules["mimics"]` 注入假模块（`tools/test_all.py:52-93`）；全 in-memory 假 Mimics（`tools/fake_mimics_flow_test.py:382`）；Tier 3 用 pytest + tmp_path（`test_research_pipeline.py`）。
- 命名避坑：现有 `tools/fewshot_strategies.py` 是"**训练预设**"语义，本模块用 `hardcase_*` 前缀，避免与 MONAI "Strategy" 语义混淆（内部类可叫 `Strategy` 但文件名区隔）。

---

## 3. 设计总览

### 3.1 核心映射：MONAI 双抽象 → 三层架构

```
┌─────────────────────────── MONAI Label 概念 ────────────────────────────┐
│  ScoringMethod ──update_image_info──▶ Datastore ◀──get_image_info── Strategy │
└──────────────────────────────────────────────────────────────────────────┘
              │                            │                        │
              ▼ (Tier 3, torch)            ▼ (Tier 2, 文件)          ▼ (Tier 2, 纯 py)
   src/active_learning/scoring.py   workspace/scores/<organ>/   tools/hardcase_ranking.py
   (Epistemic / TTA / Heuristic)    <case>.json (info-tag)      (读分→归一化→RRF→排序队列)
              │                                                        │
              │ 由 Tier2 cmd_score 逐 case 调度 (GPU锁+状态JSON)         │
              ▼                                                        ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Tier 1 (Py3.5) fewshot_mimics.py: "Rank Hard Cases" 按钮               │
   │   → Popen `pipeline score-batch` → poll 状态 → 启动复查面板             │
   │ tools/hardcase_review_panel.py (PySide6): 排序队列表格 + 双击打开 case  │
   └──────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策（含性价比裁决）

| 决策 | 选择 | 理由 |
|---|---|---|
| 架构 | **A：Scoring/Strategy 双抽象跨三层** | 最忠实 MONAI；重算(Tier3)/可测排序(Tier2)/轮询 UI 干净解耦；可重排序而不重打分 |
| 与 MONAI 一致的层次 | **Vendor MONAI 的 AL ABC 文件（①b），不依赖 `monailabel` 整包** | 详见 §3.3 的数据支撑取舍。逐字拷入 `scoring.py`/`strategy.py`（Apache-2.0，保留 license 头），Datastore 只留需要的 3 方法 → 安装零负担 + 字面同一份类定义（忠实度最高） |
| 打分算法库 | **引入 `monai` core**（transforms/sliding_window_inference，如需要） | 相对独立的算法依赖，只进 Tier 2/3 的 py3.10 打分 env，不进 Mimics Py3.5 |
| 主打分后端（2026-08-01 修正） | **nnU-Net 单模型（已有）+ DINOv3（已有），统一 TTA-VVC + 几何** | 训练成本受限（`fold=0` 单折）→ 不训练集成、不引入需训练的 MONAI bundle；两个已有后端走同一可插拔接口，`supports_dropout` 均设 False |
| Epistemic/MC-Dropout | **默认关闭；仅当后端显式带 dropout 才可用** | 现有后端（nnU-Net 推理模型、DINOv3）无推理期 dropout 层；不为打分改模型结构。MONAI 契约保留（`supports_dropout` 标志），将来有带 dropout 自训网络时自动启用 |
| DINOv3 地位 | **TTA-only 后端（与 nnU-Net 平级）** | 其解码器无 dropout（已验证）；TTA-VVC 可复用 `_predict_with_tta` 增强栈——同时作为**统一增强组件**（nnU-Net 后端复用同一增强栈，不开其镜像 TTA） |
| 多器官组织（2026-08-01 新增） | **按器官算分 → 病例级聚合 + "最差器官"标签** | ModelMap 每器官独立模型；标注者打开一个病例通常标多个器官 → 队列按病例排；体积先验按器官查表 |
| 融合 | **RRF rank fusion 默认** | 免调、抗重尾；攒够 30-50 标签后可切 logistic 加权 |
| 几何启发式 | **纳入为默认信号之一** | 零概率成本，抓"自信但全错"的粗失败（概率分数盲区）；无模型也能排序 |
| 冷启动多样性（2026-08-01 新增） | **仅完全无模型的器官启用：DINOv3 特征 + 覆盖最大选第一批** | 有弱模型（含粗分割）的器官直接用 TTA+几何；MedCAL-Bench 证明 DINOv2 家族分割 AL 特征最强 |
| 交互修正环节 | **nnInteractive（已有），见 §1.5** | 结构错误点两下重做；不集成 MedSAM2/DeepEdit（3D CT/MR 无增量、依赖更重） |

### 3.3 "是否引入 monailabel 整包"的数据支撑裁决

用户明确要求权衡**整包安装难度**与**不引整包的开发难度**两端。以下基于对 `Project-MONAI/MONAILabel@main` 真实源码的核实（非推测）。

**先纠正一个此前的错误论断**：之前称"import monailabel 的接口会连带拖入 fastapi"——**这是错的，已核实并收回**。真实 import 链是干净的：`monailabel/__init__.py` 只 import `os,sys,._version`；`interfaces/__init__.py` 与 `interfaces/tasks/__init__.py` 是**空文件**（仅 license）；`scoring.py`/`strategy.py` 只 `from abc import ...` + `from monailabel.interfaces.datastore import Datastore`；`datastore.py` 只依赖 `abc,enum,typing`（纯标准库）。运行时导入这几个 ABC **不会**引入 server 世界。

真正的成本在两处，各有硬数据：

**① 安装难度**（`monailabel/requirements.txt` 实测）：`pip install monailabel` 会拉入 `fastapi==0.110.2`、`uvicorn`、`pydantic`、`pynetdicom==2.0.2`、`dicomweb-client[gcp]`、`highdicom`、`girder-client`、`google-auth`、`passlib/pyjwt/bcrypt`，以及两个离线杀手：`SAM-2 @ git+https://github.com/facebookresearch/sam2.git@...`（**直接从 GitHub 装**，离线不可得）和 `numpymaxflow==0.0.7`/`ninja==1.11.1.1`（**需现场 C++ 编译**）。对本项目的离线 win/py3.10 可移植部署（`setup_offline.bat`+`wheels/`）是真实重障碍。

**② 开发难度**（真实 ABC 体量）：

| ABC | 抽象方法数 | 继承成本 |
|---|---|---|
| `ScoringMethod` | 1（`__call__`）+ `__init__`/`info` | 极低（~20 行） |
| `Strategy` | 1（`__call__`）+ `__init__`/`info` | 极低（~20 行） |
| **`Datastore`** | **13 个**（2026-08-01 勘误：经源码复核，`datastore.py` 为 144 行、13 个 `@abstractmethod`、共 25 个方法定义——旧记 20 个不准确） | **很高**——继承即须实现全部 13 个（或遍地 `NotImplementedError`），而本模块只需 `get_unlabeled_images`/`get_image_info`/`update_image_info` 三个 |

反直觉结论：**继承 monailabel 的 `Datastore` ABC 反而比自写 3 方法的文件版 store 更费事更丑**；整包能省的只是 `ScoringMethod`/`Strategy` 那 ~40 行 trivial 样板。且**三个 scoring 实现（Epistemic/TTA/Heuristic）无论哪条路都得自己写**——monailabel 的 `EpistemicScoring` 深度绑其 `InferTask`/`network`/`model_ts`，无法整段复用，"引整包省开发量"的收益极小。

**三方对比**：

| 路径 | 安装难度 | 开发难度 | MONAI 忠实度 |
|---|---|---|---|
| ① 自定义同签 ABC | ✅ 零新增（只需 `monai` core） | ✅ 低：ABC 样板 ~40 行 | 语义/签名/info-tag 一致 |
| **①b Vendor 那几个文件（选定）** | ✅ 零新增 | ✅ 最低：类定义逐字相同，datastore 只留 3 方法 | **最高**（字面同一份类定义） |
| ② 依赖 monailabel 整包 | ❌ 高：SAM-2 走 git、numpymaxflow/ninja 需编译 | ◐ 省 40 行样板，却被迫实现 20 方法 Datastore ABC + 绑其版本 | 最高但代价失衡 |

**裁决：①b**。在 `external/dinov3-medical-seg/src/active_learning/_monai_contract.py`（或分 `scoring.py`/`strategy.py`）中**逐字 vendor** MONAI 的 `ScoringMethod`/`Strategy` 类定义（Apache-2.0，保留原始 license 头 + 来源 URL 注释）；`FileScoreStore` 实现精简版 `Datastore` 契约的 3 个方法。安装侧只依赖 `monai` core（网络本就需要），离线打包零新增负担；忠实度上就是 MONAI 的同一份类定义。逃生舱：类同名同签，将来若要迁到整包，实现类几乎可直接搬。

---

## 4. 组件详细设计

### 4.1 Tier 3：可插拔打分后端接口

**新文件**：`external/dinov3-medical-seg/src/active_learning/`（新 package）——注意：这个 package **不依赖 DINOv3 特定架构**，只依赖"后端接口"。

#### 4.1.1 `backend.py` — 可插拔后端协议

```python
# 后端接口（Protocol），任何模型只要实现它就能被打分
class ScoringBackend(Protocol):
    supports_dropout: bool     # True → Epistemic MC-Dropout 可用
    supports_tta: bool         # True → TTA-VVC 可用
    num_classes: int

    def predict_probability(self, image_path: str) -> np.ndarray:
        """单次前向，返回前景概率体 P∈[0,1]^{D,H,W}（多类则 [C,D,H,W]）。"""

    def predict_probability_samples(self, image_path: str, n: int, *,
                                     mode: str) -> Iterator[np.ndarray]:
        """产出 n 个概率体样本。
        mode='dropout' → MC-Dropout（需 supports_dropout）
        mode='tta'     → 复用可逆增强（需 supports_tta）
        """
```

**两个具体后端**：
- `monai_bundle_backend.py`：加载 MONAI Model Zoo bundle（或自训 checkpoint），`supports_dropout=True`（DynUNet/UNETR 带 dropout），`supports_tta=True`。MC-Dropout 用 `enable_dropout()`（只把 `Dropout*` 层置 `.train()`，BN/LN 保持 eval），复用 MONAI `sliding_window_inference`。
- `dinov3_backend.py`：包装现有 `src/inference.py`，`supports_dropout=False`，`supports_tta=True`（复用 `_predict_with_tta`，改为保留整叠）。

后端选择由打分请求的 `backend` 字段决定，工厂在 `backend.py::create_backend(spec)`。

#### 4.1.2 `_monai_contract.py` — Vendor 的 MONAI ABC（逐字拷贝）

**逐字 vendor** MONAI 的两个基类（Apache-2.0，保留原 license 头 + 来源 URL 注释）——签名与 MONAI **完全一致**（`__call__(self, request, datastore)`）：

```python
# Vendored verbatim from Project-MONAI/MONAILabel@main (Apache-2.0):
#   monailabel/interfaces/tasks/scoring.py, .../strategy.py
class ScoringMethod(metaclass=ABCMeta):
    def __init__(self, description): self.description = description
    def info(self): return {"description": self.description}
    @abstractmethod
    def __call__(self, request, datastore):  # datastore = FileScoreStore
        pass

class Strategy(metaclass=ABCMeta):
    def __init__(self, description): self.description = description
    def info(self): return {"description": self.description}
    @abstractmethod
    def __call__(self, request, datastore):
        pass
```

**后端如何进入 scoring**（保持签名忠实的同时接入可插拔后端）：`ScoringBackend` 通过 `request["backend"]`（工厂在 scoring 内部 `create_backend(request["backend"])`）传入，**不改 `__call__` 签名**。这与 MONAI 一致——MONAI 的 scoring 也是从 `request` 取 `model`/`network` 名再自行加载。

三个具体 scoring（继承上面 vendored `ScoringMethod`，info-tag 键与 MONAI 一致）：
- `EpistemicScoring` → `{"epistemic_entropy": float}`。N 次 dropout 前向，逐体素 `H[\bar p]`，`nanmean`。仅当后端 `supports_dropout`；否则返回 `{"epistemic_entropy": None, "epistemic_status": "unavailable_no_dropout"}`。
- `TTAScoring` → `{"tta_vvc": float}`。复用 TTA 整叠，Welford 在线累加算 $\sigma/\mu$（VVC 标量）+ 前景/边界 `H(\bar p)`（`tta_boundary_entropy`）。
- `HeuristicScoring` → `{"n_components": int, "frag_ratio": float, "border_frac": float, "vol_zscore": float, "empty_flag": bool}`。纯 numpy/scipy，从二值掩码算，**无需概率、无需 torch**（可无 GPU 单测）。

**数学（全部可单测，输入合成概率体有已知答案）**：
| 信号 | 公式 |
|---|---|
| TTA-VVC | $v_m=|\{p^{(m)}\ge\tau\}|$；$\text{VVC}=\sigma_{\{v_m\}}/\mu_{\{v_m\}}$ |
| 边界熵 | $\tfrac1{|S|}\sum_{v\in S}H(\bar p_v)$，$S$=表面体素 |
| 前景熵 | $\tfrac1{|F|}\sum_{v\in F}H(\bar p_v)$ |
| 碎片率 | $r_{\text{frag}}=(|F|-|\text{largest}|)/|F|$；$K$=`scipy.ndimage.label` 连通域数 |
| 边界接触 | $\text{border\_frac}$=触及 FOV 面的前景体素占比 |
| 体积 z-score | $z=(V-\mu_{\text{organ}})/\sigma_{\text{organ}}$，先验来自配置或训练集预测体积 |

#### 4.1.3 `scripts/score_case.py` — Tier 3 CLI 入口

仿 `scripts/infer.py`（line 34-66）。参数 `--backend-spec`（JSON：类型+checkpoint+config）`--input`（case 图像）`--methods`（epistemic,tta,heuristic）`--output`（分数 JSON）`--tta-samples`/`--dropout-samples`。stdout 打印分数 JSON。**不改 `infer.py` 主流程**，避免影响现有推理。

### 4.2 Tier 2：编排 + Datastore + Strategy

#### 4.2.1 `tools/fewshot_pipeline.py` 新增 `cmd_score_batch`

新 subparser `score-batch`（仿 `cmd_infer`，line 2279）：
- 输入：`--ts-root` `--organs`（列表，替代单 `--organ`）`--case-ids`（或全 discover）`--backend-spec` `--methods` `--job-id`。
- 流程：`discover` 出 case 列表 → **逐 case × 逐器官**（每个器官用其对应模型）：`acquire_gpu_lock_for_job` → `Popen scripts/score_case.py` → 器官级分数写 `workspace/scores/<organ>/<case>.json`（**这就是 MONAI 的 `update_image_info`**）→ `update_status` 更新批处理进度（`scored/total`）。
- 全部算完后调用 §4.2.3 的 ranking（**器官级融合 → 病例级聚合**），产出排序队列 JSON。
- 状态 JSON schema：`mimics_hardcase_job.v1`（新），字段含 `kind:"score_batch"`、`total`、`scored`、`failed`、`queue_path`。

#### 4.2.2 Datastore 抽象（轻量文件版）

**新文件** `tools/hardcase_datastore.py`：实现 §4.1.2 vendored `Datastore` 契约中本模块**实际用到的 3 个方法**（其余 17 个抽象方法不继承 vendored `Datastore` ABC，而是自定义一个精简协议 `ScoreStore`，避免被迫实现 add_image/save_label/json 等无关方法——这正是 §3.3 裁决"不继承 20 方法 ABC"的落地）：
```python
class FileScoreStore:  # 实现精简 ScoreStore 协议（get_unlabeled_images/get_image_info/update_image_info）
    def get_unlabeled_images(self) -> list[str]      # discover 出、无 final 掩码的 case
    def get_image_info(self, case_id) -> dict         # 读 scores/<organ>/<case>.json
    def update_image_info(self, case_id, info: dict)  # 原子合并写回（write_json_atomic）
```
方法名与 MONAI `Datastore` **逐字一致**（`get_unlabeled_images`/`get_image_info`/`update_image_info`），语义一致。标注状态：case 目录下是否存在人工核准掩码（`final` tag 语义）→ 决定是否在未标注池。**复用** `runtime_common.write_json_atomic` 语义（Tier2 版）。

#### 4.2.3 `tools/hardcase_ranking.py` — 具体 Strategy（纯 Python，核心可测层）

继承 §4.1.2 vendored `Strategy`（**不重新定义 ABC**），签名与 MONAI 一致 `__call__(self, request, datastore)`（`datastore` 传 `FileScoreStore`）：

```python
from active_learning._monai_contract import Strategy  # vendored MONAI ABC

class RankFusionStrategy(Strategy):
    """读所有 case 的 info-tag 分数 → robust z-score/RRF 融合 → 降序。
    返回 {"id": case_id, ...}，无候选返回 None（MONAI 契约）。
    去优先化近期已 serve（serve 时间戳，仿 MONAI）。"""

class SingleSignalStrategy(Strategy):
    """key='epistemic_entropy' 或 'tta_vvc'，降序取 Top-N。等价 MONAI Epistemic/TTA 策略。"""
```
（注：批量分诊场景下，队列 = 对候选池按分数排序的**完整有序列表**；MONAI 的 `__call__` 返回单个 `{"id":...}` 是"取下一例"语义，本模块额外提供 `rank_all(request, store) -> list` 产出整条队列供面板展示，二者共用同一融合逻辑。）

**融合算法**（可脱离 torch/mimics 单测）：
- **RRF**（默认）：$\text{RRF}(c)=\sum_j w_j/(k+r_j(c))$，$k\approx60$，$r_j$=信号 $j$ 上的排名。
- **robust z + logistic**（进阶）：$z_j=(x_j-\text{median}_j)/(1.4826\,\text{MAD}_j)$，clip 到 ±3，加权和；权重可由 ≥30-50 个人工好/坏标签 logistic 拟合。
- 分档：分数分位数 → `HIGH/MED/LOW` 三档 + 每 case 的"原因"字符串（哪些信号触发，如"碎片+边界糊"/"触及FOV边界"）。
- **病例级聚合**（多器官，2026-08-01 新增）：器官级 RRF 融合 → 病例总分 = 各器官分数融合（默认等权，可切"最差器官优先"）→ 队列每个病例一行，附 `worst_organ`（扣分最重器官）标签供标注者先修。
- 产出 `workspace/queues/<job_id>.json`（schema `mimics_hardcase_queue.v1`，跨器官）：排序 case 列表 + 各器官各信号明细 + 病例总分 + `worst_organ` + tier + reason。

默认信号集：`{ tta_vvc, tta_boundary_entropy, vol_zscore, frag_ratio, border_frac }`（相关信号只留一个代表，避免冗余加权；按器官计算，体积 z 分数查器官先验表）。

### 4.3 Tier 2 UI：`tools/hardcase_review_panel.py`

PySide6 外部进程（Tkinter 回退），仿 `fewshot_status_viewer.py`：
- 读 §4.2.1 的批处理状态 JSON（进度条 `scored/total`）+ §4.2.3 的排序队列 JSON。
- 表格：`# | case | tier | score | 最差器官 | vvc | Usurf | comp | border | reason`，按 score 降序，tier 用颜色（● HIGH 红 / ◐ MED 黄 / ○ LOW 灰）——**最差器官列**告诉标注者重点修哪个器官（多器官病例先修最差的、其余快速确认）。
- **抗锚定交互（2026-08-01 新增）**：默认先显示模型结果，但提供明显的"拒绝/重做"入口；低置信区域周期性强制复查（automation bias 缓解；依据：病理实验 AI 建议权重 0.44、CHI 2021 先独立判断再出示 AI 的 RCT）。
- 交互：`[Refresh]` `[Export CSV]` `[Re-rank ▾]`（切 RRF/单信号/logistic，**只重排不重算**）；**双击行**写 `selection.json`（`{"action":"open_case","case_id":...}`）供 Tier 1 poll 后在 Mimics 打开该 case。
- QSS 主题沿用 `"Segoe UI","Microsoft YaHei"`，`CloseAwareMainWindow` 模式。

**布局示意**：
```
┌─ Hard Case Review — liver (23 cases) ───────────────────────────┐
│ [██████████░░] scoring 18/23   [Refresh] [Export CSV] [Re-rank▾]│
├──┬───────────┬──────┬──────┬──────┬─────┬──────┬────────────────┤
│# │ case      │ tier │score │ VVC  │Usurf│ comp │ reason         │
├──┼───────────┼──────┼──────┼──────┼─────┼──────┼────────────────┤
│1 │ case_0421 │●HIGH │ 0.94 │ 0.71 │0.55 │  4   │碎片+边界糊     │
│2 │ case_0088 │●HIGH │ 0.88 │ 0.12 │0.20 │  1   │触及FOV边界     │
│3 │ case_0193 │◐MED  │ 0.61 │ 0.33 │0.41 │  1   │边界不确定      │
│…                                                                │
└─────────────────────────────────────────────────────────────────┘
      双击行 → 在 Mimics 打开该 case
```

### 4.4 Tier 1：`runtime_py35/fewshot_mimics.py` 新增入口（Py3.5-clean）

- 新按钮常量 `BUTTON_RANK_HARDCASES = "Rank Hard Cases..."`，加入 `main()` 的按钮列表（line 3045）与 dispatch（line 3049+）。
- 新函数 `_launch_score_batch_job(config, ts_root, organ, backend_spec, case_ids=None)`：仿 `_launch_inference_job`（line 1789），构造 `pipeline score-batch` 命令 → `_launch_process` → 写 `mimics_hardcase_job.v1` 状态 → `_start_monitor`。
- 新 `_launch_hardcase_review_panel(config, ts_root, organ, job_id)`：`Popen` §4.3 面板（仿 `_launch_external_status_viewer`，line 2825）。
- 新 monitor tick 分支：poll 面板写的 `selection.json`，若 `action=="open_case"` 则在 Mimics 里加载该 case（复用现有 case 打开逻辑 / `mimics_import`）。
- **不做任何重计算**，全部 launch+poll+apply。

### 4.5 配置

新文件 `hardcase_config.json`（仿 `fewshot_config.json`）：
```json
{
  "backends": {
    "monai_bundle": {"type": "monai_bundle", "bundle_path": "", "supports_dropout": true},
    "dinov3": {"type": "dinov3", "supports_dropout": false}
  },
  "default_backend": "monai_bundle",
  "default_methods": ["tta", "heuristic"],
  "epistemic_samples": 5,
  "tta_samples": 8,
  "fusion": "rrf",
  "organ_volume_priors": {},
  "review_ui_mode": "external",
  "review_ui_fallback_to_text": true,
  "gpu_lock_timeout_seconds": 3600
}
```

---

## 5. 端到端数据流

```
1. [Tier1] 用户点 "Rank Hard Cases..."，选 organ + backend
2. [Tier1] _launch_score_batch_job → Popen `pipeline score-batch --organ liver --backend monai_bundle`
3. [Tier2] cmd_score_batch: discover cases → 逐 case:
       acquire_gpu_lock → Popen scripts/score_case.py → 写 scores/liver/<case>.json  (= update_image_info)
       update_status(scored/total)
4. [Tier2] 全算完 → RankFusionStrategy 读所有分数 → RRF 融合 → 写 queues/liver/<job>.json
5. [Tier1] monitor 检测到 job done → _launch_hardcase_review_panel
6. [UI]   读队列 JSON → 排序表格；用户双击 case_0421 → 写 selection.json
7. [Tier1] monitor 检测到 selection → 在 Mimics 打开 case_0421 供人工标注/修正
8. 人标完 → case 进 final 池 → 下次 score-batch 自动从未标注池排除（主动学习闭环）
```

---

## 6. 测试策略

| 层 | 测试 | 位置 |
|---|---|---|
| Tier 3 算法 | pytest：喂合成概率体，验证 VVC/熵/碎片/border/z-score 有已知答案；MC-Dropout 用 mock 后端（固定 N 个概率体）验证熵公式 | `external/dinov3-medical-seg/tests/test_active_learning.py`（新），挂在 `test_postprocess_foreground` 旁 |
| Tier 3 后端 | mock backend（合成概率体）验证 `supports_dropout=False` 时 Epistemic 返回 `unavailable` | 同上 |
| Tier 2 排序 | 纯 Python 单测：喂固定分数 JSON，验证 RRF/robust-z/分档/reason 输出稳定；无 torch/mimics | 加入 `tools/test_all.py` |
| Tier 2 Datastore | 验证 `get_unlabeled_images`/`update_image_info` 原子合并 | `tools/test_all.py` |
| Tier 1 流程 | 假 Mimics（`fake_mimics_flow_test.py`）走通 launch→poll→open_case，mock pipeline 子进程 | `tools/fake_mimics_flow_test.py` |
| 语法 | 全 `.py` L1 compile（Tier1 必须 Py3.5 语法通过） | `tools/test_all.py` L1 |

**验证 rank 有效性**（可选，后续）：小验证集上算 failure-detection AUROC / Spearman ρ vs 真 Dice。

---

## 7. 分阶段实施

- **Phase 0（验证实验，硬门控，2–3 天）**：20–50 例有真值病例，**逐器官**测 Spearman ρ / failure-detection AUROC + 人时实测（队列顺序 vs 随机顺序各 10 例）。决策树：ρ>0.40 且 AUROC>0.70 → 全量实施；ρ 0.20–0.40 → 只用 TTA-VVC+几何；ρ<0.20 → 只做几何启发式；模型 Dice<0.3 → 放弃不确定性路径。数据不足的器官只启用几何。**与三点主线中的线 A/B 无依赖，可并行推进。**
- **Phase 1（骨架 + 免费信号）**：`ScoringBackend` 接口 + `nnunet_backend` + `dinov3_backend`（均 TTA-only，`supports_dropout=False`）+ **统一增强组件**（复用 `_predict_with_tta` 轻量增强栈）+ `TTAScoring` + `HeuristicScoring` + 算法单测。产出：能对现有模型算 VVC+几何分数。
- **Phase 2（Strategy + 队列 + 编排）**：`FileScoreStore` + `hardcase_ranking`（RRF + 病例级聚合 + 最差器官）+ `cmd_score_batch`（多器官）+ 状态 JSON。产出：命令行能出排序队列 JSON。
- **Phase 3（可选：带 dropout 后端）**：引入 `monai` core + `monai_bundle_backend` + `EpistemicScoring`（MC-Dropout）+ Model Zoo bundle 下载/离线打包——**仅在用户提供带 dropout 的自训/预训练网络时启用**；默认跳过（训练成本约束下无此资产）。
- **Phase 4（UI + Tier1）**：`hardcase_review_panel`（含最差器官列 + 抗锚定交互）+ Tier1 按钮/monitor/open_case + 假 Mimics 测试。产出：Mimics 内完整可用。
- **Phase 5（进阶，可选）**：logistic 加权融合、半监督扩量（选择性伪标签 + 经验回放）、标签质量检测接入（复用本模块打分设施：已有标注的可疑度 = 重标优先级）、RCA/AUROC 验证。

---

## 8. 风险与权衡

| 风险 | 缓解 |
|---|---|
| **模型太弱 → 不确定性 = 噪声**（AL 文献一致发现：Dice < ~0.4-0.5 时不确定性无信息量 [Ma 2024; COLosSAL 2023]） | **硬门控**：若模型 Dice < 0.4 或 Spearman ρ < 0.2 → `ScoringMethod` 返回 `"unavailable_model_weak"`，系统自动退化到纯几何启发式。门控必须在 Phase 1 实验中校验 |
| **随机选样是极难击败的基线**（5+ 独立论文同意：大多数 AL 方法未一致优于 random） | 本模块是 **Failure Detection**（排名）而非经典 AL loop（选样→重训），对"比 random 好"的要求相对宽松；但 Phase 1 仍必须有 ρ > 0.2 |
| **纯不确定性导致冗余选择**（连续切片内不确定性高度相关） | 批量分诊场景：整池排序，不去重——单 case 复查队列天然化解切片冗余（每个 case 一条） |
| `monai` core 离线打包体积大（拖 torch 生态） | 只进 Tier2/3 的 py3.10 env，不进 Mimics Py3.5；Phase 1-3 不依赖 monai，nnU-Net 后端用已有环境 |
| MC-Dropout 在无 dropout 训练的网上失校准 | 硬门控 `supports_dropout`；DINOv3 后端禁用 Epistemic，返回 `"unavailable_no_dropout"` |
| OOD/不确定性分数与真实退化不相关（Vasiliuk 2023） | Phase 1 强制验证 Spearman ρ / AUROC；ρ < 0.2 → 弃用该信号 |
| 单模型不确定性不如集成强壮（文献一致结论） | TTA-VVC 是单模型下最强信号；**fold=0 单折（成本约束）→ 不做集成**；如有历史遗留多模型权重可零成本启用分歧信号 |
| GPU 批量打分耗时长 | 复用 `acquire_gpu_lock_for_job` 串行 + 进度条；`tta_samples` 可配置；多器官 = 病例数 × 器官数 次推理（粗分割模型可先筛掉无目标器官病例，可选优化） |
| 与现有 `fewshot_strategies.py`（训练预设）命名混淆 | 新模块统一 `hardcase_*` 前缀 |
| 模型质量一般（Dice 0.3–0.5，2026-08-01 补充） | 不确定性不可靠时主动学习路径自动降级（硬门槛），但**预标注+确认模式与几何排序仍然有效**——"改"而非"画"的收益不依赖主动学习 |

### 8.1 最小质量门槛与退化路径

系统强制实现三级退化路径（`ScoringMethod.__call__` 返回值中的 `status` 字段驱动）：

```
Level 0（✅ 全信号可用）：模型 Dice > 0.5 或 Spearman ρ > 0.3
  → TTA-VVC + 边界熵 + 几何启发式 → RRF 融合

Level 1（◐ 退化 — TTA 关或不确定无信息量）：ρ < 0.2 或 TTA 不可用
  → 仅单次前向熵（如果有） + 几何启发式 → RRF 融合
  → status: "degraded_uncertainty_noisy"

Level 2（❌ 不可用 — 模型太弱）：Dice < 0.4 且无改善前景
  → 仅几何启发式
  → status: "unavailable_model_weak"

Level 3（❌❌ AL 完全不可用）：候选池 < 20 例或无模型
  → 不启动模块。返回明确错误信息。
  → status: "unavailable_insufficient_data"
```

**关于"AL 不可用时应考虑的其他方法"**：当不确定性信号无信息量时，以下替代方案可作为备选：
- 纯几何启发式（已在所有 Level 默认运行）
- RCA（Reverse Classification Accuracy）预测 per-case Dice——需要小 GT 参考库，但更可靠
- 学习式 QA 网预测分割质量——需要一次性训练，但对新 case 推理极快
- 放弃自动化 → 人力全 review——在数据/模型条件不足时，这是负责任的工程判断

---

## 9. 交付物清单

**新增文件**：
- `external/dinov3-medical-seg/src/active_learning/_monai_contract.py`（**vendored** MONAI `ScoringMethod`/`Strategy` ABC，Apache-2.0 license 头 + 来源 URL）
- `external/dinov3-medical-seg/src/active_learning/{__init__,backend,scoring}.py`
- `external/dinov3-medical-seg/src/active_learning/backends/{monai_bundle_backend,dinov3_backend}.py`
- `external/dinov3-medical-seg/scripts/score_case.py`
- `external/dinov3-medical-seg/tests/test_active_learning.py`
- `tools/hardcase_datastore.py`（`FileScoreStore` + 精简 `ScoreStore` 协议）、`tools/hardcase_ranking.py`、`tools/hardcase_review_panel.py`
- `hardcase_config.json`
- `docs/hardcase_active_learning.md`（用户文档）
- `NOTICE` / 第三方许可声明更新（记录 vendored MONAI 文件的 Apache-2.0 归属）

**修改文件**：
- `tools/fewshot_pipeline.py`（+`cmd_score_batch` + subparser）
- `runtime_py35/fewshot_mimics.py`（+按钮/launch/monitor/open_case，Py3.5-clean）
- `tools/test_all.py`、`tools/fake_mimics_flow_test.py`（+测试）

**明确不改**：`src/inference.py` 主流程、`src/models/`（除非 Phase 5 自训网络）、现有 few-shot/nnInteractive 流程。
