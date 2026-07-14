# Mimics 难例分诊模块设计（MONAI-Mirrored Active Learning）

- **日期**: 2026-07-14
- **状态**: 设计待评审
- **作者**: Claude (与 ShijianRuan 协作 brainstorming)
- **目标**: 在 Mimics 脚本库中集成一个"识别疑难/苦难/不确定病例"的主动学习模块，按难度对一批 case 排序，让人优先复查/标注最难的例，从而减少标注工作量。

---

## 1. 背景与研究基础

### 1.1 需求

用户在 Mimics（Materialise 医学影像分割工具）里已经有一套 DINOv3 few-shot 分割与 nnInteractive 交互分割工作流。现在希望增加"主动学习/难例识别"能力：对一批未标注或待复查的 case，用模型跑推理并算**不确定性/难度分数**，产出一个**按难度排序的复查队列**，人优先处理最难/最可能出错的例。

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
4. `src/inference.py` 的 `_predict_with_tta`（line 69-80）**已经在跑 TTA**，但 `torch.stack(...).mean(dim=0)` 后把方差丢弃了 → TTA-VVC 几乎零成本可得。
5. 现有三层架构 + 文件通信 + 外部 UI + JSON 轮询范式（见 §2）。

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
| 打分算法库 | **引入 `monai` core**（DynUNet/UNETR/sliding_window_inference/transforms） | 相对独立的算法依赖，只进 Tier 2/3 的 py3.10 打分 env，不进 Mimics Py3.5 |
| 主打分后端 | **带 dropout 的 MONAI 网络** | 让 MONAI 旗舰 Epistemic MC-Dropout 开箱即用；不绑 DINOv3 |
| DINOv3 地位 | **降为可选的 TTA-only 后端** | 其解码器无 dropout（已验证），MC-Dropout 不可用；但 TTA-VVC 可复用 `_predict_with_tta` |
| 网络来源 | **MONAI Model Zoo 预训练 bundle 起步 + 预留自训接口** | 最快跑通全链路，无需自己训练；两条路走同一可插拔后端接口 |
| 融合 | **RRF rank fusion 默认** | 免调、抗重尾；攒够 30-50 标签后可切 logistic 加权 |
| 几何启发式 | **纳入为默认信号之一** | 零概率成本，抓"自信但全错"的粗失败（概率分数盲区） |

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
| **`Datastore`** | **20 个**（name/datalist/get_image/add_image/save_label/remove_label/status/json/refresh/get_dataset_archive…） | **很高**——继承即须实现全部 20 个（或遍地 `NotImplementedError`），而本模块只需 `get_unlabeled_images`/`get_image_info`/`update_image_info` 三个 |

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
- 输入：`--ts-root` `--organ` `--case-ids`（或全 discover）`--backend-spec` `--methods` `--job-id`。
- 流程：`discover` 出 case 列表 → **逐 case**：`acquire_gpu_lock_for_job` → `Popen scripts/score_case.py` → 分数写 `workspace/scores/<organ>/<case>.json`（**这就是 MONAI 的 `update_image_info`**）→ `update_status` 更新批处理进度（`scored/total`）。
- 全部算完后调用 §4.2.3 的 ranking，产出排序队列 JSON。
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
- 产出 `workspace/queues/<organ>/<job_id>.json`（schema `mimics_hardcase_queue.v1`）：排序 case 列表 + 各信号明细 + tier + reason。

默认信号集：`{ tta_vvc, tta_boundary_entropy, vol_zscore, frag_ratio, border_frac }`（相关信号只留一个代表，避免冗余加权）。

### 4.3 Tier 2 UI：`tools/hardcase_review_panel.py`

PySide6 外部进程（Tkinter 回退），仿 `fewshot_status_viewer.py`：
- 读 §4.2.1 的批处理状态 JSON（进度条 `scored/total`）+ §4.2.3 的排序队列 JSON。
- 表格：`# | case | tier | score | vvc | Usurf | comp | border | reason`，按 score 降序，tier 用颜色（● HIGH 红 / ◐ MED 黄 / ○ LOW 灰）。
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

- **Phase 1（骨架 + 免费信号）**：`ScoringBackend` 接口 + `dinov3_backend`（TTA-only）+ `TTAScoring` + `HeuristicScoring` + 算法单测。复用现有 DINOv3，先不引 monai。产出：能对 DINOv3 已训模型算 VVC+几何分数。
- **Phase 2（Strategy + 队列 + 编排）**：`FileScoreStore` + `hardcase_ranking`（RRF）+ `cmd_score_batch` + 状态 JSON。产出：命令行能出排序队列 JSON。
- **Phase 3（MONAI 后端）**：引入 `monai` core + `monai_bundle_backend` + `EpistemicScoring`（MC-Dropout）+ Model Zoo bundle 下载/离线打包。产出：MONAI 旗舰方法开箱即用。
- **Phase 4（UI + Tier1）**：`hardcase_review_panel` + Tier1 按钮/monitor/open_case + 假 Mimics 测试。产出：Mimics 内完整可用。
- **Phase 5（进阶，可选）**：logistic 加权融合、自训带 dropout 网络接口、RCA/AUROC 验证、成批 core-set 去冗余（Suggestive Annotation）。

---

## 8. 风险与权衡

| 风险 | 缓解 |
|---|---|
| `monai` core 离线打包体积大（拖 torch 生态） | 只进 Tier2/3 的 py3.10 env，不进 Mimics Py3.5；Phase 1-2 不依赖 monai，可先交付 DINOv3 TTA 路径 |
| Model Zoo bundle 的类别/解剖与用户数据不匹配 | bundle 起步仅用于跑通链路 + 演示；Phase 5 自训接口对齐用户器官 |
| MC-Dropout 在无 dropout 训练的网上失校准 | 硬门控 `supports_dropout`；DINOv3 后端直接禁用 Epistemic，返回 `unavailable` 而非假分数 |
| OOD/不确定性分数与真实退化不相关（Vasiliuk 2023） | 多信号 RRF 融合而非单信号；Phase 5 用 AUROC/Spearman 在验证集上校验 |
| GPU 批量打分耗时长 | 复用 `acquire_gpu_lock_for_job` 串行 + 进度条；单 case 打分 = N passes，可配置 `tta_samples` |
| 与现有 `fewshot_strategies.py` 命名混淆 | 新模块统一 `hardcase_*` 前缀 |

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
