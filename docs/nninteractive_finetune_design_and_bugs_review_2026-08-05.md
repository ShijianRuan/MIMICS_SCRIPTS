# nnInteractive 集成设计调研与 Bug 根因分析

> 2026-08-05 复核修订：本文原始审查记录保留作为问题来源，但实现结论以本段为准。当前官方 `nnInteractiveInferenceSession.set_image()` 只执行 nonzero-bbox z-score；它不按 spacing 对整卷图像重采样。提示时才按模型 plan patch size 做 prompt-centred crop/resize。因此不应为了“对齐官方”额外重采样整卷训练数据。当前实现统一为 canonical RAS 无插值重排、严格 shape/affine 配对、保留原 spacing、官方归一化。合成 Initial Mask 已移除，只保留空 Mask 和真实已有 Mask；空 Mask 首轮 Point Set 批量一次预测，已有 Mask 和后续修正逐提示预测；验证分别报告两种起点的基线 Dice、AUC 和 AUC 增益，任一实际存在的起点模式退化都不会自动替换当前模型。

**日期**: 2026-08-05
**范围**: Mimics-Script 的 nnInteractive 集成（交互流程、微调策略、方位处理、四个已知 bug、模型隔离、数据量不一致）
**方法**: 逐文件代码追踪（`nninteractive_bridge.py`、`runtime_py35/nninteractive_mimics.py`、`resource_locks.py`、`external/nninteractive-finetune/`、`tools/nninteractive_finetune_pipeline.py`）+ 设计文档对照

---

## 一、Mimics 中 nnInteractive 的使用逻辑 vs 原版（Q1）

### 1.1 实际执行逻辑（以代码为准）

交互链路由三层组成，注意**文档意图与代码实现有出入**：

| 层 | 行为 |
|---|---|
| Mimics UI 层 | 每次采集到新提示（一次 Point Set 可含多个正负点）→ 追加进 `state["interactions"]` → 全量重新入队后台 worker（`runtime_py35/nninteractive_mimics.py:4605-4606`） |
| 后台 worker | 收到完整 interactions 列表，在同一 remote session 内**按顺序重放**（`nninteractive_bridge.py:2150-2243`） |
| Session 内 | 空 Mask 的第一组 Point Set 先累积多个点、最后一点触发一次预测；已有 Mask 或任何后续修正则逐点预测，使每个修正都能看到上一轮 previous segmentation |

所以当前实现明确区分“建立首个结果”和“修正已有结果”：

- 空 Mask 的第一组提示允许同时采集多个正负点。前 N-1 点只写提示通道，最后一点运行一次预测，避免用户在还没有可修正结果时白等 N 次推理。
- 只要进入时 Mask 非空，或当前会话已经产生过结果，Point Set 中每个点都按顺序预测。后一提示以刚得到的 Mask 为 previous segmentation，不把多个纠错意图压成一次静态猜测。

关键细节是：

- **重放起点**（`_apply_interactions`，`nninteractive_bridge.py:2233-2243`）：
  - `incremental_interaction_replay` **默认开启**（`nninteractive_config.json:11`，`nninteractive_bridge.py:2067-2068`）：当新增交互是旧交互的严格前缀且初始分割没变时，`_incremental_start_index`（:2219-2231）返回增量起点，**只重放新增交互，直接从上一轮预测结果继续**（session 状态保持）。
  - 前缀变化 / 初始分割变化 / 强制 reset 时：`session.reset_interactions()` + 重新应用初始分割（`_apply_initial_segmentation`，:2109-2123），**从进入工具时的初始 mask（或空）重放全部提示**。
- **初始 mask 处理**：空 mask 传 `None`（`runtime_py35/nninteractive_mimics.py:2560`，`nninteractive_bridge.py:2101-2106` 双重判空）；非空 mask 通过 `session.add_initial_seg_interaction(..., run_prediction=False)` 写入 prev_seg 通道。
- 文档 `docs/nninteractive_mimics.md:115` 声称"每次重放从初始分割开始、不从上一预测结果开始"——这只在**非增量路径**成立，与当前默认配置（增量开启）不符。

### 1.2 与原版 nnInteractive 的语义区别

原版的常见交互是用户新增一个提示后运行一步，但 API 同时支持 `run_prediction=False` 来先累积提示。Mimics 只在空 Mask 的首轮使用该能力；后续仍保持官方的逐轮修正语义。本地 worker 持有同一 remote session，正常增量路径不重新上传图像，也不重复计算旧提示。

### 1.3 合理性评估

| 问题 | 结论 |
|---|---|
| 空 Mask、首轮一次多提示是否合理？ | **合理，但只限首轮**。此时没有上一轮预测可供逐点纠错，批量提示主要用于描述目标；一次预测可显著减少首轮等待。|
| 为什么后续不能继续批量？ | 后续提示表达的是对当前结果的纠错。逐点运行才能让后一提示看到前一点更新后的 prev_seg；因此已有 Mask 和首轮后的所有 Point Set 均为逐点预测。|
| 逐点 `run_prediction=True` 的代价？ | 每个纠错点一次推理，但这是修改阶段保持语义正确所需的成本。增量 worker 不重传图像、不重放旧提示，避免了额外开销。|
| 有初始 mask 时逐提示生成合适吗？ | **更合适**。初始 mask 经 `add_initial_seg_interaction` 写入 prev_seg 通道后，后续每个点都是"基于当前分割的编辑"，这恰是模型原生训练的编辑语义（原版 `supports_initial_label=True`） |
| 增量重放 vs 每次从初始重放？ | 增量重放（默认开）省算力、符合交互直觉；代价是 session 状态会**累积误差**——某一点点偏了，后续全部建立在其上。已有点错后的 reset 路径（fingerprint 前缀变化会触发全量重放，`_incremental_start_index` :2225-2230），但这依赖指纹严格相等，未覆盖"同指纹但用户想撤销重做"的情况。建议：提供显式"从头开始"操作时强制 `force_reset=True`（已有 Discard 路径从当前 mask 开新会话，可用） |

---

## 二、微调 vs 原版 CLoPA 训练策略（Q2）

### 2.1 差异总表

原版 nnInteractive 的 CLoPA 训练协议未开源（README 明确，`external/nninteractive-finetune/README.md:151-153`），本仓库只有原版**推理会话**与网络重建 stub。因此对比以"CLoPA 论文历史协议"（README 转述：每样本固定 5 次互动、paired-click、10 epochs）为基准。

| 维度 | 原版/CLoPA | 微调版 | 证据 | 合理性 |
|---|---|---|---|---|
| 交互预算 | 固定 5 步 | 1~5 步，短交互权重 `[0.35,0.25,0.20,0.12,0.08]`，`short_interaction_probability=0.7` | `config.py:40-46`; `prompts.py:157-203`; `trainer.py:521-553` | ✅ **合理**。Mimics 标注者实际就给 1-3 个点，训练分布匹配使用分布是正确思路，与"点击若干点出好效果"目标一致。风险：1-2 步占比过高可能弱化深度迭代修正能力 |
| 纠错点采样 | paired-click（一步一正一负两点） | `official_single`：每个错误只放一个点，按体积加权选连通分量 | `prompts.py:412-434`; `config.py:41`; `README.md:111-115` | ✅ 基本合理。真实标注者就是单点点击；paired-click 是教学信号（同时学欠分割与过分割）。**轻微风险**：单点采样可能对"同时欠+过"的错误学得不均衡，但 0.7 的短预算下影响有限 |
| 初始 mask | 原版支持（`add_initial_seg_interaction`），训练协议以空开始为主 | 只保留空 Mask 与真实 Initial Mask；`general` 可混合，`refine_existing` 要求真实草稿 | `prompts.py`; `trainer.py`; `README.md` | ✅ 输入语义可解释，训练与真实标注流程一致 |
| 合成初始 mask | 无此机制 | **已移除** | `prompts.py` | ✅ 不再用最终标签人工制造草稿，避免验证指标与真实上游模型分布脱节 |
| 提示类型 | points+scribble+bbox+lasso 全覆盖 | **只训练 clicks**，非 clicks 直接配置拒绝 | `config.py:114-118`; `README.md:25-28` | ⚠️ **最大的能力退化风险点**，详见 2.3 |
| 数据增强 | nnU-Net 公共训练组合 | 旋转/缩放、噪声/模糊、亮度/对比度、低分辨率、Gamma、镜像 | `config.py`; `data.py` | ✅ 概率、范围、缩放方向和插值语义已对齐公开的 nnU-Net/batchgeneratorsv2；不声称复刻未公开训练代码 |
| 方位 | 无强制 | 强制 canonical RAS + 严格 geometry（fail-closed） | `data.py:131-153, 180-192`; `config.py:150-154` | ✅ 合理（详见第三节） |
| 可训练参数 | 全量 | `clopa_in`：仅 InstanceNorm affine（27,648 参数）；`clopa_conv`：+ 部分 Conv3d（145,216）；`full`：全量 | `model.py:261-322` | ✅ **刻意设计的 PEFT 低遗忘方案**（详见 2.3） |
| Loss/优化器 | — | Dice+CE（未加权）+ Adam + AMP | `losses.py:9-28`; `trainer.py:404-409` | ✅ 与原版推理归一化一致（nonzero-bbox z-score） |

### 2.2 Mimics 使用微调模型的方式是否合理

**合理，证据链完整**：
1. 导出即原版格式：`checkpoint.py:61-140` 复制官方 `plans.json/dataset.json/inference_info.json`，只替换 `network_weights` → 原版 `initialize_from_trained_model_folder` 直接加载（`evaluate.py:231-233` 已验证）。
2. 信道布局逐一对齐：prev_seg=0、points=(3,4)、共 7 通道（`prompts.py:12-15` vs 原版 `inference_session.py:242-251`）。
3. 推理用原版 session 代码，`add_initial_seg_interaction`/`add_point_interaction` 是原版原生机制。
4. 加载前审计 `input_channels==7、output_channels==[2]`（`model.py:181-183`）。

### 2.3 固有能力退化分析（原版 vs 我们的方式）

**原版方式（全参微调 / CLoPA 协议）**：
- 若对单任务全参微调 → **灾难性遗忘**：scribble/bbox/lasso 通道行为、其他器官/域的分割能力都会退化。这是全参微调在分割基础模型上的公认风险。
- 原版协议本身是"任务适配"设计，同样存在此风险，只是它用多数据集 + 全提示类型训练来对冲。

**我们的方式（clopa_in：仅 InstanceNorm affine 可训练）**：
- 属于参数高效微调（PEFT）思路，**遗忘风险显著更低**：27,648 个 affine 参数只做特征图的逐通道缩放/平移调制，不动卷积权重，编码器冻结。
- **但有一个容易被忽略的传播面**：InstanceNorm affine 是**跨通道、跨提示类型共享**的。调整它们会整体改变所有交互类型对输入的表征响应——不只是点提示。由于参数量小，扰动温和，scribble/bbox 大概率只是轻微漂移而非失效，但**未被训练覆盖的行为没有保证**。README 已诚实标注 scribble/bbox 不作为已验证功能（`README.md:25-28`）。
- `clopa_conv` 额外解冻 encoder stem、stage0、最后 decoder stage、最后 seg 层卷积——对改动能力的拟合更强，遗忘风险也更高，应作为"更强但更危险"档位使用。

**结论**：我们的微调方式在"保留固有能力"上**优于**原版全参方式；代价是适配容量有限（27k 参数）。建议补充一个**回归评测**：微调前后用官方权重在同一批 scribble/bbox 样本上跑一遍，量化漂移，作为发布 gate。

### 2.4 有初始 mask 需要微调的场景

当前设计只使用真实草稿。`refine_existing` 会跳过没有真实 Initial Mask 的病例；`general` 中有草稿的病例按 `initial_mask_probability` 选择真实草稿或空起点，没有草稿的病例只能从空起点开始。这样不会把从最终标签构造的理想化错误分布冒充真实上游模型输出。

**原版微调方式是否支持从空初始 mask 开始？** 支持。原版推理天然支持空初始（`set_target_buffer(zeros)` + 点交互），训练协议也从空开始迭代。

---

## 三、方位统一与跨方位数据（Q3）

### 3.1 现状：两端都做了 RAS 归一化

- **微调数据端**：`data.py:131-152` 用 `nib.as_closest_canonical` 强制转到 **canonical RAS**（纯轴置换+翻转，无损）；qform/sform 冲突即拒绝（fail-closed）；image/label/初始 mask 必须同 shape + 同 affine（`data.py:180-192`，绝不隐式重采样掩盖不匹配）。
- **推理端（task_model）**：`runtime_py35/nninteractive_mimics.py:2600-2604` 只对 task_model 设 `model_input_space="canonical_ras"`；bridge 用 `canonical_ras_buffer_mapping`（`nninteractive_bridge.py:1065-1093`）把 Mimics 网格翻到 RAS 再喂给模型，预测结果经 `platform_to_mimics` 翻回（:1056-1062）。
- **官方模型**：保持当前 Mimics 数组顺序；官方 session API 不接收 affine/spacing，也不会替调用方自动做整卷方位或 spacing 重采样。它只做 nonzero-bbox z-score 和提示中心局部 crop/resize。

### 3.2 "其他方位的数据集装成 .mcs 再用微调模型"怎么办

**可以正常用**，前提是 Mimics 的 `voxel_to_ras` 元数据正确：
- Mimics 内部统一用 RAS 世界坐标表达（`mimics_bridge.py:1577-1578` 显式 `LPS_TO_RAS @ dicom_affine_lps`；`mimics_export.py:92,98` 声明 `world_coordinate_system: RAS`；label 与图像同网格导出 `export_space="source_image"`）。
- 任意方位的数据集导入 Mimics 后，bridge 会依据 `voxel_to_ras` 矩阵把输入翻到 RAS 再推理，结果翻回 Mimics 网格——**方位差异被无损映射吸收**。
- 历史坑：旧版本曾把 SimpleITK 的 LPS affine 直接存进 `source_voxel_to_ras`（差一整轴翻转），已有修复工具 `runtime_py35/fix_source_affine_metadata.py`。若遇到几何校验失败，先跑它。

### 3.3 spacing 与官方会话的复核结论

复核当前安装的官方 `nnInteractiveInferenceSession` 后，原审查中“`set_image` 按 plan target spacing 重采样”的判断不成立。官方 `set_image` 只按 nonzero bounding box 计算全图 z-score；收到提示后才提取 prompt-centred crop，并把 crop resize 到 configuration patch size。session API 不接收 spacing，也没有整卷 spacing 重采样。

因此当前正确契约是：canonical RAS 只做轴置换/翻转；保留源 spacing 元数据但不额外插值；图像、最终标签和 Initial Mask 必须同 shape/affine；训练固定体素 patch 与推理 prompt-centred crop 在上下文大小上可能不同，但二者使用相同方向与强度定义。盲目加入整卷 spacing 重采样反而会产生新的标签插值和训练/部署重复处理。

另一个容易被版本默认值掩盖的差异是提示通道衰减。当前官方 session 在能力文件未声明时可能采用自己的 `interaction_decay` 默认值，而 CLoPA 微调配置使用显式值。任务模型导出现在会把训练实际使用的 `point_radius` 和 `interaction_decay` 固化到 `inference_info.json`，保证部署 session 读取到同一组值，而不是随安装版本漂移。

---

## 四、如何达到"点击若干点就出好初始效果"（Q4）

当前实现采用 CLoPA 原论文的 paired 纠错：每个训练交互步在对应错误存在时各加入一个前景点和背景点，然后预测一次。常规训练使用 1-8 步的温和长尾，验证检查 1/3/5/8 步；它比旧版 1-5 步更能暴露后期退化，但不会把论文用于研究分析的 100 步评估成本带进每轮训练。已有真实 Initial Mask 会按来源以及相对最终标签的 Dice/precision/recall/体积比分层，空起点与不同质量草稿分别报告。

数据增强默认使用可公开核验的 nnU-Net 训练组合及其概率/范围：3D 旋转缩放、噪声、模糊、亮度、对比度、低分辨率模拟、Gamma 与镜像。实现额外核对了缩放矩阵方向、偶数 patch 的半体素中心、图像线性插值、二值标签的线性类别概率决策、Gaussian reflect padding/官方核尺寸和低分辨率的 nearest-exact→trilinear 语义。nnInteractive 原始训练代码没有公开，因此这里是依据其 ResEnc-L/nnU-Net 训练基础实现的数学与协议对齐版本，而不是声称逐随机数复刻内部训练器。Mimics 训练界面的 `Anatomy mirroring` 默认为 Auto：任务或目标 Mask 名含 left/right、L/R、左/右时禁用 canonical RAS axis 0 的左右翻转，其余两个轴仍可增强；解析结果写入配置和日志。

在此基础上按优先级补充：

1. **用真实标注轨迹训练（收益最大）**：在 Mimics 里记录真实标注者的点击序列，重放为训练样本（interaction trajectory 采样），替代合成误差点采样。合成误差与真实纠错的分布差异是"几点击效果"与"真实手感"之间的最大鸿沟。
2. **首点效果作为训练监控指标**：训练时单独统计"空初始 mask + 第一个点后"的 DSC，作为与最终 DSC 并列的发布指标——直接对齐你的目标。
3. **首点/前几点给模型更多信息**：确认 `clopa_in` 档位下 27k 参数有足够的首点表达能力；若首点效果不足，优先试用 `clopa_conv`（145k 参数）而非 full。
4. **预标注衔接**：用 DINOv3/大模型预标注结果作为初始 mask 输入（pipeline 已支持 `initial_mask_probability` + 真实草稿），微调模型负责"首点精修"——这是现实中成本最低的路径。
5. **修掉 3.3 的 spacing 缺口**，否则任何"几点击"效果在异源数据上都会打折扣。
6. **数据量与多样性 > 策略微调**：每个器官/任务 >= 10-20 例配对数据，覆盖不同 spacing/窗宽，比调任何超参都有效。
7. **保留最小 scribble/bbox 训练**（哪怕只占 5-10% 样本），维持编辑能力不漂移；同时建立 2.3 提到的回归评测。

---

## 五、四个 Bug 根因与杜绝方案（Q5）

### 总览：四个 bug 高度耦合，根子在"server 生命周期失败不视为可恢复事件"

| Bug | 阶段 | 根因类别 |
|---|---|---|
| 1. GPU 僵尸锁 | initialize_or_poll | 存活性判断在 Windows 下不可靠 → 死进程锁不清理 |
| 2. add_point_interaction 500 | async_prediction | 输入越界/会话失效，且 500 不被归类 → 无限重试 |
| 3. Transaction 构造失败 | 写回 Mimics | 双分支 fallback 前提错误 + 失败时 mask 不写 |
| 4. WinError 10061 | async_prediction | server 已死但无自动重启路径（只覆盖 "at capacity"） |
| 5. 模型串台隐患 | 切换模型 | official 身份 cache key 不含 checkpoint sha |

### Bug 1：GPU is already in use（僵尸锁）

**根因链**：锁记录的是 **server 进程 pid**（`nninteractive_bridge.py:839-844` 把锁转移给子进程）。锁清理依赖 `_is_stale` → `process_matches(pid, process_start_marker)` 判死；但 `_process_matches_server` 在 **Windows 下读不到命令行时返回 True（信任 PID）**（`nninteractive_bridge.py:462-471`），PowerShell CIM 超时/权限异常返回 None 也走信任路径 → **已死进程被误判为存活 → 锁永不释放** → 下个 job 报 "gpu is busy ... (pid 27100)"（27100 就是上一轮死掉的 server）。

**已实现**：
- 状态文件新增独立的 `server_heartbeat_epoch`。watchdog 只有在携带 ownership token 的 `/healthz` 检查成功后才刷新它，不能用普通任务活动时间冒充服务存活。
- `_process_command_line` 在 Windows 依次尝试 `psutil`、PowerShell CIM 和 WMIC；全部不可读时，不再无限信任 PID，只接受带 token 的实时健康检查、90 秒内的成功健康心跳、与最长首次加载时间一致的有限启动宽限期，或带进程启动标记且仍在超时预算内的活动推理。最后一项只用于保锁，防止健康端点在长推理中暂时不可达时提前放行其他 GPU 任务。
- PID 不存在或命令行明确不匹配模型路径与 ownership token 时立即判定为旧状态；清理仍按“先退休状态文件，再释放同 token GPU 锁”的顺序执行。

### Bug 2：add_point_interaction 500

**根因**（server 源码不在仓库，从客户端可控制输入反推）：
- **坐标越界**：`_apply_point_set`/`_apply_interaction` 的点/框坐标经 buffer_mapping（RAS 翻转）后可能越出平台网格 → server 越界 500。
- **crop 与 bbox 尺寸不一致**：scribble/lasso 走 `_iter_2d_interaction_crops`（:1501-1532），仓库已有 "Interaction crop shape mismatch" 守卫（:1668-1671），说明这是已知可触发路径。
- **会话失效**：server 以 `--max-sessions 1` 启动（:795），异步 worker 间抢占唯一 session 或 session 状态被 reset 后，再 POST 也会 500。
- **500 不被归类**：`_FATAL_ERROR_SUBSTRINGS`（:2432-2446）没有 "500"，`_is_recoverable_worker_error` 判其可重试 → **反复重试同一 500，用户看到的就是反复失败**。

**已实现**：
- 点和轨迹坐标在发送前本地校验；只钳制最多一个体素的显示边缘舍入误差，明显越界直接抛本地 RuntimeError，不把错误提示悄悄移动到其他解剖位置。
- 越界致命错误同时识别 `outside image bounds`、`out of bounds`、`IndexError` 与 `index error`，不会再为确定性的坐标错误无谓重启 server。
- 把 HTTP 500/`HTTPError` 并入致命错误处理：触发 **server 重启 + 本次交互重放**（复用 `_connect_and_upload` 已有的 capacity-restart 模式 :2015-2044），而不是无限重试。
- 500 时把 server 日志尾部（`_server_log_path` :148-150）一并回传，让错误可定位。

### Bug 3：Transaction 构造失败（mask 未被修改）

**错误原文**：`Named constructor: object() takes no parameters; compatibility constructor: __init__() missing 1 required positional argument: 'transaction_name'`

**根因分析**（`runtime_common.py:79-92` 的双分支 fallback）：
- 两个分支的报错**自相矛盾**：named 分支报 "object() takes no parameters"（说明类的 C 层 `__new__`/tp_new 拒收位置参数），no-arg 分支报 "missing transaction_name"（说明存在 Python 层 `__init__(self, transaction_name)`）。同一个类不可能同时满足两者——**`mimics.Transaction` 不是普通 Python 类，而是脚本库兼容层包装类**（C 层构造器拒参 + Python 层 init 要求名字）。
- 我们的 fallback 前提"named 构造 TypeError ⇒ 旧版无参构造，试一下 no-arg"是**错误启发式**：named 的 TypeError 根本不是签名不匹配，而是包装类的 `__new__` 行为。
- 官方 API 文档（`docs/Mimics_API_Documentation_CN.md:1182-1198`）显示 Transaction 基类 `object`、未文档化构造参数——构造方式本身就需要按 Mimics 版本探测。
- **最恶劣的后果**：事务启动失败 → `_set_mask_from_u8` 抛 RuntimeError → **mask 根本没写入**（"the Mask was not changed"），用户白点了一轮。

**已实现**：按 named、no-argument、显式 `__new__ + __init__(name)` 三种兼容形式尝试构造。若 Mimics 的包装绑定仍全部拒绝，则向 Mimics log 写入 warning，并对已经过 shape/hash/活动项目校验的结果执行无事务直写。这样不会因为 Transaction 包装差异丢失整轮 AI 结果；能正常建立事务时仍保留单步 Undo 和异常 rollback。

### Bug 4：WinError 10061 连接被拒

**根因**：10061 = 端口 1527 **没有监听进程**（server 没起来或已死）。三条链：
1. **watchdog 空闲回收**：worker 只在 `__init__` 里 `_ensure_server` 一次（:2565），之后复用；watchdog 按 `server_idle_timeout_seconds`（默认 1800s）回收 server（:727-735）。投递提示间隔超过 idle timeout → 下次 predict 直接 10061。
2. **无自愈路径**：`_is_capacity_error` 只认 "server is at capacity"（:1829-1831）；10061/URLError 不被识别 → 不触发 `_ensure_server` 重建 → worker 一直对死 server 发请求。
3. **启动失败无重试**：`_start_server` 异常只做清理（:834-837），不自动换端口/换 python 重试。

**已实现**：`_BridgeSessionContext.predict` 外层有一次性自愈守卫。对连接拒绝、连接中断和 server 500，只有持有 ownership token 的本地 worker 才能停止并重建自己创建的 server，然后重新上传图像并完整重放本次 interactions；最多重试一次，避免永久循环或误杀其他服务。

### Bug 5：官方/微调权重隔离与切换

**机制现状（基本正确）**：
- server 层：`_ensure_server` 校验 `model_dir/device/fold/checkpoint_sha256` 四要素，不匹配即终止旧 server 重启（`nninteractive_bridge.py:921-954`）；checkpoint 身份不符拒绝启动（:964-971）。
- worker 层：`_worker_cache_key = image_key + "::" + model_identity`，identity 含 `profile_id|model_id|checkpoint_sha256`（`runtime_py35/nninteractive_mimics.py:863-916`）→ 切模型即切 worker。
- 切换前 `_retire_different_model_workers` 清理异模型 worker（:3317-3345），"自杀式换模型"是刻意设计。

**边界与防护**：
1. official 身份仍使用固定 key，但 task model 身份包含 `profile_id|model_id|checkpoint_sha256`，所以官方模型与微调模型并不会共用 worker。剩余边界仅是“运行中原地替换官方 checkpoint 文件”；这属于不支持的部署操作，应先停止后台服务再替换权重。
2. 模型中心默认隐藏 `failed/corrupt/incompatible` 版本；即使用户显示失败历史，`model_is_usable()` 仍禁用其 `Set Active` 操作。

**操作要求**：不要在后台服务运行时原地覆盖官方 checkpoint。通过模型中心切换注册的 task model；若要维护官方权重，先运行 Stop Background Services，再替换并重新启动。

---

## 六、Original image dataset 与 Saved .mcs folder 数据量不一致（Q6）

**处理原则：只含图像、无配对 mask 的 case 直接跳过，不做半监督利用。**

具体逻辑（`tools/nninteractive_finetune_pipeline.py`）：
1. **单 case 跳过**：source_mode="mcs" 时 `_find_exported_label(staging, case_id)` 找不到匹配 target Mask → log `Skipped {case} because its saved .mcs project has no matching target Mask.` 并 `continue`（:1019-1028）——这正是你说的"不一定每个数据都标注了"的落地处理。
2. **硬性下限**：全部 case 都没有配对 label → 直接报错 `No selected case has both a source image and a matching target Mask.`（:1141-1144），不允许无标注训练。
3. **验证集兜底**：划分出的 val case 无匹配 Mask 时，把最后一个 train case 重赋为 val（:1170-1177）。
4. 训练端同样硬校验：manifest 每行 image/label 必须存在，缺失即 `FileNotFoundError`（`data.py:71-84`）。

**含义与建议**：
- 未标注 case 永远不会进入训练（不是半监督），这与微调场景匹配——**微调必须要有配对标注**。
- 想利用未标注数据的路径：先 prelabel（DINOv3/官方 nnInteractive 粗标），保存该真实草稿，再人工修正成最终 Target Mask；两者以同病例配对进入 existing-Mask 训练通道。
- 若选中的 case 里很多没有标注，Mimics 端建议在训练设置页**显示"已配对/总数"比例**（如 8/15），避免用户困惑为什么训练集比选的数据少。

---

## 七、行动清单（按优先级）

| 优先级 | 事项 | 对应问题 |
|---|---|---|
| 已完成 | `predict` 外层统一自愈守卫（server 死→重建→重放一次），500/10061 一并归入 | Bug 2、4 |
| 已完成 | Transaction 兼容构造；全部构造失败时记录警告并回退无事务直写 | Bug 3 |
| 已完成 | 锁存活性加入 token 健康检查与独立心跳证据；命令读取失败不再无限信任 PID | Bug 1 |
| 已澄清 | task model 身份已包含 checkpoint；运行中不支持原地替换 official checkpoint | Bug 5 |
| 已完成 | 发送前校验坐标/shape，超界在 bridge 本地报错；失败结果保留 server 日志路径 | Bug 2 |
| 已完成 | 固化官方输入契约：canonical RAS 无插值重排、严格 affine、无额外 spacing 重采样 | Q3 |
| 已完成 | 仅空 Mask 首轮 Point Set 批量预测；已有 Mask 和后续修正逐点预测 | Q1 性能与语义 |
| 已完成 | `refine_existing` 全部病例缺少有效真实草稿时给出专用诊断，不再误报为缺少 Target Mask | 数据准备诊断 |
| 已完成 | 删除无消费端的 synthetic Initial Mask CLI；旧配置键改为明确拒绝并给出迁移说明 | 配置一致性 |
| P2 | 录制真实标注轨迹作训练数据；首点 DSC 作为监控指标 | Q4 |
| P2 | scribble/bbox 回归评测作为发布 gate；少量混合训练防漂移 | Q2 退化 |
| P3 | 训练设置页显示已配对/总数比例 | Q6 |
