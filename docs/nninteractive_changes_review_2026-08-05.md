# 本地改动审查报告（2026-08-05）

**审查对象**: 用户声称已完成的 5 项修复（27 个文件，+10208/-9491 行，其中 MimicsHelp_MD 大量改动为行尾噪声）
**方法**: 逐文件 diff 审查 + 原版 nnInteractive 源码交叉验证 + 独立跑测试套件 + Python 3.5 语法检查

---

## 总体结论

**5 项声称全部成立或核心语义成立**，无严重新引入问题。审查中发现 1 个中低严重度缺口（越界错误黑名单覆盖不全）和 4 个低优先级改进点。测试实测：微调包 39 passed、nnInteractive 专项 167 passed + 1 skipped、综合回归 436 OK（从仓库根目录运行）、runtime_py35 18 文件 Python 3.5 语法 0 错误。

另外，本次改动**纠正了我上一份审查报告中的一个错误判断**（spacing 问题），详见声称 3。

---

## 声称逐条核对

### 1. 多提示逻辑 — ✅ 已实现

**空 Mask 首个 Point Set 批量预测**（`nninteractive_bridge.py:1779-1796`）：
- interaction 携带 `prediction_policy`：`"initial_empty_batch"`（空起点首提示）或 `"sequential"`（默认）。
- 批量模式下前 N-1 点 `run_prediction=False`、最后一点预测一次；逐点模式行为不变。
- **已核实原版 API 语义**：`add_point_interaction(..., run_prediction=False)` 只写入点通道、不触发预测、返回 None（原版 `inference_session.py:1181-1209`），批量模式成立。N 点 Point Set 从 O(N) 次全图预测降为 O(1)。

**Mimics 侧条件**（`runtime_py35/nninteractive_mimics.py:4601-4608` async、`:4751-4757` sync）：`首个提示 && base_pixel_count==0` → 批量；其余（非空初始 mask、已有预测结果、后续提示）→ 逐点。`base_pixel_count` 在 `_start_async_job` 新增（:3687）。与声称完全一致。

**一致性细节**：policy 随 interaction dict 持久化（`_persist_interaction`），因此 undo/reset/指纹失配全量重放时批量策略保持一致，不会串模式。单点 Point Set 无行为变化（`index == len(points)-1` 时即最后一点）。

### 2. Initial Mask 策略 — ✅ 已实现

- 合成初始 mask（腐蚀/膨胀/平移/轴截断/稀疏切片）生成函数**完全移除**，无残留执行路径（prompts.py 全量检索确认；scipy ndimage 仅剩点核 EDT 与连通域用途）。
- `initial_mask_probability`（默认 0.5）新语义 = **使用真实草稿的概率**；`provided_initial_mask_probability` 从配置删除，旧键加载时 pop 迁移（`config.py:256-260`）。
- **refine_existing 两级过滤**：manifest 生成级跳过无真实草稿病例（`tools/nninteractive_finetune_pipeline.py:1095-1103`）+ trainer prepare 级再过滤（`trainer.py:418-425, 443-458`）；无可用训练病例时报错（:423-425、:454-458、pipeline:1169-1172）。
- **旧配置迁移幂等**：pipeline 请求层 synthetic→none（`pipeline:931-939`）+ config 键剥离（`config.py:256-260`），二次 resume 无副作用；不扰动已注册模型。
- 无真实草稿时回退空起点（测试 `test_unavailable_supplied_mask_falls_back_to_empty_not_synthetic` 覆盖）。

### 3. 训练与推理一致性 — ✅ 已实现（且修正我上轮报告的错误）

- 无损 canonical RAS：`as_closest_canonical`（`data.py:151`），无插值重采样路径；shape/affine/qform-sform 严格校验 fail-closed（`data.py:140-150, 338-352`）。
- nonzero-bbox z-score 训练/推理统一。
- **保留源 spacing 是对的**：我逐行核实原版推理 session——`inference_session.py:731` 明确 "spacing is accepted for API compatibility but currently [ignored]"，server 端预处理只有 z-score + 非零包围盒（`server/app.py:140` `_finish_preprocessing_and_initialize_interactions`），**官方 session 确实不做整卷 spacing 重采样**。我上轮报告基于"server 按 nnU-Net plan 重采样"的假设不成立，本次"保留源 spacing"与推理行为一致，是正确的决定。
- **point_radius / interaction_decay 导出有效**：`checkpoint.py:49-54` 写入 `inference_info.json`，且原版 `_load_capability_and_runtime_defaults`（`inference_session.py:625-629`）**确实读取**这两个字段。训练默认 decay 0.9 vs 原版默认 0.98 的差异被显式导出消除——这是我上轮报告提到的真实风险点，已闭合。

### 4. 验证 AUC — ✅ 核心达成（一处近似实现）

- evaluate 分别输出 `empty_mask_trajectory_auc` / `real_initial_mask_trajectory_auc`、基线 Dice（空起点基线定义为 0，`evaluate.py:240`）、相对基线增益（`evaluate.py:358-398`）。
- 最终比较逐模式执行（`_quality_result`，`pipeline:1853-1938`）：empty-vs-empty、real-vs-real 独立对比。
- **auto-promote gate 五重条件**（`pipeline:1939-1946`）：① `comparable_case_ids` 非空（成对门槛）② 全局 delta ≥ 门槛 ③ 无全局严重回归 ④ `mode_severe` 为空（含 **起始 Mask 基线变化**检测 :1873-1880、逐模式逐 case 回归）⑤ `mode_not_improved` 为空（逐模式 delta 门槛）。任一不满足 → `_register_model` 不自动提升。
- **`validation_count` 已修复**：改读 post-skip 的验证 manifest（`pipeline:2208-2211`），不再用原始 request 行数（修复了我上轮报告点名的计数高估问题）。
- ⚠ 近似点：成对门槛 `comparable_case_ids` 基于 blended 轨迹交集，未逐模式强制配对；但两模式均无数据的模式被跳过（无数据可比，可辩护），单侧有数据的模式会被 `starting_matches` 拦截（`mode_severe: starting_mask_baseline_changed`）——实际比表面更严格。

### 5. 运行时问题 — ✅ 已实现

**Transaction 修复**（`runtime_py35/runtime_common.py:77-120`）：
- 三阶构造：named → no-arg → **显式 `__new__` + `__init__(name)`**（针对"C 层 tp_new 拒参 + Python 层 `__init__` 要求 transaction_name"的包装类——正是上轮报告分析的矛盾签名模式）。
- 全部失败 → **warning + 回退无事务直写**（`operation()`），保住已验证 mask 写入，不再出现"Mask was not changed"。测试覆盖（`tools/test_all.py:317-333`）。

**server 自愈**（`nninteractive_bridge.py:2106-2149, 2257-2276`）：
- `_server_failure_is_recoverable`：黑名单（OOM/cuda/cudnn/checkpoint/corrupt/outside bounds）→ 不可恢复；白名单（10061/connection refused/remote closed/500/internal server error）→ 可恢复。
- `_restart_owned_server`：**只重启自己持有的 server**（校验 `auto_start_server` + 127.0.0.1 + `owned_state_path` + ownership token），终止旧 server → 清 state → `_ensure_server` 重建 → 重新 `_connect_and_upload` → **只重放一次**（第二次失败直接传播）。
- 重放正确性：重建后 `_applied_initial_key=None` + `_applied_interaction_fingerprints=[]` → 从初始分割全量重放；`initial_platform` 是内存数组，重启后仍可用。
- **无无限重试**：二次失败进入 worker 层 `_FATAL_ERROR_SUBSTRINGS`（新增 500 类，:2579-2581）→ worker 以 code 2 退出 → Mimics 侧感知死 worker 弹"Start new session"。错误"只重放一次"的声称成立。
- `finally` 块安全：`_set_server_operation_active` 对缺失 state 文件 no-op（:682-684），不会重建已删除的 state。
- **UI 同步**：start_empty 强置 `none`（pipeline:950-951），模型中心 refine_existing 要求非 none 源（model_center:1154-1161），方向一致。

---

## 发现的问题（按严重度）

| # | 严重度 | 问题 | 位置 | 建议 |
|---|---|---|---|---|
| 1 | 中低 | **越界错误黑名单覆盖不全**：黑名单只含 "outside image bounds"，但越界真实报错形态是 `IndexError: index 512 is out of bounds for axis 0 with size 400`（不匹配）→ 越界会被误判可恢复，触发一次无谓的 server 重启+重放后才失败。声称"越界不被误判"只部分成立 | `nninteractive_bridge.py:2106-2123` | 黑名单补 "out of bounds"/"index error"；更稳的是发送前本地钳制坐标（上轮报告 P1 项，仍未做） |
| 2 | 低 | refine_existing 全跳过时报错文案是通用错误（"No selected case has both a source image and a matching target Mask"），不指明清原因（真实草稿缺失），排障方向误导 | `tools/nninteractive_finetune_pipeline.py:1169-1172` | 区分文案：缺 label vs 缺真实初始 mask |
| 3 | 低 | per-mode 成对门槛为近似实现：两模式均无数据的模式不参与 gate（可辩护）；`validation_count` 是 post-skip manifest 数，不是 post-prepare 数（trainer 内 refine_existing 还会因空/与 target 相同拒绝初始 mask 再过滤） | `pipeline:1846, 2208-2211` | 可选：gate 用 `real_initial_mask_validation_cases` 等 post-prepare 计数 |
| 4 | 低 | 旧配置中除 `provided_initial_mask_probability` 外的 `synthetic_*` 键仍被 `_merge` 静默保留（无行为、落灰），未随本次迁移一并 pop | `config.py:73-80, 256-260` | 一并 pop 或 warn |
| 5 | 低 | `cli.py:153-158` 保留隐藏的 `--provided-initial-mask-probability`（resume 兼容），已无任何消费端 | `cli.py:153-158` | 清理（确认无旧控制器依赖后） |
| 6 | 信息 | 批量模式语义注意：empty-start 批量预测与逐点预测并非严格等价（中间点看不到前一点的 prev_seg），对"一次采集多点"的标注意图更贴切；训练覆盖多点通道（预算 1-5 步内累积），无问题 | — | 无需改，实机验收时对比手感 |

---

## 测试实测结果

| 套件 | 命令 | 结果 |
|---|---|---|
| 微调包 | `pytest tests/`（external/nninteractive-finetune） | **39 passed** ✓ |
| nnInteractive 专项 | `pytest tools/test_nninteractive_bridge_prompts.py tools/test_mimics_nnint_deep.py tools/test_mimics_nnint_functional.py` | **167 passed, 1 skipped** ✓ |
| 综合回归 | `python tools/test_all.py`（**必须从仓库根目录跑**） | **436 tests OK** ✓ |
| Python 3.5 语法 | `ast.parse(feature_version=(3,5))` × 18 文件（runtime_py35） | **0 错误** ✓ |

注意：从 `tools/` 目录运行 test_all.py 会因相对路径（`fewshot_config.json`、`external/dinov3-medical-seg/`）产生 **8 个假失败**——是 cwd 问题不是代码问题，从仓库根目录运行全过。用户声称的 "436 passed" 实测吻合；"212 passed" 与实测 206+1 相近（可能含其他测试文件或环境差异）。

新增测试质量确认：`test_point_set_rejects_unknown_prediction_policy`、`test_server_transport_failures_are_recoverable_once`（10061 分类+单次重试+不可恢复路径）、`test_mimics_transaction_binding_failure_keeps_validated_write_available`、`test_legacy_synthetic_mix_key_is_removed_when_loading`、`test_inference_prompt_contract_is_explicit_and_preserves_capabilities` 等均真实覆盖新行为，非凑断言。

---

## 遗留验收项

与用户声明一致，以下必须在 Mimics 实机（Windows）验收：
1. Transaction 真实绑定（三阶构造哪一阶生效）与失败回退时 mask 确实写回。
2. 500/10061 后 server 重启 + 重放一次的真实体验（含 GPU 锁重新获取）。
3. 批量预测模式在真实 Mimics 交互中的手感（多点一次采集 vs 逐点）。
4. Windows 下僵尸锁修复（心跳第二证据）——注意：**本次改动未包含心跳/`_process_command_line` 兜底修改**（上轮报告 Bug 1 的 P0 建议），若用户声称"已修复僵尸锁"则未完全落实，但自愈重启路径（`_restart_owned_server` 内 `_terminate_owned_server` + state 清理）部分缓解了该问题。建议确认。

**文档同步核查**：`docs/nninteractive_mimics.md`（16 行）、`README.md`（42 行）、架构文档（14 行）的改动与代码一致（MimicsHelp_MD 的 17982 行变动疑似行尾符噪声，建议 `git diff --stat` 单独确认后提交）。
