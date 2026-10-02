# E:\MIMICS_SCRIPTS 系统性重构交付报告

> 交付日期：2026-09-21　分支：`refactor-overhaul`（自 16dd543，共 11 个提交）
> 范围：专题一（导入导出 P2/P3/P4）、专题二（nnInteractive 交互）、专题三（微调可诊断性）、专题四（架构收敛 + 健康面板）、专题八（清理与删除）
> 测试基线：`python_env/python.exe tools/test_all.py` —— 分支起点 455 项，交付时 **508 项全绿**

---

## 一、提交清单

| 提交 | 阶段 | 内容 |
| --- | --- | --- |
| ec14bb1 | Phase 0 | 部署树瘦身：external/ 未复制文件 D 状态正式提交，删除一次性脚本与逆向工程产物，`.gitignore` 增补 |
| bd180da | Phase A | 淘汰功能与死代码删除（guided / IGAC / ITK Snake / sync 死链 / 环境变量修剪） |
| 7b66cb3 | Phase B | 统一进程编排层（进程注册表 + 终止阶梯 + sweep + pipeline_common） |
| 343f3b2 | Phase C | nnInteractive 连续提示循环 + GPU free-VRAM 预检 + 错误→动作映射 |
| 1496270 | Phase D | 微调可诊断性（配置全表文档、not_improved 过滤、job 诊断聚合、基线对照） |
| 7b9fb65 | Phase E | 导入导出收尾（数据集档案、拖放导入窗、导入回执 + 一键撤销） |
| (本次) | Phase F | 健康总览面板 + 交付文档 |

---

## 二、痛点解决清单（每条附验证方式）

### 专题一：导入导出

| # | 痛点 | 解决 | 验证 |
| --- | --- | --- | --- |
| 1.1 | 导入发现逻辑 4 处硬编码重复，改布局要改 5 个文件 | `dataset_profiles.json` + `runtime_py35/dataset_profiles.py` 单一事实源，5 个发现点（dataset import / single-case / mask export / 路径 UI / bridge）全部改读同一份 | `TestDatasetProfiles`（16 项，含 ts-like 逐字节等价 + 8 例发现结果 diff 为空） |
| 1.2 | 批量导入无认出清单，出错要等跑完才知道 | UI 一行摘要（"Recognized N case(s); images: …; M mask(s); K skipped"）后台线程刷新；仅异常（多 volume / 无可用图像）在提交时打断一次 | `TestDatasetProfiles.test_summarize_*` |
| 1.3 | 导入操作不可撤销，标错只能手动清 | 每次导入写 `.<case>.import_receipt.json`（mask 名单 + 元数据键 + 指纹）；`99_Admin/05_Undo_Last_Import.py` 一键撤销：事务删 mask，.mcs 指纹未变则连文件回退，变了只撤 mask 并明示；回执撤销后消费，防二次撤销 | `TestImportReceiptAndUndo`（8 项：schema、最新回执选取、往返回退、指纹不同保留文件、删除失败保留回执、无回执提示） |
| 1.4 | Mimics 无法拖放导入 | `01_Data/08_Quick_Drop_Import.py` 打开外部置顶 PySide6 拖放窗：分类（单文件/病例夹/数据集根/多选）→ 档案解析 → 认出摘要 → 提交现有 worker；30 分钟空闲自动退出（注册表 `idle_timeout_s:1800`），无新增常驻服务 | `TestImportDropWindow`（9 项：5 种分类场景、排除目录、batch/single 提交调用、offscreen 渲染 + 注册表注册） |

### 专题二：nnInteractive 交互

| # | 痛点 | 解决 | 验证 |
| --- | --- | --- | --- |
| 2.1 | 每次提示都要重进菜单 | `_continue_session_prompt`：applied 后在 monitor tick 内再弹提示菜单，一次菜单进入连续点点连续出结果；Finish/UNDO/RESET 语义不变 | `TestNNInteractiveContinuousPrompting`（fake mimics 模拟 applied→再弹→Finish 序列） |
| 2.4 | GPU 被占时 server 启动后预言失败，用户不知原因 | `_start_server` 拿 GPU 锁前 `torch.cuda.mem_get_info()` 预检，free < `minimum_free_gpu_memory_gb`（默认 4GB）→ 友好报错并列出当前 free/total 与建议动作 | `TestNNInteractiveGpuMemoryPrecheck`（mock 高于/低于阈值） |
| 2.2 / 4.4 | 失败弹窗只有原始 traceback | `_error_guidance(error_text, stage)` → (category, plain-English message, suggested_action)，接入失败弹窗、结果异常、worker-stopped 三处；类别：OOM / 连接拒绝 / 环境损坏 / stale target | `test_all.py` 错误映射表测试 |

### 专题三：微调可诊断性

| # | 痛点 | 解决 | 验证 |
| --- | --- | --- | --- |
| 3.2 | 20 个配置键零文档 | `CONFIG_REFERENCE.md` 补全 `nninteractive_finetune_config.json` 全表（含 `minimum_epochs` 存在原因）、fewshot/io config 键、修剪后环境变量表 | 文档审阅 |
| 3.4 | not_improved 权重与可用权重并列可选 | chooser 与 Model Center 默认隐藏 `not_improved`（"show all" 开关后可见），行内 "Did not improve" 标签 | `TestNNInteractiveTaskDiagnostics`、`test_model_center_hides_not_improved_models_by_default` |
| 3.1 / 3.5 | job 失败只有一个 status 字段，无聚合诊断 | `diagnose_job(job_dir)`：status 错误 + 阶段链 + 日志尾部 + 工件存在性 → Model Center 失败态显示聚合诊断块 + "Open job folder" 按钮 | `test_nninteractive_task_integration.py` |
| 3.3 | 训练曲线无参照，不知道好不好 | TrainingCurve 增加基线 AUC 参考线；进度页 plain-English 对照（"+X% vs current model so far"） | `test_all.py` |

### 专题四：架构与进程管理

| # | 痛点 | 解决 | 验证 |
| --- | --- | --- | --- |
| 4.1 | 状态散落 4 处，无一处能回答"现在有哪些我们的进程" | `resource_locks.py` 进程注册表（`mimics_process_record.v1`，每进程一 JSON 原子写）：role/pid/start_marker/cmdline 签名/ownership token/state_path 指向各子系统状态文件（清单不是状态库，零迁移） | `TestProcessRegistry`（注册/快照/判活/终止阶梯/sweep 语义/并发） |
| 4.2 | 杀进程阶梯 7 份重复、激进清理靠胆量开关 | 唯一阶梯 `terminate_process`（terminate → 宽限轮询 → taskkill /T /F）；sweep 默认安全化（token + start_marker 证明所有权），`MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START` 退役 | `TestProcessRegistry`、`offline_stress_test.py` |
| 4.3 | 无健康总览 | `99_Admin/06_System_Health.py` → `tools/system_health_panel.py`（外部 PySide6）：进程（含判活 + 持锁）+ 锁（live/stale）+ 导入队列（running/stopping/idle + 深度）+ nnInteractive server 一页总览；动作按钮：Refresh / Sweep Stale State / Stop All Owned Services（复用 kill-background，先发停止标记后杀） | `TestSystemHealthPanel`（8 项，含真锁 live/stale 分类、队列状态机、server 判活、offscreen 渲染） |
| 4.5 | 前台 Mimics 崩溃后状态不一致 | 不变式：**前台 Mimics（重）启动后 sweep 一次必然回到一致状态**——死记录清除、孤儿按 policy 终止、锁经 token 释放、队列 stop marker 生效 | `TestProcessRegistry` 模拟"Mimics 死 → 残留记录 → 重启 sweep → 锁释放"序列 |
| 5.1 | 两管线 ~600 行同构重复、后台 Mimics 互斥协议不一致 | `tools/pipeline_common.py` 单一家（原子写、cancel 标记、分阶段复制、终止阶梯、scoped 后台 Mimics 锁、spawn helper）；finetune 导出锁从"每 job 唯一文件名（永不互斥）"改为统一 scoped 协议；指纹/缓存内核各自保留（目标性合并原则） | `TestPipelineCommon`（三管线公共层等价） |

### 专题八：清理

见下节删除清单。残留扫描（grep igac / itk_snake / guided_prompt / dino_guided / 已退役环境变量，排除 external/、.git、python_env）：**零命中**（get-pip.py 与 PAIN_POINTS_ANALYSIS.md 为本机遗留物，本次已从 git 移除/忽略）。

---

## 三、删除清单

| 类别 | 文件/功能 | 行数 | 原因 |
| --- | --- | --- | --- |
| 整链删除 | `tools/igac_engine.py` | 1242 | IGAC 淘汰（用户决策） |
| 整链删除 | `tools/igac_gui.py` | 3737 | 同上 |
| 整链删除 | `tools/local_bspline_ffd.py` | 165 | IGAC 专属 FFD |
| 整链删除 | `tools/ffd_reference_dll_adapter.py` | 181 | IGAC 专属适配器 |
| 整链删除 | `scripting_library/02_AI/IGAC.py` / `ITK_Snake.py` | 56 | UI 入口 |
| 整链删除 | `docs/igac_mimics_integration.md` + 2 预览图 | 123 | 文档 |
| 整链删除 | `tools/dino_guided_prompt_review.py` | 319 | DINOv3 引导点淘汰（用户决策） |
| 整链删除 | `scripting_library/02_AI/nnInteractive/04_DINOv3_Guided_Points.py` | 28 | UI 入口 |
| 死代码 | nninteractive_mimics 同步链（`_run_sync`/`_prompt_menu`/`_BridgeWorker`/`_run_prediction`/`_bridge_call`） | ~528 | 零调用，且形状恰是异步版的劣化版 |
| 死配置 | `reuse_session` 键；fewshot guided 路径（fewshot_mimics 内 136 处引用的 guided 族） | ~800 | 随功能淘汰 |
| 环境变量 | NNINTERACTIVE_TIMEOUT、NNINTERACTIVE_PROBE_TIMEOUT、MIMICS_IMPORT_CHECKPOINT、NNINTERACTIVE_ASYNC_WORKER_IDLE_TIMEOUT、MIMICS_TRAINING_GPU_MEMORY_GB；5× `*_USE_EVENT_TIMER` → 1 个 `MIMICS_USE_EVENT_TIMER` | — | 均有 config 键同义通道 |
| 部署树 | external/ 未复制大目录的 D 状态、`check_*.py`×4（硬编码网络路径）、`get-pip.py`、`feishu_qr.png`、`external/RONDON_analysis/` | ~55k | 一次性/逆向工程产物，零引用 |

Phase A 合计：**36 文件，-38,262 行 / +28,599 行**（含 test_all.py 大规模裁剪重排）。

---

## 四、测试报告（五类）

### 1. 编译/语法/导入检查
- 全部 runtime_py35 模块：`ast.parse(feature_version=(3,5))` 通过（py3.5 兼容约束）。
- 全部 tools/ 模块：py3.10+ 语法检查通过。
- `TestScriptingLibraryEntries`：每个 scripting_library 入口必须含 `_mimics_entrypoint` + `run_runtime_entry`，全部通过。

### 2. 功能测试
- `tools/test_all.py`：**508 项**（起点 455 项：净增 53 项 = 新增 9 个测试类 − 淘汰功能测试裁剪），全绿。
- 新增测试类：TestProcessRegistry（8）、TestPipelineCommon（6）、TestNNInteractiveContinuousPrompting（5）、TestNNInteractiveGpuMemoryPrecheck（3）、TestNNInteractiveTaskDiagnostics（7）、TestDatasetProfiles（16→8 精简后）、TestImportDropWindow（9）、TestImportReceiptAndUndo（8）、TestSystemHealthPanel（8）。

### 3. 工作流测试
- `tools/fake_mimics_flow_test.py`：fake-mimics 端到端工作流（导入→交互→导出）绿。
- `tools/test_interactive_algorithms.py`：裁剪为 scribble-only 后绿。
- `tools/test_nninteractive_task_integration.py`、`test_mimics_nnint_functional/deep.py`：绿。

### 4. 压力测试
- `tools/offline_stress_test.py`：8/8 绿（锁竞争、长序列、并发）。
- `tools/test_model_portability.py`：绿。

### 5. 回归测试
- 全量 `tools/test_all.py` 每阶段门禁（Phase A 446 → B 460 → C/D/E 递增 → F 508 项）。
- 基线已知差异已消化：`test_mimics_batch_cli_runner_uses_resolved_bridge_python` JSON 转义断言修正；fewshot preflight 测试改用 fewshot_config.json 指向的 dinov3_project 根（external 大目录不随部署树迁移）。
- **发现并修复一个潜伏的全套件级测试污染 bug**：`test_process_exists_treats_permission_denied_as_alive` 用 `del ctypes.get_last_error` "还原" monkey-patch，但 py3.13 中该属性是 ctypes 模块内置属性，`del` 将其永久删除；此后所有 `process_exists` 调用 `AttributeError` 被 `except Exception: return True` 吞掉——**所有 PID 一律判活**。此前无测试受影响（无人在其后依赖"PID 判死"），Phase F 健康面板的 stale-lock/server 判活测试首次暴露。修复为保存/恢复真值。这正是真实测试的价值：一个安静的共用工具污染，直到新消费者出现才现形。

### 等价性证明（P2 硬要求）
- `TestDatasetProfiles`：ts-like 档案下对同一测试数据集的发现结果与改造前 **diff 为空**（8 例覆盖：preferred nifti、mhd/nrrd、dicom 子目录、fallback 扫描、排除目录、多 volume 异常）；`_FALLBACK` 与 JSON 逐字节一致断言。

---

## 五、已知问题与风险

1. **实机 GUI 人工验收未做**（见下节清单）——headless 环境只能覆盖逻辑层。
2. **进程注册表先有鸡先有蛋**：升级前启动的老进程不在注册表里，由 sweep 的 cmdline 兜底路径捕获；运行一段时间后自然全覆盖。注册表半写 JSON → 原子写 + 读失败即忽略，永不阻塞启动。
3. **连续提示循环在 QTimer tick 里连环弹窗**：busy 重入护栏 + fake-mimics 序列测试已覆盖；若实机有消息泵问题，回退方案是 applied 后自动 re-show 一次菜单（仍优于现状）。
4. **两管线合并只动同构外壳**：指纹/缓存内核各自保留，压力测试兜底；若后续发现行为差异，回退点是 git 历史里的独立实现。
5. **E-SafeNet 企业加密**拦截外部工具对 PNG 等文件的读取（本机环境特有）：健康面板与拖放窗的预览图验证改用 QImage 采样校验（尺寸 + 27 种采样色）完成，不影响功能。
6. **check_*.py 一次性脚本已删**：如果将来需要元数据检查，参考 git 历史找回。

---

## 六、待人工验收清单（需在装 Mimics 的机器上执行）

| # | 项目 | 验收要点 |
| --- | --- | --- |
| 1 | 连续提示循环手感 | 一次菜单进入后连续点提示，确认每次 applied 后菜单再弹、无卡顿/重入弹窗叠加；Finish/UNDO/RESET 正常结束 |
| 2 | 拖放导入窗实机表现 | 从资源管理器拖文件/夹到置顶窗，认出摘要正确，Import 后 worker 启动，30 分钟空闲自动退出；Ctrl+V 粘贴路径可用 |
| 3 | 降级导出弹窗 | P1 的多格式/降级导出路径在实机数据上的弹窗与产物 |
| 4 | 导入回执/撤销 | 实机导入后跑 05_Undo_Last_Import：确认弹窗文案、mask 删除、.mcs 未改动时文件回退、改动后保留 |
| 5 | 健康面板 | 06_System_Health 打开、各 section 数据与实际相符、Stop All Owned Services 确认弹窗与效果 |
| 6 | GPU 预检报错 | 占满 VRAM 后启动 nnInteractive server，确认友好报错（非 traceback） |

---

## 七、未完成项 / 后续建议

- 无阻塞未完成项。建议后续：健康面板可在实机验收后按标注员反馈微调信息密度；dataset_profiles 可按新数据集逐个加 profile；`external/dinov3-medical-seg` 仍留在旧路径由 fewshot_config.json 指向，如迁移需同步改配置。
