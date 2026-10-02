# 交付报告 2 —— 四专题深挖（UI / LOG / 进程管理 / 多机可迁移性）

**日期**： 2026-09-21
**分支**： `refactor-overhaul`
**上一轮交付**： 16dd543..b7f590d（DELIVERY_REPORT.md，509 测试绿）
**本轮门禁**： `tools/test_all.py` 513 项全绿 + `test_model_portability.py` 6 项 + `test_migrate_root.py` 13 项 + `test_collect_diagnostics.py` 8 项

---

## 一、变更清单（按专题）

### 专题 1 多机多标注者可迁移性（Phase 1）

**目标**： 部署树拷到新机器/新盘符后零手工改路径。

| 变更 | 文件 | 说明 |
| --- | --- | --- |
| 配置相对化 | fewshot_config.json / nninteractive_config.json / nninteractive_finetune_config.json | `dinov3_project`→`external/dinov3-medical-seg`；`workspace_dir`→`nninteractive_task_models`；`official_model_dir`→`python_env/models/nnInteractive_v1.0`（消除 legacy `nninteractive_env` 名） |
| 内容归位 | external/dinov3-medical-seg/、nninteractive_task_models/ | 从旧家 `E:\mimics_script_offline` 搬入 dinov3 **代码子集**（197 文件，含 `src/data/`，排除 58G data/）与 task 权重（4.7G，35 文件全量）+ registry.json + project_bindings.json；**不搬** 47G cache/（可再生）与 jobs/（历史归档留旧家） |
| setup_offline.bat 重生成 | setup_offline.bat（由 package_portable._generate_offline_bat 生成） | 目标 `python_env`；步骤 5 由 tkinter 安装改为 PySide6 校验（Tk 后端已删） |
| 一键迁移命令 | **tools/migrate_root.py**（新） | `--old-root X [--absolute] [--dry-run] [--reset-runtime]`：改写 6 个根配置绝对路径→根相对；修 `~/.mimics_script/fewshot_model_index.json`（basename 匹配重写）；nnunet 注册表 **flag-not-drop**（死路径行标 `status:"missing_path"` 保留，UI 提示"模型目录缺失——运行 migrate 或重新导入"）；重生成 setup_offline.bat；输出迁移报告 |
| 迁移指南 | **docs/MIGRATION.md**（新） | 迁移步骤 + 兼容性矩阵（registry v2 relpath / manifest path_reference 双记录 / python_env embeddable 整体可拷 / 只依赖 python.exe 不依赖 Scripts\*.exe） |

**修复的测试阻塞**： 24 项 TestNewFeatures + 2 项 bridge preflight 测试原先因 dinov3 项目缺失而红，内容归位后全绿。

### 专题 2 LOG 与缓存治理（Phase 2）

**目标**： 无无限增长；隐私脱敏；一键诊断包。全部沿用现有钩子模式，未建新框架。

| 目标 | 接入点 | 保留策略 |
| --- | --- | --- |
| finetune `jobs/` 无限增长 | `job_retention_days`（原死键）→ `_cleanup_terminal_artifacts` + 管线启动清理；status.json 保留供 Model Center 历史 | 按配置天数（默认 14） |
| `debug_out` 面包屑 | mimics_import `_checkpoint_record` 写入时顺带清理 | 14 天 |
| `drop_import/` 拖放残留（含隐私路径） | append/import 路径清理 | 30 天 |
| `append_jobs/` | append_masks_batch 启动清理 | 14 天 |
| `health_panel_*.log` | external_window_launcher.prune_logs | 最近 10 个 |
| `setup_env.log` | 套用现有 rotate_log | 5MB×3 |
| `import_queues/mcs_output_*` | 活队列保护（active marker mtime < 7 天 **或** 消费者锁存在 → 跳过），挂 clear_all_caches 与启动 sweep | 清 runner/日志/`__pycache__`，目录保留 |
| `%TEMP%\mimics_script_fewshot_logs` 回退日志 | fewshot 管线启动 cleanup | 30 天 |
| finetune `cache/`（47G 级） | 新配置键 `prepared_cache_retention_days`（默认 30） | 过期 bucket 清理 |
| **导入回执**（用户决策） | create_mcs_batch 写回执时按 mtime 清理 | **30 天** |
| 诊断包 | **tools/collect_diagnostics.py**（新）+ **99_Admin/07_Collect_Diagnostics.py** 入口 + runtime_py35/collect_diagnostics_mimics.py | zip 上限 10MB（超限拒发）；各日志尾 2MB；路径只留最后 2 级；case ID 保留（模型诊断需要）；Bearer/api-key 掩码 |

### 专题 3 UI 收敛（Phase 3）

**目标**： UI 永不阻塞 Mimics GUI；删重复；报错说人话。

1. **健康面板自卡修复**（system_health_panel.py）: `collect_health` 与 `do_sweep` 移入 daemon 线程 + result queue + QTimer 轮询（全库既有模式）；sweep 完成改非模态状态行。
2. **拖放窗闭环**（import_drop_window.py）: Browse 改 `_AsyncPathDialog`（进程隔离）；新增 Recent drops 活动列表（轮询 `.mimics_runtime/drop_import` 状态 JSON，逐拖显示 Importing/Done/Failed + 一句话错误，点行开日志）。
3. **异常直达人话 ×5**: 新 `runtime_py35/external_window_launcher.py` 的 `error_guidance()`——python 找不到→"环境损坏，跑 Admin > Setup/Repair"；脚本缺失→"安装不完整，重新解包"。fewshot_mimics（训练设置窗/模型选择器/状态窗）+ nnunet_mimics（训练/状态窗）全部接入。
4. **ScribblePrompt 记住上次选择**（interactive_algorithms_mimics.py）: 每提示 2 次问答→1 次确认（默认上次 plane/sign），配置键 `remember_last_plane_sign`。
5. **删 Tk 后端**（用户决策，-3000 行级）: fewshot_training_setup_ui.py 删整个 Tk `TrainingSetupApp`（~810 行）；fewshot_status_viewer.py 同删；setup_env.py 探测删 tkinter；`MIMICS_DINOV3_GUI_BACKEND` 环境变量全库清除。共享常量上移模块级，Qt 类自持。
6. **启动器合并**: import_drop_mimics.py + system_health_mimics.py → external_window_launcher 参数化核心 + 各 ~20 行薄壳；fewshot_mimics `_launch_gui_process` → runtime_common.launch_external_gui_process（新增 `extra_pythonpath` 参数保持 PYTHONPATH 行为）。
7. **JSON 帮手收敛**: pipeline_common 增 `read_json`/`write_json_best_effort`；fewshot_pipeline、mimics_batch_cli 的 `write_json_atomic` 变委托；io_path_setup_ui 的 ctypes `process_exists` 副本 → import resource_locks（顺带修复旧副本漏判 WAIT_OBJECT_0 / ERROR_INVALID_PARAMETER 的缺陷）。

### 专题 4 进程管理补缺（Phase 4）

1. **nnU-Net trainer 进注册表**（tools/nnunet_pipeline.py）: Popen 处 register_process（role trainer，cancel_path=state_path），退出 unregister——健康面板/kill-background 全覆盖。
2. **官方路径源缓存生成移出 GUI 线程**（nninteractive_mimics.py）: 官方模型路径统一设 `defer_source_alignment_to_worker`（对齐 task_model 路径既有机制），`prepare_source_fastpath` 移入 async worker——消除最长 1800s 的 GUI 线程同步执行。
3. **删死代码**: `mimics_import.call_bridge`（零调用，communicate(timeout=600)）删除 + 回归测试断言防复活。
4. **调度模型文档化**（用户决策：保持 FIFO + 单 GPU 锁）: docs/MIMICS_PROJECT_ARCHITECTURE_CN.md 新增"任务调度与并发模型"一节——import 队列/导出 job/GPU 锁三域队列图、交互快速通道、不加优先级调度的理由。

---

## 二、删除清单

| 删除项 | 位置 | 理由 |
| --- | --- | --- |
| Tk `TrainingSetupApp` 类（~810 行） | tools/fewshot_training_setup_ui.py | 用户决策：Tk 后端删除，PySide6 在 python_env 恒在 |
| Tk 状态窗实现 | tools/fewshot_status_viewer.py | 同上 |
| tkinter 探测与回退分支 | tools/setup_env.py | Tk 已删，探测无意义 |
| `MIMICS_DINOV3_GUI_BACKEND` 环境变量 | 全库（原 fewshot_training_setup_ui / fewshot_status_viewer / 文档） | 后端选择逻辑随 Tk 删除 |
| `mimics_import.call_bridge` | runtime_py35/mimics_import.py | 零调用死代码（审计核实；communicate(timeout=600) 有 GUI 阻塞风险） |
| 双胞胎启动器主体 | runtime_py35/import_drop_mimics.py、system_health_mimics.py | 合并进 external_window_launcher，各留 ~20 行薄壳 |
| ctypes `process_exists` 副本 | tools/io_path_setup_ui.py | 委托 resource_locks（且旧副本有漏判缺陷） |
| 6 处 `write_json_atomic`/`read_json` 副本 | tools/fewshot_pipeline.py、tools/mimics_batch_cli.py | 收敛到 pipeline_common canonical 实现 |

**残留扫描（零命中验证）**: `MIMICS_DINOV3_GUI_BACKEND` / `import tkinter` / `TrainingSetupApp`（非 Qt）/ `call_bridge`（mimics_import）在 tools/、runtime_py35/、docs/ 均零命中。

---

## 三、测试报告

| 套件 | 数量 | 结果 |
| --- | --- | --- |
| tools/test_all.py 全量 | 513 | **全绿** |
| tools/test_model_portability.py | 6 | 全绿 |
| tools/test_migrate_root.py（新） | 13 | 全绿 |
| tools/test_collect_diagnostics.py（新） | 8 | 全绿（含脱敏断言、超限拒发、活队列不清） |

关键新增/改写测试：
- **test_migrate_root.py**: 旧根配置改写断言、fewshot index basename 修复、nnunet missing_path 标注、setup_offline.bat 重生成、--reset-runtime。
- **test_collect_diagnostics.py**: 患者路径脱敏（留最后 2 级）、Bearer/api-key 掩码、case ID 保留、超限拒发（BUNDLE_MAX_BYTES patch，确定性）、health_panel 日志入包且脱敏。
- **TestLifecycleAndRetention（32 项）**: 每条保留钩子的伪造 mtime prune 测试——含"活队列不清"（active marker 新鲜 → 跳过）。
- **TestNewFeatures 改写项**: Tk 测试全部重定向到 `ui.EPOCH_CHOICES` 模块级常量与 `QtTrainingSetupApp`（offscreen QApplication + object.__new__ 模式）；`test_health_panel_entry_exists` 扩展断言 external_window_launcher + 双薄壳引用。
- **回归防复活**: mimics_import.call_bridge 删除断言（hasattr 检查）。
- **Phase 4**: nnunet trainer 注册（fake Popen）、官方路径 defer 断言 ×2。

**测试基建修正**: test_collect_diagnostics 超限测试原用随机 hex 试图制造不可压缩内容，但 hex 可 2:1 压缩导致断言依赖 zlib 版本——改为 patch `BUNDLE_MAX_BYTES=1`，确定性通过。

## 四、已解决痛点对照（每条附验证）

| 痛点 | 验证 |
| --- | --- |
| 配置写死旧机绝对路径，换机必坏 | migrate_root dry-run/实跑测试 13 项；本机实跑 `--old-root E:/mimics_script_offline` 成功（bat 重生成、零 nninteractive_env 引用） |
| 24 项测试因 dinov3 项目缺失而红 | TestNewFeatures 215 项全绿 |
| 日志/缓存 9 处无限增长 | TestLifecycleAndRetention 32 项 prune 测试 |
| 隐私路径泄漏进日志/回执 | collect_diagnostics 脱敏断言 ×5 |
| 健康面板扫描卡死 GUI | TestSystemHealthPanel 8 项（线程化断言） |
| 拖放窗无反馈闭环 | TestImportDropWindow 11 项（Recent drops 状态机） |
| 原始异常（WinError 2 等）直达标注者 | error_guidance 结构断言（TestNewFeatures） |
| ScribblePrompt 每提示 2 次问答 | interactive_algorithms remember_last_plane_sign 测试 |
| nnU-Net trainer 不在注册表（kill 不达） | TestProcessRegistry 8 项 + fake Popen 注册测试 |
| 官方路径源缓存生成卡 Mimics GUI 1800s | defer_source_alignment_to_worker 断言 ×2 |
| 3 处 stale 配置指向已不存在目录 | 配置值断言（TestNewFeatures 配置解析测试） |

## 五、已知问题与残留

1. **大数据集仍在旧家**: dinov3 58G `data/` 与实验产物未搬（按设计——测试用 preflight 验证只需代码+模型；需要时用户可在配置中改绝对路径指向大数据位置）。ts_root 数据集根如迁移需另行拷贝（migrate_root 报告中列为手动步骤）。
2. **nninteractive cache/（47G）留旧家**: 可再生；新机首次推理会重建（首次较慢）。
3. **GUI 人工验收待办**（同上轮）: 健康面板/拖放窗/训练设置窗的真人操作验收仍待标注者执行。
4. **robocopy 平台注意**: 本轮实际用 Python copytree 落地（robocopy 在自动化环境不可用）；手工迁移时 Windows 上 robocopy 仍是首选。

## 六、本轮关键决策记录

- **registry flag-not-drop**（而非删行）: 模型行含训练历史元数据，路径缺失不代表模型作废——标注者跑 migrate 或重新导入即可恢复，静默删除不可逆。
- **import_queues 活判据用合取**: active marker 新鲜度（< 7 天）**或**消费者锁存在即视为活——宁可漏清不可误伤正在消费的队列。
- **超限诊断包拒发而非截断**: 截断会静默丢内容误导诊断；明确报错让用户先清日志。
- **hex 压缩测试改 patch 法**: 测试不应依赖 zlib 压缩率的实现细节。
