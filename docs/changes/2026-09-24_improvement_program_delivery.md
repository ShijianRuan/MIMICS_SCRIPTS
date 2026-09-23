# 六点改进计划交付报告

> 分支 `refactor-overhaul`，2026-09-20 至 2026-09-24。
> 每点一个独立 commit，全部离线门禁绿；GPU 相关项列入人工验收清单。

## 总览

| # | 主题 | Commit | 规模 | 状态 |
|---|------|--------|------|------|
| P1 | 运维日志/诊断包覆盖面修复 | `1c6e77f` | 2 文件 +314/-41 | ✅ 完成 |
| P2 | 死代码清理 + 设置 GUI 化（窗宽窗位编辑器） | `1fc0778` | 6 文件 +392/-2 | ✅ 完成 |
| P3 | 配置编辑器 GUI | `70d2352` | 6 文件 +507/-2 | ✅ 完成 |
| P4 | 三类模型统一管理（AI Model Manager） | `1996110` | 8 文件 +1409/-14 | ✅ 完成 |
| P5 | 远程训练 v2（SSH/Docker，全框架统一） | `bb8f73d` | 15 文件 +1031/-45 | ✅ 完成 |
| P6 | 标注者痛点（错误引导/批次面板/skip 提醒） | `97e5dde` | 10 文件 +990/-15 | ✅ 完成 |
| P7 | 测试体系（回归矩阵+策略文档+冒烟扩展） | `744ff09` | 4 文件 +439/-3 | ✅ 完成 |
| P8 | 入口收敛+文档收尾+交付报告 | 本 commit | — | ✅ 完成 |

## 各点交付明细

### P1 诊断包覆盖面（`1c6e77f`）
- 修复诊断包未命中的真实日志：按实际文件名收集 setup_env/mimics_export/
  nninteractive 双日志尾 + 健康面板/drop 窗口/批次状态/外部窗 stderr 最新 2 份。
- job 目录快照：5 类 job 根（flexict/nnunet/nninteractive tasks/export/import）
  按 status.json mtime 排序，活跃任务 + 失败任务各取最新 5 个入包。
- `--focus-latest-failure`：失败排查模式，只收最近 1 个失败 job + 其日志。
- 脱敏保持不变（路径留末两级 / Bearer、api-key 掩码 / case ID 保留）。

### P2 死代码清理 + 窗宽窗位编辑器（`1fc0778`）
- 删根目录 4 个 check_*.py（硬编码病人路径的一次性排查脚本）、
  dinov3-medical-seg 下 8 个 diag_*.py、mimics_export 528 行同步模式。
- 窗宽窗位预设编辑器 GUI（`window_level_editor_mimics` + `06_Window_Edit_Presets`）：
  手输 W/L、增删预设、改关键词，替代手改 JSON。

### P3 配置编辑器（`70d2352`）
- `config_editor_mimics` + `08_Edit_Configs`：图形化编辑 5 个用户配置
  （nninteractive / finetune / io / flexict / interactive_algorithms），
  带类型校验与写入前备份，用户不再需要手改 JSON。

### P4 AI Model Manager（`1996110`）
- `model_manager_mimics` + `09_Manage_AI_Models`：三类模型（nnInteractive /
  FlexiCT / nnU-Net）统一入口——模型包一键导入、默认模型切换、失效模型清理、
  回滚。此前三类模型散在三处各自管理。

### P5 远程训练 v2（`bb8f73d`）
- 推倒重做（不纠结旧版）：`remote_compute`（servers.json + known_hosts +
  Credential Manager 密码）+ `remote_training_controller` + 容器化 worker。
- 三框架统一：FlexiCT 训练/推理、nnInteractive 微调、nnU-Net 训练/批量推理
  都走同一条 SSH/Docker 路径；训练窗 Compute tab 选本地/远程，体验一致。
- 安全约束全部落地：密码永不落盘明文（Credential Manager）、SSH 指纹首连
  确认/变更拒绝、容器 `--network none` + HF offline。

### P6 标注者痛点（`97e5dde`）
- **错误→动作映射**：import/export 每个失败对话框都带「Suggested action」
  （环境坏了→修复环境；磁盘满→清理；网络盘断→重连；源数据无效→看诊断文件；
  背景 Mimics 起不来→Stop All Owned Services 后重试）。
- **批次状态面板**（`09_Show_Batch_Status`）：6 类 on-disk 记录（import runs/
  队列/export jobs/前台任务/append/drop import）一个窗口看全，2 秒刷新，
  一键打开任务文件夹。
- **skip-existing 主动提醒**：选 Skip 策略但目标已有标注时弹三选一
  （继续 Skip / 改 Overwrite / 返回），不再静默跳过。

### P7 测试体系（`744ff09`）
- `tools/run_regression_matrix.py`：25 套件一命令矩阵，smoke(~1min)/fast(~10min)/
  full(~30min) 三档，JSON 工件含每套件命令/耗时/退出码/输出尾。
- `docs/TEST_STRATEGY.md`：五层测试策略 + 改动→门禁速查 + 资源清单。
- fake-mimics 流测试：runtime 模块导入覆盖 11→27（全部 runtime_py35 文件），
  新增批次状态聚合流测试；修复 stop-background 过时契约。
- 门禁：test_all 411 项 OK。

### P8 入口收敛 + 文档收尾（本 commit）
- **入口审计落地方式**：菜单文件不动（避免破坏使用习惯），以
  `docs/mimics_entry_guide.md` 为导航层收口——补齐缺失入口（Quick Export /
  Quick Drop Import / ScribblePrompt / nnInteractive 三入口细化 / FlexiCT 与
  nnU-Net 远程能力标注），每个入口写清前置条件；nnU-Net 03（查看）标注
  「可停止任务」与 04 的分工；窗宽窗位入口标注行为已收敛点（choose 内含
  reset、auto 无匹配回退 choose）。
- **CONFIG_REFERENCE.md 补齐**：`interactive_algorithms_config.json`（9 键）、
  `window_level_presets.json`（结构+编辑入口）、远程计算配置 `servers.json`
  （存储位置/安全模型/字段表）、训练 worker env 变量契约
  （nnUNet_* / FLEXICT* / MIMICS_REMOTE_CONFIG_DIR 等 12 个）。
- 本交付报告。

## 测试结果

| 门禁 | 结果 |
|------|------|
| test_all.py 全量 | **411 tests OK**（~19 min） |
| 回归矩阵 fast（23/24 套件） | 通过；`flexict_pkg` 因机器内存提交额度 >95%（用户任务占用 ~20GB）safetensors mmap 失败（Windows error 1455），属环境限制非代码回归，已写入 TEST_STRATEGY.md 资源清单 |
| fake-mimics 流测试全量 | 13/13 通过 |
| fake_mimics_flow_test --only stop（修复后） | 通过 |

## 遗留人工验收清单（GPU/实机依赖，无法离线自动化）

| 项 | 操作 | 通过标准 |
|----|------|----------|
| FlexiCT 训练冒烟 | 训练窗：8-case 肾 2D，NUM_EPOCHS=2 | job completed、loss 下降、模型注册 |
| 端到端主动学习闭环 | pair 训练 → AL 一轮 → Mimics overlay | 不确定度双带 mask 可见可保存 |
| AI Model Manager GUI | 三类模型包各导入一次，切换默认/回滚 | 切换生效、失效模型可清理 |
| 远程 FlexiCT 实机 | 远程 profile 训练 1 例 + 推理 | 产物带 remote provenance 注册 |
| nnInteractive 实机 | 官方 + 自定义模型各 1 例 | 预测 mask 应用无仿射告警 |
| 批次状态面板 GUI | 混合导入/导出后打开 | 6 类记录可见、文件夹按钮可用 |

## 维护提示

- 日常回归：`python_env/python.exe tools/run_regression_matrix.py --profile fast`；
  改 runtime_py35 后至少跑 `--profile smoke`。
- 测试策略与资源清单：`docs/TEST_STRATEGY.md`。
- 配置速查：`CONFIG_REFERENCE.md`；入口速查：`docs/mimics_entry_guide.md`。
