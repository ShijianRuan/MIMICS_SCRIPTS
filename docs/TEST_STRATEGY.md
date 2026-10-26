# 测试策略（Test Strategy）

> 一句话：**改什么 → 跑哪层 → 谁来跑**。所有离线层可一条命令跑完
> （`tools/run_regression_matrix.py`）；GPU 层为人工验收项，按清单执行。

## 分层总览

| 层 | 内容 | 规模 | 触发时机 | 执行者 |
|----|------|------|----------|--------|
| L1 单元/契约 | `tests/test_all.py`（410+ 项，含全部 Py3.5 语法、entrypoint、错误引导、配置） | ~16 min | 提交前全量；改动小可先跑目标类 | 开发 |
| L2 集成套件 | 18 个 `tests/test_*.py` + `integrations/flexict-finetune/tests/test_flexict_pkg.py`（各框架独立套件：flexict/nnunet/nninteractive/SLIP/remote/算法/可移植性/诊断/迁移/UI 偏好） | ~10 min | 每次提交 | 开发 |
| L3 冒烟流（fake mimics） | `tests/fake_mimics_flow_test.py`（imports/append/entrypoint/window/export/nninteractive/taskmodels/stop 8 组，27 个 runtime 模块导入 + 6 类记录聚合） | ~1 min | 改 runtime_py35/ 或桥接层后立刻跑 | 开发 |
| L4 离线压测 | `tests/offline_stress_test.py`（资源锁竞争、取消、坐标映射、增量重放） | ~1 min | 改锁/并发/映射代码后 | 开发 |
| L5 GPU 人工验收 | 训练冒烟、pair→AL→overlay 端到端、AI Model Manager GUI、远程 FlexiCT 实机、SLIP/nnInteractive 实机 | 按清单 | 发版前 | 标注者/开发 |

## 一键矩阵

```bash
# 冒烟（~1 min）：改完 runtime_py35 / bridge 最快的有效门禁
python_env/python.exe tools/run_regression_matrix.py --profile smoke

# 快速（~10 min）：全部集成套件 + 冒烟 + 压测（不含 test_all）
python_env/python.exe tools/run_regression_matrix.py --profile fast

# 全量（~30 min）：快速 + test_all.py 全量
python_env/python.exe tools/run_regression_matrix.py --profile full

# 单独跑某几个套件
python_env/python.exe tools/run_regression_matrix.py --only flow_export nnunet_integration

# 列出全部套件
python_env/python.exe tools/run_regression_matrix.py --list
```

结果工件（JSON，含每套件命令/耗时/退出码/输出尾部）写入
`.mimics_runtime/regression/<时间戳>_<profile>.json`——红灯跑可直接附给 bug 报告。

## 各层覆盖什么、不覆盖什么

**L1 test_all.py**（`python_env/python.exe tests/test_all.py`）
- 覆盖：全部 tools/ 业务逻辑、Py3.5 兼容（runtime_py35 全部文件的语法
  walk）、scripting_library 入口路由、错误→动作引导文案、窗宽窗位、
  导入/导出核心、交互算法、打包清单完整性。
- 不覆盖：真实 Mimics API（用 fake mock）、真实 GPU、GUI 事件循环
  （PySide6 窗口只做构造断言）。

**L2 集成套件**（每套独立可跑，退出码即门禁）
- `test_flexict_integration.py` — FlexiCT job 生命周期、trainer env 契约、
  远程执行 13 项契约
- `test_flexict_common.py` / `test_flexict_pkg.py`（独立仓）— 配置/注册表/
  backbone/解码器/权重加载/不确定度
- `test_nnunet_integration.py` — nnU-Net 训练/推理管线、job 记录、
  主动学习失败路径（<2 折拒绝、空池、全例网格剔除、排名不一致、
  use_folds 透传）
- `test_nninteractive_bridge_prompts.py`（pytest 风格）— 桥接 prompt 顺序契约
- `test_nninteractive_task_integration.py` / `test_mimics_nnint_deep.py` /
  `test_mimics_nnint_functional.py` — nnInteractive 三套：任务路由/边界/功能
- `test_remote_training.py` — SSH/Docker 远程训练契约（不连真实服务器）
- `test_slip_mimics.py` — SLIP Mimics 侧会话失败路径（54 项）：影像校验
  （int8 拒绝、CT 缺 HU 映射拒绝、MR/其他模态接受及其强度导出——
  MR 映射/直通、CT HU 逆变换）、mask 缓冲区/gpu.lock 拒绝、异步结果
  状态机（waiting/applied/restart/ended 全分支，含 worker 死亡、半写
  结果、身份不匹配、手动改 mask、空预测拒绝后撤销计数同步）、命令入队、
  会话关闭清理（空草稿删除）、监控超时、
  会话目标选择（自动草稿/多选拒绝/写模式对话/与 nnInteractive 互斥）
- `test_slip_bridge_reconcile.py` — SLIP worker 侧发布失败调和（结果写盘
  失败后同一提示重试只补发布不重复推理、不同提示先回滚孤儿、撤销失败后
  重放/补发布、推理中途异常毒化会话拒绝后续命令）；fake 模块直测
  `_SlipSession`，无需 GPU，已入 smoke/fast
- `test_interactive_algorithms.py`、`test_model_portability.py`、
  `test_collect_diagnostics.py`、`test_geometry_manifest_regressions.py`、
  `test_migrate_root.py`、`test_ui_preferences.py`、
  `test_ui_quality.py`（UI 非人工质量门禁：主题调色板完整性、
  WCAG AA 对比度、offscreen 真渲染 + CJK 字形验证）
- `test_review_workflow_regressions.py` — 使用文档接口的严格 Mimics 容器桩，
  验证撤销确认/删除、所选 Image 导出来源、批量任务身份、远程前置校验和
  AL 部分失败回滚；已加入 fast/full。它不能替代真实 Mimics API 验收。

**L3 fake mimics 流测试**
- 覆盖：runtime 模块在 fake `mimics` 命名空间下的导入（27 个模块——
  语法回归在这里最先暴露）、Scripting Library 共享入口、导出 buffer 流、
  外部 I/O 窗口路由、nnInteractive buffer/draft 流、stop 锁清理、
  批次状态面板 6 类记录聚合。
- 不覆盖：真实 Mimics 事件、真实进程树。

**L5 GPU 人工验收清单**（见 `docs/flexict_mimics.md` §验收 与
`docs/windows_end_to_end_acceptance_2026-08-01.md`）

| 项 | 命令/操作 | 通过标准 |
|----|-----------|----------|
| 训练冒烟 | FlexiCT 训练窗，8-case 肾 2D，NUM_EPOCHS=2 | job completed，loss 下降，产物注册 |
| 端到端闭环 | pair 训练 → AL 一轮 → Mimics 内 overlay 双带 mask | 不确定度带出现在 Mimics，mask 可保存 |
| AI Model Manager | 导入三类模型包各一，切换默认，回滚 | 默认模型切换生效，失效模型可清理 |
| 远程 FlexiCT | 选远程 profile 训练 1 例 + 推理 | 产物带远程 provenance 下载注册 |
| nnInteractive 实机 | 官方模型 + 自定义模型各跑 1 例交互 | 预测 mask 应用无仿射告警 |
| SLIP 实机 | `python_env/python.exe tests/manual/run_slip_bridge_gpu_acceptance.py`（Totalsegmentator s0000，需 slip_models 权重） | 16 项契约全过（单发+异步、predict/undo/reset、错误路径、close、gpu.lock 互斥与释放），dice 在 0.10–0.45 区间。模型本身不限模态（论文 Table 4），但实机验收目前仅在 CT 上执行过；MR 等其他模态的实机验收待补 |
| SLIP 分发包人工检查（R91-5/SL05） | 打包标准与 with-env 合成包各一并解压 | 包内含根级 slip_bridge.py；完整 SLIP 发布模式含可校验的 slip_models 资源；缺件时解压预检明示缺失项而非报发布成功 |
| SLIP 无 GPU 机器拒绝路径（R91-6/SL06） | 在无 CUDA 的机器上跑 SLIP/01_Annotate 入口 | 在创建草稿/导出缓冲之前被明确拒绝并提示需要 CUDA GPU，不产生任何半途产物 |
| nnU-Net AL 端到端（R93-1） | nnU-Net 训练窗对 Totalsegmentator 只读小子集逐折训练 2 折 → `02_AI/nnUNet/05_Active_Learning_Review` 排序未标注池 → 复核窗口双击开病例/叠加不确定度带/标记状态 | ranking.csv 排序合理；不确定度双带出现在 Mimics；annotation_state.json 状态回流；全程不写源数据集路径 |
| 分发工程预测（D20/R95-4） | 管线创建的 .mcs 复制到脱离数据集的目录（源路径不可达）→ nnU-Net 与 FlexiCT 预测各跑 1 例 | 预测窗口提示体素模式（仅本机）；预测完成 mask 叠加回工程；网格恒等快路无重采样损失；job 目录 input/ 内含 project_image.raw |

## 资源清单（运行测试需要什么）

| 层 | Python | GPU | 磁盘 | 网络 | 数据/权重 |
|----|--------|-----|------|------|-----------|
| L1–L4 全部 | 仅 `python_env/`（torch 2.6.0+cu124 已含） | 不需要 | <2 GB 临时 | 不需要 | 不需要（全部合成数据/fixture） |
| L5 训练冒烟 | `python_env/` | 12 GB（RTX 3060 级即可，2D） | ~5 GB | 不需要 | 肾 8-case 标注 + flexict_2d 预训练权重（576 MB） |
| L5 端到端 | `python_env/` | 12 GB（2D）/ 24+ GB（3D pair） | ~15 GB | 不需要 | 同上 + flexict_3d 权重（共 1.1 GB） |
| L5 远程 | 本地不需要 GPU | 远程 Docker 容器 | 远程 ~10 GB | SSH 到 GPU 服务器 | 服务器端 /models/flexict 双 backbone |

**内存注意**：L2 中的 `flexict_pkg`（权重加载测试）需要 ~1.5 GB 可用内存
提交额（pagefile + RAM）；若机器同时跑大内存任务（提交额度 >95%），
safetensors mmap 会以 Windows 错误 1455（页面文件太小）或段错误失败——
这是环境限制，不是代码回归，等内存空闲后重跑即可。

预训练权重位置：`integrations/flexict-finetune/weights/flexict_{2d,3d}/model.safetensors`
（不入 git；setup 时拷入或用 AI Model Manager 导入）。

## 变更→门禁速查

| 改了什么 | 至少跑 |
|----------|--------|
| `runtime_py35/**`、`scripting_library/**` | smoke 矩阵（或全量 flow）+ test_all |
| `tools/flexict_*` | fast 矩阵（flexict_integration + common） |
| `tools/nnunet_*` / `tools/nninteractive_*` | fast 矩阵对应套件 + flow_nninteractive |
| `tools/mimics_*` / `runtime_py35/mimics_import|mimics_export` | flow_imports + flow_export + test_all |
| `tools/remote_*` / 容器定义 | remote_training + flexict_integration（远程契约） |
| 打包/部署（package_portable、setup_env） | full 矩阵 + 便携包人工解压检查 |
| 文档 | 无门禁（链接检查可选） |

## 已知约定

- `test_all.py` 单跑某类需用 importlib 加载（`-m unittest tools.test_all.类`
  因路径问题不可用）——用矩阵的 `--only` 或直接全量。
- fake flow 测试的 `--only` 组与矩阵套件名一一对应（`flow_*`）。
- GPU 套件刻意不存在于自动化矩阵：任何需要 GPU 的验证都是 L5 人工项，
  避免 CI 环境隐性依赖。
