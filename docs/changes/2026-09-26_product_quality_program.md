# 五点产品质量改进计划交付报告

> 2026-09-25 至 2026-09-26，五点要求全部落地。门禁：test_all.py 全量绿、test_remote_training 81/81、回归矩阵 fast 档通过。
> 前置交付：`2026-09-24_improvement_program_delivery.md`（P1-P8）。

## 五点要求 → 交付对照

| # | 要求 | 交付 | 状态 |
|---|------|------|------|
| 1 | 减少本地配置/路径修改操作 | Phase 2：路径默认记住 + 首次引导图形化 + 机器特定硬编码清除 | ✅ |
| 2 | 测试覆盖所有功能 | Phase 3：收敛冒烟 + 权重可加载校验 + 全流程贯通 + GPU 争用握手 | ✅ |
| 3 | 速度/性能/流畅度/稳定性 | Phase 1：状态查看器异步化 + 目录指纹缓存 + worker scratch 脱离 %TEMP% | ✅ |
| 4 | 远程训练设计完善 | Phase 1b-d：代码漂移检测 + 容器重挂接 + 监控上限 + 传输缓存 | ✅ + 实机验收 |
| 5 | 训练收敛性 + 推理正确性 | Phase 3a 收敛冒烟 + Phase 4 实机 6/6 验收 | ✅ |

## Phase 1 — 稳定性/性能

- **1a 状态查看器异步化**：3 个查看器（nnunet/flexict/batch）2000ms 同步刷新改后台线程收集 + queue + QTimer 排空；GUI 冻结源消除。
- **1b 远程镜像/代码漂移检测**：job 启动前本地 vs 容器内 `/app` 内容指纹比对，strict 拒跑 / warn 记录（同 `remote_weights_verify` 模式）；同镜像 ID 结果缓存。**实机验证**：206 验收 step code_drift 指纹 MATCH（`0245c539…`），秒级。
- **1c 容器重挂接**：orphaned_remote 状态下 "Re-attach" 动作——校验容器 labels + running + 远端 status.json 后重启监控循环；远端已完成未下载场景直接走下载路径。
- **1d 监控健壮化 + 传输效率**：重连上限（默认 30 次后转 `remote_unreachable`）、>24h 日志静默 + 容器 running 时 `remote_stall_suspected` 提示、自定义基础模型 per-sha256 上传缓存、远端模型指纹缓存（1.1GB 免重复哈希）。
- **1e 本地目录指纹缓存**：`path_signature` manifest 缓存（键=路径+顶层 mtime+文件数，7 天重验）。

## Phase 2 — 配置简化

- **2a 路径默认记住**：上次成功提交的 dataset_root/输出路径自动回填（默认开启，保留清除入口）。
- **2b 首次引导图形化**：环境问题检测 → PySide6 引导窗（缺什么/多大下载/一键修复/进度条）；migrate_root 图形化一键迁移；删 `flexict_common.py` 的 `R:/` 硬编码回退。

## Phase 3 — 测试覆盖

- **3a 收敛冒烟**（`test_training_convergence.py`，full 档）：合成 3-case 2-epoch，断言 loss 下降 + checkpoint 可 torch.load + 推理非空 mask。**抓到 3 个真实训练 bug**（trainer KeyError、Windows DA 崩溃、waiting_for_gpu 卡死）。
- **3b 模型包权重校验**：导入时 checkpoint `torch.load(map_location='cpu')` 头部校验，损坏拒绝导入。
- **3c 全流程贯通**：同 dataset train→register→infer→AL 排序→overlay 请求文件，每步交接状态一致。
- **3d GPU 争用握手**：双进程真实锁争用，waiting_for_gpu→training 变迁正确。

## Phase 4 — 远程真实服务器验收（206）

**6/6 PASS**，详见 `2026-09-26_remote_acceptance.md`。要点：

- 全链路实机贯通：漂移检测 → 缓存上传（二次零上传）→ GPU 锁容器训练（1672s）→ 产物回传注册 → 权重校验 → 本地推理（GPU 75s，mask 37551 voxels）。
- **7 个真实 bug 被验收暴露并全部修复**——全部是 mock 测试无法覆盖的实机问题（含 flexict_pipeline 缺 prepared 分支、models[] 容器路径泄漏、规划器 batch_size OOM）。
- 206 运维沉淀：镜像 patch+commit 秒级更新法（不重建，buildkit 缓存被 k3s 回收且联网解析超时）。

## 实机暴露的环境问题（已代码化防护）

| 问题 | 防护 |
|------|------|
| %TEMP% 被 IT 周期清理，>1h 推理中途 scratch 被删 | worker TMP/TEMP 指向 job 目录内（de0a07b） |
| 崩溃 worker 留 ~10GB 孤儿进程 → 后续加载 os error 1455 | 验收/生产推理默认 GPU 路径（7a8d532） |
| CPU 推理 1132 层 ~70 分钟且原生崩溃（0xC0000005）非确定性 | 同上；GPU 75s（56×） |

## 维护提示

- 日常回归：`python_env/python.exe tools/run_regression_matrix.py --profile fast`。
- 远程验收复跑：`python_env/python.exe tools/remote_acceptance_checklist.py --profile <id> --dataset-root <nnU-Net 格式 case 目录>`。
- 206 镜像更新用 patch+commit（见验收报告运维要点），改代码后必须更新镜像再跑远程 job。
