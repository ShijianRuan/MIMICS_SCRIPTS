# 远程真实服务器验收报告（Phase 4）

> 服务器：10.8.168.206（2× RTX A4000 16GB，nerdctl 2.2.1，k3s 管理的 containerd，镜像命名空间 `mimics-ai`）
> 验收窗口：2026-09-25 至 2026-09-26；数据：Totalsegmentator 肾 kidney_left，8 case（7 训练 + 1 留出），只读复制自源路径
> 执行：`tools/remote_acceptance_checklist.py --profile 206_root --dataset-root ... --output D:\mimics_acceptance_run9`

## 最终结果：6/6 PASS

| # | 步骤 | 结果 | 耗时 |
|---|------|------|------|
| 1 | preflight（SSH/镜像/GPU/远程根布局） | ✅ | 2.0s |
| 2 | code_drift（镜像内管线代码 = 本 checkout，指纹 `0245c539…`） | ✅ | 1.6s |
| 3 | dataset（8 case 就位） | ✅ | 0s |
| 4 | train_remote（FlexiCT 2D 2-epoch，`--network none` 容器 + GPU 锁） | ✅ | 1672s |
| 5 | verify_weights（checkpoint torch.load 通过） | ✅ | 45s |
| 6 | local_inference（注册模型本地推理留出例，mask 37551 voxels） | ✅ | 131s（推理本体 75s GPU） |

产物报告：`D:\mimics_acceptance_run9\acceptance\20260925T231249_remote_acceptance.json`（local_inference 条目已按 2026-09-26 复验结果更新并注明）。

## 验收过程中发现并修复的 7 个真实 bug

程序化验收的价值直接体现在这里——7 个 bug 全是 mock 测试无法覆盖的实机问题：

| # | bug | 修复 |
|---|-----|------|
| 1 | 验收脚本直调 `fp.run_training()` = 本地 worker 路径（在本机 3060 上训练而非远程） | 改走 `fp.create_flexict_job()` 生产分发 |
| 2 | `_status_update(..., **code_identity)` 重复关键字崩溃 | dict merge + setdefault |
| 3 | nerdctl 把 warning 行打到 stdout，`_container_labels` JSON 解析失败 | 取最后一个 `{...}` 行 |
| 4 | `log_sync_state` 在 run() 失败路径未绑定 → NameError | `_monitor_remote` 前初始化回退 dict |
| 5 | **`flexict_pipeline.run_training` 缺 `label_source=="prepared"` 分支**（nnunet_pipeline 有）——远端容器按别名找标注全 404 | 补齐分支，直接消费 `prepared_cases` |
| 6 | FlexiCT 2D 注意力模型用规划器默认 batch_size（bs=37）在 16GB A4000 OOM | 程序化调用显式给小 batch_size（验收用 bs=4） |
| 7 | 控制器下载产物后 status.json 的 `models[]` 仍是远端容器路径 `/job/...`，下游推理找不到模型 | 注册后用本地 manifest 重写 models[] |

另有两个实机环境问题（非代码 bug）在本机推理阶段暴露并修复：
- **%TEMP% 被 IT 周期清理**：>1h 的 CPU 推理中途 scratch 被删，导出步骤 FileNotFoundError → worker TMP/TEMP 改指向 job 目录内（commit de0a07b）。
- **推理默认走 GPU**：1132 层 CT CPU 需 ~70 分钟且在内存紧张工作站上不可靠（原生 0xC0000005 ×2、os error 1455 提交内存耗尽），GPU 75 秒完成、56× 加速（commit 7a8d532）。

## 206 服务器运维要点（沉淀给后续使用者）

- **镜像更新不要重建**：buildkit pip 层缓存被 k3s 回收（~70min/次），且 buildkit 尝试联网解析 base 镜像 metadata（dockerproxy.net 超时直接失败）。用 **container patch + commit**（秒级）：
  `nerdctl -n mimics-ai run -d --name img-patch --entrypoint sleep mimics-ai-runtime:1.0 infinity` → `nerdctl cp <file> img-patch:/app/<rel>` → `nerdctl commit img-patch mimics-ai-runtime:1.0` → `nerdctl rm -f img-patch`，以代码指纹 MATCH 验证。
- FlexiCT checkpoint 每个约 2GB（best+final 各一），单模型下载约 8GB；共享节点磁盘约束下远端 job 用完即删。
- 远端数据上传走 per-case 内容寻址缓存（`cache/<user>/datasets/<sha256>/`），同数据二次验收零上传。

## 结论

五点产品质量改进计划的远程训练链路（Phase 4）**验收通过**：镜像漂移检测、数据上传输缓存、GPU 锁容器化训练、产物回传注册、权重校验、本地推理全链路在真实服务器上贯通。所有修复已提交（`4e23062`..`7a8d532`），门禁全绿（test_all.py exit 0，test_remote_training 81/81）。
