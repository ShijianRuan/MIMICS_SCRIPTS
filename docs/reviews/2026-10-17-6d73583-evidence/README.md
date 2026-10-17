# 6d73583 复审证据

所有当前观察对应源码 `6d735832db9847e98d60fd99eab4f241b21c6d28`。后续报告提交SHA与源码SHA分别记录，旧证据不被覆盖。

## 🔬 边界

review_probes.py 使用真实回执补写、预测发布/恢复、DICOM几何、拆分、FIFO状态持久化与完整AL计算链。原生Mimics、模型执行和GPU预约为边界替代；全为临时合成数据。

ui_scenarios.py 创建真实Qt状态窗口并执行实际恢复回调，只在文件摘要读处加200ms慢盘延迟。3张画面已人工查看；无真实患者、非Windows截图。退出反例只控制线程先后，不实际杀进程。

## ▶️ 复现

使用audit-environment.json记录的隔离依赖，从该源码仓库根目录执行：

```sh
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-17-6d73583-evidence/review_probes.py --out /tmp/review-observations.json
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-17-6d73583-evidence/ui_scenarios.py --out /tmp/review-ui
```

临时合成数据自动清理，不启动模型、训练、Mimics或环境修复。依赖不足须另记，不能把未运行算通过。

## 🗂️ 记录

- [核心观察](observations.json)、[Qt观察](ui-observations.json)、[回归汇总](regression-results.json)。
- [探针日志](probes.log)、[Qt场景日志](ui-scenes.log)、[UI回归原日志](ui.log)、[FlexiCT原日志](flexict.log)。
- suite-exits.json记录9组选择；runtime-changed.log另执行 `tests/test_all.py TestCreateMcsBatch TestEdgeCases TestBridgeDicomLoading`。总共10组选择；执行可有交集。
- [环境](audit-environment.json)、[源码与证据哈希](manifest.json)、[独立事实核对](fact-check.json)。
- [主报告](../2026-10-17-6d73583-revalidation.md)、[结构化记录](../2026-10-17-6d73583-revalidation.json)。

本机路径已在文本日志中替换为占位符，截图路径只对应临时合成工作区。没有把缺torch错误和Windows用例跳过说成通过。
