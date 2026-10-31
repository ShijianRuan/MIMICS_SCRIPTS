# 2026-10-31 修复与外部模型调查证据

产品源码：`33e738fca65526156791e03b5acf999f9d9928b3`。各文件摘要及 Python 3.5/3.13 语法检查记录见 [source-files.json](source-files.json)。[主报告](../2026-10-31-nninteractive-fixes-external-nnunet.md)与[外部模型设计](../2026-10-31-nnunet-external-model-design.md)分别区分已实施修改与待实施能力。

| 文件 | 验证内容与边界 |
|---|---|
| [regression-matrix.json](regression-matrix.json)、[日志](regression-matrix.log) | 最终源码的 17 组明确选择的离线回归；包含 3 项 Windows GUI 专属跳过。不是完整 fast/full |
| [targeted-tests.log](targeted-tests.log) | 155 项 nnInteractive/源图映射/生命周期/GUI 主线程测试；日志保留实际运行时的未发布源码身份；相关文件在最终提交中逐字节相同，见验证范围 |
| [validation-scope.json](validation-scope.json) | 源码身份、实际环境和未执行范围；不重标旧日志为新的提交 |
| [capture_ui.py](capture_ui.py)、[截图日志](qt-capture.log)、[测量](qt-observations.json) | 真实 macOS PySide6 离屏控件，使用合成状态/注册表及依赖替身；没有执行真实训练或推理 |
| [完成提示截图](test_completion_long_copy_wraps_within_viewport.png)、[长 ID 截图](test_task_model_chooser_full_id_readable_and_copyable.png) | 853×440 控件渲染，人工检查正文可见及按钮/进度条无重叠；不能替代 Windows DPI/焦点验收 |
| [worker-lifecycle.mmd](worker-lifecycle.mmd) | 已实施保护流程的可编辑 Mermaid 源码 |
| [upstream-sources.json](upstream-sources.json) | 官方 nnU-Net 调研时主线 SHA、日期、固定 URL 和内容摘要；没有下载模型 |
| [fact-check.json](fact-check.json)、[verification.json](verification.json) | 另行核对结论、文档链接及提交/证据一致性 |
| [evidence-sha256.json](evidence-sha256.json) | 证据摘要清单，排除清单自身 |

新增合成回归的可执行源位于 [test_worker_cache_safety.py](../../../tests/test_worker_cache_safety.py)，模型包受限加载反例位于 [test_model_portability.py](../../../tests/test_model_portability.py)。它们使用临时目录、小型数组、实际文件租约、数值 checkpoint 与模拟进程身份；不会运行模型、训练或处理病例。进程终止由替身拦截。

复跑新增安全回归：

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python tests/test_worker_cache_safety.py
```

Qt 截图复跑：

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python docs/reviews/2026-10-31-fixes-evidence/capture_ui.py
```

脚本会覆盖本目录中的当前源码截图，因此对未来源码运行时应另存到新的证据批次，保留这批旧证据身份。前轮反例保留在 `2026-10-31-09d5948-evidence`，不能在变更后的源码上重跑并覆盖成“同一批结果”。
