# eee725b 审查证据说明

所有产品结论固定到 `eee725b45ec42b7c045def95e4682089a3ea13a2`，基线为 `e33dc74b93cb57b83f9fc863dc6da018a9cecafd`。[主报告](../2026-10-22-eee725b-revalidation.md)与[问题卡](../2026-10-22-eee725b-revalidation.json)区分实际合成复现和源码风险。

## 内容

- `source-scope.json`：新增提交、文件与来源限定；`upstream-comparison.json`：固定官方 SHA 的 15 个文件摘要，换行差异单列。
- `regression_runner.py`、`regression-results.json`、各套件 `*-result.json`/`.log`：隔离离线回归，265 通过、1 失败、3 跳过，共 269 方法执行，存在选择重叠。
- `slip_probes.py`、`observations.json`、`probes.log`：真实消费者/桥接边界、锁、小型实际 zip、临时恢复目录与提取的上游坐标方法。Mimics/模型边界替代，不是模型或病例端到端验证。
- `ui_probe.py`、`ui-observations.json`、`ui.log`、`setup-progress-new-check.png`：真实离屏 Qt 进度窗口及 GUI 线程慢读取；状态是合成的，没有运行修复。
- `regression-diagnostics.json`：POSIX 夹具失败与 detector 未变的边界；`probe-harness-initial.log`：首次 AST 脚本包装错误，修正后再运行成功，不计入产品缺陷。
- `slip-session-contract.mmd`：可编辑修复建议图；`manifest.json`：源码和证据摘要；`verify_evidence.py`/`fact-check.json`：生成后的独立静态核对。

## 安全复现

在固定源码的独立检出与已有依赖环境中运行；脚本会写本目录结果，需先复制证据以保留原记录。源码已变化时停止，不能把新输出覆盖旧 SHA。当前使用现有 `/Users/ruanshijian/miniconda3/bin/python3`，不自动安装或修改依赖。

```sh
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 /Users/ruanshijian/miniconda3/bin/python3 docs/reviews/2026-10-22-eee725b-evidence/regression_runner.py
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 /Users/ruanshijian/miniconda3/bin/python3 docs/reviews/2026-10-22-eee725b-evidence/slip_probes.py
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 /Users/ruanshijian/miniconda3/bin/python3 docs/reviews/2026-10-22-eee725b-evidence/ui_probe.py
python3 docs/reviews/2026-10-22-eee725b-evidence/verify_evidence.py
```

主回归的子套件保留非零退出记录；合计不能当作全部通过。复现依赖现有 numpy/torch/SimpleITK/PySide6 等，不要求加载 SLIP 权重。不要运行真实病例 GPU acceptance 脚本来完成本离线复现。

本目录 JSON/日志内仓库与临时路径已脱敏为 `<repo>`/`<synthetic-dir>`；原始测试输出的尾随空格保留。脚本、产物与测试边界均可检查，未触碰真实病例和实际 Python 安装。
