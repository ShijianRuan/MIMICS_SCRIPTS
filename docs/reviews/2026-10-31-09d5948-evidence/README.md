# 2026-10-31 审查证据

_源码 `09d5948988c4efd481ddb15d162551c644aebaef`；所有数据均为临时合成文件。_

---

## 🔎 身份与结果

[主报告](../2026-10-31-09d5948-revalidation.md) · [问题记录](../2026-10-31-09d5948-revalidation.json) · [提交范围](source-scope.json) · [验证汇总](validation-summary.json) · [事实核对](fact-check.json) · [时序图源文件](adoption-order.mmd)

- `regression-matrix.json` / `.log`：未改动测试的 14 个选择套件，13 通过、1 失败，矩阵退出 1。失败套件 `ui_workflow_regressions` 有 30 项中的 2 项失败；GUI 另有 3 项 Windows 锁检查跳过。没有将失败记录改为成功。
- `nninteractive-targeted.log`：148 项 nnInteractive/桥接/资源生命周期与几何用例，退出 0。它与矩阵可能覆盖同类函数，不能相加宣称独立端到端覆盖。
- `observations.json`、`reproduction.log`、`reproduce_review.py`：6 个合成反例。断言验证“缺口确实存在”，脚本退出 0 不是产品安全回归通过。模型/服务器未启动，终止进程调用被记录器拦截。
- `capture_ui_failures.py`、`ui-failure-observations.json`、`ui-capture.log`、PNG：在原始失败断言处保留实际 Qt 控件，随后再泵事件循环 0.6 秒并截图。没有跳过断言让原测试变绿。模型选择器截图显示当前 ID 本来就能放入视口；完成页说明则确实被垂直裁切。
- `syntax.json`：29 个变更 Python 文件语法检查；runtime 文件按 Python 3.5 语法解析。
- `source-files.json` / `evidence-sha256.json`：源码与交付文件摘要。`verify_evidence.py` 复核链接、范围、卡片/观察对应关系、失败保留与源码身份；它不执行训练或完整模型验收。

日志中的仓库、现有解释器与临时路径替换为占位符。源码 SHA、退出码、用例名和错误内容保留。截图仅有合成任务和模型 ID，不含患者资料。

## 🧪 安全重跑

从对应仓库根目录、使用已具备 numpy/nibabel/SimpleITK/PySide6/pytest 的现有解释器运行。完整选择命令见 `validation-summary.json`。为每次新运行选择新的证据目录，避免把其他源码结果写到本次旧证据上；没有安装/升级依赖的步骤。

```sh
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 \
MIMICS_REVIEW_ROOT=<repo> <existing-python> \
<copied-evidence>/reproduce_review.py <fresh-output>/observations.json

QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 \
MIMICS_REVIEW_ROOT=<repo> <existing-python> <copied-evidence>/capture_ui_failures.py

python3 docs/reviews/2026-10-31-09d5948-evidence/verify_evidence.py
```

复制截图脚本到新目录后再运行，因为它默认在脚本所在目录写观察和图片。反例脚本启动前校验关键源码仍匹配被审提交；源码变更时会拒绝复用旧证据身份。

测试没有真实 Mimics/Windows/CUDA、实际服务器、模型预测、训练或临床数据。PID 活性使用本诊断进程，真正终止从未执行；没有刻意诱发操作系统回收 PID。首次/重复运行耗时和实际分割效果仍需目标机验证。
