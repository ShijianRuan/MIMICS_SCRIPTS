# 2026-10-25 主动学习复核证据

远端源码为 `c5947bf681647c295e11a7096163733986bc9ba9`，最终实现源码为 `4af03938284e59ac2673746c65ed70dfdbc47c44`。本目录均为合成数据、临时文件或离屏 Qt 检查，没有临床病例、真实模型推理、训练或实际环境修复。

[主报告](../2026-10-25-c5947bf-active-learning.md) · [结构化记录](../2026-10-25-c5947bf-active-learning.json) · [提交范围](source-scope.json) · [验证汇总](validation-summary.json) · [可编辑流程图](active-learning-loop.mmd)

## 证据身份

- `baseline-*.log`：拉取后的原始远端版本，SLIP 77 项、nnU-Net 154 项、FlexiCT 149 项。不是修改后重新生成的基线。
- `regression-matrix.json` / `.log`：实现提交后运行的 12 个选定套件，556 项报告用例、3 项 Windows 跳过，套件退出码均为 0。推理器和 Mimics 仍有替身边界。
- `new-regressions.log`：提交前工作区的 16 项新增反例；相同用例在最终矩阵再次执行。真实 NIfTI 转换、临时发布/恢复文件与 Qt 窗口均实际运行；没有执行模型。
- `receipt-final.log`、`affected-core-final.log`：31 项回执/队列与 59 项规范化/缓冲用例（1 项 Windows 跳过）。运行时工作区相关源码字节与实现提交一致。
- `remote-fix-revalidation.log`：8 项 GPU 修复脚本的隔离临时包用例；该脚本从远端到实现提交没有改变，没有切换本机 torch。
- `setup-progress-final-0.log` 至 `-4.log` / `setup-progress-stability.json`：4 项进度窗口检查在 5 个独立进程运行，退出码均为 0。修正的是测试中 QApplication、读取线程与控件的清理。
- `affected-core-initial.log` 保留最初外部解释器未发现造成的失败；`setup-progress-initial-crash.log` 保留测试打印 OK 后进程退出 139 的失败。这两份来自修正前工作区，不属于最终源码快照。后续成功不能抹去这两次过程失败。
- `*-review*.png` 和 `ui-observations*.json`：实际 Qt 窗口，820×650 逻辑尺寸，1 倍及 1.5 倍缩放，合成病例名。已查看截图；宽度/控件数据只辅助判断，不替代视觉检查。
- `source-files.json` 记录相关源码字节摘要，`evidence-sha256.json` 记录本批交付摘要，`verify_evidence.py` 复核源码、链接、范围、原始退出码与截图尺寸。它是交付一致性检查，不是模型验收。

日志中的本机仓库、解释器和临时数据路径已替换为占位符；源码 SHA、用例名、结果与异常内容保留。证据里的历史失败不会在后续重跑时自动改为成功。

## 重跑方式与范围

使用已有的 Python 3.13、numpy、nibabel、SimpleITK、PySide6 环境。`<existing-python>` 指已经具备依赖的解释器。以下从仓库根目录运行；为新运行选择新的输出目录，避免覆盖本目录带有源码身份的历史证据。没有要求安装或升级依赖。

```sh
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 \
MIMICS_BRIDGE_PYTHON=<existing-python> <existing-python> \
tools/run_regression_matrix.py --only nnunet_integration flexict_integration \
flexict_common flexict_pkg slip_mimics slip_bridge geometry_manifest_regressions \
review_workflow_regressions active_learning_review gui_smoke ui_quality cross_workflow \
--timeout 180 --output <fresh-output>/regression-matrix.json

PYTHONPATH=tests:tools:. QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp \
PYTHONDONTWRITEBYTECODE=1 MIMICS_BRIDGE_PYTHON=<existing-python> \
<existing-python> -m unittest test_all.TestCreateMcsBatch \
test_all.TestMimicsBridgeNiftiNormalization test_all.TestMimicsBridgeBufferMapping \
test_all.TestGpuRepairLauncher test_all.TestSetupProgressWindow -v
```

截图脚本 [render_ui.py](render_ui.py) 默认写在自身目录；如需重跑，请先复制脚本到新目录，设置 `MIMICS_REVIEW_ROOT` 为对应源码工作区，并分别以 `QT_SCALE_FACTOR=1` 和 `QT_SCALE_FACTOR=1.5 AL_CAPTURE_SUFFIX=-scale-1_5` 启动两个进程。脚本用真实 Qt 加载合成文件；没有连接 Mimics。这里的 Unix 环境变量语法是本次 macOS 检查命令，不是目标 Windows 安装步骤。

```sh
python3 docs/reviews/2026-10-25-c5947bf-evidence/verify_evidence.py
```

完整训练收益、标注时间比较、目标 Windows 锁行为、Mimics 原生对象与 CUDA 执行均未验证。下一例只选择列表行；保存工程、确认标签和启动训练仍由用户明确操作。流程图中的患者验证/测试集排除是选池要求，目前没有自动检测池重叠。
