# 2026-10-13 复核证据

_对应业务源码 `96448aee155f2009624b5a93fc08e64e31b41c6e`；未修改业务代码_

## 📋 内容与边界

| 文件 | 用途 |
|---|---|
| observations.json / review_probes.py | 18 组观察（包含空拆分拒绝的正向结果），支撑 17 张问题卡中的函数级反例；不能把观察组数当 bug 数 |
| ui-observations.json / ui_scenarios.py | 真实 Qt 长版本、中文标签、完成导航、远程隐藏项、实际批量扫描/提交 |
| 7 张 PNG | 合成场景截图；不是 Windows/Mimics 原生截图 |
| metrics-default.json / metrics-font125.json | 当前版本各 150 状态测量；未检测到测量场景的按钮内容越界，不能证明全部文本可读 |
| control-and-entry-inventory.json | 当前 28 入口、控件构造/信号绑定；真实分发回归另有日志 |
| regression-results.json / *.log.txt | 17 组主要套件与诊断重跑；缺依赖/失败/跳过均保留 |
| source-anchors.json / manifest.json | 关键函数位置、业务源码与证据 SHA-256，固定源码身份 |

所有输入/工程/状态均在临时目录。训练和作业 _launch、实际 worker 提交、模型读取/登记 fixture、原生 Mimics delete/transaction 都按脚本注明在边界替换。文件读写、真实扫描器、发布函数、拆分函数、GUI 控件和 OS 锁按现有代码执行。没有真实推理、训练、SSH、模型下载或环境 repair。

## 🔁 复跑

需要隔离 Python 环境：Python 3.12.14、PySide6-Essentials 6.12.0、numpy 2.3.5、nibabel 5.4.2、pydicom 3.0.2、SimpleITK 2.5.6、scipy 1.18.1、pytest 9.1.1。当前审查环境没有 torch/PyYAML 和已安装的 nninteractive_finetune；因此相关用例不能算完成。产品 Windows/Python 3.13 与真实 API 必须另外验收。

从仓库根目录运行（python 指向上述隔离环境；输出目录位于仓库外）：

```sh
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-13-evidence/review_probes.py --out /tmp/mimics-review-replay/observations.json
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-13-evidence/ui_scenarios.py --out /tmp/mimics-review-replay/ui
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-12-ui-evidence/render_ui_audit.py --out /tmp/mimics-review-replay/render
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-12-ui-evidence/render_ui_audit.py --out /tmp/mimics-review-replay/font125 --font-scale 1.25
```

套件命令选择器见 regression-results.json：大部分直接 `python <command_selector>`；source-path-rerun 含 PYTHONPATH 前缀单独使用。macOS 如 Qt 临时目录路径异常可设置 TMPDIR=/private/tmp，Windows 不使用该路径。观察脚本用于记录行为，没有将已知缺陷伪装成通过测试；开发修复后应将对应期望写入正式回归并重新生成证据。

## 📐 测量解释

Offscreen 平台支持布局和内容宽度检查，缺少真实窗口管理器/字体/DPI/焦点与宿主坐标。font-scale=1.25 修改主题字号，不等于 Windows 缩放系统。R12 的长 token 超过摘要区在截图中可直接观察，即使按钮测量无越界也不能关闭。R13 fixture 隔离异步注册刷新，以完成状态与 not_improved 行验证导航/过滤；不是启动真实训练。

锁竞争探针持有真实本机 OS 文件锁，调用 Mimics 侧标记函数测返回时间；原生事件循环未运行，主线程调用链由源码核对。Undo 探针验证实际判定与删除调度，原生 delete 为 list.remove，真实可恢复性仍须目标机验证。DICOM 几何探针使用合成 pydicom header 和真实推断函数，dcmread 是受控头边界。远程组丢失通过真实打包函数与重建契约/拆分验证，不涉及服务器连接。

## 🔐 完整性与历史

manifest 中 source_commit 是被审查源码，不是随后文档提交的 hash；tracked_source_sha256 对应该源码快照，artifacts_sha256 对应本轮发布证据。报告/JSON/索引未覆写旧 evidence manifest 或旧业务源码身份。日志占位 `<repo>`、`<test-state>`、`<synthetic-state>`、`<audit-python>` 只用于剔除本机路径；日志行末空白已做格式归一，测试内容和结果未改变。
