# e325e3a 审查证据

## 📌 身份与边界

源码固定为 e325e3a8a91c7c64e2c68e1b8b117f6cf08fc31f；范围 b609be3..e325e3a。仅合成文件和真实 Qt offscreen，无实际病例、模型训练/推理、环境安装或修复。

[报告](../2026-10-18-e325e3a-revalidation.md) · [结构化记录](../2026-10-18-e325e3a-revalidation.json) · [范围](source-scope.json) · [摘要](manifest.json) · [独立事实校验](fact-check.json)。

## 🧪 复现

在仓库根目录、具备 audit-environment.json 中依赖的现有 Python 上，以下命令只对合成临时目录工作。重新执行会改写本目录日志，历史证据应复制到新源码批次后运行，不能覆盖原 SHA 输出。

```sh
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp python3 docs/reviews/2026-10-18-e325e3a-evidence/regression_runner.py
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp python3 docs/reviews/2026-10-18-e325e3a-evidence/review_probes.py --out docs/reviews/2026-10-18-e325e3a-evidence/observations.json
QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp python3 docs/reviews/2026-10-18-e325e3a-evidence/ui_scenarios.py --out docs/reviews/2026-10-18-e325e3a-evidence
python3 docs/reviews/2026-10-18-e325e3a-evidence/verify_evidence.py
```

## 📂 内容

- review_probes.py / observations.json / probes.log：实际函数边界；S01 时钟回拨、S02 发布屏障、S06 单层方向、T04 拒写等；模型/GPU 为桩。
- ui_scenarios.py / ui-observations.json / ui-scenes.log / 5 张 PNG：实际 Qt 853×440；慢摘要读取及事件循环；关窗恢复另开真实子进程。
- restore_exit_child.py：首个真实 rename 后注入暂停，正常关闭 Qt 最后窗口并退出，由父进程检查 current/old/backup/status；不使用 kill。
- regression_runner.py / regression-results.json / 每套件日志及 result JSON：10 组选择，698 次方法执行、693 通过、2 断言失败、3 Windows 跳过；可能重叠，非完整 E2E。
- restore-lifecycle.mmd：可编辑的建议状态图，非实际实现证据。
- audit-environment.json：当前宿主版本；未修复/安装环境。

路径在日志/观测里替换成 <repo>、<synthetic-state>、<synthetic-ui> 等；截图仅显示合成病例。Windows/Mimics/GPU/真实 SSH 与医学有效性均未验收。
