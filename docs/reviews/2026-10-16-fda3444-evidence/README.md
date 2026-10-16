# 2026-10-16 fda3444 审查证据

_源码 `fda3444427360dfba0515648da1b7b04bd1eb94b`；合成数据；业务代码与正式测试源码未修改_

## 📋 范围与使用

- `review_probes.py` / `observations.json`：真实快照/队列/发布/撤销判定/拆分/源几何/标签合成/锁/主动学习交接的反例与正向复核。
- `ui_scenarios.py` / `ui-observations.json` / PNG：真实 Qt 长值、导航、隐藏配置、中文请求构建与批量预检。模型注册行为使用合成 fixture，提交/推理/SSH 边界替换。中文页面只构建请求，不显示伪造的产品成功状态。
- `metrics-default.json` / `metrics-font125.json`：旧渲染器在当前源码执行，各 150 状态；字号放大不等于 Windows DPI。
- `control-and-entry-inventory.json`：28 入口、20 UI 源模块的当前 AST 绑定；前门与家族分发测试通过不等于原生功能验收。
- `regression-results.json` / 日志：14 组主要套件与单独的无绘图诊断；错误/失败/跳过不隐藏，类级 skip 不从执行的方法数里再扣一次。
- `manifest.json`：当前生产文件与证据 SHA-256。

## 🔁 环境与复跑

审查解释器为 Python 3.12.14，隔离环境依赖见 `audit-environment.json`。缺 torch、matplotlib 与真实权重；这些均不是生产环境缺陷结论。原临时环境缺少完整可执行依赖后，重新建立了仓库外审查环境；没有运行产品 setup/repair 或改业务配置。

从仓库根运行（python 指向审查环境，输出在仓库外）：

```sh
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-16-fda3444-evidence/review_probes.py --out /tmp/mimics-review-replay/observations.json
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-16-fda3444-evidence/ui_scenarios.py --out /tmp/mimics-review-replay/ui
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-12-ui-evidence/render_ui_audit.py --out /tmp/mimics-review-replay/render
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-12-ui-evidence/render_ui_audit.py --out /tmp/mimics-review-replay/font125 --font-scale 1.25
```

主要测试命令数组见 regression-results.json；使用对应解释器替换 `python`。macOS 可设 TMPDIR=/private/tmp；Windows 使用自身临时目录。

## 🔬 边界解释

S01 用真实 `_capture_mask_snapshot` 的对象返回值喂给真实 `create_mcs_batch.main` 队列，只有原生 create/save 写入合成 .mcs 为 stub；不是只用 AST 猜测类型。S04 原生 Mask delete/transaction 用列表 stub，实际所有权/内容判定和删除调度不替换。S02/S03 为真实文件发布，旧/新文件都只在临时目录；截断例是 gzip 正常而 NIfTI 体素 payload 缺失。

S06 pydicom.dcmread 返回合成 header，几何检查/affine 构造为真实函数。S07 使用真实别名匹配与 `_materialize_case`，小 NIfTI 合成后实际读取标签值。S09 只把模型执行替换成同样错源的两份结果，当前 folder 输出与 uncertainty 链路使用原函数。S10 用真实 run_batch 两次和真实目录分析器，捕捉失败后旧产物仍入排名。

Mimics 侧标记和外部 UI 点击函数的锁竞争使用真实本机文件锁；UI 对象外壳是最小 fixture。没有真实宿主事件循环，关于回调同步性的结论同时核对源码。Windows msvcrt、UNC、字体/DPI/焦点和 GPU/SSH 完整链路未运行。

原始 pkg 测试的绘图依赖失败保留；额外诊断仅在测试进程把 run_batch 的 save_plots 设 False，与生产 AL 调用一致，选中两测试通过。不能称该套件全部通过。源码/工件路径在日志中以 `<repo>`、`<audit-python>`、`<synthetic-state>`、`<test-state>`、`<temporary-state>` 占位；行末空白已归一，测试结果和错误内容未改。
