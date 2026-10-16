# 336e99d 复审证据与复现说明

全部证据对应 `336e99d6bbe1b01a192a9a450a4af29adf0f3ee4`，日期按 Asia/Shanghai 为2026-10-17。后续文档提交SHA不等于被审源码SHA。

## 🔬 证据边界

- review_probes.py 使用真实队列、发布、撤销判定、DICOM几何、标签查找、split writer、状态持久化和完整AL数据链；仅原生Mimics创建/删除、模型执行和GPU预约为边界替代。
- gzip计数在实际reader返回处累计字节；没有运行真实大病例，没有把合成耗时外推。
- 同秒重试控制目录命名时钟；新目录部分失败只在共识保存处注入OSError；其他计算/排名链真实执行。异步反序只控制线程先后，实际JSON写入与文件锁不替代。
- ui_scenarios.py 为真实PySide6/offscreen控件与合成状态；7张PNG已人工检查，未使用患者数据，不是Windows实机截图。
- observations.json、ui-observations.json是本轮事实；旧证据保留原源码身份，不覆盖。日志中本机临时目录被替换为占位符，便于提交且避免误认为存在的持久病例路径。

## ▶️ 复现

在该源码版本的隔离审查环境，具备numpy/nibabel/pydicom/SimpleITK/scipy/PySide6（精确版本见audit-environment.json），从仓库根目录执行：

```sh
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-17-336e99d-evidence/review_probes.py --out /tmp/review-observations.json
QT_QPA_PLATFORM=offscreen python docs/reviews/2026-10-17-336e99d-evidence/ui_scenarios.py --out /tmp/review-ui
```

临时目录中的合成数据自动删除。命令不启动真实Mimics/模型，不修复产品环境；若环境缺依赖，须单列记录，不把未执行算通过。

10组测试命令由suite-exits.json记录前9组；runtime-changed.log额外执行 `tests/test_all.py TestCreateMcsBatch TestEdgeCases TestBridgeDicomLoading`。regression-results.json保留执行/通过/失败/错误/跳过计数；具体失败原因见主报告，不忽略原始失败。

## 🗂️ 文件

- [观察记录](observations.json)、[Qt观察](ui-observations.json)、[回归汇总](regression-results.json)。
- [探针日志](probes.log)、[Qt场景日志](ui-scenes.log)、[FlexiCT日志](flexict.log)、[UI回归日志](ui.log)。
- [源文件与证据哈希](manifest.json)、[独立事实核对](fact-check.json)。
- [主报告](../2026-10-17-336e99d-revalidation.md)、[结构化处理卡](../2026-10-17-336e99d-revalidation.json)。
