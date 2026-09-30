**证据如何阅读与复跑**

业务基线：`36eac04`。本目录是评审产物，不是生产测试目录，也不包含修复。源码链接固定到该提交；后续开发应在修复提交上复跑相应期望行为。结果中的路径已做可移植化处理（`<REPO>`、`<TEMP>`、`<REVIEW_PYTHON>`），没有改变错误类别、计数或断言结果。脚本使用自身位置查找仓库，不依赖审查者的用户名。

**证据类型与限制**

| 文件 | 做了什么 | 不能证明什么 |
|---|---|---|
| 2026-09-29-evidence/reproduce_findings.py | 实际环境函数+受控探针、合成DICOM header、原函数线程路径、实际Qt状态、backbone读取路径 | 不是完整多帧解码、真实Mimics线程稳定性或有权重模型推理 |
| 2026-09-29-evidence/reproduce_ux.py | 原AL函数+真实临时buffer；非空列表与错误覆盖局部执行 | 不是AL完整端到端任务；错误覆盖探针的摘要写入由桩代表 |
| 2026-09-29-evidence/reproduce_api_contract.py | 缺get_active_project、仍有文档接口的宿主桩 | 不宣称所有Mimics版本都没有该接口；不证明实际数据已丢失 |
| 2026-09-30-evidence/reproduce_scenarios.py | 12个真实场景最小化探针；F22实际运行准备管线并使用临时NIfTI/硬链接；其它宿主/进程边界用桩 | 不是修复后验收；F28没有模拟真实Windows多进程锁；F29没有训练模型 |
| 两张PNG | 正式ui_theme、真实PySide6控件、合成行数据、macOS/Fusion离屏截图 | 不代表Windows字体/DPI、多显示器、读屏或完整任务体验 |
| 两份regression-results.json | 实际运行既有回归套件的退出码与输出尾部 | 通过率不等于产品覆盖率；fake宿主不能保证实际API存在 |
| flexict-test-output.txt | 第二轮失败分类的完整输出 | 缺torch/权重不等同生产实现必然失败；macOS路径别名不是Windows功能回归 |

截图是当轮布局证据；复现脚本验证控件状态，不自动重建这两张截图。未来截图回归应固定平台、字体、DPI、数据和窗口尺寸。

**隔离验证环境**

本轮使用 Python 3.12.14，numpy 2.3.5、pytest 9.1.1、scipy 1.18.1、nibabel 5.4.2、pydicom 3.0.2、PyYAML 6.0.3、SimpleITK 2.5.6、PySide6-Essentials 6.11.2；未安装 torch，不含模型权重。临时环境不属于项目支持的生产依赖组合。以上是复现实验记录，不要求将项目运行环境升级到这些版本。

在独立 Python 环境准备这些依赖后，从仓库根运行。第二轮文件删除/覆盖仅针对脚本创建的 TemporaryDirectory；请保留该隔离方式，不替换成真实病例路径。脚本执行当前业务函数，未来业务行为变化时需重新检查边界桩，不能未经审查当通用诊断器使用。

```sh
python docs/reviews/2026-09-29-evidence/reproduce_findings.py
python docs/reviews/2026-09-29-evidence/reproduce_ux.py
python docs/reviews/2026-09-29-evidence/reproduce_api_contract.py
python docs/reviews/2026-09-30-evidence/reproduce_scenarios.py
```

第二轮脚本包含5个“预期仍有缺陷”的断言，其余场景输出观察值。修复后这些断言应该不再成立，因此不要把脚本退出0解释为产品正确；测试人员应将它们改写成[T矩阵](test-plan.md)定义的正向门禁，并放到正常测试目录。这项改写属于下游开发，本次未做。

第一轮实际矩阵选择：

```sh
python tools/run_regression_matrix.py --only gui_smoke geometry_manifest_regressions nninteractive_bridge_prompts interactive_algorithms collect_diagnostics --timeout 180 --output <external-results-path>
```

第二轮实际矩阵选择：

```sh
python tools/run_regression_matrix.py --only nnunet_integration flexict_integration model_portability cross_workflow flow_imports flow_export flow_append --timeout 180 --output <external-results-path>
```

命令里的 `<external-results-path>` 是占位符，执行前替换为临时目录文件，不要覆盖仓库内已保存的审查证据。现有完整矩阵还包含训练等成本较高的测试，不因本次审查而自动要求在用户工作环境执行全部套件。

**历史记录与来源**

第一轮 F09 复核的是已有 R61-4；F14 包括 R61-8。R61-3 原同步等待已改为异步，本次发现的是新生命周期/目标复核问题。R61-22 的相关文件在基线已跟踪；R61-26 的默认路径已 defer，不能继续照抄“默认每次阻塞1800秒”。这些更正不代表历史清单其他项都已重新关闭。外部来源的访问日期和用途见两轮报告；本轮新增的 region/ignore/TotalSegmentator 资料按官方文档核对，Materialise 产品页只取得检索摘要并明确保留限制。

完整性清单保存在[artifact-manifest.json](artifact-manifest.json)，列出文档与证据的 SHA-256，以及涉及业务源码的基线 hash。它用于防止交接时误混版本，不是对模型效果或整仓库质量的认证。
