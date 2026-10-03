# 2026-10-03 证据、环境与复跑说明

业务提交：`f02cfad4a7b030ac017e0ba30ce8f6f6911f500e`，分支 `al-review-loop`。证据仅用合成数据和临时文件，未操作真实 Mimics、GPU、患者数据或用户远程训练任务。日志中的本机路径已替换占位符，行尾空白已清理；内容错误和测试结果保留。

## 执行结果

13 组首次运行共 **842 项测试**：832 通过、5 个 assertion failure、4 个 error、1 跳过。随后只复跑了有环境原因的失败项：

- 5 项 `/var` 与 `/private/var` 路径别名断言，在设 `TMPDIR=/private/tmp` 后全部通过，未改产品代码；不据此宣称 Windows 路径测试已做。
- 2 项 MRI 归一化测试最初找不到本地 integration package，补正确 PYTHONPATH 后仍因缺 Torch 无法运行；另 1 项 FlexiCT preflight 缺 Torch。这 3 项保持“环境阻塞”，没有安装大模型依赖来制造通过。
- **1 项真实产品失败：**系统健康窗口的 QSS `.format` 报 KeyError（F36）。日志是 [extended.txt](extended.txt)。
- 因此计入这 5 项复跑结果后，842 项的归类是 **837 通过、3 环境阻塞、1 产品失败、1 跳过**，不是全绿。复跑不是新的独立测试数量。

| 组 | 首次测试数 | 首次结果 |
|---|---:|---|
| targeted（旧问题相关 test_all 类） | 93 | 通过 |
| extended（几何、导入、会话、健康等） | 205 | 202 通过，3 errors；其中 F36 为产品错误 |
| gui_smoke | 38 | 通过，含真实 Qt 非空 AL/模型选择 |
| ui_quality | 4 | 通过，不等于整套 UI 精美/无错误 |
| nnunet_integration | 100 | 通过 |
| flexict_integration | 105 | 100 通过，4 路径断言失败，1 缺 Torch |
| nninteractive_task_integration | 11 | 通过 |
| cross_workflow_transitions | 2 | 通过，仍是离线边界替代 |
| geometry_manifest_regressions | 6 | 通过 |
| remote_training | 104 | 通过，**没有启动远程训练** |
| mimics_nnint_deep | 75 | 74 通过，1 路径断言失败 |
| mimics_nnint_functional | 91 | 90 通过，1 跳过 |
| standalone build_dataset | 8 | 通过，但未覆盖 F29 物理位置不变的变换 |

完整结构化结果、每组日志链接和环境版本见 [regression-results.json](regression-results.json)。`PySide6` 的 distribution metadata 不可用，版本读取实际导入模块为 6.11.2；不能将该字段误记为“没有 Qt”。

## 独立探针与截图

[audit_probes.py](audit_probes.py) 产生 [scenario-results.json](scenario-results.json)，共 19 条观察：F01 两种、F13 一种、F16 两种、F19 两种、F20 一种、F28 两种、F29 一种、F30 一种、F31 三种、F32–F35 各一种。

探针记录当前行为，**不是断言坏行为“应该如此”的回归测试**。开发修复后把对应 RA 验收规则转换为正向测试，不应把这些当前输出当黄金答案。探针导入现有 test_all 的 Mimics 桩以装载生产模块，随后为每项场景只替换宿主/进程边界；这些桩不证明真实 API 存在。严格 Image 契约另有一个只提供 linked_objects 的桩，暴露接口依赖问题。

主要证据方法：

- 发布冲突使用真实临时文件；其中检查后写入通过拦截 replace 调用确定性模拟，不需要随机竞争。
- 导出用可区别的 8 字节体素数组，真实执行选取/命名/manifest 代码；宿主读取由小数据替代；多 Image 探针到 manifest 构造为止。
- AL 探针真实走来源/网格/身份检查，替换外部进程与线程启动；确认错误病例可进入 converting 且完成 guard 放行，未执行实际 GPU/Mask 写入。
- 数据集 split 探针真实写入/读取 manifest 和 splits，NIfTI 文件用字节占位，因为该阶段只复制文件、不解码。F29 则用真正 nibabel NIfTI，保持世界物理目标相同，改变存储朝向。
- F01 多帧文件只是 header fixture，并非完整合法 Enhanced CT；结论限定为错误文件计层策略，完整解码仍需要 L3 fixture。
- F34 使用真实 Qt `button.click()` 和 excepthook，不是直接调用刷新函数。

[render_ui.py](render_ui.py) 产生真实生产窗口、合成列表和临时设置。保留 6 张代表截图；[ui-metrics.json](ui-metrics.json) 还保存 9 个窗口尺寸实验的按钮布局。图中的病例与模型是虚构数据，日期 1970 等来自合成排序 fixture，**不是产品日期错误**。模型窗口的状态文字明确写合成模型；不把被替换的后台扫描输出作为缺陷证据。

截图为 macOS Qt offscreen/Fusion；Windows 字体、标题栏、DPI、原生对话框和 Mimics 叠加显示仍待实测。已有截图已经证明按钮截断等本地布局问题，不能外推所有 Windows 环境表现完全相同。

## 复跑

在隔离环境准备 numpy、nibabel、pydicom、SimpleITK、PySide6、pytest。此次实际版本见 JSON。运行时设置 `PY` 为自己的审查解释器，`REPO` 为固定提交 checkout，`OUT` 为仓库外临时目录；这些变量都不是系统配置变量。不要通过运行旧报告脚本覆盖新证据。

```bash
"$PY" "$REPO/docs/reviews/2026-10-03-evidence/audit_probes.py" --repo "$REPO" --out "$OUT"
"$PY" "$REPO/docs/reviews/2026-10-03-evidence/render_ui.py" --repo "$REPO" --out "$OUT"
```

生产测试从 repo 根运行：

```bash
"$PY" tests/test_all.py TestImportReceiptAndUndo TestFlexiCTActiveLearningApply TestEnvironmentRepairSemantics TestForeignScriptIntegration TestImportDropWindow TestGuiThreadBlockingContract TestUiThemePalette TestStreamedVoxelBuffers TestWindowLevel TestScriptingLibraryEntries
"$PY" tests/test_all.py TestRuntimeCommon TestMimicsBridgeNiftiNormalization TestMimicsBridgeBufferMapping TestVerifyMedicalGeometry TestNNInteractiveBridgeSourceImage TestCreateMcsBatch TestNNInteractiveContinuousPrompting TestNNInteractiveSessionWriteMode TestBridgeDicomLoading TestSourceImagePathEquivalence TestDatasetProfiles TestSystemHealthPanel TestBatchPredictCommand
"$PY" tests/test_gui_smoke.py
"$PY" tests/test_ui_quality.py
"$PY" tests/test_nnunet_integration.py
"$PY" tests/test_flexict_integration.py
"$PY" tests/test_nninteractive_task_integration.py
"$PY" tests/test_cross_workflow_transitions.py
"$PY" tests/test_geometry_manifest_regressions.py
"$PY" tests/test_remote_training.py
"$PY" tests/test_mimics_nnint_deep.py
"$PY" tests/test_mimics_nnint_functional.py
"$PY" integrations/flexict-finetune/tests/test_build_dataset.py
```

只复跑路径别名相关失败（macOS）：

```bash
TMPDIR=/private/tmp "$PY" tests/test_flexict_integration.py TestActiveLearningReviewUI.test_materialized_pool_disambiguates_slug_collisions TestActiveLearningReviewUI.test_materialized_pool_records_source_paths TestInferenceLifecycle.test_completed_inference TestMainRouting.test_operation_infer_routes_to_run_inference
TMPDIR=/private/tmp "$PY" tests/test_mimics_nnint_deep.py PipelineErrorRecoveryTests.test_cancelled_job_cleans_partial_model
PYTHONPATH=integrations/nninteractive-finetune/src "$PY" tests/test_all.py TestSourceImagePathEquivalence.test_mri_rescale_restoration_preserves_negative_background_contract TestSourceImagePathEquivalence.test_task_model_mri_uint16_quantization_preserves_zscore_input
```

最后一行在本次环境仍缺 Torch，已如实保留错误。没有更改生产测试来跳过错误，也没有把不存在的依赖变成假模块使它“通过”。

## 完整性与适用提交

[entry-inventory.json](entry-inventory.json) 从 AST 提取本次 27 个真实菜单、runtime 路由、参数和源码行号；入口覆盖文字见场景交接。本轮 [artifact-manifest.json](artifact-manifest.json) 保存新增材料和当前总入口/backlog 的哈希，以及关键生产源码在固定提交的哈希。

上两轮的 `../artifact-manifest.json` 是历史证据，不应拿其业务源码哈希与 88 次提交后的工作树逐项比较后宣称“审查污染代码”。本轮也不改写历史业务基线。当前提交仅审查文档/脚本/截图，不修复上述缺陷；真正的修复和目标环境验收由下游据本轮记录实施。
