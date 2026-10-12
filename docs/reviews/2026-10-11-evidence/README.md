# 本轮证据与复跑边界

审查开始 2026-10-11，收尾 2026-10-12。代码基线 `a34805e`；领导汇报单独先提交 `4d6d8df`。修复代码提交 `4e2cba4`，精确 SHA-256 在[机器记录](../2026-10-11-revalidation.json)；实际测试对应包含本轮修复的工作树，不是声称基线原代码通过。

## 文件

- [regression-results.json](regression-results.json)：每项命令、规模、结论和日志尾部；不是完整矩阵。
- `*.log.txt`：末55行日志，经本机目录/临时目录占位，去除行尾空白。完整原日志在本轮隔离临时目录，没有包含患者数据。
- [workflow-observations.json](workflow-observations.json)：六条未关闭行为观察；记录实际输出，不把错误行为设为验收预期。
- [workflow_probes.py](workflow_probes.py)：真实生产函数、合成小 NIfTI、受控 DICOM header/宿主边界，随附复跑脚本。
- [严格回归](../../../tests/test_review_workflow_regressions.py)：11项。最初8项修复前1通过/5失败/2错误；增加AL两种失败分支及所选Image来源贯通后11项通过。初始日志不是当前11项全套的基线运行。

## 结论怎么读

本轮明确通过：严格工作流11、runtime目标类82、nnU-Net106、GUI80、远程契约134、跨流程2、几何6、UI基础质量4、Py3.5语法3、batch CLI目标类（规模见JSON）、export fake flow 4条、导出目标7项。远程契约使用桩，不连接服务器；GUI是离屏真实控件，不是Mimics窗口。

FlexiCT集成128项有1项因缺torch报错；模型迁移22项有19项因缺torch报错。两套保留非零结果，记为环境受阻，没有计为整套通过。没有安装/运行模型权重、CUDA、Mimics或SSH。没有跑全量回归和训练收敛；历史R79全量记录不计入本轮。任何目标系统功能/质量结论都需要实际Windows/Mimics/GPU验证。

缓存修复前，已有 `test_reexported_identical_labels_hit_case_cache` 在macOS失败；原因是硬链接改变源ctime，而非标签内容改变。修复后106项通过，新增私有缓存测试同时验证源签名/字节不变。这个平台现象不宣称Windows已复现。

大小写碰撞的原基线负对照见[脚本](export_collision_baseline.py)及日志：加载Git中`a34805e`原模块，仅在隔离进程运行已有F32测试。本机不区分大小写磁盘上原代码失败，修复后同一测试通过；不会改仓库源码。在区分大小写磁盘上原问题可能不出现。

## 复跑

使用已具备 numpy/nibabel/pydicom/SimpleITK/PySide6/pytest 的外部Python。项目标准外部环境是Windows `python_env/python.exe`；本轮为了隔离使用macOS Python3.12，没有替换仓库业务环境或配置。

```bash
# 自选evidence输出，原观察文件不要直接覆写
python docs/reviews/2026-10-11-evidence/workflow_probes.py --repo . --out /tmp/workflow-observations.json
python tests/test_review_workflow_regressions.py
python tests/test_all.py TestImportReceiptAndUndo TestRuntimeCommon TestBridgeDicomLoading TestStreamedVoxelBuffers TestFlexiCTActiveLearningApply
python tests/test_nnunet_integration.py
python tests/test_gui_smoke.py
```

macOS离屏复跑可设置 `QT_QPA_PLATFORM=offscreen TMPDIR=/private/tmp`；Windows按真实环境跑UI，不将离屏结果替代DPI/主线程/事务验收。新严格套件已注册fast/full矩阵，可用 `tools/run_regression_matrix.py --only review_workflow_regressions`。

## 探针具体证明什么

| 探针 | 实际函数与替换边界 | 观察 / 不能外推 |
|---|---|---|
| DICOM_nonuniform / duplicate | `_read_dicom_series_headers`、shape/affine实际函数；dcmread返回合成Dataset | 0,1,3→0,1.5,3；0,1,1→0,1,2；不证明Mimics原生导入也错 |
| FlexiCT_frozen_validation | 实际raw准备与最终split写入；四例2³NIfTI | 原val c0最终入train；没有实际训练GPU模型 |
| FlexiCT_all_groups_validation | 实际split helper，两患者各两期，指定各一期val | 扩组得到空train；后续trainer可能报错，不等于成功训练 |
| remote_prediction_existing_output | 实际远程结果发布函数；下载文件/旧结果是合成字节 | 旧文件直接被替换；没有SSH传输，没有宣称用户文件已受损 |
| undo_default_source_output_discovery | 实际receipt discovery；temp配置、default输出receipt | receipt存在而未发现；没有打开真实Mimics工程 |

没有运行旧2026-10-03探针覆盖历史证据。旧探针/旧测试曾给mask添加实例delete、Image添加masks，不应再用那些假边界证明当前宿主API兼容。请优先严格容器桩，再在真实版本验证。
