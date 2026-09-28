# Mimics-Script Windows 端到端验收指南

> 文档日期：2026-08-01
> 适用范围：当前仓库代码、Windows 10/11、Mimics、NVIDIA GPU 训练与推理
> 目标：验证功能正确性、医学几何一致性、GUI 流畅度、后台生命周期和不同功能之间的资源切换。

## 1. 当前审查结论

本地自动化测试已经覆盖导入导出桥接、异步状态机、进程停止、GPU 与后台 Mimics 资源、nnInteractive、nnU-Net、ScribblePrompt、FlexiCT 和远程训练控制等代码路径。

当前自动化结果（本文档编写时的快照；最新数字以
`python_env\python.exe tools\run_regression_matrix.py --profile full`
的实时输出为准，工件写入 `.mimics_runtime/regression/`）：

- 全量测试（test_all.py + 集成套件）：数百项通过，个别因本机条件跳过。
- fake Mimics 端到端流程、资源锁、GPU 排队/取消、日志轮换、checkpoint
  清理和 nnInteractive 映射压测套件全部通过。
- Python 编译检查通过。
- 150% DPI 下的 PySide6 离屏布局和缩放检查通过。
- 合成数据几何往返检查通过，Dice 为 1.0。

这些结果说明代码具备进入 Windows 实机验收的条件，但不能替代真实 Mimics、真实许可证、真实 GPU、真实医学数据和 Windows 显示系统的最终验证。特别是下列内容必须在 Windows 上确认：

- Mimics API 的实际行为和不同 Mimics 版本差异。
- 后台 Mimics 是否受到许可证或单实例限制。
- `.mcs`、原始图像和导出 mask 的真实方位、尺寸及边界一致性。
- NVIDIA 驱动、CUDA wheel 和显存释放。
- Windows 100%、125%、150%、200% 缩放下的窗口质量。
- 网络盘、权限受限目录、杀毒软件和长路径环境下的文件操作。

## 2. 是否需要安装软件和依赖

结论：**不需要安装系统 Python，但并非完全免安装。**

完整离线包使用 Windows 嵌入式 Python，并把所有 Python 包安装到项目目录内的 `nninteractive_env`。首次部署仍需要运行 `setup_offline.bat`，否则 PySide6、PyTorch、nnInteractive、nnU-Net、SimpleITK 等功能无法工作。

| 项目 | 是否必须 | 安装或准备方式 |
|---|---:|---|
| Mimics | 必须 | 正常安装并准备有效许可证；版本需支持当前 Scripting API。 |
| 系统 Python | 不需要 | 使用离线包中的 `python/` 创建项目内独立环境。 |
| `nninteractive_env` | 必须 | 首次运行 `setup_offline.bat` 自动创建和安装依赖。 |
| `wheels/` | 离线部署必须 | 必须包含与 Windows、Python 和 GPU 版本匹配的完整 wheel。 |
| NVIDIA 驱动 | GPU 功能必须 | 安装与所用 PyTorch/ONNX Runtime 兼容的驱动。 |
| CUDA Toolkit | 通常不需要 | PyTorch 和 ONNX Runtime wheel 自带所需运行库；只有其他自编译扩展需要时才安装。 |
| Visual C++ Redistributable | 建议安装 | 部分 Windows 二进制 wheel 依赖；缺失时可能出现 DLL 加载失败。 |
| 模型权重 | 对应功能必须 | nnInteractive、ScribblePrompt、FlexiCT 等权重必须放在规定目录。 |
| Docker、NVIDIA Container Toolkit | 仅远程模式必须 | 安装在远程服务器，不需要安装在普通本地标注机。 |
| SSH/Paramiko | 仅远程模式必须 | `paramiko` 由 Python 环境安装；服务器需启用 SSH。 |
| 编译器或 Visual Studio Build Tools | 正常离线包不需要 | 前提是所有依赖均提供预编译 wheel，不在目标机源码编译。 |

### 2.1 推荐部署方式：完整离线包

完整离线包至少应包含：

```text
mimics_script_offline/
├── python/                 Windows 嵌入式 Python
├── wheels/                 完整 Windows wheel 集合
├── setup_offline.bat
├── scripting_library/
├── runtime_py35/
├── tools/
├── external/
├── remote/
├── nninteractive_env/models/
└── *.json                  运行配置
```

建议将整个目录放在本机 SSD，例如 `E:\mimics_script_offline`，不要直接放在公盘、同步盘或权限受限目录。这样可以降低小文件 I/O、临时文件原子替换、模型加载和杀毒扫描带来的等待与 `WinError 5` 风险。

首次部署：

```bat
cd /d E:\mimics_script_offline
call setup_offline.bat
```

该脚本会执行以下操作：

1. 从 `python/` 创建自包含的 `nninteractive_env`。
2. 在本地环境中准备 pip。
3. 从 `wheels/` 离线安装依赖，不访问互联网。
4. 验证 PySide6、PyTorch、CUDA、nnInteractive、nnU-Net、SimpleITK、ONNX Runtime 等依赖。
5. 检查默认 ScribblePrompt 和 FlexiCT 权重。

安装完成后运行完整性检查：

```bat
nninteractive_env\python.exe tools\package_portable.py check
```

只有全部必需项显示 `[OK]` 后，才进入 Mimics 验收。

### 2.2 仅有源码目录时

仅执行 `git clone` 或复制源码目录并不足以离线运行，因为源码通常不包含：

- Windows 嵌入式 Python。
- 完整的二进制 wheels。
- 体积较大的初始模型权重。
- 已构建的 `nninteractive_env`。

有网络时，可以从 Mimics 运行：

```text
99_Admin/01_Setup_Repair_Environment.py
```

无网络时，必须先在兼容的 Windows 开发机准备完整环境和模型，再生成离线包：

```bat
nninteractive_env\python.exe tools\package_portable.py check
nninteractive_env\python.exe tools\package_portable.py offline-bundle
```

生成目录默认为项目同级的 `mimics_script_offline`。由于 wheel 与操作系统、CPU 架构、Python ABI 和 CUDA 版本有关，离线包应在 Windows x64 环境构建并在同类 Windows 机器使用。

`pack --with-env` 只适用于源机器和目标机器的操作系统、架构、Python 与 CUDA 环境完全一致的情况；通常更推荐通过 `setup_offline.bat` 在目标机重建项目内环境。

## 3. Windows 测试机准备

### 3.1 最低准备项

- Windows 10 或 Windows 11 x64。
- Mimics 和有效许可证。
- NVIDIA GPU；建议至少 12 GB 显存，用于 3060 级训练场景。
- 最新稳定 NVIDIA 驱动。
- 本地 SSD 至少保留 100 GB 空间；大批量数据和训练建议更多。
- 一套小型测试数据，以及至少一套包含倾斜、翻转或非标准 affine 的几何测试数据。
- 若测试远程功能：可 SSH 登录的 Linux GPU 服务器、Docker 和 NVIDIA Container Toolkit。

### 3.2 Mimics 配置

1. 在 Mimics 中打开 Scripting Library。
2. 添加 `E:\mimics_script_offline\scripting_library`。
3. 确认菜单按以下顶层分类显示：`01_Data`、`02_AI`、`03_Review`、`99_Admin`。
4. 先运行 `99_Admin/01_Setup_Repair_Environment.py` 的检查功能，不要直接开始大任务。
5. 确认 Mimics 日志中只显示英文运行日志；本验收文档使用中文不影响运行日志规范。

### 3.3 GPU 检查

```bat
nvidia-smi
nninteractive_env\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nninteractive_env\python.exe -c "import onnxruntime as o; print(o.get_available_providers())"
```

预期：

- `torch.cuda.is_available()` 为 `True`。
- ONNX Runtime provider 包含 `CUDAExecutionProvider`。
- 若只有 CPU provider，默认 ONNX 路径会显著变慢，应先修复环境，不进入性能验收。

## 4. 测试数据和证据留存

建议准备以下数据：

- 1 例标准轴向 CT，带 2 至 3 个 mask。
- 1 例 MRI，验证非 CT 强度和显示路径。
- 1 例倾斜 CT 或非恒等方向矩阵数据。
- 1 例单文件 `.mhd` 与对应 raw 数据。
- 1 例 DICOM 文件夹。
- 1 个批量数据集，至少 5 例，其中人为放入 1 个损坏病例。
- 1 个 multi-label mask。
- 1 例含已有初始草稿 mask 和最终标签的 nnInteractive 微调样本。

每项验收保留：

- Mimics 版本、Windows 版本、GPU 和驱动版本。
- 入口名称、操作时间和任务 ID。
- Mimics 日志和外部 GUI 截图。
- 状态 JSON、错误日志及输出文件路径。
- 原图、导出 mask 和参考标签的 shape、affine、orientation、Dice。
- 卡顿时长、最大 GPU 显存、最大内存和磁盘增长。

## 5. 外部窗口、路径选择和 DPI

分别在 Windows 显示缩放 100%、125%、150% 和 200% 下测试：

1. 打开导入路径窗口。
2. 打开导出路径窗口。
3. 打开 FlexiCT 训练设置、状态查看窗口。
4. 打开 nnInteractive 自定义模型中心。
5. 打开 nnU-Net 训练、推理和状态窗口。

验收标准：

- 文字、按钮、图表和路径输入框不重叠、不截断。
- 不需要手动拉伸窗口才能看到主操作按钮。
- 可以直接粘贴路径，不强制遍历大盘。
- 浏览大型本地盘或网络盘时，外部窗口可以等待，但 Mimics GUI 必须继续响应。
- 外部 GUI 关闭或崩溃后，Mimics 有英文错误日志和 stderr 路径。
- 窗口不会错误隐藏在 Mimics 后方，也不会以低分辨率方式拉伸。

## 6. 数据导入和导出

### 6.1 单例导入

入口：`01_Data/02_Import_Single_Case.py`

依次验证：

- 单个 NIfTI 文件。
- 单个 MHD 文件。
- 包含 DICOM 序列的文件夹。
- TotalSegmentator 风格病例目录。
- `Images only`、`All masks`、指定一个或多个 mask。

预期：

- 路径窗口提交后立即返回，Mimics 不黑屏。
- 进度窗口从发现、准备、DICOM/MCS 创建到完成持续更新。
- 第一例准备完成后即可进入 `.mcs` 创建，不等待所有病例准备完成。
- 选择 Stop 后有真实停止路径，不出现 `This task has not exposed a stop path yet`。
- 成功后 `.mcs` 可打开，图像与 mask 对齐。
- 失败时保留 job 目录和英文诊断日志，不关闭前台 Mimics。

### 6.2 批量导入

入口：`01_Data/01_Import_Dataset.py`

验证：

1. 批量导入 5 例数据。
2. 在第 3 例放入损坏图像或不可读取 mask。
3. 分别选择全部 mask、仅图像、指定 mask 名称。
4. 第 1 例 `.mcs` 创建后立即打开标注，同时后台继续后续转换。

预期：

- 单例失败只记录该病例失败，后续病例继续。
- 中间 DICOM 和工作目录按生命周期清理，不无限增长。
- 已完成 `.mcs` 不被后续失败删除。
- 前台标注不被后台发现、准备或 `.mcs` 创建阻塞。
- 停止队列后，不再启动新病例；当前可安全终止阶段按状态结束。

停止入口：`01_Data/03_Stop_Import_Queue.py`

### 6.3 导入 mask

入口：`01_Data/04_Import_Masks.py`

验证 NIfTI、MHD、NRRD、multi-label mask，以及与图像不同方向但物理空间相同的 mask。

预期：

- 外部进程读取和重采样，Mimics 仅分块应用结果。
- 每个标签成为可编辑、可见、颜色明确的 Mimics mask。
- 连续点击入口不会启动多个并发导入修改同一项目。
- 用户可以取消；取消后已完成部分和未完成部分状态清晰。
- mask 与图像在物理空间对齐，不因 RAS/LPS 差异产生镜像。

### 6.4 导出 mask

入口：`01_Data/06_Quick_Export_Masks.py`

验证：

- 指定输出目录。
- 指定一个、多个或全部 mask。
- 导出到新目录。
- 明确选择覆盖原始 `segmentations` 目录。
- 后台 Mimics 可用和不可用两种情况。
- 大批量任务的进度、隐藏、重新查看和停止。

预期：

- 输出目录由用户选择，不受导入用 `mimics_output_dir` 污染。
- 导出结果与原始图像 shape、affine 和物理位置一致。
- 倾斜图像只执行必要的一次几何映射，标签使用最近邻插值。
- 前台 Mimics 保持响应。
- 后台 Mimics 受到许可证限制时给出明确原因和可执行方案。
- 停止入口仅终止本项目创建的导出任务。

停止入口：`01_Data/05_Stop_Mask_Export.py`

快速导出入口：`01_Data/06_Quick_Export_Masks.py`

快速导出也必须执行同样的几何验证，不能因为使用当前项目 API 而直接把 Mimics buffer 错写为原图 affine。

### 6.5 几何验收命令

使用只读工具比较原图和导出 mask：

```bat
nninteractive_env\python.exe tools\verify_medical_geometry.py ^
  --image E:\data\case001\ct.nii.gz ^
  --mask E:\exports\case001\liver.nii.gz
```

有参考标签时：

```bat
nninteractive_env\python.exe tools\verify_medical_geometry.py ^
  --image E:\data\case001\ct.nii.gz ^
  --mask E:\exports\case001\liver.nii.gz ^
  --reference E:\data\case001\segmentations\liver.nii.gz
```

验收标准：shape、affine、orientation 和物理包围盒一致；参考标签相同的往返场景 Dice 应接近 1.0。

## 7. 标注复查工具

### 7.1 光标位置识别 mask

入口：`03_Review/01_Identify_Mask_At_Cursor.py`

预期：

- 隐藏和显示的 mask 都参与识别。
- 大量 mask 时先用 bbox 过滤，避免同步读取全部 voxel buffer。
- 结果显示扫描数量和耗时摘要。
- `Click Again` 和 `Finish` 行为清晰，Esc 可退出。
- 等待点击期间不会破坏其他 Mimics 工具状态。

### 7.2 窗宽窗位

入口：

- `03_Review/02_Window_From_Selected_Mask.py`
- `03_Review/03_Window_Choose_Preset.py`
- `03_Review/04_Window_Undo_Last.py`

预期：

- 依据规范化 mask 名自动匹配预设。
- CT 与 MRI 使用合理的强度范围，不把 HU 直接当作 Mimics GV。
- 所有 lower/upper contrast point 均在当前图像允许范围内。
- Undo 恢复上一次设置。
- Reset Full Range 不崩溃，并恢复当前图像实际完整范围。

## 8. ScribblePrompt

入口：`02_AI/Annotate_With_ScribblePrompt.py`

使用：在 Mimics 采集前景/背景点击、涂画或一个前景 box，外部模型完成 2D 提示分割，再把结果返回 Mimics。

预期：

- 检查点缺失时明确提示，不静默降级成未知算法。
- 所有提示位于同一二维切片；首次只点击时由用户选择视图平面。
- CT 与 MRI 均走与模型匹配的强度预处理。
- 上一轮 logits 只有在 Mask 未被人工修改时才复用。
- 结果可继续编辑，可更新选中 mask 或创建副本。

## 9. nnInteractive

入口：

- `02_AI/nnInteractive/01_Annotate_Official_Model.py`
- `02_AI/nnInteractive/02_Annotate_Custom_Model.py`
- `02_AI/nnInteractive/03_Train_and_Manage_Custom_Models.py`

### 9.1 官方模型标注

验证点：

- 启动入口后立即在后台准备图像 worker，不把加载时间转移成 Mimics 黑屏。
- 每轮新增 1 至 3 个提示时，后台按交互顺序逐轮执行；后一轮基于前一轮预测。
- 已有 mask 作为初始 mask 时，后续交互实际基于该状态继续修正。
- Lasso、前景点、背景点、bbox 等支持状态与模型训练能力一致。
- 结果完成后不增加纯通知弹窗，而是直接询问 `Update Selected Mask` 或 `Create Editable Copy`。
- idle timeout、服务退出和 GPU 锁等待都有明确英文状态。

### 9.2 自定义模型

验证点：

- 同一 mask 任务存在多个微调模型时，必须显示模型选择，不默认使用第一个。
- 选择界面显示模型策略、训练时间、验证结果和兼容提示。
- 自定义模型推理日志明确显示实际加载的模型 ID 和权重路径。
- 不允许静默回退到官方权重。
- CLoPA-IN、CLoPA-CN 等名称不绑定 CT，MRI 任务也可配置。
- 模型历史在训练完成后自动刷新，进度条按实际阶段和 epoch 更新，不提前到 100%。

### 9.3 微调训练

验证：空 mask 开始、已有真实草稿、模拟初始草稿三种场景。

预期：

- `Initial Mask source` 由用户选择；未选择真实草稿时才按配置模拟。
- 最终标签和初始草稿分别导出、缓存，并各自通过 affine 对齐到训练图像网格。
- 同一数据再次训练时命中本地与远程数据缓存，不重复复制和打包不变病例。
- 训练中的提示模拟覆盖缺失分割、过分割、欠分割和混合误差，不固定为机械的 5 click。
- 取消后训练进程、GPU 锁、状态文件和子进程顺序一致，不出现永久 `stopping`。

## 10. FlexiCT 微调与推理

入口：

- `02_AI/FlexiCT/01_Train_Model.py`
- `02_AI/FlexiCT/02_Predict_Current_Case.py`
- `02_AI/FlexiCT/03_Active_Learning_Review.py`
- `02_AI/FlexiCT/04_Show_Status_and_Stop.py`

验收重点：

- 训练设置窗口布局无重叠，Dataset、Training 分区清楚；读到已保存的
  dataset/label 路径后自动扫描病例，不把空表误读为路径错误。
- 病例列表展示分割质量指标（Dice、HD 等）和上一轮模型版本，支持按质量排序
  挑选下一轮微调病例（主动学习闭环）。
- 训练窗口实时显示 epoch、loss、validation 指标和预计剩余时间。
- 状态窗口内可停止训练任务；停止后的训练续接（continue training）入口可用。
- 推理前验证模型 manifest、权重和预处理配置；推理结果的 affine 和 shape
  能正确映射回当前 Mimics 图像。
- 结果由用户选择更新选中 mask 或创建 `AI_` 可编辑副本。

## 11. nnU-Net

入口：

- `02_AI/nnUNet/01_Train_Model.py`
- `02_AI/nnUNet/02_Predict_Current_Case.py`
- `02_AI/nnUNet/03_Show_Status_Models.py`

验收重点：

- 支持多标签任务，不按单器官模型处理。
- Dataset ID、label mapping、configuration、fold、trainer、epochs 等配置真实传入工作流。
- `epochs` 不再是死配置，状态中的 epoch 总数与实际 trainer 一致。
- 不修改用户全局 `.bashrc` 或系统环境变量；所有路径通过子进程 env 注入。
- 训练数据转换、planning、preprocessing、training 和模型导出形成完整闭环。
- 状态 JSON 有阶段、进度、日志、错误和终态。
- 停止时先写 cancel，再终止 worker，GPU 锁绑定真实 GPU worker PID。
- Mimics 重启后恢复应用预测结果时，不重复创建已经应用的 mask。
- 推理前验证模型 label schema、spacing/geometry 和输入兼容性。
- 多标签预测按 label mapping 创建或更新对应 mask。

## 12. 本地与远程训练一致性

远程功能是可选增强，不得改变本地训练默认行为。

远程服务器需要：

- Linux、NVIDIA GPU 和可用驱动。
- Docker、NVIDIA Container Toolkit。
- 可 SSH 登录的地址、端口和用户凭据。
- 预先构建的统一训练镜像，例如 `mimics-ai-runtime:1.0`。
- 持久化 Remote work folder，用于数据缓存、模型缓存、日志和结果。

分别对 FlexiCT、nnInteractive 和 nnU-Net 执行：

1. 第一次训练，观察本地打包、上传、容器启动、训练和权重回传。
2. 不修改数据再次训练，确认直接复用本地归档和远程缓存。
3. 只修改 1 例标签，确认只重传变化部分。
4. 指定不同 GPU，确认设备选择生效。
5. 中断网络并恢复，确认状态可诊断且不会误报完成。
6. 取消任务，确认远程容器退出或销毁，GPU 释放，持久缓存保留。
7. 在另一台客户端复用共享模型，确认模型 manifest 不包含不可迁移的绝对本地路径。

预期：

- 数据内容指纹不包含本机绝对路径和无关文件。
- 相同内容换到另一台机器仍能命中相同缓存。
- 远程缓存命中后不再执行本地全量 copy 和 tar。
- `Preparing data` 明确标记是在本地、上传、远程解包还是远程训练预处理。
- 日志、错误、容器 ID、GPU ID 和结果下载状态均可在状态窗口查看。
- 远程失败不影响本地模式；用户可以立即切回本地训练。

## 13. 资源切换、停止和恢复

本项目不采用“只要有一个后台任务，其他功能全部禁止”的全局串行策略。只有真正共享且不可并发访问的资源才互斥：Mimics mask/image buffer、同一任务目录、同一 GPU 和正在被修改的 Python 环境。后台文件准备、医学几何转换、不同输出目录的后台 Mimics 和纯 CPU 计算可以并行。

### 13.1 功能切换矩阵

| 功能 | Mimics buffer | GPU | 后台 Mimics | 切换行为 |
|---|---|---|---|---|
| 单例/批量数据导入 | 不占用当前项目 buffer | 不占用 | 按输出队列隔离 | 可继续标注和运行不冲突的 AI；单病例失败不得中断后续病例。 |
| Mask 导入 | 外部准备阶段不占用；结果写入阶段整批独占 | 不占用 | 不需要 | 多个标签作为一个原子应用批次，不能被 AI 回填或导出插入；切换项目后暂停，回到原项目再继续。 |
| 当前项目 Mask 导出 | 仅逐个读取 buffer 时独占 | 不占用 | 当前项目快速导出不需要 | 所有 buffer 落盘后立即释放租约，后续 affine 转换不妨碍标注、Mask 导入或 AI 回填。 |
| FlexiCT 训练 | 只有从 `.mcs` 获取新标签时由后台 Mimics 读取 | 全局互斥 | 按导出任务隔离 | 训练在外部低优先级进程运行；GPU 冲突显示占用者并排队/停止，不允许以 OOM 结束。 |
| FlexiCT 推理 | 结果返回时短暂独占 | 全局互斥 | 不需要 | 只向启动时的项目和图像回填；项目切换后结果等待或拒绝写入。 |
| nnInteractive | 图像/初始 mask 快照及结果回填时短暂独占 | 复用推理服务并持有全局 GPU 锁 | 不需要 | 长训练占用 GPU 时在采集提示前拒绝启动；空闲推理服务收到其他 GPU 任务请求时主动释放。 |
| nnInteractive 微调 | 标签导出阶段受 buffer/后台 Mimics 规则约束 | 全局互斥 | 视数据源而定 | 与其他训练共享 GPU 规则；配置窗口和训练任务都阻止环境被中途修改。 |
| nnU-Net 训练/推理 | 标签导出或预测回填时短暂占用 | 全局互斥，并附加数据集锁 | 视数据源而定 | 同一数据集写入不并发；预测只回填到启动时的项目。 |
| ScribblePrompt | 输入快照和结果回填时短暂独占 | 全局互斥 | 不需要 | 等待 GPU 时请求空闲 nnInteractive 服务让出显卡；项目切换后等待原项目，不误写当前项目。 |
| 环境 Check | 不占用 | 不占用 | 不需要 | 只读，可在其他任务运行时执行。 |
| 安装、修复、解压环境 | 不占用 | 不占用 | 不需要 | 任何任务、外部功能窗口或资源锁仍活跃时拒绝开始，避免运行中替换依赖。 |

Windows 后台训练/推理子进程使用 `BELOW_NORMAL_PRIORITY_CLASS`，降低 CPU 调度和大量预处理对 Mimics 前台操作的干扰。该设置不降低 GPU 算法精度，也不改变模型训练参数。

### 13.2 交叉功能测试

按以下顺序做交叉功能测试：

1. 批量导入运行时启动 FlexiCT 训练。
2. nnInteractive 推理结束后立即启动 FlexiCT 训练。
3. FlexiCT 训练运行时尝试 nnInteractive 和 nnU-Net GPU 推理。
4. 导出运行时启动另一个导入或导出。
6. 每个阶段分别点击专项 Stop 和全局 Stop。

验收标准：

- 不共享 GPU 的 CPU/I/O 任务可以并行，但不得同时写同一项目或同一 job 目录。
- GPU 任务冲突时显示等待原因、占用者和可停止入口，不以 CUDA OOM 结束。
- 多后台 Mimics 被许可证允许时可以并行；受限时清晰排队或失败，不关闭前台 Mimics。
- 后台任务的 PID、创建者、job ID 和资源锁可以追踪。
- 终止顺序为：请求取消、等待优雅退出、必要时终止子进程、确认锁可释放、写终态、清理 monitor。
- 全局停止只终止本项目创建的进程，不影响 Mimics 本身和其他软件进程。

专项停止入口包括：

- `01_Data/03_Stop_Import_Queue.py`
- `01_Data/05_Stop_Mask_Export.py`
- `02_AI/nnUNet/03_Show_Status_Models.py`（状态窗口内停止任务）
- `02_AI/FlexiCT/04_Show_Status_and_Stop.py`（状态窗口内停止任务）
- nnInteractive 模型中心中的停止操作。
- `99_Admin/03_Stop_All_Owned_Services.py`

缓存清理入口：`99_Admin/02_Clear_Cache.py`

清理前应显示将删除的缓存类别和范围；运行中的任务使用的缓存不得删除。模型权重、用户标注、原始数据和未确认导出结果不得作为普通缓存清理。

### 13.3 项目切换与全局停止

- 在后台结果尚未完成时打开另一个 `.mcs`，结果必须保持待处理或安全丢弃，不得写入新项目。
- 返回原项目并激活原图像后，允许仍在有效期内的 Mask 导入或 ScribblePrompt 结果继续应用。
- `Stop All Owned Services` 必须先写取消标记，再拆除 monitor，随后终止仍存活的受管进程，最后清除本 Mimics 会话持有的 buffer 租约和陈旧跨进程锁。
- 全局停止之后立即运行 Mask 导入、Mask 导出和任一 AI 入口，不能出现永久 busy、永久 `stopping` 或“任务无停止路径”。

## 14. 故障注入测试

至少执行以下异常场景：

- 删除一个 wheel 后运行 `setup_offline.bat`。
- 暂时移走一个必需模型权重。
- 将输出目录设为只读。
- 在 JSON 原子替换时由杀毒软件短暂占用目标文件。
- 中途关闭外部 PySide6 窗口。
- 中途终止 worker，但保留 monitor。
- 中途终止 monitor，但保留 worker。
- 网络断开、SSH 断开和远程容器被杀。
- 后台 Mimics 无许可证或启动后立即退出。
- 导入数据中混入损坏病例。

每个场景必须满足：

- 前台 Mimics 不闪退、不黑屏、不永久等待。
- 用户看到简明状态，详细信息写入英文日志。
- 错误具有可执行的下一步，不只显示 traceback。
- 可以停止、重试、编辑设置后重试，或安全切换到其他功能。
- 进程、GPU、许可证、临时文件和资源锁最终可回收。

## 15. 最终验收记录

建议按下表逐项签字：

| 模块 | 功能正确 | GUI 流畅 | 进度清晰 | 可停止/恢复 | 几何正确 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 环境安装与修复 | 待测 | 待测 | 待测 | 待测 | 不适用 | 待记录 |
| 单例导入 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| 批量导入 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| Mask 导入 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| Mask 导出 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| 窗宽窗位与复查工具 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| ScribblePrompt | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| nnInteractive 官方模型 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| nnInteractive 自定义模型 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| FlexiCT 训练与推理 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| nnU-Net 训练与推理 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| 远程训练与推理 | 待测 | 待测 | 待测 | 待测 | 待测 | 待记录 |
| 全局停止与缓存清理 | 待测 | 待测 | 待测 | 待测 | 不适用 | 待记录 |

任何一项出现以下情况都不应判为通过：

- Mimics 黑屏、闪退或长时间无响应。
- mask 与原图镜像、翻转、错位或边界被非必要重采样破坏。
- 任务显示完成但没有输出。
- 任务停止后 GPU、后台 Mimics、worker 或锁未释放。
- 推理静默加载错误模型或回退到默认模型。
- 外部 GUI 在常见 DPI 下遮挡主按钮或无法操作。
- 失败只有弹窗，没有日志和可恢复路径。
