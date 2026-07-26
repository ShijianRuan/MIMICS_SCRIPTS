# Mimics-Script 项目架构与功能入口指南

本文面向第一次接触本项目的开发者，目标是回答以下问题：

- Mimics Scripting Library 中每个入口做什么；
- 一次操作会经过哪些 Python 模块、后台进程和状态文件；
- 为什么耗时工作不会直接运行在 Mimics GUI 线程；
- 图像与 Mask 如何在原始数据、Mimics 和 AI 模型之间保持空间一致；
- DINOv3 与 nnInteractive 模型如何训练、注册、推理和跨机器迁移；
- 出现异常时应查看哪里、如何停止以及如何继续开发。

本文描述的是当前代码架构。实际部署目标为 Windows + Mimics，项目开发与多数离线测试也可在 macOS/Linux 完成。

## 1. 总体架构

项目将运行环境分为两层。

### 1.1 Mimics 内控制层

目录：`scripting_library/`、`runtime_py35/`

职责：

- 调用 Mimics API；
- 读取当前 Image、Mask、选中对象和项目 metadata；
- 采集点、框、套索等交互提示；
- 启动外部 Python 进程；
- 使用定时器轮询 JSON 状态；
- 将外部结果分步应用回 Mimics；
- 在 Mimics 日志中显示关键状态；
- 提供取消、停止、回退和异常恢复路径。

约束：

- 保持 Mimics 内置 Python 3.5 兼容；
- 不在 GUI 线程中导入 PyTorch、SimpleITK、nibabel、PySide6 等重依赖；
- 不在 GUI 线程中执行大目录扫描、模型加载、训练、推理或大数组变换；
- Mimics API 只在 Mimics 进程中调用，不从外部进程跨进程直接操作 Mimics 对象。

### 1.2 外部工作层

目录：`tools/`、`external/`、根目录桥接脚本

职责：

- PySide6 界面；
- 数据发现和医学图像读取；
- DICOM/NIfTI/MHA/MHD/NRRD 转换；
- 几何变换与 Mask 最近邻重采样；
- DINOv3、nnInteractive 训练和推理；
- 模型注册、打包和导入；
- 状态、日志、缓存和资源锁管理。

外部进程优先使用项目的 `nninteractive_env`，而不是系统 Python。环境由 `99_Admin/01_Setup_Repair_Environment.py` 创建或修复。

### 1.3 基本调用模式

```text
Scripting Library 入口
        |
        v
runtime_py35/_mimics_entrypoint.py
        |
        v
Mimics 控制模块
        |
        +--> 快速 Mimics API 操作
        |
        +--> Popen 外部 Python / 后台 Mimics
                    |
                    +--> JSON 状态 + 轮转日志 + 结果文件
                    |
Mimics 定时器 <-----+
        |
        v
分步应用结果、显示完成/失败状态、清理任务
```

`runtime_py35/_mimics_entrypoint.py` 是所有菜单脚本的统一薄入口。它负责定位 `runtime_py35`、按名称加载真正的运行模块，并将入口指定的 action 或 function 分发给该模块。

## 2. 主要目录

| 路径 | 作用 |
|---|---|
| `scripting_library/` | Mimics 菜单中可见的入口，只做轻量分发 |
| `runtime_py35/` | Mimics 内控制器、定时器、状态机、Mimics API 操作 |
| `tools/` | 外部 UI、任务编排、CLI、测试和部署工具 |
| `external/dinov3-medical-seg/` | DINOv3 医学分割训练与推理核心 |
| `external/nninteractive-finetune/` | nnInteractive 任务模型微调核心 |
| `mimics_bridge.py` | 通用医学图像导入、导出和空间转换桥 |
| `nninteractive_bridge.py` | nnInteractive 服务、图像会话、推理与结果转换桥 |
| `.mimics_runtime/` | 本机运行状态、任务目录和临时文件，不提交 Git |
| `logs/` | 本机轮转日志，不提交 Git |
| `nninteractive_task_models/` | 本机 nnInteractive 任务模型工作区，不提交 Git |

## 3. Data 功能入口

### 3.1 `01_Data/01_Import_Dataset.py`

用途：批量发现数据集病例并转换为 `.mcs`。

调用链：

```text
01_Import_Dataset.py
  -> runtime_py35/mimics_import.py
  -> tools/io_path_setup_ui.py
  -> 外部 discovery / mimics_bridge.py prepare
  -> 输出目录中的 prepared queue
  -> runtime_py35/create_mcs_batch.py（后台 Mimics）
```

设计要点：

- 路径选择与大目录发现位于外部 PySide6 进程；
- 支持 Images only、All masks、指定一个或多个 Mask；
- preparation 与 `.mcs` 创建采用流式队列，第一例准备完成后即可开始创建，不等待全部病例；
- 每例有独立工作目录和错误记录，单例失败不会中止后续病例；
- producer lease 防止同一输出队列被多个导入生产者相互覆盖；
- 后台 Mimics 负责必须由 Mimics API 完成的 `.mcs` 创建；
- 可通过 `04_Stop_Import_Queue.py` 停止本项目创建的导入队列。

关键状态：

- `.mimics_runtime/import_runs/`：当前前台 Mimics 发起的导入任务；
- 输出目录下的 `_import_jobs/` 或运行时队列目录：病例准备和 `.mcs` 创建状态；
- `job_state.json`、队列 active/done/stop 标记：进度、终态和停止请求；
- `mimics_import.log`：导入主日志，按大小轮转。

### 3.2 `01_Data/02_Import_Single_Case.py`

用途：导入一个图像文件、一个 DICOM 文件夹或一个病例文件夹。

调用链：

```text
02_Import_Single_Case.py
  -> runtime_py35/mimics_import.py("single_case")
  -> tools/io_path_setup_ui.py
  -> tools/single_case_import_worker.py
  -> mimics_bridge.py
  -> 与批量导入相同的 prepared queue / create_mcs_batch.py
```

单例入口与批量入口共用准备、队列、后台 Mimics 和状态机制，避免维护两套不一致的转换逻辑。外部 worker 负责耗时读取和准备；Mimics 内只轮询进度。用户点击 Stop 时，控制器会终止已暴露的外部进程并写入停止标记。

### 3.3 `01_Data/03_Export_Masks.py`

用途：将当前项目或一批已保存 `.mcs` 中的 Mask 导出为与源图像网格一致的标签文件。

调用链：

```text
03_Export_Masks.py
  -> runtime_py35/mimics_export.py
  -> tools/io_path_setup_ui.py
  -> 当前项目：Mimics timer 分步读取 buffer
  -> 批量项目：后台 Mimics 打开 .mcs
  -> mimics_bridge.py do_convert
  -> dataset_manifest.json
```

设计要点：

- 输入 `.mcs` 路径、原始数据路径和标签输出路径由用户明确选择；
- 用户可导出全部 Mask 或指定 Mask 名称；
- 当前打开项目优先在当前 Mimics 内分步导出，避免不必要地启动第二个 Mimics；
- 批量读取其他 `.mcs` 时使用后台 Mimics；
- 导出的 Mask 先从 Mimics buffer 恢复到 Mimics 物理网格，再以最近邻插值映射到源图像网格；
- 输出 shape、affine、方向和源图像一致；
- `dataset_manifest.json` 记录 case、图像、标签、`.mcs` 和几何信息之间的关系；
- 用户选择的独立输出目录不会修改原始标签；只有明确选择原始标签目录并允许覆盖时才覆盖。

停止入口：`01_Data/06_Stop_Mask_Export.py`。

### 3.4 `01_Data/04_Stop_Import_Queue.py`

用途：只停止 Mimics-Script 创建的导入准备和 `.mcs` 创建队列。

实现：`runtime_py35/mimics_stop_background.py:main_stop_import`

它不会终止当前前台 Mimics，也不会终止非本项目创建的 Python/Mimics 进程。停止过程先写 stop marker，再尝试优雅终止已登记进程，最后清理任务监控器和锁。

### 3.5 `01_Data/05_Import_Masks.py`

用途：将 NIfTI、MHA、MHD、NRRD 等分割文件导入当前活动 Image。

调用链：

```text
05_Import_Masks.py
  -> runtime_py35/mask_import.py
  -> 外部文件选择器
  -> mimics_bridge.py prepare_masks_for_grid
  -> Mimics timer 每次应用一个 Mask buffer
```

设计要点：

- 外部进程读取文件和执行空间变换；
- 二值 Mask 生成一个 Mimics Mask；
- 多标签文件按非零 label 拆分为多个 Mimics Mask；
- 通过源 Mask affine 和当前 Mimics voxel-to-RAS 矩阵显式对齐；
- 标签只使用最近邻插值，防止生成不存在的类别；
- Mimics timer 一次应用一个结果并调用 GUI 更新，避免一次性写入多个大 Mask；
- 有并发防护、超时和取消路径，避免重复点击同时修改同一项目。

### 3.6 `01_Data/06_Stop_Mask_Export.py`

用途：只停止当前 Mimics-Script Mask 导出任务。

实现：`runtime_py35/mimics_stop_background.py:main_stop_export`

它通过当前导出任务保存的 PID、ownership token、status 和 stop marker 定位目标，不进行全局进程枚举式误杀。

### 3.7 `01_Data/07_Quick_Export_Masks.py`

用途：快速导出当前打开项目中的全部 Mask。

实现：`runtime_py35/mimics_export.py:quick_export_main`

此入口复用标准导出的几何和转换逻辑，不维护独立的“简化坐标转换”。它适合当前病例快速保存；需要选择 Mask、批量 `.mcs` 或自定义数据关系时使用标准 Export Masks。

## 4. DINOv3 Few-Shot 功能入口

### 4.1 组件关系

```text
runtime_py35/fewshot_mimics.py
  |-- tools/fewshot_training_setup_ui.py
  |-- tools/fewshot_status_viewer.py
  |-- tools/fewshot_model_chooser.py
  |-- tools/fewshot_pipeline.py
          |-- runtime_py35/mimics_export.py
          |-- external/dinov3-medical-seg/
          |-- tools/resource_locks.py
```

`fewshot_mimics.py` 只负责 Mimics 上下文、外部进程启动、状态轮询和预测 Mask 应用。训练、数据物化和模型推理由 `fewshot_pipeline.py` 在外部环境完成。

### 4.2 `02_AI/DINOv3/01_Train_Model.py`

用途：打开统一训练设置界面并启动一个 DINOv3 少样本训练任务。

实现：

- Mimics 侧：`fewshot_mimics.main(BUTTON_TRAIN_MODEL)`；
- UI：`tools/fewshot_training_setup_ui.py`；
- 编排：`tools/fewshot_pipeline.py train`；
- 训练核心：`external/dinov3-medical-seg/scripts/train.py`。

训练数据可来自：

- 原始数据集中的现有标签；
- 用户指定的独立标签导出目录；
- 已保存 `.mcs` 中最新标注的 Mask；
- 上述来源生成的 `dataset_manifest.json`。

当需要从 `.mcs` 导出标签时，先做 Mask 名称预检，再启动昂贵的后台导出。训练界面保存的是任务配置和可迁移数据引用；训练产物的模型清单保存相对 checkpoint/config 路径。

### 4.3 `02_AI/DINOv3/02_Predict_Current_Case.py`

用途：使用当前器官的推荐模型预测当前病例。

实现：`fewshot_mimics.main(BUTTON_PREDICT)`

行为：

- 根据选中 Mask 名称确定器官；
- 在当前数据集和全局模型索引中解析推荐模型；
- 推理前显示实际模型身份，不使用不可见的隐式失败模型；
- 将当前 `.mcs` metadata 对应的源图像作为模型输入；
- 外部预测生成带 affine 的 NIfTI；
- bridge 将预测映射回当前 Mimics grid；
- 结果完成后允许 `Update Selected Mask` 或 `Create Editable Copy`。

### 4.4 `02_AI/DINOv3/03_Predict_Choose_Model.py`

用途：先选择一个已训练或已导入模型，再预测当前病例。

实现：

- Mimics 控制器：`fewshot_mimics.main(BUTTON_PREDICT_MODEL)`；
- 外部选择器：`tools/fewshot_model_chooser.py`。

适用于同一器官存在多个模型版本、不同数据域模型或需要回退到上一版本的情况。

### 4.5 `02_AI/DINOv3/04_Show_Status_Results.py`

用途：查看当前任务相关的训练/推理进度、日志、曲线、结果和模型版本。

实现：`tools/fewshot_status_viewer.py`

状态查看器运行在独立 PySide6 进程中，不阻塞 Mimics。它支持自动刷新、保持日志尾部阅读位置、失败任务重试、编辑参数后重试，以及 DINOv3 模型包导入/导出。

### 4.6 `02_AI/DINOv3/05_Stop_AI_Task.py`

用途：停止当前 DINOv3 训练或推理任务。

实现：`fewshot_mimics.main(BUTTON_STOP)`

停止顺序：

1. 写入 cancel marker；
2. 允许 worker 在检查点安全退出；
3. 超过 grace period 后终止已登记的子进程树；
4. worker 写入 `cancelled` 或可诊断的 `failed` 终态；
5. 释放 GPU 锁并清理监控器。

## 5. nnInteractive 功能入口

nnInteractive 被明确分成“官方通用模型”和“自定义任务模型”，二者共用可靠的提示采集与异步应用引擎，但模型来源互不覆盖。

### 5.1 `02_AI/nnInteractive/01_Annotate_Official_Model.py`

用途：使用官方通用 nnInteractive 模型进行交互标注。

实现：

- Mimics 控制：`runtime_py35/nninteractive_mimics.py`；
- 外部服务与推理：`nninteractive_bridge.py`；
- 官方模型：`nninteractive_env/models/nnInteractive_v1.0`。

支持提示：

- 前景/背景点；
- scribble；
- box；
- lasso；
- undo；
- reset；
- 从已有 Mask 开始；
- 放弃当前 AI session。

图像 worker、模型服务和预处理在外部 Python 中运行。提示采集前可启动图像会话准备，提示提交后 Mimics 只轮询结果。预测完成后，用户可更新选中 Mask 或创建可编辑副本。

### 5.2 `02_AI/nnInteractive/02_Annotate_Custom_Model.py`

用途：使用某个任务专用微调模型进行交互标注。

实现：

- 任务解析与模型选择：`runtime_py35/nninteractive_finetune_mimics.py`；
- 模型选择 UI：`tools/nninteractive_task_model_chooser.py`；
- 提示与异步推理：复用 `runtime_py35/nninteractive_mimics.py`；
- 模型服务：复用 `nninteractive_bridge.py`，通过 model profile 指定 checkpoint。

任务解析优先级：

1. 当前 Mask 上已记录的 task/model metadata；
2. Mask 名称与 task aliases；
3. 当前项目绑定；
4. 只有一个兼容任务时自动选择；
5. 仍不唯一时打开任务模型选择器。

自定义模型不会替换官方模型。调用的是同一交互引擎的 `run_with_model_profile` 路径，model profile 中包含 task、model、checkpoint checksum 和输入契约。

### 5.3 `02_AI/nnInteractive/03_Train_and_Manage_Custom_Models.py`

用途：训练、监控、选择、导入和导出 nnInteractive 任务模型。

实现：

- Mimics 控制：`runtime_py35/nninteractive_finetune_mimics.py`；
- 模型中心：`tools/nninteractive_task_model_center.py`；
- 数据解析：`tools/nninteractive_task_common.py`；
- 训练编排：`tools/nninteractive_finetune_pipeline.py`；
- 训练核心：`external/nninteractive-finetune/`；
- 模型包：`tools/ai_model_bundle.py`。

模型中心包含：

- Task 与 Target Mask；
- 原始数据、`.mcs`、独立标签目录的数据来源；
- 可选病例筛选；
- 训练状态、日志和结果；
- 模型版本、当前推荐模型；
- 导入/导出可迁移模型包。

工作区默认位置：`nninteractive_task_models/`。

## 6. Review 功能入口

### 6.1 `03_Review/01_Identify_Mask_At_Cursor.py`

用途：点击图像位置，识别该位置属于哪些 Mask，无论 Mask 当前是否可见。

实现：`runtime_py35/mask_identifier.py`

策略：

- 先用 bounding box 排除不可能命中的 Mask；
- 对可能命中的 Mask 读取或复用缓存的 voxel buffer；
- 显示 Mask 名称、颜色和可见状态；
- 一次点击结束后明确选择 `Click Again` 或 `Finish`，避免长期占用 Mimics 点击状态机。

### 6.2 `03_Review/02_Window_From_Selected_Mask.py`

用途：根据选中 Mask 的规范化名称自动匹配窗宽窗位预设。

实现：`runtime_py35/window_level_mimics.py("auto")`

预设来源：`window_level_presets.json`。找不到名称匹配时给出明确提示，不静默应用错误预设。

### 6.3 `03_Review/03_Window_Choose_Preset.py`

用途：手动选择预设。

实现：`window_level_mimics.py("choose")`

### 6.4 `03_Review/04_Window_Undo_Last.py`

用途：恢复执行本项目窗宽窗位操作前保存的上一状态。

实现：`window_level_mimics.py("undo")`

### 6.5 `03_Review/05_Window_Reset_Full_Range.py`

用途：将当前 Image 对比度恢复到 Mimics 允许的完整灰度范围。

实现：`window_level_mimics.py("reset")`

输入值会根据 Mimics 当前 Image 的合法 GV 范围裁剪，避免 lower/upper contrast point 越界。

## 7. Admin 功能入口

### 7.1 `99_Admin/01_Setup_Repair_Environment.py`

用途：安装或修复项目外部 Python、离线 wheels、PySide6、CUDA 依赖和模型目录。

实现：

- Mimics 控制：`runtime_py35/setup_environment.py`；
- 外部安装器：`tools/setup_env.py`；
- 可移植部署检查：`tools/package_portable.py`。

安装过程在外部进程运行，状态通过 JSON/日志反馈，不在 Mimics 内执行 pip。

### 7.2 `99_Admin/02_Clear_Cache.py`

用途：清理已终止任务的缓存、中间文件和过期状态。

实现：`mimics_stop_background.clear_cache_main`

清理前会检查任务活跃状态，不删除仍在使用的工作目录、模型或用户数据。

### 7.3 `99_Admin/03_Stop_All_Owned_Services.py`

用途：异常情况下停止所有由当前 Mimics-Script 安装创建的后台服务和任务。

实现：`runtime_py35/mimics_stop_background.py`

识别依据包括：

- 运行状态文件记录的 PID；
- ownership token；
- 允许的脚本根目录和命令行；
- nnInteractive server state；
- import/export/few-shot job registry。

该入口不以进程名称直接杀死所有 Python 或 Mimics，因此不会主动终止前台 Mimics和其他软件创建的后台进程。

### 7.4 `99_Admin/04_Fix_Source_Affine_Metadata.py`

用途：修复旧版 `.mcs` 中不正确或缺失的源 NIfTI affine metadata。

适用场景：旧项目在 AI 推理时报告源图像 affine 与当前 Mimics 项目不匹配。

脚本会：

- 读取当前活动 Image 的源路径；
- 从磁盘读取真实 NIfTI affine；
- 校验 source shape；
- 更新 metadata；
- round-trip 验证；
- 保存项目。

新导入项目不应依赖此修复入口。

## 8. 图像与 Mask 的空间契约

### 8.1 RAS、LPS 与数组

NIfTI 数组本身没有“RAS 数组”或“LPS 数组”的固有属性。空间含义由 `voxel index -> world coordinate` 的 affine 决定。

- DICOM 和 Mimics 的世界坐标语义是 LPS；
- NIfTI 通常通过 affine 表达 voxel 到 RAS；
- 项目内部统一保存显式的 voxel-to-RAS 矩阵用于数学运算；
- 需要与 DICOM/Mimics API 交互时，显式进行 RAS/LPS 世界坐标换算；
- 不通过“看到 LPS 就盲目翻两次轴”的方式处理方向。

### 8.2 导入时保存的 metadata

活动 Image 上的关键 metadata：

| 键 | 含义 |
|---|---|
| `mimics_script.source_image_path` | 原始图像路径 |
| `mimics_script.source_case_dir` | 原始病例目录 |
| `mimics_script.source_image_shape` | 原始 voxel shape |
| `mimics_script.source_voxel_to_ras_matrix` | 原始 voxel 到 RAS |
| `mimics_script.mimics_voxel_to_ras_matrix` | Mimics buffer voxel 到 RAS |
| `mimics_script.mimics_to_source_index_matrix` | Mimics voxel 到源 voxel 的索引变换 |
| `mimics_script.source_image_modality` | CT/MR 等模态信息 |

### 8.3 最小重采样原则

- 图像能由标准 DICOM 几何表达时，保留原始采样网格；
- 只有包含无法用经典单序列 DICOM 几何表达的剪切等情况时，才生成一次可表达的目标网格；
- Mask 使用完整 affine 映射到该目标网格；
- 标签一律使用最近邻插值；
- 导出时从 Mimics grid 映射回原始图像 grid；
- 训练和推理均以源图像空间为数据契约，Mimics 只作为显示与编辑空间。

因此，训练数据来自原始标签还是 `.mcs` 导出标签，不应产生两套方向定义：后者在导出时已经被恢复到对应源图像网格。

### 8.4 `dataset_manifest.json`

当图像、`.mcs` 和标签不在同一目录时，不依赖文件夹猜测，而使用 manifest 记录关系。

它解决：

- 标签导出到自定义目录；
- 不同病例的标签位于不同目录；
- 数据集移动到另一台机器；
- 训练只选择部分病例；
- 同一病例存在原始标签和新导出标签。

manifest 中优先保存相对路径；数据集整体迁移后，解析器会从 manifest 所在位置重新解析。

## 9. 后台任务生命周期

每个长任务都应遵循以下状态机：

```text
created -> starting -> running/waiting
                       |       |
                       |       +-> stopping -> cancelled/failed
                       +-> completed/failed/cancelled
```

基本规则：

- 状态先落盘，再向 Mimics 日志显示；
- 完成、失败、取消是终态；
- `stopping` 必须包含进程是否仍存活、正在等待什么和下一次升级动作；
- 锁只在 worker 已退出或确认不再访问资源后释放；
- 状态终态写入、结果原子发布、锁释放和 monitor 清理必须保持固定顺序；
- JSON 使用临时文件加 `os.replace`，Windows/网络盘失败时有重试和可读 fallback；
- 日志限制大小并保留有限数量备份；
- 清理只处理终态或可证实的 stale 任务。

### 9.1 资源协调

- GPU：DINOv3 与 nnInteractive 共享资源锁，避免同时占满显存；
- nnInteractive 服务可以在 DINOv3 训练前按需释放 GPU；
- 后台 Mimics：任务按实际输出队列和 `.mcs` 访问关系协调，不假设所有机器都只能单实例；
- 同一输出目录的 import producer 使用 lease 防止互相覆盖；
- 同一个活动项目的 Mask 导入、导出和 AI 结果应用有本地 monitor 防重入。

## 10. nnInteractive 脑模型跨 Windows 迁移

当前本机模型：

- Task：`Brain Extraction (MR T1)`；
- Task ID：`brain_extraction`；
- Model ID：`brain_clopa_in_v1`；
- 本机模型目录：`nninteractive_task_models/tasks/brain_extraction/models/brain_clopa_in_v1`。

模型 checkpoint 是 PyTorch 格式，不绑定 macOS 路径或 macOS Python 环境。正确迁移方式是导出模型包，而不是复制整个 `nninteractive_env` 或手工修改 registry。

### 10.1 在源机器导出

Mimics/GUI 方式：

1. 打开 `02_AI/nnInteractive/03_Train_and_Manage_Custom_Models`；
2. 选择 `Brain Extraction (MR T1)`；
3. 打开模型版本区域；
4. 选择 `brain_clopa_in_v1`；
5. 点击 Export Model Package；
6. 得到一个 `.zip` 模型包。

命令行方式：

```bash
python tools/ai_model_bundle.py export-nninteractive \
  --workspace nninteractive_task_models \
  --task-id brain_extraction \
  --model-id brain_clopa_in_v1 \
  --output brain_clopa_in_v1.model.zip
```

模型包包含：

- `checkpoint_final.pth`；
- nnInteractive plans、dataset 和 inference metadata；
- task/model 身份；
- 输入预处理契约；
- checkpoint SHA-256；
- 不含源机器绝对路径。

### 10.2 在 Windows 目标机器准备环境

1. 部署与模型包兼容的最新 Mimics-Script 代码；
2. 在 Mimics 中运行 `99_Admin/01_Setup_Repair_Environment`；
3. 确认 `nninteractive_env`、PyTorch/CUDA 和官方 nnInteractive 运行依赖安装成功；
4. 不要复制 macOS 的 `nninteractive_env`，Windows 必须使用 Windows Python 和 Windows wheels；
5. 将模型 `.zip` 复制到 Windows 任意普通用户可读目录。

### 10.3 在 Windows 导入

Mimics/GUI 方式：

1. 打开 `02_AI/nnInteractive/03_Train_and_Manage_Custom_Models`；
2. 进入模型版本区域；
3. 点击 Import Model Package；
4. 选择模型 `.zip`；
5. 导入器校验结构和 SHA-256；
6. 将其设为该 Task 的当前模型。

命令行方式：

```bat
nninteractive_env\python.exe tools\ai_model_bundle.py import-nninteractive ^
  --workspace nninteractive_task_models ^
  --bundle D:\Models\brain_clopa_in_v1.model.zip ^
  --set-current
```

导入后本机 registry 只记录：

```text
tasks/brain_extraction/models/brain_clopa_in_v1
```

不会记录源 Mac 的 `/Users/...` 路径。

### 10.4 在 Mimics 中使用

1. 打开目标 MR 脑病例 `.mcs`；
2. 选中已有脑 Mask，或创建并规范命名为 Brain 的 Mask；
3. 运行 `02_AI/nnInteractive/02_Annotate_Custom_Model`；
4. 系统根据 Mask alias、项目绑定或选择器解析 Brain Task；
5. 使用点、scribble、box 或 lasso 提示；
6. 结果完成后更新选中 Mask，或创建可编辑副本。

目标病例的采集方向可以与训练集不同。输入会根据各自 affine 转为 canonical RAS，再执行同一输入契约：

```text
源图物理值
  -> canonical RAS
  -> nonzero spatial bounding-box z-score
  -> nnInteractive task model
  -> 预测映射回当前 Mimics grid
```

这里的 nonzero bounding box 来自图像非零区域，不依赖真实 Mask。训练与推理使用同一实现。

### 10.5 实际验证结果

当前真实脑模型已完成以下离线 round-trip：

- 从现有工作区导出 368.1 MB 模型包；
- 导入全新空工作区；
- 重建 Task 和推荐模型；
- checkpoint SHA-256 校验通过；
- model profile 解析通过；
- 导入 registry 不包含源工作区绝对路径。

这验证了模型文件和注册设计的跨机器可迁移性。由于开发机没有 Mimics，仍需在目标 Windows + Mimics 上完成一次真实 GPU 推理验收，重点检查环境版本、显存和结果应用；这不需要重新训练模型。

## 11. DINOv3 模型跨机器迁移

DINOv3 使用相同的模型包工具：

```bash
python tools/ai_model_bundle.py export-dinov3 \
  --manifest /path/to/model/manifest.json \
  --output liver_model.model.zip
```

Windows 导入：

```bat
nninteractive_env\python.exe tools\ai_model_bundle.py import-dinov3 ^
  --workspace D:\Dataset\fewshot_models ^
  --bundle D:\Models\liver_model.model.zip ^
  --set-latest
```

打包器会：

- 将 checkpoint 与推理 config 放入同一包；
- 展平 config 的 `_base_` 依赖；
- 删除训练机数据路径和实验目录；
- 以相对路径重建本机 manifest；
- 更新本机全局模型索引。

DINOv3 预训练 encoder 属于目标机器的标准运行资产，不应重复塞入每个任务模型包。

## 12. 配置文件

| 文件 | 作用 |
|---|---|
| `mimics_io_config.json` | 导入/导出、buffer mapping、后台 Mimics 与路径默认值 |
| `nninteractive_config.json` | 官方 nnInteractive 服务、超时、worker 与结果策略 |
| `nninteractive_finetune_config.json` | 自定义任务模型工作区、训练和模型中心默认值 |
| `fewshot_config.json` | DINOv3 环境、训练策略、资源和 UI 默认值 |
| `window_level_presets.json` | Mask 名称与窗宽窗位预设 |

配置原则：

- 配置中的路径只作为默认值，用户明确选择优先；
- 可迁移模型 manifest 不保存本机训练数据绝对路径；
- runtime 状态不能写入配置文件；
- 新配置键应有默认值、类型校验和旧键迁移逻辑。

## 13. 日志与排障

先判断问题属于哪一层：

1. 入口没有启动：检查 Mimics logging 和外部 UI stderr；
2. UI 启动失败：检查 `.mimics_runtime/io_setup/*_stderr.log`；
3. 导入/导出失败：检查对应 output/log 目录和 per-case failed record；
4. AI 等待：查看 status 中的 `waiting_for_gpu`、`waiting_for_background_mimics` 或 worker PID；
5. 结果方位错误：核对 source path、source affine、Mimics affine 和 manifest；
6. 模型无法加载：先执行模型 audit/checksum，不要直接重训；
7. 异常残留：优先使用功能自己的 Stop；最后使用 `99_Admin/03_Stop_All_Owned_Services`。

错误处理要求：

- 用户可见消息说明“失败在哪一阶段、下一步做什么、日志在哪里”；
- Python traceback 写日志，不用一大段内部栈替代用户消息；
- 失败必须落终态，不能永久停留在 running/stopping；
- Mimics message box 只用于必须确认或必须立即知道的结果；
- 正常启动和连续进度优先写 Mimics logging 或外部状态窗口。

## 14. 测试与开发

### 14.1 通用测试

```bash
python tools/test_all.py
```

覆盖：

- import/export 几何；
- 最小重采样；
- Mask 多标签和最近邻；
- 后台状态机与停止；
- DINOv3 训练/推理参数编排；
- nnInteractive worker、提示和结果应用；
- Windows 进程存活与锁处理。

### 14.2 关键专项测试

```bash
python tools/test_geometry_manifest_regressions.py
python tools/test_mimics_nnint_functional.py
python tools/test_mimics_nnint_deep.py
python tools/test_nninteractive_task_integration.py
python tools/test_model_portability.py
python tools/fake_mimics_flow_test.py
```

### 14.3 没有 Mimics 时

Fake Mimics 测试用于验证：

- Mimics API 调用顺序；
- timer 轮询；
- 对话框分支；
- Mask 创建/更新；
- 任务停止和终态；
- 不在 GUI 主路径执行重计算。

它不能替代以下 Windows 实机验收：

- Mimics 不同版本的实际 API 行为；
- 大体积 `get_voxel_buffer` / `set_voxel_buffer` 时间；
- MimicsResearch 许可证与多实例策略；
- 显存释放；
- OS 窗口层级；
- 真实图像显示方向和交互提示位置。

## 15. 新增功能的实现检查表

新增一个 Mimics 功能时至少检查：

1. Scripting Library 入口是否只有轻量分发；
2. Mimics 内代码是否兼容 Python 3.5；
3. 大目录、图像 IO、模型和数组操作是否在外部进程；
4. 是否使用项目 `nninteractive_env`；
5. 是否有 created/running/terminal 状态；
6. 是否有用户可达的 Stop/Cancel；
7. 外部进程崩溃后是否能在有限时间内进入 failed；
8. 是否只清理本功能拥有的进程和文件；
9. 锁释放是否发生在 worker 停止访问资源之后；
10. 日志是否轮转，错误是否包含诊断路径；
11. 是否会和 import/export/DINOv3/nnInteractive 争用同一资源；
12. 是否使用统一 UI theme 和命名；
13. Mask 更新是否允许安全副本；
14. 空数据、损坏文件、网络盘权限和重复点击是否有测试；
15. 几何转换是否使用 affine，而不是写死轴翻转。

## 16. 推荐阅读顺序

新开发者建议按以下顺序阅读代码：

1. `scripting_library/01_Data/02_Import_Single_Case.py`
2. `runtime_py35/_mimics_entrypoint.py`
3. `runtime_py35/mimics_import.py`
4. `mimics_bridge.py`
5. `runtime_py35/dataset_manifest.py`
6. `runtime_py35/nninteractive_mimics.py`
7. `nninteractive_bridge.py`
8. `runtime_py35/nninteractive_finetune_mimics.py`
9. `tools/nninteractive_task_model_center.py`
10. `tools/nninteractive_finetune_pipeline.py`
11. `runtime_py35/fewshot_mimics.py`
12. `tools/fewshot_pipeline.py`
13. `runtime_py35/mimics_stop_background.py`
14. `tools/resource_locks.py`
15. `tools/test_all.py` 和专项回归测试

阅读时始终区分三个空间：源图像 grid、Mimics grid、模型 canonical grid；也始终区分三个进程角色：前台 Mimics、外部 Python worker、后台 Mimics。多数复杂问题都来自把其中两个混为一谈。
