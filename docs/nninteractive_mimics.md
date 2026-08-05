# nnInteractive for Mimics Research 21

## 1. 结论

nnInteractive 可以作为 Mimics Research 21 的独立交互分割工具，但必须采用双进程结构：

- Mimics Python 3.5 负责选择图像和目标 Mask、采集提示、写回结果；
- 外部 Python 3.10+ 环境负责 PyTorch 和 nnInteractive 推理；
- 两者通过临时体素 buffer 和 JSON 调用协议连接。

该功能不依赖病例包、任务队列、Registry 或任何特定标注流程。只要 Mimics 当前打开了图像，用户就可以运行它。已有标注流程中的病例也可以使用同一个入口，不需要专门集成。

用户入口统一命名为 **nnInteractive**，窗口标题为 **nnInteractive Segmentation**。不再使用含义宽泛的 “AI Refine”。

## 2. 为什么不能直接安装进 Mimics Python

Mimics Research 21 的脚本环境基于 Python 3.5。nnInteractive 当前要求 Python 3.10+，并依赖现代 PyTorch，因此不能在 Mimics 解释器内运行。

Mimics 侧只做轻量工作：

1. 读取当前 active image 的灰度 buffer；
2. 读取 source Mask，并创建或恢复结果 AI Draft；
3. 通过 Mimics 原生 API 采集交互；
4. 调用外部 bridge；
5. 把预测结果写回目标 Mask。

## 3. 入口与 GUI 边界

Mimics 21 官方脚本 API 支持：

- 将脚本注册到 `Script -> Scripting Library`；
- 通过脚本一键运行；
- 打开 `Edit Mask`、点选、Spline 等交互工具。

现有公开 API 没有提供向 `Segment` 或 `Advanced Segment` 工具栏注册自定义图标、按钮或 Ribbon 命令的接口。因此当前可靠入口是：

```text
Script -> Scripting Library -> 02_AI -> nnInteractive
  -> 01_Annotate_Official_Model
```

这已经是单击启动，不需要打开 Editor、Console 或命令行。若以后要进入 Mimics 原生分割工具栏，需要单独向 Materialise 确认扩展 SDK 或厂商支持，不能把它当作 Python API 已有能力。

## 4. 目标 Mask 的选择

运行前，用户在 Project Tree 中选择一个 source Mask 或已有 AI Draft：

- 选择普通非空 Mask：自动创建 `<Mask name> - AI Draft`，原 Mask 不被修改；
- 选择空 Mask：直接把这个空 Mask 作为结果 Mask；
- 选择带 `nninteractive.role=ai_draft` 的 Mask：继续原地修正该 Draft；
- 没有选择 Mask：自动创建新的空 `nnInteractive Result` Draft；
- 选择多个 Mask：脚本停止并要求只选一个，避免把结果写错对象；
- source/target Mask 不属于 active image：脚本停止，不做隐式跨图像绑定。

普通非空 Mask 启动推理时不再询问输出方式。首个 AI 结果完成后只显示一次
`Prediction Ready` 选择：`Update Selected Mask` 在所选 Mask 上继续修正；
`Create Editable Copy` 保留原 Mask，并把结果写入新的 Draft。这个选择同时承担
完成通知，不会在应用后再弹成功提示。两种方式都支持继续追加提示、Undo 和
Reset；选择 Copy 前也不会同步复制大 Mask。

## 5. Mimics 交互与 nnInteractive 提示的映射

### 5.1 已实现的提示

| nnInteractive 提示 | Mimics 采集方式 | 转换方式 |
| --- | --- | --- |
| Include/Exclude point set | 多次 `mimics.indicate_coordinate(confirm=False)` | 空 Mask 的会话首轮可一次收集多个正负点并只预测一次；已有结果上的修正按点逐轮预测 |
| Include scribble | 临时 Prompt Mask + `activate_edit_mask(..., "Ellipse", "Draw")` | 把用户画出的 Ellipse 区域裁剪为 Scribble mask |
| Exclude scribble | 同上 | 写入 Scribble negative channel |
| Foreground box | `mimics.measure.indicate_distance_measurement()` | 两端点作为二维矩形对角点，转换为半开区间 bbox |
| Foreground lasso | `mimics.analyze.indicate_spline()` | 要求 `Spline.closed=true`，栅格化闭合轮廓 |

Box 和 Lasso 在当前 Mimics UI 中固定为前景提示，不再额外询问 Foreground/Background。nnInteractive API 虽然存在 negative box/lasso channel，但在实际修正中排除区域可用 Exclude Points 或 Exclude Scribble 更清楚，也避免每次使用范围提示都增加一次选择。

### 5.2 各提示的几何门禁

- Point Set 至少包含一个点，可以在同一批中交替添加 Include 和 Exclude 点；采集期间以绿色/红色临时 Point 显示位置，可删除最后一个点；
- Box 的两个端点必须位于同一个轴对齐切片，并且在另外两个方向上形成非零矩形；
- 当前 Mimics Lasso 入口要求显式闭合、至少包含三个不同体素点，并位于一个 axial、coronal 或 sagittal 切片；
- Scribble 可位于一个或多个切片。多切片区域会拆成若干二维 crop，全部写入后只执行一次预测。

当前 nnInteractive v1 权重明确支持 `bbox2d`、不支持真正的 3D box。nnInteractive 的 Scribble/Lasso API 接收 mask 和 `interaction_bbox`；Mimics 侧把这两类用户交互定义为二维编辑语义，因此不会把斜穿三维空间的 Spline 静默近似成 Lasso。

### 5.3 为什么 Scribble 使用 Ellipse Prompt Mask

Mimics 21 的 `activate_edit_mask()` 没有公开自由画笔类型，只公开 `Ellipse`、`Rectangle`、`Lasso`、`FloodFill` 和 `LiveWire`。它也不返回每次 stroke 的轨迹、Draw/Erase 历史或工具参数。

因此当前最接近“涂抹”的可靠方式是：

1. 脚本创建空 Prompt Mask；
2. 以 `Ellipse + Draw` 打开 Edit Masks；
3. 用户在需要包含或排除的位置画一个或多个区域；
4. 脚本读取非空区域并作为 Scribble mask；
5. 立即删除 Prompt Mask；
6. 目标 Mask 只接收模型结果，不被提示绘制污染。

Windows 实机仍需确认 Mimics 21 的一次 Ellipse 编辑会话能否连续画多个区域。如果只能画一个，功能仍然正确，用户可重复添加 Scribble；只是一次提示的覆盖范围较小。

### 5.4 连续修正与原版 nnInteractive 的关系

原版 nnInteractive 在一个 inference session 中维护：

- 初始分割或当前 previous segmentation；
- 按顺序累积的所有提示；
- 每次预测写回的 target buffer。

Mimics 集成会在外部 image worker 中保留同一 remote session 和已经上传、预处理的图像。正常追加提示时只发送新增提示；Undo、Reset、提示历史变化或 worker 重启时，才从进入会话时保存的 Initial Mask 有序重放：

1. 进入工具时保存一次目标 Mask，作为本次工具会话的 initial segmentation；
2. 首次建立 remote session 后上传图像并保留 target buffer；
3. 正常追加时只执行新增提示，让它读取上一轮 prediction 形成的 previous segmentation；
4. 需要重放时先 reset，再注入同一份 initial segmentation，并按原顺序执行提示；
5. 把最终 target buffer 写回 Mimics Mask。

因此连续修正不是“把刚预测出的 Mask 再作为新的 initial segmentation”。后者会在每次提示时重置交互通道并累积模型误差。只有退出工具并重新启动时，当前 Mimics Mask 才成为下一次工具会话的新 initial segmentation。

空 Mask 的首个 Point Set 是一个批量初始化事件：前 N-1 个点以 `run_prediction=False` 写入，最后一个点触发一次预测，以减少首次等待。已有 Initial Mask、已有预测结果后的 Point Set，以及任何后续纠错，全部按点 `run_prediction=True` 顺序执行，保证每一点都看到上一轮结果。多切片 Scribble 仍按切片顺序执行，因为每个新交互都可能改变下一轮的 previous segmentation。

## 6. 与旧 diff 方案的区别

旧方案把目标 Mask 的增删变化解释为正负 Scribble，需要保存完整 baseline。它存在以下问题：

- 所有提示都被降级成区域 diff；
- 无法保留 point、box、lasso 的真实语义；
- baseline 占用磁盘并产生额外 I/O；
- baseline 刷新后无法撤销某一条提示；
- 必须依赖此前生成的 runtime 和初始 baseline；
- 用户在目标 Mask 上画提示时会暂时破坏结果。

当前方案：

- 不保存长期 baseline；
- 进入工具时只在临时目录保存一次初始目标 Mask；
- 当前工具会话中按顺序保存提示；
- 每次从初始目标 Mask 重放提示；
- 支持 **Undo Last Prompt**；
- 支持 **Reset To Start**；
- 工具退出时删除图像、初始 Mask 和 prompt 临时文件；
- 最终结果保留在 AI Draft/空目标 Mask 中，普通 source Mask 保持不变。

重新选择 AI Draft 启动 nnInteractive 时，当前 Draft 成为新的 initial
segmentation。上一会话的提示历史不会继续保留，但分割结果不会丢失。

## 7. 用户工作流

1. 在 Mimics 中打开任意项目。
2. 激活要处理的 image set。
3. 在 Project Tree 选择原始 Mask 或已有 AI Draft；也可以不选，由脚本创建新 Draft。
4. 运行 `Script -> Scripting Library -> 02_AI -> nnInteractive -> 01_Annotate_Official_Model`。
5. 选择 **Add Points**、**Paint Scribble**、**Draw Box** 或 **Draw Lasso**。
6. Add Points 中可连续加入 Include/Exclude 点，绿色/红色标记会保留到 Run/Discard；必要时使用 Remove Last Point。
7. Paint Scribble 只需选择一次 Include 或 Exclude，然后在临时 Mask 中绘制。
8. Box 和 Lasso 默认为前景，不再显示正负选择。
9. 脚本调用外部 nnInteractive，结果自动写回 AI Draft，并显示非阻塞成功提示。
10. 继续增加提示，或使用 **Undo Last Prompt** / **Reset To Start**。
11. 选择 **Finish**，按正常 Mimics 方式保存项目。

第一次推理需要启动模型服务，通常比后续提示慢。脚本会复用正在运行的本机服务。

服务由 bridge 创建并记录所有权，不会根据一个来源不明的 PID 直接终止进程。每次提示都会刷新活动时间；默认连续 30 分钟没有推理请求后，独立 watchdog 会核对 PID、启动命令、模型路径和所有权 token，再关闭自己启动的服务并释放 GPU。受管服务持有 `<repo>/.mimics_runtime/locks/gpu.lock`，DINOv3 训练和推理会等待这把锁而不是同时争抢 CUDA 显存。如果 watchdog 自己崩溃，默认启动清理和 DINOv3 的 GPU 等待逻辑都只会在 state 文件、ownership token 和 idle timeout 同时满足时清理该受管 server。nnInteractive 官方的 `idle-timeout` 只回收 client session，本集成没有把它误当作服务退出机制。

`start_server.bat` 只用于人工诊断。默认自动管理模式下，诊断结束后应按 Ctrl+C 停止它再运行 Mimics；若确实要长期连接手工启动的服务，需要在 `nninteractive_config.json` 中显式设置 `auto_start_server: false`。自动模式不会接管或结束来源不明的进程；默认端口 `1527` 已被占用时，会为本次受管服务选择另一个空闲本地端口，并把实际地址写入状态和日志。

### 7.1 启动、等待与日志

Mimics 侧不再直接相信“某个 `python.exe` 文件存在”。第一次使用时会先检查外部解释器版本，并确认 `numpy`、`nibabel`、`torch` 和 `nnInteractive` 都能被发现。DICOM source-image fast path 还需要外部环境安装 `pydicom`。环境变量和配置文件仍可覆盖默认位置，但指向一个错误或不完整环境时会立即给出解释器路径和缺失包，不会继续走到 `ConnectionRefusedError` 才暴露问题。

设备默认值是 `auto`：

- 有可用 CUDA 时使用 `cuda:0`；
- 没有可用 CUDA 时自动回退到 CPU；
- 显式要求 CUDA 但工作站没有可用 GPU 时，默认也回退到 CPU，并在 Mimics Log Panel 和日志中记录；
- 只有设置 `allow_cpu_fallback: false` 时才会因为 CUDA 不可用而阻断。

fold 默认值也是 `auto`。bridge 让 nnInteractive 从实际存在的 `fold_*` 目录自动发现模型；只有一个 `fold_0` 时不会再错误地强制 `fold all`。

同一次 `nnInteractive` 工具运行只创建一个外部 worker 和一个远程 inference session。影像上传与 nnInteractive 预处理只执行一次；后续提示、Undo 和 Reset 使用 `reset_interactions()` 后重放提示，不再重复上传整幅影像。worker 会在标注者选择第一种提示前开始初始化，使模型加载尽量与交互操作重叠。

Windows 子进程使用无控制台窗口和独立进程组启动，因此正常操作不会再弹出 bridge、server 或 watchdog 黑色终端窗口，也不会继承 Mimics 所在控制台的 Ctrl+C 广播。

默认等待预算按阶段拆开：

| 阶段 | 默认上限 |
| --- | ---: |
| 外部环境检查 | 180 秒 |
| 模型服务首次启动 | 600 秒 |
| 影像上传与预处理 | 1800 秒 |
| 单次预测 | 1800 秒 |
| 兼容的一次性 bridge 总预算 | 4200 秒 |

可在 `nninteractive_config.json` 中分别调整 `environment_probe_timeout_seconds`、`server_startup_timeout_seconds`、`set_image_timeout_seconds`、`prediction_timeout_seconds` 和 `bridge_timeout_seconds`。不要只缩短总超时，否则 CPU 工作站可能在正常计算中被误判失败。

诊断信息固定写到模型目录同级的 `logs/`：

- `logs/nninteractive_mimics.log`：Mimics 侧启动、等待、设备选择和结果摘要；
- `logs/nninteractive_bridge.jsonl`：bridge 的阶段、实际解释器、设备、端口和 traceback；
- `logs/nninteractive_worker.stderr.log`：持久 worker 的标准错误；
- `.nninteractive_server.log`：模型服务启动、权重加载和服务端异常；
- `.nninteractive_server.json`：受管服务 PID、实际端口、设备、fold 和所有权信息。

Mimics 会自动打开 Log Panel，并在推理开始、CPU 回退和完成时写入用户日志。失败弹窗会直接给出失败阶段和上述日志路径，不再只显示 `WinError 10061`。

## 8. 代码结构

| 文件 | 职责 |
| --- | --- |
| `scripting_library/02_AI/nnInteractive/01_Annotate_Official_Model.py` | 官方模型标注入口 |
| `runtime_py35/nninteractive_mimics.py` | source/Draft 选择、提示采集、临时文件和结果写回 |
| `adapters/mimics/nninteractive_bridge.py` | 外部 Python 中加载图像、重放提示并调用 nnInteractive |
| `scripts/setup_nninteractive_env.py` | 在 Windows 上联网安装独立环境 |
| `scripts/build_nninteractive_bundle.py` | 在 Windows 上构建含环境、权重、bridge 和 Mimics 脚本的离线包 |

`nnInteractive` 入口不会导入平台 Console，也不会读取平台 runtime。

## 9. Windows 安装

### 9.1 联网安装

在项目根目录使用 Python 3.10+：

```powershell
python scripts\setup_nninteractive_env.py --cuda cu124 --device auto
```

安装脚本可恢复“权重已经下载、但虚拟环境不完整”的中断状态：重建 Python 环境时暂存并恢复 `nninteractive_env\models\`，不重新下载已有权重。

然后将 Mimics 的 Scripting Library 路径设为：

```text
<project>\adapters\mimics\scripting_library
```

### 9.2 离线包

必须在 Windows 机器上构建 Windows 包：

```powershell
python scripts\build_nninteractive_bundle.py
```

解压后，把 Mimics Scripting Library 指向：

```text
<extract-root>\nninteractive_env\mimics\scripting_library
```

虚拟环境、PyTorch wheel 和 Python 可执行文件不能从 macOS 直接复制为 Windows 运行环境。

### 9.3 与标注工作包合并（推荐）

平台操作者可以为标注者生成一个同时包含标注入口和 nnInteractive 的单一工作目录。标注者只需在 Mimics 中配置一次 Scripting Library 路径，就能同时使用六个 `Labeling_*.py` 标注脚本和 `nnInteractive` AI 工具。

**平台操作者执行**：

```powershell
# 1. 安装 nnInteractive 环境（只需做一次）
python scripts\setup_nninteractive_env.py --cuda cu124 --device auto

# 2. 导出工作包（自动包含 nnInteractive 脚本）
sp review export-worklist `
  --registry D:\platform_registry `
  --output-root D:\transfer\batch_001 `
  --limit 30
```

`export-worklist` 在检测到仓库中存在 nnInteractive 脚本时，会自动把它们复制进工作包。

**工作包目录（标注者收到）**：

```text
D:\transfer\batch_001\
  Labeling_Open_Next_Case.py
  Labeling_Case_Navigation.py
  Labeling_Submit_Complete.py
  Labeling_Submit_or_Report_Issue.py
  Labeling_View_Task_List.py
  Labeling_Save_Recovery_Backup.py
  nnInteractive.py              ← AI 工具入口（自动包含）
  nninteractive_bridge.py       ← Bridge 脚本（自动包含）
  runtime_py35/
    sp_common.py
    sp_open_review.py
    sp_review_console.py
    sp_save_checkpoint.py
    sp_submit_review.py
    nninteractive_mimics.py     ← AI 工具实现（自动包含）
  cases/
    case_001/
    case_002/
  worklist_manifest.json
  worklist_progress.json
```

外部 Python 环境和模型仍须单独放到标注者机器上。推荐把离线 bundle 解压到标注工作包的父目录下（脚本自动发现），或通过环境变量显式指定：

```powershell
setx NNINTERACTIVE_PYTHON "D:\nninteractive_env\python\python.exe"
setx NNINTERACTIVE_MODEL_DIR "D:\nninteractive_env\models\nnInteractive_v1.0"
```

如果离线 bundle 解压到工作包平级目录 `..\nninteractive_env\`，`nninteractive_mimics.py` 的自动发现机制会找到它。不设置环境变量也能运行。

**标注者**：在 Mimics 中 `File → Preferences → Scripting → Scripting Library` 指向 `D:\transfer\batch_001\` 即可使用全部 7 个入口。

## 10. 实机验证清单

在 Mimics Research 21 Windows 实机上依次验证：

1. Scripting Library 中出现 `nnInteractive`。
2. 任意非平台项目可以直接运行。
3. 选中已有 Mask 后结果写回正确 Mask。
4. 未选 Mask 时可创建新结果 Mask。
5. 一个 Point Set 中可混合多个 Include/Exclude 点；空 Mask 首轮只执行一次预测，已有 Mask 或后续修正逐点预测；Run、Discard 和异常退出后均无残留 Point。
6. Point 使用 `confirm=False` 时单击即可返回，没有额外 OK。
7. Ellipse Scribble 的正负语义正确；检查一次 Edit Masks 会话能否画多个区域。
8. Distance Measurement 两端点正确生成二维 bbox，测量对象随后被删除。
9. 开放 Spline 被拒绝；闭合、单切片 Spline 能生成完整 Lasso 轮廓。
10. 在 Ellipse/Spline/Distance Measurement 中 Cancel，确认脚本返回且不采用未确认提示。
11. Undo 只撤销最后一个提示事件；一个 Point Set 作为整体撤销。
12. Reset 恢复进入工具时的 Mask。
13. 切换 image set 后不会写错图像或 Mask。
14. 工具退出后没有残留 Prompt Mask、Spline 或 Distance Measurement。
15. 第一次和后续推理耗时可接受。
16. Mimics 保存、关闭、重新打开项目后结果仍存在。

还必须覆盖本次启动与等待修复：

- 临时把配置中的 `python` 指向一个没有 nnInteractive 的解释器，确认在连接服务前阻断，并显示错误解释器路径；
- 在无 NVIDIA GPU 或禁用 GPU 的机器上保持 `device: auto`，确认日志记录 CPU 回退且能够完成小体积预测；
- 使用只有 `fold_0` 的官方权重，确认服务命令没有 `--fold all`，能够自动加载；
- 预先占用 `127.0.0.1:1527`，确认受管服务选择其他本地端口且提示仍可完成；
- 强制关闭 Mimics 后重新打开，确认残留状态不会误杀无关 PID，旧服务不可复用时会安全重建；
- 连续添加两个提示，确认第二次没有重新执行 `set_image`/整幅影像预处理；
- 观察任务管理器，确认 bridge、server 和 watchdog 不弹出终端窗口；
- 分别查看 Mimics Log Panel、`logs/nninteractive_mimics.log`、`logs/nninteractive_bridge.jsonl`、`logs/nninteractive_worker.stderr.log` 和 `.nninteractive_server.log`，确认失败阶段可追踪。

在这些检查通过前，应把功能视为工程验证版，而不是生产标注能力。

## 11. 已知限制

- Mimics 21 公开 Python API 不能注册原生 Segment 工具栏图标。
- `activate_edit_mask()` 文档没有声明 Cancel 时抛出的异常类型。代码处理 `mimics.UserInterrupted`、正常返回空 Mask 和其他异常，并始终删除 Prompt Mask 和临时目录；但 Cancel 是否可能正常返回带部分编辑的 Mask，必须实机验证。
- Mimics 21 未公开自由画笔 API；Scribble 目前由 Ellipse Draw 区域近似。
- Box 和 Lasso 的 Mimics 入口只暴露前景语义；负向修正使用 Exclude Point/Scribble。
- Spline Lasso 只接受轴对齐二维切片，不把任意三维曲线近似为 Lasso。
- Mimics 21 API 明确公开 `Spline.geometry_points` 和 `Spline.points`。若运行时两者都不可读或不足两个不同体素点，脚本会显示明确错误，不再静默忽略。
- 一个工具会话内提示可撤销；退出后只保留结果，不保留提示历史。
- 推理期间 Mimics 会等待外部 bridge 返回。
- 自动启动的服务使用本机 bearer token，并只复用带匹配所有权记录、模型目录、设备和 fold 配置的进程；默认端口被其他服务占用时会改用空闲本地端口，不会杀死来源不明的进程。
- Mimics Research 21 的公开 Python API 没有可更新的原生进度条，也没有说明 `timer` 事件的调度周期和脚本返回后的生命周期。为了在预测完成后立即安全写回 Mask，当前版本仍在单次预测期间同步等待。代码通过隐藏终端、提前启动 worker、复用影像预处理、扩大分阶段超时和显示 Log Panel 降低影响；真正的后台推理加自动写回必须经过 Windows 实机验证后再启用，不能仅凭未说明语义的 `timer` 回调实现。
- 无法识别的 Mimics voxel buffer dtype 会阻断并要求补充工作站验证，不会默认猜成 `int16`。
- 当前官方模型权重的具体许可必须以实际下载版本携带的 license 为准，不能仅根据代码仓库许可推断用途。

## 12. 图像强度、预处理和后处理

### 12.1 Mimics 输出的强度

Mimics 21 API 文档明确说明：

- `ImageData.get_voxel_buffer()` 返回 16-bit Gray Value 三维数组；
- Mimics Python API 中涉及图像强度的方法统一使用 Gray Value；
- `mimics.segment.GV2HU()` 和 `HU2GV()` 用于 Gray Value 与 HU 的转换。

因此该 buffer 既不能简单称为原始 DICOM stored value，也不是直接的 HU 数组。

官方模型的 Mimics-buffer 路径原样传递 Mimics Gray Value，不主动转换成 HU。自定义任务模型则会根据可用的项目强度 metadata 恢复其训练使用的 source value。nnInteractive 2.4.2 在 `set_image()` 后会：

1. 把图像转换为 float；
2. 查找非零区域；
3. 用非零区域的均值和标准差对整幅图像做 z-score；
4. 不在这一阶段按 spacing 对整幅图像重采样。

若 Gray Value 与源值是正向线性关系，而且非零空间包围盒不变，z-score 后的值理论上相同。零背景一旦因为 intercept 或量化变成非零，统计区域可能扩大到整幅图像，影响就可能很大。因此自定义 MRI 模型不能只依赖“z-score 会抵消”这一假设。

Source-image fast path 不读 Mimics buffer，而是由外部 Python 读取原始 NIfTI 或 DICOM。该路径会把 source 图像重采样到 Mimics voxel grid 后，再按 ImageData metadata 选择强度空间：当前 NIfTI 导入流程会写成 CT 派生 DICOM，因此使用 Mimics 侧传入的 `HU2GV(0)` 与 `HU2GV(1)` 推导出的线性 HU-to-GV 转换；原始 CT DICOM 也使用该转换；明确的非 CT DICOM 默认保留 source values，避免把 MR 等数据的背景零值错误转换为非零。这样 source path 与 Mimics-buffer path 尽量保持同一 Gray Value 语义，同时避免对非 CT 数据做错误 HU 假设。

仍需实机验证：

1. 在已知 CT 位置读取 `image.get_grey_value()` 和 voxel buffer；
2. 用 `GV2HU()` 转换并与外部 DICOM Rescale Slope/Intercept 结果对比；
3. 记录 `GV2HU(0)`，确认零值在该项目中的物理含义；
4. 比较 Mimics buffer 与外部 NIfTI 在 z-score 后的数值分布；
5. 检查 Pixel Padding、截断、饱和或导入转换是否造成非线性差异。

只有发现非线性差异或零值区域明显不一致时，才应在 bridge 中增加更复杂的项目级强度校正。直接假设“必须把 Mimics buffer 转 HU”并不严谨；但 source-image fast path 必须转换到 Mimics GV，才能和 Mimics API 返回的 buffer 保持一致。

### 12.2 nnInteractive 原生预处理

当前实现固定安装并验证 `nninteractive==2.4.2`，避免 `>=` 版本漂移改变内部行为。

nnInteractive 的推理预处理包括：

- 非零区域统计和全图 z-score；
- 以最新提示为中心裁剪局部 patch；
- 超出图像范围时做 constant padding；
- 图像使用 trilinear resize 到模型 patch size；
- previous segmentation 使用 nearest resize；
- Scribble/Lasso 等交互通道使用 area resize；
- 对可能在下采样中消失的细提示先做 dilation；
- AutoZoom 在预测变化触及 patch 边界时逐步扩大视野。

这意味着调用方不应再次自行做固定 spacing 重采样、强度标准化或 patch 裁剪，否则会与模型内部逻辑叠加。

### 12.3 nnInteractive 原生后处理

nnInteractive 2.4.2 的默认输出路径包括：

- 对网络输出做 `argmax` 得到离散预测；
- AutoZoom 后把 coarse prediction 缩放回对应区域；
- 二值 coarse mask 使用 trilinear resize 后以 `0.5` 阈值离散化；
- 根据 coarse prediction 与 previous segmentation 的差异规划 refinement patches；
- 在原图坐标空间把局部 refinement 结果写回 target buffer。

默认推理代码没有执行通用的最大连通域、孔洞填充、器官拓扑约束或解剖学冲突处理。若具体任务需要这些规则，应作为明确、可关闭的后处理层单独设计，不能假设 nnInteractive 已经处理。

## 13. Current Mimics Integration Behavior

### 13.1 Prompt-Time Responsiveness

The Mimics foreground process must still call Mimics APIs such as
`get_voxel_buffer()` and `set_voxel_buffer()` on the Mimics side. These calls
cannot be made fully asynchronous without risking Mimics object lifetime and
thread-safety issues.

The integration therefore moves the expensive nnInteractive work out of Mimics
and starts image-session preparation before prompt capture. The foreground
Mimics process only performs the unavoidable buffer export/import operations,
then external Python loads the model, uploads the image, preprocesses it and
runs inference.

`mimics.update_gui()` is used only to let Mimics process pending UI messages
before or after a short foreground operation. It does not make a long Mimics API
operation asynchronous. `disable_update_gui()` is used only around short
`set_voxel_buffer()` writes so partial mask refreshes do not cause flicker or
black-screen style redraw stalls. It must always be paired with
`enable_update_gui()` in `finally`.

### 13.2 First Prediction Latency

The first prediction can still be slower than later predictions because the
external worker may need to start the nnInteractive server, load the model,
upload the full Mimics image buffer and let nnInteractive run its native
`set_image()` preprocessing. Later prompts reuse the same image worker when
possible, so they should not repeat full image upload/preprocessing.

The default async worker idle timeout is now 3600 seconds. `idle_timeout` means
that the prepared AI worker waited too long without producing a prompt result
and exited to release resources; it is not itself a prediction failure reason.
If a ready worker expires before the next prompt, the next session starts again
from the current Mask.

### 13.3 Existing Mask Semantics

If a non-empty manual Mask is selected, it is exported as the initial
segmentation snapshot and a separate `<name> - AI Draft` receives predictions.
The source Mask is never overwritten. Selecting an existing AI Draft or an
empty Mask continues in place. Each prediction resets the nnInteractive
interactions, applies the immutable initial snapshot, then replays all prompts.

The bridge passes Mimics Gray Value voxels as a raw float32 image array to
nnInteractive. It does not apply an extra HU normalization, fixed-spacing
resampling or patch crop on the Mimics side. Those operations belong to
nnInteractive's native `set_image()` and inference pipeline; duplicating them in
the integration would change model behavior.

### 13.4 Scripting Library Layout

The Scripting Library entries are organized by workflow:

- `01_Data`: dataset import and mask export.
- `02_AI`: nnInteractive and DINOv3 few-shot actions.
- `03_Review`: mask identification, window/level presets, undo and reset.
- `99_Admin`: background-service cleanup.

Visible entries are intentionally thin wrappers. They all go through
`runtime_py35/_mimics_entrypoint.py`, which performs runtime path setup and
loads the target runtime module. The wrappers do not call `importlib.reload()`
on every click, because reload can discard in-memory monitors for background
inference or training jobs while those jobs are still expected to report back to
Mimics.

DINOv3 exposes one external `Train Model` setup entry, separate recommended
and explicit-model prediction intents, `Show Status Results`, and `Stop AI
Task`. This preserves one-click common prediction without a mode popup.

### 13.5 Source-Image Fast Path For Prewarming

Prewarming must not simply move a long Mimics GUI stall from after the prompt
to before the prompt. The expensive image load, model load, image upload and
nnInteractive preprocessing should happen in the external Python worker.

For `.mcs` files created by the batch import pipeline, the background Mimics
creator stores source image metadata on the imported ImageData:

- `mimics_script.source_image_path`
- `mimics_script.source_image_kind`
- `mimics_script.source_image_shape`
- `mimics_script.source_image_index_space`
- `mimics_script.source_image_modality`
- `mimics_script.source_world_coordinate_system`
- `mimics_script.mimics_world_coordinate_system`
- `mimics_script.source_to_mimics_world_matrix`
- `mimics_script.source_voxel_to_ras_matrix`
- `mimics_script.mimics_voxel_to_ras_matrix`
- `mimics_script.mimics_to_source_index_matrix`

When this metadata points to a valid source NIfTI or DICOM folder, and its
index-space contract plus world-coordinate contract are complete,
`nnInteractive` sends `image_path`, `image_source_kind`,
`image_expected_shape`, `interaction_shape`, the RAS/LPS metadata, source
modality and the selected source-to-Mimics intensity transform to the external
bridge. The recorded source shape may differ from the open Mimics image shape:
the bridge uses the stored source and Mimics voxel-to-RAS matrices to resample
the source image into the actual Mimics grid before inference. Mimics then skips
`ImageData.get_voxel_buffer()` and does not write a full raw image buffer on the
foreground GUI thread.

The fast path is based on Mimics voxel index space, not only patient view labels.
Prompts collected through `image.get_voxel_indexes()` are `(x, y, z)` indexes.
For NIfTI imported through this pipeline, the image may undergo a lossless axis
permutation or flip while it is oriented for DICOM/LPS. This changes voxel index
order but does not interpolate image values or change physical locations. The
source NIfTI shape and RAS affine and the derived-DICOM/Mimics grid are recorded
separately. NIfTI masks are read in their original on-disk order with their RAS
affine and are mapped once into the actual Mimics grid using physical
coordinates; they are not independently reoriented to LPS and then transformed
a second time. For DICOM source folders, source and Mimics world
coordinates are both LPS; the external worker reads the selected series, sorts
slices by `ImagePositionPatient` along the slice normal, transposes each pixel
plane from `(Rows, Columns)` to `(Columns, Rows)`, and stacks the result as
`(Columns, Rows, Slices)`. If the source shape differs and affine metadata is
available, the worker resamples into the open Mimics image shape; if multiple
DICOM series are present and no shape match can identify the intended series,
the worker fails instead of guessing. Internally, the bridge first aligns source
data to the Mimics grid, applies the selected source-to-Mimics intensity
transform, then applies the configured Mimics-to-platform axis
mapping before calling nnInteractive. Shape checks compare platform shape to the
mapped Mimics shape, so non-identity axis mappings are not rejected by comparing
two different coordinate systems.

For imported `.mcs` projects, the background `.mcs` creator derives the actual
Mimics voxel grid after DICOM import by calling `ImageData.get_voxel_center()`.
Mimics/DICOM patient coordinates are LPS, so the stored matrix is converted to
RAS for consistent math with nibabel NIfTI affines. If this actual grid differs
from the prepared grid, masks are mapped directly from their original files into
the measured Mimics grid before `Mask.set_voxel_buffer()` is called; the
intermediate prepared buffer is not resampled a second time. The creator
does not blindly reshape equal-size buffers because that can create diagonal
mirrors or swapped axial/coronal/sagittal views.

If the source image voxel grid differs from the Mimics voxel grid but both
`source_voxel_to_ras_matrix` and `mimics_voxel_to_ras_matrix` are present, the
external worker computes `source_index = inv(source_voxel_to_RAS) *
mimics_voxel_to_RAS * mimics_index` and resamples the source image into the
Mimics voxel grid in the background. In that case the tool does not fall back to
foreground Mimics buffer export merely because the source orientation differs.
It falls back or fails only when the required geometry is missing or ambiguous
for example a DICOM folder containing multiple possible series.

Mask-to-image resampling inside the conversion bridge uses RAS affines
internally. NIfTI mask affines from nibabel are already RAS, and DICOM image
geometry is converted from LPS to RAS before resampling. The bridge therefore
does not convert mask affines to LPS during resampling; doing so would mix
coordinate systems and can create left-right/anterior-posterior mirroring.

The derived classic-DICOM writer preserves axial, coronal, sagittal, oblique,
and regular gantry-tilt stacks without image interpolation. Gantry tilt is
represented by the full per-slice `ImagePositionPatient` step, including its
in-plane component. Automatic image resampling is reserved for a genuinely
non-orthogonal in-plane row/column basis, which classic DICOM pixel geometry
cannot represent. With `MIMICS_DICOM_RESAMPLE_MODE=never`, that unsupported
geometry is rejected instead of emitting a geometrically invalid DICOM series.
If Mimics itself normalizes a valid imported stack, the
background creator measures the live Mimics voxel centers and maps source masks
directly to that measured grid.

If source metadata is absent, unsupported, or points to a file that does not
exist on the current workstation, the official model follows `image_input_mode`.
Custom task models use the separate `task_model_image_input_mode` policy:

- `auto` (default) prefers a readable local source image and otherwise exports
  the image already embedded in the open `.mcs`;
- `source` is strict and stops if the source file cannot be used;
- `mimics` always uses the project image buffer.

In `auto` mode, recorded UNC/network paths are not synchronously probed from the
Mimics GUI thread. This prevents a moved project or unavailable share from
freezing prompt collection. A short `get_voxel_buffer()` snapshot is made before
the prompt; model loading, image upload, canonical-RAS orientation and
preprocessing continue in the external image worker.

The custom-model fallback keeps the exact open Mimics geometry. CT buffers are
converted back through the available HU/GV linear transform. New MR imports
persist their derived-DICOM rescale mapping and value ranges in `.mcs` metadata.
The external worker reconstructs approximate source values and snaps residuals
within half a quantization step back to zero, preserving nnInteractive's nonzero
normalization crop. Older projects try the retained DICOM Rescale Slope and
Intercept tags. If neither mapping exists, inference still runs with raw GV and
records `raw_gv_zscore_best_effort`; it never fails merely because the mapping is
missing. The runtime records `image_input_provenance`, recovery basis and
`image_intensity_compatibility` in job and bridge logs.

The managed nnInteractive server is kept warm until its configured idle timeout
or until `Stop Background Services` is run. This avoids turning every prompt
sequence into a cold model-load path. Mimics logging reports timing breakdowns
for image load/resampling, server readiness, `set_image`, `set_target`, prompt
application and total elapsed time so slow runs can be diagnosed without opening
raw JSON logs.

Before-prompt prewarm supports both a validated source image and the portable
Mimics-buffer fallback. Mimics performs one short local buffer snapshot; source
recovery, canonical orientation, model loading, upload and preprocessing then
continue in the external worker while the user collects prompts.

### 13.7 DINOv3 候选点引导

`02_AI/nnInteractive/04_DINOv3_Guided_Points.py` 提供一个有人工确认的组合入口：DINOv3 不直接覆盖 Mask，而是根据 TTA 前景概率提出最多 3 个自动前景点和 3 个自动背景点。最大 26-连通区域必保留；额外区域只有在相对体积、物理距离、概率和 TTA 一致性均可信时才保留，最多两个。前景点来自高概率、低方差的区域深部，背景点只来自所有预测区域外的稳定低概率壳层；弱候选不会通过宽松回退被强行变成提示。

建议点以 RAS 世界坐标写出，Mimics 侧显式转换为 LPS 后创建临时 Point。标注者可以在 Mimics 中移动、添加或删除这些点；自动建议的 1～3 点上限不限制手工点数。确认时会重新读取 Point 的当前位置，而不是使用最初的数组索引；因此用户修改后的坐标才是实际送入 nnInteractive 的坐标。

审核窗口可以选择官方模型或当前任务下任一完整且支持 point prompt 的微调模型。确认后复用既有异步 image worker、initial Mask、结果目标选择、undo/reset 和生命周期管理。一个 `point_set` 内的多个点在 bridge 中按顺序执行，每一步预测都继承上一轮结果。取消审核会删除临时 Point，不修改选中 Mask，也不会保留 GPU 锁；成功提交后，Point 会在推理结果应用或任何终止路径自动删除，DINO 临时 Mask 与建议 JSON 也会清理。

Empty initial Masks use an empty-mask state marker instead of exporting a full
zero-valued mask buffer. Undo and reset restore such sessions with `mask.clear()`.
Existing non-empty Masks are still exported as the initial segmentation because
nnInteractive needs that current user-edited state.

## 14. References

- [nnInteractive 官方仓库](https://github.com/MIC-DKFZ/nnInteractive)
- [nnInteractive server-client 文档](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md)
- [nnInteractive API v2 变更说明](https://github.com/MIC-DKFZ/nnInteractive/blob/master/API_CHANGES_v2.md)
- [nnInteractive 2.4.2 on PyPI](https://pypi.org/project/nninteractive/2.4.2/)
- [Mimics Research 21 Python API 本地手册](../references/mimics/api_21/Mimics_API_Documentation.md)
