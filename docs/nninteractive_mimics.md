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
2. 读取或创建目标 Mask；
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
Script -> Scripting Library -> nnInteractive
```

这已经是单击启动，不需要打开 Editor、Console 或命令行。若以后要进入 Mimics 原生分割工具栏，需要单独向 Materialise 确认扩展 SDK 或厂商支持，不能把它当作 Python API 已有能力。

## 4. 目标 Mask 的选择

运行前，用户在 Project Tree 中选择一个目标 Mask：

- 恰好选择一个 Mask：在该 Mask 上继续分割或修正；
- 没有选择 Mask：脚本可创建新的 `nnInteractive Result`；
- 选择多个 Mask：脚本停止并要求只选一个，避免把结果写错对象；
- 目标 Mask 不属于 active image：脚本停止，不做隐式跨图像绑定。

已有 Mask 会作为 nnInteractive 的 initial segmentation。空 Mask 则从提示开始生成新分割。

## 5. Mimics 交互与 nnInteractive 提示的映射

### 5.1 已实现的提示

| nnInteractive 提示 | Mimics 采集方式 | 转换方式 |
| --- | --- | --- |
| Include/Exclude point set | 多次 `mimics.indicate_coordinate(confirm=False)` | 一次收集任意数量的正负点；最后统一触发一次预测 |
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

Mimics 集成的外部 bridge 每次调用都会关闭 remote session，因此不能直接保留服务器端 session。当前实现采用等价的有序重放：

1. 进入工具时保存一次目标 Mask，作为本次工具会话的 initial segmentation；
2. 每次新增提示后，新建 remote session；
3. 注入同一份 initial segmentation；
4. 按原顺序重放此前所有提示，每个提示事件保持原来的预测边界；
5. 把最终 target buffer 写回 Mimics Mask。

因此连续修正不是“把刚预测出的 Mask 再作为新的 initial segmentation”。后者会在每次提示时重置交互通道并累积模型误差。只有退出工具并重新启动时，当前 Mimics Mask 才成为下一次工具会话的新 initial segmentation。

Point Set 是一个提示事件：多个正负点以 `run_prediction=False` 写入，最后一个点触发一次预测。多切片 Scribble 同理，所有二维 crop 写入后只触发一次预测。

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
- 最终结果只保留在用户选中的 Mimics Mask 中。

重新启动 nnInteractive 时，当前 Mask 成为新的 initial segmentation。上一会话的提示历史不会继续保留，但分割结果不会丢失。

## 7. 用户工作流

1. 在 Mimics 中打开任意项目。
2. 激活要处理的 image set。
3. 在 Project Tree 选择一个目标 Mask；也可以不选，由脚本创建新 Mask。
4. 运行 `Script -> Scripting Library -> nnInteractive`。
5. 选择 **Add Points**、**Paint Scribble**、**Draw Box** 或 **Draw Lasso**。
6. Add Points 中可连续加入 Include/Exclude 点，绿色/红色标记会保留到 Run/Discard；必要时使用 Remove Last Point。
7. Paint Scribble 只需选择一次 Include 或 Exclude，然后在临时 Mask 中绘制。
8. Box 和 Lasso 默认为前景，不再显示正负选择。
9. 脚本调用外部 nnInteractive，结果自动写回目标 Mask。
10. 继续增加提示，或使用 **Undo Last Prompt** / **Reset To Start**。
11. 选择 **Finish**，按正常 Mimics 方式保存项目。

第一次推理需要启动模型服务，通常比后续提示慢。脚本会复用正在运行的本机服务。

服务由 bridge 创建并记录所有权，不会根据一个来源不明的 PID 直接终止进程。每次提示都会刷新活动时间；默认连续 30 分钟没有推理请求后，独立 watchdog 会核对 PID、启动命令、模型路径和所有权 token，再关闭自己启动的服务并释放 GPU。nnInteractive 官方的 `idle-timeout` 只回收 client session，本集成没有把它误当作服务退出机制。

`start_server.bat` 只用于人工诊断。默认自动管理模式下，诊断结束后应按 Ctrl+C 停止它再运行 Mimics；若确实要长期连接手工启动的服务，需要在 `nninteractive_config.json` 中显式设置 `auto_start_server: false`。自动模式不会接管或结束来源不明的进程；默认端口 `1527` 已被占用时，会为本次受管服务选择另一个空闲本地端口，并把实际地址写入状态和日志。

### 7.1 启动、等待与日志

Mimics 侧不再直接相信“某个 `python.exe` 文件存在”。第一次使用时会先检查外部解释器版本，并确认 `numpy`、`nibabel`、`torch` 和 `nnInteractive` 都能被发现。环境变量和配置文件仍可覆盖默认位置，但指向一个错误或不完整环境时会立即给出解释器路径和缺失包，不会继续走到 `ConnectionRefusedError` 才暴露问题。

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
| `adapters/mimics/scripting_library/nnInteractive.py` | Mimics Scripting Library 独立入口 |
| `adapters/mimics/runtime_py35/nninteractive_mimics.py` | 目标选择、提示采集、临时文件和结果写回 |
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
5. 一个 Point Set 中可混合多个 Include/Exclude 点，并且只执行一次预测；Run、Discard 和异常退出后均无残留 Point。
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

当前集成原样传递 Mimics Gray Value，不主动转换成 HU。原因是 nnInteractive 2.4.2 在 `set_image()` 后会：

1. 把图像转换为 float；
2. 查找非零区域；
3. 用非零区域的均值和标准差对整幅图像做 z-score；
4. 不在这一阶段按 spacing 对整幅图像重采样。

若 Gray Value 与 HU 是正向线性关系，z-score 后的值理论上相同。直接转换 HU 通常不会改变模型输入，反而可能改变“哪些体素等于零”的判定。

仍需实机验证：

1. 在已知 CT 位置读取 `image.get_grey_value()` 和 voxel buffer；
2. 用 `GV2HU()` 转换并与外部 DICOM Rescale Slope/Intercept 结果对比；
3. 记录 `GV2HU(0)`，确认零值在该项目中的物理含义；
4. 比较 Mimics buffer 与外部 NIfTI 在 z-score 后的数值分布；
5. 检查 Pixel Padding、截断、饱和或导入转换是否造成非线性差异。

只有发现非线性差异或零值区域明显不一致时，才应在 bridge 中增加显式强度转换。现在直接假设“必须转 HU”并不严谨。

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

## 13. 依据

- [nnInteractive 官方仓库](https://github.com/MIC-DKFZ/nnInteractive)
- [nnInteractive server-client 文档](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md)
- [nnInteractive API v2 变更说明](https://github.com/MIC-DKFZ/nnInteractive/blob/master/API_CHANGES_v2.md)
- [nnInteractive 2.4.2 on PyPI](https://pypi.org/project/nninteractive/2.4.2/)
- [Mimics Research 21 Python API 本地手册](../references/mimics/api_21/Mimics_API_Documentation.md)
