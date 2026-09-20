# Mimics 外部工具标注能力扩展 — 痛点深挖分析

> 基于代码事实调研（2026-09-19）。所有痛点均对应实际实现，非猜测。
> 调研覆盖：导入导出、nninteractive 原始分割、nninteractive 微调、外部进程架构。

---

## 系统在干什么（所有人能懂的版本）

Mimics 是商业医学影像标注软件，本身能看图、勾画 mask、存成自己的 `.mcs` 工程文件。但它的 AI 能力几乎为零，支持的文件格式也少。所以我们做了一套"外挂"：

- Mimics 里嵌着一个很老的 Python（3.5，只能用基础功能，装不了 AI 库）；
- 我们在它旁边放了另一个完整的 Python（3.13，能跑 PyTorch、AI 模型）；
- Mimics 把活儿"递条子"给这个外部 Python 干，干完了再把结果拿回来。

这套"递条子"的机制，是后面所有痛点的总根源。

---

## 专题一：数据的导入导出

**现状纠正**：导入其实不只支持 DICOM。NIfTI、MHA、MHD、NRRD 都能进，mask 也能进，一图一 mcs 设计扎实（原子写、崩溃恢复、几何对齐校验都做了）。导出目前只有 `.nii.gz`。**痛点不在"完全没做"，而在"做得不够顺手、不够通用、不够面向标注员"。**

### 痛点 1.1：导入流程对标注员太重，不是"打开就用"
导入一个病例要走：外部弹窗 → 选目录/文件 → 选 mask 来源 → 确认 → 等后台 Mimics 批量创建 mcs。**完全没有拖拽**（全库零匹配）。

- **产品视角**：流程通但门槛高。"拖进去就打开"是任何看图软件的本能预期，做不到就显得难用。
- **标注者视角**：每次导入面对复杂外部窗口，要懂"图像根目录""mask 来源""输出目录"等概念，学习成本高、步骤多、出错不知哪一步错。
- **开发者视角**：拖拽缺失是 Mimics 脚本 API 拿不到原生拖拽事件所致，只能走外部窗口。补偿方案把"配置成本"全压给了用户。

### 痛点 1.2：对"特定数据集布局"有硬编码依赖，换数据集就别扭
代码写死"TS-like"约定：首选文件名 `ct.nii.gz`/`mri.nii.gz`，mask 固定从 `segmentations/` 子目录找，DICOM 固定找 `dicom/` 子目录。有兜底扫描，但约定泄漏到通用工具里。

- **产品视角**：通用性打折。换数据集结构靠兜底扫描，不报错但行为不可预期。
- **标注者视角**：最怕"不报错但选错了"。兜底扫描选错图、配错 mask，可能标了半天才发现底图不对。
- **开发者视角**：首选名/子目录约定散落在至少 4 个文件（import/export/bridge/UI），改一处同步四处，可维护性差。

### 痛点 1.3：导出格式只有 nii.gz，且强依赖"源图像元数据"
导出只能出 `.nii.gz`，且**强制要求**从 mcs 读回导入时的源图像路径和几何矩阵才能还原回源网格。元数据丢失则导出直接失败。

- **产品视角**：格式覆盖窄。下游可能要 DICOM-RT、STL、mhd，现在一条都没。强依赖源元数据意味着"不能脱离原始数据导出"。
- **标注者视角**：导出失败提示"original source image is required"——看不懂为什么导出还要原图。
- **开发者视角**：几何对齐必要，但"无源元数据则拒绝导出"是过严契约，应允许"按 Mimics 当前网格直接导出"作降级路径。

### 痛点 1.4：可回退性是流程级的，不是操作级的
系统在"不产生半成品文件"上做得好（原子写、崩溃隔离），但**没有面向标注员的"撤销本次导入/导出"**。

- **产品视角**：原子性保证文件系统不脏，不保证用户操作可逆。缺"导入事务"概念。
- **标注者视角**：标错能 Undo（Mimics 原生），但"导入错了一堆 mask"没有一键撤回。
- **开发者视角**：技术上可做（记录本次导入创建的 mask 列表，提供回滚），目前没做。

---

## 专题二：nninteractive 原始权重分割

**现状**：工程化做得非常重——异步 worker、HTTP 服务、看门狗、GPU 锁、崩溃恢复、错误分类都有。**痛点在于"为了稳，把交互做重了"。**

### 痛点 2.1：点击层级太多，每个提示都要重新跑入口
最简单的"点一下出结果"要点 4-5 次：菜单跑脚本 → 选提示类型 → 采集点 → 运行。**每加一个提示，要重新从菜单跑一遍入口**。同步模式被禁用（卡 GUI），异步后没法常驻后台写回通道。

- **产品视角**：体验硬伤。同类产品是"点一次进入交互模式，连续点点连续出结果"，我们像"每次都要重新启动工具"。
- **标注者视角**：频繁点击 = 累 = 慢 = 出错。一天点几百次菜单，疲劳导致漏点、错点。
- **开发者视角**：根因是 Mimics 脚本 API 没有原生后台写回，只能 timer 轮询。异步架构对，但"每提示重跑入口"可优化——让 worker 常驻、Mimics 侧维护轻量"继续提示"循环。

### 痛点 2.2：环境问题对标注员完全是黑盒，坏了就停摆
torch/cuda/nnunetv2 版本门禁一堆，但**环境出问题标注员既看不懂也修不了**。错误有 stage 和日志路径，但对非技术用户是天书。环境探测只记录不阻断，真正炸在 worker 启动时才暴露。

- **产品视角**：环境是基础设施，不该是标注员日常故障源。
- **标注者视角**："点了一下没反应/弹英文错误"——只能喊人。Setup_Repair 藏得深，修不好不知下一步。
- **开发者视角**：探测到位，但"探测到缺包"和"引导修复"之间断了。probe 应直接驱动修复流程。

### 痛点 2.3：跨机型/跨用户部署仍然重
有 setup_offline.bat 和打包工具，但 venv 不可移植（打包默认不含 Lib），换机器要重建环境。torch wheel 带 cu124 对显卡有要求。

- **产品视角**：多站点多标注员规模化部署，每台机都要走环境搭建，成本高。
- **标注者视角**：新机器开箱到能用，中间隔"装环境"的墙。
- **开发者视角**：已尽力做离线包，但 Python 生态可移植性天花板就在这，架构层约束。

### 痛点 2.4：显存没有预检查，OOM 是"标到一半炸"
没有显存容量预检查（全库无 mem_get_info）。显存不够时推理跑到一半 OOM，worker 退出，标注员看到"nnInteractive Failed"。

- **产品视角**：应在"启动任务前"就告知"显存跑不了"，而非"跑到一半失败"。
- **标注者视角**：标了十分钟崩了，前面交互状态丢失要重来，可回退性差。
- **开发者视角**：技术上可做（torch.cuda.mem_get_info 预估 + patch size 自适应），目前空白。

---

## 专题三：nninteractive 微调

**现状**：四层进程、数据回流、参数约束、训练监控、权重注册、任务路由全做了。**痛点在"复杂度全部暴露给少数能操作它的人，且闭环仍有断点"。**

### 痛点 3.1：微调数据回流链路长，任一环几何不对齐就硬失败
数据从 mcs 标注 → 后台导出 → DICOM 转 NIfTI → mask 重采样到图像网格 → canonical 校验 → 缓存 → 训练。**任一步 shape/affine 不一致直接抛错**。

- **产品视角**：几何严格对，但错误发生在链路深处，用户看不到全链路，失败只看到技术错误不知哪步怎么修。
- **标注者视角**：基本碰不到，一旦失败反馈是"训练起不来，原因看不懂"。
- **开发者视角**：每个拒绝点带修复指引文案，但链路串行跨进程，中间任一进程崩溃要跨多个 job 目录翻日志。可观测性是"有日志但分散"，非"一条链路一目了然"。

### 痛点 3.2：微调参数刻意收窄到 6 项，但"为什么这么设"用户不知道
UI 只暴露策略、epoch、训练目标、起始模型、镜像、数据选择。学习率/batch/patch 全固定。有意设计但**用户不知道固定值含义和边界**，想调没地方、想理解没解释。

- **产品视角**：收窄降门槛是好事，但缺"这些参数意味着什么"的说明层。对研究型用户收窄成限制。
- **标注者视角**：基本不微调，主要是技术用户用。"epoch 4-20"是个数，"为什么不能 30"没解释。
- **开发者视角**：固定超参记进 manifest 是好实践。但配置项散在 nninteractive_finetune_config.json，**没收录进 CONFIG_REFERENCE.md**——文档缺口。

### 痛点 3.3：训练监控做了三层镜像，但"训练到底好不好"对非专家不直观
有 trainer_status → pipeline status → UI 曲线，画了 validation AUC。但**指标是 trajectory AUC（多步点击轨迹平均 Dice）**，对非 ML 背景难理解。"AUC 0.56 好不好？要不要重训？"无参照系。

- **产品视角**：监控数据全，缺"翻译层"——把 AUC 翻译成"这次比原来好/差，建议采用/重训"。
- **标注者视角**：想知道"新模型会更准吗"，不是"AUC 是多少"。
- **开发者视角**：已做基线 vs 候选对照（delta_auc、严重回退检测、自动选用），是四专题里最完整的，痛点主要是"呈现给非专家"不够。

### 痛点 3.4：权重管理和任务路由解决了，但"权重从哪来、能不能用"仍需用户操心
有权重注册表、sha256 校验、任务→模型路由。但**权重"来源可信度"和"适用范围"对用户不透明**——not_improved 或 corrupt 权重还在注册表里，用户可能误选。

- **产品视角**：路由自动化是好方向，但"推荐模型"和"历史失败模型"混在一张表，需更强"只给你能用的"过滤。
- **标注者视角**：标注时不想选模型，想"系统自动给最合适的"。歧义时弹 chooser，不知怎么选。
- **开发者视角**：registry 设计合理。但 corrupt/incompatible 权重应自动归档或隐藏，而非和可用权重并列。

### 痛点 3.5：微调出 bug 的排障门槛极高
一个 job 目录有 job.log / trainer.log / status.json / request.json / export_config.json / staging/。**要懂这套结构才能定位问题**。远程训练还多一层 SSH/Docker 日志。

- **产品视角**：没"一键诊断失败任务"入口，出问题靠人翻目录。
- **标注者视角**：完全碰不了。
- **开发者视角**：日志保留策略好。但缺"聚合视图"——把一个 job 的所有日志/状态/错误并排呈现的工具。

---

## 专题四：架构设计（外部进程管理）

**最该反思的一块。** 进程生命周期管理做得极重——ownership token、heartbeat、PID+启动时间防回收、看门狗、supervisor、三套杀进程路径。**这些复杂度不是过度设计，而是"松散外部进程 + 文件中转"架构自带的税。问题是：这个架构是不是最优解？**

### 痛点 4.1：文件中转通信是脆弱性总根源
Mimics ↔ 外部 Python 全是 JSON 文件 + 定时轮询，没有 RPC、没有 socket 长连接（除 nninteractive HTTP）。意味着：没有主动通知（靠 timer 翻文件）、状态会漂移（写一半被读、轮询窗口错位）、延迟天然存在（0.25-2s 轮询粒度）。

- **产品视角**：用户感知的"慢""卡""偶尔没反应"，很多不是计算慢，是通信慢。
- **标注者视角**：点之后等零点几秒才有反应，累积就是"这软件不跟手"。
- **开发者视角**：文件中转优点是**崩溃可恢复**（状态在磁盘，进程死了重启能接着读），这是比 RPC 强的地方。代价是上述脆弱性。真实架构权衡，不是 bug。

### 痛点 4.2：进程生命周期管理"很重但很散"
PID 防回收、token 鉴权、heartbeat、watchdog、supervisor 都做了，但**分散在多文件，靠约定维系一致性**。杀进程有三套路径（ctypes / PowerShell taskkill / reaper 线程），语义一致靠人记。外部 Python 发现逻辑在至少 7 个文件重复，路径列表有细微差异。

- **产品视角**：对用户不可见，但维护成本高 = 改进慢 = 新痛点修得慢。
- **标注者视角**：间接受害——架构散，修一个 bug 可能漏改另一处，导致"明明修了怎么还出问题"。
- **开发者视角**：技术债。应有统一"进程编排层"（统一启动/健康检查/终止/发现接口），现在没这层抽象。

### 痛点 4.3：没有统一"健康检查 + 自愈"面板，用户没统一回滚办法
有 99_Admin 下修复入口（修环境、停服务、清缓存、修 affine），但**是独立工具，不是"系统健康总览"**。用户不知道"现在系统什么状态、哪些进程活着、哪些锁被谁占、要不要全清重来"。

- **产品视角**：缺"控制台"——一眼看所有后台进程/锁/队列状态，一键"重置到干净状态"。现在是"哪坏点哪"，非"系统级自检自愈"。
- **标注者视角**：出问题不知点哪个修复入口。"AI 不工作了"可能是环境坏/进程僵死/锁没释放/显存占满——分不清，只能挨个试或喊人。
- **开发者视角**：状态信息都在（server.json/worker_status/lock payload/queue status），只是没聚合呈现。技术上完全可做总览面板。

### 痛点 4.4：错误分类对用户是黑盒，没有"错误→动作"映射
系统内部有错误分类（fatal/可恢复/容量错误/OOM/连接拒绝），但**没传递成"用户该做什么"**。OOM 和"服务没启动"对标注员都是同一个"Failed"弹窗。

- **产品视角**：每个错误应带"建议动作"——OOM 建议"减 patch/关别的程序"，连接拒绝建议"点重启服务"，环境坏建议"点修复环境"。
- **标注者视角**：现在弹窗只说"失败了"，不说"怎么办"。这是支持成本高的主因。
- **开发者视角**：错误分类内部已有（_FATAL_ERROR_SUBSTRINGS 等），只是没接到用户引导层。接通成本低、收益高。

### 痛点 4.5：Mimics 自身崩溃后没有统一回滚
后台 Mimics 崩溃有 supervisor 兜底（隔离病例+重启），但**前台 Mimics（标注员正用的）崩了，外部进程/锁/队列状态悬空**。没"前台 Mimics 重启后自动清理并恢复到一致状态"机制。

- **产品视角**：标注员最怕"软件崩了，重启后一团乱"。需要"重启即自愈"。
- **标注者视角**：崩了重启，可能发现 GPU 被上个僵尸进程占着，AI 用不了，一脸懵。
- **开发者视角**：启动时有 stale 锁清理（_cleanup_stale_processes），但激进杀进程 opt-in（怕误杀）。需更聪明的"前台 Mimics 启动 = 安全收敛所有状态"逻辑。

---

## 跨专题根因总结（最重要）

所有痛点都是以下四条根因的变体：

**根因一：外部能力是"外挂"而非"内嵌"，所有外部世界的不稳定都直接砸到标注员头上。**
Mimics 封闭，AI 在外面。中间"递条子"的桥是脆弱性来源，也是体验损失来源（慢、重、卡）。→ 痛点 2.1、4.1、4.3

**根因二：复杂度做了，但"翻译"没做。系统内部很精密，对外呈现很粗糙。**
错误分类、几何校验、权重路由、AUC 评估——内部都有，但标注员看到的是英文弹窗和技术错误。缺"把系统能力翻译成标注员动作"的中间层。→ 痛点 1.2、2.2、3.3、4.4

**根因三：通用性被"特定数据集约定"和"过严契约"侵蚀。**
硬编码文件名/子目录、导出强依赖源元数据、无降级路径。→ 痛点 1.2、1.3

**根因四：状态是分散的、恢复是局部的，缺系统级一致性。**
进程/锁/队列/权重状态散落各处，有局部恢复无全局收敛，用户没"一键回到干净状态"的把握。→ 痛点 1.4、3.5、4.2、4.3、4.5

---

---

## 专题五：few-shot 微调与 DINOv3 引导点

**现状**：标注员面对 **13 个 AI 入口**（nnInteractive 4 个 + DINOv3 5 个 + IGAC/Snake/Scribble 3 个 + nnUNet 4 个）。背后是两套并行训练管线。

### 痛点 5.1：13 个 AI 入口，无概念导览，标注员无法自行选对入口
DINOv3 预测入口（`DINOv3\02`）和 DINOv3 引导点入口（`nnInteractive\04`）在两个不同菜单文件夹，但都是"选器官 Mask → 后台跑 DINOv3 推理"，前者直接出掩膜、后者出建议点再转 nnInteractive。标注员不读文档无法区分。guided 工作流在文档里只字未提。

- **产品视角**：入口爆炸 + 概念泄漏。DINOv3 掩膜模型 vs nnInteractive 任务模型 vs 官方模型——这是产品级概念泄漏到了 UI 层。
- **标注者视角**：想"让 AI 帮我标这个器官"，面对 13 个入口不知道点哪个。选错了白等一次推理。
- **开发者视角**：交互层面不重复（04 复用 nnm 的 point_set），但**训练层面真实重复**：`fewshot_pipeline.py` 与 `nninteractive_finetune_pipeline.py` 两套并行管线都做 .mcs 标签导出、样本选择、GPU 锁、后台 Mimics 调度、取消协议，仅模型头不同。两个模型目录、两套状态文件。

### 痛点 5.2：guided 点流程长、双窗口切换、3+3 上限无解释
标注员路径 = 选器官 Mask → 菜单 04 → 等后台 DINOv3 推理 → 外部窗口 +/- 调点 → 点 Run → 再等 nnInteractive。点在 Mimics 里（3D 视图），控制在复核窗口里（另一进程），跨窗口往返 + 350ms 轮询延迟。复核窗口只说"at most three of each"，标注员不知道这是 `prompt_proposals.py` 的硬编码参数，也无法在 UI 调。

- **产品视角**：若 DINOv3 建议点不准，标注员等于多等一次推理才进入和入口 01 完全一样的流程，价值密度不确定。
- **标注者视角**：双窗口切换累；加/删点有延迟；5 秒 ack 超时说明作者已知这条链路会丢消息。
- **开发者视角**：ack 协议、clamp 越界点等稳定性细节做得不错，但本质是为"AI 在 Mimics 外"付的延迟与稳定性税。

### 痛点 5.3：四套"给提示→出掩膜"心智模型不兼容
nnInteractive（3D 点集）、ScribblePrompt（scribble/box + 强制平面选择）、IGAC（直接拖拽边界）、ITK Snake 各一套。IGAC 是唯一"直接在图上拖"的交互，体验最好，但只做精修不做初始分割，标注员无法当主力。

- **产品视角**：交互范式碎片化，没有统一的"一次进入、连续交互"主链路。
- **标注者视角**：学一种交互只能用于一个入口，学四种才能覆盖全部场景。
- **开发者视角**：四套交互各自独立、共享 GPU 锁，技术上是隔离的，但产品上缺一层统一交互抽象。

---

## 专题六：预处理 / 后处理 / 工具链诊断

### 痛点 6.1：nnInteractive 后处理为零，是硬伤
nnInteractive 推理结果坐标映射后直接写字节进 mask，**全链路无 threshold / 连通域清理 / 空洞填充 / 平滑**。模型出什么标注员收什么，离群小连通域、空洞全靠手工修。而后处理代码在研究脚本里已经写好（`predict_z_local.py` 的按半区保留最大连通域、`run_two_stage.py` 的 keep_largest_component），却没接进标注链路。

- **产品视角**：预标注-人工修正是标注效率核心场景，后处理缺失让 AI 输出的"毛刺"全压给标注员手工修。
- **标注者视角**：AI 给一堆碎小假阳性，一个个删比从头标还累。
- **开发者视角**：能力现成（研究脚本里有），接进链路成本不高，但没接。DINOv3 的 keep_largest_component 默认关且锁死在训练期配置，推理时不能切换。

### 痛点 6.2：预处理自动但出错反馈是"文件路径"不是"下一步动作"
预处理对标注员自动执行（几何重采样、HU 编码保留），但出错时提示"Review mimics_import.log and _failed_cases.json before retrying"——指向文件让用户自己翻，无应用内查看器。更隐蔽的是强度回退 `raw_gv_zscore_best_effort` 静默降级只留 WARNING 日志，标注员无感知。

- **产品视角**：自动是好的，但"静默降级"和"出错翻文件"都是把排障成本压给标注员。
- **标注者视角**：导入失败看到一串英文 + 文件路径，不知道下一步干什么。
- **开发者视角**：几何校验精密，但错误文案是给开发者的（"classic DICOM cannot represent..."）。

### 痛点 6.3：诊断脚本三处散落，从未产品化
根目录 4 个 `check_*.py`（硬编码病人路径）、`external/dinov3-medical-seg` 下 8 个 `diag_*.py`（硬编码远程 Linux 路径）、`.mimics_runtime/` 下临时脚本——证明同一类问题（affine、z-spacing、RTSS 引用）被反复手工排查。通用工具只有 `verify_medical_geometry.py` 一个，且 CLI-only、未进入口文档。

- **产品视角**：诊断能力有且被反复需要，但每次数据问题都催生一个一次性脚本，从没整合成产品功能。
- **标注者视角**：完全碰不到，数据有问题只能等开发者来排查。
- **开发者视角**：技术债。同一几何问题反复手工排查，说明缺一个"数据健康检查"产品入口。

### 痛点 6.4：一键自检存在但只在开发者手里
`tools/test_all.py`（约 450 个测试）一条命令能跑全量回归，但没有 GUI/菜单入口、无 bat 包装。99_Admin 的 Check 只查环境不查几何/链路。标注员的"系统健康"视角仍是空白。

- **产品视角**：自检能力技术上全有，但标注员够不着。
- **标注者视角**："系统到底好不好"无法自验，只能等用的时候炸。
- **开发者视角**：测试覆盖很强，但只覆盖无 Mimics/GPU 的逻辑层。

---

## 专题七：批处理与标注版本管理

### 痛点 7.1：版本管理与标注工具链完全断层（最大痛点）
版本体系设计严谨（回退链 v2.3→v2.1→v2、逐器官回退、manifest、21 个测试），但**只活在训练数据转换层**。Mimics 导出永远写 v1 的 `segmentations/` 目录，对版本目录零感知。标注员想"发一个 v2.1"必须手工建目录搬文件；想看"这个器官现在是哪个版本"没有任何工具。两套系统各干各的。

- **产品视角**：这是最严重的断层。版本管理设计再好，标注员用不上，等于没有。标注在 Mimics 里改完导出会直接覆盖 v1，版本管理全靠人在导出后手工搬运。
- **标注者视角**：想回退到旧版本？没有操作，只能找开发者改 toml 配置。想发新版本？手工建目录。完全感受不到"版本管理"的存在。
- **开发者视角**：`segmentations_v\d`、`annotation_version` 全部只存在于 `external/nnunet_segmentation_workflow/`，Mimics 导出侧零感知。两套系统各自正确，中间断了。

### 痛点 7.2："批量跑 AI 标注"缺失
能批量建 mcs、批量导出，但**没有"选一批 case 一键预标注"**。而预标注-人工修正恰恰是标注效率的核心场景。现有 AI 全是逐例交互式或训练式。

- **产品视角**：批处理覆盖了数据流转，没覆盖标注本身。最高价值的"批量预标注"是空白。
- **标注者视角**：100 个 case 要标，只能一个个打开、一个个点 AI、一个个修。没有"先让 AI 全跑一遍，我只修错的地方"。
- **开发者视角**：技术上可做（队列 + 逐例调 nninteractive 自动提示），但需要设计"无交互自动预标注"策略（如用 DINOv3 建议点自动起手）。

### 痛点 7.3：无协作模型，多人标注会互相覆盖
没有任务分配、没有 case 级归属锁。两个标注员同时打开同一 case 的 mcs 各自改，最后导出互相覆盖，系统不拦。只能靠口头约定。

- **产品视角**：团队协作场景下，这是数据安全风险。
- **标注者视角**：不知道别人在标什么，可能白标。
- **开发者视角**：有文件资源锁防冲突，但无 case 级归属锁防协作覆盖。

### 痛点 7.4：续跑是隐式的，失败子集无法单独重跑
依赖指纹/skip-existing 的"重跑即续跑"，无 resume 动词（除微调外），无"只重跑失败 case"命令。失败 5 例混在 200 例里，用户得自己判断。导出默认 skip-existing 是双刃剑：防误覆盖（好），但重标后忘开 overwrite 会静默不更新，只有明细 JSON 里 "skipped_existing" 字样提示，不弹窗不汇总。

- **产品视角**："不报错但没生效"是标注员最怕的静默失败。
- **标注者视角**：重标后导出，以为成功了，其实文件没更新，下游用了旧标注。
- **开发者视角**：skip-existing 策略合理，但缺"有文件被跳过"的显式汇总提醒。

### 痛点 7.5：导入/导出批次无任务面板
AI 训练有专门 viewer（DINOv3 04_Show_Status、nnUNet 03_Show_Status），但最常发生的导入/导出批任务没有任何任务列表 UI。只能看一次性日志和弹窗。换机器/关会话，历史批次就"看不见"了，只能翻目录找 JSON。

- **产品视角**：可见性倒挂——低频的 AI 训练有面板，高频的导入导出没有。
- **标注者视角**：导出 200 个 case，跑到一半关了电脑，第二天不知道进度到哪了。
- **开发者视角**：job 记录分散在 `.mimics_runtime/{import_runs,export_jobs,append_jobs,ui_tasks}` 四处，没聚合。

---

## 跨专题根因总结（更新版）

在原四条根因基础上，新增/强化：

**根因五：产品入口爆炸 + 概念泄漏，标注员面对 13 个 AI 入口无导览。**
DINOv3 掩膜模型 / nnInteractive 任务模型 / 官方模型 / 三种交互算法，技术架构上的分离（两套训练管线、四套交互）直接泄漏成 UI 上的入口爆炸。→ 痛点 5.1、5.3、6.1

**根因六：版本管理与标注工具链是两个世界，从未打通。**
版本体系在训练转换层自成一体，Mimics 导出对版本零感知，标注员够不着。→ 痛点 7.1

**根因七：能力存在但未产品化——后处理、诊断、自检、批量预标注全是"开发者手里的半成品"。**
后处理代码在研究脚本里、自检有 450 个测试但没菜单、诊断脚本散落三处、批量预标注是空白。→ 痛点 6.1、6.3、6.4、7.2

**根因八：协作与可见性缺失——单人单机假设。**
无任务分配、无 case 归属锁、导入导出无面板、静默跳过不提醒。系统假设一个标注员在一台机器上独立工作。→ 痛点 7.3、7.4、7.5

---

## 优先解决方向（按"标注员收益 / 实现成本"排序）

1. **AI 入口收敛 + 概念导览**（根因五，痛点 5.1）：把 13 个入口收敛成"标注 / 训练 / 管理"三件事，入口选择由系统根据上下文自动推荐。标注员从"面对 13 个入口"变成"系统给我最合适的"。收益极大、成本中（主要是 UI 重组 + 文档）。
2. **版本管理打通标注链路**（根因六，痛点 7.1）：让 Mimics 导出感知版本目录，标注员能在导出时选"发新版本/覆盖/回退"，并在 UI 看到每个器官当前版本。这是当前最大的功能断层，收益极大、成本中。
3. **错误→动作映射 + 健康总览面板**（根因二、四，痛点 4.3、4.4、6.2）：把现有错误分类接上"建议动作"，把分散的进程/锁/队列/批次状态聚合成控制台。标注员从"出问题就喊人"变成"出问题看面板点按钮"。
4. **nnInteractive 后处理接入 + 批量预标注**（根因七，痛点 6.1、7.2）：把研究脚本里现成的后处理（连通域、填洞）接进标注链路并做成推理时可选项；新增"选一批 case 一键预标注"批处理。预标注-人工修正是标注效率核心场景。
5. **交互点击层级优化**（根因一，痛点 2.1、5.3）：让 nninteractive worker 常驻、Mimics 侧维护"继续提示"循环，把"每提示重跑入口"降到"一次进入、连续交互"；逐步统一四套交互范式。
6. **导入拖拽 + 通用化去硬编码**（根因三，痛点 1.1、1.2）：外部窗口更傻瓜化；硬编码首选名/子目录约定收敛成"可配置 + 智能识别"。
7. **显存预检查 + 导出降级路径 + skip 提醒**（痛点 2.4、1.3、7.4）：启动任务前预检显存、导出无源元数据时降级到 Mimics 网格直接导出、skip-existing 时显式汇总提醒。投入小、能消除多个硬伤。
8. **协作模型 + 批次面板**（根因八，痛点 7.3、7.5）：case 级归属锁防互相覆盖、导入导出批任务面板。从"单人单机"走向"团队协作"。

---

## 专题八：无用 / 过度的设计与功能（可清理清单）

> 基于代码事实核实（grep 引用 + 文件存在性）。分三类：**死代码/一次性残留（建议删除）**、**重复实现（建议合并）**、**过度设计（建议精简）**。

### A. 死代码与一次性残留（建议删除或归档）

1. **文档引用已删除的脚本**（死引用）：`docs\nninteractive_mimics.md` 引用 `scripts\setup_nninteractive_env.py` 和 `scripts\build_nninteractive_bundle.py`，但 `scripts/` 目录已删除。文档 §7.1 关于 "probe 缺包即阻断" 的描述与现行代码（probe 只记录不阻断）也不符。→ **清理：更新文档，删死引用**。

2. **根目录 4 个 check_*.py 一次性脚本**：`check_liver_meta.py`、`check_rtss.py`、`check_vmi_meta.py`、`check_zspacing.py`——全部硬编码网络共享路径（`\\isi-sh\HSW\...`），无 argparse，全库无任何代码/文档引用（grep 仅命中自身）。是开发者排查特定 Spectral CT 数据的即弃脚本。→ **清理：移到 archive/ 或删除**。

3. **external/dinov3-medical-seg/ 下 8 个 diag_*.py**：`diag_ghost.py`、`diag_sliding.py`、`diag_patch_fp.py`、`diag_intensity_shift.py`、`diag_invert.py`、`diag_aug_compare.py`、`diag_spatial_only.py`、`diag_trigger.py`——硬编码远程 Linux 路径（`/home/wenwen_zhang/kidney_experiments_portable/...`），服务于肾脏 T2b elastic bug 排查。grep 显示 `diag_aug_compare` 只命中自身，其余同理。纯研究一次性脚本。→ **清理：移到 research_archive/ 或删除**。

4. **.mimics_runtime/ 下临时诊断脚本**：`check_affines.py`、`q_s_all.py`、`q_s_diff.py`、`compare_brain_masks.py`、`test_mimics_start.py`——用裸 struct 手解 NIfTI 头排查 brain mask affine，与 `tools/verify_medical_geometry.py` 功能重复。是临时排查残留。→ **清理：删除（运行时目录本就不该长期存脚本）**。

5. **nninteractive_mimics.py 的同步模式死代码链（约 528 行，占文件 10%）**：`_run_with_config`（4885 行）里同步分支已被短路——`execution_mode` 配置键全库（代码/文档/json）再无任何出现，没有任何方式能走进 `_run_sync`。死亡链条：`_run_sync`（4665–4808，144 行）无人调用 → `_prompt_menu`（4650–4663）仅 `_run_sync` 调 → `_BridgeWorker` 类（2836–3039，204 行）仅 `_run_sync` 调 → `_run_prediction`（3040–3076）仅 `_run_sync` 调 → `_bridge_call`（2691–2819，129 行）仅 `_run_prediction` 调。连带可删 `reuse_session` 死配置键（4689 行，全库唯一读取点在死代码内）。git 历史显示这套同步代码从初始提交即存在、禁用分支也是初始提交即写入——**从未在当前形态下可达过**。→ **清理：删除 `_run_sync` 链 + reuse_session，约 -528 行，零风险**。

6. **external/MedDINOv3/ 整个目录（478 个 py 文件）**：是个完整的 nnU-Net 克隆 + MedDINOv3 研究代码。grep 显示它只被自己的脚本（`ext_trainer/kidneymeddinov3_trainer.py`、`eval_kidney_meddinov3.py` 等）和 dinov3-medical-seg/ext_trainer 引用，**标注流程（runtime_py35、tools）零引用**。readme 明确是 "Work in progress" 的研究项目。→ **清理：这是研究侧产物，不该混在标注工具仓库里。建议移出或 git submodule 隔离**。

7. **dinov3-medical-seg/ext_trainer/ 下的研究 trainer**：`kidney_252_trainer.py`、`flexict_primus.py` 等——tools/ 下零引用，是研究实验的 ext_trainer 残留。→ **清理：同上，研究侧归档**。

### B. 重复实现（建议合并到公共层）

8. **外部 Python 解释器发现逻辑重复 8 处**：`_python_exe`/`_find_external_python`/`_bridge_python`/`_runtime_paths`/`_environment_root` 这套"找 nninteractive_env/python.exe"的逻辑在 `runtime_py35/` 下 **8 个文件**重复实现（mimics_import、mimics_export、mask_import、nninteractive_mimics、create_mcs_batch、interactive_algorithms_mimics、setup_environment、fix_source_affine_metadata），路径列表彼此有细微差异。→ **合并：抽到 runtime_common.py 一份**。

9. **多套原子写实现（9–10 份，2 组逐行复制）**：runtime_common、resource_locks（与前者几乎逐行相同）、fewshot_pipeline、fewshot_status_viewer（与 fewshot_pipeline 逐行相同）、nninteractive_task_common、nninteractive_bridge、mimics_bridge（NIfTI 专用）、io_path_setup_ui，外加 `_copy_label_atomic`/`_copy_file_atomic` 逐行相同。各版本参数不一致（12 vs 20 次重试、有无 SMB 兜底），fewshot_pipeline 版**缺 SMB 兜底反而最脆**。→ **收敛到 resource_locks 一份**（唯一被 py3.5/py3.13 双端导入的公共模块），NIfTI 写保留专用外壳但重试环复用。

10. **多套进程存活检测（2 份完整重复 + 9 个透传包装 + 2 个合理增强）**：完整实现 3 份——runtime_common.process_exists、resource_locks.process_exists（与前者几乎逐行相同，双端零依赖原因可接受）、io_path_setup_ui.process_exists（旧式老代码残留，应删改 import）；透传包装 9 个（各 2–7 行纯噪音，docstring 重复解释 handle 截断 bug 教训写了 4 遍，应全删直接 import）；增强版 2 个应保留——`_process_matches_server`（命令行+token+heartbeat 三重匹配，判"是我们的 server 吗"）、`_lock_payload_is_live`→`process_matches`（PID+进程创建时间戳防 PID 复用，锁正确性依赖它）。→ **删 io_path_setup_ui 版 + 9 个包装；保留 2 份双端实现和 2 个增强**。

11. **三套杀进程路径**：ctypes `TerminateProcess`（mimics_import 启动清理）、PowerShell `taskkill /T /F`（stop/supervisor）、`terminate_process_async`（runtime_common reaper 线程）。语义一致靠约定。核实结论：**不是"三选一"，是"三种场景 × 每种 2–8 份副本"**。ctypes 路径两段近乎逐行相同（mimics_import L851–915 vs nninteractive_mimics L5077–5135），可删换 taskkill；PowerShell 脱离式有不可替代语义（延迟后跨进程复核锁 token 再杀）但 stop_background 内三变体 ~80% 相同可合一；reaper 阶梯被重写约 7 份（fewshot_pipeline / fewshot_status_viewer 逐行相同 / fewshot_mimics / nninteractive_mimics / nninteractive_bridge 两处 / mcs_creation_supervisor / nninteractive_finetune_pipeline），应收敛为一份。→ **ctypes 删；阶梯收敛一份；PowerShell 保留机制、内部去重**。

12. **两套训练管线重复（约 500–600 行同构逻辑）**：`tools/fewshot_pipeline.py`（5102 行，DINOv3）与 `tools/nninteractive_finetune_pipeline.py`（2527 行，nnInteractive）逐项对应：标签导出启动器（265 vs 214 行）、标签缓存指纹机制（166 vs 208 行）、GPU 锁、原子复制（`_copy_label_atomic` vs `_copy_file_atomic` **逐行相同**）、终止阶梯、cancel 标记。且 `runtime_common.background_mimics_command`（L945）已有统一的后台 Mimics 命令构造器，两条管线都没用它。**隐患**：fewshot 后台 Mimics 锁按输出目录 scope 互斥，finetune 导出锁是每 job 唯一文件名（`wait_seconds=0`，永不冲突）——两条管线连后台 Mimics 互斥协议都没统一。现状是 fewshot_pipeline 已成事实公共层（nnunet/finetune/igac 等都从它 import 单个函数），等于为 5102 行模块引入重依赖。→ **抽 pipeline_common（导出启动器/标签缓存/GPU 锁/终止阶梯/update_status），顺手修锁互斥分歧**（痛点 5.1 同源）。另有 4 份 `safe_slug`（fewshot_pipeline / fewshot_status_viewer / fewshot_training_setup_ui / nninteractive_task_common）应一并收敛。

### C. 过度设计（建议精简）

13. **环境变量开关过多且散落（逐项判定）**：
   - **可删（无人用/默认即最优/与 config 重复）**：`NNINTERACTIVE_TIMEOUT`/`NNINTERACTIVE_PROBE_TIMEOUT`（全库无任何地方设置，对应 config 键已存在）、`MIMICS_IMPORT_CHECKPOINT`（关闭崩溃面包屑没价值）、`NNINTERACTIVE_ASYNC_WORKER_IDLE_TIMEOUT`（与 config 键同义且默认值不一致 3600 vs 900，第三个设置通道纯增混乱）、`MIMICS_TRAINING_GPU_MEMORY_GB`（与 config `default_gpu_memory_gb` 同义第二通道）。
   - **应合并为 1 个**：5 个 `*_USE_EVENT_TIMER`（IMPORT/EXPORT/MASK_IMPORT/IO_SETUP 等）是同一"Mimics 版本兼容：事件订阅会崩默认 Win32 SetTimer"的 opt-in 复制 5 份，全未文档化。
   - **应文档化或移入 config**：`MIMICS_DICOM_RESAMPLE_MODE`（改变输出几何语义，仅顺带提到 never，axial 无人验证）、`MIMICS_MASK_RESAMPLE_METHOD`（nearest/distance 完全未文档化，默认 nearest 已最优，要 distance 的人无法发现）、`MIMICS_DISABLE_GPU_LOCK`（关闭 GPU 锁违背设计初衷的危险出口却最不透明，零文档）。
   - **保留必要**：`MIMICS_BACKGROUND_EXE`（后台 Mimics 找不到的核心排障出口）、`MIMICS_AUTO_CLEANUP_ON_START`、`MIMICS_IMPORT_VERBOSE_LOG`（已文档化）。

14. **CONFIG_REFERENCE.md 文档缺口（4 类）**：(A) `nninteractive_finetune_config.json` 20 个键零覆盖（最大缺口）；(B) `fewshot_config.json` 缺约 17 个键（base_config、default_batch_size、default_gpu_memory_gb、default_label_source、default_lora_alpha/rank 等）；(C) `mimics_io_config.json` 的 mimics_background_exe 不在表格、`window_level_presets.json` 完全未提、`interactive_algorithms_config.json` 顶层键无表格；(D) 约 20 个环境变量未收录（MIMICS_DICOM_RESAMPLE_MODE、MIMICS_MASK_RESAMPLE_METHOD、MIMICS_DISABLE_GPU_LOCK、NNINTERACTIVE_* 路径覆盖类、5 个 event-timer flag 等）。覆盖良好的：`nninteractive_config.json`（25 键全有表格）、fewshot_config 的 Runtime/Retention 段。

15. **mimics_io_config.json 形同虚设**：只有 `mimics_output_dir`、`mimics_background_exe` 两个键，当前值均为空。实际配置记忆靠 `%LOCALAPPDATA%\Mimics-Script\ui_state\io_paths.json`。这个文件要么删，要么和实际生效的配置统一。

### 清理优先级

- **高收益低风险**（直接删/归档）：A1-A7（死代码与一次性残留）。这些不碰任何活路径，删了仓库立刻清爽。
- **中收益中风险**（需测试）：B8-B12（重复实现合并）。合并时要用 test_all.py 兜底。
- **需设计决策**：C13-C15（过度设计精简）。涉及"环境变量收敛到 config"的产品决策。
