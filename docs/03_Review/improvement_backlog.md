# 自主迭代账本（唯一状态源）

> 本文件由 `/iterate` 自主迭代循环维护。协议见根目录 `CLAUDE.md`。
> 每轮迭代开始先读本文件；所有状态以此为准，不依赖对话记忆。
> R1（2026-09-25）完成首轮 8 视角并行盲审，以下为去重定级后的完整状态。

## 使用说明（勿删）

- 状态列：`待办` / `进行中(第N轮)` / `已解决` / `已删除` / `需用户决策` / `阻塞(等资源)`
- 每个待办条目必须包含：痛点描述、根因、影响视角、严重度、验收标准
- 验收标准一经写入不得追溯放宽；已解决必须附 commit + 测试工件证据

## 待办（按优先级排序）

> 2026-09-29 R61 四视角评审后，本区从"已清空"重新填充：P0×3 + P1×10 +
> P2/P3×8 + 已拍板待实施×4。历史已完成条目移至区尾"以下为历史待办区"。

### P0（2026-09-29 四视角评审新增，均经人工源码核验，证据链闭合）

- [x] **R61-1【P0】nnInteractive 标注会话启动即 NameError**〔R61 当轮
  解决，commit `201b516`〕：恢复 `_prompt_buttons_for_profile`（与
  bd180da^ 逐字一致）；新增不 monkeypatch 的真实路径测试
  （执行 `_async_prompt_menu` 到 question_box，断言按钮串）+ 空
  profile 抛错测试；顺手修正两处指向不存在入口 "Stop All Owned
  Background Services" 的文案（真实入口为 99_Admin/03_Stop_All_Owned_
  Services）并把本文件加入 B4 残留扫描表（R61-20 一并关闭）。
  smoke 8/8（`20260929T124245`）。
- [x] **R61-2【P0】指纹过期重导入静默覆盖已标注 .mcs**〔R61 当轮
  解决，commit `0be6ba5`〕：指纹文件缺失时改查 dataset_manifest
  provenance（与 .mcs 同目录、剪枝后存活）——记录一致 → 跳过 + 回写
  指纹自愈；记录不同 → 真源变更照旧重处理；无记录 → 保留 .mcs 跳过
  并日志告知强制重导入方法。三个失败路径测试（一致跳过自愈 / 无记录
  跳过 / 真变更重处理）。fast 27/27（`20260929T130542`）。
- [x] **R61-3【P0】FlexiCT 主动学习蒙版转换在 GUI timer 内同步阻塞最长
  10 分钟**〔R61 当轮解决，commit `a2d3244`〕：复刻预测流 wait_bridge
  两段式——tick 只 Popen + 标记 "converting"；daemon 线程 communicate
  写 result.json；后续 tick 调 `_al_finish_conversion` 在 GUI 线程上屏
  （所有 Mimics API 调用保持线程亲和）。"converting" 中间态对 UI 的
  request_updates（只读 applied/failed）和 _al_pending_requests（只拾
  pending）均透明，不会双启动/双上报。`_al_open_case` 的 open_project
  保留在 tick：Mimics API 线程亲和无法移出，且仅用户显式 open 动作
  触发。AST 契约测试钉住 communicate 不得回归 + applied/failed 两个
  失败路径测试。fast 27/27（`20260929T131846`）。

### P1（2026-09-29 四视角评审新增）

- [ ] **R61-4【P1】模型管理器死按钮**（UX 视角代理发现，人工核验属实）：
  `tools/model_manager_ui.py` 的 "Use This Model"(:361-363) 与
  "Remove Broken Entry"(:364-366) 按钮在表格刷新时被
  `setEnabled(False)`(:455-456)，但全文件无任何选中信号连接
  （无 itemSelectionChanged/cellClicked/currentItemChanged）也无任何
  `setEnabled(True)`——两个按钮永久禁用，功能不可达。
  验收：选中行后按钮可用；点击 "Use This Model" 真正切换 current 模型。
- [ ] **R61-5【P1】导入失败弹窗指向工作目录而非真实失败位置**
  （`mimics_import.py:2988-2995/3027-3036/868-871`）：用户看到的失败
  路径是内部工作目录，无法定位自己数据的问题。验收：弹窗显示
  用户可识别的原始输入路径。
- [ ] **R61-6【P1】Stop 无确认且会停掉所有队列**（`mimics_import.py:
  1413-1471`）。验收：Stop 前确认对话框（复用 question_box）；只停
  当前队列（或明确告知影响范围）。
- [ ] **R61-7【P1】输出路径"粘滞"**（`io_path_setup_ui.py:753-760/
  859-877`）：换输出目录后部分流程仍指向旧路径。验收：全流程输出
  路径一致。
- [ ] **R61-8【P1】拖拽批量导入状态冻结 "Importing"**
  （`import_drop_window.py:302-307`）：导入失败/完成后状态不更新。
  验收：状态反映真实进度（完成/失败/跳过）。
- [ ] **R61-9【P1】数据集根被当成单个 case 接受**
  （`io_path_setup_ui.py:205-236`）。验收：目录直接拖入时提示"请选择
  具体病例文件"或正确展开。
- [ ] **R61-10【P1】P0-1 修复后将暴露的模态弹窗风暴**
  （`nninteractive_mimics.py:1963-1993`）：每 case 一次模态弹窗。
  验收：会话级单次提示替代逐 case 弹窗。
- [ ] **R61-11【P1】os.walk 无剪枝扫全盘 + 同步 kill 阶梯在扫描线程**
  （`mimics_stop_background.py:264-267`、`:1234→1189-1215`）。验收：
  扫描剪枝 .mimics_runtime 深层目录；kill 移出扫描线程或加限速。
- [ ] **R61-12【P1】全量蒙版 SHA-256 在 GUI 线程计算**
  （`nninteractive_mimics.py:1933-1936`）。验收：复用
  `interactive_algorithms_mimics.py:151-166` 的 `_stream_buffer` 异步
  模式，GUI 不冻结。R61-13 契约扫描白名单同时登记了
  interactive_algorithms_mimics.py:148、mimics_export.py:1170/:1179、
  nninteractive_mimics.py:1835/:1913/:1936/:2468 的存量 tobytes()，
  修复时逐条从白名单删除。
- [ ] **R61-26【P1】nnInteractive 同步桥接往返阻塞 GUI（最长 1800s）**
  〔R61-13 契约扫描落地时发现存量〕：`_call_mimics_bridge`
  （nninteractive_mimics.py:1379-1396）Popen + communicate
  （timeout 至 1800s）可从 `_async_monitor_tick`/`_tick` 调用闭包
  可达。验收：改用 wait 线程 + result 轮询模式（R61-3 先例），
  契约扫描白名单 nninteractive_mimics.py:1389/:1395 删除。
- [x] **R61-13【P1】最高杠杆测试资产：GUI 线程阻塞源码契约扫描**
  〔R61 当轮解决，commit `afff2bc`〕：`TestGuiThreadBlockingContract`
  AST 扫描——遍历 runtime_py35 全部 `*_tick` 函数**及其同模块调用
  闭包**（纯 tick 级扫描抓不到 R61-3 那种"tick → 普通函数 →
  communicate"间接链，已用修复前代码验证调用闭包版能报红），禁止
  communicate()/tobytes() 出现在嵌套线程函数之外。存量 9 处白名单
  登记附待办编号（7 处 tobytes → R61-12；2 处 nnInteractive 同步
  桥接 communicate → **新登记 R61-26**），修复时逐条删除。已验证
  扫描对修复前代码报红（防回归能力实证）。

### P2/P3（2026-09-29 四视角评审新增，精选）

- [ ] **R61-14【P2】nnunetv2 版本门禁失配**：安装 2.8.0 但
  `setup_env.py:279` 要求 ≥2.8.1 → 触发假修复/打包失败提示。
- [ ] **R61-15【P2】653MB python_env/models 污染**
  （`nninteractive_bridge.py:179-185`）。
- [ ] **R61-16【P2】ScribblePrompt 检查点未纳入环境体检**
  （env_guidance collect_issues 不含该检查点缺失场景）。
- [ ] **R61-17【P2】FlexiCT Stop 无确认**（`flexict_status_viewer.py:
  279-285`）。
- [ ] **R61-18【P2】无根 README / 快速上手文档**（产品视角 N3）。
- [ ] **R61-19【P3】R60 重编号后活文档编号漂移**：
  `task_lifecycle_and_safety_policy_CN.md:21-23/:86`、
  `mimics_real_data_validation.md:365` 仍是旧编号——活文档测试只查
  存在性不查内容，需同步或加引用一致性测试。
- [x] **R61-20【P3】B4 残留扫描表缺 nninteractive_mimics.py**
  〔R61 当轮解决，随 R61-1 commit `201b516`〕：已加入扫描表，同时
  修正了两处指向不存在入口的 "Stop All Owned Background Services"
  文案（真实入口 99_Admin/03_Stop_All_Owned_Services），扫描表以该
  死名作为禁止串防止回潮。
- [ ] **R61-21【P3】UX 精修项批**（UX 代理 #6-13）：配色碎片化、无窗口
  图标、nnInteractive auto-apply 文案矛盾（:2715 vs :4060-4066）、
  env_guidance model_dir 死胡同、io_path_setup_ui ~200 行死代码
  （:396 后）等，详见 UX 代理报告。

### 已拍板待实施（用户 2026-09-29 拍板）

- [ ] **R61-22 收编外来脚本**：4 个未跟踪工具
  （runtime_py35/inspect_mcs_batch.py、sync_missing_masks_batch.py、
  tools/inspect_mcs_projects.py、sync_missing_masks.py）+ 外来
  .mimics_runtime 脚本（含 copy_to_z.py），git add 纳入版本管理，
  补资源锁/保留期/测试/账本登记。**copy_to_z.py 的 Z: 数据根写入路径
  单独审视后收编或改写**。
- [ ] **R61-23 批量推理立项下轮实施**：mimics_batch_cli 加 predict
  子命令（4 个外来脚本为需求实证）。
- [ ] **R61-24 UI 语言统一中文**：注意编码防乱码（GBK 控制台陷阱），
  部分技术错误信息可保留英文。涉及入口/按钮/对话框字符串。
- [ ] **R61-25 D19 FlexiCT 增训**（低优先级）：训练表单加"从上次
  checkpoint 继续"选项，透传到 `_spawn_flexict_worker`
  （`flexict_pipeline.py:851` 硬编码 False，trainer 侧已支持）。

### 以下为历史待办区（已全部完成，保留历史）

- [x] **B20【P0→P1 定级见条目】stage worker 进程注册静默失效（孤儿安全网不存在）**
  〔2026-09-28 并行审计（进程生命周期专审）发现 1，代 agent ID 见轮次报告 R45。
  原始定级 P0：这张安全网对"恰好孤儿过的那类进程"（nnU-Net/FlexiCT
  stage worker）不存在；因触发需先发生挂起/超时（概率性），按"高频真实
  卡点"标准折 P1，但它是历史 FlexiCT 孤儿事故因果链的一环，修复优先级
  按 P1 最高位排。〕
  - 痛点：`tools/nnunet_pipeline.py:951-962` 以 role `nnunet_{stage}`
    （nnunet_preprocess/nnunet_train/nnunet_infer）调 `register_process`，
    但 `resource_locks.py:236-248` 的 `VALID_PROCESS_ROLES` 不含这三个
    role，`register_process` 对未知 role 抛 ValueError（resource_locks.py:
    290-291）且被 `except Exception: ownership_token = ""` 吞掉——注册
    永远静默失败。健康面板与 mimics_stop_background 的注册表击杀对
    stage worker 完全不可见（与 949 行注释的承诺相反）。
  - 根因：worker spawn 时用了未注册进白名单的自造 role 名；白名单是
    数据不是逻辑，加名即修。
  - 影响视角：标注者/运维（孤儿发生后无自动恢复，只能手动查杀）。
  - 验收标准：① 三个 role 加入 `VALID_PROCESS_ROLES`；② `_spawn_worker`
    的注册失败分支将错误写进 job.log（不再静默）；③ 防回归测试：spawn
    一个带真实 role 的注册（mock 进程句柄），断言 `register_process` 不
    抛且 role 在白名单内；断言 VALID_PROCESS_ROLES 包含三个 stage role。
    ④ smoke + fast 门禁通过。
  - R46 解决（commit `c190b23`）。**实施中发现比审计更糟**：注册调用传了
    API 根本不存在的 `job_id=` kwarg——每次注册先抛 TypeError（还轮不到
    白名单检查），自始被 bare except 吞掉，安全网从写下的第一天就不存在。
    修复：① 调用改传 `extra={"job_id": ...}`（resource_locks 既有模式）；
    ② 三个 role 入 `VALID_PROCESS_ROLES`；③ 注册失败写 job.log 不再静默；
    ④ 测试 `test_stage_worker_roles_are_registered`（真实注册往返）+
    `test_spawn_worker_registration_matches_api`（源契约：调用点每个 kwarg
    必须存在于 API 签名——未来 API 漂移在测试里响而不是生产里静默）。
    验收 ①–④ 全达成；smoke 8/8（20260928T020933）+ fast 27/27
    （20260928T021412）。

- [x] **B21【P1】本地训练链路对"活着但不动"零防御（worker/控制器无挂起超时）**
  〔2026-09-28 并行审计（进程生命周期专审）发现 2 + FlexiCT 审计 P2-3 同源。
  历史 FlexiCT preprocess 孤儿挂死（R41/R42 轮报告记录）的直接形态：CPU
  4.45s、gate 文件在、无结果 JSON、永不退出。〕
  - 痛点：控制器轮询 `tools/nnunet_pipeline.py:998` 只判 `poll() is None`
    与取消标志，无进展/挂起超时；`tools/nnunet_stage_worker.py`（111 行）
    无自超时、无父存活检查。nnU-Net 的 spawn.Pool 在内存压力下
    OpenBLAS 死亡后"静默重生尸体，阶段 0 CPU 永远挂住"（nnunet_pipeline.py:
    869-875 注释自证）。对照正例：`tools/interactive_algorithms_worker.py:
    57-68` 的 ScribblePrompt worker 已有父存活检查模式。
  - 根因：stage worker 设计时假设"外部进程自会退出"，未覆盖挂起形态。
  - 影响视角：标注者（训练卡死无任何提示，GPU 被占，Stop 后 worker
    仍可能存活）。
  - 验收标准：① `nnunet_stage_worker.py` 加总时长上限（从 request.json
    或环境变量读，train 对齐 Mimics 侧 14 天 monitor deadline，其余阶段
    按既有 gate/超时先例取值），超时写 result JSON `status=worker_timeout`
    后退出；② worker 主循环加父进程存活检查（复用
    interactive_algorithms_worker.py:57-68 模式）；③ 控制器收到
    worker_timeout 走既有失败路径（不改控制器）；④ 防回归测试：
    worker 挂起（sleep 超限）→ 超时分支写 worker_timeout 并退出；
    父进程死亡（mock）→ worker 自退。⑤ smoke + fast 门禁通过。
  - R48 解决（commit `1c4f7cd`）。实施：`nnunet_stage_worker.py` 加
    `_watchdog_trigger`/`_start_watchdog`——daemon 线程每 5s 检查
    ① 总时长预算（`_STAGE_BUDGET_SECONDS`：train 14 天对齐 Mimics 侧
    monitor deadline，preprocess/infer 24h 对齐推理 monitor deadline；
    spec 可带 `stage_deadline_epoch` 覆盖）② 父进程存活
    （`process_exists`，`nnunet_pipeline._spawn_worker` spec 新增
    `parent_pid`）。触发即写 result JSON `{"status": "error",
    "error": "worker_timeout: ..." / "parent_gone: ..."}` 后
    `os._exit(3)`——主线程阻塞在 stage 函数里永远收不到信号，只能
    进程级退出；控制器既有失败路径（`result.get("status") != "ok"` →
    RuntimeError）原样消费，控制器零改动（验收 ③）。测试 3 个：
    子进程实跑真 watchdog + 挂死主线程 → 退出码 3 + worker_timeout；
    死父 pid → 退出码 3 + parent_gone；`_watchdog_trigger` 单元三分支。
    （第一版用 PYTHONPATH shim 假 Action4 模块，因 worker 自身
    `sys.path.insert(0, WORKFLOW)` 在 import 时先于 PYTHONPATH 解析，
    批量运行时缓存预热即穿帮、结果不确定；改为子进程直接驱动真
    watchdog + sleep，不依赖任何模块遮蔽。）验收 ①–⑤ 全达成。

- [x] **B22【P1】回归矩阵超时只杀直接子进程，detached 孙进程全部存活**
  〔2026-09-28 并行审计（进程生命周期专审）发现 3。历史孤儿事故的放大器。〕
  - 痛点：`tools/run_regression_matrix.py:111-123` 的 `subprocess.run(timeout)`
    杀掉 test python 后，套件 spawn 的 detached 控制器（CREATE_NEW_
    PROCESS_GROUP，nnunet_jobs.py:41）与 stage worker 均存活；
    `tools/test_training_convergence.py:178-192` 的 `_wait_for_completion`
    1800s 超时后只返回状态、不杀 worker。
  - 根因：矩阵按"套件进程=全部工作"假设设计，未覆盖套件内 detached
    孙进程。
  - 影响视角：开发者/CI（孤儿进程跨轮残留，污染后续门禁）。
  - 验收标准：① `test_training_convergence._wait_for_completion` 超时
    分支复用既有 `nnunet_jobs.stop_job` 清理自己启动的任务（最小方案，
    不进生产路径）；② 防回归测试或走查记录证明超时分支确实调用了
    清理；③ fast 门禁通过（training_convergence 在 full，跑 full 或单独
    套件验证）。
  - R47 解决（commit `b948da1`）。**实施方案优于原验收**：`_wait_for_
    completion` 经核实是死代码（两测试同步调 run_training，无轮询等待），
    真正的孤儿源头是矩阵超时杀 test python。改为在 `run_regression_
    matrix.py` 的 TimeoutExpired 分支调 `resource_locks.sweep_processes`——
    比"每个套件各自清理"覆盖面更广（任何套件的任何 detached 孙进程，
    只要注册过且父死即被收）。配套：stage worker 注册补 `parent_pid`
    （finetune trainer 同型），sweep 的 parent-gone 检测由此生效；防回归
    测试 `test_sweep_terminates_stage_worker_orphaned_by_dead_parent`
    （注册 worker + 死父 → sweep 终止 → 进程真退 → 注册表清空）。验收
    ②以测试达成（强于走查记录）；③ fast 27/27（20260928T022257）+
    smoke 8/8（20260928T021833）。

- [x] **B23【P1】Stop Background 两处列表缺口：remote_training_controller
  无击杀标记，FlexiCT 监视器无人停止**
  〔2026-09-28 并行审计（进程生命周期专审）发现 5。两个子项都是纯列表/
  镜像改动。〕
  - 痛点：① `runtime_py35/mimics_stop_background.py:25-66` 的 MARKERS 不含
    `remote_training_controller.py`，PowerShell cmdline 扫描击杀找不到它；
    ② `_stop_inprocess_monitors`（:100-127）只处理 io_setup_mimics 与
    nnunet_mimics 的 `_MONITORS`，`flexict_mimics._MONITORS` 无人停止——
    "Stop All Owned Services"对 FlexiCT 监视器不生效。
  - 根因：remote_training_controller 与 flexict 监视器接入晚于
    mimics_stop_background 的列表建立。
  - 影响视角：标注者（"停止所有后台服务"承诺未完全兑现）。
  - 验收标准：① MARKERS 追加 remote_training_controller.py；②
    `_stop_inprocess_monitors` 增加 flexict_mimics 分支（照抄 nnunet 的
    8 行）；③ 防回归测试：源契约断言 MARKERS 含该标记 + flexict 在
    `_stop_inprocess_monitors` 源内被处理；④ smoke + fast 门禁通过。
  - **审计描述勘误**（实施时核实）：子项 ② 字面描述"FlexiCT 监视器
    无人停止"不实——`_stop_inprocess_monitors` 的通用循环
    （mimics_stop_background.py:160 原编号）一直处理
    `("flexict_mimics", "_MONITORS", "_stop_monitor")`。**真实缺口**是：
    nnunet 有专属分支在脱离监视器*之前*给活跃任务写 `control.json`
    cancel 标记（外部控制器、尤其远程容器得以安全收尾），FlexiCT 没有
    ——它的控制器只能死于后面无协商的 PowerShell 强杀。
  - R49 解决（commit `192ac8f`）。实施：① MARKERS 追加
    `remote_training_controller.py`（nnunet_jobs.py:136/325 与
    flexict_pipeline.py:1606 以 `python <tools>/remote_training_controller.py`
    启动，cmdline 含文件名，标记即生效）；② nnunet 的 27 行
    pre-detach cancel 块提取为共享 `_cancel_controllers_before_detach
    (module_name)`（FlexiCT 与 nnU-Net 共用同一 job_dir/status.json +
    control.json 布局，fallback `dirname/status.json` 同样命中真标记），
    `_stop_inprocess_monitors` 对两模块各调一次——比验收标准 ②"照抄
    8 行"更省（删重复实现而非新增）。③ 测试：MARKERS 契约加
    remote_training_controller.py；`test_stop_all_signals_tasks_...
    ` 升级为断言两模块都先 cancel 后 detach；新增
    `test_stop_all_writes_flexict_cancel_before_detaching_monitor`
    （行为级：fake flexict_mimics 模块 + 状态 JSON → control.json
    cancel 先于 `_stop_monitor`）。验收 ①–④ 全达成。


- [x] **A2【P1】回归测试污染生产状态（用户数据被测试覆盖/垃圾无限累积）**〔已转已解决区，R3〕

- [x] **A3【P1】nnInteractive 官方模型获取路径缺失：新标注者第一天必撞断点**
  〔已转已解决区，R6（检测+引导+报错文案已闭环；获取/分发方式本身仍为 D9 需用户决策）〕

- [x] **A4【P1】"Relink the source image metadata"指向不存在的动作**〔已解决，
  R35：用户 2026-09-27 指示核查 Fix Affine 逻辑；核查证明 Fix Affine 逻辑本身无
  问题但与 Relink 报错互斥（Resolve_prediction_context 的最后回退正是 Fix
  Affine 的前置），故文案按故障分岔：未链接→指回 01_Data 导入入口 + 重新导入
  指引；stale affine（nnunet_pipeline.validate_materialized_source_geometry，
  Fix Affine 真正可修复的场景）→点名 99_Admin > 04_Fix_Source_Affine_Metadata。
  三处文案源契约 + 渲染断言测试钉住；走查记录进 R35 轮次报告 §2。commit 与
  工件见轮次报告 R35。commit `cb4a890`，fast 28/28
  （`20260927T123006_fast.json`）。〕

- [x] **A5【P1】nnU-Net 训练表单要求 ML 工程师概念（Dataset ID/Fold/Trainer/Patch size/GPU devices 等）**
  〔R8 按验收标准括号明示的**最小方案路径**（默认值+说明）解决，见已解决区。
  注意：A5 的解决**不是**表单已完成结构收敛——"必填少/高级项折叠"的结构重设计
  仍完整保留在 D2 需用户决策，本条关闭范围仅限"三项有合理默认与一行说明 +
  走查记录"。〕
  - 痛点：`tools/nnunet_training_setup_ui.py` 表单直接泄漏 nnU-Net 原生概念，与
    nnInteractive 微调刻意收窄到 6 项的设计哲学相反。填错即训练失败或训错数据。
  - 根因：表单是 nnU-Net CLI 参数的直译，无默认值策略与解释层。
  - 影响视角：标注者/技术用户（训练是核心产品流程之一，表单事实上只对懂 nnU-Net
    的人可用）。
  - 验收标准：表单项收敛为"必填少而有解释，高级项折叠且有默认值"；至少 Dataset ID/
    Fold/GPU devices 三项有合理默认与一行说明；走查记录进账本。〔表单如何收敛属
    产品设计，方案先入"需用户决策"D2 或按最小方案（默认值+说明）实施〕

- [x] **A6【P1】五个"Stop"入口不可区分 + FlexiCT 等待提示指向 nnU-Net 的停止入口**
  〔①已解决（R7，见已解决区）；②仍为 D3 需用户决策，且新增：pending 推理结果的
  主动丢弃路径（flexict_mimics.stop_running_task 的 pending 分支）经 R7 审查确认
  无菜单入口可达——是否为其增加可达入口（结果本会自动保留+应用，丢弃需求真实
  频率存疑）并入 D3 一并裁决〕
  - 痛点：同一动作词散在 5 处（01_Data/04、01_Data/06、02_AI/nnUNet/04、99_Admin/03、
    FlexiCT 菜单内 Stop），作用域各不同；标注员面对"AI 卡住了"不知点哪个，最可能
    点最猛的 Stop All 误杀其他任务。且 `flexict_mimics.py:486-491` 等待提示写 "Use
    04 Stop Running Task to cancel"——指向 nnUNet 目录下的入口，点了会得到
    "No running nnU-Net task was found."
  - 根因：入口按子系统粒度拆分而非按用户意图组织；跨家族文案引用入口编号。
  - 影响视角：标注者（故障时刻选择困难 + 误杀风险；"AI 卡住"是每天可能遇到的）。
  - 验收标准：① FlexiCT 等待/错误文案不再引用 nnUNet 入口编号（改指自身动作菜单
    或通用停止入口）；② 入口合并/重组方案入"需用户决策"D3（涉及菜单结构变更）；
    ①有测试或走查记录。

- [x] **A7【P1】Mask Identifier 等待点击期间切换工具可能崩溃 Mimics**〔已解决，
  R37：核查确认窗口最小化设计（非模态预检 + 非模态结果框、仅显式 Click Again 后
  短暂重入模态）2026-07 已落地；Mimics 21 无非模态点采集 API，脚本层无法进一步
  收缩窗口。按验收标准"无可行代码方案"分支收尾：预检文案补明后果（crash + 丢
  未保存工作，符合 task_lifecycle 数据风险须说明的规范）；平台限制登记
  docs/MIMICS_PROJECT_ARCHITECTURE_CN.md §8.1；flow 测试钉住后果措辞与非模态
  标志防静默回退。commit `f40a77e`，flow 15/15、smoke 8/8
  （`20260927T130448_smoke.json`）、fast 28/28
  （`20260927T130941_fast.json`）。〕
  - 痛点：`runtime_py35/mask_identifier.py:330-353` `mimics.indicate_coordinate()` 模态
    阻塞期间切换工具（缩放/平移）会破坏 Mimics 工具状态机可致崩溃（注释自认）；现有
    缓解只是预检对话框英文小字把责任转嫁给标注员。
  - 根因：Mimics 脚本 API 限制（模态点击采集无保护）；代码层无法完全消除，只能缩小
    危险窗口/强化提示。
  - 影响视角：标注者（审阅工具每日使用，习惯性滚轮缩放即触发，未保存标注丢失）。
  - 验收标准：评估缩小窗口的可行方案（如点击前最后一次确认、提示中文化置顶）；
    若无可行代码方案，把风险提示按 task_lifecycle 规范改造并在文档登记为平台限制。
    （崩溃依赖误操作且有缓解，定 P1 边缘——若 Mimics API 有替代采集方式则可根治）
  - C6-8 补记（R59，2026-09-28）：nnInteractive 的模态 indicate_coordinate
    （`runtime_py35/nninteractive_mimics.py:2025-2037` 前后）与 ScribblePrompt
    的点击/涂抹/框选采集（`interactive_algorithms_mimics.py`）同属该崩溃
    风险面。ScribblePrompt 已在 R59 补同款后果提示（C6-7）；nnInteractive
    的取点窗口与 mask_identifier 同为"短暂重入模态"设计（每次 prompt 前
    有非模态确认），已按同一平台限制处置，不另立项。

- [x] **A9【P1】nnU-Net 任务目录无任何 retention/清理（磁盘无界增长）**〔已转已解决区，R5〕
  - 痛点：nnU-Net workspace（`~/.mimics_script/nnunet`，jobs/models/cache 三子目录）
    对 sweep/retention/prune grep 零命中——flexict_pipeline:499 与
    nninteractive_finetune_pipeline:2327 均有 `_sweep_expired_jobs`，唯独 nnU-Net 没有；
    每次训练的 raw 预处理产物永久累积。
  - 根因：retention 收敛时漏掉 nnU-Net 路径；CONFIG_REFERENCE 无对应保留天数条目。
  - 影响视角：标注者（磁盘满→导入/导出失败，见 P6 已建的磁盘满错误映射）+ 开发者。
  - 验收标准：① nnunet jobs 加与 flexict 对齐的 `_sweep_expired_jobs`（保留天数进
    config + CONFIG_REFERENCE）；② 新增伪造 mtime prune 测试（对照 TestLifecycleAndRetention
    模式，补上 B9 缺的 prune 测试资产）；③ fast 绿。
  - **R5 范围裁定（架构审查发现）**：本条目验收标准只覆盖 jobs/ 清扫；痛点原文的
    主要磁盘增长源——`runtime/`（nnUNet_raw/preprocessed/results，按 Dataset 目录跨
    job 复用）与 `cache/source_grid/<task_id>`（可重建缓存）——仍无 retention，残留
    登记为下方 A11。

- [x] **A11【P1】nnU-Net runtime 三目录与 cache/source_grid 无 retention（A9 残留，磁盘无界增长主源）**
  - 痛点：A9 的 jobs sweep 只清 `jobs/`（实测 flexict 工作区 jobs 仅 473K，而
    runtime/ 4.8G、cache/ 145M——两个家族的 runtime 都从不清扫）。nnU-Net 的
    `runtime/nnUNet_raw|preprocessed|results` 按 Dataset 目录跨 job 复用（删除涉及
    preprocess 复用语义与 `_assert_dataset_id_available`），`cache/source_grid/<task_id>`
    是可重建缓存。磁盘满→导入/导出失败（P6 已建错误映射）。
  - 根因：retention 收敛时只对齐了 jobs 维度；runtime/cache 维度两个家族（nnU-Net
    与 FlexiCT）都漏。
  - 影响视角：标注者（磁盘满失败）+ 开发者（flexict 工作区已实证 4.8G 累积）。
  - 验收标准：① `cache/source_grid` 按 mtime 清扫（可重建缓存，参照
    `_sweep_expired_prepared_cache` 模式，保留天数进 config）；② runtime 三目录的
    清扫方案需设计（Dataset 在用判定——被已注册模型引用的 Dataset 不得删），先出
    方案过架构审查再实施；③ 伪造 mtime 测试；④ fast 绿。
  - **R9 解决（commit 3f16d0f）**：架构审查（独立架构师代理，裁定"修改后通过"，
    4 项必改全部落实：cache 任务树保护、预处理复用分支重打 manifest 时间戳、
    挂钩 except Exception 包裹、wrapper 健壮性——无 dataset_id 的 infer/AL 请求跳过、
    registry 行缺 dataset_name 时按 dataset_id+task_id 重建、显式 config 参数）。
    实现：`nnunet_common.sweep_dataset_retention` / `sweep_source_grid_cache_retention`
    共享清扫器（mtime 窗口 + 保护集）；两家族各自的 `_protected_dataset_names`
    （注册模型 ∪ 非终态 job 的 dataset 名 + task id）与 `_sweep_runtime_and_cache`
    wrapper，挂在全部 5 个 job finally 块（nnunet 2 + flexict 3）的
    `_sweep_expired_jobs` 旁。config 新键 `runtime_retention_days` /
    `source_grid_cache_retention_days`（默认 30，0 禁用，CONFIG_REFERENCE 两节
    同步）。测试：TestSweepRuntimeAndCache nnunet 9 条 + flexict 5 条（全部
    tmp 目录 + 显式 config + 伪造 mtime：超龄未引用删/注册模型保/在跑 job 保
    （含 cache 任务树）/新近保/0 禁用/空任务目录回收/models 不碰/缺
    dataset_name 重建兜底）。fast 门禁 26/26 全绿（工件
    `20260926T065602_fast.json`）。首扫预告：首个 job 触发时可能一次性 rmtree
    数 GB（本机 flexict 工作区约 2.4G 无引用 Dataset），发生在 worker 进程内，
    不阻塞 GUI。
  - 非阻塞观察（架构审查记录）：全局注册表跨工作区同名 Dataset 过度保护（安全
    方向）；僵尸非终态 job 在 reconcile 前持续保护其 dataset（方向安全）。

- [x] **A10【P1】nnU-Net 预处理 worker 在内存紧张机上 OpenBLAS 崩溃→静默死锁（训练永久挂起）**〔已转已解决区，R4〕

### P2

- [x] **B24【P2】用户文档残留 4 个已删入口（R41/R43 未同步文档面）**
  〔2026-09-28 并行审计：nnU-Net 审计发现 1 + Review 审计发现 1 + 数据
  导入导出审计发现 1，三方汇总去重。〕
  - 痛点：`docs/mimics_entry_guide.md:9,31,43,53`（Export Masks / nnU-Net
    Stop Running Task / Window Reset Full Range / Fix Source Affine
    Metadata 四行）；`docs/scripting_library_workflows.md:16,26,40,64,224`；
    `docs/mimics_real_data_validation.md:87,322`；`docs/task_lifecycle_and_
    safety_policy_CN.md:26`；`CONFIG_REFERENCE.md:314`；`docs/MIMICS_
    PROJECT_ARCHITECTURE_CN.md:309`（与同文件 :234 的正确记载自相矛盾）。
    标注者按文档找入口必落空。R41/R43 只更新了架构文档，漏了速查表与
    活文档。
  - 根因：删入口轮的验收清单只含代码/测试/架构文档，未含全部面向
    用户的活文档。
  - 影响视角：标注者（文档可信度）。
  - 验收标准：上述全部位置改为现存入口或注明合并去向（Fix Affine →
    预测报错路径）；新增源契约测试（进 test_all.py 既有 B4 同型扫描或
    单列）断言活文档集合中不出现已删入口名；fast 门禁通过。
  - R50 解决（commit `0a7462c`）。实施：账本列出的 6 份文档全部修正
    ——`mimics_entry_guide.md` 删 4 行死条目；`scripting_library_workflows.md`
    的入口清单按现存 21 入口重排（01_Data 补 07/08/09、02_AI 补
    FlexiCT×4+ScribblePrompt、99_Admin 补 05–10，删 03/nnUNet-04/
    03_Review-05），切换表 `02 AI/nnUNet/04 Stop Running Task` 改指
    状态窗口；`mimics_real_data_validation.md` 的 `03_Export_Masks` →
    `07_Quick_Export_Masks`、删 `05_Window_Reset_Full_Range`（对话框内
    Reset 动作的文字保留，它是真实功能）；`task_lifecycle_..._CN.md`
    nnU-Net 停止入口改状态窗口；`CONFIG_REFERENCE.md` Export Masks →
    Quick Export Masks；`MIMICS_PROJECT_ARCHITECTURE_CN.md` 删与 4.3
    重复的 4.7 小节（其中"使用标准 Export Masks"指已删入口）。
    **审计清单外补漏**：`windows_end_to_end_acceptance_2026-08-01.md`
    4 处（254/322/420/512 行）同型死引用一并修正。架构文档内
    "原 …删除"迁移注记属刻意保留的历史记载，测试放行。新测试
    `test_living_docs_do_not_reference_deleted_entries`（6 份活文档 ×
    4 个已删入口名，放行显式历史注记行）。验收标准全达成。

- [x] **B25【P2】Setup/Repair 离线包指引文件名拼错（mimcs）**
  〔2026-09-28 并行审计（Review 工具）发现 3。标注者照指引放置文件后
    仍报"未找到"，死循环。〕
  - 痛点：`runtime_py35/setup_environment.py:503` 弹窗让用户放置
    "**mimcs**_script_portable.zip"，`_find_portable_archive()`（:84-87）
    实际查找 "**mimics**_script_portable.zip"。
  - 根因：手写文件名笔误；无对照测试。
  - 影响视角：标注者（离线安装路径直接失败）。
  - 验收标准：改正拼写；测试断言弹窗文案与 `_find_portable_archive`
    的目标文件名一致（同一常量或源契约断言）；smoke + fast 门禁通过。
  - R50 解决（commit `0a7462c`）。实施：新常量
    `PORTABLE_ARCHIVE_NAME = "mimics_script_portable.zip"`，弹窗文案与
    `_find_portable_archive` 的三处候选路径共用同一常量（顺带消除三处
    硬编码重复）；测试 `test_portable_archive_dialog_names_the_file_
    the_finder_searches` 走真实 `main("extract")` 分支断言弹窗含正确
    文件名且无 "mimcs" 残留（须 mock `active_runtime_blockers`——本机
    有用户自己的 sync_missing_masks 锁在跑，真实守卫会先拦截）。验收
    标准全达成。

- [x] **B26【P2】03_Review 三个窗宽窗位入口无"无活动图像"守卫**
  〔2026-09-28 并行审计（Review 工具）发现 4。〕
  - 痛点：`runtime_py35/window_level_mimics.py:389-396` main 不查
    `_active_image()`；无图像时 `_apply_preset → _set_contrast_points`
    抛未捕获异常直达脚本错误，或静默无效果。同目录 mask_identifier.py:
    288-296 有友好提示先例。
  - 验收标准：main() 加同 mask_identifier 型活动图像检查（无图像弹
    友好提示并返回）；flow 测试断言无活动图像时三个入口均弹提示不抛
    异常；smoke + fast 门禁通过。
  - R51 解决（commit `c398339`）。实施：`main()` 入口统一加
    `_active_image()` 守卫——四个动作（auto/choose/undo/reset）终点
    都是 `set_contrast`，一处守卫全覆盖；无图像弹 mask_identifier 同款
    提示并返回 1。flow 测试在既有 window 用例中扩展无图像退化段：四
    动作各自断言返回 1、弹提示、不开预设对话框、对比度不动。验收
    标准全达成。

- [x] **B27【P2】Quick Export 未保存工程抛裸 RuntimeError**
  〔2026-09-28 并行审计（Review 工具）发现 5。〕
  - 痛点：`runtime_py35/mimics_export.py:2596` `raise RuntimeError(
    "Save the current Mimics project before exporting masks.")` 经
    `quick_export_main`（:3756-3758）→ run_runtime_entry 无捕获，用户
    看到脚本层错误而非对话框。同文件 `__main__` 分支有 Fatal Error
    包裹先例。
  - 验收标准：该分支改 message_box + return 1；flow 测试断言未保存
    工程时弹窗且不抛；smoke + fast 门禁通过。
  - R52 解决（commit `da8b69e`）。实施：`_launch_external_export_setup`
    的未保存工程分支由裸 `raise RuntimeError` 改为 message_box（说明
    原因：快速导出基于已保存的 .mcs 才能在 Mimics 关闭后继续）+
    return 1，与同文件 `__main__` 分支的 Fatal Error 包裹同型。
    flow 测试在 io_setup 路由用例中扩展退化段：`project_path` 置空后
    断言返回 1、弹窗、不抛。验收标准全达成。

- [x] **B28【P2】nnInteractive 0 体素空结果被静默应用（可抹掉已有标注）**
  〔2026-09-28 并行审计（nnInteractive 审计）发现 F3。〕
  - 痛点：`runtime_py35/nninteractive_mimics.py:4467` 应用 + :4498-4504
    仅 log WARNING。"更新所选 mask"原位模式下全空 mask 会清掉用户已画
    内容且无任何对话框。
  - 验收标准：`_handle_async_result` 对 0 体素结果先弹确认（说明应用
    会清空当前 mask，提供 不应用/仍应用 选项），拒绝则不 apply；
    flow 测试两分支；smoke + fast 门禁通过。
  - R53 解决（commit `c1506d2`）。实施：`_handle_async_result` 在
    应用前（`_choose_completed_result_target` 之前、image/target/project
    校验之后）对 `foreground_voxels == 0` 弹确认：说明应用会清空当前
    Mask 并给出改点建议（前景点靠近结构中心、背景点放远）；"Don't
    Apply" 置 session ready 不 apply；"Apply Empty Result" 走原路径。
    flow 测试两分支：拒绝则不触碰 mask 缓冲且 session 回 ready；接受
    分支走 apply（拒绝消费掉 pending_sequence 后重查不重复弹窗）。
    验收标准全达成。

- [x] **B29【P2】FlexiCT 推理监控不认 abandoned 终态（最长 24h 空轮询）**
  〔2026-09-28 并行审计（FlexiCT 审计）P2-1。〕
  - 痛点：`runtime_py35/flexict_mimics.py:577` 推理分支终态只有
    ("failed","cancelled")，训练分支 :557 含 abandoned。远程任务被放弃后
    本地监控走到 24h deadline，告警文案误导。
  - 验收标准：:577 终态集合与训练分支对齐（加 abandoned），复用训练
    分支提示文案；测试断言 abandoned 状态触发 _stop_monitor + 正确文案；
    smoke + fast 门禁通过。
  - R54 解决（commit `978bfc1`）。实施：`_monitor_tick_locked` 推理
    分支终态元组加 "abandoned"，abandoned 弹窗复用训练分支的失败句式
    "FlexiCT prediction failed.\n\n{error}"（error 为空时用 "The remote
    task was abandoned." 兜底），failed/cancelled 原文案不变。
    `tools/test_flexict_integration.py` 新增 TestInferenceMonitorTerminal
    States：abandoned 与 failed 各一例，直接驱动 `_monitor_tick_locked`，
    断言 `_stop_monitor`（从 _MONITORS 移除）+ 弹窗文案。验收标准
    全达成。

- [x] **B30【P2】FlexiCT"至少两个病例"文案与校验不一致（单病例可进空训练）**
  〔2026-09-28 并行审计（FlexiCT 审计）P2-2。〕
  - 痛点：`tools/flexict_training_setup_ui.py:607-610` 只查非空，文案写
    "at least two cases"；单病例走到 `flexict_pipeline.py:299` 时
    train=0、val=1，浪费一次 GPU 锁与完整空训练。
  - 验收标准：条件改 `len(request["cases"]) < 2`（一行）；测试断言
    单病例被拒且报错与文案一致；fast 门禁通过。
  - R55 解决（commit `b5b31d8`）。实施：`tools/flexict_training_setup_ui.py`
    校验条件由 `not request["cases"]` 改 `len(request["cases"]) < 2`，
    报错文案不变（本来就写 "at least two cases"）。测试
    `test_flexict_single_case_submission_is_rejected`（test_gui_smoke.py，
    离屏 Qt）：mock `_request` 返回单病例，走真实 `_submit` → 后台
    校验线程 → `_poll_submission`，断言未提交且状态栏显示该文案。
    验收标准全达成。

- [x] **B31【P2】remote_unreachable 被 Mimics 侧监控改写为 orphaned_remote**
  〔2026-09-28 并行审计（nnU-Net 审计）发现 2。〕
  - 痛点：`runtime_py35/nnunet_mimics.py:412-417` 豁免元组缺
    "remote_unreachable"（对照 tools/nnunet_jobs.py:180 已含）。远程
    重连预算耗尽后 10s 内状态被改写，重连指引文案丢失（Re-attach 仍
    可用，消息降级）。
  - 验收标准：415 行元组补 "remote_unreachable"；测试断言该状态不被
    改写；smoke + fast 门禁通过。
  - R56 解决（commit `401b187`）。实施：`runtime_py35/nnunet_mimics.py`
    `_managed_job_process_stopped` 豁免元组补 "remote_unreachable"，
    与 `tools/nnunet_jobs.py:180` 的 reconcile 豁免集对齐。测试
    `test_remote_unreachable_is_not_rewritten_by_stopped_process_check`
    断言该状态（含死 PID、超过 10s 龄期）不被判为 stopped。验收标准
    全达成。

- [x] **B32【P2】远程监控 RemoteCommandError 无限重试无预算**
  〔2026-09-28 并行审计（进程生命周期专审）发现 6。无人干预时控制器
    本地进程永活、任务状态永悬。〕
  - 痛点：`tools/remote_training_controller.py:3950-3970` docker 控制
    命令持续失败时无重试预算循环（对照网络错误走 _reconnect_session
    有 RECONNECT_MAX_ATTEMPTS）。
  - 验收标准：该分支加与 reconnect 相同的尝试预算，超限抛既有
    RemoteServerUnreachable（落入非终态+reattach 路径，零新状态）；
    测试断言预算耗尽路径；fast 门禁通过。
  - R57 解决（commit `eb16503`）。实施：`_monitor_remote` 新增
    `docker_error_attempt` 计数（成功轮次归零，与 `reconnect_attempt`
    同型），`RemoteCommandError` 分支先写
    `remote_control_unavailable` 状态（含 `remote_docker_error_attempt`）
    再判预算，超限抛既有 `RemoteServerUnreachable`，由 `run()` 的既有
    handler 落为非终态 `remote_unreachable`（B31 刚保护过该状态不被
    Mimics 侧改写）——零新状态。测试
    `test_docker_control_failures_are_bounded` 用全失败的 Session 驱动
    真实 `_monitor_remote`，断言异常文案、终态
    `remote_control_unavailable`、`remote_docker_error_attempt` 达到
    RECONNECT_MAX_ATTEMPTS；用推进式假时钟消除 10s 真实等待。

- [x] **B33【P2】Undo Last Import 前台做全量 SHA-256 大文件 I/O 无提示**
  〔2026-09-28 并行审计（Review 工具）发现 6。铁律 1 风险。〕
  - 痛点：`runtime_py35/import_undo_mimics.py:228` `_mcs_fingerprint`
    （.mcs 可达数百 MB 全量哈希）+ :127 open_project + :231 save_project
    全在 GUI 线程，无预告。
  - 验收标准：弹非阻塞"正在打开/校验工程"提示（复用既有 waiting
    消息通道）；指纹阶段耗时可控性评估结论入轮次报告（若工程文件
    大到哈希秒级以上，考虑挪后台线程或跳过指纹改为 mtime+size 弱
    校验——需在轮次报告论证取舍）；flow 测试 + smoke + fast 门禁。
  - R58 解决（commit `5fba9c4`）。实施：删除前经
    `log_user_message`（INFO，既有日志面板通道，与
    collect_diagnostics 同型）预告 "opening and verifying <file>; Mimics
    may pause for a moment"，失败静默。指纹评估结论：本机基准
    500MB=0.41s、1GB=1.15s（~1.2GB/s，硬件加速 SHA-256）——数百 MB
    的 .mcs 哈希亚秒级，远低于"秒级以上"阈值，**不挪后台线程**
    （会引入消息驱动续体，复杂度不对价）；open/save_project 本身是
    Mimics API，只能在 GUI 线程调，预告即是对这段停顿的正解。测试
    `test_undo_announces_blocking_steps_in_log_before_deleting` 钉住
    顺序：通知必须在删除开始前落地（mid-flight 快照断言）。

- [x] **B1【P2】nninteractive_bridge.py 顶层 import nibabel，冷启动 15-32 秒**（R11 解决，commit 4dafac0）
  - 实施与验收对照：`import nibabel` 从模块顶层移入仅有的两个使用函数
    （`canonical_ras_buffer_mapping`、`load_image_nifti`），与 mimics_bridge.py
    既有懒加载模式一致。`-X importtime` 复测冷 import **0.57s**（验收标准
    <3s），import 链中已无 nibabel/pydicom；懒加载路径功能验证通过
    （identity affine 映射正确）。smoke 8/8
    （20260926T072638_smoke.json）、fast 26/26
    （20260926T073130_fast.json），含 flow_nninteractive 套件。
  - 全库核查确认无外部 `bridge.nib` 引用，其余模块本就是函数内懒加载。
- [x] **B2【P2】运行时状态无界增长（import_queues 147 目录无 prune / guard 文件 345 个）**（R12 解决，commits 6933c08 + 1e9c9f8 + c1967fa + ec9e539）
  - (a) `.mimics_runtime/import_queues/` 按输出目录哈希建目录永不清理（export 侧有
    `_prune_local_export_jobs` 100 个/14 天，import 侧无对应）；(b) `resource_locks.py:592`
    guard 文件按输出目录哈希生成、sweep 永不删，实测 345 个且被健康面板 10s 一次
    反复列举。验收：复用 export prune 模式补 import 队列清理 + sweep 按 mtime 清理
    超龄 guard；各配伪造 mtime 测试；fast 绿。〔与 A2 的垃圾清理合并执行〕
  - 实施与验收对照：
    - **guard mtime 活跃戳**：`_MutationGuard.__enter__`（resource_locks.py）与
      `_open_resource_guard`（runtime_common.py）在成功取得 OS 锁后 `os.utime` 戳
      guard mtime——此前 mtime 永远停在创建时间，热资源 guard 会被误判闲置。
    - **`sweep_stale_guards`**（resource_locks.py，30 天默认）：三重门（idle mtime /
      无 sibling .lock / 无竞争 try-acquire）+ 平台删除顺序（POSIX 锁内 unlink；
      Windows close→re-stat→remove，FileNotFoundError=成功）。挂在
      `sweep_processes` 末尾（import 启动守护线程/健康面板即自动执行）。修复了
      实现期间发现的 `"a+b"` 打开位置在 EOF、msvcrt.locking 按当前位置锁字节的
      偏移 bug（必须先 `seek(0)`，与真实持有者一致）。
    - **`_prune_import_queues`**（mimics_import.py，14 天）：心跳超龄（或无心跳且
      目录 mtime 超龄）+ 无活 pid 消费者 + 内部 guard 无竞争 → rmtree + 同步删除
      mcs_queues 注册表行（防止 Stop 复活）。挂在 `_mark_mcs_queue_active`
      （每次导入启动，与 export prune 同款时机）。
    - **测试隔离**：fake_mimics_flow_test main()、cross_workflow setUp、
      nnunet/flexict integration `__main__` 均设 `MIMICS_RESOURCE_LOCK_DIR`；
      `GPU_LOCK_PATH` import 时常量改为按调用解析的 `_gpu_lock_path()`
      （mimics_label_export.py + nninteractive_bridge.py），并删除遗留孤儿常量
      `RESOURCE_LOCK_DIR`。
    - **回归测试**：`TestStaleGuardSweep` 4 例（闲置无 sibling 删 / 新鲜保留 /
      有 sibling 保留 / 持锁竞争保留）+ `TestImportQueuePrune` 5 例（超龄心跳+
      注册行删 / 新鲜保留 / 活 pid 保留 / 无心跳 mtime 回退删 / 无心跳新鲜保留），
      均用单字节 guard + 伪造 mtime，进 test_all.py（fast profile 覆盖）。
    - 门禁：smoke 8/8（20260926T084419_smoke.json）+ fast 26/26
      （20260926T091053_fast.json）；一次性生产清理 **464 → 2 guard**
      （保留 2 个为有 sibling .lock 的活资源锚），记录详见轮次报告 R12。
- [x] **B3【P2】_first_mcs_monitor_tick 每 2 秒全目录扫描（batch 模式可存活 7 天）**（R15 解决，commit `63d2aaa`）
  - `mimics_import.py:3081` batch 分支每 tick `sorted(os.listdir(output_dir))` 全目录
    stat，网络盘+大批量时每次 tick 都一次全扫描。验收：batch 分支只读 batch status
    +降频扫描；单测覆盖；smoke 绿。
  - 实施与验收对照：等待首个 .mcs 的 batch 分支改为读本地 `_mcs_batch_status.json`
    （单文件廉价读，且本就不在输出共享盘上）作门控——无完成案例时全目录扫描降频至
    30s 一次（`last_scan_epoch` 戳）；`completed>=1` 立即扫描（首个 .mcs 通知是
    监视器存在的目的，不得被降频拖慢）；首通知之后 tick 原本就只读 status 文件。
    单案例路径（target_mcs 指定）不受影响。测试：fake-mimics flow 驱动真实
    `_first_mcs_monitor_tick`——首 tick 扫描、30s 窗内无完成不重扫、completed=1
    立即扫描且通知触发、首通知后不再扫描（覆盖全部三个降频语义 + 通知不延迟）。
    smoke 8/8（20260926T120703）、fast 26/26（20260926T121039）。
- [x] **B4【P2】错误文案残留漏网：甩环境变量名/命令行/JSON 文件名/入口编号**（R10 解决，commit a68ef5a）
  - 原状：`mask_identifier.py:306`（MIMICS_MASK_IDENTIFIER_VISIBLE_ONLY）、
    `setup_environment.py:510`（"Run: python tools/package_portable.py"）、
    `mimics_import.py:2999`（翻 mimics_import.log 和 _failed_cases.json）、多处
    "Use 04 Stop Import Queue"（入口编号脆引用）。
  - 实施：13 处用户可见文案全部改为"可执行动作"——隐藏 mask 提示改为"在 Mask
    页签显示隐藏 mask 后重试"；offline-bundle 命令行改为"请联系配置本工作站的
    人准备离线包"（setup_environment.py + setup_env.py 两处）；6 处
    "04 Stop Import Queue" 改为弹性引用"Stop Import Queue (01 Data menu)"
    （名称+菜单，不依赖编号，含 2 处跨行字符串拼接的漏网点）；3 处
    "check mimics_import.log" 改为指向导入输出文件夹的 logs 子文件夹。
  - 防回归：`tools/test_all.py` 新增 `test_user_facing_text_has_no_developer_
    residue`——tokenize 扫描五个模块的全部字符串字面量，断言禁用形式
    （"Run: python"/"disable MIMICS_"/"04 Stop Import Queue"/
    "check mimics_import.log"）永不回归，并断言弹性引用形式存在。
  - 保留例外：mimics_import.py:755-786 disk_full/source_data_invalid 指引提到
    `_failed_cases.json`——该文件位于用户可见的导入输出文件夹内，用户可直接
    打开，属可用指引而非开发者残留，且 test_all.py:3739 既有断言依赖。
  - 证据：smoke 8/8（20260926T071054_smoke.json）、fast 26/26
    （20260926T071842_fast.json）。
- [x] **B5【P2】Window Undo 拿不到上次窗宽窗位时静默降级为全量重置**（R18 解决，commit `51fed2e`）
  - `window_level_mimics.py:371-375` `undo_last()` 无 `previous_contrast` 时仅日志
    WARNING 即 `reset_full_range()`——按"撤销"得到"重置"，无告知。验收：降级时弹
    确认或明确提示；测试覆盖。
  - 实施与验收对照：降级路径改为先弹确认（"Reset to Full Range / Cancel"，说明
    上次窗宽窗位未保存于本次会话），Cancel 时显示保持不变；测试扩入既有
    window flow（Cancel 不改对比度 + 接受后恢复全量，两分支断言）。
    smoke 8/8（20260926T124203）、fast 26/26（20260926T124558）。
- [x] **B6【P2】Setup/Repair Environment 动作菜单运维化（五选一英文选择题）**（R19 解决，commit `0e5ed6f`）
  - `setup_environment.py:418-452` Offline Install/Extract/Check/Repair/Scratch 五选
    一，环境已坏的标注员第一屏是选择题。验收：默认推荐动作置顶+其余折叠或加解释；
    走查记录。
  - 实施与验收对照：菜单按工作站实际状态排序——有已装环境（find_external_python
    命中）置顶 Check（先验证再改动），无环境有离线包置顶 Offline Install，无环境
    无离线包置顶 Extract Archive；首行给出推荐+理由，其余动作保留一行说明（question_box
    无折叠能力，取"加解释"分支）。测试：4 状态矩阵断言按钮顺序与推荐文案。
    smoke 8/8（20260926T125242）、fast 26/26（20260926T125627）。走查：测试即走查
    记录（4 状态×按钮序+推荐行断言）。
- [x] **B7【P2】Undo Last Import 在打开其他工程时拒绝执行 + 回执一次性**（R20 解决，commit `3cc624e`）
  - `import_undo_mimics.py:110-115` 要求先关当前工程；回执消费后撤销机会一次性，
    中途搞错无法再撤。验收：文案明示两步操作与回执消费语义；工程约束如可放宽
    （按指纹校验即可安全撤销）则放宽；测试覆盖。
  - 实施与验收对照：确认弹窗加 Notes 两行（需先关其他工程 + 一次性语义及
    失败可重试）；拒绝对话框改为两步指引（Step 1 关工程 / Step 2 重跑，明示
    记录未被消费）。工程约束**不放宽**：mask 删除按回执清单在目标工程内执行，
    指纹校验只保护文件回滚路径，不保护 mask 删除目标正确性——放宽会引入误删
    其他工程同名 mask 的风险。测试：确认文案断言 + 拒绝对话框两步文案断言。
    smoke 8/8（20260926T125952）、fast 26/26（20260926T130413）。
- [x] **B8【P2】前台批量导出走到一半弹"Batch Export Disabled"死路径弹窗**（R21 解决，commit `51ed9e5`）
  - `mimics_export.py:1172-1181` 前台路径弹英文架构说明并要求"再启动一次"。验收：
    前台入口直接路由到后台路径（或启动时即提示），不再出现"做一半重来"；测试/走查。
  - 实施与验收对照：根因核实——该弹窗所在的整条前台批次链是**死代码**：
    `_start_export_monitor` 零调用点（main() 批次路径只走
    `_launch_background_batch_export`），下游 `_export_monitor_tick`/
    `_start_next_batch_export`（含该弹窗）/`_start_win32_export_monitor`
    只被它调用；`batch_queue` 参数从未被真实传入。按铁律 4 整链删除
    （-199 行），弹窗随之消失且不会"做一半重来"（前台批次路径已不存在，
    批次只在后台进程执行）。测试：防复活断言（弹窗文案 + 4 个函数名 +
    batch_queue 不得重现）。smoke 8/8（20260926T131312）、
    fast 26/26（20260926T131650）。
- [x] **B9【P2】测试资产完整性：孤儿套件未注册 + prune 钩子零防回归 + 交付报告证据失真**（(a) 已解决 R34——用户 2026-09-27 拍板"删掉，但先核查功能是否有其他测试覆盖"；核查结论：两套孤儿测试均能跑通（47 项 / 35 项，31s+0.3s）且所覆盖功能**无任何其他测试覆盖**——nnint 引擎本体（config 校验/trainer 循环/数据/prompt/评估/任务锁，每次自定义模型训练经 `nninteractive_finetune_pipeline.py:76` 启动）与 Action1 标注版本回退链（独立 CLI convert 阶段专属）皆然；用户改选**注册进矩阵**而非删除（矩阵已有 flexict_pkg 集成包先例）：`run_regression_matrix.py` 新增 `nninteractive_finetune_pkg`（pytest，32s）与 `nnunet_annotation_version`（unittest，1s）两行，进 fast+full；fast 28/28（`20260927T121139_fast.json`），commit `460b9b1`。
  - (a) `integrations/nninteractive-finetune/tests/`（47 项）与
    `nnunet_segmentation_workflow/test_annotation_version.py` 不在任何 profile，从不
    运行；(b) ~~`prune_import_receipts`/`_prune_old_checkpoints`/`prune_logs`/
    `_prune_local_export_jobs` 四个 retention 钩子零直接测试（DELIVERY_REPORT_2 声称
    的 32 项 prune 测试现仅 26 项且零 prune——证据链失真）~~ → R32 已解决：四钩子
    各配伪造 mtime 测试（进 test_all.py TestLifecycleAndRetention，full 门禁覆盖），
    full 28/28（工件 20260926T225405_full.json，commit `24c1fd4`）；(c) ~~`test_atomic_write_
    survives_kill_signal` 名不副实（未模拟 kill）~~ → R28 已改名重写为
    `test_atomic_write_publishes_exactly_one_complete_file`（真实截断文件对照原子
    发布，commit `54bbcbb`，见 TB-01 登记项）。验收：孤儿套件注册进矩阵或入账
    废弃删除（需用户决策 D7）~~ → R34 注册进矩阵（`nninteractive_finetune_pkg` +
    `nnunet_annotation_version`，fast+full）；~~四个 prune 钩子各配伪造 mtime 测试~~；~~矩阵
    fast 绿~~（本轮改 test_all.py 自身，按 R28/R30 先例用 full）。
- [x] **B10【P2】PID 回收防护无端到端测试 + 生产 registry 已有真实回收案例 + fewshot 遗留状态将被迁移**（R22 解决，commit `450b748`）
  - (a) `resource_locks.py:194-201/406-464` start_marker 机制无"记录指向被回收 PID
    → 必须拒杀"的测试；生产 136 条记录中 6 条 PID 已被复用（25336=OpenConsole.exe
    等），风险真实存在；(b) `.mimics_runtime/fewshot_mimics_state.json`（DINOv3 删除
    前测试遗留）将被 `_migrate_old_settings` 改名成用户正式设置。验收：补 PID 回收
    端到端测试（含 terminate/sweep 拒杀断言）；删除 fewshot 遗留文件并加防复活
    （迁移函数遇指向不存在路径的旧状态应跳过）测试。
  - 实施与验收对照：(a) 端到端测试进入 TestProcessRegistry——注册真实子进程，
    篡改记录 start_marker 模拟 PID 回收，断言 `process_is_live` 判死、
    `terminate_process` 拒杀（子进程存活）、`sweep_processes` 只清记录不杀进程；
    (b) 本工作站核实两处旧文件均不存在（无可删对象）；修复迁移函数本身——
    记忆的数据集根全部不存在的旧状态**跳过迁移**（不再改名为用户正式设置），
    至少一个根仍存在的正常迁移保留；两分支测试覆盖。
    smoke 8/8（20260926T132829）、fast 26/26（20260926T133217）。
- [x] **B11【P2】(c 项遗留) 失败作业远程目录无限保留**（R23 解决，commit `ebb777e`）
  - (a)(b) 已于 R1 解决（见已解决区）；(c) 失败作业远程目录按设计无限保留（诊断
    用），但在共享服务器上违反"任务结束即清理"红线，无管理入口。验收：失败诊断
    目录提供显式清理入口（验收清单结束时可选清理）。
  - 实施与验收对照：验收清单加 `--cleanup-failed-dirs` 可选入口——run 结束时
    `step_cleanup_failed_jobs` 列出 `<remote_root>/jobs/<owner>/` 下残留作业目录
    （find 排除 .tar/.part），逐个走 `_remove_remote_job`（路径校验拒逃逸，
    连同 .tar/.tar.part 兄弟文件一并删除）；单条拒删不中断其余。默认不清理
    （保留诊断），入口可选符合验收措辞。测试 3 个进 AcceptanceChecklistTests
    （正常清理、拒删继续、flag 接线源扫描）。smoke 8/8（20260926T134646）、
    fast 26/26（20260926T135114）。
- [x] **B12【P2】远程新配置字段无 UI 无文档 + remote_root 默认值违反红线**（R13 解决，commit `20fd859`）
  - `container_runtime`/`container_namespace`/`remote_code_verify`/`remote_weights_verify`/
    `remote_cache_retention_days` 只能手改 servers.json；`remote_root` 默认
    `$HOME/mimics-ai`（root 用户落 `/root/mimics-ai`，直接违反"只能用
    /userdata/shijian_ruan/"红线，代码不提醒）。验收：Compute tab 表单暴露上述字段；
    remote_root 默认或校验对齐红线（至少首次使用时校验并警告）；CONFIG_REFERENCE +
    remote_training_design.md 补文档；gui_smoke/远程套件绿。
  - 实施与验收对照（架构师"修改后通过"7 项必改全部落实）：
    - **表单暴露**：ServerProfilesDialog 新增 5 行（Container runtime 组合框 /
      nerdctl namespace 行编辑 / Code+Weights verification 组合框 /
      Server cleanup after 天数控）+ 各自 tooltip；校验组合框项标注"(default)"，
      防止用户在不知情下降级安全校验；对话框 660→820 高度适配。✔
    - **红线对齐（首次使用即警告）**：test_connection 在 SSH 用户为 root 时于结果
      payload 附带通用措辞警告（"On a shared server, use the folder your
      administrator assigned... Current work folder: <绝对路径>"），UI 状态区以
      "⚠"前缀渲染；连接成功消息补上此前完全缺失的**解析后绝对工作目录**显示。
      不硬编码任何机器路径（红线是一台用户提供的机器的规则，写入发布代码违反
      铁律 4）；normalize_profile 保持纯校验器语义不动。✔
    - **顺带修复数据丢失 bug**：`_profile_values` 此前丢弃这 5 个字段——手改
      servers.json（如 container_runtime='nerdctl'）后从对话框点 Save 会静默
      重置为默认值。表单补齐后此路径关闭，离屏回归测试断言 5 个手改值全部
      经 `_profile_values` 存活。✔
    - **checklist 提示**：nerdctl 排障提示从"手改 servers.json"改为指 UI 表单
      （否则立即复现 B12 自己的痛点）。✔
    - **文档**：CONFIG_REFERENCE 修正两个错误键名（image→runtime_image、
      remote_workspace→remote_root）+ 5 字段全量文档（默认值/用途/共享服务器
      指引）+ "全部字段可在 UI 编辑"声明；remote_training_design.md Compute
      节补一行高级字段清单。✔
    - **测试**：离屏 ServerProfilesDialog 往返测试（5 手改值存活 + remote_root
      保留）+ root 警告源码级防回归；fast 26/26
      （20260926T112424_fast.json，含 remote_training 44.8s 与 gui_smoke）。✔
- [x] **B13【P2】MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START"退役"不彻底**（R14 解决，commit `550d7dc`）
  - DELIVERY_REPORT 宣布退役，但 `runtime_common.py:1127-1129` 仍读该环境变量、
    CONFIG_REFERENCE.md:264 仍文档化、多处 docstring 仍引用。用户照文档配置得到已
    退役行为。验收：读点与文档全删或明确标注退役；grep 零活引用；smoke 绿。
  - 实施与验收对照：走"全删"路线——`aggressive_auto_cleanup_enabled` 读取点 +
    两模块 wrapper + 两处启发式杀进程分支（ctypes OpenProcess/TerminateProcess
    循环、PowerShell 命令行扫描、TEMP 陈旧锁清扫）全删；所有权可证的清理
    （registry sweep、陈旧锁清理、owned-server 空闲超时）原样保留为唯一终止者。
    CONFIG_REFERENCE 行删除。防回归：旧 `test_windows_cleanup_declares_pointer_
    sized_process_handles`（守护对象即被删代码块）改写为 `test_startup_cleanup_
    has_no_aggressive_kill_path`——三模块源码级断言 flag 名/helper 名/OpenProcess
    裸杀不复活（防回归断言里的字面 flag 名是唯一存活引用，两处 docstring 已改为
    不点名措辞避免自撞）。活引用 grep 零命中（余下仅为 DELIVERY_REPORT 历史陈述
    与测试断言本身）。smoke 8/8（20260926T114815）、fast 26/26（20260926T115223）。
- [x] **B14【P2】文档漂移三处**（R14 解决，commit `550d7dc`）
  - 实施与验收对照：(a) ScribblePrompt checkpoint 路径 `external/`→`integrations/`
    两处（README_MIMICS.md + CONFIG_REFERENCE.md，与 download_scribbleprompt_
    checkpoint.py 实际路径一致）；(b) mimics_entry_guide.md 管理表补 Environment
    Guidance 行（入口 99_Admin/10 实存，env_guidance_mimics 标题核对）；(c)
    flexict_pipeline.py:215 与 test_flexict_integration.py:186 两处 MedDINOv3 陈旧
    引用改为中性措辞（审计只记了一处，实施时 grep 发现测试文件里还有同款——
    一并清理，未放宽验收标准）。grep 复查零残留。
- [x] **B15【P2】CLAUDE.md 含服务器明文密码且未 gitignore**（R14 解决，commit `550d7dc`）
  - 实施与验收对照：CLAUDE.md 加入 .gitignore（带注释说明含凭据、严禁入库）；
    `git status` 确认 CLAUDE.md 从 untracked 列表消失，`git check-ignore CLAUDE.md`
    命中。密码改环境引用方案不采用（CLAUDE.md 需对每个新会话自含生效，改环境
    引用会把红线约束变成"配置对了才生效"）。
- [x] **B17【P2】do_discover case 目录兜底扫描无上限**（R16 解决，commit `d180a9f`）
  - `mimics_bridge.py:2664` 对无首选图像的 case 全量列目录（DICOM 平铺可达数万文件）；
    `io_path_setup_ui.py:224` 同型扫描有 511 封顶，bridge 侧没有。验收：镜像 511
    封顶；单测；smoke 绿。
  - 实施与验收对照：兜底扫描从 `sorted(os.listdir)` 全量枚举改为 `os.scandir`
    早停——遇 `.dcm` 或第 512 项即停，与 `discover_single_source` 完全同型
    （含注释互指）。测试：fake-scandir 证明首轮枚举后不再继续（复用 UI 侧
    同型测试模式），既有 mask_selection 三态测试回归通过。smoke 8/8
    （20260926T121717）、fast 26/26（20260926T122054）。
- [x] **B18【P2】io_path_setup_ui 路径输入无去抖 + 状态面板最小化时仍 2s 全量扫描**（R17 解决，commit `9ebcf43`）
  - (a) `io_path_setup_ui.py:934` textChanged 每字符触发数据集扫描线程（加 300-500ms
    QTimer 去抖）；(b) batch/nnunet 状态窗不可见时仍每 2s 扫描（实测单次 1.7s），
    不可见时降频至 10s 或暂停。验收：两处改完 + gui_smoke 绿。
  - 实施与验收对照：(a) textChanged 改接 400ms 单发 QTimer 去抖（`recognition_debounce`），
    连发/粘贴只触发最后一次改动后的单次扫描；(b) `viewer_refresh.BackgroundRefresh`
    在每次周期 tick 时检查父窗口可见性——不可见/最小化时切 10s 心跳，重新可见时恢复
    2s 并立即刷新一次；显式 `request()`/`refresh_now()` 不受可见性影响（gui_smoke
    既有测试在未 show 的窗口直调 request，语义保持）。可见性在 tick 处判定而非
    hideEvent/showEvent monkey-patch（Qt 虚函数分发不保证触达实例属性，且 tick 处
    判定确定性可测）。三处状态窗（batch/nnunet/flexict）经由共享 helper 一次性覆盖。
    测试：新增 3 个（隐藏窗降频+request 直通、可见父对象保 2s、去抖接线断言）。
    gui_smoke 18/18（含新测试）、fast 26/26（20260926T123645_fast.json）。
- [x] **B19【P2】测试盲区 TB-01~TB-08 修复排期**（见下方登记表；TB-01~TB-08
  全部已解决——TB-08 于 R31 关闭，B19 整体关闭）

### P3 / 代码质量（可合并为"死代码与重复收敛"主题轮处理）

- [x] **C6【P3】2026-09-28 并行审计 P3 汇总（10 条，按修复成本升序）**
  〔六个审计代理的 P3 级发现，逐条入账待办。文件级证据已在轮次报告
  R45 附录存档。〕R59 解决（commit `027812f`）——C6-1～C6-10 全部实施
  （C6-8 为账本补记；C6-10 中菜单重排/改名部分拆入 D9 需用户决策）。
  - **C6-1 nnInteractive owned-server schema 漂移**：`runtime_py35/
    nninteractive_mimics.py:568` 只匹配 v2，bridge 写 v3
    （nninteractive_bridge.py:1019）——Mimics 启动清扫对现行 schema
    失效。改一行同时收 v2/v3。→ `_cleanup_stale_owned_servers` 的
    schema 检查改为收 v2/v3 元组；测试
    `test_owned_server_sweep_accepts_current_schema`。
  - **C6-2 nnInteractive worker 默认日志相对路径**
    （nninteractive_bridge.py:2893 `Path("nninteractive_bridge.jsonl")`，
    早期错误写进任意 CWD）。改 tempfile 绝对路径。→ `_worker_main`
    默认 `tempfile.gettempdir()`；测试
    `test_bridge_worker_log_default_is_absolute`。
  - **C6-3 训练/推理 setup 监控 1h deadline 静默脱管**（nnunet_mimics.py:181
    + flexict_mimics.py:144；表单开超 1h 后提交，完成弹窗丢失）。tick 里
    发现状态越过 opening 时刷新 deadline（或超时弹指引到状态查看器）。
    → deadline 到期时若监控是 *_setup 且其控制器进程仍存活，deadline
    顺延 1h 而非停监控；两模块测试
    `test_setup_monitor_deadline_extends_while_window_alive` /
    `_stops_when_window_dead`。
  - **C6-4 R43"已在运行"边缘分支吞原始报错**（fix_source_affine_metadata.py:
    367-381 main() 已在运行返回 0 → offer 返回 True → 上层不显示原始失败
    对话框）。main() 已在运行返回非 0，让上层照常显示原始失败。→
    main() 已在运行分支改返回 2；测试
    `test_affine_repair_already_running_returns_nonzero`。
  - **C6-5 同模型双图像并发 vs --max-sessions 1 未验证**（nninteractive
    runtime 只拦不同模型 busy worker）。先补并发双图像测试进
    test_mimics_nnint_deep.py，实测互相踢会话再加前置拦截。→ 代码级
    核实确证互踢：server claim() 满载即 503 不驱逐（app.py:196），
    远端会话构造即占租约（remote_session.py:192），bridge 容量自愈
    重启直接终止共享服务器（nninteractive_bridge.py:2298）——第二图像
    worker 必然摧毁第一图像在途预测。实施：`_same_model_busy_workers`
    前置拦截（在途预测阻止启动，对话框说明单会话限制）+
    `_retire_idle_image_workers`（预热 worker 按模型+图像 AND 语义保留，
    其余空闲者在启动新 worker 前退役；`_retire_different_model_workers`
    重构为调用它）。测试放 fake_mimics_flow_test.py（模块需 fake mimics
    注入才能 import，深测文件无此设施）：三段断言（在途阻止+文案、
    同图像不拦、空闲异图像 worker 退役且本图像 worker 保留）。
  - **C6-6 FlexiCT 表单不自动扫病例**（flexict_training_setup_ui.py:204-206,
    365；默认空表易被误读为路径错）。_load_context 读到 dataset 路径后
    自动调一次既有 _rescan_cases_async。→ `_load_context` 末尾按
    dataset+label 均非空触发自动扫描。
  - **C6-7 ScribblePrompt 取点缺 A7 同款后果提示**
    （interactive_algorithms_mimics.py:525-538）。复用 A7 文案。→
    提示循环文案补 A7 同款"光标等待期间勿切换工具"后果段。
  - **C6-8 A7 登记补记**：nnInteractive 模态 indicate_coordinate
    （nninteractive_mimics.py:2025-2037）与 A7 同源崩溃风险未登记——账本
    补记，不改代码。→ 已在 A7 条目下补记（ScribblePrompt R59 已修，
    nnInteractive indicate_coordinate 为同平台限制，不另立条目）。
  - **C6-9 用户文案出现环境变量名/原始路径/错误入口名**：mimcs 拼写
    （见 B25）；`tools/mimics_label_export.py:941-944`、`runtime_py35/
    mimics_export.py:537-541`、`mimics_import.py:2726-2729` 的
    MIMICS_BACKGROUND_EXE 环境变量措辞（应指向配置项）；`mimics_stop_
    background.py:1428-1433` 对话框展示内部 stop_log JSON 路径；
    `mimics_import.py:3030-3034` 入口名 "Stop All Owned Background
    Services" 与实际 "Stop All Owned Services" 不符；`mimics_export.py:2495`
    "Review mimics_export.log" 未给路径且 B4 测试扫描列表不含此文件；
    `setup_environment.py:524-526` 开发者向 CLI 指引漏网。→ 全部改为
    指向配置项/入口名/日志路径的措辞；B4 残留测试扩 5 文件；修复过程
    中再捕获两处同型泄漏（mimics_stop_background.py `_stop_background_tick`
    弹窗 Report 行、超时弹窗 "Check: {1}" stop_log 路径）一并修复，
    B4 禁词精化为 standalone `"Report: ` / `"Check: ` 字面量。
  - **C6-10 文档/账本一致性**：CONFIG_REFERENCE.md:40 undo 状态路径写错
    （ui_state/ → .mimics_runtime/）；99_Admin 编号跳 04、03_Review 跳 05、
    01_Data 跳 03（是否重排需用户拍板——改名动肌肉记忆）；ScribblePrompt.py
    是 21 入口中唯一非动词开头名。菜单重排/改名单列需用户决策项。
    〔R50 补充：`windows_end_to_end_acceptance_2026-08-01.md` §10 DINOv3
    整节 + §13.2 "DINOv3/05_Stop_AI_Task" 为死内容（DINOv3 已按宪法
    淘汰并删实现）；该文档"当前自动化结果"小节的测试计数也已过期。
    同文档其余死入口引用已在 R50 修正，DINOv3 节待下轮清理或整份
    文档按现状重写（需先决定这份 2026-08-01 验收文档的定位）。〕→
    CONFIG_REFERENCE.md 路径改正；验收文档 DINOv3 死内容全部清除
    （§1/§2/§5/§10/§12/§13.1/§13.2/签字表，替换为 FlexiCT 现行入口），
    "当前自动化结果"改为指向回归矩阵实时输出的说明；living-docs 测试
    加 `"02_AI/DINOv3"` 禁引标记。菜单重排/改名维持 D9 需用户决策。

- [x] **C1【P3】死代码清理**：R36 关闭（commit `ae64029`）。删除前逐项 grep 全库
  核实零引用（def + 调用方 + 测试 + 文档）；删除 mimics_label_export 死四件套
  （acquire_gpu_lock_for_job / materialize_label_on_source_grid /
  plan_mcs_label_cache / publish_mcs_label_cache 及连环死的 mcs_label_fingerprint、
  _cached_case_label、_copy_label_atomic、_path_stat_signature，共 289 行）、
  pipeline_common.write_json_best_effort、flexict_pipeline._parse_training_progress
  + EPOCH_MATCH、nnunet_pipeline._split_cases、setup_env._gui_wheels_available、
  nninteractive_mimics._remove_state_file、mimics_export 的
  _active_background_export_lock 与 _release_background_export_lock、根目录
  quick_import.py / quick_export.py 死入口（同名流程在 scripting_library/01_Data/
  有活跃入口：02_Import_Single_Case、08_Quick_Drop_Import、07_Quick_Export_Masks）。
  净 -396 行。**审计条目两处失实已核实并纠正**：write_json_best_effort 并不在
  `__all__`（在 __all__ 的是 write_json_atomic）；"mimics_export 三个 _background_
  export 函数"实际只有两个死函数（_watch_ 与 _finalize_ 均在活跃调用链上，已保留）。
  验证：全部改动模块 import/py_compile 通过；smoke 8/8
  （20260927T125214_smoke.json）；fast 28/28（20260927T125734_fast.json）。
- [x] **C2【P3】断裂引用**：R38 关闭（commit `97c1f4e`）。(a) 删除三处永不命中的
  `adapters/mimics/...` fallback 候选（mimics_import.py ×2 + mimics_export.py ×1，
  真实路径在候选 1/2 必命中，adapters 候选纯误导）；全库 grep 确认代码与文档零
  残留。(b) fewshot 断裂引用按验收标准"归档标注"处理：难例分诊 spec（从未实施）
  加归档横幅（注明未实施 + few-shot 模块 2026-09-23 删除 + 实施规划文档已不存在 +
  现状以 MIMICS_PROJECT_ARCHITECTURE_CN.md 为准）；nninteractive-finetuning spec
  依赖行标注参考模块已删；active_learning 调研文档指向行同样标注。历史文档不重写
  正文（与 b632d70 的"历史文档保持原样"决策一致）。smoke 8/8
  （`20260927T131551_smoke.json`）、fast 28/28
  （`20260927T132104_fast.json`）。
- [x] **C3【P3】重复实现收敛**（R40）：逐项核实后**收敛 1 组、豁免 4 组**，
  每项结论如下：
  - **log 轮转 ×4（runtime_py35）→ 收敛**：create_mcs_batch / mimics_export /
    mimics_import / nninteractive_mimics 四份 `_rotate_log_file`/`rotate_log`
    函数体逐字节相同（仅默认常量不同），且四文件均已 import runtime_common。
    收敛为 `runtime_common.rotate_log_file`（默认 5MB/3 份；nninteractive 的
    10MB 默认由其调用点参数保留），四处原函数变为一行委托。同理
    `tools/mimics_label_export.rotate_log`（pathlib 版）收敛进
    `pipeline_common.rotate_log`。**豁免**：nninteractive_bridge（单文件独立
    部署铁律，见其文档既有豁免）；setup_env（bootstrap 阶段模块绑定
    LOG_FILE）；remote_worker（流式句柄半关闭设计，非文件轮转同构）。
    新增 2 个防回归测试钉住轮转边界语义（shift 链、backups=0 只删主日志、
    缺失路径不抛）。
  - **safe_identifier ×2 → 豁免**：remote_compute（字符循环 + `_+` 坍缩 +
    保留 Unicode）vs nnunet_common（纯正则 ASCII + 96 字符截断）。差分测试
    确认输出实质不同（`a--b__c`、中文、超长串均分叉）；语义差异已固化在
    两套服务器/任务的磁盘路径里。pipeline_common docstring 2026-09 重构时
    已明文豁免 safe_slug 变体，本项同理。
  - **_parse_axes/_parse_flips ×2 → 豁免**：mimics_mask_apply（py3.5 运行时，
    无 default 参数）vs mimics_label_export（py3.13 桥接，带 default）。
    分属两个互不 import 的 Python 层级，收敛需跨层引入依赖，违背改动
    最小化。
  - **_mimics_log ×9 → 豁免**：逐个比对函数体，回退行为实质分化——吞掉/
    返回 bool/print 回退/前缀 print/仅 message 签名（mask_identifier）/
    MIMICS_IMPORT_USE_MIMICS_LOG 环境开关（mimics_import，防宿主不稳）。
    假收敛会改掉这些刻意差异（mimics_import 的 opt-in 是防 Mimics 崩溃的
    保护），豁免。
  证据：commit `af3d201`；smoke 8/8（`20260927T155956_smoke.json`）、
  fast 28/28（`20260927T160614_fast.json`）、full 30/30
  （`20260927T164913_full.json`）。
- [x] **C4【P3】.mimics_runtime 迭代垃圾**：R38 关闭。删除 15 个 tmp_*.py、
  6 个 commit_msg*.txt、box_check*、test_all 计划/日志、gate_p*/flexict 测试日志、
  setup_env.log、convergence 调试日志、debug_out/ 内非 checkpoint 残留
  （full_suite_*.txt、health_preview*.png）。debug_out/ 剩余 326 个全部为
  mimics_import_checkpoint_* 面包屑——受 14 天 prune 钩子管辖（R12/B9-b 落地，
  R32 补测试），非迭代垃圾，保留。纯操作性清理，无代码改动、无 commit；
  checkpoint prune 测试回归通过。
- [x] **C5【P3】弱断言测试**：R39 关闭（commit `6145eb2`）。两个测试重写为真实
  断言：test_cancelled_job_cleans_partial_model 现断言 partial model 目录真被删、
  删除入报、且 workspace 外路径被拒（"outside job workspace" 安全契约——旧测试
  完全未守护，重写时首跑即失败暴露该契约值得钉住）；test_prepare_manifest_with_
  no_cases_raises 从"内联复制管线过滤逻辑再断言副本结果"改为真调
  pipeline._run_label_export 空病例列表，断言 RuntimeError 文案 + guard 前无
  staging 残留。test_mimics_nnint_deep 75/75、fast 28/28
  （`20260927T133501_fast.json`）。
- [x] **C6【P3】待核实**：R39 核实完毕，两项均**保留现状**（不删不改默认）：
  (a) defer_source_alignment_to_worker=False 逃生门——键不在 CONFIG_REFERENCE、
  不在默认 config，用户可发现性为零，实际无人用；但它是 worker 对齐路径
  （GUI 线程安全修复）的配置级回滚保险，删除收益（~2 行分支）远小于回滚
  能力损失，且有专门测试（test_official_model_sync_alignment_still_available_
  when_configured）钉住行为。按铁律 9 风险不对称原则保留，结论记录于此。
  (b) import 启动清扫 opt-in（MIMICS_IMPORT_STARTUP_CLEANUP）——docstring 明示
  "某些 Mimics 构建上 Win32 锁探测/进程检查可致宿主不稳定"，且清扫本体已由
  健康面板 10s 周期与 guard sweep 覆盖（R12），改默认开启违背铁律 1 的
  保守取向。保留 opt-in。
- [x] **C8【P3】零测试数据写入模块补覆盖**：verify_medical_geometry.py 补覆盖
  完毕（R39，commit `8f960bc`）——test_all.py 新增 TestVerifyMedicalGeometry
  6 测试：匹配 grid exit 0、affine/shape 错位 exit 2（fail-closed）、源几何
  不可读 RuntimeError、NRRD mask LPS→RAS 转换语义（identity LPS 方向 →
  RAS diag(-1,-1,1) 不得与 identity 源几何误判匹配）、reference mask Dice
  路径。证据：full 30/30（`20260927T155018_full.json`，test_all 468 项含新
  6 项）。register_brain_model.py、prepare_kidney_3d_append.py 的覆盖**随 D5
  决策**（若 D5 判保留则补覆盖，若判删除则无需）。

## 进行中

（截至 R61 修复阶段（2026-09-29）：四视角评审的 3 个 P0（R61-1/2/3）+
两项测试资产（R61-13 契约扫描、R61-20 扫描表补缺）已当轮修复关闭，
commits `201b516` / `0be6ba5` / `a2d3244` / `afff2bc`，smoke 8/8
（`20260929T124245`）+ fast 27/27（`20260929T130542`、`20260929T131846`）。
契约扫描落地时新登记存量 P1 **R61-26**（nnInteractive 同步桥接阻塞）。
待办区剩余：P1 队列 R61-4~R61-12、R61-26 + P2/P3 精选 + 已拍板待实施
（R61-22 收编、R61-23 批量推理、R61-24 UI 中文、R61-25 FlexiCT 增训）。
**工作区事件登记**：R61 期间 E: 盘一度 100% 满（剩 2.2MB），fast 门禁中
2 个 test_all 用例 Errno 28 瞬时失败、单跑即绿——判定为磁盘满瞬态而非
回归；用户释放空间后恢复（6.3G 可用）。此前状态：

截至 R60（2026-09-28）：**待办区与需用户决策区均已清空**。D8 拍板
"维持现状不杀"（关闭，零改动）；D9 拍板"重排编号 + 改名"，R60 已完成
（commit `761c941`，14 入口重编号 + ScribblePrompt 改名 + 17 处引用
同步，fast 27/27 + full 29/29）。此前待办区已清空——P1 修复链 B20→B22→B21→B23
（R46–R49）闭合历史 FlexiCT 孤儿事故因果链 ①②③；P2 队列 B1–B33 全部
关闭（R50–R58）；P3 批次 C6-1～C6-10 关闭（R59 `027812f`，全量门禁
29/29 + test_all 483）。剩余条目均不在自主迭代权限内：
- **需用户决策**：D1–D6 已全部拍板关闭；仍开放的是 D8（monitor deadline
  到期是否升级为主动杀，推荐维持现状）、D9（菜单编号重排与
  ScribblePrompt 改名）、D11 AI 采纳度度量等产品指标项，以及需求缺口
  提案 D7/D10–D19（未立项不实施）。
- **阻塞（等资源）**：实机 GUI 人工验收清单（DELIVERY_REPORT 两轮遗留
  6 项 + D17 的 Undo 实机验证）——需装有 Mimics 的机器与人工操作；
  远程 FlexiCT 实机验收（P5 遗留）。
- **工作区异常**：2026-09-28 会话中出现 4 个非本项目迭代创建的未跟踪
  文件（runtime_py35/inspect_mcs_batch.py、runtime_py35/
  sync_missing_masks_batch.py、tools/inspect_mcs_projects.py、
  tools/sync_missing_masks.py）——来源不明，未触碰、未提交，待用户确认
  归属与处置。）

## 已解决（保留历史，勿删）

- [x] **A5【P1】nnU-Net 训练表单要求 ML 工程师概念（最小方案路径）**
  - R8 解决（commit 5039fff），按验收标准括号明示的最小方案（默认值+说明）实施，
    **结构收敛（必填项缩减、高级项折叠）完整保留 D2 需用户决策**，本条不宣称
    表单已"收敛"。
  - 实施：`nnunet_training_setup_ui.py` 新增 `_hint()` helper（同型复制既有
    hint 模式）+ 7 处一行说明（Dataset ID / Fold / Validation fraction /
    Preprocessing workers / GPU count / GPU devices / Epochs），四项关键
    hint 存为属性供测试断言。默认值此前已齐（Dataset ID 701 自动建议、
    Fold 0、GPU count 1、GPU devices 留空），本轮补的是可见解释层——
    标注者不悬停，tooltip 不够。
  - 验收对照：至少 Dataset ID/Fold/GPU devices 三项有合理默认与一行说明——
    四项均有（+GPU count）；文案经架构审查逐条核事实（C1 否决点：GPU devices
    原"留空=用所有 GPU"文案有误，实际留空=自动选择且列出的设备数必须与
    GPU count 匹配——这正是提交校验会拒的规则，此前只能靠失败发现；C2：
    validation fraction 注明 Fold=all 时不适用；C5：Fold 措辞只用已验证
    事实：五折之一 held out、all=顺序训五个，来自 `_split_folds` 与
    run_training 源码核证）。走查记录=本条 + 轮次报告。
  - 测试：`test_gui_smoke.py::TestTrainingSetupPathMemory` 新增
    `test_nnunet_form_fields_carry_one_line_explanations`（离屏实例化，断言
    四个 hint 属性存在非空 + GPU devices 规则文案 + Fold 事实），套件
    15/15；fast 门禁（gui_smoke 在 fast 内）见轮次报告。
  - **D2 结构收敛跟进**（R44，commit 见轮次报告）：按用户拍板"折叠为
    '高级设置'区"，Training 组整体（Trainer/Epochs/Fold/Validation fraction/
    Preprocessing workers/GPU count/GPU devices/Pretrained checkpoint/
    Continue/TTA）折叠到"Advanced training settings"切换按钮后，默认收起；
    所有字段默认值与 R8 一行说明原样保留，提交时直接采用默认。测试
    `test_nnunet_training_group_collapsed_by_default` 钉住默认收起 + 切换
    语义（isHidden 断言，离屏窗口不用 isVisible）。

- [x] **A6①【P1】FlexiCT 等待/错误提示指向不可达/不适用的停止入口**
  - R7 解决（commit a33e632）。实施前勘察发现比账本记载更严重：不止
    文案指向 nnUNet 入口（工作区不相交，必得 "No running nnU-Net task was
    found."），**FlexiCT 的 stop/status 动作本身无任何菜单入口可达**（三个入口
    均固定 action，chooser 永不出现）。修复：
    ① 新增 `02_AI/FlexiCT/04_Show_Status_and_Stop.py`（action BUTTON_STATUS →
    外部状态查看器，其 Stop 按钮接真实 FlexiCT workspace 的 stop_job；对齐 nnU-Net
    "Show Status & Models 并可停止任务"先例，不新增第 6 个 Stop 名义入口）；
    ② 资源等待日志（waiting_for_gpu 等）改指新入口；③ 两条结果等待日志（job 已
    completed、查看器 Stop 按钮对终态禁用——架构审查否决点：指过去是新死胡同）
    改为陈述真实机制：结果保留、目标项目重开/占用结束后自动应用（resume 机制
    兜底）；④ 四处文档同步（entry guide、task_lifecycle §2 表新增 FlexiCT 行、
    flexict_mimics.md §3.4 + 组件表、架构文档四入口表）。
  - 验收对照：① 文案不再引用 nnUNet 入口编号——测试
    `test_waiting_messages_reference_reachable_stop_paths`（负向 grep "04 Stop
    Running Task" + 三条新文案正向断言）+ `test_shells_route_to_flexict_mimics`
    增 04 入口行。门禁：TestEntryRouting 4/4、smoke 8/8
    （`20260926T052121_smoke.json`）、fast 26/26
    （`20260926T052851_fast.json`）。
  - **范围裁定**：pending 结果主动丢弃路径不可达（stop_running_task pending
    分支无入口）登记入 A6②/D3——结果自动保留+应用，丢弃需求频率存疑，是否
    值得加入口由用户裁决；菜单重组（②）仍 D3。

- [x] **A3【P1】nnInteractive 官方模型获取路径缺失：新标注者第一天必撞断点**
  - R6 解决（commit 6c5bd3d）。① `env_guidance.collect_issues()` 新增
    `nninteractive_model` 检测：候选链**镜像运行时解析链**（NNINTERACTIVE_MODEL_DIR
    环境变量 → nninteractive_config.json `model_dir` → 找到的外部环境 models 目录 →
    python_env/nninteractive_env 兜底；复用 tools 侧 `official_model_dir` 会因 config
    文件/键不同产生误报漏报，架构审查修正 1），可用性判定与
    `nninteractive_mimics._model_folds` 同型（fold_*/checkpoint_final.pth）；severity
    bad、fix_action ""（信息型，未知动作会渲染坏按钮）。② 两种运行时报错（目录缺失/
    fold 缺失）均给出"从分发源拷贝 + 见 99_Admin → Environment Guidance"指引（py3.5
    .format）；③ `docs/mimics_entry_guide.md` 官方模型入口行同步前置条件。
  - 验收对照：① 检测有测试（TestEnvGuidance 新增 4 个：缺失报 bad / 已装静默 /
    环境变量覆盖 / config 键覆盖，均 pop/restore NNINTERACTIVE_MODEL_DIR 防真实环境
    泄漏；既有空断言测试同步建模型目录）；② 报错文案含可执行获取指引；③ 入口表
    已同步。门禁：TestEnvGuidance 12/12、test_migrate_root 9/9、smoke 8/8
    （`20260926T045239_smoke.json`）、fast 26/26（`20260926T045940_fast.json`）。
  - **范围裁定**：检测+引导+报错文案已闭环；获取/分发方式本身（下载/离线包/共享
    目录）仍为 D9 需用户决策，不在本轮。行为变化披露：无官方模型的机器上 System
    Health 会持续显示"1 environment issue(s)"——预期行为（问题真实存在）。

- [x] **A9【P1】nnU-Net 任务目录无任何 retention/清理（磁盘无界增长）**
  - R5 解决（commit fa0bcf6）。① `nnunet_pipeline._sweep_expired_jobs` 镜像 flexict
    版（:513）：终态 job 过保留天数清内容留 status.json；调用点在 run_training 与
    run_inference 两处 finally（对齐 flexict 三控制器 finally 模式，架构审查修正 1）；
    保留天数进新建 `nnunet_config.json`（单键 `job_retention_days` 默认 30，0 禁用；
    nnU-Net 此前无 config 文件，最小 loader 镜像 flexict_common.load_config）+
    CONFIG_REFERENCE.md 同步；TERMINAL_STATES 取自 nnunet_common（含 abandoned）。
    ② 伪造 mtime prune 测试 6 个（TestSweepExpiredJobs：清内容留 status / 新近保留 /
    在跑不删 / reattaching 不删 / retention=0 禁用 / models+cache 显式不受影响断言），
    全部 tmp 目录 + 显式 config dict（不依赖默认 config 路径）；套件 73/73 绿。
    ③ fast 门禁 26/26 全绿（工件 `20260926T042044_fast.json`）。
  - **范围裁定**：jobs/ 清扫已闭环；痛点原文的主要磁盘源（runtime 三目录 +
    cache/source_grid）残留为新条目 A11——runtime 涉及 Dataset 在用判定与 preprocess
    复用语义，需单独设计方案，本轮不扩范围。

- [x] **A10【P1】nnU-Net 预处理 worker 在内存紧张机上 OpenBLAS 崩溃→静默死锁（训练永久挂起）**
  - R4 解决（commit b33a399）。① `_worker_environment` 对全部 stage、全平台 setdefault
    `OMP/MKL/OPENBLAS_NUM_THREADS=1`（与 mimics_batch_cli 先例及 nnU-Net 官方
    run_training.py 入口同型）；架构审查发现的缺口同步修复：`flexict_worker_environment`
    （preprocess/train）与 `flexict_infer_environment`（infer）整体 monkeypatch 替换
    `_worker_environment`（flexict_pipeline.py :596/:983/:1197 三处），FlexiCT 预处理走
    同一条 spawn.Pool 指纹提取链，故三处全部镜像封顶；② 本机复跑
    `tools/test_training_convergence.py` 独立通过（2/2 OK，1261s，无挂起——R2 时 3600s
    超时死锁）；③ 防回归测试 3 个：test_nnunet_integration（三 stage 断言）+
    test_flexict_integration（worker env 三 stage + infer env 各一），套件 67/67、74/74 绿；
    ④ full 门禁 28/28 全绿（工件 `20260926T035220_full.json`，含 training_convergence
    1231s 通过）。封顶随
    spec environment 流入远程容器（共享服务器更需封顶，符合资源红线）；Phase 4 远程
    验收（10add23）跑在封顶之前，train stage 行为等价（Action3 已在进程内 setdefault
    同三变量），无需重跑远程验收。
  - 残余风险：上游 nnU-Net Pool 对崩溃 worker 静默 respawn 不可控，封顶是"不让 worker
    崩"而非根治 respawn；若未来出现其他崩溃源（如显存耗尽），同型挂起仍可能出现
    （上游问题，已在此登记）。

- [x] **A2【P1】回归测试污染生产状态（用户数据被测试覆盖/垃圾无限累积）**
  - R3 解决（commit 6e85b65）。① 两测试隔离：test_all 的 batch CLI runner 测试改用
    `MIMICS_IMPORT_RUNTIME_DIR`（库自带隔离钩子，运行时读取）指向临时目录，
    fake_mimics_flow_test 的 entrypoint 测试 patch 缓存模块 `_state_path`（与 :902
    同型）+ try/finally 恢复；② 生产垃圾清理：`.mimics_runtime/import_queues/`
    153 个目录（删除前逐一核查：74 空 + 77 引用 Temp\mimics_test_* + 2 引用
    Temp\cli_dbg_*，无一真实生产队列）、`.mimics_runtime/fewshot_mimics_state.json`
    （内容全部为 fake-mimics 临时数据集路径，不删会被 `_migrate_old_settings` 迁入
    正式设置）、`.mimics_runtime/window_level_state.json`（内容为 fake 场景 0-100
    对比度，即测试数据）全部删除；③ 防复活断言：两测试各自快照生产目录/文件前后
    一致（`_snapshot_dir` / bytes 比较）；④ smoke 8/8
    （20260926T005016_smoke.json）+ fast 26/26（20260926T015355_fast.json）绿。
  - 范围裁定：痛点中"locks/ 345+ guard"由 B2 承接（B2 含 guard mtime 清理 +
    import 队列 prune，是同一痛点的产品侧修复）；测试侧隔离与一次性清理由 A2
  关闭。架构审查另有 2 项建议已顺手落实：:902 兄弟测试补 finally 恢复（模块
  单例不再遗留指向已删临时目录的 patch）。

- [x] **A1【P0】Collect Diagnostics 在 Mimics 主线程同步阻塞最长 300 秒**
  - R2 解决（commit c651d9a）。`collect_bundle()` 改为 daemon 线程 + 非阻塞 timer
    轮询（镜像 `mimics_stop_background.clear_cache_main` 模式：Win32 SetTimer →
    PyQt5 QTimer → daemon 线程三级回退），立即返回并先弹"后台收集中"提示；重复
    点击合并进在跑任务。验收：① 代码走查——collect_bundle 源码无 subprocess.run，
    子进程调用仅在 worker `_collect_in_background` 内 ✅；② 防回归测试 2 个
    （source-walk 断言线程/timer 模式 + fake-mimics 端到端验证后台流程）✅；
    ③ smoke 8/8 绿（工件 20260925T221806_smoke.json，含 flow_imports 的 py3.5
    可导入检查）✅；④ 因属 P0/核心流程，另跑 `--profile full`
    （`20260925T235429_full.json`，27/28：除 training_convergence 外全绿；该套件
    超时经 R3 查明为本机 OpenBLAS 死锁环境问题、非 A1 改动引入，已立 A10 跟踪，
    A1 验收不以该套件为证据）。

- [x] **A8【P1】远程验收清单：超时不停止训练容器 + verify_weights 假 PASS**
  - R1 解决（commit 4e23062）。① 超时/异常/KeyboardInterrupt 统一经
    `_request_stop()` 写 `control.json {"action":"stop"}` 并等终态至多 180s（走
    controller 自有通道，不做并发 CLI cancel）；② verify_weights/local_inference
    改用 `_resolve_local_model_dir()`——`status["model"]["model_dir"]` 本地注册
    回写路径，缺失/容器路径/不存在目录一律 FAIL，空 checkpoint glob 也 FAIL；
    ③ 单测：AcceptanceChecklistTests（超时 stop、缺目录 FAIL、坏 checkpoint
    FAIL、未注册路径 FAIL）；④ test_remote_training 81/81 绿。
- [x] **B11(a)(b)【P2】远程 .part 残片与 job 归档永不清理**
  - R1 解决（commit 4e23062）。30 天缓存清理补 `*.part`/`*.tmp`（7 天阈值）；
    `_remove_remote_job` 同时删 `jobs/<id>.tar` 与 `.tar.part`；单测
    RemoteJobCleanupTests。残留 (c) 失败诊断目录管理入口单列待办。
- [x] **B16【P2】验收 preflight nerdctl 回退与后续步骤不一致**
  - R1 解决（commit 4e23062）。回退逻辑删除：runtime 与 profile 不匹配是
    profile 配置错误，preflight 直接 FAIL 并提示在 profile 里设
    container_runtime='nerdctl'。
- [x] **C7【P3】远程小杂项**
  - R1 解决（commit 4e23062）。--skip-reattach 死参数与 docstring 第 7 步删除；
    step_dataset 假 staging 目录删除；`_sync_remote_nnunet_curve` 补 `..` 拒绝
    （单测）；json.dumps 注释故障模式改正（strict 永拒/warn 误报）。
    **豁免项**：`_append_log` 轮转——日志为 per-job 文件，随 job 目录清理有界，
    加轮转属防御式编程，按宪法复杂度预算豁免（架构师审查时认可该判断）。

## 已删除项（附理由）

- [x] **D1 三个重复菜单入口**（用户 2026-09-27 拍板"上述三个可以删除，
  但是需要明确是否保留的命名合理，目标一致，用户方便操作"）
  - R41 解决（commit `5ca3623`）。删 `03_Review/05_Window_Reset_Full_Range`
    （Choose Preset 对话框内已有同一 `reset_full_range()` 按钮 + undo 回退）、
    `02_AI/nnUNet/04_Stop_Running_Task`（状态查看器内已有完整 Stop 流）、
    `01_Data/03_Export_Masks`（07_Quick_Export_Masks 为超集）。
  - 保留入口审计：03_Review 剩 Choose/From Selected Mask/Undo/Edit Presets
    （一事一入口，动词开头）；nnUNet 剩 Train/Predict/Show Status and Models
    （Stop 内置于状态查看器，与 FlexiCT 04_Show_Status_and_Stop 同构）；
    01_Data 导出仅 07_Quick_Export_Masks 一个活入口。命名一致、目标不重叠。
  - 验收：smoke 8/8（20260927T221402_smoke.json）、fast 27/27
    （20260927T224936_fast.json）、full 29/29——training_convergence 首跑
    3600s 超时（flexict preprocess worker 孤儿挂起，环境性；单独重跑
    1012.8s OK，且 nnunet 家族同轮通过、R41/R42 未触碰 flexict 训练路径；
    详见轮次报告 R41/R42）。
- [x] **D5 死模块**（用户 2026-09-27 拍板"确保删除的不是会使用的功能就好"）
  - R42 解决（commit `a3371f0`）。**审计勘误先行**：账本原记"整目录死代码"
    有误——`Action2/3/4`、`ResampleImageAndMask.py`、`trainers/`、
    `ModelMap.toml` 是活代码（nnunet_stage_worker 直接 import），全部保留。
  - 实删：standalone TOML CLI（AutoSegmentationFramework + SetEnvionmentVariables
    + Action1 convert + Action5 evaluate + 其专属 helper ImageConvertor/
    DicomToMhd/MaskOperation + 8 个 Config_*.toml + 2 个 ModelMap 变体 + gpu*.sh
    + temp.sh + SmallTools.ipynb + ANNOTATION_VERSIONING.md）、
    test_annotation_version.py（连矩阵注册一起删，测的是死 convert 阶段的
    标注版本回退）、`tools/nifti_to_dicom.py`、`tools/prepare_kidney_3d_append.py`。
    删除前逐文件全仓引用扫描，零活引用。
  - 净 -8653 行。验收：fast 27/27 + full 29/29（同 D1 工件）。
- [x] **D4 Fix Affine 菜单入口**（用户 2026-09-27 拍板"合并进报错路径后删入口"）
  - R43 解决（commit `cb0f898`）。合并而非单纯删除：标注者此前要"读报错→
    记住菜单路径→切 99_Admin 找工具"，正是引导式失败路径要消除的流程。
  - 实施：`nnunet_pipeline.py` 的 `validate_materialized_source_geometry` 抛错
    带稳定标记 `source geometry mismatch`（本地/远程 nnU-Net 与 FlexiCT 共用
    该校验器，一处改全覆盖）；`fix_source_affine_metadata.py` 新增
    `offer_repair_for_prediction_failure()`（检测标记→一次询问→接受则启动
    既有非阻塞修复流；绝不 raise，失败不遮蔽原始报错）；nnunet_mimics /
    flexict_mimics 预测失败分支先查标记，接受即修复、拒绝即原报错不变。
    删除 `99_Admin/04_Fix_Source_Affine_Metadata.py` 入口；修复模块本体保留
    （仍是修复实现，且注册在 mimics_stop_background 的监视列表）。
  - 验收：smoke 8/8（20260928T010326_smoke.json）、fast 27/27
    （20260928T011902_fast.json）；test_all 更新
    `test_prediction_context_errors_point_to_real_repair_paths` + 新增
    `test_prediction_failure_offers_repair_on_geometry_mismatch`
    （无关错误不弹窗/无活动图不弹窗/接受即修复/拒绝不动）。

## 需用户决策 / 阻塞（等资源）

需求缺口提案格式（需求视角审计的产出放这里，等用户立项）：

> **需求**：（一句话描述没被满足的真实需要）
> **使用者/频率**：（谁、多久遇到一次）
> **不解决会怎样**：（现在的变通方式及其代价）
> **最小可行方案**：（不做大，先满足核心场景的方案）

### 重复入口与功能价值存疑（需用户逐项拍板，默认不擅动）

- **D1 入口合并候选**：✅ 已拍板并完成（R41，见"已删除项"）。
- **D2 nnU-Net 训练表单收敛方案**（A5 的产品设计面：收窄到哪些必填项、哪些折叠）。
  ✅ 已拍板"折叠为'高级设置'区"（2026-09-27），已完成（R44，见"已删除项"
  同级记录于"已解决"区 A5 条目下）。
- **D3 五个 Stop 入口的重组方案**（A6 的产品设计面：按用户意图重组 vs 仅靠导航层）。
  ✅ 已拍板维持现状（2026-09-27）。关闭。
- **D4 `99_Admin/04_Fix_Source_Affine_Metadata` 退役条件**：✅ 已拍板"合并进报错
  路径后删入口"（2026-09-27），已完成（R43，见"已删除项"）。
- **D5 死模块处置**：✅ 已拍板并完成（R42，见"已删除项"）。
- **D6 nnU-Net/FlexiCT 菜单组织统一**：✅ 已拍板维持现状（标注者视角分析：
  两框架各自入口结构与任务重量匹配，2026-09-27）。关闭。
- **D8 monitor deadline 到期是否升级为主动杀**（2026-09-28 并行审计发现 5
  的拆出项）：当前 nnunet_mimics/flexict_mimics 的 monitor deadline（14 天/
  24h）到期只停监视器、不停外部任务，日志记一句"the external task was not
  stopped"。升级为主动杀有误杀长训练（合法跑到 deadline 附近）的风险；
  B20/B21/B23 落地后三层兜底闭合，主动杀的必要性下降。✅ 已拍板**维持现状
  不杀**（2026-09-28，用户确认推荐项）。关闭。
- **D9 菜单编号重排与 ScribblePrompt 改名**（C6-10 拆出）：R41/R43 删除后
  01_Data 跳 03、03_Review 跳 05、99_Admin 跳 04；ScribblePrompt.py 是 21
  个入口中唯一非动词开头名。重排/改名会让标注者肌肉记忆失效，需用户拍板
  是否值得。✅ 已拍板**重排编号 + 改名**（2026-09-28）。R60 解决（commit
  `761c941`）——14 个入口 `git mv` 编号连续化 + ScribblePrompt →
  Annotate_With_ScribblePrompt；17 处活文档/测试引用同步；带日期的
  DELIVERY_REPORT*/docs/changes 历史记录不追改。

### 需求缺口提案（R1 需求视角审计，均 P2 建议、未立项不实施）

- **D7 AL Review UI "Open Case" 在外部看图软件打开 NIfTI 而非 Mimics 内打开 .mcs**
  （使用者：AL 用户每 case 一次；变通：人肉到 mcs_output 翻找；最小方案：复用
  apply_request 握手让 Mimics 打开对应 mcs）。
- **D8 数据集级标注进度总览**（"1184 例里哪些已标/未标/跳过"，数据已在
  dataset_manifest + annotation_state，只差只读面板；可与 D9 复用 batch_status_viewer
  框架）。
- **D9 复核（QA）工作流**：标注→复核→打回状态流转（需用户先答：是否存在复核流程、
  团队规模是否 ≥2）。
- **D10 AI 预测 vs 人工标注差异可视化与量化**（导出明细附 Dice；Review 菜单
  "Compare Mask With Draft"）。
- **D11 AI 采纳度度量**（记录 Update/Copy/丢弃计数 + 修正 Dice，Model Center 加现场
  采纳统计列；与 D10 同管道，建议捆绑评估）。
- **D12 导出漂移检测**（.mcs 改了、导出标签没跟上：训练 Scan Data 时对已导出 case
  加 stale 列 + 一键重导；manifest 时间戳数据已具备）。
- **D13 空/异常 mask 导出与训练校验**（=TB-04 的产品面；导出明细与 Scan Data 统计
  体素数，0 或 <100 标黄确认）。
- **D14 远程共享服务器忙闲预检**〔已解决，R33：用户 2026-09-27 拍板"要硬阻断"。
  Test Connection 扩展（Check & Show Load）已于 R29/TB-06 落地（nvidia-smi 利用率
  + 逐卡点名软提示）；R33 补齐启动时硬管控——`assert_gpus_not_busy` 门（复用
  R29 判定逻辑与共享 GPU_QUERY_COMMAND）在 run() 两处执行：远端目录建立后、
  任何上传之前（被挡任务只花一次 SSH 往返）+ 容器启动前再查一次（数据集上传
  可达数小时，挤占时刻是启动那一刻）。策略键 `gpu_busy_policy`（block 默认/
  warn/off，UI 表单 + CONFIG_REFERENCE 同步；warn 模式写状态文件而非仅日志）。
  查询失败 fail-closed 抛错。架构审查"修改后通过"10 项必改全落实。Test
  Connection 保持软诊断并加"Block 模式下将拒绝启动"预告。commit `e6fad56`，
  fast 26/26（`20260927T020155_fast.json`）。docker ps 维度未做（架构审查
  范围外，无先例需求）〕
- **D15 模型与权重分发链路**（FlexiCT backbone 2×576MB 仍靠手工拷贝；诊断包无传递
  通道；最小方案：backbone 纳入 ai_model_bundle 家族 + 诊断包"复制到共享目录"按钮）。
- **D16 跨标注员标签命名词汇表**（Mask 名即导出标签名，命名不一致训练映射即断；
  Model Center 手填 alias 框就是变通痕迹；最小方案：任务级 alias 表持久化 + 新任务
  预填）。
- **D17 AI 覆盖写入前快照回退**（task_lifecycle §5.5 承认脚本事务是否进 Mimics 原生
  Undo 从未实机验证；第一步零代码：在目标 Mimics 版本实机验证一次，不可用则加
  "Restore Last Overwritten Mask"）。
- **D18 孤儿测试套件处置**〔已解决，R34：用户 2026-09-27 初拍板"删掉但先核查
  覆盖"，核查发现两套覆盖的功能无任何其他测试且引擎本体在用，用户改选注册进
  矩阵；见 B9-a 登记项〕。
- **D19 FlexiCT 增训（continue_training）缺口**〔2026-09-28 R45 审计 P2-4 补登：
  `tools/flexict_pipeline.py:851` 训练参数硬编码 `"continue_training": False`，
  而 nnU-Net 路径（`integrations/nnunet_segmentation_workflow/Action3_Train.py:51`）
  已暴露该参数。使用者：想用新标注病例增量训练已有 FlexiCT 模型的标注者；
  频率：每次主动学习迭代（周级）。不解决会怎样：每次"增训"实际从 backbone
  从头训练，烧掉整轮 GPU 时长，与主动学习工作流的核心价值矛盾；变通是接受
  全量重训。最小可行方案：训练表单加"从上次 checkpoint 继续"选项，透传到
  `_spawn_flexict_worker` 参数（trainer 侧已支持，见
  `integrations/flexict-finetune/tests/test_flexict_pkg.py:239` 注释）。未立项
  不实施〕。✅ 2026-09-29 用户拍板：**立项，低优先级**（登记为待办
  R61-25，先修完本轮 P0/P1 再做）。

### 已拍板（2026-09-29，四视角评审后）

- **外来脚本处置**：✅ 用户拍板**收编**——4 个未跟踪工具
  （runtime_py35/inspect_mcs_batch.py、sync_missing_masks_batch.py、
  tools/inspect_mcs_projects.py、sync_missing_masks.py）+ .mimics_runtime
  外来脚本（含 copy_to_z.py）纳入版本管理，补资源锁/保留期/测试/账本
  登记（R61-22）。copy_to_z.py 的 Z: 数据根写入路径（新建
  `label_task_mr_synced` 目录，非 v201 内部）单独审视后收编或改写。
- **批量推理**：✅ 用户拍板**立项，下轮实施**（R61-23）——mimics_batch_cli
  加 predict 子命令。
- **UI 语言**：✅ 用户拍板**统一中文，但注意编码问题不要显示乱码，部分
  错误可以英文显示**（R61-24）——技术错误（traceback、异常类名等）
  保留英文防 GBK 乱码，界面文案统一中文。
- **D19 FlexiCT 增训**：✅ 立项低优先级（R61-25）。

### 阻塞（等资源）

- 实机 GUI 人工验收清单（DELIVERY_REPORT 两轮遗留 6 项 + D17 的 Undo 验证）——需
  装有 Mimics 的机器与人工操作。
- 远程 FlexiCT 实机验收（P5 遗留）——需按 B12 先修 remote_root 红线问题再执行。

## 测试盲区登记表

> 测试深度审计发现的盲区在此登记，修复后转"已解决"。评估维度：
> 发生概率 / 用户损失 / 当前防护。

### TB-01 导出/导入大文件（NIfTI/DICOM 序列）写一半被硬杀后残留与恢复【已解决，R28，commit `54bbcbb`】
- **场景**：导出 mask（`mimics_bridge.write_mask_nifti`）、导入转换
  （`nifti_to_derived_dicom` 逐 slice 写 DICOM）过程中进程被杀/断电/磁盘满。
- **链路位置**：`runtime_py35/mimics_export.py` → `mimics_bridge.py` 导出流；
  `runtime_py35/mimics_import.py` → derived DICOM 写盘。
- **现有防护**：JSON 状态写入原子（`write_text_atomic`，test_all 有覆盖）；
  但 NIfTI/DICOM 数据文件本身的半写残留无测试（现有
  `test_mask_nifti_publish_is_atomic...` 只 mock os.replace 失败，未模拟
  写入中途被杀后磁盘上的半文件被下一轮当作有效输入）。derived DICOM
  `_validate_derived_dicom_series` 会校验 slice 完整性（代码级防护），但
  无自动化测试。
- **发生概率/损失**：概率中（后台进程被杀/机器断电在实际部署发生过）；
  损失=白干活+残留脏文件可能污染下一轮导入（数据损坏级）。
- **建议测试形态**：fake 流中模拟"写第 N 个 slice 时 kill"，重启后断言
  阶段目录被清理/标记 failed，且重跑产出完整结果；断言无半写文件被
  后续 discover 当作有效 case。（与 B9-c 的名不副实 kill 测试合并解决）
- **R28 核实结论**：纯测试轮（产品零改动）。逐读源码核实三条路径的
  防护已成立——(1) NIfTI 导出原子发布（tempfile.mkstemp 临时名 →
  nib.save → os.replace，杀进程只留临时名，最终名要么旧要么新）；
  (2) derived DICOM 逐 slice 直写**确实半写可见**，但被三重隔离兜住：
  每次导入独立 run 目录（import_runs/<ts>_<pid>_<uuid>）、manifest 仅
  校验通过后写入、消费端（create_mcs_batch）以 manifest 存在性为唯一
  门（且 >5s 陈旧 descriptor 丢弃），错误路径 `_cleanup_work_dir`
  rmtree；(3) 批量导入监视器错误分支已调 `_cleanup_work_dir`。缺的
  只是这些保证的防回归测试。
- **R28 实施（commit `54bbcbb`，含 B9-c 合并解决）**：新增 6 个测试改动（产品 0）：
  1. `test_derived_dicom_kill_mid_slices_leaves_unpublished_residue_and_
     rerun_recovers`——FileDataset.save_as 写到第 4 个 slice 时抛
     KeyboardInterrupt：断言残留只落在隔离 run 目录（slice_0001-0004，
     最后一个为截断小文件）、重跑产出完整 8-slice series；
  2. `test_derived_dicom_validation_rejects_incomplete_series`——
     校验器三失败路径（slice 数不匹配/空文件/SeriesInstanceUID 不匹配）
     各自 RuntimeError + 完整 series 通过；
  3. `test_mask_nifti_kill_mid_write_never_publishes_partial_file`——
     nib.save 中途杀：既有完整导出逐字节不变、临时名从未变成最终名、
     重跑正常发布 (4,4,4)；
  4. `test_partial_work_dir_without_manifest_is_never_converted`——
     无 manifest 的半成品 work_dir 永不被 create_mcs_batch 消费
     （descriptor 陈旧化丢弃路径）；
  5. `test_bridge_error_cleans_up_partial_derived_dicom_work_dir`——
     监视器错误分支：failed+1、work_dir 被 rmtree、下一 case 启动；
  6. B9-c：`test_atomic_write_survives_kill_signal`（名不副实，从未
     模拟 kill）改名 `test_atomic_write_publishes_exactly_one_complete_
     file` 并重写——真实构造截断 JSON 证明读者必须拒绝，对比原子写
     全有或全无。
- **实施与验收对照**："写第 N 个 slice 时 kill"→ 测试 1（KeyboardInterrupt
  恰在 slice_0004 写后抛出）；"重启后断言阶段目录被清理/标记 failed"→
  测试 5（错误路径 rmtree + failed 计数）与测试 4（无 manifest 不消费）；
  "重跑产出完整结果"→ 测试 1/3 断言重跑 series 完整（8 slices / 正确
  shape）；"无半写文件被后续 discover 当作有效 case"→ 测试 2（校验器
  拒截断）+ 测试 4（manifest 门）。
- **证据**：full 门禁 28/28 绿、test_all 456/456 OK（991s，
  `.mimics_runtime/regression/20260926T172647_full.json`）。
- **顺带修复（R12 遗留测试破损）**：R12 commit `1e9c9f8` 给 server
  state 加 `gpu_lock.path` 后 FakeLock 未同步，两个 nninteractive 启动
  失败路径测试（releases/retains lock）自 R12 起一直 error——fast
  门禁不含 test_all 故不可见，本轮 full 门禁跑出后修复（FakeLock 补
  `path` 属性）。教训已入账：改产品属性时 grep 测试 fake。
- **残余风险**：run 目录里的半写 DICOM 残留会保留到下一次成功导入或
  手动清理（隔离目录内、无 manifest、不会被消费）——风险可接受，
  不引入自动清扫（防御式编程）。

### TB-02 同目录多 DICOM series 混放（含缺失 SeriesInstanceUID）的导入选择【已解决，R24】
- **场景**：一个 case 目录里混入两个 series（如平扫+增强、或拷贝残留），
  或 slice 缺 `SeriesInstanceUID`/`ImagePositionPatient`。
- **链路位置**：`nninteractive_bridge._select_dicom_records` /
  `load_image_dicom_folder` / `mimics_bridge.is_dicom_folder`。
- **核实结论**：疑点属实，是真实 P1 数据损坏 bug——mock 复现确认全缺 UID
  的两个 series 在全部三条选择路径（无 shape / shape 匹配 / mismatch 允许）
  静默混叠成一个 6-slice 乱序 volume。
- **实施与验收对照**（R24）：`_dicom_group_key` 缺 UID 时改按
  `(StudyInstanceUID, SeriesNumber)` 复合分组（SeriesNumber=0 保留为真值）；
  新增 `_reject_interleaved_missing_uid_groups`——非真 UID 组内出现两条同
  （2 位小数取整）ImagePositionPatient 即 fail-closed（覆盖平扫+增强同格
  与拷贝残留双场景；真 UID 组豁免，多 UID 同格仍走既有 "Multiple DICOM
  series" 报错）。架构师审查通过（3 项加固全部落实：None/0 哨兵区分、
  检查覆盖所有复合组而不仅全缺组、无 IPP 记录跳过）。测试 6 个进
  TestBridgeDicomLoading。smoke 8/8（20260926T140745）、
  fast 26/26（20260926T141207）。
- **残余风险登记**：两个偏移网格 series（平扫 z=0,10…、增强 z=2.5,7.5…）
  缺 UID 且同 (Study, SeriesNumber) 时无重复 IPP，仍可能混入同组——
  expected_shape 路径由最终 shape 复查兜底；间距均匀性检查属启发式，
  误报风险大于收益，不实施（架构师裁定）。

### TB-03 路径含中文/空格/超长路径（>260）端到端链路【已解决，R25（长路径除外）】
- **场景**：标注者数据集目录或输出目录含中文（中国医院环境常态）、
  空格、嵌套超过 MAX_PATH。
- **链路位置**：全链路（import → bridge → export → 训练 worker →
  远程上传命令拼接）。
- **实施与验收对照**（R25，commit `8a3e5b6`）：(1) fake flow 新增
  `unicode and spaces paths flow`——在 `患者数据 2026/标注 输出/` 根下跑完
  mask buffer 导出（含中文名 mask）+ manifest 回读 + prepared mask 导入
  apply 全链；(2) 远程契约测试 `test_remote_job_removal_quotes_spaces_and_
  unicode_in_paths`——remote_root 含空格+中文时 `shlex.quote` 拼出的删除
  命令经 `shlex.split` 反解析必须还原为完全一致的三个目标路径（证明无
  词切分/无 shell 重解释）。全部通过无需产品代码改动（链路本身健壮，
  缺的只是防护测试）。smoke 8/8（20260926T142037）、
  fast 26/26（20260926T142453）。
- **残余（未解决，另立）**：超长路径（>260）——Windows 无长路径处理代码，
  需 `\\?\` 前缀或注册表 longPathsEnabled，涉及面广且现代 Win10 + Py3.13
  默认启用，实际风险低；如需处理应作为独立条目评估。

### TB-04 空 mask / 全零 label 病例进入训练【已解决，R26】
- **场景**：标注者导出时某 case 的 mask 全空（漏标），或 label 文件
  存在但全零，进入 nnU-Net/FlexiCT 训练。
- **核实结论**（架构师审查纠正认知）：本地数据集路径其实已有防护——
  `prepare_source_grid_cases` 默认 `empty_case_policy="skip"`，空 label
  case 被跳过并逐 case 报告（`skipped_case_details`）；真正的洞在
  远程/prepared 分支（两 pipeline 的 `run_training`），那里只查文件存在。
- **实施与验收对照**（R26，commit `b75126a`）：在 `build_training_data_
  profile`（两 pipeline × 两执行路径的唯一共享内容门，已逐 case 加载
  image）同步加载每个 label，全零即 RuntimeError（含 case id + label
  路径 + 重导/移除指引）；不信任缓存 foreground_voxels 快路径（陈旧
  缓存缺键会误拒好 case，架构师裁定）。profile 增补 `label_foreground_
  voxels`（min/median/max，加性键不影响指纹/缓存身份）。测试 3 个：
  直接单测（全零拒 + 正常过含统计）、flexict prepared 路径集成测试
  （镜像 `test_prepared_case_missing_file_fails_before_training`）。
  smoke 8/8（20260926T144313）、fast 26/26（20260926T144750）。
- **产品面（D13，仍需用户决策）**：导出明细与 Scan Data 统计体素数、
  0/<100 标黄确认——产品级 UI 增强，未包含在本轮（fail-closed 门已满足
  验收"拒绝或明确标记"）。

### TB-05 远程训练中网络中断在"上传/下载中段"（非日志轮询阶段）【已解决 R27】
- **场景**：上传数据集压缩包或下载训练产物到一半时 SSH 断开/服务器重启。
- **链路位置**：`remote_training_controller` 传输路径（`_upload`/
  download）与 `_reconnect_session`。
- **现有防护**：较好——续传测试存在（`test_upload_resumes_existing_remote_
  part`、`test_download_resumes_existing_local_part`）、SHA-256 校验拒绝
  损坏归档（`test_corrupt_download_is_rejected_before_model_publish`）、
  重连预算有界（`test_reconnect_attempts_are_bounded`）、reattach 覆盖
  本地控制器死亡。缺口：没有测试模拟"传输函数执行中途抛
  socket.error"后控制器进入什么状态、半传残片是否影响重试语义（现有
  续传测试构造的是"已有残片"的静态场景，不是中断时序）。
- **发生概率/损失**：概率中；损失=可控（有防护）但中断时刻的状态机
  未验证，最坏=卡在非终态需要人工清理（白干活）。
- **建议测试形态**：mock SFTP write 在第 N 块后抛 OSError，断言状态落
  到可重试/可 reattach 的态、无锁残留、重试后产物 SHA 校验通过。
- **R27 核实结论**：纯测试轮（产品零改动）。逐读控制器源码核实：
  全部三类传输位点已各自包裹重连重试循环——数据集分片上传
  （remote_training_controller.py:4310）、作业归档上传（4378）、产物
  下载两处（fresh-run 1892 / reattach 2157），外加监控（3970）与容器
  启动（4446）共 6 处 `except (RemoteComputeError, EOFError, OSError,
  socket.error) → _reconnect_session` 循环；`.part` 续传语义保证重试
  从断点续传而非重来；中断时状态先落 `reconnecting_remote`（非终态，
  remote_state_unknown=True），重连成功恢复原状态——即验收所要求的
  "可重试/可 reattach 的态"在产品里已成立，缺的只是中断时序的防回
  归测试（与 TB-03 同型：链路健壮，防护缺测试）。
- **R27 实施（commit 5ee0045）**：新增 4 个测试（tools/test_remote_
  training.py，91 通过）：
  1. `test_upload_interrupted_midway_keeps_part_for_retry`——SFTP 写句柄
     第 1 块后抛 OSError：`.part` 恰持有已落盘字节、目标文件不存在、
     健康重试从断点续传（progress 首回调=断点）且完成后无 `.part` 残留；
  2. `test_download_interrupted_midway_keeps_part_for_retry`——下载孪生：
     远端读第 1 块后死，本地 `.part` 恰为收到的字节前缀，重试续传完成
     且产物逐字节等于源；
  3. `test_download_interrupted_midway_recovers_via_reconnect`——控制器
     级端到端：驱动 `_finalize_completed_remote_job`，模拟下载中途
     OSError：断言状态先落 `reconnecting_remote`（reattach 兼容，非终态
     failed）、重连后恢复并完成（status=completed）、`remote_reconnect_
     attempt=1`、下载恰 2 次（1 失败 + 1 续传）、产物 SHA-256 校验通过
     并发布（manifest 逐字节一致）、无 `.part` 残留；
  4. `test_transfer_sites_wrap_interruptible_calls_in_reconnect_retry`——
     源契约防复活：控制器源码中重连重试块 ≥5 处（当前 6），防止未来
     重构静默删掉传输位点的重试循环。
- **实施与验收对照**：验收"mock SFTP write 在第 N 块后抛 OSError"→
  测试 1/2 的 `_RemoteHandle` 恰在 N 块后抛 OSError；"断言状态落到可
  重试/可 reattach 的态"→ 测试 3 断言 `reconnecting_remote` + 恢复原
  状态 + reattach 的 TERMINAL 集不含该态；"无锁残留"→ 传输阶段容器
  未启动、锁目录尚未创建（容器 4423 行才标记 may_exist），测试 3 亦断
  言无 `.part` 残留与成功清理路径；"重试后产物 SHA 校验通过"→ 测试
  3 端到端经 `_download_artifact` 的 SHA-256 比对（remote digest ==
  local digest）并发布。
- **证据**：`tools/test_remote_training.py` 91/91 OK（47s）；fast 门禁
  26/26 绿（20260926T152910）。
- **残余风险**：监控期控制文件上传（3834）与曲线同步（3740）不在重试
  循环内，但均被外层监控循环的 3970 重试块捕获（曲线同步 IO 异常单独
  try/except 吞掉），非传输主干——不构成白干活路径，不另行立项。

### TB-06 远程共享 GPU 被他人任务占用时的行为【已解决，R29（软提示，commit `c4cc660`）；启动时硬管控已由 R33/D14 落地（见上 D14 登记项）】
- **场景**：服务器 GPU 正被别人占用（无 MIMICS 锁），本项目远程任务启动。
- **链路位置**：`tools/remote_worker.py _remote_gpu_lock`（fcntl 锁，只管
  本项目自己的容器）；`remote_compute.py` 只查 GPU 存在（index/uuid/name/
  memory.total），不查 memory.used/utilization。
- **现有防护**：项目自有容器间互斥有实现；对"他人进程"零防护零测试。
  CLAUDE.md 红线要求"先查后用、不得挤占"，但工具层面无检查。
- **发生概率/损失**：概率高（共享服务器）；损失=挤占他人任务违反红线
  + 自己任务 OOM/极慢（白干活+协作风险）。
- **建议测试形态**：契约测试断言启动前查询 `nvidia-smi --query-gpu=...
  memory.used,utilization.gpu` 并在占用超阈值时给出明确提示或排队；
  （是否做成硬阻断属需求决策 D14，先测软提示。）
- **R29 实施与验收对照**（commit `c4cc660`，架构审查"修改后通过"，5 项必改全落实）：
  - **落点**：`test_connection`——TB-06 登记的链路位置即此查询所在；
    D14 明文"Test Connection 扩展为 Check & Show Load"；控制器 run()
    为无人值守后台流，提示无人看见。"启动前查询"按"用户提交启动前的
    检查时刻"落实（Test Connection 即该时刻）；启动时查询/硬阻断/排队
    仍属 D14，未在本轮实现也未声称已实现（测试名与文档如实标注
    "at server test time"）。
  - **查询扩展**：nvidia-smi 加 `memory.used,utilization.gpu` 两列；
    CSV 解析提取为纯函数 `parse_gpu_lines`（6 列全 split；`[Not
    Supported]` 与非数值→0；短行跳过——MIG 物理卡 memory 字段常见
    Not Supported，必改项 2）。
  - **软提示**：纯函数 `gpu_busy_warning(gpus, gpu_device)`——显存占用
    ≥80% 或利用率 ≥50%（模块常量，非配置项，D14 可再配置化）；选中
    具体卡时只评估该卡；**选中卡匹配不到任何行（MIG UUID 只以物理卡
    出现）时保守回退评估全部卡**（必改项 3）；**total≤0 不做除法**
    （必改项 1，MIG 服务器上否则崩溃）；**多卡忙时逐卡点名双指标**
    （必改项 4）；与 root 警告合并进既有单一 `warning` 字段（UI 的 ⚠
    渲染与 B12 回归测试零改动）。
  - **UI**：GPU 下拉项与连接摘要追加 `(N% used)`，用户能看到提示所指
    的负载。
  - **测试**（test_remote_training.py +5）：契约测试（查询含
    memory.used,utilization.gpu + test_connection 调 gpu_busy_warning +
    warning 字段连续性）、parse 纯函数（Not Supported/短行/空）、
    gpu_busy_warning（空闲→""、选中空闲卡不因他卡忙告警、选中忙卡点名
    显存指标、auto 多卡逐卡点名双指标）、MIG 无匹配回退、零总显存除零
    守卫。96/96 通过。
  - **证据**：`tools/test_remote_training.py` 96/96 OK（47.7s）；fast 门禁
    26/26 绿（`20260926T203525_fast.json`）。

### TB-07 导入进行中同时触发导出（同一数据双向操作）【已解决，R30，commit `d670709`】
- **场景**：后台导入某 case 转换中，用户对同一批数据发起导出/训练取数。
- **链路位置**：`runtime_py35/mimics_import.py` 队列锁 vs
  `mimics_export.py` 导出锁——两者锁命名空间不同。
- **现有防护**：锁测试丰富（producer lease、destination lock、GPU 锁、
  stop 互不误伤），但**没有**一条测试是"import 进行中对同一 case 目录
  触发 export"。`test_mask_export_stop_does_not_target_background_import`
  只测 stop 语义不重叠，不测数据竞争。
- **发生概率/损失**：概率中；损失=读到半转换的 staging 目录或文件被
  双方同时移动（白干活或个别文件损坏）。
- **建议测试形态**：fake flow 里启动 import 监视器，转换未完成时调
  export 入口，断言 export 要么等待要么基于已发布 manifest 跳过该 case，
  且不触碰 `.publishing_*` staging 目录。
- **R30 调查结论（实施前核实）**：两流数据面几乎不相交——
  (1) import 的转换产物全部落在**本地隔离 run 目录**
  （`_rt()` → import_queue_runtime_dir，mimics_import.py:128-138 明文
  "No queue/status JSON is written to the selected output share...
  The output folder contains only final .mcs files"），output share 只出现
  最终 .mcs；(2) export 的 discover_ts_cases（mimics_export.py:3127）排除
  export_root/output_dir/label_staging_dir/label_output_root，不扫 import
  的任何目录；(3) `.publishing_*` staging 只存在于训练侧
  （nnunet_pipeline.py:498/789/1261、flexict_pipeline.py:452——case cache
  物化/数据集 staging/模型发布），import/export 流自身均无此命名。
  真实交集仅一处：**import 的 case X 尚未产出 .mcs 时，export 对该 case
  走"缺 .mcs"失败/跳过分支**（mcs_paths 映射缺失 → 失败原因记录）。
- **R30 实施（纯测试轮，产品 0）**：`tools/test_all.py` 新增
  `test_export_during_ongoing_import_skips_unpublished_case`——fake flow
  中注册活跃 import monitor（busy 转换中，work_dir 含半转换 slice），
  同批数据驱动 `run_background_batch_export`：
  1. 未发布 case（无 .mcs）被跳过且带明确原因
     （`_failed_exports` 记录 ".mcs file not found"），不崩溃不读半文件；
  2. 已发布 case（.mcs 存在）正常被 export 打开继续——两分支同测；
  3. import 的隔离 work_dir 内容逐文件逐字节不变；
  4. 全树无 `.publishing_*` staging 被创建或触碰（该命名属训练 prep，
     import/export 流自身不产生——验收该条款按此落实）；
  5. 活跃 import monitor 在并发 export 后原样存活（export 不动 import
     的内存态）。
- **实施与验收对照**："fake flow 里启动 import 监视器"→ 注册
  `_IMPORT_MONITORS` busy monitor（busy=True + 未来 deadline + work_dir
  半成品，同型于 R28 监视器驱动测试）；"转换未完成时调 export 入口"→
  case_pending 无 .mcs 时直调 `run_background_batch_export`；"export 要么
  等待要么基于已发布 manifest 跳过该 case"→ 产品语义是"未发布即跳过
  并记录原因、已发布即正常处理"，两分支分别断言（skip 原因 +
  open_project 恰一次且为已发布 case）；"不触碰 .publishing_* staging
  目录"→ 全树 rglob 断言零 `.publishing_*` 存在。
- **证据**：test_all.py 单测 1/1 + TestNewFeatures 类 120/120；
  full 门禁 28/28（含 test_all 457/457），工件
  `.mimics_runtime/regression/20260926T213230_full.json`。

### TB-08 单 slice 病例 / 极端几何（1×N×N、异常 spacing）过链路【已解决，R31，commit `7dca7f7`】
- **场景**：单 slice CT（薄层或损坏导出只剩一张）、spacing 为 0 或负、
  shape 含 1 的维度。
- **链路位置**：`_unit_axis`（零 spacing 有 raise）、nnU-Net 预处理、
  nnInteractive prompt 平面逻辑。
- **现有防护**：`test_unit_axis_zero_spacing_raises` 存在；`test_2d_patch_
  has_two_dimensions`、trailing singleton channel 有一条；但"单 slice
  volume 走完 import→convert→训练 discover"无测试。nnU-Net 对 z=1 的
  3d_fullres 预处理会失败或产出退化 patch，错误是否可读未验证。
- **发生概率/损失**：概率低-中；损失=晦涩深层报错（白干活，用户难以
  自行定位）。
- **建议测试形态**：shape=(1,20,20) 的 case 过 discover+convert+训练
  请求规范化，断言要么被显式拒绝并带人话报错，要么完整走通。
- **R31 调查结论**：三条链路逐一核实——①导入侧（create_mcs_batch.py
  874-916）已有防护：manifest 期望 volume 而候选全单 slice 时先按宽松
  分组重试，仍失败则人话报错拒存坏 .mcs；②训练侧缺口：`prepare_source_
  grid_cases` 无 shape 检查，`build_training_data_profile`（两家族本地+
  远程 prepared 四条路径共用的 choke point，TB-04 空 label 门同址）只查
  维度>0 与 spacing 正值——1×N×N 会放行进 nnU-Net 预处理（即 TB-08 的
  深层失败）；③预测侧 `validate_model_input_compatibility` 同为 >0 检查
  （1 维维度放行）——不在 TB-08 冻结验收范围内，留观察记录防重复发现；
  nnInteractive 交互 helper（_iter_2d_interaction_crops/_polyline_to_mask）
  纯 numpy 几何切片，1 维维度安全。
- **实施与验收对照**：在 `build_training_data_profile` 既有"valid 3D"检查
  之后新增单 slice fail-closed 门（任一维度 ≤1 即拒，报错点名 case + 形状
  + 两条下一步动作，人话）；架构审查**通过**（独立架构师代理裁定通过，
  两条建议均采纳：条件式去冗余 int() 与既有惯用法对齐、报错措辞改配置
  中立——FlexiCT 2d trainer 也过此门；确认远程 prepared 分支与 FlexiCT
  767 行调用点同被覆盖、`<=1` 阈值无误伤真实薄层病例、2d trainer 不需
  按配置豁免——profile 声明 spatial_dimensions=3 且分布界会被 z=1 污染）。
  测试：`test_training_data_profile_rejects_single_slice_case`（正常 6³ +
  1×20×20 混合 rows，断言点名拒单 slice case）+ `test_training_data_
  profile_rejects_invalid_spacing`（零 spacing affine 锁住既有 1319 行
  行为，此前无测试；奇异矩阵经 set_sform 存储因 nibabel 拒收奇异 affine）。
- **证据**：test_nnunet_integration 86/86（含新测试 2 个）；fast 门禁
  26/26，工件 `.mimics_runtime/regression/20260926T220314_fast.json`。
- **残余观察**（不新增待办）：预测输入侧 dim==1 仍放行
  （validate_model_input_compatibility:1411），属预测入口语义（TB-08 验收
  范围外）；若未来发现真实预测单 slice 需求再议。

## 性能基线表（每轮更新数字，不凭感觉）

技术指标：

| 指标 | 基线值 | 最近值 | 轮次 | 备注 |
|------|--------|--------|------|------|
| 冒烟门禁耗时 | 63.9 s（8/8 通过） | 63.9 s | R1 | 工件 `20260925T192245_smoke.json`；历史 2026-09-24 套件合计仅 5.9 s——测量受磁盘缓存/杀软影响显著，应记录多次均值 |
| 快速门禁耗时 | 套件合计 2073 s（≈35 min 墙钟），26/26 通过 | 同 | R1 | 工件 `20260925T195215_fast.json`；TEST_STRATEGY 标称 ~10 min 偏保守 |
| 全量门禁耗时 | 待建立（标称 ~16+ min） | - | - | 无历史工件 |
| nninteractive_bridge 冷 import | 15.8 s / 31.9 s（两次） | - | R1 | nibabel→pydicom 链占 90%+，见 B1 |
| mimics_bridge 冷 import | 7.7 s / 11.9 s（两次） | - | R1 | numpy 占 6.1 s |
| flexict_pipeline 冷 import | 2.2 s | - | R1 | 健康 |
| 批次面板单次采集（collect_batch_rows） | 1686 ms | - | R1 | 147 队列目录+136 进程记录下；每 2 s 一次，见 B2/B18 |
| 运行时状态规模 | locks 345 guard / import_queues 147 目录 / processes 136 记录 | - | R1 | 见 B2 |
| 单例推理耗时 | 待建立 | - | - | 需 GPU，标注待建立方法 |
| 批量导入耗时 | 待建立 | - | - | 需真实数据 |
| 内存峰值 | 待建立 | - | - | 需确定测量方式 |

产品指标（标注者实际体验，产品价值审计的量化依据）：

| 指标 | 基线值 | 最近值 | 轮次 | 备注 |
|------|--------|--------|------|------|
| 新标注者标出第一个病例的操作步数 | 待建立 | - | - | 从打开 Mimics 到 mask 可保存；R1 产品审计估环境装好后 5-10 min，但官方模型获取是断点（A3） |
| 新标注者标出第一个病例的前置配置数 | 待建立 | - | - | 需手工改文件/路径/理解的项数，目标趋近 0 |
| 核心工作流步数：导入单病例→打开 | 待建立 | - | - | |
| 核心工作流步数：选 mask→导出 | 待建立 | - | - | |
| 权重分发：拿到模型包→可用于推理的步数 | 待建立 | - | - | 目标"一键导入即用"；自训模型已闭环，官方模型/FlexiCT backbone 是断点（A3/D15） |
| 拿到新数据集→首次导入成功需了解的概念数 | 待建立 | - | - | |

## 外部资源使用登记（服务器 / 真实数据集）

> 每次 touch 服务器或 Totalsegmentator 数据集都记一行：做了什么、用了哪些
> 路径、是否守住红线（服务器只在 /userdata/shijian_ruan/、数据集只读）、
> 清理结果。红线见 CLAUDE.md"外部资源与红线"。

| 日期 | 轮次 | 资源 | 操作内容 | 路径 | 红线确认 | 清理结果 |
|------|------|------|----------|------|----------|----------|
| 2026-09-25 | R1 | 无 | 本轮纯代码审计，未连接服务器、未触数据集 | - | - | - |
| 2026-09-29 | R61 | 无 | 四视角评审，未连接服务器、未触数据集。**发现**：外来文件 .mimics_runtime/copy_to_z.py 向 Z:\ImageAnalysisData\1-CT\Segmentation\data\ 下新建 label_task_mr_synced 目录写入（v201 兄弟目录，非 v201 内部）——非本轮所为、未触碰该文件，红线边缘操作已提请用户裁决（拍板"收编"，写入路径将单独审视） | .mimics_runtime/copy_to_z.py（只读检查） | 本轮只读未写 | 未动任何外来文件 |

## 轮次报告存档

轮次报告已独立归档至 `docs/03_Review/round_reports.md`（R1 起迁移，避免本文件
过度膨胀）。
