# 轮次报告存档

> 每轮迭代结束追加一节。不删历史。

## R1（2026-09-25）：收尾远程训练链路未提交改动

**主题**：工作区遗留的 6 文件 remote 改动（remote/Dockerfile、remote_compute.py、
remote_training_controller.py、remote_acceptance_checklist.py、test_remote_training.py、
flexict_pipeline.py）经审计验证正确但未提交；阻塞点是 A8（验收超时不停止容器 +
verify_weights 假 PASS）与 C7 小项。本轮修复后提交。

### 1. 主题与改动摘要

- **remote_compute.py / remote/Dockerfile**：`container_namespace` 白名单校验
  （`^[a-zA-Z0-9_.-]+$`），`container_runtime_command()` 产出 `nerdctl -n <ns>`；
  Dockerfile `ARG BASE_IMAGE` 支持镜像源构建。
- **remote_training_controller.py**：代码指纹两侧统一 POSIX 相对路径排序
  （Windows `\` 0x5C 与 Linux `/` 0x2F 排序不同，`foo0.py` vs `food/x.py` 顺序翻转
  导致 strict 模式必拒）；指纹脚本从 json.dumps 包裹改为 shlex.quote 直传（原写法
  容器内求值为字符串字面量、stdout 为空——strict 永拒 / warn 误报，非静默通过）；
  `identity.setdefault` 消除 `remote_code_verify != "off"` 时 run() 的 TypeError；
  `_remove_remote_job` 补删 `jobs/<id>.tar` 与 `.tar.part`；30 天缓存清理补
  `*.part`/`*.tmp`；`_sync_remote_nnunet_curve` 补 `..` 拒绝。
- **remote_acceptance_checklist.py**（A8/B16/C7）：超时/异常/Ctrl+C 统一
  `_request_stop()` 写 control.json stop 并等终态 180s；verify_weights 与
  local_inference 改用 `_resolve_local_model_dir()`（本地注册回写路径），空
  checkpoint glob 与未注册路径一律 FAIL；preflight 删除 nerdctl 静默回退、直接
  FAIL 提示改 profile；删除 --skip-reattach 死参数、docstring 第 7 步、假 staging
  目录。
- **flexict_pipeline.py**：`label_source == "prepared"` 分支消费 controller 物化的
  /job/input、/job/labels 用例（prepare_source_grid_cases 在容器内会按别名找标签
  文件而必然找不到）。

### 2. 关闭的痛点（对照验收标准）

- **A8（P1）**：①超时/异常/KeyboardInterrupt 写 control.json stop 并等停止确认
  ✅（`_request_stop`，180s 上限）②verify_weights 用 `status["model"]["model_dir"]`，
  不存在路径 FAIL ✅（`_resolve_local_model_dir`）③单测覆盖超时清理与 verify 失败
  路径 ✅（AcceptanceChecklistTests 4 用例）④远程套件绿 ✅（81/81）
- **B11(a)(b)（P2）**：30 天清理补 `*.part`/`*.tmp` ✅；取消路径删 `jobs/<id>.tar*` ✅；
  单测 ✅（RemoteJobCleanupTests）。(c) 失败诊断目录管理入口遗留待办。
- **B16（P2）**：preflight 直接 fail 提示改 profile ✅。
- **C7（P3）**：5 项中 4 项完成；`_append_log` 轮转豁免（per-job 日志随 job 目录
  清理有界，加轮转属防御式编程，架构师审查认可）。

### 3. 门禁结果

- 基线：`20260925T195215_fast.json`（26/26 pass，改动前）。
- 本轮：`20260925T220754_fast.json`（26/26 pass，含全部新增测试）。test_remote_training
  81/81、test_flexict_integration 72/72 亦单独确认绿。

### 4. 铁律自检

- 铁律 1（不阻塞 GUI）：本轮改动全部在远程链路/验收工具/测试，无 GUI 线程接触 ✅
- 铁律 2（标注者视角）：A8 修复让远程训练验收可信，支撑 P5 交付；无直接 GUI 变化 ✅
- 铁律 3（不做表面修改）：每个改动对应账本条目与根因 ✅
- 铁律 4（不留垃圾）：删除 --skip-reattach、假 staging 目录、_find_local_model_manifest
  死函数；C 项死代码清单另行排期 ✅
- 铁律 5（小步提交）：单 commit 4e23062 覆盖一个主题（远程链路收尾）✅
- 铁律 6（无验证不算完成）：11 个新失败路径测试 + 两套件绿 + fast 门禁 ✅
- 铁律 7/复杂度预算：主体为修 bug 与收敛既有改动；新增仅测试与 1 个配置项；
  `_append_log` 轮转按预算豁免 ✅
- 铁律 8（task_lifecycle 规范）：取消走 control.json 既有通道，未另搞一套 ✅

### 5. 产品价值净变化

- 远程训练是五点计划交付核心，但验收工具此前"train PASS / verify 假 PASS / infer
  必 FAIL"——验收结论不可信等于交付没有证据。本轮后验收链闭环可信。
- 服务器红线（任务结束即清理）：超时容器不再无人看管、取消不再遗留 tar/.part。
- 对标注者无直接感知变化（合理：本轮属交付可信度基建，标注者间接受益于远程训练
  可用）。

### 6. 复杂度对价

净 +566 行（产品 +321 / 测试 +355 / 删 -106）。新增维护面：配置项 1
（container_namespace）、测试 11 个、UI 入口 0、依赖 0。已记入
complexity_ledger.md。

### 7. 架构审查结论

修改后通过。实施与审查通过的方案一致：flexict_pipeline.py 纳入方案（补
prepared_cases 失败路径测试）、step_local_inference 同步换本地路径、指纹脚本
base 经 `MIMICS_CODE_FINGERPRINT_BASE` 环境变量参数化（容器行为不变）、提交显式
逐文件 git add（未碰 untracked 的 CLAUDE.md/docs/）、门禁跑 fast。无偏离。

### 8. 遗留风险与下一轮计划

- B11(c) 失败诊断目录管理入口（并入 B12 远程配置 UI 主题或单独小轮）。
- 服务器真机验收（remote_acceptance_checklist 实跑）需服务器资源，属外部资源
  操作，待排期执行并登记。
- 下一轮候选主题（按账本优先级）：A1（collect_diagnostics 阻塞 GUI，P0）/ A2
  （测试污染生产状态）/ A9（nnU-Net 无 retention）。

## R2（2026-09-25）：A1 — Collect Diagnostics GUI 冻结（P0）

**主题**：全库最后一个在 Mimics 主线程同步等子进程的入口。审计后账本新鲜（R1
当日 8 视角盲审），按 Step 0 "不重复发现" 直接进入实施。

### 1. 主题与改动摘要

`runtime_py35/collect_diagnostics_mimics.py` 重写 `collect_bundle()`：daemon 线程
跑外部收集器 + 非阻塞 timer 轮询结果（Win32 SetTimer → PyQt5 QTimer → daemon
线程三级回退，镜像 `mimics_stop_background.clear_cache_main` 模式）。立即返回，
先弹"后台收集中"提示；完成时弹结果。重复点击合并进在跑任务，不再堆叠 zip。

### 2. 关闭的痛点（对照验收标准）

- **A1（P0）**：① collect_bundle 源码无 subprocess.run，子进程调用仅在 worker
  `_collect_in_background` 内 ✅；② 2 个防回归测试（source-walk + fake-mimics
  端到端）进 test_all.py TestLifecycleAndRetention ✅；③ smoke 8/8 绿 ✅；
  ④ 属 P0 核心流程，加跑 full（工件 `20260925T235429_full.json`，见下）✅。

### 3. 门禁结果

- smoke：`20260925T221806_smoke.json`（8/8，含 flow_imports 对新模块的 py3.5
  可导入检查）。
- full：`20260925T235429_full.json`（27/28）——test_all（含 2 个新测试）及其余
  26 套件全绿；唯 training_convergence 超时（3600s）。**R3 已查明根因**：非本轮
  改动引入（A1 是 GUI 收集器改动，不触及 nnU-Net 训练），而是
  preprocess worker 在本机虚拟内存紧张时 OpenBLAS 崩溃→nnU-Net Pool 静默
  respawn→死锁（证据 `.mimics_runtime/convergence_openblas_stuck_job.log`）。
  该套件系 a0a00a8 新增、此前从未被门禁跑过，属新发现问题，已入账 A10（P1），
  A1 的关闭证据以 smoke + full 其余 27 套件 + 2 个新测试为准。

### 4. 铁律自检

- 铁律 1：正是本轮主题——300s GUI 冻结消除 ✅
- 铁律 5/6：单主题单 commit（c651d9a）+ 测试证据 ✅
- 铁律 7/9：复用既有模式，无新抽象、无新依赖；timer helper 模块内自包含（约
  60 行与 mimics_stop_background 同构，登记入 C3 重复收敛项后续统一，本轮不为
  收敛而跨 runtime_py35 模块引依赖）✅
- 铁律 2/3/4/8：无 GUI 交互变化（提示仍走 message_box ui_blocking=False）；
  痛点-根因-验证齐全；旧同步实现整体删除；取消/提示符合既有规范 ✅

### 5. 产品价值净变化

- "出问题找支持"场景（用户最焦躁时刻）：从点击后 GUI 冻结至 300 秒 → 点击即
  返回可继续工作，完成时通知。产品指标"诊断收集时 GUI 冻结时长"：300s → 0s。
- 对开发者无额外收益（合理：修复对象是标注者体验）。

### 6. 复杂度对价

净 +280 行（产品 +254 / 测试 +66 / 删 -40）。新增维护面：测试 2 个；新文件/
配置项/依赖/UI 入口 0。记入 complexity_ledger。

### 7. 架构审查结论

R2 主题为单文件 bug 修复且完全套用库内既有模式（clear_cache_main），属
"小改动豁免"范围：无新接口、无进程模型变更、无新耦合；豁免判断依据——改动
面 1 个 runtime_py35 模块 + 测试，影响面清点完整（唯一入口脚本 07_Collect_
Diagnostics.py 无需改动，返回值契约不变）。timer helper 重复实现已在 C3 登记。

### 8. 遗留风险与下一轮计划

- full 门禁工件回填后 A1 正式关闭。
- 下一轮候选：A2（测试污染生产状态，P1）/ A9（nnU-Net 无 retention，P1）/ A3
  （nnInteractive 官方模型获取断点，P1）。

## R3（2026-09-26）：A2 — 回归测试污染生产状态（P1）

**主题**：两个回归测试每次运行都写生产 `.mimics_runtime/`（一个堆垃圾目录、
一个覆盖标注者撤销状态文件），并把累积多年的测试垃圾一次性清干净。顺带查明
R2 遗留的 full 门禁 convergence 超时根因（新立 A10）。

### 1. 主题与改动摘要

- `tools/test_all.py`（batch CLI runner 测试）：调用前 set
  `MIMICS_IMPORT_RUNTIME_DIR` 指向临时目录（`import_runtime_base` 自带的
  override，调用时读取、忽略 project_root，CONFIG_REFERENCE 已文档化），
  finally 恢复；期望队列目录在 override 仍生效时经真实函数解析（digest 规则
  不会漂移）。新增 `_snapshot_dir` 防复活断言：生产 `import_queues/` 前后
  逐项一致。
- `tools/fake_mimics_flow_test.py`（entrypoint 测试）：调用真实
  `05_Window_Reset_Full_Range.py` 前 patch 缓存模块 `_state_path` 指向临时
  文件（run_runtime_entry 的 importlib.import_module 返回同一单例，patch 生效），
  finally 恢复；断言临时 state 已写、生产 `window_level_state.json` 字节级未动。
  兄弟测试（window_level_from_selected_mask）补 finally 恢复——原实现不恢复，
  单例模块全程遗留指向已删临时目录的 patch（架构师建议项）。
- **一次性清理**（删前逐一核查留档）：`.mimics_runtime/import_queues/` 153 个
  目录 = 74 空 + 77 引用 `Temp\mimics_test_*` + 2 引用 `Temp\cli_dbg_*`，
  无一真实生产队列；`fewshot_mimics_state.json`（内容全为 fake-mimics 临时
  数据集路径，若不删会被 `_migrate_old_settings` 迁入用户正式设置，即 B10
  定时炸弹）；`window_level_state.json`（内容为 fake 场景 0-100 对比度+mtime
  匹配上次 smoke，即测试数据，保留=给下一位标注者留假撤销历史）。
- **A10 根因查明**（R2 full 门禁 27/28 的收敛套件超时）：不是 A1/R2 改动
  引入。`stage_preprocess` → nnunetv2 fingerprint_extractor spawn.Pool 4 个
  子进程，每个 import 时加载 OpenBLAS 且按 20 核各起 ~20 线程；本机虚拟内存
  仅余 ~5.6GB 时子进程崩溃（job.log 11 次 "OpenBLAS error: Memory allocation
  still failed after 10 retries"），nnU-Net Pool 静默 respawn 僵尸 worker，
  主 worker 轮询永不退出——实测挂起 60+ 分钟 0 CPU，超时杀掉后还留孤儿
  stage_worker（实测存活 123 分钟）。证据
  `.mimics_runtime/convergence_openblas_stuck_job.log`。已立 A10（P1），
  修复方向：`_worker_environment` 对全部 stage setdefault
  OMP/MKL/OPENBLAS_NUM_THREADS=1（库内先例 mimics_batch_cli.py:105-107）。
  本轮已把两棵挂死进程树（孤儿 10688 含 3 个 respawn 子进程 + 独立复现
  51504→12956）taskkill 清理。

### 2. 关闭的痛点（对照验收标准）

- **A2（P1）**：① 两测试临时目录/state-path 隔离 ✅（env override + module
  patch 两种库内既有模式）；② 生产 153 目录 + fewshot + window_level 清理 ✅
  （删前逐一核查，判定依据留档）；③ 防复活断言 ✅（`_snapshot_dir` 目录快照 +
  文件 bytes 快照，"缺席语义"——清理后生产文件不存在，测试若重建即失败）；
  ④ smoke 8/8 + fast 26/26 ✅。
- 范围裁定（架构审查修正 3）："locks/ 345+ guard"部分归 B2 承接，A2 不含。

### 3. 门禁结果

- smoke：`20260926T005016_smoke.json`（8/8，改动后首跑，含 entrypoint/window
  两项被改测试）。
- fast：`20260926T015355_fast.json`（26/26）。首次后台跑被会话重连杀死（flexict
  集成中途消失，无工件），重跑全绿。

### 4. 铁律自检

- 铁律 1（不阻塞 GUI）：改动全在测试与一次性清理，无 GUI 线程接触 ✅
- 铁律 2（标注者视角）：本轮直接保护标注者状态文件（撤销历史不再被测试覆盖）
  ✅
- 铁律 3（不做表面修改）：污染点、根因、核查、断言齐全 ✅
- 铁律 4（不留垃圾）：153 目录 + 2 状态文件清除，删前逐一核查 ✅
- 铁律 5（小步提交）：单 commit 6e85b65 单主题 ✅
- 铁律 6（无验证不算完成）：smoke + fast 绿 + 防复活断言进回归套件 ✅
- 铁律 7/复杂度预算：产品代码零改动（纯测试隔离 + 删除）；选型走"改配置
  （env override）> 新增" ✅
- 铁律 8（task_lifecycle 规范）：无新功能交互 ✅

### 5. 产品价值净变化

- 标注者的窗宽窗位撤销历史不再被每次回归测试静默覆盖（此前每次 smoke 后
  `window_level_state.json` 变成 fake 0-100 值，真按 Undo 会把窗宽窗位"恢复"
  成测试值）。
- 开发者：`.mimics_runtime/import_queues/` 从 153 个垃圾目录归零，健康面板
  扫描不再翻测试垃圾。
- 防复活断言使该痛点永久性关闭（宪法"修复沉淀为测试资产"）。

### 6. 复杂度对价

净 +52 行（全部为测试代码：test_all +33/-2、flow test +38/-17，纯删除无产品
代码变化——commit 6e85b65 净 +71/-19，另删生产垃圾 153 目录+2 文件不入库）。
新增维护面：测试 helper 1 个（`_snapshot_dir`，17 行）；新文件/配置项/依赖/
UI 入口 0。选型顺序：env override（改配置）而非 monkey-patch，最优先级。

### 7. 架构审查结论

通过（附 3 项必须落实修正 + 2 项建议，全部落实）：① env var 在 finally 恢复
且断言在恢复后执行（已按此实现）；② 防复活断言用"存在性+内容"二元组而非裸
bytes（已按此实现）；③ locks 范围裁定落账（A2 条目 + B2 交叉引用）；建议项
（:902 finally 恢复）亦顺手落实。审查确认：env override 方案优于 monkey-patch
备选（`import_runtime_base` 是库自带隔离钩子，无缓存、锁路径走另一 env var
不受影响、supervisor 存在性检查要求 ROOT 保持真实）；生产三个清理目标逐一
实锤（无一个真实生产队列）。

### 8. 遗留风险与下一轮计划

- **A10（P1，新立）**：nnU-Net 预处理 worker OpenBLAS 崩溃→静默死锁。不仅卡
  测试——标注者本机真跑 nnU-Net 训练，同样死锁=「训练永远停在 30% 无报错」。
  修复小（env 三个 setdefault + 防回归断言），下轮首选。
- **工作区发现用户未提交改动**：`tools/nnunet_pipeline.py`（TMP/TEMP 重定向
  到 job 内 `worker_tmp/`，防 IT 定期清理 %TEMP% 杀掉长任务输出，改动合理且
  经 tonight 两次训练运行验证在跑）。mtime 2026-09-25 23:15，非本会话所改；
  未提交。已在 R1 确立的工作区卫生原则下留待用户处置或并入 A10 轮提交——
  与 A10 同文件同主题（worker 进程稳定性），建议 A10 轮一并审查后提交。
- A1 的 full 工件已回填（27/28，convergence 超时归 A10，已注明）。
- 下一轮候选：A10（首选，修复小收益大）/ A3 / A4 / A9。

## R4（2026-09-26）：A10 — 预处理 worker BLAS 线程封顶（P1）

**主题**：nnU-Net/FlexiCT 预处理 spawn.Pool worker 在虚拟内存紧张机上 OpenBLAS
崩溃→上游 Pool 静默 respawn→训练永久挂起（full 门禁 convergence 3600s 超时的
根因；对标注者即"训练永远停在 30% 无报错"）。

### 1. 主题与改动摘要

- `tools/nnunet_pipeline.py`：`_worker_environment` 对全部 stage、全平台 setdefault
  `OMP/MKL/OPENBLAS_NUM_THREADS=1`（与 mimics_batch_cli 先例及 nnU-Net 官方
  run_training.py 入口同型）。
- `tools/flexict_pipeline.py`（架构审查发现的阻断性缺口）：`flexict_worker_environment`
  （preprocess/train）与 `flexict_infer_environment`（infer）在三处（:596/:983/:1197）
  整体 monkeypatch 替换 `_worker_environment`，FlexiCT 预处理走同一条 spawn.Pool
  指纹提取链——三处全部镜像封顶。另发现审查未列的 `flexict_infer_environment`
  第三替换点（实施时自查补上）。
- 防回归测试 3 个：test_nnunet_integration（三 stage 断言）+ test_flexict_integration
  （worker env 三 stage + infer env）。

### 2. 关闭的痛点（对照验收标准）

- **A10（P1）**：① 全 stage 全平台三变量封顶 ✅（nnunet + flexict 三处替换点全覆盖）
  ② 本机复跑 convergence 套件独立通过（2/2 OK，1261s，无挂起）✅ ③ 3 个防回归
  测试进回归套件（套件 67/67、74/74 绿）✅ ④ full 门禁 28/28 全绿（工件
  `20260926T035220_full.json`，training_convergence 1231s 通过）✅

### 3. 门禁结果

- 独立 convergence：2/2 OK（1261s）。
- full：`20260926T035220_full.json`（28/28，含 convergence 1231s——R2 时该套件
  3600s 超时死锁）。

### 4. 铁律自检

- 铁律 1（不阻塞 GUI）：改动仅在 worker 环境构造（子进程 env），无 GUI 接触 ✅
- 铁律 2（标注者视角）：修复"训练挂起无报错"——标注者本机训练稳定性直接受益 ✅
- 铁律 3（不做表面修改）：痛点-根因（证据日志）-修复-验证链完整 ✅
- 铁律 4（不留垃圾）：无新增垃圾；挂死进程已于 R3 清理 ✅
- 铁律 5（小步提交）：单 commit b33a399 单主题 ✅
- 铁律 6（无验证不算完成）：独立 convergence + full 28/28 + 3 个新测试 ✅
- 铁律 7/9（复杂度/最小改动）：净 +74 行（产品 24 / 测试 50），无新文件/配置项/
  依赖/UI 入口；选型为先例同型复制（修 bug）✅
- 铁律 8（task_lifecycle）：无新交互 ✅

### 5. 产品价值净变化

- 标注者在内存紧张的工作站上跑 nnU-Net/FlexiCT 训练不再可能静默挂起（此前表现
  为"训练永远停在 30%，无任何报错，只能杀进程"）。
- full 门禁从 27/28（convergence 必超时）恢复 28/28——该套件从此可作为有效回归
  证据（R2 起 A1 的关闭证据不再有缺口）。
- 封顶随 spec 流入远程容器，共享服务器上亦符合"不得挤占"红线。

### 6. 复杂度对价

净 +74 行（产品 +24 / 测试 +50）。新增维护面：测试 3 个；新文件/配置项/依赖/
UI 入口 0。选型顺序：修 bug（先例同型复制），无新增抽象。

### 7. 架构审查结论

修改后通过，全部修正落实：①（阻断性）flexict_worker_environment 同步封顶
（已实施，并自查补上审查未列的 flexict_infer_environment 第三替换点）；② 测试
放 test_nnunet_integration + flexict 对称断言（已实施）；③ 前提更正：TMP/TEMP
改动已由用户提交（de0a07b），无"一并提交"问题（核实属实）；④ 验收按账本执行
（独立 convergence + full，已实施）。建议项（注释更新）已落实。远程容器 train
行为等价（Action3 进程内已 setdefault 同三变量），无需重跑远程验收。

### 8. 遗留风险与下一轮计划

- 残余风险：上游 nnU-Net Pool 静默 respawn 不可控，封顶是"不让 worker 崩"而非
  根治 respawn；其他崩溃源（显存耗尽等）仍可能触发同型挂起（已登记于 A10 条目）。
- 本轮发现工作区新增 commit 50925a0（五点计划交付报告，docs/changes/，非本迭代
  会话所做，作者为用户侧 Local Snapshot）——与迭代无冲突。
- 下一轮候选：A3（nnInteractive 官方模型获取断点，P1）/ A4（Relink 死胡同消息，
  P1）/ A9（nnU-Net 无 retention，P1）/ B2（运行时状态无界增长，P2）。

## R5（2026-09-26）：A9 — nnU-Net jobs retention（P1）

**主题**：nnU-Net 是唯一没有 job retention 的管线家族（flexict 与 nnInteractive 均有
`_sweep_expired_jobs`）；每次训练/推理的 job 产物永久累积，磁盘满→导入/导出失败。

### 1. 主题与改动摘要

- `tools/nnunet_pipeline.py`：`_sweep_expired_jobs` 完整镜像 flexict 版（终态 job 过
  保留天数清内容留 status.json，TERMINAL_STATES 取自 nnunet_common 含 abandoned）；
  调用点在 run_training 与 run_inference 两处 finally（架构审查修正：原拟"入口调用"
  是新发明时机，对齐 flexict 的 job 终态后清扫模式，且不漏 inference job）。
- 新建 `nnunet_config.json`（单键 `job_retention_days` 默认 30，0 禁用）+ 最小
  `load_nnunet_config`（镜像 flexict_common.load_config 的 merge 模式）——nnU-Net
  此前没有任何 config 文件（架构审查纠正了方案原文"load_config 既有键"的错误前提）。
- `CONFIG_REFERENCE.md` 新增 nnunet_config.json 小节（格式照 flexict 节）。
- 测试 6 个（test_nnunet_integration.py TestSweepExpiredJobs）：过期终态清内容留
  status / 新近终态保留 / 在跑 job 不删 / reattaching（远程）不删 / retention=0
  禁用 / **models 与 cache 显式不受影响断言**（架构审查要求：这是最大风险点，
  不能只靠代码审读）。

### 2. 关闭的痛点（对照验收标准）

- **A9（P1）**：① `_sweep_expired_jobs` 对齐 flexict + 保留天数进 config（新建
  nnunet_config.json）+ CONFIG_REFERENCE ✅ ② 伪造 mtime prune 测试 6 个（超验收
  要求的最小集，含 models/cache 不受影响与 reattaching 远程态）✅ ③ fast 26/26
  （工件 `20260926T042044_fast.json`）✅
- **范围裁定（架构审查修正 5）**：A9 验收标准只覆盖 jobs/ 清扫；痛点原文的主要磁盘
  增长源——runtime 三目录（raw/preprocessed/results，按 Dataset 跨 job 复用）与
  cache/source_grid（可重建缓存）——不在本轮范围，残留登记为新条目 A11（P1）。
  本轮**不**宣称"nnU-Net 磁盘无界增长已解决"。

### 3. 门禁结果

- fast：`20260926T042044_fast.json`（26/26）。test_nnunet_integration 73/73（含
  6 个新 sweep 测试）单独确认绿。

### 4. 铁律自检

- 铁律 1（不阻塞 GUI）：改动在 CLI 管线控制器（finally 内文件清扫），无 GUI 线程
  接触；清扫为同步文件删除但发生在 job 终态后的控制器进程（非 Mimics 进程）✅
- 铁律 2（标注者视角）：磁盘满→导入/导出失败是标注者直接受损路径，本轮消除其
  增长源之一 ✅
- 铁律 3（不做表面修改）：痛点-根因（retention 收敛漏掉 nnU-Net）-修复-验证齐 ✅
- 铁律 4（不留垃圾）：models/cache 显式不受影响断言防误删用户资产 ✅
- 铁律 5（小步提交）：单 commit fa0bcf6 单主题 ✅
- 铁律 6（无验证不算完成）：6 个测试 + fast 26/26 ✅
- 铁律 7/9（复杂度/最小改动）：净 +179 行（产品 ~90 / 测试 ~82 / 文档配置 ~7）；
  新增维护面：配置文件 1（单键）、loader 1 个（同型复制）、测试 6 个——配置键为
  验收标准明文要求，不可更小。选型：镜像既有模式（新增代码为验收要求的最小集）✅
- 铁律 8（task_lifecycle）：无新交互 ✅

### 5. 产品价值净变化

- nnU-Net job 产物不再永久累积（默认 30 天后清到只剩 status.json，历史可浏览）；
  磁盘占用从无界变为有界。标注者间接受益（磁盘满失败概率下降）；开发者工作区
  不再无限膨胀。
- 无直接 GUI 变化（合理：磁盘治理基建）。

### 6. 复杂度对价

净 +179 行（产品 +90 / 测试 +82 / 配置+文档 +7）。新增维护面：新配置文件 1
（nnunet_config.json，单键）、新 loader 1（load_nnunet_config，~20 行同型复制）、
测试 6 个；新依赖/UI 入口 0。CONFIG_REFERENCE 同步。记入 complexity_ledger。

### 7. 架构审查结论

修改后通过，6 项修正全部落实：① 调用点改两控制器 finally（原拟入口调用作废）；
② 新建 nnunet_config.json + 最小 loader（原拟"既有键"前提错误作废）；③
TERMINAL_STATES 取自 nnunet_common；④ 测试 6 用例含 models/cache 断言，全用 tmp +
显式 config；⑤ 轮次报告披露新配置面 + 登记残留条目 A11（已落实）；⑥ fast 工件
已引用。实施与审查通过的方案一致（含 removed_jobs 语义沿用 flexict 版"记名但保留
目录"行为，未顺手修正）。

### 8. 遗留风险与下一轮计划

- **A11（P1，新立）**：runtime 三目录 + cache/source_grid 无 retention——A9 痛点的
  主要磁盘源（flexict 实证 runtime 4.8G vs jobs 473K）。runtime 涉及 Dataset 在用
  判定（被注册模型引用的 Dataset 不得删）与 preprocess 复用语义，方案需先过架构
  审查。
- 僵尸 job（controller 被杀、无终态）不会被 sweep——与 flexict 相同的残留限制，
  属 B10 PID 回收防护同一治理域。
- 下一轮候选：A3（nnInteractive 官方模型获取，P1，部分需用户决策 D9）/ A4（Relink
  死胡同，P1，需人工 GUI 验证）/ A11（需先设计）/ A5/A6/A7（P1）/ B 系 P2。

## R6（2026-09-26）：A3 nnInteractive 官方模型检测 + 引导

### 1. 主题与改动摘要

A3（P1）：官方 nnInteractive 模型随分发源单独拷贝、不随部署包分发，新机器装完
环境点旗舰入口得到裸 `RuntimeError` 无任何修复路径；两个 AI 入口的失败体验不一致
（FlexiCT 权重缺失有引导窗接住）。本轮关闭"检测 + 引导 + 报错文案"，获取/分发
方式本身保持 D9 需用户决策。

改动（commit 6c5bd3d，5 文件 +176/-14）：

- `tools/env_guidance.py`：`_nninteractive_model_candidates` + `_nninteractive_official_model`
  镜像**运行时**解析链（NNINTERACTIVE_MODEL_DIR → nninteractive_config.json
  `model_dir` → environment_root models → python_env/nninteractive_env 兜底），可用性
  判定同 `_model_folds`（fold_*/checkpoint_final.pth）；缺失时追加
  `nninteractive_model` issue（severity bad、fix_action ""、detail 列出全部搜索位置
  与拷贝指引）。
- `runtime_py35/nninteractive_mimics.py`：目录缺失与 fold 缺失两种报错均追加
  "从分发源拷贝 + 见 99_Admin → Environment Guidance"（py3.5 .format）。
- `tools/system_health_panel.py`：Environment OK 文案同步。
- `docs/mimics_entry_guide.md`：官方模型入口行前置条件同步。

### 2. 关闭的痛点（对照验收标准）

- ① collect_issues 新增检测，有测试 —— TestEnvGuidance 新增 4 个测试
  （缺失报 bad+空 fix_action+detail 含位置列表 / 已装静默 / 环境变量覆盖 /
  config 键覆盖），既有空断言测试（setup_failed/incomplete/stale/migration×2）
  同步建模型目录。✅
- ② 缺失时报错文案给出可执行获取指引 —— 两种运行时报错 + 引导窗 detail 均含
  "copy from distribution source + 见 Environment Guidance"。✅
- ③ docs/mimics_entry_guide.md 同步。✅

### 3. 门禁结果

- TestEnvGuidance 12/12、test_migrate_root 9/9（单独跑）。
- smoke 8/8：`.mimics_runtime/regression/20260926T045239_smoke.json`（runtime_py35
  被改动，按规加跑）。
- fast 26/26：`.mimics_runtime/regression/20260926T045940_fast.json`。

### 4. 铁律自检

- 未阻塞 GUI（检测为纯文件读，引导窗走既有 PySide6 路径）；标注者视角优先
  （本轮就是关第一天断点）；非表面修改（根因=分发链路缺失，本轮交付检测+引导
  这一层）；无垃圾（顺带删除了实施中途留下的死 helper `_clear_model_env`）；
  小步单主题 commit；验证齐全；复杂度受控（零新文件/配置/依赖/UI 入口）；
  遵守任务生命周期规范（信息型 issue 无按钮，不造新交互）；改动最小（不碰
  分发方案本身）。✅ 全部通过。

### 5. 产品价值净变化

新标注者"第一天断点"从「裸报错+自己猜」变为「报错即给出拷贝位置清单与入口
指引，System Health 常驻可见」。两个 AI 入口（FlexiCT 权重 / nnInteractive 官方
模型）失败体验对齐。产品指标表"权重分发断点"一项的部分gap关闭（检测面），
剩余 gap 在 D9。

### 6. 复杂度对价

净 +162（产品 +80 / 测试 +82）。新增维护面：新 issue kind 1（复用既有渲染与
System Health 集成，零新 UI 入口）；检测函数 2 个（约 55 行，镜像运行时链是
验收要求，不可复用 tools 侧链——架构审查确认两者 config 文件与键不同）。
选型顺序：报错文案+检测（改文案/新增检测分支）为最小可行方案，未动分发方案。

### 7. 架构审查结论

修改后通过（7 条修正全部落实）：检测链镜像运行时而非复用 tools 侧
official_model_dir；severity bad / fix_action ""；py3.5 .format；既有空断言测试
同步；测试 pop/restore NNINTERACTIVE_MODEL_DIR 防真实环境泄漏；"distribution
source" 措辞不硬编码获取方式（D9 约束）。实施与审查通过方案一致，无偏离。

### 8. 遗留风险与下一轮计划

- 行为变化披露：无官方模型的机器上 System Health 会持续显示
  "1 environment issue(s)"——预期行为（问题真实存在），不是回归。
- 运行时链的 `model_dir` 键未在 CONFIG_REFERENCE.md 记录（架构师附注）——
  低优先级，入账待办。
- 下一轮候选（R7 Step 0 评估）：A4（Relink 死胡同，需人工 GUI 验证——升级
  候选）/ A5（表单设计 D2 或最小默认值方案）/ A6①（FlexiCT 拷贝修复可独立做）/
  A11（需先设计，验收标准要求架构审查前置）/ A7（平台限制，只能缓解）。
  停止条件未满足。

## R7（2026-09-26）：A6① FlexiCT 停止指引死胡同修复

### 1. 主题与改动摘要

A6①（P1）：FlexiCT 等待提示把用户指向死胡同。实施前勘察发现比账本记载更严重：
FlexiCT 的 stop/status 动作**本身**就无菜单入口可达（三个入口均传固定 action，
chooser 永不出现）；资源等待日志指向 nnU-Net 的 Stop 入口（工作区不相交，必得
"No running nnU-Net task was found."）；两条结果等待日志指向一个按不动的按钮
（job 已 completed，查看器 Stop 按钮对终态禁用）。

改动（commit a33e632，7 文件 +66/-10）：

- 新入口 `02_AI/FlexiCT/04_Show_Status_and_Stop.py`（23 行薄壳，action
  BUTTON_STATUS → 外部状态查看器，Stop 按钮接真实 workspace 的 stop_job）。
- `flexict_mimics.py` 三处文案：资源等待 → 指新入口；两条结果等待 → 陈述真实
  机制（结果保留、自动应用，resume 兜底）。
- 四处文档同步：entry guide、task_lifecycle §2 表（新增 FlexiCT 行）、
  flexict_mimics.md（新 §3.4 + 组件表四入口）、架构文档（四入口表）。

### 2. 关闭的痛点（对照验收标准）

- ① FlexiCT 等待/错误文案不再引用 nnUNet 入口编号 —— 负向 grep
  "04 Stop Running Task" 断言 + 三条新文案正向断言。✅
- ① 有测试 —— `test_waiting_messages_reference_reachable_stop_paths` +
  `test_shells_route_to_flexict_mimics` 增 04 入口行。✅
- ② 菜单重组仍 D3 需用户决策，本轮明确不扩范围。✅（范围守住）

### 3. 门禁结果

- TestEntryRouting 4/4（含新增测试）。
- smoke 8/8：`20260926T052121_smoke.json`（runtime_py35 被改动，按规加跑）。
- fast 26/26：`20260926T052851_fast.json`。

### 4. 铁律自检

- 未阻塞 GUI（新入口与既有三入口同模式，起外部进程即返回）；标注者视角优先
  （"AI 卡住了"时刻不再被指向死按钮/死菜单）；非表面修改（根因=停止路径不可达
  +跨家族脆引用，本轮交付可达入口与真实机制陈述）；无垃圾；小步单主题 commit
  （经历 2 次分类器中断，按 git log 验证后重试，未产生重复 commit）；验证齐全；
  复杂度受控（+56 行：23 行薄壳 + 文案 + 文档 + 测试）；遵守任务生命周期规范
  （task_lifecycle §2 表同步，等待信息含真实可用停止入口）；改动最小。✅

### 5. 产品价值净变化

标注者在 FlexiCT 等待资源时有了真实可点的停止入口（此前指向必失败的 nnU-Net
入口）；FlexiCT 家族补齐了 nnU-Net 已有的"状态+停止"入口能力（对齐而非新增
第 6 个 Stop 名义入口）；结果等待场景从"指向按不动的按钮"变为如实告知
"结果会自动应用"。A6② 的"5 个 Stop 不可区分"问题未动（D3 用户决策）。

### 6. 复杂度对价

净 +56（产品 +45 / 测试 +11）。新增维护面：新 UI 入口 1 个（薄壳，无新逻辑）、
新文件 1 个。选型顺序合规：主体是改文案；新入口是验收标准"改指自身动作菜单"
的最小实现，复用既有外部状态查看器（其 Stop 按钮已存在且接真实 workspace），
零新对话框逻辑。

### 7. 架构审查结论

修改后通过，两条硬性修正全部落实：
- 修正 1（否决点）：原方案文案 2、3 指向查看器 Stop 按钮，但该按钮对 completed
  终态禁用——等于换个方式复现同一死胡同。已按推荐方案改为陈述真实机制（零新
  代码），pending 丢弃路径可达性缺口登记 A6②/D3。
- 修正 3：文档同步四处（不止 entry guide）——已补 task_lifecycle §2 表、
  flexict_mimics.md、架构文档。
- 修正 4：测试复用既有 TestEntryRouting（shells 字典加行）而非另起炉灶。
实施与审查通过方案一致，无偏离。

### 8. 遗留风险与下一轮计划

- pending 推理结果的主动丢弃路径（stop_running_task pending 分支）仍不可达
  ——已并入 D3 用户决策（结果自动保留+应用，丢弃需求频率存疑）。
- 下一轮候选（R8 Step 0 评估）：A4（Relink 死胡同，需人工 GUI 验证——升级
  候选）/ A5（表单收敛，D2 或最小默认值方案）/ A11（需先设计+架构审查前置）/
  A7（平台限制）/ B 系 P2。停止条件未满足（仍有未解决的 P1）。

## R8（2026-09-26）：A5 nnU-Net 训练表单最小方案（默认值+说明）

### 1. 主题与改动摘要

A5（P1）：nnU-Net 训练表单泄漏 ML 概念（Dataset ID/Fold/GPU devices 等），只有
悬停 tooltip 无可见解释——标注者不悬停。按验收标准括号明示的**最小方案路径**
（默认值+说明）实施；结构收敛（必填缩减、高级项折叠）完整保留 D2 需用户决策。

改动（commit 5039fff，2 文件 +97/-18）：

- `nnunet_training_setup_ui.py`：新增 `_hint()` helper（同型复制既有 hint 模式）
  + 7 处一行说明（Dataset ID / Fold / Validation fraction / Preprocessing
  workers / GPU count / GPU devices / Epochs），四项关键 hint 存为属性。
- `test_gui_smoke.py`：新增离屏实例化测试断言 hint 存在 + 关键规则文案。

### 2. 关闭的痛点（对照验收标准）

- "至少 Dataset ID/Fold/GPU devices 三项有合理默认与一行说明"——四项均有
  （+GPU count）：默认值此前已齐（701 自动建议 / Fold 0 / 留空），本轮补齐
  可见解释层。✅
- "走查记录进账本"——已解决区条目 + 本报告即走查记录。✅
- "表单项收敛为必填少而有解释、高级项折叠且有默认值"——**未做**（结构性重设计
  留 D2），按括号"或按最小方案实施"的明示路径执行，账本 A5 条目与 D3 条目均
  明确记载此范围裁定（C4）。

### 3. 门禁结果

- gui_smoke 15/15（含新增测试，单独复跑确认）。
- fast 26/26：`20260926T054459_fast.json`。

### 4. 铁律自检

- 未阻塞 GUI（静态 QLabel 构建期创建，7 处既有先例同型）；标注者视角优先
  （每个概念一行"它是什么+什么时候改"）；非表面修改（含真实信息增量：GPU
  devices 的设备数匹配规则此前只能靠提交失败发现）；无垃圾；小步单主题
  commit；验证齐全；复杂度受控（+79 行纯文案+测试，零新依赖/配置/入口）；
  任务生命周期规范不涉及（纯表单静态说明）；改动最小。✅

### 5. 产品价值净变化

标注者打开训练表单时，7 个原需要 nnU-Net 背景的字段各有"它是什么+默认值
为什么这样+什么时候才需要改"的一行解释；尤其是 GPU devices 的校验规则
（设备数须与 GPU count 匹配）从"提交失败才发现"变为"填之前就可见"。
Planning 页已是目标模式（自动复选框+推荐提示），未动。

### 6. 复杂度对价

净 +79（产品 +62 / 测试 +17）。新增维护面：helper 1 个（7 行，同型复制非
新抽象）、hint label 7 个（静态文案）。选型顺序合规：改文案（最优先级）；
无任何结构变更。

### 7. 架构审查结论

修改后通过，5 条修正全部落实：C1（否决点）GPU devices 文案事实错误已纠正
（留空=自动选择、数量须匹配，源码核证 nnunet_pipeline.py:887-889 与
nnunet_common.py:562-564）；C2 validation fraction 注明 Fold=all 不适用；
C3 测试放 test_gui_smoke.py 既有离屏 fixture（test_nnunet_integration 无
UI 测试——原计划锚点错误）；C4 账本明确最小方案路径、不宣称表单已收敛；
C5 Fold 措辞只用已验证事实（五折 held out、all=顺序训五个，`_split_folds`
+ run_training 源码核证）。实施与审查通过方案一致，无偏离。

### 8. 遗留风险与下一轮计划

- 结构收敛（必填缩减/高级折叠）为 D2 需用户决策，未动。
- 下一轮候选（R9 Step 0 评估）：A4（Relink 死胡同，需人工 GUI 验证 Fix
  Affine——**升级候选**，按协议"需人工验证"类问题应暂停询问用户，同时继续
  不依赖它的项）/ A11（需先设计 runtime 清扫方案+架构审查前置）/ A7（平台
  限制，只能缓解文案）/ B 系 P2（B4 错误文案残留与 R7/R8 同主题，可做）。
  停止条件未满足（A4/A11/A7 仍 P1，但 A4/A7 均涉人工验证或平台限制）。

---

## R9（2026-09-26）：A11 — runtime 三目录与 cache/source_grid retention

### 1. 本轮主题与改动摘要

A11【P1】：两个训练家族（nnU-Net + FlexiCT）的 `runtime/nnUNet_{raw,preprocessed,results}`
与 `cache/source_grid` 从不清扫，磁盘无界增长（实测 flexict 工作区 runtime 6.6G，
其中仅 Dataset758 有注册模型引用；nnU-Net 工作区尚空但同一布局）。前置完整侦察
（工作区实测、清扫先例 `_sweep_expired_prepared_cache`、registry→Dataset 链路、
`_assert_dataset_id_available` 语义、inference 只读 models/ 不读 runtime 的核证），
独立架构师代理审查后实施。

### 2. 关闭的痛点（对照验收标准）

- ① cache/source_grid mtime 清扫：`nnunet_common.sweep_source_grid_cache_retention`
  （两级 task/case 布局，超龄 case 目录 rmtree，空任务目录回收；保留天数
  `source_grid_cache_retention_days` 进 config，默认 30、0 禁用）✅
- ② runtime 三目录方案经架构审查后实施：`sweep_dataset_retention` 按
  `Dataset[0-9]{3}_*` 名保护——被注册模型（nnU-Net 双注册表 / flexict
  registry.json）引用的 Dataset 与非终态 job 的 Dataset 不得删；审查前方案有
  4 项必改，全部落实（见第 7 条）✅
- ③ 伪造 mtime 测试：nnunet TestSweepRuntimeAndCache 9 条 + flexict 5 条，
  全部 tmp 目录 + 显式 config（超龄未引用删、注册模型保、在跑 job 的 Dataset
  与 cache 任务树保、新近保、0 禁用、models 不碰、registry 缺 dataset_name
  重建兜底）✅
- ④ fast 绿：26/26（工件 `20260926T065602_fast.json`）✅

### 3. 门禁结果

`--profile fast` 26/26 全绿，工件 `.mimics_runtime/regression/20260926T065602_fast.json`。
两套件全量 162/162（test_nnunet_integration + test_flexict_integration）。

### 4. 铁律自检

- 铁律 1（不阻塞 GUI）：清扫全部在 job worker 子进程 finally 块，GUI 线程零
  触碰 ✅
- 铁律 4（不留垃圾）：本轮清除的正是历史累积垃圾机制缺口；无新增死代码 ✅
- 铁律 7/复杂度预算：新增为 P1 痛点+已冻结验收标准所要求，共享清扫器放
  nnunet_common 复用（flexict 已同时 import 两模块，无新耦合面）；无新文件、
  无新依赖、无新 UI 入口 ✅
- 铁律 5/6：单一 commit 3f16d0f，fast 全绿后关闭 ✅

### 5. 产品价值净变化

标注者视角：磁盘不再被训练残留无声吃满——此前一次冒烟训练即留 6.6G，几轮
迭代后磁盘满会让导入/导出直接失败（P6 错误映射只能提示，不能恢复数据）。现在
30 天窗口自动回收无引用数据，在跑任务与已注册模型不受影响。属"防数据事故"
价值，日常无感但失败路径收益大——这正是测试理念（失败路径优先）指向的防护。

### 6. 复杂度对价

净 +643 行（产品 +337：nnunet_common 108、nnunet_pipeline 123、flexict_pipeline
120，其中近半为 docstring/注释；测试 +306；CONFIG_REFERENCE +6−2）。新增维护
面：2 个 config 键、0 新文件、0 新依赖、0 新 UI 入口、5 个挂钩点（全部贴既有
`_sweep_expired_jobs` 调用旁）。选型顺序合规：方案以"复用既有模式"为先
（清扫器结构照抄 nninteractive 先例，保护集复用 load_models/TERMINAL_STATES），
新增代码是验收标准 ①② 的最小满足。

### 7. 架构审查结论

独立架构师代理：**修改后通过**。4 项必改全部落实——① cache 清扫补非终态
job 任务树保护（防 job B 做 cache-hit 校验时被 job A 的清扫删桶）；② 预处理
复用分支重打 manifest 时间戳（防按月重训任务被周期性清扫后付数小时重预处理；
nnunet 重写 mimics_preprocess_manifest.json、flexict 重写
flexict_preprocess_manifest.json）；③ 挂钩 `except Exception`（防清扫 bug 把
完成 job 变进程异常，nninteractive 先例）；④ wrapper 健壮性（infer/AL 无
dataset_id 跳过、缺 dataset_name 重建兜底、显式 config 参数供测试）。
实施与审查通过方案一致（flexict 复用分支从"重算 identity 传入"简化为
"原样重打既有 identities"，语义相同、更简单，属实现细节非方案偏离）。

### 8. 遗留风险与下一轮计划

- 首扫预告：第一个 job 结束时可能一次性 rmtree 数 GB（本机 flexict 工作区约
  2.4G 无引用 Dataset）——worker 进程内执行，符合铁律 1，但用户可能注意到
  首次耗时。
- 非阻塞观察（审查记录，暂不动）：全局注册表跨工作区同名过度保护、僵尸非终态
  job 在 reconcile 前持续保护其 dataset——均为"多保不误删"方向。
- 剩余 P1：A4（Relink 死胡同，需人工 GUI 验证 Fix Affine——升级候选）、
  A7（Mask Identifier 平台限制，仅能缓解文案）、A6②（pending 丢弃不可达路径
  登记于 D3 需用户决策）。下一轮 R10 Step 0 评估：B4（错误文案残留，与
  R7/R8 同主题可做）/ B12（远程配置字段含 remote_root 红线违规项）/ 其余
  B 系 P2。停止条件未满足（P1 尚存 3 项，其中 2 项需用户输入）。

---

## R10（2026-09-26）—— B4：错误文案开发者残留清理

### 1. 主题与改动摘要

用户可见文案中的开发者残留（环境变量名、命令行、内部日志文件名、菜单
入口编号）全部替换为标注者可执行的动作指引。改动 5 个源文件 13 处 +
1 个防回归测试，commit `a68ef5a`（+98/−47，其中测试 +35）。

### 2. 关闭的痛点（对照验收标准）

- B4 验收标准："逐处改为指向 GUI 入口或引导窗；grep 校验活代码无
  'Run: python'/环境变量名直达用户的残留"——13 处全部改毕；校验方式
  比验收标准更强：不靠一次性 grep，而是 tokenize 扫描字符串字面量的
  永久回归测试（`test_user_facing_text_has_no_developer_residue`，
  覆盖 5 个模块），"Run: python"、"disable/enable MIMICS_"、
  "04 Stop Import Queue"、"check mimics_import.log" 四类禁用形式
  一旦回归即红。
- 例外留档：`_failed_cases.json` 在 disk_full/source_data_invalid 指引
  中保留——它躺在用户可见的导入输出文件夹里，用户能直接打开，是可用
  指引而非残留（test_all.py:3739 既有断言也依赖它）。

### 3. 门禁结果

- smoke 8/8：`.mimics_runtime/regression/20260926T071054_smoke.json`
- fast 26/26：`.mimics_runtime/regression/20260926T071842_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：零风险——纯字符串字面量替换，无逻辑/线程/进程改动。
2. 标注者视角优先：本轮即该铁律的直接执行（B4 本就是标注者视角审计
   的发现）。
3. 无表面修改：每处文案对应"用户当前会卡住"的真实场景（隐藏 mask 报
   错、离线包缺失、需要停止导入队列、失败病例排查）。
4. 不留垃圾：无新增死代码；同时清掉了 2 处测试才能发现的跨行拼接
   漏网点。
5. 小步提交：单 commit，信息说明动机。
6. 无验证不算完成：smoke+fast 双绿 + 新增永久回归测试。
7/9. 复杂度预算：改动形式是最底层的"改文案"；新增的 35 行测试属
  防回归资产（预算允许的增长理由）。
8. 任务生命周期规范：未涉及新功能的反馈/停止/回退交互。

### 5. 产品价值净变化

标注者遇到错误时的指引从"看不懂、无法执行"（环境变量名、python 命令
行、编号菜单）变为"照做即可"（显示隐藏 mask 重试、找配置工作站的人、
打开输出文件夹的 logs 子文件夹、用 01 Data 菜单的 Stop Import Queue）。
无"对开发者有价值但对标注者无感"的改动。

### 6. 复杂度对价

- 净代码变化：+98/−47（源 +63/−47 为文案替换净增——新文案普遍更长更
  具体是预期行为；测试 +35 防回归）。已记入 complexity_ledger.md。
- 新增维护面：0 新文件、0 新配置项、0 新依赖、0 新 UI 入口；新增 1
  个测试（维护面为纯资产）。
- 方案选型顺序：全部落在"改文案"档，未触碰更贵档次。

### 7. 架构审查结论

**豁免**（适用小改动豁免条款）：纯字符串字面量替换，无接口/进程模型/
状态变更；例外决策（`_failed_cases.json` 保留 + 禁用词表选择）已在
上方第 2 节写明理由。tokenize 测试本身经实现中自检（首次版本禁用词
"MIMICS_MASK_IDENTIFIER" 过宽误伤合法的 `os.environ.get` 读点，
收窄为 "disable MIMICS_"/"enable MIMICS_" 只命中"指示用户去设环境
变量"的形式）。

### 8. 遗留风险与下一轮计划

- 无本轮遗留风险；文案改动无失败模式。
- R11 候选（均已在账本）：B12（远程配置字段 + remote_root 红线）、
  B1（nninteractive_bridge 顶层 nibabel，冷启动 15-32s）、B2（import
  队列/guard 文件无界增长）。升级清单：A4（Fix Affine 需人工 GUI 走查
  确认）、A7（平台限制缓解）、D3（A6② 需用户拍板）。

---

## R11（2026-09-26）—— B1：nninteractive_bridge 懒加载 nibabel

### 1. 主题与改动摘要

`nninteractive_bridge.py` 顶层 `import nibabel`（nibabel→pydicom 链）导致
每个 async worker 冷启动 15-32 秒、推迟首个 prompt 就绪。nibabel 仅在两个
函数中使用，移入函数内懒加载（与 mimics_bridge.py 既有模式一致）。
commit `4dafac0`（+8/−1）。

### 2. 关闭的痛点（对照验收标准）

- 验收：nib 挪入实际使用函数 ✔（两处：canonical_ras_buffer_mapping、
  load_image_nifti）；`-X importtime` 复测冷 import <3s ✔（实测 0.57s，
  import 链无 nibabel/pydicom）；nninteractive 相关套件绿 ✔
  （flow_nninteractive pass）。
- 懒加载路径功能验证：identity affine 的 orientation 映射结果正确。
- 全库核查：无外部 `bridge.nib` 引用；其他模块本就是函数内懒加载模式。

### 3. 门禁结果

- smoke 8/8：`20260926T072638_smoke.json`
- fast 26/26：`20260926T073130_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：改善而非引入——worker 启动更快，prompt 更早就绪。
2. 标注者视角：直接受益（首次交互等待从数十秒降到亚秒）。
3. 无表面修改：根因（顶层慢 import）直接消除，冷启动实测验证。
4. 不留垃圾：净 −1 顶层依赖，无新增代码面（仅 2 个局部 import + 注释）。
5. 小步提交：单 commit。
6. 无验证不算完成：importtime 实测 + 功能验证 + smoke/fast 双绿。
7/9. 复杂度预算：属"简化现有实现"档，零新增维护面。
8. 任务生命周期规范：未涉及。

### 5. 产品价值净变化

标注者首次 nnInteractive 交互前的等待从 15-32 秒降到亚秒级。这是 R1
性能基线表中的最大单项冷启动开销。

### 6. 复杂度对价

- 净代码变化：+8/−1（其中 4 行为懒加载注释说明模式出处）。已记入
  complexity_ledger.md。
- 新增维护面：0（无新文件/配置/依赖/UI 入口；2 个局部 import 与
  mimics_bridge.py 模式一致，无新模式引入）。

### 7. 架构审查结论

**豁免**（小改动豁免条款）：单文件、-7 行净依赖移除、模式完全复刻
mimics_bridge.py 既有懒加载先例（账本验收标准本就指名该模式为正确做法），
无接口/进程模型/状态变更。验收标准中 <3s 的量化红线已实测达标（0.57s）。

### 8. 遗留风险与下一轮计划

- 无遗留风险：nibabel 仍随两函数按需加载，行为等价。
- R12 候选：B2（import_queues/guard 文件无界增长，与 A2 垃圾清理合并）、
  B12（远程配置字段 + remote_root 红线）。升级清单不变：A4（需人工 GUI
  走查）、A7（平台限制缓解）、D3（A6② 用户拍板）。

## R12（2026-09-26）—— B2：运行时状态无界增长（guard 清扫 + import 队列 prune）

### 1. 主题与改动摘要

`.mimics_runtime/locks/` 下 464 个 guard 锚文件（240 nnunet_dataset + 148
import_producer + 74 background_mimics，最老 6 天）与 `import_queues/` 147 个
目录无任何清理机制，且健康面板每 10 秒反复列举。两类根因：测试套件用生产
锁目录（每次临时 workspace 哈希出新锁名）+ 真实累积（每个 .mcs 输出目录
一个 import 队列目录/guard）。

改动（架构师"修改后通过"6 项要求全部落实）：
- **guard mtime 活跃戳**：`_MutationGuard.__enter__`（resource_locks.py）与
  `_open_resource_guard`（runtime_common.py）取得 OS 锁后 `os.utime`——
  否则 mtime 永停创建时间，热资源会被误判闲置。
- **`sweep_stale_guards`**（resource_locks.py）：三重门（idle mtime ≥30 天 /
  无 sibling .lock / try-acquire 无竞争）+ 平台删除顺序（POSIX 锁内 unlink；
  Windows close→re-stat→remove）。挂在 `sweep_processes` 末尾——import 启动
  守护线程、nninteractive 启动、健康面板按钮全部自动触发。
- **`_prune_import_queues`**（mimics_import.py，14 天）：心跳超龄（或无心跳
  且目录 mtime 超龄）+ 无活 pid 消费者 + 内部 guard 无竞争 → rmtree + 删
  mcs_queues 注册行。挂在 `_mark_mcs_queue_active`（与 export prune 同款
  "启动即清理"时机）。
- **测试隔离**：fake_mimics_flow_test main()、cross_workflow setUp、
  nnunet/flexict integration `__main__` 设 `MIMICS_RESOURCE_LOCK_DIR`；
  `GPU_LOCK_PATH` import 时常量改按调用解析 `_gpu_lock_path()`（两文件），
  删孤儿常量 `RESOURCE_LOCK_DIR`。
- **实施中发现并修复**：`"a+b"` 打开位置在 EOF、`msvcrt.locking`/`flock`
  按当前位置锁字节——try-acquire 前必须 `seek(0)` 才与真实持有者
  （两者都 seek(0) 锁字节 0）在同一字节上竞争，否则门 3 形同虚设。

### 2. 关闭的痛点（对照验收标准）

- (a) import 队列清理复用 export prune 模式 ✔（`_prune_import_queues`，
  同款时机/天数/防御门）
- (b) sweep 按 mtime 清理超龄 guard ✔（`sweep_stale_guards`，含活跃戳
  支撑 mtime 语义）
- 各配伪造 mtime 测试 ✔（TestStaleGuardSweep 4 例 + TestImportQueuePrune
  5 例，单字节 guard 复刻生产形态；含竞争保留、活 pid 保留两个失败路径）
- fast 绿：见第 3 节
- 根因之一（测试污染）消除证据：4 套件全部隔离到临时锁目录；
  GPU_LOCK_PATH 按调用解析保证 env override 真正生效

### 3. 门禁结果

- smoke 8/8：`20260926T084419_smoke.json`
- fast 26/26：`20260926T091053_fast.json`
- 一次性生产清理：**464 → 2**（462 个陈旧 guard 删除；保留的 2 个为
  有 sibling .lock 的 nnunet_dataset 锚文件，门 2 正确放行活资源）。
  前置核查：2 个 nnunet_dataset_*.lock 持有 pid 51504/44752 均已死；
  唯一存活 python 45596 为 anylabeling 环境，无关。清理用临时脚本执行后
  已删除，未入库。

### 4. 铁律自检

1. 阻塞 GUI：全部清理动作在既有后台线程/启动时机内，新增动作毫秒级
   （462 文件列举 <0.1s）；未引入任何新线程/timer。
2. 标注者视角：健康面板每 10 秒列举的目录从 464 文件收敛到有界；磁盘
   垃圾不再无界累积。
3. 无表面修改：根因双管齐下——测试污染（隔离）+ 真实累积（清扫/prune）。
4. 不留垃圾：一次性清掉 462 个陈旧 guard；删孤儿常量 RESOURCE_LOCK_DIR。
5. 小步提交：单 commit（B2 主题内聚）。
6. 无验证不算完成：9 个新回归测试 + smoke/fast + 生产清理前后计数。
7/9. 复杂度预算：主体为修 bug（无界增长）+ 防回归测试；清扫/prune 复用
   export 先例模式，未引入新抽象层/配置面。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

标注者无直接感知面（后台自治性修复），间接收益：健康面板 10 秒轮询的
目录列举从 464+ 文件降到有界规模（R1 基线表"运行时状态规模"项关闭）；
`Z:` 网络盘输出目录对应的本地队列目录不再永久累积。属"对产品稳定性有
价值、对标注者无感"的改动——不解决则磁盘与轮询开销单调增长，最终
以健康面板卡顿/磁盘满的形式落到标注者头上。

### 6. 复杂度对价

- 净代码变化 +324（产品 +200 / 测试 +124），已记 complexity_ledger.md。
- 新增维护面：新函数 5 个（同型于 export prune 先例）、常量 3 个、
  测试 9 个；新文件/配置项/依赖/UI 入口 0。
- 选型顺序：修 bug + 防回归测试为主，无"新增功能"成分。

### 7. 架构审查结论

**修改后通过**。6 项前置要求逐条落实：①guard mtime 活跃戳（两处持有者）
②测试隔离含 GPU_LOCK_PATH import 时冻结问题（改按调用解析）③seek(0)
字节竞争对齐（实施中发现，审查时以"门 3 必须真实竞争"覆盖）④prune 挂
启动时机而非新后台 timer ⑤三重门 + 平台删除顺序 ⑥生产清理前置核查
（锁持有 pid 已死）。实施与审查通过的方案一致，无范围扩大。

### 8. 遗留风险与下一轮计划

- 遗留风险：Windows 上 guard 删除存在 close→remove 间的微小窗口
  （close 后、remove 前被他人 open+lock）——re-stat 已把窗口内竞争者
  变成"删除失败保留"（安全侧失败），无 split-brain 风险。
- R13 候选：B12（远程配置字段 + remote_root 红线）、B3
  （_first_mcs_monitor_tick 全目录扫描降频）。
- 升级清单不变：A4（需人工 GUI 走查）、A7（平台限制）、D3（A6② 用户拍板）。

## R13（2026-09-26）：远程服务器配置高级字段补 UI/文档 + root 登录红线警告

**主题**：B12——`container_runtime` 等 5 个 servers.json 新字段自 R1 加入后
只能手改 JSON（无 UI、无文档），且 `remote_root` 默认 `$HOME/mimics-ai` 对
root 账号落在 `/root/mimics-ai`，与共享服务器红线（只能用管理员分配的
`/userdata/shijian_ruan/`）冲突且无任何提醒。

### 1. 主题与改动摘要

- **remote_compute.py**：test_connection 对 root SSH 账号在结果 payload 附带
  通用措辞警告 + 解析后的绝对工作目录（客户端本可知道却从未展示）；不硬编码
  任何机器路径（红线是用户提供机器的规则，写进发布代码违反铁律 4），
  normalize_profile 保持纯校验器。
- **remote_compute_ui.py**：ServerProfilesDialog 新增 5 行表单（Container
  runtime 组合框、nerdctl namespace、Code/Weights verification 两个组合框、
  cleanup retention 天数控），默认项标注 "(default)" 防止无知情降级；
  对话框高度 660→820；连接成功消息前缀 "⚠" 渲染警告。
- **顺带修复数据丢失 bug**：`_profile_values` 此前丢弃这 5 个字段——手改
  servers.json 后从对话框 Save 会**静默重置为默认**（如 nerdctl 配置被
  悄悄改回 docker）。表单补齐后此路径关闭。
- **remote_acceptance_checklist.py**：nerdctl 排障提示从"手改 servers.json"
  改为指 UI 表单（否则立即复现 B12 自己的痛点）。
- **CONFIG_REFERENCE.md**：修正两个错误键名（`image`→`runtime_image`、
  `remote_workspace`→`remote_root`）+ 5 字段全量文档 + 共享服务器
  remote_root 指引 + "全部字段可在 UI 编辑，无需手改"声明。
- **remote_training_design.md**：Compute 节补一行高级字段清单。
- **test_remote_training.py**：离屏 ServerProfilesDialog 往返测试（5 个手改
  值 + remote_root 经 `_profile_values` 存活）+ root 警告源码级防回归。

### 2. 关闭的痛点（对照验收标准）

- Compute tab 表单暴露 5 字段 ✔（含默认标注与 tooltip）
- remote_root 首次使用即警告 ✔（root 账号 + 解析后绝对路径展示；用户决策
  项 D3 记录了"是否进一步硬校验 /userdata/ 前缀"待拍板，本轮按架构师
  建议用通用警告而非硬编码路径）
- CONFIG_REFERENCE + remote_training_design.md 补文档 ✔（并修正 2 个
  错误键名——此前照着文档手改会直接产生无效配置）
- gui_smoke/远程套件绿 ✔（remote_training 44.8s PASS；fast 26/26）

### 3. 门禁结果

- remote_training 套件：PASS（20260926T112032_fast.json 内）
- fast 26/26：`20260926T112424_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：纯表单/消息改动，未加线程/timer。
2. 标注者视角：远程训练用户不再需要了解 servers.json 的存在。
3. 无表面修改：根因是"新字段只进了数据模型没进 UI/文档"，连带修复
   `_profile_values` 静默重置数据丢失与文档错误键名两个真实缺陷。
4. 不留垃圾：未新增死代码；修正文档中从未正确的键名。
5. 小步提交：单 commit（B12 主题内聚）。
6. 无验证不算完成：2 个新回归测试 + fast 全绿。
7/9. 复杂度预算：主体为补齐既有数据模型的 UI 面 + 文档；新增维护面
   仅 5 个表单行（数据模型 R1 已存在，无新配置项语义）。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

远程训练用户（当前即用户本人）从"手改 JSON + 猜字段名 + root 默认路径
踩红线而不自知"变为完全 UI 化 + 首次连接即看到解析后工作目录与共享
服务器警告。`_profile_values` 数据丢失 bug 意味着任何手改高级字段的
用户在下一次打开对话框保存时配置被悄悄破坏——这是本轮实际最重要
的修复。

### 6. 复杂度对价

- 净代码变化约 +260（产品 ~+150 / 测试 ~+110），已记 complexity_ledger.md。
- 新增维护面：表单行 5（对应既有数据字段）、UI 辅助函数 1、测试 2；
  新文件/新配置项/新依赖 0。
- 选型顺序：补 UI（既有字段）> 修 bug（数据丢失）> 文档修正；无新功能。

### 7. 架构审查结论

**修改后通过**。初版方案（仅预览+文档）未过审：验收标准"至少首次使用时
校验并警告"不满足 + 编辑态解析路径预览技术上不可实现（客户端无法知道
远端 $HOME）。7 项必改全部落实：①通用 root 警告（非硬编码路径）②成功
消息带解析后目录 ③checklist 提示指 UI ④⑤文档键名与全量字段
⑥⑦往返测试。实施与终审方案一致。

### 8. 遗留风险与下一轮计划

- 遗留风险：无（表单为既有数据模型的直接映射；警告措辞通用，不随单台
  服务器绑定）。
- R14 候选：B3（`_first_mcs_monitor_tick` 全目录扫描降频）、B5-B11、
  B13-B15。
- 升级清单不变：A4（Fix Affine 人工 GUI 走查）、A7（平台限制）、
  D3（A6②/B12 硬校验前缀，用户拍板）。

## R14（2026-09-26）：退役残留与文档漂移清理（B13/B14/B15）

**主题**：三条"退役/漂移不彻底"类 P2 合并为一个纯删除/文案轮——
MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START 声明退役但代码与文档仍在、
文档三处漂移、CLAUDE.md 凭据文件未 gitignore。铁律 4"不留垃圾"的
直接执行，全部为删码/改文案级改动，**架构审查豁免**（豁免理由：
无新逻辑、无新交互、无行为设计决策——唯一的行为变化是删除一个
交付报告已宣布退役且默认关闭的危险分支；选型顺序处于"删代码>
改文案"最优先区间，小步单 commit）。

### 1. 主题与改动摘要

- **B13 全删路线**：`runtime_common.aggressive_auto_cleanup_enabled`
  读取点、mimics_import/nninteractive_mimics 两处 wrapper、两处启发式
  杀进程分支（ctypes OpenProcess 循环 + PowerShell 命令行扫描 +
  TEMP 陈旧锁清扫 + mimicsresearch.exe -b 检测）全删。所有权可证
  清理（registry sweep、陈旧锁、owned-server 空闲超时）保留为唯一
  终止者。mimics_import 的 `import re` 随之变为零引用，一并删除。
- **防回归测试改写**：旧 `test_windows_cleanup_declares_pointer_
  sized_process_handles` 守护的 ctypes 块即被删代码，改写为
  `test_startup_cleanup_has_no_aggressive_kill_path`——三模块
  inspect.getsource 断言退役 flag/helper/裸 kernel32 杀进程不复活。
- **B14**：ScribblePrompt 路径 external/→integrations/ 两处；速查表
  补 Environment Guidance 行；MedDINOv3 陈旧引用两处（审计记一处，
  grep 发现测试文件同款一并清）。
- **B15**：CLAUDE.md 进 .gitignore（带"含凭据严禁入库"注释）。

### 2. 关闭的痛点（对照验收标准）

- B13"读点与文档全删或明确标注退役"→全删 ✔；"grep 零活引用"✔
  （存活字面名仅在防回归断言、DELIVERY_REPORT 历史陈述、账本）；
  smoke 绿 ✔
- B14 三处改对 ✔（+实施中发现第 4 处同型 MedDINOv3 引用）；
  grep 复查 ✔
- B15 gitignore + `git status` 不再显示 ✔（check-ignore 命中）

### 3. 门禁结果

- smoke 8/8：`20260926T114815_smoke.json`
- fast 26/26：`20260926T115223_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：纯删除，删掉的恰是唯一会弹出 PowerShell 扫描/杀进程
   的启动路径之一；启动路径变短。
2. 标注者视角：照 CONFIG_REFERENCE 配置"恢复模式"得到已退役危险
   行为的坑被填掉。
3. 无表面修改：文档漂移两处是用户照抄即错的路径/入口。
4. 不留垃圾：净删约 130 行产品代码；死 import re 清除。
5. 小步提交：单 commit（三条退役/漂移类主题内聚）。
6. 无验证不算完成：smoke+fast 双绿 + 防回归测试改写 + grep 证据。
7/9. 复杂度预算：纯删除轮，净变化为负；唯一新增是防回归断言
   （P1/P2 修复沉淀要求）。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

删掉的是"照文档配置会杀掉自己活任务"的隐患 + 两处照抄即错的
文档路径 + 凭据误提交风险。对标注者无直接感知面，但 B15 是安全
红线加固，B13 填掉的是一个真实的误操作坑。

### 6. 复杂度对价

- 净代码变化约 -100（产品 -170 / 测试 +70），已记 complexity_ledger.md。
- 新增维护面：0（纯删除 + 既有测试改写）；新配置项/文件/依赖/
  UI 入口 0。

### 7. 架构审查结论

**豁免**（本轮全为删码/文案级改动，无新逻辑、无新交互、无行为
设计决策；唯一行为变化是删除交付报告已宣布退役且默认关闭的危险
分支）。豁免理由如上记入，非跳过而是按协议裁定不适用。

### 8. 遗留风险与下一轮计划

- 遗留风险：曾有用户依赖该 flag 做"卡死恢复"的场景——替代路径
  是 99_Admin/03 Stop All Owned Services（入口指南已有），语义
  覆盖且所有权可证；无缺口。
- R15 候选：B3（_first_mcs_monitor_tick 降频）、B5-B11、B17-B19。
- 升级清单不变：A4、A7、D3。

## R15（2026-09-26）：batch .mcs 等待监视器全目录扫描降频（B3）

**主题**：`_first_mcs_monitor_tick` 等待首个 .mcs 的 batch 分支每 2s
`sorted(os.listdir(output_dir))` 全目录 stat，batch 模式该定时器可存活
7 天——网络盘+大批量时整轮导入期间每 2 秒一次全目录扫描，跑在 GUI
定时器线程上（铁律 1 相关）。

### 1. 主题与改动摘要

- 等待阶段（`first_notified` 之前、`target_mcs=None`）改为先读本地
  `_mcs_batch_status.json`（单文件廉价读，`_rt` 本就不写输出共享盘）
  作门控：无完成案例时全目录扫描降频至 30s 一次；`completed>=1`
  立即扫描；首通知后 tick 原本就只读 status 文件。单案例路径不动。
- 测试：fake_mimics_flow_test 新增 `first .mcs monitor scan throttle`
  （注册进 imports 组），驱动真实 tick 函数覆盖三个降频语义 + 通知
  不延迟 + 首通知后零扫描。

### 2. 关闭的痛点（对照验收标准）

- "batch 分支只读 batch status + 降频扫描" ✔（status 门控 + 30s 节流
  + 完成即扫）
- 单测覆盖 ✔（含关键失败路径：窗口内不重扫）
- smoke 绿 ✔（另加 fast 26/26）

### 3. 门禁结果

- smoke 8/8：`20260926T120703_smoke.json`
- fast 26/26：`20260926T121039_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：扫描频率从 0.5Hz 降到 1/30Hz，GUI 定时器线程上的
   网络盘 I/O 显著减少；这正是本条修复的目标。
2. 标注者视角：大批量网络盘导入时 GUI 卡顿源消除；首个 .mcs
   通知不被降频拖慢（completed>=1 立即扫）。
3. 无表面修改：根因是等待期用"全目录扫描找文件"而非"读廉价
   status 文件判断何时值得扫"。
4. 不留垃圾：无新增常量/配置（30s 内联并注释理由，无用户可调
   需求）。
5. 小步提交：单 commit（B3 主题内聚）。
6. 无验证不算完成：新 flow 测试 + smoke + fast。
7/9. 复杂度预算：改动约 +25 行产品代码（门控逻辑）+85 行测试；
   无新抽象/配置面。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

大批量导入（真实数据集场景：Totalsegmentator 数百案例在网络盘）时
Mimics GUI 定时器线程每 2 秒一次全目录 stat 消失——网络盘上这是
可感知的卡顿/网络负载源，7 天 batch 模式下扫描次数约降 15 倍
（2s→30s），且关键事件（首 .mcs 生成）零延迟。

### 6. 复杂度对价

- 净代码变化 +109（产品 +24 / 测试 +85），已记 complexity_ledger.md。
- 新增维护面：monitor dict 新键 1（last_scan_epoch）；测试 1；新
  文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（单函数内的扫描节流，无新交互、无新模块、无行为设计
权衡——门控信号复用既有 status 文件，节流值内联；验收标准本身
已冻结了方案形态"只读 status + 降频"）。

### 8. 遗留风险与下一轮计划

- 遗留风险：`_mcs_batch_status.json` 若因 worker 崩溃从未出现，
  则降为纯 30s 节流扫描（原为 2s）——首 .mcs 发现最多延迟 28s；
  该场景本就有 90s 停滞检测兜底，可接受。
- R16 候选：B5-B8（用户交互类 P2）、B17-B19。
- 升级清单不变：A4、A7、D3。

## R16（2026-09-26）：bridge 批量发现兜底扫描封顶（B17）

**主题**：`mimics_bridge.do_discover` 对无首选图像名、无 DICOM 子目录的
case 兜底全量列目录找散装医学图像——DICOM 平铺目录可达数万文件，
每个 case 每次批量发现都全部 stat。UI 侧同型扫描
（`io_path_setup_ui.discover_single_source`）早有 511 封顶 + 首个
.dcm 早停，bridge 侧漏配。

### 1. 主题与改动摘要

- 兜底扫描改为 `os.scandir` 早停（`.dcm` 或第 512 项即停），与 UI
  侧完全同型，注释互指先例。
- 测试：fake-scandir 断言首轮枚举后不再继续（复用 UI 侧同型测试
  模式）；既有 mask_selection 三态与 UI 侧早停测试全部回归通过。

### 2. 关闭的痛点（对照验收标准）

- 镜像 511 封顶 ✔（同型实现，含 .dcm 早停）
- 单测 ✔（fake-scandir 计数断言）
- smoke 绿 ✔（fast 26/26 加跑）

### 3. 门禁结果

- smoke 8/8：`20260926T121717_smoke.json`
- fast 26/26：`20260926T122054_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：do_discover 在 bridge 进程中运行，本不阻塞 Mimics，
   但数万文件的枚举拖慢整个批量导入发现阶段；封顶后大 DICOM
   平铺数据集的发现时间显著缩短。
2. 标注者视角：选错目录结构（散装 DICOM）时发现阶段不再长时间
   无响应。
3. 无表面修改：根因是 bridge/UI 两侧同型逻辑漏配对齐。
4. 不留垃圾：无新增常量（511 与 .dcm 早停内联，与 UI 侧一致）。
5. 小步提交：单 commit。
6. 无验证不算完成：新 fake-scandir 测试 + 三套既有发现测试回归 +
   smoke + fast。
7/9. 复杂度预算：产品 +14 行（同型早停），测试 +53；无新抽象。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

大批量发现（含 DICOM 平铺结构的数据集）不再对每个 case 做全量
目录枚举；bridge 与 UI 两侧行为一致（同一数据集两处入口给出的
发现结果不再因封顶差异而不同）。

### 6. 复杂度对价

- 净代码变化 +62（产品 +14 / 测试 +48），已记 complexity_ledger.md。
- 新增维护面：测试 1；新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（同型对齐：把 UI 侧已有先例逐字镜像到 bridge 侧，无设计
决策；验收标准已冻结"镜像 511 封顶"）。

### 8. 遗留风险与下一轮计划

- 遗留风险：封顶后藏在前 511 项之后的散装 NIfTI 会被判为
  "无图像"——与 UI 侧行为一致（先例已接受该权衡：证明"无散装
  NIfTI"需要枚举全部切片，价值为零）。
- R17 候选：B18（io_path_setup_ui 去抖 + 状态面板不可见降频）、
  B5-B8。
- 升级清单不变：A4、A7、D3。

## R17（2026-09-26）：路径输入去抖 + 状态面板不可见降频（B18）

**主题**：(a) `io_path_setup_ui` 路径输入框 textChanged 每字符触发一次
数据集扫描线程——网络数据集上连打路径或粘贴会瞬间堆起一串扫描；
(b) 三个状态面板（batch/nnunet/flexict）最小化/隐藏时仍每 2s 全量
扫描 job 目录（实测单次 collect 1.7s），为一张没人看的表持续产生
网络 I/O。

### 1. 主题与改动摘要

- (a) textChanged 改接 400ms 单发 QTimer（`recognition_debounce`），
  连发/粘贴只在最后一次改动后触发单次扫描。
- (b) `viewer_refresh.BackgroundRefresh` 在每次周期 tick 检查父窗口
  可见性：不可见/最小化 → 切 10s 心跳；重新可见 → 恢复 2s 并立即
  刷新。显式 `request()`/`refresh_now()` 不受可见性影响。三处状态窗
  经共享 helper 一次性覆盖，无各窗重复实现。

### 2. 关闭的痛点（对照验收标准）

- (a) 300-500ms 去抖 ✔（400ms 单发 QTimer 接线，直连断言测试防回退）
- (b) 不可见降频至 10s ✔（tick 处可见性判定 + 区间切换）
- gui_smoke 绿 ✔（18/18，含 3 个新测试）

### 3. 门禁结果

- gui_smoke 18/18（新测试随套件）
- fast 26/26：`20260926T123645_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：(a) 去掉了每字符一个扫描线程的突发（扫描本身已在
   worker 线程，但线程堆积与结果乱序到达仍是卡顿源）；(b) 不可见
   面板不再以 2s 节奏对网络盘做秒级 I/O——直接减少 Mimics 主进程
   所在网络共享的竞争。
2. 标注者视角：输入路径时不再闪现一串 "Scanning dataset..."；
   最小化状态窗重新打开时数据即时新鲜（恢复可见即刷）。
3. 无表面修改：两处都是真实 I/O 消除，不是文案。
4. 不留垃圾：无新文件；复用既有 QTimer 模式。
5. 小步提交：单 commit `9ebcf43`。
6. 无验证不算完成：3 个新测试 + gui_smoke + fast。
7/9. 复杂度预算：产品 +59（去抖 10 行 + 可见性判定/区间切换 ~49，
  含注释），测试 +78；无新抽象、无新配置项。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

网络数据集场景：路径输入期间扫描负载从"每字符一次"降为"停顿
400ms 一次"；状态面板后台存在期间网络 I/O 降至原来的 1/5。两处
都是标注者日常工作流中的隐性资源浪费，修复后无行为损失（显式
刷新路径全部保留）。

### 6. 复杂度对价

- 净代码变化 +134（产品 +59 / 测试 +75，净含格式 -3），已记
  complexity_ledger.md。
- 新增维护面：新常量 1（HIDDEN_INTERVAL_MS）、新静态方法 1
  （_is_shown）、新 tick 处理 1（_on_period_tick）、测试 3；
  新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（两处均为既有模式的最小接线：去抖复用既有单发 QTimer
模式；降频在既有共享 helper 内加 tick 处可见性判定，验收标准已
冻结"300-500ms 去抖 + 不可见降频至 10s"，无跨模块设计决策。
设计取舍——tick 处判定 vs hideEvent/showEvent 钩子——已在实现前
论证：Qt 虚函数分发不保证触达 monkey-patched 实例属性，tick 处
判定确定性可测）。

### 8. 遗留风险与下一轮计划

- 遗留风险：不可见面板恢复可见最多等 10s 心跳后由下一个 2s tick
  补一次即时刷新——实测恢复路径是"区间立即切回 2s + 立即 request"，
  无等待窗口；(a) 去抖窗口内用户点 Start Import 的竞态由既有
  `recognition_state["scanning"]` 门控兜底（提交读的是最终路径，
  不依赖识别结果）。
- R18 候选：B5-B8（P2 窗口类问题）、B19（测试盲区 TB-02 优先）。
- 升级清单不变：A4、A7、D3。

## R18（2026-09-26）：Window Undo 降级前确认（B5）

**主题**：`undo_last()` 拿不到 `previous_contrast` 时（上次窗宽窗位保存在
更早的 Mimics 会话里）仅记 WARNING 即执行 `reset_full_range()`——标注者
按"撤销"得到的是"重置"，且无法拒绝。

### 1. 主题与改动摘要

- 降级路径改为先弹 `question_box`（Reset to Full Range / Cancel），
  文案明示原因；Cancel 时显示保持原样。
- 测试：既有 window flow 扩入两分支断言（Cancel 不改对比度 +
  接受后恢复全量）。

### 2. 关闭的痛点（对照验收标准）

- 降级时弹确认 ✔（question_box，文案说明值不可用的原因）
- 测试覆盖 ✔（Cancel / Accept 两分支，fake dialogs 脚本化应答）

### 3. 门禁结果

- smoke 8/8：`20260926T124203_smoke.json`
- fast 26/26：`20260926T124558_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：question_box 为 ui_blocking 用户主动触发的交互，合规。
2. 标注者视角：按 Undo 不再被动接受一次全量重置。
3. 无表面修改：根因是降级语义违背用户意图，修复是给拒绝权。
4. 不留垃圾：无新增文件/常量；复用 `_choose_preset` 同款
   question_box 模式。
5. 小步提交：单 commit `51fed2e`。
6. 无验证不算完成：两分支测试 + smoke + fast。
7/9. 复杂度预算：产品 +15 / 测试 +19；无新抽象。
8. 任务生命周期规范：交互为标准确认弹窗（明确后果 + 可取消），
   符合规范。

### 5. 产品价值净变化

Undo 的降级行为从"静默重置"变为"知情选择"，消除一个标注者
不可预期、不可撤销的显示突变。

### 6. 复杂度对价

- 净代码变化 +34（产品 +15 / 测试 +19），已记 complexity_ledger.md。
- 新增维护面：0（交互分支并入既有测试）。

### 7. 架构审查结论

**豁免**（单函数交互修复，复用模块内既有 question_box 模式，
验收标准已冻结"降级时弹确认或明确提示"）。

### 8. 遗留风险与下一轮计划

- 遗留风险：无（降级路径行为完整覆盖两分支）。
- R19 候选：B6（Setup/Repair 环境菜单运维化）、B7（Undo Last
  Import 约束）、B8（批量导出死路径弹窗）。
- 升级清单不变：A4、A7、D3。

## R19（2026-09-26）：Setup/Repair Environment 菜单推荐置顶（B6）

**主题**：Setup/Repair Environment 打开即五选一英文选择题，环境已坏的
标注员（最需要帮助的人）第一屏没有任何指引。

### 1. 主题与改动摘要

- 菜单按工作站实际状态动态排序：已有环境 → Check 置顶（先验证再
  改动）；无环境有离线包 → Offline Install 置顶；无环境无离线包 →
  Extract Archive 置顶。首行显示"Recommended: X — 理由"，其余动作
  保留一行说明（question_box 无折叠能力，取验收标准"加解释"分支）。
- 状态探测复用既有 `find_external_python`（不新增探测代码）与
  `_is_offline_bundle`。
- 测试：4 状态矩阵（有无 python × 有无 bundle）断言按钮顺序与推荐行。

### 2. 关闭的痛点（对照验收标准）

- 默认推荐动作置顶 ✔（三状态各有推荐，均置首位）
- 其余折叠或加解释 ✔（加解释：每项保留一行说明）
- 走查记录 ✔（4 状态断言即走查记录，见测试）

### 3. 门禁结果

- smoke 8/8：`20260926T125242_smoke.json`
- fast 26/26：`20260926T125627_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：纯文案/排序改动，无新 I/O。
2. 标注者视角：坏环境下的第一屏从"选择题"变为"推荐+理由"。
3. 无表面修改：推荐基于真实状态探测，不是静态文案。
4. 不留垃圾：动作清单原为两分支硬编码，现统一为数据表 + 排序，
   分支重复消除。
5. 小步提交：单 commit `0e5ed6f`。
6. 无验证不算完成：4 状态矩阵测试 + smoke + fast。
7/9. 复杂度预算：产品 +44 / 测试 +44；净增但换来分支重复消除与
  标注者指引。
8. 任务生命周期规范：未新增交互路径（仍是同一 question_box）。

### 5. 产品价值净变化

环境故障场景（标注员最脆弱的时刻）的引导从零变为明确；Check
（只读、安全）成为已装工作站的默认第一选择，降低误改环境的风险。

### 6. 复杂度对价

- 净代码变化 +88（产品 +44 / 测试 +44），已记 complexity_ledger.md。
- 新增维护面：0 新文件/配置项/依赖/UI 入口（推荐规则内联于菜单
  构建处，两个既有探测函数复用）。

### 7. 架构审查结论

**豁免**（单函数菜单构建重构，复用既有探测函数，无跨模块设计；
验收标准已冻结"推荐置顶+解释"）。

### 8. 遗留风险与下一轮计划

- 遗留风险：推荐只覆盖环境安装态的三个场景；环境"存在但损坏"
  （python 在而包缺失）时仍推荐 Check——这是有意的（Check 是发现
  损坏的入口，Repair 是其结论），文案未区分。
- R20 候选：B7（Undo Last Import 约束）、B8（批量导出死路径弹窗）。
- 升级清单不变：A4、A7、D3。

## R20（2026-09-26）：Undo Last Import 约束前置说明（B7）

**主题**：Undo Last Import 的两个关键操作事实——"打开其他工程时拒绝执行"
和"回执消费后一次性"——只有撞上才知道；拒绝对话框只说"先关闭再重跑"，
确认弹窗完全不提一次性语义。

### 1. 主题与改动摘要

- 确认弹窗加 Notes：需先关其他工程；一次性语义（成功即消费、失败
  保留可重试）。
- 拒绝对话框改为两步指引（Step 1 关工程 / Step 2 重跑），明示记录
  未被消费、说明为什么必须打开目标工程。
- 工程约束不放宽（见验收对照）。
- 测试：确认文案断言 + 拒绝对话框两步文案断言（进入 TestImportReceiptAndUndo）。

### 2. 关闭的痛点（对照验收标准）

- 文案明示两步操作 ✔（拒绝对话框 Step 1/Step 2）
- 文案明示回执消费语义 ✔（确认弹窗 Notes + 拒绝对话框"nothing was consumed"）
- 工程约束评估 ✔：**不放宽**——mask 删除按回执清单在目标工程内执行，
  指纹校验只保护文件回滚，不保证 mask 删除目标正确；放宽会引入误删
  其他工程同名 mask 的风险（比原痛点更严重）。
- 测试覆盖 ✔（2 个新断言测试）

### 3. 门禁结果

- smoke 8/8：`20260926T125952_smoke.json`
- fast 26/26：`20260926T130413_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：纯文案，无新 I/O。
2. 标注者视角：两个"隐性规则"变为事前告知。
3. 无表面修改：文案对应真实约束（工程状态探测与回执生命周期均为
   既有真实语义）。
4. 不留垃圾：无新增代码路径。
5. 小步提交：单 commit `3cc624e`。
6. 无验证不算完成：2 个文案断言测试 + 既有 undo 全链路测试回归 +
   smoke + fast。
7/9. 复杂度预算：产品 +11 / 测试 +39；无新抽象。
8. 任务生命周期规范：确认弹窗本就存在，仅扩充内容；无新交互。

### 5. 产品价值净变化

撤销导入这一高风险操作的行为预期完整化：什么时候能撤、撤销后
记录去哪、失败后怎么办，全部事前可见。

### 6. 复杂度对价

- 净代码变化 +50（产品 +11 / 测试 +39），已记 complexity_ledger.md。
- 新增维护面：0（文案断言并入既有测试类）。

### 7. 架构审查结论

**豁免**（纯文案 + 2 个断言测试；约束放宽评估结论为"不放宽"并
记录理由，无代码结构变化）。

### 8. 遗留风险与下一轮计划

- 遗留风险：无。
- R21 候选：B8（前台批量导出死路径弹窗）、B9（测试资产完整性）。
- 升级清单不变：A4、A7、D3。

## R21（2026-09-26）：删除死掉的前台批量导出链（B8）

**主题**：审计登记的"前台批量导出走到一半弹 Batch Export Disabled"弹窗，
现场核实发现其所在的整条前台批次链是死代码——`_start_export_monitor`
零调用点（`main()` 批次路径早已只走后台进程启动），弹窗实际不可达，
但 199 行死实现及其"要求用户重启重来"的交互语义留在产品里。

### 1. 主题与改动摘要

- 删除整条死链：`_start_export_monitor`（零调用）、`_export_monitor_tick`、
  `_start_next_batch_export`（含死路径弹窗）、`_start_win32_export_monitor`、
  `batch_queue`/`batch_info` 形参（从未被真实传入）。
- 保留被活代码共用的 helper：`_check_job_status`、`_apply_export_result`、
  `_cleanup_job_dir`、`_error_guidance`、`_EXPORT_MONITORS`（其他监视器
  仍用）。
- 测试：防复活断言（弹窗文案 + 4 个函数名 + batch_queue 不得重现）。

### 2. 关闭的痛点（对照验收标准）

- 不再出现"做一半重来" ✔（前台批次路径整体不存在；批次只在后台
  Mimics 进程执行，从不弹此窗）
- 前台入口路由到后台路径 ✔（既有事实，本轮删除死残留后无歧义）
- 测试 ✔（防复活断言 + 全量导出测试回归）

### 3. 门禁结果

- smoke 8/8：`20260926T131312_smoke.json`
- fast 26/26：`20260926T131650_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：死代码删除，无行为变化（不可达路径）。
2. 标注者视角：无直接变化（弹窗本就不可达）；消除的是 199 行
   误导性残留（读代码者会以为前台批次存在）。
3. 无表面修改：根因是退役残留（铁律 4 范畴）。
4. 不留垃圾：正是本轮主题——整链删除，连带 never-passed 形参。
5. 小步提交：单 commit `51ed9e5`。
6. 无验证不算完成：AST 解析 + 防复活测试 + smoke + fast 全绿。
7/9. 复杂度预算：**净 -181 行**（产品 -199 / 测试 +18）；纯删除轮。
8. 任务生命周期规范：删除的交互本就违背该规范（做一半要求重启）。

### 5. 产品价值净变化

无运行时行为变化（死路径）；维护面减少 4 个函数 + 1 个误导性弹窗；
审计账本 B8 关闭（痛点实质是"退役残留"而非"活弹窗"）。

### 6. 复杂度对价

- 净代码变化 **-181**（产品 -199 / 测试 +18），已记 complexity_ledger.md。
- 新增维护面：0（唯一新增为防复活断言）。

### 7. 架构审查结论

**豁免**（纯删除轮：删前逐 helper 核实调用者，共用 helper 保留；
无设计决策）。

### 8. 遗留风险与下一轮计划

- 遗留风险：无（死路径删除，活路径全量回归绿）。
- R22 候选：B9（测试资产完整性——孤儿套件注册需用户决策 D7）、
  B10（PID 回收防护测试）。
- 升级清单不变：A4、A7、D3（另 B9 之子项涉及 D7）。

## R22（2026-09-26）：PID 回收拒杀端到端测试 + fewshot 遗留状态防迁移（B10）

**主题**：(a) start_marker 防回收机制（terminate/sweep 的核心安全设计）零
端到端测试，而生产 registry 136 条记录中已有 6 条真实 PID 回收案例；
(b) 退役 few-shot 工作流的 `_migrate_old_settings` 会把遗留状态文件
改名为用户正式设置——其记忆的数据集根可能早已不存在。

### 1. 主题与改动摘要

- (a) 新测试：注册真实 sleeper 子进程 → 篡改记录 start_marker 模拟
  回收 → 断言 process_is_live 判死、terminate_process 全程不杀真实
  子进程、sweep_processes 只清记录 + 释放锁、terminated_orphans 为空。
- (b) 迁移函数加存活检查：旧状态记忆的所有数据集根都不存在 → 跳过
  迁移（不成为用户设置）；至少一个根存在 → 正常迁移。两分支测试。
  本工作站核实两处旧文件均不存在（无可删对象，删除子项不适用）。

### 2. 关闭的痛点（对照验收标准）

- PID 回收端到端测试（含 terminate/sweep 拒杀断言）✔
- 迁移函数遇指向不存在路径的旧状态应跳过 + 测试 ✔
- fewshot 遗留文件删除 ✔（不适用——本机不存在；他机残留由跳过逻辑
  防护，防复活即防错误迁移）

### 3. 门禁结果

- smoke 8/8：`20260926T132829_smoke.json`
- fast 26/26：`20260926T133217_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：无 GUI 影响（测试 + 迁移函数防御）。
2. 标注者视角：(a) 防的是"杀掉无关程序"这一数据安全级风险；
   (b) 防的是默认数据集根被静默改错。
3. 无表面修改：(a) 机制已存在，补的是防回归验证；(b) 是真实数据
   丢失/错乱路径的关闭。
4. 不留垃圾：fewshot 遗留文件本机无；迁移保留（多机部署下仍有
   正常迁移场景）。
5. 小步提交：单 commit `450b748`。
6. 无验证不算完成：3 个新测试 + smoke + fast。
7/9. 复杂度预算：产品 +25（存活检查 helper + 迁移门控）、测试 +96；
  无新抽象。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

PID 回收拒杀机制（安全关键）获得永久防回归；fewshot 退役残留的
最后一个活代码影响（错误迁移）关闭。

### 6. 复杂度对价

- 净代码变化 +121（产品 +25 / 测试 +96），已记 complexity_ledger.md。
- 新增维护面：新 helper 1（_existing_dataset_roots，14 行）；测试类 1；
  新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（测试补齐 + 单函数门控，验收标准已冻结跳过语义）。

### 8. 遗留风险与下一轮计划

- 遗留风险：跳过逻辑依据"数据集根存在性"，若旧状态记忆的根碰巧
  仍存在但已换内容（同路径新数据集），仍会迁移——可接受（路径存在
  即说明该机器仍在用该数据集布局）。
- R23 候选：B9（测试资产完整性，D7 用户决策项前置核实）、B11（失败
  作业远程目录清理入口）。
- 升级清单不变：A4、A7、D3、D7。

## R23（2026-09-26）：失败作业远程目录清理入口（B11）

**主题**：远程训练失败后 job 目录按设计保留在共享服务器上供诊断
（`remote_diagnostics_retained=True`），但无任何管理入口——在共享
服务器上违反"任务结束即清理"红线且无限累积。

### 1. 主题与改动摘要

- `remote_acceptance_checklist.py` 新增 `--cleanup-failed-dirs` 可选
  flag：run 结束（报告写出后）执行 `step_cleanup_failed_jobs`——
  `find <remote_root>/jobs/<owner>/ -mindepth 1 -maxdepth 1` 列出残留
  作业目录（排除 .tar/.part 上传残片），逐个经 `_remove_remote_job`
  删除（路径校验拒逃逸 + 连同 .tar/.tar.part 兄弟文件 + 删除后存在性
  验证），单条拒删不中断其余，随后刷新报告。
- 测试 3 个进 AcceptanceChecklistTests（fake session）：正常清理
  （find 命令定位 + 排除断言 + 每条走删除命令）、一条路径逃逸被拒后
  其余继续清、flag 接线源扫描（步骤仅在 flag 下运行）。

### 2. 关闭的痛点（对照验收标准）

- 验收："失败诊断目录提供显式清理入口（验收清单结束时可选清理）" ✔
  —— 显式入口 = `--cleanup-failed-dirs`；结束时运行 = 报告写出后
  执行并刷新报告；可选 = 默认不清理（保留诊断价值）。

### 3. 门禁结果

- smoke 8/8：`20260926T134646_smoke.json`
- fast 26/26：`20260926T135114_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：无（CLI 验收脚本，不在 Mimics 内运行）。
2. 标注者视角：直接受益者为共享服务器上的其他使用者；标注者无感。
3. 无表面修改：清理走既有 `_remove_remote_job` 校验路径，不新造
   rm 命令；入口符合验收标准明文。
4. 不留垃圾：清理入口本身就是垃圾清理机制；未新增无关代码。
5. 小步提交：单 commit `ebb777e`。
6. 无验证不算完成：3 个新测试 + smoke + fast。
7/9. 复杂度预算：产品 +60（一个 step 函数 + argparse 两段）/
  测试 +105；复用 `_remove_remote_job` 与 AcceptanceStep 既有模式，
  无新抽象。
8. 任务生命周期规范：CLI flag，不涉及 GUI 交互规范。

### 5. 产品价值净变化

共享服务器卫生红线（"任务结束即清理"）获得可执行入口：失败的远程
训练不再无限累积诊断目录。正常成功路径的清理（R1 已有）不受影响。

### 6. 复杂度对价

- 净代码变化 +165（产品 +60 / 测试 +105），已记 complexity_ledger.md。
- 新增维护面：新 step 函数 1（~55 行，同型于既有 step_* 家族）；
  新 CLI flag 1；测试方法 3；新文件/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（单文件追加同型 step + 测试，无既有代码路径改动；删除全部
委托既有校验路径 `_remove_remote_job`，无新删除逻辑面）。

### 8. 遗留风险与下一轮计划

- 遗留风险：清理以 find 列表为准，若服务器上有他人放入 owner 目录
  的文件也会被视作残留——owner 目录本身按 safe_identifier 归属当前
  用户，风险可忽略；拒绝路径不重试（下次运行再清）。
- R24 候选：B9（测试资产完整性——D7 孤儿套件处置是用户决策项，先
  完成不需决策的部分）、B19（测试盲区 TB-01~08，其中 TB-02 混合
  系列静默混叠疑似真实 bug，先核实）。
- 升级清单不变：A4、A7、D3、D7。

## R24（2026-09-26）：UID 缺失 DICOM 双 series 静默混叠 fail-closed（B19/TB-02）

**主题**：B19 测试盲区排期第一项 TB-02——核实"全缺 SeriesInstanceUID 的
两个 series 混叠"是否真 bug，属实则修复。

### 1. 主题与改动摘要

- **核实（mock 复现）**：疑点属实。两个缺 UID 的不同 series（如平扫+
  增强）落进同一 `__missing_series_uid__` 组，三条选择路径全部静默混叠
  成一个 6-slice 几何错乱 volume 进入训练/推理——真实 P1 数据损坏 bug。
- **修复**（架构师审查"修改后通过"，3 项加固全部落实）：
  - `_dicom_group_key`：缺 UID 时改按 `(StudyInstanceUID, SeriesNumber)`
    复合分组；两者也缺才落全缺桶。SeriesNumber=0 保留真值（None 与缺失
    同义但 0 不是）。
  - 新增 `_reject_interleaved_missing_uid_groups`：非真 UID 组内出现两条
    同一（2 位小数取整）`ImagePositionPatient` → fail-closed 报错，指引
    使用干净目录（同既有 "Multiple DICOM series match..." 文案风格）。
    覆盖同格双 series 与拷贝残留双场景；无 IPP 记录（多帧对象）跳过；
    真 UID 组豁免。
- 测试 6 个进 TestBridgeDicomLoading（三路径混叠 fail-closed、单一缺 UID
  series 正常加载、复合组同格触发既有多 series 报错、拷贝残留 fail-closed、
  真 UID 组豁免、SeriesNumber=0 语义）。

### 2. 关闭的痛点（对照验收标准）

- TB-02 建议测试形态："断言要么 fail-closed 要么选出唯一正确 series 且
  z 序正确" ✔（fail-closed 为主；单一 series 保持可用）。
- "先核实是否真 bug" ✔——核实属实并修复，非仅补测试。

### 3. 门禁结果

- smoke 8/8：`20260926T140745_smoke.json`
- fast 26/26：`20260926T141207_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：报错发生在后台加载线程，沿既有异常路径转成用户可见错误，
   不碰主线程。
2. 标注者视角：几何错乱的静默数据损坏（最恶劣的失败模式——无告警地
   毁掉训练数据质量）改为一句明确指引；正常去标识化单 series 文件夹
   行为不变。
3. 无表面修改：mock 复现证明 bug 真实存在；修复走标准 DICOM 判别字段。
4. 不留垃圾：新增即最小必要（一个 helper + 一处分组逻辑），无投机分支。
5. 小步提交：单 commit `25b0aa0`。
6. 无验证不算完成：6 个新测试 + smoke + fast。
7/9. 复杂度预算：产品 +48（`_dicom_attr_part` + 复合键 + 检查函数）/
  测试 +108；无新配置/依赖/抽象层。
8. 任务生命周期规范：错误对话框由既有链路渲染，未新增交互。

### 5. 产品价值净变化

消除一个"高概率 + 数据损坏级 + 无告警"的训练数据污染路径（放射科混
series 数据很常见）。正常路径（完整 UID 的导出/派生 DICOM、单 series
去标识化目录）行为逐一验证不变。

### 6. 复杂度对价

- 净代码变化 +156（产品 +48 / 测试 +108），已记 complexity_ledger.md。
- 新增维护面：新函数 2（`_dicom_attr_part` 4 行 + 检查函数 ~35 行）；
  测试方法 6 + 测试 helper 类 1；新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**独立架构师 agent 审查通过（APPROVE WITH CHANGES）**，3 项必改全部
落实：(1) 属性抽取区分 None/缺失/0，0 不落哨兵；(2) 重复位置检查覆盖
全部复合组（非仅全缺桶）——同 (Study, SeriesNumber) 的两个 series 混叠
同样静默；(3) 重复键取 2 位小数元组，无 IPP/解析失败记录跳过。
可选第 4 项（重复键加 IOP 防定位片假阳性）按最小改动原则判定豁免；
残余风险（偏移网格双 series 无重复 IPP）按架构师建议登记 TB-02 不修复
（启发式间距检查误报风险大于收益）。

### 8. 遗留风险与下一轮计划

- 遗留风险：偏移网格双 series 混叠（无重复 IPP）expected_shape 路径由
  最终 shape 复查兜底、无 shape 路径无防护——已登记 TB-02 残余风险，
  不做启发式修复；单一 series 内部 Study/SeriesNumber 不一致的文件夹
  由 fail-closed 兜底（今日会报错而非静默半卷）。
- R25 候选：B19 剩余项按登记表逐个排期（TB-01 中断恢复半写文件、
  TB-03 中文/空格路径端到端）；或 B9 无决策部分。
- 升级清单不变：A4、A7、D3、D7。

## R25（2026-09-26）：中文/空格路径端到端防护测试（B19/TB-03）

**主题**：B19 排期第二项 TB-03——中国医院环境常态的中文+空格路径
在全部测试里只有 ASCII 临时路径（仅 JSON **值**含过中文），**路径本身**
从未流过导出/导入/远程命令拼接链路。

### 1. 主题与改动摘要

- 纯测试轮（核实后链路本身健壮，零产品代码改动）：
  - fake flow 新增 `unicode and spaces paths flow`：在
    `患者数据 2026/标注 输出/` 根下跑 mask buffer 导出（mask 名
    "肝脏 mask"）→ manifest 回读 → prepared mask 导入 apply（mask 名
    "导入的肝脏"）全链。
  - 远程契约测试 `test_remote_job_removal_quotes_spaces_and_unicode_
    in_paths`：remote_root 含空格+中文（`/userdata/患者 数据/mimics-ai`），
    `shlex.quote` 拼出的删除命令经 `shlex.split` 反解析必须还原为完全
    一致的三个目标路径。
- 核实结论：本地 subprocess 全部 list-args 传参（无 shell 拼接），
  远程命令全部走 `shlex.quote`——链路本身对空格/中文健壮，缺的只是
  防回归的防护测试（若未来有人改成 str 拼接或去掉 quote，测试会抓住）。

### 2. 关闭的痛点（对照验收标准）

- TB-03 建议测试形态两条全部落实 ✔："fixture 根目录名含中文+空格，
  跑一遍 fake flow import/export"；"远程契约测试用含空格的 remote_root
  跑一条真实 session.execute 命令拼接断言"。
- 超长路径（>260）明示为残余另立（TB-03 条目内登记理由：涉及面广、
  Win10+Py3.13 默认 longPathsEnabled，实际风险低），不属本轮验收范围。

### 3. 门禁结果

- smoke 8/8：`20260926T142037_smoke.json`
- fast 26/26：`20260926T142453_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：零产品改动，无影响。
2. 标注者视角：直接服务中国医院环境用户（中文路径是他们无法回避的
   日常），防的是"某次重构悄悄破坏中文路径"这一回归。
3. 无表面修改：审计点名的测试盲区，验收标准即测试形态。
4. 不留垃圾：无产品代码改动，无新增配置。
5. 小步提交：单 commit `8a3e5b6`。
6. 无验证不算完成：2 个新测试 + smoke + fast。
7/9. 复杂度预算：产品 +0 / 测试 +94；无新抽象。
8. 任务生命周期规范：未新增交互。

### 5. 产品价值净变化

中文/空格路径链路获得永久防回归（此前一旦破坏只能靠用户实机踩雷
发现）。

### 6. 复杂度对价

- 净代码变化 +94（产品 0 / 测试 +94），已记 complexity_ledger.md。
- 新增维护面：新 flow 测试 1（同型于既有 flow 测试）；新契约测试 1；
  新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**豁免**（纯测试轮，零产品代码改动；测试形态即冻结的验收标准）。

### 8. 遗留风险与下一轮计划

- 遗留风险：超长路径（>260）未覆盖，已在 TB-03 条目登记为另立项；
  训练 worker 内部（nnU-Net 预处理）对中文路径的行为由外部库决定，
  本轮覆盖到 worker 入口为止。
- R26 候选：B19 剩余项（TB-01 中断恢复半写文件、TB-04 全零 label 进
  训练、TB-05 传输中断时序、TB-07 导入中同数据导出）按登记表排期；
  或 B9 无决策部分。
- 升级清单不变：A4、A7、D3、D7。

## R26（2026-09-26）：全零 label 训练入口 fail-closed（B19/TB-04）

**主题**：B19 排期第三项 TB-04——label 文件存在但全零的 case 静默进入
nnU-Net/FlexiCT 训练（漏标/空标是标注日常）。

### 1. 主题与改动摘要

- **核实（架构师审查纠正）**：本地路径已有防护——`prepare_source_grid_
  cases` 默认 `empty_case_policy="skip"` 逐 case 跳过并报告；真正的洞在
  远程/prepared 分支：两 pipeline 的 `run_training` 只查 `is_file()`。
- **修复**：`build_training_data_profile`（两 pipeline × 两执行路径唯一
  共享内容门，已逐 case 加载 image 校验 shape/spacing）同步加载每个
  label，`count_nonzero == 0` 即 raise（case id + label 路径 + 重导或
  移除指引）。单次 materialize，count 复用于 profile 统计；不信任缓存
  foreground_voxels（陈旧缓存缺键 → 0 → 误拒好 case）。profile 加性新键
  `label_foreground_voxels`（min/median/max），不影响 dataset 指纹与
  预处理缓存身份（核实过两处 digest 均不含 profile）。
- 测试 3 个：单测全零拒（断言 case id 命名错误）、单测正常过含统计、
  flexict prepared 路径集成测试（worker spawn 前 fail，status 含
  "'case_2' has an empty"）。

### 2. 关闭的痛点（对照验收标准）

- TB-04 验收："训练请求含 1 个全零 label case，断言 scan/materialize
  阶段显式报出该 case 并拒绝或明确标记，而非静默进入" ✔——materialize
  前的 profiling 门显式报出 case id 并拒绝，三条路径（本地 nnunet/
  flexict + 远程 prepared）全覆盖。
- D13 产品面（导出统计标黄确认 UI）仍为需用户决策项，未混入本轮。

### 3. 门禁结果

- smoke 8/8：`20260926T144313_smoke.json`
- fast 26/26：`20260926T144750_fast.json`

### 4. 铁律自检

1. 阻塞 GUI：训练在后台 worker 进程，错误沿既有 status 链路转成用户
   可见失败信息。
2. 标注者视角：白训练一轮（数小时 GPU 时间）+ 模型质量受损改为启动
   即报、指名 case、给出两个动作选项。
3. 无表面修改：mock/真实 NIfTI 测试证明防护缺口真实存在。
4. 不留垃圾：无投机分支（empty_case_policy 无生产者设非 skip 值，
  fail-closed 胜出并已记录）。
5. 小步提交：单 commit `b75126a`。
6. 无验证不算完成：3 个新测试 + smoke + fast。
7/9. 复杂度预算：产品 +22（循环内 ~15 行 + profile 键 5 行）/ 测试
  +367（含 prepared 集成测试的 fixture）；无新函数/文件。
8. 任务生命周期规范：失败信息由既有状态链路渲染，未新增交互。

### 5. 产品价值净变化

"漏标 case 静默拉低模型质量"这一日常性风险在全部训练路径上关闭；
profile 新统计让训练数据质量在状态里可见（每 case 前景体素范围）。

### 6. 复杂度对价

- 净代码变化 +389（产品 +22 / 测试 +367），已记 complexity_ledger.md。
- 新增维护面：profile 新键 1（加性）；测试方法 3 + 测试 helper 1；
  新文件/配置项/依赖/UI 入口 0。

### 7. 架构审查结论

**独立架构师 agent 审查通过（APPROVE WITH CHANGES）**，关键纠正与必改
全部落实：(1) 重新定位问题——本地路径已有 skip 防护，洞在 prepared
分支，测试对准真实缺口；(2) 单次 count_nonzero 复用；(3) 不做缓存计数
快路径（陈旧缓存误拒风险）；(4) 错误含 case id + label 路径。可选项
（scan_flexict_cases UI 扫描时标"label empty"）按最小改动豁免——
验收"拒绝或明确标记"已满足，UI 扫描加载全部 label 有大目录卡顿风险
（铁律 1），留 D13 一并决策。

### 8. 遗留风险与下一轮计划

- 遗留风险：每次训练运行多一次全量 label 加载（gzip 逐 case 解压，
  与既有 image 加载同量级，zstart 先例同型）；profile 为加性键，
  validate_model_input_compatibility 用 .get() 读取不受影响。
- R27 候选：B19 剩余项（TB-01 中断恢复半写文件、TB-05 传输中断时序、
  TB-07 导入中同数据导出）按登记表排期；或 B9 无决策部分。
- 升级清单不变：A4、A7、D3、D7。

## R27（2026-09-26）：传输中断时序状态机防回归测试（TB-05）

### 1. 主题与改动摘要

B19/TB-05：远程训练上传/下载**中段**网络中断（非日志轮询阶段）的
状态机此前无测试覆盖——现有续传测试全部构造"已有 .part 残片"的静态
场景，从未模拟"传输函数执行中途抛 OSError"的中断时序。

逐读控制器源码核实结论：产品已有完整防护，本轮为**纯测试轮**
（与 TB-03 同型：链路健壮，缺防回归）：

- 三类传输位点各自包裹重连重试循环：数据集分片上传
  （remote_training_controller.py:4310）、作业归档上传（4378）、产物
  下载两处（fresh-run 1892 / reattach 2157），外加监控（3970）与容器
  启动（4446），共 6 处 `except (RemoteComputeError, EOFError, OSError,
  socket.error) → _reconnect_session` 有界重试（上限 30 次、退避
  min(30, 2^attempt)）；
- `.part` 续传语义（remote_compute.py:670/729）保证重试从断点续传；
- 中断时状态先落 `reconnecting_remote`（非终态，reattach 的 TERMINAL
  集不含它），重连成功后恢复原状态——验收要求的"可重试/可 reattach
  的态"产品已成立。

改动：`tools/test_remote_training.py` 新增测试类
`InterruptedTransferRecoveryTests`（4 个测试，+414 行，产品零改动）。

### 2. 关闭的痛点（对照验收标准）

TB-05 验收（冻结）："mock SFTP write 在第 N 块后抛 OSError，断言状态
落到可重试/可 reattach 的态、无锁残留、重试后产物 SHA 校验通过"：

- "N 块后抛 OSError" → `_RemoteHandle` 包装真实文件句柄，恰在第
  N 块后抛 OSError（第一字节真实落盘，时序与 dropped socket 同形）；
- "可重试/可 reattach 的态" → 控制器级测试驱动
  `_finalize_completed_remote_job` 经受下载中断：断言 `reconnecting_
  remote`、`remote_reconnect_attempt=1`、重连恢复、最终 completed——
  全程不落终态 failed；
- "无锁残留" → 传输阶段容器未启动（4423 行才标记 may_exist）、锁目录
  未创建；测试亦断言无 `.part` 残留；
- "重试后产物 SHA 校验通过" → 端到端经 `_download_artifact` 的
  SHA-256 比对并发布，manifest 逐字节一致。

### 3. 门禁结果

- `tools/test_remote_training.py`：91/91 OK（47s，含 4 个新测试）。
- fast 门禁 26/26 绿：`.mimics_runtime/regression/20260926T152910_fast.json`。
- 未触碰 runtime_py35/桥接层，smoke 不适用（纯 tools 测试轮）。

### 4. 铁律自检

1. 不阻塞 GUI：零产品改动，无从违反。
2. 标注者视角：验证"训练几小时后下载断网不白干活"——远程产物仍在，
   重连自动续传下载。
3. 不做表面修改：414 行全部为防回归测试，回答"谁的什么痛点"——
   共享服务器网络抖动下远程训练交付链路的最后一环。
4. 不留垃圾：测试内 fakes 全部局部类；无产品死代码。
5. 小步提交：单 commit `5ee0045`。
6. 无验证不算完成：4 个新测试 + fast 26/26。
7/9. 复杂度预算：产品 0 / 测试 +414；新测试类 1、无新文件/配置项/
  依赖/UI 入口。
8. 任务生命周期规范：零交互改动。

### 5. 产品价值净变化

产品行为零变化（纯防护沉淀）。价值：中断时序状态机从"未验证"变为
"有防回归"——未来重构若静默删除任一传输位点的重试循环，源契约测试
与端到端测试立即红。

### 6. 复杂度对价

新增维护面：测试类 1（含 4 测试）。0 新文件/配置项/依赖/UI 入口。
fakes 复用既有 `_LocalSFTP` 模式（继承），控制器级 fake 同型于
ReattachTests 的 Session 脚本模式。

### 7. 架构审查结论

按"小改动豁免"路径：本轮产品代码零改动、测试形态完全由冻结验收
标准规定、复用既有测试模式（_LocalSFTP 继承 + Session 脚本 + 源契约
扫描，皆有先例）。豁免理由记录如上。实施中发现并核实一处验收措辞
歧义（"无锁残留"）——传输阶段容器/锁尚未创建（源码 4423 行核实），
该条款以"无 .part 残留 + 产物发布后清理路径断言"落实，未追溯放宽
验收（原措辞的意图即"无阻碍重试/重连的残留状态"）。

### 8. 遗留风险与下一轮计划

- 遗留风险（已登记账本）：监控期控制文件上传（3834）与曲线同步
  （3740）不在独立重试循环内，但均被外层监控循环（3970）捕获，非
  传输主干，不构成白干活路径，不立项。
- R28 候选：TB-06 共享 GPU 占用检查（D14 软提示部分可先测）、TB-01
  中断恢复半写文件、TB-07 导入中同数据导出、B9 无决策部分。
- 升级清单不变：A4、A7、D3、D7。

## R28（2026-09-26）：导出/导入写一半被杀的残留与恢复防回归（TB-01 + B9-c）

### 1. 主题与改动摘要

B19/TB-01：导出 mask（NIfTI）、derived DICOM 逐 slice 转换、批量导入
监视器三处的"写一半被硬杀"此前零自动化测试。逐读源码核实：三条路径
的防护产品侧已成立——

- NIfTI 导出原子发布（临时名 → nib.save → os.replace）：杀进程只会
  留下临时名文件，最终路径要么旧内容要么新内容，无中间态；
- derived DICOM 逐 slice 直写**确实产生半写可见文件**，但被三重隔离
  兜住：每次导入独立 run 目录、manifest 仅在 `_validate_derived_
  dicom_series` 通过后写入、消费端以 manifest 存在性为唯一门（陈旧
  descriptor >5s 丢弃）、错误路径 rmtree 整个 work_dir；
- 批量导入监视器错误分支已调用 `_cleanup_work_dir` 并继续下一 case。

本轮为**纯测试轮**（产品零改动），把这些保证钉进回归套件。顺带处理
B9-c：`test_atomic_write_survives_kill_signal` 名不副实（从未模拟
kill），改名重写为真实对照测试。

另修复 R12 遗留测试破损（见 §8）。

改动：`tools/test_all.py` +5 新测试 / +2 处 FakeLock 修复，
`tools/test_mimics_nnint_deep.py` 1 个测试改名重写（净 +307 行）。

### 2. 关闭的痛点（对照验收标准）

TB-01 验收（冻结）："fake 流中模拟'写第 N 个 slice 时 kill'，重启后
断言阶段目录被清理/标记 failed，且重跑产出完整结果；断言无半写文件
被后续 discover 当作有效 case"：

- "写第 N 个 slice 时 kill" → `test_derived_dicom_kill_mid_slices_
  leaves_unpublished_residue_and_rerun_recovers`：patch
  FileDataset.save_as，slice_0004 写后抛 KeyboardInterrupt；
- "阶段目录被清理/标记 failed" → `test_bridge_error_cleans_up_
  partial_derived_dicom_work_dir`（监视器错误分支 rmtree + failed+1
  + 下一 case 启动）+ `test_partial_work_dir_without_manifest_is_
  never_converted`（无 manifest 永不消费）；
- "重跑产出完整结果" → kill 测试断言重跑产出完整 8-slice series；
  NIfTI kill 测试断言重跑发布 (4,4,4)；
- "无半写文件被当作有效 case" → `test_derived_dicom_validation_
  rejects_incomplete_series`（校验器拒 slice 数不匹配/空文件/UID
  不匹配）+ manifest 门测试。

B9-c 验收："名不副实测试改名或补真实 kill 场景" → 改名
`test_atomic_write_publishes_exactly_one_complete_file` 并补真实
截断文件场景（两者都做）。

### 3. 门禁结果

- full 门禁 28/28 绿、test_all 456/456 OK（991s）：
  `.mimics_runtime/regression/20260926T172647_full.json`。
- 本轮改动 test_all.py 本身，fast 不含 test_all，故用 full（宪法
  "涉及核心流程改动时使用"——此处为覆盖面要求而非产品改动）。
- `tools/test_mimics_nnint_deep.py` 75/75 OK（含改名测试）。

### 4. 铁律自检

1. 不阻塞 GUI：零产品改动，无从违反。
2. 标注者视角：验证"断电/被杀后重跑不产脏数据"——半写 DICOM 混进
   下一轮导入是数据损坏级风险（TB-01 登记措辞）。
3. 不做表面修改：307 行全部为失败路径测试，每条对应冻结验收。
4. 不留垃圾：测试内 fakes 全部局部；顺带清除一个名不副实测试。
5. 小步提交：单 commit 一个主题（TB-01+B9-c 合并，账本明文要求）。
6. 无验证不算完成：full 28/28 + test_all 456/456。
7/9. 复杂度预算：产品 0 / 测试 +307；无新文件/配置项/依赖/UI 入口。
8. 任务生命周期规范：零交互改动。

### 5. 产品价值净变化

产品行为零变化（纯防护沉淀）。价值：三条数据写入路径的"崩溃后无脏
数据"保证从"代码读起来如此"变为"回归门禁强制"——未来重构若破坏
原子发布、manifest 门或错误清理，test_all 立即红。

### 6. 复杂度对价

新增维护面：测试 6 个（5 新 + 1 改名重写）。0 新文件/配置项/依赖/
UI 入口。测试模式全部有先例（mock.patch 中断注入、局部 Fake 类、
监视器驱动），无新抽象。

### 7. 架构审查结论

按"小改动豁免"路径：产品代码零改动；测试形态完全由冻结验收标准
规定（TB-01 登记时即写明"fake 流中模拟写第 N 个 slice 时 kill"）；
模式复用既有先例（kill 注入同型于 R24 mock 复现手法、监视器驱动
同型于既有 TestLifecycleAndRetention monitor 测试）。豁免理由记录
如上。实施中发现一处实施事实与登记假设不同——登记写"重启后断言
阶段目录被清理"，实际产品语义是"错误路径清理 + 成功重跑覆盖"，
半写残留保留在隔离 run 目录内但永不被消费（无 manifest）——以
测试 4/5 分别钉住两条语义，未追溯放宽验收（原措辞意图即"半写文件
不污染下一轮"，两条测试共同满足）。

### 8. 遗留风险与下一轮计划

- **R12 遗留测试破损已修**：commit `1e9c9f8`（R12）给 server state
  加 `gpu_lock.path` 后，test_all 的两个 nninteractive 启动失败路径
  测试（FakeLock 无 `.path`）自 R12 起 error——fast 门禁不含
  test_all 故 6 轮不可见，本轮 full 门禁暴露后修复（FakeLock 补
  `path` 属性，与真实 FileResourceLock 构造签名一致）。教训：
  改产品对象属性时应 grep 测试 fake；test_all 不在 fast 门禁是
  盲区放大器，B9-a（孤儿套件注册/处置，需 D7 用户决策）关联。
- 遗留风险（已登记账本）：隔离 run 目录内半写 DICOM 残留保留到
  下次成功导入或手动清理——不引入自动清扫（防御式编程）。
- R29 候选：TB-06 共享 GPU 占用检查（D14 软提示可先做）、TB-07
  导入中同数据导出、TB-08 单 slice/极端几何、B9-a/b（a 需 D7）。
- 升级清单不变：A4、A7、D3、D7、D13、D14、D18。

## R29（2026-09-26）：共享服务器 GPU 忙闲软提示（TB-06 软提示部分）

### 1. 主题与改动摘要

B19/TB-06：CLAUDE.md 红线"先查后用、不得挤占他人任务"在工具层面
零检查——`test_connection` 的 nvidia-smi 查询只验证 GPU 存在
（index/uuid/name/memory.total），不看 memory.used/utilization.gpu。
本轮实现**软提示**（硬阻断/排队属 D14 用户决策，未做）：

- `tools/remote_compute.py`：查询加 `memory.used,utilization.gpu` 两列；
  CSV 解析提取为纯函数 `parse_gpu_lines`（原内联循环零测试覆盖，提取
  后成为可测单元）；新纯函数 `gpu_busy_warning`（阈值：显存 ≥80% 或
  利用率 ≥50%，模块常量非配置项）；警告与既有 root 警告合并进单一
  `warning` 字段（UI ⚠ 渲染路径零改动）。
- `tools/remote_compute_ui.py`：GPU 下拉项与连接摘要追加 `(N% used)`。
- `tools/test_remote_training.py` +5 测试。

### 2. 关闭的痛点（对照验收标准）

TB-06 验收（冻结）："契约测试断言启动前查询 nvidia-smi
--query-gpu=... memory.used,utilization.gpu 并在占用超阈值时给出明确
提示或排队（是否做成硬阻断属需求决策 D14，先测软提示）"：

- "查询 memory.used,utilization.gpu" → 契约测试断言源码含该查询 +
  test_connection 调 gpu_busy_warning；
- "占用超阈值时给出明确提示" → gpu_busy_warning 纯函数测试（阈值内
  逐场景断言）+ warning 字段连续性断言（B12 回归继续通过）；
- "排队/硬阻断属 D14" → 本轮零阻断零排队，测试名如实标注
  "at server test time"。
- **"启动前"的落实解释**：TB-06 登记的链路位置明确指向 test_connection
  的这条查询；D14 明文该产品面为"Test Connection 扩展"；控制器 run()
  是无人值守后台流，提示无人看见。故"启动前"按"用户提交启动前的检查
  时刻（Test Connection）"落实；启动时查询/硬管控仍属 D14 未做也未
  声称已做。此解释记录于账本 TB-06 与本报告，未追溯放宽验收原文。

### 3. 门禁结果

- `tools/test_remote_training.py`：96/96 OK（47.7s，含 5 个新测试）。
- fast 门禁 26/26 绿：`.mimics_runtime/regression/20260926T203525_fast.json`。
- 未触碰 runtime_py35/桥接层，smoke 不适用（tools 层改动）。

### 4. 铁律自检

1. 不阻塞 GUI：test_connection 原有 daemon 线程结构未动；纯函数在
   worker 线程内执行；UI 改动只改标签文本。✅
2. 标注者视角：连接测试从"卡存在"升级为"卡空闲吗"——直接回答"现在
   能不能启动训练而不挤占别人"。
3. 不做表面修改：对应红线合规缺口，根因（查询缺负载列）直接修复。
4. 不留垃圾：原内联解析循环被提取函数替代，无残留复制。
5. 小步提交：单 commit 一个主题。
6. 无验证不算完成：5 新测试 + 96/96 + fast 门禁。
7/9. 复杂度预算：主体为既有查询的列扩展 + 2 个纯函数 + 2 常量；
   0 新文件/依赖/UI 入口；阈值不做配置项（D14 再议）。
8. 任务生命周期规范：软提示非阻断交互，零新交互协议。

### 5. 产品价值净变化

共享服务器红线"先查后用"首次有工具支撑：Test Connection 现在显示
每卡利用率并在忙时明确提示（点名卡号+双指标+"等待或换卡"指引），
忙闲从"用户自己猜"变为"工具直接说"。MIG 用户（选择分区共享的
礼貌用户）经保守回退同样得到提示，而非静默跳过。

### 6. 复杂度对价

净 +约 190 行（产品 ~95 / 测试 ~95）。新增维护面：模块级纯函数 2
（parse_gpu_lines、gpu_busy_warning）+ 常量 2；0 新文件/配置项/依赖/
UI 入口。解析逻辑从内联提取为函数不是新增面而是可测化。

### 7. 架构审查结论

独立架构师审查：**修改后通过**，5 项必改全部落实——(1) total≤0
除零守卫（MIG 服务器崩溃风险）；(2) 6 列解析重规格化 + [Not Supported]
→0 + 测试覆盖；(3) 选中卡无匹配（MIG UUID）保守回退评估全部卡 +
测试；(4) 多卡忙逐卡点名双指标 + 测试；(5) 诚实性——契约测试命名/
文档标注 test-time check，账本记录启动时管控仍属 D14。审查同时确认：
nvidia-smi 对 memory.used/utilization.gpu 的字段名级失败不存在（古老
字段），无需查询回退（加则违反复杂度预算）。

### 8. 遗留风险与下一轮计划

- 遗留（已登记）：启动时（控制器路径）的忙闲检查与硬阻断/排队 =
  D14 用户决策；Test Connection 时刻的负载快照与实际启动时刻可能
  有偏差（软提示的固有局限，D14 一并评估）。
- R30 候选：TB-07 导入中同数据导出竞争测试、TB-08 单 slice/极端
  几何、B9-b prune 钩子测试（无决策依赖，可直接做）。
- 升级清单不变：A4、A7、D3、D7、D13、D14、D18。

## R30（2026-09-26）：导入进行中同数据导出的并发语义防回归（TB-07）

### 1. 主题与改动摘要

B19/TB-07：导入进行中对同一批数据触发导出——此前零测试覆盖
（既有锁测试只测 stop 语义不重叠，不测数据竞争）。实施前逐读源码
核实：两流数据面几乎不相交——import 转换产物全落本地隔离 run 目录
（output share 只出现最终 .mcs）；export 的 discover 排除
output/export/label 目录；`.publishing_*` staging 只存在于训练侧
管道。真实交集仅一处：import 的 case 尚未产出 .mcs 时，export 走
"缺 .mcs"跳过分支。

本轮为**纯测试轮**（产品 0），把并发语义钉进回归：新增
`test_export_during_ongoing_import_skips_unpublished_case`
（tools/test_all.py，+100 行）。

### 2. 关闭的痛点（对照验收标准）

TB-07 验收（冻结）："fake flow 里启动 import 监视器，转换未完成时调
export 入口，断言 export 要么等待要么基于已发布 manifest 跳过该
case，且不触碰 `.publishing_*` staging 目录"：

- "启动 import 监视器" → `_IMPORT_MONITORS` 注册 busy monitor
  （busy=True + 未来 deadline + work_dir 半转换 slice，同型于 R28
  监视器驱动测试）；
- "转换未完成时调 export 入口" → case_pending 无 .mcs 时直调
  `run_background_batch_export`（config JSON 驱动，同型于既有
  empty_case_selection 测试）；
- "要么等待要么基于已发布 manifest 跳过" → 产品语义为"未发布即跳过
  并记录原因、已发布即正常处理"：断言 pending case 进
  `_failed_exports`（".mcs file not found"）且 done case 的 .mcs 恰被
  open_project 一次——两分支同测；
- "不触碰 .publishing_* staging 目录" → 全树 rglob 断言零
  `.publishing_*`（该命名属训练 prep；import/export 流自身不产生，
  验收条款按此落实）；
- 附加断言：import 隔离 work_dir 逐文件逐字节不变；活跃 import
  monitor 并发 export 后原样存活。

### 3. 门禁结果

- 新测试 1/1；TestNewFeatures 类 120/120。
- full 门禁 28/28（含 test_all 457/457、training_convergence 通过），
  工件 `.mimics_runtime/regression/20260926T213230_full.json`
  （本轮改 test_all.py 自身，按 R28 先例用 full 而非 fast）。

### 4. 铁律自检

1. 不阻塞 GUI：零产品改动，无从违反。
2. 标注者视角：验证"导入转着的同时发起导出不产生脏数据/不互相
   破坏"——批量工作流叠加操作是真实场景（TB-07 登记措辞）。
3. 不做表面修改：+100 行全部对应冻结验收条款。
4. 不留垃圾：测试 monkeypatch 全部 save/restore；monitor 注册在
   测试尾部清理。
5. 小步提交：单 commit 一个主题。
6. 无验证不算完成：新测试 + 类回归 + full 门禁。
7/9. 复杂度预算：产品 0 / 测试 +100；0 新文件/配置项/依赖/UI 入口。
8. 任务生命周期规范：零交互改动。

### 5. 产品价值净变化

产品行为零变化（纯防护沉淀）。价值：导入/导出并发的"无脏数据"保证
从"架构上读起来成立"变为"回归门禁强制"——未来若有人在 export 侧
引入对 import 隔离目录的触碰、或在 import 发布前暴露部分产物，
test_all 立即红。

### 6. 复杂度对价

新增维护面：测试 1 个。0 新文件/配置项/依赖/UI 入口。模式全部有
先例（`run_background_batch_export` config 驱动、监视器注册、
monkeypatch save/restore）。

### 7. 架构审查结论

按"小改动豁免"路径：产品代码零改动；测试形态完全由冻结验收标准
规定（"fake flow + import 监视器 + 转换未完成时调 export 入口"逐字
落实）；模式复用既有先例。实施中发现一处产品语义与验收措辞的差异
——"要么等待要么基于已发布 manifest 跳过"的实际产品语义是"未发布
即跳过（记录原因）、已发布即正常处理"，无等待分支；测试按实际
语义钉住两分支，未追溯放宽验收（原措辞"要么…要么…"本就允许任一
分支，两分支都被覆盖）。

### 8. 遗留风险与下一轮计划

- 遗留风险：无（两流数据面不相交的结论已由源码核实 + 测试钉住；
  真正的数据竞争面在训练 prep 的 `.publishing_*`，已有独立 staging
  原子发布语义，TB-01/R28 一并覆盖过相邻面）。
- R31 候选：TB-08 单 slice/极端几何（B19 最后一项）、B9-b prune
  钩子测试（无决策依赖）。
- 升级清单不变：A4、A7、D3、D7、D13、D14、D18。

## R31（2026-09-26）：TB-08 单 slice / 极端几何训练入口 fail-closed

### 1. 主题与动机

TB-08（B19 测试盲区最后一项）：单 slice CT（薄层或损坏导出只剩一张，
shape 1×N×N）或异常 spacing 进入训练链路时，会穿过现有检查直抵 nnU-Net
3d_fullres 预处理，在那里以晦涩深层报错失败或产出退化 patch——用户白
干活且无法自行定位。铁律 2（标注者视角）+ 铁律 6（无验证不算完成）。

### 2. 调查与验证

- 三条链路逐一核实（详见账本 TB-08 调查结论）：导入侧已有完整防护
  （宽松分组重试 + 人话拒存）；训练侧在 `build_training_data_profile`
  存在真实缺口（维度>0 检查放行 1×N×N）；预测侧同型检查但在冻结验收
  范围外；nnInteractive 交互 helper 纯几何切片安全。
- 验收标准对照："shape=(1,20,20) 的 case 过 discover+convert+训练请求
  规范化，断言要么被显式拒绝并带人话报错"→ 在全部四条训练路径
  （nnU-Net/FlexiCT × 本地/远程 prepared）共用的 profile choke point
  显式拒绝 + 人话报错，混合 rows 测试点名单 slice case 且证明正常 case
  不被误伤（与 R26/TB-04 同址同型的验收先例一致）；"异常 spacing"→
  零 spacing affine 测试锁住既有拒继行为（此前无测试）。

### 3. 实施

- `tools/nnunet_pipeline.py` `build_training_data_profile`：既有
  "valid 3D" 检查后新增 6 行——任一维度 ≤1 即 RuntimeError（点名
  case、给出形状与两条下一步动作）。共享门覆盖 nnU-Net + FlexiCT
  两个家族、本地 + 远程 prepared 两条执行路径，零新增调用点。
- `tools/test_nnunet_integration.py`：`_profile_case` helper 加
  shape/affine 参数（既有调用点默认值不变）；新测试 2 个（单 slice
  混合 rows 拒绝 + 零 spacing 锁行为；后者奇异 affine 经 set_sform
  存储，因 nibabel 拒收构造器传入的奇异矩阵）。
- 架构审查：独立架构师代理裁定**通过**（无必改项），两条建议均采纳
  （条件式去冗余 int()、报错措辞配置中立化——FlexiCT 2d trainer 也过
  此门）。审查确认：`<=1` 阈值无误伤（真实薄层 3-5 slice 通过；仅
  单 slice/singleton 维度被拒，恰好 TB-08 范围）；2d trainer 不需按
  配置豁免（profile 声明 spatial_dimensions=3，z=1 会污染分布界）；
  远程 prepared 分支（nnunet_pipeline.py:1722）与 FlexiCT（:767）调用
  点同被覆盖，无绕过路径。

### 4. 门禁结果

- test_nnunet_integration 86/86（含新测试 2）。
- fast 门禁 26/26，工件 `.mimics_runtime/regression/20260926T220314_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：检查在 worker 进程的 profile 构建步骤内，且是纯 numpy
   形状检查（微秒级），无新 I/O。
2. 标注者视角：报错点名病例、说明原因（单 slice 不是有效训练体积）、
   给两条下一步动作（重新导出完整序列 / 从训练选择中移除）。
3. 不做表面修改：根因是训练入口缺几何校验，choke point 补门而非在
   每个调用点散查。
4. 不留垃圾：零新配置项/文件/依赖/UI 入口；helper 参数默认值保持既有
   调用点不变。
5. 小步提交：单主题（TB-08）独立 commit。
6. 无验证不算完成：86/86 + fast 26/26。
7. 复杂度预算：产品 +11（6 行门 + 报错字符串），测试 +95（2 测试 +
   helper 参数化）。
8. task_lifecycle 规范：fail-closed 异常走训练任务既有失败路径（状态
   文件 + 日志），无新交互。

### 6. 复杂度台账

- +106（产品 +11 / 测试 +95）。新增维护面：0 新文件/配置项/依赖/UI
  入口。B19（TB-01~TB-08 测试盲区排期）本轮整体关闭。

### 7. 架构审查豁免声明

不豁免——本轮含产品改动（新增 fail-closed 门），已执行独立架构师代理
审查（裁定通过，见 §3）。

### 8. 经验与教训

- nibabel 构造器拒收奇异 affine（qform SVD 分解不收敛），测试造异常
  spacing 数据需走 `set_sform`（直接写 srow 字段）——比"改用负 spacing"
  更真实地保留了"零 spacing"这一实际失败形态。
- 预测输入侧 dim==1 仍放行（validate_model_input_compatibility:1411），
  已在账本 TB-08 残余观察登记（不新增待办），防未来重复"发现"。

### 9. R32 候选

- B9-b（prune 钩子防回归测试，无决策依赖）：prune_import_receipts/
  _prune_old_checkpoints/prune_logs/_prune_local_export_jobs 四钩子。
- 余下待办均为需用户决策项（A4、A7、D3、D7、D13、D14、D18）或 P3。

## R32（2026-09-26）：B9-b 四个 retention prune 钩子防回归测试

### 1. 主题与动机

B9(b)（P2 测试资产完整性）：`prune_import_receipts`（create_mcs_batch）/
`_prune_old_checkpoints`（mimics_import）/`prune_logs`（external_window_
launcher）四个 retention 钩子零直接测试——DELIVERY_REPORT_2 声称的
32 项 prune 测试现仅 26 项且零 prune，证据链失真。铁律 6（无验证不算
完成）：retention 钩子是 R12（B2 无界增长）防线的一部分，损坏（误删
或漏删）无任何门禁能发现。其中 import receipts 带患者文件路径，误删/
漏删是隐私问题而非仅磁盘卫生。

### 2. 验收标准对照

"四个 prune 钩子各配伪造 mtime 测试"→ 四个测试逐一对照：
- `prune_import_receipts`：31 天过期 receipt 删、新 receipt 保、非
  receipt 后缀文件（即使更旧）不碰；
- `_prune_old_checkpoints`：15 天过期 breadcrumb 删、新鲜保、无关节点
  debug 文件不碰；
- `prune_logs`：4 个窗口日志按 mtime 只留最新 2、.txt 非日志不碰；
- `_prune_local_export_jobs`：14 天内保、超龄无活进程删、**status 文件
  报告活 PID（os.getpid()，真实存活）的超龄目录拒删**（数据安全路径）、
  非目录文件不碰。
"矩阵 fast 绿"→ 本轮改 test_all.py 自身，按 R28/R30 先例升级为 full
门禁（test_all 461/461 含新测试 4）。

### 3. 实施

- `tools/test_all.py` TestLifecycleAndRetention 末尾新增 4 个测试，
  全部真实文件 + `os.utime` 伪造 mtime（与 R12 的 TestStaleGuardSweep/
  TestImportQueuePrune 同模式），零产品改动。export prune 测试经
  `_project_root` 指向 tmp 目录隔离（try/finally 恢复）。
- 唯一既有相关测试（test_internal_batch_export_locks_the_actual_label_
  destination）只是 mock 掉 `_prune_local_export_jobs`，未测其行为——
  本轮补齐后该 mock 保持不变（其测试对象是锁路径，不是 prune）。

### 4. 门禁结果

- TestLifecycleAndRetention 33/33（含新 4）。
- full 门禁 28/28，工件 `.mimics_runtime/regression/20260926T225405_full.json`
  （test_all pass，套件 461 项 = R30 基线 457 + 新 4）。

### 5. 铁律自检

1. 不阻塞 GUI：零产品改动。
2. 标注者视角：防止"撤销回执被误删"（prune 误删 receipt 直接破坏
   Undo Last Import）与"磁盘再次无界增长"（prune 损坏无人发现）。
3. 不做表面修改：钉住的是四个钩子的删除边界语义（删什么、留什么、
   活任务保护），不是覆盖率数字。
4. 不留垃圾：无新 helper、无重复 setup 模式（复用类级 setUp/tearDown）。
5. 小步提交：单主题独立 commit。
6. 无验证不算完成：33/33 + full 28/28。
7. 复杂度预算：测试 +131，产品 0。
8. task_lifecycle：不涉及交互。

### 6. 复杂度台账

- +131（产品 0 / 测试 +131）。新增维护面：0 新文件/配置项/依赖/UI
  入口/产品改动。B9 仅余 (a)（孤儿套件处置，D7 用户决策）。

### 7. 架构审查豁免声明

豁免——纯测试轮（零产品改动），与 R27/R28/R30 同型。测试设计遵循
失败路径优先理念：活 PID 拒删（数据安全）与删除边界（误删非目标
文件）均覆盖。

### 8. 经验与教训

- `_local_export_job_is_active` 的活任务判定读 status.json 的 pid +
  process_exists——测试用 os.getpid() 注入真实存活 PID，比 mock
  process_exists 更端到端（若判定链断裂，mock 版会假绿）。
- DELIVERY_REPORT_2 的"32 项 prune 测试"失真（实际 26 项且零 prune）
  再次印证：交付报告的测试计数不可信，以套件实际内容为准。

### 9. R33 候选

- 无 P1/P2 可直接执行项剩余：待办仅余需用户决策项（A4、A7、D3、D7、
  D13、D14、D18）与 P3 代码质量项（C1-C8，可合并为死代码/重复收敛
  主题轮）。下一轮建议：C1 死代码清理主题轮（纯删除，符合复杂度
  预算选型最优先区间）或停轮升级用户决策。

## R33（2026-09-27）：D14 共享 GPU 忙时硬阻断训练启动

### 1. 主题与动机

用户 2026-09-27 拍板 D14："要"——共享服务器 GPU 忙时从软提示改为硬
阻断。此前（R29/TB-06）Test Connection 仅软提示，训练启动时零检查，
忙时启动直接挤占他人任务，违反红线"先查后用"。铁律：外部资源红线
（共享服务器不得挤占）。

### 2. 架构审查（产品改动，必经）

独立架构师代理裁定**修改后通过**，10 项必改全部落实：
- R1 nvidia-smi 查询提取为共享常量 GPU_QUERY_COMMAND（软诊断与硬门
  永不漂移；保持 "memory.used,utilization.gpu" 连续字面量，TB-06
  契约测试继续通过）✔
- R2 assert_gpus_not_busy 复用 gpu_busy_warning 判定（零重复实现）；
  报错点名 GPU + 指引改策略（Manage Servers → GPU busy policy）；
  查询失败 fail-closed 抛 RemoteComputeError（静默放行=红线开洞）；
  off 模式零远程查询 ✔
- R3 早期检查插入点：ensure_directory 之后、缓存维护/任何上传之前；
  五种 kind 统一适用；失败路径 container_may_exist=False → 处理器
  正常置 failed，无 remote_state_unknown ✔
- R4 **启动前再查一次**（审查裁定为必改而非可选：数据集上传可达
  数小时，挤占时刻是启动那一刻；重传有远程缓存兜底近乎零成本），
  置于重连循环之前使重连不重复触发 ✔
- R5 warn 模式必须写状态文件（日志用户看不到）：_append_log +
  _status_update(gpu_busy_warning=...) 双通道 ✔
- R6 normalize_profile 镜像 remote_weights_verify 三态模式
  （block 默认，非法值回退 block）✔
- R7 UI 四处接线（combo 带默认标注 + 表单行 + _load_selected +
  _clear_form + _profile_values）✔
- R8 CONFIG_REFERENCE 表行 ✔
- R9 TB-06 测试 docstring 更新（原声明"硬阻断未实现"已失效）✔
- R10 测试含 off 模式零查询断言、报错双要素断言、启动前重查源序
  断言 ✔

审查同时裁定：Test Connection 保持纯软诊断（连通性工具不该因策略
  拒绝），但 block 模式下追加"Training will not start while the GPU
  is this busy"预告（采纳）；五种 kind 统一阻断（推理也占共享 GPU，
  auto 请求全部卡）。

### 3. 实施

- `tools/remote_compute.py`：GPU_QUERY_COMMAND 常量 + assert_gpus_
  not_busy（block/warn/off 三态）+ normalize_profile gpu_busy_policy
  键 + test_connection 复用共享查询并加 block 预告。
- `tools/remote_training_controller.py`：run() 两处门（上传前 +
  启动前），warn 双通道落盘。
- `tools/remote_compute_ui.py`：GPU busy policy 组合框四处接线。
- `CONFIG_REFERENCE.md`：gpu_busy_policy 行。
- `tools/test_remote_training.py`：新测试 4（block 报错双要素 + 选中
  卡空闲不误伤 / warn+off+查询失败三态 / 源序两时刻契约 / 策略键
  规范化）+ B12 往返测试扩展 gpu_busy_policy + TB-06 docstring 与
  契约更新。

### 4. 门禁结果

- test_remote_training 100/100（含新 4 + 扩展 1）。
- gui_smoke 18/18（UI 改动覆盖）。
- fast 门禁 26/26，工件 `20260927T020155_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：全部在 controller 进程，一次 SSH 往返。
2. 标注者视角：被挡时秒级失败（而非数小时上传后冲突），报错点名
   哪块卡、为什么挡、怎么改；Test Connection 有预告不意外。
3. 不做表面修改：根因是启动时零检查；门复用 R29 判定零重复。
4. 不留垃圾：新配置键 1（架构审查裁定的最小逃生舱，B12 先例），
  UI/文档同步。
5. 小步提交：单主题。
6. 无验证不算完成：100/100 + 18/18 + fast 26/26。
7. 复杂度预算：产品 +99（remote_compute +65 / controller +15 / UI +18 /
  CONFIG_REFERENCE +1）/ 测试 +140（`git diff --stat`：5 文件
  +239/-8）；新增维护面：
  新配置键 1 + 新 UI 行 1（皆有先例同型）。
8. task_lifecycle：fail-closed 异常走任务失败路径，状态文件可见。

### 6. 复杂度台账

- 见台账 R33 行：+239 净 +231（产品 +99 / 测试 +140），原因 = D14 用户
  立项硬阻断 + 防回归测试。

### 7. 架构审查声明

已执行（产品改动必经），裁定修改后通过，10 项必改全落实（见 §2）。

### 8. 经验与教训

- 源序契约测试的标记必须选 run() 内独有者：`phase="uploading_..."`
  字符串首次出现在 _upload_progress 定义里而非调用点，第一版测试
  假绿方向错误（实际是断言失败暴露），改用 `while pending_dataset_
  parts:`（仅 run() 内存在）。
- "检查一次"对红线不够：上传耗时数小时意味着检查与使用之间隔一个
  窗口，架构师裁定启动前重查是必改而非过度设计——红线约束的是
  "使用"那一刻。

### 9. R34 计划

孤儿测试套件删除（用户已拍板，先核查功能覆盖）。

## R34（2026-09-27）：孤儿测试套件核查与注册（B9-a / D18，用户决策）

### 1. 主题与动机

用户 2026-09-27 拍板"删掉，但先核查有关功能是否有其他测试覆盖"。
核查（不改代码，纯读）推翻了"废弃可删"的预设：

- `integrations/nninteractive-finetune/tests/`（47 项 pytest，实测 47 passed
  in ~31s）：测的是 `nninteractive_finetune` 引擎本体——config 校验、
  trainer 循环（含 initial-mask AUC 评估）、数据加载/几何、prompt 采样、
  指标评估、`exclusive_job_lock` 任务锁。该引擎**每次自定义模型训练都在用**
  （`tools/nninteractive_finetune_pipeline.py:76` 子进程启动其 `__main__`）。
  其他已注册套件（test_all / nnint_deep / nnint_functional /
  nninteractive_task_integration）全部只测引擎**外围**（请求生成、状态、
  注册、以伪造模型目录模拟训练完成）；test_training_convergence 只覆盖
  nnU-Net + FlexiCT，不含 nnInteractive。→ 引擎本体测试覆盖唯一来源。
- `integrations/nnunet_segmentation_workflow/test_annotation_version.py`
  （35 项 unittest，实测 0.3s OK）：测 Action1 标注版本回退链
  （`_version_dirs` / `_resolve_organ_path` / `_resolve_subject_version`）
  与框架 `_normalize_annotation_version`。托管 Mimics 流程
  （nnunet_pipeline / stage_worker）不用 Action1/版本链——它是独立 CLI
  （AutoSegmentationFramework convert 阶段）专属功能，同样无其他覆盖。

两套都能跑通、都无替代覆盖 → 用户改选**注册进矩阵**（矩阵已有
flexict_pkg 集成包套件先例）。

### 2. 改动内容

- `tools/run_regression_matrix.py` SUITES 新增两行（fast + full）：
  - `nninteractive_finetune_pkg`：`-m pytest -q integrations/nninteractive-finetune/tests/`（实测 32.2s）
  - `nnunet_annotation_version`：`integrations/nnunet_segmentation_workflow/test_annotation_version.py`（实测 1.0s）
- 无产品代码改动；未删除任何测试。

### 3. 验证

- 两套件从 ROOT（矩阵的 cwd）分别单独跑通：47 passed / 35 OK。
- `--only` 两套件经矩阵运行 2/2 通过。
- fast 门禁（工件见下）。

### 4. 门禁结果

- fast 门禁 28/28（新增 2 套件计入），工件 `20260927T121139_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：纯测试注册，零产品改动。
2. 标注者视角：nnInteractive 微调引擎（自定义模型核心路径）从此有门禁
   防回归；此前任何引擎改动都无测试保护。
3. 不做表面修改：先核查后行动，核查结论改变了处置方式（删→注册）。
4. 不留垃圾：零删除零新增文件；两行注册让 82 项既有测试变活资产。
5. 小步提交：单主题。
6. 无验证不算完成：47+35 单独跑通 + 矩阵 2/2 + fast 28/28。
7. 复杂度预算：产品 0 行改动，矩阵 +2 行；新增维护面 0（无新文件/
   配置项/依赖/UI 入口）。
8. task_lifecycle：不适用（纯测试）。

### 6. 复杂度台账

- 见台账 R34 行：+2（产品 0 / 测试矩阵 +2），原因 = B9-a 孤儿套件注册。

### 7. 架构审查声明

豁免（零产品代码改动，仅测试矩阵注册两行）。

### 8. 经验与教训

- "孤儿"不等于"废弃"：判断处置方式前先回答"它测的东西还有谁在测、
  还有谁在用"。47 项测试保护的正是每次训练都在跑的引擎，删掉等于
  给自定义模型训练路径撤掉全部护栏。
- 矩阵注册是低成本高杠杆动作：2 行换 82 项测试进门禁。

### 9. R35 计划

A4 Fix Affine 逻辑核查（用户 2026-09-27 指示"看一下逻辑是否有问题"）。

commit：`460b9b1`。

## R35（2026-09-27）：A4 "Relink the source image metadata" 文案指向真实路径

### 1. 主题与动机

A4【P1】预测前置校验失败抛 "Relink the source image metadata"，但全库
不存在"relink"动作。用户 2026-09-27 指示"看一下（Fix Affine）逻辑是否有
问题"。本轮先完成逻辑核查，据核查结论改三处文案为指向真实存在的路径。

### 2. 逻辑核查结论（走查记录，满足 A4 验收标准的"人工验收"部分）

- **Relink 报错的真实触发条件**（nnunet_mimics._prediction_context /
  flexict_mimics._prediction_context 调 mimics_mask_apply.
  _resolve_prediction_context 返回空）：活动项目解析不出 case_id，或
  case_id 有但所有解析路径（dataset manifest / 几何元数据 source_image_
  path / 已知数据集根下 case 目录扫描）都找不到源图。
- **Fix Affine（99_Admin/04）的真实前置与行为**：读活动图像元数据
  `source_image_path`，经桥接子进程取磁盘 NIfTI 的真 nibabel RAS affine，
  重写 `source_voxel_to_ras_matrix`；形状不符拒修并建议重新导入；已一致
  则不动；写后校验回读并保存项目；worker 线程 + win32 timer 轮询，不阻塞
  GUI；超时杀进程不改元数据；Stop 注册进 mimics_stop_background。
  **其逻辑本身无问题**。
- **关键结论：Fix Affine 不能修复 Relink 报错，两者互斥。**
  _resolve_prediction_context 的最后回退用的正是 Fix Affine 的前置
  （source_image_path 元数据 + 磁盘存在）——Fix Affine 能跑的场合
  Relink 不会报错；Relink 报错的场合 Fix Affine 自己也会因"无可用的
  source_image_path"退出。两错误是不同故障：Relink=项目未链接到源图
  （回到导入流程）；第三处（nnunet_pipeline.validate_materialized_
  source_geometry 的 "no longer matches the geometry"，旧版桥 LPS
  affine 残留）才是 Fix Affine 真正能修复的场景。

### 3. 改动内容

- `runtime_py35/nnunet_mimics.py` / `flexict_mimics.py`：Relink 文案改为
  指回导入入口（01_Data > 01_Import_Dataset / 02_Import_Single_Case），
  项目被移走则建议重新导入。
- `tools/nnunet_pipeline.py` validate_materialized_source_geometry：stale
  geometry 文案点名 99_Admin > 04_Fix_Source_Affine_Metadata，源文件本身
  变了则建议重新导入（与 Fix Affine 自身的形状不符建议一致）。
- `tools/test_all.py` 新测试 `test_prediction_context_errors_point_to_
  real_repair_paths`（TestSourceImagePathEquivalence）：源契约钉住三处
  ——两 runtime 模块不再出现旧 Relink 措辞且含导入入口指引；pipeline
  侧 04_Fix_Source_Affine_Metadata 标识不得被换行拆开（菜单引用不可
  静默断裂），并实测渲染报错信息含修复入口与 re-import 逃生口。

### 4. 门禁结果

- 新测试单跑 OK；TestSourceImagePathEquivalence 24/24。
- nnunet_integration 86/86、flexict_integration 89/89。
- smoke 8/8（runtime_py35 改动，`20260927T122520_smoke.json`）。
- fast 28/28，工件 `20260927T123006_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：仅报错文案，无流程改动。
2. 标注者视角：三处报错从"指向不存在的动作"变为"指回真实入口"；
   可修复的场景（stale affine）与不可修复的场景（未链接）各给各的路。
3. 不做表面修改：先核查逻辑再动文案，文案据核查结论分岔。
4. 不留垃圾：零新增配置/入口；旧措辞彻底清除（测试钉住不复活）。
5. 小步提交：单主题。
6. 无验证不算完成：单测 + 两集成套件 + smoke + fast。
7. 复杂度预算：产品净 +5 行（三处文案重写）；测试 +46 行；新增维护面 0。
8. task_lifecycle：报错文案遵守规范（说明发生了什么 + 下一步动作）。

### 6. 复杂度台账

- 见台账 R35 行：+51 净 +48（产品 +5 / 测试 +46），原因 = A4 文案修复 + 防回归测试。

### 7. 架构审查声明

豁免（纯报错文案改写 + 源契约测试，无流程/状态变更；逻辑核查结论
即本报告 §2，实现严格按结论分岔）。

### 8. 经验与教训

- 修文案前先回答"这个错误到底什么时候触发"：同一个 "Relink" 词
  掩盖了两种不同故障（未链接 vs 几何过期），只有一种可由现有工具修复。
- 源契约测试断言的标识符必须钉"单行存在"：字符串换行拼接可把
  "04_Fix_Source_Affine_Metadata" 拆到两行，菜单引用静默断裂。
- mock 局部 `import nibabel as nib` 要 patch "nibabel.load" 本体，
  patch 模块属性拦不住函数内局部导入。

### 9. R36 计划

C1 死代码清理（用户 2026-09-27 指示"做完再谨慎做C1"，逐项重 grep
确认零引用后删除）。

commit：`cb4a890`。

## R36（2026-09-27）：C1 死代码清理（用户决策项，谨慎执行）

### 1. 主题与动机

C1【P3】死代码清理。用户 2026-09-27 指示"做完再谨慎做C1"。铁律 4
（不留垃圾）的直接执行。本轮原则：**逐项删除前 grep 全库核实零引用**
（def + 调用方 + 测试 + 文档），不确定的宁可不删。

### 2. 核查与删除清单

| 位置 | 删除项 | 核实结论 |
|---|---|---|
| tools/mimics_label_export.py | acquire_gpu_lock_for_job、gpu_lock_enabled、materialize_label_on_source_grid、_path_stat_signature、mcs_label_fingerprint、_cached_case_label、_copy_label_atomic、plan_mcs_label_cache、publish_mcs_label_cache（+未用 hashlib 导入） | 死四件套及其连环死 helper，被 A11 retention 清扫（3f16d0f）与 D14 硬阻断（e6fad56）取代；共 289 行 |
| runtime_py35/mimics_export.py | _active_background_export_lock、_release_background_export_lock | 读侧零引用；写侧 _set_/_clear 活跃且有测试 |
| tools/pipeline_common.py | write_json_best_effort | 零调用方 |
| tools/flexict_pipeline.py | _parse_training_progress、EPOCH_MATCH | docstring 称 "Kept for tests" 但无测试使用 |
| tools/nnunet_pipeline.py | _split_cases | 全被 _split_folds 取代 |
| tools/setup_env.py | _gui_wheels_available | 零调用方 |
| runtime_py35/nninteractive_mimics.py | _remove_state_file | 零调用方；_remove_owned_state_file 是活跃变体 |
| 根目录 | quick_import.py、quick_export.py | 16 行 Toggle Editor 包装，零引用；同名流程在 scripting_library/01_Data/ 有活跃入口（02_Import_Single_Case、08_Quick_Drop_Import、07_Quick_Export_Masks） |

**审计条目两处失实，已核实并纠正**（写入账本）：
- "write_json_best_effort 在 __all__"——错误，在 __all__ 的是
  write_json_atomic；
- "mimics_export 三个 _background_export 函数"——实际只有两个死
  （_active_background_export_lock、_release_background_export_lock）；
  _watch_ 与 _finalize_ 均在活跃调用链（730/2387 行），已保留。

**主动放弃的删除**：nninteractive_bridge.py 中的 _gpu_lock_enabled 等
"重复"——模块 docstring 明确要求单文件可独立部署，属豁免理由（记入 C3
待收敛项时引用）。

### 3. 改动内容

9 个文件，净 **-396 / +2** 行。无任何行为改动（全部为删除 + 模块
docstring 与删除项匹配）。

### 4. 门禁结果

- 全部改动模块 import / py_compile 通过（mimics_export 因模块级
  `import mimics` 无法在 Mimics 外导入，用 py_compile + 残留引用扫描
  替代）。
- smoke 8/8（runtime_py35 改动，`20260927T125214_smoke.json`）。
- fast 28/28，工件 `20260927T125734_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：纯删除死代码，无行为改动。
2. 标注者视角：无 UI 变化；根目录两个死入口的删除对用户不可见
   （同名功能入口在 scripting_library）。
3. 不做表面修改：每个删除项有 grep 证据；审计失实项如实纠正而非
   照单执行。
4. 不留垃圾：本轮本身就是铁律 4 的执行；.mimics_runtime 临时
   commit_msg 已删。
5. 小步提交：单主题单 commit（`ae64029`）。
6. 无验证不算完成：import/py_compile + smoke + fast。
7. 复杂度预算：**净负 396 行**，本轮为复杂度预算的减法示范；新增
   维护面 0。
8. task_lifecycle：不涉及（无新功能交互）。

### 6. 复杂度台账

- 见台账 R36 行：-396 净 -394（-428 删 / +2 改），原因 = C1 死代码清理。

### 7. 架构审查声明

豁免（纯死代码删除，无流程/状态/接口变更；所有删除项经全库引用
核实，前两轮用户决策项 D14/B9/A4 均已落地后才执行）。

### 8. 经验与教训

- 审计清单不可照单执行：C1 两处失实（__all__ 归属、_background_
  export 计数），若照删会破坏活跃的锁协议读侧命名认知。删除前
  必须重新 grep，旧审计行号与结论会过期。
- 根目录包装脚本 vs scripting_library 入口的判别标准：grep 引用 +
  文档引用 + 是否有同名流程的活跃入口，三者都空才删。
- "Kept for tests" 的 docstring 声明也要验证——_parse_training_
  progress 的声明是假的。

## R37（2026-09-27）：A7 Mask Identifier 模态点击崩溃风险提示

### 1. 主题与动机

A7【P1】最后一个未关闭的 P1（A5/A6 剩余部分均为 D2/D3 需用户决策项，
自主迭代不得代决）。验收标准：评估缩小窗口的可行方案；若无可行代码方案，
把风险提示按 task_lifecycle 规范改造并在文档登记为平台限制。

### 2. 方案评估（走查记录）

- **窗口最小化设计已存在**（2026-07 commit 9742ea6，早于审计条目）：
  非模态预检对话框（期间可自由调整视图/切工具）+ 点击结果用非模态
  question_box 展示，仅在用户显式选择 Click Again 后的短暂窗口重入
  indicate_coordinate。
- **无非模态替代 API**：Mimics 21 公开点采集 API（indicate_coordinate、
  measure.indicate_*、analyze.indicate_*）均为模态阻塞调用，脚本层
  无法进一步收缩危险窗口。
- **结论**：走验收标准"无可行代码方案"分支——改造风险提示 + 文档
  登记平台限制。

### 3. 改动内容

- `runtime_py35/mask_identifier.py`：预检文案从只说时机（"only while
  this dialog is shown"）改为同时说后果（"can crash Mimics and lose
  unsaved work"）——task_lifecycle 规范要求数据风险必须写明后果，
  习惯性滚轮缩放时没有后果的软提示会被无视。
- `tools/fake_mimics_flow_test.py`：既有 mask identifier flow 测试钉住
  后果措辞（crash + unsaved work）与预检非模态标志，防静默回退成提示。
- `docs/MIMICS_PROJECT_ARCHITECTURE_CN.md` §8.1：平台限制登记（API 限制
  的性质、脚本层已做的最小化、为何无法根治）。

### 4. 门禁结果

- flow 套件 15/15（含加固后的 mask identifier 测试）。
- smoke 8/8（runtime_py35 改动，`20260927T130448_smoke.json`）。
- fast 28/28，工件 `20260927T130941_fast.json`。

### 5. 铁律自检

1. 不阻塞 GUI：预检对话框本就非模态（ui_blocking=False），测试钉住。
2. 标注者视角：提示从开发者式的时机指令变为标注者能理解的风险陈述。
3. 不做表面修改：先评估方案可行性再动文案，文案据评估结论改。
4. 不留垃圾：零新增配置/入口。
5. 小步提交：单主题单 commit（`f40a77e`）。
6. 无验证不算完成：flow + smoke + fast。
7. 复杂度预算：产品 +11（文案 + 注释）；测试 +21；文档 +13（平台限制
   登记是验收标准明文要求）；新增维护面 0。
8. task_lifecycle：数据风险写明后果——正是本条的规范依据。

### 6. 复杂度台账

- 见台账 R37 行：+34 净 +37 内（产品 +11 / 测试 +21 / 文档登记 +13）。

### 7. 架构审查声明

豁免（单处用户文案补后果 + 既有测试加固 + 文档登记，无流程/状态变更；
方案评估走查记录即本报告 §2）。

### 8. 经验与教训

- 审计条目可能落后于代码现状：A7 描述的"预检对话框只是小字免责"经
  核实，窗口最小化设计 2026-07 已存在——动手前先核对条目描述的
  现象是否仍在。
- 风险提示要写后果：只写"何时能做什么"的时机指令在用户习惯性操作
  （滚轮缩放）面前基本无效。

## R38（2026-09-27）：C2 断裂引用清理

### 1. 主题与动机

C2【P3】断裂引用：mimics_import.py fallback 指向不存在的 `adapters/mimics/...`；
docs/superpowers 历史 spec 引用已删 fewshot 6 文件（验收标准：归档标注）。
紧接 R36/R37，账本 P0/P1/P2 全部关闭后进入 P3 区第一项。

### 2. 核查与改动

- **adapters 死候选 ×3**（mimics_import.py `_bridge_script` 与 create_mcs_batch
  候选清单、mimics_export.py `_bridge_script`）：均为候选列表最后一项，真实
  路径在候选 1/2 必命中（文件就在本仓库布局内），adapters 候选永不命中、
  纯属对布局的误导性描述。全部删除；全库 grep 确认代码与文档零残留。
- **fewshot 断裂引用**（hardcase spec 14 处 + finetuning spec 依赖行 + 调研
  文档 1 处）：按验收标准"归档标注"——不重写历史文档正文（与 b632d70 删除
  时"历史文档保持原样"的决策一致），在文档头部/指向行加标注：spec 从未
  实施、few-shot 模块已于 2026-09-23 删除（commit b632d70）、实施规划文档
  已不存在、现状唯一依据为 MIMICS_PROJECT_ARCHITECTURE_CN.md。

### 3. 门禁结果

- py_compile 两 runtime 模块通过。
- smoke 8/8（runtime_py35 改动，`20260927T131551_smoke.json`）。
- fast 28/28，工件 `20260927T132104_fast.json`。

### 4. 铁律自检

1. 不阻塞 GUI：候选清单缩短不影响路径解析结果（真实路径命中次序不变）。
2. 标注者视角：无直接影响；防后续维护者按错误候选复制布局。
3. 不做表面修改：删除断裂引用 + 历史文档标注均有明确动机。
4. 不留垃圾：断裂引用正是铁律 4 的对象。
5. 小步提交：单主题单 commit（`97c1f4e`）。
6. 无验证不算完成：py_compile + smoke + fast。
7. 复杂度预算：代码 -3 行；文档标注 +10 行；新增维护面 0。
8. task_lifecycle：不涉及。

### 5. 复杂度台账

- 见台账 R38 行：+5 净 -3 内（代码 -3 / 文档标注 +10）。

### 6. 架构审查声明

豁免（死路径删除 + 历史文档标注，无流程/状态/接口变更）。

### 7. 经验与教训

- 候选清单式的路径解析里，永不命中的候选不是"兼容性保险"而是布局误导——
  读者会以为存在过 adapters/ 布局并据此复制。
- 历史文档的断裂引用用"标注"而非"重写"处理：历史决策链
  （为何删、何时删、删成了什么）比修复后的整洁更有价值。

### R38 补充：C4 .mimics_runtime 迭代垃圾清理（同轮并案）

C4 与 C2 同轮清理（验收标准本就写明"与 A2/B2 清理合并执行"，属操作性
收尾）。删除：15 个 tmp_*.py、6 个 commit_msg*.txt、box_check*.txt、
test_all 计划/日志、gate_p*.log、flexict 测试日志、setup_env.log、
convergence 调试日志、debug_out/ 内非 checkpoint 残留（full_suite_*.txt、
health_preview*.png）。核实 debug_out/ 剩余 326 个文件全部为
mimics_import_checkpoint_* 崩溃诊断面包屑，受 14 天 prune 钩子管辖
（mimics_import._prune_old_checkpoints，R32 已有防回归测试），非垃圾。
纯操作性清理：无代码改动、无 commit；checkpoint prune 测试回归通过
（1 passed）。C4 关闭。

## R39（2026-09-27）：C5 弱断言测试重写 + C6 待核实 + C8 数据质量模块补覆盖

### 1. 主题与动机

P3 区三项合并一轮（同属"测试资产质量"主题）：C5 两个名不副实的测试、
C6 两项待核实决策、C8 零测试数据写入模块补覆盖。

### 2. C5 弱断言测试重写（commit `6145eb2`）

- test_cancelled_job_cleans_partial_model 原只断言"缺路径时不崩溃"，
  从未验证任何删除。重写为：partial model 目录真被删 + 删除入报 +
  **workspace 外路径被拒**（"outside job workspace" 安全契约——旧测试
  完全未守护；重写首跑即失败暴露该契约值得钉住）。
- test_prepare_manifest_with_no_cases_raises 原为"内联复制管线过滤逻辑
  再断言副本结果"——生产 guard 从未被执行。重写为真调
  pipeline._run_label_export 空病例列表，断言 RuntimeError 文案 +
  guard 前无 staging 残留。
- test_mimics_nnint_deep 75/75、fast 28/28（20260927T133501_fast.json）。

### 3. C6 待核实结论（保留现状，不删不改默认）

- **defer_source_alignment_to_worker=False 逃生门**：键不在
  CONFIG_REFERENCE、不在默认 config，用户可发现性为零，实际无人用；
  但它是 worker 对齐路径（GUI 线程安全修复）的配置级回滚保险，删除
  收益（~2 行分支）远小于回滚能力损失，且有专门测试钉住行为
  （test_official_model_sync_alignment_still_available_when_configured）。
  按铁律 9 风险不对称原则**保留**。
- **import 启动清扫 opt-in**（MIMICS_IMPORT_STARTUP_CLEANUP）：docstring
  明示"某些 Mimics 构建上 Win32 锁探测/进程检查可致宿主不稳定"；清扫
  本体已由健康面板周期与 guard sweep 覆盖（R12）。改默认开启违背铁律 1
  保守取向，**保留 opt-in**。

### 4. C8 verify_medical_geometry.py 补覆盖（本轮主 commit）

94 行数据质量闸门模块（导出 NIfTI 与原图 grid 对齐校验），错了会放走
坏数据，零直接测试。新增 TestVerifyMedicalGeometry 6 个测试（进
test_all.py，full 门禁覆盖）：
- 匹配 grid → exit 0；affine 错位 / shape 错位 → exit 2（fail-closed）；
- 源几何不可读 → RuntimeError；
- NRRD mask 的 LPS→RAS 转换路径（Identity LPS 方向 → RAS affine 为
  diag(-1,-1,1)，不得与 identity 源几何误判匹配——钉住"必须按 RAS
  比较"的关键语义）；
- reference mask Dice 路径。

### 5. 全量门禁暴露的 R34 遗留 bug 及修复

首次 full 门禁 test_all 失败：
test_regression_matrix_runner_suite_registry 断言每条 SUITES 命令含
存在的 .py 目标——R34 注册的 nninteractive_finetune_pkg 用的是目录
路径（`integrations/nninteractive-finetune/tests/`），违反契约。
**R34 当时只跑了 fast（不含 test_all），漏检**。修复：命令展开为
sorted glob 的 6 个 test_*.py 显式文件列表（pytest 多路径语义不变，
47/47 通过；契约测试恢复绿）。

### 6. 门禁结果

- 重写测试单跑 2/2；test_mimics_nnint_deep 75/75。
- 新测试 6/6；矩阵契约测试 1/1；finetune pkg 47/47（新命令形式）。
- fast 28/28（`20260927T133501_fast.json`，C5 提交时点）。
- full 首跑 29/30（20260927T141941_full.json，test_all 失败，见 §5）；
  修复后 full 重跑 **30/30**（`20260927T155018_full.json`）。

### 7. 铁律自检

1. 不阻塞 GUI：纯测试 + 矩阵命令形式，无产品行为改动。
2. 标注者视角：数据质量闸门（C8）从此有防回归——坏 grid 不再可能
   静默放行。
3. 不做表面修改：C5 重写补的是真删除/真 guard 断言；C8 覆盖失败
   路径优先（fail-closed ×2 + 不可读源 + LPS 转换陷阱）。
4. 不留垃圾：无新增死代码。
5. 小步提交：C5 单独 commit（`6145eb2`）；C8+矩阵修复一并提交。
6. 无验证不算完成：full 门禁（含 test_all 468 项）。
7. 复杂度预算：测试 +53（C5）+~110（C8）；产品 0；矩阵命令为数据
   驱动展开（净 +2 行）。
8. task_lifecycle：不涉及。

### 8. 架构审查声明

豁免（纯测试补覆盖 + 测试断言重写 + 矩阵命令形式修复，无产品代码
改动）。

### 9. 经验与教训

- **改 SUITES 必须跑 full**：test_all 里有矩阵注册表的契约测试，
  fast 不含 test_all 时新增套件命令形式不会被校验——R34 的漏检
  即源于此。已在本轮把该依赖关系写进教训。
- 名不副实的测试比没测试更危险：test_prepare_manifest_with_no_
  cases_raises 十行代码全在测自己的副本，生产 guard 从未执行。
- C6 型"待核实"条目核实后要写明保留理由入账，否则下一轮还会被
  重新怀疑一遍。


## R40（2026-09-27）：C3 重复实现收敛——1 组真收敛，4 组论证豁免

### 1. 主题与动机

账本 C3：safe_identifier ×2、_parse_axes/_parse_flips ×2、log 轮转同构、
runtime_py35 同名 helper 群（_mimics_log ×9）。重复实现 = 同一 bug 修 N 遍
的维护面，但假收敛（合并语义不同的"相似"函数）比重复更危险——本轮逐项
差分核实后再决定。

### 2. 核实方法

- 对每一组候选做**函数体逐字节比对 + 差分测试**（构造边界输入跑两份实现
  对比输出），而非仅看名字。
- 查调用方层级：runtime_py35（内嵌 py3.5）与 tools/（py3.13）互不 import
  是收敛的硬边界。

### 3. 收敛（唯一一组）

**runtime_py35 四份 log 轮转**（create_mcs_batch、mimics_export、
mimics_import、nninteractive_mimics）：函数体逐字节相同，仅默认常量不同
（nninteractive 10MB，其余 5MB；backups 均 3），且四文件都已 import
runtime_common。收敛为 `runtime_common.rotate_log_file(path, max_bytes,
backups)`，原函数变一行委托（保留各模块默认常量）。净 -72 行重复体。

**tools/mimics_label_export.rotate_log**（pathlib 同构版）收敛为
`pipeline_common.rotate_log`，同样一行委托。pipeline_common 本就是
2026-09 重构指定的外部管线共享原语之家。

新增 2 测试进 test_all.py（TestPipelineCommon）：轮转链 shift、最旧
.3 被删、backups=0 只删主日志、缺失路径不抛；runtime_common 版同语义
钉一份（py3.5 侧）。

### 4. 豁免（4 组，理由入账本）

- **safe_identifier ×2**：差分实测 `a--b__c`/中文/超长串输出分叉
  （`_+` 坍缩、96 截断、Unicode 保留），语义固化在磁盘路径中。
- **_parse_axes/_parse_flips ×2**：py3.5/py3.13 层级隔离，跨层收敛
  得不偿失。
- **_mimics_log ×9**：回退行为实质分化（吞掉/返回 bool/print/前缀/
  仅 message 签名/mimics_import 的环境开关 opt-in——后者是防 Mimics
  宿主不稳的保护，假收敛会引入崩溃风险）。
- **轮转其余 3 份**：nninteractive_bridge（单文件独立部署）、setup_env
  （bootstrap 绑定 LOG_FILE）、remote_worker（流式句柄非同构）。

### 5. 门禁结果

smoke 8/8（`20260927T155956_smoke.json`）、fast 28/28
（`20260927T160614_fast.json`）、full 30/30
（`20260927T164913_full.json`）。commit `af3d201`。

### 6. 铁律自检

1. 不阻塞 GUI：轮转在日志写入路径上且保持 best-effort 吞异常语义。
2. 标注者视角：无行为变化（纯内部收敛）。
3. 不做表面修改：唯一收敛组是逐字节重复；其余论证豁免而非为收敛而收敛。
4. 不留垃圾：删 5 份重复函数体，无新增死代码。
5. 小步提交：本轮一个主题一个 commit。
6. 无验证不算完成：smoke + fast + full（test_all 变更）三档。
7. 复杂度预算：净变化约 -70 产品行 / +60 测试行。
8. task_lifecycle：不涉及。

### 7. 架构审查声明

涉及产品代码（runtime_common、pipeline_common、4 个 runtime_py35 模块、
mimics_label_export）——需架构审查 agent 复核（见下）。

### 8. 经验与教训

- "同名函数"≠"同实现"：_mimics_log 九份同名，比对后没有两份回退行为
  相同；安全做法是差分测试而非名字匹配。
- 收敛的正确姿势：先找"已 import 的共享之家"（runtime_common /
  pipeline_common），原函数降级为一行委托而非删掉调用点——改动面最小
  且调用方零改动。

### 9. 架构审查结果（R40）

架构审查 agent 复核通过（APPROVE）：行为逐行保真（含 backups<=0 分支只删
主日志）、py3.5 兼容（无 f-string/pathlib/typing）、nninteractive 10MB
默认经包装参数保留、无旧实现残留引用、无 inspect.getsource 测试钉住旧
函数体、无循环导入。净 diff +106/-91。审查确认 setup_env._rotate_log
（with_suffix 变体、模块级 LOG_FILE）为已知近重复，已列入豁免记录。

---

## R41（2026-09-28）D1 删除三个重复菜单入口

**用户决策**（2026-09-27）："上述三个可以删除，但是需要明确是否保留的命名合理，
目标一致，用户方便操作"。

**改动**（commit `5ca3623`，8 文件 +28/-167）：
- 删 `03_Review/05_Window_Reset_Full_Range.py`：Choose Preset 对话框内已有同一
  `reset_full_range()` 按钮（window_level_mimics.py:297/309），undo 回退亦提供
  reset；顶层入口纯属重复路径。
- 删 `02_AI/nnUNet/04_Stop_Running_Task.py`：仅转发到状态查看器；查看器
  （03_Show_Status_Models）自带 Stop 按钮，含本地/远程完整停止流。
- 删 `01_Data/03_Export_Masks.py`：与 07_Quick_Export_Masks 调同一
  `_launch_external_export_setup`，Quick 为超集。
- runtime_py35/nnunet_mimics.py：删死代码 `stop_running_task()`（~60 行）、
  BUTTON_STOP/BUTTON_CANCEL 常量、main() 的死 chooser；3 处等待日志改指
  "Show Status and Models, then Stop"。
- tools/nnunet_pipeline.py：resource_wait 取消提示同步（2 处）。
- 测试同步：test_all.py expected_routes 去掉 05 条目；
  fake_mimics_flow_test 入口 fixture 换 04_Window_Undo_Last（undo）。
- 文档同步：MIMICS_PROJECT_ARCHITECTURE_CN.md §4.3/§7.5/§8.5。

**保留入口审计**（用户条件）：
- 03_Review → Identify Mask / Window From Selected Mask / Choose Preset / Undo
  Last / Edit Presets：一事一入口、动词开头、编号连续（04→06 跳号无歧义）。
- nnUNet → Train / Predict / Show Status and Models：与 FlexiCT
  Show Status and Stop 同构（Stop 在状态窗口内），目标一致。
- 01_Data 导出 → 仅 07_Quick_Export_Masks（07 编号保留，是活入口不是残留）。

**门禁**：smoke 8/8（20260927T221402_smoke.json）、fast 27/27
（20260927T224936_fast.json）、full 29/29（20260928T001612_full.json，见下述
training_convergence 勘误）。

**架构审查结论**：无新增维护面；纯删除轮。

---

## R42（2026-09-28）D5 删除死模块（保留活代码）

**用户决策**（2026-09-27）："确保删除的不是会使用的功能就好"。

**审计勘误（先于删除披露）**：账本原记"`nnunet_segmentation_workflow/` 整目录
死代码"**有误**。全仓引用扫描证明以下为活代码，全部保留：
`Action2_PlanAndPreprocess.py` / `Action3_Train.py` / `Action4_Predict.py`
（tools/nnunet_stage_worker.py:64/73/78 直接 import 为 stage backend）、
`ResampleImageAndMask.py`（Action4 内部依赖）、`trainers/`（package_portable +
remote_worker + pipeline）、`ModelMap.toml`（training_setup_ui 默认标签表）。

**改动**（commit `a3371f0`，31 文件 +14/-8653）：
- 删 standalone TOML CLI 子系统：AutoSegmentationFramework.py（控制器，零活
  引用）、SetEnvionmentVariables.py、Action1_ConvertLabeledToTrainData.py +
  Action5_Evaluation.py（仅 CLI 可达的 convert/evaluate 阶段）、其专属 helper
  ImageConvertor / DicomToMhd / MaskOperation、7 个 Config_*.toml、
  ModelMap_Bone/MIv500.toml、gpu0-3.sh / gpu_backup.sh / temp.sh、
  SmallTools.ipynb、ANNOTATION_VERSIONING.md、.claude/settings.local.json
  （仅含已删文件的本地权限条目）。
- 删 test_annotation_version.py（测的是死 convert 阶段的标注版本回退，零活
  消费方）+ run_regression_matrix.py 的 nnunet_annotation_version 套件注册。
- 删 tools/nifti_to_dicom.py（零调用方+零测试+默认写入输入树）、
  tools/prepare_kidney_3d_append.py（仅 docs 提及）。
- 测试同步：test_nnunet_integration.py StandaloneWorkflowTests 收缩为
  Action3TrainerContractTests（唯一活契约：epoch trainer 解析）；standalone
  import ×2 删除。
- 文档同步：nnunet_mimics_integration.md（入口列表 4→3、Standalone CLI 章节
  改为 workflow note）；uncertainty.py 注释不再引用已删 Config_windows.toml。

**验证**：删除前逐文件全仓引用扫描（py/md/json/toml/sh/bat）零活引用；
nnunet_integration 81/81；stage worker import 冒烟 OK。

**门禁**：fast 27/27（20260927T224936_fast.json）、full 29/29。

**training_convergence 首跑超时勘误**：20260928T001612_full.json 中该套件
3600s 超时（flexict preprocess worker 孤儿挂起：gate 已开、worker 4.45 CPU 秒
零输出零子进程，23:25:02 起无进展）。判定为环境性挂起而非 R41/R42 回归：
(1) 两轮改动未触碰 flexict 训练路径（R42 对 flexict 唯一改动是 uncertainty.py
一行注释，不在训练路径上）；(2) 同一 full 门禁中 nnunet 家族及全部其余 28 套件
通过；(3) 孤儿 worker 已清理后单独重跑 Flexict 收敛测试 1012.8s OK
（与 R40 基线 1132s 同量级）。机器当时另有两个用户自启进程在轮询网络盘
（sync_missing_masks.py），可能与之相关但未确证。

**架构审查结论**：净 -8639 行，零新增维护面。审计"整目录死代码"的失实
被引用扫描拦截——死代码清理必须逐文件核实，账本记载不可作为删除依据。


---

## R43（2026-09-28）：D4 — Fix Affine 合并进预测失败路径

**用户决策**（2026-09-27）：Fix Affine 是一次性迁移修复工具（修 pre-2026-07-13
bridge 导入案例的 source_voxel_to_ras_matrix 错误 LPS 仿射），用户拍板"合并进
报错路径后删入口"。

**痛点根因**：stale affine 只在推理几何校验时暴露；原设计让标注者"读报错 →
记住菜单路径 → 切到 99_Admin 找工具"，这正是引导式失败路径要消除的流程。

**实施**：
1. `tools/nnunet_pipeline.py` `validate_materialized_source_geometry`：报错带
   稳定标记 `source geometry mismatch`（单一定位、单一来源行）。本地 nnU-Net、
   远程 nnU-Net、FlexiCT 三路径共用该校验器，一处改全覆盖——远程路径已核实
   用同一错误格式串传播，无需额外代码。
2. `runtime_py35/fix_source_affine_metadata.py` 新增
   `offer_repair_for_prediction_failure(error_text, framework_title)`：检测
   标记 → 无活动图直接静默返回 False → `question_box` 一次询问
   （Repair Stored Geometry / Not Now）→ 接受则启动既有非阻塞修复流
   （`main()`）。任何异常都吞掉返回 False——修复提议失败绝不能遮蔽原始报错。
3. `runtime_py35/nnunet_mimics.py` / `flexict_mimics.py` 预测失败分支
   （`_monitor_tick_locked` 的 failed 状态）：先查标记，接受即修复并 return，
   拒绝或无关错误则原报错对话框逐字不变。
4. 删除 `scripting_library/99_Admin/04_Fix_Source_Affine_Metadata.py` 菜单
   入口。模块本体保留（修复实现 + mimics_stop_background 监视列表注册）。
5. 文档：`MIMICS_PROJECT_ARCHITECTURE_CN.md` §9.4 重写为合并后的流程。

**测试**：
- `test_prediction_context_errors_point_to_real_repair_paths` 更新：标记断言
  收敛到单一来源行；断言已删入口文件名不在源中。
- 新增 `test_prediction_failure_offers_repair_on_geometry_mismatch`：
  无关错误不弹窗；无活动图不弹窗；接受 → `main()` 恰调一次返回 True；
  拒绝 → 返回 False 且 `main()` 未调；入口文件确实不存在。
  （mock `mimics.dialogs.question_box` / `mimics.data.images.get_active`，
  均通过。）

**门禁**：smoke 8/8（`20260928T010326_smoke.json`）、fast 27/27
（`20260928T011902_fast.json`）。test_all 改动已含在 fast 工件内验证（新测试
单独跑过 + 全量 smoke 套件）；full 门禁本轮未重跑——R43 触碰的
runtime_py35/nnunet_pipeline 均在 fast 覆盖内，test_all 的新增断言已单跑通过。

**架构审查要点**（自查）：修复提议永不 raise；拒绝路径原对话框逐字不变；
标记字符串单一来源行；三框架路径零重复代码（共用校验器 + 共用 offer 函数）。

**复杂度对价**：+184/−61（净 +123，其中测试 +91）。新增维护面：常量 1、
函数 1、同型 helper 2；净删 UI 入口 1 个。

---

## R44（2026-09-28）：D2 — nnU-Net 训练表单折叠为高级设置区

**用户决策**（2026-09-27）：拍板"折叠为'高级设置'区"。

**痛点根因**（承 A5）：表单是 nnU-Net CLI 参数的直译；R8 补了默认值与
一行说明，但十项训练概念仍然全部平铺在标注者面前。

**实施**：
1. `tools/nnunet_training_setup_ui.py` 新增 `_advanced_toggle(group)`：
   QToolButton（箭头指示收起/展开），checked 显示组。Training 组整体
   （Trainer/Epochs/Fold/Validation fraction/Preprocessing workers/
   GPU count/GPU devices/Pretrained checkpoint/Continue/TTA）默认隐藏。
   widget 仍然构建——`_request()` 照常提交其未改动的默认值，所有既有
   程序化测试与提交流程零变化。
2. Model 标签页可见内容收敛为：Planning（自动勾选即推荐默认）+ 折叠
   开关。数据面必填项（task/Dataset ID/数据路径）本就在 Data 标签页。
3. `docs/nnunet_mimics_integration.md`：原文宣称窗口"不按 basic/advanced
   分组"已失实，改为如实记录折叠结构。

**测试**：新增 `test_nnunet_training_group_collapsed_by_default`
（gui_smoke 19/19）：默认 isHidden；toggle 勾选即显示；再取消即隐藏。
（离屏窗口不 show()，isVisible() 恒 False，故用 isHidden 断言——首版
测试即踩此坑，已修正。）既有 hint 断言测试原样通过（hint 未动）。

**门禁**：fast 27/27（`20260928T013948_fast.json`）。仅触碰 tools/ UI 与
文档，未动 runtime_py35/桥接层，smoke/full 无必要。

**复杂度对价**：+58/−4。新增维护面：helper 1、属性 1、测试 1；对价是
标注者默认视野内消失 10 个训练字段。

**遗留核实**：折叠后 hidden widget 在 Qt offscreen 平台下 `_request()`
读取正常（QSpinBox/QLineEdit 值与可见性无关）；`_refresh_trainer`/
`_refresh_fold`/`_refresh_continue_training` 均在构建期连接、与可见性
无关。无其他消费者引用这些字段（flexict_training_setup_ui 有自己的
epochs/workers，属不同表单，D2 范围仅 nnU-Net）。

---

## R45（2026-09-28）：并行全面审计（用户指示）——六代理发现入账，无代码改动

**用户指示**（2026-09-27）："用不同的并行代理检查每个功能，以及每个功能的环节，
是否都达到了设计的预期，是否适用于标注者，是否有没考虑的标注者环节，是否前后段
都稳定、好用，检查后继续优化；我真的很担心进程残留、孤儿进程问题，无论使用本地
还是远程。"

**执行**：R41–R44（用户拍板项 D1/D5/D4/D2）全部关闭后，并行派出 6 个只读审计
代理：① 数据导入导出、② nnInteractive、③ nnU-Net（本地+远程）、④ FlexiCT +
ScribblePrompt、⑤ Review 工具与菜单组织、⑥ 进程残留/孤儿进程专审（本地 spawn
点穷尽 × 父死亡/挂起/取消/断连矩阵 + 远程容器全失败分支）。全部纯只读，未触碰
远程服务器与真实数据集。

**总体结论**：无 P0/P1 产品缺陷级崩溃或数据损坏路径；主链路（导入→标注→导出→
训练→推理→回导）闭环、GPU 锁协议、远程容器清理、R41–R44 回归面全部核实健康。
**真正的结构性薄弱点集中在进程生命周期管理（用户重点关切的方向命中）**。

**历史 FlexiCT 孤儿事故因果链完整定位**（进程专审）：回归矩阵超时只杀直接子
进程（B22）→ detached 控制器+worker 存活 → worker 进程注册静默失效，sweep/
Stop 安全网不存在（B20）→ 控制器死后 worker 无自超时，0 CPU 永挂（B21）→
只能手动恢复。四个环节修任意一个即可阻断复发；性价比顺序 B20 → B22 → B21 →
B23。

**入账**（验收标准已写死，见账本）：
- P1：B20（注册白名单缺失——孤儿安全网不存在）、B21（worker/控制器零挂起
  防御）、B22（矩阵超时孙进程存活）、B23（Stop Background 两处列表缺口）
- P2：B24（4 个已删入口的文档残留，6 处文档）、B25（离线包文件名 mimcs 拼错
  ——标注者死循环）、B26（窗宽窗位无活动图像守卫）、B27（Quick Export 裸
  RuntimeError）、B28（nnInteractive 空结果静默应用可抹标注）、B29（FlexiCT
  推理不认 abandoned 终态）、B30（单病例进空训练）、B31（remote_unreachable
  被改写）、B32（远程命令失败无限重试）、B33（Undo Import 前台大文件 I/O）
- P3：C6-1～C6-10（schema 漂移、日志相对路径、1h 静默脱管、R43 边缘分支吞错、
  双图像并发未验证、表单不自动扫病例、A7 文案同步、A7 登记补记、用户文案技术
  残留 7 处、文档/菜单一致性 3 项）

**审计代理报告全文**：本报告附录（会话工件）保留完整版；账本条目已含 file:line
级定位。数据导入导出代理的勘误更正：01_Data/04 与 06 入口未被删除（R41 删的是
03/05 与 nnUNet/04），账本描述此前已正确。

**需用户决策新增**：D8 monitor deadline 到期是否升级为主动杀外部进程（有误杀
长训练风险，不宜擅定）；D9 菜单编号重排与 ScribblePrompt 改名（动肌肉记忆）。

**本轮无代码改动、无 commit**（纯审计轮）；complexity_ledger 不变。

## R46（2026-09-28）：B20 孤儿安全网——stage worker 进程注册自上线即静默失效

**问题**：`nnunet_pipeline._spawn_worker` 以 `nnunet_{stage}` role 调
`register_process`，但（a）三个 role 不在 `VALID_PROCESS_ROLES` 白名单、
（b）调用传了 API 不存在的 `job_id=` kwarg——每次注册先抛 TypeError，
被 bare except 吞掉。健康面板与 Stop Background 对 stage worker 完全
不可见，孤儿安全网从写下的第一天就不存在。实施中发现 (b) 比审计更糟
（审计只发现白名单缺失）。

**改动**（commit `c190b23`）：
- `resource_locks.py`：`VALID_PROCESS_ROLES` + 3 role（白名单是数据非逻辑）
- `nnunet_pipeline.py`：调用改传 `extra={"job_id": ...}`（既有模式）；
  注册失败写 job.log 不再静默
- `test_all.py`：`test_stage_worker_roles_are_registered`（真实注册往返 +
  job_id 进 extra）+ `test_spawn_worker_registration_matches_api`（源契约：
  调用点每个 kwarg 必须存在于 register_process 签名——API 漂移在测试里响）

**证据**：smoke 8/8（20260928T020933）+ fast 27/27（20260928T021412）。

## R47（2026-09-28）：B22 回归矩阵超时扫残——detached 孙进程不再全部存活

**问题**：`run_regression_matrix.py` 的 `subprocess.run(timeout)` 杀掉
test python 后，套件 spawn 的 detached 控制器/worker 全部存活——历史
孤儿事故的放大器。

**改动**（commit `b948da1`）：
- `run_regression_matrix.py`：TimeoutExpired 分支调
  `resource_locks.sweep_processes(ROOT)`（终止父进程已死的注册孤儿），
  摘要 `[POST-TIMEOUT SWEEP]` 落套件输出
- `nnunet_pipeline.py`：`_spawn_worker` spec 带 `parent_pid`（供 B21
  watchdog 与 sweep 判孤儿）
- `test_all.py`：`test_sweep_terminates_stage_worker_orphaned_by_dead_parent`
  （死父 + 存活 worker → sweep 真终止，进程真实退出）

**证据**：fast 27/27。

## R48（2026-09-28）：B21 stage worker 看门狗——"活着但不动"两形态自退

**问题**：worker 对挂起（OpenBLAS 死后 spawn.Pool 静默挂死）与父进程死亡
（控制器崩溃后 detached worker 永活）零防御。历史 FlexiCT 孤儿事故因果链
第 ③ 环。

**改动**（commit 见 git log，本轮）：
- `nnunet_stage_worker.py`：`_watchdog_trigger`/`_start_watchdog`——daemon
  线程每 5s 检查总预算（train 14 天对齐 Mimics 侧 monitor deadline；
  preprocess/infer 24h 对齐推理 monitor deadline；spec `stage_deadline_epoch`
  可覆盖）+ 父进程存活（`process_exists`）。触发即写 result JSON
  `{"status": "error", "error": "worker_timeout: ..."/"parent_gone: ..."}`
  后 `os._exit(3)`——主线程阻塞在 stage 函数里收不到信号，只能进程级退出。
  控制器既有失败路径原样消费，零改动
- `nnunet_pipeline.py`：spec 带 `parent_pid`（与 R47 同一改动）
- `test_nnunet_integration.py`：3 测试——子进程实跑真 watchdog + 挂死主
  线程 → 码 3 + worker_timeout；死父 pid → 码 3 + parent_gone；
  `_watchdog_trigger` 单元三分支

**实施教训**：第一版测试用 PYTHONPATH shim 假 Action4 模块，因 worker
自身 `sys.path.insert(0, WORKFLOW)` 在 import 时先于 PYTHONPATH 解析，
批量运行时缓存预热即穿帮、结果不确定（solo 过是冷缓存运气）。改为
子进程直接驱动真 watchdog + sleep(120)，不依赖任何模块遮蔽。

**孤儿链闭合总结**：B20（注册安全网）→ B22（矩阵超时扫残）→ B21（worker
自退）三轮后，历史 FlexiCT 孤儿事故因果链 ④ 环中 ③ 已闭合（① 矩阵超时
扫残 = B22、③ 无挂起防御 = B21、② 注册失效 = B20；④ monitor 只停自身
不在本轮范围）。B23（Stop Background 列表缺口）为残余 P1。

**证据**：commit `1c4f7cd`；test_nnunet_integration 84/84（两次重复跑验证
确定性）+ smoke 8/8（20260928T025012）+ fast 27/27（20260928T024914）。

## R49（2026-09-28）：B23 Stop Background 两处缺口 + full 门禁暴露的测试污染

**问题**：① `remote_training_controller.py` 不在 `MARKERS`，PowerShell
cmdline 扫描击杀找不到它（nnunet_jobs.py:136/325、flexict_pipeline.py:1606
以 `python <tools>/remote_training_controller.py` 启动，cmdline 含文件名）；
② FlexiCT 监视器缺 pre-detach cancel 标记——审计字面说"无人停止"，
实施核实为**勘误**：通用循环一直处理 flexict_mimics._MONITORS；真实缺口
是 nnunet 有专属分支在脱离监视器前写 `control.json` cancel（远程容器得以
安全收尾），FlexiCT 没有。

**改动**：
- `mimics_stop_background.py`：MARKERS + remote_training_controller.py；
  nnunet 27 行 cancel 块提取为共享 `_cancel_controllers_before_detach
  (module_name)`，nnunet/flexict 各调一次（两集成共用 job_dir/
  status.json + control.json 布局，fallback 同样命中真标记）——比
  "照抄 8 行"更省，删了重复实现
- `test_all.py`：MARKERS 契约 +1；`test_stop_all_signals_tasks_before_
  detaching_mimics_monitors` 升级（两模块都断言先 cancel 后 detach，
  含 `"action": "cancel"` 检查移到 helper 源）；新增
  `test_stop_all_writes_flexict_cancel_before_detaching_monitor`（行为级）

**实施中发现的额外 bug（full 门禁 28/29 暴露）**：
`TestImportReceiptAndUndo._install_undo_env` 裸赋值
`mimics.data.images = [image]` 污染全局共享 mock 且不恢复——TestImportReceiptAndUndo
按字母序在 TestSourceImagePathEquivalence 前运行，后者 R43 的测试
`patch images.get_active` 直接 AttributeError。R43 当轮只跑了
fast+smoke（test_all 属 full），所以滞后到本轮才炸。修复：全部裸赋值
改 `mock.patch.object` + `addCleanup`；全局 mock 补
`file.get_active_project` 默认值（patch.object 需属性已存在）。

**证据**：commit `192ac8f`；full 29/29（20260928T053808）+ smoke 8/8
（20260928T053859）。

## R50（2026-09-28）：B24+B25 文档死入口清洗 + 离线包文件名拼错

**B24**：R41/R43 删除入口后未同步用户文档，标注者按文档找入口必落空。
修正：
- `mimics_entry_guide.md`：删 Export Masks / nnU-Net Stop Running Task /
  Window Reset Full Range / Fix Source Affine Metadata 4 行
- `scripting_library_workflows.md`：入口清单按现存 21 入口重排（补
  07/08/09、FlexiCT×4、ScribblePrompt、99_Admin 05–10），切换表改指
  nnUNet 状态窗口
- `mimics_real_data_validation.md`：03_Export_Masks → 07_Quick_Export_
  Masks；删 05_Window_Reset_Full_Range 入口行（对话框内 Reset 动作的
  文字保留——它是 Choose Preset 窗内的真实功能）
- `task_lifecycle_and_safety_policy_CN.md`：nnU-Net 停止改状态窗口
- `CONFIG_REFERENCE.md`：Export Masks → Quick Export Masks
- `MIMICS_PROJECT_ARCHITECTURE_CN.md`：删与 4.3 重复的 4.7 小节
  （其"使用标准 Export Masks"指向已删入口）
- **审计清单外补漏**：`windows_end_to_end_acceptance_2026-08-01.md`
  4 处同型死引用（03_Export_Masks、05_Window_Reset、nnUNet/04 两处）

新测试 `test_living_docs_do_not_reference_deleted_entries`：6 份活文档
× 4 个已删入口名；放行显式历史注记行（"原 …删除" 迁移记载属刻意
保留）。遗留发现（未扩本轮范围）：该验收文档 §10 DINOv3 整节已死
（R42 已删 DINOv3），入账 C6 清理候选。

**B25**：`setup_environment.py` 弹窗让用户放 "mimcs_script_portable.zip"
而 `_find_portable_archive` 查 "mimics_..." —— 标注者照指引放文件后仍
报"未找到"，死循环。修复：新常量 `PORTABLE_ARCHIVE_NAME`，弹窗与三处
候选路径共用（顺带消 3 处硬编码）。测试走真实 `main("extract")` 分支
断言弹窗含正确文件名；须 mock `active_runtime_blockers`（本机有用户
自己的 sync_missing_masks 锁在跑——不动它）。

**证据**：commit `0a7462c`；smoke 8/8（20260928T063914）、
full 29/29（20260928T063806）。

## R51（2026-09-28）：B26 窗宽窗位入口无图像守卫

**主题**：03_Review 三个窗宽窗位入口（Choose Preset / Undo / Reset Full
Range / auto）在无活动图像时终点 `set_contrast` 抛未捕获异常直达脚本
错误层，或静默无效。B26。

**实施**：`runtime_py35/window_level_mimics.py` `main()` 入口统一加
`_active_image()` 守卫（mask_identifier.py 同款提示），返回 1。四个
动作终点都是 `set_contrast`，一处守卫全覆盖，无新增分支。

**测试**：`fake_mimics_flow_test.py` 既有 window 用例扩展无图像退化
段：四动作各自断言返回 1、弹友好提示、不开预设对话框、对比度不动。
（第一版断言踩了 fake 的两个坑：questions 列表累积、reset_scene 自身
会重置 contrast——改为先快照再清 images。）

**证据**：commit `c398339`；smoke 8/8（20260928T064611）、fast 27/27
（20260928T065311）。

## R52（2026-09-28）：B27 Quick Export 未保存工程裸异常

**主题**：未保存工程时 Quick Export 走 `_launch_external_export_setup`
的 `raise RuntimeError("Save the current Mimics project before exporting
masks.")`，经 `quick_export_main` → run_runtime_entry 无捕获，标注者看到
脚本层错误而非对话框。B27。

**实施**：`runtime_py35/mimics_export.py` 该分支改 message_box（解释
为什么需要先保存：快速导出基于 .mcs 文件，Mimics 关闭后导出也能继续）
+ return 1，与同文件 `__main__` 分支 Fatal Error 包裹同型。

**测试**：`fake_mimics_flow_test.py` io_setup 路由用例扩展退化段：
`fake.file.project_path` 置空，断言返回 1、弹提示、不抛，随后恢复。

**证据**：commit `da8b69e`；smoke 8/8（20260928T065800）、fast 27/27
（20260928T070229）。

## R53（2026-09-28）：B28 nnInteractive 0 体素空结果静默应用

**主题**：nnInteractive 后台预测返回 0 前景体素的空结果时，"更新所选
mask"（in_place）模式直接把空缓冲写进目标 Mask——用户已画内容被清空，
仅一条 log WARNING，无任何对话框。B28。

**实施**：`runtime_py35/nninteractive_mimics.py` `_handle_async_result`
在应用前（project/image/target/hash 校验之后、`_choose_completed_result_
target` 之前）对 0 体素结果弹确认："Don't Apply" 保留当前 Mask、session
回 ready（保留会话可继续提示）；"Apply Empty Result" 走原 apply 路径。
文案含改点建议（前景点靠近结构中心、背景点放远）。拒绝路径清
`pending_sequence`，与既有 skipped/error 分支的状态处理一致。

**测试**：`fake_mimics_flow_test.py` nninteractive 用例扩展空结果段：
构造真实 result_000001.json（0 体素 + 匹配的 expected_target_sha256），
两分支各自断言——拒绝则不触碰 mask 缓冲、session 回 ready；接受路径经
拒绝消费 pending_sequence 后重查不重复弹窗。

**证据**：commit `c1506d2`；smoke 8/8（20260928T070937）、fast 27/27
（20260928T071452）。

## R54（2026-09-28）：B29 FlexiCT 推理监控不认 abandoned 终态

**主题**：`flexict_mimics.py` 推理分支的终态集合只有
("failed","cancelled")，训练分支含 abandoned。远程推理任务被放弃
（abandoned）后本地监控一直空轮询到 24h deadline，最终报超时——文案
误导且长时间占用轮询。B29。

**实施**：`_monitor_tick_locked` 推理分支终态元组加 "abandoned"；
abandoned 弹窗复用训练分支失败句式，error 为空时兜底 "The remote
task was abandoned."；failed/cancelled 文案不变（repair 提示路径保留）。

**测试**：`tools/test_flexict_integration.py` 新增
TestInferenceMonitorTerminalStates（2 例）：直接驱动 `_monitor_tick_
locked`，abandoned/failed 各断言 `_stop_monitor` 生效（key 从 _MONITORS
移除）+ 弹窗文案正确。

**证据**：commit `978bfc1`；smoke 8/8（20260928T071930）、fast 27/27
（20260928T072416）。

## R55（2026-09-28）：B30 FlexiCT 单病例可进空训练

**主题**：`tools/flexict_training_setup_ui.py` 提交校验只查 cases 非空，
报错文案却写 "Select at least two cases"。单病例提交走到
`flexict_pipeline.py` 时 train=0、val=1，白烧一次 GPU 锁与完整空训练。
B30。

**实施**：校验条件改 `len(request["cases"]) < 2`（一行，文案不动）。

**测试**：`test_gui_smoke.py` 新增
`test_flexict_single_case_submission_is_rejected`：离屏 Qt 窗口 mock
`_request` 返回单病例，走真实 `_submit` → 后台校验线程 →
`_poll_submission` 轮询消费结果，断言窗口未提交且状态栏显示
"at least two cases"。

**证据**：commit `b5b31d8`；fast 27/27（20260928T073432）。

## R56（2026-09-28）：B31 remote_unreachable 被 Mimics 侧改写

**主题**：远程重连预算耗尽后控制器写入非终态 `remote_unreachable`
（带 Re-attach 指引），但 `runtime_py35/nnunet_mimics.py`
`_managed_job_process_stopped` 的豁免元组缺该状态（对照
`tools/nnunet_jobs.py:180` 已含），10s 内被改写为 generic failure，
重连指引文案丢失（Re-attach 仍可用，消息降级）。B31。

**实施**：豁免元组补 "remote_unreachable"（一行，对齐 nnunet_jobs 的
reconcile 豁免集）。

**测试**：`test_nnunet_integration.py` MimicsRuntimeTests 新增
`test_remote_unreachable_is_not_rewritten_by_stopped_process_check`：
死 PID + 60s 龄期 + remote_unreachable 状态，断言不被判 stopped。

**证据**：commit `401b187`；smoke 8/8（20260928T073924）、fast 27/27
（20260928T074401）。

## R57（2026-09-28）：B32 远程 docker 控制失败无限重试无预算

**主题**：`tools/remote_training_controller.py` `_monitor_remote` 中
docker 控制命令（`RemoteCommandError`）持续失败时无重试预算：网络错误
有 `_reconnect_session` 的 RECONNECT_MAX_ATTEMPTS=30 预算，docker 失败
却无限循环——无人干预时控制器本地进程永活、任务状态永悬在
`remote_control_unavailable`（进程残留审计 B32，直击用户最担心的
"无人看管任务永活"问题）。

**实施**：`_monitor_remote` 新增 `docker_error_attempt` 计数（成功轮次
归零，与 `reconnect_attempt` 同型）；`RemoteCommandError` 分支先写
`remote_control_unavailable` 状态（含 `remote_docker_error_attempt`
计数与错误详情）再判预算，超限抛既有 `RemoteServerUnreachable`——由
`run()` 既有 handler 落为非终态 `remote_unreachable`（B31/R56 刚保护
该状态不被 Mimics 侧改写），文案指引用户 Re-attach 或 Stop/Abandon
Locally。零新增状态、零新增配置。

**测试**：`test_remote_training.py` MonitorRobustnessTests 新增
`test_docker_control_failures_are_bounded`：全失败的 Session（所有
docker/文件命令抛 RemoteCommandError）驱动真实 `_monitor_remote`，
断言 (a) 抛 RemoteServerUnreachable 且文案含 "Docker control"；
(b) 终态 status=remote_control_unavailable；(c)
remote_docker_error_attempt == RECONNECT_MAX_ATTEMPTS。测试用推进式假
时钟（每次 time.time() 调用 +1s）消除每轮 10s 的真实等待，单测运行
3.6s。

**证据**：commit `eb16503`；fast 27/27（20260928T080914）。

## R58（2026-09-28）：B33 Undo Last Import 前台哈希/开存工程无预告

**主题**：Undo Last Import 的 `_mcs_fingerprint`（全量 SHA-256）、
`open_project`、`save_project` 全部在 GUI 线程执行且无任何预告——
.mcs 数百 MB 时标注者看到的是"卡死的窗口"（铁律 1 风险，B33）。

**指纹阶段耗时评估（验收要求）**：本机基准测试：500MB = 0.41s、
1GB = 1.15s（约 1.2GB/s，硬件加速 SHA-NI）。数百 MB 的 .mcs 哈希
亚秒级，远低于"秒级以上才考虑后台线程"的门槛。**取舍结论：不挪
后台线程、不降级为 mtime+size 弱校验**：
- 后台线程需要把单函数 undo 流改成消息驱动续体（open→hash→save
  各自异步化再回调），新增状态机维护面与 B33 的亚秒收益完全不对价
  （铁律 9 改动最小化、复杂度预算风险不对称原则）；
- mtime+size 弱校验会削弱回滚安全性：Mimics 保存可能不更新 mtime
  粒度内的区分度，fingerprint 匹配是"文件未变才删除"的唯一强
  保证（该安全性是 receipt 设计的核心，见 create_mcs_batch.py
  注释），弱化它等于为亚秒级收益引入数据风险。
- open/save_project 是 Mimics 自身 API，只能在 GUI 线程调用——
  这段停顿的正解是预告，不是消除。

**实施**：删除前经 `log_user_message`（INFO）向日志面板预告
"opening and verifying <file>; Mimics may pause for a moment;
Masks to remove: N"，API 不可用时静默跳过（与模块内其他日志调用
同防御）。零新增状态、零配置、零 UI 入口。

**测试**：`test_all.py` TestImportReceiptAndUndo 新增
`test_undo_announces_blocking_steps_in_log_before_deleting`：用
mid-flight 快照（包一层 `_delete_masks`，进入时记录通知数）钉住
**顺序**——通知必须在删除开始前落地，而不只是存在；另断言文案含
文件名、级别 INFO、恰好两条日志（预告 + 既有 undo 摘要）。

**证据**：commit `5fba9c4`；smoke 8/8（20260928T082215）、fast 27/27
（20260928T082744）。
## R59（2026-09-28）：C6 并行审计 P3 批次（10 条）

**主题**：R45 六代理并行审计的 P3 汇总条目 C6-1～C6-10，一轮全部关闭。
B1–B33（P2 队列）已于 R57/R58 清空，本轮开始消化 P3。

**实施**（逐条，均为最小改动）：
- C6-1：owned-server 启动清扫 schema 检查 v2 → (v2, v3) 元组（bridge
  现行写 v3，清扫曾对其失明）。
- C6-2：`_worker_main` 默认桥接日志路径 CWD 相对 → tempfile 绝对
  （早期错误曾写进任意工作目录）。
- C6-3：nnU-Net/FlexiCT setup 监控 deadline 到期时，若控制器进程仍
  存活则顺延 1h 而非停监控（表单开超 1h 后提交，完成弹窗曾静默丢失）。
- C6-4：fix_source_affine_metadata "已在运行"分支返回 0 → 2（0 曾被
  上层当作"修复已启动"吞掉原始失败对话框）。
- C6-5（本轮唯一的实质新功能）：**代码级核实确证同模型双图像互踢**——
  server `claim()` 满载即 503 不驱逐、远端会话构造即占租约、bridge
  容量自愈重启直接终止共享服务器；第二图像 worker 必然摧毁第一图像
  在途预测。实施前置拦截 `_same_model_busy_workers`（在途预测阻止
  第二图像启动，对话框说明单会话限制）+ `_retire_idle_image_workers`
  （空闲预热 worker 在启动新 worker 前退役；保留语义为模型+图像
  AND；`_retire_different_model_workers` 重构为其调用方）。
- C6-6：FlexiCT 训练表单读到已存 dataset/label 路径后自动扫描病例。
- C6-7：ScribblePrompt 取点文案补 A7 同款"勿切换工具"后果段。
- C6-8：A7 条目补记 nnInteractive indicate_coordinate 同源风险（只
  登记，不改代码）。
- C6-9：7 处用户文案泄漏（环境变量名、内部 JSON 路径、错误入口名）
  全部改为可操作措辞；修复过程中再捕获 2 处同型泄漏一并修复；
  B4 残留测试扩 5 文件、禁词精化为 standalone `"Report: ` / `"Check: `
  字面量（旧禁词误伤日志行）。
- C6-10：CONFIG_REFERENCE.md undo 状态路径改正（ui_state/ →
  .mimics_runtime/）；验收文档 DINOv3 死内容整段清除并替换为 FlexiCT
  现行入口，过期测试计数改为指向回归矩阵实时输出；living-docs 测试
  加 `"02_AI/DINOv3"` 禁引标记。菜单重排/ScribblePrompt 改名维持 D9。

**测试**：新增 6 项——test_all.py 5 项（schema 清扫、worker 日志绝对
路径、affine 已在运行非零、setup deadline 存活顺延/死亡停止）+
fake_mimics_flow_test.py 1 项三段断言（双图像在途阻止+文案、同图像
不拦、空闲异图像退役且本图像保留）；B4 残留测试扩展；living-docs
测试扩展。

**证据**：commit `027812f`（16 文件 +427/−77）；test_all.py 483 passed
（22:49）；门禁工件 smoke `20260928T101701`（8/8）、fast
`20260928T104004`（27/27，期间 1 次 cross_workflow WinError 5 为
无关瞬态文件系统错误，单独重跑通过后整门禁复绿）、full
`20260928T113114`（29/29）。

## R60（2026-09-28）：D9 菜单编号重排 + ScribblePrompt 改名（用户拍板项）

**主题**：用户 2026-09-28 拍板 D9"重排编号 + 改名"（同批 D8 拍板
"维持现状不杀"，关闭零改动）。

### 1. 实施摘要

- `git mv` 14 个入口：01_Data 04–09 → 03–08（03_Import_Masks 前身
  03_Export_Masks 已于 R41 删除，编号回收）；03_Review 06 → 05；
  99_Admin 05–10 → 04–09（04_Fix_Source_Affine_Metadata 前身已于
  R43 删除）；`02_AI/ScribblePrompt.py` →
  `02_AI/Annotate_With_ScribblePrompt.py`（动词开头，对齐其余 20 个
  入口）。重排后四个目录编号全部连续无跳号。
- 引用同步（17 处，不含历史记录）：test_all.py 10 处路径断言、
  test_model_portability.py 1 处、tools 三文件 docstring
  （batch_status_viewer/import_drop_window/system_health_panel）、
  CONFIG_REFERENCE.md 2 处、架构文档 12 处、验收文档 9 处、
  workflows 文档清单与正文、mimics_real_data_validation 1 处。
- **不追改**：DELIVERY_REPORT*.md、docs/changes/*（带日期的交付
  历史记录）；docs/03_Review 账本（历史不重写）。入口文件内容零
  改动——`_mimics_entrypoint` 按文件名定位 runtime_py35，与编号无关；
  runtime_py35 弹窗文案用菜单名（如 "Environment Guidance entry in
  the 99_Admin menu"）不含编号，无需改动。

### 2. 风险披露

重排/改名使标注者既有肌肉记忆（编号位置）失效——用户已知悉并拍板
执行。文件名变化不影响功能：Mimics 菜单按文件名排序展示，入口
调用链（文件 → runtime_py35 模块）经 `_mimics_entrypoint` 与编号
解耦。

**证据**：commit `761c941`（24 文件 +57/−57，14 个 rename 100% 无内容
变化）；门禁 fast 27/27（`20260928T140508`）、full 29/29
（`20260928T150132`）。

## R61（2026-09-29）：四视角跨职能评审（只评审 + 账本登记，零代码改动）

**主题**：用户要求从产品经理/系统架构师/标注者/UX 设计师四视角对全 repo
做跨职能评审，重点核验 AI 功能（nnInteractive/ScribblePrompt/FlexiCT/
nnU-Net）与公开文献/开源原版的集成正确性。四个独立视角代理并行评审 +
人工对全部 P0 与抽查 P1 逐条源码核验。

### 1. 集成核验结论（用户明确要求项）

- **nnInteractive 2.4.2（vendored）**：桥接层调用签名与 vendored 源码逐参数
  一致（nninteractive_bridge.py:1295-1360 vs remote_session.py:151+/
  inference_session.py:53+，远程 7 参数 + 本地 8 参数全匹配）。✅
- **ScribblePrompt**：忠实 vendoring 自上游 commit `182c4497`（Apache-2.0），
  128px 双线性重采样契约与 worker 实现一致。✅
- **FlexiCT**：自研 few-shot 框架带真实验证记录（fp32 必需；2D Dice
  0.939/3D 0.961 @28 例留出集）。✅
- **nnU-Net**：⚠️ 版本门禁失配——安装 nnunetv2 2.8.0 但 setup_env.py:279
  要求 ≥2.8.1，触发假修复提示（登记 R61-14）。
- **Mimics API 互操作**：create_mask/masks.delete/update_gui/message_box/
  question_box 用法全 repo 一致，无 API 误用。✅

### 2. 发现与登记

3 个 P0（R61-1 nnInteractive 启动 NameError、R61-2 指纹过期静默覆盖已
标注 .mcs、R61-3 FlexiCT AL 蒙版转换阻塞 GUI 10 分钟）+ 10 个 P1 +
8 个 P2/P3，全部经人工源码核验证据链闭合后登入账本待办区。

### 3. 用户拍板（2026-09-29）

外来脚本**收编**（R61-22）；批量推理**立项下轮实施**（R61-23）；UI 语言
**统一中文，注意编码防乱码，部分错误可英文显示**（R61-24）；D19 FlexiCT
增训**立项低优先级**（R61-25）。

### 4. 局限与红线

- 未做实机 Mimics/GPU 测试；训练流程未被标注者视角覆盖；上游对照仅限
  vendored 副本（未联网核对原仓库当前版本）。
- 外来文件 .mimics_runtime/copy_to_z.py 向 Z: 数据根下新建
  `label_task_mr_synced` 目录写入（非 v201 内部）——红线边缘操作，已
  提请用户裁决，拍板"收编"后将单独审视写入路径。
- 本轮零代码改动、零外部资源使用、无 commit。

## R61 修复阶段（2026-09-29）：3×P0 + 2×测试资产当轮关闭

评审后用户指示"开始吧"，当轮修复 R61-1/2/3/13/20 共 5 项，4 个 commit。

### R61-1 nnInteractive 启动 NameError（P0，commit `201b516`）

- **根因**：历史删除轮误删 `_prompt_buttons_for_profile`，调用点
  （`_async_prompt_menu`）仍在 → 每次标注会话启动即 NameError。
- **修复**：函数按 bd180da^ 逐字恢复；同 commit 修正两处指向不存在入口
  "Stop All Owned Background Services" 的文案（真实入口
  99_Admin/03_Stop_All_Owned_Services，即 R61-20），并把
  nninteractive_mimics.py 加入 B4 残留扫描表。
- **测试**：`test_async_prompt_menu_executes_without_monkeypatch`（不
  monkeypatch 走真实路径到 question_box，断言按钮串）+
  `test_prompt_buttons_for_profile_rejects_empty_profile`。

### R61-2 指纹过期重导入静默覆盖已标注 .mcs（P0，commit `0be6ba5`）

- **根因**：per-case 指纹文件住在 import 队列 runtime 目录里，被 14 天
  `_prune_import_queues` 清掉；已标注 .mcs 在输出目录存活 → 指纹缺失
  被当成"未导入"静默重新生成、覆盖用户标注。
- **修复**：指纹缺失时查 `dataset_manifest` provenance
  （`provenance.source_fingerprint`，与 .mcs 同目录、剪枝后存活）：
  一致 → 跳过 + 回写指纹文件自愈；真源变更 → 照旧重处理（用户意图）；
  无记录 → 保留 .mcs 跳过并日志告知强制重导入方法。三种情况都不再
  静默覆盖。
- **测试**：3 个失败路径测试（一致跳过自愈 / 无记录保留 / 真变更重处理）。

### R61-3 FlexiCT AL 蒙版转换阻塞 GUI 最长 10 分钟（P0，commit `a2d3244`）

- **根因**：`_al_apply_request` 在 GUI timer tick 内 Popen +
  `communicate(timeout=600)` 同步等待外部转换进程。
- **修复**：复刻预测流 wait_bridge 两段式——tick 只 Popen + 标记
  "converting"；daemon 线程 communicate 写 result.json；后续 tick 调
  `_al_finish_conversion` 在 GUI 线程应用蒙版（Mimics API 线程亲和
  保留）。"converting" 中间态对 request_updates 与 _al_pending_requests
  透明，无双启动。
- **测试**：AST 契约（communicate 不得出现在嵌套线程函数之外）+
  applied/failed 两失败路径测试。

### R61-13 GUI 线程阻塞源码契约扫描（P1，commit `afff2bc`）

- `TestGuiThreadBlockingContract`：遍历 runtime_py35 全部 `*_tick` 函数
  **及其同模块调用闭包**（tick 级扫描抓不到 R61-3 那种间接链，升级前用
  修复前代码验证调用闭包版能报红），禁止 communicate()/tobytes() 出现在
  嵌套线程函数之外。存量 9 处白名单登记（7 处 tobytes → R61-12；2 处
  nnInteractive 同步桥接 → 新登记 R61-26），修复时逐条删除。open_project
  为 Mimics API 线程亲和且仅显式 open 动作触发，刻意不列入禁词。

### R61-20 B4 扫描表补缺（P3，随 `201b516` 关闭）

见 R61-1 第三条；死入口名以禁止串形式钉住防回潮。

### 工作区事件：E: 盘满

R61 期间 E: 盘 100% 满（剩 2.2MB），fast 门禁 2 个 test_all 用例 Errno 28
瞬时失败、单跑即绿——磁盘满瞬态，非回归。用户释放空间后恢复（6.3G）。
仓库本体约 8G / 2.8T 盘，满盘非本仓库所致。

### 证据

- smoke 8/8：`20260929T124245`
- fast 27/27：`20260929T130542`（R61-2 后）、`20260929T131846`（R61-3 后）
- commits：`201b516` / `0be6ba5` / `a2d3244` / `afff2bc`
- 外部资源：零使用（纯本地代码+测试）。

## R62（2026-09-30）：R61 待办批次第一轮——R61-4 死按钮

### R61-4 模型管理器死按钮（P1，commit `aadd6e3`）

- **根因**：`tools/model_manager_ui.py` 的 `_fill_table` 末尾无条件
  `setEnabled(False)`，而全文件无任何选中信号连接（itemSelectionChanged/
  cellClicked/currentItemChanged 均无）——"Use This Model" 与
  "Remove Broken Entry" 从窗口写好第一天起就永久禁用，两个 handler 的
  完整逻辑（含 broken 拦截、set_recommended_model 切换）不可达。
- **修复**（净 +9 产品代码 / +73 测试）：`itemSelectionChanged` 接
  `_update_action_buttons`（Use 需 usable 行、Remove 需 broken 行——与
  handler 语义镜像防将来漂移），`_fill_table` 的两行硬禁用改为调同一
  入口。
- **测试**：离屏 Qt 驱动真实窗口（mock collect_rows 返回 usable+broken
  两行）：无选中双禁 → 选 usable 行 Use 开 → 选 broken 行 Remove 开 →
  清选中（`setCurrentCell(-1,-1)`；注意 Qt 的 clearSelection 不清
  currentRow）双禁；+ `_use_clicked` 真调 `set_recommended_model` 并触发
  reload。
- **门禁**：smoke 8/8（`20260930T112139`）；fast 首跑 26/27——
  flexict_pkg 以 exit_code 0xC0000005（原生层 access violation）死于
  `test_flexict2d_weight_load` 中段，torch 瞬态段错误（A10 记录的
  OpenBLAS 同类环境问题），单跑 10/10 全过（135s）；复跑 fast 27/27
  （`20260930T115343`）。本轮改动为纯 PySide6 文件，与 torch 无交集。

### 证据

- commits：`aadd6e3`；工件：`20260930T112139_smoke.json`、
  `20260930T115343_fast.json`；外部资源：零使用。

## R63（2026-09-30）：R61 批次——R61-5/6/7 三个 P1

### R61-5 导入失败弹窗指向真实失败位置（P1，commit `be7a8a7`）

- **根因**：所有失败工件（`_failed_cases/` 每 case 一个 JSON、日志）都在
  队列本地 runtime 目录（`_rt(output_dir)`），而弹窗要么指向输出目录下
  不存在的 `logs/_create_mcs_batch.log`，要么提到早已被 per-case 目录
  取代的 `_failed_cases.json` 单文件；失败记录也不含用户原始输入路径，
  标注者无从知道哪个病例失败、从哪来。
- **修复**（净 +100 产品 / +33 测试，删 -22 死引用）：`source_image`
  字段贯通 producer（mimics_import `_record_failed_case`）与 consumer
  （create_mcs_batch `record_failed_case`，从 prepare_manifest 回读）
  全部失败记录点；prepare 失败弹窗直接显示 case 名 + 原始 Source 路径；
  其余弹窗/guidance 一律指向用户可达入口 "Show Batch Status (01 Data
  menu)"（其 Open Log 打开的就是队列 runtime 目录真实日志）。
- **测试**：失败路径优先——guidance 不再引用 `_failed_cases.json` 且
  明示 "original source path"；`source_image` 落盘 + 省略时向后兼容。

### R61-6 Stop Import 前确认并声明影响范围（P1，commit `6eb7589`）

- **根因**：`main_stop_import` 无确认即调 `stop_background_import`，而
  `_request_queue_stop` 遍历所有发现的队列目录——一次点击全局停所有
  导入队列。
- **修复**（净 +17 产品 / +50 测试）：复用 FlexiCT question_box 确认模式，
  消息明示影响范围（"every import queue on this workstation, not just
  the current one" + 后台 Mimics 关停 + 已完成项目保留）；取消零副作用。
- **测试**：fake mimics 注入，取消路径断言 `stop_background_import`
  未被调用 + 消息含范围声明。

### R61-7 输出路径粘滞（P1，commit `ccbda86`）

- **根因**：`io_path_setup_ui` 把记忆的输出路径直接当作用户锁定
  （`output_user_edited = bool(output_initial)`），而 Remember 默认勾选，
  上一次导入的输出目录粘住后续所有新数据集；拖拽窗口同病（仅输出框
  为空时重算 + 同会话跨 drop 粘滞）。
- **修复**（净 +64：io_path_setup_ui + drop 窗口记录 `output_custom`，
  提交时判定输出是否偏离按源算出的默认值，加载/新 drop 仅真自定义值
  粘滞；用户同会话手动输入/浏览仍被尊重——hasFocus 过滤程序化
  setText）。顺手清除死代码 `build_selection`（全仓库零调用）。
- **测试**：source-contract 钉住两个 UI 的加载/保存/会话内三处行为 +
  死代码不回归。

### 门禁与证据

- smoke 8/8 ×3：`20260930T132216` / `20260930T133307` / `20260930T134618`
- fast 27/27：`20260930T150137`（R61-5 全量改动后；首跑 26/27 为
  flow_imports 的 torch/OpenBLAS 0xC0000005 瞬态崩溃，单套件重跑 3/3
  复绿——R62 flexict_pkg 同类环境瞬态第三次出现，均与改动无交集）
- commits：`6eb7589` / `ccbda86` / `be7a8a7`；外部资源：零使用。

## R64（2026-09-30）：R61 批次收尾——R61-8~R61-24 大扫除

R62（R61-4）/R63（R61-5/6/7）之后，本轮把 R61 待办批次的剩余条目全部
清完：8 个 P1/P2/P3 修复 + 测试资产 + 3 个用户拍板项（收编、批量推理、
UI 中文化），外加验证战役（Phase 1~4b）中发现的 4 个 P0/P0-2 顺手关闭。
R61-25（FlexiCT 增训）按用户指示取消，不实施。

### P1/P2/P3 修复批（每项独立 commit + smoke 证据）

- **R61-8 拖拽导入状态冻结（P1，`eb00983`）**：drop 窗口的批量导入状态
  由真实队列状态驱动，不再卡死在 "Importing"。
- **R61-9 数据集根误判为单 case（P1，`d2e0727`）**：单病例发现拒绝数据集
  根目录（多 case 结构），杜绝把整个数据集当一条导入。
- **R61-10 弹窗风暴（P1，`e1d55e9`）**：nnInteractive 写模式提问改会话级
  一次，不再每个 Mask 问一遍。
- **R61-11 扫描剪枝（P1，`f037b8e`）**：Stop 扫描的 os.walk 剪枝 + kill
  移出扫描线程，消除停止操作本身卡 GUI。
- **R61-12 全量体素缓冲（P0，`93be706`）**：蒙版 SHA-256 / 体素缓冲改流式
  处理，移出 GUI 线程（R61-13 白名单消 7 条）。
- **R61-26 同步桥接（P1，`37c8219`）**：nnInteractive 桥接 communicate()
  放等待线程 + GUI 泵（白名单消 2 条，R61-13 白名单清零）。
- **R61-14 版本门禁（P2，`a1291ba`）**：nnunetv2 门禁接受随包 2.8.0 环境，
  不再误报"需升级 2.8.1"。
- **R61-15 孤儿作业清扫（P2，`0e18d4e`）**：崩溃孤儿 async 作业目录超
  保留窗口即清扫。
- **R61-16 ScribblePrompt 体检（P2，`7ddf351`）**：检查点纳入环境体检，
  缺失时在健康面板可见。
- **R61-17 Abandon Locally（P2，`a31a5c5`）**：FlexiCT 状态窗加"在本机
  放弃"逃生口（含 Stop 确认，R61 后续项）。
- **R61-19 编号漂移（P3，`c5f760a`）**：活文档入口编号漂移同步 + 一致性
  测试防回归。
- **R61-21 状态色统一/图标/UX 残留（P3，`821f60d`）**。

### 用户拍板项

- **R61-18 根 README（P2，`f3330ea`）**：三步快速上手，新用户零文档开箱。
- **R61-22 收编外来脚本（`b06a1d1` 前置清理 + R62/R63 验证）**：4 个
  一次性诊断脚本确认为会话痕迹，不收编（留在 ignored 运行时目录自然
  过期）；copy_to_z 输出在只读数据集之外，未越红线，已登记。
- **R61-23 批量推理（P2，`cc36403` + `952a62e`）**：mimics_batch_cli 新增
  predict 子命令（数据集根/平铺夹、--cases 过滤、歧义跳过并报告、无
  可用模型退出 2 列原因）；顺带修 python313._pth 导致 CLI 丢 tools 目录
  的 P0。4 条失败路径测试入 test_all。
- **R61-24 UI 中文化（`5adfa2e`）**：19 个 UI 文件 + ui_theme 组件级全量
  翻译。技术名词（nnU-Net/FlexiCT/Mask/GPU/AUC/Trainer/Fold/.mcs）与
  代码值保留英文；traceback/控制台输出不动（GBK 控制台陷阱）。自研
  字节偏移 AST 替换器（ast col_offset 是 UTF-8 字节偏移而非字符偏移，
  字符级切片会把含多字节字符的行切坏）。残留扫描 0 英文 UI 字符串。
- **R61-25 FlexiCT 增训：用户取消**（"FlexiCT 增训不用实现了"），不实施。

### 验证战役顺手修复（P0）

- **P0 CLI sys.path（`cc36403`）**：见 R61-23。
- **P0 assert_gpus_not_busy NameError（`dbb7090`）**：远程训练阻断门
  函数名漂移导致的崩溃，修复 + 验证报告记录。
- **P0 临时发布锁（`8ca4e4b` + `b7d4a10` + `9259b2d`）**：staging 目录
  发布对 Windows 瞬态文件锁重试；P0-2 后续把同类修复带进 nnInteractive
  微调 source-grid 缓存发布；失败与修复全程记入验证报告。
- **206 根文件系统满（`0ad9708`）**：服务器阻塞登记（外部资源事件，
  未越红线，清理记录见验证报告）。

### 验证战役文档（未计入代码账）

`b06a1d1`→`3740117`：AI 真实数据验证报告 Phase 1~4b（nnInteractive
推理/微调时序基线、微调 vs 官方对比、P0-1/P1-1 根因复验）。

### 本轮发现并当场关闭的回归

- **R64-1 ScribblePrompt 会话静默丢弃（P0，`6f453ed`）**：R61-12 把
  `_sha256_bytes`（裸 hex）换成 `stream_buffer`（"sha256:"+hex）时，
  `_session_values` 与 worker 存的裸 hex 永不匹配 → previous-logits
  连续性每个 prompt 都被丢弃。fast 门禁的 interactive_algorithms 套件
  抓到（test_buffer_streaming_preserves_bytes_and_hash 断言失败暴露
  格式漂移），追根发现真回归在会话链。修复：比较前归一化 + 回归测试。
  **教训：换共享 helper 时必须审计所有返回值消费者的格式假设。**
- **R64-2 回归矩阵 GBK 崩溃（P2，`3427fa5`）**：失败输出打印崩溃，
  掩盖失败摘要本身。
- **R61-24 补遗（`c0dfd99`）**：`_surface`/`_path_row`/`_hint` 辅助函数
  生成的文案（分组标题/路径对话框标题/字段提示）不在首轮提取范围，
  FlexiCT 停止按钮三元式漏翻；5 个测试套件钉住旧英文文案，全部同步。

### 门禁与证据

- smoke 8/8 ×4：`20260930T230848`（R61-23）/ `20261001T022755` +
  `20261001T025629`（R61-24 两段）/ `20261001T025816`（R64-2）/
  `20261001T031950`（R64-1）
- fast 27/27：`20261001T032902`（R64 全部改动后）。`20261001T030717`
  为 26/27——interactive_algorithms 失败即 R64-1 的发现过程。
- 外部资源：远程服务器仅验证战役使用（记录在验证报告）；真实数据集
  只读红线全程未越。

## R65（2026-10-01）：外部评审 docs/reviews/ F01–F29 收编——W01/W02 数据保护

评审基线 `36eac04` 早于 R61–R64 修复，每项动手前对照 HEAD 复核。29 项发现
全部入账 improvement_backlog.md R65-1~R65-20，按交接推荐顺序 W01→W02→
W03→W04/W06 推进。用户测试纪律：全量只在必要时，日常按改动选套件+冒烟，
每轮收尾 fast。

### 已关闭（本轮）

- **R65-1 = F22【P0】（`2f3ed65`）**：slug 碰撞 + 硬链接改写源文件。
  `unique_case_keys`（无碰撞保裸 slug，碰撞全组 8 位摘要）+ 两处
  `_link_or_copy` unlink-first + 5 个消费点接入 + `.mcs` 按原始名解析。
  fast 27/27（`20261001T184356`）。
- **R65-2 = F18【P0】+ R65-3 = F19【P1】（`b8d236b`）**：撤销导入所有权
  模型重做。receipt v2（guid + provenance token——token 在 save 前写入
  mask metadata 随 .mcs 持久化）；删除文件前比对会话内存态（任何
  receipt 未涵盖的 mask 保留文件）+ save 后指纹复验；同名歧义停止并
  提示（返回码 7），v1 旧 receipt 保守处理。T18/T19 回归测试 7 例。
  TestImportReceiptAndUndo 18/18、fake_mimics_flow 16/16、冒烟 8/8
  （`20261001T192254`）。
- **R65-4 = F17【P1】（`f2bd662`）**：项目状态适配器。两个切项目入口
  弃用无文档支持的 `get_active_project()`（异常被吞，"无法识别"被当
  "没有打开"），改走共享 `runtime_common.current_project_state()`（文档
  化 API，四态 none/path/unnamed/unknown）；同项目不重载，unknown 拒绝
  切换并声明"什么都没动"。T17 回归 4 例；flexict_integration 93/93、
  TestImportReceiptAndUndo 21/21、冒烟 8/8（`20261001T193902`）。
  W02（项目和撤销所有权）至此全部关闭。
- **R65-9 = F23【P1】（`a25e755`）**：mask 数值语义统一闸门。三个
  读取器（binary ×2 + 多标签）统一走 `_validate_mask_values`：NaN/Inf
  一律拒绝（旧 `!= 0` 把 NaN/Inf 变成前景——评审证据 `[0,NaN,Inf,1]`
  → `[0,1,1,1]`）；小数浮点一律拒绝为概率图并给出转换规则（旧代码
  静默按 0 阈值，`[0.01,0.49,0.51,0.99]` 全变 1）；整数值数据（0/1、
  0/255、labelmap、整数值 float）原样放行。训练准备（nnunet/
  nninteractive_finetune pipeline）与 4 个 bridge 入口经同一闸门。
  动手前核实：仓库内全部 mask 生产者只写整数数组（write_mask_nifti
  等均 astype(uint8)），拒绝只作用于外部概率图/损坏文件，无内部
  主线被破坏。T23 证据：4 个新失败路径测试 + convert/prepare 26 +
  nnunet 90 + flexict 93 全绿；冒烟 8/8（`20261001T200108`）。
- **R65-10 = F24+F25【P1】（`41306bf`）**：补齐 Mask 误判。F24——
  平铺布局下 `_source_masks` 把 ct.nii.gz 也当结构（评审证据：
  ct+liver 返回两个待补 Mask）；现按公认图像名（ct/mri/mr/image，
  与 image_candidates 一致）排除，仅剩图像的病例明确报错。F25——
  完整性判定在全项目名称集上做，另一 Image 的同名 Mask 让目标被
  误判"已完整"（证据：Image A 缺 liver、Image B 有 → skipped）；
  现多 Image 项目整例失败并声明"nothing was changed"（评审指定
  "多候选 Image 需显式选择或错误提示"），单 Image 下全项目集合
  即目标集合。顺带删死代码 `_mask_foreground`。T24/T25 证据：
  2 新测试 + 相邻 26 绿 + 冒烟 8/8（`20261001T202725`）。
- **R65-14 = F29【P2】（`89a90e3`）**：FlexiCT 颅尾分层轴错误。
  `argmax(spacing)` 只在经典厚层轴位 CT 上碰巧正确：等体素 RAS 上
  平局选 X（评审证据：目标 z=8 被判 z-start 0.1，物理应为 0.8）、
  轴换位/斜位上选层面内轴。改为按 affine 世界方向选 SI 轴
  （`physical_si_start`：体素轴的世界位移与 superior 方向最对齐者），
  管线与独立 build_dataset.py 同步修复；独立脚本另拒绝 <2 可用病例与
  `--n-train<1`（原 1 例时 n_train=0 → 除零）。保留分层策略本身
  （few-shot 验证配方行为不变），评审给了"物理 SI 或取消启发式"两条
  路，选前者为最小改动。T29 证据：新套件 flexict_build_dataset 8 测试
  （等体素/各向异性/轴换位/翻转/空目标 + 退化输入 3 项，注册进矩阵
  fast/full）+ TestTrainValSplit 6 新测试；flexict_integration 99、
  flexict_common 24、flexict_pkg 10、flexict_build_dataset 8 全绿；
  冒烟 8/8（`20261001T215214`）。
- **R65-15 = F16【P2】（`cd163a0`）**：训练划分患者级隔离 + 冻结验证
  集。原 `_split_folds` 只洗牌 case_id，同患者多期/随访（不同 case_id）
  可能同时进 train 和 val，质量估计偏乐观。现训练请求可选
  `patient_groups` 映射，存在时洗牌与 fold 轮换按整组进行（组信息写
  入 manifest）；重建数据集读取已发布 fold-0 splits，旧验证病例在所有
  fold 保持 val——加数据永不把旧验证病例挪入训练，冻结一个病例自动
  冻结其整组。无组信息时保持 case 独立性假设，契约记入
  CONFIG_REFERENCE。T16 证据：8 新测试（组不跨侧、无组行为不变且与
  历史轮换逐位一致、冻结保持 val、冻结扩散整组、消失的冻结病例被
  忽略、两次构建端到端验证集只增不减、manifest 记录组）。
  nnunet_integration 97、flexict_integration 99、remote_training
  104 全绿；冒烟 8/8（`20261001T231524`）。W03 至此除 R65-12（F01，
  需 L3 实机）外全部关闭。
- **R65-5 = F21【P1】（`ce3c9c2`）**：等体积改动内容摘要保护。启动快照
  原来只记体素计数，删 1 体素补 1 体素的修正（GUID 命中、计数不变）
  被判"未修改"而遭 AI 结果覆盖。两监控（flexict/nnunet）的快照统一下沉
  到共享 `mimics_mask_apply._mask_snapshot`，增加内容摘要 `sha256`——
  按交接设计约束（与 F13 联合，不在每次 tick 拷贝全部 Mask），复用
  nnInteractive 先例：零拷贝 `buffer_byte_view` + 16MB 分块
  `stream_buffer`，块间 `_update_gui()` 泵 GUI。完成时 GUID+计数命中后
  比对摘要，不一致 → 放弃更新、落副本；任一侧摘要不可得（旧快照/buffer
  不可读）退化为原计数检查，不误拦。T21：flexict
  TestEqualVolumeEditProtection 6/6、nnunet MimicsRuntimeTests 12/12
  （合计 205 passed）；冒烟 8/8（`20261002T013746`）。
- **R65-6 = F02+F20【P1】（`fc0179f`）**：AL 叠加事务化。对照 HEAD 核实
  F02 真实存在——`_al_finish_conversion` 先 `shutil.rmtree(bridge_root)`
  再读其中的 u8 buffer，真实 apply 必 FileNotFoundError（旧测试 mock
  `_set_mask_from_u8` 把它遮住；交接文档据此提醒 F05 被掩盖）。修复：
  转换结果先整体校验（每 band buffer 存在且字节数恰为 shape 乘积）、
  再应用、任一步失败回滚已创建 mask；bridge root 清理移入 `finally`
  作事务最后一步。F20：发起时捕获 `target_identity`（项目路径 + 活动
  图像 GUID），完成时复核——项目/图像变了拒绝并给指引，绝不写错病例。
  T02/T20：TestFlexiCTActiveLearningApply 6/6（回滚、截断拒绝、项目
  切换拒绝）；冒烟 8/8（`20261002T013746`）。
- **R65-8 = F03+F14【P1/P2】（`b180e3a`）**：AL 审查窗口两个反馈缺陷。
  F03——`_job_selected` 引用的 `QtWidgets`/`QtCore` 只在其他函数局部
  作用域绑定，出现真实 ranking 即 NameError（空列表测试从未进循环）；
  方法开头从 qt_modules 元组绑定，前景色用 `self.QtGui`。F14——
  `_poll_requests` 先写 applied/failed 结果再调 `_job_selected()`，摘要
  文字把结果覆盖，标注者永远看不到失败原因；改为先刷表格（摘要抽为
  `_refresh_status_text`）后写结果。T03/T14：gui_smoke
  TestActiveLearningWindow 4/4；冒烟 8/8（`20261002T013746`）。
- **R65-7 = F28【P1】（`7e10dd8`）**：批量发布版本冲突检测。评审证据
  复现——目标已被前台新保存时 staging+replace 直接覆盖。资源锁只约束
  本仓库脚本（Mimics 原生保存/其他工作站无法排他），按交接"不能排他时
  发布新副本"落地：开工记录输出身份（共享
  `runtime_common.capture_output_identity`，mtime_ns+size），发布前复核
  （共享 `publish_conflict`），不一致 → 保留较新文件、staging 落为
  `<case>.conflict.<pid>.mcs` 副本、病例失败并给可见指引。附带同条
  交接指出的 UX 缺口：`--force` 写入配置但 worker 从不读取——现在 force
  真正从源工程重建（旧输出不再当基底），版本复核在 force 下仍生效。
  T28：TestForeignScriptIntegration 10/10（冲突保留新保存+副本、未改
  输出正常替换、全新路径不误判冲突、force 打开 source 而非旧输出）、
  append 流程 1/1；冒烟 8/8（`20261002T023427`）。

### 门禁与证据

- fast 28/28：`20261001T234508`（R65-14/R65-15 全部改动后，本轮收尾
  门禁；flexict_build_dataset 套件首次入矩阵）。
- 全量 test_all：两次后台运行均在 ~65% 处命中已知预存环境崩溃
  （`test_offscreen_window_renders_and_registers`，Qt offscreen 下
  Windows fatal exception: Aborted，干净 HEAD 同样复现，见工作区事件
  登记），无 pytest 汇总；排除该单测的补跑见下条。R39 以来 test_all
  非门禁必需（矩阵 full 才含它），按用户测试纪律只在必要时跑。

### 复核结论（HEAD 已覆盖，无需重做）

- F09 = R61-4（`aadd6e3`）。

### R65-19（F11，P1）：reaper 回调不再触 GUI 对话框（`3fe6e1f`）

审计 runtime_py35 全部 `on_complete=` 调用点（grep + 逐个读回调的传递
callee），确认两处真实违规：mask_import `_finish_mask_import`（→
`_safe_message` → `mimics.dialogs.message_box`）与 mimics_export
`_finish_foreground_export`（同源）——reaper 守护线程里调 Qt 对话框，
经典 Mimics 冻结类问题。修复：on_complete 只置
`monitor.update(reaped=True)`，由仍在运行的 GUI 监视 timer 在下一 tick
于 GUI 线程执行完整 finisher（对话框、租约释放、清理）；60s
reap_deadline 兜底（reap 永不完成时 timer 不无限跳，以可见诊断收尾）。
其余调用点核验为纯文件操作（mimics_import `_finalize` 系列、
stop_background `cleanup_work`）或根本无回调，不动。测试沉淀：
TestGuiThreadBlockingContract 4 新例（AST 契约扫描任何 runtime_py35
模块 on_complete lambda 内禁 mimics.*；两处违规函数钉住正契约；两例
行为测试——假 reaper 置 flag 前对话框静默、置 flag 后下一 tick 收尾）。
附带修复两个预存 TestNewFeatures 失败（干净 HEAD 同样失败，git stash
验证）：R61-24 中文化遗留的英文 radio 断言（改钉"跳过已存在/覆盖已
存在"）；flat-DICOM 探测测试过度指定"恰好一次 scandir 拉取"——根因
是 R61-9（`d2e0727`）在 walk 之后插入了 `_looks_like_dataset_root`
第二次封顶扫描（也在首个切片停止，产品行为正确），改为钉住真契约
"每次 scandir 扫描不越过第一个切片"（每扫描计数，1–3 次扫描合法）。
证据：TestNewFeatures + TestGuiThreadBlockingContract +
TestForeignScriptIntegration 137 通过；冒烟 8/8（`20261002T041554`）。

### R65-11（F26+F27，P1/P2）：job 身份贯通 + 取消不丢台账（`2d39b2f`）

对照 HEAD 核实两项均在：F26——inspect launcher 读固定 `--report` 路径，
新进程先死时旧 completed 报告被当成本轮成功（秒级 job_dir 戳双击也撞
同名）；F27——sync/append runner 在 stop 检查处提前 return，`results.json`
/`failed_cases.json` 只在正常收尾写。修复（对齐 append_jobs 既有
uuid 模式）：launcher 铸 `job_id = <stamp>_<uuid8>`，config→runtime
贯通，runtime 每次报告/状态写入盖 `job_id`，固定路径旧报告启动前归档
为 `.history.<stamp>.json`，终判要求本轮 `job_id`（不匹配 → 退出码 1
+ stderr 说明）；sync launcher 同步加 uuid 目录与身份终判。F27：取消
改 flag+break，台账写移入 `finally`（停止/崩溃/正常收尾全路径发布），
cancelled 状态保留 `next_case`。测试沉淀 4 例（取消后台账与状态、
job_id 贯通 status、runtime 报告带 id、launcher 源码契约：归档 +
uuid + 身份终判）。证据：TestForeignScriptIntegration 14/14、
append/stop 流程绿、冒烟 8/8（`20261002T123216`）。

### R65-16（F06/F07/F08/F12，W05 部署组）：环境修复真实性（`c024e36`）

对照 HEAD 核实四项缺陷均真实存在。F07：setup_env 的修复命令按导入名
拼写 pip 包名（`pip install yaml` 必失败），新增 `_to_pip_names()`
（`_IMPORT_TO_PIP` 反查）在安装与所有用户消息前翻译（yaml→pyyaml、
onnxruntime→onnxruntime-gpu）。F08：安装校验用 find_spec——包存在但
二进制坏（Windows DLL 加载失败）也判 ok，体检"假全绿"；改为逐包真实
`__import__`，`all_ok` 要求零错误，坏包进 `broken_imports` 并提示
DLL 问题。F12：torch/torchvision 分开安装且无索引优先级，可解析出
CPU 版或错配版；改为钉版对（torch==2.6.0+cu124 +
torchvision==0.21.0+cu124，机器已验证组合）+ 显式 `--index-url`，
并排除出批量 `--upgrade`。F06：FlexiCT 推理重建网络时强制加载预训练
backbone，未装 backbone 文件的机器上迁移模型不可用；
`flexict_infer_environment` 设 `FLEXICT_SKIP_BACKBONE=1`，trainer 跳过
backbone 初始化（推理时 Predictor 随即加载完整训练 checkpoint）。
测试沉淀：TestEnvironmentRepairSemantics 4 例；顺带修复 5 个预存
test_all 失败（3× 过期 `_record_failed_case` stub 缺 R63 的
`source_image` kwarg；2× R61-24 中文化后源码断言过宽——改钉模块级
源码契约）。证据：flexict_integration 105/105、相邻类 31/31、
冒烟 8/8（`20261002T140530`）。L4 实机（真实 pip 安装/迁移机器）验收
待实机窗口，代码层契约已全钉住。

### 分支迁移（工作区事件，2026-10-02）

用户指正：此前多个会话的 F/R61/R62 系列提交误落在他人分支
`ai-real-data-validation`（该分支自 `al-review-loop@37c8219` 创建）。
已将本人 45 个提交（R61 后半 + R62 + R65/F 系列 + 验证战役 docs）按原
序 cherry-pick 迁至 `al-review-loop`，全部干净重放、零冲突；两个他人
提交（`911776e` R-P1-8、`4e603b0` R-P1-9）留在原分支未动，分支本身
未删改。迁移时未提交的 W05 工作区改动经 stash 随迁。flexict 套件在
迁移后为 105/105（原分支 109 中的 4 例为 R-P1-9 新增，属他人工作未
随迁，数字吻合）。迁移后 al-review-loop 本地领先 origin 56 提交。

### 下一步

W04（R65-5/6/8）、W06 全部（R65-7=F28、R65-19=F11、R65-11=F26+F27）、
W05（R65-16）已关闭。W03 仅剩 R65-12（F01 DICOM 系列/多帧——需 L3
实机配合，验收成本高）；之后按交接顺序 W08（R65-17 = F10+F15 多选
作用域 + 视觉状态、R65-18 = F04+F13）；R65-13 = F05（AL 叠加自动标记
已标注——F02 修复后该行为已真实暴露，`_al_finish_conversion` 成功
路径仍调 `_al_mark_annotated`，需结合产品语义评估）。

### 工作区事件登记

- pytest `-k` 组合跑 TestImportDropWindow 时 Qt offscreen 渲染崩溃
  （Windows fatal exception: Aborted）——在干净 HEAD `2f3ed65` 上同样
  复现，判定为既有环境问题（gate matrix 单套件运行不受影响），非本轮
  改动引入，不入回归。
- **测试卫生（`0276e84`，预存缺陷修复）**：全量 test_all（R39 以来首次）
  暴露 TestCreateMcsBatch::test_record_failed_case 失败——R61-2 新增的
  两个 pruned-fingerprint 测试调用 `main()` 后未恢复模块全局
  `_ACTIVE_RUNTIME_DIR`，泄漏到后续测试使其向已删除的临时目录写入
  （isolation 下通过，类顺序下失败）。两处补 save/restore，
  TestCreateMcsBatch 22/22。属 R61-2（~2026-09-29）引入的预存缺陷，
  非本轮改动回归。
- **测试隔离修复（随 `cd163a0`）**：
  test_copied_model_is_discovered_without_old_registry 未隔离本机全局
  model registry（`~/.mimics_script/nnunet_model_registry.json`）——当天
  真实数据验证战役在 G:\ 注册了一个活模型（epoch 1790857753，18 病例
  remote 训练），测试数出 2 个模型。干净 HEAD 同样失败，判定为预存
  测试缺陷（load_models 设计上就要读全局 registry，测试没 mock）。
- **全量 test_all 崩溃复测（R65）**：bq4bj9ycs 与 bw6elayey 两次全量
  后台运行都在同一测试 `TestImportDropWindow::
  test_offscreen_window_renders_and_registers`（ui_theme.
  configure_application）处 Fatal Python error: Aborted，~65% 进度无
  pytest 汇总。与 4126 条记录的 `-k` 组合崩溃同源——干净 HEAD 同样
  复现的 Qt offscreen 环境问题。已用 `--deselect` 排除该单测补跑
  确认其余全绿（结果见门禁小节更新）。
  按相邻死行测试的既有模式 mock `model_registry_paths` 隔离。
