# AI 功能真实数据有效性验证报告（2026-09-30）

分支：`ai-real-data-validation`（基于 ccbda86）
工作区：`G:\mimics_ai_validation\`（E: 空间不足，模型/工作区放 G:）
数据：`Z:\ImageAnalysisData\1-CT\Segmentation\data\Totalsegmentator_dataset_v201`（只读红线，全程未写入）
远程：10.8.168.206（profile `206_root`，remote_root=`/userdata/shijian_ruan/mimics-ai`，只使用该目录）

验证目标：三个 AI 功能（nnU-Net 多器官训练+推理、FlexiCT liver 少样本微调+推理、
nnInteractive 交互标注+任务微调）各有一份可信的"真实有效"证据链：任务完成 →
训练收敛 → held-out 推理 → 与 Totalsegmentator GT 的 Dice 对比。

## 选例清单（nibabel 逐例校验 4 器官 mask 非零体素）

- 训练 18 例：s0001, s0004, s0006, s0009, s0010, s0011, s0012, s0013, s0014,
  s0015, s0016, s0019, s0022, s0024, s0028, s0029, s0030, s0031
- held-out 4 例：s0038, s0039, s0040, s0042
- **选例方法结论**：按文件大小/存在性筛例不可靠——s0000/s0002/s0003 有标签
  文件但 mask 全空（~43KB 压缩空数组），s0005/s0007/s0008 连标签文件都没有。
  必须加载后数非零体素。

## 外部资源使用登记

| 日期 | 资源 | 操作 | 红线遵守 | 清理 |
|---|---|---|---|---|
| 2026-09-30 | 206 服务器 | 只读预检：GPU（2×A4000 空闲）、磁盘 429GB、目录布局、镜像清单 | 未写任何远端文件 | — |
| 2026-09-30 | Z: 数据集 | nibabel 只读加载 22 例做选例校验 | 未写入 | — |
| 2026-09-30 | 本地 flexict_models | sweep_dataset_retention 清理过期冒烟残留（~0.7GB，可重建中间产物，超 30 天保留期；Dataset758 注册保护未动） | — | 已完成 |
| 2026-09-30 | 206 服务器 | nnU-Net 远程训练 job（本报告 Phase 2 节） | 见 Phase 2 | 见 Phase 6 |

发现待清理：206 残留容器 `mimics-ai-root-train_20260925T183409_50564969`
（5 天前 Exited(1)，非本次验证产生）→ Phase 6 处理。

## Phase 1：真实数据导入（完成）

- 命令：`mimics_batch_cli.py prepare-import`（`01_Import_Dataset.py` 背后同一
  实现），后台真实 Mimics（`D:\Mimics Research 21.0\MimicsResearch.exe`）。
- **结果：22/22 成功，0 失败**；每例 receipt 确认 4 器官 mask 全部创建。
- **耗时 14.8 min（≈40s/例）** —— 账本性能基线"批量导入耗时"首次实测
  （22 例真实 CT，后台 Mimics 复用一个进程）。
- 工件：`G:\mimics_ai_validation\mcs\`（receipt + .mcs 工程）。

## 环境阻塞：206 根分区 100% 满（2026-09-30 21:26 发现）

- **现象**：远程训练到容器启动一步必然失败——k3s/containerd 崩溃循环
  （`/run/k3s/containerd/containerd.sock: connection refused`，systemd 已
  自动重启 261 次，主进程 exit 255）。
- **根因**：根分区 `/dev/sdb3` 100G 用满（剩 20K）。占用大户：
  `/root/.conan2` 37G、`/log/aims-resmgr-agent` 27G、`/opt/_extracted_*`
  18G（多版本安装包解压残留）。/userdata（3.0T，余 429G）正常，
  我方目录仅 695M。
- **处理**：全部在允许路径 `/userdata/shijian_ruan/` 之外，按红线不动，
  需服务器管理员清理根分区（建议：清 /log/aims-resmgr-agent 旧日志、
  /opt/_extracted_* 旧版残留、/root/.conan2 缓存）。
- **影响**：Phase 2 远程训练、Phase 3 FlexiCT 远程训练暂停在远程启动
  一步；两任务的本地阶段（标签导出、数据准备）不受影响照常完成并缓存，
  服务器恢复后重提交即可续跑。验证顺序调整为先做 Phase 4（本机 GPU）。

## Phase 2：nnU-Net 多器官远程训练（进行中，阻塞于 206 磁盘满）

- 任务：liver+spleen+kidney_left+kidney_right 4 标签单模型，
  label_source=**mcs_refresh**（从已保存 .mcs 后台导出 mask——此前从未实机
  验收过的组合），execution_backend=remote，profile 206_root，3d_fullres，
  50 epochs，18 训练例。
- job：`train_20260930T205202_36ca9b92`（workspace
  `G:\mimics_ai_validation\nnunet_workspace`）
- 验收判据：任务 completed；loss 明显下降；validation Dice 上升趋势；模型
  下载注册（mimics_model_manifest.json 远程溯源）。held-out 推理 Dice：
  liver/spleen ≥ 0.7，kidney ≥ 0.5。
- （待补充：训练曲线、耗时、code-drift 警告实际表现、Dice 表）

## Phase 3：FlexiCT liver 少样本远程微调（待做）

- 判据：completed、loss 下降、checkpoint_best 注册；held-out 推理
  liver Dice ≥ 0.6。

## Phase 4a：nnInteractive 官方模型交互标注（完成，全部达标）

- **链路**：完全复刻 Mimics 侧协议——`nninteractive_bridge.py --async-worker`
  文件队列（initialize.json → command_000001.json → prediction_*.u8），
  真实加载官方模型 `python_env/models/nnInteractive_v1.0`，本机 GPU
  （cuda:0，RTX 3060），真实推理服务器
  （nnInteractive.inference.server.main，端口 1527）。
- **交互**：1 个正点（GT 质心体素）+ 1 个负点——模拟标注者最小操作。
- **结果**（4 例 held-out，liver + kidney_left）：

| 例 | liver Dice | kidney_left Dice |
|---|---|---|
| s0038 | 0.8931 | — |
| s0039 | 0.9380 | 0.8859 |
| s0040 | 0.9528 | 0.9646 |
| s0042 | 0.9709 | 0.9646 |

- **判据全部达标**：liver ≥0.8（实际 0.89–0.97），kidney ≥0.6
  （实际 0.89–0.96）。结论：**官方模型交互标注真实有效**。
- **耗时基线**：服务器冷启动 ~300s + 图像加载 ~200s（首例）；服务器
  热状态下单次点 prompt 推理 **1.7–9s**（含首例 34.6s 的 warmup）。
  服务器空闲后由 watchdog 按超时自动回收，无孤儿进程。
- **有效性之外的观察**（记录待入账）：cold-start 到首个可用预测需
  ~9 分钟（服务器启动 5min + 图像加载 3min + 首次推理 35s），标注者
  第一次点 prompt 的等待体验值得在产品层面评估（warm pool/预加载）。
- 工件：`G:\mimics_ai_validation\nninteractive_official\job_*\results\`
  （result_*.json + prediction_*.u8）、`logs\nninteractive_bridge.jsonl`。

## Phase 4b：nnInteractive 任务微调（训练完成，质量门通过）

- **链路**：`nninteractive_finetune_pipeline.py run`（`03_Train_and_Manage_Custom_Models.py`
  Model Center 背后同一实现），prepared 数据源，CLoPA-IN 策略，
  官方模型为基座，本机 GPU。
- **规模**：6 训练例 + 2 验证例（few-shot liver），20 epochs，
  实测 78.9 分钟（RTX 3060）。
- **训练收敛**：loss 从 ~0.30 降至 ~0.15-0.25 区间；validation AUC
  从 0.9725 升至 0.9776（best epoch 16）。
- **质量门（`_model_is_usable`）结果：通过**——
  - baseline（官方模型）AUC 0.9747
  - candidate（微调模型）AUC 0.9779，**delta +0.0032，无严重退化模式**
  - `qualifies: true` → **`new_model_selected`**，模型注册进
    registry.json（checkpoint SHA-256
    `45858d26…`，父模型 official，质量记录完整）
- **历史对照**：此前唯一一次真实 GPU 收敛实验为负结果
  （qualifies: false）。本次为**首次正结果**：few-shot 微调在真实
  CT liver 上带来可测量的提升且质量门正确放行。
- 首跑撞出 P0-2 同款 bug（source-grid 缓存 publish WinError 5），
  已修复（commit b7d4a10）后重跑成功。
- **微调 vs 官方基线直接对比**（同一协议：1 正点 + 1 负点，held-out
  liver，官方基线数字来自 Phase 4a）：

| 例 | 官方模型 Dice | 微调模型 Dice | delta |
|---|---|---|---|
| s0039 | 0.9380 | 0.9539 | +0.016 |
| s0040 | 0.9528 | 0.9660 | +0.013 |
| s0042 | 0.9709 | 0.9841 | +0.013 |

- **结论**：微调模型在同一 held-out 例上稳定小幅优于官方模型
  （3/3 例 delta +0.013~0.016），与质量门的 AUC delta（+0.0032，
  qualifies: true）方向一致。**few-shot 任务微调真实有效且带来可测量
  提升**——项目历史上首次（此前唯一真实实验为负结果）。

## 过程中发现的问题（待按账本协议入账）

> 注：同事对 improvement_backlog.md / round_reports.md 的改动尚未提交，
> 账本条目待其提交后补录，先在此记录全文，避免污染其未提交 diff。

### P0-3 远程训练控制器启动即 NameError（已修复，commit dbb7090）

- **痛点**：远程训练 job 本地阶段全部完成后，容器启动前一刻死于
  `NameError: name 'assert_gpus_not_busy' is not defined`——GPU 忙闲门
  函数未随主导入分支导入，任何远程训练/推理 job 100% 无法启动容器。
- **根因**：`remote_training_controller.py` 的两个导入分支中只有降级
  分支（flat-module）列了 `assert_gpus_not_busy`，实际生效的
  `tools.remote_compute` 分支漏了该名字。
- **修复**：主分支补上该导入；新增源码契约测试钉住两个分支都必须
  包含它（`test_controller_imports_assert_gpus_not_busy_from_both_branches`）。
- **证据**：remote_training 套件 156s 全绿；修复前 job
  train_20260930T211247_4734b4a7 于 18/18 数据准备后死于该错误。

### P0-1 mcs_refresh 训练在锁等待处自取消（已修复，commit 4d90b7d）

- **痛点**：任何 label_source=mcs_refresh 的训练 job（本地或远程）在
  "Exporting selected Masks from saved Mimics projects" 阶段必然以
  `ResourceLockCancelled: Cancelled while waiting for background_mimics lock`
  失败——用户从 Mimics 向导发起的 mcs_refresh 训练 100% 无法启动。
- **根因**：`tools/mimics_label_export.py` 三处以
  `Path(cancel_path).is_file()` 判断取消，但 job 管线传入的是
  control.json——该文件**创建 job 时即存在**（内容 `{"action":"run"}`
  表示运行中），按存在性判断即恒为"已取消"。
- **修复**：新增 `cancel_requested()`：JSON 文件按内容判断
  （action∈{cancel,stop} 才算取消），非 JSON 标记文件按存在性判断，
  缺失不算取消；三处调用点替换。
- **防回归测试**：`tools/test_all.py::test_cancel_requested_control_json_conventions`
  覆盖 5 种约定组合（control.json run/cancel、缺失、旧标记、None）。
- **证据**：修复后同一请求重提交，job 顺利通过锁获取进入真实导出
  （修复前同阶段必失败）；冒烟 flow_imports/flow_export + fast
  nnunet_integration 门禁绿。

### P0-2 staging 目录发布遇瞬时文件占用即整 job 失败（已修复，commit 8ca4e4b）

- **痛点**：第一次真实 mcs_refresh 训练在 preparing_data 阶段（14/18 例）因
  `os.replace` 发布 s0028 缓存目录时 WinError 5（拒绝访问）整体失败——
  Windows 杀毒/索引服务对新写入的 nii.gz 树瞬时持有句柄，几小时级训练
  job 因一瞬占用报废。
- **根因**：4 处发布点（nnunet_pipeline 的 case 缓存/数据集/模型注册 +
  flexict_pipeline 模型注册）用裸 `os.replace`，无重试；而
  `write_json_atomic` 早有同款问题并已带 20 次退避重试——目录发布漏掉了。
- **修复**：`nnunet_common.replace_with_retry`（复用 write_json_atomic 的
  重试形状），替换 4 处调用点。
- **防回归测试**：`test_replace_with_retry_recovers_from_transient_lock`、
  `test_replace_with_retry_raises_after_persistent_denial`（失败发布不丢
  staging 树）。
- **证据**：smoke 门禁 8/8 绿；修复后同请求重提交
  （train_20260930T211247_4734b4a7）越过原失败点。

### P1-1 死进程遗留的 background_mimics 锁不被清扫（未修，待入账）

- **痛点**：宿主 Mimics/进程异常退出后，其持有的 background_mimics 锁
  文件残留；后续 job 等锁时 `sweep_processes` 只释放"进程注册表内有
  记录"的进程锁，死进程不在注册表 → 锁永不释放，下一个导出任务
  只能等锁超时。
- **实测**：验证过程中发现 pids 36812/31828/36144 已死但锁残留，需手工
  删除锁文件才能继续。
- **拟验收标准**：锁等待循环检测到持有者 pid 已死且不在注册表时，
  主动回收该锁（或 sweep 覆盖此情况）；防回归测试进套件。
- **状态**：待按账本协议正式入账后修复。

## 性能基线首次填数

| 基线项 | 数值 | 条件 |
|---|---|---|
| 批量导入耗时 | 14.8 min（≈40s/例） | 22 例真实 CT，后台 Mimics 单进程复用，Z: 网络盘读取 |
| 单例推理耗时 | 待测 | Phase 2 held-out 推理时测 |
