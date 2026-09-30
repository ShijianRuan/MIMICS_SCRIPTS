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

## Phase 4：nnInteractive（待做）

- 4a 官方模型点 prompt 交互标注（本机 GPU）：liver Dice ≥ 0.8、
  kidney ≥ 0.6；mask 经 `_set_mask_from_u8` 链路写回 .mcs。
- 4b 任务微调（本机 GPU，few-shot liver）：微调后点 prompt Dice 不低于
  官方基线；复核 `_model_is_usable` 质量门行为。

## 过程中发现的问题（待按账本协议入账）

> 注：同事对 improvement_backlog.md / round_reports.md 的改动尚未提交，
> 账本条目待其提交后补录，先在此记录全文，避免污染其未提交 diff。

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
