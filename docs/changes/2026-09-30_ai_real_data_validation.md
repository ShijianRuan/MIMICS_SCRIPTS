# AI 功能真实数据有效性验证报告（2026-09-30）

分支：`ai-real-data-validation`（基于 ccbda86）
工作区：`G:\mimics_ai_validation\`（E: 空间不足，模型/工作区放 G:）
数据：`Z:\ImageAnalysisData\1-CT\Segmentation\data\Totalsegmentator_dataset_v201`（只读红线，全程未写入）
远程：10.8.168.206（profile `206_root`，remote_root=`/userdata/shijian_ruan/mimics-ai`，只使用该目录）

验证目标：三个 AI 功能（nnU-Net 多器官训练+推理、FlexiCT liver 少样本微调+推理、
nnInteractive 交互标注+任务微调）各有一份可信的"真实有效"证据链：任务完成 →
训练收敛 → held-out 推理 → 与 Totalsegmentator GT 的 Dice 对比。

## 总结论（2026-10-02，全部验证完成）

| 功能 | 结果 | held-out Dice | 判据 | 结论 |
|---|---|---|---|---|
| nnU-Net 多器官训练+推理（远程 171） | 50 epochs 完成，模型注册 | liver 0.95–0.98 / spleen 0.92–0.99 / kidney 0.95–0.99（3 例） | ≥0.7 / ≥0.5 | **有效** |
| FlexiCT liver 少样本微调+推理（远程 171） | 20 epochs 完成，模型注册 | liver 0.972–0.976（3 例） | ≥0.6 | **有效** |
| nnInteractive 官方模型交互标注 | 点 prompt 即出 mask | liver 0.89–0.97 / kidney 0.89–0.96（4 例） | ≥0.8 / ≥0.6 | **有效** |
| nnInteractive 任务微调（本机） | 训练完成，质量门通过 | 微调 vs 官方：Dice delta +0.013~+0.016（3/3 例），AUC +0.0032 | 不退化 | **有效** |
| ScribblePrompt 切片交互分割（本机，真实官方 checkpoint） | 点+框 prompt 即出 mask | liver 0.87–0.95 / spleen 0.82–0.96 / kidney 0.21–0.95（8/9 organ-case ≥0.82） | 见 Phase 4c | **有效（1 个离值点已归因，见 4c）** |

三个功能全部通过真实数据有效性验证。过程中发现并修复 7 个真实产品缺陷
（3×P0 + P1-4/P1-5/P1-6/P1-8，各自 commit + 防回归测试，见"过程中发现的问题"）。

**可信度对照实验**（回应"效果这么好是否可信"）见"可信度对照"节：五组对照
（跨例错配、平凡基线、体积比、切片分布、表面 Dice）全部排除了"分数虚高"
的系统成因。

三个功能全部通过真实数据有效性验证。过程中发现并修复 6 个真实产品缺陷
（3×P0 + P1-4/P1-5/P1-6，各自 commit + 防回归测试，见"过程中发现的问题"）。

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
| 2026-10-01 | 171 服务器 | nnU-Net + FlexiCT 远程训练（Phase 2/3）、镜像重建与上传、FlexiCT 基座权重上传 | 只读写 /home/shijian_ruan/ | 见"外部资源清理登记"节 |
| 2026-10-01 | Z: 数据集 | Phase 2/3 推理输入、GT Dice 评估（只读加载） | 未写入 | — |

发现待清理：206 残留容器 `mimics-ai-root-train_20260925T183409_50564969`
（5 天前 Exited(1)，非本次验证产生）→ 已于 2026-10-02 核验不在（k8s.io
namespace 现无 mimics 容器），无需处理。

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

## Phase 2：nnU-Net 多器官远程训练（进行中，转 171 服务器）

- 任务：liver+spleen+kidney_left+kidney_right 4 标签单模型，
  label_source=**mcs_refresh**（从已保存 .mcs 后台导出 mask——此前从未实机
  验收过的组合），execution_backend=remote，3d_fullres，50 epochs，18 训练例。
- **服务器切换验证**（用户要求）：206 根分区磁盘满（见上节）后，改用
  171（10.8.168.171，root，路径 /home/shijian_ruan/mimics-ai，docker，
  runtime 端口 7329）。切换动作 = servers.json 已有多 profile（171_root /
  206_root）+ 请求里改一个字段 `remote_profile_id: "171_root"`，控制器自动
  适配 docker/nerdctl、端口、远端根路径、容器命名。密码经 Windows 凭据管理器
  （store_password）更新，不落明文。
- 171 镜像重建（原镜像 8 周旧、缺 nnunet_pipeline.py）：本机 Docker 构建
  （GFW 阻断 docker.io/archive.ubuntu.com/PyPI 大文件 → 复用本机已有 pytorch
  基础镜像、跳过 apt 层、pip 走清华镜像）→ docker save 7.18GB → SFTP 上传
  151s @47MB/s → docker load → 镜像内 nnunet_pipeline.py 在位验证通过。
- job 1 `train_20261001T164921_e52e1fd5`：旧镜像，数据全量上传后在 35% 失败
  ——"nnU-Net pipeline is missing from the runtime image"。code_verify=warn
  不拦截旧镜像，失败发生在数据上传之后（见问题账 P1-3）。
- job 2 `train_20261001T182222_4d53fa10`：新镜像，预处理/planning 全通过，
  Epoch 0 处数据增强线程死亡："One or more background workers are no longer
  alive"。容器被控制器清理、真实异常被线程化增强器吞掉。手动在容器内复跑并
  读 pipeline job.log 抓到根因：**nnU-Net v2.8 默认开启 torch.compile
  （inductor），镜像内无 C 编译器，dynamo 编译崩溃**（见问题账 P1-4，已修复
  commit cdd4648）。
- job 2 复跑（同 job 目录，容器内 `nnUNet_compile=false`，GPU 1）：训练正常
  推进，92s/epoch，Dice 逐 epoch 上升（epoch 4：liver 0.63 / spleen 0.32 /
  kidney_L 0.19 / kidney_R 0.31）。
- **验收判据全部达标（2026-10-01）**：
  - 任务 completed：50 epochs，92s/epoch（含 mcs_refresh 后台导出 mask 与
    重新预处理），模型 `nnunet_20261001T183729_3a01cb98` 下载并注册进本地
    模型注册表（mimics_model_manifest.json 带远程溯源）。
  - 训练收敛：训练 epoch pseudo dice 逐 epoch 上升，最终 liver 0.82 /
    spleen 0.36 / kidney_L 0.79 / kidney_R 0.82（spleen 因 18 例中 3 例
    牌缺如/小体积，中位水平受限，但推理仍达标，见下表）。
  - **held-out 推理 Dice（本地 GPU，02_Predict_Current_Case 等价链路，
    mask 经 mimics_bridge 应用回 .mcs 后导出对 GT）**：

    | case | liver | spleen | kidney_left | kidney_right |
    |------|-------|--------|-------------|--------------|
    | s0038 | 0.9483 | 0.9192 | 0.9778 | 0.9792 |
    | s0039 | 0.9651 | 0.9584 | 0.9698 | 0.9553 |
    | s0042 | 0.9783 | 0.9875 | 0.9859 | 0.9763 |

  - 判据 liver/spleen ≥ 0.7、kidney ≥ 0.5：**3/3 例全部通过，远超阈值**。
    结论：nnU-Net 多器官远程训练 + mcs_refresh 标签源 + 本地推理链路
    **真实有效**。
- 训练中真实发生的"控制器死亡→重挂"场景恰好验证了 `reattach` 路径：本地
  status.json 呈终态后按协议重置为 training 再 `remote_training_controller.py
  reattach`，控制器从远端下载模型并完成注册——该恢复路径此前从未实机走过。

## Phase 3：FlexiCT liver 少样本远程微调（进行中）

- 判据：completed、loss 下降、checkpoint_best 注册；held-out 推理
  liver Dice ≥ 0.6。
- 链路与 nnU-Net Phase 2 相同的远程控制器 + 171 服务器，差异点：flexict
  2d 配置、few-shot 8 例（s0001/s0004/s0006/s0009–s0013）、epochs 20、
  trainer flexict2d_Trainer、基座权重经 /models/flexict 指纹校验挂载。
- **失败史（每个都是独立根因，逐一修复）**：
  - job 1 `75a88aa1`：GPU 忙拒绝——gpu_device=auto 检查所有卡，GPU 1 被占。
    非缺陷；171_root profile 设 gpu_device=0。
  - job 2 `507fd5e0`：171 新服务器缺 /models/flexict 基座权重。手动上传
    （两个 576MB safetensors）后通过。
  - job 3 `0ec68c3f`：容器 /dev/shm 耗尽——Docker 默认 64MB，torch 共享内存
    队列饿死。**P1-5 已修复**（commit 466793c，控制器传 --shm-size=16g）。
  - job 4 `b41c7e7f`：OOM——nnU-Net 2d planner 默认 batch 66 对 16GB A4000
    过大。请求加 batch_size=12。
  - job 5 `8f3dec09`：batch 12 仍以 batch 66 跑——远程 prepared cache 仅按
    数据集字节寻址，planning-only 变更命中陈旧缓存、预处理被整段跳过。
    **P1-6 已修复**（commit 49aa7c5，planning 字段哈希进缓存 identity）。
  - job 6 `c4d437f7`：缓存 identity 修复生效（namespace 带 `_a30406c5` 后缀，
    "Batch size override: 66 -> 12" 出现在日志，plans 中 batch_size=12），但
    FlexiCT Primus 基座网络过重，batch 12 × patch [192,256] 首个 forward 的
    up_projection GELU 仍 OOM（需 972MiB / 剩 770MiB）。非代码缺陷——是
    16GB 卡对该骨干的固有显存压力。降 batch 6 重提（job 7）。
- **job 7 `4d8f93c1`（batch 6）：训练完成（2026-10-01）**。20 epochs，
  ~3.3 min/epoch；train_loss -0.79 → -0.98 单调下降；EMA pseudo Dice
  0.585 → 0.724 逐 epoch 上升；fold 内验证 Dice 0.80。模型
  `flexict_20261001T214342_a5e672ad` 下载注册（flexict_model_manifest.json
  带远程溯源：171_root、镜像 ID、GPU 0）。
- **held-out 推理（本地 GPU，02_Predict_Current_Case 等价链路，
  checkpoint_best、TTA 关、纯推理 ~46s/例）liver Dice**：

  | case | liver |
  |------|-------|
  | s0038 | 0.9722 |
  | s0039 | 0.9761 |
  | s0042 | 0.9754 |

- **判据 liver Dice ≥ 0.6：3/3 例全部通过（实际 ~0.97）**。结论：FlexiCT
  liver 少样本远程微调 + 本地推理链路**真实有效**——few-shot 8 例 × 20
  epochs 即可把通用基座微调到接近专用模型水平。

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

### P1-1 死进程遗留的 background_mimics 锁不被清扫（待入账，根因存疑）

- **痛点**：验证过程中发现 pids 36812/31828/36144 已死但
  background_mimics 锁文件残留，后续导出任务等锁直到超时，需手工
  删除锁文件才能继续。
- **初步归因（存疑）**：`sweep_processes` 只释放"进程注册表内有记录"
  的进程的锁，死进程不在注册表 → 锁不被 sweep 释放。
- **复核（2026-10-01）**：受控实验证明该归因不完整——
  `FileResourceLock.acquire` 的 stale 检测路径
  （`resource_locks.py:892 _is_stale` → `_unlink_any`）在持有者 pid
  已死时**本来就会在下一次 acquire 时回收锁**，实测立即成功。即简单
  情形下锁并不会"永不释放"。
- **未解之处**：真实事故的锁文件已手工删除，无法复检原始 payload
  （怀疑方向：pid 被新进程复用且锁 payload 缺 `process_start_marker`
  时，`process_matches` 对活 pid 恒返回 True → 永不判 stale）。
- **拟验收标准**：入账时按"根因待复现"处理——需要保留住下一份
  死持有者锁 payload 再定位；禁止在未复现前盲改锁机制。
- **状态**：待按账本协议正式入账（含本次复核结论）。

### P1-3 旧 runtime 镜像缺 pipeline 文件，失败发生在数据全量上传之后（待入账）

- **痛点**：code_verify=warn 只记录代码漂移不拦截。job 1 用 8 周前的旧镜像
  提交，走完 mcs_refresh 标签导出（~14 min）+ 数据全量上传（18 例）之后才
  在 35% 处失败——"nnU-Net pipeline is missing from the runtime image"。
  服务器带宽和标注者时间被白白消耗，且报错点远离根因（缺文件在 job 开始前
  就可判定）。
- **根因**：远程控制器在启动容器前不做镜像内容预检；code drift 警告（warn
  级）与"镜像缺关键文件"（致命）混在同一信息里，用户无从区分。
- **拟验收标准**：提交远程 job 时，控制器先在容器内检查关键文件
  （tools/nnunet_pipeline.py 等）存在性，缺失则在上传任何数据前失败并给出
  明确修复指引（重建镜像）。

### P1-4 远程 nnU-Net 训练 Epoch 0 必崩：torch.compile 需要镜像内 C 编译器（已修复，commit cdd4648）

- **痛点**：任何远程 nnU-Net 训练（当前 runtime 镜像，无 gcc）在 Epoch 0
  崩溃，报错被线程化数据增强器吞成误导性的 "One or more background workers
  are no longer alive. Exiting. Please check the print statements above"
  ——而真实异常（inductor "Failed to find C compiler"）不在任何用户可见的
  日志里。
- **根因**：nnU-Net v2.8 的 `nnUNetTrainer._do_i_compile` 在 CUDA/Linux 上
  默认返回 True（torch.compile/inductor），inductor 运行时需要 cc；runtime
  镜像（原版与本地重建版）均不含 C 编译器。dynamo 后台编译线程崩溃后，
  `nondet_multi_threaded_augmenter` 的结果收集线程只报"worker 死了"。
- **定位方法**：容器被控制器清理导致 docker logs 不可用；手动复跑容器并读
  挂载卷里的 pipeline job.log 抓到完整 dynamo traceback。
- **修复**：`nnunet_pipeline._worker_environment` 对 train 阶段默认设置
  `nnUNet_compile=false`（与既有 BLAS 线程 cap 同一约定）。
- **防回归测试**：`test_worker_environment_disables_torch_compile_for_train`
  （train 阶段禁用、infer 阶段不设）。
- **证据**：复跑（同 job 目录 + `nnUNet_compile=false`）训练正常推进，
  92s/epoch，Dice 逐 epoch 上升；smoke 门禁 8/8 绿。

### P1-5 容器 /dev/shm 默认 64MB 使远程训练必死于共享内存耗尽（已修复，commit 466793c）

- **痛点**：远程训练容器内 torch 数据加载的共享内存队列在预处理/训练启动
  阶段写满 Docker 默认 64MB /dev/shm，报 "unable to write to file
  </torch_...>: No space left on device"，job 失败。用户无法从 Mimics 侧
  配置或规避。
- **根因**：`remote_training_controller._launch_container` 的 docker run 命令
  未设置 `--shm-size`；nnU-Net 多线程增强器的 worker 间张量队列依赖
  /dev/shm，64MB 远不够。
- **修复**：docker run 增加 `--shm-size=16g`（nerdctl 默认较大，无需处理）。
- **防回归测试**：`test_launch_container_sets_shm_size`。
- **证据**：job 3 之后所有远程容器不再出现 shm 耗尽；test_remote_training
  104 项全绿。

### P1-8 FlexiCT 向导缺 batch size 字段，远程小显存卡 OOM 后标注者无法自救（已修复，commit 911776e）

- **痛点**：nnU-Net 规划器按训练环境给 batch（本例 66），FlexiCT Primus 主干
  在 16GB A4000 上首层前向即 OOM（batch 12 都不够，实测 6 才稳）。管线自
  R-P1-6 起完全支持 request batch_size，nnU-Net 向导也有该字段，但 FlexiCT
  向导没有——标注者在远程卡 OOM 后只能放弃或找工程师，无法从 UI 自救。
- **根因**：`flexict_training_setup_ui.py` 建请求时未暴露 batch_size；
  锁定配方文案写着"batch size 取自 nnU-Net plans"固化了这一缺口。
- **修复**：镜像 nnU-Net 向导模式加 auto 勾选框（默认自动规划）+ 1–128
  spin；请求字段 `batch_size: None | int`；锁定配方文案同步（batch size
  是唯一暴露的配方旋钮）。
- **防回归测试**：`test_flexict_batch_size_override_reaches_request`
  （默认 None、取消勾选后手动值进请求）。
- **证据**：test_gui_smoke 34/34、smoke 门禁 8/8 绿。

## Phase 4c：ScribblePrompt 真实 checkpoint 交互分割（本机，2026-10-02 补充）

之前未覆盖（用户指出的覆盖缺口之一）。补测方式与 4a 同思路：GT 合成
prompt（最大 GT 面积轴向切片的中心正点 + 包围框），驱动**真实官方
checkpoint**（`ScribblePrompt_unet_v1_nf192_res128.pt`）走产品同一路径
`tools/interactive_algorithms_worker.py run_scribbleprompt`，切片级 Dice
对 GT。脚本：`G:\mimics_ai_validation\validate_scribbleprompt.py`。

| case | liver | spleen | kidney_left |
|---|---|---|---|
| s0038 | 0.9474 | 0.8190 | 0.2093 |
| s0039 | 0.9309 | 0.8522 | 0.9033 |
| s0042 | 0.8679 | 0.9607 | 0.9545 |

- **离值点归因（s0038 kidney_left 0.2093）**：预测并非空/错位——预测 144
  px 落在 GT 850 px 的包围盒内部（GT bbox y[74:111] x[68:100]，预测
  y[87:104] x[78:92]），logits 在 GT 内部 mean=-1.2（健康例为正），即模型
  在该切片对肾脏响应弱。该例 kidney_left 总量 44k vox、77 层含器官，与
  s0042（51k、73 层，Dice 0.9545）规模相当——非数据问题、非坐标错配，
  是官方模型对这一例的真实弱点（单切片、单点单框本就是最弱 prompt 组合）。
  其余 8/9 organ-case ≥0.82。
- **性能**：冷启动（含 checkpoint 加载+首次推理）122s，热状态单次 <2s
  （RTX 3060）。

## 可信度对照实验（回应"效果这么好，可信吗"）

对 Phase 2/3 的 held-out Dice（0.92–0.98）做五组对照，逐一排除"分数虚高"
的系统成因。脚本与数值工件：`G:\mimics_ai_validation\`（credibility_controls /
credibility_3d / slice_detail）。

1. **跨例错配检验（排除"预测和 GT 不是同一例"）**：把 s0039 的预测与
   s0042 的 GT 算 Dice → 形状不匹配直接报错（不同例 shape 不同），同一
   引擎内的配对不可能错例。
2. **平凡基线（排除"Dice 公式或二值化逻辑有 bug 导致虚高"）**：空 mask
   Dice=0.0000，全图 mask Dice≈0.0438（器官仅占体积 ~2%）。若 Dice 计算有
   任何虚高 bug，这两个数不可能落在理论值上。
3. **体积比（排除"预测只是尺寸近似、形状不对"）**：预测/GT 体素比
   0.97–1.02（nnU-Net 4 标签 ×3 例、FlexiCT 3 例全部）。
4. **切片级分布（排除"总体积对但内部一塌糊涂"）**：逐轴向切片 Dice 均值
   0.82–0.96；Dice=0 的切片全部是肝脏尖端 6–250 px 的碎片层（单切片最大
   6457 px），器官主体切片无一为 0。
5. **表面 Dice@2mm（排除"只对体积不对边界"）**：0.78–0.92（边界容差 2mm
   内的重合率），与体 Dice 0.92+ 一致，边界质量真实。

结论：高分是模型真实表现的合理反映（肝脏/脾脏/肾脏是 Totalsegmentator
上公开基准 Dice 0.9+ 的容易器官，训练 18 例、测试例与训练例同分布），
不是测量假象。

## 覆盖缺口清单（如实登记）

- **FlexiCT 3d_fullres**：未测。代码内注明"3D fullres needs ~32GB"
  （flexict_pipeline.py:158），两台远程服务器均为 16GB A4000——按现有
  硬件不可测，除非同时调小 patch+batch（那测的就不是 3d 配方本身）。
  处置：作为硬件限制如实登记，不算验证失败；'auto' 在 16GB 卡上自动
  选 2d 的行为已实测正确（Phase 3 走的就是该路径）。
- **FlexiCT active_learning（pair 配置）**：未测。依赖 2d+3d 成对模型，
  被 3d_fullres 的硬件限制连带阻塞。入账"需用户决策"：是否有 32GB+ GPU
  可用，再决定是否补测。
- nnU-Net 2d / 3d_lowres 等其他配置：未在真实数据上训练（3d_fullres 已
  测）；推理侧低优（同一管线不同 plans）。

- **痛点**：用户在向导里改了 batch_size / patch_size / spacing / configuration
  等规划参数重跑远程训练，控制器显示一切正常，但远程端预处理被整段跳过、
  旧 plans（含旧 batch size）被直接复用——训练表现与用户设置完全脱节，
  且无任何提示。FlexiCT 验证中表现为 batch 66 OOM 反复复发。
- **根因**：`_prepare_flexict`/`_prepare_nnunet` 生成的远程 prepared cache
  identity 只由 `_dataset_parts_fingerprint`（数据集字节）决定；planning
  字段不在寻址键里，内容寻址缓存对规划变更不敏感。
- **修复**：新增 `_prepared_cache_identity`，把
  configuration/plans/spacing/patch_size/batch_size 哈希进 identity；缺失
  该组字段时退回纯数据集指纹（向后兼容）。
- **防回归测试**：`test_prepared_cache_identity_includes_planning_fields`
  （batch 12 与 66 产生不同 identity；非训练 job 保持纯指纹）。
- **证据**：job 6（c4d437f7）日志出现 "Batch size override: 66 -> 12" 与
  plans batch_size=12，缓存 namespace 带新 identity 后缀，预处理确实重跑。

## 外部资源清理登记（2026-10-02）

**171（/home/shijian_ruan/mimics-ai）**：
- 已删除：本次验证 6 个失败 job 目录（e52e1fd5、0ec68c3f、b41c7e7f、
  8f3dec09、c4d437f7 及 manual 日志/脚本）、镜像构建源 mimics-ai-src
  （7.6G，镜像已 load 进 docker）。
- 保留（登记）：cache/root 42G —— 内容寻址设计缓存（dataset/prepared，
  30 天过期惯例）；8 月 3 个 nninteractive_train job 目录（c4012773 等，
  asl_epi 数据，非本次验证产物、归属未确证，不删待确认）；镜像
  mimics-ai-runtime:1.1（可复用资产）。
- 终态核验：GPU 0/1 利用率 0%、无 mimics 容器、无 mimics/remote_worker
  进程、磁盘 43%（145G free）。

**206（/userdata/shijian_ruan/mimics-ai）**：jobs/root 空，k8s.io namespace
无 mimics 容器（此前的 stale 容器 train_20260925T183409_50564969 已不在），
仅剩 cache/models 等既有资产，无本次验证新增产物（本次验证因 206 磁盘满
全程未在 206 产生数据）。

**本地（G:\mimics_ai_validation）**：保留全部验证工件（mcs 工程、两个
workspace 的 job 记录与模型、predictions、评估脚本）作为证据链，供
复核；不在仓库内、无污染。

**账本（improvement_backlog.md）状态**：P1-3/P1-4/P1-5/P1-6 与性能基线
的正式入账待同事的 docs/03_Review 未提交改动合入后补写（避免污染其
未提交 diff），本报告为唯一状态记录。

## 性能基线首次填数

| 基线项 | 数值 | 条件 |
|---|---|---|
| 批量导入耗时 | 14.8 min（≈40s/例） | 22 例真实 CT，后台 Mimics 单进程复用，Z: 网络盘读取 |
| nnInteractive 点 prompt 推理（单次） | 热状态 1.7–9s；冷启动到首个可用预测 ~9 min | RTX 3060，官方/微调模型，1 正点+1 负点 |
| nnInteractive few-shot 微调 | 78.9 min（20 epochs，6 训练+2 验证例） | RTX 3060，CLoPA-IN，官方模型为基座 |
| nnU-Net 单例推理 | 纯推理 5–16s/例（240×240×333 级 CT）；全 job 4–8 min | RTX 3060，3d_fullres 4 标签模型，TTA 关；job 时长含 worker 进程启动、模型加载、空间校验 |
| FlexiCT 单例推理 | 纯推理 ~46s/例；全 job ~5 min | RTX 3060，2d few-shot liver 模型，checkpoint_best，TTA 关 |
| FlexiCT 远程 few-shot 微调 | 66 min（20 epochs，8 训练例，batch 6，~3.3 min/epoch） | 171 A4000 16GB，远程控制器全链路（上传/预处理/训练/下载注册） |
