# AI 标注能力边界总报告（2026-10-03）

> 交付对象：标注者与产品决策者。本文回答一个问题：**这套外挂式 AI 标注
> 工具在真实数据上到底能做什么、不能做什么、依据是什么。**
> 全部结论都有可复查的证据链（详见各引用节），无 mock、无粉饰。

依据：`docs/changes/2026-09-30_ai_real_data_validation.md`（真实数据验证
主线报告，1229 例 Totalsegmentator CT + 真实 Mimics + 171/206 远程 GPU +
本机 RTX 3060，2026-09-30 至 10-03 完成）。

## 一、各功能能力边界一览

| 功能 | 已验证能力（真实数据证据） | 边界/未验证 | 结论 |
|---|---|---|---|
| **数据导入**（含批量） | 22 例真实 CT 批量导入 .mcs 工程，~40s/例；异常（缺标签/冲突）有确认门 | 纯 DICOM 网络流、>100 例级批量未测 | **可用** |
| **nnU-Net 多器官训练+推理** | 4 标签（liver/spleen/双肾）50 epochs 远程训练收敛；held-out Dice liver 0.95–0.98 / spleen 0.92–0.99 / kidney 0.95–0.99 | 仅 3d_fullres 配置实测；2d/3d_lowres 未训（同一管线，低优）；训练例仅 9 例（少样本下限场景） | **有效** |
| **困难器官多标签**（pancreas/adrenal/esophagus/gallbladder） | 同管线 4 困难器官训练+推理，大器官判据达标；小器官（肾上腺）结果见专节 | 小器官几何与类别不均衡是固有难点 | **部分有效（如实）** |
| **FlexiCT 少样本微调（单标签）** | 8 例 liver few-shot：2d Dice 0.972–0.976、3d 0.903–0.951（16GB 卡 patch/batch 覆写自救后） | 3d 默认 planner patch 超 16GB，必须手动覆写；非 liver 器官未测 | **有效** |
| **FlexiCT 多分类（≤4 标签）** | 2026-10-03 新增能力：liver+spleen 一模型，liver 劣化 ≤0.004，spleen ≥0.936 | 小器官/4 标签上限场景未测（测试盲区） | **有效（PoC 级）** |
| **nnInteractive 官方模型交互** | 点 prompt 即出 mask：liver 0.89–0.97 / kidney 0.89–0.96 | 通用模型对非常见器官/病灶无承诺；仅本机 GPU | **有效** |
| **nnInteractive 任务微调** | few-shot 微调不退化且小幅提升（Dice delta +0.013~+0.016，3/3 例）；质量门拦截不达标模型 | 提升幅度小（few-shot 天花板）；长期增训价值待 D19 | **有效** |
| **ScribblePrompt 切片交互** | 真实官方 checkpoint 点+框 prompt：8/9 organ-case ≥0.82 | 1 个离值点（kidney 0.21）已归因（含气干扰）；仅本机 | **有效（1 离值已归因）** |
| **远程训练链路（171）** | FlexiCT/nnU-Net 全链路（上传→预处理→训练→下载注册）多轮真实跑通；GPU 忙闲门、重试、看门狗实测 | 206 镜像为 2026-09-26 快照 + code_verify=warn（按原样验证，见主线报告）；mcs_refresh 标签源远程组合已验收（修复 P0-1 后） | **可用** |
| **批量预测** | 命令行 CLI 已有（R61-23）；**Mimics GUI 内无入口** | 需求缺口 D20，需用户决策是否立项 | **缺口已登记** |

## 二、可信度

"效果这么好是否可信"——五组对照实验（跨例错配、平凡基线、体积比、切片
分布、表面 Dice）**全部排除了分数虚高的系统成因**（主线报告"可信度对照
实验"节）。另：所有 Dice 均为 held-out 例（训练未见），TTA 关闭，无 cherry-pick。

## 三、过程中发现并修复的真实缺陷（证据均在主线报告）

- 3×P0：远程控制器启动 NameError（100% 阻断远程训练）、mcs_refresh 锁
  等待自取消（100% 阻断该标签源训练）、以及 2026-10-03 追加的
  **PySide6 6.11 drawPolygon varargs 移除导致所有 GUI 窗口启动路径原生
  abort**（ui_theme._application_icon，潜伏 P0，本分支已修，
  **al-review-loop 主线同样存在需同步**）。
- P1-4/5/6/8/9/10：远程 prepared cache 对规划参数不敏感（改 batch size
  后静默复用旧 plans）、零 mask 无声导入（本轮加确认门）、FlexiCT 权重
  缺失提示不指路 GUI 等——全部修复 + 防回归测试。
- P1-11（all-fail export reads as success）：已登记待入账。

## 四、明确不做/不承诺的

- 小器官（肾上腺级）少样本精度不承诺；困难器官结论见专节。
- nnU-Net 2d/3d_lowres、FlexiCT pair（active_learning）配置未在真实数据
  上走到（覆盖缺口清单已登记，pair 需用户决策）。
- 206 镜像不更新（快照即现有流程）；验证 206 时守共享红线。
- 纯 GUI 向导交互（点击流）不在自动验证范围（gui_smoke 套件另覆盖渲染）。

## 五、给标注者的一句话结论

**三大 AI 系列（nnU-Net 全自动、FlexiCT 少样本、nnInteractive/ScribblePrompt
交互）在真实 CT 数据上均验证有效**：训练能收敛、推理 mask 与 GT Dice 达
标、全链路（Mimics 入口→训练→推理→回传 .mcs）真实跑通。已知边界
（小器官、配置覆盖、批量预测入口）如实登记在案，无隐藏缺口。
