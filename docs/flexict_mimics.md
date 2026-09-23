# FlexiCT few-shot 分割 for Mimics Research 21

## 1. 结论

FlexiCT 是本项目的 few-shot（少样本）医学图像分割框架：用少量标注病例（8 例起）
微调一个 ViT 预训练 backbone，产出 2D / 3D 分割模型，可直接对当前 Mimics 病例
推理并把预测应用为 Mimics Mask。它取代了此前的 DINOv3 few-shot 实现。

三种用法共用同一套代码：

- **Mimics 菜单**（标注者视角，本文重点）：`Script -> Scripting Library -> 02_AI -> FlexiCT`
  - `01 Train Model` — 选择病例与标签，训练模型
  - `02 Predict Current Case` — 用已训练模型预测当前病例
  - `03 Active Learning Review` — 主动学习标注审核
- **独立仓库 CLI**：`integrations/flexict-finetune/`（见其 README，Linux GPU 机器
  零 Mimics 依赖）
- **Python API**：`tools/flexict_pipeline.py` 的 job-dir 模式（创建 job 目录 +
  request.json，运行 main()）

## 2. 双进程结构

与 nnInteractive 集成相同：Mimics 内置 Python 3.5 只做轻量工作（选图像、写
Mask、监控状态），PyTorch 训练/推理全部在外部 `python_env`（Python 3.13）后台
进程中进行，两者通过 job 目录里的 JSON 状态文件与 `mimics_bridge.py` 通信。

```
Mimics (Py3.5)                    python_env (Py3.13)
runtime_py35/flexict_mimics.py    tools/flexict_pipeline.py
  ├─ 启动/恢复监控器                ├─ run_training      (训练 job)
  ├─ 轮询 status.json              ├─ run_inference     (预测 job)
  ├─ 轮询 AL apply_requests/       └─ run_active_learning (AL job)
  └─ mimics_bridge 应用 mask          └─ nnUNet_extTrainer + FlexiCT trainer
```

## 3. 入口与流程

### 3.1 训练（01 Train Model）

1. 在 Mimics 中打开包含标注 Mask 的病例，或直接在训练窗 Data 页选择病例
   （可多选，来自标注流程导出的数据集）。
2. 选择标签名（如 `kidney_left`）——训练用单标签二值化（前景=该 Mask，背景=其余）。
3. 选择 configuration：
   - `2d` — 只训练 2D 模型（12GB GPU 推荐，快）
   - `3d_fullres` — 只训练 3D 模型（显存要求高，见 §6）
   - `pair` — 顺序训练 2D + 3D 两个模型，**主动学习要求 pair**
4. 高级折叠区（一般不用动）：epochs（默认 150）、单侧器官 mirror 关闭、验证
   case 数（默认 3）。

训练配方（学习率、优化器、deep_supervision、TTA、fp32）锁定为验证过的值，
不在 UI 暴露——全肾 2D 0.956 / 3D 0.960 Dice 的成绩即出自该配方。改动配方
属于 `integrations/flexict-finetune/` 仓库变更，见其 README。

阶段：`exporting_labels → preparing_data → preprocessing → waiting_for_gpu →
training → (training_3d) → registering → completed`。模型注册进
workspace `flexict_models/registry.json`，pair 的两个模型共享 `pair_id`。

**远程训练（Compute 页）**：训练窗口的 Compute 页默认 **This workstation**
（本地）。选择已保存的远程服务器后，训练在远程 GPU 容器内执行——同一份
请求、同一套 source-grid 数据准备、同一份锁定配方与模型 manifest，只是
stage worker 在容器里跑。FlexiCT 预训练 backbone 从服务器 `/models/flexict`
只读挂载，上传前做内容指纹校验（与本地 backbone 不一致即拒绝）。训练完成
后 pair（或单模型）下载注册进本地 registry，带完整 remote 溯源字段，后续
预测/主动学习照常本地使用。详见 `docs/remote_training_design.md`。

### 3.2 预测（02 Predict Current Case）

1. 当前病例已打开且有 verified grid（第一次导入时的网格契约）——没有契约时
   拒绝应用，防止把预测贴到错误网格上。
2. 模型下拉列表按推荐模型排序（显示验证 Dice 与日期），默认选 recommended。
3. 输出模式：应用为 Mimics Mask（`FlexiCT <label>`）、保存 NIfTI、或两者。
4. 批量文件夹模式：对文件夹内每个 NIfTI 输出一份预测 NIfTI。

推理固定用 `checkpoint_best.pth`（验证 Dice 选出的）且 TTA 关闭——与验证配方
一致。预测 NIfTI 在 source grid 上，由 `mimics_bridge.py` 重采样到当前 live
grid 后经 mask buffer 事务应用；网格不匹配的预测会被拒绝而不是硬贴。

预测窗口同样带 Compute 选择：默认本地；选远程时当前病例图像与所选模型一起
上传，容器内推理后预测 NIfTI 下载回本地，经同样的网格校验与 buffer 路径
应用——与本地预测完全同一条 Mimics 侧路径。

### 3.3 主动学习（03 Active Learning Review）

前提：一个 **pair**（同 dataset fingerprint 的 2D+3D 模型对）。没有 pair 时
AL 窗会列出缺什么，不会静默降级成单模型。

一轮循环：

1. **准备**：AL job 把未标注池（数据集根里的病例）物化到 `job_dir/input/`，
   逐病例记录 source geometry（`input_geometries.json`）。
2. **双端预测**：2D 和 3D 两个模型分别对全池预测（GPU 锁内，锁定配方）。
3. **不确定度**：`uncertainty/` 模块以 `disagreement` 法逐体素计算两模型分歧，
   `analyze_uncertainty_dir` 按 integrated 分数排序输出 `ranking.csv`。
4. **不确定度带**：把 uint8×10 不确定度图阈值化为两条带状 mask
   （moderate / high），预生成在 `uncertainty/bands/`。阈值自适应数据：
   2 模型 disagreement 的最大级别是 5（1 − 1/2，×10），默认 high=6 永远不
   触发，因此 high 自动 clamp 到数据里实际出现的最大级别，moderate 保持在其下。
5. **Review UI**（外部 PySide6 窗口）：排序表（排名/case/分数/不确定体积/
   状态 new|annotated|skipped，持久化到 `annotation_state.json`）。每行操作：
   - **Open Case** — 系统默认程序打开该病例 NIfTI
   - **Overlay Uncertainty**（双击行同效）— 写 apply request，切回 Mimics 由
     AL monitor 在 Mimics 内应用 high+moderate 两条带 mask
   - **Apply Consensus Mask** — 应用两模型共识 mask（`FlexiCT Consensus`）
   - **Mark Annotated / Mark Skipped** — 更新状态，指导下一轮标注优先级
6. **闭环**：标注完 top case 后把它们加入训练集，重训 pair，再跑下一轮 AL。
   顶部表格每列均可导出 CSV（`annotation_progress.csv`）。

### 3.4 应用请求的握手（为什么要点两次）

Review UI 是 Mimics 外部进程，不能碰 `mimics.*` API。它把请求写成
`job_dir/apply_requests/<case>_<what>_<ms>.json`，Mimics 内的 AL monitor 每
2 秒轮询：校验当前 active image 的 source geometry 与 job 记录一致 → 调
mimics_bridge 重采样 → 事务内应用 mask → 把请求标为 applied/failed。因此点
Overlay 后需要切回 Mimics，monitor 会弹窗告知结果。

## 4. 代码结构

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| 独立仓 | `integrations/flexict-finetune/` | FlexiCT backbone/trainer/uncertainty，零 Mimics 依赖 |
| 配置/注册表 | `tools/flexict_common.py` | 13 键配置、dataset-id 750–799、registry、pair 解析 |
| job runner | `tools/flexict_pipeline.py` | train / infer / active_learning 三类 job |
| 训练 UI | `tools/flexict_training_setup_ui.py` | 三 Tab（Data/Training/Jobs） |
| 预测 UI | `tools/flexict_prediction_setup_ui.py` | 当前 case 预测 + 批量模式 |
| AL UI | `tools/flexict_active_learning_ui.py` | 排序表 + apply request 写入 |
| 状态查看器 | `tools/flexict_status_viewer.py` | job 列表 + 日志查看 |
| Mimics 入口 | `runtime_py35/flexict_mimics.py` | Py3.5 监控器（train/infer/AL request） |
| 菜单脚本 | `scripting_library/02_AI/FlexiCT/` | 三个薄壳入口 |
| 测试 | `tools/test_flexict_integration.py`（56 项）+ `tools/test_flexict_common.py` + `integrations/flexict-finetune/tests/` | 离线测试 |

## 5. Windows 安装

- Python 环境：`python_env/`（Python 3.13 embeddable）已含全部依赖
  （nnunetv2 2.8.0、torch 2.6.0+cu124、safetensors、einops、nibabel、scipy）。
  用 `Admin > Setup Environment`（`tools/setup_env.py`）安装/校验。
- 预训练权重：`integrations/flexict-finetune/weights/flexict_{2d,3d}/model.safetensors`
  （各 ~576MB，fp32）。不入 git、不进 pack 包，需从分发源拷贝。
  `resolve_pretrained_dir` 解析顺序：配置绝对路径 → integrations/weights →
  `R:\` 只读回退。
- nnunetv2 ≥2.8 时 trainer 通过 `nnUNet_extTrainer` 环境变量挂载，无需拷进
  site-packages；≤2.5.2 的安装方式见独立仓 README。

## 6. 已知限制

- **12GB RTX 3060**：3D fullres 的 fp32 ViT 可能 OOM（大卡上 bs2 实测 32GB）。
  auto 配置默认选 2d；需要 3D 时减小 patch size（见独立仓 README）。
- **fp32 锁定**：RoPE 在 fp16 下 NaN，训练/推理全程 fp32。
- **AL 不确定度分辨率**：2 模型 disagreement 只有 0 / 0.5 两档（×10 后为
  0/5），moderate 与 high 带在 pair 模式下覆盖同一批体素——这是二值分歧的
  数学上限，不是 bug；需要更细分辨率时给 pair 加第三端（3 模型即可产生
  1/3、2/3 档）。
- **GPU 冒烟**：训练冒烟（8-case 肾 2D、NUM_EPOCHS=2）与端到端 AL 手动跑
  依赖 GPU 空闲，属人工验收项（见 docs/windows_end_to_end_acceptance 文档
  的同类清单）。

## 7. 验证成绩（参考）

全肾（Totalsegmentator 协议）few-shot：

| 配置 | Dice |
| --- | --- |
| FlexiCT 2D | 0.956 |
| FlexiCT 3D | 0.960 |

配方细节与肝验证见 `integrations/flexict-finetune/docs/validation.md`。
