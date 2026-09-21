# DINOv3 微调框架对照审查与 Mimics 集成说明

## 1. 审查范围

本次审查逐项对照了 `/Users/ruanshijian/research/dino v3` 中的研究记录、
`dinov3seg` 原型、配置与合成数据运行产物，并检查了 Mimics-Script 当前的：

- 数据指纹、样本划分、体积/patch/2.5D 数据路径；
- DINOv3 主干、2D/3D decoder、Frozen/LoRA 训练路径；
- 训练状态、模型注册、推理配置固化和 Mimics 回填；
- 本地及远程训练的共同配置入口。

研究目录中的原型适合验证想法，但不是可以直接替换产品代码的上游实现。
它主要使用合成数据，且 `eval_3d.csv` 将实际写入的 HD95 列标成了
`iou_*`。因此本次只采用能由当前代码、现有实验或确定的软件缺陷支持的改动。

## 2. 当前框架已经具备、无需重复移植的能力

当前 Mimics 框架已经覆盖了研究原型中的主要有效能力，并在以下方面更完整：

1. 支持 Frozen Feature 2D、2.5D 和多种 3D decoder，不只是一种切片堆叠头。
2. 支持 Frozen、LoRA 及历史模型兼容，并把最终架构计划固化到模型清单。
3. 使用数据指纹决定 full/patch、patch 大小和切片方向，支持稀疏目标采样。
4. 训练和推理共用强度处理、canonical RAS、模型 Z/Y/X 和源图恢复约定。
5. 已有 per-run 特征缓存、缓存清理、GPU 锁、取消、状态 JSON 和远程执行。
6. 已有 Dice、HD95、ASSD、surface Dice、lesion 指标和连通域后处理代码。
7. 已有滑窗推理和镜像 TTA 内核；此前只是通用 Mimics 训练界面没有暴露 TTA。

## 3. 本次确认并修复的真实问题

### 3.1 Windows DataLoader 的随机流可能重复

`VolumeAugmentation` 自己持有 `random.Random`。Windows 使用 spawn worker 时，
多个 worker 可能从相同状态开始，产生重复增强。现在每个 worker 会分别设置
Python、NumPy、PyTorch 和 augmentation RNG，同时保持相同训练 seed 可复现。

### 3.2 学习率步数少算梯度累积余数

训练循环会对不足一个完整 accumulation group 的最后几批执行一次 optimizer
step，但 scheduler 原先用向下取整计算总步数。少样本训练很容易触发这个差异。
现在每个 epoch 的有效步数使用 `ceil(batch_count / grad_accumulation)`，与实际
optimizer step 完全一致。

### 3.3 Frozen Feature 2D 的 cosine 更新晚一个 epoch

原先 epoch 结束后才设置该 epoch 的学习率，因此调度整体延后一轮。现在进入
epoch 前设置学习率，第 1 轮使用基础学习率，后续按实际 epoch 进度衰减。

### 3.4 早停在最小观察轮数之前偷偷消耗 patience

原先早期验证未改善也会累计计数，可能一到 `min_epochs` 就立即停止。现在
`min_epochs` 之前只显示验证结果，不消耗 patience。由 Mimics 新建的训练任务还会
明确写入 `patience: 0`，避免继承研究 base config 中不可见的早停配置；用户设置的
Epochs 就是实际训练预算。

### 3.5 通用模型缺少可选镜像平均入口

训练设置新增 **Mirror averaging (slower)**，默认关闭。开启后会在选定视图的
平面内执行原图、两次单轴镜像和双轴镜像，共四次预测并平均。该选项写入最终
模型配置，后续本地或远程推理都会复用同一设置。

Frozen Feature 2D 的已验证实现本来就固定执行四向镜像平均，因此界面会隐藏该
开关，避免重复计算或产生两套互相冲突的设置。

## 4. 暂不移植的内容

以下内容不是被遗漏，而是当前没有足够证据证明适合默认产品路径：

- **DINOv2 自动回退**：会破坏基础权重身份和模型可复现性；离线产品应失败并明确
  提示缺失权重，而不是静默换模型。
- **新增 FPN/UPerNet 或 poly scheduler**：研究目录只有原型/合成运行，没有在当前
  多器官、少样本、3060 约束下的对照结果。未验证选项不加入 Mimics GUI。
- **半监督、伪标签和不确定性采样**：会显著增加数据状态、阈值和质量控制负担，
  当前材料没有临床数据上的有效性与失败边界。
- **跨任务永久特征缓存**：可以减少重复编码，但需要解决权重哈希、预处理指纹、
  并发引用和磁盘回收。当前 per-run 缓存生命周期更可控，不贸然扩大缓存范围。
- **把合成数据分数作为架构结论**：只能证明代码能运行，不能证明器官分割有效。

## 5. Mimics 中的调用链

训练入口为 `02_AI/DINOv3/01_Train_Model.py`：

1. Mimics 入口只启动外部设置窗口，不在 Mimics Python 进程中扫描或训练。
2. PySide6 窗口收集参数，`append_training_args()` 将所有公开选项写入外部命令；
   `mirror_tta` 也包含在 `strategy-options-json` 中。
3. `fewshot_pipeline.py` 校验样本、生成数据指纹、解析兼容架构并写出最终 YAML。
4. 外部 DINOv3 Python 运行训练；状态和曲线通过 JSON/日志增量返回，不阻塞 GUI。
5. 最佳权重、portable `config.yaml`、有效配置和输入契约写入模型 manifest。
6. `02_Predict_Current_Case.py` 或 `03_Predict_Choose_Model.py` 加载同一 manifest，
   所以训练时确定的归一化、方向、滑窗和镜像平均不会在推理时丢失。
7. 预测先恢复到原始图像网格，再由 Mimics bridge 映射到当前打开图像的 buffer。

本地和远程训练共用步骤 2 至 7。远程任务只是更换执行位置，不使用另一套模型
配置，因此该改动不影响现有本地训练，也不制造本地/远程推理差异。

## 6. 验证与仍需 Windows/Mimics 实测的边界

自动测试应覆盖：

- augmentation reseed 的确定性；
- gradient accumulation 有余数时的 scheduler 步数；
- strategy 到 `tta_axes` 的方向映射；
- Mimics GUI 选项到 CLI、YAML 和模型 manifest 的透传；
- 所有公开 2D/2.5D/3D 架构组合仍可生成后端配置。

仍需在目标 Windows 机器验证两点：

1. 使用同一病例分别关闭/开启镜像平均，确认结果方向相同、耗时约为普通推理的
   四倍以内，并记录 Dice/边界指标后再决定是否为特定任务默认开启。
2. 在 Mimics 中完成一次训练、关闭并重新打开项目后推理，确认模型 manifest 的
   portable 配置被读取，Mask 与原图在三视图中重合。

这些是运行环境和真实数据验证，不应通过合成数据分数代替。
