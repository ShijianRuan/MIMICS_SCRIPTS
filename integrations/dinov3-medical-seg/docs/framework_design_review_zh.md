# DINOv3 医学分割框架设计复核

## 结论

当前公共训练界面收敛为少量有明确含义的选择，底层仍保留旧模型兼容能力：

| 用户选择 | 新模型实际实现 | 推理行为 |
|---|---|---|
| Auto | 根据训练数据 spacing 指纹选择 2D 或 3D | 固化训练时解析结果 |
| 2D | 四层 DINOv3 特征 + 2D 伪金字塔解码 | 逐层预测并还原 3D 体积 |
| 3D Standard | 多层特征融合 + 低分辨率 3D 上下文 | 全体积或滑窗由采样策略决定 |
| 3D High detail | 2D 边界恢复分支 + 轻量 3D 连续性分支 | 两个分支的 logits 学习融合 |
| Frozen | 冻结 DINOv3，仅训练解码器 | 加载同一 backbone 和解码器 |
| LoRA | 在每层 attention 的 Q/V 投影插入 LoRA | 加载 LoRA 与解码器权重 |

这不是声称某个组合对所有器官都最佳。Auto 是安全起点，最终精度仍需用固定验证集比较。

## 设计依据

1. [DINOv3 官方实现](https://github.com/facebookresearch/dinov3)把 ViT/16 作为高质量稠密特征提取器，并为 LVD 权重给出 ImageNet mean/std 归一化。框架先把医学图像映射到 `[0,1]`，再使用同一 ImageNet 归一化进入 backbone。
2. [SegDINO](https://github.com/script-Yang/SegDINO)和 [SegDINO v2](https://github.com/script-Yang/segdino_v2)都利用不同 transformer 深度的特征，而不是只读取最后一层。2D 公共解码器按 `H/2、H/4、H/8、H/16` 构造伪金字塔并自顶向下融合。
3. 12 层 ViT-S/B 使用 `[2,5,8,11]`，24 层 ViT-L 使用 `[4,11,17,23]`；更深模型按同样相对深度扩展。模型加载时会对照 `config.json` 校验层号，避免模型尺度与解码器输入静默错配。
4. [DinoUNet](https://github.com/yifangao112/DinoUNet)建立在 nnU-Net 式医学分割流程上；[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)强调数据指纹、2D/3D 配置、预处理、训练和推理的一体化。本框架只把指纹用于起始规划，不把启发式规则包装成已验证结论。
5. [DINO-Med3D](https://arxiv.org/abs/2606.18886)指出 2D foundation backbone 用于体数据时需要切片上下文、跨层连续性和边界细节恢复。3D High detail 因此采用双分支，但不在高分辨率上保存大通道 3D feature volume，以适应 RTX 3060 级显存。

## Mimics 到训练和推理的同一契约

训练链路：

1. 从原始病例目录读取 image。
2. 将来自原始目录、导出目录或 `.mcs` 的 Mask 映射到原图物理网格。
3. NIfTI 只做 canonical RAS 轴置换和翻转；发生 spacing 重采样时，image 和 label 使用同一变换。
4. 根据模态执行强度处理。
5. 按 `height,width` 等比例缩放并补边到 DINOv3 输入尺寸。
6. 复制或构造三通道，再执行 DINOv3 官方 ImageNet 归一化。

推理读取注册模型自己的 `config.yaml` 和 input contract，重复步骤 3 至 6；预测先去除补边，再还原到源图网格，最后由 Mimics bridge 映射到当前 Mimics image grid。推理不会根据新病例重新选择训练维度、decoder、窗口或 resize 方式。

## 模态与强度

GUI 默认使用 `Automatic from dataset`：

- 优先读取导入时写入 `dataset_manifest.json` 的 `source_modality`。
- DICOM 使用 Modality 元数据。
- 规范的 `ct.nii.gz`、`mr.nii.gz`、`mri.nii.gz` 可作为次级线索。
- 无可靠信息时使用 Other 的逐病例稳健 percentile，不冒充 CT。
- 同一训练任务同时检测到 CT 和 MR 时停止并要求拆分或人工指定。

CT 默认固定窗口 `[-1024,1024]`；MR/Other 默认使用图像自身 `0.5%–99.5%` percentile。两者随后都进入 ImageNet 归一化。解析后的模态和参数会写入可迁移模型配置，不依赖训练机器路径。

## GUI 约束关系

- 2D：3D quality 和 sub-volume 自动禁用。
- 3D High detail：解析为双分支 decoder；Standard 解析为轻量 3D decoder。
- Frozen：LoRA rank/alpha 禁用。
- LoRA：rank/alpha 启用，并在启动时校验每层 Q/V 是否成功插入。
- Patch：训练使用 patch；推理自动使用滑窗。
- Full volume：推理自动走整卷，不额外让用户选择。
- Batch size `Auto`：按 GPU 和架构保守建议；显式数值原样传给本地或远程训练。
- Sub-volume 与 batch size 大于 1 不允许组合，避免含义和内存行为冲突。
- 公共 Loss 只保留 Automatic、Dice+Focal、Dice+CE；纯 CE 仍可读取旧实验配置，但不再作为新任务推荐项。

## 建议消融顺序

不要对所有选项做笛卡尔积。每个任务固定同一 5-shot 训练集、验证集和随机种子，按下面顺序淘汰：

1. **维度筛选**：2D 对 3D Standard；Auto 只作为规划准确性对照。
2. **3D 结构**：仅当 3D 胜出或边界指标不足时，比较 Standard 与 High detail。
3. **适配方式**：在胜出结构上比较 Frozen 与 LoRA。
4. **采样方式**：大器官比较 Full 与 Adaptive；小/细长目标比较 Patch 与 Adaptive。
5. **输入尺寸与显存**：只在胜出组合上比较 224、256、320；记录真实 batch、峰值显存和耗时。

至少报告 Dice、HD95、surface Dice、空预测率、单病例推理时间和峰值显存。使用三个 seed 或三个独立 fold 后再决定默认策略；单个病例或单个 seed 只能用于排除崩溃和明显塌陷。

## 尚未由代码保证的部分

- 数据指纹只能选择合理起点，不能保证 Auto 比强制 2D/3D 更准。
- 新 3D High detail 已通过真实 DINOv3 前向/反向和形状验证，但医学任务精度仍需消融实验支持。
- CT 固定窗口是跨病例复用的稳健默认值，不等价于器官特定最佳窗。
- LoRA Q/V 是受控参数高效方案，不代表 rank 8 对所有任务最优。
