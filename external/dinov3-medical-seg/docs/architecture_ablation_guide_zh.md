# DINOv3 2D/3D 架构与消融指南

## 当前公开训练架构

新任务只公开三个入口意图，具体实现由训练数据指纹解析：

| 选择 | 实际实现 | 可修改项 | 自动锁定项 |
|---|---|---|---|
| Auto | 按最粗 spacing 与最细 spacing 的比值选择 2D 或 3D | Sampling、微调方式、图像尺寸、epoch、学习率、batch | 解码器名称、切片方向、推理方式 |
| 2D | `scale_aware2d`，多层 DINO 特征构造显式 2D 空间金字塔 | Full/Patch、Frozen/LoRA、图像尺寸、训练参数 | 3D quality 固定 Standard；Sub-volume 关闭 |
| 3D Standard | `context3d_lite`，多层语义融合后进行轻量 3D 上下文建模 | Full/Patch、Frozen/LoRA、图像尺寸、训练参数 | 具体 decoder 名称 |
| 3D High detail | `context3d_hybrid`，2D 边界恢复分支与轻量 3D 连续性分支 | Full/Patch、Frozen/LoRA、图像尺寸、训练参数 | 具体 decoder 名称 |

2D 模式仍接收完整 3D 图像。它逐层编码和解码切片，再按原顺序堆叠为 3D
结果。Patch 表示先从 3D 图像采样一个局部区域，并不表示只裁一张 2D 小图。

旧模型清单中的 `feature_unet2d`、`segformer3d` 等实现仍可加载、推理和重试，
但不再作为新任务的公开组合项。旧 `feature_unet2d` 继续使用原有 ONNX、256
输入和 stretch 预处理，不会被新规则迁移。

## Auto 与资源规则

Auto 使用 canonical RAS 后的 spacing，但不假设厚层轴一定是 Z：

1. 找出 spacing 最大的轴作为候选切片轴。
2. 最大/最小 spacing 比值不低于 2.5 时选择 2D。
3. 其余情况选择 3D Standard。
4. Target patch 覆盖率在候选切片轴的正交平面上计算。

`Batch size = Auto` 在训练真正执行的设备上读取显存。因此远程训练会读取远程
GPU，而不是沿用 Mimics 工作站的显存。手工输入的 batch 不会被静默改小。

全体积病例可有不同深度。编码器共享 padded batch，但 3D 解码器会按每例真实
深度独立解码，最后只对 logits 补齐；补齐标签使用 ignore index，损失和指标均
忽略该区域。Sub-volume 与 batch 大于 1 不能同时启用，界面会禁用该组合；需要
更大有效 batch 时使用 gradient accumulation。

新架构使用 `fit_pad`：保持切片平面长宽比，缩放后填充。训练模型清单记录
`resize_mode`，推理按同一规则裁掉填充并返回原图网格。输入契约已升级为 v2；
旧 v1 模型仅在其配置仍为 stretch 时兼容。

## 第一阶段：只比较三个架构

目的不是一次搜索所有参数，而是在同一任务上回答“2D、轻量 3D、细节 3D
谁更合适”。

固定以下条件：

- 同一组 5 个训练病例和同一组验证病例。
- 相同 canonical RAS 数据、mask、输入尺寸、loss、sampling 和增强。
- Frozen backbone、20 epochs、相同学习率和 scheduler。
- Batch 固定为 1，或三个 cell 均使用同一手工值。不要让 Auto 造成架构间
  batch 不同。
- 先使用 3 个 seeds；不要同时比较 LoRA。

三个 cell：

| Cell | Training dimension | 3D quality |
|---|---|---|
| A | 2D | Standard |
| B | 3D | Standard |
| C | 3D | High detail |

筛选指标至少包括 mean Dice、HD95、surface Dice、空预测率、峰值显存和单 epoch
时间。若 Dice 差值小于 0.01，优先选择空预测率更低、HD95 更稳定且模型更简单的
cell。只有筛选胜者才进入独立 fold 确认。

## 第二阶段：只在胜者上比较微调

固定第一阶段胜出的架构，比较：

- Frozen
- LoRA，固定 rank 8、alpha 16

仍使用相同病例、seed、训练长度和预处理。LoRA 只有在独立验证 Dice 提升且没有
明显增加空预测或 HD95 长尾时才成为默认；否则保留 Frozen。

## 第三阶段：按数据指纹决定是否比较 Patch

大而致密目标先保留 Full。只有目标 patch 覆盖不足、前景极低或 Full 出现全背景
塌陷时，再比较 Full 与 fingerprint Patch。不要同时改变 loss、架构和 patch，
否则无法判断改进来源。

Patch 比较必须报告：

- 远端假阳性和 HD95，而不只看 Dice。
- 每例 recall、precision 和空预测率。
- 滑窗重叠区域的融合是否与训练配置一致。

## 从 Mimics 执行

1. 在 DINOv3 训练窗口固定同一批病例、Target Mask 和验证集。
2. 分别选择 2D、3D Standard、3D High detail，其他设置保持一致。
3. 每次训练使用独立任务 ID；在 Status 中导出完整配置、metrics history 和模型
   manifest。
4. 远程 GPU 可以把 Batch 改大，但正式架构消融仍应保持三个 cell 的 batch
   一致。另开一组吞吐测试评估 Batch 1/2/4。

命令行或远程自动化应保存以下审计字段：

```text
training_dimension
quality_mode
resolved decoder
support/validation case IDs
seed
requested/resolved batch size
GPU memory planning budget
input contract and architecture plan hash
```

## 结论边界

当前代码验证证明三个架构能够完成真实 DINOv3 前向、反向、保存和重建，并保证
训练/推理预处理一致；它不等于已经证明某个架构在特定器官上精度最好。推荐值
必须由上述受控消融和独立验证集结果决定。

架构依据：

- [DinoUNet](https://github.com/yifangao112/DinoUNet)：多层 DINO 特征、显式
  空间层级与高分辨率细节恢复。
- [SegDINO v2](https://arxiv.org/abs/2606.17972)：DINO 中间层同 token 网格时，
  需要显式伪空间金字塔，而不是把语义层级误当成原生尺度。
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)：由数据指纹与显存共同规划 patch、
  topology 和 batch，同时允许受控覆盖。
- [DINO-Med3D](https://arxiv.org/abs/2606.18886)：医学 3D 场景需要显式空间与
  体上下文设计，不能只堆叠 2D 特征后假设已经获得 3D 建模。
