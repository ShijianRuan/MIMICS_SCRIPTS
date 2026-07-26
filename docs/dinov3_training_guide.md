# DINOv3 少样本训练指南

> 基于 TimeSlice 7.0.5 实测对比编写。本指南帮助你选择正确的训练路径和参数。

## 两条训练路径

### Frozen Feature 2D（cached_slices）— 推荐

**适用场景**：少样本（1-10 个标注 case）、快速迭代、追求与 TimeSlice 一致的效果。

**工作原理**：
1. 冻结 DINOv3 ViT-S/16 编码器（ONNX 推理，256×256 输入，输出 16×16 grid）
2. 对每张切片编码一次，缓存特征到磁盘
3. 训练轻量 U-Net 解码器（DoubleConv + InstanceNorm + pixel-unfold skip）
4. 训练时直接读缓存特征，不重复跑编码器

**强制约束**（与 TimeSlice 7.0.5 一致）：
- 微调方法：frozen（编码器冻结）
- 切片轴：axial
- 梯度累积：1（真实切片 batch）
- 混合精度：关闭
- 3D 子体积：关闭
- 损失：CrossEntropy（匹配 TimeSlice）
- 图像尺寸：256×256（ONNX 固定）

**推荐超参**（`timeslice_match` profile）：
- lr=0.001, epochs=50, batch_size=4, weight_decay=0.0001
- 验证间隔=5, cosine 调度器, warmup=0

### Volume Pipeline（扩展探索）

**适用场景**：有更多标注数据（>10 case）、需要 3D 上下文、探索非 TimeSlice 架构。

**可选 decoder**：segformer3d / token_pyramid3d / dpt3d / linear3d / mlp_probe / conv2d

**可选微调**：frozen / lora / adapter / full

**特点**：
- 支持 3D patch 采样、多尺度特征、LoRA/Adapter 微调
- 不与 TimeSlice 一致，是独立扩展
- PyTorch 后端（需 HuggingFace 权重，ImageNet 归一化）

## Decoder 选择建议

| 器官类型 | 推荐 decoder | 理由 |
|---|---|---|
| 小器官（肾上腺、主动脉） | feature_unet2d | 2D 切片独立处理，避免 3D 卷积过平滑 |
| 大器官（肝脏、脾脏） | feature_unet2d 或 segformer3d | feature_unet2d 速度更快；segformer3d 多尺度更精确 |
| 快速验证 | linear3d | 最轻量（~0.1M 参数），快速看效果 |
| 低内存设备 | mlp_probe | 内存占用最小 |
| 追求最高精度 | segformer3d + lora | LoRA 微调编码器 + 3D 多尺度 |

## 关键参数调优

### 学习率
- **TimeSlice 实测 lr=0.001**（balanced/timeslice_match profile 默认）
- 过高（>0.005）：训练不稳定，loss 震荡
- 过低（<0.0001）：收敛慢，50 epochs 可能欠拟合

### Epochs
- **TimeSlice 实测 epochs=50**
- 有 early stopping（min_epochs=12, patience=5）
- 小器官可能需要更多 epochs（early stopping 会自动处理）

### 验证
- **TimeSlice 实测 validation_interval=5**
- 少样本下验证 Dice 波动大，建议保持 5（每 5 epoch 验证一次）
- val_fraction=0.2（20% 数据用于验证）

### 图像尺寸
- **ONNX 固定 256×256**（与 TimeSlice 一致）
- PyTorch 后端可自定义（224/256/384/512），但 256 是平衡点
- DINOv3 用 RoPE，支持可变尺寸，224 权重在 256 上有效

## 常见问题

### Q: 为什么 ONNX 是 256 但权重是 224？
A: 权重 224 是 HuggingFace DINOv3 原生训练尺寸（Meta AI 预训练）。ONNX 256 是从同一份权重重新导出的，DINOv3 用 RoPE 天然支持可变尺寸。TimeSlice 7.0.5 实测也用 256。

### Q: cached_slices 时 GUI 禁用了很多选项？
A: 这些选项与 Frozen Feature 2D 不兼容（如 lora 微调、非 axial 切片、3D patch）。GUI 会自动调整并记录到告警——点击 Start 后在 Status 面板可看到"以下参数已被调整"。

### Q: 训练失败怎么办？
A: 检查状态查看器的错误信息。常见原因：
- 后台 Mimics 许可冲突（关闭其他 Mimics 窗口）
- GPU 锁等待（nnInteractive 占用 GPU）
- mask 名称不匹配（确保 .mcs 中的 mask 名称与 organ 匹配）

### Q: 何时启用 target_spacing 重采样？
A: 当训练/测试数据的体素间距不一致时（临床 CT 常见，切片厚度 0.5-5mm 差异）。TimeSlice 默认按 ref_spacing 重采样以保证特征一致性。如果数据集间距统一，可以不启用。

## 实测对比总结

| 维度 | TimeSlice 7.0.5 | Mimics-Script | 一致性 |
|---|---|---|---|
| 输入尺寸 | 256×256 | 256×256 | ✓ |
| grid | 16×16=256 patches | 16×16=256 patches | ✓ |
| decoder | DoubleConv+InstanceNorm+skip | FrozenFeatureUNet2D | ✓ |
| 归一化 | clip(0.5-99.5)+z-score | normalize_timeslice_casewise | ✓ |
| 优化器 | AdamW | AdamW | ✓ |
| 调度器 | cosine_lr | cosine_epoch | ✓（粒度不同） |
| loss | CrossEntropy | CrossEntropy | ✓ |
| 翻转增强 | H/W flip | _random_flip | ✓ |
| lr | 0.001 | 0.001 | ✓ |
| epochs | 50 | 50 | ✓ |
| 重采样 | ref_spacing | 可选 target_spacing | ✓（已恢复） |
