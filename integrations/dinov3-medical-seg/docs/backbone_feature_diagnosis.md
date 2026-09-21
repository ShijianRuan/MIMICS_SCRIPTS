# DINOv3 Backbone 特征可用性诊断报告

> 日期：2026-07-10
> 诊断脚本：`external/dinov3-medical-seg/scripts/diagnose_backbone_features.py`
> 关联实验：`E:/total_test/fewshot_models/runs/liver/train_20260710T121337_ad7db296`（frozen + linear3d，20 epoch，best DSC=0.0252）

## 背景

在 DINOv3 医学分割微调中观察到：

- 另一项目 DINOv3 微调 1 epoch 即达到 dice>0.9，收敛极快。
- 本项目收敛慢、上限低（kidney_left frozen+segformer3d，15 epoch 仅 0.47；liver frozen+linear3d，20 epoch 仅 0.025）。
- 直觉判断"预训练权重没有被利用"。

为定位根因，对一个 frozen + linear3d 线性探针实验（DSC 仅 0.025）做特征可用性诊断：**到底是 backbone 特征对前景/背景没有判别力，还是特征可用但探针/优化被训塌了？**

## 诊断方法

脚本 `diagnose_backbone_features.py`：

1. 加载一张真实 CT slice（s0001 liver，含器官的最大轴向切片）+ 对应二值 mask。
2. 按 trainer 完全相同的链路送入 frozen DINOv3 backbone：单通道 → repeat 成 3 通道 → 归一化 → interpolate 到 224 → backbone。
3. 提取 4 层特征图（out_indices=[2,5,8,11]，每层 768 通道）。
4. 测量"前景 vs 背景"在特征空间的可分性：
   - discrimination ratio = `|mean_fg − mean_bg| / (std_fg + std_bg)`，跨通道平均。
   - 同时统计"强信号通道数"（ratio > 0.1 的通道数 / 768）。
5. 对比三种输入归一化链路，隔离归一化是否为瓶颈：
   - **repeat3 + imagenet_norm**（当前 trainer 实际使用的链路）
   - repeat3 + ct_norm（3 通道一致，CT 自身 mean/std）
   - repeat3 + no_norm（原始 [0,1]）

判读标准：ratio < ~0.05 → 特征不可分（瓶颈在特征链路）；ratio > ~0.2 → 特征可分（瓶颈在 decoder/优化）。

## 结果

| 输入链路 | 平均可分性 ratio | 强信号通道(level3) |
|---|---|---|
| repeat3 + imagenet_norm（**当前**） | **0.2399** | 605/768 |
| repeat3 + ct_norm | 0.2404 | 627/768 |
| repeat3 + no_norm | 0.2453 | 612/768 |

每层特征前景/背景均值均有明显差异，数百个通道对前景/背景有区分度。**三种归一化方案几乎没有差异**（0.240 / 0.240 / 0.245）。

## 结论

1. **backbone 特征是可用的**：ratio ~0.24，远高于"不可分"阈值，前景/背景在特征空间线性可分。预训练权重确实被加载并提供了有用信号。
2. **归一化链路不是瓶颈**：imagenet / ct / 无归一化三者特征可分性几乎相同。之前怀疑的"单通道 repeat3 + imagenet 归一化造成 OOD"**被证伪**。`input_normalization` 配置项不影响特征质量。
3. **线性探针 DSC 仅 0.025 的根因不在特征，而在训练动力学**：特征给了信号，但探针被训塌了。liver run 的 dice_loss 卡在 0.85（dice≈0.15）不动、ce_loss 卡在 0.10，是优化塌缩（预测几乎全背景）的典型表现，而非"特征没有信号"。

## 修正后的根因排序

排除：特征链路、归一化、单通道处理、backbone 权重加载。

剩余主因（按可能性）：

1. 🔴 **class_weights 过激**：`compute_class_weights`（`tools/fewshot_pipeline.py:1122`）按逆频率计算前景权重，硬编码上限 `max_weight=20.0`。`losses.py` 对 Dice **和** CE 都乘 class_weights（双重加权），导致 Dice loss = `1 − 20×dice`，dice>0.05 后 loss 即转负并爆炸（实测训练 loss 跑到 −2.0）。小 decoder/探针对这种尺度爆炸的梯度极脆弱。
2. 🟡 **batch_size=1 + 无随机种子**：`trainer.py` 无 `torch.manual_seed`，DataLoader `shuffle=True` 无 generator → 梯度极度噪声 + 不可复现（同配置两次 run best_dsc 0.47 vs 0.41）。
3. 🟡 **有效 batch=1 下 lr=1e-3 偏大**，对小 decoder 不稳。

## 后续动作

- ✅ 已回退 `mimics_lora_segformer3d.yaml` 中无效的 `input_normalization` 改动（实测无影响）。
- ✅ 降低 class_weights 上限：`max_weight` 20 → 5，并暴露到 `fewshot_config.json`。
- ✅ 修复 Dice 重复加权：`losses.py` 的 DiceLoss 不再乘 class_weights（Dice 本身是区域均衡指标，仅 CE 加权）。
- ⏳ 待办：trainer 固定随机种子（manual_seed + DataLoader generator + cudnn.deterministic）。

## 复现

```bash
cd external/dinov3-medical-seg
"E:/mimics_script_offline/nninteractive_env/python.exe" scripts/diagnose_backbone_features.py \
    --ct E:/total_test/s0001/ct.nii.gz \
    --label E:/total_test/s0001/segmentations/liver.nii.gz \
    --model ./models/dinov3-vitb16
```

可换 `--label` 为其它器官（kidney_left.nii.gz 等）验证结论一致性。
