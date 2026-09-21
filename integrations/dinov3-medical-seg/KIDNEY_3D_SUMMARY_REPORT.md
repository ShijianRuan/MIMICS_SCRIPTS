# Kidney_left 5-shot 分割实验总结报告

**任务**：kidney_left 5-shot（5 训练 / 3 验证 / 100 测试）CT 分割，在 nnU-Net 2.5.2 框架内对比不同预训练 backbone + 架构 + 训练策略。
**目标**：找到最佳预训练框架用于 5-shot 微调。2D 已收敛（0.8478），3D 差强人意（当前最佳 0.7726），本报告聚焦 3D 的探索与诊断。
**数据**：Dataset901(5-shot训练) / Dataset902(100例测试)，CT，单侧器官(kidney_left)，前景占比 0.1-0.7%。
**更新日期**：2026-08-17

---

## 一、全部实验结果总览（100例 test）

### 2D（已收敛）
| 实验 | 预训练/架构 | mean | median | fails | HD95 |
|---|---|---|---|---|---|
| **meddinov3_ct3m** | MedDINOv3 CT-3M 全参 | **0.8478** | 0.950 | 10 | 11.7 |
| flexict2d | FlexiCT-2D 全参 | 0.8435 | 0.956 | 9 | 10.6 |
| meddinov3_ct3m_frozen | CT-3M 冻结encoder | 0.6909 | 0.832 | 26 | 18.1 |
| original_dinov3 | DINOv3 自然图 全参 | 0.6954 | 0.863 | 26 | 28.0 |
| originaldinov3_frozen | DINOv3 冻结 | 0.6749 | 0.872 | 28 | 19.8 |
| feat_b_frozen_single | feature_unet2d 冻结 | 0.4989 | 0.613 | 44 | 184.5 |

**2D 结论**：CT-3M（医学CT预训练）全参微调最佳(0.8478)。冻结 encoder 全面差于全参（掉 0.16），证明 5-shot 下全参微调优于冻结。自然图 DINOv3 不如医学 CT 预训练。

### 3D（探索中，核心问题）
| 实验 | 预训练/架构 | 策略 | mean | median | fails | HD95 | 诊断 |
|---|---|---|---|---|---|---|---|
| **flexict3d (exp4)** | FlexiCT-3D ViT | multiscale,3e-5,150ep | **0.7726** | 0.942 | 18 | 23.2 | 当前3D最佳 |
| flexict3d_v2 (exp8) | FlexiCT-3D ViT | 投影decoder(3456→864) | 0.5926 | 0.883 | 38 | 27.4 | 投影瓶颈丢多尺度 |
| flexict3d_off (T1) | FlexiCT-3D ViT | 对齐官方(ConvT+1e-4+300ep+clip1) | 0.5179 | 0.732 | 46 | 41.1 | 5-shot高LR过拟合 |
| dinounetr (exp9) | DINOv2 ViT-L | MONAI UNETR | 0.4913 | 0.523 | 49 | 63.3 | 自然图预训练不适配CT |
| merlin | Merlin CNN I3ResNet | 3D-CT CLIP,fp16bug修复后 | 0.2896* | 0.196 | 73 | 171.7 | CNN-152 5-shot欠训练 |
| flexict3d_aug (T2b) | FlexiCT-3D ViT | 强增强(elastic bug,重训中) | 待验证 | - | - | - | elastic bug已修复重训 |

*merlin 0.2896 是 fp16 bug 版；修复后 val 0.30，100例待补

**3D 当前最佳仍是 exp4 (0.7726)**，但比 2D (0.8478) 差 0.075。所有 3D 改进尝试都没突破。

---

## 二、3D 为什么普遍差于 2D（四个根因）

1. **patch 覆盖不全**（结构性）：3D 用 96³@1.5mm=144mm patch，肾脏~120mm 勉强覆盖，边界 patch 多。2D 逐切片无此问题。
2. **5-shot 数据效率低**：3D 模型参数多、有效 crop 少，5 例数据严重不足。2D 同样数据切成更多 slice，等效样本多。
3. **decoder 上采样**：3D 用 trilinear（省显存）非 ConvTranspose，边界重建弱（HD95 23 vs 2D 11）。
4. **超参未适配 5-shot**：官方超参为大数据设计，5-shot 下需低 LR + 短训练（T1 验证）。

---

## 三、3D 各实验深度诊断

### exp4 (0.7726) —— 当前最佳，基线
FlexiCT-3D backbone（CT专用 ViT）+ multiscale decoder（3456宽无投影，4×带宽）+ trilinear + vit_lr 3e-5 + 150ep + grad_clip 12 + L-R mirror fix。
EMA 0.8979 但 test 0.7726，gap 0.125 = 轻度过拟合（5-shot+3同分布val）。

### exp8 (0.5926) —— 投影 decoder 更差
exp4 的 multiscale(3456宽) vs exp8 的投影(3456→864)。源码级确认：multiscale 保留 4× decoder 通道带宽，5-shot 下细节更全。投影瓶颈丢了多尺度信息。**结论：5-shot 下 decoder 带宽比投影精简更重要。**

### T1 (0.5179) —— 对齐官方超参适得其反
对齐官方 FlexiCT：ConvTranspose3d + vit_lr 1e-4(原3e-5) + grad_clip 1.0(原12) + 300ep(原150)。
- 训练健康（EMA 单调 0→0.8457，无 NaN）
- 但 46 例全空(dice=0)，且这些 case exp4 能做 0.9+
- **根因**：5-shot 下高 LR+长训练让 backbone 过拟合 5 个训练 case，对偏离形态预测全背景。官方超参为大数据(几百例+1000ep)设计，不能照搬到 5-shot。
- **证伪**：实现/超参对齐不是 3D 差的解，反而更差。

### exp9 dinounetr (0.4913) —— 自然图预训练不适配
DINOv2 ViT-L（自然图像预训练）+ MONAI UNETR decoder。自然图预训练的 features 与 CT 差距大，5-shot 微调无法弥补。**CT 专用预训练(FlexiCT) >> 自然图(DINOv2)。**

### Merlin (0.2896) —— CNN 5-shot 欠训练 + fp16 bug
I3ResNet-152（CNN，ResNet152 inflate 3D）+ 3D-CT CLIP 对比学习预训练 + UNetDecoder。
- **第一次 0.2896 有 fp16 bug**：Merlin I3ResNet forward 在 fp16 溢出(inf→nan)，trainer 只覆盖了 train_step 的 fp32，没覆盖 validation/predict。val_loss 从 epoch17 起 nan，正式 validation Dice 0.0。
- **agent 深度诊断推翻了我的"轴错位"误判**：ramp probe 证明网络轴完全正确对齐，(2,2,1) stride 正确补偿各向异性下采样。
- **修复后**（fp32 validation + 禁 mirror TTA）：val_loss 正常，Validation Dice 0.30，EMA 0.597。
- **真实性能弱**：CNN-152 层深，5-shot 严重欠训练。论文优势是"10%数据对 nnU-Net 从头训的相对优势"，我们的对照 exp4 是更强的 FlexiCT 预训练，Merlin 无相对优势。
- 100例 fp32 predict 因 OOM(和训练抢RAM)搁置，val 0.30 结论已够。

### T2a 128³ (EMA 0.771) —— 大 patch 5-shot 不稳定
exp4 配置 + patch 128³@1.0mm（覆盖更广）。Pseudo dice 剧烈波动 [0.33-0.81]，EMA 仅 0.771。
**根因**：大 patch 在 5-shot 上 crop 多样性不足 + 有效位置少 + 梯度方差大，加剧不稳定性。**高分辨率大 patch 方向在 5-shot 失败。**

### T2b 强增强 (elastic bug, 重训中) —— 过分割是代码 bug
exp4 配置 + 强数据增强（elastic + brightness/contrast/gamma 加宽 + sim_lowres + noise + oversample 0.5）。
- **第一次训练**：Pseudo dice 0.969 看似突破，但 Validation 0.398、训练 case 也 0.419——严重过分割(FP 是 TP 3倍)。
- **诊断**：训练 case 也差 = 非过拟合 = 训练/推理不一致 = 代码 bug。
- **逐一排除**：mirror TTA ❌ / fp16 ❌ / checkpoint ❌ / 推理流程 ❌（exp4 同流程 0.97）。唯一差异：elastic。
- **真凶**：batchgeneratorsv2 SpatialTransform 的 `elastic_deform_scale=(0.05,0.1)` 传 2 值（误当随机范围），但 API 要求 per-axis（长度=3）。`zip(deformation_scales, patch_size)` 截断成 2 个 sigma，第 3 轴 `sigmas[2]` 越界 → IndexError → dataloader worker 崩溃 → 返回损坏 batch（非"未变形"，是垃圾数据）→ 模型学坏 → Pseudo dice 虚高 → 推理过分割。
- **修复**：关 elastic，保留其余强增强重训。Epoch 0 EMA 0.8729 无崩溃，修复确认。待完成验证 100 例是否突破 0.7726。

---

## 四、关键方法论教训

1. **Pseudo dice 高 ≠ 模型好**：5-shot + 3 同分布 val 下，EMA/Pseudo dice 反映训练拟合度，不是泛化。训练/推理不一致会虚高（T2b/Merlin）。必须看 100 例 test + 训练 case 表现。
2. **训练 case 也差 = 代码 bug，不是过拟合**：过拟合是"训练好未见差"，训练 case 都差说明训练/推理不一致，必是 bug（T2b）。
3. **强增强需逐 transform 验证不崩**：batchgeneratorsv2 API 误用静默崩溃，nnU-Net 吞 worker stderr，要主动复现（直接调 transform + dataloader）。
4. **5-shot 不能照搬大数据超参**：高 LR + 长训练加剧过拟合（T1）。低 LR(3e-5) + 短训练(150ep) + 松 grad_clip(12) 更鲁棒。
5. **5-shot 大 patch 不行**：crop 多样性不足致训练不稳定（T2a）。
6. **CT 专用预训练 >> 自然图**：FlexiCT-3D(0.77) >> DINOv2(0.49)。医学域预训练关键。
7. **冻结 encoder 全面差于全参**：5-shot 下全参微调保留+适配预训练知识更好（2D 验证）。

---

## 五、当前结论与下一步

**3D 当前最佳**：exp4 flexict3d 0.7726（FlexiCT-3D + multiscale + 低LR + 150ep）。
**3D < 2D gap**：0.075，主因 5-shot 过拟合（数据多样性不足）。

**正在进行**：
- **T2b 重训**（elastic 关，其余强增强）：验证强增强（真正的，非 bug 版）能否突破 0.7726。这是当前最有希望的方向——强增强直击 5-shot 数据多样性不足。
- Merlin 100例 fp32 待补（OOM 搁置，val 0.30 结论已够）。

**收敛逻辑**：
1. T2b 重训完成 → 若突破 0.7726 → 强增强是 3D 解，定 T2b 为赢家
2. 若 T2b ≈ exp4 → 5-shot 过拟合难以靠增强解决 → exp4 0.7726 是 FlexiCT-3D 在 5-shot 的天花板附近
3. 后续可探索：test-time 策略 / 更优 3D backbone（若有CT专用3D预训练）/ 承认 5-shot 3D 天花板

---

## 附：预训练 backbone 对比（回答"什么预训练权重+架构少样本更好"）

| 预训练类型 | 架构 | backbone | 3D最佳 | 2D最佳 |
|---|---|---|---|---|
| 医学CT专用 | ViT 3D | FlexiCT-3D | **0.7726** | 0.8435 |
| 医学CT专用 | ViT 2D | MedDINOv3 CT-3M | - | **0.8478** |
| 3D-CT对比学习 | CNN I3ResNet | Merlin | 0.2896*(bug)/0.30(val) | - |
| 自然图 | ViT-L | DINOv2 | 0.4913 | 0.6954 |

**结论**：5-shot 下，**医学CT专用预训练 + ViT 架构 + 全参微调**最优。CNN(深152层) 5-shot 欠训练，自然图预训练不适配CT。3D 受限于 5-shot 数据效率，暂逊 2D。
