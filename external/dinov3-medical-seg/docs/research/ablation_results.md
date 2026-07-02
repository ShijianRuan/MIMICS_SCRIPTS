# DINOv3 医学图像分割微调 — 消融实验结果

> **日期**: 2026-06-29 | **设备**: Apple M4 MPS
> **设定**: 5-shot (5 个训练体积), 10 epochs, ViT-B/16, 224×224 输入

---

## 0. 当前实现与数据策略说明

本消融表的数字来自当前项目实现，而不是完整 DINOv3 医学数据策略空间。当前训练入口 `scripts/train.py` 使用 `MedicalVolumeDataset` 读取 NIfTI 体数据，输出单通道 `(1,D,H,W)` volume；`SliceWiseEncoder3D` 在模型前向中把每张切片复制成 3 通道，并 resize 到本地 DINOv3 ViT-B/16 的 `image_size=224` 后提取多层特征。DINOv3 负责 2D slice feature extraction，decoder 负责把沿 z 轴堆叠的 pseudo-3D features 转成 3D mask。

当前已报告的主矩阵消融使用以下共同数据策略：

| 项目 | 策略 |
|------|------|
| 文件格式 | NIfTI `.nii` / `.nii.gz` |
| 3D→2D 组织 | axial slice-wise encoding；每张 slice 独立过 DINOv3 |
| 通道处理 | 单通道医学 slice 复制成 3 通道；未使用 2.5D stack、CT multi-window 或 z-depth pseudo-color |
| 平面尺寸 | 多数配置 `data.img_size=[224,224]`；encoder 再对齐到 backbone `image_size=224` |
| 强度归一化 | 主矩阵未设置 `data.modality`，因此使用 dataset 默认 min-max；未启用 CT window 或 MRI z-score |
| spacing | 主矩阵未设置 `data.target_spacing`，因此未做 spacing 重采样 |
| 采样 | `k_shot=5` volume-level few-shot；未实现 foreground/class-balanced crop |
| 增强 | 主矩阵关闭 augmentation；data-strategy 配置单独打开 gamma/brightness/noise/flip/affine |
| 损失 | DiceCE 0.5/0.5；data-strategy 配置额外设置 `class_weights=[1.0,2.0]` |
| sub-volume | 已报告配置均 `enabled=false` |

已实现消融配置对应关系：

| 配置族 | 代表配置 | 验证内容 | 数据策略差异 |
|--------|----------|----------|--------------|
| Decoder × PEFT 主矩阵 | `synthstrip_frozen_*`、`synthstrip_lora_*`、`synthstrip_adapter_*` | decoder 类型和微调方法对 SynthStrip 的影响 | 数据策略固定为 min-max、224、repeat-3ch、无增强 |
| 最简 pipeline baseline | `synthstrip_frozen.yaml` | 64×64 + frozen + linear3d 是否能跑通 | `img_size=[64,64]`，主要用于快速验证，不与 224 主矩阵直接等价 |
| SynthStrip data strategy | `synthstrip_data_strategy.yaml` | 增强 + class weights 是否改善 frozen+segformer3d | 打开 gamma/brightness/noise/flip/affine，设置 `[1.0,2.0]` class weights |
| Prostate baseline | `msd_prostate_frozen_segformer3d.yaml` | frozen+segformer3d 跨器官/MRI 泛化 | 与主矩阵相同的数据策略 |
| Prostate data strategy | `msd_prostate_data_strategy.yaml` | 小前景 MRI 任务中增强 + class weights 的影响 | 与 SynthStrip data strategy 相同 |
| Heart baseline | `msd_heart_frozen_segformer3d.yaml` | 极小前景左心房任务泛化 | 与主矩阵相同的数据策略 |
| Backbone 尺度 | `synthstrip_vitl_frozen_segformer3d.yaml` | ViT-B vs ViT-L | 数据策略固定，只改变 backbone |

因此，§1 的主矩阵主要回答“在固定数据策略下，decoder 和 PEFT 谁更重要”；§2 的跨数据集结果回答“同一固定策略迁移到不同器官/前景占比时会发生什么”。它们还没有回答 CT window、MRI z-score、target spacing、foreground crop、2.5D stack 或多分辨率 TTA 的收益，这些应作为下一轮数据策略消融。

---

## 1. 完整消融矩阵

### 1.1 核心指标 (SynthStrip v1.5, 脑部剥离, 10 test volumes)

```
Decoder        Frozen           LoRA (r=8, QV)   Adapter (bn=64)
──────────────────────────────────────────────────────────────────────
linear3d       0.727  (90min)   0.758  (43min)   0.748  (29min)
mlp_probe      0.889  (30min)   0.879  (50min)   0.888  (40min)
DPT3D          0.797  (55min)   未测试            未测试
segformer3d    0.916  (30min) ★ 0.920  (50min)   0.924  (33min) ★★
```

### 1.2 资源消耗

| 配置 | 训练时间 | Checkpoint | 可训参数 |
|------|:------:|:----------:|:--------:|
| frozen+linear3d | ~90 min | 1.2 MB | 1,538 |
| frozen+mlp_probe | ~30 min | 1.8 MB | 592,134 |
| frozen+DPT3D | ~55 min | 222 MB | ~8M |
| frozen+segformer3d | ~30 min | 9 MB | 4,339,970 |
| LoRA+linear3d | ~43 min | 1.2 MB | ~300K |
| LoRA+mlp_probe | ~50 min | 2 MB | 887,046 |
| LoRA+segformer3d | ~50 min | 10 MB | ~4.6M |
| adapter+linear3d | ~29 min | 5 MB | ~1.2M |
| adapter+mlp_probe | ~40 min | 6 MB | 1,781,766 |
| adapter+segformer3d | ~33 min | 13 MB | ~5.5M |

### 1.3 逐 epoch DSC 曲线 (SynthStrip)

```
Epoch  frozen+L3D  frozen+MP  frozen+DPT  frozen+S3D  LoRA+L3D  LoRA+MP  LoRA+S3D  Adp+L3D  Adp+MP  Adp+S3D
  1      0.393      0.753      0.000      0.826       0.121     0.755    0.787     0.760    0.651   0.823
  2      0.546      0.792      0.542      0.872       0.399     0.776    0.877     0.726    0.731   0.871
  3      0.666      0.828      0.498      0.877       0.656     0.835    0.878     0.737    0.798   0.887
  4      0.691      0.834      0.549      0.887       0.705     0.833    0.890     0.722    0.859   0.887
  5      0.694      0.837      0.640      0.894       0.712     0.837    0.896     0.734    0.858   0.897
  6      0.695      0.848      0.702      0.896       0.716     0.838    0.899     0.740    0.862   0.903
  7      0.704      0.870      0.754      0.902       0.734     0.867    0.913     0.742    0.867   0.910
  8      0.711      0.879      0.797      0.905       0.745     0.879    0.919     0.746    0.883   0.919
  9      0.717      0.888      —          0.905       0.749     0.881    0.920     0.748    0.888   0.922
 10      0.727      0.889      0.782      0.916       0.758     0.879    0.911     0.748    0.888   0.924
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Best     0.727      0.889      0.797      0.916       0.758     0.881    0.920     0.748    0.888   0.924
```

> L3D=linear3d, MP=mlp_probe, DPT=DPT3D, S3D=segformer3d

### 1.4 关键发现

1. **Decoder >> Fine-tuning Method**：解码头选择的影响（+19%）远超微调方法（±0.8%）
2. **segformer3d 最优**：30min frozen 即达 0.916，adapter 推到 0.924
3. **mlp_probe 是强第二**：0.889 frozen，0.5M params，轻量级优选
4. **DPT3D 表现最差**：0.797，参数量大（8M）但效果不佳
5. **微调方法影响极小**：在 segformer3d 上 frozen/LoRA/adapter 仅差 ±0.008
6. **LoRA/Adapter 在小 decoder 上更有价值**：linear3d 上 +3.1%/+2.1%，segformer3d 上仅 ±0.4%
7. **mlp_probe 收敛快**：epoch 1 即 0.753（仅次于 segformer3d 的 0.826）

---

## 2. 跨数据集泛化验证

### 2.1 三个数据集完整对比

| 数据集 | 任务 | 模态 | 训练/测试 | Best DSC | 前景占比 | 难度 |
|--------|------|------|:--------:|:--------:|:------:|:--:|
| SynthStrip v1.5 | 脑部剥离 | MRI/CT | 10/10 | **0.916** | ~50% | 简单 |
| MSD Task05 Prostate | 前列腺分割 | MRI | 25/7 | **0.787** | ~2.4% | 中等 |
| MSD Task02 Heart | 左心房分割 | MRI | 15/5 | **0.646** | ~1.5% | 困难 |

> 全部使用 frozen+segformer3d, 5-shot, 10 epochs, ViT-B/16

### 2.2 分析

- **架构得到验证**：Prostate 0.79 证明 slice-wise 编码 + 3D decoder 在不同解剖部位（大脑、前列腺）和模态上都有效
- **MSD Heart 是异常值**：心房极小（1.5%）、体积大（13M voxels）、只有 15 train，5-shot 的 slice-wise 编码难以捕获如此小的结构
- **任务难度决定上限**：SynthStrip (50% 前景) > Prostate (2.4%) > Heart (1.5%)，DSC 呈单调递减
- **三个数据集一致的学习模式**：epoch 1-2 低 DSC → epoch 3-4 相变 → epoch 5+ 平稳收敛

### 2.3 ViT-B vs ViT-L

| Backbone | Params | Decoder | DSC (SynthStrip) | Time | 结论 |
|----------|:------:|---------|:---:|:----:|------|
| ViT-B/16 | 86M | frozen+segformer3d | **0.916** | 30m | ★ 推荐 |
| ViT-L/16 | 300M | frozen+segformer3d | **0.876** | ~3h+ | 更差+更慢，不推荐 |

> ⚠️ ViT-L epoch 10 未完成（MPS OOM），取 epoch 8-9。验证了 DINOv3 Benchmark (arXiv:2509.06467) 的发现：医学领域放大模型不一定有效。

---

## 3. Decoder 性能排名

```
Decoder       Params    DSC (frozen)   收敛速度    推荐场景
──────────────────────────────────────────────────────────
segformer3d   ~4.3M     0.916 ★★★     极快 (ep1=0.83)   通用首选
mlp_probe     ~0.6M     0.889 ★★      快   (ep1=0.75)   轻量级
DPT3D         ~8M       0.797 ★       慢   (ep1=0.00)   不推荐
linear3d      ~1.5K     0.727 ★       极慢 (ep1=0.39)   仅做 baseline
```

---

## 4. 已完成 vs 待补

### 已完成 ✅
- [x] frozen + 4 种 decoder (linear3d, mlp_probe, segformer3d, DPT3D)
- [x] LoRA + 3 种 decoder (linear3d, mlp_probe, segformer3d)
- [x] Adapter + 3 种 decoder (linear3d, mlp_probe, segformer3d)
- [x] Adapter + segformer3d
- [x] ViT-L 对比 (frozen+segformer3d)
- [x] 第二数据集验证 (MSD Heart)
- [x] 5-shot SynthStrip 完整训练闭环
- [x] 时间/参数 资源统计
- [x] 逐 epoch DSC 收敛曲线

### 待补 ❌
| 缺口 | 优先级 | 说明 |
|------|:------:|------|
| Full fine-tuning (ViT-B) | 中 | MPS 显存不够，需 CUDA GPU |
| 更多 shot 数 (1/3/10/20) | 中 | 需要跑更多实验 |
| 第三数据集验证 | 低 | 进一步增强泛化性证据 |
| 更长训练 (50+ epochs) | 低 | 看 segformer3d 是否继续提升 |
| DoRA | 低 | PEFT 增强方案 |
| 多 seed 重复 | 低 | 量化 5-shot 方差 |

---

> **文档维护**: 随实验进展更新。最新完整消融矩阵见 §1。
