# DINOv3 权重、模态与空间方向契约

## 目标

训练和推理必须保存并复用同一套输入契约。模型不能根据当前文件名、当前
patch 或当前 Mimics 视图临时改变预处理，也不能把不同预训练骨干误认为
具有相同的层数、通道数或 patch stride。

## 预训练权重

新的 2D 和 3D 方案支持本地 Hugging Face 格式的 DINOv3 ViT：

- 目录必须包含 `config.json`、`preprocessor_config.json`，以及单文件或
  分片 `safetensors` 权重。
- 层数、隐藏维度、patch stride、图像均值和标准差均从所选权重目录读取。
- 四个中间层按实际 transformer 深度计算。12 层模型使用
  `[2, 5, 8, 11]`，24 层模型使用 `[4, 11, 17, 23]`；更深模型使用相同
  相对深度规则，不再固定写死。
- 图像高宽必须可以被所选权重的 patch stride 整除。
- `scale_aware2d`、`context3d_lite` 和 `context3d_hybrid` 的输入通道来自
  实际 `hidden_size`，因此可以用于 ViT-S、ViT-B 和 ViT-L。

这不等于支持任意视觉权重。ConvNeXt、原生 Meta `.pth`、普通 ViT 或只
输出最后一层特征的 ONNX，接口均不同，启动前会被拒绝。旧
`feature_unet2d` 是单独保留的兼容路径，只支持其导出时固定的
ViT-S/16 最后一层 ONNX 特征。

Hugging Face 的 DINOv3 backbone 接口允许选择输出 stage，并返回所选
stage 的 feature maps；本项目据此实现按真实层深选层，而不是假定所有
模型都有 12 层。参考：
[DINOv3ViTBackbone](https://huggingface.co/docs/transformers/main/model_doc/dinov3)。

## 模态与强度

界面保留 `Automatic`、`CT`、`MRI` 和 `Other`：

- 所有病例都有一致、可靠的导入或 DICOM metadata 时，`Automatic`
  使用该模态。
- 所有病例均没有可靠 metadata 时，`Automatic` 使用通用非 CT 策略。
- 只有部分病例能识别，或同一批数据混有 CT 与 MRI 时，不再猜测，要求
  用户显式选择或拆分训练。

当前输入策略为：

- CT：固定物理值窗口映射到 `[0, 1]`；三窗输入只允许用于 CT。
- MRI/Other：在整病例图像上计算稳健百分位并映射到 `[0, 1]`。
- 随后复制或构造三通道，再使用所选 DINOv3 processor 的均值和标准差。

整病例归一化发生在 ROI、patch 和滑窗之前。因此同一病例中的不同 patch
不会各自重新估计强度范围。该规则同时用于 2D、3D、全图、patch、训练和
推理，并写入模型的 `input_contract`。

模态并不能保证一个预处理对所有任务最优。CT 固定宽窗和 MRI 稳健百分位
是通用、安全的默认值，不替代任务级验证。nnU-Net 同样让 channel/modality
影响 normalization，并要求训练和推理通道保持一致：
[nnU-Net dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)。

## 方向、视图与输出恢复

新方案的统一空间流程为：

1. 用 NIfTI affine 同时重排图像和标签到 canonical RAS。
2. NIfTI 数组从 `(X, Y, Z)` 转为模型张量 `(Z, Y, X)`。
3. 在 canonical RAS 中定义 axial、coronal 和 sagittal。
4. 可选 spacing 重采样时，图像使用连续插值，标签使用最近邻，并强制落在
   同一个目标网格。
5. 推理结果先恢复到原图 canonical 网格，再恢复原始轴序、shape 和 affine。
6. 应用到 Mimics 时，bridge 使用保存的源图 affine，把 RAS 结果映射到
   当前 Mimics/DICOM LPS 网格。

NiBabel 将 world axes 定义为 RAS+，并提供 `as_closest_canonical` 完成
轴交换和翻转：
[NiBabel image orientation](https://nipy.org/nibabel/image_orientation.html)。
医学分割框架也常在网络前显式统一 orientation；例如 MONAI 提供
`Orientationd`，nnU-Net 的 `NibabelIOWithReorient` 可统一到 RAS。

当前 3D 方案仍是 2D DINOv3 slice encoder 加 3D context decoder，不是原生
3D transformer。因此 view plane 对 2D 和 3D 都有意义：

- axial：沿 canonical Z 编码切片；
- coronal：沿 canonical Y 编码切片；
- sagittal：沿 canonical X 编码切片。

`Auto` 根据 canonical RAS 下的物理 spacing 选择最粗的轴作为切片深度，
适合各向异性数据。3D decoder 在同一 slice-space 中聚合上下文，输出后再
逆置换回 `(Z, Y, X)`。

## 已知边界

- affine 缺失、奇异或本身错误的数据无法仅凭像素数组推断真实病人方向，
  必须在导入或数据修复阶段处理。
- 跨数据集复用允许不同原始轴序，因为每例都会独立 canonicalize；但模态、
  通道语义和模型保存的强度契约必须一致。
- ViT-L 或更大的权重在接口上兼容，不代表 3060 的显存一定足够。实际可用性
  还受输入尺寸、LoRA、切片批量、patch 和 decoder 影响；本地资源不足时应
  使用冻结骨干、patch/sub-volume 或远程 GPU。
- 旧 ONNX 兼容模型保留原生轴向和固定输入特征契约，不作为新训练的默认
  多尺度方案。
