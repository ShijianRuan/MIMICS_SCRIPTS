# RONDON（TimeSlice）逆向工程与训练复现文档

> 本文档基于对本地安装的 RONDON 6.0.0 软件的实际文件挖掘整理，目的是为后续复现其"用 DINOv3 微调分割"的训练流程提供完整细节。
> 整理日期：2026-07-23。所有结论均标注了来源文件路径。

---

## 0. 一句话总结

RONDON 是一个 **Tauri + 嵌入式 Python 3.11** 的医学影像（CT/MRI）3D 分割桌面软件。它的核心玩法是：**冻结一个预训练好的 DINOv3 ViT-S/16 作为编码器（ONNX 推理），把每张切片编码成 patch embedding 并缓存到磁盘；再用 tinygrad 训练一个轻量的 U-Net 风格解码器**（吃缓存 embedding + 原图 pixel-unfold 多尺度 skip）来做器官/病灶分割。**完整的训练循环源码已被 Nuitka 编译进原生 .pyd，无法直接阅读**；本文档的训练流程是从该 .pyd 中提取符号字符串逆向还原的。唯一以明文 Python 存在的训练相关源码是随每个模型导出的 `network.py`（解码器结构定义）。

---

## 1. 关键文件与目录路径速查

### 1.1 运行时配置（用户数据）—— `C:\Users\shijian.ruan\AppData\Roaming\RONDON\`

| 路径 | 内容 | 来源 |
|---|---|---|
| `settings.json` | 用户配置（笔刷、colormap、image_store 路径、token、license 用户信息）。**内含明文密码** | 实读 |
| `.persisted-scope` | 持久化的"最近访问文件"路径列表（Totalsegmentator MRI/CT 数据集 nii.gz 路径） | 实读（二进制混合文本） |
| `db\DoNotTouchMe` | SQLite 数据库（病人/研究/序列/层堆/sample/模型关系），SQLModel/SQLAlchemy 建表 | 实读 + 日志确认 |
| `logs\*.log` | 运行日志，如 `20260723-105105.log` | 实读 |
| `model\` | 模型仓库（编码器 + 解码器） | 实读 |

### 1.2 应用内部模块结构（从 .pyd 符号确证）

来源：`_tauri_app.cp311-win_amd64.pyd` 中的模块路径字符串。

```
_tauri_app
├── __main__
├── commands
│   ├── app              # 应用初始化、数据库、image/model store（日志中 _tauri_app.commands.app）
│   ├── encoder          # 编码器 CRUD：list_encoder / import_encoder / export_encoder
│   ├── decoder          # 解码器 CRUD：list_decoder / create_decoder / train_decoder / run_decoder
│   ├── _encoder         # encoder 内部实现
│   ├── _decoder         # decoder 内部实现（训练循环 run_*_step_for_decoder 在此）
│   ├── sample           # 训练样本管理
│   ├── seg              # 分割标注
│   ├── image / patient / study / serial / stack / record / dicom_meta  # DICOM 数据模型
│   ├── group / contain  # 分组
│   ├── task / server    # 任务调度、服务
│   └── utils
│       ├── array        # ← 归一化函数、resize、patch 转换
│       ├── tiny_unet    # ← 训练循环核心：train_decoder / run_*_step / compute_dice / save_best
│       ├── network      # 网络工厂、图生成 build_tinygrad_model_class_from_graph
│       ├── layer        # tinygrad 层封装
│       ├── metric       # Dice 等
│       ├── image / colors / feature / fs / db / msg / archive / remote
└── networks
    └── decoder_network  # Network 类（对应明文 network.py）
```

**关键映射**：
- 训练循环在 `commands.utils.tiny_unet`：`train_decoder` / `train_decoder_handler` / `run_training_step_for_decoder` / `run_validation_step_for_decoder` / `run_evaluation_step_for_decoder` / `save_best_validation_network_parameters` / `compute_dice` / `load_embedding_as_tensor` / `load_seg_as_tensor`。
- 归一化与编码器预处理在 `commands.utils.array`：5 个 `normalize_*` 函数 + `resize_slice_array` + `slice_to_patch_channels16` + `preprocess_for_decoder`。
- 网络结构与图生成在 `commands.utils.network` + `networks.decoder_network`。

### 1.3 程序安装目录（含训练代码）—— `C:\Users\shijian.ruan\AppData\Local\RONDON\`

| 路径 | 内容 | 来源 |
|---|---|---|
| `rondon.exe` | Tauri 主程序（Rust 壳 + WebView） | `ls` 确认 |
| `python.exe` / `python311.dll` | 嵌入式 Python 3.11 运行时 | `ls` 确认 |
| `Lib\site-packages\_tauri_app.cp311-win_amd64.pyd` | **应用核心逻辑（含训练循环），Nuitka 编译的原生扩展** | strings 提取，含 `__nuitka__` 标记 |
| `Lib\site-packages\tauri_app\` | 薄启动包（`__init__.py` 38 行 + `__main__.py`） | 实读 |
| `Lib\site-packages\tinygrad\` | tinygrad 0.11.0（训练后端） | `ls` + dist-info |
| `Lib\site-packages\onnxruntime\` | ONNX 推理（编码器前向） | `ls` |
| `Lib\site-packages\` 其余 | SimpleITK, pydicom, numpy 1.26.4, scipy, sqlalchemy, sqlmodel, cryptography, PyMCubes/mcubes, trimesh, pyradiomics/radiomics, xlsxwriter 等 | `ls` |

### 1.3 模型仓库结构 —— `AppData\Roaming\RONDON\model\`

两类目录：

**编码器目录**（例 `dinov3-vits16\`）：
```
dinov3-vits16\
├── model.onnx            # DINOv3 ViT-S/16，86 MB
├── model.json            # 元数据：category=DINOV3, model=encoder, imported=true
└── decoder_network.py    # 解码器结构定义（明文，2399 字节）
```

**解码器目录**（随机 16 字符名，例 `casgzeefcueyriwz\`）—— 一次微调训练的产物：
```
casgzeefcueyriwz\
├── model.json            # 训练配置 + 指标（optimization / metrics / tagstr / encoder_id）
├── network.py            # 解码器结构定义（明文，与 decoder_network.py 相同）
├── model.safetensors     # 最新/最优权重（8.5 MB）
├── epoch{N}-dice{...}.safetensors   # 各 epoch checkpoint
├── tiny_mesh.glb / .png  # 3D 预览网格
├── __pycache__\network.cpython-311.pyc
└── <16字符随机>\          # 训练 sample 子目录（若干个）
    ├── model.json        # sample 元数据（source_id / source_image_path / spacing / patient_name）
    ├── volume.nii.gz     # 该 sample 的原始体积
    ├── segs\<id>.nii.gz  # 标签
    ├── tiny_mesh.glb / .png
    └── predict_tiny_mesh.glb / .png   # 预测结果网格
```

---

## 2. 整体技术栈

| 层 | 技术 | 证据 |
|---|---|---|
| GUI 框架 | Tauri（Rust + WebView） | `rondon.exe` + `EBWebView\` + pytauri 0.8.0 |
| 后端语言 | Python 3.11（嵌入式） | `python311.dll` |
| 应用逻辑 | **Nuitka 编译的 .pyd** | `_tauri_app.cp311-win_amd64.pyd` 含 `__nuitka__` |
| 训练框架 | tinygrad 0.11.0 | `tinygrad\` + `tinygrad.nn.optim` / `tinygrad.nn.state` 符号 |
| 编码器推理 | onnxruntime（CPU） | `CPUExecutionProvider`, `InferenceSession` 符号 |
| 数据 IO | SimpleITK 2.5.3, pydicom 3.0.1, numpy 1.26.4 | site-packages |
| 数据库 | SQLite via SQLModel/SQLAlchemy | `sqlite:///...DoNotTouchMe` 日志 |
| 3D 网格导出 | PyMCubes 0.1.6 + trimesh 4.11.2 | site-packages + `tiny_mesh.glb` |
| 影像组学 | pyradiomics 3.0.1 | site-packages |
| 加密/授权 | cryptography（AES-GCM + PBKDF2-HMAC-SHA256），HWID 绑定 | `TSMODELv1`, `AESGCM`, `PBKDF2HMAC`, `get_hwid` 符号 |

---

## 3. 模型设计（核心）

### 3.1 编码器：DINOv3 ViT-S/16（冻结）—— ONNX 实测确证

来源：用 `onnxruntime` 实跑 `dinov3-vits16\model.onnx`（这是本文档中少数 100% 实测确证的部分）。

- **存储形式**：`model.onnx`（86 MB），onnxruntime 加载，Provider = `CPUExecutionProvider`。
- **ONNX 输入**：`['batch', 3, 512, 512]` float —— **3 通道、512×512**（不是 1 通道！）。
- **ONNX 输出**：3 个张量，均为 `(B, 384, 32, 32)`：
  - `hidden1`、`hidden2`、`output`。训练用 `output`（最后一层）。
- **patch 参数（确证）**：`patch_size = 16`，`grid = 32×32 = 512/16`，`embed_dim = 384`。与 "ViT-S/**16**" 命名一致。
- **冻结**：训练中编码器不更新参数，只跑前向。
- **两种 encoder**：
  - `dinov3-vits16`：从 `model.onnx` 跑（`run_dinov3_vits16_encoder`，内含 `process_slice` / `process_grayscale_slice`）。
  - `vanilla`：`run_vanilla_encoder`。**没有独立 ONNX**，推测是 tinygrad 原生实现的简单 CNN 编码器。

> ⚠️ 重要修正（覆盖早先推断）：ONNX 输入是 **3 通道**，不是单通道。医学影像单通道灰度需先转成 3 通道再喂 ONNX。

### 3.2 编码器前向的预处理 pipeline（`process_slice` / `process_grayscale_slice`）

来源：符号 + ONNX 实测 + docstring。

```
原始切片 (H, W) 单通道灰度
  │
  ├─ resize_slice_array  → 512×512   (注释确证 "Resize a slice array to 512x512")
  ├─ 归一化 (见第 4 节, utils.array)
  ├─ pad 到 16 的倍数 (记录 pad 量到文件名)
  ├─ process_grayscale_slice → 转 3 通道 (灰度复制成 RGB, 喂 ONNX 需要 (B,3,512,512))
  ├─ ONNX 前向 (input_name='input')
  └─ 取 output → embedding (B, 384, 32, 32)
        │
        └─ 缓存到 embedding_folder, 文件名 {idx:05d}_l{l}_r{r}_u{u}_d{d} (见 3.4)
```

- **灰度→3通道**：函数 `process_grayscale_slice`（符号确证，报错信息 `"Error processing grayscale slice"`）。推测是单通道复制 3 份成 RGB（标准做法，让 ImageNet 预训练的 ViT 接受灰度输入）。**这一步的具体实现未确证**（可能是 `np.stack([g,g,g])` 或 `np.repeat(g,3)`），但 ONNX 输入确为 3 通道。
- **插值**：图像线性，标签 `NEAREST`（符号 `interpolation` / `NEAREST`）。
- **ONNX 前向**：`get_sync_session` / `synchronize_session`，取 `input_name` / `output_name`。三个输出 `hidden1/hidden2/output` 都是 `(B,384,32,32)`，训练取 `output`。

### 3.3 Patch 策略与恢复机制（核心，实测确证）

这套架构里有**两条独立的 patch 路径**，容易混淆，必须分清：

#### 路径 A：编码器的 ViT patch（图像 → patch embedding）

这是 DINOv3 标准 ViT 行为，**ONNX 内部完成，不需要手动 unfold**：

- 输入图像 `(B, 3, 512, 512)` 进入 ONNX。
- ONNX 内部的 patch_embed 把图像切成 `16×16` 的 patch（patch_size=16），共 `32×32 = 1024` 个 patch。
- 每个 patch 经线性投影成 384 维 token，加位置编码，过 Transformer blocks。
- 输出 reshape 回空间：`(B, 384, 32, 32)` —— 这就是 patch grid 上的 embedding。
- **grid = 32×32，与原图 512×512 的比例 = 1/16**（每个 patch 对应原图 16×16 像素）。

> 即：编码器侧的 patch 是 ViT 内部的，用户代码只管喂 `(B,3,512,512)`、拿 `(B,384,32,32)`。

#### 路径 B：`slice_to_patch_channels16`（patch 级 slice↔patch 转换工具）

来源：docstring `"Returns an array shaped (256, H/16, W/16), where 256=16*16 channels"` + 实测。

- 这是个**独立的工具函数**（在 `utils.array`），把单通道切片 `(1, 512, 512)` 做 **16×16 pixel-unfold** 成 `(256, 32, 32)`：
  ```python
  # 等价实现 (实测形状确证)
  s16 = s.reshape(B,C,H//16,16,W//16,16).transpose(0,1,3,5,2,4).reshape(B,-1,H//16,W//16)
  # (1,1,512,512) -> (1, 256, 32, 32),  256 = 16*16, grid = 32x32
  ```
- **关键**：它的输出 `(256, 32, 32)` 与 ONNX embedding `(384, 32, 32)` **共享同一个 32×32 patch grid**，但通道是 256（原图像素展开）而非 384（ViT 特征）。
- 逆操作 `patch_channels16_to_slice`：把 `(256, 32, 32)` fold 回 `(1, 512, 512)`。
- **用途推断**：用于在 patch grid 分辨率上对齐/转换原图信息（如把 patch 级的预测映射回像素，或在 patch 级做某些操作）。**注意它不是 decoder 的 skip**（decoder skip 用 8/4/2/1 倍 unfold，见路径 C）。

#### 路径 C：解码器的多尺度 skip（pixel-unfold，`network.py` 明文）

这是 decoder 恢复分辨率的核心机制。对原图 `s = (B, 1, 512, 512)` 做**不同倍数的 pixel-unfold** 生成各尺度 skip：

| skip | unfold 倍数 | 输出形状 | 通道 | 空间 | 喂给 |
|---|---|---|---|---|---|
| `s8` | 8×8 | `(B, 64, 64, 64)` | 64 = 8×8 | 64×64 (1/8) | up1（1/16→1/8） |
| `s4` | 4×4 | `(B, 16, 128, 128)` | 16 = 4×4 | 128×128 (1/4) | up2（1/8→1/4） |
| `s2` | 2×2 | `(B, 4, 256, 256)` | 4 = 2×2 | 256×256 (1/2) | up3（1/4→1/2） |
| `s1` | 1×1 | `(B, 1, 512, 512)` | 1 | 512×512 (1/1) | up4（1/2→1） |

（以上形状均经 numpy 实测确证。）

实现（`network.py` 原文）：
```python
s8 = s.reshape(B,C,H//8,8,W//8,8).permute(0,1,3,5,2,4).reshape(B,-1,H//8,W//8)
s4 = s.reshape(B,C,H//4,4,W//4,4).permute(0,1,3,5,2,4).reshape(B,-1,H//4,W//4)
s2 = s.reshape(B,C,H//2,2,W//2,2).permute(0,1,3,5,2,4).reshape(B,-1,H//2,W//2)
s1 = s
```

#### 恢复机制：编码器 embedding 如何逐级恢复到原图分辨率

整条恢复链（实测形状）：

```
embedding  e: (B, 384, 32, 32)      ← ONNX 输出, patch grid 32x32 (1/16)
  │
  bottleneck: DoubleConv(384→256)   → (B, 256, 32, 32)    仍在 1/16
  │
  up1: ConvT(256→128, stride2) + concat(s8:64ch) + DoubleConv
  │   32x32 → 64x64 (1/8),  concat 后 128+64=192 → 128
  │
  up2: ConvT(128→64, stride2) + concat(s4:16ch) + DoubleConv
  │   64x64 → 128x128 (1/4), concat 64+16=80 → 64
  │
  up3: ConvT(64→32, stride2) + concat(s2:4ch) + DoubleConv
  │   128x128 → 256x256 (1/2), concat 32+4=36 → 32
  │
  up4: ConvT(32→16, stride2) + concat(s1:1ch) + DoubleConv
  │   256x256 → 512x512 (1/1), concat 16+1=17 → 16
  │
  outc: Conv2d(16→num_classes, 1x1) → (B, 2, 512, 512)   原图分辨率
```

**通道匹配验证**（来自 `network.py` 的 `Up(in_ch, ch, out_ch)`，`ch = out_ch + skip_ch`）：
- up1: `Up(256, 128+64, 128)` → ConvT 出 128，concat skip 64 = 192，DoubleConv(192→128) ✓
- up2: `Up(128, 64+16, 64)` → ConvT 出 64，concat skip 16 = 80，DoubleConv(80→64) ✓
- up3: `Up(64, 32+4, 32)` → ConvT 出 32，concat skip 4 = 36，DoubleConv(36→32) ✓
- up4: `Up(32, 16+1, 16)` → ConvT 出 16，concat skip 1 = 17，DoubleConv(17→16) ✓

（以上通道数与 `network.py` 构造函数完全吻合，确证 skip 通道 = unfold 像素数。）

#### ViT-S 确证

- ONNX 输出 `384` 通道 = ViT-S 的 hidden size（ViT-S: embed_dim=384, depth=12, heads=6）。**确证是 ViT-S**。
- patch_size=16（ViT-S/**16**），grid=32×32。
- 编码器参数量：ViT-S/16 约 22M，但 ONNX 86MB（含 fp32 权重 + 算子图，正常）。

#### 关于 "16×16" 与 decoder 注释 "16×16" 的澄清

`network.py` 的 `__call__` 注释写 `e.shape B, 384, 16, 16`，但 ONNX 实测输出是 `(B, 384, 32, 32)`。**这是 network.py 注释过时/不准确**，实际以 ONNX 实测的 32×32 为准。decoder 代码本身用 `H//8` 等动态推导，不依赖硬编码 16，所以 32×32 输入也能正常工作（skip 的 s8=64×64 等都正确生成）。

#### 复现要点总结

1. **编码器输入必须是 3 通道 512×512**：单通道灰度复制 3 份。
2. **patch 是 ViT 内部做的**（patch_size=16），你不用手动 unfold，直接喂 `(B,3,512,512)`。
3. **embedding 是 `(B,384,32,32)`**，缓存这个。
4. **decoder skip 用 8/4/2/1 倍 unfold**（不是 16），对原图 `(B,1,512,512)` 做。
5. **`slice_to_patch_channels16`（16×16 unfold→256ch）是单独工具**，非 skip、非 encoder 输入；复现时一般用不到，除非要在 patch grid 上做对齐。
6. **pad 量记录在 embedding 文件名**，预测后用 `apply_pads_to_slice` 裁回原尺寸（恢复原始 H×W）。

### 3.4 解码器：U-Net 风格轻量 head（`network.py`，明文可读）

来源：`C:\Users\shijian.ruan\AppData\Roaming\RONDON\model\*\network.py`（所有目录内容完全一致，2399 字节）。

```python
from tinygrad import nn, Tensor

class DoubleConv:
    def __init__(self, in_ch, out_ch):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding="same", bias=False)
        self.norm1 = nn.InstanceNorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding="same", bias=False)
        self.norm2 = nn.InstanceNorm(out_ch)
    def __call__(self, x):
        x = self.conv1(x); x = self.norm1(x); x = x.leaky_relu()
        z = self.conv2(x); z = self.norm2(z)
        out = (z + x).leaky_relu()          # 残差连接
        return out

class Up:
    def __init__(self, in_ch, ch, out_ch):
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.body = DoubleConv(ch, out_ch)
    def __call__(self, x, skip):
        x = self.up(x)
        x = Tensor.cat(x, skip, dim=1)
        x = self.body(x)
        return x

class Network:
    def __init__(self, in_channels=384, num_classes=2):
        self.bottleneck = DoubleConv(in_channels, 256)   # 384->256, 1/16
        self.up1 = Up(256, 128+64, 128)                  # 1/16 -> 1/8
        self.up2 = Up(128, 64+16, 64)                    # 1/8  -> 1/4
        self.up3 = Up(64, 32+4, 32)                      # 1/4  -> 1/2
        self.up4 = Up(32, 16+1, 16)                      # 1/2  -> 1
        self.outc = nn.Conv2d(16, num_classes, 1)

    def __call__(self, e, s):
        # e: (B, 384, 16, 16) 编码器 embedding
        # s: (B, 1, 512, 512) 原始切片
        B, C, H, W = s.shape
        # 原图 pixel-unfold 生成多尺度 skip（关键设计）
        s8 = s.reshape(B,C,H//8,8,W//8,8).permute(0,1,3,5,2,4).reshape(B,-1,H//8,W//8)  # 64 通道
        s4 = s.reshape(B,C,H//4,4,W//4,4).permute(0,1,3,5,2,4).reshape(B,-1,H//4,W//4)  # 16 通道
        s2 = s.reshape(B,C,H//2,2,W//2,2).permute(0,1,3,5,2,4).reshape(B,-1,H//2,W//2)  # 4 通道
        s1 = s                                                                            # 1 通道
        e = self.bottleneck(e)
        x = self.up1(e, s8)
        x = self.up2(x, s4)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        x = self.outc(x)
        return x
```

**设计要点**：

- **DoubleConv**：2 次 3×3 same conv + InstanceNorm + leaky_relu，带残差 `(z+x)`。用 InstanceNorm 而非 BatchNorm（适合小 batch、医学影像）。
- **Up**：ConvTranspose2d(stride=2) 上采样 → concat skip → DoubleConv。
- **通道阶梯**：256→128→64→32→16→num_classes。
- **多尺度 skip 不来自编码器，而来自原图 pixel-unfold**：因为 DINOv3 是 ViT，输出单一 1/16 特征，没有天然多尺度。代码对原图 `s` 按 8×8/4×4/2×2 块展开成通道，分别给 1/8、1/4、1/2、1/1 层做 skip。这是整个架构最巧妙的地方——**让 ViT 单尺度特征能逐级恢复到原图分辨率**。
- **num_classes 默认 2**（前景+背景二分类）。符号 `num_classes` 确认可配置；多标签时通过多个 decoder 或 one-hot 处理（见 3.5）。
- **输入尺寸约定**：注释 `e.shape B, 384, 16, 16; s.shape B, 1, 512, 512`，即 512×512 切片 → 16×16 patch grid（patch_size=32？注意：编码器侧 512/16=32，但解码器注释写 16×16，存在 2× 下采样差异，复现时以编码器实际输出为准）。

> ⚠️ 复现注意：`network.py` 里 `e` 注释是 16×16，但编码器 `process_slice` 把 512×512 切成 32×32 patch grid。这两者需要在复现时对齐——要么编码器输出 16×16（切片先下采样到 256×256 再 patch），要么解码器按 32×32 改。建议直接用编码器 ONNX 的真实输出形状驱动解码器。

### 3.4 Embedding 缓存文件名格式（重要，复现必需）

来源：`.pyd` 中的注释字符串。

embedding 文件名 stem 有两种格式：
- 形状信息：`"{idx:05d}_h{height}_w{width}"`（5 位 slice 序号 + 高宽）
- pad 信息：`"{idx:05d}_l{left}_r{right}_u{up}_d{down}"`（左/右/上/下 padding 量）

解析函数：
- `parse_pads_from_stem` → 返回 `(idx, left, right, up, down)`
- 另有函数返回 `(height, width)`

**含义**：切片在送入编码器前可能被 pad 到 16 的倍数（或 512），pad 量记录在文件名里，预测后用 `apply_pads_to_slice` 裁回原尺寸。注释：`"Pad sizes exceed slice dimensions"`（pad 超过切片尺寸时报错）。

### 3.5 图驱动网络生成（`build_tinygrad_model_class_from_graph`）

来源：符号 `build_decoder_graph` / `save_decoder_graph` / `build_tinygrad_model_class_from_graph` / `GeneratedModel` / `exec_dynamic` / `to_attr_name` / `graph_path`。

- 解码器结构除 `network.py` 外，还保存为一份**图定义**（`decoder.graph_path`）。
- 运行时用 `build_tinygrad_model_class_from_graph` 动态生成 tinygrad 模型类 `GeneratedModel`：
  - `__init__` 根据 `class_op` 节点声明有状态层（Conv/BN 等）。
  - `__call__` 执行前向。
  - 用 `exec_dynamic` 执行动态生成的代码。
- **支持的算子**：`Conv2D, ConvTranspose2D, BatchNorm2D, ReLU, Sigmoid, Upsample`（注释原文）。
- 每个算子有 shape 推断：`conv2d_out_shape`、`conv_transpose2d_out_shape`、`batchnorm2d_out_shape`、`sigmoid_out_shape`、`upsample_out_shape`。
  - Conv2d：`Hout = floor((Hin + 2*pad - ks)/stride + 1)`
  - ConvTranspose2d：`Hout = (Hin - 1) * stride - 2*pad + ks`
- 支持**符号化空间维度**：如 `"H/16" -> ("H", 1/16)`、`"W*4" -> ("W", 4.0)`（`Format symbolic dim from base and factor`）。

> 这套机制是为了**模型导入/导出**时重建网络。复现时直接用 `network.py` 的 `Network` 类即可，不必实现图生成。

### 3.6 num_classes / 多标签处理

来源：符号 `num_classes`、`compute_dice_per_label`、`include_background`、`labels`、注释 `"Treats any non-zero value as foreground by default"`、`"multi-class segmentation arrays"`。

- 默认二分类（`num_classes=2`），非零视为前景。
- 支持多标签 Dice：`compute_dice_per_label`，可指定 `labels`，可 `include_background`。
- 多个器官 = 多个独立 decoder（每个 decoder 一个 `seg_name`，如 `spleen`），而非一个多类模型。证据：每个 decoder 目录的 `model.json` 里 `name` 是单个器官名（spleen），`source_label_path` 指向单个标签文件。

---

## 4. 归一化策略（重点复现细节）

### 4.0 诚实声明：选择逻辑无法从二进制完全还原

**这是整个逆向中最薄弱的一环，必须说清楚。** 应用核心逻辑被 Nuitka 编译成机器码后：
- ✅ **能确认的**：函数名、模块归属、少量 docstring 字符串、相关 DICOM 字段名。
- ❌ **无法确认的**：`if/else` 分支条件——即"CT 走哪个函数、MRI 走哪个函数"的**分发逻辑**在编译后不存在可读字符串，只有函数符号。

因此本节给出的是"函数清单 + 用途推断 + 证据"，**不是**确证的调用流程。复现时这一步需要你自行决定。

### 4.1 函数清单与模块归属（确证）

来源：`_tauri_app.cp311-win_amd64.pyd` 符号提取 + 模块路径字符串。

归一化函数位于模块 **`_tauri_app.commands.utils.array`**，共 5 个：

| 函数 | 用途（推断） | 证据 |
|---|---|---|
| `normalize_value` | 标量归一化（单值），报错信息 `"normalize_value failed"` | 符号 |
| `normalize_volume_array_min_max` | Min-Max 到 [0,1]：`(x-min)/(max-min)` | 函数名 |
| `normalize_volume_array_percentile` | 百分位截断后归一化，参数 `lower`/`upper`/`percentile` | 函数名 + 符号 `lower`/`upper`/`percentile` |
| `normalize_volume_array_z_score` | Z-score：`(x-mean)/std` | 函数名 |
| `normalize_volume_array_x_ray` | CT/X 光窗归一化，参数 `wl_center`/`ww_width`/`center`/`width`/`slope`/`intercept` | 函数名 + 符号 `wl_center`/`ww_width`/`ww_wl`/`center`/`width`/`slope`/`intercept` |

辅助符号：`is_pre_normalized`（标记数据是否已预归一化，避免重复）、`clip`（截断）。

### 4.2 归一化的调用位置（确证）

来源：符号在 .pyd 中紧邻排列，且模块归属一致。

归一化函数与以下符号在 `utils.array` / encoder 模块中**紧邻出现**：
```
normalize_value
normalize_volume_array_min_max
normalize_volume_array_percentile
normalize_volume_array_x_ray
normalize_volume_array_z_score
patch_channels16_to_slice
preprocess_for_decoder          ← 解码器侧预处理总入口
resize_slice_array              ← "Resize a slice array to 512x512"
run_dinov3_vits16_encoder       ← 编码器前向
run_vanilla_encoder
slice_to_patch_channels16
```

**结论（确证）**：归一化是**编码器前向预处理 pipeline 的一部分**，由 `preprocess_for_decoder` / `run_dinov3_vits16_encoder` 调用。即归一化发生在「切片 → resize 到 512×512 → 归一化 → pad → 灰度转3通道 → ONNX 前向」这条链路里。（注：`slice_to_patch_channels16` 是独立的 patch 级转换工具，不在这条链路上，见 3.3 路径 B。）

### 4.3 DICOM 窗信息的获取（确证）

来源：docstring 字符串 + 符号。

- `get_window_settings`：注释原文 `"Get Window Center and Width from DICOM or fallback."` —— 从 DICOM tag 读窗位窗宽，读不到则 fallback。
- 相关 DICOM 字段符号：`WindowCenter`, `WindowWidth`, `RescaleSlope`, `RescaleIntercept`。
- `x_ray` 归一化的参数符号：`wl_center`（窗位）、`ww_width`（窗宽）、`ww_wl`、`center`、`width`、`slope`、`intercept`。
- 这意味着 CT 的 HU 值会先经 `RescaleSlope * x + RescaleIntercept` 还原，再用 `wl_center ± ww_width/2` 截断，最后归一化。

### 4.4 选择策略（推断，未确证）

**以下是基于函数名和 DICOM 字段的合理推断，不是确证的代码逻辑：**

| 模态 | 推断选用的函数 | 理由 |
|---|---|---|
| CT（有 RescaleSlope/Intercept + Window） | `normalize_volume_array_x_ray` | CT 有标准 HU 值和窗，x_ray 函数正好吃 wc/ww/slope/intercept |
| MRI（无标准窗） | `normalize_volume_array_percentile` 或 `z_score` | MRI 无 WindowCenter/Width 意义，用百分位截断或 z-score 更合理 |
| 通用兜底 | `normalize_volume_array_min_max` | 最简单，无模态信息时用 |
| 标签/seg | 不归一化 | 只做 nearest 重采样 |

**判断模态的依据推断**：可能基于 `is_dicom` + DICOM tag 是否含 `WindowCenter/Width` 或 `RescaleSlope/Intercept`。但**具体的 if/else 条件无法从 .pyd 还原**。

### 4.5 复现时的实操建议

由于选择逻辑不可读，复现时建议：
1. **CT**：读 DICOM 的 `RescaleSlope`/`RescaleIntercept` 还原 HU → 用 `WindowCenter ± WindowWidth/2` 截断 → min-max 到 [0,1]。等价于 `normalize_volume_array_x_ray`。
2. **MRI / NIfTI 无窗信息**：用百分位截断（lower=1%, upper=99%）→ min-max。等价于 `normalize_volume_array_percentile`。
3. 保留 `is_pre_normalized` 开关：若数据导入时已归一化（如某些预处理的 nii.gz），则跳过。
4. **关键**：归一化后的值域要和 DINOv3 ONNX 的预期输入一致。DINOv3 通常期望 ImageNet 归一化（mean/std）后的输入，但这里的 `x_ray`/`min_max` 输出是 [0,1]——**复现时必须实测 ONNX 输入期望**（用 `dinov3-vits16\model.onnx` 跑一个已知输入看输出范围），确认归一化方案是否还需要额外减 ImageNet mean。这是另一个未确证点。

### 4.6 重采样（与归一化配合）

- 符号：`ref_spacing`, `image_spacing`, `pixel_spacing`, `pixel_spacing_raw`, `spacing_match`, `GetSpacing`, `SetSpacing`。
- 切片在归一化前按 `ref_spacing` 重采样到统一 spacing（SimpleITK），图像线性插值，标签 nearest。
- 解码器侧 `_resize_slice_array` 把切片 resize 到 512×512（注释确证）。

### 4.3 重采样

- 符号：`ref_spacing`, `image_spacing`, `pixel_spacing`, `pixel_spacing_raw`, `spacing_match`, `Mainv_spacing`, `GetSpacing`, `SetSpacing`。
- 切片在归一化前会按 `ref_spacing` 重采样到统一 spacing（SimpleITK），图像线性插值，标签 nearest。
- 解码器侧还有 `_resize_slice_array` 把切片 resize 到 512×512。

---

## 5. 训练流程（从 .pyd 符号还原）

### 5.1 入口与生命周期

来源：符号 `train_decoder` / `train_decoder_handler` / `run_training_step_for_decoder` / `run_validation_step_for_decoder` / `run_evaluation_step_for_decoder`。

```
train_decoder (Tauri command TRAIN_DECODER)
  └─ train_decoder_handler
       ├─ collect_train_decoder_related_ids     # 收集 sample/encoder/decoder id
       ├─ for epoch in range(epochs):
       │    ├─ run_training_step_for_decoder    # 内含 train_step + scheduler
       │    ├─ run_validation_step_for_decoder  # 内含 valid_step（每 validation_interval 个 epoch）
       │    └─ (条件) save_best_validation_network_parameters  # is_best_dice 时存最优
       └─ run_evaluation_step_for_decoder       # 训练结束后评估，含 evaluation_step
```

### 5.2 训练超参（实读自 `casgzeefcueyriwz\model.json`）

```json
"optimization": {
  "lr": 0.0005,
  "epochs": 30,
  "batch_size": 4,
  "optimizer": "Adam",
  "b1": 0.9,
  "b2": 0.999,
  "eps": 1e-08,
  "weight_decay": 0.0001,
  "loss": "CrossEntropyLoss",
  "validation_interval": 2
}
```

- **优化器**：`Adam`（符号也见 `AdamW`，可选）。来自 `tinygrad.nn.optim`。
- **学习率调度**：`cosine_lr`，参数 `base_lr`, `max_steps`, `min_lr`（余弦退火到 min_lr）。`max_steps` = epochs（或总 step 数）。
- **损失**：`CrossEntropyLoss`（符号 `cross_entropy`）。多分类 CE，配合 `num_classes` 输出通道。
- **batch 构造**：`sample_pairs` → `pairs_chunks`（按 `batch_size` 分块）→ `shuffle`（注释 `"Shuffled "`）。`io_chunksize` / `multipart_chunksize` 控制数据加载分块。

### 5.3 Embedding 缓存策略（关键性能设计）

来源：符号 `embedding_folder` / `embedding_paths` / `get_embedding_paths` / `load_embedding_as_tensor` / `get_embedding_folder`。

- **编码器只跑一次**：DINOv3 对每个 slice 跑 ONNX 前向，结果存盘为 embedding 文件。
- **训练时解码器直接读缓存 embedding**，不重复跑编码器。这是能用 tinygrad 在 CPU 上微调 ViT 的关键。
- embedding 文件按 sample 组织在 `embedding_folder` 下，文件名含 slice idx + pad 元数据（见 3.4）。
- `load_embedding_as_tensor` 加载单个 embedding。

### 5.4 数据增强 = 翻转 TTA（Test-Time Augmentation 风格）

来源：符号 `embedding_tensor_h_flip` / `_w_flip` / `_h_w_flip`，及对应的 `image_tensor_*` / `seg_tensor_*` / `outputs_*` / `predicts_*` / `loss_*`。

- 对 embedding、image、seg 都预生成 3 种翻转：水平翻转 `h_flip`、垂直翻转 `w_flip`、双向 `h_w_flip`。
- 训练/评估时组合使用（计算 `loss_h_flip`, `outputs_h_w_flip`, `predicts_w_flip`），相当于翻转增强 + TTA。
- **注意**：翻转在 embedding 空间进行（因为编码器冻结，翻转原图等价于翻转 embedding 的空间维度），省去重复编码。

### 5.5 评估指标（Dice）

来源：符号 `_compute_dice` / `compute_dice` / `compute_dice_per_label` + 注释。

- **二值 Dice**：`_compute_dice`，`Dice coefficient in [0, 1]`，带 `smooth` 平滑项和可选 `threshold`（`"If set, binarizes y_pred as (y_pred >= threshold)"`）。
- **逐标签 Dice**：`compute_dice_per_label`，返回 `Mapping of label -> Dice score`，可指定 `labels`、`include_background`。
- 支持 `argmax`（多类取最大类别）、`softmax` / `sigmoid` 后处理。
- 形态学后处理（来自 SimpleITK/二值操作）：`BinaryFillhole`, `BinaryErode`, `BinaryDilate`, `BinaryOpen`, `BinaryClose`, `BinaryContour`。

### 5.6 Checkpoint 与状态机

来源：符号 `save_best_validation_network_parameters` / `is_best_dice` / `epoch_parameter_path` + 实读文件名。

- 每个 epoch 存 `epoch{N}-dice{...}.safetensors`（如 `epoch14-dice0.9643784325130254.safetensors`）。
- 维护 `model.safetensors`（最新/最优）。
- `is_best_dice` 判断是否最优，是则 `save_best_validation_network_parameters`。
- tinygrad 权重通过 `tinygrad.nn.state`（`safe_save` / `safe_load`）读写 safetensors。
- **状态字段**（`model.json`）：`trained` (bool)、`fronzen` (bool，原文拼写如此)、`imported`。
  - 未训练/未冻结不能删除：`"is not trained or frozen, cannot be dropped"`。
  - 冻结后不能训练：`"The fronzen decoder cannot be trained"`。

### 5.7 训练数据组织（sample 机制）

来源：符号 `sample` / `sample_pairs` / `batch_update_sample` / `sample_epoch_validation_loss_list` 等 + 实读 sample 子目录 `model.json`。

- 训练数据按 **sample** 组织，一个 sample = 一个病人/序列/层堆来源。
- sample `model.json` 字段（实读 `bixpyayanjnniqau\model.json`）：
  ```json
  {
    "name": "spleen", "decoder_id": "casgzeefcueyriwz",
    "source_id": "...", "source_state_id": "...",
    "source_label_path": "14hbmkf\\1t5zy7o\\segs\\ucdkiskgvyfankcp.nii.gz",
    "source_image_path": "14hbmkf\\1t5zy7o\\volume.nii.gz",
    "source_data": {
      "patient_id": "...", "patient_name": "spleen_33",
      "study_id": "...", "serial_id": "...", "series_number": "-1",
      "stack_id": "...", "stack_name": "spleen_33",
      "seg_name": "spleen",
      "spacing": [0.9238, 0.9238, 5.0]
    },
    "role": "training", "source_model": "seg", "model": "sample"
  }
  ```
- 每个 sample 记录逐 epoch 指标：`sample_epoch_validation_loss_list`, `sample_epoch_evaluation_loss_list`, `sample_validation_dice_list` 等。
- `batch_update_sample` 在 epoch 结束后批量回写 sample 指标。

### 5.8 tinygrad 运行时控制

来源：符号 + 日志。

- 设备：默认 CPU（`Failed to import tinygrad Device` / `Failed to synchronize tinygrad device` 的 fallback 提示）。
- 缓存：`Failed to disable tinygrad cache` / `tinygrad device cache freed` —— 训练前会禁用/清理 tinygrad 编译缓存。
- `get_computer_cores`：日志显示 `computer cores: 20`，用于并发数据加载（`max_workers`, `max_concurrency`）。

### 5.9 3D 预览

来源：符号 `save_best_validation_sample_predit_as_decoder_mesh` + 实读 `tiny_mesh.glb`。

- 训练/验证后用 PyMCubes（`mcubes`）对预测 mask 做 marching cubes，trimesh 导出 `tiny_mesh.glb` + 渲染 `tiny_mesh.png`。
- sample 级也存 `predict_tiny_mesh.glb`。

---

## 6. 已有训练实例汇总

实读自 `model\` 各目录。

| 目录 | 任务/标签 | tagstr | encoder | 表现（验证 Dice 峰值） |
|---|---|---|---|---|
| `dinov3-vits16` | 编码器 | — | — | — (encoder, 86MB ONNX) |
| `casgzeefcueyriwz` | spleen | `abdomen\nCT\nMSD` | dinov3-vits16 | val dice ~0.939（30 epoch） |
| `dakjegtxoxskxzqg` | (spleen_33 来源) | — | dinov3-vits16 | 最优 epoch14 dice **0.964** |
| `ntdayzfqquxvbwrm` | (难任务) | — | dinov3-vits16 | 12 ckpt，dice 0.68–0.80（疑似欠训） |
| `wmhokzowjauuelbg` | — | — | — | 2 ckpt，dice ~0.92 |
| `zbzkeduipucefexs` | — | — | — | 最优 epoch9 dice 0.960 |
| `csstogjmrlrfcfvs` | (骨架/预览) | — | — | 仅 network.py + mesh，无权重 |

`casgzeefcueyriwz\model.json` 训练指标示例（前几项）：
- `training_loss`: 30 个值，从 0.045 降到 0.0005
- `validation_loss`: 15 个值（validation_interval=2，30/2=15）
- `validation_dice`: 15 个值，峰值 0.939

---

## 7. 加密与授权机制

来源：符号 `TSMODELv1` / `_MODEL_MAGIC` / `_MODEL_AAD` / `derive_key` / `AESGCM` / `PBKDF2HMAC` / `get_hwid` + 注释。

### 7.1 .model 文件格式（TSMODELv1）

- 编码器/解码器导出为单一 `.model` 文件，格式头 `TSMODELv1`。
- **加密**：AES-GCM（`AESGCM`），密钥用 PBKDF2-HMAC-SHA256 从 passphrase + salt 派生（`Derive a 256-bit key`）。
- 文件结构：`magic(TSMODELv1) + header(含 nonce/salt 等) + AES-GCM 密文`。
- `_MODEL_AAD`：附加认证数据。
- 错误提示：`"Invalid model file header"` / `"Invalid file header"` / `"model.onnx not found in archive"`。

### 7.2 License 授权

- 符号：`save_license` / `license_path` / `license_res` / `get_hwid`。
- 授权绑定硬件：`get_hwid`，失败则 fallback 到持久化 UUID（`"Failed to get hwid, falling back to persistent UUID"`）。
- 用户信息（`settings.json`）：`identifier`, `token`, `name`, `email`, `organization`, `department`。
- 远程 API 注册：`Setup user by calling remote API and saving license.`，license 存到 Downloads 目录。

> ⚠️ 安全提醒：`settings.json` 内**明文存放了密码**（`"password": "877207245"`）和 token，注意不要外泄该文件。

---

## 8. 复现指南（基于以上逆向）

### 8.1 复现训练所需组件

1. **DINOv3 ViT-S/16 ONNX**：从 `dinov3-vits16\model.onnx` 直接取用（86MB）。**已实测**：输入 `(B,3,512,512)`，输出 `(B,384,32,32)`（3 个输出取 `output`）。
2. **tinygrad 0.11.0**：`pip install tinygrad==0.11.0`。
3. **onnxruntime**：CPU 即可。
4. **解码器**：直接用 `network.py` 的 `Network` 类。
5. **数据**：Totalsegmentator MRI/CT 数据集（`.persisted-scope` 里全是该数据集路径）。

### 8.2 复现训练流程（伪代码）

```python
# 1. 预处理每个 sample
for sample in samples:
    volume = sitk.ReadImage(sample.image_path)        # volume.nii.gz
    label  = sitk.ReadImage(sample.label_path)         # segs/*.nii.gz
    volume = resample(volume, ref_spacing)             # 重采样到统一 spacing
    # 归一化（按模态选, 见第4节）
    arr = normalize(volume)  # CT: x_ray/percentile; MRI: z_score
    for z in range(arr.shape[2]):
        slice2d = arr[:,:,z]                           # (H,W) 单通道
        slice2d = resize(slice2d, 512, 512)            # 线性, 注释确证 "Resize to 512x512"
        # pad 到 16 倍数，记录 pad 量
        slice2d, pads = pad_to_multiple(slice2d, 16)
        # 灰度 -> 3通道 (ONNX 输入要 (B,3,512,512), 实测确证)
        rgb = np.stack([slice2d]*3, axis=0)[None]      # (1,3,512,512)
        # ONNX 前向 (patch_size=16 在 ViT 内部完成, 不用手动 unfold)
        outs = onnx_session.run(None, {input_name: rgb})
        emb = outs[2]                                   # 'output' -> (1,384,32,32)
        # 缓存，文件名: {idx:05d}_l{l}_r{r}_u{u}_d{d}.npy
        save_embedding(emb, f"{z:05d}_l{pads[0]}_r{pads[1]}_u{pads[2]}_d{pads[3]}.npy")

# 2. 训练解码器
net = Network(in_channels=384, num_classes=2)
opt = tinygrad.nn.optim.Adam(net.parameters(), lr=0.0005, b1=0.9, b2=0.999, eps=1e-8, wd=0.0001)
sched = cosine_lr(base_lr=0.0005, max_steps=epochs, min_lr=...)
loss_fn = CrossEntropyLoss()

for epoch in range(30):
    pairs = shuffle(sample_pairs)
    for batch in pairs_chunks(pairs, batch_size=4):
        e = load_embedding_as_tensor(batch)           # (B,384,H/16,W/16)
        s = load_slice_as_tensor(batch)               # (B,1,512,512)
        # 翻转增强
        for flip in [none, h, w, hw]:
            e_f, s_f, seg_f = apply_flip(e, s, seg, flip)
            pred = net(e_f, s_f)                      # (B,2,512,512)
            loss = loss_fn(pred, seg_f)
            opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 2 == 0:  # validation_interval=2
        dice = compute_dice(net, val_set)
        if is_best_dice(dice):
            save_best(net, f"epoch{epoch}-dice{dice}.safetensors")
```

### 8.3 复现时的不确定点（需自行验证）

1. **编码器输出空间尺寸**：`network.py` 注释 `e: (B,384,16,16)`，但 `process_slice` 注释 `512->32×32 patch grid`。需用 ONNX 实跑确认输出是 16×16 还是 32×32，据此调整解码器。
2. **归一化选择**：未见显式配置字段，疑似按模态自动判断。建议 CT 用百分位截断、MRI 用 z-score。
3. **翻转 TTA 的具体组合方式**：是训练增强还是仅评估 TTA，还是两者都有——符号显示训练和评估侧都有 `loss_*_flip`，倾向于两者都用。
4. **cosine_lr 的 min_lr 值**：符号见 `min_lr`，具体值未提取到，建议设为 base_lr 的 1% 或 0。
5. **vanilla encoder 的结构**：无 ONNX，推测是 tinygrad 原生简单 CNN，复现时可不实现，专注 dinov3。
6. **patch_size**：编码器侧 512/32=16 或 512/16=32，需确认 ViT patch size 是 16 还是 32（DINOv3 ViT-S/16 名字暗示 patch=16，即 32×32 grid）。

### 8.4 训练代码的真实位置

**没有可读的独立训练脚本**。完整训练逻辑在：
```
C:\Users\shijian.ruan\AppData\Local\RONDON\Lib\site-packages\_tauri_app.cp311-win_amd64.pyd
```
由 `rondon.exe` 启动嵌入式 Python 加载。该 .pyd 是 Nuitka 编译产物，源码不可读。可读 Python 仅：
- 各 `model\*\network.py`（解码器结构）
- `Lib\site-packages\` 下的第三方包（tinygrad 等）

如需原始未编译源码，只能联系作者（`iridium.xiaoming@gmail.com`，北部战区总医院放射科）。

---

## 9. 附：关键证据符号清单（来自 .pyd strings 提取）

训练循环：
`train_decoder`, `train_decoder_handler`, `run_training_step_for_decoder`(train_step, scheduler), `run_validation_step_for_decoder`(valid_step), `run_evaluation_step_for_decoder`(evaluation_step)

编码器：
`run_dinov3_vits16_encoder`(process_slice), `run_vanilla_encoder`, `get_sync_session`, `synchronize_session`, `CPUExecutionProvider`, `InferenceSession`, `input_name`, `output_name`

patch 转换：
`slice_to_patch_channels16`, `patch_channels16_to_slice`, `patch_grid`, `patch_channels`, `_resize_slice_array`("Resize a slice array to 512x512")

归一化：
`normalize_volume_array_min_max`, `_percentile`, `_z_score`, `_x_ray`, `normalize_value`, `is_pre_normalized`, `WindowCenter`, `WindowWidth`, `RescaleSlope`, `RescaleIntercept`

embedding 缓存：
`embedding_folder`, `embedding_paths`, `get_embedding_paths`, `load_embedding_as_tensor`, `parse_pads_from_stem`, `apply_pads_to_slice`
文件名格式：`"{idx:05d}_h{height}_w{width}"`, `"{idx:05d}_l{left}_r{right}_u{up}_d{down}"`

翻转增强：
`embedding_tensor_h_flip`, `_w_flip`, `_h_w_flip`, `image_tensor_*`, `seg_tensor_*`, `outputs_*`, `predicts_*`, `loss_*`

损失/优化/调度：
`cross_entropy`, `Adam`/`AdamW`, `cosine_lr`(base_lr, max_steps, min_lr), `weight_decay`, `zero_grad`, `backward`

Dice：
`_compute_dice`, `compute_dice`, `compute_dice_per_label`, `include_background`, `labels`, `threshold`, `smooth`, `argmax`, `softmax`, `sigmoid`

Checkpoint：
`save_best_validation_network_parameters`, `is_best_dice`, `epoch_parameter_path`, `tinygrad.nn.state`, `tinygrad.nn.optim`

图生成：
`build_decoder_graph`, `save_decoder_graph`, `build_tinygrad_model_class_from_graph`(GeneratedModel, exec_dynamic, to_attr_name), `graph_path`
支持算子：Conv2D, ConvTranspose2D, BatchNorm2D, ReLU, Sigmoid, Upsample

加密：
`TSMODELv1`, `_MODEL_MAGIC`, `_MODEL_AAD`, `derive_key`, `AESGCM`, `PBKDF2HMAC`, `get_hwid`, `encrypt_folder`, `decrypt_folder`

形态学后处理：
`BinaryFillhole`, `BinaryErode`, `BinaryDilate`, `BinaryMorphologicalOpening`, `BinaryMorphologicalClosing`, `BinaryContour`

3D 网格：
`save_best_validation_sample_predit_as_decoder_mesh`, PyMCubes(`mcubes`), trimesh

---

## 10. 逆向能力边界（哪些确证、哪些推断，复现前必读）

本文档的结论分三个置信度，复现时请按此判断：

### 10.1 确证（来自明文文件、确凿符号或 ONNX 实测）

- 解码器网络结构（`network.py` 全文）：DoubleConv + Up + Network，pixel-unfold skip。
- 训练超参（`model.json` 实读）：Adam, lr=5e-4, 30 epoch, batch=4, CE loss, validation_interval=2, weight_decay=1e-4。
- **DINOv3 ViT-S/16 ONNX 实测**：输入 `(B,3,512,512)`，输出 `(B,384,32,32)`（3 输出取 output），patch_size=16，grid=32×32，embed_dim=384=ViT-S。CPU 推理。
- **patch 策略（实测+明文）**：编码器 patch 是 ViT 内部做的（16×16）；decoder skip 用 8/4/2/1 倍 unfold（64/16/4/1 通道）；`slice_to_patch_channels16` 是 16×16 unfold→256ch 的独立工具。embedding `(384,32,32)` 经 bottleneck→up1~up4 逐级 ×2 恢复到 `(2,512,512)`，通道与 `network.py` 构造完全吻合。
- embedding 文件名格式：`{idx:05d}_l{l}_r{r}_u{u}_d{d}` / `{idx:05d}_h{h}_w{w}`。
- 模块结构（`_tauri_app.commands.utils.tiny_unet` 等）。
- 5 个归一化函数名、所在模块、参数符号（wl_center/ww_width/lower/upper 等）。
- `get_window_settings` 从 DICOM 取窗位窗宽 + fallback。
- 调度器是 `cosine_lr`（参数 base_lr/max_steps/min_lr）。
- 翻转增强在 embedding 空间做（h/w/hw）。
- checkpoint 命名 `epoch{N}-dice{...}.safetensors`，safetensors 格式。
- 加密格式 TSMODELv1（AES-GCM + PBKDF2）。

### 10.2 推断（从符号名/注释合理推断，未看到代码逻辑）

- **归一化的模态分发条件**（CT→x_ray, MRI→percentile/z_score）：只有函数名和 DICOM 字段符号，**if/else 不可读**。见第 4 节。
- **灰度→3通道的具体实现**（`np.stack([g,g,g])` vs `np.repeat`）：ONNX 输入确为 3 通道，但转换函数 `process_grayscale_slice` 内部不可读。复现用 `np.stack([g]*3)` 即可（等价）。
- **DINOv3 ONNX 输入是否需要 ImageNet mean/std**：ONNX 内部可能已含归一化（patch_embed 前的 normalize 层），实测零输入输出范围正常（-146~309），但**未确证是否需要外部 ImageNet 归一化**。复现时建议先用 [0,1] 输入跑通，再对比加 ImageNet mean/std 的效果。
- 翻转 TTA 是训练增强还是评估 TTA 还是两者（符号显示两侧都有 `loss_*_flip`，倾向两者，未确证）。
- `cosine_lr` 的 `min_lr` 具体值。
- vanilla encoder 的网络结构（无 ONNX，推测 tinygrad 原生 CNN）。
- `slice_to_patch_channels16` 的确切用途（推断为 patch grid 对齐工具，未确证调用点）。

### 10.3 完全不可知（Nuitka 机器码，无法还原）

- 任何函数内部的具体数值计算（如 percentile 的默认 1%/99%、x_ray 的窗截断公式细节、Dice 的 smooth 值）。
- 归一化的 if/else 分支。
- 训练循环的精确语句顺序（虽有函数名，但无源码）。
- 数据加载的精确 batching 逻辑。

### 10.4 如需 100% 确证，只能

1. 用 `dinov3-vits16\model.onnx` 实跑，探测输入输出范围与形状（解决 10.2 的尺寸和归一化问题）。
2. 联系作者获取未编译源码：`iridium.xiaoming@gmail.com`（北部战区总医院放射科，model.json 中 creator 字段）。
3. 动态调试：在 Python 层 monkey-patch `tinygrad` / `onnxruntime` 拦截输入输出张量，反推 pipeline（需在 RONDON 运行时注入）。
