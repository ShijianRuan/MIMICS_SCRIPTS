# RONDON 函数级深度分析（补充文档）

> 本文档是对 `RONDON_REVERSE_ENGINEERING.md` 的补充，基于对编译模块
> `C:\Users\shijian.ruan\AppData\Local\RONDON\Lib\site-packages\_tauri_app.cp311-win_amd64.pyd`
> 的系统性字符串提取（共 7577 个唯一字符串，存于 `C:\Users\shijian.ruan\_pyd_strings.txt`）。
>
> **方法说明**：应用核心被 Nuitka 编译成原生 .pyd，源码不可读。但 Nuitka 保留了：
> - 源文件路径（`_tauri_app\commands\utils\tiny_unet` 等）→ 还原完整源码树
> - 类/方法限定名（`uSample.get_pairs` 等）→ 还原类结构
> - 函数 docstring（`    Returns an array shaped...`）→ 还原函数语义
> - 异常/日志消息（`uEpoch `、`uShuffled `）→ 还原运行时叙事
>
> **局限**：数值常量（smooth、eps、lr 具体值）是机器码立即数，不以字符串存在，无法提取。
> 反射加载（import _tauri_app）因依赖完整 Tauri+pydantic+sqlmodel 运行时而失败，未能拿到函数签名。

---

## 1. 完整源码文件树（36 个 .py，从路径字符串还原）

```
_tauri_app/
├── __init__.py
├── __main__.py
├── commands/
│   ├── __init__.py
│   ├── app.py              # 应用初始化、DB、image/model store、license
│   ├── decoder.py          # 解码器 CRUD + train/run/download/upload/export
│   ├── encoder.py          # 编码器 CRUD + import/export
│   ├── _decoder.py         # decoder 内部实现
│   ├── _encoder.py         # encoder 内部实现
│   ├── sample.py           # 训练样本（含 get_pairs/get_embedding_paths）
│   ├── seg.py              # 分割标注（944+ 行，最大模块）
│   ├── image.py            # 图像导入
│   ├── patient.py / study.py / serial.py / stack.py / record.py / dicom_meta.py  # DICOM 数据模型
│   ├── group.py / contain.py  # 分组关系
│   ├── task.py             # 任务队列（add_task/_run_task/_start_next_from_queue）
│   ├── server.py           # 远程服务
│   └── utils/
│       ├── __init__.py
│       ├── array.py        # ★ 归一化 + resize + patch 转换 + tensor 加载
│       ├── tiny_unet.py    # ★ 训练循环核心
│       ├── network.py      # ★ 网络工厂 + 图生成
│       ├── layer.py        # ★ tinygrad 层封装 + shape 推断
│       ├── metric.py       # ★ Dice 评估
│       ├── image.py        # 图像处理
│       ├── feature.py      # 影像组学（pyradiomics）
│       ├── archive.py      # .model 加解密
│       ├── remote.py       # 远程 API（api.timeslice.space）
│       ├── db.py / fs.py / colors.py / msg.py
└── networks/
    └── decoder_network.py  # Network 类（= 明文 network.py）
```

> **关键认知**：之前以为训练代码"完全不可读"——实际 Nuitka 保留了源文件路径和大量 docstring，能还原出函数级语义，只是函数体（数值逻辑）不可读。

---

## 2. 数据模型类（从 `u<Class>.<attr>` 还原）

### 2.1 Decoder（解码器，SQLModel）
属性/方法：`Config, model, folder, folder_id, network_path, parameter_path, graph_path, tiny_mesh_path, tiny_mesh_preview_path, training_log_path`
- `network_path` → `network.py`
- `parameter_path` → `model.safetensors`
- `graph_path` → `graph.json`
- `training_log_path` → `training_log.log`

### 2.2 Encoder（编码器）
属性：`Config, model, folder, folder_id, model_path, meta_path, decoder_network_path`
- `model_path` → `model.onnx`
- `meta_path` → `model.json`
- `decoder_network_path` → `decoder_network.py`

### 2.3 Sample（训练样本，核心）
属性/方法：`Config, model, folder, folder_id, embedding_folder, preprocessed_label_folder, tiny_mesh_path, tiny_mesh_preview_path, predict_tiny_mesh_path, predict_tiny_mesh_preview_path`
方法：
- `get_embedding_folder` / `get_embedding_paths` — 定位 embedding 缓存
- `get_preprocessed_label_folder` / `get_preprocessed_label_paths` — 定位预处理后的标签
- `get_pairs` — 生成 (embedding, label) 训练对（含 `<genexpr>` 生成器）
- `get_status` — 样本状态

### 2.4 其他模型
- `Stack`: `volume_path, seg_folder, geo_path, get_preview_paths`
- `Seg`: `volume_path, folder, model`
- `Patient/Study/Serial/Group/Contain/Record/Task`: 各自 `model, folder_id` 等
- `DatabaseManager`: `__init__, get_session, get_sync_session, close, reset_database`

---

## 3. 网络类（从 `a<Class>` 和 `u<Class>.<method>` 还原）

| 类 | 来源 | 作用 |
|---|---|---|
| `DoubleConv` | network.py 明文 + .pyd | 2×conv+InstanceNorm+残差 |
| `Up` | network.py 明文 + .pyd | ConvTranspose + concat skip + DoubleConv |
| `Down` | .pyd（`uDown.__init__/__call__`） | 下采样块（vanilla encoder 用，非明文） |
| `UNet2D` | .pyd（`uUNet2D.__init__`） | 完整 U-Net（vanilla encoder 的候选结构） |
| `Network` | network.py 明文 | DINOv3 解码器 head |
| `VIT` | .pyd（`aVIT`） | ViT 标签/枚举类（无内部结构符号，非完整实现） |
| `GeneratedModel` | .pyd（图生成） | `build_tinygrad_model_class_from_graph` 动态生成的类 |

**vanilla encoder 推断**：`run_vanilla_encoder` + `UNet2D`/`Down` 存在 → vanilla encoder 是 tinygrad 原生的 U-Net（非 ViT），与 DINOv3 路径并列。`VIT` 仅作 encoder 类型枚举（`dinov3-vits16` vs `vanilla`）。

---

## 4. 训练循环叙事（从日志/异常消息还原）

### 4.1 入口与任务队列
```
train_decoder (Tauri command TRAIN_DECODER)
  → create_train_decoder_task
  → add_task (任务队列: _run_task, _start_next_from_queue, _count_running, _has_running)
  → _train_decoder_handler
       ├─ collect_train_decoder_related_ids   ("Failed to collect train decoder related ids")
       ├─ "Number of training samples: "
       ├─ "Run training on device " + device_id
       ├─ "Shuffled " + N + " training samples into " + M + " batches"
       ├─ for epoch:
       │    "Epoch " + n
       │    "Training: "    → run_training_step_for_decoder.train_step + scheduler
       │    "Validation: "  → run_validation_step_for_decoder.valid_step  ("validation time cost")
       │    "Save best validation network parameters at epoch " + n
       │    _log_call_back / validation_call_back   (回调前端进度)
       └─ "Training run finished"
```

### 4.2 训练步（`run_training_step_for_decoder.train_step`）
- 内嵌 `scheduler`（`cosine_lr`）
- 日志 `" at step "`、`" batches"`、`" samples missed"`
- 失败：`"Failed to process sample "`、`" has no valid training data"`

### 4.3 验证/评估步
- `run_validation_step_for_decoder.valid_step`：`"Validation: "`，产出 `validation_dice`/`validation_loss`
- `run_evaluation_step_for_decoder.evaluation_step`：`"Evaluation: "`，`" has no valid evaluation data"`
- `is_best_dice` → `save_best_validation_network_parameters`（存 `model.safetensors`）
- `save_best_validation_sample_predit_as_decoder_mesh`（存最优预测的 3D 网格）

### 4.4 解码器推理（`run_decoder`）
- `_run_decoder_handler`，内含 `run_decoder.<locals>.step`
- 命令 `RUN_DECODER`，`auto_seg`（批量自动分割，含 `_on_success`/`_run_decoder_handler`/`call_back`）

### 4.5 任务系统
`add_task` 管理并发：`_count_running` / `_count_running_by_category` / `_has_running` / `_run_task` / `_start_next_from_queue`。训练是异步任务，通过 `call_back` 回调前端。`is_constrained` 限制并发数（按 `computer cores: 20`）。

---

## 5. 数据预处理与加载（`array.py`，从 docstring 还原）

### 5.1 函数清单与 docstring
| 函数 | docstring |
|---|---|
| `slice_to_patch_channels16` | "Convert a 2D slice into 16x16 patch channels." → `(256, H/16, W/16)` |
| `patch_channels16_to_slice` | "Convert 16x16 patch channels back into a 2D slice." |
| `resize_slice_array` | "Resize a slice array to 512x512." |
| `apply_pads_to_slice` | "Apply padding to a slice array." pads=(left,right,up,down) |
| `parse_pads_from_stem` | "Parse slice index and pad metadata from embedding filename stem." → (idx,l,r,u,d) |
| `parse_shape_from_stem` | "Parse shape from embedding filename stem." → (height,width) |
| `get_window_settings` | "Get Window Center and Width from DICOM or fallback." |
| `process_grayscale_slice` | 灰度→3通道（报错 "Error processing grayscale slice"） |
| `preprocess_for_decoder` | 预处理总入口（内含 `process_sample`），日志 "Preprocess: " |
| `normalize_value` | 标量归一化（"normalize_value failed"） |
| `normalize_volume_array_min_max` | Min-Max |
| `normalize_volume_array_percentile` | 百分位截断（参数 lower/upper/percentile） |
| `normalize_volume_array_z_score` | Z-score |
| `normalize_volume_array_x_ray` | CT 窗归一化（参数 wl_center/ww_width/slope/intercept） |
| `load_embedding_as_tensor` | 加载 embedding（`*.npz` 格式） |
| `load_seg_as_tensor` | 加载标签 |

### 5.2 embedding 缓存格式（确证）
- **格式**：`*.npz`（numpy 压缩），不是 .npy
- **文件名**：`{idx:05d}_l{left}_r{right}_u{up}_d{down}` 或 `{idx:05d}_h{height}_w{width}`
- **校验**：`"has embeddings and slices with different names"`（embedding 与 slice 名不匹配时报错）、`"Embedding files not found at"`、`"Embedding folder not found at"`

### 5.3 翻转增强（TTA，确证在 embedding 空间）
符号：`embedding_tensor_h_flip` / `_w_flip` / `_h_w_flip`，及 `image_tensor_*` / `seg_tensor_*` / `outputs_*` / `predicts_*` / `loss_*`。
- 训练和评估侧都有 `loss_*_flip` → **训练增强 + 评估 TTA 都用**（之前标注为推断，现倾向确证两者都用）。

### 5.4 预处理标签
- `get_preprocessed_label_folder` / `get_preprocessed_label_paths`：标签也预处理后缓存（与 embedding 配对），存于 `preprocessed_label_folder`。

---

## 6. Dice 评估（`metric.py`，从 docstring 完整还原签名）

### 6.1 `compute_dice`（二值）
```
y_true, y_pred : np.ndarray
    Ground truth / Predicted segmentation array (or probabilities).
smooth : float
    Small value to avoid division-by-zero.
threshold : Optional[float]
    If set, binarizes y_pred as (y_pred >= threshold).
empty_score : float
    Score returned when both masks are empty.
Returns: Dice coefficient in [0, 1].
```
- 默认 `"Treats any non-zero value as foreground"`。
- 空对空返回 `empty_score`，避免除零。

### 6.2 `compute_dice_per_label`（多类）
```
y_true, y_pred : Integer-labeled segmentation arrays of the same shape.
labels : Optional[Iterable[int]]
    Specific labels to evaluate. If None, uses union of labels in inputs.
include_background : bool
    Whether to include label 0 in evaluation.
smooth : float
    Small value to avoid division-by-zero in extreme cases.
threshold : Optional[float]
    If threshold provided, applies to y_pred (useful for probabilities).
empty_score : float
    Score for empty-vs-empty masks for a label.
Returns: Mapping of label -> Dice score. ("Compute Dice per label for multi-class masks.")
```
- "Functions are robust to empty masks"。

> **数值未提取**：smooth、empty_score、threshold 的具体数值是机器码立即数，无法从字符串获取。复现建议 smooth=1e-6（标准做法），empty_score=1.0。

---

## 7. 网络图生成机制（`network.py` + `layer.py`）

### 7.1 图结构
- `build_decoder_graph` / `save_decoder_graph` → 存 `graph.json`（`"save decoder graph to"`、`"Decoder graph saved"`）
- 图元素：`nodes_list` / `edges_list` / `input_nodes` / `output_nodes` / `class_op` / `op_type` / `in_edges` / `out_edges`
- DAG 执行：`"Builds a DAG execution plan from edges."`、`"Runs the plan in __call__, supporting multiple inputs/outputs"`

### 7.2 动态类生成
`build_tinygrad_model_class_from_graph`：
- `GeneratedModel.__init__`：`"Declares stateful layers in __init__ based on class_op nodes."`
- `GeneratedModel.__call__`：执行 DAG
- `exec_dynamic`：动态执行生成的代码
- `to_attr_name`：节点名→合法属性名

### 7.3 支持的算子与 shape 推断（`layer.py`）
算子：`Conv2D, ConvTranspose2D, BatchNorm2D, ReLU, Sigmoid, Upsample`
shape 公式（docstring 确证）：
- Conv2D：`Hout = floor((Hin + 2*pad - ks)/stride + 1)`
- ConvTranspose2D：`Hout = (Hin - 1)*stride - 2*pad + ks`
- BatchNorm2D：`"does not change the shape"`
- Upsample：`"Nearest-neighbor upsample with scale factor"`（最近邻）
- Sigmoid/ReLU：元素级，形状不变（`sigmoid_out_shape` / `relu_out_shape`）

符号维度：`"H/16" -> ("H", 1/16)`、`"W*4" -> ("W", 4.0)`，`"If B is missing, set to 'B'; if H/W missing, leave symbolic"`、`"Ensure shape is a list of four elements [B, C, H, W]"`。

---

## 8. 加密与授权（`archive.py`）

### 8.1 .model 格式（TSMODELv1）
- 头：`TSMODELv1`（`_MODEL_MAGIC`），附加认证数据 `_MODEL_AAD`
- 加密：`AESGCM`（AES-256-GCM），密钥 `derive_key` = `PBKDF2HMAC-SHA256`（passphrase + salt，派生 256-bit key）
- 结构：`add_header`（magic + header 含 nonce/salt）+ AES-GCM 密文
- 错误：`"Invalid model file header"` / `"Invalid file header"` / `"Path header is required"`
- 操作：`encrypt_folder` / `decrypt_folder` / `encrypt_bytes_to_file` / `decrypt_bytes_from_file`

### 8.2 License
- `save_license` → 存到 Downloads 目录
- HWID 绑定：`get_hwid`，失败 fallback 到持久化 UUID
- 远程 API：`api.timeslice.space`（`server.py` / `remote.py`），`setupUser` 调用
- license 内容：`timeslice.license`

---

## 9. 远程服务与上传（`remote.py` + `server.py`）

- API 域名：`api.timeslice.space`
- 上传：`upload_decoder` / `upload_encoder`（含 `_handler`/`call_back`），用 boto3 上传到 R2（S3 兼容）
- 下载：`download_decoder` / `download_encoder`（`_handler`/`_import_decoder`/`call_back`），含进度 `"Downloading X%"`
- 日志上报：`"Failed to send log via webhook"`、`"The log file will be sent to the developer"`
- User-Agent：`Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/91`

---

## 10. 影像组学（`feature.py`）

- `RadiomicsFeatureExtractor`（pyradiomics），`extract_radiomics_features`
- `radiomics_seg` 命令（含 `_on_success`/`_run_radiomics_handler`/`call_back`）
- `_RadiomicsProgressHandler`（`__init__`/`emit` 进度）
- 失败：`"RadiomicsFeatureExtractor not available"`

---

## 11. tinygrad 运行时控制

- 设备：`tinygrad Device`，`"Run training on device "` + `device_id`
- GPU 检测：`"Could not retrieve GPU information"`（说明会尝试 GPU，但实测默认 CPU）
- 缓存：`"Failed to disable tinygrad cache"`、`"cache can be freed"` / `"cache cannot be freed"`、`"tinygrad device cache freed"`
- 同步：`"Failed to synchronize tinygrad device"`
- 优化器：`Adam` / `AdamW`（`tinygrad.nn.optim`）
- 权重 IO：`tinygrad.nn.state`（safetensors）
- 调度：`cosine_lr`（参数 base_lr/max_steps/min_lr）

---

## 12. 仍未确证的点（诚实清单）

| 项 | 状态 | 复现建议 |
|---|---|---|
| 归一化模态分发条件 | 不可读（机器码 if/else） | CT→x_ray, MRI→z_score/percentile |
| 归一化数值（百分位 1%/99% 等） | 不可读（立即数） | 1%/99% 标准值 |
| Dice 的 smooth/empty_score/threshold 值 | 不可读 | smooth=1e-6, empty_score=1.0 |
| cosine_lr 的 min_lr | 不可读 | base_lr 的 1% |
| 灰度→3通道实现 | 不可读 | np.stack([g]*3) 等价 |
| vanilla encoder(UNet2D)结构 | 不可读 | 用标准 UNet 复现 |
| train_step 内部语句 | 不可读 | 按超参+CE loss+Adam 复现 |
| 翻转 TTA 组合方式 | 倾向两者都用 | 训练增强+评估 TTA |
| DINOv3 ONNX 是否需 ImageNet mean/std | 未确证 | 实测 [0,1] vs ImageNet 归一化对比 |

---

## 13. 如何进一步突破（如需 100% 确证）

1. **动态拦截**（最有效）：在 RONDON 运行时用 `PYTHONSTARTUP` 或 sitecustomize 注入 monkey-patch，拦截 `tinygrad`/`onnxruntime`/`numpy` 的输入输出张量，打印实际归一化值、形状、loss 值。需在 `rondon.exe` 启动的嵌入式 Python 里注入。
2. **Nuitka 反编译**：用 Nuitka 常量 blob 解析工具（如 `nuitka-extractor`）从 .pyd 提取常量表，可能拿到更多默认参数值。
3. **ONNX 实测**：已做（输入 (B,3,512,512)→输出 (B,384,32,32)）。
4. **联系作者**：`iridium.xiaoming@gmail.com`。
