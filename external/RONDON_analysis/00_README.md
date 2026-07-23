# RONDON 逆向工程产物总索引

本文件夹存放对本地安装的 **RONDON 6.0.0**（医学影像分割软件，内部代号 TimeSlice）的全部逆向分析产物。
所有文件已集中于此，**不会散落到 `C:\Users\shijian.ruan\` 根目录**。

整理日期：2026-07-23

---

## 文件夹结构

```
C:\Users\shijian.ruan\RONDON_analysis\
├── 00_README.md                      ← 本文件（总索引 + 每个文件的来源）
│
├── docs\                             ← 分析文档
│   ├── 01_REVERSE_ENGINEERING.md       架构总览（含 patch 策略 ONNX 实测）
│   └── 02_DEEP_FUNCTION_ANALYSIS.md    函数级深度分析（源码树 + 类结构 + 训练叙事）
│
├── source_recovered\                 ← 解析出的源码
│   ├── README.md                       来源与可靠性说明
│   ├── plaintext\                      明文真源码（直接拷贝，100% 可信）
│   │   ├── decoder_network.py
│   │   └── network.py
│   └── reversed\                       从编译模块逆向还原（函数级，非可运行源码）
│       ├── source_tree.txt
│       ├── utils_tiny_unet.py.recovered.txt
│       ├── utils_array.py.recovered.txt
│       ├── utils_metric.py.recovered.txt
│       ├── utils_network_layer.py.recovered.txt
│       └── models_and_utils.py.recovered.txt
│
├── evidence\                         ← 原始证据
│   ├── pyd_strings.txt                 从 .pyd 提取的 7577 个字符串
│   └── model_json_samples\             实读的 model.json 样本
│       ├── encoder_dinov3-vits16.json
│       ├── decoder_casgzeefcueyriwz.json
│       └── sample_bixpyayanjnniqau.json
│
└── scripts\                          ← 工具脚本
    └── probe_onnx.py                   ONNX 输入输出探测脚本
```

---

## 每个文件的来源

### docs/

| 文件 | 来源方法 | 内容 |
|---|---|---|
| `01_REVERSE_ENGINEERING.md` | 实读明文文件 + ONNX 实测 + .pyd 符号提取 | 整体架构、技术栈、模型设计、patch 策略、归一化、训练流程、加密、复现指南 |
| `02_DEEP_FUNCTION_ANALYSIS.md` | .pyd 系统性字符串提取（7577 个） | 36 个 .py 源码树、数据模型类、网络类、训练叙事、Dice 签名、图生成机制 |

### source_recovered/plaintext/（明文真源码）

| 文件 | 拷贝自 | 说明 |
|---|---|---|
| `decoder_network.py` | `AppData\Roaming\RONDON\model\dinov3-vits16\decoder_network.py` | DINOv3 解码器 head，2399 字节，未修改 |
| `network.py` | `AppData\Roaming\RONDON\model\casgzeefcueyriwz\network.py` | 与上同内容（每个模型目录各带一份），未修改 |

> 这是 RONDON 唯一以明文存在的训练相关源码。其余源码已被 Nuitka 编译。

### source_recovered/reversed/（逆向还原）

| 文件 | 对应原 .py | 还原方法 |
|---|---|---|
| `source_tree.txt` | 整个 `_tauri_app` 包 | `grep -aoE '_tauri_app[\\][a-z_\\]+' .pyd` 提取源文件路径 |
| `utils_tiny_unet.py.recovered.txt` | `commands/utils/tiny_unet.py` | 提取函数名 + 日志/异常消息还原训练叙事 |
| `utils_array.py.recovered.txt` | `commands/utils/array.py` | 提取函数名 + docstring 还原归一化/patch 语义 |
| `utils_metric.py.recovered.txt` | `commands/utils/metric.py` | 提取完整 docstring 还原 Dice 签名 |
| `utils_network_layer.py.recovered.txt` | `commands/utils/network.py` + `layer.py` | 提取算子名 + shape 公式 docstring |
| `models_and_utils.py.recovered.txt` | 数据模型 + archive/remote/feature/task | 提取 `u<Class>.<attr>` 限定名还原类结构 |

> 详见 `source_recovered/README.md`。

### evidence/

| 文件 | 来源 | 说明 |
|---|---|---|
| `pyd_strings.txt` | `grep -aoE '[ -~]{4,}' _tauri_app.cp311-win_amd64.pyd \| sort -u` | 7577 个唯一字符串，是所有逆向结论的原始证据 |
| `model_json_samples/encoder_dinov3-vits16.json` | 拷贝自 `model\dinov3-vits16\model.json` | 编码器元数据（含训练超参来源） |
| `model_json_samples/decoder_casgzeefcueyriwz.json` | 拷贝自 `model\casgzeefcueyriwz\model.json` | 解码器训练配置 + 指标（lr/epochs/loss/dice） |
| `model_json_samples/sample_bixpyayanjnniqau.json` | 拷贝自 `model\casgzeefcueyriwz\bixpyayanjnniqau\model.json` | 训练样本元数据 |

### scripts/

| 文件 | 用途 | 运行方式 |
|---|---|---|
| `probe_onnx.py` | 探测 DINOv3 ONNX 输入输出形状，确证 patch 策略 | `"C:\Users\shijian.ruan\AppData\Local\RONDON\python.exe" probe_onnx.py` |

---

## 关键源文件路径（RONDON 安装位置）

逆向所依据的原始文件都在这两个位置：

1. **配置/模型仓库**（用户数据）：
   `C:\Users\shijian.ruan\AppData\Roaming\RONDON\`
   - `settings.json`、`db\DoNotTouchMe`、`logs\`、`model\`

2. **程序安装目录**（含编译后训练代码）：
   `C:\Users\shijian.ruan\AppData\Local\RONDON\`
   - `rondon.exe`（Tauri 主程序）
   - `python.exe`（嵌入式 Python 3.11）
   - `Lib\site-packages\_tauri_app.cp311-win_amd64.pyd` ← **训练代码真身（Nuitka 编译）**
   - `Lib\site-packages\tinygrad\`、`onnxruntime\` 等

---

## 逆向能力边界（哪些可信、哪些不可读）

### 确证（明文文件 / ONNX 实测 / .pyd 字符串）
- 解码器网络结构（`plaintext/network.py`）
- 训练超参（`evidence/model_json_samples/`）
- DINOv3 ONNX：输入 `(B,3,512,512)` → 输出 `(B,384,32,32)`，patch_size=16（`scripts/probe_onnx.py` 实测）
- patch 策略：ViT 内部 patch + decoder 8/4/2/1 倍 unfold skip
- 36 个 .py 源码文件树
- 5 个归一化函数 + Dice 签名 + 图算子 shape 公式
- 加密格式 TSMODELv1（AES-GCM + PBKDF2）

### 不可读（Nuitka 机器码立即数，无法从字符串还原）
- 归一化的模态分发 if/else 条件
- 数值常量（smooth、eps、百分位 1%/99%、cosine 的 min_lr）
- 函数体内部语句
- vanilla encoder（UNet2D）的具体结构

### 如需 100% 确证（突破机器码限制）
1. **动态拦截**：RONDON 运行时注入 monkey-patch，拦截 tinygrad/onnxruntime 张量打印实际值
2. Nuitka 常量 blob 反编译
3. ONNX 实测（已做）
4. 联系作者：`iridium.xiaoming@gmail.com`（北部战区总医院放射科）

完整说明见 `docs/01_REVERSE_ENGINEERING.md` 第 10 节、`docs/02_DEEP_FUNCTION_ANALYSIS.md` 第 12-13 节。

---

## 阅读建议

1. 先看 `docs/01_REVERSE_ENGINEERING.md` 建立整体认知（架构 + patch 策略 + 复现指南）
2. 再看 `docs/02_DEEP_FUNCTION_ANALYSIS.md` 了解函数级细节
3. 复现训练时参考 `source_recovered/plaintext/network.py`（真源码）+ `reversed/` 下的还原清单
4. 验证 patch 策略用 `scripts/probe_onnx.py`
5. 任何结论都可回溯到 `evidence/pyd_strings.txt` 核验
