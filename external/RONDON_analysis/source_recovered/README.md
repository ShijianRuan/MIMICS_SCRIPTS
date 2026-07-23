# 解析出的源码（source_recovered/）

本目录存放从 RONDON 中解析出的源码。分两类，**可靠性不同，务必区分**：

## 1. `plaintext/` —— 明文真源码（100% 完整可信）

这些是 RONDON 随每个模型导出的明文 Python 文件，**直接从磁盘拷贝**，未经任何修改，是真正的可运行源码。

| 文件 | 拷贝自 | 内容 |
|---|---|---|
| `decoder_network.py` | `AppData\Roaming\RONDON\model\dinov3-vits16\decoder_network.py` | DINOv3 解码器 head（DoubleConv + Up + Network） |
| `network.py` | `AppData\Roaming\RONDON\model\casgzeefcueyriwz\network.py` | 与 decoder_network.py 内容完全一致（每个模型目录都带一份） |

> 这两个文件内容相同（2399 字节），是唯一以明文存在的训练相关源码。
> 详见 `docs/01_REVERSE_ENGINEERING.md` 第 3.4 节。

## 2. `reversed/` —— 从编译模块逆向还原（函数级，非可运行源码）

应用核心逻辑被 **Nuitka 编译**成原生 .pyd（机器码），源码不在磁盘。本目录通过提取 .pyd 中的元数据（源文件路径、类/方法限定名、docstring、日志消息）**还原出"有哪些文件 + 每个文件的函数/类清单 + 函数语义"**，但**函数体（数值逻辑）不可读**。

| 文件 | 对应原 .py | 还原内容 | 可靠性 |
|---|---|---|---|
| `source_tree.txt` | 整个包 | 36 个 .py 的完整文件树 | 路径 100% 确证 |
| `utils_tiny_unet.py.recovered.txt` | `commands/utils/tiny_unet.py` | 训练循环函数 + 运行时叙事 | 函数名/叙事确证，函数体不可读 |
| `utils_array.py.recovered.txt` | `commands/utils/array.py` | 归一化 + patch 转换 + tensor 加载 | 函数名/docstring 确证，数值不可读 |
| `utils_metric.py.recovered.txt` | `commands/utils/metric.py` | Dice 评估（完整签名） | 签名/docstring 确证，数值不可读 |
| `utils_network_layer.py.recovered.txt` | `commands/utils/network.py` + `layer.py` | 图生成 + 算子 shape 公式 | 算子/公式确证 |
| `models_and_utils.py.recovered.txt` | 数据模型 + archive/remote/feature/task | 类结构 + 命令清单 | 类名/属性确证，方法体不可读 |

### 逆向方法（可复现）

```
目标二进制：C:\Users\shijian.ruan\AppData\Local\RONDON\Lib\site-packages\_tauri_app.cp311-win_amd64.pyd

# 提取所有可打印字符串（7577 个，存于 ../evidence/pyd_strings.txt）
grep -aoE '[ -~]{4,}' _tauri_app.cp311-win_amd64.pyd | sort -u

# 还原源码文件树
grep -aoE '_tauri_app[\\][a-z_\\]+' _tauri_app.cp311-win_amd64.pyd | sort -u

# 还原类/方法结构（u 前缀 = unicode 字符串常量）
grep -oE '^u[A-Z][A-Za-z_]+\.[a-z_][A-Za-z0-9_.]*' pyd_strings.txt | sort -u

# 还原函数语义（docstring，4 空格缩进）
grep -aoE '    [ -~]{25,}' _tauri_app.cp311-win_amd64.pyd | sort -u
```

### 可靠性图例

- **确证**：直接来自明文文件、ONNX 实测、或 .pyd 中的字符串常量。
- **推断**：从符号名/docstring 合理推断，未见代码逻辑。
- **不可读**：Nuitka 机器码立即数（数值常量、if/else 分支），无法从字符串还原。

完整的能力边界见 `docs/01_REVERSE_ENGINEERING.md` 第 10 节。

## 反射加载尝试（失败记录）

曾尝试 `import _tauri_app` 做运行时反射以获取完整函数签名，但该模块依赖完整 Tauri+pydantic+sqlmodel 运行时环境，stub 代价过高（加载到 `commands/seg.py` 第 944 行卡在 pydantic schema 生成）。故退回字符串提取路线。详见 `docs/02_DEEP_FUNCTION_ANALYSIS.md` 开头。
