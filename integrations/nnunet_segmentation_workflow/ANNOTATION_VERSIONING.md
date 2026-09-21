# 标注版本管理设计文档

> 为 TotalSegmentator → nnUNet 转换流程（Action1_ConvertLabeledToTrainData）引入标注版本管理，
> 使同一病例可挂载多版标注，训练时按版本逐器官回退查找 mask。

---

## 1. 背景与动机

### 1.1 现状问题

Action1_ConvertLabeledToTrainData 把 TotalSegmentator 格式数据转成 nnUNet 格式。转换前，
每个病例的标注只有一个版本：

```
s0001/
├── ct.nii.gz
└── segmentations/            ← 唯一的标注目录（硬编码）
    ├── liver.nii.gz
    ├── kidney_right.nii.gz
    └── ...（每个器官一个二值 mask）
```

原代码在 `Action1_ConvertLabeledToTrainData.py` 中**硬编码**读取 `segmentations/` 目录：

```python
mask_paths = [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map]
```

整个流程没有任何"标注版本"概念：
- `labeled_dataset = ["Totalsegmentator_dataset_v201"]` 里的 `v201` 只是文件夹名前缀，不是版本号
- `dataset.json` 里 `release: "2.0"` 是静态字面量，与输入无关
- 同一病例无法挂载多版标注，无法选择"用哪一版"训练

### 1.2 需求

- 同一病例可挂载多版标注（大版本全量修订 + 小版本增量修订个别器官）
- 训练时指定一个目标版本，逐器官在该版本的回退链中找最新可用文件
- 小版本只放改动的器官，缺失的器官自动从低版本补齐
- 向后兼容：不指定版本时行为与历史完全一致

### 1.3 设计目标

- **最小侵入**：复用现有转换/合并函数（`combine_labels` 等），只在"mask 路径解析"环节插入版本逻辑
- **可追溯**：每个 case 各器官实际取自哪版，写入 manifest CSV + 控制台告警
- **容错**：目标版本目录缺失/某器官全链缺失时告警不中断，回退到可用版本

---

## 2. 版本体系

### 2.1 版本号语义

**版本号数字本身区分全量/增量**（用户决策）：

| 类型 | 格式 | 含义 | 目录名 |
|---|---|---|---|
| 大版本 | `v1` / `v2` / `v3` …（整数） | 全量重标，权威基准 | `segmentations/`(=v1)、`segmentations_v2/`、`segmentations_v3/` … |
| 小版本 | `v2.1` / `v2.3` …（点号分层） | 某大版本的增量补丁，只改个别器官 | `segmentations_v2.1/`、`segmentations_v2.3/` … |

**命名约定**：
- `v1` ≡ 基础版 ≡ `segmentations/`（无后缀，不重命名，兼容现有数据）
- `vN`（N≥2）≡ `segmentations_vN/`
- `vN.M` ≡ `segmentations_vN.M/`

### 2.2 磁盘目录结构示例

```
s0001/
├── ct.nii.gz
├── segmentations/            ← v1 基础版（全量）
│   ├── liver.nii.gz
│   └── kidney_right.nii.gz
├── segmentations_v2/         ← v2 大版本（全量重标）
│   └── liver.nii.gz
├── segmentations_v2.1/       ← v2.1 小版本（增量：仅 spleen）
│   └── spleen.nii.gz
└── segmentations_v2.3/       ← v2.3 小版本（增量：liver 修订）
    └── liver.nii.gz
```

---

## 3. 回退链算法（核心）

### 3.1 规则

| 指定版本 | 回退链 | 说明 |
|---|---|---|
| 空 / `v1` | `segmentations/` | 基础版，与历史一致 |
| `v3`（大版本） | `segmentations_v3/` → `segmentations_v2/` → `segmentations/` | 大版本主链向低回退 |
| `v2.1`（小版本） | `segmentations_v2.1/` → `segmentations_v2/` → `segmentations/` | 先回所属大版本，再大版本链 |
| `v2.3`（小版本） | `segmentations_v2.3/` → `segmentations_v2.1/` → `segmentations_v2/` → `segmentations/` | **同大版本小版本按序叠加** |
| `v1.2`（小版本） | `segmentations_v1.2/` → `segmentations/` | v1 是最低大版本 |
| `v3.1`（小版本） | `segmentations_v3.1/` → `segmentations_v3/` → `segmentations_v2/` → `segmentations/` | 跨大版本不取小版本 |

### 3.2 关键规则详解

**① 小版本按序叠加（解决问题 5 的一致性风险）**

同大版本下，minor 递增是**时间叠加序**——minor 大的修订建立在 minor 小的基础上。指定 `vN.M` 时，
扫描磁盘上同大版本下 minor ≤ M 的所有小版本目录，按 minor **降序**插入链：

```
指定 v2.3 → 扫描到 v2.1, v2.3 → 降序 → [v2.3, v2.1, v2, v1]
```

这样 v2.3 改的 liver + v2.1 改的 spleen 都保留。若两个小版本都改了同一器官，链中 minor 大的
更靠前，`_resolve_organ_path` 先命中它 → **最新修订优先**。

> ⚠️ 前提约定：同大版本下小版本必须按时间顺序编号且语义上是"叠加修订"，minor 数值需严格递增。
> 不要用 v2.1 和 v2.3 表示两个平行独立的修订分支——那会造成语义混乱。

**② 大版本不叠加小版本**

指定大版本 `vN`（无 minor）时，链**不含任何小版本**：`[vN, v(N-1), ..., v1]`。
大版本是全量基准节点，语义上"回到这个全量快照"，不掺入小版本的增量。

**③ 跨大版本不取小版本**

`v3.x` 的链含 v3 的小版本 + v3 大版本 + v2 大版本 + v1，**不含 v2 的小版本**。
因为 v2.1 是 v2 的修订，不是 v3 的祖先。小版本叠加只发生在"指定版本所属大版本"内部。

**④ 不向更高的其他大版本回退**

指定 `v2.1` 时链不含 v3，即使 v3 物理存在。v2.1 是基于 v2 的修订，逻辑上不应叠加 v3 的内容。

**⑤ 只扫描实际存在的目录**

回退链中只包含磁盘上**实际存在**的小版本目录。若指定 `v2.5` 但磁盘上只有 `v2.1`，链是
`[v2.1, v2, v1]`（v2.5 不进链）——这正是"目标版本目录缺失，告警回退"的容错行为。

### 3.3 回退链总公式

```
指定 vN.M（小版本）:
  [所有 segmentations_vN.M' (M' ≤ M 且实际存在, 按 M' 降序)]
  + [segmentations_vN]
  + [segmentations_v(N-1), ..., segmentations_v2]
  + [segmentations]                          # v1

指定 vN（大版本）:
  [segmentations_vN]
  + [segmentations_v(N-1), ..., segmentations_v2]
  + [segmentations]                          # v1
  # 不含任何小版本
```

---

## 4. 逐器官解析

### 4.1 查找逻辑

`_resolve_organ_path` 对每个器官（roi），在回退链中**逐目录**查找 `{roi}.nii.gz`，返回第一个
存在的路径：

```python
def _resolve_organ_path(subject_path, roi, seg_chain, record=None):
    for seg_dir in seg_chain:
        candidate = seg_dir / "{}.nii.gz".format(roi)
        if candidate.exists():
            if record is not None:
                record.append((roi, seg_dir.name))   # 记录来源版本
            return candidate
    # 全链缺失: 回退基础版路径, 由下游 combine_labels 走 'Missing' 打印
    if record is not None:
        record.append((roi, None))
    return subject_path / "segmentations" / "{}.nii.gz".format(roi)
```

- **优先级**：链中靠前（版本高）的目录优先，符合"最新修订优先"
- **全链缺失**：返回基础版路径（不存在），下游 `combine_labels` 打印 `Missing: ...`，该器官标为背景
- **record 回调**：调用方传入列表，收集 `(roi, source_version)` 用于告警与 manifest

### 4.2 端到端示例

针对第 2.2 节的目录结构，指定 `annotation_version="v2.3"`：

| 器官 | 回退链查找过程 | 实际取自 |
|---|---|---|
| liver | v2.3/liver.nii.gz 存在 → 命中 | `segmentations_v2.3/` |
| spleen | v2.3/无 → v2.1/spleen.nii.gz 存在 → 命中 | `segmentations_v2.1/` |
| kidney_right | v2.3/无 → v2.1/无 → v2/无 → segmentations/存在 → 命中 | `segmentations/`(v1) |

**这正是问题 5 的修复**：旧规则下 spleen 会回退到 v2 大版本（丢失 v2.1 的修订），新规则叠加
回退使 v2.1 的 spleen 保留。

---

## 5. 多数据集版本对齐（问题 4 修复）

### 5.1 根因

`AutoSegmentationFramework.ReadConfigFile()` 构造两个**独立**列表：

| 列表 | 来源 | 语义 | 长度 |
|---|---|---|---|
| `dataset_path` | `labeled_dataset` | 源数据集（可多个合并） | = labeled_dataset 数量 |
| `nnUNet_path` | `train_dataset` | 输出模型（独立） | = train_dataset 数量 |

校验只要求 `len(segment_list_name) == len(train_dataset)`，**不要求** `labeled_dataset` 与
`train_dataset` 等长。历史 `convertdata()` 把整个 `datasets` 列表传给每个 index 的 `convert()`，
隐含语义是"所有 labeled_dataset 合并进每个 train_dataset"。

因此 `annotation_version` 是**源数据集（labeled_dataset）的属性**，必须按 `labeled_dataset`
数量广播。初版实现误按 `train_dataset` 数量广播并用 `ann_versions[index]`（index 是 train_dataset
索引）取值，导致多数据集时索引错位。

### 5.2 修复方案

**① 框架层**（`AutoSegmentationFramework.convertdata`）：

```python
ann_versions = _normalize_annotation_version(
    config["PATHS"].get("annotation_version"), len(datasets)   # datasets = labeled_dataset 列表
)
for index in _indices(config, dataset_index):                  # index 遍历 train_dataset
    action.convert(
        datasets,
        ...,
        annotation_version=ann_versions,    # 传整个列表, 不再用 ann_versions[index]
    )
```

**② 转换层**（`convert()` 内）按 subject 的源数据集定位版本：

`generate_train_test_dataset` 合并所有源数据集的 case，但 case 路径保留了源标识——
`Path(subject).parent.name` 即源数据集目录名。新增 `_resolve_subject_version`：

```python
def _resolve_subject_version(subject, dataset_path, ann_versions):
    if isinstance(ann_versions, (list, tuple)):
        parent_name = Path(subject).parent.name
        for idx, dpath in enumerate(dataset_path):
            if Path(dpath).name == parent_name:
                return ann_versions[idx] if idx < len(ann_versions) else None
        return None      # 找不到源, 退化为只读 segmentations/
    return ann_versions   # 单值, 所有源共用（向后兼容）
```

主循环内每个 subject 先定位版本再建回退链：

```python
for subject in tqdm(...):
    subject_path = Path(subject)
    subj_ver = _resolve_subject_version(subject, dataset_path, annotation_version)
    seg_chain = _version_dirs(subject_path, subj_ver)
    ...
```

### 5.3 对齐约定

- `annotation_version` 列表按 **`labeled_dataset`**（源数据集）顺序对齐，**不是** `train_dataset`
- 单值 → 所有源数据集共用（最常见场景）
- 多数据集合并时，每个源数据集的 case 用各自版本

---

## 6. 告警与追溯（问题 1/3 修复）

### 6.1 告警（不中断）

`convert()` 主循环为每个 case 收集器官来源后，调用 `_emit_version_warnings`，三类情况分别告警：

1. **目标版本目录缺失**：该 case 没有任何器官取自目标版本目录（目标目录不存在）
   ```
   [告警] 以下 case 的目标版本目录不存在，已回退到更低版本（未中断）：
     - Totalsegmentator_v201_s0001(train): 目标版本 v2.3 目录 segmentations_v2.3/ 缺失
   ```
2. **器官回退到非目标版本**：列出实际来源版本
   ```
   [告警] 以下器官回退到了非目标版本（最新修订优先，未中断）：
     - Totalsegmentator_v201_s0001(train): kidney_right 取自 segmentations (目标 v2.3)
   ```
3. **器官全链缺失**：所有版本目录都没该器官，标为背景
   ```
   [告警] 以下器官在所有版本目录均缺失，该器官标为背景（未中断）：
     - Totalsegmentator_v201_s0001(train): femur_left
   ```

告警汇总按 case 打印，避免逐器官刷屏。**不中断转换**，保证可用数据继续处理。

### 6.2 Manifest 追溯

`_write_annotation_manifest` 写 `<nnunet_path>/annotation_manifest.csv`：

| 列 | 含义 |
|---|---|
| `case` | case 名（`{源数据集名}_{subject}`） |
| `split` | `train` / `test` |
| `roi` | 器官名 |
| `source_version` | 实际取自的版本目录名（如 `segmentations_v2.1`），空表示全链缺失 |
| `target_version` | 该 case 指定的目标版本（如 `v2.3`） |

**防覆盖设计**（问题 3）：每个 `train_dataset` 的 `nnunet_path` 不同 → manifest 天然分文件，
无跨数据集覆盖。`convert()` 单进程顺序执行，无并发写入。

---

## 7. 数据流总览

```
Config_*.toml
  [PATHS] labeled_dataset = [...]          ──┐
           annotation_version = "v2.3" 或 [...]│
  [MODEL] train_dataset = [...]              │
           segment_list_name = [...] → ModelMap.toml → class_map
                                             │
AutoSegmentationFramework.ReadConfigFile()   │
  dataset_path   = labeled_dataset 展开       │
  nnUNet_path    = train_dataset 展开         │
  ann_versions   = 按 labeled_dataset 广播 ←─┘
                                             │
convertdata()  ── 对每个 train_dataset(index) 调用 convert(datasets, ..., ann_versions)
                                             │
convert(dataset_path, nnunet_path, class_map, ..., annotation_version=ann_versions)
  ├─ generate_train_test_dataset(dataset_path)   # 合并所有源数据集, 80/10/10 划分
  │
  ├─ for subject in train+val:
  │    ├─ subj_ver = _resolve_subject_version(subject, dataset_path, ann_versions)
  │    ├─ seg_chain = _version_dirs(subject_path, subj_ver)
  │    ├─ organ_record = []
  │    ├─ mask_paths = [_resolve_organ_path(subject_path, roi, seg_chain, record=organ_record)
  │    │                 for roi in class_map]
  │    ├─ _record_manifest(manifest_rows, case_name, "train", organ_record, subj_ver)
  │    └─ combine_labels / resample_and_combine_labels(...)   # 合并多器官 mask → 单 label volume
  │
  ├─ for subject in test:  (同上, split="test")
  │
  ├─ _emit_version_warnings(manifest_rows)              # 告警
  ├─ _write_annotation_manifest(nnunet_path, manifest_rows)   # annotation_manifest.csv
  └─ generate_json_from_dir_v2(..., annotation_version)      # dataset.json + splits_final.json
                                             │
                                             ▼
nnUNet_raw/<train_dataset>/
  ├── imagesTr/<case>_0000.nii.gz
  ├── labelsTr/<case>.nii.gz              ← 多标签合并 volume（按版本回退生成）
  ├── imagesTs/, labelsTs/
  ├── dataset.json                        ← 含 annotation_version 字段（追溯）
  └── annotation_manifest.csv             ← 每个 case 各器官来源版本
nnUNet_preprocessed/<train_dataset>/splits_final.json
```

---

## 8. 配置说明

### 8.1 Config_*.toml 新增字段

在 `[PATHS]` 段、`labeled_dataset` 下方：

```toml
[PATHS]
labeled_path = "/data1/segmentationForTrain/labeled"
labeled_dataset = ["Totalsegmentator_dataset_v201"]

# 标注版本：指定后逐器官沿回退链查找 mask（仅影响取哪个目录的器官文件，不改 class_map）。
#   大版本 v2/v3   = 全量重标, 目录 segmentations_v2/ 等; v1 ≡ segmentations/(基础版)
#   小版本 v2.1    = v2 的增量补丁(只改个别器官), 目录 segmentations_v2.1/
#   回退规则:
#     大版本 vN   → [vN, v(N-1), ..., v1]            （大版本是全量基准, 不叠加小版本）
#     小版本 vN.M → [vN.M'(M'<=M且存在,降序), vN, ..., v1]  （同大版本小版本按序叠加, 跨大版本不取小版本）
#   例 "v2.3" → segmentations_v2.3/ → segmentations_v2.1/ → segmentations_v2/ → segmentations/(v1)
#       （v2.3 改的 liver + v2.1 改的 spleen 都保留; 同器官冲突取 minor 大的最新修订）
#   留空/"v1" → 只读 segmentations/（默认，与历史行为一致）
#   目标版本目录缺失或某器官回退时打印告警(不中断)，并写 annotation_manifest.csv 追溯每个器官来源。
# 多数据集时可为列表，按 labeled_dataset(非 train_dataset)顺序对齐：annotation_version = ["v2.1", "v3"]
annotation_version = ""
```

### 8.2 典型用法

| 场景 | 配置 | 含义 |
|---|---|---|
| 不启用版本管理 | `annotation_version = ""` | 只读 `segmentations/`，与历史一致 |
| 切换到大版本 v2 | `annotation_version = "v2"` | 全量用 v2，缺器官回退 v1 |
| 用 v2.3 增量修订 | `annotation_version = "v2.3"` | v2.3 + v2.1（叠加）+ v2 + v1 |
| 多数据集各用各版 | `annotation_version = ["v2.1", "v3"]` | 按 labeled_dataset 顺序对齐 |

### 8.3 class_map 不随版本变化（问题 2）

不同版本器官集合变化（如 v2 新增 spleen）**不处理**：class_map 仍是全局唯一（来自
`ModelMap.toml` 的 `segment_list_name`）。某版缺某器官时该器官走 Missing（带告警，标为背景）。
若需按版本配不同 class_map，需扩展 `ModelMap.toml` 按版本分节，当前不支持。

---

## 9. 函数参考

均在 `Action1_ConvertLabeledToTrainData.py`：

| 函数 | 行号 | 职责 |
|---|---|---|
| `_scan_minor_versions(subject_path, major)` | 357 | 扫描磁盘上 `segmentations_v{major}.{minor}` 目录，返回 minor 整数集 |
| `_version_dirs(subject_path, annotation_version)` | 374 | 构建逐器官回退链（大小版本叠加算法） |
| `_resolve_subject_version(subject, dataset_path, ann_versions)` | 428 | 多数据集时按 subject 源数据集定位版本 |
| `_resolve_organ_path(subject_path, roi, seg_chain, record=None)` | 446 | 在回退链中取第一个存在的器官文件，record 记录来源 |
| `_record_manifest(manifest_rows, case_name, split, organ_record, target_version)` | 468 | 收集 case 各器官来源到 manifest_rows |
| `_emit_version_warnings(manifest_rows)` | 480 | 三类告警（缺失/回退/全链缺），不中断 |
| `_target_version_dir_name(target_version)` | 526 | 版本字符串 → 目录名（告警比对用） |
| `_write_annotation_manifest(nnunet_path, manifest_rows)` | 534 | 写 annotation_manifest.csv |
| `convert(..., annotation_version=None)` | 567 | 主转换入口，串联上述逻辑 |

`AutoSegmentationFramework.py`：

| 函数 | 行号 | 职责 |
|---|---|---|
| `_normalize_annotation_version(value, count)` | 287 | 把 annotation_version 标量/列表广播为与 labeled_dataset 等长的列表 |
| `convertdata(config, dataset_index=None)` | 223 | 读取并广播 annotation_version，传列表给 convert() |

---

## 10. 已修复的 5 个缺陷

| # | 缺陷 | 性质 | 修复 |
|---|---|---|---|
| 1 | 目标版本目录缺失静默回退 | 风险 | 告警不中断：`_emit_version_warnings` 打印缺失/回退/全链缺三类告警 |
| 2 | 无法处理器官集合变化 | 限制 | 不处理，class_map 全局唯一，缺器官走 Missing（带告警） |
| 3 | manifest 多进程写入覆盖 | 预防 | 按 train_dataset 分目录分文件，单进程顺序写 |
| 4 | convert() 多数据集索引错位 | **真实 bug** | annotation_version 按 labeled_dataset 对齐，`_resolve_subject_version` 按 subject 源数据集定位 |
| 5 | 混合版本标注一致性风险 | 语义 | 小版本按序叠加回退：v2.3 → v2.1 → v2 → v1，minor 降序叠加 |

---

## 11. 向后兼容性

- `annotation_version` 缺省 / 空字符串 → `_version_dirs` 返回 `["segmentations"]` → 等价历史行为
- `convert()` 的 `annotation_version=None` 默认值 → 旧调用方（不传该参数）行为不变
- `annotation_version` 接受单值（旧）或列表（新多数据集），单数据集场景两者等价
- 现有 `segmentations/` 目录无需重命名/迁移
- `labelsTs`（测试集 GT）同样按版本回退生成，训练与评估自动用同一版 GT，一致
- `evaluation()`（框架 line 328）无需改动——读的是 convert 已固化的 `labelsTs`
- class_map 不变

---

## 12. 验证

### 12.1 单元测试

```bash
python -m unittest test_annotation_version -v
```

`test_annotation_version.py` 共 21 个用例，覆盖：

- **回退链算法**（`TestVersionDirs`，11 个）：空/v1、大版本链、大版本不含小版本、小版本叠加链、
  minor≤M 过滤、同器官冲突取最新、不向更高大版本回退、跨大版本不取小版本、v1.x、非法版本报错
- **逐器官解析**（`TestResolveOrganPath`，6 个）：基础版、大版本回退、**v2.3 叠加 spleen 来自 v2.1**
  （问题 5 核心）、v2.1 不取 v2.3、record 回调、全链缺失返回基础版
- **多数据集版本定位**（`TestResolveSubjectVersion`，4 个）：单值广播、列表按 parent.name 定位、
  未知源返回 None、None 值（问题 4）

### 12.2 端到端验证

1. **问题 5（叠加）**：构造 case，`segmentations/`(liver+kidney)、`segmentations_v2/`(全量)、
   `segmentations_v2.1/`(spleen)、`segmentations_v2.3/`(liver 修订)。设
   `annotation_version="v2.3"` 跑 convert，检查 `labelsTr`：spleen 来自 v2.1、liver 来自 v2.3、
   kidney 来自 v1。
2. **问题 1（告警）**：删除 `segmentations_v2.3/`，重跑，确认打印"目标版本目录缺失"告警且不中断、
   输出回退到 v2.1+v2+v1。
3. **问题 3（manifest）**：检查 `<nnunet_path>/annotation_manifest.csv` 生成、内容含每个 case 各器官
   的 source_version；多 train_dataset 时各目录独立 manifest 无覆盖。
4. **问题 4（多数据集）**：Config 设两个 `labeled_dataset`、`annotation_version=["v2.1","v3"]`，
   确认每个源数据集的 case 用各自版本（检查 manifest 的 source_version 列）。
5. **兼容回归**：不设 `annotation_version`，输出与改造前 `np.array_equal` 一致。

### 12.3 运行命令

```bash
# 单独跑转换阶段
python AutoSegmentationFramework.py convert --config Config_Template.toml

# 完整流程（转换→预处理→训练→预测）
python AutoSegmentationFramework.py run --config Config_Template.toml
```

---

## 13. 设计决策记录

> 本节记录关键设计岔路的决策与理由，供后续维护时理解"为什么这么设计"。

### 13.1 为什么用"平行子目录"而非"逐器官后缀"或"独立版本根目录"

- **平行子目录**（选定）：`segmentations/` + `segmentations_v2/` 并排。最小侵入，只需把硬编码
  目录名改成可配置，语义清晰。
- 逐器官后缀（`liver.nii.gz` / `liver.v2.nii.gz`）：粒度最细但 Action1 和 ModelMap 都要改，复杂。
- 独立版本根目录：几乎不改代码但无版本语义，且同名 case 合并会冲突。

### 13.2 为什么小版本要叠加而非只回大版本

旧规则"小版本只回所属大版本"会导致**小版本修订累积丢失**：v2.1 改 spleen、v2.3 改 liver，
指定 v2.3 时 spleen 回退到 v2 大版本（而非 v2.1），v2.1 的 spleen 修改丢失（问题 5）。
叠加回退使同大版本的小版本修订都能保留，符合"小版本是增量补丁、按时间叠加"的直觉。

### 13.3 为什么大版本不叠加小版本

大版本是全量基准快照。指定大版本 v2 表示"回到 v2 这个全量状态"，不应掺入 v2.1/v2.3 的增量
（那些增量是 v2 之后的修订）。若用户想用增量，应指定具体小版本。

### 13.4 为什么跨大版本不取小版本

v2.1 是基于 v2 的修订，不是 v3 的祖先。v3.x 的链若含 v2.* 会造成语义混乱（v3 的状态叠加 v2
的修订）。小版本叠加严格限制在"指定版本所属大版本"内部。

### 13.5 为什么 annotation_version 按 labeled_dataset 对齐而非 train_dataset

`annotation_version` 描述的是"源数据用哪版标注"，是源数据集（labeled_dataset）的属性。
`labeled_dataset`（源，可多个合并）与 `train_dataset`（输出模型）是两个独立维度，长度可不等。
按 train_dataset 对齐会在多数据集时索引错位（问题 4）。

### 13.6 为什么器官集合变化不处理

按用户决策：保持 class_map 全局唯一。处理器官增减需为每版配独立 class_map，会使 ModelMap 和
config 结构显著复杂化，当前收益不足。缺器官走 Missing + 告警，足够安全。

### 13.7 为什么告警不中断

转换常在批量数据上跑，单个 case 版本目录缺失不应让整批失败。告警 + manifest 让用户事后能
发现并修正，同时保证可用数据继续处理。
