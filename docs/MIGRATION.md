# 迁移指南（MIGRATION.md）

> 适用版本：refactor-overhaul 分支 2026-09 之后（配置默认相对路径）
> 目标：把整套 Mimics-Script 部署（代码 + 环境 + 模型 + 配置）从一台机器/一个根目录迁到另一台，迁移后核心功能可用，无需改代码。

## 一、可迁移性总览（兼容性矩阵）

| 资产 | 位置 | 可迁移性 | 说明 |
| --- | --- | --- | --- |
| 代码 + 配置 | 部署树根 | ✅ 整树拷贝 | 所有配置默认相对路径（基于项目根解析） |
| Python 环境 | `python_env/` | ✅ 整目录拷贝 | Python 3.13 embeddable 发行版（非 venv），`python313._pth` 全相对路径，无 home= 锚定 |
| nnInteractive 官方模型 | `python_env/models/nnInteractive_v1.0/` | ✅ 随 python_env 拷贝 | `official_model_dir` 配置指向此处 |
| nnInteractive 微调任务模型 | `nninteractive_task_models/` | ✅ 拷贝 `tasks/` + `registry.json` | registry v2 用 workspace 相对路径 `model_relpath`，拷贝即生效 |
| `cache/`、`jobs/`（微调） | 同上目录 | ⛔ 不迁移 | 可再生（prepared 缓存）/历史日志，体积大 |
| nnU-Net 模型 | workspace `models/` + 全局注册表 | ✅ 拷贝 + manifest 扫描 | 注册表死路径行标 `missing_path`（不再静默丢弃），manifest 扫描自动重建 |
| FlexiCT few-shot 模型 | workspace `flexict_models/`（registry + Dataset 训练产物） | ✅ 拷贝整个 `flexict_models/` | registry 用 workspace 相对路径；成对模型共享 `pair_id`，拷贝即识别 |
| FlexiCT 预训练 backbone | `integrations/flexict-finetune/weights/`（2×~576MB） | ⚠️ 不在 pack 包内 | gitignore 且 package_portable 排除；需手工拷贝或从分发源重新获取 |
| 数据集（ts_root / mcs） | 任意位置 | ✅ 拷贝 | `dataset_manifest` path_reference 双记录 |
| `.mimics_runtime/` 状态 | 部署树内 | ⛔ 不迁移 | 全部机器本地（锁/进程记录/队列标记），`migrate_root --reset-runtime` 清除 |
| `Scripts/*.exe` 入口脚本 | `python_env/Scripts/` | ⚠️ 迁移后失效 | pip.exe 等 wrapper 内嵌绝对路径——项目代码只调 `python.exe`，不依赖它们 |

## 二、标准迁移步骤

### A 机（导出）

```
# 1. 打包（代码 + 配置 + 外部权重 + 任务模型，不含 python_env）
python_env/python.exe tools/package_portable.py pack

# 或带环境（仅当两机 OS/CUDA/Python 完全一致）
python_env/python.exe tools/package_portable.py pack --with-env

# 2. 离线安装包（目标机无网络时）
python_env/python.exe tools/package_portable.py offline-bundle
```

### B 机（导入）

```
# 1. 解压到任意目录，如 D:\MIMICS_SCRIPTS
# 2. 有网：跑 Mimics 菜单 Admin > Setup Environment（或 tools/setup_env.py）
#    无网：双击 setup_offline.bat
# 3. 迁移机器相关路径（若旧机根目录与不同）：
python_env/python.exe tools/migrate_root.py --old-root E:\旧机安装根
# 4. 若拷贝过 .mimics_runtime（不推荐），加 --reset-runtime 清状态
```

### 手工最小迁移（不用 package_portable）

1. 整树拷贝（含 `python_env/`、`nninteractive_task_models/`）
2. 跑 `migrate_root.py --old-root <旧根>`
3. 打开 Mimics，菜单进入 Admin > System Health 确认无 stale 状态

## 三、migrate_root.py 做什么 / 不做什么

**做**：
- 根配置 JSON 中指向旧根的绝对路径 → 改为相对路径（或 `--absolute` 改新绝对路径）
- 重新生成 `setup_offline.bat`（当其仍指向 legacy `nninteractive_env`）
- `--reset-runtime` 清 `.mimics_runtime` 机器本地状态
- `--dry-run` 只打印计划不写盘

**不做**（输出报告里会提示）：
- 搬运文件本身（权重/数据目录用拷贝或 package_portable）
- 修复 nnU-Net 全局注册表死路径行（它们被标记 `missing_path` 并在状态查看器可见，manifest 扫描自动重建可用行；也可手工删 `~/.mimics_script/nnunet_model_registry.json` 让其重建）

## 四、迁移后验证清单

1. `Admin > System Health`：进程/锁/队列干净，无 stale
2. `02_AI > nnInteractive > Manage Custom Models`：任务模型列表正常（registry v2 relpath）
2a. `02_AI > FlexiCT > 01 Train Model`：模型表能列出（flexict_models/registry.json 正常加载）
3. 任一病例：导入 → 提示 → 导出 全流程走一遍
4. （可选）`python_env/python.exe tests/test_model_portability.py`

## 五、版本不一致场景

| 场景 | 行为 |
| --- | --- |
| 旧配置仍含绝对路径 | 解析链照常工作（绝对路径优先），migrate_root 可修 |
| 新配置遇到旧代码（回滚） | 相对路径由旧代码的 ROOT 解析同样支持（nninteractive_task_common.py:147 早已实现） |
| registry v1（旧 workspace） | nninteractive 注册表 v1→v2 在保存时升级，读取时走 legacy `model_dir` → `model_relpath` 回退链 |
| Python 环境版本差异 | `Scripts/*.exe` wrapper 失效但项目不依赖；库版本由 setup_env 校验（torch 2.6.0+cu124、nnunetv2 2.8.x） |
| `nninteractive_env` 旧名 | 全库双候选名回退（`python_env` 优先），migrate_root 重生成 bat |

## 六、跨机使用模型（常见问题）

- **A 机训练的模型 B 机能用吗？** 能。nnInteractive 微调模型 registry 用相对路径；nnU-Net 模型 manifest 扫描。两者都随目录拷贝即识别。
- **A 机导入的 mcs B 机能导出吗？** 能。mcs 自包含；导出配置不写死 mcs 路径（每次任务重新发现）。
- **换机要手工改路径吗？** 配置已默认相对路径。仅当数据集根（ts_root）也换了位置时，重新在 UI 里选一次数据集根即可。
