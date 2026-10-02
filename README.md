# MIMICS_SCRIPTS — Mimics AI 标注扩展

为医学影像标注软件 **Mimics**（Materialise）提供外挂式 AI 标注能力：

- **数据导入导出**：拖拽/浏览/粘贴导入任意数据并转 `.mcs`、掩码注入与
  导出、任务状态与逐行停止
- **AI 分割**：nnInteractive（官方模型 + 自训模型）、nnU-Net 训练与推理、
  ScribblePrompt 交互分割、FlexiCT 少样本训练与主动学习
- **辅助工具**：窗宽窗位预设、系统健康检查、诊断包收集

Mimics 内嵌的 Python 3.5 只做轻量调度（`runtime_py35/`）；所有重活
（GPU 推理、训练、文件转换）由外部 Python 3.13 环境（`python_env/`）在
独立进程完成，**绝不阻塞 Mimics 主界面**。

## 快速上手（三步）

1. **安装环境**（每台工作站一次）
   - 有分发安装包（含 `python\` 与 `wheels\`）：双击 `setup_offline.bat`
   - 或在 Mimics 中：`Scripting → Add Scripting Library` 选择
     `scripting_library\` 文件夹，然后运行菜单
     `99_Admin → 01_Setup_Repair_Environment`
   - 之前装过但报环境问题？运行 `99_Admin → 09_Environment_Guidance`，
     它会逐项列出问题并给出修复入口。

2. **挂载脚本库**：Mimics 中 `Scripting → Add Scripting Library`，
   选择本目录的 `scripting_library\` 文件夹。菜单会多出 01_Data /
   02_AI / 03_Review / 99_Admin 四组入口。

3. **开始标注**：
   - 打开项目、激活图像 → `02_AI → nnInteractive → 01_Annotate_Official_Model`
   - 用点/框/涂鸦提示，几秒出分割草稿，确认后应用为 Mask
   - 每个入口的用途与前置条件见 [docs/mimics_entry_guide.md](docs/mimics_entry_guide.md)

## 常见问题

| 症状 | 去哪 |
|------|------|
| 环境缺失/安装失败/目录迁移/权重缺失 | `99_Admin → 09_Environment_Guidance` |
| 端到端功能验证清单（真实 Mimics 机器） | [docs/mimics_real_data_validation.md](docs/mimics_real_data_validation.md) |
| 用户可改配置项说明 | [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) |
| 某个入口的详细工作流 | [docs/scripting_library_workflows.md](docs/scripting_library_workflows.md) |
| 后台任务的反馈/停止/回退规范 | [docs/task_lifecycle_and_safety_policy_CN.md](docs/task_lifecycle_and_safety_policy_CN.md) |
| 各 AI 框架的集成细节 | `docs/{nninteractive,nnunet,flexict,scribbleprompt}_mimics.md` |

## 仓库结构

```
scripting_library/   Mimics 菜单入口（01_Data / 02_AI / 03_Review / 99_Admin）
runtime_py35/        Mimics 内嵌 Python 3.5 侧的调度与状态监控
tools/               外部 Python 侧的桥接、UI 与流水线
tests/               测试套件（test_all.py 等全部测试文件）
integrations/        独立集成仓（flexict-finetune、ScribblePrompt 等）
docs/                使用文档、验收清单与迭代账本（03_Review/）
python_env/          外部 Python 3.13 环境（Setup 生成，不入库）
*.json               根目录配置（见 CONFIG_REFERENCE.md）
```

## 开发与回归

```powershell
# 冒烟（~1 min）
python_env\python.exe tools\run_regression_matrix.py --profile smoke
# 快速（~10 min，每轮迭代结束必须通过）
python_env\python.exe tools\run_regression_matrix.py --profile fast
# 全量（~30 min）
python_env\python.exe tools\run_regression_matrix.py --profile full
```

测试策略与门禁定义见 [docs/TEST_STRATEGY.md](docs/TEST_STRATEGY.md)。
