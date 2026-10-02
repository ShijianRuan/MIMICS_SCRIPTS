# Phase 2b: GUI-first environment bootstrap & guidance

> 分支 `refactor-overhaul`，2026-09-25。五点产品质量计划 Phase 2 第二项。

## 问题

部署不是总处于零配置状态。环境缺失/安装失败/换机器迁移/FlexiCT 权重未装这四类情形
此前只有两条出路：终端报错（要求标注者开命令行），或者用户自己知道该跑哪个脚本。
配置文件本身默认相对路径、无需手填（P2 迁移专题已解决），真正的摩擦在"异常发生时
的引导"。

## 交付

### 1. 环境引导窗口（新增 `tools/env_guidance.py`）

纯函数 + GUI 两层：

- `collect_issues(project_root)` — 纯读、无 Qt、可无头单测。检测五类问题：
  - `python_missing`（python_env 不在 → 建议 Setup Environment；有离线 bundle 则
    建议 Offline Install）
  - `setup_failed` / `setup_incomplete`（setup_env_state.json 的 error/incomplete
    态，24h 内才报，过期视为陈旧）
  - `migration_pending`（根配置 JSON 里存在本机不存在的绝对 Windows/UNC 路径 →
    推断旧根，建议一键 migrate_root）
  - `flexict_weights`（flexict-finetune 仓库存在但权重未放、且无配置覆盖 →
    信息性提示预期位置）
- `show_dialog()` — PySide6 对话窗，每个问题一个 GroupBox：说明 + 修复按钮；
  修复按钮经同一 setup_environment 编排启动 worker（或 migrate_root），反馈
  以内嵌文本呈现，不弹终端。

### 2. System Health 面板集成

`tools/system_health_panel.py` 快照新增 `environment` 节（后台线程收集，与既有
四节一致），面板新增 "Environment" section：问题行 + "Guidance" 按钮直达引导窗。
摘要行追加 "N environment issue(s)"。

### 3. Mimics 菜单入口

- `runtime_py35/env_guidance_mimics.py` — external_window_launcher 薄壳。
- `scripting_library/99_Admin/10_Environment_Guidance.py` — 菜单项 "Environment
  Guidance"。
- `external_window_launcher` 缺 Python 提示改为指向 Environment Guidance。

### 4. 删除机器特定泄漏

`tools/flexict_common.py` 的 `R:/flexict-finetune/weights` 硬编码回退已删
（仅在唯一一台研究工作站上存在的共享路径）。现在权重缺失时错误信息明确列出两个
可选位置（config 覆盖 / repo weights/），引导窗会接住并解释。

## 测试

- `test_all.py` 新增 `TestEnvGuidance`（8 用例：五类问题检测、陈旧 state 忽略、
  迁移检测的活路径/相对路径豁免、config 覆盖静默）+ health panel 2 用例
  （快照含 environment 节、菜单入口存在）。
- flexict 两套件（24+71）回归绿；fake_mimics_flow_test 13 pass（runtime_py35
  新文件语法在 py3.5 兼容）。

## 明确不做

- setup_env.py 本身不加 pending-issue JSON——`setup_env_state.json` 已经是既有的
  状态文件，引导窗直接消费它，不再引入第二个真源。
- migrate_root.py 不加自动启动——由引导窗按需拉起（保守：只有用户点了按钮才动
  配置文件）。
