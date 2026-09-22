# Mimics-Script 功能入口速查表

## 数据导入导出

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| Import Dataset | 批量转换数据集为 .mcs | 数据集文件夹（含案例子目录 + 医学图像） |
| Import Single Case | 单个案例 → .mcs | 一个图像文件 / 案例文件夹 / DICOM 目录 |
| Import Masks | 外部掩码文件注入当前项目 | **项目已打开** + 已激活图像 + 掩码文件（.nii.gz/.mha/.nrrd） |
| Export Masks | 当前项目掩码导出为 .nii.gz | **项目已保存为 .mcs** + 已激活图像 + 有掩码 |
| Stop Import Queue | 停止正在运行的批量导入 | 有批量导入正在运行 |
| Stop Mask Export | 停止当前 Mimics-Script 掩码导出 | 有掩码导出正在运行 |

## AI 分割

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| nnInteractive Segmentation | 以选中掩码为提示做 AI 分割 | **项目已打开** + 已激活图像 + 已选中掩码 + 环境已设置 |
| FlexiCT Train Model | 少样本（8 例起）训练单器官分割模型 | 已导出标注的数据集 + 环境已设置 + 预训练权重已就位 |
| FlexiCT Predict Current Case | 用 FlexiCT 模型预测当前病例并应用为 Mask | **项目已打开** + 已激活图像 + 已有训练好的模型 + verified grid |
| FlexiCT Active Learning Review | 双模型分歧排序未标注池并叠加不确定度带 | 已有 2D+3D 模型对（pair）+ 未标注数据池 |
| nnU-Net Train Model | 训练多类别分割模型 | 数据集（含已导出的掩码）+ 环境已设置 + 推荐 GPU |
| nnU-Net Predict Current Case | 用训练好的模型推理当前案例 | **项目已打开** + 已激活图像 + 已有训练好的模型 |
| nnU-Net Show Status & Models | 查看任务进度/模型列表 | 有启动过的 nnU-Net 任务 |
| nnU-Net Stop Running Task | 停止当前训练或推理 | 有正在运行的 nnU-Net 任务 |

## 审阅工具

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| Identify Mask At Cursor | 点击视图查看该位置的掩码名 | **项目已打开** + 已激活图像 + 有掩码 |
| Window From Selected Mask | 根据掩码名自动匹配窗宽窗位 | **项目已打开** + 已激活图像 + 已选中掩码 |
| Window Choose Preset | 手动选择窗宽窗位预设 | **项目已打开** + 已激活图像 |
| Window Undo Last | 撤销上次窗宽窗位 | 有可撤销的窗宽窗位操作 |
| Window Reset Full Range | 恢复图像全灰度范围 | **项目已打开** + 已激活图像 |

## 管理

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| Setup / Repair Environment | 安装/修复外部 Python 环境 | 网络（首次约需 5–15 GB） |
| Clear Cache | 清除所有临时文件和缓存 | 无 |
| Stop All Owned Services | 强制停止所有后台进程 | 无（切换功能前遇锁资源时使用） |
| Fix Source Affine Metadata | 修复旧版 bridge 导致的仿射矩阵错误 | **项目已打开** + 已激活从 NIfTI 导入的图像 + 源文件仍在 |

## 快速参考

- **遇锁资源**：先点「Stop All Owned Services」，再开新功能
- **停止导出**：使用「Stop Mask Export」，它只停止本项目创建的掩码导出进程
- **掩码识别时**：仅在结果对话框显示时切换工具，光标等待点击时不要切
