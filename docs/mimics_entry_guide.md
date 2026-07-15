# Mimics-Script 功能入口速查表

## 数据导入导出

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| Import Dataset | 批量转换数据集为 .mcs | 数据集文件夹（含案例子目录 + 医学图像） |
| Import Single Case | 单个案例 → .mcs | 一个图像文件 / 案例文件夹 / DICOM 目录 |
| Import Masks | 外部掩码文件注入当前项目 | **项目已打开** + 已激活图像 + 掩码文件（.nii.gz/.mha/.nrrd） |
| Export Masks | 当前项目掩码导出为 .nii.gz | **项目已保存为 .mcs** + 已激活图像 + 有掩码 |
| Stop Import Queue | 停止正在运行的批量导入 | 有批量导入正在运行 |

## AI 分割

| 入口 | 作用 | 需要满足的条件 |
|------|------|--------------|
| nnInteractive Segmentation | 以选中掩码为提示做 AI 分割 | **项目已打开** + 已激活图像 + 已选中掩码 + 环境已设置 |
| DINOv3 Train Model | 训练小样本分割模型 | 数据集（含已导出的掩码）+ 环境已设置 + 推荐 GPU |
| DINOv3 Predict (Latest Model) | 用最新模型推理当前案例 | **项目已打开** + 已激活图像 + 已选中掩码 + 已有训练好的模型 |
| DINOv3 Predict (Choose Model) | 手动选择模型后推理 | 同上，多个模型时弹出选择窗口 |
| DINOv3 Show Status | 查看任务进度/日志/训练曲线 | 有启动过的 DINOv3 任务 |
| DINOv3 Stop AI Task | 停止当前训练或推理 | 有正在运行的 AI 任务 |

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
- **导出无法取消**：导出在独立 Mimics 进程中运行，任务管理器终止
- **掩码识别时**：仅在结果对话框显示时切换工具，光标等待点击时不要切
