# Mimics 后台任务反馈、停止与回退规范

本文规定 Mimics-Script 新增功能的统一交互和生命周期原则。目标是在不阻塞
Mimics GUI 的前提下，让用户知道任务是否运行、等待什么、如何停止，以及关键
写入失败后会留下什么状态。

## 1. 低打扰反馈原则

- 任务提交成功后只写一条英文 Mimics 日志，不增加“已开始”确认弹窗。
- 阶段、病例或进度发生变化时更新现有状态窗口并写一条英文日志。
- 资源等待或长阶段超过 30 秒后给出一次进度；状态不变时最多每 60 秒重复一次。
- 等待信息必须包含等待对象、已等待时间和可用的停止入口。
- 只有失败终态、覆盖选择、强制停止和数据风险需要弹窗。
- 完成结果需要用户选择写入位置时，结果选择弹窗本身就是完成通知，不再追加
  第二个“成功”弹窗。

## 2. 资源等待与停止入口

| 功能 | 可能等待的资源 | 用户可见反馈 | 释放方式 |
| --- | --- | --- | --- |
| 数据导入 | 外部转换、后台 Mimics、首个 `.mcs` | 导入状态、病例计数、60 秒日志 | `04 Stop Import Queue` |
| Mask 导入 | 外部空间对齐、活动项目、Mask buffer | 30/60 秒日志 | 再次运行 `05 Import Masks` 后选择停止 |
| Mask 导出 | Mask buffer、外部空间转换、后台 Mimics | Mask 计数、阶段、60 秒日志 | `06 Stop Mask Export` |
| DINOv3 | GPU、后台 Mimics、远程 GPU、结果写回 | Status 窗口、epoch/metric、60 秒日志 | `05 Stop AI Task` |
| nnInteractive 推理 | GPU worker、顺序提示推理、结果写回 | worker stage、sequence、60 秒日志 | 再次运行 nnInteractive 后丢弃会话，或停止后台服务 |
| nnInteractive 微调 | GPU、本地或远程训练 | Task Models 状态、loss/AUC、日志 | `Pause and Release GPU` 或 `Stop Training` |
| nnU-Net | GPU、Dataset ID、远程任务、结果写回 | Status 窗口、阶段、60 秒日志 | `04 Stop Running Task` |
| ScribblePrompt | GPU 或 Mask buffer | 阶段日志 | 再次运行同一入口后选择 `Stop` |

“暂停”只在训练器具备可靠 checkpoint 恢复语义时提供。DINOv3、nnU-Net、导入
和导出当前提供取消，不伪装成暂停；取消后资源必须在子进程确认退出后释放。

远程任务无法连接服务器时，普通“停止”不能伪装成已经释放远程 GPU。状态窗口会
将同一个停止按钮改成 **Abandon Locally**：它只结束本机等待，保留服务器、容器
名称和“远程进程可能仍在运行”的警告，供管理员后续核查。该按钮只在远程状态
未知时出现，并需要一次确认，不增加正常流程的操作负担。

远程传输显示总量、速率和预计剩余时间；模型下载前检查本地下载与解包空间。
容器日志按固定大小轮转，防止长期训练无限占用服务器磁盘。

## 3. 关键写入与回退

### 3.1 Mimics Mask

- Mask 导入遇到同名 Mask 时不再覆盖，自动创建 `名称 - Imported`、
  `名称 - Imported 2` 等可编辑副本。
- DINOv3、nnInteractive、nnU-Net 和 ScribblePrompt 的结果
  在完成后选择 `Update Selected Mask` 或 `Create Editable Copy`。
- 关闭选择框、目标 Mask 在后台运行期间发生变化、项目或活动图像不一致时，默认
  不覆盖现有工作；根据功能创建副本、等待原项目，或丢弃结果。
- 最终 `set_voxel_buffer()` 写入使用 `mimics.Transaction`。写入异常时调用事务
  回滚；成功写入作为单一事务提交。是否出现在 Mimics 原生 Undo 列表中需要在
  目标 Mimics 版本上按本文第 5 节验证。

### 3.2 导出文件

- 默认导出到用户选择的位置并跳过同名文件。
- 用户明确选择 `Overwrite existing` 时，先在同目录写完整临时 NIfTI，再通过
  原子替换发布。临时写入或替换失败时，已有文件保持不变。
- 导出 NIfTI 同时显式写入 qform 和 sform，且数组、shape、affine 必须对应原图
  网格；几何不明确时停止，而不是生成方向未知的标签。

### 3.3 模型与任务状态

- 新模型先写任务目录并完成校验，再更新模型注册表或 latest 指针。
- 远程下载先进入暂存路径，校验后发布；发布失败保留可恢复备份。
- 停止标记先写入，子进程确认退出后才发布 `cancelled` 并释放 GPU/后台 Mimics
  锁。无法确认退出时保持 `stopping`，不得提前释放锁让新任务抢占资源。

## 4. 错误提示分级

- **日志**：可自动恢复、等待、重试、清理失败但主任务仍可继续。
- **非阻塞弹窗**：任务失败、任务完成但包含失败病例、孤儿远程任务、停止超时。
- **需要决策的弹窗**：覆盖已有 Mask、强制终止仍存活的进程、清理可能仍被使用
  的运行环境。
- 每个失败终态必须保存错误、阶段、日志路径、相关 PID 或远程任务标识；不能只
  显示弹窗而不留日志。

## 5. Windows 实机验收

以下项目需要在真实 Mimics Windows 环境验证：

1. 启动一个占用 GPU 的训练，再启动另一 AI 任务。后者应立即显示持有者，60 秒
   后仍有进度，停止后可以取得 GPU。
2. 导入 20 例数据，令其中一例转换失败。队列应继续，首个成功 `.mcs` 可先使用，
   `04 Stop Import Queue` 应停止未开始病例并最终释放后台进程。
3. 在 Mask 导入期间切换项目。结果应等待原项目，不写入当前项目；再次运行入口
   可以停止。
4. 导入与现有 Mask 同名的标签。原 Mask 体素不得变化，新 Mask 名应带
   ` - Imported`。
5. 对已有 Mask 运行一次 AI 预测，分别验证 `Create Editable Copy` 和
   `Update Selected Mask`。后者完成后检查 Mimics 原生 Undo 是否可恢复更新前
   内容；若目标版本不记录脚本事务到 Undo，需要反馈版本号和结果以增加持久化
   回退入口。
6. 选择 `Overwrite existing` 导出，然后模拟输出目录只读或占用。导出应失败并
   记录日志，原 NIfTI 的校验和必须不变，目录中不得留下临时文件。
7. 训练完成后确认任务保持 `completed`，模型可立即刷新并选择；摘要日志失败不得
   把已完成任务改写成 Failed。

## 6. 自动化验证

本地测试覆盖：等待通知节流、Mimics 事务提交与异常回滚、同名 Mask 安全副本、
NIfTI 原子覆盖失败保护、qform/sform 写入，以及 nnInteractive 微调完成终态。
真实 Mimics 的事务 Undo 行为、超大 Mask 单次写入耗时和 Windows/SMB 文件占用
仍以第 5 节实机结果为最终依据。
