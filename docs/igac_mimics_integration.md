# IGAC 与 Mimics 集成说明

## 目标

IGAC 用于在已有 3D Mask 基础上进行交互式主动轮廓修正。Mimics 只负责启动时读取当前图像和 Mask，以及用户确认后回填最终结果；显示、交互和 LGDF 计算均在外部 PySide6 进程中运行，不持续占用 Mimics GUI 线程。

## 用户操作

1. 在 Mimics 中选中一个 Mask，运行 `02_AI/IGAC.py`。
2. 外部窗口准备完成后保持暂停，Mask 不会自行变化。
3. 默认使用 `Boundary drag (FFD)`：把鼠标移到青色 Mask 轮廓附近，待边界点显示为黄色后按下，拖到目标边界并松开。拖动期间实时显示局部 3D B-spline 形变，松开后 IGAC 才进行有上限的边界吸附。
4. 需要涂抹约束时切换到 `Correct by stroke`，在错误区域点击并按住滑动：
   - 在 Mask 外的缺失区域操作，自动转换为 Add；
   - 在 Mask 内的多余区域操作，自动转换为 Barrier；
   - 该操作是连续笔画，不是边界点拖拽；需要精确移动现有轮廓时使用 FFD。
   - 第一次明确跨界后锁定整笔语义，直到松手不再改变。
5. 普通笔画会实时形成 Add/Barrier 约束并触发 LGDF；FFD 拖动只做轻量局部预览，避免耗时演化阻塞鼠标，松开后再拟合到稳定条件并自动暂停。两种操作都只有一个可见结果，并支持 `Undo last` 精确撤销。
5. 使用 `Stop` 可立即停止后续迭代并保留当前预览，`Undo last` 可完整恢复本次点击或笔画之前的轮廓和约束，`Reset` 恢复打开窗口时的 Mask。
6. 检查三个正交视图后，选择更新原 Mask 或创建可编辑副本，再点击 `Apply to Mimics`。关闭或取消窗口不会修改 Mimics。

## 算法边界

本实现保留 IGAC 的局部高斯分布拟合（LGDF）水平集演化和 Add、Erase、Barrier 交互语义。用户输入只修改 IGAC 约束和水平集初始状态，后续边界变化由 LGDF 能量、曲率和正则项计算。

`Correct` 是输入翻译层，不是新分割算法。它根据按下时的二维 Mask 快照，把一整次手势转换为连续 3D capsule 形式的 Add 或 Barrier 约束。若拖动跨越初始轮廓，后台会先恢复本笔按下前的精确状态，再按跨界后的语义重放完整轨迹，确保不会残留相反约束。转换后的约束图若与原版手工输入相同，LGDF 的计算结果也相同。自动判定不可能理解用户未表达的解剖意图，因此仍提供显式覆盖和单步撤销，但不要求用户在常规修正中频繁切换工具。

`Boundary drag (FFD)` 是独立交互，不是把涂抹换一个名字。鼠标按下必须捕获当前二维切片上的真实 Mask 边界点；拖动位移在显示平面内定义，但紧支撑三次 B-spline 在物理毫米空间中同时形变邻近切片。每个预览都从鼠标按下时的三维水平集快照重算，不累计插值误差。位移过大时自动扩大影响范围以降低形变场折叠风险，松开后再由同一个 IGAC LGDF 内核吸附图像边界。

实现直接形变有符号距离场，不经过 Mask 转网格再栅格化，因此不会额外引入一次表面离散化。FFD 只负责把现有边界移动到合理邻域；弱边界是否能继续贴合仍受图像对比、初始 Mask、LGDF 参数和最大自动位移限制，不应把 FFD 当作新的全自动分割模型。

本入口不再包含以下非 IGAC 行为：

- 搜索图像梯度峰值作为拖拽终点；
- 使用插值场直接扭曲 `phi` 或 Mask；
- 把固定若干次迭代误认为已经收敛；
- 鼠标松开后无限运行。

如果需要“两点之间沿梯度寻找最优 2D 路径”，应使用 LiveWire/Intelligent Scissors。这是另一类算法，不应混入 IGAC。

## 实时性设计

普通笔画按轨迹批量合并；FFD 鼠标移动只保留尚未处理的最新指针位置。拖动期间不运行昂贵的 LGDF 循环，只刷新局部 FFD 结果；鼠标松开后才启动可停止、可自动收敛的吸附。该顺序优先保证指针跟手和结果可控，同时仍让最终轮廓由图像项参与调整。

- GUI 线程只负责绘制图像、约束覆盖和接收鼠标事件。
- 灰度底图只在切片、方向或窗宽窗位变化时重建；轮廓动画帧只更新轻量覆盖层，避免重复进行整幅浮点 RGB 合成。
- 高频画刷段在 worker 入口合并，同一笔内连续、不丢点，但不会为每个鼠标事件堆积一个计算任务。
- 每轮最多执行配置的少量 LGDF 迭代，然后重新检查用户命令；因此 `Stop`、`Undo last` 和关闭窗口不必等待一个长迭代批次。
- 显示帧按固定时间间隔发布，过时的鼠标段合并处理，不追赶无意义的旧帧。
- 鼠标按下时保存一个精确的设备内撤销快照。只保留最近一次操作，限制额外显存占用。

“立即反馈”包含两个层次：手势覆盖由 GUI 立即显示；约束写入后的 Mask 和真正的 3D LGDF 边界根据 GPU/CPU 速度逐帧更新。它是连续动画式拟合，不是假装成毫秒级梯度吸附。单次 LGDF 迭代仍是不可中断的响应上限，因此大 ROI 的实际帧率必须在目标 GPU 上验收。

## 自动停止

停止条件基于实际二值 Mask 的变化，而不是固定迭代次数：

1. 每个计算周期统计本周期改变的体素数及其占 ROI 的比例；
2. 达到最少观察轮次后，连续多个周期低于变化阈值才判定稳定；
3. 迭代数和运行时间仅作为防止弱边界持续演化的安全上限；
4. 达到安全上限时保留当前预览、显示警告并暂停，不自动提交结果。

默认参数位于 `interactive_algorithms_config.json`：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `display_interval_seconds` | `0.06` | 外部窗口最大约 16 FPS 的结果刷新间隔；计算较慢时自动降速 |
| `iterations_per_cycle` | `1` | 每次检查命令前执行的 LGDF 迭代数 |
| `convergence.minimum_iterations` | `3` | 开始判断稳定前的最少观察迭代 |
| `convergence.stable_cycles` | `3` | 连续稳定周期数 |
| `convergence.changed_voxel_ratio` | `0.00001` | 允许变化体素占 ROI 的比例 |
| `convergence.maximum_iterations` | `80` | 单次交互的迭代安全上限 |
| `convergence.maximum_seconds` | `15` | 单次交互的时间安全上限 |
| `parameters.max_displacement_mm` | `10` | 相对初始 Mask 的自动位移壳层 |

## 数据和坐标

- 图像、初始 Mask、约束和结果始终位于同一个 Mimics voxel grid。
- 外部窗口优先使用可获得的原始 CT HU 或 MRI 信号；不可用时使用具有 metadata 关系的 Mimics buffer 表达。
- 窗宽窗位只改变显示，不改变 LGDF 使用的归一化强度。
- spacing 用于高斯邻域、导数、曲率、距离场和画笔的物理毫米计算。
- 结果返回前再次验证 shape、病例身份和目标 Mask。目标 Mask 在 IGAC 打开期间被修改时，只允许创建副本，不覆盖较新的人工工作。

## 资源和失败处理

IGAC 默认优先使用 CUDA，并与其他 AI 功能共享 GPU 锁。空闲 nnInteractive 服务可在竞争时主动释放模型；真实推理正在运行时只显示等待原因，不强杀结果。CPU 回退只允许用于受限 ROI。

外部窗口关闭、取消、失败或应用结果后都会释放 GPU 锁和计算引用。Mimics 中唯一无法后台化的是启动时读取 voxel buffer 和最终一次性写回 Mask；两处均保持为短事务。中间结果不会逐帧回写 Mimics，因此 IGAC 失败不会留下半个 Mask。

## 适用范围

适合已有轮廓基本正确、局部需要扩张、阻挡泄漏或改善强度不均匀边界的 CT、MRI 和其他灰度体数据。它不负责从空 Mask 识别器官，也不能在完全不可见、严重伪影或相邻组织统计分布相同的区域推断解剖语义。此类情况应先由 nnInteractive、DINOv3 或 nnU-Net 产生初稿。

## Windows 实机验收

1. 打开窗口后等待 10 秒，未操作时 Mask 不变化。
2. `Correct` 在 Mask 外自动补入、在 Mask 内自动移除；从内部跨到外部整笔重判为补入，从外部跨入整笔重判为移除。
3. 跨界重判会移除先前相反约束；第一次跨界后保持锁定，即使实时轮廓越过鼠标轨迹也不会中途反转。
4. 鼠标按下和移动期间可看到手势及轮廓更新，不必等待抬起。
5. 快速拖动无断点，持续操作不会产生越来越长的响应延迟。
6. `Boundary drag (FFD)` 只有在轮廓点高亮后才能启动；远离边界按下不改变 Mask，也不消耗一次撤销。
7. 拖动期间轮廓跟随指针，松开后状态显示 `Snapping dragged boundary`，随后在收敛或安全上限处自动暂停。
6. 抬起后自动稳定并暂停；继续等待时 Mask 不再变化。
7. `Stop` 在当前 LGDF 小周期结束后生效，`Undo last` 精确恢复上一笔，`Reset` 恢复初始 Mask。
8. 达到时间或迭代安全上限时显示警告但不提交结果。
9. `Discard` 不修改 Mimics；`Apply to Mimics` 只产生一次可撤销事务。
10. nnInteractive 完成后立即打开 IGAC，空闲模型及时释放，不出现无原因等待。

## 参考

- [IGAC 官方实现](https://github.com/cwkx/IGAC/)
- [Interactive GPU Active Contours for Segmenting Inhomogeneous Objects](https://link.springer.com/article/10.1007/s11554-017-0740-1)
- [Live-Wire-ing the Insight Toolkit with Intelligent Scissors](https://insight-journal.org/browse/publication/230/)
- [AnatomySketch-Software（交互和可用性参考）](https://github.com/DlutMedimgGroup/AnatomySketch-Software)
- [AnatomySketch 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9712876/)

AnatomySketch 仓库未提供顶层开源许可证。公开可下载或仅用于学术研究不等于允许复制、修改或再分发其源码和 DLL。因此本项目没有提交其二进制，也不在运行时依赖它。`tools/ffd_reference_dll_adapter.py` 只供已获得相应授权的用户在 Windows 子进程中做数值对照；默认 IGAC 使用本项目的 PyTorch FFD。
