# 可直接粘贴到 Codex Goal 的执行指令

请在 **MechET** 代码仓库执行一个完整的、以**现有电子流树搜索输出**为数据来源的 NMI 正文指标分析与绘图任务。不要把这理解为再写一个计划，也不要恢复 RSI、重新训练模型或提交新的 Taiji 推理任务。

**首先完整阅读本 PR 的三个文件：**

1. [完整执行协议](NMI_R1_R2_TREE_EVIDENCE_GOAL_20261016.md)
2. [Figure 2/3 与树展示设计](NMI_R1_R2_FIGURE_DESIGN_20261016.md)
3. [统一配色、字体与 RDKit 绘图样式](../../configs/figures/mechet_nmi_tree_style_v1.yaml)

同时核对 [MechET-paper PR #29](https://github.com/wangyu-sd/MechET-paper/pull/29) 的 R1/R2 正文和实验矩阵，不得将原有 URSA/RetroCast/PaRoutes 完整路线指标或 FlowER H2 组合泛化指标移除。

**执行而非讨论：**

- 识别当前机器上实际存在的 tree/MCTS/beam/rollout/route JSONL 和记录清单，读取现有格式、尽可能完整地解析真实 parent-child、动作、化学状态、分数、terminal、预算和真实 timing。对缺失字段标注 unavailable；不要从 GT、图像、理论示意或推测中重造树。
- 实现规范化树节点/动作/终止反应/路线数据层、自动汇总和对应脚本，产生现有完整数据上的 morphology、edge chemistry、candidate ranking gap、独立化学证据和效率分解。
- 对化学可行性采用有层次的评价：executor accepted 是符号规则；局部 motif/结构变化是描述性证据；Tier A/B 文献必须指向**完整反应**的具体独立论文/专利来源，而不是给未验证的内部符号状态贴“真实机理”标签。没有文献不等于反应不可能，缺失信息不要默认判失败。
- **重点交付审美优秀的图而不是表格堆砌**：真实搜索树点线概览、真实 RDKit 分子图的关键分叉放大、电子源与电子汇以及键变化、真实终止前体、R1 在实际预算下的完整路线/排序/计算效率证据、R2 原始的组合泛化/复杂反应证据。采用不等宽、不等高的科学编辑排版，白底低饱和蓝灰/青绿/赭色配色，字体一致，矢量 PDF+SVG、300dpi 预览 PNG，并生成 Extended Data。
- 真实输入不足以填正式 R1/R2 全量实验时，仍使用全部已有树结果生成高质量 \`observed\` 图和代码；不可虚构 URSA/RetroCast 全量结果、H2 27,104 个模型输出或化学分子/文献。将缺失 cohort 记录成明确的后续资产依赖即可。
- 代码需要可以确定性重现，添加有意义的自动测试，并实际在现有数据上运行。每个 panel 的来源要能追到 run、target、node、action、route 和 DOI/patent（如有）；生成文件清单、英文 Nature 风格 captions、真实结果小结。

**完成定义：** 打开提交包含真实分析代码、绘图代码、图形风格配置、测试、运行说明、真实输出预览与 figure manifests 的 PR；清楚说明已使用哪一批原始数据（完整的 n/路径/source SHA），缺什么数据，以及哪些主图可以直接用于 MechET-paper PR #29。提交前检查图片缩放至 NMI 正文 183mm 宽后的真实可读性和化学结构/电子转移准确性。

请持续完成这个固定目标，不要每完成一个阶段再重新讨论是否需要改变 R1/R2、设计新模型、扩大训练任务或做小规模探索性实验。小型程序测试样例允许，但论文展示不能用模拟结果。

**此 Goal 的目标就是把已生成的电子流树结果转变为科学可解释、视觉专业、可直接进入 NMI 正文的 Figure 2/3 和相应的定量证据。**
