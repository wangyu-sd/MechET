# Codex Goal — 请直接执行 PR #87 后续的 R1/R2 Baseline 对照与 NMI Results

请在 `wangyu-sd/MechET` **当前这条 PR 分支**工作；它直接基于 [PR #87](https://github.com/wangyu-sd/MechET/pull/87) 的真实搜索树分析代码和数据成果。**不要仅仅提出新方案或再做审稿意见总结。请实际运行分析、实现缺失的评估器和对照、生成图表并提交实现 PR。**

首先完整读取：

1. [冻结的完整实验、现有数据与交付指标](NMI_R1_R2_MATCHED_BASELINE_GOAL_20261017.md)
2. [NMI Figure 2 / Figure 3 / Extended Data 正式版式](NMI_R1_R2_MATCHED_FIGURES_20261017.md)
3. [可机读的现有 run ID、分母、baseline 与统计协议](../../configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml)
4. 现有 [PR #87 数据和图表结果](../results/nmi_tree_evidence_20261016/README.md)、`docs/analysis/NMI_TREE_EVIDENCE_20261016.md`、`src/mechet/analysis/` 和 [paper PR #29](https://github.com/wangyu-sd/MechET-paper/pull/29) 的 R1/R2 实验协议。

**本轮科研目标：把 MechET 已经生成的 170 个非零库存闭合目标、14,040 个 FLOW 分叉上下文、71 个多路线目标和 3,672 个训练未见程序签名转化成有独立 baseline 比较的、能直接写进 NMI 正文的科学发现。**

务必真正完成以下工作：

**A · 先做现有结果的 target-paired 直接对照。** 对已有 MCTS 的 171 个相同目标 ID，连接两组已完成的 460-ID Greedy 输出，形成分别按 PaRoutes、RetroCast、URSA Drugs/Expert 统计的 both solved / MCTS-only / greedy-only / neither、精确胜负目标列表、完整路线图去重、新反应中心、新前体和电子流分叉来源，并把 MCTS 的整套 460 分母库存闭合下界保留。已有 greedy 和 MCTS 预算不同，不得把描述性相对数量当成等预算提升；但必须充分挖掘并展示真实 MCTS-only 新断键和完整路线案例，而不止报告限制。

**B · 把 R2 的程序新颖性变成真正的化学收益分析。** 按正确 actor 训练暴露区分 ordinary 和 nonordinary 程序，比较相同 ID 上 Greedy 与 MCTS 的完整电子程序新颖性、不同化学后继、不同 canonical 结构前体、不同断键、最终是否组成库存闭合路线；分析复杂度和 RetroCast 2–5 步分层。保留历史 604/6,553 和 H2 27,104 的独立训练谱系，不能拿全量训练的 0.6B 冒充 H2-only 泛化。

**C · 立刻用已保存的 mech-USPTO 1,711 个 MCTS 候选池以及两组完整 3,120 个 Greedy 结果，统一正式结构前体端点评价并重新计算 Pool Recall、NLL Top-1/5/10、Ranking Gap、MRR。** 修正旧 `rxn_prod_min` 代理口径，同时保留代理结果供溯源；先使用相同冻结候选池比较简单无 GT 排序器，不要重新跑耗时的 MCTS 仅为了改排名。

**D · 真正完成缺失的核心 Baseline。** 优先复用现有 independent-K 和传统完整反应级规划器资产；若不存在可比输出，在相同 Stage-II Qwen、相同目标、库存、chemistry executor、采样协议和预冻结预算下运行必要的独立完整轨迹采样和常规候选级规划对照。实现并实际输出 full R1 460 / 当前已完成 MCTS 171-ID 交集的对照；所有未完成项提供准确目标清单和可执行命令。不要无限制重跑长达每目标数小时的 MCTS，不要换基础模型或训练 RSI。若当前机器没有资源，E1/E2 的真实实测对照和高质量图必须先完整交付，不能因为缺少 GPU 就停在计划。

**E · 交付可直接进入 NMI 的比较性 Figure 2 和 Figure 3。** Figure 2 主体展示 target-paired baseline rescue、库存闭合/预算曲线（只有真实时间或成本序列才能画）以及一条真正由电子流关键分支产生的替代断键与完整合成路线。Figure 3 主体展示完整程序新颖性 → 化学上不同的前体/断键 → 库存可达的化学效用，保留真正独立的 H2 研究定位和复杂反应化学案例。画分子用真实 RDKit 结构和源–汇电子箭头，白底、低饱和蓝灰／青绿／赭色，严谨但不要满页免责说明。沿用既有 NMI 183mm 矢量 PDF/SVG 和 300dpi PNG 的美学标准，别做默认三栏卡片式 AI 图。

**F · 提交可复现交付物。** 至少包括逐 ID 对比 CSV/JSON、实际指标表（精确 n/分母及 paired CI）、源路径和 SHA manifest、按化学价值选择的实际正向 case registry、图对应的来源证据、Figure 2/3、Extended Data、英文 Nature 风格 captions、数据和运行脚本、自动测试。不要覆盖 PR #87 的旧图或改变原始结果哈希。最终简洁汇报最强的观察性优势、已补齐的等预算证据、剩余不可用的 H2/化学数据以及正式 paper PR #29 中哪些图可以更新。

请**持续完成此固定目标**，不要每完成一部分就要求重新讨论论文方向，也不要改掉 R1 四类应用、R2 的 H2 严格泛化或原有 16 个工作包。所有主要优势必须来自实际输出、同源结构和真实化学案例。