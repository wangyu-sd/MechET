# Codex Goal — R1/R2 用真实文献验证电子流搜索得到的反应和完整路线（Literature-first）

请直接在当前 **MechET PR #89** 分支继续执行，不要只是输出实验设想，也不要重新训练或扩大耗时的 MCTS。**新的主科学问题已明确：评估模型自主构造的每一条真实合成反应是否有文献实验先例，以及由这些反应组成的合成路线是否具有可追溯的化学证据。参考反应物一致性、NLL Top-K 和 Ranking Gap 不是本轮主要评价指标。**

完整读取并以此为优先级：

1. **[LGRV-v1 正式协议](MECHET_LITERATURE_GROUNDED_REACTION_ROUTE_PROTOCOL_20261018.md)** — 必须执行的完整步骤、Tier A/B/C/D 判据、反应检索流程、化学与机理证据区分、路线证据指标。
2. [更新后的 Baseline / R1/R2 执行目标](NMI_R1_R2_MATCHED_BASELINE_GOAL_20261017.md)。
3. [Figure 2/3 化学主导的正式版式](NMI_R1_R2_MATCHED_FIGURES_20261017.md) 和 `configs/figures/mechet_nmi_tree_style_v1.yaml`。
4. **最新原始结果优先**：[PR #88 的进度及逐目标索引](https://github.com/wangyu-sd/MechET/blob/docs/experiment-progress-no-ranking-20261017/docs/results/EXPERIMENT_PROGRESS_20261017.md)；PR #87 提供已有的树/执行器证据、化学案例和图形脚本；论文 PR #29 保留原定 R1、R2、R3 五 Results 故事。

**不能沿用旧任务的过时数据范围：**
- 最新 R1：200/460 完成报告，其中 **199 条非零完整 stock-AND 闭合目标**，另外一个目标本来就在 stock；还有 260 个目标未完成。原 PR #87 171/460 是历史快照，保留复现，但不当最新状态。
- 最新单反应搜索：完整 **3,120/3,120** 测试目标，每个最多 **10 次完整电子轨迹尝试**，其中 **3,022 个目标至少有一个完整终止前体**。对这些真实生成的前体进行反应文献检索，而不是以仅 **300 个 reference-hit** 为中心。
- 两组完整 460 Greedy 路线输出继续作为源数据；同目标的 Greedy/MCTS 反应步骤，必须用完全相同的化学规范化和文献检索进行比较。

## 实际执行内容（持续完成固定目标）

**A. 提取真实反应，而非参考答案。** 从 R1 每条模型产生的完整路线提取所有 `structural precursor(s) → product` 合成反应步骤；从 31K 每个目标全部可用的已结束候选提取结构前体。保留 IMPORT 辅助组分、FLOW 电子来源/汇及映射，但不要把单个 FLOW 箭头当成文献可验证的完整合成反应。建立可去重的 reaction key、route key、反应中心和模型原始路径；未完成的输出保持缺失标记。

**B. 进行实际结构检索与实验原始来源核实。** 优先使用有结构化输入/产出信息的 Open Reaction Database、公开 USPTO/Lowe 反应提取数据；匹配确切反应物和产物后，回到原始专利的实际制备实施例或论文 experimental / Supporting Information。对没有完全相同结构的候选按真实反应中心、局部官能团和相容底物条件检索可类比反应。记录 DOI/专利号、实施例/段落、实际 reactants、products、conditions、源数据版本、检索匹配方式和模型训练来源重叠可能性。

**C. 双层化学证据，绝不能用 LLM 自己编造。** 对完整反应标注 A（相同底物—产物的真实实施先例）、B（有真实实验的密切化学类比）、C（证据未定）、D（确证化学冲突），再独立标明 `machine_indexed / source_passage_checked / human_reviewed / unresolved`。专利宽泛权利要求或文献标题不能直接记 A；没有检索到文献也不能算 D。对内部电子 FLOW 另记 `formal / known-mechanism-compatible / source-explicit / unresolved`，实验上有完整反应先例不等于每一步符号电子箭头都被实验观察。现有 Cabozantinib 的 `US20160022662A1` 只证明酸氯—芳胺偶联结构转化。

**D. 先统计 R1 文献支持的整条路线。** 主指标包括 `StockSolved@B`、`AnyExactStepPrecedent@B`、`AllStepExactPrecedentSolved@B`、`PrecedentCoveredSolved@B`（所有步骤 A/B）、`RouteStepEvidenceCoverage`、`LiteratureSupportedDisconnection`、相同目标上 MCTS-only 支持的新断键/路线，全部基于冻结 460 分母，同时报告当前 200 已保存目标的证据覆盖。不把任何尚未检索的步骤写成化学无效。路线中只要有一条 C，不能声称“所有反应均有文献支持”。

**E. R2 研究电子流组合的实际化学意义。** 将训练未见的电子操作组合映射到去重的真实完整反应/前体/新断键，再对这些完整反应查文献来源，形成 `unseen program → structurally novel transformation → Tier A/B supported reaction → stock-connected route`。用相同目标、相同检索范围的 Greedy 作为比较。历史签名 604/6553 与当前 actor 603/6491 不可混用；RetroCast 原 C0/C1/C2 参考机制标签仍为 Unknown，H2 27104 是独立历史训练谱系。

**F. 正式 NMI 图和案例。** Figure 2 显示库存可达路线中的实验先例覆盖、Greedy/MCTS 同目标对照和一条真实 source-linked 完整反应路线；Figure 3 显示电子流程序组合如何产生有实验先例的化学变换。可复用 R3 Figure 4 的详细 source-supported 不同前体展示。一个大而清晰的 RDKit 分子案例 + 原始 FLOW 箭头 + 各合成反应步骤的 DOI/专利实施例，比一张挤满点和线的树更重要。低饱和蓝灰/青绿/赭色、白底、可编辑 183mm PDF/SVG、300dpi PNG、完整图注和逐反应 source ID，不要默认卡片风格和长篇审稿式提醒。

**G. 交付真正运行的代码/数据。** 包含文献反应结构索引/检索、正反应结构标准化、来源段落解析与匹配、分层 evidence records、整条路线 A/B 覆盖汇总、同 ID Greedy/MCTS 比较、分子图/机理一致性、自动测试、准确 n/分母、来源 SHA 和可复现命令。对全量已有可访问输出运行，不做假阳性来源。若当前机器没有 ORD/USPTO 数据，先检查有无已下载索引，再提供公开来源地址并实现合法可复现的下载/解析适配器；当前能检索到的结果照常完成，不要变成只写计划。

不启动新 GPU 训练、不重跑昂贵 MCTS、不恢复 RSI、不删原 R1/R2/R3 任务、不主动 merge 论文 PR。重点是**从已有真实推理结果里挖掘有实验文献依据的化学优势和合成应用价值**。