# Retro-E · 完整 NMI 结果规划（模拟最终成文版）

**版本：2026-11-23。重要警示：下文 Retro-E 所有预期成绩均为假设性研究目标，非实际测试结果。** 外部模型仅在写有 `REAL-PUBLISHED` 的行提供原论文已经报告的成绩；`PLANNED-RERUN` 的模型数值仍然是假设值，不能引用为公开成绩。建议与配套 Excel 的 25 张工作表一起使用。

## 0. 论文核心主张和数据包总量

科学问题：电子级决策能否比完整反应级决策发现更多可信的多步合成路线，以及长期路线成败能否训练出更好的电子策略？

- **6 张主图**：Figure 1 方法、Figure 2 系统性能、Figure 3 电子搜索、Figure 4 ReFIT-PO、Figure 5 效率蒸馏、Figure 6 复杂合成与化学证据。
- **主文评测**：URSA Expert 100 + Drugs 100；CbyD 冻结 500（拓展 2,902）；RetroBench 5,838；96 同 Actor 搜索配对；499 synthesis-needed VALID；另有 CbyD 关键断键、电子对照、库存扰动和化学证据。
- **独立化学**：3,120 反应（频繁 2,100、稀有 700、未见组合 320）；机理挑战 300；真实同电子状态对照 240；执行失败修复 150；库存扰动 240；盲评 80 条路线/模型；AI 生成分子 512（可选主文）。
- **CbyD 原始规模**：3,502 参考路线、56,640 反应步骤、2,902 个不同目标，文献路线最长线性序列中位数 13 步。采样与 Reference/Market 等存在目的差异，不应直接把所有数据集数字相加。

## 1. Result 1 / Figure 2：现代多步逆合成系统性能

### 1.1 URSA — 两个公开 100 目标集合

以下固定采用 **URSA-minor-1.1.0-U2 / ChemCensor v1.2.0** 的历史对照。外部原始论文预算与电子模型物理调用量不同；重跑需另报真实 GPU 秒。

| Benchmark | Model | 类型 | Solv-0 | Solv-1 | Solv-2 | MV-Solv（新） |
|---|---|---|---:|---:|---:|---:|
| URSA Expert (N=100) | RetroChimera-MCTS | REAL-PUBLISHED | 99 | 78 | 32 | 未知 |
| URSA Expert | AiZynthFinder-MCTS | REAL-PUBLISHED | 65 | 60 | 29 | 未知 |
| URSA Expert | Retro* | REAL-PUBLISHED | 57 | 53 | 22 | 未知 |
| URSA Expert | DMS-Wide | REAL-PUBLISHED | 85 | 26 | 15 | 未知 |
| URSA Expert | Stage-II SFT | SIMULATED-TARGET | 83 | 58 | 20 | 14 |
| URSA Expert | ReFIT policy | SIMULATED-TARGET | 86 | 65 | 28 | 22 |
| URSA Expert | Same-actor Reaction-first | SIMULATED-TARGET | 91 | 72 | 34 | 26 |
| **URSA Expert** | **Retro-E Electron** | **SIMULATED-TARGET** | **94** | **81** | **41** | **34** |
| URSA Drugs (N=100) | RetroChimera-MCTS | REAL-PUBLISHED | 100 | 83 | 60 | 未知 |
| URSA Drugs | AiZynthFinder-MCTS | REAL-PUBLISHED | 80 | 73 | 54 | 未知 |
| URSA Drugs | Retro* | REAL-PUBLISHED | 76 | 68 | 53 | 未知 |
| URSA Drugs | DMS-Wide | REAL-PUBLISHED | 87 | 55 | 43 | 未知 |
| URSA Drugs | Stage-II SFT | SIMULATED-TARGET | 85 | 69 | 38 | 31 |
| URSA Drugs | ReFIT policy | SIMULATED-TARGET | 89 | 75 | 49 | 40 |
| URSA Drugs | Same-actor Reaction-first | SIMULATED-TARGET | 94 | 79 | 56 | 46 |
| **URSA Drugs** | **Retro-E Electron** | **SIMULATED-TARGET** | **97** | **87** | **67** | **57** |

**关键目标**：Retro-E 在 URSA Expert Solv-2 41/100，超过 RetroChimera 原论文 32/100；Drugs Solv-2 67/100，高于 60/100。但 N=100 的比较必须配对重跑与计算不确定度，不能仅凭 +9/+7 分直接推断统计显著。

### 1.2 CbyD 全合成 — 500 个全新独立目标

这张表中的所有模型成绩都是 **模拟同集合重跑结果**；目前没有这些模型的原论文 CbyD500 分数。采用同一库存与独立 URSA/ChemCensor 评分。主文冻结 500，未来扩展到 2,902。

| Model | Solv-0 /500 | Solv-1 /500 | Solv-2 /500 | 文献首步战略断键 SDR@1 | 至少两步连续文献匹配 |
|---|---:|---:|---:|---:|---:|
| RetroChimera-MCTS (模拟重跑) | 36 (7.2%) | 27 (5.4%) | 17 (3.4%) | 82 (16.4%) | 35 (7.0%) |
| AiZynthFinder-MCTS (模拟重跑) | 26 (5.2%) | 19 (3.8%) | 11 (2.2%) | 78 (15.6%) | 25 (5.0%) |
| DirectMultiStep-Wide (模拟重跑) | 28 (5.6%) | 18 (3.6%) | 9 (1.8%) | 91 (18.2%) | 29 (5.8%) |
| Stage-II | 25 (5.0%) | 17 (3.4%) | 9 (1.8%) | 93 (18.6%) | 34 (6.8%) |
| ReFIT policy | 30 (6.0%) | 22 (4.4%) | 14 (2.8%) | 99 (19.8%) | 39 (7.8%) |
| Reaction-first Search | 39 (7.8%) | 31 (6.2%) | 22 (4.4%) | 111 (22.2%) | 48 (9.6%) |
| **Retro-E Electron** | **52 (10.4%)** | **43 (8.6%)** | **34 (6.8%)** | **133 (26.6%)** | **61 (12.2%)** |

按文献最长线性路径 LLS 分层，CbyD500 计划包含 100 条 5–9 步、200 条 10–14 步、200 条 ≥15 步的不同目标。假设 RetroChimera 在三层 Solv-0 为 13/17/6、Solv-2 为 8/7/2；Retro-E 分别为 19/23/10 和 12/15/7。注意 CbyD 文献路线本身并不必然终止于冻结库存，故不能以是否复现原始文献闭合作为 Solv-0 唯一判据。此前用于开发的 27 个目标要从主测试中排除。

### 1.3 RetroBench — 现代 Agentic-RL 同任务比较

| System | Route Top-1 | Route Top-5 | 状态 |
|---|---:|---:|---|
| R³ (ACL 2026) | **43.7%** | **58.7%** | REAL-PUBLISHED |
| FusionRetro | 37.5% | 51.2% | REAL-PUBLISHED |
| Stage-II | 2,245/5,838 = 38.5% | 3,060/5,838 = 52.4% | SIM |
| ReFIT-PO | 2,485/5,838 = 42.6% | 3,270/5,838 = 56.0% | SIM |
| Reaction-first | 2,510/5,838 = 43.0% | 3,370/5,838 = 57.7% | SIM |
| **Retro-E** | **2,715/5,838 = 46.5%** | **3,560/5,838 = 61.0%** | **SIM** |

**Route Top-k ≠库存闭合率**。应使用 R³ 原论文的参考路线等价协议，不可与 URSA 混算。

## 2. Result 2 / Figure 3：电子级决策的因果作用

### 2.1 同 Actor 96 目标配对（32 linear / 32 convergent / 32 long）

| Physical electron call budget | Independent closure | Reaction-first closure | Electron closure | ELRG（净增） |
|---:|---:|---:|---:|---:|
| 16 | 6/96 | 7/96 | 9/96 | +2/96 = 2.1 pp |
| 32 | 9/96 | 11/96 | 15/96 | +4/96 = 4.2 pp |
| 64 | 13/96 | 15/96 | **23/96** | **+8/96 = 8.3 pp** |
| 128 | 17/96 | 19/96 | 30/96 | +11/96 = 11.5 pp |

在 B=64：两者都闭合 13、仅电子闭合 10、仅反应闭合 2、两者失败 71，总和 96。Reference 1,084 的补充模拟配对为 both152、E-only53、R-only12、neither867，因此反应级164、电子级205，净增41。

### 2.2 电子决策价值、修复与泛化

| Experiment | N | 计划完成值 | 定义 |
|---|---:|---:|---|
| 冻结同状态竞争电子决策 | 240 | 70 个出现不同长程合成结局 | 真实合法 FLOW、独立后续搜索预算 |
| TRAIN 可观察同状态胜负比较 | ≥350 对 | 真实执行而非教师虚拟生成 | 目前实际只观察约48对 |
| 失败状态修复 | 150 | Retry 12、Reaction rewind 19、Electron rewind 32 条完整恢复 | 同一预算，不只看合法候选 |
| 未见电子组合恢复 | 320 | Stage-II 80、ReFIT 106；RC 重跑假设99 | 不同于前向 FlowER 任务 |
| 独立机理挑战 | 300 | ReFIT 199、Retro-E 207 个关键电子事件一致 | 另测原子映射可重放 278/300 |

3120 个单步泛化样本的分层设计为 Frequent 2100、Rare 700、Unseen combinations 320；每个模型真实运行相同测试，详细模拟成绩已在 Excel 13 表给出。

## 3. Result 3 / Figure 4：ReFIT-PO 长期信用分配

### 3.1 499 个固定需合成目标

| Policy | Solved@64 | Solved@112 | 证据 |
|---|---:|---:|---|
| Stage-II SFT | 67/499（模拟） | **72/499（真实）** | REAL @112 / SIM @64 |
| Adaptive Step64 | 未披露准确同口径 @64 | **71/499（真实）** | REAL-INTERNAL |
| Route-level RL | 73/499 | 77/499 | SIM |
| ReFIT without history credit | 76/499 | 80/499 | SIM |
| **Full ReFIT-PO** | **89/499** | **96/499** | **SIM** |

B=112 的模拟配对：both60、ReFIT-only36、SFT-only12、neither391，净增24。若真实结果未形成显著的独立改善，则这节不能主张成功学习。

### 3.2 电子动作偏好

计划冻结 240 个完全不进入策略训练的竞争状态。模型偏好最终闭合动作的模拟成功数：Stage-II 132/240=55.0%、RouteRL 151/240=62.9%、No-history 158/240=65.8%、Full ReFIT 173/240=72.1%。当前真实 PR117 截至 update125 为 47 个比较组、48 对成功/失败分支，覆盖极小，不能解释为已达到目标。

## 4. Result 4 / Figures 5–6：效率、复杂合成及外部化学证据

### 4.1 蒸馏与真实计算成本——全为 SIM

| Method | Physical call cap | Complete targets on Reference1084 | Total GPU-hours (SIM) | GPU-sec / solved |
|---|---:|---:|---:|---:|
| Stage-II policy | 64 | 126 | 45 | 1286 |
| ReFIT policy | 64 | 145 | 47 | 1167 |
| Retro-E Full Search | 64 | **205** | **94** | **1651** |
| Student policy | 32 | 183 | 31 | 610 |
| Student policy | 64 | **205** | **45** | **790** |
| Student + Search | 64 | 210 | 60 | 1029 |

如果 205 条闭合路线相同，Full Search 的 94 GPUh 对 Student 的 45 GPUh = 2.09x 训练后摊销计算收益。这需要真实同硬件运行才能成立。目前并没有成功训练出来的 Student，允许根据真实数据取消蒸馏主图。

### 4.2 Synthesis-Evidence Dark Space（可选择放主文）

公开 AI 生成分子需要先锁定真实 512 个 ID。模拟同一库存和 URSA 评价：

| Model | Solv-0 /512 | Solv-2 /512 | MV-Solv /512 | 全部步骤有独立先例的路线 |
|---|---:|---:|---:|---:|
| RetroChimera-MCTS（模拟重跑） | 49 | 35 | 24 | 15 |
| Stage-II | 33 | 22 | 14 | 8 |
| Reaction-first | 42 | 31 | 24 | 19 |
| **Retro-E Electron** | **78** | **65** | **52** | **34** |

### 4.3 库存扰动与可信路线多样性

在预先选择的 240 个对比模型原库存均闭合的目标上，删除同一关键原料。模拟恢复数：RetroChimera 39、Reaction-first 35、Electron-level 61。CbyD500 中模型 Solv-2 的分母和可信路线类别需区分：RetroChimera 17 个 Solv-2 目标，22 条可信不同路线类别；Reaction-first 22、30；Retro-E 34、49。

### 4.4 与 RetroChimera Nature 专家可信度对齐

**外部真实发表**：RetroChimera 在其精选困难案例中，经筛选后的 291 个反应步骤，96.9% 获专家接受；10 条复杂路线中 9 条整条认可。它们与新测试目标的抽样策略不同，不能直接当作我们的共同对照。

**Retro-E 模拟目标**：每个系统独立盲抽 80 条已完整生成且符合预注册条件的路线（理想情况下 ≥3 位专家盲评），Retro-E 期望整条路线被认可 56/80；独立反应先例 278/320 步；URSA Solv-2 64/80；本地原生电子全程重放 80/80。针对外部模型的 MV-Solv 不得自动赋 0，而应调用同一独立电子检查器。

## 5. 主文 6 张 Figure 的成文结构

- **Figure 1（3 panels）**：Retro-E 电子流决策、执行器、ReFIT-PO 概念图与真实化学示例。
- **Figure 2（4 panels）**：URSA 200（强外部标准）→ CbyD 500（真实全合成）→ CbyD 难度分层 → RetroBench 5838（新近 RL 对照）。
- **Figure 3（4 panels）**：96 同 Actor 配对路线结果 → 240 同电子状态真实后果 → 3120 未见电子组合 → 150 失败修复。
- **Figure 4（4 panels）**：局部教师与长程结果冲突 → 499 VALID 实际后训练收益 → 独立竞争状态偏好 → 后果距离/深度。
- **Figure 5（3 panels）**：搜索经验蒸馏 → 相同解决率下真实 GPU 成本 → 重复候选与长尾效率。
- **Figure 6（4 panels）**：CbyD 关键战略断键 → 库存扰动重规划 → 复杂化学结构路线案例 → 独立盲评与先例支持。AI 设计分子作为可选补充或扩展主图。

主文总计 **22 panels**，可将大表移入 Extended Data / Supplementary。

## 6. 研究重点、当前缺口及执行门槛

最重要的正式结果目标不是 25 张表每列都达到想象值，而是：

1. **URSA**：Retro-E Expert Solv-2 至少实测超过 RetroChimera 原文32%，目标41%；Drugs 目标67% 对原文60%。必须复现同一 U2 评分和对齐成本。
2. **CbyD**：在冻结的500个不同复杂目标上得到明确高于同批重跑强外部模型的 Solv-2 与关键战略断键恢复，目标34/500及133/500。
3. **Electron vs Reaction**：96配对同 Actor 目标上出现电子独有解决显著多于反应独有解决，目标10对2；扩展 Reference1084 目标，增量41。
4. **ReFIT-PO**：499固定 VALID 的实际 @112 闭合从72提升到96，而不是当前已知的72→71。
5. **Chemistry credibility**：将库存闭合、URSA Solv-2、机制重放、文献先例和专家盲评五种不同证据分开报告。

初步执行顺序应先确定 CbyD 数据、URSA 环境及外部模型的真实性能，再进行96同Actor配对与RL验收；不应在未得到真实结果前把 SIM 填入正式论文。

## References / Code

- RetroChimera Nature 2026: https://www.nature.com/articles/s41586-026-11160-9
- RetroChimera official code/checkpoints: https://github.com/microsoft/retrochimera
- URSA official evaluator: https://github.com/insilicomedicine/URSA
- URSA public target sets: https://huggingface.co/datasets/insilicomedicine/URSA-benchmarking-sets
- URSA publicly documented protocol/results: https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
- CbyD v2026.11 data/tree conversion: https://github.com/chemistry-by-design/cbyd
- CbyD arXiv: https://arxiv.org/abs/2611.24996
- R³ ACL 2026: https://aclanthology.org/2026.acl-long.1745/
- AbSynth JACS 2026: https://doi.org/10.1021/jacs.6c08434
- Current Retro-E running record: https://github.com/wangyu-sd/MechET/pull/117