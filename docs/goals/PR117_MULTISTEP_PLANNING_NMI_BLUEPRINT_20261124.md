# Retro-E · 多步逆合成规划后训练与 NMI 正文 6 图实战蓝图（升级定稿版）

**版本：2026-11-25 CST (v3.0 · 2027 范式定稿版：Proof-Carrying 电子流定理证明 + 电子分叉相变 + 串联折叠 + 分子可合成性修复 + 多目标 Pareto 路由)**  
**核心科学立意（Paradigm Shift）**：从「反应级模式检索（Reaction-Level Pattern Retrieval）」迈向 **「可验证、可调试、可组合、可修复的可执行机理规划（Proof-Carrying Executable Mechanistic Planning）」**——把有机电子流状态机升维为有机化学的形式化编译器（Chemical Lean），在物理守恒汇编语言上实现：
1. **Proof-Carrying Synthesis**：每条合成路线均为机器逐行校验、附带双向机理证人（PR18 + 独立 FlowER 正向 + 官方 ChemCensor）的可执行物理证明；
2. **Critical Electronic Bifurcation Landscape（电子分叉相变景观）**：在精确相同的瞬态中间电子态上，量化仅差 1 对电子流向引发的全局合成树命运相变（Fig 3 & Fig 4）；
3. **Reaction-Boundary Dissolution（涌现式一锅法串联/多米诺折叠）**：打破离散“人名反应”边界，用连续电子流折叠文献多步级联转化（Fig 5）；
4. **Mechanistic Synthesizability Surgeon（机理级分子可合成性外科修复）**：面对 AI 从头生成的难合成“暗空间”分子，精准定位卡死电子步并给出最小原子微调修复建议，打通“逆合成 ↔ 生成式药物设计”闭环（Fig 6d）；
5. **Inference-Time Multi-Objective Pareto Steering**：基于严格守恒带来的精确整树原子经济性、绿色合规与起始原料简化度，免重训切换 **MedChem 模式** 与 **Process 模式**（Fig 6a/e）。
**状态：用户确认定型方案（进入 3–4 天 8×H20 集中执行阶段）**  
**核心原则**：完全自主（不依赖难以复现的 RetroChimera 专有权重）、所有主表数据均在 3–4 天预算内真实跑完、严格区分内部验证与独立测试、绝不将未实测的模拟目标写入正式结果。

---

## 1. 现状诊断与战略转向（为什么必须立刻切换）

### 1.1 前期短步训练（`round-000204`）的实测定位
- 已安全停止的 8×H20 Streaming 任务（`meteor_mechet_pr116_adaptive_streaming_8h20_20261124_12`，最新完整落盘 `round-000204`，`complete_sha256: 4cc1dfed79ebca034d2b7d45b2325559612333fb3c2a521398191914a62d3a9f`）累计覆盖 26,112 个 D3 目标。
- **关键诊断**：
  1. **80% 算力耗在短步片段**：约 80% 目标在 1–2 个 `FINISH` 即截断为 `segment_complete_continuation_unknown`，本质是局部短步适应/蒸馏，而非面向完整 AND/OR 闭合的多步规划后训练；
  2. **探索严重枯竭**：`stepwise.temperature = 0.25` 叠加尖锐的 SFT 基座导致策略熵仅 `0.0066`，同状态 K=2/K=4 分叉重复率极高；
  3. **全量 27,736 目标审计证实“撞库水分”与 1 步占比过高**：64.4%（17,850）被短步截断；在 3,008 条 `stock_solved` 中 **66.5%（2,001）仅为 1 步反应**，仅 128 条同时满足 `depth ≥ 2` 且 `min_pr18 ≥ 0.0`。
- **资产保留**：`round-000204` 完整保留，直接作为论文四阶消融中的 **“短步局部适应（Short-Horizon Adaptation）”** 对照基线。

### 1.2 对齐 2025–2026 前沿逆合成规划的 4 大核心瓶颈

吸收 **FlowER (*Nature 2025*)** 的“物理守恒 + 组合泛化”叙事与 **RetroChimera (*Nature 2026*)** 的“化学可信度优先（URSA `Solv-2`）+ 分层难度 + 复杂案例深度剖析”实验设计，将 Retro-E 精准锚定在现有反应级黑盒模型（RetroChimera / DirectMultiStep / R³ / RetroAgent / RetrOrchestrator）无法解决的 **4 大前沿瓶颈**：

| 前沿核心瓶颈 | 反应级黑盒模型的局限 | Retro-E（逆向电子流规划）的破局机制 |
|---|---|---|
| **1. 高闭合率 ≠ 高化学可信度**<br>（URSA Expert 上 RetroChimera `Solv-0=99% → Solv-2=32%`，DMS `85% → 15%`） | 黑盒字符串/模板在搜索期不校验原子/电子守恒与正向可逆性，大量路线死于隐式副反应与价键幻觉。 | **可执行电子流状态机 + 在线双向 CRN-JEPA (PR18)**：每步编译为严格电子守恒基元程序并同步验证逆向+正向可逆一致性，拉高 `Solv-2 / Solv-0` 留存率与 `MV-Solv`；并引入**独立第三方 FlowER 正向模型 + 官方 ChemCensor 双重交叉验证**消除“自评”质疑。 |
| **2. 局部机理可行 ≠ 全局多步可合成**<br>（Local–Global 错位） | 单步模型与短步微调只优化局部似然；实测局部打分器在 >60% 关键分叉点上偏好下游成环或死路的分支。 | **长程多步规划后训练（100% 深度 ≥ 2 完整路线 + 机理门控 ReFIT-PO）**：将终局 AND/OR 闭合结果反传至同电子态竞争动作，逆转局部短视偏好，涌现非单调预备步、汇聚拆解与战略成环断开。 |
| **3. 稀有/未见组合、复杂天然产物与 AI 从头设计分子泛化崩塌** | 模板库覆盖有限；SMILES 生成在复杂多环手性骨架与 AI 生成的“合成暗空间”上迅速失真。 | **基元电子流的组合泛化**：有机极性反应由有限基元电子推动组合而成，在 CbyD 200–300 天然产物与 200–300 AI 从头生成药物候选分子上展现显著 OOD 泛化优势。 |
| **4. 粗粒度反应级搜索、失败死循环与缺乏药物化学可控性** | 必须等完整前体生成后才分叉，遇错丢弃整步重来；无法在搜索期响应工艺/SAR 约束。 | **亚反应级电子态分叉 + 冲突导向回跳 + 约束可控规划**：在未完成反应态保留竞争电子态并按预算展现更陡的 Test-Time Scaling 曲线；支持禁断键/禁用试剂规避与 SAR 共同关键中间体发散规划。 |

---

## 2. 面向现实应用的奖励函数设计（Mechanistically-Gated Long-Horizon Reward）

$$R(\tau) = \underbrace{\mathbb{I}[\text{all-stock closed}] \cdot G_{\text{mech}}(\tau) \cdot G_{\text{stereo}}(\tau) \cdot \phi_{\text{LLS}}(\tau)}_{\text{机理与现实可用性门控的终局规划回报}} + \underbrace{R_{\text{discharge}}(\tau)}_{\text{受终局约束的库存消解进度}} - \underbrace{C_{\text{conflict}}(\tau)}_{\text{成环与重复加倍惩罚}}$$

1. **木桶机理门控 $G_{\text{mech}}(\tau)$**：取整条闭合路线中最弱一步的双向/正向 PR18 得分 $q_{\text{mech}}(\tau) = \min_{e \in \tau} s(e)$；$q_{\text{mech}} \le -0.50$ 压至 `0.20`，$q_{\text{mech}} \ge 0.10$ 平滑升至 `1.0`（参考路径验证加成 `+0.20`），彻底杜绝低质撞库。
2. **旁观手性守恒门控 $G_{\text{stereo}}(\tau)$**：未参与电子重排的旁观手性中心无故丢失构型时施以 `0.70` 衰减。
3. **步骤经济性与汇聚性因子 $\phi_{\text{LLS}}(\tau)$**：汇聚路线 `+0.05`，LLS > 4 每增一步温和衰减 `0.08`，抑制冗长直链与无意义绕圈。
4. **分级冲突惩罚 $C_{\text{conflict}}(\tau)$**：祖先成环（`ROUTE_CYCLE`）与同状态重复拒绝施以 `2.5×` 加倍惩罚。
5. **主动 K2 分叉与同状态反事实信用（ReFIT-PO）**：根部与关键电子决策步主动开启 `K=2` 探索（`low_score_threshold = 0.05`），利用语义去重后的同状态竞争结果反传长程优势。

---

## 3. 长程多步规划后训练配置（8×H20 正在执行）

- **热启动与锚定**：从 `round-000204` 热启动（`initial_adapter`），KL 参考策略严格锁定原始 `Stage-II SFT`（防偏离原子守恒基座）。
- **课程目标池**：仅保留 `stratum ≥ 2`（彻底剔除 1 步送分题），**100% `lane="full", max_finishes=0`**。
- **关键超参**：`max_optimizer_steps = 96`、`lr = 1e-5`、`kl_beta = 0.01`、`route_weight = 0.35`、`history_weight = 1.0`、`score_max_candidates = 64`、`max_slates = 64`。
- **并发与启动优化**：解锁 IO-bound 目标并发上限至 512 并发活跃目标（解决原先 4.26% KV cache 饥饿）；32 线程并行 `verify_contract`；复用已验证 H20 kernel unit 与 CRN 校准。
- **内部选点验证**：`validation_steps = [0, 32, 64, 96]` 在 **512 clean VALID** 上跟踪选点，锁定主模型 **`Retro-E-Plan`**（该内部选点轨迹仅放入 Supplementary）。

---

## 4. 3–4 天 8×H20 全量实战排期（含全量升级与五大升华实验，纯机时 ~68 小时 ≈ 2.8 天）

**严格数据隔离铁律（主文 100% 纯公开独立测试集，零内部集污染）**：
- **仅放 Supplementary / Extended Data（内部选点）**：`512 clean VALID` 仅作训练选点日志，**不进入正文任何主图/主表**。
- **正文 Figure 2–6（100% 公开独立测试集）**：
  - **RetroCast Reference 1,084**（全量 1,084 目标：既做 `SFT → 204 → Retro-E-Plan` 消融，也做全量 `C0` vs `C4` vs `C2` 控制器配对与 Test-Time Scaling）；
  - **URSA Expert 100 + Drugs 100**（全量 200 目标，官方 `ChemCensor` + 独立冻结 **FlowER** 正向双重交叉验证）；
  - **CbyD 200–300**（从 3,502 条复杂天然产物中按 LLS 5–9 / 10–14 / ≥15 预注册分层抽取 200–300 个，严格排除开发集 27 根）；
  - **Steerable Constraint & SAR Common-Intermediate Panel (~40 分子)**（3–5 个药物骨架系列，验证禁断键/禁用试剂规避与共同关键中间体发散）；
  - **AI-Generated "Synthesizability Dark Space" (200–300 分子)**（公开 AI 从头生成药物候选分子集）。
  - *（可选加餐：若 Day 4 仍有富余 8×H20 机时，补充 **PaRoutes n5 10,000** 高通量扩展基准表。）*

| 日程 | 核心任务（全部 8×H20 实测 / CPU 后分析） | 8×H20 耗时 | 交付里程碑 |
|---|---|---:|---|
| **Day 1–2** | • **96 步长程多步规划 RL**（`meteor_mechet_pr116_multistep_plan_v2_8h20_20261124_03`），自动在 `0/32/64/96` 步记录 512 clean VALID 内部选点日志（Supplementary）<br>• *CPU 并行准备*：完成 URSA/ChemCensor 适配器、Test-Time Scaling 前缀分析器、涌现合成策略解码器、CbyD 200–300 / SAR 40 / AI Dark Space 200–300 冻结清单与 8×H20 全卡并行评测脚本 | ~16h | 主模型 `Retro-E-Plan` + 全套 Day 3–4 评测脚本与冻结面板就绪 |
| **Day 3** | • **Reference 1,084 全量模型消融**（`SFT` vs `短步 204` vs `Retro-E-Plan`，8 卡全 Actor + 本卡 PR18 提速）<br>• **Reference 1,084 全量控制器配对**（固定 `Retro-E-Plan`：`C0 原生独立` vs `C4 反应级分叉` vs `C2 亚反应电子态分叉+冲突回跳`，同步记录完整搜索轨迹以零成本提取 **升华① Test-Time Scaling `B=8..256`**）<br>• *CPU 零额外 GPUh*：运行 **升华② 涌现合成策略定量解码**（非单调复杂度、汇聚度、成环断开、电子推动基序迁移） | ~24h | **Fig 2a/b**、**Fig 3b/c/d/e**、**Fig 4a/b/c/d/e** 全量 1,084 公开独立数据 |
| **Day 4** | • **URSA Expert 100 + Drugs 100**（官方 `ChemCensor` 得 `Solv-0/1/2` + `MV-Solv`，对标已发表官方文献值）<br>• **升华③ 独立第三方双重正向交叉验证**：冻结 **FlowER (*Nature 2025*)** 正向模型 + 官方 **ChemCensor** 双重校验新颖 `MV-Solv` 路线（~1.5h）<br>• **CbyD 200–300 复杂天然产物分层评测**（~8h）<br>• **升华④ 可控约束与 SAR 共同中间体发散规划**（3–5 药物骨架 ~40 分子，~2h）<br>• **升华⑤ AI 从头生成分子“合成暗空间”评测**（200–300 公开 AI 生成药物候选，~2.5h）<br>• **公开集全量压力测试**：Reference 1,084 真实 `ROUTE_CYCLE` 失败态全量恢复 + Reference 1,084 / URSA 200 闭合路线全量库存断供重规划（~4h）<br>• *可选加餐*：若机时有余，跑 PaRoutes n5 10,000（Extended Data） | ~26h | **Fig 2c/d/e**、**Fig 5a/b/c/d/e**、**Fig 6a/b/c/d/e** 全部真实数据 + 升华证据包，同步 `MechET-paper` |

---

## 5. 正文 6 张高密度多 Panel 主图设计（共 30 Panels，100% 公开独立测试集）

### Figure 1｜Retro-E 框架总览：亚反应级可执行电子流与长程机理规划（5 Panels）
- **a. 范式对比**：反应级黑盒（SMILES/模板）vs 亚反应级可执行电子流（`IMPORT → FLOW₁..k → FINISH` + AND/OR 树）。
- **b. 在线双向微观护栏（PR18）**：`S_before → 临时执行 → S_after` 同步评估逆向分 `s_inv` 与经化学回放验证的正向逆算子分 `s_fwd`。
- **c. 亚反应级分叉与冲突回跳**：真实分子在未完成反应态分出 A/B 两条合法电子流，A 下游触发 `ROUTE_CYCLE` 后精准回跳至 B 并闭合库存。
- **d. 现实约束机理门控奖励**：$G_{\text{mech}} \times G_{\text{stereo}} \times \phi_{\text{LLS}}$ 过滤撞库幻觉 + 保守电子流归因流图。
- **e. 长程多步规划后训练闭环**：`Stage-II SFT → 短步局部适应 (round-204) → 100% 完整深度 ≥ 2 长程机理门控 ReFIT-PO (Retro-E-Plan)`。

### Figure 2｜标准多步基准性能、化学可信度留存与独立双重正向校验（5 Panels）
- **a. Reference 1,084 `Solved@Budget` 主曲线**：`SFT → 短步(204) → Retro-E-Plan` 公开全量独立跃升（带 95% bootstrap CI，附 RetroCast 官方文献参考线）。
- **b. Reference 1,084 拓扑分层柱状图**：Linear (600) / Convergent (400) / Long (84) 分层对比，突出长链与汇聚增益。
- **c. URSA Expert (100) + Drugs (100) 阶梯衰减对比**：`Solv-0 → Solv-1 → Solv-2 → MV-Solv`，对标已发表官方文献值（RetroChimera-MCTS / AiZynthFinder / Retro* / DMS-Wide）的断崖衰减与 Retro-E 的高留存率。
- **d. 升华③ 独立第三方双重正向交叉验证（Zero Self-Judging）**：在 URSA 200 + Reference 1,084 新颖 `MV-Solv` 路线上，对比**冻结 FlowER (*Nature 2025*) 正向预测 Top-1/Top-3 命中率**与**官方 ChemCensor 通过率**，并展示 `Solv-2 / Solv-0` 留存率气泡图。
- **e. 真实完整多步汇聚路线案例（RDKit 高清图）**：1 条 4–5 步同时通过 `Solv-2`、FlowER 独立正向校验与 `MV-Solv` 的完整电子流合成树。

### Figure 3｜电子分叉相变景观（Electronic Bifurcation Landscape）、长程信用反纠偏与涌现策略（5 Panels）
- **a. 临界电子分叉相变案例（Critical Electronic Bifurcation）**：在精确相同的瞬态中间电子态上，仅差 1 对电子流向（1 个 `FLOW` 动作），支路 A 下游成环失败，支路 B 5 步高汇聚闭合到库存（附电子箭头相变全景图）。
- **b. Local–Global 错位与反纠偏统计**：同状态成败对比组中局部打分器偏好失败分支的比例 vs 加入 ReFIT-PO 长程信用后的逆转率。
- **c. 公开独立集（Reference 1,084 + URSA 200）多步消融对比**：对比 `SFT → 短步(204) → Retro-E-Plan` 的 `depth ≥ 2` 闭合率与木桶机理门控闭合率（`512 clean VALID` 选点轨迹移至 Supplementary）。
- **d. 公开测试集竞争分叉点策略概率迁移**：在 **Reference 1,084** 全部真实观测到的同电子态竞争分叉点上，对比 SFT / 短步(204) / `Retro-E-Plan` 的 $\Delta \log \pi(a_{\text{win}}) - \Delta \log \pi(a_{\text{lose}})$ 配对分布。
- **e. 升华② 涌现合成策略定量解码**：对比 `SFT → 204 → Retro-E-Plan` 在 Reference 1,084 + CbyD 上的四类高层合成策略涌现——**非单调预备步（BertzCT/重原子数先升后降）占比、汇聚度指数、战略成环断开占比、高阶电子推动基序分布迁移**。

### Figure 4｜亚反应级电子态调试：Test-Time Compute Scaling 与冲突精准回跳（5 Panels，全量 1,084 升级）
- **a. 升华① 全量 Reference 1,084 Test-Time Compute Scaling Law**：固定 `Retro-E-Plan` 权重，在完整搜索预算 `B = 8 → 16 → 32 → 64 → 128 → 256`（对数坐标）上对比 `C0 原生独立` vs `C4 反应级分叉` vs `C2 亚反应电子态分叉+冲突回跳` 的 `Solv-0 / MV-Solv` 缩放斜率。
- **b. 全量 Reference 1,084 配对胜负矩阵**：`Electron-only (C2)` vs `Reaction-only (C4)` 净胜与 McNemar 检验（按 Linear 600 / Convergent 400 / Long 84 拆分）。
- **c. 探索多样性与有效分支率对比**：全量 1,084 上对比 `SFT` vs `Retro-E-Plan` 的同状态不同合法电子后继数、策略熵与随预算扩展的去重路线增益。
- **d. 公开测试集真实成环失败态全量恢复率**：在 **Reference 1,084** 全部真实发生的 `ROUTE_CYCLE` 失败态上，对比 `Retry` vs `Reaction Rewind` vs `Electron Backjump` 的完整恢复率。
- **e. 冲突导向回跳时序图（Debugger Call Timeline）**：30+ 重原子分子从成环触发到精准回跳责任电子步并全树闭合的逐调用轨迹图。

### Figure 5｜复杂天然产物全合成（CbyD 240）、串联折叠与多步稀有断键泛化（5 Panels）
- **a. 公开多步路线全量频率分层泛化**：对 **Reference 1,084 + CbyD 240** 全部真实多步预测反应步骤，按训练库先例频率全量分层（`Frequent / Rare / Novel Electron Disconnections`），报告电子流可执行率、双向可逆一致率与独立 FlowER 正向校验率。
- **b. CbyD 240 复杂天然产物按 LLS 分层表现**：5–9 / 10–14 / ≥15 步的文献首步战略断键 `SDR@1`、连续两步匹配、完整闭合与旁观手性保持率。
- **c. 范式高光 · 涌现式一锅法串联折叠（Reaction-Boundary Dissolution，RDKit 大图）**：展示 Retro-E 打破“人名反应”离散边界，用连续电子流一步折叠文献多步级联/汇聚转化（含非单调预备步）的电子推动机理。
- **d. 天然产物案例 2（竞争断键对比 + 化学证据分层）**：新颖电子守恒路线 vs 文献路线对照 + CbyD 240 全部预测步骤的证据分层（精确先例 / 类似物先例 / FlowER+PR18 双重验证新颖可行）。
- **e. 复杂多环骨架的旁观手性守恒与步骤经济性剖析**：CbyD 240 上按手性中心数（0 / 1–2 / 3–5 / ≥6）分层的旁观手性保持率与 LLS 压缩比。

### Figure 6｜真实药物化学闭环：多目标 Pareto、SAR 发散与「机理级分子可合成性修复」（5 Panels）
- **a. 多维现实合成实用性雷达图（含精确原子经济性与绿色合规）**：在公开测试集（Reference 1,084 + URSA 200）上综合对比 `SFT`、`短步(204)`、`Retro-E-Plan`、`Retro-E-Plan + C2` 在 **闭合率、可信度留存 (`Solv-2/Solv-0`)、手性保持、步骤经济性 (LLS)、精确原子经济性 (Exact AE)、绿色合规与起始原料简化度** 全维度的领先。
- **b. 公开测试集全量库存断供扰动重规划**：对 **Reference 1,084 + URSA 200** 全部闭合路线屏蔽关键非平凡砌块后的绕路重闭合率与 Top-K 不同关键断键策略数。
- **c. 升华④ 可控约束规划与 SAR-40 共同关键中间体发散**：5 个经典临床药物骨架系列（40 分子）上的**禁断键/禁用试剂规避成功率**，以及将整个 SAR 系列收敛至**同一晚期共同中间体**后的总合成步数节约（附 4 分子共享核心中间体发散树图）。
- **d. 升华⑤ AI 合成暗空间评测 + 「机理级分子可合成性外科修复（Synthesizability Surgeon）」**：在 250 个公开 AI 生成药物候选分子上对比 `Solv-0` vs `Solv-2 / MV-Solv`；并挑选 2–3 个被电子流精准归因出卡死电子步的失败 AI 分子，展示**最小原子微调后瞬间从不可合成变为 3 步高 `Solv-2` 闭合**的反事实修复案例。
- **e. 免重训多目标 Pareto 切换（MedChem vs Process）与真实算力前沿**：展示同一模型仅切换偏好权重即可在 **MedChem（最短 LLS）** 与 **Process（最高机理稳健 + 精确原子经济性 + 零重金属）** 间平滑移动，并附 `GPU-sec / Solved Route` 效率账本。

---

## 6. Extended Data / Supplementary 清单
1. **ED Fig 1**：`512 clean VALID` 训练过程选点轨迹（step 0 / 32 / 64 / 96）与短步 `round-000204` 全量 27,736 目标诊断分布。
2. **ED Fig 2**：Test-Time Compute Scaling Law 分层曲线（按 Linear / Convergent / Long 与 URSA Expert / Drugs 拆解）。
3. **ED Fig 3**：独立 FlowER 正向模型 + ChemCensor 双重校验失败案例剖析（揭示哪类副反应/区域选择性冲突仍具挑战）。
4. **ED Fig 4**：SAR 共同中间体发散全部 3–5 个药物骨架系列完整路线图与 AI 生成分子高/低 SA-score 代表案例。
5. **ED Table 1（可选加餐）**：PaRoutes n5 (10,000) 高通量扩展基准结果（若 Day 4 机时允许）。
