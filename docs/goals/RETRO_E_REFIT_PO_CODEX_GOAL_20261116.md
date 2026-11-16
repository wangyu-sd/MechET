# Retro-E ReFIT-PO：同状态电子流分支信用分配与失败动作惩罚

> **Research execution authority added 2026-11-16:** [Progressive RL, benchmarking and required final ReFIT search-to-policy distillation](RETRO_E_PROGRESSIVE_RL_BENCHMARK_SEARCH_DISTILLATION_20261116.md).
> For the **Standard / controlled +F benchmark** conditions, the new plan's explicit **NO RetroChimera** requirement overrides the older optional RetroChimera sections in this document; the optional existing teacher integration described below applies only to a separately labelled **Pro/open-enhanced** condition. Forward FlowER is optional only in +F, and final TRAIN-only search-to-policy distillation is mandatory. PR109 remains docs-only; active training is untouched.

**Codex implementation goal | 2026-11-16**

## 0. 执行摘要

本任务是在现有 SFT-native Process GRPO 上实现一个受 ReFIT 采样规范支持的**同状态兄弟分支条件信用分配**机制，暂定名称：

**ReFIT-PO — ReFIT-sampled Policy Optimization**

ReFIT-PO 不是新的 RL 框架，不替换 GRPO，不修改 ReFIT 的采样职责，也不把 Teacher、Sampler 和 Optimizer 混成一个模块。它只扩展现有 Process GRPO 的 Advantage 构造：当多个不同的 FLOW 决策从同一个精确化学决策状态出发时，比较它们各自的后续路线收益，并把条件 Advantage 分配给产生分支差异的 FLOW 动作。

本轮同时明确 FLOW 失败处理：

- Executor 明确拒绝的非资源性 FLOW 是一个真实失败动作；
- 单条采样 rollout 在首次非资源性拒绝后立即终止；但 ReFIT 将失败记录在 action edge 上，父节点仍可回溯并重新采样其他动作；
- 失败 FLOW 动作获得现有有界 validator failure penalty；
- FLOW 失败、无有效 FLOW 的 premature FINISH、同状态 repeated FINISH 均产生有界负向信用；失败 rollout 的 route task reward 置零，并撤回此前未完成路线的部分 progress；
- 失败分支作为负向样本保留在同状态兄弟分支组中；
- 失败 edge 不会杀死 parent chemical state；ReFIT 可以从 parent 或最近 OPEN ancestor 重新展开，其他兄弟分支继续运行；
- GPU、显存、上下文、超时等资源删失不是化学失败，不得处罚。

**FlowER 本轮明确不接入。** FlowER 是后续可选的正向机理验证器；本任务不增加 FlowER 推理服务、依赖、队列、模型加载或训练损失。RetroChimera 保留为当前逆向反应知识来源，可提供已缓存且可可靠映射的分支证据；它不替代真实路线回报。

本 PR 仅包含设计与 Codex 实施规格，不修改生产训练代码、不启动训练、不更改当前运行任务或正式训练合同。

---

## 1. 当前代码基线与必须保留的资产

实现前先读取 GitHub 最新代码，不以本文件中的 SHA 代替实时检查。

当前设计所依据的基线：

- PR #107：SFT-native Process GRPO，当前审查快照 HEAD 为 944780f6e24410f654f115bebe5f6f8fb2b142cd。
- PR #108：首次非资源性拒绝即终止的后续修复分支；它需要与 #107 最新 HEAD 对齐，不能假设已经合并。
- 主要实现：
  - src/mechet/route_process_reward.py
  - src/mechet/route_native_observation.py
  - docs/audits/native_process_grpo_20261115/source/src/mechet/route_policy_episode.py
  - docs/audits/native_process_grpo_20261115/source/src/mechet/external_route_rl.py
  - 对应的训练入口、行为记录、恢复和测试。

必须保留：

1. Stage-II 0.8B SFT Policy 初始化和冻结 Reference。
2. 原有 Route Environment、Executor、IMPORT/FLOW/FINISH 动作协议。
3. 原有多步 AND/OR 前体义务、参考反应边、库存闭合和路线终局评价。
4. 原有 Process Reward 的可信参考进展、终局路线奖励、有界 validator cost。
5. 原有 G4 独立路线采样、Action-level Return-to-Go 和 Dr.GRPO clipped token loss。
6. ReFIT 作为分支采样/轨迹组织规范。**ReFIT 不负责化学打分，不负责定义 Reward，不负责 Policy Update。**
7. RetroChimera 作为逆向反应知识来源。FlowER 延后。
8. 现有固定 VALID 人群、采样预算和评价规则；不得通过改变 VALID 或预算制造收益。

不要引入 DPO、Critic、PRM、大型 Reward Model、Hierarchical RL 或新的训练框架。不要重新训练 Stage-II 基础模型。

---

## 2. 四个职责必须严格分离

| 组件 | 职责 | 不负责 |
|---|---|---|
| ReFIT | 组织同一状态下的候选动作分支、后续 rollout 和轨迹记录 | 不定义化学正确性、Reward 或策略梯度 |
| RetroChimera | 提供反应模板、原子定位、Graph Edit、前体候选等逆向证据 | 不决定 ReFIT 如何采样 |
| 多步路线数据 + Executor | 提供参考反应边、真实状态转移、库存闭合和路线结果 | 不替代 Student Policy |
| ReFIT-PO / Process GRPO | 使用轨迹收益和同状态分支比较更新 Student | 不替代 ReFIT 搜索协议 |

RetroChimera 的分支证据可以作为有界的辅助过程信号或已可靠对齐的 OPD 监督；缺失证据必须 mask。**RetroChimera 没有覆盖某个分支，不等于该分支化学错误。**

FlowER 不属于本轮实施范围。未来接入时，应作为独立的正向机理证据源，不改变 ReFIT-PO 的基本定义。

---

## 3. ReFIT-PO 的核心对象：精确决策状态与兄弟 FLOW 分支

### 3.1 决策节点定义

同状态不是“canonical SMILES 一样”这么宽松。只有模型实际条件相同，才允许组成一个 branch group。

建议节点身份至少绑定：

    decision_node_key = hash(
        target_id,
        tree_id,
        reaction_occurrence_id,
        exact_prompt_token_ids,
        tool_schema_hash,
        local_action_history_hash,
        route_context_hash,
        behavior_policy_version,
        executor_contract_version,
    )

其中：

- 必须是同一个 target 和 tree；
- 必须是同一个 reaction occurrence；
- 必须有相同的完整 Policy Observation / Prompt token IDs；
- 必须有相同的局部 accepted-action history；
- 必须有相同的路线上下文和工具定义；
- 必须来自同一个 behavior policy version；
- Atom Mapping 必须保留并正确规范化，不能仅凭原子编号相同判断状态相同。

不同局部历史、不同 precursor occurrence、不同路线义务、不同 Prompt 或不同 Policy Version，不得合并为一个 sibling group。

### 3.2 第一版只对 FLOW 决策启用条件分支比较

对父节点 v，收集从完全相同状态独立采样得到的语义不同的 FLOW 动作：

\[
\mathcal B_{\mathrm{FLOW}}(v)=
\{a_{v,1},\ldots,a_{v,K_v}\}
\]

第一版只对 apply_electron_flow 建立 ReFIT-PO sibling group。IMPORT、FINISH 以及没有足够 FLOW 兄弟分支的动作继续使用原有 G4 Action-level Return-to-Go。

语义重复的动作不能被伪装成不同的化学分支。需要基于当前状态、原子映射和 Executor 解析结果做语义去重；重复采样可保留为 rollout 重复次数，但不得让同一个语义动作因为文字格式不同而在 branch baseline 中获得额外权重。

每个 sibling action 都必须保存：
- request / action ID；
- parent node ID；
- branch ID；
- occurrence ID；
- 完整 prompt 与生成 token IDs；
- 每个生成 token 的真实 behavior log-probability；
- action-level sampling probability（若可用）；
- Executor receipt；
- state-before / state-after；
- action semantic key；
- policy version。

### 3.3 ReFIT 采样与 on-policy 概率

ReFIT 是采样规范，不得把 PUCT 分数、节点访问次数或确定性搜索选择误当成行为策略概率。

用于 GRPO Policy Loss 的 sibling actions 必须来自已记录的 behavior policy sampling distribution。保存实际生成概率和 token log-probability。若某些搜索分支由确定性排序或未知 proposal distribution 选出，不能直接当作 on-policy 样本；应留作搜索分析，或在有真实 proposal probability 时进行明确的重要性修正。

第一版优先使用同一 behavior policy、同一解码合同，在同一父节点独立采样候选 FLOW。ReFIT 的节点选择/预算管理保持原样，不因 ReFIT-PO 改写采样规范。

---

## 4. FLOW 失败必须产生负向信用

这是本次必须实现的行为，不是可选诊断。

### 4.1 哪些情况属于失败

失败必须根据 Executor receipt、当前 occurrence 的有效 FLOW 历史和精确状态判断，至少分别识别：

- **Invalid FLOW**：Executor 明确拒绝非法 source/sink、BE transition、bond/charge change、atom reference 或其他化学无效操作。
- **Premature FINISH**：当前 reaction occurrence 尚无任何 accepted、productive、state-changing FLOW，就提交 FINISH。
- **Repeated FINISH**：同一 occurrence、同一精确状态下重复提出已拒绝的相同语义 FINISH。
- **Repeated failed FLOW**：同一 occurrence/state 下重复提出已拒绝的相同语义 FLOW。
- **Other non-resource rejection**：无效 IMPORT、格式错误及其他由环境/Executor 明确拒绝的动作。
- **Verified non-progressive FLOW**：只有 Executor contract 明确禁止 no-op，且独立状态重建确认没有任何允许的 BE/charge/graph transition 时才可处罚；不得仅凭 canonical SMILES 相同判定。

一次有效 FLOW 后 FINISH 可以合法；不得把只需一个 elementary FLOW 的反应误判为 premature FINISH。成功 FINISH 会创建新的 precursor occurrence，新 occurrence 按自己的 FLOW 历史判断。统计必须分开报告上述类别。

### 4.2 哪些情况不是化学失败

以下情况必须标记为 censored / unknown，不发负奖励：

- CUDA OOM、GPU 不可用；
- 资源估算超限；
- context/token 容量删失；
- worker / scheduler / storage 故障；
- 超时或基础设施错误；
- 无法判断是资源故障还是 Executor 化学拒绝的异常。

分类不明确时 fail closed：标记为 unknown/censored，不要猜测为化学错误。

### 4.3 单条 rollout 首次拒绝即终止；ReFIT 父节点仍可重新展开

1. 保留失败动作的原始生成文本、token IDs、behavior log-probs、request ID 和真实 Executor error。
2. 将失败动作 edge 标记为 `REJECTED_TERMINAL`；Executor 未改变状态时，不创建虚假 child chemical state。
3. 当前 sampled rollout 在首次非资源性拒绝处终止，不在同一个 Episode 中追加失败恢复 Prompt 或连续 RETRY。
4. 将失败语义动作 key 写入 parent node 的 rejected-action ledger。
5. **Parent chemical state 不自动变成 FAILED。** 尚有未探索动作时标记 `OPEN_WITH_REJECTIONS`。
6. ReFIT 可回到失败 edge 的 parent node，从相同保存状态/Observation 独立采样另一个动作；若 parent 已耗尽，则回溯到最近 OPEN ancestor。
7. 新分支复用已接受的共享前缀，只生成新的 suffix；共享前缀 action receipt 不得复制进入 Policy Loss。
8. 如果模型再次采样同一已拒绝的语义动作，不重新调用 Executor；保存这次新生成的 tokens/log-probs，记录 duplicate rejected proposal，并使用现有 repeat_extra 产生有界重复成本。
9. 只有当候选动作或预设 expansion budget 耗尽且没有新动作时，才将 parent node 标记为 `EXHAUSTED`。
10. 同一树的其他 sibling branches 独立继续。

这不是 failure replay：旧失败不会作为模型可见的恢复示范，也不会在同一 rollout 中反复重试。它是 ReFIT 从保存的 parent state 发起新的 sibling sampling。失败边保留为真实负向训练样本，不能只保留成功分支。详见第 13 节的规范状态机和回溯协议。

### 4.4 失败奖励：复用现有有界 validator cost，不重复叠加

现有 Process GRPO 已经有有界 validator cost：

\[
C_t=1-\exp(-E_t/4)
\]

\[
\eta=\frac{0.05}{\max(L,1)}
\]

\[
r_t^{failure}=-\eta(C_t-C_{t-1})
\]

其中普通非资源性拒绝的首个错误权重 E 增加 1；明确的路线循环按现有合同增加 2。第一版保留现有错误分类与权重，不额外添加一个重复的 FLOW-only penalty，避免对同一次 Executor rejection 双重处罚。

对于首个普通 FLOW rejection：

\[
r_t^{failure}
=
-\frac{0.05}{\max(L,1)}
\left(1-e^{-1/4}\right)
\]

这是小额、明确、可审计的负向动作奖励。它与失败路线的 route terminal correction 共同起作用；不能仅靠放大惩罚系数解决失败循环。

### 4.5 失败 rollout 撤销未完成的部分路线进展

若一条具体 root-to-leaf rollout 在 Invalid FLOW、Premature FINISH 或其他非资源性拒绝处终止，则该条 rollout 的 route task reward 为 0：

[
B_{mathrm{failed rollout}}=0
]

此前尚未形成完整路线的 partial route progress 必须在该 rollout 的 reward ledger 中撤销：

[
r_t^{rollback}=-q_{t-1}
]

最终失败 rollout 的 reward sum 只保留有界 validator failure cost：

[
sum_u r_u=-eta C_T
]

这个 rollback 只作用于该条失败 rollout，不能回滚其他已经成功闭合的 sibling route，也不能把失败 sibling 的结果传播成整个 ReFIT tree 的失败。parent node 仍可由 ReFIT 重新展开。

### 4.6 已有离线证据（不是新训练收益）

PR #107 的 2026-11-16 首次拒绝离线反事实审计已提供以下证据：

- 前两轮 128 个 TRAIN targets × G4，共 512 条保存轨迹；
- 首次拒绝截断后，模型动作调用从 14,781 降到 9,651，减少 5,130 次（34.71%）；
- 参考核验成功轨迹保持 15/15，对应目标保持 5/5；
- 库存闭合轨迹从 186 降至 176，有 10 条闭合轨迹依赖重试；
- 原来 80 条失败但仍获得正分的轨迹均变为负分；
- 失败路线优于保留闭合候选的 G4 组从 8 降为 0；
- 这是保存轨迹截断、重评分和 CPU 前缀重放，不是新模型生成、RL 更新或性能提升实验。

Codex 应保留这个 trade-off：首错即停减少无效动作且没有丢失本批参考核验成功轨迹，但会放弃一部分依赖重试才闭合的库存路线。不要将其写成已经证明策略性能提升。

---

## 5. ReFIT-PO：分支价值与条件 Advantage

### 5.1 分支 suffix return

对于父节点 v 的 sibling FLOW action a_j，对其后续完成的 rollout m 定义：

\[
Z_{v,j}^{(m)}
=
\sum_{u=t}^{T_{v,j,m}} r_{v,j,m,u}
\]

这里从当前 FLOW 分支动作开始累计，包含：
- 当前 FLOW 动作奖励；
- 后续 IMPORT / FLOW / FINISH 奖励；
- RetroChimera 的有效、可映射辅助证据（若启用）；
- 后续 precursor occurrences 的路线进展；
- 完整路线库存闭合/终局奖励；
- 失败终止或其他有界失败成本。

失败 FLOW branch 没有后续 rollout，其 Z 就是该失败动作的负向 terminal return（若此前有未完成进展，包含撤销 correction）。

### 5.2 分支价值估计

每个 sibling action 使用相同数量 M 的独立 continuation rollouts，避免因搜索访问次数不同造成分支权重偏差：

\[
Q(v,a_j)=
\frac{1}{M}\sum_{m=1}^{M}Z_{v,j}^{(m)}
\]

第一版可以从 M=1 开始；如果预算允许，再增加相同的 M。 不允许把 PUCT visit count 直接当作 Q 的统计权重。不同 sibling 预算不相等时，必须记录真实样本数和估计不确定性；不满足预先约定的 matched-budget 条件时，不启用该 sibling group 的条件 Advantage。

### 5.3 Same-state Branch Advantage

对于 K_v≥2 个语义不同、有效且完整评价的 sibling actions：

\[
\bar Q(v)=\frac{1}{K_v}\sum_{j=1}^{K_v}Q(v,a_j)
\]

\[
\boxed{
A^{branch}(v,a_j)=Q(v,a_j)-\bar Q(v)
}
\]

保留 mean-centering，不做 reward standard-deviation scaling，不做动作长度归一化。

这个 baseline 只在精确相同的父节点内成立。它比较的是相同 Policy Conditioning 下不同 FLOW 决策的后续价值。

如果 sibling group 中任何一个分支是资源删失或评价不完整，第一版应 mask 整个 sibling group 的 conditional comparison，不能只保留成功/已完成分支再重新计算均值，以免产生 survivor-selection bias。对应的独立 G4 Episode 如果本身有效，仍可使用原 G4 Advantage。

如果一个 sibling group 只有一个语义动作、少于两个有效分支或存在状态绑定不一致，则不计算 Branch Advantage。

---

## 6. 如何与现有 G4 Action-level Process GRPO 结合

原有 G4 不替换。ReFIT-PO 只在有严格同状态 sibling group 的 FLOW 决策点，提供更合适的条件 Advantage。

对唯一的 policy decision edge e：

\[
\boxed{
\tilde A(e)=
\begin{cases}
A^{branch}(v,e), & e\text{ 是有效同状态 FLOW sibling group 的根动作}\\
A^{G4}(e), & \text{否则，且原 G4 样本有效}\\
\text{mask}, & \text{没有任何有效 Advantage 来源}
\end{cases}
}
\]

其中原有：

\[
G_{i,t}=\sum_{u=t}^{T_i}r_{i,u}
\]

\[
A_{i,t}^{G4}=G_{i,t}-\frac14\sum_{j=1}^{4}G_{j,t}
\]

**禁止把 A_branch 与 A_G4 直接相加。** 二者都包含路线未来收益，直接相加会重复计算。

规则：

1. 同状态兄弟组的分叉 FLOW 根动作使用 A_branch，替代该动作原本的 G4 Advantage。
2. 后续动作在自己的父节点若形成有效 sibling group，则使用自己的 A_branch；否则使用原 G4 Action-level Advantage。
3. 分叉前的共享前缀动作只保留一个唯一行为记录，不因有多个后续叶节点而复制 token loss。
4. 一个真实生成的 action receipt 最多进入 Policy Loss 一次。
5. 不允许将共享前缀的同一生成 token 复制到多个 root-to-leaf paths 后重复更新。
6. 共享前缀的未来价值可以用于诊断或节点价值估计，但不能伪装成多次独立的行为采样。
7. 同一 sibling group 的不同 action 必须来自同一个 behavior policy version。

### 6.1 Policy loss

保持现有 token-level clipped GRPO/Dr.GRPO loss：

\[
\rho_{e,k}=
\frac{\pi_\theta(y_{e,k}\mid o_e,y_{e,<k})}
{\pi_{\mathrm{old}}(y_{e,k}\mid o_e,y_{e,<k})}
\]

\[
\mathcal L_{PG}
=
-\frac{1}{C}
\sum_{e,k}m_{e,k}
\min\left[
\rho_{e,k}\tilde A(e),
\operatorname{clip}(\rho_{e,k},1-\epsilon,1+\epsilon)\tilde A(e)
\right]
\]

同一个动作的所有有效生成 token 使用该动作的同一个 Advantage。Prompt tokens、Executor outputs 和 tool results 不参与 policy loss。

保留现有：
- clip epsilon；
- frozen Stage-II reference KL；
- fixed loss normalizer；
- token masks；
- behavior log-probability receipts；
- G4 whole-group censoring；
- optimizer/checkpoint/resume 语义。

不要因为 ReFIT-PO 引入 sequence-level reward 就改成 sequence-level loss 或 GSPO。

### 6.2 行为概率约束

Sibling actions 必须有真实的行为策略来源。每个 action 的完整生成 token log-probabilities 必须保留。

如果 ReFIT 的某个分支来自确定性排序、PUCT 选择或其他 proposal distribution，而不是已记录的 behavior policy sampling：
- 不得假装其行为分布等于 π_old；
- 没有真实 proposal probability 时，不得将其直接送入 on-policy policy loss；
- 可保留为搜索/评估数据；
- 只有在实际 proposal probability 可恢复且重要性修正定义明确时，才另行支持。

第一版优先使用同一 behavior policy 在同一父节点独立采样 FLOW siblings，避免引入额外 off-policy correction。

---

## 7. RetroChimera 的范围；FlowER 明确延期

### 7.1 本轮保留 RetroChimera

RetroChimera 可为同一个 branch 提供：
- template classification；
- atom localization；
- Reaction SMARTS / graph edits；
- precursor candidate support；
- ensemble ranking。

只对能够通过 atom mapping、当前 BE state 和实际 Executor transition 可靠对齐的证据计算辅助分支分数。

建议将其写成有界的 branch evidence potential：

\[
\Phi_R(b_t)
\]

\[
r_t^R=\lambda_R[\Phi_R(b_t)-\Phi_R(b_{t-1})]
\]

要求：
- 只根据实际 branch state / Executor transition 更新；
- 对同一 template/edit/center evidence 去重，防止重复领取；
- 缺少 teacher coverage 时 mask，不记负；
- 原始分数和校准/归一化结果分开存储；
- RetroChimera 证据不能压过完整路线的真实结果；
- 不把原始 ensemble softmax 当作校准的化学正确概率。

如果当前代码还没有可靠的 branch-to-template mapping，先实现证据存储与审计，不得为了完成任务而伪造动作级 teacher labels。

### 7.2 FlowER 本轮不做

不实现：
- FlowER worker；
- FlowER async queue；
- FlowER model loading；
- forward endpoint recovery reward；
- forward BE trajectory reward；
- FlowER OPD。

Evidence schema 可以预留 teacher source/version 字段，但本轮不得引入 FlowER 依赖或阻塞训练。

### 7.3 最终 Reward

保留现有 route reward 和 bounded validator failure cost。增加 ReFIT-PO 条件 Advantage 不等于增加新的 Reward 类型。

本轮动作奖励结构：

\[
r_{i,t}=
r_{i,t}^{route}
+\lambda_R\Delta\Phi_{i,t}^{R}
+\lambda_I\Delta\Phi_{i,t}^{import}
-r_{i,t}^{failure}
\]

其中：
- route reward 继续使用现有可信参考进展和终局路线合同；
- RetroChimera 仅使用已可靠对齐的证据；
- IMPORT 仅在片段的 productive contribution 被后续实际反应确认后回填；
- Executor rejection 使用现有有界 failure cost；
- FLOW acceptance 本身不产生正奖励；
- Teacher unsupported 不自动产生负奖励；
- FlowER 暂不参与。

不要在本 PR 中凭空指定新的 reward 权重。优先复用现有合同；新增 RetroChimera 项必须先给出分布审计、缺失率、覆盖率和与 route reward 的重复性分析。

---

## 8. 机器可读 Branch Evidence Schema

每个唯一 decision edge 至少保存：

    {
        "tree_id": "...",
        "target_id": "...",
        "route_episode_id": "...",
        "decision_node_id": "...",
        "parent_node_id": "...",
        "branch_id": "...",
        "reaction_occurrence_id": "...",

        "state_hash": "...",
        "observation_hash": "...",
        "prompt_token_ids_hash": "...",
        "local_history_hash": "...",
        "route_context_hash": "...",
        "tool_schema_hash": "...",

        "behavior_policy_version": "...",
        "executor_contract_version": "...",

        "action_type": "apply_electron_flow",
        "semantic_action_key": "...",
        "generated_token_ids": [...],
        "behavior_logprobs": [...],
        "action_logprob": ...,

        "accepted": True,
        "failure_class": None,
        "resource_censored": False,

        "state_before_hash": "...",
        "state_after_hash": "...",
        "verified_transition": {...},

        "retro_evidence": {...},
        "route_reward_events": [...],
        "failure_reward": 0.0,

        "suffix_rollout_count": 1,
        "branch_value": None,
        "branch_advantage": None,
        "advantage_source": None
    }

保存 branch evidence 时必须有完整 SHA / contract binding，恢复时验证：
- parent node 是否存在；
- edge 是否属于该 parent；
- prompt/state/history hash 是否一致；
- policy version 是否一致；
- 行为 token 与 log-probability 长度是否一致；
- 每个 unique edge 是否只进入 policy loss 一次；
- teacher evidence 是否绑定到正确 occurrence。

---

## 9. Codex 实施顺序

### Task 0 — 代码审计，不先改代码

从最新 PR #107 HEAD 和 #108 分支开始检查：
- 当前 ProcessRouteReward；
- 首次拒绝即停的 Episode 行为；
- route failure reward 是否真正撤销之前的 partial progress；
- 当前 optimizer 的 action/token sequence binding；
- ReFIT 当前采样输出的真实数据结构；
- 是否已有可复用的 tree/branch/node ID；
- 共享前缀和行为概率保存位置。

不得假设 #108 已合并；先基于最新 #107 代码核对并整合，避免重复实现或覆盖新提交。

### Task 1 — 失败类型识别、负向信用与 ReFIT 回溯

实现并测试：
- 非资源性 FLOW rejection 立即终止当前 sampled rollout；
- 无有效、状态改变 FLOW 就 FINISH 时识别并惩罚 premature FINISH；
- 同一 occurrence/state 下重复 FINISH 或重复已拒绝语义动作，使用现有 repeat_extra；
- failed action 保留 token / behavior receipt；
- 失败 rollout 的 route task reward=0，撤销该 rollout 此前未完成路线的 partial progress；
- 复用现有 bounded validator cost，不重复叠加 FLOW-specific penalty；
- 资源删失 reward=None，不标记为化学失败；
- 失败 edge 终止，但 parent node 仍可 OPEN_WITH_REJECTIONS；
- ReFIT 可回到 parent 或最近 OPEN ancestor 采样其他动作；
- 同语义失败动作不重新执行 Executor；duplicate proposal 保留 token/log-prob、有界计费；
- 兄弟分支独立继续。

### Task 2 — 同状态节点识别与失败状态机

实现：
- exact decision context hash；
- parent/child/branch ID；
- semantic action key；
- policy version binding；
- shared prefix 唯一记录；
- OPEN、OPEN_WITH_REJECTIONS、EXHAUSTED、SOLVED、EXPANDED、REJECTED_TERMINAL、CENSORED 状态；
- rejected semantic-action ledger；
- backtrack target 选择（失败动作 parent 优先，其次最近 OPEN ancestor）；
- 有界 duplicate proposal / expansion ceiling；
- 不能把相同 canonical SMILES 但不同 history/context 合并。

### Task 3 — Branch suffix return 和 conditional Advantage

实现：
- 每个 sibling action 的 matched continuation rollout；
- branch value Q(v,a)；
- mean-centered A_branch；
- resource-censored sibling group 整组 mask；
- 语义重复动作去重；
- 无有效 sibling group 时回退原 G4；
- branch Advantage 替代而非叠加 G4 Advantage；
- unique edge token loss，不重复训练共享前缀。

### Task 4 — 接入现有 Process GRPO

只修改必要的数据结构和 Advantage 选择逻辑：
- 不改 optimizer 框架；
- 不改模型结构；
- 不改 reference policy；
- 不改固定 VALID；
- 不改多步路线数据集；
- 不改 ReFIT 的采样规范；
- 保留完整行为概率和 checkpoint/resume 绑定。

### Task 5 — RetroChimera evidence

优先复用现有部署模型和缓存。
只有可靠对齐的 branch evidence 才能进入辅助 reward / OPD。
没有映射证据时记录 unavailable，不制造标签。
FlowER 完全延期。

### Task 6 — 报告

输出：
- docs/results/REFIT_PO_DESIGN_IMPLEMENTATION_REPORT.md
- 机器可读的 branch statistics JSON
- branch group 数量、有效兄弟分支数、平均分支数；
- FLOW rejection 次数和比例；
- failed branch / accepted branch 的 return 分布；
- 同状态 branch advantage 分布；
- teacher coverage / missingness；
- route closure / reference-supported route；
- unique policy edges 与 token 数；
- shared-prefix 去重统计；
- resource-censored group 数；
- 与原 G4 Process GRPO 的差异。

不以非零 Advantage 数量或教师分数上涨代替 VALID 性能提升。

---

## 10. 必须新增的针对性测试

### Same-state grouping
1. 相同 prompt/state/history/route context/policy version 可以组成 sibling group。
2. 相同 canonical molecule 但不同 local history 不可组成同一组。
3. 不同 precursor occurrence 不可合并。
4. 不同 policy version 不可合并。
5. 同一语义 FLOW 的不同文字格式不能被当成不同化学分支。

### Branch return / Advantage
6. K=2 的手工 Q 值验证 branch advantage。
7. 所有 sibling Q 相同则 advantage 全零。
8. 一个 failed FLOW sibling 得到负 terminal return，并参与 sibling mean。
9. 资源删失 sibling 导致整个 conditional group mask，不得只对幸存分支重算均值。
10. continuation budget 不等时拒绝未声明的比较。
11. 同一个 unique edge 只能产生一次 policy-loss entry。
12. 共享前缀不因多个叶节点重复进入 loss。

### Failure semantics
13. 非资源性 FLOW rejection 立即终止当前 branch。
14. failed FLOW token / behavior log-probs 仍被保留。
15. 失败路线的 route task reward 为零。
16. 先前 partial route progress 被撤销，最终总 reward 只剩有界 failure cost。
17. 失败 sibling 不终止其他 sibling。
18. RESOURCE_CENSORED 不得获得负 reward。
19. Executor 接受 FLOW 不自动产生正 reward。
20. Teacher unsupported 不自动判为负。

### Policy loss / behavior provenance
21. branch Advantage 应用到该 FLOW action 的有效生成 token。
22. prompt 和 tool output 不进入 policy loss。
23. 共享 token 不重复计算。
24. behavior logprob 数量与生成 token 一致。
25. 未知 proposal probability 的确定性搜索分支不得伪装成 on-policy 样本。
26. 冻结 reference 无梯度。
27. 现有 G4 fallback 与原实现数值一致。

只运行与本次改动直接相关的本地测试和固定运行环境测试。**不要花时间处理无关 GitHub CI 失败；CI 状态不是本任务的研究阻塞条件。**

---

## 11. 固定的非目标与安全边界

本任务不做：

- FlowER 集成；
- DPO；
- Critic / PRM；
- Hierarchical RL；
- 新的 PPO/GRPO 训练框架；
- 新的 ReFIT 采样算法；
- 用 PUCT visit count 伪造 policy probability；
- 把 ReFIT 变成 Reward；
- 把 RetroChimera / FlowER 变成 Student；
- 重新训练 Stage-II；
- 修改正在运行的正式任务；
- 擅自启动新正式训练；
- 缩小 TRAIN / VALID；
- 修改 VALID targets、解码参数、G4、seed 或预算；
- 将库存闭合直接称为化学正确；
- 将 Teacher unsupported 直接判负；
- 将资源删失当作化学失败。

Codex 可以实现代码、测试、文档和离线回放，但不得提交或启动新的正式训练，除非用户另行明确授权。

---

## 12. 最终验收标准

本任务完成时，必须能够回答并通过测试：

1. 给定一个完全相同的电子化学决策状态，系统能否识别其不同 FLOW 兄弟分支？
2. 每个分支是否有独立、可追溯的后续路线收益？
3. FLOW 被 Executor 拒绝时，是否立即终止该分支并获得负向信用？
4. 失败分支是否与成功兄弟分支共同参与同状态条件比较？
5. 资源删失是否被正确 mask，而不是被误判为化学失败？
6. 完整路线结果是否能回传到真正产生分支差异的 FLOW 动作？
7. 共享前缀是否只训练一次？
8. 是否保持原有 G4 Process GRPO 作为 fallback？
9. 是否严格使用真实 behavior log-probabilities？
10. 是否没有把 FlowER、DPO、Critic 或新 RL 框架提前引入？

最终方法的核心定义：

\[
\boxed{
\text{ReFIT-PO}
=
\text{Existing Process GRPO}
+
\text{Same-state FLOW Branch-relative Advantage}
+
\text{Fail-fast Bounded FLOW Failure Cost}
}
\]

它要解决的问题是：

**在同一个化学状态下，比较不同逆向电子流决策的真实后续路线价值，并明确惩罚 Executor 拒绝的无效 FLOW，从而让长期合成结果更精确地作用于电子流层级的策略决策。**


---

## 13. 2026-11-16 clarification — Failure-aware ReFIT backtracking and FINISH penalties (normative)

本节是对前文“失败分支立即终止”“不允许重复尝试”等表述的精确定义；若前文存在歧义，以本节为准。

**Fail-fast applies to one sampled rollout, not to the parent chemical state or the whole ReFIT tree.** 一个被拒绝的动作是失败 edge，不意味着它的 parent state 没有其他合法后继。

### 13.1 必须处罚的动作类型

本轮失败惩罚不局限于 FLOW rejection。至少区分：

1. **Invalid FLOW**：Executor 明确拒绝 source/sink、BE transition、bond/charge change、atom reference 或其他化学非法操作。
2. **Premature FINISH**：当前 reaction occurrence 尚无任何 accepted、productive、state-changing FLOW，就提交 FINISH。
3. **Repeated FINISH**：同一 reaction occurrence、同一精确状态下，重复提出此前已拒绝的相同语义 FINISH。
4. **Repeated failed FLOW**：同一 occurrence/state 下再次提出此前已拒绝的相同语义 FLOW。
5. **Other non-resource rejection**：无效 IMPORT、格式错误、过早 FINISH 等其他由环境/Executor 明确拒绝的动作。
6. **Verified non-progressive FLOW**：只有在 Executor contract 明确禁止 no-op，且状态重建确认没有任何允许的 BE/charge/graph transition 时才可处罚；不能仅凭 canonical SMILES 相同判定。

**合法的一次 FLOW 后 FINISH 不受惩罚。** 这里的“一步就 FINISH”专指没有任何有效 FLOW 就立即 FINISH，不是只需要一个 elementary FLOW 的合法反应。

若 FINISH 在新 occurrence 中发生，必须根据新 occurrence 自己的 FLOW 历史判断，不能因前一个 occurrence 已经 FINISH 而处罚。

### 13.2 复用现有有界 validator cost，不叠加重复 penalty

现有 `route_process_reward.py` CONFIG 已包含普通错误、重复错误和循环错误的权重规则。Codex 必须先核对最新代码及 PR #108 的合同，再决定如何映射，不得自行发明另一套惩罚系数。

- 首次 Invalid FLOW：按普通非资源性拒绝计一次成本。
- 首次 Premature FINISH：按普通非资源性拒绝计一次成本。
- 同一状态/occurrence/semantic action 的重复 FINISH 或重复 FLOW：使用现有 `repeat_extra` 增量。
- 明确循环：使用现有 `cycle_cost`。
- 不额外再加一个 FLOW-specific penalty，以免同一拒绝被重复扣分。

如果当前环境会接受“没有有效 FLOW 的 FINISH”，必须增加环境侧 FINISH precondition guard，生成可审计的 synthetic rejection receipt；保留模型真实生成的 FINISH token 和 behavior log-prob。若 Executor 已经拒绝，则复用其真实 receipt，不能重复处罚。

### 13.3 失败 edge 与 parent node 的状态必须分开

建议的状态机：

| 对象 | 状态 | 含义 |
|---|---|---|
| Decision node | `OPEN` | 尚有可探索动作 |
| Decision node | `OPEN_WITH_REJECTIONS` | 已有拒绝动作，但仍可探索其他动作 |
| Decision node | `EXHAUSTED` | 候选动作/展开预算耗尽 |
| Decision node | `SOLVED` | 找到满足当前路线目标的有效后续 |
| Action edge | `EXPANDED` | Executor 接受并产生真实 child state |
| Action edge | `REJECTED_TERMINAL` | 非资源性拒绝，只终止这个动作分支 |
| Action edge | `CENSORED` | 资源/基础设施删失，不是化学失败 |

每个 parent node 保存：

```python
node.rejected_actions[semantic_action_key] = {
    "failure_class": "INVALID_FLOW | PREMATURE_FINISH | REPEATED_FINISH | ...",
    "failure_receipt_id": "...",
    "failure_cost_weight": 1.0,
    "repeat_count": 0,
}
```

拒绝动作没有改变状态，因此：
- 失败 edge 设为 `REJECTED_TERMINAL`，不创建虚假的 child chemical state；
- parent node 标记为 `OPEN_WITH_REJECTIONS`，而不是全局 FAILED；
- ReFIT 优先回到该 parent node，使用完全相同的保存状态和 Policy Observation 采样其他候选；
- 如果该 parent 已无未探索动作，则回溯到最近的 `OPEN` / `OPEN_WITH_REJECTIONS` ancestor；
- 只有候选空间或预设 expansion budget 耗尽，才标记 `EXHAUSTED`。

### 13.4 “重新采样”与“失败重试”必须严格区分

**允许：**
- 一条 rollout 在失败动作处结束；
- ReFIT 把失败动作记录到 parent node 的 rejected-action ledger；
- 从 parent 或最近 OPEN ancestor 建立一条新的 sibling rollout；
- 新 rollout 复用已接受的共享前缀，只生成新的 suffix；
- 同一树中的其他 sibling 独立继续。

**不允许：**
- 在同一 rollout 里把失败 assistant/tool exchange 拼入下一轮 Prompt 让模型修复；
- 无上限地在同一状态重复生成同一个失败动作；
- 将失败轨迹放入成功 replay buffer；
- 将一个 failed edge 等同于 parent chemical state failed；
- 因一个 sibling 失败而终止其他 sibling。

如果策略再次采样同一个已拒绝的 semantic action：
- 保留这次新生成的 token/log-prob；
- 记录 `DUPLICATE_REJECTED_PROPOSAL`，关联此前失败 receipt；
- 不再次调用 Executor；
- 按现有 `repeat_extra` 增加一次有界重复成本；
- 达到有限 duplicate-proposal ceiling 后停止该节点的重复采样，并在没有其他候选时标记 `EXHAUSTED`。

这不是 failure replay buffer：模型没有看到旧失败作为恢复示范；这是新采样动作被环境依据失败动作账本拒绝。

### 13.5 失败 rollout 的 route reward 与其他 sibling 隔离

对某条具体失败 rollout：
- terminal route task reward 为 0；
- 撤回该 rollout 中此前尚未完成完整路线的 partial route progress；
- 保留失败动作的 bounded validator cost；
- 因此该失败 rollout 的 reward sum 只保留有界失败成本：
  \[
  \sum_t r_t=-\eta C_T
  \]

这个 rollback 只作用于该条失败 rollout 的 reward ledger。不能回滚其他已经成功闭合的 sibling route，也不能把失败 sibling 的 route failure 传播成整棵 ReFIT tree 的失败。

资源删失标记为 `CENSORED`：不施加化学负奖励、不将其当作负 branch Q；如需恢复，只能按资源恢复协议处理，不伪装为化学重新采样。

### 13.6 对 ReFIT-PO 条件 Advantage 的影响

在同一个 parent node 下：
- 一个有效完成的 sibling 有真实 suffix route return；
- 一个 Invalid FLOW / Premature FINISH sibling 有负向 terminal return；
- 一个 repeated failed proposal 有现有 repeat_extra 对应的负向 terminal return；
- 一个资源删失 sibling 是 unknown，不能当成负值。

若 sibling group 含资源删失或其他未完成、不可比较的分支，第一版整组 mask conditional Advantage，回退到可用的独立 G4 Process GRPO；不得只保留幸存分支后重算均值。

同一 parent node 的失败动作和成功动作可以参与同状态比较；但共享前缀仍然是唯一行为记录，只能进入 Policy Loss 一次。

### 13.7 新增必须报告的失败/回溯统计

至少分别统计：
- Invalid FLOW count/rate；
- Premature FINISH count/rate；
- Repeated FINISH count/rate；
- Repeated failed FLOW count/rate；
- Other non-resource rejection；
- Verified non-progressive FLOW（若该类别在当前 Executor contract 下成立）；
- resource-censored branch；
- rejected edges per parent；
- parent nodes marked OPEN_WITH_REJECTIONS；
- parent re-expansion count；
- backtracks to immediate parent；
- backtracks to earlier OPEN ancestor；
- exhausted nodes；
- duplicate proposals blocked from Executor；
- failure followed by successful alternative sibling；
- failure penalty sum and rollback amount。

不能把这些全部合并成一个 failure rate。

### 13.8 Additional required tests for FINISH guards and ReFIT backtracking

1. FINISH as the first action in a new reaction occurrence is rejected as `PREMATURE_FINISH`, retains the sampled FINISH tokens/log-probs, and incurs exactly one ordinary bounded validator cost.
2. One accepted productive state-changing FLOW followed by FINISH is accepted; do not impose a two-FLOW minimum.
3. A successful FINISH creates a new occurrence; a FINISH in that new occurrence is not classified as repeated solely because the tool name is unchanged.
4. A repeated semantic FINISH at the same occurrence/state is classified as `REPEATED_FINISH` and uses the existing `repeat_extra`, without calling Executor a second time.
5. A repeated semantic failed FLOW at the same occurrence/state is classified as repeated failure and uses the existing `repeat_extra`, without re-executing Executor.
6. A failed action marks its edge `REJECTED_TERMINAL` and parent `OPEN_WITH_REJECTIONS`; it does not mark the parent `FAILED`.
7. ReFIT returns to the immediate parent and samples an alternative action; the alternative can be accepted and expanded.
8. If the immediate parent has no remaining candidates, ReFIT backtracks to the nearest OPEN ancestor.
9. A node becomes `EXHAUSTED` only after all known semantic candidates / the declared expansion budget are exhausted.
10. A failed sibling does not terminate or roll back a successful sibling's route reward.
11. A failed rollout rolls back only its own unfinished partial progress, including its own copy of the shared prefix reward ledger; it must not mutate another sibling's ledger.
12. Re-sampling an already rejected semantic action creates a duplicate-proposal event with the new behavior tokens/log-probs, applies bounded repeat cost, and does not invoke Executor again.
13. Duplicate proposal sampling has a finite ceiling; it cannot create an infinite loop.
14. A resource-censored edge is `CENSORED`, not `REJECTED_TERMINAL`, and does not close the parent node as chemically exhausted.
15. A failure followed by a valid alternative sibling is represented as one failed edge plus one expanded edge under the same parent.
16. If the configured behavior policy is deterministic and cannot produce distinct sibling actions, Codex must report this as an interface/coverage limitation. It must not silently change temperature/top-p or fabricate proposal probabilities; any permitted sampling change requires an explicit separately frozen behavior-policy contract.

---

## 14. Penalty scope and causal credit propagation (normative follow-up)

**本节补充第 13 节：局部失败成本、失败 rollout 的路线收益、同状态兄弟分支的价值备份是三种不同量。不得混为“全路径都被处罚”。**

### 14.1 三个层级，三种不同的处理

| 层级 | 对象 | 规则 |
|---|---|---|
| Local action / edge | 真实失败的 IMPORT、FLOW、FINISH 或格式错误 | 仅该失败 action edge 记录一次 validator cost；保留全部 generated tokens/logprobs |
| Sampled rollout | 包含失败 edge 的这一条 root-to-leaf continuation | 该条 rollout 终止；未解决库存义务则完整路线任务收益为 0；用终局 correction 撤销 provisional route progress |
| ReFIT parent/tree | 共享的有效化学状态、共享前缀、其他兄弟分支 | 不施加失败标签，不回滚共享状态，不提前关闭 parent，不给兄弟分支扣分；从 parent 或其他 OPEN ancestor 继续展开 |

不因为一个后继动作失败，就将此前通过 Executor 的 IMPORT/FLOW/FINISH 重新标成化学非法。只有当后续证据能够直接证明此前的动作本身非法，才能对该动作建立独立错误证据。

### 14.2 数学定义：局部 penalty 与终局 outcome 不同

令该 rollout 在 t_f 的最后一个动作被明确非资源性拒绝；q_t 是 PR107 的参考进展势，c_t 是原有 validator cost 增量。

对沿途动作定义 route-process ledger：

\[
r_t^{route}=(q_t-q_{t-1})-\eta(C_t-C_{t-1})
+\mathbb 1[t=t_f](B-q_t).
\]

在失败 rollout 上设 B=0，且拒绝的最后动作不增加 q_t，于是：

\[
r_{t_f}^{route}=-\eta\Delta C_{t_f}-q_{t_f-1},
\qquad
\sum_{t\le t_f}r_t^{route}=-\eta C_{t_f}.
\]

这里的 \(-q_{t_f-1}\) 是**终局进展冲销**，不是把过去每条已接受边再逐一罚一遍。失败 rollout 的总收益是负向的；而此前有效动作的局部 chemistry receipt 仍然有效。

如果存在尚未兑现的机制/IMPORT 辅助 shaping，必须为每一种新增分量声明 failure-terminal treatment：不得让已知失败 rollout 通过 provisional auxiliary rewards 获得虚假的净正收益；也不得在每个历史动作额外乘一遍失败惩罚。独立 verified local chemical facts 仍可保留为诊断，具体是否进入最终 reward 应以冻结的显式合同定义。

### 14.3 对先前动作的 policy-gradient 信用如何传播？

**局部错误惩罚只记在失败 edge；失败路径的最终低 return 会通过 Return-to-Go 影响该条 rollout 中较早动作的 Advantage。** 这是 RL 的自然长期信用传播，不等于先前每个动作被判定非法。

\[
G_t=\sum_{u=t}^{t_f}r_u,\qquad
A_t=G_t-b(s_t).
\]

对于固定已接受前缀共享的其他成功 sibling，它们有各自的完整 suffix return；它们的 reward ledger 不受失败 sibling 修改。共享前缀只对应一次真实动作 receipt，不允许通过复制失败/成功叶节点的前缀重复计算 policy loss。

若较早状态存在多个合法后续，不能只凭一个失败 continuation 将该状态判定为零价值；正确的 parent value 是其合法后续 rollout 的期望收益。若只有一个失败样本，报告估计不确定性，不声称它证明此前整个路径错误。

### 14.4 FINISH 惩罚边界与实际 Executor

FINISH 判断是 **per occurrence** 的：

1. **没有任何被接受的有效 state-changing FLOW 就请求 FINISH**：明确 nonproductive/premature，按一次普通拒绝处罚；这与仓库已有的 NONPRODUCTIVE_FINISH guard 一致。
2. **已有一次有效 FLOW 且形成非平凡、合法、非循环的前体**：可以 FINISH；不设置“至少两个 FLOW”门槛。
3. **虽有 FLOW 但 precursor endpoint 没有净变化**：已有 NONPRODUCTIVE_FINISH / endpoint guard 拒绝；不可仅因工具调用次数大于零就放行。
4. **先前被拒绝的 FINISH 在同一 occurrence、相同 Executor state 和语义下再次被独立采样**：标记 REPEATED_FINISH；保存新行为 token 作为错误提议，用 rejection ledger 阻止重复 Executor 调用，并按现有有界 repeat cost；不是在原失败 rollout 内附反馈继续生成。
5. **成功 FINISH 后进入新 occurrence**：计数重置。新 occurrence 第一个 FINISH 仍需检查该 occurrence 自己的有效机理状态，不因其他 occurrence 先前结束而放行。
6. **目标已在库存**：环境自动闭合，不应产生模型 FINISH 处罚事件。

禁止用“重复了 finish 这个工具名”作为错误条件；必须检查同一精确状态、同一 reaction occurrence 和此前拒绝的语义动作。

### 14.5 Backtracking 是搜索控制，不是 failure-repair rollout

ReFIT 接收到 rejected edge：

    record_failure(edge, error, token_logprobs)
    mark(edge, REJECTED_TERMINAL)
    mark(parent, OPEN_WITH_REJECTIONS)
    # Reject the sampled continuation; preserve the valid parent state.
    next_parent = parent if has_untried_budget(parent) else nearest_open_ancestor(parent)
    expand_new_sibling(next_parent, frozen_behavior_policy)

不要在同一 rollout 内通过失败反馈修复并继续采样；另开 sibling continuation，复用已验证的祖先状态与已接受前缀。重复候选可以作为新采样的错误提议计入统计和有界成本，但绝不能把拒绝动作从策略训练证据中静默删除。

如使用 action masks、reject-sampling、去重后的强制不同候选或 PUCT 排序，动作实际 proposal distribution 可能改变。Codex 必须验证 behavior provenance：若无法计算真实 proposal probability，就只允许搜索评估/诊断，不能直接按标准 on-policy GRPO 使用伪造的 token denominator。第一版应优先同一 frozen behavior policy 的真实独立采样，并显式记录被拒绝、重复或未选择的候选。

### 14.6 Codex additional acceptance tests

- One failed FLOW creates exactly one local rejection cost, one failed rollout terminal correction, and does not mutate valid ancestor receipts.
- A failed suffix gives low/negative return to the specific sampled path, but successful siblings keep their original route returns.
- For a parent with one failed child and one successful child, branch Q/Advantage reflects both; parent is not FAILED.
- No FLOW then FINISH is penalized; exactly one productive FLOW then FINISH is not penalized.
- Same occurrence/state repeated rejected FINISH is blocked from Executor but retains the new proposal log-probs; a FINISH at a distinct occurrence is not automatically duplicate.
- Truncated/context/resource-censored errors carry reward=None (or matching censor protocol), not a chemical penalty.
- Shared-prefix tokens appear once in the optimizer input, even if different successors yield success and failure.
- If auxiliary shaping exists, the sum of rewards on a failed rollout obeys the frozen terminal treatment (no unintentional net positive failure exploitation).
- Backtracking from a terminal failed edge to the exact parent or nearest OPEN ancestor preserves Executor state, atom mapping, accepted history, and original SFT-native observation.
- Unknown/masked proposal probabilities are never substituted with plain model log-probs in the on-policy ratio.
