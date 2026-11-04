# PR98 路线级逆电子流后训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从已训练的 Qwen3.5-0.8B Stage-II 出发，用跨反应、全部前体到购买库的完整路线回报更新全路线电子决策，独立保留 TRAIN 电子轨迹监督。

**Architecture:** 一个 electron actor、原机制执行器、冻结 stock 和确定性的外层义务调度器。在线采样使用相互独立的完整路线 episode，不借用 MCTS 的优胜路径；GT CE 与模板诊断分开。

**Tech Stack:** 现有 Python / RDKit、PR96 Qwen3.5 官方 XML 协议、PyTorch / Transformers / PEFT，以及已验证的 HF 生成路径；不预设旧 vLLM 0.8.5 支持新模型。

**Spec:** [PR98 最新路线级设计](../specs/2026-10-27-stage3-multievent-gt-template-design.md)，依据提交 `e7bce5d4ecb0b35acd13213231b6741c43ec8b3a`；[本轮完整结构连通性审计](../../results/PR98_ROUTE_TRAIN_CONNECTIVITY_20261027.md)。

**状态：计划已写，以下代码任务均未实现。** 本次推送只包含文档及实际 CPU 审计结论。不启动训练，不更改当前作业，不改 paper。本计划的接口和优化细节是供确认的设计选择，不是已测最优配置。

## Global Constraints

- 保持“内层 IMPORT/FLOW/FINISH，外层所有非库存结构前体继续拆解”；FINISH 只结束一个反应。
- 父权重分别是各自 PR96 Stage-II：31k History660 / `cd5f2f9d…ab49ca`，FlowER History89442 / `2c98a9bf…e5f453`；完整 SHA 以设计及原合同为准，不接 TEST-trained 权重。
- TRAIN 来源分别为 10,137 / 243,753 个 accepted source-arrow 反应；保留原始分母、隔离与哈希。新路线构造不修改原 SFT 文件。
- 冻结 460 个评价 ID、457 个不同目标身份及各自 stock；不使用其参考路径、模型候选或 MCTS 缓存训练。
- GT CE 只来自获准 TRAIN；模板默认只诊断，权重为 0。无新增 critic、backbone、模板 ID 预测或静默混训。
- 所有新输出、缓存、临时测试文件放 Ceph；运行前 source `configs/runtime/shared_host_storage.env`。没有明确授权不运行模型 smoke/pilot 或提交 GPU 任务。
- 后续任务仍须 meteor 前缀、成功案例的腾讯镜像与 Ceph init、持续默认 POD stdout/stderr；本 PR 不生成提交请求。

## Review Focus

1. 多个前体必须全部闭合；一条分支到货、初始目标已在库存、环与无变化不能产生非零步成功回报。
2. 跨反应的 atom map 是局部名字；同一中间体连接时不得把上一反应编号直接当成下一反应编号。
3. 独立 episode 不共享候选或搜索决策；跨批次重排不应重复 RNG，采集与更新不能混用权重版本。
4. 预算终止、模型拒绝、物理删失分开；跳过删失样本可能引入选择偏差，不能宣称无偏完整根分布结果。
5. 带电物、whole-salt、重复组分和对称立体表达保持原协议；模板未知不当负例，stock 闭合不当化学真值。

## 先明确两个理论细节

令根目标为 `x`，冻结库存为 `S`，完整路线 episode 为 `tau`，其中全部模型生成 token 为 `y_u`，每个 token 的实际观察为 `o_u`。外层调度固定为确定性环境规则，不加入需要学习的新动作。

主目标是 `J(theta) = E_x E_tau~pi_theta [R_stock(tau)]`。在没有删失、使用新鲜采样的理想条件下，其 score-function 梯度是 `E[(R-b(x)) * sum_u grad log pi_theta(y_u|o_u)]`。因此：

- **不能未经说明地把每条路线除以它自身的 token 数，再宣称仍是完全相同的路线成功目标。** 这会改变不同长度路径的权重。本计划建议均匀平均目标和 episode、路线内部对策略 token 求和，再除固定常数；内存微批量不改变该求和。若选择按实际长度归一化，须显式标为不同的 surrogate，并做长度偏差对照。这是对设计原“有效动作归一化”表述的精确化提议。
- 同一目标 K 条独立路线用 leave-one-out 基线 `b_i = sum_{j!=i} R_j/(K-1)`，不除回报样本标准差；K 至少 2。全同回报组的端点优势为 0，保留 KL/GT 的独立记账，不能伪造 RL 贡献。

实做可用 token-wise clipped surrogate + 固定父策略 KL + 独立 GT CE；clip 是稳定更新近似，不保证每次提升真实路线成功率。KL 采用有明确采样分布的估计，不能在复用旧样本多轮更新时把无校正估计称为当前策略精确 KL。首版每次采集后只做一轮策略更新，更新完成就废止该行为版本的在线样本；进一步复用需要单独定义重要性校正。

终止条件不同于删失：预先声明的 episode 计算预算用完且未闭合，任务回报为 0；OOM、缺失回执、物理上下文或作业中断为 `resource_censored`，优先原状态续作。未能续作则单列，不把它们当化学负例；报告含删失分母的成功下界与删失率，不能只报成功完成子集的高准确率。

## 文件边界与实现顺序

参考实现分散在 PR97 与已有 MCTS 工作区，代码实施前应从其**固定提交**只引入需要的依赖，不能把本计划分支误认为已经包含训练器。保留旧入口，新增 route-specific 模块，不热改运行依赖。

| 新增文件 | 单一职责 |
|---|---|
| `scripts/audit_pr98_route_train.py` | 可重跑的完整 TRAIN 参考超图与证据清单 |
| `src/mechet/route_training_contract.py` | 来源、行为策略、库存、预算及权限契约 |
| `src/mechet/route_policy_episode.py` | 独立完整路线的义务和动作状态机 |
| `src/mechet/route_policy_sampling.py` | Qwen3.5 原协议、滚动 batch、逐 token 回执 |
| `src/mechet/route_policy_objective.py` | 端点优势、全路线 loss、KL 和独立 GT CE |
| `scripts/train_pr98_route_policy.py` | 分轮刷新策略、训练游标、检查点和日志 |
| `scripts/eval_pr98_route_policy.py` | 同预算独立采样与 Tree 的 2×2 对照 |

### Task 1：正式 TRAIN 路线构造与来源验收

**Files:** Create `scripts/audit_pr98_route_train.py`, `src/mechet/route_training_contract.py`; Test `tests/test_pr98_route_train.py`。

**Interfaces:** `audit_train(contract: dict, output: Path) -> dict`；产出 source-bound 反应边清单、root 清单、stock SHA、每条见证的源 ID/字节定位、隔离清单及完整报告。只允许 `model_execution=false`，不装载 actor。

- [ ] 写 failing tests：双前体缺一不闭合；纯环不闭合；重复组分保留；stock 中整盐不拆；评价目标作为中间反应产物也不能进入新路线 GT；同名不同电荷不能连；有一步/两步替代解分别计数。
- [ ] 运行 `python -m pytest tests/test_pr98_route_train.py -q --basetemp=<唯一Ceph目录>`，确认新接口测试失败。
- [ ] 实现图传播和逐边重新执行；图身份是连接条件，不重用跨反应 map。分别验证每条参考程序能从本反应的主产物起点执行到其结构前体，保留新映射对应见证；反应条件兼容性另标未验证。
- [ ] 完整扫描两个原 TRAIN；报原始行数、不同根数、最短深度、任意多反应见证根数、无法连接原因及验收失败数。只把**整条**参考程序已验收者用于路线 CE；其余原 TRAIN 根仍可用于在线探索，不能捏造路线答案。
- [ ] 单元测试通过后提交。正式构造结果另成有哈希的 Ceph artifact；本轮 3–6 / 3,871–6,833 只是最短深度≥2的连通性观察，不能当正式构造验收数。

### Task 2：独立完整路线 episode

**Files:** Create `src/mechet/route_policy_episode.py`; Test `tests/test_route_policy_episode.py`。

**Interfaces:** `RouteEpisode(contract, public_root, stock, uid, seed)`；`next_decision() -> DecisionRequest | None`；`accept(receipt: ActionReceipt) -> None`；`finish_report() -> RouteEpisodeReport`。请求包含 episode/反应/动作 UID、当前真实 node 和权重版本，永不包含 GT。

- [ ] 写 failing tests：`P -> A+B` 中 A 到货但 B 未闭合仍不得成功；两条反应后全部到货成功；stock 根零步不给奖；祖先循环失败，共享支路不误判成循环；非法动作不能回退换另一候选再冒充原独立 episode。
- [ ] 运行对应测试确认失败。
- [ ] 实现确定性调度：当前反应做到 FINISH 后，按规范化分子键和 occurrence ID 排序处理未解决前体；每个 occurrence 保留多重性/祖先。新反应以公开分子重新构建 root 与局部压缩历史，不向提示添加新路线字段。
- [ ] 明确结束状态 `stock_solved / already_stock / rejected_action / cycle / nonproductive_finish / budget_exhausted / resource_censored`；正奖须同时有非零步真实反应和 AND 证据。拒绝无 FLOW 或结构不变的 FINISH，日志仍保存原始输出。
- [ ] 测试通过后提交；证明复用的是执行/库存语义而非 MCTS 的路径选择。

### Task 3：0.8B 采样与行为概率

**Files:** Create `src/mechet/route_policy_sampling.py`; Test `tests/test_route_policy_sampling.py`。

**Interfaces:** `collect_route_groups(actor, tokenizer, tasks, contract) -> list[RouteGroup]`；每组同根同 stock、K 个独立 episode。`ActionReceipt` 保留原始 messages、prompt/generated IDs、每 token 行为 logprob、EOS、执行前后状态、行为版本及时间。

- [ ] 写 failing tests：Qwen3.5 `messages_for(node, True)` + `tool_prefix` 与 Stage-II 逐 token 一致；mask 只覆盖生成 XML 到 EOS；无 GT 标号预填；批次重排不会复用同一 episode 的随机流；相邻权重版本不能混组。
- [ ] 运行对应测试确认失败。
- [ ] 复用新版 HF 装载/原始批次保存，新增逐 token logprob 与 episode 私有 RNG；T=1、top-p=1、top-k=0、无 penalty，使采样概率与训练重算契约一致。重算时禁用随机 dropout，并验证冻结参数下逐 token 概率一致；长度截断不得丢弃后偷偷重采，生成前缀及预算必须保留，按可续作或删失处理。小型随机初始化网络用于软件概率/梯度对照，不装载项目 actor 做模型实验。
- [ ] GPU 生成与 CPU execute 分离滚动入队，16 CPU worker / GPU 仅为待测上界而非性能承诺；所有批次先保存原始输出，再执行和打分。可共享只读模型、stock、确定性模板检查；不共享随机候选、执行候选分支、MCTS N/W。KV 复用只在真实上下文相同时允许。
- [ ] 通过软件测试后提交。另记录每类 CPU/GPU/I/O 用时；不宣称已集成 vLLM 或获得未实测加速。

### Task 4：全路线更新、GT 与模板诊断

**Files:** Create `src/mechet/route_policy_objective.py`; Test `tests/test_route_policy_objective.py`。

**Interfaces:** `route_advantages(group: RouteGroup) -> list[float]`；`route_policy_loss(records, actor, reference, contract) -> LossBreakdown`；`LossBreakdown` 独立报告 endpoint PG、reference KL、GT CE、template diagnostics 和各自有效计数。

- [ ] 写 failing tests：两个反应中的所有生成动作均有策略梯度；工具/输入/GT 无策略梯度；[0,1] 的 leave-one-out 优势为 [-1,1]，[0,0] 为 [0,0]；相同内容不同 padding 不改变梯度；跨 rank 空样本归一化与单进程一致。
- [ ] 运行对应测试确认失败。
- [ ] 按前述目标定义求和 token logprob，目标/episode 均匀平均；把归一化常数、clip、KL 采样估计显式写进契约。GT CE 只来自 Task 1 接受的 TRAIN 参考，用现有 assistant mask，按单独的监督 token 分母归一化，不把其下降归因于 RL。
- [ ] 模板首版 `eta=0`，诊断检查逆 `s->s'` 的正向 `s'->s`；支持/未知/明确矛盾/资源不足分开。非法动作可作为回报 0 的采样终止，不能仅因带电而非法；循环/重复 IMPORT 不得刷奖励。全失败组保留 0 PG，KL/CE 是否更新分别记录。
- [ ] 通过软件梯度和离子/模板方向回归后提交；不引入 value model，不拿 MCTS 找到的赢家当独立策略样本。

### Task 5：训练编排与可靠恢复

**Files:** Create `scripts/train_pr98_route_policy.py`; Test `tests/test_pr98_route_training.py`。

**Interfaces:** `train_route_policy(contract_path: Path) -> None`。依赖 Tasks 1–4；采集期间冻结行为版本，完成采集后更新，再发布完整新版本。契约必须冻结来源、父模型、stock、根清单、K、每 episode 预算、GT/KL 权重及计算分母，不接受缺省旧 EARHO 配置。

- [ ] 写 failing tests：恢复时更换 actor/source/stock/prompt 任一 SHA 都拒绝；写入中断不把部分 checkpoint 当完整；未闭合 episode 恢复不得重复奖励/覆盖原始输出；不能在非 TRAIN 根上更新。
- [ ] 运行对应测试确认失败。
- [ ] 保存 adapter、optimizer、调度、各 rank RNG、行为版本、全部待办路线、样本 UID 和哈希索引；原子发布 complete manifest 后才更新 latest 指针。旧 episode 只能按旧行为版本续采，不在新 actor 下续采后声称同策略。
- [ ] 首个正式轨道建议 FlowER；31k 独立保留，不用扩大 GT 标签或混合语料掩盖其多步连通性稀疏。所有原 TRAIN 根维持独立记账，闭合参考子集仅是课程/监督来源；课程抽样若改变目标分布，必须另报分布，不称为均匀全 TRAIN 优化。
- [ ] 通过软件恢复测试后提交。CPU fixture 通过不等于模型增益；无用户授权不提交真实训练。

### Task 6：同预算评测与发布

**Files:** Create `scripts/eval_pr98_route_policy.py`; Test `tests/test_pr98_route_evaluation.py`。

**Interfaces:** `evaluate_routes(eval_contract: dict, actors: dict) -> dict`；四个条件为 Stage-II/Route-RL × independent/tree。同一目标、各自冻结 stock、同 token/执行预算，保留未解决和删失分母。

- [ ] 写 failing tests：所有 460 ID 恰好一次汇总；一个 stock 分支不能算 solved；新 actor 换搜索预算不能当纯训练增益；零步到货另计；多重性保持；删失不从表中消失。
- [ ] 运行对应测试确认失败。
- [ ] 输出 Solved@Budget、首次发现曲线、实际合成深度与反应数、资源删失、独特完整路线、模板诊断和配对 bootstrap CI；单反应端点保持各原始人口独立报告，不使用 ranking Top-K 作主指标。
- [ ] 对完整原始 I/O、行为概率、路线证据、成本与报告做来源哈希核查；保留模板/stock 不是独立化学真值的说明。
- [ ] 验证并提代码 PR，后续只有获得训练/评测授权才运行正式协议。不能把本计划的预期行为填写为已有结果。

## 本轮交付与下一次批准点

本轮已完成最新设计阅读、原接口核查、两个 accepted TRAIN 文件及三个 stock 的完整结构连通性扫描，并推送本实施计划。六个代码任务仍未开始，不把 DOCS ONLY PR 偷换成训练提交。

下一次应先确认本计划对优化归一化、删失以及 FlowER 优先轨道的处理，再依次实现 Tasks 1–6；预算和权重在正式契约中冻结、不是在对话中宣称最优。保留现有全部训练/MCTS，不自动 merge 本 PR 或改 paper。
