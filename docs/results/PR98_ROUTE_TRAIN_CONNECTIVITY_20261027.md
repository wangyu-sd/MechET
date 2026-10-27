# PR98：路线级后训练的 TRAIN 连通性与接口审计

2026-10-27。对应 PR98 设计提交 `e7bce5d4ecb0b35acd13213231b6741c43ec8b3a`。

**本次实际完成：** 全量扫描两个 PR96 accepted TRAIN 文件，校验源文件 SHA，构造 TRAIN 反应超图，并在三个冻结库存中计算 AND 闭合与最小反应深度。仅使用 CPU，未加载模型、未生成预测、未更新权重、未提交或停止太极任务。

**本次尚未完成：** 拼接路线逐边重新执行、跨边原子身份对应、路线级在线采样、梯度更新及效果评估。因此以下数量是参考结构的连通性支持，不是已验收的路线训练集，也不是模型准确率。

## 1. 数据与身份规则

数据根目录：`/aaa/fionafyang/buddy1/whaleywang/MechET/`。

| TRAIN 文件 | 完整扫描反应行数 | SHA-256 |
|---|---:|---|
| `data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01/inverse/train.jsonl` | 10,137 | `9d008f435412df07e7714460c0b875fbd47ba0d5f3b7adddef702889fea5a76a` |
| `data/pr96_flower_mainonly_source_arrows_v2_20261023_06/inverse/train.jsonl` | 243,753 | `e2574d2037171f9091ca03aec194cb9dae0a1cd539c491c12cacdf72204cc166` |

两者均为已命名的 source-arrow accepted 条件，不能称为未经限定的公开全集。扫描哈希与现有两份 `outputs/submissions/pr96_*_earho_train*_20261025_02/contract.json` 一致；两份文件解析错误均为 0。本次不改写源数据或其训练分母。

只读取冻结 `all_targets.jsonl` 中的公开产物，用于排除 460 个评价 ID 对应的 457 个不同化学身份。未读取其参考路线用于拼接、偏好或正例构造。

| 来源 | 与冻结评价产物相同的源反应行 | 排除后参考边行数 | 不同产物身份数 |
|---|---:|---:|---:|
| mech31k | 1 | 10,136 | 10,129 |
| FlowER | 54 | 243,699 | 202,569 |

这是**新路线任务的构造审计**，不是对原 SFT 再筛选。这里只排除评价产物身份；尚未完成设计要求的必要反应边重叠检查，不能声称已完成全部训练/评测隔离。

身份键复用 `proof_program._canonical_unmapped` 的逐组分规范化，使用 RDKit 2026.03.4：去 atom map，保留立体、同位素和电荷，按既有端点评分处理可移除的显式氢。组分排序不丢失重复组分。字符串键能证明正向相同，不能穷尽所有对称立体表达的等价性，因此连通性计数受当前键定义限制。

每条边使用 `target_smiles` 和 `structural_precursor`，不把 `expected_precursor` 的全部环境分子都当作路线义务。结构混合物整体若恰好在 stock，按既有 whole-salt 规则保留；否则每个结构组分都是必须闭合的义务。无结构变化的单子边不用于拼接。电子参考步数取自 `metadata.trace_plan.steps`，不是本轮新的执行结果。

## 2. 冻结库存

库存目录：`data/route_benchmarks/frozen_r1_r2_20261013/stocks/`。本轮读取规范化文件并重新核对 SHA，未更改或合并库存。

| 库存 | 不同条目数 | 规范化文件 SHA-256 |
|---|---:|---|
| URSA | 255,436 | `d2b40f5eaffc6d21d1db347f2f2685c0fb06c89fa76ba9c293beb84c624e73d2` |
| buyables publication | 313,458 | `5a54e8c5a3d28bfd5ca09d0017c157ed357659797a09483de1b1a259b9b858cd` |
| n5 publication | 13,306 | `0a96cb8c5f2115d94e4c9a299b1085f76c9afbfa6ef5e89178d29b5cce434c59` |

只提交哈希和统计，不重新分发库存原文。沿用冻结 manifest 的许可及用途约束。

## 3. 全量参考超图的闭合结果

从 stock 开始做最小不动点：一条参考反应的**所有**前体已闭合，才能闭合其产物。深度为 `1 + max(所有前体深度)`，同一产物保留最小值，stock 深度为 0。按深度优先队列传播，不引入任意化学深度上限；没有有限 stock 证据的环不能自行变成成功。

| 来源 | stock | 非库存不同根数 | 可闭合不同根数 | 最短深度 1 | 最短深度 ≥2 | 未闭合根数 |
|---|---|---:|---:|---:|---:|---:|
| mech31k | URSA | 9,942 | 2,299 | 2,296 | 3 | 7,643 |
| mech31k | buyables | 9,933 | 2,136 | 2,133 | 3 | 7,797 |
| mech31k | n5 | 10,109 | 1,016 | 1,010 | 6 | 9,093 |
| FlowER | URSA | 191,544 | 40,173 | 33,525 | 6,648 | 151,371 |
| FlowER | buyables | 191,584 | 38,561 | 31,728 | 6,833 | 153,023 |
| FlowER | n5 | 201,510 | 17,498 | 13,627 | 3,871 | 184,012 |

FlowER 最短深度分布：

| stock | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| URSA | 33,525 | 5,258 | 1,116 | 247 | 26 | 1 | 0 |
| buyables | 31,728 | 5,418 | 1,081 | 291 | 39 | 4 | 0 |
| n5 | 13,627 | 3,155 | 532 | 110 | 55 | 16 | 3 |

mech31k 的闭合最短深度只有 1 和 2。**“最短深度≥2”不是“所有存在两步路线的根总数”**：一个根可能同时有一步和多步解，本次表格只将它计入最短深度 1。后续正式构造器需另报具有至少一个多反应证据的根数，不能把这里的 3–6 误写成全部可构造多步路线数。

本轮扫描/规范化/图传播用 16 CPU 进程；mech31k 约 2.62 秒，FlowER 约 70.49 秒，不含共享库存初始化。这不是模型采样、电子执行或后训练的速度。

## 4. 可追溯的两条结构拼接证据

这些是 TRAIN 参考结构的连接例子，**不是模型预测，也未在本轮验收为整条化学路线**。每个反应保留自己的参考映射空间，不能直接拼接动作编号。

### mech31k，URSA / buyables，深度 2

1. `10506`：`COCCCCCCOc1ccc(C#N)cc1` → `CO` + `N#Cc1ccc(OCCCCCCBr)cc1`；参考含 2 个电子事件。
2. `25184`：`N#Cc1ccc(OCCCCCCBr)cc1` → `BrCCCCCCBr` + `N#Cc1ccc(O)cc1`；参考含 2 个电子事件。

三个义务叶节点 `CO`、`BrCCCCCCBr`、`N#Cc1ccc(O)cc1` 均在对应冻结 stock。

### FlowER，n5，深度 2

1. `flower_mech_proof_train_148483`：`BrCCCCCCOCCCCSc1ccccc1` → `BrCCCCCCBr` + `OCCCCSc1ccccc1`；参考含 2 个电子事件。
2. `flower_mech_proof_train_8478`：`OCCCCSc1ccccc1` → `OCCCCCl` + `Sc1ccccc1`；参考含 2 个电子事件。

三个义务叶节点 `BrCCCCCCBr`、`OCCCCCl`、`Sc1ccccc1` 均在冻结 n5 stock。

每条候选路线在进入监督前仍需逐边重新执行、验证新的公开起点、保持所有分支与重复组分，并记录 GT 代码来源。结构闭合不证明条件兼容、选择性或实验可行性。

## 5. 接口结论与实施建议

- 旧 `independent_route_control.py` 的独立 episode 语义可参考，但其 `next_request()` 使用旧 system/policy prompt；`independent_route_runtime.py` 固定旧 adapter SHA `40204c7a…`、Qwen3 工具前缀和 vLLM 0.8.5。不能直接给它换成 Qwen3.5-0.8B 权重运行。
- 新 `infer_pr96_mcts.py` 已有 Qwen3.5 模型装载、`SampleBackend`、真实原始输出保存；`pr96_mcts_protocol.py` 已有新 actor 与原执行器的衔接。这部分应复用，而不是宣称所有路线接口都不存在。
- 但这个 backend 当前主要给出动作累计 logprob，训练还需要可逐 token 验证的行为 logprob、独立 episode RNG 和更新边界；现有 MCTS 中的共享候选/cache/N/W 不能进入新在线采样。
- 原 `train_pr96_earho.py` 的 `collect()` / `optimize()` 明确只保存和更新 `first_output`。新路线损失不能直接换 reward 或增大 horizon 就实现。
- 建议先完成 FlowER 路线任务的正式可执行构造与完整 episode 软件验收。31k 保留为独立轨道，不假造多步 GT、不与 FlowER 默默混训；其余 TRAIN 根仍可作为在线待解目标，但没有已知路线不等于必然不可解。

下一步按同 PR 的实施计划完成可重跑的正式构造器、独立路线 episode 和全路线策略梯度。**本轮是 Phase A 的完整结构连通性扫描及接口审计，不是 Phase A 所有执行/隔离条件都已验收，更不是路线 RL 已有效。**
