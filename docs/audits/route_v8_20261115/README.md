# 当前多步路线 RL 的问题审计：v8、截断、奖励信号与推理成本

这条 PR 记录当前已经发生的问题、对应代码和修复顺序。它叠在 [PR #105](https://github.com/wangyu-sd/MechET/pull/105) 上，仅增加审计资料和部署源码的只读快照。没有修复运行中的任务，没有重启训练，也没有宣称新的加速或验证集收益。

## 1. 当前到底在跑什么

正式任务为 `meteor_mechet_stage2_external_route_drgrpo_4h20_zjk_20261114_action_v8_02`，实例 `8b1d8208a1cfaa4801a1d8ff12bb110c`，4 张 H20。模型从修正数据训练的 **Qwen3.5-0.8B History-89442** 开始，固定 KL reference 同源；当前不是 8B 续训。外部多步数据的完整协议为 **23,730 TRAIN / 2,986 VALID**，每个目标独立生成 4 条路线，运行一轮数据遍历。

当前条件为 `route_action_feedback_v8`；有效后端为 **原生 HF 批量采样**，合同中没有 `route_memory`。`generation_batch_size=64` 是容量上限，不能据此声称实际每次都在跑 64 条。参见 [合同字段投影](protocol_projection.json)、[实际 episode 工厂](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/scripts/train_external_route_rl.py#L33-L55)、[实际采样后端选择](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/scripts/train_external_route_rl.py#L124-L204)、[实际批次容量划分](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_generation_capacity.py#L26-L46)。

这些是已明确过滤并冻结的外部多步目标，不应称为未过滤的 FlowER full。R1/R2 与 SimpRetro 仍为保护的评估数据。

## 2. 结果与证据范围

截至 **2026-11-15 16:26 CST** 的这份固定审计覆盖完成的前 **16 次更新**。复现程序逐一读取 16 个 complete receipts，核对合同；另外实际计算第 16 个 adapter 文件 SHA，确认与 receipt 一致且不同于原始 parent。统计来自保存的 TRAIN 轨迹，没有新推理或重打奖励。

| 指标 | 前 16 轮累计 |
| --- | ---: |
| 已访问的训练目标 / 完整 TRAIN 分母 | 1,024 / 23,730（4.32%） |
| 路线轨迹 | 4,096 |
| 模型决策记录 | 79,817 |
| 提交的反应程序 | 5,789 |
| 库存闭合轨迹 | 667 |
| 参考支持的闭合轨迹 | 24 |
| 因 censor 而排除的 G4 组 | 542 / 1,024（52.9%） |
| 组内优势非零的组 | 58 / 1,024（5.7%） |

**没有完整 VALID summary，不能据此声称训练带来提升。** 累计 censor 包括资源或保护数据等未知结果；此次只对第 15 轮做了具体原因拆分，不能将其原因比例外推到全部 16 轮。这些是部分 TRAIN 采集结果，轨迹不是独立目标分母；库存闭合也不等于独立的化学可行性验证。模型决策记录按保存的 records 计数，包含无输出的资源 censor 记录，不能直接等同于 GPU 新生成次数。数据见 [aggregate.json](aggregate.json)。

为了避免混用轮次，下面的耗时、截断拆分和输入例子固定取**第 15 轮**，不是第 16 轮的结果。它包含完整 64 组 / 256 条轨迹。

## 3. P0：把完整留档做成了每次完整输入

当前 `_messages` 把原始目标、所有 pending obligations、已提交反应、stock leaves 和整份 `ledger` 重新序列化到 system 消息，再追加当前状态。`model_view` 确实删除了重复的 native system 和部分派生 capsule，也做了文本表去重，但仍遍历并返回全部历史状态、历史局部观测、过去动作和事件。这不是递归嵌套旧完整提示；问题是持续增大的历史表整体再次进入模型。

对应代码：[完整历史进入提示](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L137-L217)；[全量 model_view](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_event_history.py#L177-L240)；[每次尝试写入历史](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_event_history.py#L144-L154)。

一个已保存的真实例子：模型做了 34 次决策、提交 5 条反应后，下一次输入达到 **33,033 tokens**，超过该条件的 32,768 容量。当前 user 观测只有 **1,282 字符**，但 system 消息为 **69,684 字符**；其中 ledger 占 **66,805 字符**，历史观测约 27,365 字符、历史动作约 14,051 字符、事件约 22,751 字符。字符数是组成分析，不能当作 token 数直接相加。

**修复方向：**落实 PR105 已有的 G1–G3 合同：完整原始事件仍归档，模型读取确定性的 Working / Failure / Route-Proof / Episodic 视图。当前结构、有效原子键清单、必要的 AND 分支义务必须保留；过去失败按当前状态和 occurrence 关联，不把所有旧 inventory 每次展开。 optional memory 用真实部署 tokenizer 计量。该改造必须使用新的 observation 版本，不能假称与 v8 提示字节等价，也不能删除原始行为概率证据。

**验收：**在现有完整保存轨迹上验证重放/恢复、分支 multiplicity 和当前状态优先级；长历史的 optional memory 有界；必要当前状态无法容纳时仍明确 censor。CPU 重放通过不等于新策略有 GPU 或路线收益。

## 4. P0：两个不同截断源，最后导致整组样本排除

当前每次允许输出的长度是 `min(max_action_tokens, max_context - prompt_length)`；`max_action_tokens=2048`、`max_context=32768`。没有全路线累计 token 上限，但存在这两个逐调用边界。参见 [room 计算与实际 generate](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_sampling.py#L198-L307)、[每次动作的输出容量](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py#L129-L135)、[截断转为 resource_censored](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L287-L380)。

第 15 轮 70 条 censored 轨迹拆分如下：

| 具体原因 | 条数 | 代码中的表现 |
| --- | ---: | --- |
| 输入已无剩余上下文 | 45 | `PHYSICAL_CONTEXT_LIMIT` |
| 动作输出耗尽输入后剩余的上下文 | 12 | `ACTION_OUTPUT_CONTEXT_LIMIT` |
| 输入仍有空间，但单动作到达 2,048 输出上限 | 13 | 同样标为 `ACTION_OUTPUT_CONTEXT_LIMIT` |

所以之前把所有 70 条都解释为“输入历史太长”是不准确的。后两类需要看原始 prompt/completion 长度才能区分。70 条轨迹影响了 38 / 64 个 G4 组；任一 sibling 的奖励是 `None` 就排除整组，见 [whole-G4 censor 规则](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py#L18-L35)。这里是未知结果，不得改写成化学失败或给 0 分来提高样本利用率。

**修复方向：**先处理提示膨胀，另行解决合法动作未生成完就到单动作上限的问题。增加上限、延续生成或改变停止条件都需要新的明确协议和真实概率记录；不能事后拼接结果、丢弃失败 sibling 或伪造未截断轨迹。本 PR 没有修改现有上限。

**验收：**分别报告输入溢出、剩余上下文不足、单动作上限、有效 G4 比例；验证合法完整 tool call 的边界行为，保留原始 EOS/stop/token/logprob 证据和恢复语义。

## 5. P1：当前没有使用已有 vLLM 路径，HF 生成占主要成本

代码已经存在 `VllmRouteSampler`、持久化 Engine 子进程和真实行为概率回执接口；它没有被这份正式合同启用。合同没有 `inference_backend=vllm`，`collect_round` 选择 HF。当前 HF 每个决策重新 render/tokenize 完整提示，再调用 `model.generate`；`use_cache=True` 用于单次生成内部，没有把跨工具决策的 `past_key_values` 传给下一次调用。

代码入口：[后端实际分支](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/scripts/train_external_route_rl.py#L124-L204)、[原生批量 sampler](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_io.py#L44-L49)、[HF 完整输入生成](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_sampling.py#L198-L307)、[已有 vLLM sampler](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_backend.py#L184-L212)、[显式启用 vLLM 的配置入口](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_contract.py#L114-L118)。

第 15 轮 rank00 的观测：路线采集 4,898.32 秒，其中 `sampling.generate` 4,617.08 秒；`learner.total` 714.41 秒；checkpoint 发布 5.77 秒。日志 flush 覆盖 159.92 秒，也值得优化，但不是这份窗口中的首要成本。phase 存在嵌套，不能把全部值直接相加；这里没有测量 device prefill/decode 的拆分，不能据此断言其比例。

**修复方向：**复用已有 vLLM 集成做同 actor / 同完整工作量的验收，检查真实 GPU LoRA buffer、权重版本同步、原始行为概率、actor/reference/rollout 显存共驻和更新梯度。不要只换配置字段就上线，不额外占用一套 GPU 服务，也不能只因历史测试失败就断言 vLLM 当前不可用。历史 vLLM 迁移没有完成，本次审计没有重跑它的 GPU 验证，也没有测得新的速度收益。

已有检查和接口：[实际 LoRA buffer 审计](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_adapter.py#L107-L168)；[adapter 版本 lease](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_protocol.py#L19-L37)；[token/logprob 回执校验](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_protocol.py#L58-L76)；[运行时和依赖绑定](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_contract.py#L12-L111)。

## 6. P1：v8 缓存了失败执行，仍不能阻止失败提议消耗 GPU

v8 将当前 source-bond inventory、FINISH 可用性和同状态失败写回输入；相同反应和结构下的语义等价失败动作复用拒绝回执。`semantic_action` 忽略解释文字，并规范化部分无序表示，但不会重排电子流顺序或替模型选择新动作。

因此“换措辞再做一次错误动作”可以不再调用化学执行器，却仍然经过了模型生成。这是执行去重，不是模型提议去重；本 PR 没有统计当前完整训练中的提议重复率，不能把旧的两目标诊断计数套到本次全量任务。

代码：[动作语义签名](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_action_feedback.py#L10-L38)、[当前状态失败查找](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_action_feedback.py#L41-L53)、[当前约束反馈](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_action_feedback.py#L61-L77)、[重复失败复用分支](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L287-L380)。

FINISH 不可用时，schema 不提供它，执行器仍会拒绝模型强行提出的调用；并非 logits 硬屏蔽。需要有电子流、结构前体发生改变并避开祖先循环。FINISH 后只提交当前反应，后续前体继续从新的 executor 状态展开。连续 8 次拒绝是当前恢复终止规则；它不是 8 步路线深度上限。

代码：[动态 FINISH 工具列表](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_action_feedback.py#L80-L83)、[productive / cycle 检查](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L219-L231)、[连续拒绝上限](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L233-L236)、[反应提交和新前体队列](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py#L382-L404)。

## 7. P1：路线奖励很稀疏，当前没有步级信用分配

当前评分是确定性 executor / reference 规则，没有训练 reward model。参考路线仅在采集之后进入隐藏 scorer，不注入模型提示。实际奖励为：

$$
R(\tau)=\begin{cases}
\mathrm{None}, & \text{资源或保护数据 censor}\\
0, & \text{路线图无效}\\
1, & \text{结构前体全部库存闭合且所有反应边有参考支持}\\
0.2\min\!\left(\frac{S}{\max(L_{ref},1)},1\right), & \text{其他图有效路线}
\end{cases}
$$

$S$ 为从可信祖先延伸、匹配参考边的不重复反应数；$L_{ref}$ 为该目标参考步数。不可信分支上偶然命中参考边不获得可信进展。图有效但最终未完成的轨迹可以保留 partial reward；拒绝一个动作没有单独的固定惩罚。

评分比较的是规范化的 `product → precursor multiset` 反应边，不是电子机理动作逐条等于 GT。前体来自 executor 的结构端点投影，不把整套环境试剂当作路线义务。库存闭合但参考未覆盖的路线是未独立核验，不应直接宣称化学错误。

代码：[实际 scorer](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py#L57-L116)；[反应边签名](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_data.py#L28-L29)；[结构端点与辅助参与物拆分](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/endpoints.py#L81-L92)。

每个目标采样四条完整路线，使用 $A_i=R_i-\frac14\sum_jR_j$，不除以标准差。任一 `None` 排除整组；同一条轨迹中的所有生成动作共享同一个最终优势。clip=0.2、KL beta=0.02，动作 token 求和采用固定 C=1024 和有效组数/G4 归一化。没有按当前动作是否执行成功就立即加分，也没有已部署的步级因果信用分配或 teacher CE。

代码：[组均值优势](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py#L18-L35)、[路线级优势应用到所有动作](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py#L138-L266)、[实际 clipping 和 importance-corrected KL](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_objective.py#L17-L27)。

前 16 轮只有 58 组有非零组内优势；其余可用但等分的组缺少路线偏好 PG 信号，仍可能贡献 KL。不能把“参数变了”当作已学会更好路线。

**修复顺序：**先提高可完成的 G4 比例，核验已有成功及实际恢复轨迹，用独立参考支持的路线和可验证恢复证据做输入适配；再做新的 on-policy RL。新的过程奖励、critic 或信用分配方法需单独设计和验证，本 PR 不暗中更换算法或奖励。

## 8. 记忆模块的代码基础存在，但当前训练没有启用

现有 mixin 能绑定冻结快照、精确作用域和少量 typed advice，curator 能回放 TRAIN 轨迹并区分“合法 import”与后续真正提交的恢复反应。但当前合同没有 `route_memory`，因此 episode 工厂没有选择 memory-conditioned episode；G1–G6 和记忆条件训练尚未交付。

代码：[实际启用条件](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/scripts/train_external_route_rl.py#L33-L55)、[快照绑定检查](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_memory_integration.py#L68-L74)、[现有 exact/typed advice 追加方式](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_memory_integration.py#L76-L94)、[TRAIN 回放与正例门槛](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_memory_curation.py#L35-L112)。仅给现有长提示再追加 advice 不会解决历史膨胀，必须先实现有界 public projection。

## 9. 接下来的交付顺序和验收

| 顺序 | 交付 | 必须看到的证据 |
| --- | --- | --- |
| 1 | 在 0.8B 上落实 G1–G3，分离完整档案与模型记忆视图 | 原始事件全保留；当前结构/分支正确；跨 reaction handle 不串用；真实 tokenizer 下 optional memory 有界；恢复一致 |
| 2 | 分开修复输入溢出与合法单动作截断 | 两类边界各有回归；真实 token/probability/stop 回执；whole-G4 censor 统计完整 |
| 3 | 核验 TRAIN 路线与恢复数据，做记忆输入适配 | 不混入受保护数据；库存命中不是正例通行证；无隐藏参考动作；数据分母和排除账本明确 |
| 4 | 完成已有 vLLM / 调度路径的数值、驻留和完整工作量验收 | 相同 actor/输入；真实行为概率；不增加独立 GPU 服务；有效更新/GPU 小时的实际变化 |
| 5 | 用新命名 observation 条件重新采样并做完整 VALID 对比 | actor/memory 对 G4 冻结；不复用旧提示概率；原始 v8 和权重保留；降低 censor 且路线质量有证据 |

上述都是待实施项目。当前运行代码、合同、奖励和 GPU 分配保持不变；没有新 smoke、pilot、子集训练或评估任务。全文所谓“CPU 重放”均为读取已保存证据的工程核对。

## 10. 如何复现与核对所有代码引用

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
python -S docs/audits/route_v8_20261115/reproduce.py \
  --submission-dir /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/submissions/route08b_v8_full_20261114_02 \
  --up-to-round 16 --diagnostic-round 15 \
  --output-dir /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/route_v8_saved_audit_replay
```

需要已存在的私有共享存储产物，公开包不含目标 ID、分子结构、原始提示、隐藏参考路线或挂载凭据。[复现脚本](reproduce.py) 只读保存的 TRAIN receipts，不调用模型、不重新核验化学、不改变奖励；输出为聚合 JSON。报告里的 `checkpoint_manifest_and_weight_verified` 指选定第 16 个实际 adapter 的检查，不表示此次重新加载了模型或验证了全部历史 checkpoint 权重。

所有代码引用固定到公开审计源码 commit `b4fbc4ee59c6a7665582db9463a44eaef2333aef`。这 17 个文件与部署合同中的对应 SHA 全部相符；它们位于 `docs/audits/.../source/`，仅供审阅，不进入运行时模块路径。原私有源 commit 为 `c43bba1ad1ba89a4c518ffb8a1d653cfbb53bae5`。完整原始路径、SHA、AST 函数行号见 [source_manifest.json](source_manifest.json)。

| 原始部署文件 | 冻结源码链接 | SHA-256 |
| --- | --- | --- |
| `src/mechet/route_policy_episode.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_episode.py) | `2039481b5cce4891924adad49d797befaf8621d57a580f43e176d36496012074` |
| `src/mechet/route_event_history.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_event_history.py) | `61e076dfb97ebdd92409e946c9e760c0c683b23488e29995cb170bff23ab7c9e` |
| `src/mechet/route_action_feedback.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_action_feedback.py) | `73d576ef25a542fbab160f248106e849e0d8c7e315671b2e10df1805b31845ef` |
| `src/mechet/route_policy_sampling.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_sampling.py) | `a8fc894162b6dc0060ec7e2a0e5c09c5236f33b6120cb704bc24d33cb8db9545` |
| `src/mechet/external_route_io.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_io.py) | `23f5f677d6f8f5c71fdd45c421b4315a75ea330d53d1afa8019b2d983ab77167` |
| `src/mechet/external_route_rl.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_rl.py) | `e541b807c43eb3f4b7bd1b4fac357cfa1623f8c9a1aefc5c41a5752384d8776f` |
| `src/mechet/route_policy_objective.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_policy_objective.py) | `0170c13c925158fa33b917c52cd2e7a380f1bd11edebdc34edc8d515f4e0319e` |
| `src/mechet/route_memory_integration.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_memory_integration.py) | `82d4275a412530dcae9b3bac64e3f5da33bb8a64eb5cce37d2ba74ea52527c2f` |
| `src/mechet/route_memory_curation.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_memory_curation.py) | `bdbd979c270f500e89696f17b204b2d247c86e57fbe854a3c083eef8ee298a4d` |
| `src/mechet/route_vllm_backend.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_backend.py) | `1b36ea43c592d13af92a141f36c20c44f131083fa53c5999b0783941ebefb5e2` |
| `src/mechet/route_vllm_contract.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_contract.py) | `e01c6a3bb2f3c6bac9535dc5b17d8d2b2d9bf4f7309b179b4071a6579c946f89` |
| `src/mechet/route_vllm_adapter.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_adapter.py) | `9ec56f0535b185327261a629a736ecbc5fee07b343d1d263ba8249ae1f5b3b7f` |
| `src/mechet/route_vllm_protocol.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_vllm_protocol.py) | `843e2b0c52a7003bc2e56164ee37492d6f772faa33131fa738b9e7cf60706f08` |
| `src/mechet/route_generation_capacity.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/route_generation_capacity.py) | `b558a61305b80eea18831a074838d889ec4eb85e75d18fa13f525d444a75de9d` |
| `src/mechet/endpoints.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/endpoints.py) | `2afdfbdfc828e064fa6cd2ce9dae2c693c38013e0cd47395447b998f3c415d22` |
| `src/mechet/external_route_data.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/src/mechet/external_route_data.py) | `358910be0a303b6423a2d5df0be68cdf65cf65956c90d9f7758920fd1eecd515` |
| `scripts/train_external_route_rl.py` | [逐行查看](https://github.com/wangyu-sd/MechET/blob/b4fbc4ee59c6a7665582db9463a44eaef2333aef/docs/audits/route_v8_20261115/source/scripts/train_external_route_rl.py) | `2a276a109a0f3c91f23dc753ed79dbc0a7f7a6ee85a0fe075380d9be7aa5b599` |
