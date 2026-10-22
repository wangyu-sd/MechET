# Codex Goal — MechET 0.8B：mech-USPTO-31k / FlowER 主产物-only 修正与两阶段充分训练

> **本 PR 仅提交需求，不包含任何实现代码、训练配置、启动脚本、模型权重或实验结果。** 用户会基于本 PR 单独开启 Codex Goal 执行。本 PR 不负责提交 Taiji 任务。
>
> **研究目标：** 保持 MechET 的 executable reverse electron-flow 范式，从**唯一主产物**出发，由模型自主决定何时 IMPORT 缺失参与物、产生电子源/受体及耦合 FLOW，执行器更新化学状态，最终通过 FINISH 得到前体。首先把 mech-USPTO-31k 和 FlowER **两套数据都修正到相同的主产物-only 输入口径**；然后分别使用 Qwen 0.8B 充分完成 Stage-I State-SFT；固定、轻量的真实抽样验证通过后，在各自训练轨道中自动续接 Stage-II compressed-history Trajectory-SFT。

## 0. 开始前先读；锁定模型、代码和科学边界

- 以 **已合并的 MechET PR #95** 为基础；阅读 `src/mechet/inverse_trace_data.py`、`scripts/build_mech31k_main_product_only_sft.py`、`scripts/build_natural_language_event_sft.py`、`scripts/build_natural_language_history_sft.py`、`scripts/train_tool_sft.py`、`scripts/prepare_tool_sft_arrow.py`、`scripts/run_natural_language_value_search.py`、`scripts/taiji_run_with_heartbeat.sh`、`PROJECT_MEMORY.md`、`docs/ACTIVE_ICLR_STATUS.md` 及数据 manifest。不得绕过源数据状态和参考回放 gate。
- 将“Qwen 0.8B”明确解析为官方 **Qwen/Qwen3.5-0.8B**。由 Codex 在**实际部署环境**确认官方模型名称、具体不可变 revision SHA、Transformers / PEFT / tokenizer、工具模板与 LoRA 兼容性，并写入冻结 manifest。不得捏造 revision，也不得把此前 Qwen3-0.6B adapter 当成新模型可复用权重。
- Qwen3.5 的架构、工具模板或混合注意力模块可能与 Qwen3-0.6B 不同；要求真机加载与 **assistant-only loss、模型所拥有的 tool-call prefix、参数更新、显存与生成解析** 审计通过。仅训练所需语言部分，禁止不经验证地将旧 `AutoModelForCausalLM` 训练路径当作已兼容。
- **绝对不改研究范式**：不做“先预测完整 reactants，再生成 post-hoc 电子流解释”；不引入新 RL、RSI、图模型、检索 GT、MCTS 或更换电子流执行器作为这次任务的替代方案。
- 两轨**各自一套从相同预训练基座初始化、相互独立的模型**，独立数据、训练/验证划分、checkpoint、tokenizer 缓存、日志、输出、SHA 和 Stage-I→Stage-II 继承链。不要默认混合两个数据集或借用另一轨的已训练 adapter。

## 1. 锁定两轨与硬件

| 轨道 | 数据与要求 | 指定硬件 | 执行顺序 |
| --- | --- | --- | --- |
| **A：mech-USPTO-31k** | 从 PR #95 的主产物-only/first-use IMPORT 修复重新建完整可审计数据 | **8 × A100**，严格检查 CUDA 设备数和 GPU 型号 | 数据修复→Stage-I→抽样检验通过→Stage-II→相同抽样确认 |
| **B：FlowER** | 从冻结的严格可执行机制轨迹**重新**修正主产物-only 和 IMPORT，不能直接沿用旧 SFT/history | **8 × H20**，严格检查 CUDA 设备数和 GPU 型号 | 数据修复→Stage-I→抽样检验通过→Stage-II→相同抽样确认 |

两条任务可以独立并行，**各自的 Stage-II 必须等本轨 Stage-I gate**。有卡型不符、配置/模型 SHA 不匹配或必要数据未验证时，不自动改用其他卡型，必须报错并记录阻塞原因。

## 2. 两套数据都必须先改正确

### 2.1 mech-USPTO-31k

- 原 current-compiler 严格可执行 trace-view：**TRAIN 10,152 / VALID 1,319 / TEST 1,253，合计 12,724**。该集合不是原始完整 31k 的 TRAIN 24,959 / VALID 3,120 / TEST 3,120，结果必须标注分母。
- PR #95 已证明主产物起点重新构造可行：**12,706/12,724** 回放成功；18 条 unresolved（train 15、valid 2、test 1）。不能把这 18 条标为已通过、直接删掉后宣称完整 12,724 训练成功，也不能通过消除手性、形式电荷、显式氢来“修复”标签。
- 优先查清剩余 18 条：15 条端点立体信息不一致、3 条 root stereoisomorphism/atom correspondence 失败。能在保持化学正确性下修复的必须修复；本轮无法可靠重放的病例进入**显式 quarantine ledger**，保留 reaction ID、split、原因、原始/输出状态、SHA 和独立审计。训练可使用被明确许可的“已回放通过子集”，但必须在完整 split denominator 上报告可用率，TEST 评测必须单独说明覆盖界限。
- 公共起点只能是**单主产物**；在 GT 原子映射与机理中找到从初态移除、但未来电子事件要用的组分，放进“首次使用前”的离线 IMPORT 标签。支持后续状态再次 IMPORT，不能要求每一条反应无条件第一步 IMPORT；原始 FLOW、终止前体与真实执行器结果不变。将 endpoint-only context 与电子参与物分开记录。
- 逐条做参考轨迹从公共 product-only 起点的严格回放；逐决策验证当前原子、电子事件、原子映射重编号/芳香 Kekulé/显式 H/立体，以及真实公开训练 prompt。参考终态必须与原 GT 匹配，不能以离线 reference-aware 执行结果代替线上 executor 回放。
- 保留每 split 源 manifest、原反应总数、成功回放数、失败原因和 ID、全部决策数、IMPORT/FLOW/FINISH 分布、首步工具分布、原子/模型可见信息检查、SHA；训练数据要有单独明确的授权状态。

### 2.2 FlowER

- 数据源使用冻结的 `data/flower_inverse_tool_sft_action_delta_v1` **严格可执行**反应集合：TRAIN **257,167** / VALID **2,890** / TEST **28,967**。不要和官方反应级完整 FlowER TRAIN 257,171 / VALID 2,890 / TEST 28,971 混淆。
- 对 FlowER **从头核查主产物定义和起点**。不能把旧 `flower_natural_language_event_sft_v2` 或 `flower_natural_language_event_history_v2` 原样作为已经修正的训练源；必须用相同的“单主产物 + 缺失参与物 first-use IMPORT”口径重新生成 Stage-I 参考。
- 先检查源 `target_smiles` 的连通组分情况及外加离子/副产物；对能唯一识别的最大含碳主产物，保留作为公开起点。若最大含碳组分出现歧义，不要依靠 test 反应物、未来动作或私有条件选择“最能命中”的主产物：歧义反应进入显式原因统计和独立审核。
- 被移走的其他 product-side 组分不得出现在初始可见 SMILES 或执行器私有起点；离线参考监督中需要首次参与 FLOW 前的 IMPORT。现有其他 first-use import 逻辑保留；不得双重导入或重映射到冲突原子。
- 首先在确定性 VALID 抽样上严格回放，再全量 TRAIN/VALID/TEST 重新生成、执行和统计。任何转换未回放、电子事件结果不一致、端点失配、手性/电荷/氢问题，都必须有稳定 ID 的失败清单；禁止静默过滤、从完整 GT 中偷看缺失组分进行推理，或凭非等价 bond-delta 替换 reference electronic action。
- FlowER 自己的 Stage-II history 也必须从**修正后的 FlowER Stage-I 完整 accepted-prefix 数据**重建，保持每个决策之前只能看见之前已接受的动作。

### 2.3 数据验收规则

- 全部原始反应均计入固定分母，并在 manifest 中落实 `reference_pass + unresolved = original_count`；训练子集和完整 benchmark 分开报告。
- 明确处理未解决参考（修复或隔离）；如果仍有 unresolved，**不能直接覆盖 PR #95 原来 training_allowed=false 的原始文件**；需建立新版本 manifest、quarantine 审核/训练许可协议、可验证哈希，再让训练脚本消费对应已审计子集。数据用途状态由新增审计产物决定，而非手工改几个 JSON 字段。
- TRAIN/VALID/TEST reaction IDs 不交叉；严格先后引用、actor-visible 无 GT future/隐式导入；映射、tokenizer 和 model input parity 通过。
- Data gate 是正式训练的前置条件，抽样跑通不能替代全量 audit。出现源数据错误时 Codex 应在同一 Goal 中修复，不要用“先只跑能跑的几条”宣称正式完成。

## 3. 两阶段训练配置与过程

### Stage-I：State-SFT（充分监督训练）

- 架构：**Qwen3.5-0.8B**；各轨从自己独立的 pinned official base 加载。继续使用原 MechET 单步工具 SFT；训练所有合法 IMPORT、FLOW、FINISH 决策，loss 仅计 assistant tool-call spans；状态转换由 executor 执行。不得让模型直接预测 reference successor SMILES。
- 初始化建议为现有 PEFT LoRA / 训练器兼容改造，覆盖实际 Qwen3.5 语言模型中的 attention/DeltaNet/FFN 的目标层（以官方真实模块检查为准），冻结视觉模块或不实例化视觉输入。对实际训练输入做 tool template、assistant mask、梯度非零、target modules、量化/precision 的小规模真机 preflight，**通过后直接正式训练**，不是反复新开 smoke 设计项目。
- **默认完整训练 epochs：mech31k Stage-I ≥10 epochs，FlowER Stage-I ≥5 epochs。** 不准使用较小 `max_steps` 提前结束后声称完成多个 epoch。对大数据集记录真实消耗的 unique examples、optimizer steps 和有效监督 token；梯度累积与 effective batch size 记录。
- 正式按训练进度保存 checkpoint、optimizer/scheduler/RNG 恢复信息、base/tokenizer/data SHA，保留 latest 与 validation-best。训练充分但出现退化时选择历史验证最佳 checkpoint，不能只以最后一个 epoch 作为有效结果。
- 验证损失按 epoch 计算（可用既定小样本以控制成本），每 epoch 必须产生阶段报告；模型行为验证使用**冻结的同一批真实 VALID 反应**，禁止看结果后换样本。

### Stage-I → Stage-II 的轻量真实验证（每轨独立 GO/NO-GO）

- 在 Stage-I 训练开始前用固定种子、source IDs 和机制长度/进口角色分层冻结至少 **64 个 VALID 反应**；额外留一组 64～128 VALID 用于 Stage-II 完成后的独立确认。样本必须包括 root IMPORT、少量 root FLOW、不同 FLOW 复杂度，不能只选容易的 32 例。
- 同一批 VALID 需要同时做：
  1. **Teacher-forced next-tool-action**：以 GT 当前状态和之前已接受历史为条件，报告 root IMPORT 的 tool 和分子组分 exact、FLOW event/source-sink、执行器合法率、GT successor exact、FINISH 与全反应参考决策准确率；显式区分“执行成功”和“化学后继与 GT 一致”。
  2. **真正主产物-only greedy K=1 闭环**：模型只拿单一主产物；缺失物由模型自己 IMPORT，FLOW 自己生成，executor 更新状态直到 FINISH/拒绝/超预算；与 GT 比较核心结构前体、完整组分，记录首个错误深度、工具错选、过量 IMPORT、STATE_CYCLE 和每次拒绝。**不可把参考重放 100% 当模型成功率。**
- **进入 Stage-II 的固定最低门槛**：工具生成可解析率 ≥99%；有 root IMPORT 的样本中首步工具选择正确率 ≥80%、实际缺失组分 exact ≥25%；teacher-forced FLOW 化学 successor exact ≥60%；FINISH ≥80%；64 条闭环抽样至少 **4 条 GT 结构前体命中**，且没有明显的恒定首步 IMPORT/FLOW 模式崩塌、明显 GT 泄漏或 output truncation。
- 上述阈值是工程晋级 gate，**不是论文 claim**。若未达到：优先排查实际监督质量、root IMPORT 内容/分类、model template、训练长度与已有 checkpoint，并在预定范围内继续已固定的训练；不得修改数据分母/标签来人为放行。若仍失败，明确记录本轨 Stage-I 不达标并**阻止本轨 Stage-II**，另一轨可继续，不重新设计新算法。

### Stage-II：compressed-history Trajectory-SFT（门槛通过后自动接续）

- mech31k：**≥5 个完整 epochs**；FlowER：**≥3 个完整 epochs**，分别在**自己 corrected Stage-I 数据**的 accepted-action-prefix history 版本上继续 SFT。
- Stage-II parent 必须是本轨选出的 Stage-I best checkpoint，严格绑定 adapter SHA、model/tokenizer revision、数据与执行器协议；不可使用先前 Qwen3-0.6B R0、其他数据集 adapter、或者未运行的“训练完成”占位数据。
- 历史中只有当时已接受动作和执行器状态，不包含参考未来、隐式 GT 导入、失败动作（除非协议另有验证）或虚构的历史步。必须核对 SFT prefix 与在线重建压缩历史的 token parity。
- 每 epoch 继续轻量验证 loss 与真实生成；完成后与 Stage-I 在**同一批冻结 VALID IDs**上做 teacher-forced + 产品起点闭环的配对比较，并独立报告 root IMPORT 片段召回/精确率、FLOW 后继、FINISH、Top-1、错误深度。结果若退化如实汇报，不能因为 Stage-II loss 降低就声称方法改善。
- 两阶段到位后保存正式 model lineage / logs / checkpoint / validation 与受审计 TEST 计划；没有执行完整 TEST 就明确写“未执行”。

## 4. 必须持续输出训练进度（强制要求）

不能出现只有开始/结束的空白日志；**数据构造、tokenize、Stage-I、抽样验证、Stage-II、最终验证**都要在 POD stdout 与可持久化 JSONL 进度日志持续输出：

- **启动时**：模型精确名称和 SHA、语料名称、输入分母、通过/隔离数量、主产物-only gate、8 张 GPU 的设备名、CUDA/torch/Transformers/PEFT/RDKit、全局 batch、累积次数、learning rate、scheduler、总 epochs、预估总 optimizer steps、输出目录。
- **训练中每 10–20 次 optimizer step**：时间戳、轨道、阶段、global_step/total_steps、epoch 当前小数和 epoch 进度%、train loss、LR、grad norm、samples/监督 tokens/sec、已消费总 tokens、显存/显存峰值、当前 checkpoint 与最近异常统计；可观测指标为实际采集，缺失要标 NA，不能编造。
- **最少每 60 秒**：任务存活 heartbeat（复用现有 `taiji_run_with_heartbeat.sh`）；不能用“脚本还在运行”替代进度百分比。
- **每一 epoch 结束**：VALID loss、当前最佳 checkpoint 与选择标准、checkpoint SHA、实际累计 steps、何时发起下一次验证。
- **数据处理与轻量验证**：已处理/总数、失败分类、已生成 token cache 行数及零截断检查；抽样 VALID 逐例进度、root IMPORT exact、FLOW successor、FINISH、闭环 endpoint GT hits、当前累计率以及最终是否晋级 Stage-II。
- **完整结束**：分别输出两套可下载/可分析的结果 manifest、Stage-I/II checkpoint 路径+hash、训练配置、生效模型代码 SHA、分布式设备使用证明、关键指标和未解决问题；失败必须输出可定位 error 与最后一次成功 checkpoint，自动恢复不能破坏 optimizer/RNG lineage。

## 5. Codex 执行顺序与验收

1. 读取当前 `main` 和 PR #95 的实代码/原始 manifests，冻结 official 0.8B revision 与兼容性版本；检查 8×A100/8×H20 能否运行，未通过绝不伪报 ready。
2. **先修两套数据**。mech31k 审计现有 18 条并形成可训练的可靠监督版本；FlowER 从严格轨迹全量重建主产物-only first-use IMPORT；两套校验独立、总分母完整、随机抽样真实执行器回放通过；保存 SHA/status，消除 train/test 泄漏。
3. 真机完成 Qwen3.5 tokenizer / tool prefix / assistant-only loss / LoRA 前后向梯度兼容性检查；预 tokenize 两套数据并验证 zero supervised-token truncation、正确的每类动作分布。
4. 分别在**指定 8×A100 和 8×H20** 启动 Stage-I 完整多 epoch 训练，持续输出结构化进度、可恢复 checkpoint；验证并选定 Stage-I best。
5. 每轨执行冻结 VALID 的轻量 teacher-forced + product-start K1，完成预先固定的 gate；通过则**在同一 Goal 中无须再次征询，自动启动本轨 Stage-II**，失败则记录原因并对该轨 fail-closed。
6. 完成 Stage-II 多 epoch、同一冻结 VALID 配对验证、模型与运行日志/原始预测证据归档；不得只交一个初期 smoke 或一份计划书就宣称目标完成。

**明确禁止：** 自动 merge、本 PR 提交训练任务、绕过失败的 GT replay 或 tokenizer mask、将 FlowER 严格子集写成 Full、将旧 0.6B 结果冒充本次 0.8B、把模型可执行当 GT 化学正确、在 test 上调阈值、无结果预填模型准确率。

## 当前已知基线（不是本 Goal 的新结果）

- PR #94：已有 8-epoch Qwen3-0.6B Stage-I 在含两组分的产物侧输入上，对相同 32 VALID 反应达到 28/32 GT 结构端点；**主产物-only 输入为 0/32**。
- PR #95：新构造 32/32 GT 轨迹可从单主产物回放，原 0.6B 未重训因此仍 0/32；12,706/12,724 严格参考回放，剩余 18 条记录在 ledger。
- 上述数字是**诊断基线和数据参考验证**，与这次 0.8B 在新监督下的性能完全不同。本 PR 没有任何新的 GPU 训练和准确率结果。
