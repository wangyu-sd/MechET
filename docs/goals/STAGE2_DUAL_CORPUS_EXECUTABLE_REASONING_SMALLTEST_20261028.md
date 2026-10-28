# MechET — Qwen3.5-0.8B 双 Stage-II 三项小规模验证 Codex Goal

状态：**新 PR 仅提交小测执行任务书；尚未生成新的实验结果，没有启动模型推理或训练。**
日期：2026-10-28。
基础分支：MechET PR #97，results/pr96-earho-4gpu-20261026，提交 10a5edadcb8a709ffa320adb67a1600e1460d1a2。
独立路线源：PR #84 / #89，只作为未合并的接口参考。PR #98 的路线级 RL 设计保持独立。

## 1. 目标和范围

直接用现有的两份 **Qwen3.5-0.8B Stage-II compressed-history Trajectory-SFT** 检查点，不新增训练、RL、RetroChimera 或第三方模型。分别在 mech-USPTO-31k 和 FlowER 进行三个小测：

- **P1 复杂反应恢复 / 组合诊断**：真正从主产物自主生成前体，按参考电子事件数、净键变化复杂度及 IMPORT 需求分层。不是严格 H2 unseen-composition 结论，因为这两份 0.8B 并未在 PR #80 的 H2 专用 TRAIN split 上训练。
- **P2 真实状态反馈干预**：在同一真实执行的参考前缀和同一冻结模型上比较 fresh state 与 stale observation，从而初步检验模型后续动作对真实执行反馈的依赖。GT 只构造实验的合法起点，后续不能进入模型提示、停止或选择。
- **P3 小型多步合成路线规划**：在同一 Stage-II actor 下比较 Independent complete-route sampling 和 reverse electron-flow tree search 的库存闭合、预算消耗、执行错误与资源截断。此项仅作 16 目标/轨道的可行性和趋势小测，不做正式显著性结论。

本 PR 不改已有模型或训练权重、不停止现有任务、不运行全量 H2/FlowER/460 路线实验、不改 MechET-paper；不得把之前 0.6B 搜索结果冒充本次 0.8B 小测结果。

## 2. 仅允许的两套 Stage-II 父检查点

| Track | 实际冻结父模型绝对路径 | adapter_model.safetensors SHA256 | VALID |
| --- | --- | --- | --- |
| mech31k | /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01/models/history/checkpoint-660 | cd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca | 1,319 |
| flower | /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/models/history/checkpoint-89442 | 2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453 | 2,890 |

基座 Qwen/Qwen3.5-0.8B，固定官方 revision 2fc06364715b967f1860aea9cf38778875588b17；必须使用 PR #97 的 Qwen3.5 tool prefix、first-use IMPORT、产物-only 私有初态、真实 executor、压缩历史。不得自动加载任何 Stage-III EARHO 权重，不得回退到旧 Qwen3-0.6B tokenizer、工具模板或 evaluator。

权威模型和数据协议来源：
- PR97 mech31k：docs/results/artifacts/pr96_earho_20261026/mech31k/contract.json
- PR97 flower：docs/results/artifacts/pr96_earho_20261026/flower/contract.json
- 当前 PR97 可复用的代码：src/mechet/qwen35_protocol.py、src/mechet/pr96_evaluation.py、src/mechet/pr96_eval_runtime.py、scripts/evaluate_pr96.py。
- 注意 scripts/evaluate_pr96.py 是正式验证入口，含训练硬件门控；小测应新增独立、合法的推理入口，不伪装设备类型，不修改训练保护。

mech31k VALID 原始绝对路径：
/aaa/fionafyang/buddy1/whaleywang/MechET/data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01/inverse/valid.jsonl
/aaa/fionafyang/buddy1/whaleywang/MechET/data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01/state_sft/valid.jsonl

FlowER VALID 原始绝对路径：
/aaa/fionafyang/buddy1/whaleywang/MechET/data/pr96_flower_mainonly_source_arrows_v2_20261023_06/inverse/valid.jsonl
/aaa/fionafyang/buddy1/whaleywang/MechET/data/pr96_flower_mainonly_source_arrows_v2_20261023_06/state_sft/valid.jsonl

任何数据/模型/stock 文件先按各自 contract SHA 校验；不可擅自降低手性、电荷、同位素和显式氢的评分要求，不得将两轨训练集合并或将不同 VALID 准确率混成一个分母。

## 3. 固定样本：不看预测效果选题

- P1：每轨最多 **64 个 VALID reaction IDs**。先按照参考电子 FLOW 次数 1–2、3–4、5+ 以及是否需要 IMPORT 建立真实分层，再按 SHA256(seed=17, source ID) 固定次序取样，在源数据可支持范围内尽量平衡。不能按模型成功或旧 bad case 挑选。若层不足，报告各层真实数量与选中数量，不人为补造。
- P2：每轨从上述 P1 样本里挑选最多 **32 个**，要求参考 GT 前缀可以真实执行出一个后继，且至少存在一次后续需要模型决定的动作。选择只依赖已冻结的参考合法性，不依赖本模型表现。达不到 32 则明确真实 n，不从别处选择成功例凑数。
- P3：从原有 460 个目标的 **四个固定场景各 4 个目标**，合计 16，按照固定 ID + seed17 的哈希顺序选择，**不能使用旧 MCTS 已完成的 200 个成功报告作为筛选条件**。两模型完全相同的 16 个目标和原场景 stock。

在进行任何真实模型调用前输出不可变 selection_manifest，绑定所有 ID、所属源、参考复杂度层、模型 SHA、原始数据 SHA、原料库 SHA、固定随机种子和选择算法。

P1 只进行独立 K=1 主产物自主演化，最多 **16 次工具动作/目标，每次最多 512 个生成 token**，正常 FINISH 提前停止。达到预算记资源上限/删失，而不是自动判无效化学。已有同一 Stage-II checkpoint 与完全相同协议的 VALID raw traces 可以只读复用；否则需要真实推理，不允许借用 EARHO after 或旧模型输出。

## 4. P1：复杂度和组合性小测

以每轨 64 条 reference VALID 为单位，保存完整自主生成过程和输入输出。

主要指标：
1. 真实结构前体 exact@1、完整前体混合物 exact@1、正常 FINISH 率、执行错误/截断率。
2. 按 **参考 FLOW 事件数量 1–2、3–4、5+** 的恢复率；模型生成的 FLOW 次数单独报告，不能拿它冒充参考复杂度。
3. GT 是否需要 IMPORT；根部导入/首次导入后的化学后继表现。
4. 真实重原子成断键数量、环变化、反应中心复杂度：只在可验证映射与结构变化时统计。无法映射或缺失的保留 unknown。
5. 若能从该轨自己的实际 TRAIN 反应重建同一语义的完整电子程序签名，做 TRAIN seen vs unseen 的**描述性分析**；不能称其为 PR #80 的严格 H2 unseen-composition 评测。没有可靠签名就省略此分层，不应另建 H2 训练。

输出真实分层 n、hit、误差类型，以及至少一张实测数值图。样本太少的层只报告描述性计数，不进行效应推广。

## 5. P2：真实执行状态反馈干预

两臂的真实 executor 状态都必须是**同一个真实执行到的 s(t+1)**。此前 GT 前缀只用于达到这个共同起点，模型只能看见合法的过去动作/状态，没有未来 GT。

- Fresh：向模型提供真实更新后的 CURRENT STATE / MOLECULAR INVENTORY 和原有合规压缩历史。
- Stale：向模型提供上一步旧观测 s(t) 的当前状态/库存文本，**executor 仍真实处于 s(t+1)**；模型后续发出的动作照常经过真实执行器。历史不能偷偷泄漏 s(t+1)。这是观测不一致性的压力测试，不是学习到化学因果定律的充分证据。
- 可选 history ablation：保留真实 current state，只删除可合法删除的历史摘要；如果实现成本很高，就只完成上述两臂，不扩展实验。
- 同一个 source ID、相同 GT 已执行前缀、相同随机种子、同一冻结模型和 K=1，最多 16 次后续动作/每次 512 tokens。不能把两个臂恢复到不同的私有状态来制造差异。

主要指标：下一步执行接受率、化学后继与参考一致率（仅已验证可比较情况）、FINISH 和完整端点命中率、无效状态和历史相关错误、tokens/决策数。输出双臂相同 ID 的配对表（both/fresh-only/stale-only/neither）及原始动作轨迹。

如果无法准确实施 state/history 观测控制，应标 BLOCKED；不得拿普通 teacher-forcing 与 closed-loop 两份不同实验伪称“真实反馈消融”。

## 6. P3：真实库存闭合多步路线小测

目标集、标准化与 frozen stock 来源：MechET PR #84，configs/eval/frozen_route_benchmarks_20261013.json、src/mechet/reverse_route_network.py；独立采样源码参考 PR #89 的三个未合并脚本：independent_route_control.py、independent_route_runtime.py、independent_route_cli.py（均在该 PR 的 scripts 目录，属于跨分支参考，不是本 PR 当前文件）。

**重要：这些旧 route 程序未存在于当前 PR97 stacked branch，且绑定旧 0.6B Stage-II。** Codex 只移植完成本试验必要的受审接口，同时验证新 0.8B 官方 Qwen3.5 token/tool 模板、first-use IMPORT、状态执行、whole-salt stock semantics 和权限。不得直接执行旧 checkpoint 配置，也不得合并旧 PR 数百文件。

固定方法：
- Independent full-route episodes：当前目标的多个完整尝试彼此独立，不共享已采样电子动作/搜索状态；每一个非库存结构前体均必须递归处理。
- Reverse electron-flow tree search：从同一产物与 frozen stock 出发，在真正可执行的电子状态上扩展、复用和分叉。跨目标的生成缓存若无法公平计费，则禁用或逐一核算，不凭缓存节约声称纯算法优势。

**每个目标×每个方法：最多 12,000 个新生成 output tokens、最多 128 次尝试工具动作；两个真实上限满足任意一个即停止或按规范记删失。** 两模型×16 目标×2 方法构成最多 64 个目标-方法组合。独立采样运行多个 episode 直到此预算、发现首条完整路线或本轮允许的真实资源中断。两种方法的 sampling temperature、工具协议、随机性、输入、stock 和终止定义一致。同时记录 prompt tokens、forward calls、实际执行器成本及时间，不能仅因 output-token 配额相同就宣称 FLOPs 或 GPU 秒数严格匹配。

库存闭合的唯一正例条件：至少一条非零步模型生成反应，且**所有必需结构前体叶节点均进入本场景冻结原料库（AND closure）**。起始目标已经在原料库则 separately already-in-stock，不计新合成路线。生成了一个可购买的前体但另一个未闭合，不算成功。正式 TEST 的参考路线、旧 MCTS 搜索结果或 GT 前体均不得参与推理和候选挑选。

对比按同一 ID 的实际 route_solved、首次闭合 generation tokens、反应边数、最大合成深度、工具调用、资源截断/无效终止；带完整 actor/source/work manifest。由 PR97 TRAIN 交叉审核这些外部目标及必要来源重叠；若重叠未知，只标为功能诊断，禁止声称严格外部分布外泛化。

这是小样本可行性试验，n=16/track 不支撑正式 NMI 方法优越性的统计宣称。

## 7. 交付实现、证据和绘图

Codex 在本 stacked PR 内优先做完整数据校验，再实现单一增量小测入口，建议新增名为 eval_stage2_dual_smalltest.py 的入口（实现时放入 scripts 目录），支持 prepare / run / aggregate，按 track=mech31k|flower 和 study=P1|P2|P3 选择。以上是待实现接口名称，不声称已经存在。

工程约束：
- 复用 PR97 的 tokenizer/model/executor/eval，加载真实冻结 Stage-II checkpoint，不更改训练 hardware guard；如果推理硬件不足或数据挂载不存在，给出 BLOCKED 状态而不是伪造结果。
- GPU 生成与 CPU 化学执行可按原状态顺序滚动调度，但每个模型动作必须接收真实执行后的必要状态，不能为了加速引入 GT 或跳过电子事件。
- 每次执行保留 raw prompt、input/output token IDs、模型输出、解析结果、工具错误、真实状态前后、每目标累计模型预算、GT 来源仅限评分、source / checkpoint SHA。
- 训练、验证和现有平台作业完全不变。用户允许的是这次小规模评测的范围；如实际 Taiji 需要额外资源申请或明确任务授权，必须如实报告 BLOCKED，不自动启动大型任务。
- 不生成无数据的拟合分布、预期准确率或凭空成功图。

**全部新产物放在独立服务器绝对目录：**
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/stage2_dual_corpus_smalltest_20261028

按轨道保存 selection_manifest.json、raw_calls.jsonl、episodes.jsonl、metrics.json、source_and_checkpoint_sha.json、comparison.json；P2 fresh/stale、P3 independent/tree 各有独立子目录；summary 下提供真实计数、可复现命令、缺失与删失记录。

轻量汇总与真实图（仅有实际模型输出时）纳入 PR：
- P1：反应复杂度分层的原始计数 / 恢复率图
- P2：Fresh vs stale 同目标成对比较
- P3：实际路线发现 vs 生成 token / 两模型各自分组比较

图片建议 matplotlib/RDKit，PDF/SVG/300dpiPNG；不得用漂亮图替代小样本的科学限制。

## 8. 必须完成的最小验收

1. 两个模型真实 adapter SHA 与 Qwen3.5 revision 校验通过；不存在轨道串用或误载 Stage-III。
2. 真实 VALID 先冻结 P1 每轨最多 64 IDs、P2 每轨最多 32 的合法 GT prefix；P3 真正四组各 4 固定目标，与旧实验成功状态完全无关。
3. 所有推理各有可验证 raw model calls 和真实执行事件；CPU fixture test 与 GPU 真生成证据严格分开。
4. P2 两臂起点真实执行状态相同且无未来 GT 提示；P3 相同模型的两方法按实测 token/工具预算严格对比。
5. 结果必须分开报告每轨数据、真实分母、缺失、拒绝动作、资源删失与化学未知；不报告新模型未运行的曲线、不将临时 pilot 当 NMI 正式 Test。
6. 新增必要软件测试，核对 Qwen3.5 工具协议、原料库全叶闭合和 source hash；不改 paper 或 Stage-III 训练设计。

**结果判断：** P1 看现有 Stage-II 复杂反应恢复是否随长度退化；P2 看真实执行反馈对续推是否有可测影响；P3 看新 Stage-II 的多步执行是否可用、树搜索相对独立采样有什么实测差别。无效或全零结果亦必须真实记录。若后续要扩成 NMI 正文，再用完整严格数据规模与正式匹配预算验证。

**再次声明：本 PR 提交的是小测目标与验收规范，不是完成的模型结果；尚未执行 GPU 推理或训练。**
