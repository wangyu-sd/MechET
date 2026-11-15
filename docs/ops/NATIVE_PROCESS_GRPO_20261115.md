# SFT 输入对齐与细粒度多步 GRPO

当前要修复的是两个耦合问题：v8 给策略的路线长账本、额外规则和动态工具与 History-89442 的监督输入不同；旧奖励虽然已有可信反应进展，但整条路线共享一个 advantage，不能区分哪些动作推动了路线、哪些在重复失败。新条件从修正数据训练的 Qwen3.5-0.8B Stage-II History-89442 重新开始，不继承旧 v8 的 optimizer 或 rollout。8B、EARHO、记忆框架不混入这一条件。

## 输入如何对齐

正常决策复用原 SFT 的 SYSTEM、当前执行器状态 USER、当前反应内 accepted-action 压缩历史、原始三个工具定义及 Qwen3.5 序列化。结束当前反应后，对下一个待合成前体重新构造原生单反应输入；完整多步路线和所有失败仍由环境保存。

失败重试只加入最近一次同反应、同映射状态下的失败 assistant 调用和真实 tool 错误回执，再放原生当前 USER。失败内容里的聊天边界字符转义，防止生成的 EOS 或伪造 role 标记重入 tokenizer 模板。这个失败上下文是明确的新增扩展；不声称原 SFT 已监督过这些失败案例。当前没有追加教师机理动作或重训失败 SFT。

全量输入审计使用实际修正 SFT VALID 的 2,751 个反应、21,355 条决策，逐条核对 SYSTEM/USER/tools 与 assistant 开始位置前的实际 tokenizer IDs。它验证的是格式和 accepted-prefix 对齐，不能证明每一条金标准 FINISH 都满足路线环境更严格的 productive/cycle 条件，也不是模型准确率评估。

## 采用哪种 RL

采用**可核验过程奖励 + 每个工具动作的 return-to-go + Dr.GRPO 固定归一化**，不训练新的 reward model/critic。

参考依据是 [DeepSeekMath §4.1.3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS3) 的过程监督 GRPO 和 [Dr.GRPO](https://arxiv.org/abs/2503.20783) 的奖励/长度偏差分析。这里把“步骤”定义为一次模型工具决策，奖励由执行器和 TRAIN 隐藏参考图核验；具体 baseline 是本项目的独立同目标 G4 位置基线，并非逐项复刻原论文。[GiGPO](https://arxiv.org/abs/2505.10978) 的同状态分组值得后续研究，但目前没有建立完整化学状态等价锚点，不把本实现叫 GiGPO。

## 奖励公式

令 L 为 TRAIN 参考路线反应数，S_t 为截止动作 t、位于可信祖先路径上的**去重参考支持反应边数**。只在有效 FINISH 提交了真实反应且参考核验通过后增加：

\[
q_t=0.2\min\{S_t/\max(L,1),1\}.
\]

IMPORT 合法、FLOW 可执行、重复提交和不可信祖先下面的反应，都不凭动作类型获得正分。[Rewarding Progress](https://arxiv.org/abs/2410.08146) 对进展与状态价值的区分启发了这个选择；本项目不使用该论文的学习型价值估计。

对执行器明确拒绝的非资源动作，累计错误权重 E_t：普通无效/格式错误/过早 FINISH 为 1，循环拒绝为 2，同一 occurrence、同一映射状态重复同一语义失败再加 1。说明文字不同不能逃避重复识别。定义：

\[
C_t=1-e^{-E_t/4},\qquad\eta=0.05/\max(L,1).
\]

总惩罚小于一个可信反应进展的四分之一；不会用合法动作数量稀释错误，也不奖励“故意失败后恢复”。令 b=1 表示参考支持的完整 AND 库存闭合；图无效时 b=0；其余 b=q_T。逐动作奖励为：

\[
r_t=(q_t-q_{t-1})-\eta(C_t-C_{t-1})+\mathbf1[t=T](b-q_T),
\quad\sum_t r_t=b-\eta C_T.
\]

完整成功至少 0.95，部分可信进展不超过 0.2，无进展只得到有界负成本。无效图的终端校正撤销此前进展。库存命中、参考支持、未核验的新路线分开报告；参考不覆盖不等于化学错误。资源截断和保护集命中仍为 reward=None，整组 G4 不更新，原目标保留在报告分母。

## 步级信用如何进入优化

\[
G_{it}=\sum_{u\ge t}r_{iu},\qquad
A_{it}=G_{it}-\frac14\sum_{j=1}^4G_{jt}.
\]

较早结束的兄弟路线在后续位置的 future return 为 0。位置 t 并不表示不同路线处于同一化学状态。每次动作生成的所有 token 共享该动作 A_it，prompt 和工具输出不参与策略损失。保留真实采样行为概率、clip=0.2、固定参考 KL=0.02、C=1024 和全局有效组×G4 分母，不做奖励标准差或动作长度缩放。包括自身的组均值产生与旧 mean-centered baseline 一样的 3/4 纯策略梯度比例，不是 RLOO。

原始行为 token、请求 ID、逐动作奖励、return、validator 账本和 SHA 摘要一起落盘，恢复时复核绑定。旧 v8 的观测输入不会被偷偷改成新输入再用于训练。

## 验证与证据边界

- 96 项相关 CPU 回归通过，覆盖原生输入、真实 tokenizer 边界、未开始/进行中的恢复、奖励守恒、重复/循环/无信任祖先、整组 censor 和独立数值/梯度参考；另在固定 Python3.11 训练依赖下复核核心测试。具体运行回执见随 PR 提交的 verification.json。
- 对旧 v8 已完成 18 轮、72 份 rank 文件、1,152 个 G4、4,608 条轨迹做只读重评分。参考闭合仍为 30，censor 仍为 614 组。标量奖励非恒定组从 65 增至 467，非零动作系数组 470。这只说明奖励能区分更多已有行为，不是新策略提升。
- 全量 SFT 审计回执绑定冻结数据 SHA 和最终源码 SHA，无模型 forward/optimizer。
- 正式条件：external TRAIN=23,730、固定 VALID=2,986、G4、seed17、1 epoch、4 H20；从修正 History-89442 actor 和同父固定 KL reference 开始；无 smoke/pilot。25/50/75/100% TRAIN 暴露后执行完整固定 VALID，初始基线按已冻结 deferred baseline 协议保留。

该条件保持 native HF 批次生成，不额外混入 vLLM 替换。去掉全路线重复输入可减少历史膨胀，但单动作 2,048 和物理上下文 32,768 容量边界仍需报告。部分进展奖励仍存在提前放弃的动机，最终以完整闭合和固定 VALID 判断是否有效，不以非零 advantage 数量判断。新观测只保留当前反应历史；长程路线记忆的收益需要独立输入适配实验。过程惩罚改变了目标，不声称势函数策略不变性。

## 代码与部署

新增生产模块是 `src/mechet/route_native_observation.py`、`src/mechet/route_process_reward.py`；调用点位于 `route_policy_episode.py`、`external_route_rl.py`、`prepare_external_route_rl.py`、`train_external_route_rl.py`。PR 中的 source/ 是最终部署源码的只读副本，便于直接引用行号；新模块也放在原路径。修改现有私有运行时的完整集成差异在 `patches/route_sft_process_grpo_20261115.patch`。

公开仓库此前没有上述四个运行时入口文件，不能声称仅拉取本 PR 就能独立运行整个训练栈。补丁基于保留的工作树提交 215687d（与 v8 已部署四文件逐字节一致），最终实现提交 bb4d9da；实际提交的冻结 source inventory 与数据/actor/runtime 绑定见 formal_submission.json。应用补丁需要同名基线运行时，不应把审计快照当另一套生产入口。
