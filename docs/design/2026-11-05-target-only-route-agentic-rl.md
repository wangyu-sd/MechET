# MechET 主产物条件多步路线 Agentic RL：方法、公式与实施设计

**状态：设计评审稿，2026-11-05。本文 PR 仅增加文档，不启动训练、不切换正式推理检查点。**

## 1. 建议采用什么方法

建议以 **GRPO 家族中 Dr.GRPO 的固定归一化目标作为稳健主线**，以原始 GRPO 和已有 RLOO 原型建立可解释对照，以 **Qwen GSPO** 做前沿方法对照。DAPO 的非对称裁剪和有效组采样分别作为独立消融，不将所有技巧叠加后统称为 GRPO。

理由：我们已有原生工具执行器和路线级结果核验，缺的是把完整轨迹反馈稳定地传回电子决策。现阶段不增加价值模型。算法采用已有论文与公开实现；研究贡献应落在“主产物条件、逆向电子流决策、完整 AND 路线反馈和可核验机理证据”，不能把已有优化器重新命名为新算法。[1–5]

| 方法 | 在本项目中的角色 | 有依据的部分 | 需要在 MechET 验证的部分 |
|---|---|---|---|
| RLOO | 已有本地原型及尺度对照 | 留一法回报基线，无 critic [1] | 当前 clip/KL/多轮适配的效果 |
| 原始 GRPO | 标准方法对照 | 组内相对优势与裁剪目标 [2] | 稀疏路线回报、长度和难度归一化影响 |
| Dr.GRPO | 推荐稳健主线 | 去除组内标准差和变长响应归一化 [3] | 固定尺度在不同路线长度上的优化行为 |
| GSPO | 前沿对照 | 序列似然比与序列级裁剪，Qwen 公开方法 [5] | 多次工具调用组成一条路线时的适配；不能外推 MoE 收益到本项目稠密 0.8B 模型 |
| DAPO 技巧 | 第二阶段独立消融 | Clip-Higher、动态组采样等 [4] | 有效组条件筛选带来的采样分布变化 |

**现有代码不是已完成的 GRPO/GSPO 实现。** 本地提交 `54f9760`、`2056823` 实现的是 RLOO 式优势、全轨迹 token-sum loss、clip 和固定参考策略 KL。它可用于复用采样、执行、持久化和恢复接口；新方法仍需实现和验证。尚无本轮 RL 后的化学测评结果。

## 2. 任务边界与信息隔离

初始输入只给目标主产物 (x)。后续模型观察当前待合成分子、自己执行得到的原子守恒体系和自己的动作历史。初始前体、参考路线、参考中间体和标准机理动作均不进入策略输入。

环境状态包含全部待闭合前体 occurrence、当前反应体系、历史反应图和冻结库存；模型观察只是这个状态的公开投影，因此不假设它拥有完整可观测的全局状态。待合成义务的基础调度沿用确定性控制器，不额外引入一个“规划 LLM”。

最小化学决策保持为逆向电子流和显式原子补全 IMPORT。`finish_trace` 只结束一个反应的机理程序；之后每个非库存结构前体 occurrence 都成为新的待合成义务。一个反应可以有多个电子动作，一条路线可以包含多个反应，两个计数必须分开。

训练奖励端与策略端采用两个数据平面：

| 策略/执行器可见 | 只允许奖励端读取 |
|---|---|
| 主产物、当前自生成体系、公开工具定义、自生成历史、公共库存查询 | TRAIN 参考前体、参考反应边、参考路线拓扑及分段进展基准 |

参考信息只在轨迹停止后打分，不用于填入工具参数、teacher forcing、选择标准下一步、补齐失败后缀或控制停止。外部路线无需编译成伪造的电子流标签；新训练的动作级 GT CE 权重固定为零。

训练使用独立策略 rollout；部署测评继续使用现有 neural-guided MCTS。不能把 MCTS 挑选的赢家当作独立 on-policy 样本直接套用下文目标。

## 3. 外部数据、真实分母与必须先修的数据划分

### 3.1 来源及当前 v1 快照

来源为 USPTO 专利反应中抽取、连接出的 PaRoutes 路线，经 RetroCast `v2026-06-05 / reaction-holdout-n1-n5` 整理。[7–8] 路线来自专利反应，不等于每一条抽取树都经人工完整实验核验。

我们下载的是上游 **TRAIN156,723 条路线记录**，不是将上游验证集混入 TRAIN。上游训练文件 SHA256：

```text
bae4c87e2944acd78afd5be02ffd34f41a175c046ec97f9c41daaeb19ea5b6e2
```

当前库存为已按 RDKit-2026 审计的冻结 URSA；其文件 SHA256：

```text
b1c42cc4613d5e365c73300ba53a1f7a8dc7ce3f471f423b6685511d9cab9b35
```

逐条核算如下；这些是数据构建统计，不是模型实验结果：

| 处理结果 | 路线记录数 |
|---|---:|
| 保留的库存裁剪后多步路线 | 30,370 |
| 重复路线/分支排列等价 | 1,290 |
| 主产物已在库存 | 5,369 |
| 参考路线在当前库存下不能闭合 | 87,243 |
| 购买中间体后实际只剩一步 | 11,545 |
| 原参考本身不足两步 | 20,886 |
| 包含受保护测评分子 | 20 |
| **合计** | **156,723** |

保留路线按目标合并为 30,276 个目标；v1 的根目标划分为 TRAIN27,290 / VALID2,986，私有参考反应边 75,394 条。保留完整立体/电荷身份、前体多重集和分支排列不变的路线签名。库存裁剪后须仍有至少两个实际反应，不能以电子动作深度冒充路线深度。

当前 R1/R2 的460条和 SimpRetro 的172条公开案例，共619个不同化学身份，已经作为受保护集合从参考图中排除。该结论只覆盖本轮参考数据中的这些身份；不是对历史预训练、所有测评反应或所有同专利样本均无泄漏的声明。

### 3.2 补充审计发现的问题：v1 不能直接用于最终模型选择

根目标不重叠仍不够。对 v1 私有数据库完整审计发现：

- **67 个 VALID 根目标出现在 TRAIN 参考图内部，涉及108个 TRAIN 任务。** 训练递归到该分子时，策略可能实际学习它的反应。
- TRAIN/VALID 共享 **1,348 个参考反应签名**。

因此，先前“目标独立”应严格理解为根目标不交叉，不能推广为中间体/反应独立。**正式训练和模型选择前，必须发布新的 v2 条件，不能沿用 v1 分母宣称已解决。**

v2 的固定处理顺序：

1. 冻结 v1 的2,986个 VALID 根目标及其参考反应身份，作为本轮模型选择候选集。
2. TRAIN 中移除包含这些 VALID 根目标或 VALID 反应身份的参考证据；重新检查原目标是否仍有完整、库存闭合、至少两反应的可信 witness。失去全部 witness 的任务排除并单列原因。
3. 重新计算签名、公开目标、私有奖励库和全部分母；**v2 的 TRAIN 数量目前未知，不能预填27,290或将108直接当作最终排除数。**
4. 验收 TRAIN/VALID 根目标、中间体目标暴露、参考反应交叉均符合协议；共享公开库存分子本身不等同于反应标签泄漏。
5. 运行时对 TRAIN rollout 加入 VALID 根目标和正式测评目标保护；若策略自主触及受保护的非库存义务，标记删失并报告，不能悄悄借用参考答案。
6. 保留专利/来源/route-step provenance，报告专利和骨架交叉；如另做专利分组或骨架外推划分，作为新的明确条件，不覆盖主协议。

运行阶段的保护集合必须区分用途：TRAIN 为“正式测评目标＋VALID 根”；VALID 为“正式测评目标”，不能把正在评估的 VALID 根自身挡掉。保护目标触发与资源不足分别记为 `protected_censored` / `resource_censored`，两者都移除整个 G 组的 learner 项，但原目标 ID 仍留在覆盖率和成功下界分母中。

当前 v1 的 manifest SHA：`654893dfba2d492db0ac17e854f4502124917ae798c2dec6ff61d129940d852b`。数据审计证据位于仓库工作区的 `outputs/analysis/external_route_rl_20261105/`，其中 `design_validation_overlap_audit.json` 记录上述新增交叉检查。模型文件、数据和缓存只存共享 Ceph；本 PR 不上传这些文件。

### 3.3 长尾来源

当前样本来自原始专利路线分布，没有另造长尾路线或做过采样。v2 需要发布库存裁剪后的反应深度、总反应数、AND 分叉数和反应频次分布；按深度2/3–4/≥5及线性/汇聚路线分别报告。可从同一 TRAIN 参考图提取真实连通子路线作为后续课程，但必须另报新增目标和权重，不能将随机拼接的反应链当作真实长尾证据。

## 4. 轨迹、概率与 token mask

一次工具动作本身是一段模型生成的 XML/token 序列。完整路线轨迹记为

$$
\tau_i=(a_{i,1},z_{i,1},\ldots,a_{i,T_i},z_{i,T_i}),
\qquad h_{i,t}=\operatorname{render}(\text{公开状态与自生成历史}).
$$

工具结果 (z) 由环境产生，不是模型输出。令 (m_{i,u}=1) 仅标记实际生成的 assistant 动作 token；system/user、RDKit 结果、padding 和未来参考信息的 mask 均为零。

保持当前未截断分布的 categorical 采样：`temperature=1, top_p=1, top_k=0`，每条独立轨迹独立随机流。mask 包括实际采样的 assistant 终止 token，以及最终被执行器拒绝的畸形输出；不能在零回报失败轨迹中只保留合法 token，或仅对成功轨迹更新。资源/保护删失仍按整组移除。

$$
\ell_{\theta,i,u}=\log\pi_\theta(y_{i,u}\mid h_{i,u}),\qquad
L_i=\sum_u m_{i,u},\qquad
\log p_\theta(\tau_i\mid x)=\sum_u m_{i,u}\ell_{\theta,i,u}.
$$

最后一个等式省略了与策略参数无关、且在双方相同的环境转移因子。跨反应时会重新构造局部提示；**不能把多个独立工具请求拼成一段任意 chat completion，再用错误的上下文重算概率。** 应逐决策保存原始 `prompt_token_ids`、`generated_token_ids` 和行为 logprob，按 `trajectory_id` 聚合。

采样策略 (\pi_b) 在一轮收集内冻结；KL 参考策略 (\pi_{ref}) 在整个训练条件中冻结。这两个策略角色不能混淆。

## 5. 隐藏路线奖励

令 (B(\tau)) 表示从根开始重建的 occurrence 级 AND 图完整闭合，且叶子多重集全部属于冻结库存。要求每条提交反应有合法电子执行证据、至少一次有效 FLOW；单纯 IMPORT、空 FINISH、自环和断开的库存分子不构成成功。

令 (V(\tau)) 表示每个生成反应都得到隐藏 TRAIN 参考端点支持。完整奖励为

$$
R_{\mathrm{full}}(\tau)=\mathbf1[B(\tau)\land V(\tau)].
$$

不要求生成机理与参考机理一致；外部数据也没有提供这类动作真值。支持的判定使用主产物和结构前体多重集的化学身份，保留电荷与立体信息。

### 部分反馈必须沿可信祖先连通

初始化可信 occurrence 集为根。只有“父 occurrence 已可信，且本反应得到参考支持”时才将其子 occurrence 标记可信。令 $E^{trusted}_{done}$ 为这样得到的不同反应边，$M_x$ 为冻结参考 witness 的反应数基准：

$M_x$ 在库存裁剪和 v2 清理后仍完整有效的原始 witness 集合中，取最少总反应 occurrence 数；多个最小 witness 用规范路线签名字典序打破平局并写入 manifest，不隐含求解所有可能重组路线的全局最优值。部分边去重键为“规范主产物＋保留重复项的规范前体多重集”，不依赖动作文本或原子映射号；库存、身份规则及该 witness 在同一训练条件内固定。

$$
P(\tau)=\min\!\left(\frac{|E^{trusted}_{done}|}{\max(1,M_x)},1\right),
\qquad
R(\tau)=
\begin{cases}
\bot,&\text{资源或保护目标删失};\\
0,&\text{伪造、不连通或循环证明};\\
1,&B(\tau)\land V(\tau);\\
\lambda P(\tau),&\text{其余已观察停止，}\ \lambda=0.2.
\end{cases}
$$

**这需要修正现有原型：** 原型会统计生成根图中匹配的参考边，但尚未阻止“越过一个不受支持的祖先后，再在下游拿部分奖励”。新实现必须按可信 occurrence 传播。非法下一动作可保留之前的可信进展；资源删失则不给部分奖励。

全路线必须同时闭合所有相同前体的重复 occurrence。去重可用于部分进展计数或计算缓存，不能取消真实义务。

目前完整奖励验证的是“参考支持的库存闭合”，不是任意新反应的可行性。对新颖且库存闭合但未获参考支持的路线，单列 `unsupported_closed`，不宣称化学错误，也不授予完整正奖。独立正向/文献验证器是另一个需要验证的后续模块；把同一电子编辑简单反放回去不是独立化学验证。

## 6. 推荐主线：Dr.GRPO 式全轨迹目标

对每个目标独立采样 $G=4$ 条完整轨迹。只对所有回报均可观察的组计算下面优势，并停止对奖励本身求梯度：

$$
\bar R_x=\frac1G\sum_{j=1}^G R_j,\qquad
A_i^{center}=R_i-\bar R_x,
\qquad
\rho_{i,u}(\theta)=\exp(\ell_{\theta,i,u}-\ell_{b,i,u}).
$$

定义裁剪项

$$
\Psi(\rho,A)=\min\{\rho A,\operatorname{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A\}.
$$

主线设置 $\epsilon_l=\epsilon_h=0.2$，不做组内标准差缩放，以固定 $C=1024$ 作纯梯度尺度：

$$
\mathcal L_{Dr}=-\frac1{NGC}\sum_{x,i,u}m_{i,u}
\left[\Psi(\rho_{i,u},A_i^{center})-\beta\kappa_{i,u}\right],
\qquad \beta=0.02.
$$

$N$ 是所有 rank 上可用组的全局数量，$G$ 固定。**$C$ 是归一化常数，不是总 token 上限。** 本项目将这一固定尺度目标适配到完整多工具轨迹；不能把它描述为完全未改动的单 completion 实验。

保留已测试的、在行为状态分布上计算的 token 级 KL 正则项：

$$
d_{i,u}=\ell_{ref,i,u}-\ell_{\theta,i,u},\qquad
\kappa_{i,u}=\rho_{i,u}\,[\exp(d_{i,u})-d_{i,u}-1].
$$

行为/参考 logprob 是常量；KL 中的 $\rho$ 保留梯度。这个正则项不等于对当前策略全部路线状态分布精确积分的 KL，也不声称修正了所有轨迹状态分布偏移。库的 KL 校正开关及 stop-gradient 位置必须与上式逐项对齐。[6]

### RLOO、GRPO 与主线的区别

$$
A_i^{RLOO}=R_i-\frac{\sum_{j\ne i}R_j}{G-1}
=\frac{G}{G-1}(R_i-\bar R_x),\qquad
A_i^{GRPO}=\frac{R_i-\bar R_x}{\sigma_x+\varepsilon_{num}}.
$$

因此，不做标准差缩放时，RLOO 和组均值中心化只差常数。比较时须控制有效 PG 尺度及相对 KL 权重；不能将这个系数差当作重大算法创新。原始 GRPO 对照保留按响应 $L_i$ 平均的目标；其与固定 $C$ 的差别必须明确记录。[1–3]

标准差和数值稳定项对齐固定 TRL commit 的 `nanstd` 和 `+1e-4`，不自行换成另一种总体/样本标准差约定；软件验收同时比较归一化优势与最终梯度。

四条回报全相同，RLOO 和 GRPO 的策略梯度优势都为零。不能把重复候选“增加权重”解释为创造了新的学习信号。

## 7. 前沿对照：GSPO 的完整轨迹适配

先实现不加 KL 的论文形式，并与固定版本公开实现做数值/梯度对齐。[5–6] 令优势使用标准组归一化，序列比例定义为

$$
s_i(\theta)=\exp\left[\frac1{L_i}\sum_u m_{i,u}
(\ell_{\theta,i,u}-\ell_{b,i,u})\right],\qquad
\mathcal L_{GSPO}=-\frac1{NG}\sum_{x,i}\Psi(s_i,A_i^{GRPO}).
$$

这里一个“序列”对应一条路线的全部 assistant 动作 token，而不是每个 XML 动作各自独立的一条序列。$s_i$ 是完整联合比例的几何平均，**不是未归一化的精确整条轨迹重要性权重**。这种多轮环境适配是本项目要检验的假设，不是原论文已经验证的逆合成结论。

GSPO 的裁剪尺度与 token 比例不同。第一组固定 $\epsilon_l=3\times10^{-4}$、$\epsilon_h=4\times10^{-4}$，采用 TRL 官方 paper index 给出的论文复现配置，不直接复用0.2。[6] 本项目暂不额外搜索裁剪参数；任何后续 VALID 调参范围、完整条件数与计算预算须在新条件中登记。

### 一个必须写进实验设计的等价性

当 $\theta=\theta_b$、裁剪未触发、优势相同且使用相同响应长度归一化时，原始 GRPO 与 GSPO 的首步梯度相同：

$$
\nabla\mathcal L\big|_{\theta_b}
=-\frac1{NG}\sum_{x,i}\frac{A_i}{L_i}
\sum_u m_{i,u}\nabla\log\pi_\theta(y_{i,u}\mid h_{i,u}).
$$

当前原型每次收集只进行一次全量梯度累积后更新。**只把 ratio 换成 GSPO 并继续这个单步设置，不能证明 GSPO 的裁剪/离策略稳定性优势。** 有区分力的对照采用相同冻结行为样本上的 $\mu=2$ 更新，并让 GRPO/GSPO 都使用相同 $\mu$、奖励、优势、采样量和验证集。

TRL 官方复现配方还提供将一次收集分成4个 learner mini-batch 的调度，使后续更新相对行为策略偏离。本项目先采用相同样本上的两次完整更新作受控比较；这是明确登记的调度适配，不声称逐项复现论文的计算预算。若换成4个 mini-batch，必须同时修改匹配控制组并保留全局归一化语义。

采样引擎与 learner 的 logprob 校准只在第一步、同一行为参数下进行；第二步以后真实策略变化应进入 ratio/KL 监控，不能沿用“漂移超过阈值立即报错”的原型检查将正常更新误判为引擎错误。

若加入固定参考 KL，命名为 `GSPO + common token-KL` 的任务适配，另做消融；不能声称它就是原论文不加 KL 的形式，也不能将序列比例广播的 KL 当成已证明无偏的 token 反向 KL。

### 内存受控且精确的跨工具请求梯度

不能为了 GSPO 将整条长路线的所有激活同时留在 GPU，也不能把每个工具动作各自的 ratio 当成路线 ratio。每个 learner 更新固定当前参数，采用两遍计算：

1. 第一遍 `no_grad`：逐个原始提示重算当前 logprob，将整条路线的 log-ratio **先求和再除以总 token 数**，得到 $\Delta_i=\log s_i$，计算真实标量目标及 clip gate。
2. 计算并 detach 整条路线的系数：

$$
c_i=\frac{\partial[-\Psi(\exp\Delta_i,A_i)/(NG)]}{\partial\Delta_i}.
$$

3. 第二遍只保留一个决策/micro-batch 的计算图，累积以下梯度代理：

$$
\widetilde{\mathcal L}=\sum_i\frac{\operatorname{stopgrad}(c_i)}{L_i}
\sum_u m_{i,u}\ell_{\theta,i,u}.
$$

它在该固定参数点的梯度与完整 GSPO 经验目标相同，但标量值不同，日志须保存第一遍的真实目标。两遍之间不更新参数；关闭 dropout，保持数值模式、提示和 token mask 一致；每次 optimizer step 后重新计算所有系数。零生成 token 的路线在除法前判为删失/无效。

单元验收以不同长度、多个独立提示组成的合成轨迹，对比“保留全部图直接反传”和“两遍分块反传”的参数梯度，包括正/负优势、裁剪开启与边界值。

## 8. DAPO 的取舍、删失与重复样本

第一阶段保持目标均匀覆盖，记录有效回报差异组比例、策略熵和不同可执行化学后继比例。DAPO 的 `epsilon_high=0.28` 与原始对称裁剪的比较单独做，必须挂在相同的 token-ratio、$\mu\ge2$ 条件上，不能挂在 E2 的单次全量梯度更新上形成无效消融；动态有效组采样再单独做。[4,6]

动态采样不能改变 `G=4` 的含义：四条必须是独立生成，不能从更多轨迹中只挑四条成功者，再当作原分布样本。不得用 benchmark 结果驱动 TRAIN 目标选择。有效组过滤会改变被优化的目标分布；零优势组的 PG 为零，不代表删除这些组对 KL 和整体目标也没有影响。

资源删失同样不是免费的无偏处理。若 $I_x$ 表示该组所有轨迹均被完整观察，则原型实际优化的是类似

$$
\frac{\mathbb E[I_x\,\mathcal J_x]}{\mathbb E[I_x]},
$$

而非无条件全目标期望。长路线更易删失时会产生选择偏差。主报告必须同时给全目标成功下界、删失率和按深度分层的覆盖；不能只报告保留组的成功率。本文不凭空假设已知删失概率并使用不可靠的逆概率修正。

没有总 token、反应深度或总动作数的化学截断；物理上下文/单次输出容量导致的截断均为资源删失。系统作业被中断时保留未完成状态，不转为化学负例。

无总上限不保证任意随机轨迹在有限时间内结束，因此不能承诺固定最坏轮耗时。调度器记录最老在途轨迹、让其他独立轨迹继续供给 GPU，并在外部资源终止时保留原行为版本和状态；不能为清理慢样本私自施加一个未声明的化学失败阈值。

重复处理的边界：

- 缓存相同提示的确定性编码/prefill 可复用，但各 trajectory 的采样随机流独立。
- 相同提示和 token 可复用确定性的模型前向，但最终梯度系数要分算法处理：现有 `(prompt IDs, generated IDs, behavior logprobs, advantage, G)` multiplicity 键只足够固定 C 的 token 目标。GRPO 还依赖整条路线的 $L_i$，GSPO 还依赖当前 $s_i$ 和整路线 clip gate；必须保留各轨迹系数，在路线裁剪之后求和，或按精确系数分组，不能直接照搬原型去重。
- 不同优势符号不能先相加再套裁剪，因为裁剪对优势符号非线性。
- 部署 MCTS 的节点合并、先验频次和访问计数，与 RL 的独立采样权重分开记账。

## 9. 实现边界与成熟组件复用

首选参考实现固定为 **TRL v1.17.1，commit `44809ba3f3a136f3d62859ee981fee2506bce5c9`**。[6] 已核对其公开依赖下限为 Python≥3.11、Torch≥2.6、Transformers≥4.56.2；这只证明声明范围，不能代替本项目运行时兼容验证。

策略约束继续使用已冻结的 Qwen3.5-0.8B 原生工具模板、逆向电子执行器和 LoRA。TRL 提供 `loss_type="dr_grpo"`、`scale_rewards=False`、序列比例等配置作为目标对齐参考，但 **`max_completion_length` 同时涉及采样限制，不能直接用于本项目的无总上限轨迹归一化**。常数 $C$ 与环境采样容量分离。

原生采样/执行适配器输出逐动作概率记录，再按路线聚合。标准 Trainer 的多轮工具/`rollout_func` 接口必须验证 assistant-only mask、工具输出概率排除、跨反应提示重置和恢复语义；公开接口中有实验性部分，不能声称已无缝适配。若需抽取固定版本 loss 内核，保留来源、版本及与其公式的逐元素/梯度差分测试；不重写一套未说明的新优化器。

| 文件/接口 | 现有能力 | 本设计需要的变更 |
|---|---|---|
| `external_route_data.py` / 构建入口 | v1 外部树解析、库存裁剪、目标/私有库分离 | v2 中间体与反应隔离、witness 重验、来源与深度统计 |
| `HiddenRouteReward.score` | 端点匹配、AND 多重集、删失返回 | 可信祖先传播，防止越过不支持边刷部分奖 |
| `TargetOnlyRouteEpisode` | 原生提示和动作；全前体递归 | 将 VALID 根加入保护；保持参考不参与采样/停止 |
| `route_policy_sampling.py` / native sampler | 独立请求 RNG、原始 token/logprob 记录 | 与 learner 校准；完整轨迹分组；后端切换前的等价性检查 |
| `optimize_routes` | RLOO、固定归一化、重复计算合并 | 显式优势/聚合/ratio 变体、$\mu$ 更新及 GSPO 轨迹聚合 |
| checkpoint/contract | 原子发布、数据与代码绑定、原样恢复 | 冻结 v2、算法变体和上游 SHA；旧 RLOO 合约不可直接当新算法作业 |

训练起点与 KL 参考均为已核验的 EARHO `round-00513`，adapter SHA256：

```text
2ec51123811d1386808f72fa3bb786deb0f3711edf1e1b1477a82b85f06b2a6a
```

这是 FlowER 训练模型的权重来源；本轮路线训练数据来自外部专利路线。所有对照使用相同起点。没有用户批准和新冻结条件时，不换成旧0.6B检查点或其他模型。

## 10. GPU/CPU、存储与恢复

基础资源建议为4×H20：每 GPU 一个模型 owner；每轮64个目标分至4个 rank，$G=4$，初始可供每 GPU64条序列；每 rank16个 CPU executor 槽位。它是起始配置，不是已经证明最优的利用率。

CPU 在其他独立轨迹生成时执行化学动作；ready 队列、提示长度分桶、有限 in-flight 队列负责供给 GPU。物理执行器耗时和队列长度必须测量；CPU 线程受限时再评估进程池。不能把“4个 GPU owner”误写成“只有4个总 worker”。

保留 GPU 请求与返回的持久化边界；概率/执行 receipts 批量落盘，避免每条候选单独 Ceph `fsync`。只在当前行为参数的 rollout 完成并保存证据后更新模型。第一版同步行为版本，不混合多个未校正的异步旧策略。

vLLM、SGLang、共享 prefill 或连续批次属于后端优化选项，不改变算法条件。当前 Transformers5.5 的环境不能直接启用需要5.8以上版本的连续批次选项。[6] 任一后端替换先验证模型/adapter 身份、模板、行为 logprob、工具解析和执行后继；不能以 token/s 上升替代化学正确性。

每条决策至少记录 `task_id, trajectory_id, reaction_occurrence, decision_index, behavior_version, source hashes, prompt IDs, generated IDs, behavior logprobs, executor receipt`。actor 更新后的参数与旧样本概率分开存储。

训练循环的实施骨架如下；隐藏评分始终发生在采样/停止之后：

```text
for round in frozen_full_v2_target_order:
    behavior_version = seal(current_actor)
    raw_groups = native_rollout(targets=round, independent_G=4,
                                frozen_behavior=behavior_version)
    scored_groups = hidden_TRAIN_score_after_stop(raw_groups)
    report_all_target_ids_and_censor_causes(scored_groups)
    usable = drop_whole_groups_with_resource_or_protection_censoring(scored_groups)
    N_global = all_reduce_count_of_usable_groups_before_backward()
    advantages = declared_advantage_estimator(usable)
    for update in declared_mu if N_global > 0 else no_updates:
        # old behavior logprobs never get replaced by current-actor logprobs
        if token_ratio_objective:
            backward_chunked_full_route_token_objective(usable, advantages)
        else:
            coefficients = no_grad_complete_route_ratio_and_clip(usable)
            backward_original_decisions_with_detached_route_coefficients(coefficients)
        sum_gradients_across_ranks_without_a_second_mean()
        optimizer_step()
    atomic_checkpoint_publish_or_collective_failure()
    if declared_full_VALID_boundary:
        evaluate_all_VALID_targets_without_parameter_updates()
```

全局分母必须在 backward 前确定并用于每个 micro-batch；loss 已按全局组数归一化时，梯度 all-reduce 求和，不再按 world size 或本地 token 数重复平均。空 rank 以零梯度参与相同 collective 顺序。全局没有可用组时跳过优化，但仍记录所有目标访问、删失及原子 checkpoint 边界。

恢复只复用同一冻结行为版本的原始采样；已返回但尚未完成概率记录的 token 可重算概率，不重新抽样。没有返回值的请求标删失。rank 内数值/执行错误统一传播；rank0 checkpoint 写入失败广播给全部 rank，避免其他 GPU 在 collective 上长期等待。

## 11. 预先声明的完整实验协议

所有训练/验证条件必须用审核后的完整 v2 分母，不提交小子集、smoke 或 pilot 训练/测评。CPU 合成张量与符号图测试属于软件验证，不作为化学实验结果。

| 编号 | 条件 | 用途 |
|---|---|---|
| E0 | 固定 round-00513，无新增 RL | 同推理协议基线 |
| E1 | RLOO 优势/固定尺度目标，$\mu=1$ | 同样使用 v2 数据及修正后奖励；不把旧数据差异混入优化器对照 |
| E2 | Dr.GRPO 式固定尺度，$\mu=1$，$\beta=0.02$ | 推荐主线 |
| E3 | 标准 GRPO，$\mu=2$，$\beta=0$ | GSPO 的匹配控制 |
| E4 | 标准 GSPO，$\mu=2$，$\beta=0$ | 有辨识力的前沿对照 |
| E5 | E3 的 token-ratio、$\mu=2$＋单独 Clip-Higher；再独立动态组采样 | DAPO 技巧消融，预算评审后执行 |

E3/E4 使用相同优势、奖励、独立采样组数和 learner 更新次数；报告裁剪尺度差异。若比较 GSPO+KL，则增加相同明确 KL 定义的 GRPO 对照。不能将多一次 learner pass 的收益全部归因于序列比例。

第一轮方法比较建议固定 seed17、相同完整训练目标暴露数；是否增加 seed23/42作确认，由本 PR 的实验预算评审决定。所有新增训练条件仍完整覆盖其冻结分母。模型选择只用清理后的 VALID，禁止用 R1/R2 或 SimpRetro 选超参、挑 checkpoint。

### 启动前必须冻结的训练与验证合约

默认训练为完整 v2 TRAIN 的1次目标暴露；每目标 G4，AdamW，学习率 $2\times10^{-6}$、betas=(0.9,0.999)、eps=$10^{-8}$、weight_decay=0、梯度范数上限0.5；暂用恒定学习率，不做事后未声明的 warmup/课程改动。各方法只改变表内定义的项。预算由完整条件和种子数表达，不通过缩小分母或将未完成训练改称完整条件来满足。

| 验证项 | 冻结规则 |
|---|---|
| 收集器 | 主模型选择使用独立 native rollout，每个 VALID 根 G4；不是 MCTS 赢家 |
| 提示/采样 | 与训练相同公开投影；temperature1、top_p1、top_k0；无总轨迹截断 |
| 随机流 | 由固定评估协议、task_id、candidate index、decision index 派生；actor SHA 记入 receipt，但不改变验证随机流键 |
| 隐藏信息 | VALID 专用只读评分命名空间，只在停止后读取；TRAIN 奖励访问器禁止读取 VALID 标签；评估期间无梯度/参数更新 |
| 频率 | epoch0，以及完整目标暴露进度首次达到25/50/75/100%后的已发布轮边界；每次使用完整 VALID 分母 |
| 主要选择指标 | 全 VALID 根的参考支持库存闭合率 `verified_closed@4`；删失/未观察目标保留在成功下界分母 |
| 平局顺序 | 较低删失率；再比较完整 VALID 上每根总决策请求数；仍相同选更早 checkpoint |
| 辅助指标 | 原生库存闭合、未验证新颖路线、完整路径化学证据、决策请求数、GPU小时和删失分层 |

若预算不能承担这个完整协议，先修改并审阅合约，再运行；不能运行中仅挑更易或更短的 VALID 子集。外部资源中断恢复到已发布参数及原始行为 receipts；未完成训练的访问进度如实报告，不作为完成一轮的模型结果。每个新条件保存 train/valid/stock/参考访问边界、代码、库版本、起点、种子、$\mu$ 和全部优化器参数的 hash。

最终部署以固定 neural-guided MCTS 条件分别测 R1/R2 全460和 SimpRetro 全172，库存、目标、搜索版本、参数和硬件一致。旧0.6B/A100长时间累计结果不能作为等计算量提升证据。

主要指标：

- 全目标库存闭合率，并单列参考支持、未验证新颖路线和原已库存目标。
- 成功率—**实际策略决策请求数**曲线、首次完整闭合所需请求数；电子动作数、完成反应程序数与完整路线数分开。
- 保留全部未成功目标的首解分布；不要仅在成功样本中比较平均用量。实际时间/GPU小时和 learner passes 同时报告。
- 按路线反应深度、AND 分叉和反应频次分层的闭合/删失覆盖。
- 有效组比例、重复可执行后继比例、策略熵、KL、clip fraction、梯度范数。
- GPU繁忙率、ready 队列、批次填充、CPU等待和 Ceph 延迟；GPU显存占用不代表利用率。

对正式测评输出成功证书重新核验全 AND 叶子多重集与原生电子执行证据。奖励分数上升不等于独立化学可行性提升。

## 12. 实施顺序与验收门槛

1. **数据与奖励门槛。** 发布 v2 新分母、67/108/1348交叉问题的处理结果和全图审计；修正可信祖先奖励。v1 快照和旧结果保留，禁止静默改写。
2. **标准目标门槛。** 固定 TRL commit 和本项目环境锁；用相同张量比较 loss/梯度。覆盖组方差为零、重复前体、正负优势裁剪、不同轨迹长度、$\mu=1$ GRPO/GSPO首步等价性和$\mu=2$比例更新。
3. **系统软件门槛。** 真实小张量多进程检查空 rank、全局分母、失败传播；检查完整多动作轨迹的 assistant mask、相同上下文独立 RNG、原始 receipt 恢复和 checkpoint 原子性。此阶段不调用真实模型做小规模化学测评。
4. **完整主线训练。** 仅在本 PR 获得用户评审批准、上述门槛通过后，冻结新的 Dr.GRPO 条件，再按完整分母提交正式作业。现有 RLOO 合约 `external_route_rl_20261105_02` 不能替代它。
5. **完整方法对照及正式测评。** 按已批准预算执行 E3/E4 和必要消融；VALID 选择完成后，统一正式460/172测评。任何改变起点、库存、奖励、归一化、删失或采样分布均产生新条件和新输出命名空间。

现有软件证据：原型首轮19项通过；后续集成49通过/3失败，再以5项修复/新增检查解决全部已知失败，合并覆盖54项相关检查；不是一次最终54项全绿运行。没有新方法的 GPU训练、完整模型行为对齐或化学收益证据。实施者不能把这些原型测试直接当作上述新目标的验收。

## 13. 本 PR 请审阅的决定

- 是否同意“Dr.GRPO 稳健主线＋标准 GRPO/GSPO 匹配前沿对照”，将 DAPO 技巧放在独立消融。
- 是否同意先修 v2 验证隔离与可信祖先奖励，再启动任何正式训练。
- 确认完整条件和多 seed 的计算预算；不以减小分母或复用 benchmark 标签替代预算选择。

## 参考文献与实现来源

1. Ahmadian et al., **Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs**, ACL2024. https://aclanthology.org/2024.acl-long.662/
2. Shao et al., **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**, 2024. https://arxiv.org/abs/2402.03300
3. Liu et al., **Understanding R1-Zero-Like Training: A Critical Perspective**, 2025. https://arxiv.org/abs/2503.20783 ; code: https://github.com/sail-sg/understand-r1-zero
4. Yu et al., **DAPO: An Open-Source LLM Reinforcement Learning System at Scale**, 2025. https://arxiv.org/abs/2503.14476
5. Zheng et al., **Group Sequence Policy Optimization**, 2025. https://arxiv.org/abs/2507.18071 ; Qwen official: https://qwenlm.github.io/blog/gspo/
6. Hugging Face TRL **v1.17.1 GRPO Trainer** documentation: https://huggingface.co/docs/trl/v1.17.1/grpo_trainer ; immutable GSPO reproduction settings: https://github.com/huggingface/trl/blob/44809ba3f3a136f3d62859ee981fee2506bce5c9/docs/source/paper_index.md ; immutable source: https://github.com/huggingface/trl/tree/44809ba3f3a136f3d62859ee981fee2506bce5c9 ; package metadata: https://github.com/huggingface/trl/blob/44809ba3f3a136f3d62859ee981fee2506bce5c9/pyproject.toml
7. Genheden and Bjerrum, **PaRoutes: towards a framework for benchmarking retrosynthesis route predictions**, Digital Discovery2022. https://doi.org/10.1039/D2DD00015F ; data: https://github.com/MolecularAI/PaRoutes/blob/main/data/README.md
8. RetroCast training release construction: https://retrocast.ischemist.com/dev/rationale/training-set-releases/ ; frozen upstream manifest: https://files.ischemist.com/retrocast/training-sets/paroutes/v2026-06-05/reaction-holdout-n1-n5/manifest.json
