# 8-epoch mech-USPTO-31k：产物侧信息对闭环电子流推理的影响

## 结论

同一 Qwen3-0.6B 检查点、同一批 32 条验证反应、同一 greedy 闭环执行器：
保留完整产物侧时，端点命中 28/32；只保留主产物时，端点命中 0/32。

这次检查点的训练入口也不是强制主产物-only：10,152 条训练反应中，
9,886 条（97.38%）的产物起点有两个组分，266 条（2.62%）有一个组分。
所有 10,152 条反应的首步监督都是 FLOW，首步 IMPORT 为零。

这个结果区分了两个能力：**已给定产物侧参与组分后回推机制**，以及
**仅凭主产物先补齐缺失参与物、再回推机制**。前者的 87.5% 不能当作后者的成绩。

## 数据和模型

- 本次模型是 **Stage-I state-only**，不是压缩历史 Stage-II。
- 原始底座：Qwen/Qwen3-0.6B，revision
  `c1899de289a04d12100db370d81485cdf75e47ca`。
- 加载实际 `checkpoint-4056`，trainer state 为 epoch 8.0 / global step 4,056；
  不是根目录自动重载的 epoch-7 best checkpoint。
- 权重 SHA-256：
  `bf5cacc93d2c5e64b11df585e0dd0febf790773e21d13ec0838fa8dd538cdf51`。
- 训练数据：`data/mech_uspto_31k_natural_language_event_v2/train.jsonl`，
  10,152 条可执行 trace-view 反应，展开为 32,401 条决策监督；不是
  24,959 条全量 endpoint 训练反应。
- 本次评测为复用既定的 **32 条 validation smoke**，不是全量 3,120 条 test。
  GT 轨迹包含 1–5 个电子事件、2–7 次工具决策。

[训练合同](artifacts/mech31k_state8ep_product_input_20261022/model_lineage/training_data_contract.json)、
[实际 checkpoint trainer state](artifacts/mech31k_state8ep_product_input_20261022/model_lineage/checkpoint4056_trainer_state.json)、
[adapter 配置](artifacts/mech31k_state8ep_product_input_20261022/model_lineage/checkpoint4056_adapter_config.json)。

## 我们训练时能看见什么

每个训练决策包含固定 system/tool schema，以及以下观测：

```text
TARGET PRODUCT SMILES: <源数据的产物侧 SMILES>
CURRENT STATE SMILES: <当前可见分子状态>
MOLECULAR INVENTORY
<当前原子、化学键、孤对电子的位置和属性>
Choose the single next retrosynthetic action.
```

起点来自源数据 `target_smiles`，没有统一抽取最大有机主产物的步骤。
模型输出 `import_fragments`、`apply_electron_flow` 或 `finish_trace`，执行器计算下一个状态。
这次 state-only 模型不接收压缩历史；闭环时也不接收 GT 动作或未来状态。

完整训练文件经 SHA 验证后逐行统计得到：

| 训练起点 | 反应数 | 比例 | 首步 FLOW | 首步 IMPORT |
|---|---:|---:|---:|---:|
| 一个产物侧组分 | 266 | 2.62% | 266 | 0 |
| 两个产物侧组分 | 9,886 | 97.38% | 9,886 | 0 |
| 合计 | 10,152 | 100% | 10,152 | 0 |

[训练起点全量统计](artifacts/mech31k_state8ep_product_input_20261022/model_lineage/training_root_census.json)。
这里统计的是每条反应的首个决策，不是把全部 32,401 条中间状态当作反应数。

## 配对条件

除输入组分外，权重、32 个 reaction IDs、工具语法、state-only 渲染、执行器、
FP16/no quantization、greedy K1、完整端点参考和计算预算不变。每条只生成一条路径；
没有 value model、beam、NLL reranker、GT 修复或候选挑选。

| 条件 | 主产物衍生前体结构命中 | 完整前体混合物命中 | 正常 FINISH |
|---|---:|---:|---:|
| 原始完整产物侧，两个组分 | 28/32 = 87.50% | 28/32 = 87.50% | 29/32 = 90.625% |
| 只保留主产物，一个组分 | 0/32 = 0% | 0/32 = 0% | 12/32 = 37.50% |

主产物按原始产物侧中最大的含碳连通组分选择；32 条均无重原子数平局。
去除原 atom map 后独立重建地址，保留记录中的电荷、立体信息和显式氢语义。
不改用存在已知字段问题的 `rxn_prod_min`，也不额外中和结构。
删除的组分从 target 文本和私有初态同时移除，之后只能由模型自行导入。

这 32 条原始起点各有一个较小组分：15 个 `[Cl-]`、10 个 `[OH2+]B(Br)Br`、
3 个 `[Br-]`、2 个 `CC(C)(C)OC(=O)[O-]`、1 个 `[I-]`、1 个 `CC(C)(C)[Si](C)(C)F`。
它们来自源数据产物侧，不是这次闭环推理额外注入的未来 GT。

## 具体断点

对照原始映射和首个 GT 事件，32/32 首事件都用到了被删除组分中的原子。
若要从单主产物重现这些记录轨迹，必须先获得缺失参与物。
但本次模型在主产物-only 条件下 **32/32 首步直接 FLOW，0/32 首步 IMPORT**。
其中 8 条首步即被拒绝；随后整体结果为：

| 结束方式 | 反应数 |
|---|---:|
| source/destination 键不存在 | 10 |
| 化学状态执行失败 | 5 |
| 状态循环 | 5 |
| FINISH 成功但未命中记录端点 | 12 |

这定位到一个具体的入口监督缺口，而不只是“长轨迹太长”或“epoch 不够”。
它并不证明每条非 GT 端点都化学错误，也不证明仅增加 IMPORT 监督就一定修复全程结果。
尚未开展相应的新训练或因果消融。

## 全部原始 case，不只汇总

[32 条 case 逐例索引](artifacts/mech31k_state8ep_product_input_20261022/CASES.md)。
没有只挑成功例子，也没有用合成或模拟结果替代模型输出。

| 文件 | 内容 |
|---|---|
| [完整产物侧原始生成](artifacts/mech31k_state8ep_product_input_20261022/product_side/records/raw_generations.jsonl) | 151 次真实调用的完整 prompt、input token IDs、attention mask、output token IDs、生成原文 |
| [完整产物侧执行轨迹](artifacts/mech31k_state8ep_product_input_20261022/product_side/records/episodes.jsonl) | 32 条的全部 attempted/accepted/rejected 动作、状态、错误、端点 |
| [主产物-only 原始生成](artifacts/mech31k_state8ep_product_input_20261022/main_product_only/records/raw_generations.jsonl) | 103 次真实调用，同样保留完整输入输出 |
| [主产物-only 执行轨迹](artifacts/mech31k_state8ep_product_input_20261022/main_product_only/records/episodes.jsonl) | 32 条完整模型执行记录 |
| [GT 源行](artifacts/mech31k_state8ep_product_input_20261022/product_side/selected_source_rows.jsonl) | 32 条原始参考、trace plan、电子移动、映射状态、端点 |
| [主产物选择](artifacts/mech31k_state8ep_product_input_20261022/main_product_only/selected_inputs.jsonl) | 每条原始产物侧、实际主产物输入、被删除组分 |
| [逐例统一评分](artifacts/mech31k_state8ep_product_input_20261022/main_product_only/per_reaction_endpoint_comparison.jsonl) | 归一化的预测/GT 前体，两个条件的命中情况和错误 |
| [teacher-forcing 输入](artifacts/mech31k_state8ep_product_input_20261022/teacher_forced/prepared/tasks.jsonl) | 同一批 32 条的 157 个 GT-state 决策输入和答案 |
| [teacher-forcing 原始输出](artifacts/mech31k_state8ep_product_input_20261022/teacher_forced/predictions/decisions.shard-00-of-01.jsonl) | 157 个预测调用、生成原文、逐决策评分，不是闭环端点成绩 |

### 核验与评分

两组共 254 次闭环真实调用均有原始文本和 token IDs，且已独立重建观测并回放执行。
同一 reference 上完整混合物的比较忽略 atom map 和组分顺序；没有引入宽松中和规则。
结构指标统一为含主产物原子的前体组分，对两个条件采用同一定义。
这会改变 11 条原始 all-product-side structural reference 的范围，但配对结果仍为 28 对 0。
主产物-only 原始 `evaluation.json` 的旧 structural 字段不作为统一指标；
以 [endpoint_comparison.json](artifacts/mech31k_state8ep_product_input_20261022/main_product_only/endpoint_comparison.json) 为准。

两组均没有撞生成、context 或决策上限。完整产物侧最大生成 165 tokens，
主产物-only 最大 171 tokens，上限均为 512。
全部 archive 都是原文件逐字节拷贝；未补造未保存的内容。

## 当前后续状态

用户在提交前提出主产物-only 控制，新的 4×A100 Stage-II 训练**尚未提交**。
本 PR 只归档现象和原始证据，不训练新模型、不改变正式 benchmark、不改论文结果，
不自动合并 PR。下一步应先明确主产物-only 起点的片段生成监督，再决定训练配置。
