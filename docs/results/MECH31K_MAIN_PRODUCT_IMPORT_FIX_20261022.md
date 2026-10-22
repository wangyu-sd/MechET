# mech-USPTO-31k 主产物起点与 IMPORT 监督修复

本次只修正数据入口和 IMPORT 监督，不改 backbone、Stage-II 或 RL，也不提交训练。
主产物从冻结 trace-view 产物侧选择最大的含碳连通组分；其他组分同时从可见输入与私有初态移除，
在首次参与电子事件前显式 IMPORT。电荷、立体信息和显式氢保留，原始 GT 完整前体不变。
这不是重新确认全部原始数据中哪个分子才是主要合成产物，也没有切换原始数据字段。

## 参考轨迹全量检查

范围是当前冻结编译器已有的 **12,724 条可执行 trace-view 反应**，不是全部 31,199 条原始反应。
逐条从公开的、无原始 atom map 的主产物重新建立私有地址，实际执行 IMPORT、FLOW、FINISH，
核对每个决策的模型前缀、每次电子事件后的化学状态，以及最终 GT 完整前体。

| Split | 原有反应数 | 完整验证通过 | 未解决 |
|---|---:|---:|---:|
| train | 10,152 | 10,137 | 15 |
| valid | 1,319 | 1,317 | 2 |
| test | 1,253 | 1,252 | 1 |
| 合计 | 12,724 | 12,706 | 18 |

分母没有过滤。18 条失败逐条保留，生成的数据标记为 `training_allowed=false`，不能当成全量已验收的数据直接训练。
其中 15 条回放端点的立体化学与参考不一致，3 条在公开起点与原始参考的手性原子对应检查中失败。
这些是 GT 参考程序验证的问题，不是模型生成了 18 条错误。

修正了两类比较误判：手性标签必须按邻接顺序归一化，不能直接比较 RDKit 原始 chiral tag；
对于该 opt-in 数据条件，完成端点允许私有对称原子的不同对应，但去映射后的异构 SMILES 必须完全相同。
不通过清除手性、电荷、显式氢或中和结构来提高通过率。中间状态仍逐步独立检查。

[全量 manifest](artifacts/mech31k_main_product_import_fix_20261022/full_reference_audit/manifest.json)、
[未解决样本及实际端点](artifacts/mech31k_main_product_import_fix_20261022/unresolved_reference_cases.json)。

## 同一 32 条验证反应

复用此前固定的 32 个 validation reaction IDs、实际 8-epoch Stage-I state-only
Qwen3-0.6B `checkpoint-4056`，adapter SHA-256 为
`bf5cacc93d2c5e64b11df585e0dd0febf790773e21d13ec0838fa8dd538cdf51`。
没有重新训练权重，也没有加载 Stage-II、value model 或 RL 模型。

| 检查 | 结果 |
|---|---:|
| 修正后的 GT 参考轨迹执行至原 GT 端点 | 32/32 |
| GT 首个决策为 IMPORT | 32/32 |
| GT 决策数 | 189：55 IMPORT、102 FLOW、32 FINISH |
| 旧权重闭环推理的 GT 完整前体命中 | 0/32 |
| 旧权重闭环推理的主产物衍生前体结构命中 | 0/32 |
| 模型 FINISH | 12/32 |
| 模型首个决策为 FLOW / IMPORT | 32 / 0 |

闭环使用 greedy K=1，仅从主产物开始，不提供 GT 动作、参考前缀或预定 IMPORT。
保留原来的 40 次决策、32 个导入片段、512 个生成 token 和 4,096 上下文预算。
最大实际生成长度为 171 tokens，最大输入长度 1,063 tokens，未撞到生成长度上限。
全部 103 次实际调用的提示词 token IDs、输出 token IDs、工具调用、拒绝原因和状态均保存，
并独立重放检查。实际输入与输出 token IDs 与上一次主产物-only 对照的 103 次调用全部一致。

**GT 执行通过要求端点正确；模型自行 FINISH 不代表端点正确。**
数据修正没有更新旧权重，因此本次结果不是“修正监督后重新训练的模型效果”，也没有模型精度提升的结论。

例如验证反应 `10643`，公开起点是 `COC(=O)c1cccc(CBr)c1`。
修正后的 GT 首步为 IMPORT `[Cl-]`，然后回推电子流；旧模型首步却断开 C–Br 键，
得到 `[Br-].[CH2+]c1cccc(C(=O)OC)c1`，最终未命中 GT 前体。

## 原始数据与复现边界

归档目录：[mech31k_main_product_import_fix_20261022](artifacts/mech31k_main_product_import_fix_20261022/)。

- [实际模型输入与完整原始输出](artifacts/mech31k_main_product_import_fix_20261022/records/raw_generations.jsonl)
- [32 条完整执行记录](artifacts/mech31k_main_product_import_fix_20261022/records/episodes.jsonl)
- [全部 32 条 GT 与逐决策监督](artifacts/mech31k_main_product_import_fix_20261022/ground_truth/)
- [逐条 case 索引](artifacts/mech31k_main_product_import_fix_20261022/CASES.md)
- [独立重放与端点评分审计](artifacts/mech31k_main_product_import_fix_20261022/endpoint_comparison.json)
- [模型评测配置与权重来源](artifacts/mech31k_main_product_import_fix_20261022/provenance.json)
- [原始汇总](artifacts/mech31k_main_product_import_fix_20261022/task_summary.json)
- [逐文件哈希](artifacts/mech31k_main_product_import_fix_20261022/archive_manifest.json)

原始归档不改写路径、时间、权重 SHA 或采集时的发布状态。原始模型推理使用历史运行代码 `6874ea7b`，
driver 中的绝对路径保留用于定位，不表示在任意 checkout 都能直接运行。
本 PR 从公开主线单独移植此次修复，只补齐公开主线缺少的六行公开主产物私有映射 helper，
不夹带历史运行分支的训练、RL、搜索或推理配置变更。

模型采样使用修正数据 `_03`；最终全量检查使用 `_04`。
对这同一 32 条，两版 inverse 参考、189 条 state-SFT 监督和 32 条 audit 记录逐项相同。
模型端点评分不随全量检查中其余 5 条对称映射误判的修正而改变。

CPU 数据构建命令：

```bash
PYTHONPATH=src:. python scripts/build_mech31k_main_product_only_sft.py \
  --workspace /absolute/path/to/shared/MechET \
  --output-dir /absolute/path/to/new/main_product_only_output \
  --workers 16
```

该命令需要冻结的源数据和原始 manifest；拒绝覆盖输出目录。当前仍有 18 条未解决，
所以构建程序按设计返回非零状态，而不是悄悄缩小分母。

历史运行树验证为 986 项通过、17 项跳过；本 PR 整理到公开主线后，
完整单元测试为 486 项通过、1 项跳过。两者测试树不同，不应混用测试数量。
