作者模板匹配器 v3：自由基/配位状态重放与复合事件
日期：2026-10-20

本目录是离线审计修复，不是新训练或推理实验。
最终入口：results_final/report.txt、summary.json、verification.json。
完整逐事件输出：results_final/event_matches.jsonl.gz。
原始状态、动作来源仍是 v1 的冻结 event_inputs.jsonl.gz；其 SHA256、
72 条记录导入修正及所有代码快照均由最终 manifest 固定，没有改写 GT。

修改
1. transition_repair.Event 不再因自由基电子数变化而提前拒绝匹配。
   仍执行原作者模板，并要求完整映射目标状态一致。没有删除/忽略自由基
   信息，也没有对电荷中和。配位导致的价态/自由基表示变化必须能由模板
   改键后的 RDKit 状态重放得到；旁观分子的额外变化会被拒绝。
2. 允许可明确定位的原始映射质子转移，加一个剩余核心事件组成有限分解。
   每个分量必须分别通过同一版本作者模板的完整状态重放，所有中间态
   严格串联，最终完整状态必须等于原目标。不任意拆单条电子箭头。
   单模板兼容与多模板组合兼容分别统计；组合不是完整反应阶段顺序证明。
3. 候选仅来自固定作者库及原有 pKa 展开；没有手工增加用于提高覆盖率的
   化学模板。原来的芳香/Kekulé 与显式氢修复继承自不变的 v2 实现。
4. delivery.py 仅清理历史状态标签、补齐版本/代码来源和 GT 反应级聚合。
   results/ 是原始完整数值结果；results_final/ 是标签和来源补齐后的交付。
   交付步骤断言全部数值计数不变。历史 v2 状态保存在 legacy_v2_* 字段。

当前作者库 Reaction_templates；同一完整 189,233 事件总体
条件                  分母       v2单模板    v3单模板    v3单模板或组合
GT                   117,365      96.352%      99.520%        99.621%
全部保存模型事件       71,868      80.272%      80.667%        80.713%
31k MCTS               27,886      78.950%      79.212%        79.255%
R1/R2 MCTS             27,402      77.082%      77.841%        77.888%
具体计数及全部六个模型条件见 report.txt；旧版 Reaction_templates_20250216
单独保存在 summary.json，不取两个版本的并集充当上表。

GT：113,083 -> 116,802 个单模板匹配，增加 3,719；另 118 个仅有组合证明。
原先跳过的 3,792 个自由基标记变化中，剩 73 个未被直接模板解释。
合计仍有 445 个 GT 事件没有当前版本的单模板或组合证明：
  72：含明确记录的自由氢原子导入，仍未解释；没有擅自改成 H2。
   1：其他自由基变化，仍未解释。
 372：闭壳层转换，仍未解释；没有统一归因为模板缺失或 GT 错误。
全部 563 个“未单模板匹配”事件（含 118 个组合匹配）的完整原始输入、
电子动作、结果与诊断保存在 unmatched_single_gt_inputs_and_diagnostics.jsonl.gz。

实际修复样例
  GT test_37 的第5、6步：恢复 Pd-P 第三、第四配体事件的作者模板匹配。
  GT test_123 的第1步：两次质子转移分别有作者模板证明，标签为组合匹配，
  没有虚报为一个模板。原始氢身份 62、60 在逐步证据中保留。

总体与意义
  GT 来自 28,967 条 strict compiled FlowER test 反应，共 117,365 个事件。
  官方完整 reaction test 为 28,971，另 4 条有命名的上游损坏端点无 strict trace。
  模型总体为 71,868 个去重事件，来自 37,922 个完整保存程序/124,808 次事件出现。
  这些不是所有拒绝/未完成的原始搜索提案。GT 反应级全事件覆盖另见
  summary.json 的 gt_reaction_paths，不与模型程序数混用。
  指标是作者局部转换兼容率，不是端点命中率、动力学可行率或文献支持率。
  作者模板参与生成 FlowER，因此 GT 覆盖只校准匹配实现，不是独立化学验证。
  未匹配不能直接视为化学负例；没有为达到100%而过滤或放宽整态一致条件。

验证
  新入口28项软件测试通过，包括继承到新入口的16项表示回归。
  旧实现20项软件测试也通过（48次测试执行，不代表48个互不重叠用例）。
  4,775份新增固定原子绑定证据逐条重放核对，组合每一分量及首尾边界均核对。
  完整事件处理错误0，搜索安全阈值触发0，全部历史正匹配保留。
  独立只读代码审查没有发现重要接受条件缺陷；来源/状态标签建议已在交付中处理。
  全仓 python -m pytest -q 仍有两个既存收集错误，未宣称全仓通过：
    tests/test_mech_uspto31k_causal_import.py: 缺 mechet.a7_rescue
    tests/test_nl_history_beam.py: 缺 mechet.in_place_grounded_flow

复现（仓库根目录；使用新的输出目录，不覆盖证据）
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u outputs/paper/flower_author_template_audit_20261020_v3/audit_transition_repair.py \
    --workers 12 --out /path/to/new_raw_output
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u outputs/paper/flower_author_template_audit_20261020_v3/delivery.py \
    --source /path/to/new_raw_output --out /path/to/new_final_output
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s outputs/paper/flower_author_template_audit_20261020_v3 -p 'test_*.py' -v

Python3.13.5 / RDKit2026.03.4 / 12个本地CPU进程；完整数值处理约455秒。
未调用模型、未提交/停止太极任务、未修改检查点/执行器/训练数据。
作者仓库 https://github.com/jfjoung/Mechanistic_dataset
固定提交 8bd5b38e9602c4d45485bf50076d03c292dcc486。
