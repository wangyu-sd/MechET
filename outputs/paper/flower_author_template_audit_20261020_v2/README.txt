芳香/Kekulé 与显式氢表示修正：离线作者模板匹配器
日期：2026-10-20

权威结果：results_final/summary.json、report.txt、verification.json。
逐事件匹配与重放证据：results_final/event_matches.jsonl.gz。
原始完整状态保持在上一版 results_v2/event_inputs.jsonl.gz；来源哈希见
summary.json。没有改写模型输出、电子动作、执行器、训练数据或检查点。

修复内容
1. 不再独立删除两侧氢原子；复用作者 explicit_H.py 展开规则，在完整状态
   中物化上下文氢。保留原始映射氢身份、同位素和电荷。
2. 对芳香/Kekulé 输入及作者符号化芳香中间体，尝试原规则的双侧重放。
   接受条件是重放后整份映射状态与目标一致，不是简单忽略芳香键。
3. SMARTS 的原始重原子度数等条件在相应隐式氢视图中检查。
4. 修正 72 个 GT 动作前状态遗漏其记录 imports 的审计准备问题。
   原始导入是自由氢原子，不能改成 H2；修正后均属自由基变化，仍标为
   此次闭壳层表示修复不支持。覆盖数不变。完整修正记录见
   results_final/gt_recorded_import_corrections.jsonl.gz。

同一完整事件总体：189,233 个事件，未缩小分母。
  71,868 个去重的已保存模型事件，来自六个既有推理条件；
  117,365 个 GT 事件，来自 28,967 条 strict compiled FlowER test 反应。
官方 FlowER test 为 28,971 条，另 4 条上游端点损坏记录无 strict trace。
模型事件来自完整保存程序池，不包含所有被拒绝/未完成的搜索提案。
程序链接共 37,922 条、124,808 次事件出现，重复事件不重复计入事件分母。

当前作者模板版本 Reaction_templates 的局部转换兼容率
条件              分母       修复前             修复后
GT                117,365    94,417 (80.447%)    113,083 (96.352%)
全部保存模型事件   71,868    46,980 (65.370%)     57,690 (80.272%)
31k MCTS           27,886    17,915 (64.244%)     22,016 (78.950%)
R1/R2 MCTS         27,402    16,574 (60.485%)     21,122 (77.082%)
其他 greedy 条件、旧版 Reaction_templates_20250216 及逐程序聚合在 summary.json。
两个作者版本分别报告；上表不是取版本并集。

重要解释
这是匹配器的表示层修正，不是模型精度或端点命中率的提升。
GT 兼容率校准的是匹配器灵敏度，不能作为独立化学正确性验证。
结果不证明动力学可行、文献支持或完整反应的试剂/条件与阶段顺序兼容。
自由基变化、不平衡模板仍有不支持项；无匹配不能直接判定化学错误。
原有正匹配原样保留；新增匹配必须有完整状态重放证据。
不做电荷中和、互变异构合并，也不忽略真实改键或原子映射身份。

来源
作者仓库：https://github.com/jfjoung/Mechanistic_dataset
固定提交：8bd5b38e9602c4d45485bf50076d03c292dcc486
原始模板、酸碱表及显式氢展开实现位于：
/aaa/fionafyang/buddy1/whaleywang/datasets/flower_original_mechanistic_templates_20261019/
本地冻结 FlowER 数据对应的精确生成器版本尚未确认，因此不预设应达 100%。
旧事件、模板与程序链接来源由 manifest 的 SHA256 固定。

验证
20 项软件测试通过，含 3 个实际漏匹配回归样例，以及错误电荷、额外断键、
远端双键迁移、同位素变化、原子身份交换等拒绝测试。
对所有新增匹配的 30,121 个事件、33,325 份固定原子绑定证据逐条重放复核，
全部通过；不是抽样检查。压缩文件完整性验证通过。
全仓 pytest 未通过收集，存在两个缺模块错误，未将其描述为全仓测试通过：
  tests/test_mech_uspto31k_causal_import.py: mechet.a7_rescue
  tests/test_nl_history_beam.py: mechet.in_place_grounded_flow

软件测试（从本目录执行；不调用模型）：
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -p 'test_*.py' -v

全总体离线审计首次运行命令（从仓库根执行）：
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u outputs/paper/flower_author_template_audit_20261020_v2/run_representation_audit.py \
    --out outputs/paper/flower_author_template_audit_20261020_v2/results_02 --workers 12
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u outputs/paper/flower_author_template_audit_20261020_v2/finalize_representation_audit.py
需在新输出位置运行，不覆盖本次证据。finalizer 的 SOURCE/OUT 指向首次运行目录。
运行环境：Python 3.13.5、RDKit 2026.03.4、12 CPU 进程，无 GPU/新模型调用。

目录沿革
  results/       首次准备阶段中止，ABORTED_PREPARATION.txt 已标记，不作为结果。
  results_02/    完整数值第一遍，保留原有 72 个状态准备错误的诊断记录。
  results_final/ 修正上述分类并复核全部新证据后的最终结果。
  前一天的 flower_author_template_audit_20261019_v1 为未改动的历史基线。
