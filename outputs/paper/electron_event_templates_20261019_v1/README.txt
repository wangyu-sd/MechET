逆向电子事件模板分析（2026-10-19）

入口：results/report.html
主结果：results/reconciled_metrics.json、results/template_counts.csv
逐事件原始输入输出及标注：results/event_annotations_v2.jsonl.gz
程序到事件的完整映射：results/program_event_links.jsonl.gz
完整案例：results/cabozantinib_complete_event_analysis.json
各类代表事件：results/case_gallery.json

范围是 PR89 v18 已冻结的全部 37,922 条完整程序，不是新一轮模型评测。
全部 8,520 份原始树文件和动作载荷均核对了冻结哈希。
124,808 次 FLOW 出现，按推理条件、动作前私有映射状态和原始动作参数
精确去重后为 71,868 项。没有声称这是对称不变的化学事件去重。

所有模板按逆向事件的正向对应解释，并保留原始逆向箭头。
模板名称和规则为本次人工编写的局部图规则，不是经标注验证的真值分类器。
62.285% 局部模板兼容不是化学正确率，也不是端点命中率。
4.994% 特定模板不适用均来自普通 SN2 适用性检查：3,443 项 sp2/sp 中心，
146 项三级碳中心。不能由此断言相应净反应不可能，其他机理没有被排除。
319 项自由基容器事件在本轮极性模板库外，没有标成化学错误。

版本说明：
event_annotations.jsonl.gz 与 summary.json 是第一次批处理原始输出，予以保留。
event_annotations_v2.jsonl.gz 是主结果，只修订了 5 项远端芳香性重感知的
transition_check：从 mismatch 改为 aromatic_representation_deferred。
这些事件的箭头原子本身不芳香，但闭环或断环改变了远端 1.5 阶键表示。
五项的模板名称与四分类结论均不变；没有修改任何原始程序、源树或模型结果。
其余 71,863 项标注不变。本轮没有发现可直接归因为真实状态错配的事件，
但芳香表示项仍未在本分析中独立重放证明。

复现（仓库根目录，当前 RDKit 2026.03.4）：
python -m unittest discover -s outputs/paper/electron_event_templates_20261019_v1 -p 'test_*.py' -v
python outputs/paper/electron_event_templates_20261019_v1/event_template_analysis.py --workers 12 --output NEW_OUTPUT_DIRECTORY

event_template_analysis.py 已纳入芳香表示修订逻辑；新运行不再产生上述五项
表示性 mismatch。已有 results 不会被该脚本静默覆盖。
summarize_event_templates.py 优先读取 v2，否则读取初次批处理文件。
本次没有训练、重新采样、提交平台任务、修改执行器或合并远端分支。
