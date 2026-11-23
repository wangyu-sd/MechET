# Retro-E NMI expected results — 25-sheet data export

> **SIMULATED / PLANNED. NOT ACTUAL RETRO-E TEST RESULTS.** This text export preserves the cell content of the 25-sheet expected-results workbook for PR117 planning. External numbers marked `REAL-PUBLISHED` are cited from literature, while `PLANNED-RERUN` values are hypothetical and must never be represented as completed experiments. The original styled/formula workbook is maintained as a separate conversation attachment; this is a machine-readable text extract, not the original XLSX binary.

Source workbook: `RetroE_NMI_Complete_Expected_Results_2026-11-23.xlsx` (47,688 bytes; SHA-256 `7ff511859bac584d3d72f858d1b91590cd844385b9df34d55e11b90b00ce013d`).

## 01 · 00_使用说明

```csv
index,Retro-E / NMI 完整结果规划数据包,Unnamed: 1,Unnamed: 2,Unnamed: 3
0,全部 SIM 结果是假设性论文目标，绝不是实际训练成绩；外部论文数值要严格保留原有 Benchmark 与协议。,,,
1,,,,
2,标签,是什么,是否能直接用于论文,资料链接
3,REAL-PUBLISHED,外部已发表且经资料核对的原始成绩,可引用，不能等于同配置重跑,https://www.nature.com/articles/s41586-026-11160-9
4,REAL-INTERNAL,来自 MechET PR116/117 的真实证据,可报告，注明 train/validation,https://github.com/wangyu-sd/MechET/pull/117
5,SIMULATED-TARGET,预期成文目标；假设已经完成时表格会长成什么样,不可作为已获得结果,
6,PLANNED-RERUN,公开模型在我们锁定的目标集上“预计”的模拟成绩,需要真实运行才能填入论文,https://github.com/microsoft/retrochimera
7,URSA U2,URSA-minor-1.1.0-U2；不能与 U2P2/其他版本混用,正式评价需锁定版本,https://github.com/insilicomedicine/URSA
8,Solv0/1/2,URSA：库存闭合 / 化学反应中心合法 / 严格反应合理性,全分母目标，层层包含,https://github.com/insilicomedicine/URSA
9,MV-Solv,Retro-E 拟提出的独立电子验证+Solv2,外部系统须用同一独立评价器,
10,SDR@1,CbyD 文献根部关键战略断键恢复,不可当成化学有效率,https://github.com/chemistry-by-design/cbyd
11,ELRG@B,相同 Actor、同目标、同预算的电子级vs反应级配对净闭合率,报告两侧独有成功数量,
12,预算统一,电子模型调用≠完整反应模型调用；跨模型以 GPU 秒与 wall-clock 做辅助对齐,不要把历史论文预算混入同调用图,
13,当前真实不足,Adaptive ReFIT Step64：499 VALID 72→71；24目标搜索三组均0,不能称当前已显著提升,https://github.com/wangyu-sd/MechET/pull/117
```

## 02 · 01_全部核心数字

```csv
index,Final Results · 总成绩单,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,这些数值是一套自洽的 SIMULATED-TARGET 剧本；真实 benchmark 原文数字单独留在 02 表。,,,,,,,,
1,,,,,,,,,
2,Figure,Evaluation,N,Primary metric,External / matched comparison,Retro-E target,Difference,Evidence class,Interpretation
3,Fig2a,URSA Expert,100,Solv2,RetroChimera 32/100,Retro-E 41/100,+9 pp,REAL external / SIM ours,同协议化学可信度
4,Fig2a,URSA Drugs,100,Solv2,RetroChimera 60/100,Retro-E 67/100,+7 pp,REAL external / SIM ours,高库存闭合并不等于高可信
5,Fig2b,CbyD 500,500,Solv2,RC rerun 17/500 (SIM),Retro-E 34/500,+17 routes,all SIM,外部模型必须重跑，非真实成绩
6,Fig2d,RetroBench,5838,Route Top1,R3 published 43.7%,Retro-E 46.5%,+2.8 pp,REAL external / SIM ours,非库存闭合
7,Fig3a,96 paired,96,Stock close @64,Reaction 15/96 (SIM),Electron 23/96,+8 routes,all SIM,"E-only10, R-only2"
8,Fig3a,Reference 1084,1084,Stock close @64,Reaction 164/1084 (SIM),Electron 205/1084,+41 routes,all SIM,补充验证，不是现代主benchmark
9,Fig4b,Fixed VALID,499,Stock close @112,SFT 72/499 (REAL),Full ReFIT 96/499,+24 routes,REAL baseline / SIM ours,现实 Adaptive Step64 仍为71
10,Fig4d,Held-out action states,240,Preferred right FLOW,SFT 132/240 (SIM),ReFIT 173/240,+41 states,all SIM,同输入真实后果
11,Fig5b,Reference 1084,1084,GPUh for 205 routes,Full search 94h (SIM),Student 45h (SIM),2.09x,all SIM,需要真实Student与GPU实测
12,Fig6a,CbyD strategic cuts,500,SDR@1,RC rerun 82/500 (SIM),Retro-E 133/500,+51 hits,all SIM,命中文献≠可行性
13,Fig6c,Stock perturbation,240,Recovered routes,Reaction 35/240 (SIM),Electron 61/240,+26 routes,all SIM,使用共同成功目标
14,Fig6d,Blinded expert route review,80,Fully accepted,External on same 80 TBD,Retro-E 56/80,70%,SIM only,不能拿RetroChimera选中10条混排
```

## 03 · 03_Benchmark协议

```csv
index,Dataset & Evaluation Manifest,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,外部基线的性能对齐需要同一目标 ID、库存、独立评分器及预算定义。,,,,,,,
1,,,,,,,,
2,Dataset,N,Purpose,Metric,Protocol,Role,URL,Caution
3,URSA Expert 2026,100,Public medicinal chemistry,Solv0/1/2,"U2, ChemCensor v1.2",main,https://github.com/insilicomedicine/URSA,same fixed stock
4,URSA Drugs & Clinicals,100,Public druglike targets,Solv0/1/2,"U2, ChemCensor v1.2",main,https://github.com/insilicomedicine/URSA,same fixed stock
5,CbyD v2026.11,2902,Natural products and pharmaceuticals,SolvN / SDR / evidence,500 main and full optional,main,https://github.com/chemistry-by-design/cbyd,3502 routes /56640 steps; exclude 27 dev IDs
6,RetroBench,5838,Multi-step route reconstruction,Top1/5 route accuracy,Original test route equivalence,main,https://aclanthology.org/2026.acl-long.1745/,not equal stock solved
7,Reference,1084,Internal 600/400/84 topology groups,stock solved @physical call,frozen stock + mapping,supplement,https://github.com/wangyu-sd/MechET/pull/117,older standard validation
8,Market,660,Different building-block catalog,stock solved @physical call,frozen market stock,supplement,https://github.com/wangyu-sd/MechET/pull/117,500 linear /160 convergent
9,Electronic OOD challenge,3120,Rare/unseen reaction composition,"exact precursor, executable FLOW",planned frozen subsets,mechanism,https://github.com/wangyu-sd/MechET/pull/117,must audit actual dataset before freeze
10,Mechanism expert challenge,300,Mapped electron program quality,mechanism replay/accuracy,planned verification,mechanism,https://github.com/wangyu-sd/MechET/pull/117,verify set/protocol
11,Competition FLOW,240,Heldout same-state action contrasts,closing-action preference,not RL training,mechanism,https://github.com/wangyu-sd/MechET/pull/117,only legal distinct actual successors
12,Reaction Repair,150,Actual rejected executor states,valid alternative and rescued route,matched continuation,support,https://github.com/wangyu-sd/MechET/pull/117,do not select only successful recoveries
13,Stock perturb,240,Common solved original-stock targets,recovered alternate route,fixed same removed terminals,support,https://github.com/wangyu-sd/MechET/pull/117,same intervention all models
14,Expert route review,80,Blind sampled complete routes per model,full route acceptance,3 chemists if available,application,https://www.nature.com/articles/s41586-026-11160-9,can replace by objective evidence if unavailable
```

## 04 · 02_真实外部与内部

```csv
index,Published Results & Real Internal Evidence,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,只有这一张表的 REAL 数字是当前已经获得的成绩；各行协议不同，绝不将它们当成同一排行榜。,,,,,,,
1,,,,,,,,
2,Benchmark,System,Metric,Value,Unit,Original Protocol,Evidence,Class
3,URSA Expert,RC-MCTS,Solv0,99,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
4,URSA Expert,RC-MCTS,Solv1,78,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
5,URSA Expert,RC-MCTS,Solv2,32,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
6,URSA Drugs,RC-MCTS,Solv0,100,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
7,URSA Drugs,RC-MCTS,Solv1,83,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
8,URSA Drugs,RC-MCTS,Solv2,60,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
9,URSA Expert,AZF-MCTS,Solv2,29,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
10,URSA Drugs,AZF-MCTS,Solv2,54,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
11,URSA Expert,DMS-Wide,Solv2,15,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
12,URSA Drugs,DMS-Wide,Solv2,43,/100,U2 original,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment,REAL-PUBLISHED
13,RetroBench,R3,Route Top1,43.7,%,ACL 2026,https://aclanthology.org/2026.acl-long.1745/,REAL-PUBLISHED
14,RetroBench,R3,Route Top5,58.7,%,ACL 2026,https://aclanthology.org/2026.acl-long.1745/,REAL-PUBLISHED
15,CbyD 2026,dataset,Unique targets,2902,targets,v2026.11,https://arxiv.org/abs/2611.24996,REAL-PUBLISHED
16,CbyD 2026,dataset,Routes,3502,routes,v2026.11,https://arxiv.org/abs/2611.24996,REAL-PUBLISHED
17,CbyD 2026,dataset,Steps,56640,reactions,v2026.11,https://arxiv.org/abs/2611.24996,REAL-PUBLISHED
18,CbyD 2026,dataset,Median LLS,13,steps,v2026.11,https://arxiv.org/abs/2611.24996,REAL-PUBLISHED
19,CbyD 2026,dataset,Convergent,38.6,%,v2026.11,https://arxiv.org/abs/2611.24996,REAL-PUBLISHED
20,RetroChimera Nature,RetroChimera,Expert step accept,96.9,%,291 curated steps,https://www.nature.com/articles/s41586-026-11160-9,REAL-PUBLISHED
21,RetroChimera Nature,RetroChimera,Whole route accept,9,/10,selected challenging targets,https://www.nature.com/articles/s41586-026-11160-9,REAL-PUBLISHED
22,AbSynth JACS 2026,dataset,Routes,3077,routes,published,https://doi.org/10.1021/jacs.6c08434,REAL-PUBLISHED
23,AbSynth JACS 2026,dataset,Steps,59712,steps,published,https://doi.org/10.1021/jacs.6c08434,REAL-PUBLISHED
24,PaRoutes n5,Retro*,Stock solved,97.03,%,500 original iterations,https://github.com/MolecularAI/PaRoutes,REAL-PUBLISHED
25,PaRoutes n5,MCTS,Stock solved,96.76,%,500 original iterations,https://github.com/MolecularAI/PaRoutes,REAL-PUBLISHED
26,PR117 VALID,StageII,Stock closed,72,/499,"@112, 499 needs-synthesis",https://github.com/wangyu-sd/MechET/pull/117,REAL-INTERNAL
27,PR117 VALID,Adaptive Step64,Stock closed,71,/499,"@112, 499 needs-synthesis",https://github.com/wangyu-sd/MechET/pull/117,REAL-INTERNAL
```

## 05 · 04_URSA结果

```csv
index,Figure 2a · URSA 200 Target Table,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9,Unnamed: 10,Unnamed: 11,Unnamed: 12,Unnamed: 13
0,外部值是原论文已发表 U2 结果；SIM 是 Retro-E 预期完成版。新 MV-Solv 不能把外部基线当作 0。,,,,,,,,,,,,,
1,,,,,,,,,,,,,,
2,Set,System,Evidence,N,Solv0 count,Solv0 rate,Solv1 count,Solv1 rate,Solv2 count,Solv2 rate,MV count,MV rate,Budget,URL
3,Expert,RetroChimera-MCTS,REAL-PUBLISHED,100,99,0.99,78,0.78,32,0.32,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
4,Expert,AiZynthFinder-MCTS,REAL-PUBLISHED,100,65,0.65,60,0.6,29,0.29,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
5,Expert,Retro*,REAL-PUBLISHED,100,57,0.57,53,0.53,22,0.22,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
6,Expert,DMS-Wide,REAL-PUBLISHED,100,85,0.85,26,0.26,15,0.15,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
7,Expert,Stage-II SFT,SIMULATED-TARGET,100,83,0.83,58,0.58,20,0.2,14,0.14,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
8,Expert,ReFIT Policy,SIMULATED-TARGET,100,86,0.86,65,0.65,28,0.28,22,0.22,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
9,Expert,Reaction-first,SIMULATED-TARGET,100,91,0.91,72,0.72,34,0.34,26,0.26,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
10,Expert,Retro-E Electron,SIMULATED-TARGET,100,94,0.94,81,0.81,41,0.41,34,0.34,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
11,Drugs,RetroChimera-MCTS,REAL-PUBLISHED,100,100,1,83,0.83,60,0.6,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
12,Drugs,AiZynthFinder-MCTS,REAL-PUBLISHED,100,80,0.8,73,0.73,54,0.54,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
13,Drugs,Retro*,REAL-PUBLISHED,100,76,0.76,68,0.68,53,0.53,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
14,Drugs,DMS-Wide,REAL-PUBLISHED,100,87,0.87,55,0.55,43,0.43,,,original U2,https://www.researchgate.net/publication/408523302_URSA_Chemistry-Aware_Benchmark_for_Utilitarian_Retrosynthesis_Assessment
15,Drugs,Stage-II SFT,SIMULATED-TARGET,100,85,0.85,69,0.69,38,0.38,31,0.31,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
16,Drugs,ReFIT Policy,SIMULATED-TARGET,100,89,0.89,75,0.75,49,0.49,40,0.4,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
17,Drugs,Reaction-first,SIMULATED-TARGET,100,94,0.94,79,0.79,56,0.56,46,0.46,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
18,Drugs,Retro-E Electron,SIMULATED-TARGET,100,97,0.97,87,0.87,67,0.67,57,0.57,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
```

## 06 · 05_CbyD主成绩

```csv
index,Figure 2b / 6a · CbyD 500 Complex Targets,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9,Unnamed: 10,Unnamed: 11,Unnamed: 12,Unnamed: 13,Unnamed: 14,Unnamed: 15
0,500 独立目标、统一库存及 CHEMCENSOR；此前 27 个开发目标不计入测试。RC/DMS/AZF 的数字全是未来重跑设想，不是论文数据。,,,,,,,,,,,,,,,
1,,,,,,,,,,,,,,,,
2,Model,Evidence,N,Solv0 N,Solv0 rate,Solv1 N,Solv1 rate,Solv2 N,Solv2 rate,SDR N,SDR rate,≥2 lit steps N,Rate,Status,Budget,URL
3,RC-MCTS,PLANNED-RERUN,500,36,0.072,27,0.054,17,0.034,82,0.164,35,0.07,"hypothetical rerun, NOT reported",matched cost,https://github.com/chemistry-by-design/cbyd
4,AZF-MCTS,PLANNED-RERUN,500,26,0.052,19,0.038,11,0.022,78,0.156,25,0.05,"hypothetical rerun, NOT reported",matched cost,https://github.com/chemistry-by-design/cbyd
5,DMS-Wide,PLANNED-RERUN,500,28,0.056,18,0.036,9,0.018,91,0.182,29,0.058,"hypothetical rerun, NOT reported",matched cost,https://github.com/chemistry-by-design/cbyd
6,Stage-II,SIMULATED-TARGET,500,25,0.05,17,0.034,9,0.018,93,0.186,34,0.068,no-search policy,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
7,ReFIT Policy,SIMULATED-TARGET,500,30,0.06,22,0.044,14,0.028,99,0.198,39,0.078,RL policy only,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
8,Reaction-first,SIMULATED-TARGET,500,39,0.078,31,0.062,22,0.044,111,0.222,48,0.096,same Actor controller,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
9,Retro-E Electron,SIMULATED-TARGET,500,52,0.104,43,0.086,34,0.068,133,0.266,61,0.122,stretch target,256 e-calls,https://github.com/wangyu-sd/MechET/pull/117
```

## 07 · 06_CbyD难度分层

```csv
index,Figure 2c · Complex Synthesis Stratification,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,三组按照文献最长线性反应路径 LLS 分段；每种模型合计必须等于 CbyD 主表。,,,,,,,
1,,,,,,,,
2,Reference LLS,N,Model,Evidence,Solv0,Solv1,Solv2,SDR
3,LLS 5–9,100,RC-MCTS,PLANNED-RERUN,13,10,8,18
4,LLS 10–14,200,RC-MCTS,PLANNED-RERUN,17,12,7,37
5,LLS ≥15,200,RC-MCTS,PLANNED-RERUN,6,5,2,27
6,LLS 5–9,100,Retro-E Electron,SIMULATED-TARGET,19,16,12,30
7,LLS 10–14,200,Retro-E Electron,SIMULATED-TARGET,23,19,15,58
8,LLS ≥15,200,Retro-E Electron,SIMULATED-TARGET,10,8,7,45
```

## 08 · 07_RetroBench

```csv
index,Figure 2d · RetroBench 5838 Route Top-k,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,Route Top1/5 是与参考合成树/路线的匹配指标，并非库存闭合。R3 真实发表百分比未给出原始整数，故不伪造对应人数。,,,,,,,,
1,,,,,,,,,
2,Model,Evidence,N,Top1 hits,Top1 %,Top5 hits,Top5 %,Protocol,URL
3,R3 (ACL 2026),REAL-PUBLISHED,5838,,0.437,,0.587,original paper,https://aclanthology.org/2026.acl-long.1745/
4,FusionRetro,REAL-PUBLISHED,5838,,0.375,,0.512,original paper,https://aclanthology.org/2026.acl-long.1745/
5,Stage-II,SIMULATED-TARGET,5838,2245,0.38454950325453924,3060,0.5241521068859198,original route equality,https://github.com/wangyu-sd/MechET/pull/117
6,ReFIT,SIMULATED-TARGET,5838,2485,0.42565947242206237,3270,0.5601233299075026,original route equality,https://github.com/wangyu-sd/MechET/pull/117
7,Reaction-first,SIMULATED-TARGET,5838,2510,0.4299417608770127,3370,0.5772524837273039,original route equality,https://github.com/wangyu-sd/MechET/pull/117
8,Retro-E Electron,SIMULATED-TARGET,5838,2715,0.46505652620760535,3560,0.6097978759849263,original route equality,https://github.com/wangyu-sd/MechET/pull/117
```

## 09 · 08_电子搜索曲线

```csv
index,Figure 3a · Search Solved@Budget,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4
0,96 个独立目标、32/32/32 线性/汇聚/长路线，同一 Actor 和实际调用预算；全部数值为 SIM。,,,,
1,,,,,
2,Physical e-calls,Controller,N,Closed N,Closed rate
3,16,Independent,96,6,0.0625
4,16,Reaction-first,96,7,0.07291666666666667
5,16,Electron-level,96,9,0.09375
6,32,Independent,96,9,0.09375
7,32,Reaction-first,96,11,0.11458333333333333
8,32,Electron-level,96,15,0.15625
9,64,Independent,96,13,0.13541666666666666
10,64,Reaction-first,96,15,0.15625
11,64,Electron-level,96,23,0.23958333333333334
12,128,Independent,96,17,0.17708333333333334
13,128,Reaction-first,96,19,0.19791666666666666
14,128,Electron-level,96,30,0.3125
```

## 10 · 09_电子搜索配对

```csv
index,Figure 3a · Pairwise E vs Reaction,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,总目标数 = both + E-only + R-only + neither；ELRG=(E-only - R-only)/N，可直接做 McNemar 配对检验。,,,,,,,,
1,,,,,,,,,
2,B,N,Both,Electron-only,Reaction-only,Neither,Reaction solved,Electron solved,ELRG
3,16,96,6,3,1,86,7,9,0.020833333333333332
4,32,96,9,6,2,79,11,15,0.041666666666666664
5,64,96,13,10,2,71,15,23,0.08333333333333333
6,128,96,17,13,2,64,19,30,0.11458333333333333
7,64,1084,152,53,12,867,164,205,0.03782287822878229
```

## 11 · 10_ReFIT后训练

```csv
index,Figure 4b · Actual policy training expected table,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,499 SYNTHESIS-NEEDED 目标：真实 Adaptive @112 72→71；所展示的 Full ReFIT 96 尚未出现。,,,,,,,,
1,,,,,,,,,
2,Actor,Evidence,N,Solved@64,Rate@64,Solved@112,Rate@112,Caution,Source
3,StageII baseline,REAL @112; SIM @64,499,67,0.1342685370741483,72,0.14428857715430862,@112真实72；@64仅作为方案,https://github.com/wangyu-sd/MechET/pull/117
4,Adaptive update64,REAL-INTERNAL,499,,,71,0.14228456913827656,@112真实对照72→71,https://github.com/wangyu-sd/MechET/pull/117
5,Route RL,SIMULATED-TARGET,499,73,0.1462925851703407,77,0.15430861723446893,独立权重训练,https://github.com/wangyu-sd/MechET/pull/117
6,ReFIT without history,SIMULATED-TARGET,499,76,0.1523046092184369,80,0.16032064128256512,独立权重训练,https://github.com/wangyu-sd/MechET/pull/117
7,Full ReFIT-PO,SIMULATED-TARGET,499,89,0.17835671342685372,96,0.19238476953907815,最终强目标,https://github.com/wangyu-sd/MechET/pull/117
```

## 12 · 11_ReFIT配对

```csv
index,Figure 4b · SFT vs ReFIT matched-target outcome table,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,Only-ReFIT / Only-SFT 是同一批目标的两侧独有成功数；独立真实 RL 训练与完整留出策略评估必需。,,,,,,,,
1,,,,,,,,,
2,Call budget,N,Both solved,ReFIT-only,SFT-only,Neither,SFT solved,ReFIT solved,Gain
3,64,499,55,34,12,398,67,89,0.04408817635270541
4,112,499,60,36,12,391,72,96,0.04809619238476954
```

## 13 · 12_电子信用数据

```csv
index,Figure 3b / 4c–d · Real vs Expected Branch Evidence,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5
0,现有同状态有成败结局的 FLOW 对很少；不能用训练聚合系数直接证明独立策略泛化。,,,,,
1,,,,,,
2,Item,Evidence,Count,Unit,Note,Link
3,Actual observed TRAIN same-input pairs,REAL-INTERNAL,48,pairs,update 125,https://github.com/wangyu-sd/MechET/pull/117
4,Observed TRAIN eligible groups,REAL-INTERNAL,47,groups,244168 learner groups,https://github.com/wangyu-sd/MechET/pull/117
5,Distinct mapped FLOW pairs,REAL-INTERNAL,35,pairs,as of update125,https://github.com/wangyu-sd/MechET/pull/117
6,Future TRAIN contrasting FLOW pairs,SIMULATED-TARGET,350,pairs,real executed legal alternatives,https://github.com/wangyu-sd/MechET/pull/117
7,Frozen heldout action states,SIMULATED-TARGET,240,states,no RL training overlap,
8,SFT closing-action preference,SIMULATED-TARGET,132,/240,55.0%,
9,Route RL closing-action preference,SIMULATED-TARGET,151,/240,62.9%,
10,ReFIT without history preference,SIMULATED-TARGET,158,/240,65.8%,
11,Full ReFIT closing-action preference,SIMULATED-TARGET,173,/240,72.1%,
12,Outcome-different states,SIMULATED-TARGET,70,/240,forced matched continuations,
13,Teacher favors eventual failing action,SIMULATED-TARGET,42,/70,outcome-selected subset only,
```

## 14 · 13_电子组合泛化

```csv
index,Figure 3c · Rare / Unseen composition,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,"公开 RetroChimera 没有在这些设想的 3,120 个目标上发表成绩；相应对照数字是 SIM 重跑设想。",,,,,,,
1,,,,,,,,
2,Subset,N,Model,Evidence,Exact hits,Exact %,Criterion,Source
3,Frequent,2100,RetroChimera rerun,PLANNED-RERUN,1428,0.68,Exact precursor / mapped replay,https://github.com/microsoft/retrochimera
4,Frequent,2100,Stage-II,SIMULATED-TARGET,1239,0.59,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
5,Frequent,2100,ReFIT,SIMULATED-TARGET,1302,0.62,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
6,Rare,700,RetroChimera rerun,PLANNED-RERUN,357,0.51,Exact precursor / mapped replay,https://github.com/microsoft/retrochimera
7,Rare,700,Stage-II,SIMULATED-TARGET,299,0.42714285714285716,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
8,Rare,700,ReFIT,SIMULATED-TARGET,332,0.4742857142857143,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
9,Unseen combo,320,RetroChimera rerun,PLANNED-RERUN,99,0.309375,Exact precursor / mapped replay,https://github.com/microsoft/retrochimera
10,Unseen combo,320,Stage-II,SIMULATED-TARGET,80,0.25,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
11,Unseen combo,320,ReFIT,SIMULATED-TARGET,106,0.33125,Exact precursor / mapped replay,https://github.com/wangyu-sd/MechET/pull/117
```

## 15 · 14_失败修复与库存

```csv
index,Figure 3d / 6c · Repair & Inventory robustness,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,150 个已记录执行器失败状态，240 个最初被所有方法解决的共同目标；不允许只挑成功实例。,,,,,,,
1,,,,,,,,
2,Task,N,Controller,Different legal successor,Rate,Complete rescue,Recovery rate,Evidence
3,Reaction Repair,150,Independent retry,52,0.3466666666666667,12,0.08,SIMULATED-TARGET
4,Reaction Repair,150,Reaction rewind,68,0.4533333333333333,19,0.12666666666666668,SIMULATED-TARGET
5,Reaction Repair,150,Electron rewind,91,0.6066666666666667,32,0.21333333333333335,SIMULATED-TARGET
6,Stock Perturb,240,RetroChimera search,39,0.1625,39,0.1625,SIMULATED-TARGET
7,Stock Perturb,240,Reaction-first,35,0.14583333333333334,35,0.14583333333333334,SIMULATED-TARGET
8,Stock Perturb,240,Electron-level,61,0.25416666666666665,61,0.25416666666666665,SIMULATED-TARGET
```

## 16 · 15_蒸馏与效率

```csv
index,Figure 5 · Search amortization conditional result,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,全部数值为 SIM；Student 尚未训练成功。2.09x 指全搜 94 GPUh / 相同 205 solved 的 Student 45 GPUh。,,,,,,,,
1,,,,,,,,,
2,Method,N,Physical cap,Solved N,Solved rate,Total GPUh,GPU-sec / solved,Full search cost ratio,Evidence
3,Stage-II,1084,64,126,0.11623616236162361,45,1285.7142857142858,1.2839024390243903,SIMULATED-TARGET
4,ReFIT Policy,1084,64,145,0.13376383763837638,47,1166.896551724138,1.4146341463414636,SIMULATED-TARGET
5,Retro-E Full Search,1084,64,205,0.18911439114391143,94,1650.7317073170732,1,SIMULATED-TARGET
6,Student Policy,1084,32,183,0.1688191881918819,31,609.8360655737705,2.7068450039339105,SIMULATED-TARGET
7,Student Policy,1084,64,205,0.18911439114391143,45,790.2439024390244,2.088888888888889,SIMULATED-TARGET
8,Student + Search,1084,64,210,0.1937269372693727,60,1028.5714285714287,1.6048780487804877,SIMULATED-TARGET
```

## 17 · 16_化学证据

```csv
index,Figure 6d · Chemistry evidence vs published Nature standard,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7
0,相似文献、URSA Solv-2、机理执行、专家路线认可不是一个评价指标；RetroChimera 的 9/10 是特选案例。,,,,,,,
1,,,,,,,,
2,Source,Evidence,Unit,N,Accept N,Accept rate,Definition,URL
3,RetroChimera Nature,REAL-PUBLISHED,Curated reactions,291,,0.969,Expert accepted step rate,https://www.nature.com/articles/s41586-026-11160-9
4,RetroChimera Nature,REAL-PUBLISHED,Chosen difficult routes,10,9,0.9,Fully expert-accepted route,https://www.nature.com/articles/s41586-026-11160-9
5,Retro-E expected,SIMULATED-TARGET,Blind routes per method,80,56,0.7,Full route accepted by ≥3 experts,
6,Retro-E expected,SIMULATED-TARGET,Independent reaction steps,320,278,0.86875,Independent similar literature precedent,
7,Retro-E expected,SIMULATED-TARGET,URSA route subset,80,64,0.8,Solv-2 on frozen scorer,
8,Retro-E expected,SIMULATED-TARGET,Native full electronic replay,80,80,1,Executable mechanism per route,
```

## 18 · 17_主图逐项

```csv
index,Main Manuscript Figures · 6 Figures and 22 Panels,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4
0,论文成文时每个结果面板有明确数据、证据和对照；概念图仅在 Figure1。,,,,
1,,,,,
2,Panel,Title,Dataset / N,Reported metric,Data sheet
3,Fig1a,Framework,Electron action vs reaction endpoint,Actual structural illustration,concept
4,Fig1b,Executor,FLOW/IMPORT/FINISH mapped replay,Real chemistry and engine,concept
5,Fig1c,ReFIT-PO,"Actor, long-term credit and control",Actual objective and algorithm,concept
6,Fig2a,URSA main performance,URSA Expert100 + Drugs100,Solv0/1/2 and MV,04_URSA结果
7,Fig2b,CbyD hard chemistry,500 distinct CbyD targets,Solv2 and SDR,05_CbyD主成绩
8,Fig2c,Complexity dependence,CbyD LLS 100/200/200,Solved by length,06_CbyD难度分层
9,Fig2d,Modern generative planning,RetroBench 5838,Top1/Top5,07_RetroBench
10,Fig3a,Electronic search gain,96 paired and 1084 supplemental,ELRG and unique closed,08_电子搜索曲线
11,Fig3b,Electronic decision value,240 matched same-state legal FLOW,Different full-route outcomes,12_电子信用数据
12,Fig3c,OOD electronic composition,3120 grouped reactions,New combination exact recovery,13_电子组合泛化
13,Fig3d,Repair and backtracking,150 real failure states,Valid alternative and rescue,14_失败修复与库存
14,Fig4a,Local vs long credit,350 target TRAIN contrasts,Misranking / score reversal,12_电子信用数据
15,Fig4b,Actual RL improvement,499 frozen VALID,Paired solved@112,10_ReFIT后训练
16,Fig4c,Transfer to held-out choices,240 states,Policy preference 55%→72.1%,12_电子信用数据
17,Fig4d,Delayed reward depth,200+ genuine pairs,Preference by 1/2–3/4+ steps,future real data
18,Fig5a,Search distillation,20–50k actual eligible actions,Policy training actual,conditional
19,Fig5b,GPU cost to credible route,1084 targets and URSA,GPU-sec per Solv2,15_蒸馏与效率
20,Fig5c,Efficiency diagnostics,real long-tail rollouts,Unique actions/call,PR117
21,Fig6a,Strategic disconnection,CbyD 500,SDR@1 82→133,05_CbyD主成绩
22,Fig6b,Stock re-planning,240 originally common solved,Recovery 35→61,14_失败修复与库存
23,Fig6c,Chemical synthesis examples,5–8 source-traceable real routes,Molecules + electron arrows,future real data
24,Fig6d,Independent credible chemistry,"80 blind routes, ~320 steps",Expert / precedent / Solv2,16_化学证据
```

## 19 · 18_指标字典

```csv
index,Exact Metric Definitions,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4
0,每个新指标可以放进论文 Methods，但应同时保留标准 Solv-N 指标和原始分子、方法、预算数据。,,,,
1,,,,,
2,Metric,Formula,Definition/issue,Type,Source
3,Solv-0,N_complete_stock / N_targets,"stock-terminated, valid nonzero route",URSA,https://github.com/insilicomedicine/URSA
4,Solv-1,N_legalcenter / N_targets,Solv0 + all reactions legitimate center,URSA,https://github.com/insilicomedicine/URSA
5,Solv-2,N_chemplausible / N_targets,Solv0 + strict ChemCensor,URSA,https://github.com/insilicomedicine/URSA
6,MV-Solv,N_solv2_and_independent_mapped_flow / N_targets,Use same independent validator for external systems,PROPOSED,
7,ELRG@B,(N_Eonly - N_Ronly) / N_targets,"Same Actor, same targets, same physical B",PROPOSED,
8,SDR@1,N_root_strategic_cut_hit / N_targets,Literature key cut recovery is not chemical truth,PROPOSED,https://github.com/chemistry-by-design/cbyd
9,Route Top1/5,N_reference_route_match / N_targets,"RetroBench route equality, not stock closure",RetroBench,https://aclanthology.org/2026.acl-long.1745/
10,FLOW decision value,"Q_route(s,a1)-Q_route(s,a2)",Same saved state & fixed continuation budget,PROPOSED,
11,Repair recovery,N_failed_states_full_route_rescued / N_failed_states,Do not equate new valid action with route closure,PROPOSED,
12,Stock perturb,N_reclosed_after_same_stock_removal / N_intervened,Common solved cohort only,PROPOSED,
13,Route credible diversity,Mean distinct chemically approved route classes / target,Dedup superficial electronic variants,PROPOSED,
14,GPU-sec / Solv2,total measured GPU seconds / N_solv2,Same hardware + time and overall request accounting,PROPOSED,
15,Chemist acceptance,N_routes_all_steps_accepted / N_blind_routes,Requires truly independent chemists,PROPOSED,https://www.nature.com/articles/s41586-026-11160-9
16,Step precedent coverage,N_steps_literature_matched / N_eligible_steps,Not equivalent to prospectively verified reaction,PROPOSED,
```

## 20 · 19_执行缺口

```csv
index,Workstreams & Execution Gates,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4
0,先取得真实外部基线与 CbyD 难度，再决定是否扩展更多训练。不能用模拟值替代运行状态。,,,,
1,,,,,
2,Priority,Experiment,Scale,Acceptance criteria,Real current state
3,P0,URSA download + stock + scorer freeze,200 targets,"Hashed IDs, U2/ChemCensor versions",not run
4,P0,CbyD 500 preregistration,500 distinct targets,No overlap 27 dev; route graph & key cut,not run
5,P0,External RC-MCTS/ AZF/ DMS rerun,URSA200 + CbyD500,"Independent same-target results, compute measured",not run
6,P0,96 frozen actor-matched comparison,96*3*64 max calls,Independent sampling/reaction/electron paired,24 dev all zero
7,P0,Policy checkpoint transfer verification,"499 VALID, @112",Need real Full ReFIT > baseline 72,actual update64 71
8,P1,RetroBench main model comparisons,5838 targets,Original route Top1/Top5 scorer,not run
9,P1,Electronic same-state causal dataset,350 TRAIN /240 heldout,"Distinct legal mapped FLOW, real downstream",48 current TRAIN observed pairs
10,P1,CbyD difficult routes,500 targets,"Full stock, ChemCensor, root strategic cut","27 dev calibrated, no test"
11,P2,Reaction repair and stock robustness,150 failures/240 original successes,"same intervention, matched budget",not run
12,P2,Distillation gate,"Student, real 20k+ usable records",Actual student improves cost-quality,student not trained
13,P2,Chemistry blind review,"80 per model, ideally three experts",Frozen unbiased route sampling,not run
14,P2,Full CbyD scaling,2902 unique targets,Only after 500 completed,optional
```

## 21 · 20_真实逐目标模板

```csv
index,Raw Data Contract · Target × Model × Budget × Seed,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9,Unnamed: 10,Unnamed: 11,Unnamed: 12,Unnamed: 13,Unnamed: 14,Unnamed: 15,Unnamed: 16,Unnamed: 17,Unnamed: 18,Unnamed: 19,Unnamed: 20
0,此表故意不填虚构 SMILES/目标 ID。真实原始结果填入后再由代码聚合至各 Results 表。,,,,,,,,,,,,,,,,,,,,
1,,,,,,,,,,,,,,,,,,,,,
2,target_id,canonical_smiles,benchmark,split,method,actor_sha,stock_sha,scorer_version,seed,budget,calls_real,tokens,GPU_s,wall_s,Solv0,Solv1,Solv2,MV,SDR,route_json,evidence_json
3,REPLACE_REAL_ID,,URSA-expert-2026,test,Retro-E,,,U2-ChemCensor1.2,0,256,,,,,,,,,,,
4,REPLACE_REAL_ID,,CbyD-v2026.11,test,RetroChimera-MCTS,,,U2-ChemCensor1.2,0,,,,,,,,,,,,
```

## 22 · 21_AI分子512

```csv
index,Figure 6 optional · AI-designed Molecules 512,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9,Unnamed: 10,Unnamed: 11,Unnamed: 12
0,对应先前的 Synthesis-Evidence Dark Space 方案。尚未核实最终 512 个公开生成分子 ID；所有数字包括外部模型重跑都是 SIM。,,,,,,,,,,,,
1,,,,,,,,,,,,,
2,Method,Evidence,N,Solv0 N,Solv0 rate,Solv2 N,Solv2 rate,MV N,MV rate,Full step precedent N,Full precedent rate,Precondition,Source
3,RetroChimera+MCTS,PLANNED-RERUN,512,49,0.095703125,35,0.068359375,24,0.046875,15,0.029296875,"public generated molecule sample, fixed stock",https://github.com/microsoft/retrochimera
4,Stage-II SFT,SIMULATED-TARGET,512,33,0.064453125,22,0.04296875,14,0.02734375,8,0.015625,same generated molecule sample,https://github.com/wangyu-sd/MechET/pull/117
5,Reaction-first Search,SIMULATED-TARGET,512,42,0.08203125,31,0.060546875,24,0.046875,19,0.037109375,same trained actor,https://github.com/wangyu-sd/MechET/pull/117
6,Retro-E Electron Search,SIMULATED-TARGET,512,78,0.15234375,65,0.126953125,52,0.1015625,34,0.06640625,stretch application target,https://github.com/wangyu-sd/MechET/pull/117
```

## 23 · 22_机理专家300

```csv
index,Mechanistic Challenge · PMechDB 300 expected results,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9
0,这里不是 FlowER 的正向反应预测任务；不可将 Forward FlowER Top-k 直接填作同任务基线。数据与参考标注需核验。,,,,,,,,,
1,,,,,,,,,,
2,Benchmark,Model,Evidence,N,Reference key event N,Accuracy,Mapped executable N,Executable rate,Definition,Source
3,PMechDB 300,Stage-II,SIMULATED-TARGET,300,176,0.5866666666666667,247,0.8233333333333334,key electron changes match reference; mapping replay,https://github.com/wangyu-sd/MechET/pull/117
4,PMechDB 300,ReFIT,SIMULATED-TARGET,300,199,0.6633333333333333,266,0.8866666666666667,key electron changes match reference; mapping replay,https://github.com/wangyu-sd/MechET/pull/117
5,PMechDB 300,Retro-E,SIMULATED-TARGET,300,207,0.69,278,0.9266666666666666,key electron changes match reference; mapping replay,https://github.com/wangyu-sd/MechET/pull/117
```

## 24 · 23_可信多样性

```csv
index,Credible Route Diversity · distinct chemistry not just different FLOW,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8,Unnamed: 9
0,不同电子轨迹若得到相同前体/同一战略断键，不应被重复记作新路线。跨模型独立可行性通过同一验证器确认。,,,,,,,,,
1,,,,,,,,,,
2,Dataset,Model,Evidence,N,At least one Solv0 target,Solv2 target count,Total credible route classes,Classes / target,Rule,Source
3,CbyD500,RetroChimera-MCTS,PLANNED-RERUN,500,36,17,22,0.044,credibly distinct strategic complete routes,https://github.com/chemistry-by-design/cbyd
4,CbyD500,Reaction-first,SIMULATED-TARGET,500,39,22,30,0.06,same Actor reaction-level search,https://github.com/wangyu-sd/MechET/pull/117
5,CbyD500,Retro-E Electron,SIMULATED-TARGET,500,52,34,49,0.098,same Actor electron-level search,https://github.com/wangyu-sd/MechET/pull/117
6,URSA200,Retro-E Electron,SIMULATED-TARGET,200,191,108,136,0.68,"combined two URSA subsets, distinct route classes",
```

## 25 · 24_MV_Solv

```csv
index,Proposed MV-Solv Across Benchmarks,Unnamed: 1,Unnamed: 2,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,Unnamed: 8
0,MV-Solv 是 Retro-E 拟提出的额外严格评价；外部系统不能因为无原生电子程序而被赋 0，必须运行共同验证器。,,,,,,,,
1,,,,,,,,,
2,Dataset,Model,Evidence,N,Solv2 N,MV N,MV rate,Verifier requirement,Source
3,URSA Expert,RetroChimera-MCTS,REAL-PUBLISHED / MV-PENDING,100,32,,,external independent mechanism checker not run,https://github.com/insilicomedicine/URSA
4,URSA Expert,Retro-E Electron,SIMULATED-TARGET,100,41,34,0.34,must replay mapped sequence and independently validate,https://github.com/wangyu-sd/MechET/pull/117
5,URSA Drugs,RetroChimera-MCTS,REAL-PUBLISHED / MV-PENDING,100,60,,,external independent mechanism checker not run,https://github.com/insilicomedicine/URSA
6,URSA Drugs,Retro-E Electron,SIMULATED-TARGET,100,67,57,0.57,same scoring code as external,https://github.com/wangyu-sd/MechET/pull/117
7,CbyD 500,RetroChimera-MCTS,PLANNED-RERUN,500,17,,,CbyD external baselines not yet run,https://github.com/chemistry-by-design/cbyd
8,CbyD 500,Retro-E Electron,SIMULATED-TARGET,500,34,25,0.05,500 frozen distinct targets,https://github.com/wangyu-sd/MechET/pull/117
9,Generated 512,Retro-E Electron,SIMULATED-TARGET,512,52,42,0.08203125,full route chemistry + mapped replay,
```

