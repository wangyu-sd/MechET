# PR111 加速实现与部署边界

设计：[PR111](https://github.com/wangyu-sd/MechET/pull/111)。私有运行源码提交：`f1fb6e590f41cf4b618701ab097661d5a42be478`。用户后续明确授权本地 T4 合法性/速度检查后，在原 D3/D1 任务上从完整断点切换加速版本；该授权取代早先不替换旧作业的部署限制。原数据、奖励和研究顺序保留。

## 已实现

| 改动 | 实际行为 |
|---|---|
| FLOW 化学执行缓存 | 同一映射状态、grounded动作及执行合同只计算一次；并发重复共享在途 Future。逐次重建各自历史、参数、概率分数和 token 数，循环检查独立执行。IMPORT 内存准入和 FINISH 路线边界不缓存。 |
| 额外重复后续停止 | 只停止额外 ReFIT 的已接受重复 FLOW 后续，标记 `duplicate_pruned_no_continuation`，reward/credit未知，不作为负样本，也不领取缓存 Q。原始四条 G4 路线仍完整独立执行。 |
| 动态 G4 目标领取 | 全部源行目标进入带持久化 lease 的队列；每个目标及其 G4/分支由一个 CPU owner 管理。已提交结果按哈希恢复，未完成目标可在重启后重新领取。 |
| 跨卡模型请求 | 请求通过认证内存 RPC 到同轮 actor 的任意空闲 GPU，原 CPU owner 接收真实 token/log-prob 并执行化学、分叉和评分。没有逐 token Ceph 轮询。 |
| 消除过早 NCCL 等待 | 快 owner 在 CPU 协调器等待全部目标完成，GPU worker 继续服务其他 owner；全部请求对账、GPU worker停止后才进入学习。 |
| HF 采样效率 | 新条件按长度组织生成批次；请求ID/解码位置决定 counter RNG，用批量逆CDF分类抽样代替每行GPU multinomial。原始log-prob在pin token前捕获；HF外层只选已抽出的token，不再做第二次点质量抽样。 |
| 学习批次 | 新条件配置从1提升到4，沿用长度/容量分批、每真实回执一次及原始乘数；真实前向、梯度和SGD参数更新已验证等价。 |
| 稳定性 | 单卡初始化失败和rank0汇总写盘失败均共同传播；内存准入使用全部物理执行并发数；资源/基础设施错误不缓存为化学拒绝；GPU请求有基础设施超时，非法概率有持久诊断。 |

当前阶段没有改奖励、FLOW alpha=.5、M=1/equal-semantic baseline 或隐藏参考核验。重复动作的真实回执保留；缓存复用次数不会变成额外独立成功证据。该停止策略会改变额外搜索分布，因此只作为新版本条件，不声称与旧版搜索结果完全等价。

## 旧任务的完整轮次只读审计

读取旧任务 `round-000009` 四个rank的全部已保存结果（64个源行目标），没有调用模型：

- 256条原始G4 + 3,044条额外后续 = 3,300条报告。
- 63,925个唯一真实动作请求；9,201次相同精确条件/语义动作的重复提案。
- 其中9,087次为已接受的重复动作；4,038次为保留决策节点上的额外重复FLOW提案。
- 这些不是独立成功路线次数，也不是已测得的加速倍数；跨不同历史的纯化学复用还可能更多。

脚本：[audit_refit_compute.py](../../scripts/audit_refit_compute.py)。审计：[saved_round_audit.json](../audits/pr111_acceleration_20261117/saved_round_audit.json)。

## 验证

最终相关CPU套件 **74项通过**。包括真实化学执行、并发缓存、拒绝/未知结果、全G4归属、重启恢复、双进程Gloo跨owner请求、单卡启动/汇总落盘故障传播，以及零有效本地组rank的真实梯度同步和相同参数更新。

正式任务相同的固定 Transformers5.5.0 / Torch2.6 运行包下，另外 **15项小模型数值测试通过**，覆盖原始/新抽样、FP32/BF16概率与反向传播。它们是软件数值测试，不是多步数据子集试验、benchmark或H20提速结果。两个检查集合不重叠。

初次使用宿主默认环境时，模型检查因该环境的旧Transformers没有Qwen3.5、以及MKL线程层冲突失败；已换固定运行包并设置GNU后完成上述检查。初次失败日志保留，不冒充一开始全绿。

独立审查发现的并发内存准入、单卡启动异常、汇总落盘同步错位均有先失败后通过的回归；CLI新增`--acceleration`。最后GPU线程设备绑定修复后的6项受影响集成检查也通过，与74项重叠。证据：[verification.json](../audits/pr111_acceleration_20261117/verification.json)。

## 初版准备记录（被后续断点续跑授权取代）

**H20实际吞吐、端到端更新时间和同预算路线质量尚未测得。** 不能用CPU单次化学调用减少或GPU-util上升代替正式提速。

当前选择HF fallback。真连续解码/EOS补位、跨请求prefix KV和vLLM数值准入尚未完成，不能写成已启用；可选reference-logprob缓存也未启用。新broker的native Future接口保留，但完整D3新条件当前只接受HF后端。

完整D3新条件通过`prepare_standard_route_rl.py --condition D --acceleration`单独冻结，从同一个修正Stage-II History actor开始，数据/VALID/库存保持原合同；不从旧作业中途混入改变后的采样器。准备脚本不提交作业。

继续顺序：新条件H20准入与同预算工作/等待对照 → 完整D3路线RL及固定VALID → Reference1,084和n5-10,000正式测评 → 最后TRAIN-only自搜索蒸馏及独立评测。原有D3、D1继续执行。遵守PR111不停止/原地修改旧任务及总计最多8H20的要求，本次不新增GPU作业，也没有启动smoke/pilot或小子集实验。

运行栈的私有依赖沿用已有冻结环境；修改的私有控制器以审计source镜像和补丁公开，不声称公共仓库已经包含私有数据、权重或完整运行环境。

## 2026-11-17 本地 T4 准入和同名任务续跑

运行源码 `4fca18b1a2f9f1f38035f8fb096a94a741449654`。本次允许小规模本地软件核验；正式任务仍为原全量协议。新增[迁移逻辑](../../src/mechet/route_acceleration_continuation.py)、[冻结/复制 CLI](../../scripts/prepare_acceleration_continuation.py)、[迁移测试](../../tests/test_acceleration_continuation.py)、[T4 对照脚本](../../scripts/validate_pr111_t4.py)和[最终断点核验](../../scripts/verify_acceleration_handoff_t4.py)。迁移相关 CPU 套件32项通过；重复测试不叠加成总数。

T4 不支持生产 BF16 内核，核验用 FP32/unfused；H20 仍保留原 BF16/FLA。选用 `private_pinned_argmax_v1`，保持每请求 PrivateSampling 抽样，仅移除外层对已 pin token 的再次抽样；counter categorical 保留为可选实现。四输入固定完整响应的生成对照 D3 0.996×、D1 1.006×，基本不变。4个不同真实已保存输入的 actor 前向/反向 B1→B4：D3 1.018×、D1 1.023×，梯度相对差分别1.51e-5、1.75e-5；没有做优化器更新，参考模型计算不在该计时内。不能宣称端到端或多卡提速。

首次梯度检查误用了相同 G4 输入与和为零的数值乘数，真实梯度抵消，导致相对误差判定无意义；改用4个不同目标，保留原失败记录和同一个2%阈值。固定响应仅用于速度计时，不作为 on-policy RL 数据。实际自由生成与 teacher-forced logprob 均单独核验。停源后最终 D3 step13、D1 step83 真实 adapter 各2次完整自由生成通过，logprob 最大差2.32e-6、1.22e-6。[准入数据与原始计时](../audits/pr111_acceleration_20261117/t4_continuation_admission.json)。

| 条件 | 迁移起点 | TRAIN/VALID | 新实例 |
|---|---:|---:|---|
| D3 ReFIT | optimizer13 / next_round13 |156723/8250|`8b1d891ea1cfaab201a1e94d21c428e8`|
| D1 原始 process RL | optimizer83 / next_round83 |23730/2986|`8b1d89f6a1cfaa4801a1e94d2a7e28d3`|

原实例 END、全部源 worker 锁释放、最终 complete/adapter 哈希核对后，复制 actor、optimizer、四 rank RNG、全量顺序和进度；新 complete 显式绑定新合同及源 complete。源输出不变；未提交轮次重新采集。D1 保留8次恢复/原 reward、不增加 ReFIT；D3 保留 first-rejection/ReFIT/原 reward。新增 loss_batch_size4 和显式 acceleration 配置。成功修改同名任务并回读新启动命令后启动，各4H20，总上限8。平台修改 CLI 缺认证字段会返回错误但退出0，已补充私有 Token 配置并以接口输出与配置回读共同核验；凭据不公开。

当前启动回执已接受，PENDING/容器加载/恢复后的新更新分别跟踪，seeded13/83不能算新更新。见[部署回执](../audits/pr111_acceleration_20261117/continuation_deployment.json)。只读监控已启动，无自动重提交。继续全量 RL/VALID → Reference1084/n5-10000 → 最后 TRAIN-only 搜索蒸馏。真实 H20 轮次加速和质量仍需后续同预算数据。
