# RetroBridge 本地修改记录

## 1. 对比基准

- 本地目录：`/home/estar/pxy/mechet/MechET/baselines/RetroBridge-main`
- 官方仓库：`https://github.com/igashov/retrobridge`
- 官方分支：`main`
- 官方提交：`5442b4f45edc2e956f1d1c1763bb94cefd4a3a13`
- 官方提交时间：`2024-03-26T11:07:25Z`
- 对比日期：`2026-08-27`（UTC）
- 官方提交链接：`https://github.com/igashov/RetroBridge/commit/5442b4f45edc2e956f1d1c1763bb94cefd4a3a13`

本地 RetroBridge 目录不是独立 Git checkout，而是父级 MechET 工作树中的未跟踪目录，因此无法提供一个独立的本地 RetroBridge commit。此次记录通过下载上述官方提交对应的 `main` 快照并执行逐文件比较得到。

比较核心源码时排除了以下本地运行资产：

- `.venv-retrobridge/`
- `__pycache__/` 和 `*.pyc`
- `*.log`
- 已生成的 `errorlog.txt`
- `audits/` 中的模型、处理后数据和评测产物

文件级结果：

- 修改了 12 个官方文件。
- 新增了 MechET 数据准备、审计、推理、日志、配置和测试文件。
- 没有删除官方文件。
- 官方模型主体、噪声日程和 Markov bridge 数学过程没有被重写。

## 2. 修改总览

本地修改主要解决以下问题：

1. 让 RetroBridge 能训练 FlowER full 和 Mech-USPTO-31K，而不再依赖官方 USPTO-50K 的固定原子词表和固定 10 个 dummy node 上限。
2. 保存 MechET `stable_id`，建立可审计的数据转换和统一 JSONL 推理输出。
3. 修复 padding/dummy node 引起的拉普拉斯特征分解不收敛。
4. 兼容当前 PyTorch 2.8 和 PyTorch Lightning 2.3.3。
5. 将 W&B 日志切换为 SwanLab，同时始终保留本地 CSV 日志。
6. 为大图数据增加梯度累积、可靠的 `last.ckpt` 保存和 GPU 等待启动器。

## 3. 官方文件中的修改

### 3.1 `src/data/retrobridge_dataset.py`

#### 修改内容

- 可选读取数据根目录下的 `metadata.json`。
- 从 metadata 动态设置 `atom_decoder`、原子编码字典和 `max_n_dummy_nodes`。
- 要求动态原子词表最后一个类型必须是 dummy 类型 `*`。
- 加入 `strict` 处理模式。
- 将 CSV 的 `id` 保存为每个 PyG 样本的 `stable_id`，同时保存 `source_row_idx`。
- 在 `swap=True` 时同步交换并保留 `stable_id` 和 `source_row_idx`。
- 将 `torch.load(path)` 改为 `torch.load(path, weights_only=False)`。
- 对 RDKit 解析失败、产物原子多于前体、dummy 容量不足、映射错误和图构造异常进行集中记录。
- strict 模式下不再静默丢弃失败行，而是在处理结束后抛出包含前 10 个失败原因的异常。
- 非 strict 测试集仍保留官方的 `C >> C` 占位回退行为。
- DataModule 将动态词表、dummy 容量和 metadata 传递给 `RetroBridgeDatasetInfos`。
- `possible_num_dummy_nodes` 根据动态上限生成，而不是固定为 `0..10`。

#### 修改原因

官方实现写死了 USPTO-50K 的 17 类节点词表和最多 10 个 dummy node。FlowER full 中存在官方词表之外的元素，并且前体与产物的原子数差可以明显超过 10。继续使用固定空间会导致无法编码、丢行或用错误占位样本替换测试数据。

动态 metadata 让图空间由训练数据或明确指定的官方 MIT 词表决定；`stable_id` 和 strict 模式则用于保证转换前后数量、顺序和化学图身份可审计。

#### 行为影响

- 没有 metadata 的官方 USPTO-50K 数据仍走原始兼容路径。
- 有 metadata 的 MechET 数据使用动态图空间并默认严格失败。
- 训练、验证和测试的处理后 `.pt` 格式新增了 `stable_id` 和 `source_row_idx` 字段。
- 旧的处理后缓存如果没有这些字段，应重新预处理，不能假定与新推理脚本兼容。

### 3.2 `src/features/extra_features.py`

#### 官方行为

官方代码将一个 batch 内所有图 padding 到同一节点数，并给每个 dummy node 设置相同对角值：

```python
mask_diag = 2 * max_nodes * I
```

随后直接对完整 batch 调用 `torch.linalg.eigh` 或 `torch.linalg.eigvalsh`。如果一个图有很多 dummy node，就会人为产生大量相同的非零特征值 `2 * max_nodes`。

#### 本地修改

- 删除原始 batch padding 产生的重复对角谱。
- 新增 `eigh_real_nodes()`，按有效节点数将图放入 16-node buckets，正常路径在 GPU 上批量 FP32 分解。
- bucket 内 padding 使用互不相同、且高于真实拉普拉斯谱上界的对角值，避免重新产生简并谱。
- 对孤立/dummy node 使用低于 `1e-5` 连通分量阈值的确定性 tie-break。
- 分解前再次执行 `(L + L.T) / 2`，确保数值对称。
- 增加空图检查以及 `NaN/Inf` 检查。
- 批量 FP32 分解失败时，仅将受影响 bucket 拆为逐图 GPU FP64 重试。
- GPU FP64 仍失败时回退 CPU LAPACK；CPU 仍失败时加入最大 `1e-10` jitter。
- 分解结果重新填充到 batch 原始形状，保持后续特征维度和接口不变。
- `extra_features=eigenvalues` 和 `extra_features=all` 两条路径都使用新求解器。

#### 修改原因

训练中出现过两次致命异常：

- 启动阶段：batch element 4，error code 5257。
- Epoch 8：batch element 0，error code 1。

两次错误都来自 `torch.linalg.eigh` 对高度退化或数值困难的拉普拉斯矩阵分解不收敛。单纯将 `L` 转为 float64 的中间修复仍然失败，因此最终去掉 batch padding 的重复谱、稳定孤立节点简并子空间，并加入分层回退。

#### 保持不变的语义

- 邻接矩阵仍由 `E_t[..., 1:]` 构造，即所有实际键类型都按“存在边”处理。
- 拉普拉斯矩阵仍为未归一化组合拉普拉斯 `L = D - A`。
- 最终仍使用连通分量数、前五个非零特征值、非最大连通分量指示值和前两个非零特征向量。

#### 代价

正常路径不再逐图调用 eigensolver；相近大小的图在同一 GPU kernel 中处理。只有异常 bucket 才进入逐图 FP64/CPU 路径，因此稳定性成本不再由每个 batch 承担。

### 3.3 `train.py`

#### 自定义数据指标

当 DataModule 检测到 MechET metadata 时：

- 使用 `DummyTrainMolecularMetricsDiscrete`。
- 使用 `DummySamplingMolecularMetrics`。

原因是官方详细原子级指标按 USPTO-50K 固定词表实现，动态原子词表可能造成维度不匹配。这些详细指标是诊断项，不参与 Markov bridge 训练损失。

#### checkpoint 策略

- 所有训练增加每 epoch 更新的 `last.ckpt`。
- 官方数据仍保留基于 `top_1_accuracy` 和 `top_5_accuracy` 的 top-5 checkpoint。
- MechET metadata 数据不注册 top-1/top-5 checkpoint callback，因为 dummy sampling metrics 不产生可靠的相应监控值。

原因是保证大规模训练可恢复，同时避免 custom-data 运行因不存在的监控指标而保存失败。

#### 日志系统

- 删除 W&B logger。
- 始终启用本地 `CSVLogger`。
- 默认同时启用自定义 `SwanLabLogger`。
- CLI 从 `--disable_wandb` 改为 `--disable_swanlab`。

#### 训练参数

- Trainer 新增 `accumulate_grad_batches`，默认值为 1。
- Trainer 支持配置 `devices`、`strategy`、`precision`、TF32 和 gradient clipping。
- FlowER 和 USPTO 正式配置使用 8-GPU DDP、BF16 mixed precision，global batch 均为 64。
- 自定义数据按完整 validation VLB 保存 `best.ckpt`，并支持 patience-based early stopping。
- 正式训练关闭周期性 500-step chain sampling；采样由独立 evaluation job 完成。

### 3.4 `src/frameworks/markov_bridge.py`

#### 修改内容

- 训练和验证日志显式传入 `batch_size=reactants.X.shape[0]`。
- `on_validation_epoch_end(self, outs)` 改为 `on_validation_epoch_end(self)`。

#### 修改原因

- 图 batch 是嵌套结构，Lightning 曾错误推断 batch size 为 845，显式指定后可避免 epoch 指标被错误加权。
- PyTorch Lightning 2.x 的 `on_validation_epoch_end` hook 不再接收 `outs` 参数。

Markov bridge 的 forward、VLB/CE 损失、转移概率和采样公式没有改变。

### 3.5 `src/frameworks/discrete_diffusion.py`

- 将 `on_validation_epoch_end(self, outs)` 改为 `on_validation_epoch_end(self)`。
- 原因是兼容 PyTorch Lightning 2.x hook 签名。
- 离散扩散算法和损失没有改变。

### 3.6 `src/frameworks/one_shot_model.py`

- 将 `on_validation_epoch_end(self, outs)` 改为 `on_validation_epoch_end(self)`。
- 原因是兼容 PyTorch Lightning 2.x hook 签名。
- OneShot 模型算法没有改变。

### 3.7 `src/analysis/visualization.py`

- 将 `wandb` 替换为 `swanlab`。
- 新增 `active_swanlab_run()`，没有活动 run 时不上传图片或 GIF。
- 分子图片使用 `swanlab.Image`，采样链使用 `swanlab.Video`。

原因是统一实验追踪平台，同时允许 `--disable_swanlab` 时仅写本地文件而不触发远程日志异常。

### 3.8 `src/loggers/swanlab_logger.py`（新增）

- 实现 `pytorch_lightning.loggers.logger.Logger` 接口。
- 支持 hyperparameter 序列化、逐 step metric 日志、run resume 和正常/异常结束状态。
- 仅 rank zero 创建和写入 SwanLab run。

原因是本项目使用旧命名空间 `pytorch_lightning`，而 SwanLab 的现成集成主要面向新版 `lightning` 包，因此增加了一层小型适配器。

### 3.9 `mit/train.py`

- 与主训练入口一致，将 W&B 替换为 CSVLogger + SwanLabLogger。
- CLI 改为 `--disable_swanlab`。
- MIT 训练算法、dummy metrics 和 checkpoint 逻辑没有其他变化。

### 3.10 `requirements.txt`

- 删除 `wandb==0.16.3`。
- 增加 `swanlab==0.9.7`。

其余官方固定版本保持不变。实际审计环境比官方 requirements 更新，审计记录显示使用过 Torch 2.8.0、PyG 2.7.0、Lightning 2.3.3、RDKit 2025.9.1、Pandas 2.3.3 和 NumPy 1.26.4。

### 3.11 官方 YAML 配置

修改文件：

- `configs/retrobridge.yaml`
- `configs/forwardbridge.yaml`
- `configs/digress.yaml`

三个文件只进行了日志配置替换：

```yaml
wandb_entity: null
```

替换为：

```yaml
swanlab_project: RetroBridge
swanlab_workspace: null
swanlab_mode: online
```

模型、数据、优化器和扩散超参数没有改变。

## 4. 新增的 MechET 工具

### 4.1 `prepare_mechet_retrobridge.py`

该脚本将 MechET endpoint JSONL 转换为官方 RetroBridge CSV 结构，并执行转换前兼容性检查。

主要功能：

- 读取 `train.jsonl`、`valid.jsonl` 和 `test.jsonl`。
- 使用 `precursor_mapped` 和 `product_mapped` 生成 `reactants>reagents>production`。
- 将 `stable_id` 写入 CSV 的 `id` 列。
- 检查 RDKit 可解析性、atom-map 唯一性、产物 map 是否为前体子集、映射元素一致性、原子数关系和键类型。
- 默认遇到任何不兼容行即失败，不静默修改源数据。
- `--drop-incompatible` 可显式删除不兼容行，但会在 metadata 中记录删除明细和评测分母变化。
- 支持从训练集推导原子词表，或使用论文公开的 MIT 固定词表。
- dummy 容量只从训练集最大原子差推导。
- 验证/测试出现训练空间外元素或超过训练 dummy 容量时失败。
- 支持小规模 limit 和 overfit 数据集。
- 输出 split 数量、stable ID 哈希、元素、原子差分布、失败详情和来源信息到 `metadata.json`。

### 4.2 `audit_mechet_retrobridge.py`

该脚本检查处理后的 PyG 图是否保持数据身份：

- split 数量与 metadata 一致。
- `stable_id` 唯一且顺序与 CSV 一致。
- mapped SMILES 和 unmapped SMILES 与输入一致。
- 原生元素/键图与处理后 PyG 图同构。
- 原子词表没有使用验证/测试信息。
- dummy 容量仅来自训练集。
- 记录 Python、依赖版本和仓库 revision。

审计失败时进程返回非零状态，可作为训练前置条件。

### 4.3 `sample_mechet_retrobridge.py`

该脚本为 MechET 统一评测协议新增推理入口：

- 从 checkpoint 加载 MarkovBridge。
- 检查 checkpoint 与数据的原子词表、dummy 容量一致。
- 保留每条数据的 `stable_id`。
- 每个输入执行指定数量的独立随机采样。
- 将预测图解码为 canonical SMILES，并记录有效性和解码异常。
- 输出统一 `predictions.jsonl`。
- 另存包含 NLL、ELL、随机种子、采样步数和候选有效性的 trace JSONL。
- 支持 `--limit`、`--n-steps`、`--n-samples` 和 `--use-one-hot`，便于 smoke test。

该脚本的候选 rank 当前是生成顺序。已有 smoke audit 显示原生 `sample_batch` 返回的 score 全为 0，因此不能把当前 rank 解释为经过可靠似然重排的名次。

### 4.4 `wait_for_gpu_and_train_uspto.sh`

- 只监控 GPU `0,1,4,5,6,7`。
- 每 60 秒检查一次显存使用率是否低于 1%。
- 启动前检查 Python、配置和图审计报告存在，且审计 `passed=true`。
- 使用 `flock` 防止多个 waiter 或训练进程重复启动。
- 找到空闲卡后设置 `CUDA_VISIBLE_DEVICES` 和 `MPLCONFIGDIR`，并以 `--disable_swanlab` 启动 USPTO 训练。

### 4.5 `tests/test_extra_features.py`

谱分解单元测试覆盖：

- `extra_features=all/eigenvalues` 的批量 FP32 快路径和输出 dtype。
- 不同图大小的 bucket 合并。
- 批量失败后只对受影响图执行 FP64 fallback。
- 非连续 node mask 的显式拒绝。

当前测试尚未覆盖真实 CUDA 驱动错误后的 CPU/jitter 二级 fallback、strict dataset failure 和端到端 DDP checkpoint 恢复。

## 5. 新增配置

### 5.1 `configs/mechet_retrobridge_flower_full.yaml`

- 数据集：`flower_full`
- 扩散步数：500
- 谱/环额外特征：`all`
- 分子额外特征：关闭
- 8-GPU DDP，BF16 mixed precision
- per-GPU micro-batch：8
- 梯度累积：1
- 有效 batch：64
- 模型深度：5 层
- 输出路径指向 `/data/pxy/models/RetroBridge/flower_full`

该配置用于 FlowER full 正式训练。配置注释明确说明数据准备阶段排除了部分源映射异常，并改变了 train/test 分母。

### 5.2 `configs/mechet_retrobridge_mech_uspto_31k_full.yaml`

- 数据集：`mech_uspto_31k_full`
- 扩散步数：500
- 谱/环额外特征：`all`
- 分子额外特征：关闭
- 8-GPU DDP，BF16 mixed precision
- per-GPU micro-batch：8
- 梯度累积：1
- 有效 batch：64
- 模型深度：5 层
- `resume` 默认为空，避免在新环境错误绑定旧 checkpoint。

该配置用于 Mech-USPTO-31K full。注释要求严格预处理并保留验证、测试全部行；任何转换失败应终止预处理。

### 5.3 `configs/mechet_retrobridge_flower_overfit32.yaml`

- 仅用于 milestone smoke test，不是正式 benchmark 配置。
- 将同一批 32 条训练数据复制到 train/val/test。
- 扩散步数降为 100。
- 关闭额外谱特征。
- 模型减为 3 层和更小 hidden dimensions。
- batch size 为 4，训练 30 epochs。

该配置用于验证数据管道、loss 下降、checkpoint 和推理闭环，不代表正式精度。

## 6. 新增审计和运行资产

`audits/mechet_retrobridge/flower_full/` 保存了以下可复查产物：

- 小样本和 overfit 数据及其处理后 PyG 图。
- 数据身份审计报告。
- 依赖环境锁定信息。
- FlowER full 兼容性扫描。
- 被排除源行及 removal manifest。
- overfit checkpoint、CSV 训练日志和推理输出。
- smoke test 汇总。

此外，本地还有：

- `.venv-retrobridge/`：本地 Python 环境，不属于源码修改。
- `wait_for_gpu_and_train_uspto.log`：waiter 运行日志。
- `errorlog.txt`：历史 RetroBridge 异常和修复记录。

## 7. 未修改的关键官方部分

以下核心文件与官方快照完全一致：

- `src/models/transformer_model.py`
- `src/models/layers.py`
- `src/frameworks/diffusion_utils.py`
- `src/frameworks/noise_schedule.py`
- `src/features/extra_features_molecular.py`
- `src/metrics/*`
- `sample.py`
- `predict.py`
- `mit/sample.py`
- `README.md`

因此，本地修改没有改变 Graph Transformer 主体、cosine noise schedule、Markov bridge 转移公式或官方原生采样入口。主要算法级变化仅限额外谱特征的数值求解方式，而非所选谱特征的定义。

## 8. 与官方行为不等价的地方

### 8.1 自定义数据验证指标和 checkpoint 选择

有 metadata 的数据使用 dummy train/sampling metrics，不再产生官方 top-1/top-5 validation accuracy，也不使用它们挑选最佳 checkpoint。正式结果需要通过独立统一评测脚本计算，训练目录中的 `last.ckpt` 不等价于官方按 top-5 validation accuracy 选择的 best checkpoint。

### 8.2 FlowER full 的评测分母

现有 FlowER full 配置和 audit 产物表明，转换时显式排除了不兼容源行，其中包括测试行。这会改变测试分母，不等价于“保留失败行并在评测时直接判错”。`prepare_mechet_retrobridge.py` 会记录这种变化，但记录本身不会恢复原分母。若 benchmark 要求验证/测试 failure 按错误计入，最终汇总必须把这些排除 ID 补回并作为无有效候选的错误样本，或者改造数据管道使其保留占位记录而不改变 denominator。

### 8.3 动态词表与分子额外特征

`RetroBridgeDatasetInfos.init_attributes()` 中的 valency 和 atom weight 表仍按官方固定词表定义。三个 MechET 配置都设置 `extra_molecular_features: false`，因此当前路径不会使用这些可能错位的表。若以后对动态词表启用 molecular features，需要先按动态 atom decoder 重建 valency/weight 映射。

### 8.4 谱特征稳定性与速度

当前实现消除了 batch padding 的人工重复谱，并对孤立节点作阈值内的确定性 tie-break。真实分子图仍可能有重复特征值；对应特征向量的符号和退化子空间基底并不唯一。正常路径是 bucketed GPU FP32，逐图 FP64/CPU 只用于异常 bucket。

### 8.5 CLI 和依赖兼容性

- 官方命令中的 `--disable_wandb` 在本地已失效，应使用 `--disable_swanlab`。
- 本地依赖替换为 SwanLab，直接使用官方 requirements/命令无法复现本地日志行为。
- 两个正式配置默认从头训练；恢复训练时必须显式填写已有实验名。

### 8.6 版本追踪

本地目录没有独立 `.git` revision。若这些修改需要长期复现，建议将该目录纳入 MechET Git 跟踪，或维护针对官方 commit `5442b4f...` 的 patch series。

## 9. 修改文件清单

### 官方文件已修改

```text
configs/digress.yaml
configs/forwardbridge.yaml
configs/retrobridge.yaml
mit/train.py
requirements.txt
src/analysis/visualization.py
src/data/retrobridge_dataset.py
src/features/extra_features.py
src/frameworks/discrete_diffusion.py
src/frameworks/markov_bridge.py
src/frameworks/one_shot_model.py
train.py
```

### 核心新增文件

```text
MODIFICATIONS_FROM_OFFICIAL.md
audit_mechet_retrobridge.py
configs/mechet_retrobridge_flower_full.yaml
configs/mechet_retrobridge_flower_overfit32.yaml
configs/mechet_retrobridge_mech_uspto_31k_full.yaml
errorlog.txt
prepare_mechet_retrobridge.py
sample_mechet_retrobridge.py
src/loggers/__init__.py
src/loggers/swanlab_logger.py
tests/test_extra_features.py
wait_for_gpu_and_train_uspto.sh
```

## 10. 结论

当前版本不是对 RetroBridge 模型公式的重新实现，而是一个面向 MechET benchmark 的工程适配版本。主要改动集中在动态数据空间、ID/分母审计、稳定谱分解、现代依赖兼容、实验日志和统一推理输出。需要特别注意的是，自定义数据的原生验证指标与官方 checkpoint 选择已被关闭，FlowER full 的既有转换产物还改变了测试分母；这两点会直接影响与官方结果的可比性。
