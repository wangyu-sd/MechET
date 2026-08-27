# RetroBridge FlowER small-data audit

This directory contains a Milestone 1 (100 reactions) preprocessing audit and a
Milestone 2 (32 reactions) overfit/sampling smoke test. It is not a full FlowER
benchmark result.

Run commands from `/home/estar/pxy/mechet/baselines/RetroBridge-main`.

## Environment

The local `.venv-retrobridge` inherits Torch/PyG/RDKit from the `gnn` Conda
environment and contains the additional Lightning dependencies. Exact relevant
versions are in `environment-lock.txt`. The supplied RetroBridge directory is
not a Git checkout, so a repository commit SHA cannot be recorded; the audit
report explicitly records `unavailable_not_a_git_checkout`.

## 100-reaction preprocessing audit

```bash
/home/estar/anaconda3/bin/python prepare_mechet_retrobridge.py \
  --source-dir /home/estar/pxy/mechet/MechET/data/external_baselines/flower_full \
  --output-dir audits/mechet_retrobridge/flower_full/data/audit100 \
  --train-limit 100 \
  --valid-limit 1 \
  --test-limit 1

MPLCONFIGDIR=/tmp/retrobridge-mpl \
  .venv-retrobridge/bin/python audit_mechet_retrobridge.py \
  --data-root audits/mechet_retrobridge/flower_full/data/audit100 \
  --output audits/mechet_retrobridge/flower_full/audit100_report.json \
  --stages train val test
```

The 100 frozen train reactions all survive official graph construction. The
single valid/test support rows exist only because the native data module always
constructs all three loaders. Vocabulary and dummy-node capacity are derived
from train only.

The unmodified code would reject or silently alter some of these reactions:
eight require more than 10 dummy nodes and six contain Pd. The adapter therefore
loads a train-derived atom vocabulary and dummy capacity from `metadata.json`,
preserves `stable_id` in every PyG graph, and makes conversion failures fatal.

## 32-reaction overfit test

```bash
/home/estar/anaconda3/bin/python prepare_mechet_retrobridge.py \
  --source-dir /home/estar/pxy/mechet/MechET/data/external_baselines/flower_full \
  --output-dir audits/mechet_retrobridge/flower_full/data/overfit32 \
  --overfit-size 32

MPLCONFIGDIR=/tmp/retrobridge-mpl \
  .venv-retrobridge/bin/python audit_mechet_retrobridge.py \
  --data-root audits/mechet_retrobridge/flower_full/data/overfit32 \
  --output audits/mechet_retrobridge/flower_full/overfit32_preprocess_report.json \
  --stages train val test

CUDA_VISIBLE_DEVICES=7 MPLCONFIGDIR=/tmp/retrobridge-mpl \
  .venv-retrobridge/bin/python train.py \
  --config configs/mechet_retrobridge_flower_overfit32.yaml \
  --model RetroBridge \
  --disable_swanlab
```

`--overfit-size` intentionally writes the same 32 train reactions to train,
validation, and test. This is only a leakage-positive overfit diagnostic. The
run used the native Markov-bridge VLB objective, with a smaller 918K-parameter
network and 100-step time grid. Mean train batch loss fell from 0.353674 in
epoch 0 to 0.030584 in epoch 29 (91.35% reduction).

## Fixed-budget inference and common export

The following command uses the official `use_one_hot` sampling option. The
checkpoint is locally generated and trusted; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`
is needed because Torch 2.8 otherwise refuses the Lightning checkpoint's saved
Python objects.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
CUDA_VISIBLE_DEVICES=7 \
MPLCONFIGDIR=/tmp/retrobridge-mpl \
  .venv-retrobridge/bin/python sample_mechet_retrobridge.py \
  --data-root audits/mechet_retrobridge/flower_full/data/overfit32 \
  --checkpoint audits/mechet_retrobridge/flower_full/runs/checkpoints/mechet_retrobridge_flower_overfit32_24_08_07_43_42/last.ckpt \
  --output audits/mechet_retrobridge/flower_full/predictions.jsonl \
  --trace-output audits/mechet_retrobridge/flower_full/inference_trace.jsonl \
  --n-samples 3 \
  --n-steps 100 \
  --batch-size 4 \
  --sampling-seed 42 \
  --use-one-hot
```

The budget is three independent stochastic samples per product, not a beam, so
candidate ranks record generation order. All 96 candidates parse as SMILES and
all 32 output IDs map back to the source IDs. The native sampler returns score
0 for every sample; it therefore supplies no learned ranking signal. The
one-hot likelihood accumulators are non-finite and are saved as JSON `null` in
the detailed trace.

This smoke model produced no exact reference precursor among its 96 candidates.
That does not invalidate the wiring test, but it means this run must not be
reported as an accuracy result.

## Full-data blocker found during the audit

The full 257,171-row FlowER train scan found 23 real elements, a maximum
precursor-minus-product size of 59, and 11,439 rows above the native limit of
10. It also found a small set of source mapping anomalies (category counts 9,
7, and 2, with possible overlap). By explicit decision, RetroBridge data
preparation excludes the 16 incompatible train rows and 5 incompatible test
rows. The shared source JSONL files remain unchanged. This changes the
RetroBridge train and evaluation denominators and must be reported with any
result. See `full_train_compatibility_scan.json`.

## Full FlowER conversion and training

The commands below start from the complete FlowER `flower_full` source and
apply the explicitly documented RetroBridge compatibility filter. The
converted dataset is written to `/data/pxy/data/RetroBridge/flower_full`,
while checkpoints and logs are written under
`/data/pxy/models/RetroBridge/flower_full`.

### 1. Convert the complete dataset

```bash
cd /home/estar/pxy/mechet/baselines/RetroBridge-main

/home/estar/anaconda3/bin/python prepare_mechet_retrobridge.py \
  --source-dir /home/estar/pxy/mechet/MechET/data/external_baselines/flower_full \
  --output-dir /data/pxy/data/RetroBridge/flower_full \
  --atom-vocabulary published-mit \
  --drop-incompatible
```

`published-mit` selects the fixed atom vocabulary published with the MIT
RetroBridge data instead of deriving the vocabulary from all FlowER splits.
This also covers the `Ar` and `K` atom symbols seen only in the FlowER test
split without learning information from that split.

The `--drop-incompatible` option removes the 16 anomalous train rows and 5
anomalous test rows from the generated RetroBridge CSV files only. It never
modifies the shared source JSONL files. `metadata.json` records every excluded
stable ID, its source line, and its failure category. Because five test rows
are excluded, report this denominator change with all RetroBridge metrics.

### 2. Build graph caches and audit the conversion

Run this after conversion has completed:

```bash
cd /home/estar/pxy/mechet/baselines/RetroBridge-main

MPLCONFIGDIR=/tmp/retrobridge-mpl \
.venv-retrobridge/bin/python audit_mechet_retrobridge.py \
  --data-root /data/pxy/data/RetroBridge/flower_full \
  --output /data/pxy/data/RetroBridge/flower_full/full_preprocess_report.json \
  --stages train val test
```

Do not start full training unless this audit completes successfully. Retain
`full_preprocess_report.json` with the run artifacts.

### 3. Train RetroBridge

```bash
cd /home/estar/pxy/mechet/baselines/RetroBridge-main

CUDA_VISIBLE_DEVICES=7 \
MPLCONFIGDIR=/tmp/retrobridge-mpl \
.venv-retrobridge/bin/python train.py \
  --config configs/mechet_retrobridge_flower_full.yaml \
  --model RetroBridge
```

The full-data configuration currently trains on one visible GPU. Its physical
batch size is 16 and `accumulate_grad_batches: 4`, giving an effective batch
size of 64. Change `CUDA_VISIBLE_DEVICES=7` if another GPU should be used.
Training metrics, hyperparameters, molecule images, and chain GIFs are logged
to the `RetroBridge` SwanLab project. Run `.venv-retrobridge/bin/swanlab login`
once before starting online logging. CSV metrics are retained locally as a
second logger.

Expected output locations:

- Checkpoints: `/data/pxy/models/RetroBridge/flower_full/checkpoints/<experiment_timestamp>/`
- Last checkpoint: `/data/pxy/models/RetroBridge/flower_full/checkpoints/<experiment_timestamp>/last.ckpt`
- Logs: `/data/pxy/models/RetroBridge/flower_full/logs/`
