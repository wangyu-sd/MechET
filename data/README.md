# Data

## Benchmark-universe rule

Headline one-step retrosynthesis experiments use the **complete reaction-level splits**:

- FlowER: 257,171 train / 2,890 valid / 28,971 test;
- mech-USPTO-31k: 24,959 train / 3,120 valid / 3,120 test.

Executable trace datasets are additional program-supervision views and must not replace these denominators. See [`docs/DATASET_PROCESSING_PROTOCOL.md`](../docs/DATASET_PROCESSING_PROTOCOL.md).

## FlowER full-endpoint evaluation views

`scripts/build_flower_endpoint_matched_subset.py` derives the frozen 3,080-row endpoint view corresponding to the executable-trace test from the complete 28,971-row FlowER endpoint test. Matching uses the original FlowER trajectory ID, preserves the full-endpoint reference policy, and writes a hash-bearing manifest with any trace/endpoint reference disagreements. The 3,080-row view is for trace-matched analyses only; it is not the full benchmark.

## Shipped samples

- `samples/valid_mini.jsonl` — tiny gold MECH_ET examples (linear / tree / DAG) for smoke tests.

Do **not** commit full FlowER dumps or multi-GB JSONL.

## Download upstream corpora

### 1. FlowER mechanistic dataset (`flower_new_dataset`)

Used by `scripts/build_mechet_sft.py` to build MechET trace SFT JSONL.

| Item | Link |
|------|------|
| Archive | https://doi.org/10.6084/m9.figshare.32513667 (`data.zip`) |
| Upstream code | https://github.com/FongMunHong/FlowER |

```bash
mkdir -p data/raw
unzip data/raw/data.zip -d data/raw
export FLOWER_ROOT=data/raw/data/flower_new_dataset
ls "$FLOWER_ROOT"/{train,val,test}.txt
```

Line format: `mapped_reactants>>mapped_products|sequence_idx`. Elementary steps with the same index form one mechanism graph.

### 2. USPTO-50K

Standard single-step retrosynthesis benchmark (~50k classified reactions).

| Item | Link |
|------|------|
| GLN (recommended splits) | https://github.com/Hanjun-Dai/GLN |
| DeepChem CSV | https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv |

```bash
mkdir -p data/raw/uspto50k
curl -L -o data/raw/uspto50k/USPTO_50K.csv \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv
```

### 3. mech-USPTO-31k

Mechanistic USPTO-derived elementary steps with source-to-sink arrows. The upstream reaction universe contains exactly 31,199 reactions: 24,959 train, 3,120 validation, and 3,120 test.

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision d708ff68be35fd02d2c1e183ee3d437b0b647f53 \
  --output data/raw/mech_uspto_31k
```

Build the **full reaction-level endpoint benchmark** with no executor filtering:

```bash
python scripts/build_mech_uspto_full_endpoint_sft.py \
  --data-root data/raw/mech_uspto_31k/data \
  --output-dir data/mech_uspto_31k_full_endpoint_sft
```

The builder groups elementary rows by `rxn_idx`. For each reaction it uses `elem_reac_min` from `step_idx_forward = 0` as the precursor-side reference and the unique reaction-level `rxn_prod_min` as the product. It requires all 31,199 reactions to be present and does not run the executor, discard unsupported mechanisms, or perform trace stitching.

The separate executable-trace path is documented in [`docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md`](../docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md). Its 11,429 reactions are a program-supervision subset, not the mech-USPTO benchmark denominator.

### 4. USPTO-MIT

Larger USPTO subset curated by Jin et al. (~479k reactions; forward / retro baselines).

| Item | Link |
|------|------|
| RexGen `USPTO/data.zip` | https://github.com/wengong-jin/nips17-rexgen |
| DeepChem CSV | https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv |

```bash
mkdir -p data/raw/uspto_mit
curl -L -o data/raw/uspto_mit/USPTO_MIT.csv \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv
```

## Build FlowER endpoint and trace views

### Complete reaction-level endpoint track (no trace filtering)

Use this track for the full product-only retrosynthesis denominator. It keeps all official FlowER reaction/trajectory rows: 257,171 train, 2,890 validation, and 28,971 test.

```bash
python scripts/build_flower_full_endpoint_sft.py \
  --data-root /path/to/data-containing-flower_retro-and-flower_new_dataset \
  --output-dir data/flower_full_endpoint_sft \
  --splits train valid test
```

The primary target contains whole reactant fragments sharing at least one atom map with the mapped main product. Non-contributing fragments are retained in `auxiliary_fragments`; no reaction is dropped because its mechanism cannot be compiled or replayed.

### Executable mechanism / trace track

```bash
python scripts/build_mechet_sft.py \
  --flower-root "${FLOWER_ROOT:-/path/to/flower_new_dataset}" \
  --out-dir data/mechet_sft \
  --splits train valid test

python scripts/build_mechet_sft.py --out-dir data/mechet_sft --splits train --resume
```

This path can quarantine rows at graph validation, proof compilation, and tool replay. Always report its retained count and coverage against the complete endpoint split; do not use its smaller test set as the full FlowER test set. The graph builder aggregates by trajectory ID across the entire source file; FlowER IDs are not guaranteed to occupy one contiguous block.

If you already built data in the parent `reflow` repo:

```bash
ln -s /path/to/reflow/data/orbit_mech_et_sft data/mechet_sft
```

## Overfit32 smoke slice

`configs/overfit32.yaml` expects a **tiny debug subset**, not the full SFT files:

- `data/mechet_sft/overfit32/train.jsonl` — 32 examples
- `data/mechet_sft/overfit32/valid.jsonl` — 8 examples (disjoint)

It is topology-balanced (`linear` / `tree` / `dag_branch_join`) for pipeline smoke / memorization checks before `sft_pilot.yaml`. **Not** a formal train/val split.

```bash
python scripts/make_mechet_overfit32.py \
  --src data/mechet_sft/valid.jsonl \
  --out-dir data/mechet_sft/overfit32 \
  --seed 11

python scripts/train_mechet_sft.py --config configs/overfit32.yaml
```

`manifest.json` in the output directory records IDs, seed, and topology counts.
