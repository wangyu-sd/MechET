# Data

## FlowER full-endpoint evaluation views

`scripts/build_flower_endpoint_matched_subset.py` derives the frozen 3,080-row
endpoint view corresponding to the executable-trace test from the complete
28,971-row FlowER endpoint test. Matching uses the original FlowER trajectory
ID, preserves the full-endpoint reference policy, and writes a hash-bearing
manifest with any trace/endpoint reference disagreements.

## Shipped samples

- `samples/valid_mini.jsonl` — tiny gold MECH_ET examples (linear / tree / DAG) for smoke tests.

Do **not** commit full FlowER dumps or multi-GB JSONL.

## Download upstream corpora

### 1. FlowER mechanistic dataset (`flower_new_dataset`)

Used by `scripts/build_mechet_sft.py` to build MechET SFT JSONL.

| Item | Link |
|------|------|
| Archive | https://doi.org/10.6084/m9.figshare.32513667 (`data.zip`) |
| Upstream code | https://github.com/FongMunHong/FlowER |

```bash
mkdir -p data/raw
# Download data.zip from Figshare into data/raw/, then:
unzip data/raw/data.zip -d data/raw
# Locate the directory that contains train.txt / val.txt / test.txt, e.g.:
export FLOWER_ROOT=data/raw/data/flower_new_dataset
ls "$FLOWER_ROOT"/{train,val,test}.txt
```

Line format: `mapped_reactants>>mapped_products|sequence_idx`  
Elementary steps with the same index form one mechanism graph.

#### Derive `flower_retro` (no separate download)

The public archive may contain only `flower_new_dataset`.  The
`flower_retro` directory used by the complete reaction-level endpoint track is
**not another required download**.  It is the deterministic endpoint view of
the trajectories: for each trajectory ID, in first-appearance order, take the
left side of its first elementary transition and the right side of its last
elementary transition.

```bash
export FLOWER_ROOT=data/raw/data/flower_new_dataset

python scripts/build_flower_retro.py \
  --flower-root "$FLOWER_ROOT" \
  --output-dir "$(dirname "$FLOWER_ROOT")/flower_retro" \
  --splits train valid test

ls "$(dirname "$FLOWER_ROOT")/flower_retro"/{train,val,test}.txt
```

The builder fails closed unless the official artifacts have the following
row counts and SHA-256 digests:

| Split/file | Rows | SHA-256 |
|---|---:|---|
| `train.txt` | 257,171 | `35ca595c362249c4922a9157a316c8c7fb839b5627d03c0d544abe46314615a5` |
| `val.txt` | 2,890 | `19ddcd0fd8ad5134994ce92e2556c011cba41466bb5bbde08c14f693bb138769` |
| `test.txt` | 28,971 | `5fb019a128ad9f2ec62f23fea4be22f9d5baf226147c9ec0985fba0213988e6d` |

These hashes were verified byte-for-byte against the historical local
`flower_retro` files used by the MechET full-endpoint experiments.  Use
`--allow-noncanonical` only for toy tests or explicitly documented alternate
upstream revisions.

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

Mechanistic USPTO-derived elementary steps with source-to-sink arrows. This is
the source for inverse Tool-SFT and is distinct from USPTO-50K and FlowER.

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision d708ff68be35fd02d2c1e183ee3d437b0b647f53 \
  --output data/raw/mech_uspto_31k
```

See
[`docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md`](../docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md)
for the complete replay, stitching, inverse conversion, validation, and
training procedure.

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

## Build MechET SFT from FlowER

### Complete reaction-level endpoint track (no trace filtering)

Use this track for the full product-only retrosynthesis denominator.  It keeps
all official FlowER reaction/trajectory rows: 257,171 train, 2,890 validation,
and 28,971 test.  It must not be described as executable mechanism
supervision; the replay-verified Tool-SFT data below is a separately reported
subset.

```bash
python scripts/build_flower_full_endpoint_sft.py \
  --data-root "$(dirname "$FLOWER_ROOT")" \
  --output-dir data/flower_full_endpoint_sft \
  --splits train valid test
```

The primary target contains whole reactant fragments sharing at least one atom
map with the mapped main product.  Non-contributing fragments are retained in
`auxiliary_fragments`; no reaction is dropped because its mechanism cannot be
compiled or replayed.

### Executable mechanism / trace track

```bash
python scripts/build_mechet_sft.py \
  --flower-root "${FLOWER_ROOT:-/path/to/flower_new_dataset}" \
  --out-dir data/mechet_sft \
  --splits train valid test

# Resume a partial train split:
python scripts/build_mechet_sft.py --out-dir data/mechet_sft --splits train --resume
```

This path can quarantine rows at graph validation, proof compilation, and tool
replay.  Always report its retained count and coverage against the complete
endpoint split; do not use its smaller test set as the full FlowER test set.
The graph builder aggregates by trajectory ID across the entire source file;
FlowER IDs are not guaranteed to occupy one contiguous block.

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
# After build (or symlink) has data/mechet_sft/valid.jsonl:
python scripts/make_mechet_overfit32.py \
  --src data/mechet_sft/valid.jsonl \
  --out-dir data/mechet_sft/overfit32 \
  --seed 11

python scripts/train_mechet_sft.py --config configs/overfit32.yaml
```

`manifest.json` in the out dir records ids, seed, and topology counts.
