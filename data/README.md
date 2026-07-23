# Data

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

### 3. USPTO-MIT

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

```bash
python scripts/build_mechet_sft.py \
  --flower-root "${FLOWER_ROOT:-/path/to/flower_new_dataset}" \
  --out-dir data/mechet_sft \
  --splits train valid test

# Resume a partial train split:
python scripts/build_mechet_sft.py --out-dir data/mechet_sft --splits train --resume
```

If you already built data in the parent `reflow` repo:

```bash
ln -s /path/to/reflow/data/orbit_mech_et_sft data/mechet_sft
```
