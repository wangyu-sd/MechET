# MechET

**MechET** = **Mech**anism **E**lectron-**T**ransfer CoT for retrosynthesis.

Given only a mapped **product SMILES**, MechET trains an LLM to:
1. perceive electronic endpoints / reaction centers,
2. name an inverse electron-transfer signature,
3. reconstruct the full reverse FlowER mechanism graph (chain / tree / DAG) with explicit **bond-electron deltas** (`BE_DELTA`),
4. answer with the **initial reactant system**.

This is **mechanism-guided retrosynthesis**: the graph is the verifiable chain-of-thought; the reactants are the final answer.

```text
product SMILES
    → PERCEIVE + ET_SIGNATURE
    → MECH_ET graph (STATE / RETRO_EDGE / BE_DELTA)
    → <answer> initial reactants
```

## Contributions

MechET’s novelty is **not** “we ran GRPO”. It is a **verifiable mechanism CoT** plus **self-induced process rewards** that need no external teacher at post-training time.

1. **Representation / task** — `MECH_ET v3`: from a mapped product SMILES, predict a full reverse FlowER mechanism graph (chain / tree / DAG) with explicit bond-electron transfers (`BE_DELTA`), then answer with the initial reactants. Trees and DAGs are kept; we do not collapse mechanisms to a single path.

2. **Self-induced process verification** — Given student-written `STATE` pairs, the correct \(\Delta BE\) is analytically determined (RDKit BE matrix). Matching `BE_DELTA`, electron conservation, and graph reachability yield **dense, executable process rewards** without a larger LLM teacher, a slow FlowER forward neural pass, or a learned process-RM.

3. **Self-MechVR post-training recipe** — SFT on gold MECH_ET, then **teacher-free on-policy RLVR** gated by a chemical feasible set
   \(\mathcal{F}=\{\text{format}\wedge\text{reachability}\wedge\text{e-conserved}\}\),
   with rewards for BE alignment and reactant answer. Optional edge-level credit (score each `RETRO_EDGE` when written) densifies long-horizon learning. Deployable as a **single student model**.

4. **Analysis axis** — Topology-split evaluation (linear / tree / DAG) and ablations (`−BE` / `−conserv` / outcome-only / SFT-only) isolate the value of process rewards vs sparse reactant matching.

## Why this format?

| Design | Role |
|---|---|
| FlowER elementary steps | Official DiGraph semantics (unique root, self-loop terminals) |
| `BE_DELTA` | Explicit arrow-pushing in FlowER BE-matrix units (single bond = 1) |
| `SHARED` + strip-H `STATE` | Compress long system SMILES for LLM context |
| Graph topologies | Keep trees/DAGs; do not collapse to a single path |
| Process rewards | format · reachability · BE exact · electron conservation · answer — all local |

## Post-training (Self-MechVR)

```text
SFT (gold MECH_ET)
  → on-policy rollouts
  → local verifier rewards (no external teacher)
  → GRPO / RLOO-style RLVR
```

| Signal | Source | External model? |
|---|---|---|
| format / parse | `MECH_ET v3` grammar | No |
| reachability | reverse graph walk | No |
| BE exact | \(\Delta BE(S_b)-\Delta BE(S_a)\) vs written `BE_DELTA` | No (RDKit) |
| electron conserved | \(\sum \Delta BE \approx 0\) | No |
| answer | precursors ↔ `PRECURSOR_STATE` | No |

We intentionally **avoid** slow FlowER neural forward teachers and hard-to-deploy large LLM teachers as dependencies of the main method.

## Install

```bash
git clone git@github.com:wangyu-sd/MechET.git
cd MechET
pip install -e ".[dev]"
# Needs RDKit; for training also: transformers, peft, datasets, bitsandbytes, accelerate
```

## Datasets

MechET SFT is built from **FlowER** elementary-step trajectories. USPTO-50K / USPTO-MIT are standard retrosynthesis benchmarks (not required to build MECH_ET JSONL, but useful for transfer / comparison). See also `data/README.md`.

### FlowER mechanistic dataset (`flower_new_dataset`)

Official release (Figshare): https://doi.org/10.6084/m9.figshare.32513667  
Code / prep notes: https://github.com/FongMunHong/FlowER

```bash
# Download data.zip from Figshare, then:
mkdir -p data/raw
unzip data.zip -d data/raw
# Expect: data/raw/data/flower_new_dataset/{train,val,test}.txt
# (layout may be data/flower_new_dataset/... depending on the archive)

# Point the builder at the folder that contains train.txt / val.txt / test.txt
export FLOWER_ROOT=data/raw/data/flower_new_dataset
```

Line format: `mapped_reactants>>mapped_products|sequence_idx`  
Steps sharing the same `sequence_idx` belong to one overall reaction / mechanism graph.

### USPTO-50K

Canonical sources:

- GLN (Dai et al.): https://github.com/Hanjun-Dai/GLN  
- DeepChem CSV: https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv

```bash
mkdir -p data/raw/uspto50k
curl -L -o data/raw/uspto50k/USPTO_50K.csv \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv
# Or clone GLN and copy their raw_train / raw_val / raw_test CSVs.
```

### USPTO-MIT (~479K; Jin et al.)

Canonical sources:

- RexGen: https://github.com/wengong-jin/nips17-rexgen (`USPTO/data.zip`)  
- DeepChem CSV: https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv

```bash
mkdir -p data/raw/uspto_mit
curl -L -o data/raw/uspto_mit/USPTO_MIT.csv \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv
# Or: download USPTO/data.zip from the RexGen repo and unpack into data/raw/uspto_mit/
```

## Quickstart

### 1) Inspect sample CoT

```bash
python - <<'PY'
import json
print(open("data/samples/valid_mini.jsonl").readline()[:500])
PY
```

### 2) Build SFT from FlowER

Requires `flower_new_dataset` (see **Datasets** above). Files look like `mapped_reactants>>mapped_products|sequence_idx`.

```bash
python scripts/build_mechet_sft.py \
  --flower-root "${FLOWER_ROOT:-/path/to/flower_new_dataset}" \
  --out-dir data/mechet_sft \
  --splits train valid test
```

Resume after interruption:

```bash
python scripts/build_mechet_sft.py --out-dir data/mechet_sft --splits train --resume
```

### 3) Evaluate gold CoT

```bash
python scripts/eval_mechet.py --data data/mechet_sft/valid.jsonl --limit 200
```

### 4) Train (Qwen + QLoRA)

```bash
export QWEN_MODEL_PATH=/path/to/local/qwen
python scripts/train_mechet_sft.py --config configs/overfit32.yaml
python scripts/train_mechet_sft.py --config configs/sft_pilot.yaml
```

Loss is **assistant-only** causal LM CE: user/system tokens are masked (`labels=-100`).

## CoT schema (`MECH_ET v3`)

```text
<mechanism>
MECH_ET v3
DIRECTION RETRO
TARGET_SMILES "<product>"
PERCEIVE
  ENDPOINT <label> maps=<ids>
  CENTER <a-b>
ET_SIGNATURE <name>
ET_DEMAND <name>
N_STATES n
N_EDGES m
SHARED "<spectators>"
STATE s0 "<active>"
...
TARGET_STATE s0
PRECURSOR_STATE sk
RETRO_EDGE s0 s1
  BE_DELTA
    BOND i j ±d
    LP i ±d
    CHARGE i q0 q1
</mechanism>
<answer>
<initial reactants>
</answer>
```

`BE_DELTA` on `RETRO_EDGE a→b` is \(\Delta BE = BE(b)-BE(a)\) (reverse electron redistribution).

## Repository layout

```text
src/mechet/          # core library
  mech_graph.py      # FlowER graph build / MECH_GRAPH v2 serialize
  mech_et.py         # BE matrix, MECH_ET v3, verify
  sft.py             # chat SFT formatting
  verifier.py        # process rewards
  collator.py        # assistant-only labels
scripts/             # build / eval / train
configs/             # overfit32 + pilot YAML
data/samples/        # tiny gold examples
tests/               # unit tests (needs FlowER val.txt for full suite)
```

## Tests

```bash
# Graph/BE unit tests that hit real FlowER val (edit path in tests if needed)
export PYTHONPATH=src
pytest -q tests/test_mech_et.py
```

Default FlowER val path in tests:

`/aaa/fionafyang/buddy1/whaleywang/datasets/retro/data/flower_new_dataset/val.txt`

Override by editing `FLOWER_VAL` in `tests/test_mech_et.py`.

## Relation to FlowER / ORBIT

- **FlowER** supplies elementary-step trajectories and BE-matrix semantics.
- **MechET** turns those trajectories into LLM-trainable reverse CoT for reactant prediction.
- Extracted from the broader ORBIT / Reflow research codebase as a focused, publishable track.

## Citation

If you use this code, please cite FlowER and this repository:

```bibtex
@misc{mechet2026,
  title        = {MechET: Mechanism Electron-Transfer CoT and Self-MechVR for Retrosynthesis},
  author       = {wangyu-sd},
  year         = {2026},
  howpublished = {\url{https://github.com/wangyu-sd/MechET}}
}
```

## License

MIT (see `LICENSE`).
